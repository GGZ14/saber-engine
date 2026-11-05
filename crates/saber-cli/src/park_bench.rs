//! `saber park-bench` — what parking a session actually costs on this box.
//!
//! The architecture's estimate is the thing under test: *"~5 ms restore from RAM at PCIe speed;
//! ~12 ms from the CPU-attached Gen5 drive, which makes NVMe a real second parking tier rather
//! than a cold store."* A synthetic session at prod's TP=2 rank shapes — a 72 MiB GDN state and a
//! 4k-token fp8 KV — is parked and restored `--iters` times to each tier, and the medians go in the
//! ledger next to those two numbers.
//!
//! Two things make this a gate rather than a demo:
//!
//! * the restore lands in a **different** pair of arenas from the one the park read, which is what
//!   the scheduler does (a re-admitted session gets fresh slots), and the bytes are compared
//!   against the pattern that went in;
//! * `--corrupt` flips one byte in the parking file and asserts the restore is refused with
//!   nothing reaching VRAM — the negative control for the `sha256` every image carries.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use saber_hip::{DevicePtr, HipThread};
use saber_mem::dio::OpenMode;
use saber_mem::ledger::{Ledger, LedgerSnapshot, Pool};
use saber_mem::park::{
    demote, image_digest, promote, HipParkDevice, NvmeTier, ParkDevice, ParkError, ParkTicket,
    ParkTier, PinnedHostTier, PinnedSlab, Region, RegionKind, TierStats, Verification,
    DIGEST_CHUNK,
};

/// Arguments of `saber park-bench`.
pub struct ParkBenchArgs {
    /// KV bytes in the synthetic session (4k tokens of fp8 KV at TP=2 is ~64 MiB).
    pub kv_bytes: u64,
    /// GDN state bytes (72 MiB per rank at prod's shapes).
    pub gdn_bytes: u64,
    /// Timed park/restore cycles per tier.
    pub iters: usize,
    /// Cycles discarded before timing starts.
    pub warmup: usize,
    /// Slots in each tier.
    pub slots: usize,
    /// The parking file.
    pub nvme_path: PathBuf,
    /// Pinned host memory the ledger is told the process has.
    pub pinned_budget_bytes: u64,
    /// NVMe bytes the ledger is told the parking file may hold.
    pub nvme_budget_bytes: u64,
    /// Keep the parking file instead of deleting it at the end.
    pub keep_file: bool,
    /// Run the corrupt-image negative control.
    pub corrupt: bool,
    /// HIP device ordinal within the visible set.
    pub device: i32,
    /// Threads the chunked digest may use. 0 takes the default (every logical core).
    pub hash_threads: usize,
}

/// Run the benchmark.
#[allow(clippy::too_many_lines)]
pub fn run(hip: &HipThread, args: &ParkBenchArgs) -> anyhow::Result<()> {
    hip.set_device(args.device)?;
    // The shared idle check: VRAM first, /metrics second (`crate::gpu_idle`). This bench allocates a
    // session's worth of VRAM and times a transfer, so what else is resident changes both whether
    // it fits and what it measures.
    crate::gpu_idle::require_default(hip, args.device, 0)?;
    let image_bytes = args.kv_bytes + args.gdn_bytes;
    let slot_bytes = saber_mem::dio::align_up(image_bytes);

    println!(
        "session   {} KV + {} GDN state = {} per image",
        bytes(args.kv_bytes),
        bytes(args.gdn_bytes),
        bytes(image_bytes)
    );
    println!(
        "tiers     RAM {} slots x {} pinned; NVMe {} slots in {}",
        args.slots,
        bytes(slot_bytes),
        args.slots,
        args.nvme_path.display()
    );

    // ---- VRAM: the session's home, and a SECOND pair of arenas to re-admit into ----------------
    let kv_src = hip.malloc(args.kv_bytes)?;
    let gdn_src = hip.malloc(args.gdn_bytes)?;
    let kv_dst = hip.malloc(args.kv_bytes)?;
    let gdn_dst = hip.malloc(args.gdn_bytes)?;
    println!(
        "vram      {} in 4 arenas (park reads two, restore writes the other two)",
        bytes(2 * image_bytes)
    );

    let kv_pattern: Vec<u8> = (0..args.kv_bytes).map(|i| (i % 251) as u8).collect();
    let gdn_pattern: Vec<u8> = (0..args.gdn_bytes)
        .map(|i| ((i % 241) as u8) ^ 0x5A)
        .collect();
    let stream = hip.create_stream(saber_hip::StreamFlags::NonBlocking)?;
    hip.memcpy_h2d_blocking(&kv_src, &kv_pattern, &stream)?;
    hip.memcpy_h2d_blocking(&gdn_src, &gdn_pattern, &stream)?;
    hip.memset(&kv_dst, 0xEE, &stream)?;
    hip.memset(&gdn_dst, 0xEE, &stream)?;
    hip.sync_stream(&stream)?;

    // ---- the ledger, before anything is allocated for parking -----------------------------------
    let mut ledger = Ledger::new();
    ledger.declare(Pool::HostPinned, args.pinned_budget_bytes)?;
    ledger.declare(Pool::Nvme, args.nvme_budget_bytes)?;

    // ---- pinned RAM, allocated ONCE ------------------------------------------------------------
    // One `hipHostMalloc` per tier, handed over as a slab: the tiers own their memory for the life
    // of the process (I1), which is also what lets them live behind the scheduler's `dyn Parkable`.
    let ram_bytes = args.slots as u64 * slot_bytes;
    let verify = if args.hash_threads == 0 {
        Verification::default()
    } else {
        Verification::on(args.hash_threads)
    };
    println!(
        "pinned    {} for the RAM tier + {} of NVMe staging, allocated once",
        bytes(args.slots as u64 * slot_bytes),
        bytes(slot_bytes)
    );
    println!(
        "digest    chunked sha256 on {} threads, {} chunks of {}",
        verify.threads,
        image_bytes.div_ceil(DIGEST_CHUNK as u64),
        bytes(DIGEST_CHUNK as u64)
    );
    println!();
    print_ledger("ledger before", &ledger.snapshot());

    let src = vec![
        Region::new(RegionKind::Kv, kv_src.addr(), args.kv_bytes),
        Region::new(RegionKind::GdnState, gdn_src.addr(), args.gdn_bytes),
    ];
    let dst = vec![
        Region::new(RegionKind::Kv, kv_dst.addr(), args.kv_bytes),
        Region::new(RegionKind::GdnState, gdn_dst.addr(), args.gdn_bytes),
    ];
    let mut device = HipParkDevice::new(hip, vec![&kv_src, &gdn_src, &kv_dst, &gdn_dst])?;

    // The digest alone, so the park and restore rows decompose into transfer and hash instead of
    // being one number nobody can attribute.
    let scratch = vec![0xA5u8; usize::try_from(image_bytes)?];
    let t = Instant::now();
    let _ = image_digest(&scratch, verify.threads);
    let digest_us = t.elapsed().as_secs_f64() * 1e6;
    let t = Instant::now();
    let _ = image_digest(&scratch, 1);
    let digest1_us = t.elapsed().as_secs_f64() * 1e6;
    drop(scratch);

    // ---- the runs: each tier with verification ON (the load mode) and OFF (transfer alone) ------
    let mut rows: Vec<(&'static str, &'static str, Run)> = Vec::new();
    let mut demote_us = 0.0;
    let mut promote_us = 0.0;
    let mut ram_stats = TierStats::default();
    let mut nvme_stats = TierStats::default();
    let mut session = 0u64;
    let mut control_line = None;

    for (label, policy) in [("sha256", verify), ("off", Verification::off())] {
        let ram_slab = Box::new(PinnedSlab::new(hip.host_malloc(ram_bytes)?)?);
        let mut ram = PinnedHostTier::new(ram_slab, args.slots, slot_bytes)?.verifying(policy);
        let staging = Box::new(PinnedSlab::new(hip.host_malloc(slot_bytes)?)?);
        let mut nvme = NvmeTier::create(
            &args.nvme_path,
            args.slots,
            slot_bytes,
            staging,
            OpenMode::Direct,
        )?
        .verifying(policy);
        if !args.keep_file {
            nvme = nvme.deleting_on_drop();
        }

        let ram_run = cycle(
            &mut ram,
            &mut device,
            &mut ledger,
            &src,
            &dst,
            args.warmup,
            args.iters,
            session,
        )?;
        session += (args.warmup + args.iters) as u64;
        let nvme_run = cycle(
            &mut nvme,
            &mut device,
            &mut ledger,
            &src,
            &dst,
            args.warmup,
            args.iters,
            session,
        )?;
        session += (args.warmup + args.iters) as u64;

        if policy.enabled {
            // Promotion and demotion once each, on the verified pass: this is the RAM-under-
            // pressure path, and promotion is where the drive's copy is checked on the way back.
            let parked = ram.park(9_000_000, &src, &mut device, &mut ledger)?;
            let t = Instant::now();
            let spilled = demote(&mut ram, &mut nvme, &parked, &mut ledger)?;
            demote_us = t.elapsed().as_secs_f64() * 1e6;
            let t = Instant::now();
            let promoted = promote(&mut nvme, &mut ram, &spilled, &mut ledger)?;
            promote_us = t.elapsed().as_secs_f64() * 1e6;
            ram.discard(&promoted, &mut ledger);

            // The negative control belongs on this pass: with the digest off there is nothing to
            // refuse with, which is exactly what the `off` row is there to show.
            if args.corrupt {
                // Both ends of the image: the first chunk, which must reach nothing at all, and
                // the last, which is where verifying chunk by chunk costs something.
                let first = corrupt_control(
                    &mut nvme,
                    &mut device,
                    &mut ledger,
                    &src,
                    &dst,
                    hip,
                    &kv_dst,
                    4096,
                    true,
                )?;
                let last = corrupt_control(
                    &mut nvme,
                    &mut device,
                    &mut ledger,
                    &src,
                    &dst,
                    hip,
                    &kv_dst,
                    image_bytes - 4096,
                    false,
                )?;
                control_line = Some(format!("{first}\ncontrol   {last}"));
            }
        }

        rows.push(("ram", label, ram_run));
        rows.push(("nvme", label, nvme_run));
        ram_stats = ram.stats();
        nvme_stats = nvme.stats();
        drop(ram);
        drop(nvme);
    }

    // The restores landed in the OTHER arenas: compare them against the pattern that went in.
    let mut back = vec![0u8; usize::try_from(args.kv_bytes)?];
    hip.memcpy_d2h_blocking(&mut back, &kv_dst, &stream)?;
    let kv_ok = back == kv_pattern;
    let mut back = vec![0u8; usize::try_from(args.gdn_bytes)?];
    hip.memcpy_d2h_blocking(&mut back, &gdn_dst, &stream)?;
    let gdn_ok = back == gdn_pattern;

    println!();
    println!(
        "tier  verify    n   park med µs   restore med µs   park GB/s   restore GB/s   restore p90"
    );
    for (tier, label, r) in &rows {
        println!(
            "{tier:<5} {label:<8} {:>3} {:>11.1} {:>16.1} {:>11.2} {:>14.2} {:>13.1}",
            r.n,
            r.park_med_us,
            r.restore_med_us,
            gbps(image_bytes, r.park_med_us),
            gbps(image_bytes, r.restore_med_us),
            r.restore_p90_us
        );
    }

    println!();
    println!(
        "digest    {digest_us:.1} µs on {} threads ({:.2} GB/s); {digest1_us:.1} µs on one \
         ({:.2} GB/s)",
        verify.threads,
        gbps(image_bytes, digest_us),
        gbps(image_bytes, digest1_us)
    );
    println!(
        "spill     demote RAM->NVMe {demote_us:.1} µs, promote NVMe->RAM {promote_us:.1} µs \
         (host only: the device is not touched)"
    );
    println!(
        "restored  KV {}, GDN state {} -- re-admitted into DIFFERENT arenas from the ones parked",
        yn(kv_ok),
        yn(gdn_ok)
    );
    println!(
        "readiness {} regions published per restore, as each lands",
        rows[0].2.regions_per_restore
    );
    if let Some(msg) = control_line {
        println!("control   {msg}");
    }

    println!();
    print_ledger("ledger after", &ledger.snapshot());
    println!(
        "counters  ram parks {} restores {} demotions {} | nvme parks {} restores {} refusals {}",
        ram_stats.parks,
        ram_stats.restores,
        ram_stats.demotions,
        nvme_stats.parks,
        nvme_stats.restores,
        nvme_stats.corrupt_refusals
    );

    if !kv_ok || !gdn_ok {
        anyhow::bail!("a restored session did not match the bytes that were parked");
    }
    drop(device);
    Ok(())
}

/// One tier's timings.
struct Run {
    n: usize,
    park_med_us: f64,
    restore_med_us: f64,
    restore_p90_us: f64,
    regions_per_restore: usize,
}

/// Park and restore `iters` times, after `warmup` untimed cycles.
#[allow(clippy::too_many_arguments)]
fn cycle(
    tier: &mut dyn ParkTier,
    device: &mut dyn ParkDevice,
    ledger: &mut Ledger,
    src: &[Region],
    dst: &[Region],
    warmup: usize,
    iters: usize,
    session_base: u64,
) -> anyhow::Result<Run> {
    let mut park_us = Vec::with_capacity(iters);
    let mut restore_us = Vec::with_capacity(iters);
    let mut regions = 0usize;
    for i in 0..warmup + iters {
        let session = session_base + i as u64;
        let t = Instant::now();
        let ticket = tier.park(session, src, device, ledger)?;
        let park = t.elapsed();
        let mut ready = 0usize;
        let t = Instant::now();
        tier.restore(&ticket, dst, device, ledger, &mut |_| ready += 1)?;
        let restore = t.elapsed();
        if i >= warmup {
            park_us.push(park.as_secs_f64() * 1e6);
            restore_us.push(restore.as_secs_f64() * 1e6);
            regions = ready;
        }
    }
    Ok(Run {
        n: iters,
        park_med_us: median(&mut park_us),
        restore_med_us: median(&mut restore_us),
        restore_p90_us: quantile(&mut restore_us, 0.9),
        regions_per_restore: regions,
    })
}

/// Park an image, flip one byte of the file under it, and require the restore to refuse.
///
/// `off` decides which half of the guarantee is under test. A flip inside the **first** 4 MiB
/// chunk is refused before a single byte is issued, and the destination arena's sentinel must be
/// untouched. A flip in a **later** chunk is the honest cost of verifying chunk by chunk: the
/// chunks before it have already crossed the link, so the sentinel is gone — what must still hold
/// is that the restore fails loud, that no region was ever published, and that the caller is told.
// Nine arguments because the control needs both tiers' context, the arena it probes and the two
// knobs that make it two different controls; bundling them into a struct used once would say less.
#[allow(clippy::too_many_arguments)]
fn corrupt_control(
    nvme: &mut NvmeTier,
    device: &mut dyn ParkDevice,
    ledger: &mut Ledger,
    src: &[Region],
    dst: &[Region],
    hip: &HipThread,
    kv_dst: &DevicePtr,
    off: u64,
    sentinel_must_survive: bool,
) -> anyhow::Result<String> {
    use std::io::{Read, Seek, SeekFrom, Write};

    let ticket = nvme.park(4_242, src, device, ledger)?;
    let path = nvme.path().to_path_buf();
    // A sentinel in the destination arena: if any byte of the refused restore reached VRAM, this
    // is what it overwrote.
    let stream = hip.create_stream(saber_hip::StreamFlags::NonBlocking)?;
    hip.memset(kv_dst, 0xC3, &stream)?;
    hip.sync_stream(&stream)?;

    let mut b = [0u8; 1];
    {
        let mut r = std::fs::File::open(&path)?;
        r.seek(SeekFrom::Start(off))?;
        r.read_exact(&mut b)?;
    }
    {
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)?;
        f.seek(SeekFrom::Start(off))?;
        f.write_all(&[b[0] ^ 0x80])?;
        f.sync_all()?;
    }

    // What the restore published before it refused. A region is published only when every byte
    // of it has been verified, so no published region may contain the flipped byte -- that is the
    // property, and it is checked rather than assumed.
    let mut published: Vec<(usize, u64, u64)> = Vec::new();
    let mut spans: Vec<(u64, u64)> = Vec::new();
    let mut base = 0u64;
    for region in dst {
        spans.push((base, base + region.len));
        base += region.len;
    }
    let err = match nvme.restore(&ticket, dst, device, ledger, &mut |r| {
        let (lo, hi) = spans[r.index];
        published.push((r.index, lo, hi));
    }) {
        Ok(()) => anyhow::bail!("a corrupted image restored cleanly"),
        Err(e) => e,
    };
    let ParkError::ImageCorrupt { .. } = &err else {
        anyhow::bail!("expected ImageCorrupt, got {err}");
    };
    for (index, lo, hi) in &published {
        if *lo <= off && off < *hi {
            anyhow::bail!(
                "region {index} was published although the flipped byte at {off} is inside it"
            );
        }
    }
    if sentinel_must_survive && !published.is_empty() {
        anyhow::bail!("a flip in the FIRST chunk must publish nothing");
    }

    let mut probe = vec![0u8; 4096];
    hip.memcpy_d2h_blocking(&mut probe, kv_dst, &stream)?;
    let intact = probe.iter().all(|&x| x == 0xC3);
    if sentinel_must_survive && !intact {
        anyhow::bail!("a flip in the FIRST chunk must be refused before anything reaches VRAM");
    }
    nvme.discard(&ticket, ledger);
    Ok(format!(
        "one byte flipped at offset {off} of the parking file: restore REFUSED ({}), \
         {} region(s) published (none containing the flip), VRAM sentinel {}",
        short(&err),
        published.len(),
        if intact {
            "intact"
        } else {
            "overwritten by the chunks before the corrupt one"
        }
    ))
}

fn short(e: &ParkError) -> String {
    let s = e.to_string();
    match s.find(':') {
        Some(i) => s[..i].to_string(),
        None => s,
    }
}

fn median(v: &mut [f64]) -> f64 {
    quantile(v, 0.5)
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    let idx = ((v.len() - 1) as f64 * q).round() as usize;
    v[idx]
}

fn gbps(bytes: u64, micros: f64) -> f64 {
    if micros <= 0.0 {
        0.0
    } else {
        bytes as f64 / (micros * 1e-6) / 1e9
    }
}

fn print_ledger(title: &str, snap: &LedgerSnapshot) {
    println!("{title}");
    for p in &snap.pools {
        println!(
            "  {:<14} {:>12} used of {:>12}   written {:>12}   ({} reservation(s))",
            p.label,
            bytes(p.used_bytes),
            bytes(p.capacity_bytes),
            bytes(p.written_bytes),
            p.reservations.len()
        );
    }
}

fn yn(b: bool) -> &'static str {
    if b {
        "ok"
    } else {
        "MISMATCH"
    }
}

fn bytes(n: u64) -> String {
    const UNITS: [(&str, u64); 4] = [
        ("GiB", 1 << 30),
        ("MiB", 1 << 20),
        ("KiB", 1 << 10),
        ("B", 1),
    ];
    for (u, s) in UNITS {
        if n >= s {
            return if s == 1 {
                format!("{n} B")
            } else {
                format!("{:.2} {u}", n as f64 / s as f64)
            };
        }
    }
    format!("{n} B")
}

/// Unused today, kept next to the ticket type so a future `--json` can print one.
#[allow(dead_code)]
fn ticket_line(t: &ParkTicket) -> String {
    format!(
        "session {} -> {} #{} ({}), sha {}",
        t.session,
        t.pool,
        t.id,
        bytes(t.bytes),
        hex::encode(&t.sha256[..4])
    )
}

/// One Duration to microseconds, for a future JSON emitter.
#[allow(dead_code)]
fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}
