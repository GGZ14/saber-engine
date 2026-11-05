//! `saber load` — the cold-start path, end to end, with its numbers.
//!
//! Reads one rank's pack with `O_DIRECT` at qd≥8, verifies every record's `sha256` on a host thread
//! pool, stages it through a fixed pinned ring and copies it into one VRAM arena per rank,
//! publishing each record when its bytes have landed. It is the measurement instrument for the
//! cold-start budget in `docs/ARCHITECTURE.md` (memory plane, "Cold start / model swap") and the
//! comparator for the reference server's 2.3 GB/s `default_loader`.
//!
//! Three numbers come out of it, because there are three places a byte can be held up: the drive,
//! the host memcpy into pinned staging, and the PCIe link. The drive figure is measured twice —
//! once with the pipeline attached, once by a read-only pass over the same records — so that "the
//! stream is not the bottleneck" is a measurement rather than a subtraction.

use std::path::PathBuf;
use std::time::Instant;

use saber_hip::HipThread;
use saber_mem::dio::OpenMode;
use saber_mem::ledger::{Ledger, Pool};
use saber_mem::pack::{
    Borrowed, DeliveryOrder, PackStream, PageAlignedPool, RecordRef, StreamConfig,
};
use saber_mem::upload::{HipSink, PinnedRing, Upload, UploadConfig, UploadStats};

/// Arguments of `saber load`.
pub struct LoadArgs {
    /// Pack file, or the directory holding `rank<r>.pack`.
    pub pack: PathBuf,
    /// TP rank to load.
    pub rank: u32,
    /// First table index to upload.
    pub start_record: usize,
    /// Upload only the records from `start_record` that fit in this many bytes.
    pub limit_bytes: Option<u64>,
    /// Where in the record window to start reading, as a fraction of it.
    ///
    /// `None` means front to back, which is what **measured fastest**. Rotating rank `r` of `R` by
    /// `r/R` so the two rank loaders sit a fixed distance apart in the drive's address space was
    /// the obvious idea and it lost: it makes the pair reproducible (2.44/1.77 s three times
    /// running) and slower than not doing it (best 2.22 s). Kept as a knob, off by default, with
    /// the numbers in `docs/LEDGER.md`.
    pub read_offset: Option<f64>,
    /// Use the ring size asked for, instead of sizing it from the pack. A benchmark arm.
    pub fixed_ring: bool,
    /// Staging slots.
    pub slots: usize,
    /// Bytes per staging slot.
    pub slot_bytes: usize,
    /// `O_DIRECT` reads in flight.
    pub queue_depth: usize,
    /// Bytes per `preadv`.
    pub chunk_bytes: usize,
    /// Threads in the `sha256` pool.
    pub hash_threads: usize,
    /// Host bytes the pack stream may hold in read buffers.
    pub stream_budget_bytes: u64,
    /// Skip `sha256` verification. A benchmark mode; never a load mode.
    pub no_verify: bool,
    /// VRAM the ledger is told this rank has. Not `hipMemGetInfo`: with prod at 0.95 utilisation it
    /// reports 0 free on both cards while a 520 MiB allocation still succeeds (`docs/LEDGER.md`,
    /// `saber probe` facts), so free VRAM is not a number anything may gate on.
    pub vram_budget_bytes: u64,
    /// Pinned host memory the ledger is told the process has.
    pub pinned_budget_bytes: u64,
    /// Also time a read-only pass over the same records, for the drive's own rate.
    pub io_pass: bool,
    /// Copies in flight at once. `None` is one per staging slot.
    pub events: Option<usize>,
    /// Sample this many uploaded records back out of VRAM and check their `sha256`.
    pub check_records: usize,
    /// Negative control: shift the read-back offset by this many bytes, so the check must fail.
    pub check_shift: u64,
    /// HIP device ordinal within the visible set.
    pub device: i32,
}

/// Run the upload and print the report.
pub fn run(hip: &HipThread, args: &LoadArgs) -> anyhow::Result<()> {
    hip.set_device(args.device)?;

    let cfg = UploadConfig {
        queue_depth: args.queue_depth,
        chunk_bytes: args.chunk_bytes,
        verify: !args.no_verify,
        hash_threads: args.hash_threads,
        stream_budget_bytes: args.stream_budget_bytes,
        staging_slots: args.slots,
        staging_slot_bytes: args.slot_bytes,
        start_record: args.start_record,
        limit_bytes: args.limit_bytes,
        read_offset_fraction: args.read_offset,
        // A load is the load mode: the ring is sized from the pack. `--fixed-ring` is the arm that
        // measures what that is worth.
        ring_from_pack: !args.fixed_ring,
        events: args.events,
        mode: OpenMode::Direct,
        expect_plan_hash: None,
        // `saber load` MEASURES a pack; it does not serve from it, so it states no world of its
        // own and any rank count is legitimate here.
        expect_tp: None,
    };

    let open = Instant::now();
    let upload = Upload::open(&args.pack, args.rank, cfg.clone())?;
    let open_ms = open.elapsed().as_secs_f64() * 1e3;
    let plan = upload.plan();
    let reader = upload.reader();
    let path = reader.path().to_path_buf();

    println!("pack      {}", path.display());
    println!(
        "          rank {} of {}, plan hash {}…, tier {}…",
        reader.header().rank,
        reader.header().tp_size,
        hex_prefix(&reader.header().plan_hash),
        hex_prefix(&reader.header().tier_id),
    );
    println!(
        "          {} records in the pack, {} to upload from index {}{}; \
         header+table parsed in {open_ms:.1} ms",
        plan.records_in_pack,
        plan.records,
        plan.first_record,
        match args.limit_bytes {
            Some(b) => format!(" (limit {})", bytes(b)),
            None => String::new(),
        }
    );
    println!(
        "arena     {} ({} B), one per rank, records at their own 4096-aligned pack offsets",
        bytes(plan.arena_bytes),
        plan.arena_bytes
    );
    println!(
        "staging   {} slots x {} = {} pinned, allocated once{}",
        plan.staging_bytes / args.slot_bytes as u64,
        bytes(args.slot_bytes as u64),
        bytes(plan.staging_bytes),
        match plan.staging_raised_from {
            // Said out loud, because it changes the pinned reservation: a ring that cannot run
            // past the largest record while that record is hashed makes every reader wait on one
            // core (docs/LEDGER.md, workstream K6).
            Some(asked) => format!(
                " (raised from {}: the largest record is {})",
                bytes(asked),
                bytes(plan.max_record_bytes)
            ),
            None => String::new(),
        }
    );
    {
        let (recs, chunks) = reader.chunk_summary();
        println!(
            "digests   {}",
            if chunks == 0 {
                "one sha256 per record (pack format v1, or v2 with no chunk blob)".to_owned()
            } else {
                format!(
                    "{recs} record(s) carry {chunks} chunk digests of {}; the rest is one \
                     sha256 per record",
                    bytes(u64::from(reader.chunk_bytes()))
                )
            }
        );
    }
    println!(
        "verify    {}",
        if cfg.verify {
            format!(
                "sha256 on {} host threads, before the H2D is issued",
                cfg.hash_threads
            )
        } else {
            "OFF (benchmark mode; a corrupt record would reach VRAM)".to_string()
        }
    );

    // The ledger goes first, always: over-commit is a startup error, not an OOM mid-load.
    let mut ledger = Ledger::new();
    ledger.declare(Pool::Vram(args.rank), args.vram_budget_bytes)?;
    ledger.declare(Pool::HostPinned, args.pinned_budget_bytes)?;
    let reserved = upload.reserve(&mut ledger)?;
    println!(
        "ledger    vram.rank{} {} of {} reserved; host.pinned {} of {}",
        args.rank,
        bytes(ledger.used(Pool::Vram(args.rank))),
        bytes(args.vram_budget_bytes),
        bytes(ledger.used(Pool::HostPinned)),
        bytes(args.pinned_budget_bytes),
    );

    // The staging ring IS the pack stream's buffer pool: the drive reads records into these
    // slots and the H2D sends from the same bytes. One allocation, before anything moves (I1).
    let mut staging = hip.host_malloc(reserved.staging_bytes())?;
    let ring = PinnedRing::new(&mut staging, &reserved)?;

    // Read-only passes: the drive's rate for exactly these records with no consumer attached, into
    // each kind of memory. Two numbers, because "is a page-locked destination slower to read into?"
    // is the question this design turns on, and subtracting times would only guess at it.
    let io_stream = |limit_pool: &dyn saber_mem::pack::BufferPool<'_>| -> anyhow::Result<f64> {
        let s = PackStream::with_config(
            &path,
            StreamConfig {
                queue_depth: cfg.queue_depth,
                chunk_bytes: cfg.chunk_bytes,
                order: DeliveryOrder::Completion,
                verify: cfg.verify,
                budget_bytes: cfg.stream_budget_bytes,
                hash_threads: cfg.hash_threads.max(1),
                mode: OpenMode::Direct,
                expect_plan_hash: None,
                start_record: plan.first_record,
                limit_records: Some(plan.first_record + plan.records),
                // The io pass reads what the upload reads, the way the upload reads it.
                rotate_records: cfg
                    .read_offset_fraction
                    .map(|f| ((plan.records as f64) * f.clamp(0.0, 1.0)) as usize)
                    .unwrap_or(0),
            },
        )?;
        let stats = s.run(limit_pool, &mut Borrowed(|_: RecordRef<'_>| Ok(())))?;
        Ok(stats.gigabytes_per_second())
    };
    let (io_pageable, io_pinned) = if args.io_pass {
        let pool = PageAlignedPool::new(cfg.stream_budget_bytes);
        (Some(io_stream(&pool)?), Some(io_stream(&ring)?))
    } else {
        (None, None)
    };
    println!(
        "ring      base {:#x}, {} slots x {}, {} copies in flight, largest record {} ({} the ring)",
        ring.base_addr(),
        ring.slots(),
        bytes(ring.slot_bytes() as u64),
        reserved.events(),
        bytes(plan.max_record_bytes),
        if plan.max_record_bytes <= ring.max_pinned_bytes() {
            "fits"
        } else {
            "EXCEEDS -- those records take the pageable route"
        }
    );

    let mut sink = HipSink::new(hip, &reserved)?;
    let mut first: Option<(String, u64, f64)> = None;
    let mut last: Option<(String, u64)> = None;
    // Time to the first ready record, which is the number "records publish readiness individually"
    // is actually about: a consumer can start binding this long after the load begins, not after it
    // ends.
    let t0 = Instant::now();
    let stats: UploadStats = upload.run(&mut sink, &ring, |r| {
        if first.is_none() {
            first = Some((
                r.name.to_string(),
                r.addr.as_u64(),
                t0.elapsed().as_secs_f64() * 1e3,
            ));
        }
        last = Some((r.name.to_string(), r.addr.as_u64()));
    })?;

    println!();
    println!("phase counters");
    let c = stats.counters;
    println!("  reads issued        {}", c.reads_issued);
    println!("  hashes verified     {}", c.hashes_verified);
    println!("  h2d issued          {}", c.h2d_issued);
    println!("  events waited       {}", c.events_waited);
    println!("  buffers recycled    {}", c.buffers_recycled);
    println!("  batched round trips {}", c.batches);
    println!(
        "  reader stalls       {}  (the ring applying backpressure)",
        stats.pool.stalls
    );
    println!("  idle drains         {}", c.idle_drains);
    println!("  records published   {}", c.records_published);
    println!(
        "  which path          direct {} ({}) / spilled {} ({})",
        c.records_direct,
        bytes(c.bytes_direct),
        c.records_spilled,
        bytes(c.bytes_spilled)
    );
    if let Some((name, addr, ms)) = &first {
        println!("  first ready         {name} at {addr:#x}, {ms:.1} ms into the load");
    }
    if let Some((name, addr)) = &last {
        println!("  last  ready         {name} at {addr:#x}");
    }

    println!();
    println!(
        "bytes     {} payload, {} transferred (4096-padded), peak {} held by the stream",
        bytes(plan.payload_bytes),
        bytes(plan.transferred_bytes),
        bytes(stats.peak_stream_bytes)
    );
    println!("wall      {:.3} s", stats.wall.as_secs_f64());
    println!();
    println!("throughput (GB/s = 10^9 B/s, the unit the drive bench uses)");
    if let Some(g) = io_pageable {
        println!("  SSD -> pageable, read-only pass  {g:8.2}");
    }
    if let Some(g) = io_pinned {
        println!("  SSD -> PINNED ring, read-only    {g:8.2}   (the fast path's own read rate)");
    }
    println!(
        "  SSD -> staging, in pipeline      {:8.2}   (consumer time removed; upper bound)",
        stats.ssd_to_staging_gbps()
    );
    match (stats.spill_gbps(), stats.counters.records_spilled) {
        (Some(g), n) => println!(
            "  pageable fallback                {g:8.2}   ({n} records, {:.3} s blocking)",
            stats.spill_copy.as_secs_f64()
        ),
        _ => println!(
            "  host copies                          none   ({:.0}% of bytes went straight from \
             the staging slot to VRAM)",
            stats.direct_fraction() * 100.0
        ),
    }
    match (stats.h2d_gbps(), stats.h2d_busy) {
        (Some(g), Some(d)) => println!(
            "  staging -> VRAM over PCIe        {g:8.2}   ({:.3} s of device copy time)",
            d.as_secs_f64()
        ),
        _ => println!("  staging -> VRAM over PCIe             n/a   (backend cannot time it)"),
    }
    println!(
        "  end to end                       {:8.2}",
        stats.end_to_end_gbps()
    );
    println!(
        "  host blocked: submit {:.3} s, events {:.3} s",
        stats.submit.as_secs_f64(),
        stats.event_wait.as_secs_f64()
    );

    // Read back what the device actually holds and hash it. The host tests prove the pipeline asks
    // for the right copies; only this proves the bytes landed, at the right offset, on hardware.
    if args.check_records > 0 {
        const CAP: u64 = 256 << 20;
        let n = args.check_records.min(plan.records);
        let step = (plan.records / n).max(1);
        let (mut ok, mut checked, mut skipped) = (0u32, 0u32, 0u32);
        let mut failure = None;
        for j in 0..n {
            let index = plan.first_record + j * step;
            let rec = &reader.records()[index];
            if rec.length > CAP {
                skipped += 1;
                continue;
            }
            let mut buf = vec![0u8; rec.length as usize];
            // `check_shift` is the negative control (CLAUDE.md rule 8): read the record back from
            // the wrong place and the check must fail. A gate nobody has seen fail is not a gate.
            let off = rec.data_off - plan.first_data_off + args.check_shift;
            if off + rec.length > sink.arena().len() {
                skipped += 1;
                continue;
            }
            hip.memcpy_d2h_at_blocking(&mut buf, sink.arena(), off, sink.stream())?;
            checked += 1;
            match reader.check_hash(index, &buf) {
                Ok(()) => ok += 1,
                Err(e) => failure = Some((index, e)),
            }
        }
        println!();
        println!(
            "device check  {ok}/{checked} sampled records re-read from VRAM and hashed clean\
             {}",
            if skipped > 0 {
                format!(" ({skipped} skipped, larger than {})", bytes(CAP))
            } else {
                String::new()
            }
        );
        if let Some((index, e)) = failure {
            anyhow::bail!("record {index} does not hash to its table entry after the upload: {e}");
        }
    }

    if plan.records < plan.records_in_pack {
        println!();
        println!(
            "NOTE: this was a {} prefix of a {} pack. The full per-rank load is UNTESTED while \
             prod holds both cards at 0.95 utilisation.",
            bytes(plan.arena_bytes),
            bytes(
                reader
                    .records()
                    .iter()
                    .map(saber_mem::pack::PackRecord::padded_length)
                    .sum::<u64>()
            )
        );
    }
    Ok(())
}

fn hex_prefix(h: &[u8; 32]) -> String {
    h[..4].iter().map(|b| format!("{b:02x}")).collect()
}

/// Human bytes, binary units, because every capacity on this box is quoted that way.
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

/// Parse `768M`, `8MiB`, `2G`, `1048576`. Bare `K`/`M`/`G` are binary (1024-based); the explicit
/// `KB`/`MB`/`GB` forms are decimal, because a drive's datasheet and a memory capacity do not use
/// the same thousand and pretending otherwise is how a budget ends up 7% wrong.
pub fn parse_size(s: &str) -> Result<u64, String> {
    let t = s.trim();
    if t.is_empty() {
        return Err("empty size".into());
    }
    let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return Err(format!("{s:?} does not start with a number"));
    }
    let n: u64 = digits.parse().map_err(|e| format!("{s:?}: {e}"))?;
    let unit = t[digits.len()..].trim().to_ascii_lowercase();
    let mul: u64 = match unit.as_str() {
        "" | "b" => 1,
        "k" | "kib" => 1 << 10,
        "m" | "mib" => 1 << 20,
        "g" | "gib" => 1 << 30,
        "kb" => 1_000,
        "mb" => 1_000_000,
        "gb" => 1_000_000_000,
        other => return Err(format!("unknown size unit {other:?} in {s:?}")),
    };
    n.checked_mul(mul)
        .ok_or_else(|| format!("{s:?} overflows a u64"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_parse_the_way_the_help_text_claims() {
        assert_eq!(parse_size("768M"), Ok(768 << 20));
        assert_eq!(parse_size("8MiB"), Ok(8 << 20));
        assert_eq!(parse_size("2G"), Ok(2 << 30));
        assert_eq!(parse_size("1048576"), Ok(1 << 20));
        assert_eq!(parse_size("500MB"), Ok(500_000_000));
        assert!(parse_size("").is_err());
        assert!(parse_size("M").is_err());
        assert!(parse_size("12X").is_err());
        assert!(parse_size("18446744073709551615G").is_err());
    }

    #[test]
    fn byte_rendering_is_binary() {
        assert_eq!(bytes(0), "0 B");
        assert_eq!(bytes(1023), "1023 B");
        assert_eq!(bytes(1 << 20), "1.00 MiB");
        assert_eq!(bytes(805_306_368), "768.00 MiB");
    }
}
