//! Session parking, driven with a fake VRAM so it runs on a host with no ROCm.
//!
//! The tiers under test are the real ones — a real pinned slot pool (over leaked host memory, via
//! `saber-hip`'s `test-util` feature) and a real `O_DIRECT` file with a real `posix_fallocate`.
//! Only the device half is fake, and it is fake in the one way that matters: it holds a `Vec<u8>`
//! standing in for VRAM, so a round trip can be compared **byte for byte** against what went in.
//!
//! The parking file goes under `target/` rather than `std::env::temp_dir()`: `/tmp` is tmpfs here
//! and tmpfs has no `O_DIRECT`, so a test that used it would exercise the buffered path instead of
//! the one the tier ships (`docs/LEDGER.md`, 2026-09-18).

use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use saber_hip::DeviceAddr;
use saber_mem::dio::{AlignedBuf, OpenMode};
use saber_mem::ledger::{Ledger, LedgerError, Pool};
use saber_mem::park::{
    capture_into, demote, promote, publish_from, NvmeTier, ParkDevice, ParkError, ParkTier,
    PinnedHostTier, Region, RegionKind, RegionReady, Slab, Verification, COPY_CHUNK,
};

const KV_BASE: u64 = 0x1_0000_0000;
const GDN_BASE: u64 = 0x2_0000_0000;
const KV_BYTES: u64 = 96 * 1024;
const GDN_BYTES: u64 = 40 * 1024;
const IMAGE_BYTES: u64 = KV_BYTES + GDN_BYTES;
const SLOT_BYTES: u64 = 256 * 1024;

static SEQ: AtomicU64 = AtomicU64::new(0);

/// A scratch directory on a filesystem that supports `O_DIRECT`.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let root =
            Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target")).join("saber-mem-park");
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = root.join(format!("{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A slab for a tier. Page-aligned heap memory rather than `hipHostMalloc`: this box has no ROCm,
/// and it is the same slab type `saber serve --mock` uses, so these tests exercise the shipped
/// path rather than a stand-in for it.
fn slab(bytes: usize) -> Box<dyn Slab> {
    Box::new(AlignedBuf::zeroed(bytes))
}

/// A fake VRAM: two arenas at fixed addresses, and a log of the copies it was asked for.
struct FakeVram {
    kv: Vec<u8>,
    gdn: Vec<u8>,
    d2h_calls: usize,
    h2d_calls: usize,
    h2d_bytes: u64,
}

impl FakeVram {
    fn new() -> Self {
        FakeVram {
            kv: (0..KV_BYTES).map(|i| (i % 251) as u8).collect(),
            gdn: (0..GDN_BYTES).map(|i| (i % 241) as u8 ^ 0x5A).collect(),
            d2h_calls: 0,
            h2d_calls: 0,
            h2d_bytes: 0,
        }
    }

    fn arena(&mut self, region: &Region) -> (&mut Vec<u8>, usize) {
        let addr = region.addr.as_u64();
        if addr >= GDN_BASE {
            let off = (addr - GDN_BASE) as usize;
            (&mut self.gdn, off)
        } else {
            let off = (addr - KV_BASE) as usize;
            (&mut self.kv, off)
        }
    }
}

impl ParkDevice for FakeVram {
    fn d2h(&mut self, dst: &mut [u8], region: &Region) -> Result<(), ParkError> {
        self.d2h_calls += 1;
        let len = region.len as usize;
        let (arena, off) = self.arena(region);
        dst[..len].copy_from_slice(&arena[off..off + len]);
        Ok(())
    }

    fn h2d(&mut self, region: &Region, src: &[u8]) -> Result<(), ParkError> {
        self.h2d_calls += 1;
        self.h2d_bytes += region.len;
        let len = region.len as usize;
        let (arena, off) = self.arena(region);
        arena[off..off + len].copy_from_slice(&src[..len]);
        Ok(())
    }
}

fn session_regions() -> Vec<Region> {
    vec![
        Region::new(
            RegionKind::Kv,
            DeviceAddr::NULL.offset(KV_BASE).unwrap(),
            KV_BYTES,
        ),
        Region::new(
            RegionKind::GdnState,
            DeviceAddr::NULL.offset(GDN_BASE).unwrap(),
            GDN_BYTES,
        ),
    ]
}

fn booted_ledger(pinned: u64, nvme: u64) -> Ledger {
    let mut l = Ledger::new();
    l.declare(Pool::HostPinned, pinned).unwrap();
    l.declare(Pool::Nvme, nvme).unwrap();
    l
}

#[test]
fn a_ram_round_trip_is_byte_identical_and_the_ledger_balances() {
    let mut ram =
        PinnedHostTier::new(slab((2 * SLOT_BYTES) as usize), 2, SLOT_BYTES).expect("tier");
    let mut ledger = booted_ledger(4 * SLOT_BYTES, 0);
    let mut vram = FakeVram::new();
    let (kv_before, gdn_before) = (vram.kv.clone(), vram.gdn.clone());
    let regions = session_regions();

    let ticket = ram
        .park(7, &regions, &mut vram, &mut ledger)
        .expect("park to RAM");
    assert_eq!(ticket.pool, Pool::HostPinned);
    assert_eq!(ticket.bytes, IMAGE_BYTES);
    assert_eq!(ticket.session, 7);
    assert_eq!(vram.d2h_calls, 2, "KV and GDN state leave as ONE image");
    assert_eq!(ram.len(), 1);
    assert_eq!(
        ledger.used(Pool::HostPinned),
        IMAGE_BYTES,
        "the image is billed while it is parked"
    );
    assert_eq!(
        ledger.written(Pool::HostPinned),
        IMAGE_BYTES,
        "parking is write traffic, counted"
    );

    // The session is gone from VRAM as far as the scheduler is concerned; scribble over it to
    // prove the restore is what puts the bytes back.
    vram.kv.fill(0xEE);
    vram.gdn.fill(0xEE);

    // Re-admitted into FRESH slots: the destination list is the caller's, not the parked one.
    let dst = vec![
        Region::new(
            RegionKind::Kv,
            DeviceAddr::NULL.offset(KV_BASE + 4096).unwrap(),
            KV_BYTES - 4096,
        ),
        Region::new(
            RegionKind::GdnState,
            DeviceAddr::NULL.offset(GDN_BASE).unwrap(),
            GDN_BYTES,
        ),
    ];
    // ...but only if it is the same SHAPE. A different length is refused before anything moves.
    let err = ram
        .restore(&ticket, &dst, &mut vram, &mut ledger, &mut |_| {})
        .expect_err("a destination of the wrong size must be refused");
    assert!(matches!(err, ParkError::LayoutMismatch { .. }), "{err}");
    assert_eq!(vram.h2d_calls, 0, "a refused restore copies nothing");

    let mut ready: Vec<RegionReady> = Vec::new();
    ram.restore(&ticket, &regions, &mut vram, &mut ledger, &mut |r| {
        ready.push(r)
    })
    .expect("restore from RAM");

    assert_eq!(vram.kv, kv_before, "KV came back byte-identical");
    assert_eq!(vram.gdn, gdn_before, "GDN state came back byte-identical");
    assert_eq!(ready.len(), 2);
    assert_eq!(ready[0].kind, RegionKind::Kv);
    assert_eq!(ready[1].kind, RegionKind::GdnState);
    assert_eq!(ready[1].len, GDN_BYTES);
    assert_eq!(ram.len(), 0, "the slot went back to the free list");
    assert_eq!(
        ledger.used(Pool::HostPinned),
        0,
        "a restored session holds nothing"
    );
    let stats = ram.stats();
    assert_eq!((stats.parks, stats.restores), (1, 1));
    assert_eq!(
        (stats.bytes_in, stats.bytes_out),
        (IMAGE_BYTES, IMAGE_BYTES)
    );

    // The slot is reusable, which is the whole point of a fixed pool.
    let t2 = ram.park(8, &regions, &mut vram, &mut ledger).expect("park");
    ram.discard(&t2, &mut ledger);
    assert_eq!(ram.len(), 0);
    assert_eq!(ledger.used(Pool::HostPinned), 0);
    assert_eq!(ram.stats().discards, 1);
}

#[test]
fn an_nvme_round_trip_is_byte_identical_and_counts_its_write_traffic() {
    let dir = Scratch::new("nvme");
    let staging = slab(SLOT_BYTES as usize);
    let mut nvme = NvmeTier::create(
        &dir.file("park.bin"),
        2,
        SLOT_BYTES,
        staging,
        OpenMode::Direct,
    )
    .expect("nvme tier");
    let mut ledger = booted_ledger(0, 4 * SLOT_BYTES);
    let mut vram = FakeVram::new();
    let (kv_before, gdn_before) = (vram.kv.clone(), vram.gdn.clone());
    let regions = session_regions();

    let ticket = nvme
        .park(11, &regions, &mut vram, &mut ledger)
        .expect("park to NVMe");
    assert_eq!(ticket.pool, Pool::Nvme);
    assert_eq!(ledger.used(Pool::Nvme), IMAGE_BYTES);
    // The endurance number is what the DRIVE wrote: the image padded up to 4096.
    let padded = IMAGE_BYTES.div_ceil(4096) * 4096;
    assert_eq!(ledger.written(Pool::Nvme), padded);
    assert!(nvme.path().exists());

    vram.kv.fill(0xEE);
    vram.gdn.fill(0xEE);
    let mut ready = 0usize;
    nvme.restore(&ticket, &regions, &mut vram, &mut ledger, &mut |_| {
        ready += 1
    })
    .expect("restore from NVMe");
    assert_eq!(ready, 2);
    assert_eq!(vram.kv, kv_before);
    assert_eq!(vram.gdn, gdn_before);
    assert_eq!(ledger.used(Pool::Nvme), 0);
    assert_eq!(
        ledger.written(Pool::Nvme),
        padded,
        "a restore is a read; write traffic is monotonic and unchanged"
    );
    assert_eq!(nvme.len(), 0);
}

#[test]
fn ram_fills_and_the_image_demotes_to_nvme_and_back() {
    let dir = Scratch::new("spill");
    let mut ram = PinnedHostTier::new(slab(SLOT_BYTES as usize), 1, SLOT_BYTES).expect("ram tier");
    let staging = slab(SLOT_BYTES as usize);
    let mut nvme = NvmeTier::create(
        &dir.file("park.bin"),
        4,
        SLOT_BYTES,
        staging,
        OpenMode::Direct,
    )
    .expect("nvme tier");
    let mut ledger = booted_ledger(4 * SLOT_BYTES, 8 * SLOT_BYTES);
    let mut vram = FakeVram::new();
    let (kv_before, gdn_before) = (vram.kv.clone(), vram.gdn.clone());
    let regions = session_regions();

    let first = ram.park(1, &regions, &mut vram, &mut ledger).expect("park");
    // One slot, one image: the second park is refused rather than over-committing pinned memory.
    let err = ram
        .park(2, &regions, &mut vram, &mut ledger)
        .expect_err("a full tier must refuse");
    assert!(
        matches!(
            err,
            ParkError::TierFull {
                pool: Pool::HostPinned,
                slots: 1
            }
        ),
        "{err}"
    );
    assert_eq!(
        ledger.used(Pool::HostPinned),
        IMAGE_BYTES,
        "the refused park must not leave a reservation behind"
    );

    // Under pressure the scheduler spills the idle one. No device copy is involved.
    let d2h_before = vram.d2h_calls;
    let spilled = demote(&mut ram, &mut nvme, &first, &mut ledger).expect("demote");
    assert_eq!(
        vram.d2h_calls, d2h_before,
        "demotion never touches the device"
    );
    assert_eq!(spilled.pool, Pool::Nvme);
    assert_eq!(spilled.session, first.session);
    assert_eq!(spilled.sha256, first.sha256, "the image is the same image");
    assert_eq!(ram.len(), 0);
    assert_eq!(nvme.len(), 1);
    assert_eq!(ledger.used(Pool::HostPinned), 0);
    assert_eq!(ledger.used(Pool::Nvme), IMAGE_BYTES);
    assert_eq!(ram.stats().demotions, 1);

    // And the freed slot takes the next session.
    let second = ram.park(2, &regions, &mut vram, &mut ledger).expect("park");
    ram.discard(&second, &mut ledger);

    // Promotion brings it back to RAM, verifying it on the way through.
    let back = promote(&mut nvme, &mut ram, &spilled, &mut ledger).expect("promote");
    assert_eq!(back.pool, Pool::HostPinned);
    assert_eq!(back.sha256, first.sha256);
    assert_eq!(nvme.len(), 0);
    assert_eq!(ram.len(), 1);
    assert_eq!(ledger.used(Pool::Nvme), 0);
    assert_eq!(ledger.used(Pool::HostPinned), IMAGE_BYTES);

    vram.kv.fill(0xEE);
    vram.gdn.fill(0xEE);
    ram.restore(&back, &regions, &mut vram, &mut ledger, &mut |_| {})
        .expect("restore after the round trip through NVMe");
    assert_eq!(vram.kv, kv_before, "RAM -> NVMe -> RAM -> VRAM is lossless");
    assert_eq!(vram.gdn, gdn_before);
    assert_eq!(ledger.used(Pool::HostPinned), 0);
}

#[test]
fn over_commit_is_refused_by_the_ledger_before_a_slot_is_taken() {
    let mut ram =
        PinnedHostTier::new(slab((4 * SLOT_BYTES) as usize), 4, SLOT_BYTES).expect("tier");
    // The pool is declared smaller than the tier's slots: the ledger is the authority on bytes.
    let mut ledger = booted_ledger(IMAGE_BYTES + 1024, 0);
    let mut vram = FakeVram::new();
    let regions = session_regions();

    ram.park(1, &regions, &mut vram, &mut ledger)
        .expect("first");
    let err = ram
        .park(2, &regions, &mut vram, &mut ledger)
        .expect_err("the second image does not fit the declared pool");
    match err {
        ParkError::Ledger(LedgerError::OverCommit {
            pool, want, free, ..
        }) => {
            assert_eq!(pool, Pool::HostPinned);
            assert_eq!(want, IMAGE_BYTES);
            assert_eq!(free, 1024);
        }
        other => panic!("expected an over-commit, got {other}"),
    }
    assert_eq!(ram.len(), 1, "the refused park took no slot");
    assert_eq!(vram.d2h_calls, 2, "and copied nothing");

    // An image larger than a slot is refused too, with both numbers in the message.
    let huge = vec![Region::new(
        RegionKind::Kv,
        DeviceAddr::NULL.offset(KV_BASE).unwrap(),
        SLOT_BYTES + 4096,
    )];
    assert!(matches!(
        ram.park(3, &huge, &mut vram, &mut ledger),
        Err(ParkError::ImageTooLarge { .. })
    ));
}

#[test]
fn negative_control_a_corrupt_image_is_refused_and_nothing_reaches_vram() {
    // ---- RAM: flip a byte in the slot after the park.
    let mut ram = PinnedHostTier::new(slab(SLOT_BYTES as usize), 1, SLOT_BYTES).expect("tier");
    let mut ledger = booted_ledger(4 * SLOT_BYTES, 4 * SLOT_BYTES);
    let mut vram = FakeVram::new();
    let regions = session_regions();
    let ticket = ram.park(5, &regions, &mut vram, &mut ledger).expect("park");

    // A real bit flip in the parked bytes, 100 bytes into the KV half — not a doctored ticket.
    assert!(
        ram.corrupt_for_test(&ticket, 100),
        "the fault injector must find the image it is given"
    );
    vram.kv.fill(0xEE);
    vram.gdn.fill(0xEE);
    let err = ram
        .restore(&ticket, &regions, &mut vram, &mut ledger, &mut |_| {
            panic!("a corrupt image must publish no region")
        })
        .expect_err("a corrupt image must be refused");
    match err {
        ParkError::ImageCorrupt { session, .. } => assert_eq!(session, 5),
        other => panic!("expected ImageCorrupt, got {other}"),
    }
    assert_eq!(vram.h2d_calls, 0, "not one byte reached VRAM");
    assert!(vram.kv.iter().all(|&b| b == 0xEE));
    assert_eq!(ram.stats().corrupt_refusals, 1);
    assert_eq!(
        ram.len(),
        1,
        "the image is still held, for a human to look at"
    );

    // ---- NVMe: corrupt the FILE, which is the failure this tier actually has.
    let dir = Scratch::new("corrupt");
    let path = dir.file("park.bin");
    let staging = slab(SLOT_BYTES as usize);
    let mut nvme =
        NvmeTier::create(&path, 2, SLOT_BYTES, staging, OpenMode::Direct).expect("nvme tier");
    let mut vram = FakeVram::new();
    let ticket = nvme
        .park(6, &regions, &mut vram, &mut ledger)
        .expect("park");

    {
        // A buffered writer is fine for flipping one byte; the tier still reads it with O_DIRECT.
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("reopen");
        f.seek(SeekFrom::Start(64)).unwrap();
        let mut b = [0u8; 1];
        {
            use std::io::Read;
            let mut r = std::fs::File::open(&path).unwrap();
            r.seek(SeekFrom::Start(64)).unwrap();
            r.read_exact(&mut b).unwrap();
        }
        f.write_all(&[b[0] ^ 0x80]).unwrap();
        f.sync_all().unwrap();
    }

    vram.kv.fill(0xEE);
    vram.gdn.fill(0xEE);
    let err = nvme
        .restore(&ticket, &regions, &mut vram, &mut ledger, &mut |_| {
            panic!("a corrupt image must publish no region")
        })
        .expect_err("a flipped byte on the drive must be refused");
    assert!(matches!(err, ParkError::ImageCorrupt { .. }), "{err}");
    assert_eq!(vram.h2d_calls, 0);
    assert!(vram.kv.iter().all(|&b| b == 0xEE));

    // The negative control's control: the same tier restores a clean image.
    let mut vram = FakeVram::new();
    let (kv_before, gdn_before) = (vram.kv.clone(), vram.gdn.clone());
    let clean = nvme
        .park(7, &regions, &mut vram, &mut ledger)
        .expect("park");
    vram.kv.fill(0xEE);
    vram.gdn.fill(0xEE);
    nvme.restore(&clean, &regions, &mut vram, &mut ledger, &mut |_| {})
        .expect("a clean image restores");
    assert_eq!(vram.kv, kv_before);
    assert_eq!(vram.gdn, gdn_before);
}

#[test]
fn the_tiers_refuse_geometry_o_direct_cannot_use() {
    let dir = Scratch::new("geometry");
    // A slot that is not a multiple of 4096 would make every spill fail with EINVAL; the tier says
    // so at construction instead.
    assert!(matches!(
        PinnedHostTier::new(slab(SLOT_BYTES as usize), 1, 5000),
        Err(ParkError::Config(_))
    ));
    assert!(matches!(
        PinnedHostTier::new(slab(SLOT_BYTES as usize), 0, SLOT_BYTES),
        Err(ParkError::Config(_))
    ));
    // ...and a slab too small for the slots it is asked to hold.
    assert!(matches!(
        PinnedHostTier::new(slab(SLOT_BYTES as usize), 4, SLOT_BYTES),
        Err(ParkError::Config(_))
    ));

    let staging = slab(SLOT_BYTES as usize);
    assert!(matches!(
        NvmeTier::create(&dir.file("p.bin"), 1, 5000, staging, OpenMode::Direct),
        Err(ParkError::Config(_))
    ));
}

/// How the chunked digest scales with threads, at both session sizes.
///
/// Not a gate — a measurement, and the one that decides whether overlapping the hash with the copy
/// can actually reach the copy's rate. `cargo test -p saber-mem --test park -- --ignored
/// --nocapture digest_thread_scaling`. Ignored by default because it burns every core for a few
/// seconds and its output is a table, not an assertion.
#[test]
#[ignore = "measurement, not a gate"]
fn digest_thread_scaling() {
    use saber_mem::park::image_digest;
    use std::time::Instant;

    println!(
        "cores: {} logical",
        std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(0)
    );
    for mib in [136usize, 852] {
        let bytes = mib << 20;
        // A real image's bytes, not zeroes: sha256 does not care, but the memory system does.
        let buf: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
        println!("\n{mib} MiB image");
        println!("| threads | ms | GB/s |");
        println!("|---:|---:|---:|");
        for threads in [1usize, 2, 4, 6, 8, 10, 12, 16, 20, 24] {
            // Two passes, the faster kept: the first touches cold pages.
            let mut best = f64::MAX;
            for _ in 0..2 {
                let t = Instant::now();
                let _ = image_digest(&buf, threads);
                best = best.min(t.elapsed().as_secs_f64());
            }
            println!(
                "| {threads} | {:.2} | {:.2} |",
                best * 1e3,
                bytes as f64 / best / 1e9
            );
        }
    }
}

// =================================================================================================
// Chunk-at-a-time verification (ws-k6)
// =================================================================================================

/// A session big enough to span several 4 MiB digest chunks and several 16 MiB copy chunks: the
/// small fixtures above are one chunk, which is exactly the case the pipeline collapses to.
const BIG_KV: u64 = 40 << 20;
const BIG_GDN: u64 = 12 << 20;
const BIG_IMAGE: u64 = BIG_KV + BIG_GDN;

struct BigVram {
    kv: Vec<u8>,
    gdn: Vec<u8>,
}

impl BigVram {
    fn new() -> Self {
        BigVram {
            kv: (0..BIG_KV).map(|i| (i % 251) as u8).collect(),
            gdn: (0..BIG_GDN).map(|i| ((i % 241) as u8) ^ 0x5A).collect(),
        }
    }

    fn arena(&mut self, region: &Region) -> (&mut Vec<u8>, usize) {
        let addr = region.addr.as_u64();
        if addr >= GDN_BASE {
            (&mut self.gdn, (addr - GDN_BASE) as usize)
        } else {
            (&mut self.kv, (addr - KV_BASE) as usize)
        }
    }
}

impl ParkDevice for BigVram {
    fn d2h(&mut self, dst: &mut [u8], region: &Region) -> Result<(), ParkError> {
        let len = region.len as usize;
        let (arena, off) = self.arena(region);
        dst[..len].copy_from_slice(&arena[off..off + len]);
        Ok(())
    }

    fn h2d(&mut self, region: &Region, src: &[u8]) -> Result<(), ParkError> {
        let len = region.len as usize;
        let (arena, off) = self.arena(region);
        arena[off..off + len].copy_from_slice(&src[..len]);
        Ok(())
    }
}

fn big_regions() -> Vec<Region> {
    vec![
        Region::new(
            RegionKind::Kv,
            DeviceAddr::NULL.offset(KV_BASE).unwrap(),
            BIG_KV,
        ),
        Region::new(
            RegionKind::GdnState,
            DeviceAddr::NULL.offset(GDN_BASE).unwrap(),
            BIG_GDN,
        ),
    ]
}

fn big_dst() -> Vec<Region> {
    big_regions()
}

#[test]
fn a_multi_chunk_image_round_trips_and_its_chunk_list_is_the_digest() {
    let slot = saber_mem::dio::align_up(BIG_IMAGE);
    let mut ram = PinnedHostTier::new(slab(slot as usize), 1, slot).expect("tier");
    let mut dev = BigVram::new();
    let mut ledger = booted_ledger(1 << 30, 1 << 30);

    let ticket = ram
        .park(1, &big_regions(), &mut dev, &mut ledger)
        .expect("park");

    // A 4 MiB digest chunk and a 16 MiB copy chunk: 13 of the first, 4 of the second, and a KV
    // region of 40 MiB that does NOT end on either boundary, so chunks straddle the two regions.
    assert_eq!(
        ticket.chunks.len(),
        (BIG_IMAGE as usize).div_ceil(saber_mem::park::DIGEST_CHUNK),
        "one digest per 4 MiB chunk"
    );
    assert_ne!(
        BIG_KV % saber_mem::park::COPY_CHUNK as u64,
        0,
        "the fixture must straddle"
    );

    // The list is not an independent claim: folding it has to reproduce the ticket's one value.
    assert_eq!(
        saber_mem::park::fold_chunk_digests(BIG_IMAGE as usize, &ticket.chunks),
        ticket.sha256
    );

    // ...and the whole image hashes to the same thing through the non-pipelined path.
    let mut whole = vec![0u8; BIG_IMAGE as usize];
    whole[..BIG_KV as usize].copy_from_slice(&dev.kv);
    whole[BIG_KV as usize..].copy_from_slice(&dev.gdn);
    assert_eq!(saber_mem::park::image_digest(&whole, 4), ticket.sha256);

    // Restore into a zeroed pair of arenas and compare.
    let mut back = BigVram {
        kv: vec![0; BIG_KV as usize],
        gdn: vec![0; BIG_GDN as usize],
    };
    let mut published = Vec::new();
    ram.restore(&ticket, &big_dst(), &mut back, &mut ledger, &mut |r| {
        published.push(r.kind)
    })
    .expect("restore");
    assert_eq!(published.len(), 2);
    assert_eq!(back.kv, dev.kv);
    assert_eq!(back.gdn, dev.gdn);
    assert_eq!(ledger.used(Pool::HostPinned), 0);
}

#[test]
fn a_flip_in_a_late_chunk_is_refused_and_publishes_only_the_regions_before_it() {
    // The negative control for verifying chunk by chunk. A flip in the LAST chunk is caught, and
    // what may already have been sent is exactly the regions that end before it -- never the
    // region containing the flip. `CLAUDE.md` rule 8: the clean restore below is the control's
    // control, and it must pass.
    let slot = saber_mem::dio::align_up(BIG_IMAGE);
    let mut ram = PinnedHostTier::new(slab(slot as usize), 1, slot).expect("tier");
    let mut dev = BigVram::new();
    let mut ledger = booted_ledger(1 << 30, 1 << 30);
    let ticket = ram
        .park(7, &big_regions(), &mut dev, &mut ledger)
        .expect("park");

    // The last byte of the image, which is in the GDN region and in the final digest chunk.
    assert!(ram.corrupt_for_test(&ticket, BIG_IMAGE as usize - 1));

    let mut back = BigVram {
        kv: vec![0; BIG_KV as usize],
        gdn: vec![0; BIG_GDN as usize],
    };
    let mut published = Vec::new();
    let err = ram
        .restore(&ticket, &big_dst(), &mut back, &mut ledger, &mut |r| {
            published.push(r.kind)
        })
        .expect_err("a corrupt chunk must be refused");
    assert!(matches!(err, ParkError::ImageCorrupt { .. }), "{err}");
    assert_eq!(
        published,
        vec![RegionKind::Kv],
        "the KV region ends before the flip and may be published; the GDN region contains it and \
         must not be"
    );
    assert_eq!(back.kv, dev.kv, "what was published is what was parked");
    assert_ne!(back.gdn, dev.gdn, "the region with the flip never landed");
    // The image is still held and still billed: the session stays parked.
    assert!(ledger.used(Pool::HostPinned) > 0);
    assert_eq!(ram.stats().corrupt_refusals, 1);
    ram.discard(&ticket, &mut ledger);
    assert_eq!(ledger.used(Pool::HostPinned), 0);
}

#[test]
fn a_tampered_chunk_list_does_not_let_an_image_through() {
    // The chunk digests are only trusted because they fold to the ticket's single value; a list
    // that was edited to match a corrupted image is refused by that fold.
    let slot = saber_mem::dio::align_up(BIG_IMAGE);
    let mut ram = PinnedHostTier::new(slab(slot as usize), 1, slot).expect("tier");
    let mut dev = BigVram::new();
    let mut ledger = booted_ledger(1 << 30, 1 << 30);
    let mut ticket = ram
        .park(9, &big_regions(), &mut dev, &mut ledger)
        .expect("park");

    assert!(ram.corrupt_for_test(&ticket, 8));
    // Re-hash the corrupted image and paste the honest chunk list onto the ticket, which is the
    // best an attacker with the tier's own code could do without the folded value.
    let mut whole = vec![0u8; BIG_IMAGE as usize];
    whole[..BIG_KV as usize].copy_from_slice(&dev.kv);
    whole[BIG_KV as usize..].copy_from_slice(&dev.gdn);
    whole[8] ^= 0x01;
    ticket.chunks = saber_mem::park::chunk_digests(&whole, 4);

    let mut back = BigVram {
        kv: vec![0; BIG_KV as usize],
        gdn: vec![0; BIG_GDN as usize],
    };
    let err = ram
        .restore(&ticket, &big_dst(), &mut back, &mut ledger, &mut |_| {
            panic!("a tampered ticket published a region")
        })
        .expect_err("the fold must refuse it");
    assert!(matches!(err, ParkError::ImageCorrupt { .. }), "{err}");
    assert!(back.kv.iter().all(|&b| b == 0), "nothing reached the arena");
    ram.discard(&ticket, &mut ledger);
}

// =================================================================================================
// Coalescing the copy plan (workstream K9)
// =================================================================================================

/// The page size the engine's KV arena actually strides by at prod's shapes (16 attention layers x
/// 32 KiB per block), and the size K8 measured 0.363 ms per copy for.
const PAGE: u64 = 512 * 1024;

/// `n` KV pages at consecutive arena offsets, then the GDN slot: the shape an 8k session has when
/// the block allocator hands out a clean run.
fn paged_regions(n: u64) -> Vec<Region> {
    let mut v: Vec<Region> = (0..n)
        .map(|i| {
            Region::new(
                RegionKind::Kv,
                DeviceAddr::NULL.offset(KV_BASE + i * PAGE).unwrap(),
                PAGE,
            )
        })
        .collect();
    v.push(Region::new(
        RegionKind::GdnState,
        DeviceAddr::NULL.offset(GDN_BASE).unwrap(),
        GDN_BYTES,
    ));
    v
}

/// A VRAM big enough for the paged tests, counting its calls like `FakeVram` does.
struct PagedVram {
    kv: Vec<u8>,
    gdn: Vec<u8>,
    d2h_calls: usize,
    h2d_calls: usize,
}

impl PagedVram {
    fn new(pages: u64) -> Self {
        let kv = (pages * PAGE) as usize;
        PagedVram {
            kv: (0..kv).map(|i| (i % 251) as u8).collect(),
            gdn: (0..GDN_BYTES as usize)
                .map(|i| (i % 241) as u8 ^ 0x5A)
                .collect(),
            d2h_calls: 0,
            h2d_calls: 0,
        }
    }

    fn at(&mut self, region: &Region) -> (&mut Vec<u8>, usize) {
        let addr = region.addr.as_u64();
        if addr >= GDN_BASE {
            let off = (addr - GDN_BASE) as usize;
            (&mut self.gdn, off)
        } else {
            let off = (addr - KV_BASE) as usize;
            (&mut self.kv, off)
        }
    }
}

impl ParkDevice for PagedVram {
    fn d2h(&mut self, dst: &mut [u8], region: &Region) -> Result<(), ParkError> {
        self.d2h_calls += 1;
        let len = region.len as usize;
        let (arena, off) = self.at(region);
        dst[..len].copy_from_slice(&arena[off..off + len]);
        Ok(())
    }

    fn h2d(&mut self, region: &Region, src: &[u8]) -> Result<(), ParkError> {
        self.h2d_calls += 1;
        let len = region.len as usize;
        let (arena, off) = self.at(region);
        arena[off..off + len].copy_from_slice(&src[..len]);
        Ok(())
    }
}

#[test]
fn contiguous_pages_become_one_copy_and_the_image_does_not_change() {
    const N: u64 = 32;
    let regions = paged_regions(N);
    let bytes = (N * PAGE + GDN_BYTES) as usize;

    // Unverified: the copy plan is visible directly -- one call per RUN, so two, not 33.
    let mut vram = PagedVram::new(N);
    let mut slot = vec![0u8; bytes];
    let plain = capture_into(&mut slot, &regions, &mut vram, Verification::off()).expect("capture");
    assert_eq!(
        vram.d2h_calls,
        2,
        "{N} contiguous pages plus a slot is two runs, not {}",
        N + 1
    );
    assert_eq!(
        plain.layout.len(),
        regions.len(),
        "the LAYOUT stays per page"
    );
    assert_eq!(plain.bytes, bytes as u64);

    // Verified: the remaining granularity is COPY_CHUNK, which is there on purpose -- it is what
    // overlaps the hash with the copy (K6). Still a handful of calls rather than one per page.
    let mut vram = PagedVram::new(N);
    let mut verified_slot = vec![0u8; bytes];
    let verified = capture_into(
        &mut verified_slot,
        &regions,
        &mut vram,
        Verification::default(),
    )
    .expect("capture verified");
    let groups = bytes.div_ceil(COPY_CHUNK);
    assert!(
        vram.d2h_calls <= groups + 2,
        "verified copy took {} calls for {groups} copy chunk(s) over 2 runs",
        vram.d2h_calls
    );
    assert!(
        (vram.d2h_calls as u64) < N,
        "still one call per page: {}",
        vram.d2h_calls
    );

    // The bytes and the digests are the image's, not the copy plan's.
    assert_eq!(verified_slot, slot, "the payload is identical either way");
    assert_eq!(verified.layout, plain.layout);
    assert!(verified.verified());

    // ...and the same span described as ONE region gives the same digests, which is what says the
    // coalescing is a copy-plan transform and nothing more.
    let whole = vec![
        Region::new(
            RegionKind::Kv,
            DeviceAddr::NULL.offset(KV_BASE).unwrap(),
            N * PAGE,
        ),
        Region::new(
            RegionKind::GdnState,
            DeviceAddr::NULL.offset(GDN_BASE).unwrap(),
            GDN_BYTES,
        ),
    ];
    let mut vram = PagedVram::new(N);
    let mut one_slot = vec![0u8; bytes];
    let one = capture_into(&mut one_slot, &whole, &mut vram, Verification::default())
        .expect("capture whole");
    assert_eq!(one.sha256, verified.sha256, "same bytes, same digest");
    assert_eq!(one.chunks, verified.chunks);
    assert_eq!(one_slot, verified_slot);
    assert_eq!(
        one.layout.len(),
        2,
        "and its layout is its own: two regions, so it is NOT interchangeable with the paged image"
    );

    // NEGATIVE CONTROL: the same pages in descending address order are not adjacent in image order,
    // so nothing coalesces and the image is a different image. A coalescer that ignored order would
    // pass every assertion above and fail this one.
    let mut reversed = paged_regions(N);
    let gdn = reversed.pop().expect("the slot");
    reversed.reverse();
    reversed.push(gdn);
    let mut vram = PagedVram::new(N);
    let mut rev_slot = vec![0u8; bytes];
    let rev = capture_into(&mut rev_slot, &reversed, &mut vram, Verification::default())
        .expect("capture reversed");
    assert_eq!(
        vram.d2h_calls as u64,
        N + 1,
        "descending pages must stay {} separate copies",
        N + 1
    );
    assert_ne!(
        rev.sha256, verified.sha256,
        "a different image order is a different image"
    );
    assert_ne!(rev_slot, verified_slot);
}

#[test]
fn a_fragmented_restore_is_as_many_runs_as_it_has_and_publishes_every_page() {
    const N: u64 = 32;
    let src = paged_regions(N);
    let bytes = (N * PAGE + GDN_BYTES) as usize;
    let mut vram = PagedVram::new(N);
    let mut slot = vec![0u8; bytes];
    let image = capture_into(&mut slot, &src, &mut vram, Verification::default()).expect("capture");

    // The destination: the same page COUNT and sizes, in an order that is adjacent in pairs only.
    // 16 runs of two pages, plus the slot.
    let mut dst: Vec<Region> = Vec::new();
    for i in 0..N / 2 {
        let a = KV_BASE + (N - 2 - 2 * i) * PAGE;
        dst.push(Region::new(
            RegionKind::Kv,
            DeviceAddr::NULL.offset(a).unwrap(),
            PAGE,
        ));
        dst.push(Region::new(
            RegionKind::Kv,
            DeviceAddr::NULL.offset(a + PAGE).unwrap(),
            PAGE,
        ));
    }
    dst.push(Region::new(
        RegionKind::GdnState,
        DeviceAddr::NULL.offset(GDN_BASE).unwrap(),
        GDN_BYTES,
    ));
    assert_eq!(dst.len(), src.len(), "same shape, different addresses");

    let mut ready: Vec<RegionReady> = Vec::new();
    let mut vram = PagedVram::new(N);
    publish_from(
        &slot,
        &dst,
        &mut vram,
        Verification::default(),
        &image,
        &mut |r| ready.push(r),
    )
    .expect("publish into a fragmented destination");

    // Every page is published, exactly once, in image order -- the caller binds pages, not runs.
    assert_eq!(ready.len(), dst.len(), "one ready per REGION");
    for (i, r) in ready.iter().enumerate() {
        assert_eq!(r.index, i, "published in image order");
        assert_eq!(r.kind, dst[i].kind);
        assert_eq!(r.addr, dst[i].addr);
        assert_eq!(r.len, dst[i].len);
    }
    // 16 two-page runs plus the slot: fewer calls than pages, more than the clean run's.
    assert!(
        vram.h2d_calls < dst.len(),
        "a paired destination still coalesces: {} calls for {} regions",
        vram.h2d_calls,
        dst.len()
    );
    assert!(
        vram.h2d_calls >= 17,
        "...but not below the number of runs it actually has: {}",
        vram.h2d_calls
    );

    // And the bytes landed where the destination said, not where they came from.
    let mut back = vec![0u8; bytes];
    let mut check = PagedVram::new(N);
    check.kv.copy_from_slice(&vram.kv);
    check.gdn.copy_from_slice(&vram.gdn);
    let again =
        capture_into(&mut back, &dst, &mut check, Verification::default()).expect("re-capture");
    assert_eq!(again.sha256, image.sha256, "the restored image reads back");
}
