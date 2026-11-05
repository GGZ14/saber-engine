//! The upload pipeline, driven with a fake H2D so that it runs on a host with no ROCm.
//!
//! Everything under test here is a decision the pipeline makes, not a mock of one: the fake sink
//! only remembers what it was asked to do, and the staging ring under it is the **real**
//! [`PinnedRing`], over leaked host memory (`saber-hip`'s `test-util` feature) because this box has
//! no `/opt/rocm`. What is asserted is the shape of the asking — the drive reading straight into
//! the staging slot, per-record readiness after the copy's event, the ring bounding records in
//! flight, the ledger refusing an over-commit before an arena exists, the declared spill route for
//! a record too big for the ring, and a corrupt record reaching neither.
//!
//! Fixtures go under `target/` rather than `std::env::temp_dir()`: `/tmp` is tmpfs on this box and
//! tmpfs has no `O_DIRECT`, so a test that used it would exercise a different read path from the
//! one the loader ships (`docs/LEDGER.md`, 2026-09-18).

use std::cell::RefCell;
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use saber_hip::{DeviceAddr, PinnedBuf};
use saber_mem::dio::OpenMode;
use saber_mem::ledger::{Ledger, LedgerError, Pool};
use saber_mem::pack::writer::{write_pack, PackItem};
use saber_mem::pack::{PackError, PageAlignedPool};
use saber_mem::upload::{
    CopyJob, PinnedRing, Reserved, Upload, UploadConfig, UploadError, UploadSink,
    PINNED_RESERVATION, VRAM_RESERVATION,
};

const PLAN: [u8; 32] = [0xA1; 32];
const TIER: [u8; 32] = [0x7C; 32];
const FAKE_ARENA_BASE: u64 = 0x7000_0000;

static SEQ: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let root = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target"))
            .join("saber-mem-upload");
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

/// Records of assorted sizes, including several larger than the staging slot the tests use, because
/// a record spanning slots is where the ring's join/split arithmetic lands.
fn items() -> Vec<PackItem> {
    let sizes: [usize; 6] = [4096, 40_960, 8192, 100_000, 16_384, 1];
    sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            PackItem::raw_u8(
                &format!("model.layers.{i}.weight"),
                (0..n).map(|b| (b.wrapping_mul(31) + i) as u8).collect(),
            )
        })
        .collect()
}

fn build(dir: &Scratch, name: &str) -> PathBuf {
    let path = dir.file(name);
    write_pack(&path, 1, 0, PLAN, TIER, &items()).expect("write pack");
    path
}

/// A pinned buffer over leaked, page-aligned host memory: what stands in for `hipHostMalloc` here.
fn leaked_pinned(bytes: usize) -> PinnedBuf {
    // Over-allocate and leak, then hand out the 4096-aligned interior: O_DIRECT reads land in this
    // memory, so an unaligned base would make every read fail with EINVAL — and `PinnedRing::new`
    // refuses it, which the alignment test below depends on.
    let raw: &'static mut [u8] = Vec::leak(vec![0u8; bytes + 4096]);
    let off = raw.as_ptr().align_offset(4096);
    PinnedBuf::leaked_for_test(&mut raw[off..off + bytes])
}

/// What happened, in order: what the sink was asked to do, interleaved with what the consumer was
/// told is ready. One log for both halves is what makes "published only after the copy completed"
/// an assertion rather than a belief.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Ask {
    /// An asynchronous copy out of pinned staging — the zero-copy path.
    Submit {
        dst_offset: u64,
        len: usize,
        event: usize,
        pinned: bool,
    },
    /// The declared slow path: a blocking copy out of pageable memory.
    Spill {
        dst_offset: u64,
        len: usize,
    },
    Wait {
        event: usize,
    },
    Ready {
        index: usize,
    },
}

type Log = Rc<RefCell<Vec<Ask>>>;

/// A fake H2D: no device, just a byte array standing in for the arena and a log of what it was
/// told. Events "complete" when they are waited on, which is what a single in-order copy stream
/// does.
struct FakeSink {
    events: usize,
    arena_bytes: u64,
    log: Log,
    /// Bytes as the device would hold them, so the test can compare against the pack payload.
    arena: Vec<u8>,
    /// Copies issued but not yet waited on.
    inflight: VecDeque<usize>,
    max_inflight: usize,
    /// Host addresses the sink was asked to read from, to prove they are the ring's own memory.
    sources: Vec<(usize, usize)>,
}

impl FakeSink {
    fn new(events: usize, arena_bytes: u64, log: Log) -> Self {
        FakeSink {
            events,
            arena_bytes,
            log,
            arena: vec![0u8; arena_bytes as usize],
            inflight: VecDeque::new(),
            max_inflight: 0,
            sources: Vec::new(),
        }
    }
}

impl UploadSink for FakeSink {
    fn arena_bytes(&self) -> u64 {
        self.arena_bytes
    }

    fn arena_base(&self) -> DeviceAddr {
        DeviceAddr::NULL.offset(FAKE_ARENA_BASE).unwrap()
    }

    fn event_count(&self) -> usize {
        self.events
    }

    fn submit(&mut self, copies: &[CopyJob<'_, '_>], event: usize) -> Result<(), UploadError> {
        for c in copies {
            let src = c.buffer.as_slice();
            self.sources.push((c.buffer.addr(), c.len));
            let dst = c.dst_offset as usize;
            self.arena[dst..dst + c.len].copy_from_slice(&src[..c.len]);
            self.log.borrow_mut().push(Ask::Submit {
                dst_offset: c.dst_offset,
                len: c.len,
                event,
                pinned: c.buffer.is_pinned(),
            });
        }
        self.inflight.push_back(event);
        self.max_inflight = self.max_inflight.max(self.inflight.len());
        Ok(())
    }

    fn query_event(&mut self, event: usize) -> Result<bool, UploadError> {
        // A fake copy engine that is always done: the interesting orderings are the pipeline's, and
        // "the copy has not finished yet" is exercised by `wait_event` being called at all.
        Ok(self.inflight.front() == Some(&event))
    }

    fn wait_event(&mut self, event: usize) -> Result<Option<Duration>, UploadError> {
        let front = self
            .inflight
            .pop_front()
            .expect("wait on an event with nothing in flight");
        assert_eq!(
            front, event,
            "copies must be drained in issue order; one stream completes in order"
        );
        self.log.borrow_mut().push(Ask::Wait { event });
        Ok(None)
    }

    fn copy_pageable(&mut self, dst_offset: u64, src: &[u8]) -> Result<(), UploadError> {
        let dst = dst_offset as usize;
        self.arena[dst..dst + src.len()].copy_from_slice(src);
        self.log.borrow_mut().push(Ask::Spill {
            dst_offset,
            len: src.len(),
        });
        Ok(())
    }
}

fn cfg(slots: usize, slot_bytes: usize) -> UploadConfig {
    UploadConfig {
        staging_slots: slots,
        staging_slot_bytes: slot_bytes,
        queue_depth: 2,
        chunk_bytes: 4096,
        hash_threads: 2,
        stream_budget_bytes: 1 << 20,
        mode: OpenMode::Direct,
        // These fixtures mean their ring size exactly: two of them exist to exercise backpressure
        // and the pageable spill path, which a ring sized from the pack would remove.
        ring_from_pack: false,
        ..UploadConfig::default()
    }
}

fn booted_ledger() -> Ledger {
    let mut l = Ledger::new();
    l.declare(Pool::Vram(0), 1 << 30).unwrap();
    l.declare(Pool::HostPinned, 1 << 30).unwrap();
    l
}

/// The whole rig: pack, ledger reservation, ring over leaked pinned memory, fake sink.
struct Rig {
    upload: Upload,
    reserved: Reserved,
    staging: PinnedBuf,
    log: Log,
}

impl Rig {
    fn new(path: &Path, cfg: UploadConfig) -> Result<Rig, UploadError> {
        let upload = Upload::open(path, 0, cfg)?;
        let mut ledger = booted_ledger();
        let reserved = upload.reserve(&mut ledger)?;
        let staging = leaked_pinned(reserved.staging_bytes() as usize);
        Ok(Rig {
            upload,
            reserved,
            staging,
            log: Log::default(),
        })
    }
}

#[test]
fn the_drive_reads_into_the_staging_slot_and_nothing_is_copied_on_the_host() {
    let dir = Scratch::new("direct");
    let path = build(&dir, "rank0.pack");

    // 8 slots of 64 KiB: every record in `items()` fits, so nothing may spill.
    let mut rig = Rig::new(&path, cfg(8, 64 << 10)).expect("rig");
    let plan = rig.upload.plan();
    let ring = PinnedRing::new(&mut rig.staging, &rig.reserved).expect("ring");
    let ring_lo = ring.base_addr();
    let mut sink = FakeSink::new(4, plan.arena_bytes, Rc::clone(&rig.log));

    let log = Rc::clone(&rig.log);
    let mut ready = Vec::new();
    let stats = rig
        .upload
        .run(&mut sink, &ring, |r| {
            log.borrow_mut().push(Ask::Ready { index: r.index });
            ready.push((r.index, r.name.to_string(), r.addr.as_u64(), r.len));
        })
        .expect("upload");

    // Every record took the zero-copy route.
    assert_eq!(stats.counters.records_direct, plan.records as u64);
    assert_eq!(stats.counters.records_spilled, 0);
    assert_eq!(stats.counters.bytes_direct, plan.payload_bytes);
    assert_eq!(stats.spill_copy, Duration::ZERO);
    assert_eq!(stats.direct_fraction(), 1.0);
    assert_eq!(stats.pool.pinned_records, plan.records as u64);
    assert_eq!(stats.pool.pageable_records, 0);

    // The bytes the sink was handed live inside the ring's own allocation — that is what "no host
    // copy" means, and it is checkable rather than assumed.
    let ring_hi = ring_lo + (ring.slots() * ring.slot_bytes());
    assert_eq!(sink.sources.len(), plan.records);
    for (addr, len) in &sink.sources {
        assert!(
            *addr >= ring_lo && addr + len <= ring_hi,
            "copy source {addr:#x}+{len} is not inside the staging ring"
        );
    }

    // Every record, exactly once, at the address the pack table implies.
    assert_eq!(ready.len(), plan.records);
    let reader = rig.upload.reader();
    for (index, name, addr, len) in &ready {
        let rec = &reader.records()[*index];
        assert_eq!(name, reader.name(*index));
        assert_eq!(*len, rec.length);
        assert_eq!(*addr, FAKE_ARENA_BASE + rec.data_off - plan.first_data_off);
    }

    // Published only after the copy that carries it has been waited on.
    let events = log.borrow().clone();
    let mut submitted: Vec<(u64, usize, usize)> = Vec::new();
    let mut waited: Vec<usize> = Vec::new();
    let mut seen_ready = 0usize;
    for ev in &events {
        match ev {
            Ask::Submit {
                dst_offset,
                len,
                event,
                pinned,
            } => {
                assert!(
                    pinned,
                    "a pinned ring must not hand out pageable buffers here"
                );
                submitted.push((*dst_offset, *len, *event));
            }
            Ask::Wait { event } => waited.push(*event),
            Ask::Ready { index } => {
                seen_ready += 1;
                let rec = &reader.records()[*index];
                let base = rec.data_off - plan.first_data_off;
                let (_, _, event) = submitted
                    .iter()
                    .copied()
                    .find(|(off, len, _)| *off == base && *len as u64 == rec.length)
                    .unwrap_or_else(|| panic!("record {index} published with nothing submitted"));
                assert!(
                    waited.contains(&event),
                    "record {index} published before event {event} completed"
                );
            }
            Ask::Spill { .. } => panic!("nothing should spill with a ring this size"),
        }
    }
    assert_eq!(seen_ready, plan.records);

    // And the arena holds the pack's bytes, in the right places.
    for (i, item) in items().iter().enumerate() {
        let rec = &reader.records()[i];
        let off = (rec.data_off - plan.first_data_off) as usize;
        assert_eq!(&sink.arena[off..off + item.data.len()], &item.data[..]);
    }
}

#[test]
fn the_ring_bounds_records_in_flight_and_counts_the_stall() {
    let dir = Scratch::new("backpressure");
    let path = build(&dir, "rank0.pack");

    // Two slots of 64 KiB against records up to 100 KB: the 100 KB record needs both slots, so the
    // readers have to wait for the consumer to give them back.
    let mut rig = Rig::new(&path, cfg(2, 64 << 10)).expect("rig");
    let plan = rig.upload.plan();
    let ring = PinnedRing::new(&mut rig.staging, &rig.reserved).expect("ring");
    let mut sink = FakeSink::new(4, plan.arena_bytes, Rc::clone(&rig.log));

    let mut ready = Vec::new();
    let stats = rig
        .upload
        .run(&mut sink, &ring, |r| ready.push(r.index))
        .expect("upload");

    assert_eq!(ready.len(), plan.records);
    assert_eq!(stats.counters.records_spilled, 0, "everything still fits");
    assert!(
        stats.pool.stalls > 0,
        "a 2-slot ring against a 100 KB record must have made a reader wait; \
         if this is 0 the backpressure path is not being exercised"
    );
    assert!(
        stats.pool.peak_inflight_bytes <= (2 * (64 << 10)) as u64,
        "the ring held {} bytes, more than it has",
        stats.pool.peak_inflight_bytes
    );
    assert!(
        stats.counters.idle_drains > 0,
        "with a ring this small the stream must have asked the consumer to release buffers"
    );
    let reader = rig.upload.reader();
    for (i, item) in items().iter().enumerate() {
        let rec = &reader.records()[i];
        let off = (rec.data_off - plan.first_data_off) as usize;
        assert_eq!(&sink.arena[off..off + item.data.len()], &item.data[..]);
    }
}

#[test]
fn a_record_larger_than_the_ring_takes_the_declared_slow_path() {
    let dir = Scratch::new("spill");
    let path = build(&dir, "rank0.pack");

    // One slot of 8 KiB: only records of 8192 bytes or less fit the ring; the rest must spill, and
    // must say so.
    let mut rig = Rig::new(&path, cfg(1, 8192)).expect("rig");
    let plan = rig.upload.plan();
    let ring = PinnedRing::new(&mut rig.staging, &rig.reserved).expect("ring");
    let mut sink = FakeSink::new(4, plan.arena_bytes, Rc::clone(&rig.log));

    let log = Rc::clone(&rig.log);
    let mut ready = Vec::new();
    let stats = rig
        .upload
        .run(&mut sink, &ring, |r| ready.push(r.index))
        .expect("upload");

    let big = items().iter().filter(|i| i.data.len() > 8192).count() as u64;
    let small = items().len() as u64 - big;
    assert_eq!(stats.counters.records_spilled, big);
    assert_eq!(stats.counters.records_direct, small);
    assert!(stats.direct_fraction() < 1.0 && stats.direct_fraction() > 0.0);
    assert_eq!(stats.pool.pageable_records, big);
    assert!(
        stats.spill_gbps().is_some(),
        "the fallback must report its own rate, not hide in the average"
    );

    // The spilled records went through copy_pageable, and every record still landed correctly.
    let spills = log
        .borrow()
        .iter()
        .filter(|a| matches!(a, Ask::Spill { .. }))
        .count() as u64;
    assert_eq!(spills, big);
    assert_eq!(ready.len(), plan.records);
    let reader = rig.upload.reader();
    for (i, item) in items().iter().enumerate() {
        let rec = &reader.records()[i];
        let off = (rec.data_off - plan.first_data_off) as usize;
        assert_eq!(&sink.arena[off..off + item.data.len()], &item.data[..]);
    }
}

#[test]
fn a_pageable_pool_still_works_and_reports_itself_as_the_slow_path() {
    // The workstream-K behaviour, now visible as a policy: every record spills.
    let dir = Scratch::new("pageable");
    let path = build(&dir, "rank0.pack");
    let rig = Rig::new(&path, cfg(8, 64 << 10)).expect("rig");
    let plan = rig.upload.plan();
    let pool = PageAlignedPool::new(1 << 20);
    let mut sink = FakeSink::new(4, plan.arena_bytes, Rc::clone(&rig.log));

    let mut n = 0;
    let stats = rig
        .upload
        .run(&mut sink, &pool, |_| n += 1)
        .expect("upload");
    assert_eq!(n, plan.records);
    assert_eq!(stats.counters.records_spilled, plan.records as u64);
    assert_eq!(stats.counters.records_direct, 0);
    assert_eq!(stats.direct_fraction(), 0.0);
}

#[test]
fn over_commit_is_refused_before_anything_is_allocated() {
    let dir = Scratch::new("overcommit");
    let path = build(&dir, "rank0.pack");
    let up = Upload::open(&path, 0, cfg(4, 16 << 10)).expect("open");
    let plan = up.plan();

    let mut ledger = Ledger::new();
    ledger.declare(Pool::Vram(0), plan.arena_bytes - 1).unwrap();
    ledger.declare(Pool::HostPinned, 1 << 30).unwrap();

    match up.reserve(&mut ledger) {
        Err(UploadError::Ledger(LedgerError::OverCommit { pool, want, .. })) => {
            assert_eq!(pool, Pool::Vram(0));
            assert_eq!(want, plan.arena_bytes);
        }
        other => panic!("expected a VRAM over-commit, got {other:?}"),
    }
    assert_eq!(
        ledger.used(Pool::Vram(0)),
        0,
        "a refusal must not hold bytes"
    );
    assert_eq!(ledger.used(Pool::HostPinned), 0);

    // Pinned memory is scarce too, and a failure there must not leave the VRAM half claimed.
    let mut ledger = Ledger::new();
    ledger.declare(Pool::Vram(0), 1 << 30).unwrap();
    ledger
        .declare(Pool::HostPinned, plan.staging_bytes - 1)
        .unwrap();
    assert!(matches!(
        up.reserve(&mut ledger),
        Err(UploadError::Ledger(LedgerError::OverCommit {
            pool: Pool::HostPinned,
            ..
        }))
    ));
    assert_eq!(
        ledger.used(Pool::Vram(0)),
        0,
        "the VRAM reservation must be rolled back when the pinned one fails"
    );

    // And the reservation that does fit is named, so /metrics and `explain` can see it.
    let mut ledger = booted_ledger();
    up.reserve(&mut ledger).expect("reserve");
    let snap = ledger.snapshot();
    let vram = snap.pools.iter().find(|p| p.pool == Pool::Vram(0)).unwrap();
    assert_eq!(vram.reservations[0].name, VRAM_RESERVATION);
    let pinned = snap
        .pools
        .iter()
        .find(|p| p.pool == Pool::HostPinned)
        .unwrap();
    assert_eq!(pinned.reservations[0].name, PINNED_RESERVATION);
}

#[test]
fn a_corrupt_record_reaches_neither_vram_nor_the_consumer() {
    let dir = Scratch::new("corrupt");
    let path = build(&dir, "rank0.pack");

    // Flip one byte inside the third record's payload, leaving the table untouched: the pack still
    // parses, and only the content hash can tell.
    let reader = saber_mem::pack::PackReader::open(&path).expect("open");
    let victim = 2usize;
    let victim_name = reader.name(victim).to_string();
    let off = reader.records()[victim].data_off + 17;
    drop(reader);
    {
        let mut b = [0u8; 1];
        {
            use std::io::Read;
            let mut r = std::fs::File::open(&path).unwrap();
            r.seek(SeekFrom::Start(off)).unwrap();
            r.read_exact(&mut b).unwrap();
        }
        let mut f = OpenOptions::new().write(true).open(&path).expect("reopen");
        f.seek(SeekFrom::Start(off)).unwrap();
        f.write_all(&[b[0] ^ 0x01]).unwrap();
        f.sync_all().unwrap();
    }

    let mut rig = Rig::new(&path, cfg(8, 64 << 10)).expect("rig");
    let plan = rig.upload.plan();
    let ring = PinnedRing::new(&mut rig.staging, &rig.reserved).expect("ring");
    let mut sink = FakeSink::new(4, plan.arena_bytes, Rc::clone(&rig.log));

    let mut ready: Vec<String> = Vec::new();
    let err = rig
        .upload
        .run(&mut sink, &ring, |r| ready.push(r.name.to_string()))
        .expect_err("a corrupt record must fail the load");
    match err {
        UploadError::Pack(PackError::RecordHashMismatch {
            index, ref name, ..
        }) => {
            assert_eq!(index, victim);
            assert_eq!(name, &victim_name);
        }
        other => panic!("expected RecordHashMismatch, got {other}"),
    }
    assert!(
        !ready.contains(&victim_name),
        "the corrupt record was published as ready"
    );

    // It never reached the device: the arena is untouched where that record would have gone, which
    // is the property I3 is actually about — and the check now covers the very bytes the DMA would
    // have sent, because the hash was taken from the staging slot.
    let rec = &rig.upload.reader().records()[victim];
    let off = (rec.data_off - plan.first_data_off) as usize;
    assert!(
        sink.arena[off..off + rec.length as usize]
            .iter()
            .all(|&b| b == 0),
        "corrupt record bytes were copied into the arena"
    );

    // The negative control for this negative control: the same pack, unflipped, loads clean.
    let dir2 = Scratch::new("corrupt-control");
    let clean = build(&dir2, "rank0.pack");
    let mut rig = Rig::new(&clean, cfg(8, 64 << 10)).expect("rig");
    let plan = rig.upload.plan();
    let ring = PinnedRing::new(&mut rig.staging, &rig.reserved).expect("ring");
    let mut sink = FakeSink::new(4, plan.arena_bytes, Rc::clone(&rig.log));
    let mut n = 0;
    rig.upload
        .run(&mut sink, &ring, |_| n += 1)
        .expect("clean pack loads");
    assert_eq!(n, plan.records);
}

#[test]
fn limit_bytes_takes_a_prefix_of_the_table_and_sizes_the_arena_to_it() {
    let dir = Scratch::new("limit");
    let path = build(&dir, "rank0.pack");

    let full = Upload::open(&path, 0, cfg(4, 16 << 10))
        .expect("open")
        .plan();

    // Enough for the first two records, not the third.
    let reader = saber_mem::pack::PackReader::open(&path).expect("open");
    let r = reader.records();
    let two = r[1].data_off + r[1].length - r[0].data_off;
    let three = r[2].data_off + r[2].length - r[0].data_off;
    drop(reader);

    let mut c = cfg(8, 64 << 10);
    c.limit_bytes = Some(three - 1);
    let mut rig = Rig::new(&path, c).expect("rig");
    let plan = rig.upload.plan();
    assert_eq!(plan.records, 2);
    assert_eq!(plan.records_in_pack, full.records_in_pack);
    assert!(plan.arena_bytes >= two && plan.arena_bytes < three + 4096);

    let ring = PinnedRing::new(&mut rig.staging, &rig.reserved).expect("ring");
    let mut sink = FakeSink::new(4, plan.arena_bytes, Rc::clone(&rig.log));
    let mut ready = Vec::new();
    let stats = rig
        .upload
        .run(&mut sink, &ring, |x| ready.push(x.index))
        .expect("upload");
    ready.sort_unstable();
    assert_eq!(ready, vec![0, 1]);
    assert_eq!(stats.counters.reads_issued, 2);

    // A limit below the first record is a refusal, not a zero-record success.
    let mut c = cfg(4, 16 << 10);
    c.limit_bytes = Some(1);
    assert!(matches!(
        Upload::open(&path, 0, c),
        Err(UploadError::Config(_))
    ));
}

#[test]
fn from_record_uploads_a_window_and_the_arena_mirrors_it() {
    let dir = Scratch::new("window");
    let path = build(&dir, "rank0.pack");

    let mut c = cfg(8, 64 << 10);
    c.start_record = 2;
    let mut rig = Rig::new(&path, c).expect("rig");
    let plan = rig.upload.plan();
    assert_eq!(plan.first_record, 2);
    assert_eq!(plan.records, items().len() - 2);
    assert_eq!(
        plan.payload_bytes,
        items()[2..]
            .iter()
            .map(|i| i.data.len() as u64)
            .sum::<u64>(),
        "the window's bytes, not the whole pack's"
    );

    let ring = PinnedRing::new(&mut rig.staging, &rig.reserved).expect("ring");
    let mut sink = FakeSink::new(4, plan.arena_bytes, Rc::clone(&rig.log));
    let mut ready = Vec::new();
    rig.upload
        .run(&mut sink, &ring, |r| ready.push((r.index, r.addr.as_u64())))
        .expect("upload");

    ready.sort_unstable();
    assert_eq!(
        ready.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
        vec![2, 3, 4, 5],
        "records before the window must not be read"
    );
    let reader = rig.upload.reader();
    for (index, addr) in &ready {
        let rec = &reader.records()[*index];
        assert_eq!(*addr, FAKE_ARENA_BASE + rec.data_off - plan.first_data_off);
        let off = (rec.data_off - plan.first_data_off) as usize;
        assert_eq!(
            &sink.arena[off..off + rec.length as usize],
            &items()[*index].data[..]
        );
    }

    // Past the end of the table is a refusal, not an empty success.
    let mut c = cfg(4, 16 << 10);
    c.start_record = 99;
    assert!(matches!(
        Upload::open(&path, 0, c),
        Err(UploadError::Config(_))
    ));
}

#[test]
fn the_ring_refuses_geometry_o_direct_cannot_use() {
    let dir = Scratch::new("align");
    let path = build(&dir, "rank0.pack");

    // A slot size that is not a multiple of 4096 would make every O_DIRECT read fail with EINVAL
    // at the first record; the ring says so at construction instead.
    let mut rig = Rig::new(&path, cfg(4, 5000)).expect("rig");
    match PinnedRing::new(&mut rig.staging, &rig.reserved) {
        Err(UploadError::Config(msg)) => assert!(msg.contains("4096"), "{msg}"),
        other => panic!("expected a refusal, got {other:?}"),
    }

    // A misaligned base is refused for the same reason. (The rig's own buffer is aligned, so build
    // a deliberately offset one here.)
    let raw: &'static mut [u8] = Vec::leak(vec![0u8; (4 << 20) + 4096]);
    let off = raw.as_ptr().align_offset(4096) + 1;
    let mut misaligned = PinnedBuf::leaked_for_test(&mut raw[off..off + (1 << 20)]);
    let rig = Rig::new(&path, cfg(4, 4096)).expect("rig");
    match PinnedRing::new(&mut misaligned, &rig.reserved) {
        Err(UploadError::Config(msg)) => assert!(msg.contains("aligned"), "{msg}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn the_wrong_rank_is_refused() {
    let dir = Scratch::new("rank");
    let path = build(&dir, "rank0.pack");
    match Upload::open(&path, 1, cfg(4, 16 << 10)) {
        Err(UploadError::RankMismatch { found, want, .. }) => {
            assert_eq!((found, want), (0, 1));
        }
        other => panic!("expected RankMismatch, got {other:?}"),
    }
}

/// A pack sharded for two ranks, handed to a one-rank run (ws-p9 item 4, ws-l18).
///
/// The rank check above cannot see this and never could: `rank0.pack` of a TP=2 pack IS rank 0.
/// Every record the plan binds is in it, each holding one half of its matrix, so the load, the
/// binding and every launch count downstream are green and the model answers with half its width.
/// The CONTROL is the same pack opened by a run that really is two ranks: it must still open, or
/// the refusal would be "TP=2 packs are refused" rather than "a pack must match its run".
#[test]
fn a_pack_sharded_for_another_rank_count_is_refused() {
    let dir = Scratch::new("tp");
    let path = dir.file("rank0.pack");
    write_pack(&path, 2, 0, PLAN, TIER, &items()).expect("write pack");

    let mut c = cfg(4, 16 << 10);
    c.expect_tp = Some(1);
    match Upload::open(&path, 0, c) {
        Err(UploadError::TpMismatch {
            found, want, rank, ..
        }) => assert_eq!((found, want, rank), (2, 1, 0)),
        other => panic!("expected TpMismatch, got {other:?}"),
    }

    // CONTROL: the same file, the run it was built for.
    let mut c = cfg(4, 16 << 10);
    c.expect_tp = Some(2);
    let up = Upload::open(&path, 0, c).expect("a 2-rank pack opens for a 2-rank run");
    assert_eq!(up.plan().tp_size, 2);

    // And a caller that does not state its world is not asked the question.
    let up = Upload::open(&path, 0, cfg(4, 16 << 10)).expect("no expectation, no refusal");
    assert_eq!((up.plan().rank, up.plan().tp_size), (0, 2));
}

#[test]
fn a_sink_too_small_for_the_arena_is_refused() {
    let dir = Scratch::new("mismatch");
    let path = build(&dir, "rank0.pack");
    let mut rig = Rig::new(&path, cfg(8, 64 << 10)).expect("rig");
    let plan = rig.upload.plan();
    let ring = PinnedRing::new(&mut rig.staging, &rig.reserved).expect("ring");

    let mut sink = FakeSink::new(4, plan.arena_bytes - 1, Rc::clone(&rig.log));
    assert!(matches!(
        rig.upload.run(&mut sink, &ring, |_| {}),
        Err(UploadError::Config(_))
    ));

    let mut sink = FakeSink::new(0, plan.arena_bytes, Rc::clone(&rig.log));
    assert!(matches!(
        rig.upload.run(&mut sink, &ring, |_| {}),
        Err(UploadError::Config(_))
    ));
}

// =================================================================================================
// The ring sized from the pack, and the per-rank read offset (ws-k6)
// =================================================================================================

#[test]
fn the_ring_is_sized_from_the_pack_and_says_so() {
    let dir = Scratch::new("ringsize");
    let path = build(&dir, "rank0.pack");

    // One 8 KiB slot against a 100_000-byte record: on the load path that is a ring the readers
    // would stall behind while that record is hashed, so it is raised.
    let mut asked = cfg(1, 8192);
    asked.ring_from_pack = true;
    let rig = Rig::new(&path, asked).expect("rig");
    let plan = rig.upload.plan();

    assert_eq!(
        plan.staging_raised_from,
        Some(8192),
        "the raise must be reported, not silent: it changes the pinned reservation"
    );
    assert!(
        plan.staging_bytes >= plan.max_record_bytes * saber_mem::upload::RING_HEADROOM,
        "{} of ring for a {} byte record",
        plan.staging_bytes,
        plan.max_record_bytes
    );
    assert!(
        plan.staging_bytes <= saber_mem::upload::RING_CEILING,
        "the raise is bounded"
    );

    // ...and with the policy off, the caller's size is used exactly. That is what keeps the
    // backpressure and spill tests above able to ask for a ring too small for the pack.
    let rig = Rig::new(&path, cfg(1, 8192)).expect("rig");
    assert_eq!(rig.upload.plan().staging_raised_from, None);
    assert_eq!(rig.upload.plan().staging_bytes, 8192);
}

#[test]
fn a_rotated_read_delivers_every_record_exactly_once() {
    let dir = Scratch::new("rotate");
    let path = build(&dir, "rank0.pack");

    // Rank 1 of 2 starts halfway through the window and wraps. Nothing about WHAT is uploaded may
    // change -- only the order the drive is asked for it in.
    let mut rotated = cfg(8, 64 << 10);
    rotated.read_offset_fraction = Some(0.5);
    let mut rig = Rig::new(&path, rotated).expect("rig");
    let plan = rig.upload.plan();
    let ring = PinnedRing::new(&mut rig.staging, &rig.reserved).expect("ring");
    let mut sink = FakeSink::new(4, plan.arena_bytes, Rc::clone(&rig.log));

    let mut ready = Vec::new();
    let stats = rig
        .upload
        .run(&mut sink, &ring, |r| ready.push(r.index))
        .expect("upload");

    let mut seen = ready.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(
        seen.len(),
        plan.records,
        "every record exactly once: {} delivered, {} distinct, {} in the window",
        ready.len(),
        seen.len(),
        plan.records
    );
    assert_eq!(ready.len(), plan.records, "and none twice");
    assert_eq!(stats.counters.hashes_verified, plan.records as u64);
    assert_eq!(stats.counters.records_spilled, 0);
    // The first record delivered is no longer record 0: the walk started halfway in. (Delivery is
    // in completion order, so this is a statement about the rotation, not about ordering.)
    assert!(
        ready.iter().any(|&i| i >= plan.records / 2),
        "the second half was read"
    );
}
