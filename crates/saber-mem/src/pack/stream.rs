//! The streaming pack loader: many `O_DIRECT` reads in flight, per-record readiness.
//!
//! Shape of the thing:
//!
//! ```text
//!   claim (ordered, budgeted)        hash queue                 caller's thread
//!   ┌──────────────────────┐        ┌───────────┐              ┌────────────────┐
//!   │ qd I/O worker threads│ ─────► │ hash pool │ ──ready────► │ on_record(...)  │
//!   │  preadv 1 MiB chunks │        │  sha256   │              │ buffer returned │
//!   └──────────────────────┘        └───────────┘              └────────────────┘
//!            ▲ buffers                                                 │
//!            └──────────────────── BufferPool ◄───────────────────────┘
//! ```
//!
//! * One record is read by one worker, as a sequence of `chunk_bytes` `preadv` calls. `qd` workers
//!   therefore keep `qd` reads outstanding at the drive, which is exactly the point the drive was
//!   measured at (`docs/LEDGER.md`: 1 MiB QD1 10.18 GB/s, **QD8 14.79 GB/s**).
//! * Hashing runs on its own pool, because sha256 at ~2 GB/s per core would otherwise eat the I/O
//!   threads' time and cap the stream at a third of the drive.
//! * Records are claimed strictly in table order, **and the buffer is acquired in the same critical
//!   section as the claim**. That ordering is what makes table-order delivery deadlock free: the
//!   record the consumer is waiting for always holds its buffer before any later record can take
//!   one.
//! * **Where the buffers come from is a policy** ([`super::buffers::BufferPool`]). The default
//!   [`super::buffers::PageAlignedPool`] is pageable memory pooled by exact size; the upload path
//!   passes its pinned staging ring instead, so a record is read once from the drive into the
//!   memory the H2D will send from and never copied on the host.
//! * **The callback owns the buffer while it needs it.** A pinned buffer is the source of an
//!   asynchronous H2D, so it is released when the copy's event completes, not when `on_record`
//!   returns. The callback returns `Some(buffer)` to release it immediately or `None` to keep it.
//!
//! `io_uring` is the P5 upgrade and does not change the on-disk format.

#![forbid(unsafe_code)]

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::dio::{self, OpenMode};

use super::buffers::{lock, BufferPool, PoolBuf, PoolStats};
use super::error::PackError;
use super::format::PackRecord;
use super::reader::PackReader;

/// Chunks a record needs before its hash is spread across helper threads. See `verify_buf`.
const PARALLEL_CHUNK_MIN: usize = 4;

/// Order in which completed records are handed to the callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeliveryOrder {
    /// Whatever finished first — the "records publish readiness individually" pattern, and the
    /// default: the consumer binds each record to a plan slot by name, so order buys it nothing.
    #[default]
    Completion,
    /// Table order (== ascending file offset). Deterministic, and it costs what head-of-line
    /// blocking costs: measured on the NVFP4 tier (2026-09-18, 12.2 GB pack, one 1.2 GiB
    /// `embed_tokens` record, 2 GiB budget) **10.13 GB/s against completion order's 12.39** at
    /// qd 8, and 6.33 against 10.41 with verification on. Use it when a human has to read the
    /// trace, not on the cold-start path.
    Table,
}

/// Tuning for [`PackStream`].
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Reads in flight, i.e. worker threads. 8 is the drive's knee (QD1 10.2 GB/s, QD8 the
    /// ceiling); 16 is not better and costs cores the hash pool wants.
    pub queue_depth: usize,
    /// Bytes per `preadv`. 1 MiB is the size the drive was measured with.
    pub chunk_bytes: usize,
    /// Delivery order.
    pub order: DeliveryOrder,
    /// Check each record's `sha256` before delivering it.
    pub verify: bool,
    /// Host memory a caller-built [`super::buffers::PageAlignedPool`] should hold. The stream does
    /// not read this — the pool it is given owns that decision — but it is the number a caller
    /// sizes the default pool with, so it stays here with the rest of the tuning.
    pub budget_bytes: u64,
    /// Threads hashing completed records. Ignored when `verify` is false. `sha256` runs at about
    /// 2 GB/s per core with SHA-NI, so verification — not the drive — is what bounds a verified
    /// stream: 12.39 GB/s raw against 10.41 verified at qd 8 on this box (12 physical cores).
    pub hash_threads: usize,
    /// How the file is opened.
    pub mode: OpenMode,
    /// Plan hash the pack must carry, if the caller has resolved a plan.
    pub expect_plan_hash: Option<[u8; 32]>,
    /// First table index to stream. 0 is the whole pack.
    pub start_record: usize,
    /// One past the last table index to stream. `None` runs to the end of the table.
    ///
    /// The table is ascending by `data_off`, so `[start_record, limit_records)` is a contiguous
    /// *window* of the file: a partial load reads the same bytes in the same order a full load
    /// would read them there, rather than a different access pattern that would measure something
    /// else. It bounds the *claim*, not the delivery, so backpressure and ordering are unchanged.
    pub limit_records: Option<usize>,
    /// Begin the walk this many records into the window, and **wrap**.
    ///
    /// The window is still read exactly once; only where it starts moves. It exists for one
    /// measured reason: two rank loaders reading two ~11.4 GiB files start-to-end are two readers
    /// in the same part of the drive, and they split it 2:1 and finish 0.9 s apart, after which
    /// the survivor has the drive to itself at two thirds of the rate the pair was getting
    /// (`docs/LEDGER.md`, workstream K6). Rotating rank *r* of *R* by `r * count / R` keeps the
    /// readers a fixed distance apart for the whole load.
    pub rotate_records: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        let par = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(4);
        Self {
            queue_depth: 8,
            chunk_bytes: 1 << 20,
            order: DeliveryOrder::default(),
            verify: true,
            budget_bytes: 2 << 30,
            hash_threads: par.div_ceil(2).clamp(2, 8),
            mode: OpenMode::Direct,
            expect_plan_hash: None,
            start_record: 0,
            limit_records: None,
            rotate_records: 0,
        }
    }
}

/// One verified record, with the buffer it landed in.
///
/// The buffer is **owned** by the callback for as long as it wants it: returning it releases it to
/// the pool, returning `None` keeps it. That is what lets an asynchronous H2D read a pinned buffer
/// after the callback has returned without the next read landing on top of it.
#[derive(Debug)]
pub struct RecordView<'a, 'p> {
    /// Table index.
    pub index: usize,
    /// Tensor name.
    pub name: &'a str,
    /// Table entry.
    pub record: &'a PackRecord,
    /// Exactly `record.length` bytes, in a 4096-aligned host buffer.
    pub buffer: PoolBuf<'p>,
}

impl RecordView<'_, '_> {
    /// The record's bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        self.buffer.as_slice()
    }
}

/// What the callback did with the buffer: `Some` releases it now, `None` keeps it.
pub type RecordOutcome<'p> = Option<PoolBuf<'p>>;

/// The consumer side of a stream.
///
/// A plain `FnMut` is one (there is a blanket impl below), but a consumer that *keeps* buffers —
/// the upload path keeps each one until its H2D event completes — has to implement
/// [`StreamConsumer::on_idle`] as well, and this is why:
///
/// > The stream blocks when no record is ready. If the consumer is holding buffers at that moment
/// > and the readers are waiting for exactly those buffers, nothing moves again. `on_idle` is the
/// > stream telling the consumer "I am about to block, release what you can"; a consumer that
/// > returns `false` from it is asserting it holds nothing, and that assertion is what makes the
/// > pinned pool safe to block on.
pub trait StreamConsumer<'p> {
    /// Handle one verified record.
    fn on_record(&mut self, view: RecordView<'_, 'p>) -> Result<RecordOutcome<'p>, PackError>;

    /// Called when the stream is about to block. Return `true` if something was released to the
    /// pool, and the stream will look for work again before blocking.
    fn on_idle(&mut self) -> Result<bool, PackError> {
        Ok(false)
    }
}

impl<'p, F> StreamConsumer<'p> for F
where
    F: FnMut(RecordView<'_, 'p>) -> Result<RecordOutcome<'p>, PackError>,
{
    fn on_record(&mut self, view: RecordView<'_, 'p>) -> Result<RecordOutcome<'p>, PackError> {
        self(view)
    }
}

/// One verified record, borrowed — the shape a consumer that does not keep buffers wants.
#[derive(Debug, Clone, Copy)]
pub struct RecordRef<'a> {
    /// Table index.
    pub index: usize,
    /// Tensor name.
    pub name: &'a str,
    /// Table entry.
    pub record: &'a PackRecord,
    /// Exactly `record.length` bytes.
    pub bytes: &'a [u8],
}

/// Wraps a closure that only *reads* each record: the buffer goes straight back to the pool.
///
/// The ownership-transferring form ([`StreamConsumer::on_record`]) exists for the upload path,
/// which has to hold a buffer until its DMA completes. Everything else — a verification pass, a
/// benchmark, a copy into a caller's own structure — wants this, and it keeps the closure free of
/// the pool's lifetime, which a closure cannot infer on its own.
pub struct Borrowed<F>(pub F);

impl<'p, F> StreamConsumer<'p> for Borrowed<F>
where
    F: FnMut(RecordRef<'_>) -> Result<(), PackError>,
{
    fn on_record(&mut self, view: RecordView<'_, 'p>) -> Result<RecordOutcome<'p>, PackError> {
        (self.0)(RecordRef {
            index: view.index,
            name: view.name,
            record: view.record,
            bytes: view.buffer.as_slice(),
        })?;
        Ok(Some(view.buffer))
    }
}

/// What a stream did.
#[derive(Debug, Clone, Copy)]
pub struct StreamStats {
    /// Records delivered.
    pub records: u64,
    /// Payload bytes delivered (unpadded).
    pub payload_bytes: u64,
    /// Bytes transferred from the drive (padded to 4096).
    pub transferred_bytes: u64,
    /// Wall time of the stream.
    pub elapsed: Duration,
    /// Host memory the pool was allowed to hold.
    pub budget_bytes: u64,
    /// Peak host memory actually held in flight.
    pub peak_inflight_bytes: u64,
    /// The pool's own counters, including the pinned/pageable split.
    pub pool: PoolStats,
}

impl StreamStats {
    /// Transferred GB/s (10^9 bytes per second, the unit the drive bench uses).
    #[must_use]
    pub fn gigabytes_per_second(&self) -> f64 {
        let s = self.elapsed.as_secs_f64();
        if s <= 0.0 {
            0.0
        } else {
            self.transferred_bytes as f64 / s / 1e9
        }
    }
}

/// Streaming pack loader.
#[derive(Debug)]
pub struct PackStream {
    reader: PackReader,
    cfg: StreamConfig,
}

impl PackStream {
    /// Open `path` and stream it with `qd` reads in flight and otherwise default settings.
    pub fn new(path: &Path, qd: usize) -> Result<Self, PackError> {
        Self::with_config(
            path,
            StreamConfig {
                queue_depth: qd,
                ..StreamConfig::default()
            },
        )
    }

    /// Open `path` with an explicit configuration.
    pub fn with_config(path: &Path, cfg: StreamConfig) -> Result<Self, PackError> {
        if cfg.queue_depth == 0 {
            return Err(PackError::Config("queue_depth must be >= 1".into()));
        }
        if cfg.chunk_bytes == 0 || cfg.chunk_bytes as u64 % super::format::ALIGN != 0 {
            return Err(PackError::Config(format!(
                "chunk_bytes {} must be a non-zero multiple of {}",
                cfg.chunk_bytes,
                super::format::ALIGN
            )));
        }
        if cfg.verify && cfg.hash_threads == 0 {
            return Err(PackError::Config(
                "hash_threads must be >= 1 when verify is set".into(),
            ));
        }
        let reader = match cfg.expect_plan_hash {
            Some(h) => PackReader::open_expecting(path, cfg.mode, &h)?,
            None => PackReader::open_with(path, cfg.mode)?,
        };
        Ok(Self { reader, cfg })
    }

    /// The parsed pack behind this stream.
    #[must_use]
    pub fn reader(&self) -> &PackReader {
        &self.reader
    }

    /// The configuration in force.
    #[must_use]
    pub fn config(&self) -> &StreamConfig {
        &self.cfg
    }

    /// Records this stream will deliver: the whole table, or the window
    /// `[start_record, limit_records)` names.
    #[must_use]
    pub fn record_count(&self) -> usize {
        let (start, end) = self.window();
        end - start
    }

    /// The half-open table window this stream covers.
    #[must_use]
    pub fn window(&self) -> (usize, usize) {
        let n = self.reader.records().len();
        let start = self.cfg.start_record.min(n);
        let end = self.cfg.limit_records.unwrap_or(n).clamp(start, n);
        (start, end)
    }

    /// Largest padded record in this stream's window — the floor on a blocking pool's budget.
    #[must_use]
    pub fn max_padded_in_window(&self) -> u64 {
        let (start, end) = self.window();
        self.reader.records()[start..end]
            .iter()
            .map(PackRecord::padded_length)
            .max()
            .unwrap_or(0)
    }

    /// Stream the window, handing each record to `consumer`, with buffers from `pool`.
    ///
    /// The consumer runs on the calling thread and owns each buffer until it gives it back.
    /// Returning `Err` aborts the stream and that error is returned. A closure is a consumer.
    pub fn run<'p, C>(
        &self,
        pool: &dyn BufferPool<'p>,
        consumer: &mut C,
    ) -> Result<StreamStats, PackError>
    where
        C: StreamConsumer<'p>,
    {
        let (start, end) = self.window();
        let n = end - start;
        // A pool that blocks would deadlock on a record bigger than its budget; tell it once.
        pool.accommodate(self.max_padded_in_window().max(self.cfg.chunk_bytes as u64));

        let shared = Shared {
            reader: &self.reader,
            cfg: &self.cfg,
            pool,
            start,
            end,
            rotate: if n == 0 {
                0
            } else {
                self.cfg.rotate_records % n
            },
            sched: Mutex::new(Sched { issued: 0 }),
            hashq: Mutex::new(HashQueue {
                queue: VecDeque::new(),
                io_active: self.cfg.queue_depth,
            }),
            hash_cv: Condvar::new(),
            ready: Mutex::new(Ready {
                slots: (0..end).map(|_| None).collect(),
                completed: VecDeque::new(),
            }),
            ready_cv: Condvar::new(),
            abort: AtomicBool::new(false),
            error: Mutex::new(None),
        };

        let t0 = Instant::now();
        let mut stats = StreamStats {
            records: 0,
            payload_bytes: 0,
            transferred_bytes: 0,
            elapsed: Duration::ZERO,
            budget_bytes: pool.capacity_bytes(),
            peak_inflight_bytes: 0,
            pool: PoolStats::default(),
        };

        let consumed = std::thread::scope(|scope| {
            for _ in 0..self.cfg.queue_depth {
                let s = &shared;
                scope.spawn(move || s.io_worker());
            }
            if self.cfg.verify {
                for _ in 0..self.cfg.hash_threads {
                    let s = &shared;
                    scope.spawn(move || s.hash_worker());
                }
            }
            self.consume(&shared, n, consumer, &mut stats)
        });

        stats.elapsed = t0.elapsed();
        stats.pool = pool.stats();
        stats.peak_inflight_bytes = stats.pool.peak_inflight_bytes;
        consumed.map(|()| stats)
    }

    fn consume<'p, C>(
        &self,
        shared: &Shared<'_, 'p>,
        n: usize,
        consumer: &mut C,
        stats: &mut StreamStats,
    ) -> Result<(), PackError>
    where
        C: StreamConsumer<'p>,
    {
        let mut delivered = 0usize;
        while delivered < n {
            let want = match self.cfg.order {
                DeliveryOrder::Table => Some(shared.start + delivered),
                DeliveryOrder::Completion => None,
            };
            let taken = 'take: loop {
                if let Some(x) = shared.try_take_ready(want) {
                    break 'take Some(x);
                }
                if shared.aborted() {
                    break 'take None;
                }
                // Nothing is ready and we are about to block: give the consumer the chance to
                // release the buffers it is holding first. A reader may be waiting for exactly
                // those, and a consumer asleep on a record that needs a buffer it is itself
                // holding is a deadlock, not a slow load.
                match consumer.on_idle() {
                    Ok(true) => continue 'take,
                    Ok(false) => {}
                    Err(e) => {
                        shared.fail(e);
                        break 'take None;
                    }
                }
                break 'take shared.take_ready(want);
            };
            let Some((index, buf)) = taken else {
                break; // aborted; the stored error is returned below
            };
            let rec = &self.reader.records()[index];
            let res = consumer.on_record(RecordView {
                index,
                name: self.reader.name(index),
                record: rec,
                buffer: buf,
            });
            stats.records += 1;
            stats.payload_bytes += rec.length;
            stats.transferred_bytes += rec.padded_length();
            delivered += 1;
            match res {
                // Handed back: straight into the pool, where a waiting reader is looking for it.
                Ok(Some(buf)) => shared.release(buf),
                // Kept by the caller: it releases the buffer itself once its copy has completed.
                Ok(None) => {}
                Err(e) => {
                    shared.fail(e);
                    break;
                }
            }
        }
        shared.finish();
        match shared.take_error() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

struct Sched {
    /// Records claimed so far. The index is derived from it, so a rotation is arithmetic rather
    /// than a second cursor that could disagree with the first.
    issued: usize,
}

struct HashQueue<'p> {
    queue: VecDeque<(usize, PoolBuf<'p>)>,
    io_active: usize,
}

struct Ready<'p> {
    slots: Vec<Option<PoolBuf<'p>>>,
    completed: VecDeque<usize>,
}

struct Shared<'a, 'p> {
    reader: &'a PackReader,
    cfg: &'a StreamConfig,
    pool: &'a dyn BufferPool<'p>,
    /// First and one-past-last table index this run covers.
    start: usize,
    end: usize,
    /// See [`StreamConfig::rotate_records`].
    rotate: usize,
    sched: Mutex<Sched>,
    hashq: Mutex<HashQueue<'p>>,
    hash_cv: Condvar,
    ready: Mutex<Ready<'p>>,
    ready_cv: Condvar,
    abort: AtomicBool,
    error: Mutex<Option<PackError>>,
}

impl<'p> Shared<'_, 'p> {
    fn aborted(&self) -> bool {
        self.abort.load(Ordering::Acquire)
    }

    fn fail(&self, e: PackError) {
        {
            let mut slot = lock(&self.error);
            if slot.is_none() {
                *slot = Some(e);
            }
        }
        self.abort.store(true, Ordering::Release);
        self.wake_all();
    }

    fn take_error(&self) -> Option<PackError> {
        lock(&self.error).take()
    }

    /// Stop the workers once the consumer is done (normally, or early).
    fn finish(&self) {
        self.abort.store(true, Ordering::Release);
        self.wake_all();
    }

    /// Wake everything that can be asleep, including a worker blocked inside the pool: an aborting
    /// stream that left a reader waiting on a buffer would hang the scope's join instead of
    /// reporting the error.
    fn wake_all(&self) {
        self.hash_cv.notify_all();
        self.ready_cv.notify_all();
        self.pool.wake_sleepers();
    }

    /// Claim the next record and a buffer for it, in table order.
    ///
    /// The buffer is acquired **inside** the claim's critical section on purpose: only the
    /// head-of-line record ever waits for memory, so a later record can never take the buffer an
    /// earlier one is waiting for. That is what makes table-order delivery deadlock free.
    fn claim(&self) -> Option<(usize, PoolBuf<'p>)> {
        let mut g = lock(&self.sched);
        if self.aborted() {
            return None;
        }
        let count = self.end - self.start;
        if g.issued >= count {
            return None;
        }
        // Rotated, not reordered: every index in the window is handed out exactly once, starting
        // `rotate` records in and wrapping at the end.
        let idx = self.start + (g.issued + self.rotate) % count;
        let need = self.reader.records()[idx].padded_length() as usize;
        let abort = || self.aborted();
        let mut buf = self.pool.acquire(need, &abort)?;
        buf.resize_visible(need);
        g.issued += 1;
        Some((idx, buf))
    }

    /// Give a buffer back to the pool, where a blocked reader is waiting for it.
    fn release(&self, buf: PoolBuf<'p>) {
        self.pool.release(buf);
    }

    fn io_worker(&self) {
        while let Some((idx, mut buf)) = self.claim() {
            let rec = &self.reader.records()[idx];
            let padded = rec.padded_length() as usize;
            let mut off = 0usize;
            let mut failed = None;
            while off < padded {
                let n = self.cfg.chunk_bytes.min(padded - off);
                if let Err(e) = dio::preadv_exact(
                    self.reader.file(),
                    &mut buf.as_mut_slice()[off..off + n],
                    rec.data_off + off as u64,
                ) {
                    failed = Some(PackError::io(
                        self.reader.path(),
                        format!(
                            "record {idx} ({}): read {n} bytes at {}",
                            self.reader.name(idx),
                            rec.data_off + off as u64
                        ),
                        e,
                    ));
                    break;
                }
                off += n;
            }
            if let Some(e) = failed {
                self.release(buf);
                self.fail(e);
                break;
            }
            buf.truncate(rec.length as usize);
            if self.cfg.verify {
                let mut q = lock(&self.hashq);
                q.queue.push_back((idx, buf));
                drop(q);
                self.hash_cv.notify_one();
            } else {
                self.publish(idx, buf);
            }
        }
        let mut q = lock(&self.hashq);
        q.io_active -= 1;
        drop(q);
        self.hash_cv.notify_all();
    }

    fn hash_worker(&self) {
        loop {
            let (idx, buf) = {
                let mut q = lock(&self.hashq);
                loop {
                    if let Some(item) = q.queue.pop_front() {
                        break item;
                    }
                    if q.io_active == 0 || self.aborted() {
                        return;
                    }
                    q = self
                        .hash_cv
                        .wait(q)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
            };
            // Hashed out of the buffer the drive read into — which, on the pinned path, is the
            // buffer the H2D will send from. The check therefore covers the exact bytes that reach
            // VRAM, and it happens before the copy is issued (I3).
            if let Err(e) = self.verify_buf(idx, buf.as_slice()) {
                self.release(buf);
                self.fail(e);
                return;
            }
            self.publish(idx, buf);
        }
    }

    /// Verify one record's bytes.
    ///
    /// With chunk digests the record is hashed on **many** cores instead of one, which is the whole
    /// reason they exist: `sha256` is a chain, so a 1.18 GiB record is 0.44 s of one core and no
    /// number of hash threads divides it — unless the pack says where the chunk boundaries are.
    /// Without them this is the plain whole-record hash, exactly as before.
    fn verify_buf(&self, idx: usize, bytes: &[u8]) -> Result<(), PackError> {
        let rec = &self.reader.records()[idx];
        let Some(list) = self.reader.chunk_digests(idx) else {
            let digest: [u8; 32] = Sha256::digest(bytes).into();
            if digest != rec.sha256 {
                return Err(PackError::RecordHashMismatch {
                    index: idx,
                    name: self.reader.name(idx).to_string(),
                    expected: hex::encode(rec.sha256),
                    found: hex::encode(digest),
                });
            }
            return Ok(());
        };
        let step = self.reader.chunk_bytes() as usize;
        let chunks: Vec<&[u8]> = bytes.chunks(step).collect();
        if chunks.len() != list.len() {
            // The blob was validated against the table at open, so this cannot happen without the
            // buffer being the wrong length — which is a bug here, not a corrupt pack.
            return Err(PackError::Malformed(format!(
                "record {idx}: {} bytes split into {} chunks against {} digests",
                bytes.len(),
                chunks.len(),
                list.len()
            )));
        }
        // Work comes off a shared counter rather than being sliced up front: the last chunk is
        // short, and one idle worker per short tail is exactly what the counter avoids.
        let next = AtomicUsize::new(0);
        let bad = AtomicUsize::new(usize::MAX);
        // Helpers are spawned per record, so a record has to be worth the spawns. Four chunks is
        // 16 MiB, which is ~6 ms of one core against ~30 µs of thread; below that the record is
        // hashed on the worker that already has it, and the pool's other workers keep their cores
        // for the other 1500 records. (Feeding chunk jobs into the existing pool instead of
        // spawning would remove the trade entirely; it is a bigger change to the queue and the
        // measured win did not need it.)
        let workers = if chunks.len() >= PARALLEL_CHUNK_MIN {
            self.cfg.hash_threads.min(chunks.len()).max(1)
        } else {
            1
        };
        if workers > 1 {
            std::thread::scope(|scope| {
                for _ in 0..workers {
                    let (next, bad, chunks, list) = (&next, &bad, &chunks, &list);
                    scope.spawn(move || loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        if k >= chunks.len() || bad.load(Ordering::Relaxed) != usize::MAX {
                            break;
                        }
                        let d: [u8; 32] = Sha256::digest(chunks[k]).into();
                        if d != list[k] {
                            bad.fetch_min(k, Ordering::Relaxed);
                            break;
                        }
                    });
                }
            });
        } else {
            for (k, c) in chunks.iter().enumerate() {
                let d: [u8; 32] = Sha256::digest(c).into();
                if d != list[k] {
                    bad.store(k, Ordering::Relaxed);
                    break;
                }
            }
        }
        let k = bad.load(Ordering::Relaxed);
        if k != usize::MAX {
            let lo = (k * step) as u64;
            return Err(PackError::RecordChunkMismatch {
                index: idx,
                name: self.reader.name(idx).to_string(),
                chunk: k,
                chunks: chunks.len(),
                lo,
                hi: lo + chunks[k].len() as u64,
                expected: hex::encode(list[k]),
                found: hex::encode(Sha256::digest(chunks[k])),
            });
        }
        Ok(())
    }

    fn publish(&self, idx: usize, buf: PoolBuf<'p>) {
        let mut r = lock(&self.ready);
        r.slots[idx] = Some(buf);
        r.completed.push_back(idx);
        drop(r);
        self.ready_cv.notify_all();
    }

    /// Take a delivered record if one is there, without blocking.
    fn try_take_ready(&self, want: Option<usize>) -> Option<(usize, PoolBuf<'p>)> {
        let mut r = lock(&self.ready);
        match want {
            Some(i) => r.slots[i].take().map(|buf| {
                r.completed.retain(|&j| j != i);
                (i, buf)
            }),
            None => r.completed.pop_front().map(|i| {
                let buf = r.slots[i].take().expect("completed record has no buffer");
                (i, buf)
            }),
        }
    }

    /// Wait for a delivered record: a specific index (table order) or any (completion order).
    fn take_ready(&self, want: Option<usize>) -> Option<(usize, PoolBuf<'p>)> {
        let mut r = lock(&self.ready);
        loop {
            match want {
                Some(i) => {
                    if let Some(buf) = r.slots[i].take() {
                        r.completed.retain(|&j| j != i);
                        return Some((i, buf));
                    }
                }
                None => {
                    if let Some(i) = r.completed.pop_front() {
                        let buf = r.slots[i].take().expect("completed record has no buffer");
                        return Some((i, buf));
                    }
                }
            }
            if self.aborted() {
                return None;
            }
            r = self
                .ready_cv
                .wait(r)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}
