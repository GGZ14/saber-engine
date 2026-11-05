//! Cold start: pack → pinned staging ring → VRAM, with per-record readiness.
//!
//! This is the one place in the plan where ring-buffer pipelining is the right shape
//! (`docs/ARCHITECTURE.md`, *"Transfer: overlap what is idle, never what is saturated"*): a bulk
//! transfer with **no competing compute**, where the goal is to keep the drive and the PCIe link
//! busy at the same time. In the decode path the same structure is a measured loss and is refused;
//! here it is what turns a 12 GB weight read into a few seconds.
//!
//! ```text
//!   PackStream (qd O_DIRECT readers + sha256 pool)          this module
//!   ┌─────────────────────────────────────────┐    ┌──────────────────────────────────┐
//!   │ preadv 1 MiB ─► PINNED STAGING SLOT ────┼───►│ hipMemcpyAsync straight out of   │
//!   │ sha256 verify (from that same slot)     │    │ the slot, one event per record   │
//!   │ publish in completion order             │    │ event complete → slot released   │
//!   └─────────────────────────────────────────┘    │                → RecordReady     │
//!                     ▲ slots                      └──────────────────────────────────┘
//!                     └──────────────── PinnedRing (a BufferPool) ◄────── released here
//! ```
//!
//! **The drive reads into the staging slot.** Workstream K read into pageable memory and copied
//! each record into a pinned slot on the consumer thread; that copy measured 8.3 GB/s and **66% of
//! the wall** (`docs/LEDGER.md`). Here [`PinnedRing`] *is* the pack stream's buffer pool
//! ([`crate::pack::BufferPool`]), so a record is read once, hashed where it lies, and sent to the
//! device from the same bytes. There is no host copy on the fast path at all.
//!
//! The properties that are structural rather than hoped for:
//!
//! * **Verification happens before the H2D is issued** (I3). `PackStream` hands a record over only
//!   after its `sha256` matched — and now it hashes the staging slot itself, so the check covers
//!   exactly the bytes that reach VRAM. The negative control in `tests/upload.rs` flips one byte
//!   and asserts the record is refused and never published.
//! * **The staging ring is allocated once** (I1). Slots are views into one `hipHostMalloc`
//!   ([`saber_hip::PinnedBuf::send_chunks`]); acquiring a record's buffer is bookkeeping over a
//!   free list, and when the ring is full the *reader* blocks. That is the backpressure.
//! * **A record larger than the whole ring has a declared slow path**, not a hidden one: the pool
//!   hands out pageable memory, the consumer copies it with a blocking H2D, and the
//!   `records_spilled` / `records_direct` counters say which route every record took (rule 5).
//! * **VRAM is reserved in the byte ledger before it is allocated.** [`Upload::reserve`] returns
//!   the [`Reserved`] token the sink and ring constructors demand, so "allocate first, tell the
//!   ledger later" does not typecheck.
//! * **One arena per rank.** Records land at their own 4096-aligned pack offsets, so the device
//!   layout mirrors the pack table and a record's address is `arena + data_off − first_data_off`.
//!
//! The device half is behind [`UploadSink`] so that ordering, backpressure and refusal are testable
//! on a host with no ROCm — which is this one. [`HipSink`] is the real implementation;
//! `tests/upload.rs` drives the same pipeline with a fake H2D.

#![forbid(unsafe_code)]

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use saber_hip::{
    DeviceAddr, DevicePtr, Event, EventFlags, HipThread, Job, PinnedBuf, PinnedChunk, Stream,
    StreamFlags,
};
use thiserror::Error;

use crate::dio::{AlignedBuf, OpenMode};
use crate::ledger::{Ledger, LedgerError, Pool};
use crate::pack::{
    align_up, BufferPool, DeliveryOrder, PackError, PackReader, PackStream, PinnedView, PoolBuf,
    PoolStats, RecordOutcome, RecordView, StreamConfig, StreamConsumer,
};

/// Ledger reservation name for a rank's weight arena.
pub const VRAM_RESERVATION: &str = "weights";
/// Ledger reservation name for the pinned staging ring.
pub const PINNED_RESERVATION: &str = "pack_staging";

/// Why an upload failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum UploadError {
    /// The pack itself: malformed, hash mismatch, plan-hash mismatch, I/O.
    #[error(transparent)]
    Pack(#[from] PackError),
    /// The byte ledger refused the reservation. The startup error the ledger exists for.
    #[error(transparent)]
    Ledger(#[from] LedgerError),
    /// HIP refused a call.
    #[error(transparent)]
    Hip(#[from] saber_hip::Error),
    /// The caller asked for something that cannot be done, caught before any byte moved.
    #[error("{0}")]
    Config(String),
    /// The pack is not the rank that was asked for. A rank mixup loads correct-looking weights for
    /// the wrong half of a TP model, so it is refused rather than warned about.
    #[error("pack {path} is rank {found} of {tp_size}, but rank {want} was requested")]
    RankMismatch {
        /// The file.
        path: String,
        /// Rank in the header.
        found: u32,
        /// `tp_size` in the header.
        tp_size: u32,
        /// Rank the caller asked for.
        want: u32,
    },
    /// The pack is sharded for a DIFFERENT NUMBER OF RANKS than the caller runs.
    ///
    /// Rank 0 of a 2-rank pack is rank 0, so [`UploadError::RankMismatch`] passes it: every record
    /// name the plan binds is present and every one of them is HALF a matrix. The caller then runs
    /// a model whose every sharded GEMM is missing its other half, which is not an error anywhere
    /// downstream — it is a worse answer (ws-p9 item 4, ws-l18). So the rank COUNT is checked
    /// beside the rank whenever the caller states its own.
    #[error(
        "pack {path} is sharded for {found} rank(s) but this run has {want}: rank {rank} of \
             {found} holds only its own shard of every tensor-parallel weight"
    )]
    TpMismatch {
        /// The file.
        path: String,
        /// `tp_size` in the header.
        found: u32,
        /// Ranks the caller runs.
        want: u32,
        /// The rank that was asked for.
        rank: u32,
    },
}

/// Tuning for one upload.
#[derive(Debug, Clone)]
pub struct UploadConfig {
    /// `O_DIRECT` reads in flight in the pack stream. 8 is the drive's knee.
    pub queue_depth: usize,
    /// Bytes per `preadv`.
    pub chunk_bytes: usize,
    /// Verify each record's `sha256` before it is staged. Off is a benchmark mode, never a load
    /// mode: I3 is the reason the digest is in the record.
    pub verify: bool,
    /// Threads in the hash pool.
    pub hash_threads: usize,
    /// Host memory the *pageable* spill pool may hold. The fast path does not use it at all.
    pub stream_budget_bytes: u64,
    /// Staging slots. Allocated once; the transfer path never allocates.
    pub staging_slots: usize,
    /// Bytes per staging slot. A record larger than one slot takes several **contiguous** slots;
    /// a record larger than the whole ring takes the pageable route and is counted.
    pub staging_slot_bytes: usize,
    /// First table index to upload. 0 is a full load.
    ///
    /// Exists for the same reason as `limit_bytes`, and for one more: the first 337 records of the
    /// NVFP4 tier are 4-byte scalars totalling 243 KiB, and record 337 is the 1.21 GiB
    /// `embed_tokens`. A byte limit alone therefore selects either ~0 or ≥1.2 GiB, which is not a
    /// choice this box has while prod is resident. A window starting past that record measures the
    /// pipeline on ordinary weight records.
    pub start_record: usize,
    /// Upload only the records from `start_record` that fit in this many bytes of arena.
    ///
    /// The table is ascending by `data_off`, so this is a contiguous window of the file: the same
    /// bytes in the same order a full load would read them there. It exists because prod holds both
    /// cards at 0.95 utilisation, so a full 12 GB rank cannot be resident beside it.
    pub limit_bytes: Option<u64>,
    /// Where in the record window this rank starts reading, as a **fraction** of it: rank `r` of
    /// `R` passes `r as f64 / R as f64`, and the reader walks from there and wraps.
    ///
    /// It is a property of the rank, not a tuning knob, which is why it is derived rather than
    /// asked for: two rank loaders reading two ~11.4 GiB files start-to-end are two readers in the
    /// same part of the drive. `None` keeps the old behaviour (start at the beginning).
    pub read_offset_fraction: Option<f64>,
    /// Let the staging ring be sized from the pack rather than from the caller's slot count.
    ///
    /// On by default, and on is the load mode: see [`RING_HEADROOM`] for the measurement. Off is
    /// for a caller that means its ring size exactly — the tests that exercise backpressure and
    /// the pageable spill path both want a ring too small for the pack, which is precisely what
    /// this would otherwise fix for them.
    pub ring_from_pack: bool,
    /// Copies allowed in flight at once, i.e. events. `None` means one per staging slot.
    ///
    /// It is separate from the slot count because the two bound different things: slots bound how
    /// far the *readers* may run ahead, events bound how deep the *copy queue* gets — and a deep
    /// copy queue is a tail at the end of the load, where there is no read left to overlap it.
    pub events: Option<usize>,
    /// How the pack file is opened.
    pub mode: OpenMode,
    /// Plan hash the pack must carry. A mismatch is refused before a byte is read.
    pub expect_plan_hash: Option<[u8; 32]>,
    /// Ranks the CALLER runs, when it knows. `Some(w)` refuses a pack whose header says it was
    /// sharded for a different number ([`UploadError::TpMismatch`]).
    ///
    /// `None` is "do not ask", which is right for a tool that is inspecting a pack rather than
    /// running the model in it. Every caller that is about to SERVE from the pack states its
    /// world, because the rank check alone cannot see this: rank 0 of a 2-rank pack is rank 0.
    pub expect_tp: Option<u32>,
}

impl Default for UploadConfig {
    fn default() -> Self {
        let par = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(4);
        Self {
            queue_depth: 8,
            chunk_bytes: 1 << 20,
            verify: true,
            hash_threads: par.div_ceil(2).clamp(2, 8),
            stream_budget_bytes: 2 << 30,
            staging_slots: 16,
            staging_slot_bytes: 8 << 20,
            start_record: 0,
            limit_bytes: None,
            read_offset_fraction: None,
            ring_from_pack: true,
            events: None,
            mode: OpenMode::Direct,
            expect_plan_hash: None,
            expect_tp: None,
        }
    }
}

/// What an upload is going to do, before it does any of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UploadPlan {
    /// TP rank this pack is.
    pub rank: u32,
    /// Ranks the pack was sharded for, from its header.
    pub tp_size: u32,
    /// Records that will be uploaded (the whole table, or the window `limit_bytes` allows).
    pub records: usize,
    /// First table index uploaded.
    pub first_record: usize,
    /// Records in the file.
    pub records_in_pack: usize,
    /// Payload bytes, unpadded.
    pub payload_bytes: u64,
    /// Bytes read from the drive (padded to 4096).
    pub transferred_bytes: u64,
    /// One arena per rank: `align_up(last.data_off + last.length) - first.data_off`.
    pub arena_bytes: u64,
    /// `data_off` of the first record; the arena's origin.
    pub first_data_off: u64,
    /// Pinned bytes the staging ring will hold.
    pub staging_bytes: u64,
    /// Largest padded record in the window — what decides whether anything has to spill.
    pub max_record_bytes: u64,
    /// The staging ring the caller asked for, when it was too small for the pack and was raised.
    ///
    /// `None` means the caller's own size was used. See [`RING_HEADROOM`].
    pub staging_raised_from: Option<u64>,
}

/// Proof that the ledger was told about this upload's bytes before they were allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reserved {
    plan: UploadPlan,
    slots: usize,
    slot_bytes: usize,
    events: usize,
}

impl Reserved {
    /// The plan these bytes were reserved for.
    #[must_use]
    pub fn plan(&self) -> UploadPlan {
        self.plan
    }
    /// Staging slots reserved.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.slots
    }
    /// Bytes per staging slot.
    #[must_use]
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
    /// Total pinned bytes reserved for staging.
    #[must_use]
    pub fn staging_bytes(&self) -> u64 {
        self.plan.staging_bytes
    }
    /// Copies allowed in flight at once.
    #[must_use]
    pub fn events(&self) -> usize {
        self.events
    }
}

/// One record that has landed in VRAM.
///
/// Published in **completion order**, once the event covering its copy has completed — so a
/// consumer that binds this to a plan slot is looking at bytes the device can already read.
#[derive(Debug, Clone, Copy)]
pub struct RecordReady<'a> {
    /// Table index.
    pub index: usize,
    /// Tensor name, verbatim from the checkpoint.
    pub name: &'a str,
    /// TP rank.
    pub rank: u32,
    /// Device address: `arena + data_off - first_data_off`.
    pub addr: DeviceAddr,
    /// Payload length in bytes.
    pub len: u64,
}

/// One host→device copy: `len` bytes out of `buffer` into the arena at `dst_offset`.
#[derive(Debug)]
pub struct CopyJob<'a, 'p> {
    /// Destination offset within the arena.
    pub dst_offset: u64,
    /// The buffer the record was read into. On the fast path it is pinned and the copy is a DMA
    /// straight out of it.
    pub buffer: &'a PoolBuf<'p>,
    /// Bytes to copy, from the start of the buffer.
    pub len: usize,
}

/// The device half of an upload: one arena, a queue of copies and the events that time them.
///
/// Implemented by [`HipSink`] for real hardware and by a fake in `tests/upload.rs`. Everything the
/// pipeline decides — ordering, when a buffer may be released, which route a record takes — is
/// above this trait, so the tests exercise the real decisions rather than a mock of them.
pub trait UploadSink {
    /// Bytes of arena.
    fn arena_bytes(&self) -> u64;
    /// Base device address of the arena.
    fn arena_base(&self) -> DeviceAddr;
    /// Events available. Each in-flight copy holds one.
    fn event_count(&self) -> usize;
    /// Issue `copies` and record `event` after them — in **one** round trip to the device thread.
    fn submit(&mut self, copies: &[CopyJob<'_, '_>], event: usize) -> Result<(), UploadError>;
    /// Has `event` completed? Non-blocking.
    ///
    /// The pipeline asks this before it ever waits, because a wait occupies the single HIP thread
    /// and the next copy cannot be submitted behind it.
    fn query_event(&mut self, event: usize) -> Result<bool, UploadError>;
    /// Block until `event` has completed, returning that copy's device-side time when the backend
    /// can measure it.
    fn wait_event(&mut self, event: usize) -> Result<Option<Duration>, UploadError>;
    /// The declared slow path: a record too large for the staging ring, copied out of pageable
    /// memory. Blocking, because there is no staging slot holding it that could be released later.
    fn copy_pageable(&mut self, dst_offset: u64, src: &[u8]) -> Result<(), UploadError>;
}

/// Per-phase counters. Every one of these is a number the load path can be wrong about.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UploadCounters {
    /// Records read and delivered by the pack stream.
    pub reads_issued: u64,
    /// Records whose `sha256` was checked before the copy was issued. Equals `reads_issued` unless
    /// verification was turned off, which is a benchmark mode.
    pub hashes_verified: u64,
    /// Asynchronous H2D copies issued out of pinned staging.
    pub h2d_issued: u64,
    /// Events waited on.
    pub events_waited: u64,
    /// Staging buffers returned to the pool after their copy completed.
    pub buffers_recycled: u64,
    /// Batched round trips to the HIP thread.
    pub batches: u64,
    /// Records published as ready.
    pub records_published: u64,
    /// **which_path**: records that went straight from the staging slot to VRAM, no host copy.
    pub records_direct: u64,
    /// **which_path**: records too large for the ring, copied out of pageable memory.
    pub records_spilled: u64,
    /// Bytes on the direct path.
    pub bytes_direct: u64,
    /// Bytes on the spilled path.
    pub bytes_spilled: u64,
    /// Times the stream was about to block and the consumer released a buffer first. This is the
    /// hook that makes a blocking pinned pool deadlock-free.
    pub idle_drains: u64,
    /// Completed copies reaped by a non-blocking query, i.e. without occupying the HIP thread.
    pub events_reaped: u64,
    /// Non-blocking `hipEventQuery` round trips.
    pub event_polls: u64,
}

/// What an upload did.
#[derive(Debug, Clone, Copy)]
pub struct UploadStats {
    /// The plan it ran.
    pub plan: UploadPlan,
    /// Per-phase counters.
    pub counters: UploadCounters,
    /// The buffer pool's counters, including its stalls — the backpressure, measured.
    pub pool: PoolStats,
    /// Wall time of the whole pipeline.
    pub wall: Duration,
    /// Host time spent inside `run_batch` submitting copies.
    pub submit: Duration,
    /// Host time spent blocked on an event.
    pub event_wait: Duration,
    /// Host time spent in the pageable fallback's blocking copies.
    pub spill_copy: Duration,
    /// Device-side copy time summed over the records' events, when the backend measures it.
    pub h2d_busy: Option<Duration>,
    /// Peak host bytes held by the pack stream's buffers.
    pub peak_stream_bytes: u64,
}

impl UploadStats {
    /// Bytes per second as GB/s (10^9), the unit the drive bench uses.
    fn rate(bytes: u64, d: Duration) -> f64 {
        let s = d.as_secs_f64();
        if s <= 0.0 {
            0.0
        } else {
            bytes as f64 / s / 1e9
        }
    }

    /// End-to-end GB/s: drive bytes over the whole pipeline's wall time. The number that matters.
    #[must_use]
    pub fn end_to_end_gbps(&self) -> f64 {
        Self::rate(self.plan.transferred_bytes, self.wall)
    }

    /// Drive → staging GB/s with the consumer's own time removed: what the pack stream alone would
    /// reach if the device side were free. An estimate, and labelled as one — the readers run
    /// concurrently with the consumer, so it is an upper bound on the stream's share.
    #[must_use]
    pub fn ssd_to_staging_gbps(&self) -> f64 {
        let consumer = self.submit + self.event_wait + self.spill_copy;
        Self::rate(
            self.plan.transferred_bytes,
            self.wall.saturating_sub(consumer),
        )
    }

    /// GB/s of the pageable fallback's blocking copies, when anything took that route.
    #[must_use]
    pub fn spill_gbps(&self) -> Option<f64> {
        (self.counters.bytes_spilled > 0)
            .then(|| Self::rate(self.counters.bytes_spilled, self.spill_copy))
    }

    /// Staging → VRAM GB/s over the PCIe link, from the per-record events. `None` when the backend
    /// cannot time the device side.
    #[must_use]
    pub fn h2d_gbps(&self) -> Option<f64> {
        self.h2d_busy
            .map(|d| Self::rate(self.counters.bytes_direct, d))
    }

    /// Share of the payload that never touched a host copy.
    #[must_use]
    pub fn direct_fraction(&self) -> f64 {
        let total = self.counters.bytes_direct + self.counters.bytes_spilled;
        if total == 0 {
            0.0
        } else {
            self.counters.bytes_direct as f64 / total as f64
        }
    }
}

/// One rank's cold-start upload.
#[derive(Debug)]
pub struct Upload {
    stream: PackStream,
    plan: UploadPlan,
    cfg: UploadConfig,
}

impl Upload {
    /// Open `path` for rank `rank` and work out what the upload will do.
    ///
    /// `path` is either a `rank<r>.pack` file or the pack directory holding it. Nothing is read
    /// beyond the header and the table; no device memory is touched.
    pub fn open(path: &Path, rank: u32, cfg: UploadConfig) -> Result<Upload, UploadError> {
        if cfg.staging_slots == 0 {
            return Err(UploadError::Config("staging_slots must be >= 1".into()));
        }
        if cfg.staging_slot_bytes == 0 {
            return Err(UploadError::Config(
                "staging_slot_bytes must be >= 1".into(),
            ));
        }
        let file = if path.is_dir() {
            path.join(format!("rank{rank}.pack"))
        } else {
            path.to_path_buf()
        };

        let reader = match cfg.expect_plan_hash {
            Some(h) => PackReader::open_expecting(&file, cfg.mode, &h)?,
            None => PackReader::open_with(&file, cfg.mode)?,
        };
        if reader.header().rank != rank {
            return Err(UploadError::RankMismatch {
                path: file.display().to_string(),
                found: reader.header().rank,
                tp_size: reader.header().tp_size,
                want: rank,
            });
        }
        // THE RANK COUNT, beside the rank. `rank0.pack` of a TP=2 pack passes the check above and
        // holds half of every column- and row-sharded tensor; a TP=1 run of it loads, binds every
        // record its plan names, launches every declared dispatch and answers with half a model.
        if let Some(want) = cfg.expect_tp {
            if reader.header().tp_size != want {
                return Err(UploadError::TpMismatch {
                    path: file.display().to_string(),
                    found: reader.header().tp_size,
                    want,
                    rank,
                });
            }
        }
        let mut cfg = cfg;
        let mut plan = plan_for(&reader, &cfg)?;
        // The ring is sized from the PACK, not from a flag: a ring that cannot run past the
        // largest record while that record is being hashed makes every reader wait for one core.
        // Raising it can raise the pinned reservation, and the ledger refuses loudly if that does
        // not fit — which is the intended failure, not an OOM at record 400.
        if let Some(want) = cfg
            .ring_from_pack
            .then(|| ring_for(plan.max_record_bytes, plan.staging_bytes))
            .flatten()
        {
            let asked = plan.staging_bytes;
            cfg.staging_slots = want.div_ceil(cfg.staging_slot_bytes as u64) as usize;
            cfg.stream_budget_bytes = cfg.stream_budget_bytes.max(want);
            plan = plan_for(&reader, &cfg)?;
            plan.staging_raised_from = Some(asked);
        }

        let stream = PackStream::with_config(
            &file,
            StreamConfig {
                queue_depth: cfg.queue_depth,
                chunk_bytes: cfg.chunk_bytes,
                order: DeliveryOrder::Completion,
                verify: cfg.verify,
                budget_bytes: cfg.stream_budget_bytes,
                hash_threads: cfg.hash_threads.max(1),
                mode: cfg.mode,
                expect_plan_hash: cfg.expect_plan_hash,
                start_record: plan.first_record,
                limit_records: Some(plan.first_record + plan.records),
                rotate_records: cfg
                    .read_offset_fraction
                    .map(|f| ((plan.records as f64) * f.clamp(0.0, 1.0)) as usize)
                    .unwrap_or(0),
            },
        )?;
        Ok(Upload { stream, plan, cfg })
    }

    /// What this upload will do.
    #[must_use]
    pub fn plan(&self) -> UploadPlan {
        self.plan
    }

    /// The pack behind it.
    #[must_use]
    pub fn reader(&self) -> &PackReader {
        self.stream.reader()
    }

    /// The configuration in force.
    #[must_use]
    pub fn config(&self) -> &UploadConfig {
        &self.cfg
    }

    /// Declare this upload's bytes to the ledger: the rank's arena and the pinned staging ring.
    ///
    /// Over-commit fails here, before a single allocation, with both numbers in the message. The
    /// returned token is what the sink and ring constructors demand.
    pub fn reserve(&self, ledger: &mut Ledger) -> Result<Reserved, UploadError> {
        ledger.reserve(
            Pool::Vram(self.plan.rank),
            VRAM_RESERVATION,
            self.plan.arena_bytes,
        )?;
        if let Err(e) = ledger.reserve(
            Pool::HostPinned,
            PINNED_RESERVATION,
            self.plan.staging_bytes,
        ) {
            // Leave the ledger as it was found: a failed reservation must not half-claim a pool.
            let _ = ledger.release(Pool::Vram(self.plan.rank), VRAM_RESERVATION);
            return Err(e.into());
        }
        Ok(Reserved {
            plan: self.plan,
            slots: self.cfg.staging_slots,
            slot_bytes: self.cfg.staging_slot_bytes,
            events: self
                .cfg
                .events
                .unwrap_or(self.cfg.staging_slots)
                .clamp(1, self.cfg.staging_slots.max(1)),
        })
    }

    /// Read the pack, verify it and copy it into the arena, publishing each record when its bytes
    /// are in VRAM.
    ///
    /// `pool` supplies the read buffers; pass a [`PinnedRing`] to get the zero-copy path.
    /// `on_ready` runs on this thread, in completion order. It is not given a chance to fail: a
    /// consumer that cannot bind a record has a plan problem, not a transfer problem.
    pub fn run<'p, S, F>(
        &self,
        sink: &mut S,
        pool: &dyn BufferPool<'p>,
        on_ready: F,
    ) -> Result<UploadStats, UploadError>
    where
        S: UploadSink,
        F: FnMut(RecordReady<'_>),
    {
        if sink.arena_bytes() < self.plan.arena_bytes {
            return Err(UploadError::Config(format!(
                "sink arena is {} bytes; this rank needs {}",
                sink.arena_bytes(),
                self.plan.arena_bytes
            )));
        }
        if sink.event_count() == 0 {
            return Err(UploadError::Config("sink has no events".into()));
        }

        let start = Instant::now();
        let mut pipe = Pipeline::new(self.reader(), sink, pool, on_ready, self.plan);
        let stream_stats = self.stream.run(pool, &mut pipe);
        let drain = pipe.drain_all();

        // The stashed error is the real one; the stream's is the wrapper the consumer returned.
        if let Some(e) = pipe.failure.take() {
            return Err(e);
        }
        let stream_stats = stream_stats?;
        drain?;

        let mut counters = pipe.counters;
        counters.reads_issued = stream_stats.records;
        counters.hashes_verified = if self.cfg.verify {
            stream_stats.records
        } else {
            0
        };
        Ok(UploadStats {
            plan: self.plan,
            counters,
            pool: stream_stats.pool,
            wall: start.elapsed(),
            submit: pipe.submit,
            event_wait: pipe.event_wait,
            spill_copy: pipe.spill_copy,
            h2d_busy: pipe.h2d_busy,
            peak_stream_bytes: stream_stats.peak_inflight_bytes,
        })
    }
}

/// Work out the record window, the arena span and the staging size.
/// How many times the largest record the staging ring must hold, before the ring is big enough.
///
/// **Measured, not chosen** (`docs/LEDGER.md`, workstream K6). The pack table stores one `sha256`
/// per record, so a 1.18 GiB `embed_tokens` is 0.44 s of *single-core* hashing that no number of
/// hash threads can divide — and while it hashes it occupies its own staging slots. With a ring
/// 1.7× that record the readers stall behind it and a 12 GB rank loads in 1.78-1.84 s; with a ring
/// 3.4× it, the reader stall count is **zero** and the same rank loads in 1.33 s. 6× is no better
/// than 3×, so the headroom is 3 and the ceiling below stops a pathological pack from asking for
/// the machine.
pub const RING_HEADROOM: u64 = 3;

/// The most the ring will be raised to on its own. Past this the caller states what it wants.
pub const RING_CEILING: u64 = 4 << 30;

/// The ring size this pack needs, or `None` if `have` is already enough.
fn ring_for(max_record: u64, have: u64) -> Option<u64> {
    let want = max_record.saturating_mul(RING_HEADROOM).min(RING_CEILING);
    (want > have).then_some(want)
}

fn plan_for(reader: &PackReader, cfg: &UploadConfig) -> Result<UploadPlan, UploadError> {
    let records = reader.records();
    if records.is_empty() {
        return Err(UploadError::Config("pack has no records".into()));
    }
    let start = cfg.start_record;
    if start >= records.len() {
        return Err(UploadError::Config(format!(
            "start_record {start} is past the {} records in this pack",
            records.len()
        )));
    }
    let first_data_off = records[start].data_off;
    let span = |i: usize| align_up(records[i].data_off + records[i].length) - first_data_off;

    let take = match cfg.limit_bytes {
        None => records.len() - start,
        Some(limit) => {
            let n = (start..records.len())
                .take_while(|i| span(*i) <= limit)
                .count();
            if n == 0 {
                return Err(UploadError::Config(format!(
                    "limit-bytes {limit} is below record {start}'s {} bytes ({}); \
                     a partial load still loads whole records",
                    span(start),
                    reader.name(start)
                )));
            }
            n
        }
    };

    let window = &records[start..start + take];
    let payload_bytes = window.iter().map(|r| r.length).sum();
    let transferred_bytes = window.iter().map(|r| r.padded_length()).sum();
    let max_record_bytes = window.iter().map(|r| r.padded_length()).max().unwrap_or(0);
    Ok(UploadPlan {
        rank: reader.header().rank,
        tp_size: reader.header().tp_size,
        records: take,
        first_record: start,
        records_in_pack: records.len(),
        payload_bytes,
        transferred_bytes,
        arena_bytes: span(start + take - 1),
        first_data_off,
        staging_bytes: cfg.staging_slots as u64 * cfg.staging_slot_bytes as u64,
        max_record_bytes,
        staging_raised_from: None,
    })
}

// ------------------------------------------------------------------------------------------- //
// The pinned staging ring: the pack stream's buffer pool                                        //
// ------------------------------------------------------------------------------------------- //

/// A fixed pinned staging ring that a pack stream reads **into**.
///
/// One `hipHostMalloc`, split once into `slots` chunks (I1). A record takes as many *contiguous*
/// slots as it needs — the ring is one allocation, so adjacent slots are adjacent memory — which is
/// what lets the stream hand the consumer a single contiguous buffer to DMA from. A record too big
/// for the whole ring gets pageable memory instead and is counted, because the alternative is a
/// pool that blocks forever on a request it can never serve.
///
/// Acquiring blocks while the ring is full: that is the cold-start path's backpressure, and it is
/// deadlock-free because the consumer releases everything it holds before the stream blocks
/// (`StreamConsumer::on_idle`).
#[derive(Debug)]
pub struct PinnedRing<'a> {
    base: usize,
    slot_bytes: usize,
    slots: usize,
    state: Mutex<RingState<'a>>,
    cv: Condvar,
    stalls: AtomicU64,
    pinned_records: AtomicU64,
    pinned_bytes: AtomicU64,
    spilled_records: AtomicU64,
    spilled_bytes: AtomicU64,
}

#[derive(Debug)]
struct RingState<'a> {
    /// `Some` = the slot is free and holds its chunk.
    free: Vec<Option<PinnedChunk<'a>>>,
    inflight: u64,
    peak: u64,
    spill_free: HashMap<usize, Vec<AlignedBuf>>,
    spill_allocated: u64,
}

impl<'a> PinnedRing<'a> {
    /// Build a ring over `buf`, which must be the pinned allocation the ledger was told about.
    ///
    /// Refuses a misaligned allocation or slot size rather than silently falling back to buffered
    /// I/O: `O_DIRECT` needs the address, the offset and the length all 4096-aligned, and a pool
    /// that quietly violated that would turn every read into an `EINVAL` at the first record.
    pub fn new(buf: &'a mut PinnedBuf, reserved: &Reserved) -> Result<PinnedRing<'a>, UploadError> {
        let slot_bytes = reserved.slot_bytes();
        let slots = reserved.slots();
        if slot_bytes % crate::dio::DIO_ALIGN != 0 {
            return Err(UploadError::Config(format!(
                "staging slot of {slot_bytes} bytes is not a multiple of {}; O_DIRECT reads \
                 straight into these slots",
                crate::dio::DIO_ALIGN
            )));
        }
        if buf.addr() % crate::dio::DIO_ALIGN != 0 {
            return Err(UploadError::Config(format!(
                "hipHostMalloc returned {:#x}, which is not {}-aligned; O_DIRECT cannot read into it",
                buf.addr(),
                crate::dio::DIO_ALIGN
            )));
        }
        if buf.len() < slots * slot_bytes {
            return Err(UploadError::Config(format!(
                "staging buffer is {} bytes; the ring needs {slots} x {slot_bytes}",
                buf.len()
            )));
        }
        let base = buf.addr();
        let chunks = buf
            .send_chunks(slot_bytes)
            .ok_or_else(|| UploadError::Config("staging buffer cannot be split".into()))?;
        let free: Vec<Option<PinnedChunk<'a>>> = chunks.into_iter().take(slots).map(Some).collect();
        Ok(PinnedRing {
            base,
            slot_bytes,
            slots: free.len(),
            state: Mutex::new(RingState {
                free,
                inflight: 0,
                peak: 0,
                spill_free: HashMap::new(),
                spill_allocated: 0,
            }),
            cv: Condvar::new(),
            stalls: AtomicU64::new(0),
            pinned_records: AtomicU64::new(0),
            pinned_bytes: AtomicU64::new(0),
            spilled_records: AtomicU64::new(0),
            spilled_bytes: AtomicU64::new(0),
        })
    }

    /// Host address of the ring's first slot, i.e. the pinned allocation's base.
    ///
    /// A caller can check that a buffer it was handed really is the ring's own memory — which is
    /// how the tests assert "nothing was copied on the host" rather than trusting a counter.
    #[must_use]
    pub fn base_addr(&self) -> usize {
        self.base
    }

    /// Slots in the ring.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Bytes per slot.
    #[must_use]
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Largest record the ring itself can hold.
    #[must_use]
    pub fn max_pinned_bytes(&self) -> u64 {
        (self.slots * self.slot_bytes) as u64
    }

    /// First index of a free run of `k` slots.
    fn find_run(free: &[Option<PinnedChunk<'a>>], k: usize) -> Option<usize> {
        let mut run = 0usize;
        for (i, slot) in free.iter().enumerate() {
            if slot.is_some() {
                run += 1;
                if run == k {
                    return Some(i + 1 - k);
                }
            } else {
                run = 0;
            }
        }
        None
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl<'a> BufferPool<'a> for PinnedRing<'a> {
    fn acquire(&self, padded: usize, abort: &(dyn Fn() -> bool + Sync)) -> Option<PoolBuf<'a>> {
        let k = padded.div_ceil(self.slot_bytes).max(1);
        let mut g = lock(&self.state);
        if k > self.slots {
            // Bigger than the whole ring. Pageable memory and a blocking copy, counted: this is
            // the declared slow route, not a silent degradation.
            let buf = g
                .spill_free
                .get_mut(&padded)
                .and_then(Vec::pop)
                .unwrap_or_else(|| {
                    g.spill_allocated += padded as u64;
                    AlignedBuf::zeroed(padded)
                });
            g.inflight += padded as u64;
            g.peak = g.peak.max(g.inflight);
            drop(g);
            self.spilled_records.fetch_add(1, Ordering::Relaxed);
            self.spilled_bytes
                .fetch_add(padded as u64, Ordering::Relaxed);
            return Some(PoolBuf::Pageable(buf));
        }
        loop {
            if abort() {
                return None;
            }
            if let Some(first) = Self::find_run(&g.free, k) {
                let mut chunk = g.free[first].take().expect("run starts on a free slot");
                for slot in g.free.iter_mut().skip(first + 1).take(k - 1) {
                    let next = slot.take().expect("run is free");
                    chunk = chunk
                        .join(next)
                        .unwrap_or_else(|_| panic!("ring slots {first}.. are not adjacent"));
                }
                g.inflight += (k * self.slot_bytes) as u64;
                g.peak = g.peak.max(g.inflight);
                drop(g);
                self.pinned_records.fetch_add(1, Ordering::Relaxed);
                self.pinned_bytes
                    .fetch_add(padded as u64, Ordering::Relaxed);
                return Some(PoolBuf::Pinned(PinnedView::new(chunk, first, k)));
            }
            self.stalls.fetch_add(1, Ordering::Relaxed);
            g = self
                .cv
                .wait(g)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    fn release(&self, buf: PoolBuf<'a>) {
        let mut g = lock(&self.state);
        match buf {
            PoolBuf::Pageable(b) => {
                let size = b.capacity();
                g.inflight -= size as u64;
                g.spill_free.entry(size).or_default().push(b);
            }
            PoolBuf::Pinned(view) => {
                let (first, k) = (view.slot(), view.slots());
                let mut chunk = view.into_chunk();
                g.inflight -= (k * self.slot_bytes) as u64;
                for i in 0..k - 1 {
                    let (head, tail) = chunk
                        .split_at(self.slot_bytes)
                        .expect("a k-slot chunk splits into k slots");
                    g.free[first + i] = Some(head);
                    chunk = tail;
                }
                g.free[first + k - 1] = Some(chunk);
            }
        }
        drop(g);
        self.cv.notify_all();
    }

    fn capacity_bytes(&self) -> u64 {
        self.max_pinned_bytes()
    }

    fn wake_sleepers(&self) {
        self.cv.notify_all();
    }

    fn stats(&self) -> PoolStats {
        let g = lock(&self.state);
        PoolStats {
            pinned_records: self.pinned_records.load(Ordering::Relaxed),
            pinned_bytes: self.pinned_bytes.load(Ordering::Relaxed),
            pageable_records: self.spilled_records.load(Ordering::Relaxed),
            pageable_bytes: self.spilled_bytes.load(Ordering::Relaxed),
            stalls: self.stalls.load(Ordering::Relaxed),
            peak_inflight_bytes: g.peak,
            allocated_bytes: self.max_pinned_bytes() + g.spill_allocated,
        }
    }
}

// ------------------------------------------------------------------------------------------- //
// The consumer                                                                                  //
// ------------------------------------------------------------------------------------------- //

/// One record's copy in flight: the event that marks its end and the buffer it is reading.
#[derive(Debug)]
struct InFlight<'p> {
    event: usize,
    bufs: Vec<PoolBuf<'p>>,
    ready: Vec<(usize, DeviceAddr, u64)>,
}

/// The in-flight queue, the counters and the timers.
struct Pipeline<'a, 'p, S: UploadSink, F: FnMut(RecordReady<'_>)> {
    reader: &'a PackReader,
    sink: &'a mut S,
    pool: &'a dyn BufferPool<'p>,
    on_ready: F,
    plan: UploadPlan,
    free_events: Vec<usize>,
    inflight: VecDeque<InFlight<'p>>,
    counters: UploadCounters,
    submit: Duration,
    event_wait: Duration,
    spill_copy: Duration,
    h2d_busy: Option<Duration>,
    failure: Option<UploadError>,
}

impl<'a, 'p, S: UploadSink, F: FnMut(RecordReady<'_>)> Pipeline<'a, 'p, S, F> {
    fn new(
        reader: &'a PackReader,
        sink: &'a mut S,
        pool: &'a dyn BufferPool<'p>,
        on_ready: F,
        plan: UploadPlan,
    ) -> Self {
        let events = (0..sink.event_count()).collect();
        Pipeline {
            reader,
            sink,
            pool,
            on_ready,
            plan,
            free_events: events,
            inflight: VecDeque::new(),
            counters: UploadCounters::default(),
            submit: Duration::ZERO,
            event_wait: Duration::ZERO,
            spill_copy: Duration::ZERO,
            h2d_busy: None,
            failure: None,
        }
    }

    /// Issue one record's copy, or take the declared slow route for a record that spilled.
    fn handle(&mut self, view: RecordView<'_, 'p>) -> Result<RecordOutcome<'p>, UploadError> {
        let index = view.index;
        let len = view.record.length;
        let base = view.record.data_off - self.plan.first_data_off;
        let addr = self
            .sink
            .arena_base()
            .offset(base)
            .ok_or_else(|| UploadError::Config(format!("record {index} address overflows")))?;

        if view.buffer.is_pinned() {
            if self.free_events.is_empty() {
                self.drain_one()?;
            }
            let event = self
                .free_events
                .pop()
                .expect("a drained copy frees its event");
            let t = Instant::now();
            self.sink.submit(
                &[CopyJob {
                    dst_offset: base,
                    buffer: &view.buffer,
                    len: len as usize,
                }],
                event,
            )?;
            self.submit += t.elapsed();
            self.counters.batches += 1;
            self.counters.h2d_issued += 1;
            self.counters.records_direct += 1;
            self.counters.bytes_direct += len;
            self.inflight.push_back(InFlight {
                event,
                bufs: vec![view.buffer],
                ready: vec![(index, addr, len)],
            });
            // Kept: the DMA is still reading this buffer, so the pool must not hand it out again
            // until the event says otherwise.
            return Ok(None);
        }

        // Spilled: the record is larger than the whole ring, so it was read into pageable memory
        // and there is no staging slot to DMA out of. One blocking copy, counted.
        let t = Instant::now();
        self.sink.copy_pageable(base, view.bytes())?;
        self.spill_copy += t.elapsed();
        self.counters.records_spilled += 1;
        self.counters.bytes_spilled += len;
        self.publish((index, addr, len));
        Ok(Some(view.buffer))
    }

    /// Reap every copy at the head of the queue that has already finished, without blocking.
    ///
    /// Returns how many were released. This is the cheap half of `on_idle`: copies run faster than
    /// the drive fills the ring, so most of the time the oldest one is already done and the slot
    /// can go back to a waiting reader for the price of one `hipEventQuery`.
    fn reap_completed(&mut self) -> Result<usize, UploadError> {
        let mut n = 0usize;
        while let Some(front) = self.inflight.front() {
            let event = front.event;
            self.counters.event_polls += 1;
            if !self.sink.query_event(event)? {
                break;
            }
            self.drain_one()?;
            self.counters.events_reaped += 1;
            n += 1;
        }
        Ok(n)
    }

    /// Wait for the oldest copy: release its buffers and publish its records.
    ///
    /// Copies are issued in order on one stream, so the oldest is always the first to complete —
    /// which is why a queue and one `hipEventSynchronize` is the whole scheduler here.
    fn drain_one(&mut self) -> Result<(), UploadError> {
        let Some(item) = self.inflight.pop_front() else {
            return Ok(());
        };
        let t = Instant::now();
        let busy = self.sink.wait_event(item.event)?;
        self.event_wait += t.elapsed();
        self.counters.events_waited += 1;
        if let Some(b) = busy {
            *self.h2d_busy.get_or_insert(Duration::ZERO) += b;
        }
        self.counters.buffers_recycled += item.bufs.len() as u64;
        for buf in item.bufs {
            self.pool.release(buf);
        }
        self.free_events.push(item.event);
        for r in item.ready {
            self.publish(r);
        }
        Ok(())
    }

    /// Drain everything. Called once, after the stream ends — including after it ended badly.
    ///
    /// Draining continues past a failure on purpose: a copy still in flight is reading a staging
    /// slot, and returning early would hand the caller back a ring it is about to drop, freeing
    /// that memory under a live DMA.
    fn drain_all(&mut self) -> Result<(), UploadError> {
        let mut first_err = None;
        while !self.inflight.is_empty() {
            if let Err(e) = self.drain_one() {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn publish(&mut self, (index, addr, len): (usize, DeviceAddr, u64)) {
        self.counters.records_published += 1;
        (self.on_ready)(RecordReady {
            index,
            name: self.reader.name(index),
            rank: self.plan.rank,
            addr,
            len,
        });
    }

    /// Keep the real error and hand the stream the fact that the consumer stopped.
    fn stash(&mut self, e: UploadError) -> PackError {
        let msg = e.to_string();
        if self.failure.is_none() {
            self.failure = Some(e);
        }
        PackError::Config(format!("upload aborted: {msg}"))
    }
}

impl<'p, S: UploadSink, F: FnMut(RecordReady<'_>)> StreamConsumer<'p> for Pipeline<'_, 'p, S, F> {
    fn on_record(&mut self, view: RecordView<'_, 'p>) -> Result<RecordOutcome<'p>, PackError> {
        self.handle(view).map_err(|e| self.stash(e))
    }

    fn on_idle(&mut self) -> Result<bool, PackError> {
        // 1. Release what is already done. No HIP thread is held, so the next submit is not queued
        //    behind a spinning `hipEventSynchronize`.
        let reaped = self.reap_completed().map_err(|e| self.stash(e))?;
        if reaped > 0 {
            self.counters.idle_drains += 1;
            return Ok(true);
        }
        if self.inflight.is_empty() {
            // 2. Holding nothing: the stream may block safely, because every staging slot is
            //    either free or being read by a worker.
            return Ok(false);
        }
        // 3. Copies outstanding but none finished, and no record ready: the pipeline is genuinely
        //    idle, so waiting costs nothing that could have been overlapped — and it is what makes
        //    the ring safe to block on, because the consumer always ends up holding nothing.
        self.counters.idle_drains += 1;
        self.drain_one().map_err(|e| self.stash(e))?;
        Ok(true)
    }
}

// ------------------------------------------------------------------------------------------- //
// The real device half                                                                          //
// ------------------------------------------------------------------------------------------- //

/// The real device half: one arena, a dedicated upload stream and a pool of timing events.
///
/// Every allocation happens in [`HipSink::new`] — one `hipMalloc` and `2 × events` events — and
/// nothing is allocated again while bytes move (I1). The staging memory is **not** here any more:
/// it belongs to [`PinnedRing`], because the drive reads into it.
pub struct HipSink<'h> {
    hip: &'h HipThread,
    stream: Stream,
    /// FN55c: shared, so a lazy owner (`--fn-vision lazy`) can hand the sink memory it lends.
    arena: std::rc::Rc<DevicePtr>,
    /// Recorded before a record's copy, so its device time can be read back.
    start_events: Vec<Event>,
    /// Recorded after it; this is the event a buffer's release waits on.
    end_events: Vec<Event>,
}

impl<'h> HipSink<'h> {
    /// Allocate the arena and the events for a [`Reserved`] upload.
    ///
    /// Taking the token rather than the sizes is the point: the bytes are in the ledger before HIP
    /// is asked for them.
    pub fn new(hip: &'h HipThread, reserved: &Reserved) -> Result<HipSink<'h>, UploadError> {
        let arena = std::rc::Rc::new(hip.malloc(reserved.plan().arena_bytes)?);
        HipSink::over(hip, reserved, arena)
    }

    /// FN55c: [`Self::new`] into `arena`, memory the caller already holds (the lazy tower's
    /// region, lent to the expert tier while no tower is resident). Refused when it is smaller
    /// than the plan's arena.
    pub fn over(
        hip: &'h HipThread,
        reserved: &Reserved,
        arena: std::rc::Rc<DevicePtr>,
    ) -> Result<HipSink<'h>, UploadError> {
        let plan = reserved.plan();
        if arena.len() < plan.arena_bytes {
            return Err(UploadError::Config(format!(
                "the sink's arena is {} B, the upload needs {} B",
                arena.len(),
                plan.arena_bytes
            )));
        }
        let stream = hip.create_stream(StreamFlags::NonBlocking)?;
        let mut start_events = Vec::with_capacity(reserved.events());
        let mut end_events = Vec::with_capacity(reserved.events());
        for _ in 0..reserved.events() {
            // Timing enabled: the per-record elapsed time is the only honest measurement of what
            // the link did, as opposed to what the host waited.
            start_events.push(hip.create_event(EventFlags::Default)?);
            end_events.push(hip.create_event(EventFlags::Default)?);
        }
        Ok(HipSink {
            hip,
            stream,
            arena,
            start_events,
            end_events,
        })
    }

    /// The arena, for a caller that wants to read the weights back.
    #[must_use]
    pub fn arena(&self) -> &DevicePtr {
        self.arena.as_ref()
    }

    /// The upload stream, for a consumer that wants to wait on it device-side.
    #[must_use]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }
}

impl UploadSink for HipSink<'_> {
    fn arena_bytes(&self) -> u64 {
        self.arena.len()
    }

    fn arena_base(&self) -> DeviceAddr {
        self.arena.addr()
    }

    fn event_count(&self) -> usize {
        self.end_events.len()
    }

    fn submit(&mut self, copies: &[CopyJob<'_, '_>], event: usize) -> Result<(), UploadError> {
        let start = self
            .start_events
            .get(event)
            .ok_or_else(|| UploadError::Config(format!("no event {event}")))?;
        let end = &self.end_events[event];
        let mut jobs: Vec<Job<'_>> = Vec::with_capacity(copies.len() + 2);
        jobs.push(Job::RecordEvent {
            event: start,
            stream: &self.stream,
        });
        for c in copies {
            let src = c.buffer.pinned_head(c.len).ok_or_else(|| {
                UploadError::Config(
                    "an H2D was submitted from a pageable buffer; that route is copy_pageable"
                        .into(),
                )
            })?;
            jobs.push(Job::MemcpyH2d {
                dst: self.arena.as_ref(),
                dst_offset: c.dst_offset,
                src,
                stream: &self.stream,
            });
        }
        jobs.push(Job::RecordEvent {
            event: end,
            stream: &self.stream,
        });
        // One round trip for the whole submission.
        self.hip.run_batch(&jobs)?;
        Ok(())
    }

    fn query_event(&mut self, event: usize) -> Result<bool, UploadError> {
        let end = self
            .end_events
            .get(event)
            .ok_or_else(|| UploadError::Config(format!("no event {event}")))?;
        Ok(self.hip.query_event(end)?)
    }

    fn wait_event(&mut self, event: usize) -> Result<Option<Duration>, UploadError> {
        let end = self
            .end_events
            .get(event)
            .ok_or_else(|| UploadError::Config(format!("no event {event}")))?;
        self.hip.sync_event(end)?;
        let ms = self.hip.elapsed_ms(&self.start_events[event], end)?;
        Ok(Some(Duration::from_secs_f64(f64::from(ms) / 1e3)))
    }

    fn copy_pageable(&mut self, dst_offset: u64, src: &[u8]) -> Result<(), UploadError> {
        self.hip
            .memcpy_h2d_at_blocking(self.arena.as_ref(), dst_offset, src, &self.stream)?;
        Ok(())
    }
}

impl Drop for HipSink<'_> {
    fn drop(&mut self) {
        // Drain before anything is freed. Rust drops the fields after this body, so without the
        // sync an aborted upload could free the arena while a copy is still reading a staging slot
        // — a use-after-free that would surface as a wrong weight, not a crash. A failure here is
        // logged, never panicked: this runs on the teardown path.
        if let Err(e) = self.hip.sync_stream(&self.stream) {
            tracing::error!(error = %e, "upload stream did not drain before the sink was dropped");
        }
    }
}

impl std::fmt::Debug for HipSink<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HipSink(arena {} B, {} events)",
            self.arena.len(),
            self.end_events.len()
        )
    }
}
