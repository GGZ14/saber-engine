//! The PLE n-gram row tier: the SSD tier the user named (docs/EXEC.md §10 the PLE law, §14 the
//! memory-plane law).
//!
//! The table (320,001,536 rows × 160, 102.4 GB bf16) is read IN PLACE from the checkpoint's own
//! shards through the pack's `ple/ple_index.json`: row `r` → shard `r / rows_per_shard`, local
//! `r mod rows_per_shard`, byte `shard.offset + local · row_bytes`; ONE `O_DIRECT` read of the
//! 512-aligned span covering it (512 B, or 1,024 B when it straddles a 512 B block). 16 rows per
//! token.
//!
//! ```text
//!   token ids -> (FN6's host hash) -> 16 row ids / token
//!        │
//!        ▼  dedupe within the chunk
//!   RAM row cache (LRU, `cache_rows` × row_bytes) ──hit──┐
//!        │ miss                                          ▼
//!   reader pool (qd workers, one 512/1024 B O_DIRECT read per row) ─► the staged row buffer
//!                                                     [T][16][160] bf16 = 5,120 B/token (pinned)
//!                                                     → one H2D → the card's PLE input
//! ```
//!
//! The staged buffer's layout is the contract FN6's PLE kernel reads (EXEC §14): token-major, the
//! 16 heads in hash order (8 bigram heads then 8 trigram heads, `ple_ids[t][h]`), each row the
//! table's 160 elements in checkpoint order. The `qwen_fp8` source (Qwen's first-party fp8 table,
//! 160 B rows + ONE per-table bf16 scale) is plumbed as a GATED source kind: constructing it
//! requires [`PleGate::ApprovedByUser`], which no default passes.
//!
//! Which-path counters per gather: rows requested, distinct, cache hits, NVMe reads (and how many
//! straddled), bytes read; [`PleTier::evict_all`] is the forced-miss control (the next gather must
//! read every distinct row from the drive).

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::JoinHandle;
use std::time::Instant;

use thiserror::Error;

use crate::dio::{self, AlignedBuf, OpenMode};

/// The drive's logical block: the alignment of a row read.
pub const PLE_BLOCK: u64 = 512;

/// Why the PLE tier refused.
#[derive(Debug, Error)]
pub enum PleError {
    /// The index JSON is not what the pack law says.
    #[error("ple index: {0}")]
    Index(String),
    /// The gated fp8 source without the gate.
    #[error(
        "the qwen_fp8 PLE source is a gated numerics choice (F6's gates and the user's word); \
         pass PleGate::ApprovedByUser only when both are recorded"
    )]
    Gated,
    /// A row id outside the table.
    #[error("row {row} outside the table's {rows} rows")]
    Row {
        /// The row.
        row: u64,
        /// Rows in the table.
        rows: u64,
    },
    /// I/O.
    #[error("{0}")]
    Io(String),
    /// A staging buffer of the wrong size.
    #[error("staging: {0}")]
    Staging(String),
}

/// Which source the rows come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PleSourceKind {
    /// The checkpoint's own bf16 shards in place (DEFAULT): 320 B rows.
    CkptBf16,
    /// Qwen's first-party fp8 table (GATED): 160 B rows + one per-table bf16 scale.
    QwenFp8,
}

/// The gate a gated source must pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PleGate {
    /// The default: only ungated sources.
    Default,
    /// The fp8 table's quality gate passed AND the user said yes (recorded in the LEDGER).
    ApprovedByUser,
}

/// One shard of the chosen source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleShard {
    /// Index into [`PleIndex::files`].
    pub file: u32,
    /// Byte offset of the shard's row 0 in that file.
    pub offset: u64,
}

/// The parsed `ple/ple_index.json` for one source kind.
#[derive(Debug, Clone, PartialEq)]
pub struct PleIndex {
    /// Rows in the (padded) table.
    pub rows: u64,
    /// Rows per shard.
    pub rows_per_shard: u64,
    /// Elements per row (160).
    pub row_dim: u32,
    /// Rows per token (16).
    pub rows_per_token: u32,
    /// The source kind.
    pub kind: PleSourceKind,
    /// Bytes per row (320 bf16 / 160 fp8).
    pub row_bytes: u64,
    /// The files the shards live in.
    pub files: Vec<PathBuf>,
    /// Shard `k`.
    pub shards: Vec<PleShard>,
    /// The fp8 table's one scale (1.0 for bf16).
    pub table_scale: f32,
}

impl PleIndex {
    /// Parse the index for `kind`. `checkpoint_root` replaces the index's absolute checkpoint
    /// directory (the container mounts it elsewhere); `None` keeps it.
    pub fn from_json(
        text: &str,
        kind: PleSourceKind,
        gate: PleGate,
        checkpoint_root: Option<&Path>,
    ) -> Result<PleIndex, PleError> {
        if kind == PleSourceKind::QwenFp8 && gate != PleGate::ApprovedByUser {
            return Err(PleError::Gated);
        }
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| PleError::Index(e.to_string()))?;
        let bad = |w: String| PleError::Index(w);
        let t = &v["table"];
        let num = |k: &str| {
            t[k].as_u64()
                .ok_or_else(|| PleError::Index(format!("table.{k} missing")))
        };
        let rows = num("rows")?;
        let rows_per_shard = num("rows_per_shard")?;
        let nshards = num("shards")?;
        let row_dim = num("row_dim")? as u32;
        let rows_per_token = num("rows_per_token")? as u32;
        if rows_per_shard * nshards != rows {
            return Err(bad(format!(
                "{nshards} shards x {rows_per_shard} rows != {rows}"
            )));
        }
        let key = match kind {
            PleSourceKind::CkptBf16 => "ckpt_bf16",
            PleSourceKind::QwenFp8 => "qwen_fp8",
        };
        let s = &v["sources"][key];
        let row_bytes = s["row_bytes"]
            .as_u64()
            .ok_or_else(|| bad(format!("sources.{key}.row_bytes missing")))?;
        let want = match kind {
            PleSourceKind::CkptBf16 => u64::from(row_dim) * 2,
            PleSourceKind::QwenFp8 => u64::from(row_dim),
        };
        if row_bytes != want {
            return Err(bad(format!(
                "{key} rows are {row_bytes} B, the law says {want}"
            )));
        }
        let root_of = |p: &str| -> PathBuf {
            let p = PathBuf::from(p);
            match checkpoint_root {
                Some(r) => r.join(p.file_name().unwrap_or_default()),
                None => p,
            }
        };
        let arr = s["shards"]
            .as_array()
            .ok_or_else(|| bad(format!("sources.{key}.shards missing")))?;
        if arr.len() as u64 != nshards {
            return Err(bad(format!(
                "{} shards listed, {nshards} declared",
                arr.len()
            )));
        }
        let mut files: Vec<PathBuf> = Vec::new();
        let mut shards = Vec::with_capacity(arr.len());
        let ckpt_dir = s["checkpoint"].as_str().unwrap_or("");
        for (k, sh) in arr.iter().enumerate() {
            if sh["shard"].as_u64() != Some(k as u64) {
                return Err(bad(format!("shard {k} out of order")));
            }
            let path = match kind {
                PleSourceKind::CkptBf16 => {
                    let f = sh["file"]
                        .as_str()
                        .ok_or_else(|| bad(format!("shard {k}: no file")))?;
                    root_of(&format!("{ckpt_dir}/{f}"))
                }
                PleSourceKind::QwenFp8 => root_of(
                    s["file"]
                        .as_str()
                        .ok_or_else(|| bad("qwen_fp8: no file".into()))?,
                ),
            };
            let fi = match files.iter().position(|f| *f == path) {
                Some(i) => i,
                None => {
                    files.push(path);
                    files.len() - 1
                }
            };
            let offset = sh["offset"]
                .as_u64()
                .ok_or_else(|| bad(format!("shard {k}: no offset")))?;
            shards.push(PleShard {
                file: fi as u32,
                offset,
            });
        }
        let table_scale = match kind {
            PleSourceKind::CkptBf16 => 1.0,
            PleSourceKind::QwenFp8 => s["table_scale"]["value"]
                .as_f64()
                .ok_or_else(|| bad("qwen_fp8.table_scale.value missing".into()))?
                as f32,
        };
        Ok(PleIndex {
            rows,
            rows_per_shard,
            row_dim,
            rows_per_token,
            kind,
            row_bytes,
            files,
            shards,
            table_scale,
        })
    }

    /// The read of row `r`: (file, 512-aligned offset, span length, the row's offset inside it).
    pub fn span(&self, r: u64) -> Result<(u32, u64, u32, u32), PleError> {
        if r >= self.rows {
            return Err(PleError::Row {
                row: r,
                rows: self.rows,
            });
        }
        let sh = &self.shards[(r / self.rows_per_shard) as usize];
        let byte = sh.offset + (r % self.rows_per_shard) * self.row_bytes;
        let start = byte / PLE_BLOCK * PLE_BLOCK;
        let len = (byte + self.row_bytes - start).div_ceil(PLE_BLOCK) * PLE_BLOCK;
        Ok((sh.file, start, len as u32, (byte - start) as u32))
    }

    /// Staged bytes per token.
    #[must_use]
    pub fn bytes_per_token(&self) -> u64 {
        self.row_bytes * u64::from(self.rows_per_token)
    }
}

/// One read job: the batch's submission instant (FN121's per-read timings are relative to it) and
/// its reads (position in the batch, file, offset, span, inner offset).
type ReadJob = (Instant, Vec<(u32, u32, u64, u32, u32)>);
/// A job's result: the positions, their rows, and each read's timing (FN121).
type ReadResult = Result<(Vec<u32>, Vec<u8>, Vec<ReadTime>), String>;

/// FN121: one drive read's timing inside its batch -- µs from the batch's submission to the
/// read's start (the queueing in the pool), and the read's own duration.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ReadTime {
    /// µs from the batch's submission to this read's `preadv`.
    pub start_us: f32,
    /// The read's duration, µs.
    pub dur_us: f32,
}

/// FN121: one batch of drive reads (what kind of gather issued it, its submission in unix µs,
/// the process's drive idle before it, every read's timing, the time the CALLER waited for it).
#[derive(Debug, Clone, Default)]
pub struct ReadBatch {
    /// `gather` (a synchronous gather) or `prefetch` (FN121's asynchronous candidate rows).
    pub kind: &'static str,
    /// The submission, unix µs (the process clock [`unix_us`]).
    pub submit_unix_us: u64,
    /// µs since this process's last drive read ended before the submission (its PLE readers +
    /// any reader that calls [`note_io_end`], e.g. the keep-awake); `u64::MAX` = none yet.
    pub idle_us: u64,
    /// Every read's timing, in the batch's order.
    pub reads: Vec<ReadTime>,
    /// µs the caller blocked on the batch (a gather: the whole read; a prefetch: the settle's
    /// residual wait).
    pub wait_us: f32,
}

/// FN121: the process clock -- an `Instant` as unix µs (one (Instant, SystemTime) pair a process,
/// so two processes' stamps compare to the two clocks' read jitter).
#[must_use]
pub fn unix_us(t: Instant) -> u64 {
    static BASE: std::sync::OnceLock<(Instant, u64)> = std::sync::OnceLock::new();
    let (i0, u0) = *BASE.get_or_init(|| {
        let u = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_micros() as u64);
        (Instant::now(), u)
    });
    if t >= i0 {
        u0 + (t - i0).as_micros() as u64
    } else {
        u0.saturating_sub((i0 - t).as_micros() as u64)
    }
}

/// FN121: the end of this process's last drive read (unix µs; 0 = none yet).
static LAST_IO_END_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// FN121: record a drive read's end (the PLE readers call it; so may any other reader of the same
/// drive in this process, e.g. FN90a's keep-awake).
pub fn note_io_end(t: Instant) {
    LAST_IO_END_US.fetch_max(unix_us(t), std::sync::atomic::Ordering::Relaxed);
}

/// FN121: the end of this process's last drive read, unix µs (0 = none yet).
#[must_use]
pub fn last_io_end_us() -> u64 {
    LAST_IO_END_US.load(std::sync::atomic::Ordering::Relaxed)
}

/// FN121: µs from this process's last drive read's end to `t` (`u64::MAX` = none yet).
fn idle_before(t: Instant) -> u64 {
    match last_io_end_us() {
        0 => u64::MAX,
        e => unix_us(t).saturating_sub(e),
    }
}

/// FN121: a batch handed to the pool and not yet collected ([`RowReaderPool::submit`]).
#[derive(Debug)]
pub struct Pending {
    sent: usize,
    n: usize,
    submit: Instant,
}

/// A persistent pool of reader threads, each with its own descriptors: `qd` workers = `qd` reads
/// in flight at the drive. A batch is split into `qd` contiguous sub-batches, one message each
/// way per worker, so the per-row cost is one `preadv` and a `row_bytes` copy.
#[derive(Debug)]
pub struct RowReaderPool {
    tx: Vec<Sender<ReadJob>>,
    rx: Receiver<ReadResult>,
    handles: Vec<JoinHandle<()>>,
    row_bytes: usize,
}

/// FN55c: read `buf` from `offset` until it is full or the file ends; an end before `need` bytes
/// is the loud short read (`dio::preadv_exact`'s), an end past it is a row at the file's tail.
fn read_covering(file: &File, buf: &mut [u8], offset: u64, need: usize) -> std::io::Result<()> {
    let total = buf.len();
    let mut done = 0usize;
    while done < total {
        let n = dio::preadv_once(file, &mut buf[done..], offset + done as u64)?;
        if n == 0 {
            break;
        }
        done += n;
    }
    if done < need.min(total) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("short read: {done} of {total} bytes at offset {offset} (row needs {need})"),
        ));
    }
    Ok(())
}

impl RowReaderPool {
    /// Spawn `qd` workers over `files`, each opening every file in `mode`.
    pub fn new(
        files: &[PathBuf],
        mode: OpenMode,
        qd: usize,
        row_bytes: usize,
    ) -> Result<RowReaderPool, PleError> {
        let (rtx, rx) = channel::<ReadResult>();
        let mut tx = Vec::new();
        let mut handles = Vec::new();
        for _ in 0..qd.max(1) {
            let fds: Vec<File> = files
                .iter()
                .map(|p| {
                    dio::open_read(p, mode)
                        .map_err(|e| PleError::Io(format!("{}: {e}", p.display())))
                })
                .collect::<Result<_, _>>()?;
            let (jtx, jrx) = channel::<ReadJob>();
            let rtx = rtx.clone();
            handles.push(std::thread::spawn(move || {
                let mut scratch = AlignedBuf::zeroed(4096);
                while let Ok((submit, job)) = jrx.recv() {
                    let mut pos = Vec::with_capacity(job.len());
                    let mut out = Vec::with_capacity(job.len() * row_bytes);
                    let mut tim = Vec::with_capacity(job.len());
                    let mut err = None;
                    for (p, f, off, len, inner) in job {
                        let buf = &mut scratch[..len as usize];
                        let t0 = Instant::now();
                        // FN55c: the block-aligned span of a table's LAST rows runs past the
                        // file's end (the shard is not a multiple of the block): the read ends at
                        // EOF and is complete when it covers the row
                        if let Err(e) =
                            read_covering(&fds[f as usize], buf, off, inner as usize + row_bytes)
                        {
                            err = Some(format!("row read at {off}+{len}: {e}"));
                            break;
                        }
                        let t1 = Instant::now();
                        note_io_end(t1);
                        tim.push(ReadTime {
                            start_us: (t0 - submit).as_secs_f32() * 1e6,
                            dur_us: (t1 - t0).as_secs_f32() * 1e6,
                        });
                        out.extend_from_slice(&buf[inner as usize..inner as usize + row_bytes]);
                        pos.push(p);
                    }
                    let msg = match err {
                        Some(e) => Err(e),
                        None => Ok((pos, out, tim)),
                    };
                    if rtx.send(msg).is_err() {
                        break;
                    }
                }
            }));
            tx.push(jtx);
        }
        Ok(RowReaderPool {
            tx,
            rx,
            handles,
            row_bytes,
        })
    }

    /// Workers (the queue depth).
    #[must_use]
    pub fn qd(&self) -> usize {
        self.tx.len()
    }

    /// Read every job; returns the rows in job order.
    pub fn read(&self, jobs: &[(u32, u64, u32, u32)]) -> Result<Vec<u8>, PleError> {
        self.read_timed(jobs).map(|(out, _)| out)
    }

    /// [`Self::read`] with every read's timing (FN121), in job order.
    pub fn read_timed(
        &self,
        jobs: &[(u32, u64, u32, u32)],
    ) -> Result<(Vec<u8>, Vec<ReadTime>), PleError> {
        let p = self.submit(jobs)?;
        self.complete(p)
    }

    /// FN121: hand every job to the workers WITHOUT waiting (one contiguous sub-batch a worker:
    /// up to `qd` reads in flight together); [`Self::complete`] collects them. Nothing else may
    /// use the pool between the two (the results share one channel).
    pub fn submit(&self, jobs: &[(u32, u64, u32, u32)]) -> Result<Pending, PleError> {
        let n = jobs.len();
        let submit = Instant::now();
        if n == 0 {
            return Ok(Pending { sent: 0, n, submit });
        }
        let w = self.tx.len().min(n);
        let per = n.div_ceil(w);
        let mut sent = 0;
        for (k, chunk) in jobs.chunks(per).enumerate() {
            let base = k * per;
            let job: ReadJob = (
                submit,
                chunk
                    .iter()
                    .enumerate()
                    .map(|(i, &(f, o, l, inner))| ((base + i) as u32, f, o, l, inner))
                    .collect(),
            );
            self.tx[k]
                .send(job)
                .map_err(|_| PleError::Io("a reader thread died".into()))?;
            sent += 1;
        }
        Ok(Pending { sent, n, submit })
    }

    /// FN121: collect a [`Self::submit`]ted batch: the rows and every read's timing, job order.
    pub fn complete(&self, p: Pending) -> Result<(Vec<u8>, Vec<ReadTime>), PleError> {
        let rb = self.row_bytes;
        let mut out = vec![0u8; p.n * rb];
        let mut tim = vec![ReadTime::default(); p.n];
        let mut first_err = None;
        for _ in 0..p.sent {
            match self.rx.recv() {
                Ok(Ok((pos, bytes, t))) => {
                    for (j, q) in pos.iter().enumerate() {
                        let q = *q as usize;
                        out[q * rb..(q + 1) * rb].copy_from_slice(&bytes[j * rb..(j + 1) * rb]);
                        tim[q] = t[j];
                    }
                }
                Ok(Err(e)) => first_err = first_err.or(Some(e)),
                Err(_) => first_err = first_err.or(Some("reader channel closed".into())),
            }
        }
        match first_err {
            Some(e) => Err(PleError::Io(e)),
            None => Ok((out, tim)),
        }
    }
}

impl Drop for RowReaderPool {
    fn drop(&mut self) {
        self.tx.clear();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// FN8's trial reader: the same jobs as [`RowReaderPool::read`] (file, 512-aligned span offset,
/// span length, row offset inside the span), submitted from the CALLING thread through one
/// [`dio::IoUring`] — no worker hand-off. The bytes are gated equal to the pool's
/// (`tests/ple_tier.rs`, `examples/fn8_ple.rs`).
#[derive(Debug)]
pub struct UringRowReader {
    files: Vec<File>,
    ring: dio::IoUring,
    scratch: AlignedBuf,
    row_bytes: usize,
}

impl UringRowReader {
    /// A reader over `files` (opened once, in `mode`) with a ring of `entries` slots.
    pub fn new(
        files: &[PathBuf],
        mode: OpenMode,
        entries: u32,
        row_bytes: usize,
    ) -> Result<UringRowReader, PleError> {
        let files = files
            .iter()
            .map(|p| {
                dio::open_read(p, mode).map_err(|e| PleError::Io(format!("{}: {e}", p.display())))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ring =
            dio::IoUring::new(entries).map_err(|e| PleError::Io(format!("io_uring: {e}")))?;
        Ok(UringRowReader {
            files,
            scratch: AlignedBuf::zeroed(ring.entries() as usize * 2 * PLE_BLOCK as usize),
            ring,
            row_bytes,
        })
    }

    /// Read every job; returns the rows in job order.
    pub fn read(&mut self, jobs: &[(u32, u64, u32, u32)]) -> Result<Vec<u8>, PleError> {
        let rb = self.row_bytes;
        let mut out = vec![0u8; jobs.len() * rb];
        let slot = 2 * PLE_BLOCK as usize;
        let wave = self.ring.entries() as usize;
        for (w, chunk) in jobs.chunks(wave).enumerate() {
            {
                let mut bufs: Vec<&mut [u8]> = self.scratch.chunks_mut(slot).collect();
                let mut reads: Vec<(&File, u64, &mut [u8])> = Vec::with_capacity(chunk.len());
                for (k, &(f, off, len, _)) in chunk.iter().enumerate() {
                    let file = self
                        .files
                        .get(f as usize)
                        .ok_or_else(|| PleError::Io(format!("no file {f}")))?;
                    let b = std::mem::take(&mut bufs[k]);
                    reads.push((file, off, &mut b[..len as usize]));
                }
                self.ring
                    .read_batch(&mut reads)
                    .map_err(|e| PleError::Io(format!("io_uring read: {e}")))?;
            }
            for (k, &(_, _, _, inner)) in chunk.iter().enumerate() {
                let src = &self.scratch[k * slot + inner as usize..k * slot + inner as usize + rb];
                let p = w * wave + k;
                out[p * rb..(p + 1) * rb].copy_from_slice(src);
            }
        }
        Ok(out)
    }
}

/// An LRU cache of rows in one RAM slab (allocated once).
#[derive(Debug)]
pub struct RowCache {
    row_bytes: usize,
    slab: Vec<u8>,
    map: HashMap<u64, u32>,
    key: Vec<u64>,
    prev: Vec<u32>,
    next: Vec<u32>,
    head: u32,
    tail: u32,
    used: u32,
}

const NIL: u32 = u32::MAX;

impl RowCache {
    /// A cache of `rows` rows (0 = no cache).
    #[must_use]
    pub fn new(rows: u32, row_bytes: usize) -> RowCache {
        RowCache {
            row_bytes,
            slab: vec![0u8; rows as usize * row_bytes],
            map: HashMap::with_capacity(rows as usize),
            key: vec![u64::MAX; rows as usize],
            prev: vec![NIL; rows as usize],
            next: vec![NIL; rows as usize],
            head: NIL,
            tail: NIL,
            used: 0,
        }
    }
    /// Capacity in rows.
    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.key.len() as u32
    }
    fn unlink(&mut self, s: u32) {
        let (p, n) = (self.prev[s as usize], self.next[s as usize]);
        if p != NIL {
            self.next[p as usize] = n;
        } else {
            self.head = n;
        }
        if n != NIL {
            self.prev[n as usize] = p;
        } else {
            self.tail = p;
        }
    }
    fn push_front(&mut self, s: u32) {
        self.prev[s as usize] = NIL;
        self.next[s as usize] = self.head;
        if self.head != NIL {
            self.prev[self.head as usize] = s;
        }
        self.head = s;
        if self.tail == NIL {
            self.tail = s;
        }
    }
    /// FN121: whether the row is cached (its recency untouched).
    #[must_use]
    pub fn contains(&self, row: u64) -> bool {
        self.map.contains_key(&row)
    }
    /// The row's bytes if cached (and mark it most recent).
    pub fn get(&mut self, row: u64) -> Option<&[u8]> {
        let s = *self.map.get(&row)?;
        self.unlink(s);
        self.push_front(s);
        let rb = self.row_bytes;
        Some(&self.slab[s as usize * rb..(s as usize + 1) * rb])
    }
    /// Insert a row, evicting the least recent when full.
    pub fn insert(&mut self, row: u64, bytes: &[u8]) {
        if self.capacity() == 0 || self.map.contains_key(&row) {
            return;
        }
        let s = if self.used < self.capacity() {
            self.used += 1;
            self.used - 1
        } else {
            let t = self.tail;
            self.unlink(t);
            self.map.remove(&self.key[t as usize]);
            t
        };
        let rb = self.row_bytes;
        self.slab[s as usize * rb..(s as usize + 1) * rb].copy_from_slice(bytes);
        self.key[s as usize] = row;
        self.map.insert(row, s);
        self.push_front(s);
    }
    /// Forget everything (the forced-miss control).
    pub fn clear(&mut self) {
        self.map.clear();
        self.key.iter_mut().for_each(|k| *k = u64::MAX);
        self.head = NIL;
        self.tail = NIL;
        self.used = 0;
    }
}

/// One gather's counters.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PleCounts {
    /// Row lookups requested (tokens × 16).
    pub requested: u64,
    /// Distinct rows in the gather.
    pub distinct: u64,
    /// Distinct rows served from the RAM cache.
    pub cache_hits: u64,
    /// Distinct rows read from NVMe.
    pub nvme_reads: u64,
    /// Of those, 1,024 B spans (the row straddled a 512 B block).
    pub straddles: u64,
    /// Bytes read from the drive.
    pub bytes_read: u64,
    /// Seconds in the NVMe reads.
    pub read_s: f64,
    /// Seconds for the whole gather.
    pub total_s: f64,
}

impl PleCounts {
    fn add(&mut self, o: &PleCounts) {
        self.requested += o.requested;
        self.distinct += o.distinct;
        self.cache_hits += o.cache_hits;
        self.nvme_reads += o.nvme_reads;
        self.straddles += o.straddles;
        self.bytes_read += o.bytes_read;
        self.read_s += o.read_s;
        self.total_s += o.total_s;
    }
}

/// FN121: the asynchronous candidate-row prefetch's counters (`--fn-ple-prefetch cand`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PrefetchCounts {
    /// Prefetch calls.
    pub calls: u64,
    /// Calls that submitted drive reads.
    pub batches: u64,
    /// Row ids asked (repeats counted).
    pub asked: u64,
    /// Rows already cached.
    pub cached: u64,
    /// Rows already in flight.
    pub pending: u64,
    /// Rows submitted to the drive.
    pub submitted: u64,
    /// Settles that had to WAIT for reads still in flight.
    pub waits: u64,
    /// The ms those settles waited.
    pub wait_ms: f64,
}

/// The PLE row tier: index + reader pool + RAM row cache.
#[derive(Debug)]
pub struct PleTier {
    index: PleIndex,
    pool: RowReaderPool,
    cache: RowCache,
    total: PleCounts,
    /// FN121: a prefetch in flight (the batch, its rows in job order).
    pend: Option<(Pending, Vec<u64>)>,
    pf: PrefetchCounts,
    /// FN121: the per-batch read log (`Some` = on; `--fn-ple-trace`).
    log: Option<Vec<ReadBatch>>,
}

impl PleTier {
    /// A tier over `index` with `qd` readers and a `cache_rows`-row RAM cache.
    pub fn new(
        index: PleIndex,
        mode: OpenMode,
        qd: usize,
        cache_rows: u32,
    ) -> Result<PleTier, PleError> {
        let rb = index.row_bytes as usize;
        let pool = RowReaderPool::new(&index.files, mode, qd, rb)?;
        Ok(PleTier {
            cache: RowCache::new(cache_rows, rb),
            pool,
            index,
            total: PleCounts::default(),
            pend: None,
            pf: PrefetchCounts::default(),
            log: None,
        })
    }

    /// FN121: keep a log of every drive batch ([`Self::take_read_log`]); off by default.
    pub fn set_read_log(&mut self, on: bool) {
        self.log = on.then(Vec::new);
    }

    /// FN121: the batches logged since the last take (empty when the log is off).
    pub fn take_read_log(&mut self) -> Vec<ReadBatch> {
        self.log.as_mut().map(std::mem::take).unwrap_or_default()
    }

    /// FN121: the prefetch's counters.
    #[must_use]
    pub fn prefetch_counts(&self) -> PrefetchCounts {
        self.pf
    }

    /// FN121: whether every row of `ids` is in the RAM cache (no recency change) -- the
    /// resident test at the device gate; a prefetch still in flight is NOT resident.
    #[must_use]
    pub fn resident(&self, ids: &[u64]) -> u64 {
        ids.iter().filter(|&&r| self.cache.contains(r)).count() as u64
    }

    /// FN121 (the PLE prefetch law): start reading the rows of `ids` that are neither cached nor
    /// already in flight, WITHOUT waiting -- one submission, the pool's workers' reads in flight
    /// together. The rows land in the RAM cache at the next [`Self::settle`] (every
    /// [`Self::gather`] settles first), so a prefetched row is the same bytes a gather would read.
    /// Returns the rows submitted.
    pub fn prefetch(&mut self, ids: &[u64]) -> Result<u64, PleError> {
        self.pf.calls += 1;
        self.pf.asked += ids.len() as u64;
        let mut want: Vec<u64> = Vec::new();
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let inflight: std::collections::HashSet<u64> = self
            .pend
            .as_ref()
            .map(|(_, r)| r.iter().copied().collect())
            .unwrap_or_default();
        for &r in ids {
            if !seen.insert(r) {
                continue;
            }
            if self.cache.contains(r) {
                self.pf.cached += 1;
            } else if inflight.contains(&r) {
                self.pf.pending += 1;
            } else {
                want.push(r);
            }
        }
        if want.is_empty() {
            return Ok(0);
        }
        // one batch in flight at a time (the pool's results share one channel)
        self.settle()?;
        let mut jobs = Vec::with_capacity(want.len());
        for &r in &want {
            let (f, off, len, inner) = self.index.span(r)?;
            jobs.push((f, off, len, inner));
        }
        let p = self.pool.submit(&jobs)?;
        self.pf.batches += 1;
        self.pf.submitted += want.len() as u64;
        self.pend = Some((p, want));
        Ok(self.pend.as_ref().map_or(0, |(_, w)| w.len() as u64))
    }

    /// FN121: collect a prefetch in flight into the RAM cache (blocking for reads not yet done;
    /// the wait is counted). Returns the ms waited.
    pub fn settle(&mut self) -> Result<f64, PleError> {
        let Some((p, rows)) = self.pend.take() else {
            return Ok(0.0);
        };
        let t = Instant::now();
        let submit = p.submit;
        let idle = idle_before(submit);
        let (got, tim) = self.pool.complete(p)?;
        let waited = t.elapsed().as_secs_f64() * 1e3;
        // "waited" counts only a settle that found reads still in flight (> 20 µs: the channel's
        // own hand-off when everything had landed)
        if waited > 0.02 {
            self.pf.waits += 1;
            self.pf.wait_ms += waited;
        }
        let rb = self.index.row_bytes as usize;
        let mut c = PleCounts {
            nvme_reads: rows.len() as u64,
            read_s: (Instant::now() - submit).as_secs_f64(),
            ..PleCounts::default()
        };
        for (j, &r) in rows.iter().enumerate() {
            let (_, _, len, _) = self.index.span(r)?;
            c.bytes_read += u64::from(len);
            if u64::from(len) > PLE_BLOCK {
                c.straddles += 1;
            }
            self.cache.insert(r, &got[j * rb..(j + 1) * rb]);
        }
        self.total.add(&c);
        if let Some(log) = self.log.as_mut() {
            log.push(ReadBatch {
                kind: "prefetch",
                submit_unix_us: unix_us(submit),
                idle_us: idle,
                reads: tim,
                wait_us: (waited * 1e3) as f32,
            });
        }
        Ok(waited)
    }

    /// The index.
    #[must_use]
    pub fn index(&self) -> &PleIndex {
        &self.index
    }

    /// Stage the rows of `ids` (`tokens × rows_per_token` row ids, token-major, hash order) into
    /// `staged` (`tokens × rows_per_token × row_bytes`, the contract layout).
    pub fn gather(&mut self, ids: &[u64], staged: &mut [u8]) -> Result<PleCounts, PleError> {
        // FN121: a prefetch in flight lands first (its rows are the cache's from here on)
        self.settle()?;
        let t0 = Instant::now();
        let rb = self.index.row_bytes as usize;
        if staged.len() != ids.len() * rb {
            return Err(PleError::Staging(format!(
                "{} bytes for {} rows of {rb}",
                staged.len(),
                ids.len()
            )));
        }
        let mut c = PleCounts {
            requested: ids.len() as u64,
            ..PleCounts::default()
        };
        // Distinct rows in first-appearance order.
        let mut first: HashMap<u64, u32> = HashMap::with_capacity(ids.len());
        let mut distinct: Vec<u64> = Vec::new();
        for &r in ids {
            first.entry(r).or_insert_with(|| {
                distinct.push(r);
                (distinct.len() - 1) as u32
            });
        }
        c.distinct = distinct.len() as u64;
        let mut rows = vec![0u8; distinct.len() * rb];
        let mut misses: Vec<(usize, u64)> = Vec::new();
        for (k, &r) in distinct.iter().enumerate() {
            match self.cache.get(r) {
                Some(b) => {
                    rows[k * rb..(k + 1) * rb].copy_from_slice(b);
                    c.cache_hits += 1;
                }
                None => misses.push((k, r)),
            }
        }
        if !misses.is_empty() {
            let mut jobs = Vec::with_capacity(misses.len());
            for &(_, r) in &misses {
                let (f, off, len, inner) = self.index.span(r)?;
                c.bytes_read += u64::from(len);
                if u64::from(len) > PLE_BLOCK {
                    c.straddles += 1;
                }
                jobs.push((f, off, len, inner));
            }
            let tr = Instant::now();
            let idle = idle_before(tr);
            let (got, tim) = self.pool.read_timed(&jobs)?;
            c.read_s = tr.elapsed().as_secs_f64();
            if let Some(log) = self.log.as_mut() {
                log.push(ReadBatch {
                    kind: "gather",
                    submit_unix_us: unix_us(tr),
                    idle_us: idle,
                    reads: tim,
                    wait_us: (c.read_s * 1e6) as f32,
                });
            }
            c.nvme_reads = misses.len() as u64;
            for (j, &(k, r)) in misses.iter().enumerate() {
                let b = &got[j * rb..(j + 1) * rb];
                rows[k * rb..(k + 1) * rb].copy_from_slice(b);
                self.cache.insert(r, b);
            }
        }
        for (p, r) in ids.iter().enumerate() {
            let k = first[r] as usize;
            staged[p * rb..(p + 1) * rb].copy_from_slice(&rows[k * rb..(k + 1) * rb]);
        }
        c.total_s = t0.elapsed().as_secs_f64();
        self.total.add(&c);
        Ok(c)
    }

    /// The forced-miss control: empty the row cache.
    pub fn evict_all(&mut self) {
        self.cache.clear();
    }

    /// Totals since construction.
    #[must_use]
    pub fn totals(&self) -> PleCounts {
        self.total
    }

    /// The boot line.
    #[must_use]
    pub fn boot_line(&self) -> String {
        let i = &self.index;
        format!(
            "pletier: {:?} {} rows x {} B in {} shards over {} file(s), {} rows/token = {} B/token staged | \
             qd {} | row cache {} rows ({:.1} MiB){}",
            i.kind,
            i.rows,
            i.row_bytes,
            i.shards.len(),
            i.files.len(),
            i.rows_per_token,
            i.bytes_per_token(),
            self.pool.qd(),
            self.cache.capacity(),
            f64::from(self.cache.capacity()) * i.row_bytes as f64 / f64::from(1u32 << 20),
            if i.kind == PleSourceKind::QwenFp8 {
                format!(" | GATED fp8 source, table scale {}", i.table_scale)
            } else {
                String::new()
            }
        )
    }

    /// Prometheus text for the metrics plane.
    #[must_use]
    pub fn render_prometheus(&self) -> String {
        let t = &self.total;
        let mut o = String::new();
        let _ = writeln!(o, "# TYPE saber_ple_rows_total counter");
        for (k, v) in [
            ("requested", t.requested),
            ("distinct", t.distinct),
            ("cache_hit", t.cache_hits),
            ("nvme", t.nvme_reads),
            ("straddle", t.straddles),
        ] {
            let _ = writeln!(o, "saber_ple_rows_total{{path=\"{k}\"}} {v}");
        }
        let _ = writeln!(o, "# TYPE saber_ple_bytes_read_total counter");
        let _ = writeln!(o, "saber_ple_bytes_read_total {}", t.bytes_read);
        o
    }
}

#[cfg(test)]
mod fn55c_tests {
    use super::*;

    /// FN55c: a row in a file's last partial block reads (the span past EOF covers it: the
    /// served p5_Li fault, `row read at 3200015360+1024: short read: 648 of 1024`), and a row the
    /// file does not hold is still the loud short read (the control).
    #[test]
    fn fn55c_a_row_at_the_files_tail_reads_and_a_missing_one_refuses() {
        let scratch = saber_test_scratch::ScratchDir::new("saber-fn55c-pletail");
        let p = scratch.path().join("t.bin");
        let data: Vec<u8> = (0..1672u32).map(|i| i as u8).collect();
        std::fs::write(&p, &data).unwrap();
        let f = File::open(&p).unwrap();
        let mut buf = vec![0u8; 1024];
        read_covering(&f, &mut buf, 1024, 600).unwrap();
        assert_eq!(&buf[..648], &data[1024..]);
        let e = read_covering(&f, &mut buf, 1024, 700)
            .unwrap_err()
            .to_string();
        assert!(e.contains("short read: 648 of 1024"), "{e}");
        assert!(dio::preadv_exact(&f, &mut buf, 1024).is_err());
    }
}
