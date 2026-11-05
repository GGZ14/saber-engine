//! FN18: the served route trace, the real-traffic routing profile, and the host replay of the
//! expert tier over served traffic (docs/EXEC.md §14, "the profile law"; LEDGER FN18).
//!
//! The serve plane (`--fn-route-trace PATH`, rank 0) appends one record per engine step: the
//! target's routed ids per layer (a prefill chunk as a sparse per-layer histogram, a decode / verify
//! step as the raw ids of every row, live rows first then the pad rows), the MTP drafter's routed
//! ids per pass, and the PLE row ids of every gather. Everything the memory plane decides is a
//! pure function of these ids and its own state (I2), so [`replay`] -- the real [`ExpertTier`]
//! over a counting mover -- reproduces the served tier's which-path counters EXACTLY from the same
//! warm start (the self-test, GATES FN18), and predicts them for another placement.
//!
//! Format `fn18-route-trace/1`: the 8-byte magic, then `u32 layers, u32 topk, u32 experts, u32 0`,
//! then records: `u8 kind, u8 0, u16 bucket, u16 width, u16 0, u32 rows, u32 live_rows,
//! f64 unix_seconds, u32 payload_bytes`, the payload (little endian throughout):
//!
//! * [`StepKind::Prefill`]: per layer `u16 n` then `n x (u16 expert, u16 count)` (ascending id).
//! * [`StepKind::Decode`]: `layers x rows x topk` `u16` ids, layer-major; rows `live_rows..` are pads.
//! * [`StepKind::MtpDecode`] / [`StepKind::MtpPrefill`]: `rows x topk` `u16` ids (the MTP layer).
//! * [`StepKind::Ple`]: `rows` `u64` PLE row ids (`width` = 1 for a decode gather, 0 for prefill).

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::expert_tier::{
    table_age, table_class, table_index, ExpertMover, ExpertTier, PathCounts, Placement,
    RouteProfile, TierError, TierParams, TABLE_AGES, TABLE_CLASSES,
};
use crate::xtier::XtierHeader;

/// The trace file's magic.
pub const TRACE_MAGIC: &[u8; 8] = b"FN18RTR1";
/// The routing profile format the tier reads (FN7's).
pub const PROFILE_FORMAT: &str = "fn7-route-profile/1";

/// What a trace record holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepKind {
    /// A target prefill chunk (every row live).
    Prefill,
    /// A target decode / verify step (rows = bucket x width).
    Decode,
    /// An MTP drafter pass in the decode form.
    MtpDecode,
    /// An MTP drafter pass over a prompt piece.
    MtpPrefill,
    /// A PLE gather's row ids.
    Ple,
}

impl StepKind {
    fn code(self) -> u8 {
        match self {
            StepKind::Prefill => 0,
            StepKind::Decode => 1,
            StepKind::MtpDecode => 2,
            StepKind::MtpPrefill => 3,
            StepKind::Ple => 4,
        }
    }
    fn from_code(c: u8) -> Option<StepKind> {
        Some(match c {
            0 => StepKind::Prefill,
            1 => StepKind::Decode,
            2 => StepKind::MtpDecode,
            3 => StepKind::MtpPrefill,
            4 => StepKind::Ple,
            _ => return None,
        })
    }
    /// A target step (the tier sees it).
    #[must_use]
    pub fn target(self) -> bool {
        matches!(self, StepKind::Prefill | StepKind::Decode)
    }
}

/// The trace writer (the serve plane's rank 0).
#[derive(Debug)]
pub struct TraceWriter {
    w: BufWriter<File>,
    layers: u32,
    topk: u32,
    /// Records written.
    pub records: u64,
    /// Bytes written (header + records).
    pub bytes: u64,
    last_flush: std::time::Instant,
}

fn now_s() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

impl TraceWriter {
    /// Create (truncate) a trace.
    pub fn create(
        path: &Path,
        layers: u32,
        topk: u32,
        experts: u32,
    ) -> std::io::Result<TraceWriter> {
        let mut w = BufWriter::with_capacity(4 << 20, File::create(path)?);
        w.write_all(TRACE_MAGIC)?;
        for x in [layers, topk, experts, 0] {
            w.write_all(&x.to_le_bytes())?;
        }
        Ok(TraceWriter {
            w,
            layers,
            topk,
            records: 0,
            bytes: 24,
            last_flush: std::time::Instant::now(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn head(
        &mut self,
        kind: StepKind,
        bucket: u32,
        width: u32,
        rows: u32,
        live: u32,
        payload: usize,
    ) -> std::io::Result<()> {
        self.head_t(kind, bucket, width, rows, live, payload, now_s())
    }

    #[allow(clippy::too_many_arguments)]
    fn head_t(
        &mut self,
        kind: StepKind,
        bucket: u32,
        width: u32,
        rows: u32,
        live: u32,
        payload: usize,
        t: f64,
    ) -> std::io::Result<()> {
        let mut h = Vec::with_capacity(28);
        h.push(kind.code());
        h.push(0);
        h.extend_from_slice(&(bucket as u16).to_le_bytes());
        h.extend_from_slice(&(width as u16).to_le_bytes());
        h.extend_from_slice(&0u16.to_le_bytes());
        h.extend_from_slice(&rows.to_le_bytes());
        h.extend_from_slice(&live.to_le_bytes());
        h.extend_from_slice(&t.to_le_bytes());
        h.extend_from_slice(&(payload as u32).to_le_bytes());
        // a served process is stopped by a signal: at most ~2 s of records sit in the buffer
        if self.last_flush.elapsed().as_secs_f64() > 2.0 {
            self.w.flush()?;
            self.last_flush = std::time::Instant::now();
        }
        self.w.write_all(&h)?;
        self.records += 1;
        self.bytes += (h.len() + payload) as u64;
        Ok(())
    }

    /// A target step. `ids` = the route log: `layers x rows x topk` global ids, layer-major.
    /// A prefill chunk is stored as a sparse per-layer histogram, a decode step raw.
    pub fn target(
        &mut self,
        kind: StepKind,
        rows: u32,
        live: u32,
        bucket: u32,
        width: u32,
        ids: &[u32],
    ) -> std::io::Result<()> {
        self.target_t(kind, rows, live, bucket, width, ids, now_s())
    }

    /// [`TraceWriter::target`] stamped at `t` (the recorder's writer thread: the time the rank
    /// thread handed the record over).
    #[allow(clippy::too_many_arguments)]
    pub fn target_t(
        &mut self,
        kind: StepKind,
        rows: u32,
        live: u32,
        bucket: u32,
        width: u32,
        ids: &[u32],
        t: f64,
    ) -> std::io::Result<()> {
        let per = (rows * self.topk) as usize;
        assert_eq!(ids.len(), per * self.layers as usize, "route log length");
        let mut p = Vec::new();
        if kind == StepKind::Prefill {
            for l in 0..self.layers as usize {
                let mut h: Vec<(u32, u32)> = Vec::new();
                let mut v: Vec<u32> = ids[l * per..(l + 1) * per].to_vec();
                v.sort_unstable();
                for e in v {
                    match h.last_mut() {
                        Some((x, c)) if *x == e => *c += 1,
                        _ => h.push((e, 1)),
                    }
                }
                p.extend_from_slice(&(h.len() as u16).to_le_bytes());
                for (e, c) in h {
                    p.extend_from_slice(&(e as u16).to_le_bytes());
                    p.extend_from_slice(&(c.min(65_535) as u16).to_le_bytes());
                }
            }
        } else {
            for &e in ids {
                p.extend_from_slice(&(e as u16).to_le_bytes());
            }
        }
        self.head_t(kind, bucket, width, rows, live, p.len(), t)?;
        self.w.write_all(&p)
    }

    /// An MTP drafter pass: `rows x topk` global ids.
    pub fn mtp(&mut self, kind: StepKind, rows: u32, ids: &[u32]) -> std::io::Result<()> {
        let mut p = Vec::with_capacity(ids.len() * 2);
        for &e in ids {
            p.extend_from_slice(&(e as u16).to_le_bytes());
        }
        self.head(kind, 0, 0, rows, rows, p.len())?;
        self.w.write_all(&p)
    }

    /// A PLE gather's row ids.
    pub fn ple(&mut self, decode: bool, rows: &[u64]) -> std::io::Result<()> {
        self.ple_t(decode, rows, now_s())
    }

    /// [`TraceWriter::ple`] stamped at `t`.
    pub fn ple_t(&mut self, decode: bool, rows: &[u64], t: f64) -> std::io::Result<()> {
        let mut p = Vec::with_capacity(rows.len() * 8);
        for &r in rows {
            p.extend_from_slice(&r.to_le_bytes());
        }
        let n = rows.len() as u32;
        self.head_t(StepKind::Ple, 0, u32::from(decode), n, n, p.len(), t)?;
        self.w.write_all(&p)
    }

    /// Flush the buffer (the serve plane flushes on idle).
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.w.flush()
    }
}

/// FN65: the served route RECORDER's configuration (`--fn-route-record DIR`).
#[derive(Debug, Clone)]
pub struct RecordCfg {
    /// The directory the rotated trace files land in (created).
    pub dir: std::path::PathBuf,
    /// Layers of the target.
    pub layers: u32,
    /// Routes per token.
    pub topk: u32,
    /// Experts per layer (global).
    pub experts: u32,
    /// Target steps a file holds before the recorder rotates to the next one.
    pub rotate_steps: u64,
    /// The most bytes this recorder keeps on the drive: past it the OLDEST of its own closed
    /// files is deleted (a ring; never another process's file).
    pub max_bytes: u64,
    /// The most bytes in flight between the rank thread and the writer thread: past it a record
    /// is DROPPED (counted), never waited for -- the step path never blocks on the drive.
    pub max_inflight: u64,
}

enum Rec {
    Target {
        kind: StepKind,
        rows: u32,
        live: u32,
        bucket: u32,
        width: u32,
        t: f64,
        log: Vec<u8>,
    },
    Ple {
        decode: bool,
        t: f64,
        rows: Vec<u64>,
    },
}

impl Rec {
    fn bytes(&self) -> u64 {
        match self {
            Rec::Target { log, .. } => log.len() as u64,
            Rec::Ple { rows, .. } => rows.len() as u64 * 8,
        }
    }
}

/// What the recorder's writer thread did (returned by [`TraceSink::finish`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecordStats {
    /// Files opened.
    pub files: u64,
    /// Records written (target + PLE).
    pub records: u64,
    /// Target steps written.
    pub steps: u64,
    /// Bytes written (every file, deleted ones included).
    pub bytes: u64,
    /// Files deleted by the ring (`max_bytes`).
    pub deleted: u64,
    /// Write errors (the file in hand is abandoned; the next record opens a new one).
    pub errors: u64,
}

/// FN65: the served route recorder -- the FN18 trace written OFF the rank thread. The rank thread
/// hands each record over (the route log it already read for the tier, moved, plus a timestamp)
/// through a bounded queue and never waits: a full queue drops the record and counts it. A
/// writer thread encodes, writes, flushes every ~1-2 s, rotates every `rotate_steps` target steps
/// (`routes-<unix>-<pid>-<seq>.bin`; `.part` while open) and keeps at most `max_bytes` of its own
/// files. The MTP drafter's routes are NOT recorded (they need a stream sync; the expert tier's
/// placement reads target steps only).
pub struct TraceSink {
    tx: Option<std::sync::mpsc::SyncSender<Rec>>,
    join: Option<std::thread::JoinHandle<RecordStats>>,
    inflight: std::sync::Arc<std::sync::atomic::AtomicU64>,
    max_inflight: u64,
    /// Records handed over.
    pub sent: u64,
    /// Records dropped (queue full / writer gone).
    pub dropped: u64,
    /// Rank-thread nanoseconds spent handing records over (the recorder's whole step-path cost).
    pub send_ns: u64,
    /// The target steps handed over.
    pub sent_steps: u64,
    /// The directory.
    pub dir: std::path::PathBuf,
}

impl std::fmt::Debug for TraceSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TraceSink")
            .field("dir", &self.dir)
            .field("sent", &self.sent)
            .field("dropped", &self.dropped)
            .finish_non_exhaustive()
    }
}

struct Ring {
    cfg: RecordCfg,
    cur: Option<(TraceWriter, std::path::PathBuf, std::path::PathBuf)>,
    steps_in_file: u64,
    seq: u64,
    closed: std::collections::VecDeque<(std::path::PathBuf, u64)>,
    closed_bytes: u64,
    st: RecordStats,
    stamp: u64,
}

impl Ring {
    fn open(&mut self) -> std::io::Result<()> {
        let name = format!(
            "routes-{}-{}-{:05}.bin",
            self.stamp,
            std::process::id(),
            self.seq
        );
        self.seq += 1;
        let fin = self.cfg.dir.join(&name);
        let part = self.cfg.dir.join(format!("{name}.part"));
        let w = TraceWriter::create(&part, self.cfg.layers, self.cfg.topk, self.cfg.experts)?;
        self.cur = Some((w, part, fin));
        self.steps_in_file = 0;
        self.st.files += 1;
        Ok(())
    }

    fn close(&mut self) -> std::io::Result<()> {
        if let Some((mut w, part, fin)) = self.cur.take() {
            w.flush()?;
            let b = w.bytes;
            drop(w);
            std::fs::rename(&part, &fin)?;
            self.closed.push_back((fin, b));
            self.closed_bytes += b;
            while self.closed_bytes > self.cfg.max_bytes && self.closed.len() > 1 {
                let (p, b) = self.closed.pop_front().expect("non-empty");
                let _ = std::fs::remove_file(&p);
                self.closed_bytes -= b;
                self.st.deleted += 1;
            }
        }
        Ok(())
    }

    fn put(&mut self, r: Rec) -> std::io::Result<()> {
        if self.cur.is_none() {
            self.open()?;
        }
        let (w, _, _) = self.cur.as_mut().expect("open");
        let before = w.bytes;
        match r {
            Rec::Target {
                kind,
                rows,
                live,
                bucket,
                width,
                t,
                log,
            } => {
                let ids: Vec<u32> = log
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                w.target_t(kind, rows, live, bucket, width, &ids, t)?;
                self.steps_in_file += 1;
                self.st.steps += 1;
            }
            Rec::Ple { decode, t, rows } => w.ple_t(decode, &rows, t)?,
        }
        self.st.records += 1;
        self.st.bytes += w.bytes - before;
        if self.steps_in_file >= self.cfg.rotate_steps.max(1) {
            self.close()?;
        }
        Ok(())
    }

    fn run(
        mut self,
        rx: &std::sync::mpsc::Receiver<Rec>,
        inflight: &std::sync::atomic::AtomicU64,
    ) -> RecordStats {
        use std::sync::atomic::Ordering;
        use std::sync::mpsc::RecvTimeoutError;
        loop {
            match rx.recv_timeout(std::time::Duration::from_millis(1000)) {
                Ok(r) => {
                    let b = r.bytes();
                    let res = self.put(r);
                    inflight.fetch_sub(b, Ordering::Relaxed);
                    if let Err(e) = res {
                        if self.st.errors == 0 {
                            eprintln!("fnrecord: write error ({e}); the file in hand is abandoned");
                        }
                        self.st.errors += 1;
                        self.cur = None;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    // idle: everything handed over is on the drive within ~1 s
                    if let Some((w, _, _)) = self.cur.as_mut() {
                        if w.flush().is_err() {
                            self.st.errors += 1;
                        }
                    }
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        if self.close().is_err() {
            self.st.errors += 1;
        }
        self.st
    }
}

impl TraceSink {
    /// Start the recorder (creates `cfg.dir`).
    pub fn spawn(cfg: RecordCfg) -> std::io::Result<TraceSink> {
        std::fs::create_dir_all(&cfg.dir)?;
        let (tx, rx) = std::sync::mpsc::sync_channel::<Rec>(65_536);
        let inflight = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let inf = inflight.clone();
        let dir = cfg.dir.clone();
        let max_inflight = cfg.max_inflight;
        let ring = Ring {
            cfg,
            cur: None,
            steps_in_file: 0,
            seq: 0,
            closed: std::collections::VecDeque::new(),
            closed_bytes: 0,
            st: RecordStats::default(),
            stamp: now_s() as u64,
        };
        let join = std::thread::Builder::new()
            .name("saber-routes".into())
            .spawn(move || ring.run(&rx, &inf))?;
        Ok(TraceSink {
            tx: Some(tx),
            join: Some(join),
            inflight,
            max_inflight,
            sent: 0,
            dropped: 0,
            send_ns: 0,
            sent_steps: 0,
            dir,
        })
    }

    fn send(&mut self, r: Rec, t0: std::time::Instant) {
        use std::sync::atomic::Ordering;
        let b = r.bytes();
        let ok = match &self.tx {
            Some(tx) if self.inflight.load(Ordering::Relaxed) + b <= self.max_inflight => {
                self.inflight.fetch_add(b, Ordering::Relaxed);
                match tx.try_send(r) {
                    Ok(()) => true,
                    Err(_) => {
                        self.inflight.fetch_sub(b, Ordering::Relaxed);
                        false
                    }
                }
            }
            _ => false,
        };
        if ok {
            self.sent += 1;
        } else {
            self.dropped += 1;
        }
        self.send_ns += t0.elapsed().as_nanos() as u64;
    }

    /// A target step: `log` = the route log as read for the tier (`layers x rows x topk` u32 LE,
    /// layer-major), MOVED (no copy on the rank thread).
    pub fn target_log(
        &mut self,
        kind: StepKind,
        rows: u32,
        live: u32,
        bucket: u32,
        width: u32,
        log: Vec<u8>,
    ) {
        let t0 = std::time::Instant::now();
        let n = (self.sent, self.dropped);
        self.send(
            Rec::Target {
                kind,
                rows,
                live,
                bucket,
                width,
                t: now_s(),
                log,
            },
            t0,
        );
        if self.sent > n.0 {
            self.sent_steps += 1;
        }
    }

    /// A PLE gather's row ids (copied: the caller keeps them).
    pub fn ple(&mut self, decode: bool, rows: &[u64]) {
        let t0 = std::time::Instant::now();
        self.send(
            Rec::Ple {
                decode,
                t: now_s(),
                rows: rows.to_vec(),
            },
            t0,
        );
    }

    /// Close the queue and wait for the writer (everything handed over is on the drive).
    pub fn finish(&mut self) -> Option<RecordStats> {
        self.tx = None;
        self.join.take().and_then(|j| j.join().ok())
    }
}

impl Drop for TraceSink {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

/// FN18 / FN65: where the serve plane's routes go -- FN18's synchronous trace (one file, the
/// MTP passes too; the gate form) or FN65's recorder (rotated, off the rank thread, target + PLE).
#[derive(Debug)]
pub enum RouteSink {
    /// `--fn-route-trace PATH`.
    Trace(TraceWriter),
    /// `--fn-route-record DIR`.
    Record(TraceSink),
}

impl RouteSink {
    /// Whether the MTP drafter's passes are recorded (FN18's trace only: they need a stream sync).
    #[must_use]
    pub fn wants_mtp(&self) -> bool {
        matches!(self, RouteSink::Trace(_))
    }

    /// A target step from the tier's route log (u32 LE ids, `layers x rows x topk`).
    pub fn target_log(
        &mut self,
        kind: StepKind,
        rows: u32,
        live: u32,
        bucket: u32,
        width: u32,
        log: Vec<u8>,
    ) -> std::io::Result<()> {
        match self {
            RouteSink::Trace(w) => {
                let ids: Vec<u32> = log
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                w.target(kind, rows, live, bucket, width, &ids)
            }
            RouteSink::Record(s) => {
                s.target_log(kind, rows, live, bucket, width, log);
                Ok(())
            }
        }
    }

    /// An MTP pass (FN18's trace only; the recorder ignores it).
    pub fn mtp(&mut self, kind: StepKind, rows: u32, ids: &[u32]) -> std::io::Result<()> {
        match self {
            RouteSink::Trace(w) => w.mtp(kind, rows, ids),
            RouteSink::Record(_) => Ok(()),
        }
    }

    /// A PLE gather's row ids.
    pub fn ple(&mut self, decode: bool, rows: &[u64]) -> std::io::Result<()> {
        match self {
            RouteSink::Trace(w) => w.ple(decode, rows),
            RouteSink::Record(s) => {
                s.ple(decode, rows);
                Ok(())
            }
        }
    }

    /// Flush (the trace) / drain and close (the recorder), and the end-of-run line.
    pub fn finish(&mut self) -> std::io::Result<String> {
        match self {
            RouteSink::Trace(w) => {
                w.flush()?;
                Ok(format!("fntrace: {} records, {} bytes", w.records, w.bytes))
            }
            RouteSink::Record(s) => {
                let st = s.finish().unwrap_or_default();
                let per = s.send_ns as f64 / 1e3 / (s.sent + s.dropped).max(1) as f64;
                Ok(format!(
                    "fnrecord: {} handed over ({} target steps), {} dropped | rank thread {:.3} ms total, {:.2} us a record | writer: {} files, {} records, {} steps, {:.1} MiB, {} deleted by the ring, {} errors | {}",
                    s.sent,
                    s.sent_steps,
                    s.dropped,
                    s.send_ns as f64 / 1e6,
                    per,
                    st.files,
                    st.records,
                    st.steps,
                    st.bytes as f64 / 1048576.0,
                    st.deleted,
                    st.errors,
                    s.dir.display()
                ))
            }
        }
    }
}

/// FN65: the trace files under each path (a file as given; a directory: its `*.bin` files and the
/// `*.bin.part` file a stopped recorder left, sorted by name = by recorder start then sequence).
pub fn trace_files(paths: &[std::path::PathBuf]) -> std::io::Result<Vec<std::path::PathBuf>> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_dir() {
            let mut v: Vec<std::path::PathBuf> = std::fs::read_dir(p)?
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|q| {
                    let n = q.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    q.is_file() && (n.ends_with(".bin") || n.ends_with(".bin.part"))
                })
                .collect();
            v.sort();
            out.extend(v);
        } else {
            out.push(p.clone());
        }
    }
    Ok(out)
}

/// One routed expert of a step's layer: the global id and how many LIVE / PAD rows routed it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ent {
    /// Global expert id.
    pub e: u16,
    /// Live rows that routed it.
    pub live: u16,
    /// Pad rows that routed it.
    pub pad: u16,
}

/// One trace record, decoded.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    /// Kind.
    pub kind: StepKind,
    /// Decode bucket (0 for prefill / MTP / PLE).
    pub bucket: u32,
    /// Verify width (PLE: 1 = a decode gather).
    pub width: u32,
    /// Rows.
    pub rows: u32,
    /// Live rows.
    pub live: u32,
    /// Wall clock (unix seconds) at the record.
    pub t: f64,
    /// Target steps: `layers + 1` offsets into `ents`; MTP passes: 2.
    pub off: Vec<u32>,
    /// The distinct routed experts per layer (ascending id).
    pub ents: Vec<Ent>,
    /// PLE records: the row ids.
    pub ple: Vec<u64>,
}

impl Step {
    /// The distinct routed experts of `layer` (target) or of the pass (MTP, layer 0).
    #[must_use]
    pub fn layer(&self, layer: usize) -> &[Ent] {
        &self.ents[self.off[layer] as usize..self.off[layer + 1] as usize]
    }
    /// Layers carried.
    #[must_use]
    pub fn layers(&self) -> usize {
        self.off.len().saturating_sub(1)
    }
}

/// A streaming trace reader.
#[derive(Debug)]
pub struct TraceReader {
    r: BufReader<File>,
    /// Layers.
    pub layers: u32,
    /// Routes per token.
    pub topk: u32,
    /// Experts per layer (global).
    pub experts: u32,
    /// FN104: keep a decode record's raw ids (`layers x rows x topk`, layer-major: row `j * width
    /// + t` is position `j`'s row `t`) in [`Self::raw`] (off = FN18's reader).
    pub keep_raw: bool,
    /// FN104: the last decode record's raw ids when [`Self::keep_raw`] (empty otherwise).
    pub raw: Vec<u16>,
}

fn rd<const N: usize>(r: &mut impl Read) -> std::io::Result<Option<[u8; N]>> {
    let mut b = [0u8; N];
    let mut got = 0;
    while got < N {
        let n = r.read(&mut b[got..])?;
        if n == 0 {
            return if got == 0 {
                Ok(None)
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "a truncated trace record",
                ))
            };
        }
        got += n;
    }
    Ok(Some(b))
}

fn hist_into(ents: &mut Vec<Ent>, ids: &[u16], pad: bool) {
    let start = ents.len();
    let mut v = ids.to_vec();
    v.sort_unstable();
    for e in v {
        match ents[start..].last_mut() {
            Some(x) if x.e == e => {
                if pad {
                    x.pad += 1;
                } else {
                    x.live += 1;
                }
            }
            _ => ents.push(Ent {
                e,
                live: u16::from(!pad),
                pad: u16::from(pad),
            }),
        }
    }
}

fn merge_sorted(a: &[Ent], b: &[Ent]) -> Vec<Ent> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        if j == b.len() || (i < a.len() && a[i].e < b[j].e) {
            out.push(a[i]);
            i += 1;
        } else if i == a.len() || b[j].e < a[i].e {
            out.push(b[j]);
            j += 1;
        } else {
            out.push(Ent {
                e: a[i].e,
                live: a[i].live + b[j].live,
                pad: a[i].pad + b[j].pad,
            });
            i += 1;
            j += 1;
        }
    }
    out
}

impl TraceReader {
    /// Open a trace (checks the magic).
    pub fn open(path: &Path) -> std::io::Result<TraceReader> {
        let mut r = BufReader::with_capacity(8 << 20, File::open(path)?);
        let m: [u8; 8] = rd(&mut r)?.ok_or_else(|| bad("empty trace"))?;
        if &m != TRACE_MAGIC {
            return Err(bad("not an fn18-route-trace/1 file"));
        }
        let h: [u8; 16] = rd(&mut r)?.ok_or_else(|| bad("no header"))?;
        let u = |o: usize| u32::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3]]);
        Ok(TraceReader {
            r,
            layers: u(0),
            topk: u(4),
            experts: u(8),
            keep_raw: false,
            raw: Vec::new(),
        })
    }

    /// The next record (`None` at the end).
    pub fn next_step(&mut self) -> std::io::Result<Option<Step>> {
        let Some(h) = rd::<28>(&mut self.r)? else {
            return Ok(None);
        };
        let kind = StepKind::from_code(h[0]).ok_or_else(|| bad("unknown record kind"))?;
        let u16a = |o: usize| u32::from(u16::from_le_bytes([h[o], h[o + 1]]));
        let u32a = |o: usize| u32::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3]]);
        let mut tb = [0u8; 8];
        tb.copy_from_slice(&h[16..24]);
        let (bucket, width, rows, live) = (u16a(2), u16a(4), u32a(8), u32a(12));
        let n = u32a(24) as usize;
        let mut p = vec![0u8; n];
        self.r.read_exact(&mut p)?;
        let u16s: Vec<u16> = if kind == StepKind::Ple {
            Vec::new()
        } else {
            p.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        };
        let mut st = Step {
            kind,
            bucket,
            width,
            rows,
            live,
            t: f64::from_le_bytes(tb),
            off: vec![0],
            ents: Vec::new(),
            ple: Vec::new(),
        };
        let k = self.topk as usize;
        match kind {
            StepKind::Prefill => {
                let mut at = 0;
                for _ in 0..self.layers {
                    let c = *u16s.get(at).ok_or_else(|| bad("short prefill record"))? as usize;
                    at += 1;
                    for _ in 0..c {
                        st.ents.push(Ent {
                            e: u16s[at],
                            live: u16s[at + 1],
                            pad: 0,
                        });
                        at += 2;
                    }
                    st.off.push(st.ents.len() as u32);
                }
            }
            StepKind::Decode => {
                let per = rows as usize * k;
                if u16s.len() != per * self.layers as usize {
                    return Err(bad("decode record length"));
                }
                let lv = live as usize * k;
                for l in 0..self.layers as usize {
                    let s = &u16s[l * per..(l + 1) * per];
                    let mut a = Vec::new();
                    hist_into(&mut a, &s[..lv], false);
                    let mut b = Vec::new();
                    hist_into(&mut b, &s[lv..], true);
                    st.ents.extend(merge_sorted(&a, &b));
                    st.off.push(st.ents.len() as u32);
                }
                if self.keep_raw {
                    self.raw = u16s;
                }
            }
            StepKind::MtpDecode | StepKind::MtpPrefill => {
                hist_into(&mut st.ents, &u16s, false);
                st.off.push(st.ents.len() as u32);
            }
            StepKind::Ple => {
                st.ple = p
                    .chunks_exact(8)
                    .map(|c| {
                        let mut a = [0u8; 8];
                        a.copy_from_slice(c);
                        u64::from_le_bytes(a)
                    })
                    .collect();
            }
        }
        Ok(Some(st))
    }
}

fn bad(w: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, w.to_string())
}

/// A time window of the trace (a corpus of the capture): records with `t0 <= t < t1`.
#[derive(Debug, Clone, PartialEq)]
pub struct Window {
    /// Name (the corpus / traffic class).
    pub name: String,
    /// Start (unix seconds).
    pub t0: f64,
    /// End.
    pub t1: f64,
}

impl Window {
    /// Whether the step is inside.
    #[must_use]
    pub fn holds(&self, t: f64) -> bool {
        t >= self.t0 && t < self.t1
    }
    /// Parse the capture driver's windows file (`[{"name", "t0", "t1"}, ..]`).
    pub fn parse_all(text: &str) -> Result<Vec<Window>, String> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        v.as_array()
            .ok_or("windows: not an array")?
            .iter()
            .map(|w| {
                Ok(Window {
                    name: w["name"].as_str().ok_or("window name")?.to_string(),
                    t0: w["t0"].as_f64().ok_or("window t0")?,
                    t1: w["t1"].as_f64().ok_or("window t1")?,
                })
            })
            .collect()
    }
}

/// Routing counts: `counts[layer][global expert]` (tokens routed; live rows only), plus the
/// per-step OCCURRENCE counts (steps in which the expert was routed at all, pads included -- the
/// tier's unit: one link crossing per RAM expert per step).
#[derive(Debug, Clone, PartialEq)]
pub struct Counts {
    /// Tokens (live rows) counted.
    pub tokens: u64,
    /// Steps counted.
    pub steps: u64,
    /// Routed tokens per (layer, expert).
    pub routes: Vec<Vec<u64>>,
    /// Steps routing (layer, expert).
    pub occ: Vec<Vec<u64>>,
}

impl Counts {
    /// Empty counts.
    #[must_use]
    pub fn new(layers: usize, experts: usize) -> Counts {
        Counts {
            tokens: 0,
            steps: 0,
            routes: vec![vec![0; experts]; layers],
            occ: vec![vec![0; experts]; layers],
        }
    }
    /// Add a step.
    pub fn add(&mut self, s: &Step) {
        self.steps += 1;
        self.tokens += u64::from(s.live);
        for l in 0..s.layers().min(self.routes.len()) {
            for x in s.layer(l) {
                self.routes[l][x.e as usize] += u64::from(x.live);
                self.occ[l][x.e as usize] += 1;
            }
        }
    }
    /// Add another count set.
    pub fn merge(&mut self, o: &Counts) {
        self.tokens += o.tokens;
        self.steps += o.steps;
        for (a, b) in self.routes.iter_mut().zip(&o.routes) {
            a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
        }
        for (a, b) in self.occ.iter_mut().zip(&o.occ) {
            a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
        }
    }
    /// The `fn7-route-profile/1` document the tier warm-starts from (token counts, or the
    /// per-step occurrence counts with `by_steps`).
    #[must_use]
    pub fn profile_json(&self, source: &str, by_steps: bool) -> String {
        let c = if by_steps { &self.occ } else { &self.routes };
        serde_json::json!({
            "format": PROFILE_FORMAT,
            "source": source,
            "tokens": self.tokens,
            "steps": self.steps,
            "unit": if by_steps { "steps" } else { "tokens" },
            "layers": c.len(),
            "experts": c.first().map_or(0, Vec::len),
            "counts": c,
        })
        .to_string()
    }
    /// As a [`RouteProfile`].
    #[must_use]
    pub fn profile(&self, source: &str, by_steps: bool) -> RouteProfile {
        RouteProfile {
            source: source.to_string(),
            tokens: self.tokens,
            counts: if by_steps {
                self.occ.clone()
            } else {
                self.routes.clone()
            },
        }
    }
}

/// Shannon entropy (bits) of a count row.
#[must_use]
pub fn entropy_bits(row: &[u64]) -> f64 {
    let t: u64 = row.iter().sum();
    if t == 0 {
        return 0.0;
    }
    row.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / t as f64;
            -p * p.log2()
        })
        .sum()
}

/// Fraction of a row's total the top `n` entries take.
#[must_use]
pub fn top_share(row: &[u64], n: usize) -> f64 {
    let t: u64 = row.iter().sum();
    if t == 0 {
        return 0.0;
    }
    let mut v = row.to_vec();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v.iter().take(n).sum::<u64>() as f64 / t as f64
}

fn ranks(v: &[u64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].cmp(&v[b]));
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0;
        for &k in &idx[i..=j] {
            r[k] = avg;
        }
        i = j + 1;
    }
    r
}

/// Spearman rank correlation of two count rows (ties averaged).
#[must_use]
pub fn spearman(a: &[u64], b: &[u64]) -> f64 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = ra.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in ra.iter().zip(&rb) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    if saa == 0.0 || sbb == 0.0 {
        0.0
    } else {
        sab / (saa * sbb).sqrt()
    }
}

/// The residency ceiling: the best STATIC placement of `slots` VRAM slots over one rank's
/// `(layer, local expert)` records for the given per-step occurrence counts (the tier's unit) --
/// the global water-fill with the tier's per-layer floor over the counts themselves, i.e. the
/// fraction of routed (layer, expert, step) triples a placement that knew the traffic in advance
/// would serve from VRAM. Returns (overall, per layer).
#[must_use]
pub fn static_ceiling(
    occ: &[Vec<u64>],
    rank: u32,
    per_rank: u32,
    slots: u32,
    floor: u32,
) -> (f64, Vec<f64>) {
    let layers = occ.len();
    let lo = (rank * per_rank) as usize;
    let mut ranked: Vec<(u64, usize, usize)> = Vec::new();
    for (l, row) in occ.iter().enumerate() {
        for e in 0..per_rank as usize {
            ranked.push((row[lo + e], l, e));
        }
    }
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut inv = vec![vec![false; per_rank as usize]; layers];
    let mut taken = vec![0u32; layers];
    for &(_, l, e) in &ranked {
        if taken[l] < floor {
            taken[l] += 1;
            inv[l][e] = true;
        }
    }
    let mut left = slots.saturating_sub(floor * layers as u32);
    for &(_, l, e) in &ranked {
        if left == 0 {
            break;
        }
        if !inv[l][e] {
            inv[l][e] = true;
            left -= 1;
        }
    }
    let (mut hit, mut tot) = (0u64, 0u64);
    let mut per = Vec::with_capacity(layers);
    for (l, row) in occ.iter().enumerate() {
        let (mut h, mut t) = (0u64, 0u64);
        for e in 0..per_rank as usize {
            t += row[lo + e];
            if inv[l][e] {
                h += row[lo + e];
            }
        }
        per.push(if t == 0 { 1.0 } else { h as f64 / t as f64 });
        hit += h;
        tot += t;
    }
    (
        if tot == 0 {
            1.0
        } else {
            hit as f64 / tot as f64
        },
        per,
    )
}

/// A mover that only counts (the replay: the tier's decisions do not depend on bytes).
#[derive(Debug, Default)]
pub struct CountMover {
    /// RAM -> VRAM copies.
    pub h2d: u64,
    /// VRAM -> RAM copies.
    pub d2h: u64,
    /// NVMe reads.
    pub nvme: u64,
}

impl ExpertMover for CountMover {
    fn nvme_to_ram(&mut self, reads: &[(u64, u32)]) -> Result<(), TierError> {
        self.nvme += reads.len() as u64;
        Ok(())
    }
    fn ram_to_vram(&mut self, _ram: u32, _vram: u32) -> Result<(), TierError> {
        self.h2d += 1;
        Ok(())
    }
    fn vram_to_ram(&mut self, _vram: u32, _ram: u32) -> Result<(), TierError> {
        self.d2h += 1;
        Ok(())
    }
    fn flush(&mut self) -> Result<(), TierError> {
        Ok(())
    }
}

/// The geometry-only header a replay tier is built over (the served pack's: 48 layers, 256
/// experts a rank at EP=2, one 2,617,344-B stride; the sub-record offsets do not decide anything).
#[must_use]
pub fn replay_header(rank: u32, layers: u32, per_rank: u32, stride: u64) -> XtierHeader {
    XtierHeader {
        rank,
        layers,
        first_expert: rank * per_rank,
        experts: per_rank,
        stride,
        payload: stride,
        sub: [(0, 0); 6],
        hidden: 0,
        inter: 0,
        table_off: 0,
        table_bytes: 0,
        data_off: 0,
        file_bytes: 0,
        table_sha256: [0; 32],
        plan_sha256: [0; 32],
        split_parts: 0,
        split_index: 0,
        format: crate::xtier::XtierFormat::Mxfp4E8m0,
    }
}

/// Per-bucket replay totals (decode buckets 1/2/4/8; bucket 0 = prefill chunks).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BucketTotals {
    /// Steps.
    pub steps: u64,
    /// Which-path counts.
    pub paths: PathCounts,
    /// Promotions (each one H2D of a stride).
    pub inserts: u64,
    /// FN36: D2H victim copies (FN8's law: one per promotion into a full slot).
    pub demotions: u64,
    /// FN36: victims re-read from NVMe instead (the demotion-free law).
    pub refills: u64,
}

/// A replay's result for one rank.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayOut {
    /// Per layer.
    pub per_layer: Vec<PathCounts>,
    /// Keyed by bucket (0 = prefill).
    pub buckets: std::collections::BTreeMap<u32, BucketTotals>,
    /// Steps replayed.
    pub steps: u64,
}

/// The per-bucket insert cap (`--xtier-max-inserts-by-bucket`); buckets not listed take `default`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsertLaw {
    /// Default cap (every prefill chunk and any unlisted bucket).
    pub default: u32,
    /// `(bucket, cap)`.
    pub by_bucket: Vec<(u32, u32)>,
    /// FN54: `(bucket, promotion byte budget a step)` (`--xtier-promote-budget-mib-by-bucket`);
    /// buckets not listed (and every prefill chunk) have none.
    pub budget_by_bucket: Vec<(u32, u64)>,
    /// FN54: the second-chance window (steps; 0 = off) of a prefill chunk / an unlisted bucket,
    /// and `(bucket, window)` (`--xtier-second-chance W` or `bucket:W,..`).
    pub sc_default: u32,
    /// See [`Self::sc_default`].
    pub sc_by_bucket: Vec<(u32, u32)>,
}

impl InsertLaw {
    /// The cap for a decode bucket (0 = a prefill chunk: the default).
    #[must_use]
    pub fn cap(&self, bucket: u32) -> u32 {
        if bucket == 0 {
            return self.default;
        }
        self.by_bucket
            .iter()
            .find(|(b, _)| *b == bucket)
            .map_or(self.default, |x| x.1)
    }
    /// Parse `"1:2,2:2,4:2,8:1"`.
    pub fn parse(default: u32, s: &str) -> Result<InsertLaw, String> {
        let mut by_bucket = Vec::new();
        for part in s.split(',').filter(|p| !p.trim().is_empty()) {
            let (b, c) = part
                .split_once(':')
                .ok_or_else(|| format!("'{part}': want bucket:cap"))?;
            let b: u32 = b.trim().parse().map_err(|_| format!("bucket '{b}'"))?;
            let c: u32 = c.trim().parse().map_err(|_| format!("cap '{c}'"))?;
            if ![1, 2, 4, 8].contains(&b) {
                return Err(format!("bucket {b} is not a decode bucket (1/2/4/8)"));
            }
            by_bucket.push((b, c));
        }
        Ok(InsertLaw {
            default,
            by_bucket,
            budget_by_bucket: Vec::new(),
            sc_default: 0,
            sc_by_bucket: Vec::new(),
        })
    }

    /// FN54: the second-chance window of a step's bucket (0 = a prefill chunk / unlisted: the
    /// default).
    #[must_use]
    pub fn second_chance(&self, bucket: u32) -> u32 {
        if bucket == 0 {
            return self.sc_default;
        }
        self.sc_by_bucket
            .iter()
            .find(|(b, _)| *b == bucket)
            .map_or(self.sc_default, |x| x.1)
    }

    /// FN54: add the second-chance law: `"W"` (every step) or `"4:W,8:W"` (those decode buckets
    /// only; empty / `"0"` = off).
    pub fn with_second_chance(mut self, s: &str) -> Result<InsertLaw, String> {
        let s = s.trim();
        if !s.contains(':') {
            self.sc_default = if s.is_empty() {
                0
            } else {
                s.parse().map_err(|_| format!("window '{s}'"))?
            };
            return Ok(self);
        }
        for part in s.split(',').filter(|p| !p.trim().is_empty()) {
            let (b, c) = part
                .split_once(':')
                .ok_or_else(|| format!("'{part}': want bucket:steps"))?;
            let b: u32 = b.trim().parse().map_err(|_| format!("bucket '{b}'"))?;
            let c: u32 = c.trim().parse().map_err(|_| format!("steps '{c}'"))?;
            if ![1, 2, 4, 8].contains(&b) {
                return Err(format!("bucket {b} is not a decode bucket (1/2/4/8)"));
            }
            self.sc_by_bucket.push((b, c));
        }
        Ok(self)
    }

    /// FN54: the promotion byte budget of a decode bucket (`None` = none: a prefill chunk, an
    /// unlisted bucket, or no budget flag -- today's law).
    #[must_use]
    pub fn budget(&self, bucket: u32) -> Option<u64> {
        if bucket == 0 {
            return None;
        }
        self.budget_by_bucket
            .iter()
            .find(|(b, _)| *b == bucket)
            .map(|x| x.1)
    }

    /// FN54: add the per-bucket promotion budget, `"1:A,2:B,4:C,8:D"` in MiB a step (empty = none).
    pub fn with_budget_mib(mut self, s: &str) -> Result<InsertLaw, String> {
        for part in s.split(',').filter(|p| !p.trim().is_empty()) {
            let (b, c) = part
                .split_once(':')
                .ok_or_else(|| format!("'{part}': want bucket:MiB"))?;
            let b: u32 = b.trim().parse().map_err(|_| format!("bucket '{b}'"))?;
            let c: u64 = c.trim().parse().map_err(|_| format!("MiB '{c}'"))?;
            if ![1, 2, 4, 8].contains(&b) {
                return Err(format!("bucket {b} is not a decode bucket (1/2/4/8)"));
            }
            self.budget_by_bucket.push((b, c << 20));
        }
        Ok(self)
    }
}

/// A global-id remap per layer (the owner permutation): `perm[layer][old global id]` = the id
/// under which the record lives in the re-owned pack (rank = id / per_rank).
pub type Perm = Vec<Vec<u16>>;

/// Replay one rank's tier over a trace's TARGET steps (optionally only inside `window`), from the
/// warm start of `profile` at `params`, with the per-bucket insert law and an optional owner
/// permutation. Deterministic (I2): with the served warm start and law it reproduces the served
/// counters exactly.
#[allow(clippy::too_many_arguments)]
pub fn replay(
    trace: &Path,
    rank: u32,
    per_rank: u32,
    stride: u64,
    params: &TierParams,
    profile: Option<&RouteProfile>,
    law: &InsertLaw,
    window: Option<&[Window]>,
    perm: Option<&Perm>,
) -> Result<ReplayOut, String> {
    let mut r = TraceReader::open(trace).map_err(|e| e.to_string())?;
    let h = replay_header(rank, r.layers, per_rank, stride);
    let mut tier =
        ExpertTier::new(&h, params.clone(), profile, 0, 1 << 40).map_err(|e| e.to_string())?;
    let mut mv = CountMover::default();
    let mut out = ReplayOut {
        per_layer: vec![PathCounts::default(); r.layers as usize],
        buckets: std::collections::BTreeMap::new(),
        steps: 0,
    };
    while let Some(s) = r.next_step().map_err(|e| e.to_string())? {
        if !s.kind.target() {
            continue;
        }
        if let Some(ws) = window {
            if !ws.iter().any(|w| w.holds(s.t)) {
                continue;
            }
        }
        let bucket = if s.kind == StepKind::Prefill {
            0
        } else {
            s.bucket
        };
        tier.set_max_inserts(law.cap(bucket));
        tier.set_promote_budget(law.budget(bucket));
        tier.set_second_chance(law.second_chance(bucket));
        tier.begin_step();
        let mut paths = PathCounts::default();
        for l in 0..s.layers() {
            // FN48: a row-weighted policy sees each record once per live row (the engine's
            // routed list carries every row's top-k); every other policy dedups it
            let reps = params.policy.row_gamma != 0.0;
            let routed: Vec<u32> = s
                .layer(l)
                .iter()
                .flat_map(|x| {
                    let e = u32::from(perm.map_or(x.e, |p| p[l][x.e as usize]));
                    std::iter::repeat_n(e, if reps { usize::from(x.live.max(1)) } else { 1 })
                })
                .collect();
            let c = tier
                .resolve(l as u32, &routed, &mut mv)
                .map_err(|e| e.to_string())?;
            out.per_layer[l].resident += c.resident;
            out.per_layer[l].ram += c.ram;
            out.per_layer[l].nvme += c.nvme;
            paths.resident += c.resident;
            paths.ram += c.ram;
            paths.nvme += c.nvme;
        }
        // FN36: the engine issues a demotion-free step's refills under the next step
        tier.refill(&mut mv).map_err(|e| e.to_string())?;
        let rep = tier.end_step(&mut mv).map_err(|e| e.to_string())?;
        let b = out.buckets.entry(bucket).or_default();
        b.steps += 1;
        b.paths.resident += paths.resident;
        b.paths.ram += paths.ram;
        b.paths.nvme += paths.nvme;
        b.inserts += rep.moves.inserts;
        b.demotions += rep.moves.demotions;
        b.refills += rep.moves.refills;
        out.steps += 1;
    }
    Ok(out)
}

/// FN41: the form of the MIN (Belady) oracle [`min_replay`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinForm {
    /// Today's cadence: the tier's per-layer VRAM partitions (the same warm start), promotions at
    /// the step's end only, the read-through fraction and the per-bucket insert cap, victims never
    /// routed this step. The oracle picks WHICH misses fill the cap (soonest next use) and which
    /// residents leave (farthest next use), and bypasses a miss not reused before its victim.
    Capped,
    /// The pure bound over the same partitions: no cap, no read-through; at every step's end each
    /// layer keeps the `n` records (residents + this step's misses) with the soonest next use.
    Pure,
    /// The pure bound over ONE rank-wide pool (the per-layer partition dropped).
    Global,
    /// The diagnostic of [`MinForm::Capped`]: the same victims (farthest next use) but EVERY
    /// capped miss is promoted (no bypass) -- today's admission, MIN's victim ranking.
    CappedAdmitAll,
}

/// The step-end pairing of MIN: `misses` (next use, record) ascending, victims taken from the
/// empty slots first, then from `set` in descending next use (skipping `skip`); a pair is made
/// while the miss is reused strictly before its victim. Returns (inserts, of them into full slots).
fn min_pair(
    misses: &[(u32, u32)],
    free: &mut u32,
    set: &mut std::collections::BTreeSet<(u32, u32)>,
    in_v: &mut [bool],
    skip: &dyn Fn(u32) -> bool,
    admit_all: bool,
) -> (u64, u64) {
    let (mut ins, mut full) = (0u64, 0u64);
    let mut victims = Vec::new();
    {
        let mut it = set.iter().rev().filter(|(_, r)| !skip(*r));
        for &(mn, _) in misses {
            if mn == u32::MAX && !admit_all {
                break; // never reused: bypass
            }
            if *free > 0 {
                *free -= 1;
            } else {
                match it.next() {
                    Some(&(vn, v)) if admit_all || mn < vn => victims.push((vn, v)),
                    _ => break,
                }
                full += 1;
            }
            ins += 1;
        }
    }
    for v in &victims {
        set.remove(v);
        in_v[v.1 as usize] = false;
    }
    for &(mn, m) in misses.iter().take(ins as usize) {
        in_v[m as usize] = true;
        set.insert((mn, m));
    }
    (ins, full)
}

/// FN41: the MIN (Belady) oracle over the same trace, rank, warm start and slots as [`replay`]
/// (future knowledge: the victim is the resident whose next use is farthest). The counts are the
/// tier's: `resident` = a VRAM hit, `ram` = every other route (the tier is complete); `inserts` =
/// promotions, `demotions` = promotions into a full slot (FN8's D2H). Demand-only (a record enters
/// VRAM only after a step routed it), so the bound is on a demand policy's hit rate -- what any
/// victim / admission ranking can reach. [`MinForm::Pure`] is >= every such policy at the same
/// partitions (MIN's optimality); [`MinForm::Capped`] shares today's cap and cadence.
#[allow(clippy::too_many_arguments)]
pub fn min_replay(
    trace: &Path,
    rank: u32,
    per_rank: u32,
    stride: u64,
    params: &TierParams,
    profile: Option<&RouteProfile>,
    law: &InsertLaw,
    window: Option<&[Window]>,
    form: MinForm,
) -> Result<ReplayOut, String> {
    let mut r = TraceReader::open(trace).map_err(|e| e.to_string())?;
    let layers = r.layers as usize;
    let records = layers * per_rank as usize;
    let first = rank * per_rank;
    // Pass 1: the step's distinct routed records per layer (this rank's).
    let (mut recs, mut off, mut bucket) = (Vec::<u32>::new(), vec![0u32], Vec::<u32>::new());
    while let Some(s) = r.next_step().map_err(|e| e.to_string())? {
        if !s.kind.target() {
            continue;
        }
        if let Some(ws) = window {
            if !ws.iter().any(|w| w.holds(s.t)) {
                continue;
            }
        }
        bucket.push(if s.kind == StepKind::Prefill {
            0
        } else {
            s.bucket
        });
        for l in 0..layers {
            let at = recs.len();
            for x in s.layer(l) {
                let e = u32::from(x.e);
                if e >= first && e < first + per_rank {
                    recs.push(l as u32 * per_rank + (e - first));
                }
            }
            let mut v = recs.split_off(at);
            v.sort_unstable();
            v.dedup();
            recs.extend(v);
            off.push(recs.len() as u32);
        }
    }
    let steps = bucket.len();
    // The next use of every occurrence (u32::MAX = never), and each record's first use.
    let mut nxt_of = vec![u32::MAX; recs.len()];
    let mut nxt = vec![u32::MAX; records];
    for s in (0..steps).rev() {
        let (a, b) = (off[s * layers] as usize, off[(s + 1) * layers] as usize);
        for i in a..b {
            let rec = recs[i] as usize;
            nxt_of[i] = nxt[rec];
            nxt[rec] = s as u32;
        }
    }
    // The warm start: the tier's own partitions and placement.
    let h = replay_header(rank, r.layers, per_rank, stride);
    let tier =
        ExpertTier::new(&h, params.clone(), profile, 0, 1 << 40).map_err(|e| e.to_string())?;
    let global = form == MinForm::Global;
    let parts = if global { 1 } else { layers };
    let mut free = vec![0u32; parts];
    let mut sets = vec![std::collections::BTreeSet::<(u32, u32)>::new(); parts];
    let mut in_v = vec![false; records];
    let mut size = vec![0u32; layers];
    for (l, sz) in size.iter_mut().enumerate() {
        let (base, n) = tier.layer_slots(l as u32);
        *sz = n;
        let p = if global { 0 } else { l };
        for slot in base..base + n {
            match tier.vram_record(slot) {
                Some(rec) => {
                    in_v[rec as usize] = true;
                    sets[p].insert((nxt[rec as usize], rec));
                }
                None => free[p] += 1,
            }
        }
    }
    let mut touched = vec![u32::MAX; records];
    let mut out = ReplayOut {
        per_layer: vec![PathCounts::default(); layers],
        buckets: std::collections::BTreeMap::new(),
        steps: 0,
    };
    let mut misses: Vec<Vec<(u32, u32)>> = vec![Vec::new(); parts];
    for s in 0..steps {
        let now = s as u32;
        let mut paths = PathCounts::default();
        let (mut ins, mut full) = (0u64, 0u64);
        for m in &mut misses {
            m.clear();
        }
        let cap = law.cap(bucket[s]);
        for l in 0..layers {
            let (a, b) = (
                off[s * layers + l] as usize,
                off[s * layers + l + 1] as usize,
            );
            let p = if global { 0 } else { l };
            for i in a..b {
                let rec = recs[i];
                touched[rec as usize] = now;
                if in_v[rec as usize] {
                    out.per_layer[l].resident += 1;
                    paths.resident += 1;
                    sets[p].remove(&(nxt[rec as usize], rec));
                    sets[p].insert((nxt_of[i], rec));
                } else {
                    out.per_layer[l].ram += 1;
                    paths.ram += 1;
                    misses[p].push((nxt_of[i], rec));
                }
                nxt[rec as usize] = nxt_of[i];
            }
            if matches!(form, MinForm::Capped | MinForm::CappedAdmitAll) {
                let n = size[l];
                if n == 0 || b == a || (b - a) as f64 > params.read_through * f64::from(n) {
                    continue;
                }
                let m = &mut misses[l];
                m.sort_unstable();
                m.truncate(cap as usize);
                let t = &touched;
                let all = form == MinForm::CappedAdmitAll;
                let (x, y) = min_pair(
                    m,
                    &mut free[l],
                    &mut sets[l],
                    &mut in_v,
                    &|r| t[r as usize] == now,
                    all,
                );
                ins += x;
                full += y;
            }
        }
        if matches!(form, MinForm::Pure | MinForm::Global) {
            for p in 0..parts {
                let m = &mut misses[p];
                m.sort_unstable();
                let (x, y) = min_pair(m, &mut free[p], &mut sets[p], &mut in_v, &|_| false, false);
                ins += x;
                full += y;
            }
        }
        let t = out.buckets.entry(bucket[s]).or_default();
        t.steps += 1;
        t.paths.resident += paths.resident;
        t.paths.ram += paths.ram;
        t.inserts += ins;
        t.demotions += full;
        out.steps += 1;
    }
    Ok(out)
}

/// FN48: the value a learned victim table stores per (class, age).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableMode {
    /// Minus the expected time to the next use, capped at the horizon (Belady's rule in
    /// expectation: the farthest expected next use leaves).
    Etr,
    /// The probability of a use within the horizon (the least likely leaves).
    Prob,
    /// LHD's hit density: uses within the horizon per step of (truncated) remaining lifetime.
    Density,
}

/// FN48: learn the victim table on `trace` (the tier's own touch law: every target step's
/// distinct routes of this rank, a read-through layer's flagged by the tier's partitions at
/// `params`). For every inter-reference interval of a record (class = its decayed use count with
/// `lam` + the read-through flag at the interval's start) every age it lives through is a sample
/// of (age, remaining time); intervals still open at the trace's end count as "not within the
/// horizon". Returns `TABLE_CLASSES x TABLE_AGES` values (higher = keep).
#[allow(clippy::too_many_arguments)]
pub fn learn_table(
    trace: &Path,
    rank: u32,
    per_rank: u32,
    stride: u64,
    params: &TierParams,
    profile: Option<&RouteProfile>,
    window: Option<&[Window]>,
    lam: f64,
    horizon: u64,
    mode: TableMode,
) -> Result<Vec<f64>, String> {
    let mut r = TraceReader::open(trace).map_err(|e| e.to_string())?;
    let layers = r.layers as usize;
    let records = layers * per_rank as usize;
    let first = rank * per_rank;
    let h = replay_header(rank, r.layers, per_rank, stride);
    let tier =
        ExpertTier::new(&h, params.clone(), profile, 0, 1 << 40).map_err(|e| e.to_string())?;
    let size: Vec<u32> = (0..layers).map(|l| tier.layer_slots(l as u32).1).collect();
    let gmax = 1usize << 17;
    let mut hist = vec![vec![0u64; gmax + 1]; TABLE_CLASSES];
    let mut cens = vec![vec![0u64; gmax + 1]; TABLE_CLASSES];
    let mut last = vec![u64::MAX; records];
    let mut lc = vec![0f64; records];
    let mut cls = vec![0usize; records];
    let mut now = 0u64;
    let mut v = Vec::new();
    while let Some(s) = r.next_step().map_err(|e| e.to_string())? {
        if !s.kind.target() {
            continue;
        }
        if let Some(ws) = window {
            if !ws.iter().any(|w| w.holds(s.t)) {
                continue;
            }
        }
        for (l, &n) in size.iter().enumerate() {
            v.clear();
            for x in s.layer(l) {
                let e = u32::from(x.e);
                if e >= first && e < first + per_rank {
                    v.push(l * per_rank as usize + (e - first) as usize);
                }
            }
            v.sort_unstable();
            v.dedup();
            let rt = n > 0 && v.len() as f64 > params.read_through * f64::from(n);
            for &i in &v {
                if last[i] == u64::MAX {
                    lc[i] = 1.0;
                } else {
                    let g = (now - last[i]) as usize;
                    hist[cls[i]][g.min(gmax)] += 1;
                    lc[i] = lc[i] * lam.powi(g.min(1 << 20) as i32) + 1.0;
                }
                cls[i] = table_class(lc[i], rt);
                last[i] = now;
            }
        }
        now += 1;
    }
    for i in 0..records {
        if last[i] != u64::MAX {
            cens[cls[i]][((now - last[i]) as usize).min(gmax)] += 1;
        }
    }
    let hz = horizon as usize;
    let mut out = vec![0f64; TABLE_CLASSES * TABLE_AGES];
    for c in 0..TABLE_CLASSES {
        // prefix sums over gaps: count, gap-weighted count; censored count
        let mut cn = vec![0f64; gmax + 2];
        let mut wn = vec![0f64; gmax + 2];
        let mut xn = vec![0f64; gmax + 2];
        for g in 0..=gmax {
            cn[g + 1] = cn[g] + hist[c][g] as f64;
            wn[g + 1] = wn[g] + (g as f64) * hist[c][g] as f64;
            xn[g + 1] = xn[g] + cens[c][g] as f64;
        }
        // sums over g in (a, b]: prefix[b + 1] - prefix[a + 1]
        let rng = |p: &Vec<f64>, a: usize, b: usize| p[b.min(gmax) + 1] - p[a.min(gmax) + 1];
        let mut num = vec![0f64; TABLE_AGES];
        let mut den = vec![0f64; TABLE_AGES];
        for a in 0..gmax {
            let alive = rng(&cn, a, gmax) + rng(&xn, a, gmax);
            if alive == 0.0 {
                continue;
            }
            let within = rng(&cn, a, a + hz);
            let life = rng(&wn, a, a + hz) - a as f64 * within + hz as f64 * (alive - within);
            let (val, d) = match mode {
                TableMode::Prob => (within, alive),
                TableMode::Etr => (-life, alive),
                TableMode::Density => (within, life),
            };
            let b = table_age(a as u64);
            num[b] += val;
            den[b] += d;
        }
        for b in 0..TABLE_AGES {
            out[table_index(c, b)] = if den[b] > 0.0 {
                num[b] / den[b]
            } else {
                match mode {
                    TableMode::Prob | TableMode::Density => 0.0,
                    TableMode::Etr => -(hz as f64),
                }
            };
        }
    }
    Ok(out)
}

/// The owner permutation: per layer, experts sorted by `occ` (descending), each assigned to the
/// rank whose running load is lower (ties: rank 0), `per_rank` a rank -- the balanced static split
/// of plan §6 (`layout.owner`). Returns the remap (old global id -> new global id) with every
/// rank's new ids in ascending order of the old ids (a record keeps its relative order).
#[must_use]
pub fn balanced_owner(occ: &[Vec<u64>], per_rank: u32, ranks: u32) -> Perm {
    let n = (per_rank * ranks) as usize;
    occ.iter()
        .map(|row| {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| row[b].cmp(&row[a]).then(a.cmp(&b)));
            let mut load = vec![0u64; ranks as usize];
            let mut cnt = vec![0u32; ranks as usize];
            let mut owner = vec![0u32; n];
            for &e in &idx {
                let r = (0..ranks as usize)
                    .filter(|&r| cnt[r] < per_rank)
                    .min_by_key(|&r| (load[r], r))
                    .expect("a rank has room");
                owner[e] = r as u32;
                load[r] += row[e];
                cnt[r] += 1;
            }
            let mut next = (0..ranks).map(|r| r * per_rank).collect::<Vec<u32>>();
            let mut p = vec![0u16; n];
            for (e, &o) in owner.iter().enumerate() {
                p[e] = next[o as usize] as u16;
                next[o as usize] += 1;
            }
            p
        })
        .collect()
}

/// The EP balance of the decode steps: per step per layer the distinct routed experts on each
/// rank; returns (steps, Σ max_r d_r, Σ mean_r d_r) over every (step, layer) -- the step waits
/// for the larger half at every layer's reduce.
pub fn ep_balance(
    trace: &Path,
    per_rank: u32,
    window: Option<&[Window]>,
    perm: Option<&Perm>,
) -> Result<(u64, u64, f64), String> {
    let mut r = TraceReader::open(trace).map_err(|e| e.to_string())?;
    let (mut steps, mut smax, mut smean) = (0u64, 0u64, 0f64);
    while let Some(s) = r.next_step().map_err(|e| e.to_string())? {
        if s.kind != StepKind::Decode {
            continue;
        }
        if let Some(ws) = window {
            if !ws.iter().any(|w| w.holds(s.t)) {
                continue;
            }
        }
        steps += 1;
        for l in 0..s.layers() {
            let mut d = [0u64; 2];
            for x in s.layer(l) {
                let id = perm.map_or(x.e, |p| p[l][x.e as usize]);
                d[(u32::from(id) / per_rank).min(1) as usize] += 1;
            }
            smax += d[0].max(d[1]);
            smean += (d[0] + d[1]) as f64 / 2.0;
        }
    }
    Ok((steps, smax, smean))
}

/// The PLE row cache over a trace's gathers at `rows` capacity: (requested, distinct per gather
/// summed, cache hits, drive reads, drive reads in decode gathers, decode gathers).
pub fn ple_replay(
    trace: &Path,
    rows: u32,
    window: Option<&[Window]>,
    passes: u32,
) -> Result<Vec<[u64; 6]>, String> {
    let mut cache = crate::ple_tier::RowCache::new(rows, 0);
    let mut out = Vec::new();
    for _ in 0..passes {
        let mut r = TraceReader::open(trace).map_err(|e| e.to_string())?;
        let mut c = [0u64; 6];
        while let Some(s) = r.next_step().map_err(|e| e.to_string())? {
            if s.kind != StepKind::Ple {
                continue;
            }
            if let Some(ws) = window {
                if !ws.iter().any(|w| w.holds(s.t)) {
                    continue;
                }
            }
            c[0] += s.ple.len() as u64;
            let mut seen: HashMap<u64, ()> = HashMap::with_capacity(s.ple.len());
            let mut miss = 0u64;
            for &row in &s.ple {
                if seen.insert(row, ()).is_some() {
                    continue;
                }
                c[1] += 1;
                if cache.get(row).is_some() {
                    c[2] += 1;
                } else {
                    miss += 1;
                    cache.insert(row, &[]);
                }
            }
            c[3] += miss;
            if s.width == 1 {
                c[4] += miss;
                c[5] += 1;
            }
        }
        out.push(c);
    }
    Ok(out)
}

/// FN58: how a device-resident PLE hot table is filled.
#[derive(Debug, Clone)]
pub enum PleDevPolicy {
    /// A STATIC hot set (chosen on another trace: the held-out law), never changed while serving.
    Static(std::collections::HashSet<u64>),
    /// An LRU over every row the host stages (each staged miss lands in the table beside the
    /// step's rows -- no extra bytes cross the link).
    Lru,
}

/// FN58: one gather class of [`ple_device_replay`] (b1 / b8 / all decode steps).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PleDevCell {
    /// Decode steps.
    pub steps: u64,
    /// Distinct rows a step (summed).
    pub distinct: u64,
    /// Of those, served by the device table.
    pub dev_hits: u64,
    /// Read from the drive (the RAM row cache sees only the device table's misses).
    pub drive: u64,
    /// Steps whose every row hit the device table (no host PLE work at all).
    pub host_free: u64,
    /// Steps with no drive read (the host's PLE work is RAM hits only -- FN56: ~0.02 ms).
    pub drive_free: u64,
}

/// One gather through the device table, the RAM row cache and the drive: (distinct rows, device
/// hits, drive reads).
fn ple_dev_gather(
    rows: &[u64],
    policy: &PleDevPolicy,
    ram: &mut crate::ple_tier::RowCache,
    dev: &mut crate::ple_tier::RowCache,
) -> (u64, u64, u64) {
    let mut seen: HashMap<u64, ()> = HashMap::with_capacity(rows.len());
    let (mut d, mut h, mut n) = (0u64, 0u64, 0u64);
    for &row in rows {
        if seen.insert(row, ()).is_some() {
            continue;
        }
        d += 1;
        let hit = match policy {
            PleDevPolicy::Static(s) => s.contains(&row),
            PleDevPolicy::Lru => dev.get(row).is_some(),
        };
        if hit {
            h += 1;
            continue;
        }
        if ram.get(row).is_none() {
            n += 1;
            ram.insert(row, &[]);
        }
        if matches!(policy, PleDevPolicy::Lru) {
            dev.insert(row, &[]);
        }
    }
    (d, h, n)
}

/// FN58: the PLE gather of every decode (verify) step of `trace` -- the step's rows are the PLE
/// records (`width` = 1) before its `Decode` record -- against a device hot table of `dev_rows`
/// rows filled by `policy`, the host's RAM row cache (`ram_rows`, LRU) behind it, the drive
/// behind both. Prefill gathers pass through the same planes but are not counted. Returns the
/// cells `[b1, b8, all]` (b2 / b4 count in `all` only).
pub fn ple_device_replay(
    trace: &Path,
    ram_rows: u32,
    dev_rows: u32,
    policy: &PleDevPolicy,
    window: Option<&[Window]>,
) -> Result<[PleDevCell; 3], String> {
    let mut ram = crate::ple_tier::RowCache::new(ram_rows, 0);
    let lru = matches!(policy, PleDevPolicy::Lru);
    let mut dev = crate::ple_tier::RowCache::new(if lru { dev_rows } else { 0 }, 0);
    let mut out = [PleDevCell::default(); 3];
    let mut pend: Vec<u64> = Vec::new();
    let mut r = TraceReader::open(trace).map_err(|e| e.to_string())?;
    while let Some(s) = r.next_step().map_err(|e| e.to_string())? {
        match s.kind {
            StepKind::Ple if s.width == 1 => pend.extend_from_slice(&s.ple),
            StepKind::Ple => {
                ple_dev_gather(&s.ple, policy, &mut ram, &mut dev);
            }
            StepKind::Decode => {
                let rows = std::mem::take(&mut pend);
                let (d, h, n) = ple_dev_gather(&rows, policy, &mut ram, &mut dev);
                if rows.is_empty() || window.is_some_and(|ws| !ws.iter().any(|w| w.holds(s.t))) {
                    continue;
                }
                let cls: &[usize] = match s.bucket {
                    1 => &[0, 2],
                    8 => &[1, 2],
                    _ => &[2],
                };
                for &k in cls {
                    let c = &mut out[k];
                    c.steps += 1;
                    c.distinct += d;
                    c.dev_hits += h;
                    c.drive += n;
                    c.host_free += u64::from(h == d);
                    c.drive_free += u64::from(n == 0);
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// FN58: the `n` most-gathered PLE rows of `trace` (distinct per gather, every gather counted:
/// prefill and decode) -- the static hot set a device table would hold.
pub fn ple_hot_rows(trace: &Path, n: usize) -> Result<std::collections::HashSet<u64>, String> {
    let mut cnt: HashMap<u64, u32> = HashMap::new();
    let mut r = TraceReader::open(trace).map_err(|e| e.to_string())?;
    while let Some(s) = r.next_step().map_err(|e| e.to_string())? {
        if s.kind != StepKind::Ple {
            continue;
        }
        let mut seen: HashMap<u64, ()> = HashMap::with_capacity(s.ple.len());
        for &row in &s.ple {
            if seen.insert(row, ()).is_none() {
                *cnt.entry(row).or_insert(0) += 1;
            }
        }
    }
    let mut v: Vec<(u64, u32)> = cnt.into_iter().collect();
    v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok(v.into_iter().take(n).map(|x| x.0).collect())
}

/// FN53a: one target step of the SSD-tail replay ([`tail_replay`]), handed to the caller's sink.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TailStep {
    /// Decode bucket (0 = a prefill chunk).
    pub bucket: u32,
    /// Rows of the step.
    pub rows: u32,
    /// Live rows.
    pub live: u32,
    /// Per layer, per replayed rank (in `ranks` order): tail records routed at the layer that the
    /// complete tier serves from RAM -- under the tail law each one is a drive read.
    pub reads: Vec<Vec<u16>>,
    /// Per replayed rank: tail records routed this step, whatever their path.
    pub tail_routed: Vec<u32>,
    /// Per replayed rank: tail records the tier promoted into VRAM at this step's end (under the
    /// tail law the promotion copies from the read's bounce buffer, not from RAM).
    pub tail_promoted: Vec<u32>,
}

/// FN53a: the SSD-tail replay. The real `ExpertTier` (complete: VRAM + the RAM complement) runs
/// over the trace's target steps for every rank in `ranks` in lock step; `tail[k]` marks the
/// records (`layer * per_rank + local`) of rank `ranks[k]` that would live on the drive instead of
/// in pinned RAM. The tier's decisions do not depend on where a record's bytes come from, so the
/// replay is exact for the law "a tail record served from RAM by the complete tier is an NVMe read
/// under the tail law" (a tail record promoted into VRAM copies from the read's bounce buffer; a
/// tail record demoted from VRAM is dropped). Returns the per-rank [`ReplayOut`]s (the complete
/// tier's own counters) and calls `sink` once per target step.
#[allow(clippy::too_many_arguments)]
pub fn tail_replay(
    trace: &Path,
    ranks: &[u32],
    per_rank: u32,
    stride: u64,
    params: &TierParams,
    profile: Option<&RouteProfile>,
    law: &InsertLaw,
    window: Option<&[Window]>,
    tail: &[Vec<bool>],
    mut sink: impl FnMut(&TailStep),
) -> Result<Vec<ReplayOut>, String> {
    if tail.len() != ranks.len() {
        return Err(format!(
            "{} tail sets for {} ranks",
            tail.len(),
            ranks.len()
        ));
    }
    let mut r = TraceReader::open(trace).map_err(|e| e.to_string())?;
    let layers = r.layers as usize;
    let mut tiers = Vec::with_capacity(ranks.len());
    for (k, &rank) in ranks.iter().enumerate() {
        if tail[k].len() != layers * per_rank as usize {
            return Err(format!(
                "rank {rank}: tail set of {} records",
                tail[k].len()
            ));
        }
        let h = replay_header(rank, r.layers, per_rank, stride);
        tiers.push(
            ExpertTier::new(&h, params.clone(), profile, 0, 1 << 40).map_err(|e| e.to_string())?,
        );
    }
    let mut mvs: Vec<CountMover> = ranks.iter().map(|_| CountMover::default()).collect();
    let mut outs: Vec<ReplayOut> = ranks
        .iter()
        .map(|_| ReplayOut {
            per_layer: vec![PathCounts::default(); layers],
            buckets: std::collections::BTreeMap::new(),
            steps: 0,
        })
        .collect();
    while let Some(s) = r.next_step().map_err(|e| e.to_string())? {
        if !s.kind.target() {
            continue;
        }
        if let Some(ws) = window {
            if !ws.iter().any(|w| w.holds(s.t)) {
                continue;
            }
        }
        let bucket = if s.kind == StepKind::Prefill {
            0
        } else {
            s.bucket
        };
        let mut ts = TailStep {
            bucket,
            rows: s.rows,
            live: s.live,
            reads: vec![vec![0; ranks.len()]; s.layers()],
            tail_routed: vec![0; ranks.len()],
            tail_promoted: vec![0; ranks.len()],
        };
        for (k, tier) in tiers.iter_mut().enumerate() {
            tier.set_max_inserts(law.cap(bucket));
            tier.set_promote_budget(law.budget(bucket));
            tier.set_second_chance(law.second_chance(bucket));
            tier.begin_step();
            let mut paths = PathCounts::default();
            let mut ram_tail: Vec<(u32, u32)> = Vec::new();
            for l in 0..s.layers() {
                let routed: Vec<u32> = s.layer(l).iter().map(|x| u32::from(x.e)).collect();
                for &e in &routed {
                    let Some(i) = tier.index(l as u32, e) else {
                        continue;
                    };
                    if !tail[k][i as usize] {
                        continue;
                    }
                    ts.tail_routed[k] += 1;
                    if matches!(tier.placement(l as u32, e), Some(Placement::Ram(_))) {
                        ts.reads[l][k] += 1;
                        ram_tail.push((l as u32, e));
                    }
                }
                let c = tier
                    .resolve(l as u32, &routed, &mut mvs[k])
                    .map_err(|e| e.to_string())?;
                outs[k].per_layer[l].resident += c.resident;
                outs[k].per_layer[l].ram += c.ram;
                outs[k].per_layer[l].nvme += c.nvme;
                paths.resident += c.resident;
                paths.ram += c.ram;
                paths.nvme += c.nvme;
            }
            tier.refill(&mut mvs[k]).map_err(|e| e.to_string())?;
            let rep = tier.end_step(&mut mvs[k]).map_err(|e| e.to_string())?;
            ts.tail_promoted[k] = ram_tail
                .iter()
                .filter(|(l, e)| matches!(tier.placement(*l, *e), Some(Placement::Vram(_))))
                .count() as u32;
            let b = outs[k].buckets.entry(bucket).or_default();
            b.steps += 1;
            b.paths.resident += paths.resident;
            b.paths.ram += paths.ram;
            b.paths.nvme += paths.nvme;
            b.inserts += rep.moves.inserts;
            b.demotions += rep.moves.demotions;
            b.refills += rep.moves.refills;
            outs[k].steps += 1;
        }
        sink(&ts);
    }
    Ok(outs)
}

/// FN53a: the coldest `n` RAM-placed records of a tier's warm start, by `key[record]` ascending
/// (ties: record index), as a tail set (`layer * per_rank + local`). Records the warm start puts
/// in VRAM are never in the tail (the tail shrinks the RAM complement only). `layer_cap` > 0 =
/// the LAYER-BALANCED tail: at most `layer_cap` records a layer in the first pass (a prefill
/// chunk's drive time per layer must fit under the layer's compute), the rest of `n` (if a layer
/// ran short of candidates) down the global order.
#[must_use]
pub fn coldest_ram_tail(
    tier: &ExpertTier,
    key: &[(f64, f64)],
    n: usize,
    layer_cap: usize,
) -> Vec<bool> {
    let records = tier.records() as usize;
    let mut tail = vec![false; records];
    let mut c: Vec<usize> = tier
        .warm_plan()
        .iter()
        .filter(|(_, p)| matches!(p, Placement::Ram(_)))
        .map(|(i, _)| *i as usize)
        .collect();
    c.sort_by(|&a, &b| {
        key[a]
            .partial_cmp(&key[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    if layer_cap == 0 {
        for &i in c.iter().take(n) {
            tail[i] = true;
        }
        return tail;
    }
    let per = tier.experts_geometry().1 as usize;
    let mut taken = vec![0usize; records / per.max(1) + 1];
    let mut got = 0;
    for &i in &c {
        if got == n {
            break;
        }
        if taken[i / per] < layer_cap {
            taken[i / per] += 1;
            tail[i] = true;
            got += 1;
        }
    }
    for &i in &c {
        if got == n {
            break;
        }
        if !tail[i] {
            tail[i] = true;
            got += 1;
        }
    }
    tail
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trace file in a scratch directory under `target/`, removed when the returned guard
    /// drops -- never `/tmp` (FN51h).
    fn tmp(name: &str) -> (saber_test_scratch::ScratchDir, std::path::PathBuf) {
        let d = saber_test_scratch::ScratchDir::new(&format!("fn18-rt-{name}"));
        let p = d.join("trace.bin");
        (d, p)
    }

    /// A synthetic served run: prefill chunks and decode steps of skewed routes.
    fn synth(path: &Path, layers: u32, steps: u32) {
        let mut w = TraceWriter::create(path, layers, 10, 512).unwrap();
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = |m: u32| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            // skewed: square a uniform draw
            let u = (x % 10_000) as f64 / 10_000.0;
            ((u * u) * f64::from(m)) as u32
        };
        for s in 0..steps {
            let (kind, rows, live, bucket) = if s % 20 == 0 {
                (StepKind::Prefill, 64, 64, 0)
            } else {
                let b = [1, 2, 4, 8][(s % 4) as usize];
                (StepKind::Decode, b, (b - u32::from(b > 1)).max(1), b)
            };
            let mut ids = Vec::new();
            for _ in 0..layers {
                for _ in 0..rows {
                    let mut row: Vec<u32> = Vec::new();
                    while row.len() < 10 {
                        let e = next(512);
                        if !row.contains(&e) {
                            row.push(e);
                        }
                    }
                    ids.extend(row);
                }
            }
            w.target(kind, rows, live, bucket, 1, &ids).unwrap();
            if kind == StepKind::Decode {
                w.mtp(StepKind::MtpDecode, rows, &ids[..rows as usize * 10])
                    .unwrap();
                w.ple(
                    true,
                    &(0..16 * u64::from(rows))
                        .map(|i| i % 37)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            }
        }
        w.flush().unwrap();
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn the_trace_round_trips_and_the_replay_reproduces_a_live_tier_exactly() {
        let (_dp, p) = tmp("rt");
        let layers = 4;
        synth(&p, layers, 200);
        // decode the file and check the histograms against a direct recount
        let mut r = TraceReader::open(&p).unwrap();
        let (mut targets, mut mtp, mut ple) = (0, 0, 0);
        let mut counts = Counts::new(layers as usize, 512);
        while let Some(s) = r.next_step().unwrap() {
            match s.kind {
                StepKind::Prefill | StepKind::Decode => {
                    targets += 1;
                    assert_eq!(s.layers(), layers as usize);
                    let tot: u32 = s.layer(0).iter().map(|x| u32::from(x.live + x.pad)).sum();
                    assert_eq!(tot, s.rows * 10, "every route counted once");
                    let live: u32 = s.layer(0).iter().map(|x| u32::from(x.live)).sum();
                    assert_eq!(live, s.live * 10, "live rows first");
                    assert!(
                        s.layer(0).windows(2).all(|w| w[0].e < w[1].e),
                        "ascending ids"
                    );
                    counts.add(&s);
                }
                StepKind::MtpDecode => mtp += 1,
                StepKind::Ple => {
                    ple += 1;
                    assert_eq!(s.width, 1);
                }
                StepKind::MtpPrefill => unreachable!(),
            }
        }
        assert_eq!((targets, mtp, ple), (200, 190, 190));
        // the replay == a live tier driven by the same routes (the self-test's law)
        let params = TierParams {
            vram_slots: 4 * 40,
            ram_slots: 4 * 216,
            max_inserts: 2,
            read_through: 1.0,
            ..TierParams::default()
        };
        let prof = counts.profile("synthetic", true);
        let law = InsertLaw::parse(2, "8:1").unwrap();
        let out = replay(&p, 0, 256, 4096, &params, Some(&prof), &law, None, None).unwrap();
        let h = replay_header(0, layers, 256, 4096);
        let mut live = ExpertTier::new(&h, params.clone(), Some(&prof), 0, 1 << 40).unwrap();
        let mut mv = CountMover::default();
        let mut r = TraceReader::open(&p).unwrap();
        let mut per = vec![PathCounts::default(); layers as usize];
        while let Some(s) = r.next_step().unwrap() {
            if !s.kind.target() {
                continue;
            }
            live.set_max_inserts(if s.bucket == 8 { 1 } else { 2 });
            live.begin_step();
            for l in 0..layers as usize {
                let ids: Vec<u32> = s.layer(l).iter().map(|x| u32::from(x.e)).collect();
                let c = live.resolve(l as u32, &ids, &mut mv).unwrap();
                per[l].resident += c.resident;
                per[l].ram += c.ram;
            }
            live.end_step(&mut mv).unwrap();
        }
        assert_eq!(
            out.per_layer
                .iter()
                .map(|c| (c.resident, c.ram))
                .collect::<Vec<_>>(),
            per.iter().map(|c| (c.resident, c.ram)).collect::<Vec<_>>()
        );
        assert_eq!(live.per_layer(), &out.per_layer[..]);
        // the negative control: another insert law changes the counters
        let other = replay(
            &p,
            0,
            256,
            4096,
            &params,
            Some(&prof),
            &InsertLaw::parse(2, "").unwrap(),
            None,
            None,
        )
        .unwrap();
        assert_ne!(
            other.buckets[&8].inserts, out.buckets[&8].inserts,
            "the control must differ"
        );
        // the ceiling bounds a static placement of the same size from above
        let (ceil, per_l) = static_ceiling(&counts.occ, 0, 256, params.vram_slots, 16);
        assert!(ceil > 0.0 && ceil <= 1.0 && per_l.len() == layers as usize);
        let st = replay(
            &p,
            0,
            256,
            4096,
            &TierParams {
                max_inserts: 1,
                ..params.clone()
            },
            Some(&prof),
            &InsertLaw::parse(0, "1:0,2:0,4:0,8:0").unwrap(),
            None,
            None,
        )
        .unwrap();
        let tot: PathCounts = st.per_layer.iter().fold(PathCounts::default(), |mut a, c| {
            a.resident += c.resident;
            a.ram += c.ram;
            a
        });
        let stat_hit = tot.resident as f64 / tot.routed() as f64;
        assert!(
            (stat_hit - ceil).abs() < 1e-12,
            "the in-sample static placement IS the ceiling: {stat_hit} vs {ceil}"
        );
        // the PLE cache replay: 37 distinct rows, cold then warm
        let pl = ple_replay(&p, 64, None, 2).unwrap();
        assert_eq!(pl[0][3], 37, "cold: each distinct row read once");
        assert_eq!(pl[1][3], 0, "warm: none");
        let pl0 = ple_replay(&p, 0, None, 1).unwrap();
        assert_eq!(
            pl0[0][3], pl0[0][1],
            "no cache: every distinct row per gather is read"
        );
        // the owner permutation is a bijection with per_rank ids a rank, and balances the counts
        let perm = balanced_owner(&counts.occ, 256, 2);
        for row in &perm {
            let mut v = row.clone();
            v.sort_unstable();
            assert_eq!(v, (0..512).collect::<Vec<u16>>());
        }
        let (_, m0, mean0) = ep_balance(&p, 256, None, None).unwrap();
        let (_, m1, mean1) = ep_balance(&p, 256, None, Some(&perm)).unwrap();
        assert!(
            (mean0 - mean1).abs() < 1e-9,
            "a permutation keeps the total"
        );
        assert!(
            m1 <= m0,
            "the skewed synthetic routes are balanced by it: {m1} vs {m0}"
        );
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn the_lrfu_and_admission_policies_serve_every_route_and_differ_from_lru() {
        let (_dp, p) = tmp("pol");
        synth(&p, 4, 300);
        let params = TierParams {
            vram_slots: 4 * 40,
            ram_slots: 4 * 216,
            max_inserts: 2,
            read_through: 1.0,
            ..TierParams::default()
        };
        let law = InsertLaw::parse(2, "").unwrap();
        let run = |q: &TierParams| replay(&p, 0, 256, 4096, q, None, &law, None, None).unwrap();
        let lru = run(&params);
        let lrfu = run(&TierParams {
            lrfu_lambda: 0.98,
            ..params.clone()
        });
        let adm = run(&TierParams {
            admit_window: 64,
            ..params.clone()
        });
        let routed = |o: &ReplayOut| o.per_layer.iter().map(PathCounts::routed).sum::<u64>();
        let res = |o: &ReplayOut| o.per_layer.iter().map(|c| c.resident).sum::<u64>();
        let ins = |o: &ReplayOut| o.buckets.values().map(|b| b.inserts).sum::<u64>();
        assert_eq!(
            routed(&lru),
            routed(&lrfu),
            "every route is served on one path"
        );
        assert_eq!(routed(&lru), routed(&adm));
        assert_ne!(res(&lru), res(&lrfu), "LRFU ranks victims differently");
        assert!(
            ins(&adm) < ins(&lru),
            "admission refuses one-off promotions"
        );
        assert!(
            lru.per_layer.iter().all(|c| c.nvme == 0),
            "complete tier: no NVMe"
        );
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn the_min_oracle_bounds_the_tier_and_serves_every_route() {
        // FN41's self-test: MIN (Belady) >= LRU and LRFU on the same routes, slots and warm start
        let (_dp, p) = tmp("min");
        synth(&p, 4, 400);
        let params = TierParams {
            vram_slots: 4 * 40,
            ram_slots: 4 * 216,
            max_inserts: 2,
            read_through: 0.5,
            ..TierParams::default()
        };
        let law = InsertLaw::parse(2, "").unwrap();
        let tier = |q: &TierParams| replay(&p, 0, 256, 4096, q, None, &law, None, None).unwrap();
        let min = |f| min_replay(&p, 0, 256, 4096, &params, None, &law, None, f).unwrap();
        let routed = |o: &ReplayOut| o.per_layer.iter().map(PathCounts::routed).sum::<u64>();
        let res = |o: &ReplayOut| o.per_layer.iter().map(|c| c.resident).sum::<u64>();
        let ins = |o: &ReplayOut| o.buckets.values().map(|b| b.inserts).sum::<u64>();
        let lru = tier(&params);
        let lrfu = tier(&TierParams {
            lrfu_lambda: 0.98,
            ..params.clone()
        });
        let (capped, pure, global, all) = (
            min(MinForm::Capped),
            min(MinForm::Pure),
            min(MinForm::Global),
            min(MinForm::CappedAdmitAll),
        );
        for o in [&capped, &pure, &global, &all] {
            assert_eq!(routed(o), routed(&lru), "every route is served on one path");
            assert_eq!(o.steps, lru.steps);
        }
        for t in [&lru, &lrfu] {
            assert!(res(&capped) >= res(t), "MIN at today's cap >= the tier");
            assert!(res(&pure) >= res(t), "the pure bound >= the tier");
            for (a, b) in pure.per_layer.iter().zip(&t.per_layer) {
                assert!(a.resident >= b.resident, "MIN's optimality holds per layer");
            }
        }
        assert!(res(&pure) >= res(&capped));
        assert!(ins(&all) >= ins(&capped), "bypass only removes promotions");
        // one slot, the warm start on expert 0 (never routed); the steps route 7, 9, 7, 9, ...:
        // the first miss (7) replaces 0, every 9 is bypassed (reused after 7), 7 hits at 2, 4, 6, 8
        let (_dq, q) = tmp("min1");
        let mut w = TraceWriter::create(&q, 1, 1, 512).unwrap();
        for s in 0..10u32 {
            let e = if s % 2 == 0 { 7 } else { 9 };
            w.target(StepKind::Decode, 1, 1, 1, 1, &[e]).unwrap();
        }
        w.flush().unwrap();
        let one = TierParams {
            vram_slots: 1,
            ram_slots: 255,
            max_inserts: 1,
            read_through: 1.0,
            min_layer_slots: 1,
            ..TierParams::default()
        };
        let o = min_replay(&q, 0, 256, 4096, &one, None, &law, None, MinForm::Pure).unwrap();
        assert_eq!((res(&o), ins(&o)), (4, 1));
        let o = min_replay(&q, 0, 256, 4096, &one, None, &law, None, MinForm::Capped).unwrap();
        assert_eq!((res(&o), ins(&o)), (4, 1));
        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_file(&q);
    }

    #[test]
    fn the_fn58_ple_device_table_never_changes_drive_reads_it_only_moves_hits() {
        // FN58: two b1 steps then two b8 steps of 4 rows; rows 1..=4 repeat, 100+ are novel
        let (_d, p) = tmp("fn58");
        let mut w = TraceWriter::create(&p, 1, 1, 512).unwrap();
        w.ple(false, &[1, 2, 3, 4]).unwrap();
        let steps: [(&[u64], u32); 4] = [
            (&[1, 2, 3, 4], 1),
            (&[1, 2, 3, 100], 1),
            (&[1, 2, 101, 102], 8),
            (&[1, 2, 3, 4], 8),
        ];
        for (rows, b) in steps {
            w.ple(true, rows).unwrap();
            w.target(StepKind::Decode, b, b, b, 1, &vec![0; b as usize])
                .unwrap();
        }
        w.flush().unwrap();
        let off = ple_device_replay(&p, 1024, 0, &PleDevPolicy::Lru, None).unwrap();
        let lru = ple_device_replay(&p, 1024, 1024, &PleDevPolicy::Lru, None).unwrap();
        let set: std::collections::HashSet<u64> = [1, 2].into_iter().collect();
        let st = ple_device_replay(&p, 1024, 2, &PleDevPolicy::Static(set), None).unwrap();
        // the drive reads are the compulsory ones (100, 101, 102) whatever the table
        for c in [&off, &lru, &st] {
            assert_eq!(c[2].drive, 3);
            assert_eq!(c[2].steps, 4);
            assert_eq!(c[2].distinct, 16);
            assert_eq!(c[0].steps + c[1].steps, 4);
        }
        assert_eq!(off[2].dev_hits, 0);
        assert_eq!(off[2].host_free, 0);
        // the LRU table holds every staged row: hits 4 + 3 + 2 + 4, host-free steps 1 and 4
        assert_eq!(
            (lru[2].dev_hits, lru[2].host_free, lru[2].drive_free),
            (13, 2, 2)
        );
        assert_eq!((st[2].dev_hits, st[2].host_free), (8, 0));
        let hot = ple_hot_rows(&p, 2).unwrap();
        assert_eq!(hot, [1, 2].into_iter().collect());
    }

    #[test]
    fn the_fn48_victim_arms_reduce_to_lrfu_and_stay_under_min() {
        // FN48: the neutral knobs are FN18's LRFU byte for byte; every arm serves every route on
        // one path and stays under MIN's pure bound per layer
        use crate::expert_tier::{VictimKind, VictimPolicy};
        let (_dp, p) = tmp("fn48");
        synth(&p, 4, 400);
        let base = TierParams {
            vram_slots: 4 * 40,
            ram_slots: 4 * 216,
            max_inserts: 2,
            read_through: 0.5,
            lrfu_lambda: 0.98,
            ..TierParams::default()
        };
        let law = InsertLaw::parse(2, "").unwrap();
        let tier = |q: &TierParams| replay(&p, 0, 256, 4096, q, None, &law, None, None).unwrap();
        let today = tier(&base);
        let neutral = tier(&TierParams {
            policy: VictimPolicy {
                rt_weight: 1.0,
                row_gamma: 0.0,
                prior_beta: 0.0,
                lambda_by_layer: vec![0.98; 4],
                ..VictimPolicy::default()
            },
            ..base.clone()
        });
        assert_eq!(neutral, today, "neutral knobs == today's LRFU");
        let table = learn_table(
            &p,
            0,
            256,
            4096,
            &base,
            None,
            None,
            0.98,
            64,
            TableMode::Etr,
        )
        .unwrap();
        assert_eq!(table.len(), TABLE_CLASSES * TABLE_AGES);
        let pure = min_replay(&p, 0, 256, 4096, &base, None, &law, None, MinForm::Pure).unwrap();
        let routed = |o: &ReplayOut| o.per_layer.iter().map(PathCounts::routed).sum::<u64>();
        for pol in [
            VictimPolicy {
                kind: VictimKind::Reuse,
                ..VictimPolicy::default()
            },
            VictimPolicy {
                kind: VictimKind::Table,
                table: table.clone(),
                ..VictimPolicy::default()
            },
            VictimPolicy {
                rt_weight: 0.0,
                miss_rank: true,
                row_gamma: 0.5,
                ..VictimPolicy::default()
            },
        ] {
            let o = tier(&TierParams {
                policy: pol,
                ..base.clone()
            });
            assert_eq!(
                routed(&o),
                routed(&today),
                "every route is served on one path"
            );
            for (a, b) in pure.per_layer.iter().zip(&o.per_layer) {
                assert!(a.resident >= b.resident, "MIN's pure bound holds per layer");
            }
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn statistics_are_right_on_known_rows() {
        let u = vec![1u64; 512];
        assert!((entropy_bits(&u) - 9.0).abs() < 1e-12);
        assert!((top_share(&u, 128) - 0.25).abs() < 1e-12);
        let a: Vec<u64> = (0..512).collect();
        let b: Vec<u64> = (0..512).map(|x| 1000 - x).collect();
        assert!((spearman(&a, &a) - 1.0).abs() < 1e-12);
        assert!((spearman(&a, &b) + 1.0).abs() < 1e-12);
        assert!(InsertLaw::parse(2, "3:1").is_err());
        let l = InsertLaw::parse(2, "8:0,4:1").unwrap();
        assert_eq!((l.cap(0), l.cap(1), l.cap(4), l.cap(8)), (2, 2, 1, 0));
    }

    #[test]
    fn the_tail_replay_leaves_the_tier_untouched_and_reads_only_tail_ram_routes() {
        // FN53a: the tail law is an accounting over the complete tier's own decisions
        let (_dp, p) = tmp("tail");
        synth(&p, 4, 300);
        let params = TierParams {
            vram_slots: 4 * 40,
            ram_slots: 4 * 216,
            max_inserts: 2,
            read_through: 0.5,
            lrfu_lambda: 0.98,
            ..TierParams::default()
        };
        let law = InsertLaw::parse(2, "").unwrap();
        let mut counts = Counts::new(4, 512);
        let mut r = TraceReader::open(&p).unwrap();
        while let Some(s) = r.next_step().unwrap() {
            if s.kind.target() {
                counts.add(&s);
            }
        }
        let prof = counts.profile("synth", true);
        let plain: Vec<ReplayOut> = [0u32, 1]
            .iter()
            .map(|&k| replay(&p, k, 256, 4096, &params, Some(&prof), &law, None, None).unwrap())
            .collect();
        let tier_of = |k: u32| {
            ExpertTier::new(
                &replay_header(k, 4, 256, 4096),
                params.clone(),
                Some(&prof),
                0,
                1 << 40,
            )
            .unwrap()
        };
        let key = |k: u32| -> Vec<(f64, f64)> {
            (0..4 * 256)
                .map(|i| (counts.occ[i / 256][k as usize * 256 + i % 256] as f64, 0.0))
                .collect()
        };
        let run = |n: usize| {
            let tails: Vec<Vec<bool>> = [0u32, 1]
                .iter()
                .map(|&k| coldest_ram_tail(&tier_of(k), &key(k), n, if n == 400 { 0 } else { 30 }))
                .collect();
            for (k, t) in tails.iter().enumerate() {
                assert_eq!(
                    t.iter().filter(|x| **x).count(),
                    n,
                    "rank {k}: n tail records"
                );
                let tier = tier_of(k as u32);
                for (i, on) in t.iter().enumerate() {
                    if *on {
                        let (l, g) = ((i / 256) as u32, (k * 256 + i % 256) as u32);
                        assert!(matches!(tier.placement(l, g), Some(Placement::Ram(_))));
                    }
                }
            }
            let (mut reads, mut routed) = (0u64, 0u64);
            let outs = tail_replay(
                &p,
                &[0, 1],
                256,
                4096,
                &params,
                Some(&prof),
                &law,
                None,
                &tails,
                |s| {
                    for k in 0..2 {
                        let rk: u64 = s.reads.iter().map(|x| u64::from(x[k])).sum();
                        assert!(rk <= u64::from(s.tail_routed[k]));
                        reads += rk;
                        routed += u64::from(s.tail_routed[k]);
                    }
                },
            )
            .unwrap();
            assert_eq!(
                outs, plain,
                "the tier's decisions do not depend on the tail"
            );
            (reads, routed)
        };
        assert_eq!(run(0), (0, 0), "no tail: no drive reads");
        let ram: u64 = plain
            .iter()
            .flat_map(|o| o.per_layer.iter())
            .map(|c| c.ram)
            .sum();
        let (r100, _) = run(100);
        let (r400, t400) = run(400);
        assert!(r100 <= r400 && r400 > 0 && r400 <= ram && r400 <= t400);
        let _ = std::fs::remove_file(&p);
    }

    /// FN65: the recorder (writer thread, rotation, ring) writes the SAME target records as
    /// FN18's synchronous trace -- per-layer counts equal over its files -- with no drops, and the
    /// ring deletes only its own oldest files.
    #[test]
    fn the_recorder_rotates_off_thread_and_counts_like_the_trace() {
        let (d, p) = tmp("rec");
        let layers = 4;
        synth(&p, layers, 300);
        // replay the synthetic trace's target + PLE records through the recorder
        let rec = d.join("routes");
        let mut sink = TraceSink::spawn(RecordCfg {
            dir: rec.clone(),
            layers,
            topk: 10,
            experts: 512,
            rotate_steps: 50,
            max_bytes: u64::MAX,
            max_inflight: 1 << 30,
        })
        .unwrap();
        let mut want = Counts::new(layers as usize, 512);
        let mut r = TraceReader::open(&p).unwrap();
        let mut steps = 0u64;
        while let Some(s) = r.next_step().unwrap() {
            match s.kind {
                StepKind::Prefill | StepKind::Decode => {
                    want.add(&s);
                    steps += 1;
                    // rebuild the raw route log (u32 LE) from the decoded step: per layer, the
                    // live rows then the pad rows (a prefill histogram expands to its ids)
                    let rows = s.rows as usize;
                    let mut log = Vec::new();
                    for l in 0..layers as usize {
                        let mut ids: Vec<u32> = Vec::new();
                        for x in s.layer(l) {
                            for _ in 0..x.live {
                                ids.push(u32::from(x.e));
                            }
                        }
                        let mut pads: Vec<u32> = Vec::new();
                        for x in s.layer(l) {
                            for _ in 0..x.pad {
                                pads.push(u32::from(x.e));
                            }
                        }
                        ids.extend(pads);
                        assert_eq!(ids.len(), rows * 10);
                        for e in ids {
                            log.extend_from_slice(&e.to_le_bytes());
                        }
                    }
                    sink.target_log(s.kind, s.rows, s.live, s.bucket, s.width, log);
                }
                StepKind::Ple => sink.ple(s.width == 1, &s.ple),
                _ => {}
            }
        }
        let st = sink.finish().unwrap();
        assert_eq!(sink.dropped, 0);
        assert_eq!(st.steps, steps);
        // a rotation every 50 target steps (+ one file for the PLE records after the last step)
        assert!((steps.div_ceil(50)..=steps.div_ceil(50) + 1).contains(&st.files));
        assert_eq!(st.errors, 0);
        let files = trace_files(std::slice::from_ref(&rec)).unwrap();
        assert_eq!(files.len() as u64, st.files);
        assert!(files
            .iter()
            .all(|f| f.extension().is_some_and(|e| e == "bin")));
        let mut got = Counts::new(layers as usize, 512);
        for f in &files {
            let mut r = TraceReader::open(f).unwrap();
            while let Some(s) = r.next_step().unwrap() {
                if s.kind.target() {
                    got.add(&s);
                }
            }
        }
        assert_eq!(got.steps, want.steps);
        assert_eq!(got.tokens, want.tokens);
        assert_eq!(got.routes, want.routes);
        assert_eq!(got.occ, want.occ);
        // the ring: a cap of ~2 files keeps the newest, deletes the oldest (counted)
        let ring = d.join("ring");
        let mut s2 = TraceSink::spawn(RecordCfg {
            dir: ring.clone(),
            layers,
            topk: 10,
            experts: 512,
            rotate_steps: 10,
            max_bytes: 1,
            max_inflight: 1 << 30,
        })
        .unwrap();
        for _ in 0..35 {
            s2.target_log(
                StepKind::Decode,
                1,
                1,
                1,
                1,
                vec![0u8; layers as usize * 40],
            );
        }
        let st2 = s2.finish().unwrap();
        assert_eq!((st2.files, st2.deleted), (4, 3));
        assert_eq!(trace_files(&[ring]).unwrap().len(), 1);
        // a full queue DROPS (never waits): no room in flight
        let mut s3 = TraceSink::spawn(RecordCfg {
            dir: d.join("full"),
            layers,
            topk: 10,
            experts: 512,
            rotate_steps: 10,
            max_bytes: u64::MAX,
            max_inflight: 0,
        })
        .unwrap();
        s3.target_log(
            StepKind::Decode,
            1,
            1,
            1,
            1,
            vec![0u8; layers as usize * 40],
        );
        assert_eq!((s3.sent, s3.dropped), (0, 1));
    }
}
