//! The persistent usage history: an embedded SQLite database (WAL) at
//! `~/.local/share/saber/usage.db` by default.
//!
//! - **Rows.** One row per request, keyed by a `uid`: the console proxy's ref for a proxied
//!   request (the proxy tags the engine request with it, see `saber_serve::requests`), or
//!   `e:<engine boot>:<engine id>` for a request that reached the engine directly (found by the
//!   [`poll`] fallback over the engine's `GET /v1/requests`). A proxied request is seen by BOTH
//!   paths and lands in the same row: the second arrival MERGES into the first (the proxy's
//!   timings and status, the engine's token counts), so nothing is counted twice and a re-poll
//!   after a console restart is a no-op.
//! - **Rollups.** Per minute, hour and day, per (model, key): requests, errors, tokens, sums for
//!   the means and log-bucket histograms for the quantiles (decode tok/s, TTFT). They are kept
//!   EXACTLY in step with the rows: every row insert / merge subtracts the row's old
//!   contribution and adds its new one in the same transaction. Raw rows are pruned after
//!   `raw_retention_days`; rollups are kept forever, and every long-range query (summary over
//!   days, monthly, lifetime, the Prometheus export) reads them, never the raw rows.
//! - **Writes** go through one writer thread that batches them into a transaction (at most
//!   ~150 ms of latency, never on the request path: callers only enqueue).
//!
//! No prompt or reply is stored unless `usage.store_content` is on (default OFF).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::Context;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

pub const MINUTE_MS: i64 = 60_000;
pub const HOUR_MS: i64 = 3_600_000;
pub const DAY_MS: i64 = 86_400_000;

/// Row source bits.
pub const SRC_PROXY: i64 = 1;
pub const SRC_ENGINE: i64 = 2;

// ---------------------------------------------------------------------------------------------
// log-bucket histograms

/// A fixed log-bucket histogram layout: bucket 0 = (0, base], bucket i = (base*r^(i-1),
/// base*r^i] with r = 2^(1/per_octave); the last bucket is open-ended.
#[derive(Clone, Copy, Debug)]
pub struct LogBuckets {
    pub base: f64,
    pub per_octave: u32,
    pub n: usize,
}

/// TTFT in ms: 1 ms .. ~18 min, 19 % wide buckets.
pub const TTFT_BUCKETS: LogBuckets = LogBuckets {
    base: 1.0,
    per_octave: 4,
    n: 82,
};
/// Decode tokens/s: 0.1 .. ~26k.
pub const TPS_BUCKETS: LogBuckets = LogBuckets {
    base: 0.1,
    per_octave: 4,
    n: 74,
};

impl LogBuckets {
    #[must_use]
    pub fn index(&self, v: f64) -> usize {
        if !v.is_finite() || v <= self.base {
            return 0;
        }
        let k = ((v / self.base).log2() * f64::from(self.per_octave)).ceil();
        (k.max(1.0) as usize).min(self.n - 1)
    }

    #[must_use]
    pub fn upper(&self, i: usize) -> f64 {
        self.base * (i as f64 / f64::from(self.per_octave)).exp2()
    }

    /// The value a quantile in bucket `i` reports: the bucket's geometric middle.
    #[must_use]
    pub fn representative(&self, i: usize) -> f64 {
        if i == 0 {
            self.base
        } else {
            (self.upper(i - 1) * self.upper(i)).sqrt()
        }
    }
}

/// Sparse bucket counts, stored as `i:c,i:c` text.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Hist(pub BTreeMap<u16, i64>);

impl Hist {
    #[must_use]
    pub fn decode(s: &str) -> Hist {
        let mut h = Hist::default();
        for part in s.split(',') {
            if let Some((i, c)) = part.split_once(':') {
                if let (Ok(i), Ok(c)) = (i.parse::<u16>(), c.parse::<i64>()) {
                    *h.0.entry(i).or_insert(0) += c;
                }
            }
        }
        h
    }

    #[must_use]
    pub fn encode(&self) -> String {
        self.0
            .iter()
            .filter(|(_, c)| **c != 0)
            .map(|(i, c)| format!("{i}:{c}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    pub fn add(&mut self, i: usize, by: i64) {
        let e = self
            .0
            .entry(u16::try_from(i).unwrap_or(u16::MAX))
            .or_insert(0);
        *e += by;
        if *e == 0 {
            self.0.remove(&u16::try_from(i).unwrap_or(u16::MAX));
        }
    }

    pub fn merge(&mut self, o: &Hist) {
        for (i, c) in &o.0 {
            self.add(usize::from(*i), *c);
        }
    }

    #[must_use]
    pub fn count(&self) -> i64 {
        self.0.values().sum()
    }

    /// The `q` quantile (nearest rank), or `None` with no samples.
    #[must_use]
    pub fn quantile(&self, q: f64, b: &LogBuckets) -> Option<f64> {
        let n = self.count();
        if n <= 0 {
            return None;
        }
        let rank = ((q.clamp(0.0, 1.0) * n as f64).ceil() as i64).max(1);
        let mut cum = 0;
        for (i, c) in &self.0 {
            cum += c;
            if cum >= rank {
                return Some(b.representative(usize::from(*i)));
            }
        }
        self.0
            .keys()
            .last()
            .map(|i| b.representative(usize::from(*i)))
    }
}

// ---------------------------------------------------------------------------------------------
// rows

/// One request. Every field but `uid` / `ts_ms` / `source` / `endpoint` may be unknown to one
/// of the two capture paths; a merge keeps each side's better-known fields.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageRow {
    pub uid: String,
    /// Arrival, unix ms (the proxy's when it saw the request, else the engine's).
    pub ts_ms: i64,
    /// [`SRC_PROXY`] | [`SRC_ENGINE`].
    pub source: i64,
    pub endpoint: String,
    pub model: Option<String>,
    /// The API key's label (the console proxy's key label, else the engine's).
    pub key: Option<String>,
    pub stream: Option<bool>,
    pub status: Option<i64>,
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub cached_tokens: Option<i64>,
    pub ttft_ms: Option<f64>,
    pub latency_ms: Option<f64>,
    pub decode_tps: Option<f64>,
    pub finish_reason: Option<String>,
    pub images: Option<i64>,
    pub thinking: Option<bool>,
    pub engine_boot: Option<i64>,
    pub engine_id: Option<i64>,
    /// An error message (never a body).
    pub error: Option<String>,
    /// Only with `usage.store_content` on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_body: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_body: Option<String>,
}

impl UsageRow {
    /// Merge `b` into `a` (either order of arrival). The proxy's view wins for what the client
    /// saw (arrival, timings, status, its key label, bodies); the engine's for what only the
    /// engine knows exactly (token counts, cached prefix, decode rate, finish reason).
    #[must_use]
    pub fn merged(a: UsageRow, b: UsageRow) -> UsageRow {
        let (p, e) = if b.source & SRC_PROXY != 0 && a.source & SRC_PROXY == 0 {
            (b, a)
        } else if a.source & SRC_PROXY != 0 && b.source & SRC_PROXY == 0 {
            (a, b)
        } else {
            // the same path twice (a re-poll): the newer arrival wins field by field
            (b, a)
        };
        UsageRow {
            uid: p.uid,
            ts_ms: p.ts_ms,
            source: p.source | e.source,
            endpoint: p.endpoint,
            model: p.model.or(e.model),
            key: p.key.or(e.key),
            stream: p.stream.or(e.stream),
            status: p.status.or(e.status),
            prompt_tokens: e.prompt_tokens.or(p.prompt_tokens),
            completion_tokens: e.completion_tokens.or(p.completion_tokens),
            cached_tokens: e.cached_tokens.or(p.cached_tokens),
            ttft_ms: p.ttft_ms.or(e.ttft_ms),
            latency_ms: p.latency_ms.or(e.latency_ms),
            decode_tps: e.decode_tps.or(p.decode_tps),
            finish_reason: e.finish_reason.or(p.finish_reason),
            images: e.images.or(p.images),
            thinking: e.thinking.or(p.thinking),
            engine_boot: e.engine_boot.or(p.engine_boot),
            engine_id: e.engine_id.or(p.engine_id),
            error: p.error.or(e.error),
            request_body: p.request_body.or(e.request_body),
            response_body: p.response_body.or(e.response_body),
        }
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.status.is_some_and(|s| s >= 400 && s != 499)
    }
}

/// The start (unix ms) of the day holding `ts`, days cut at `off_min` minutes east of UTC.
#[must_use]
pub fn day_start(ts: i64, off_min: i32) -> i64 {
    let off = i64::from(off_min) * MINUTE_MS;
    (ts + off).div_euclid(DAY_MS) * DAY_MS - off
}

/// (year, month 1..=12, day) of a day number since 1970-01-01 (H. Hinnant's civil_from_days).
#[must_use]
pub fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `YYYY-MM` of the local day holding `ts`.
#[must_use]
pub fn month_of(ts: i64, off_min: i32) -> String {
    let (y, m, _) = civil((ts + i64::from(off_min) * MINUTE_MS).div_euclid(DAY_MS));
    format!("{y:04}-{m:02}")
}

/// `YYYY-MM-DD` of the local day holding `ts`.
#[must_use]
pub fn date_of(ts: i64, off_min: i32) -> String {
    let (y, m, d) = civil((ts + i64::from(off_min) * MINUTE_MS).div_euclid(DAY_MS));
    format!("{y:04}-{m:02}-{d:02}")
}

// ---------------------------------------------------------------------------------------------
// rollups

/// One rollup cell's sums.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Agg {
    pub requests: i64,
    pub errors: i64,
    pub aborted: i64,
    pub prompt: i64,
    pub completion: i64,
    pub cached: i64,
    pub tps_sum: f64,
    pub tps_n: i64,
    pub pps_sum: f64,
    pub pps_n: i64,
    pub ttft_sum: f64,
    pub ttft_n: i64,
    pub latency_sum: f64,
    pub tps_hist: Hist,
    pub ttft_hist: Hist,
}

impl Agg {
    fn add_row(&mut self, r: &UsageRow, sign: i64) {
        let s = sign as f64;
        self.requests += sign;
        if r.is_error() {
            self.errors += sign;
        }
        if r.status == Some(499) {
            self.aborted += sign;
        }
        self.prompt += sign * r.prompt_tokens.unwrap_or(0);
        self.completion += sign * r.completion_tokens.unwrap_or(0);
        self.cached += sign * r.cached_tokens.unwrap_or(0);
        if let Some(v) = r.decode_tps.filter(|v| v.is_finite() && *v > 0.0) {
            self.tps_sum += s * v;
            self.tps_n += sign;
            self.tps_hist.add(TPS_BUCKETS.index(v), sign);
        }
        if let Some(t) = r.ttft_ms.filter(|v| v.is_finite() && *v >= 0.0) {
            self.ttft_sum += s * t;
            self.ttft_n += sign;
            self.ttft_hist.add(TTFT_BUCKETS.index(t), sign);
            if let Some(p) = r.prompt_tokens.filter(|p| *p > 0) {
                if t > 0.0 {
                    self.pps_sum += s * (p as f64 / (t / 1e3));
                    self.pps_n += sign;
                }
            }
        }
        self.latency_sum += s * r.latency_ms.unwrap_or(0.0);
    }

    pub fn merge(&mut self, o: &Agg) {
        self.requests += o.requests;
        self.errors += o.errors;
        self.aborted += o.aborted;
        self.prompt += o.prompt;
        self.completion += o.completion;
        self.cached += o.cached;
        self.tps_sum += o.tps_sum;
        self.tps_n += o.tps_n;
        self.pps_sum += o.pps_sum;
        self.pps_n += o.pps_n;
        self.ttft_sum += o.ttft_sum;
        self.ttft_n += o.ttft_n;
        self.latency_sum += o.latency_sum;
        self.tps_hist.merge(&o.tps_hist);
        self.ttft_hist.merge(&o.ttft_hist);
    }

    /// The JSON the API reports for a cell or a merged range.
    #[must_use]
    pub fn json(&self) -> serde_json::Value {
        let mean = |s: f64, n: i64| (n > 0).then(|| s / n as f64);
        serde_json::json!({
            "requests": self.requests,
            "errors": self.errors,
            "aborted": self.aborted,
            "error_rate": (self.requests > 0).then(|| self.errors as f64 / self.requests as f64),
            "prompt_tokens": self.prompt,
            "completion_tokens": self.completion,
            "cached_tokens": self.cached,
            "cached_ratio": (self.prompt > 0).then(|| self.cached as f64 / self.prompt as f64),
            "decode_tps_mean": mean(self.tps_sum, self.tps_n),
            "decode_tps_p50": self.tps_hist.quantile(0.5, &TPS_BUCKETS),
            "decode_tps_p95": self.tps_hist.quantile(0.95, &TPS_BUCKETS),
            "prompt_tps_mean": mean(self.pps_sum, self.pps_n),
            "ttft_mean_ms": mean(self.ttft_sum, self.ttft_n),
            "ttft_p50_ms": self.ttft_hist.quantile(0.5, &TTFT_BUCKETS),
            "ttft_p95_ms": self.ttft_hist.quantile(0.95, &TTFT_BUCKETS),
            "ttft_p99_ms": self.ttft_hist.quantile(0.99, &TTFT_BUCKETS),
            "latency_mean_ms": mean(self.latency_sum, self.requests),
        })
    }
}

/// A rollup period.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Period {
    Minute,
    Hour,
    Day,
}

impl Period {
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Period::Minute => "m",
            Period::Hour => "h",
            Period::Day => "d",
        }
    }

    #[must_use]
    pub fn ms(self) -> i64 {
        match self {
            Period::Minute => MINUTE_MS,
            Period::Hour => HOUR_MS,
            Period::Day => DAY_MS,
        }
    }

    #[must_use]
    pub fn bucket(self, ts: i64, off_min: i32) -> i64 {
        match self {
            Period::Minute => ts.div_euclid(MINUTE_MS) * MINUTE_MS,
            Period::Hour => ts.div_euclid(HOUR_MS) * HOUR_MS,
            Period::Day => day_start(ts, off_min),
        }
    }
}

/// One stored rollup cell.
#[derive(Clone, Debug, PartialEq)]
pub struct RollupCell {
    pub bucket: i64,
    pub model: String,
    pub key: String,
    pub agg: Agg,
}

// ---------------------------------------------------------------------------------------------
// the store

/// A write for the writer thread.
pub enum Op {
    Row(Box<UsageRow>),
    /// GPU energy sampled by the console, joules over the interval ending at `ts_ms`.
    Energy {
        ts_ms: i64,
        joules: f64,
    },
    /// The poller's cursor (engine boot id, last id written).
    Cursor {
        boot: u64,
        after: u64,
    },
    /// Delete rows older than `before_ms` (rollups stay).
    Prune {
        before_ms: i64,
    },
    /// Answered once every op queued before it is committed.
    Flush(mpsc::Sender<()>),
}

/// Writer counters (for `/api/usage/status`).
#[derive(Clone, Debug, Default, Serialize)]
pub struct WriterStats {
    pub rows_written: u64,
    pub rows_merged: u64,
    pub batches: u64,
    pub rows_pruned: u64,
    pub last_error: Option<String>,
    pub last_batch_ms: f64,
}

pub struct UsageStore {
    tx: Mutex<mpsc::Sender<Op>>,
    read: Mutex<Connection>,
    pub path: PathBuf,
    /// Live (the Usage page's day boundary applies without a console restart).
    off_min: std::sync::Arc<std::sync::atomic::AtomicI32>,
    stats: std::sync::Arc<Mutex<WriterStats>>,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS requests(
  uid TEXT PRIMARY KEY,
  ts_ms INTEGER NOT NULL,
  source INTEGER NOT NULL,
  endpoint TEXT NOT NULL,
  model TEXT, key TEXT, stream INTEGER, status INTEGER,
  prompt_tokens INTEGER, completion_tokens INTEGER, cached_tokens INTEGER,
  ttft_ms REAL, latency_ms REAL, decode_tps REAL,
  finish_reason TEXT, images INTEGER, thinking INTEGER,
  engine_boot INTEGER, engine_id INTEGER,
  error TEXT, request_body TEXT, response_body TEXT,
  day_off INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts_ms);
CREATE UNIQUE INDEX IF NOT EXISTS requests_engine ON requests(engine_boot, engine_id)
  WHERE engine_id IS NOT NULL;
CREATE TABLE IF NOT EXISTS rollups(
  period TEXT NOT NULL, bucket INTEGER NOT NULL, model TEXT NOT NULL, key TEXT NOT NULL,
  requests INTEGER NOT NULL, errors INTEGER NOT NULL, aborted INTEGER NOT NULL,
  prompt INTEGER NOT NULL, completion INTEGER NOT NULL, cached INTEGER NOT NULL,
  tps_sum REAL NOT NULL, tps_n INTEGER NOT NULL, pps_sum REAL NOT NULL, pps_n INTEGER NOT NULL,
  ttft_sum REAL NOT NULL, ttft_n INTEGER NOT NULL, latency_sum REAL NOT NULL,
  tps_hist TEXT NOT NULL, ttft_hist TEXT NOT NULL,
  PRIMARY KEY(period, bucket, model, key)
);
CREATE TABLE IF NOT EXISTS energy(bucket INTEGER PRIMARY KEY, joules REAL NOT NULL);
";

const ROW_COLS: &str = "uid, ts_ms, source, endpoint, model, key, stream, status, prompt_tokens, \
     completion_tokens, cached_tokens, ttft_ms, latency_ms, decode_tps, finish_reason, images, \
     thinking, engine_boot, engine_id, error";

fn row_from_sql(r: &rusqlite::Row<'_>, with_bodies: bool) -> rusqlite::Result<UsageRow> {
    Ok(UsageRow {
        uid: r.get(0)?,
        ts_ms: r.get(1)?,
        source: r.get(2)?,
        endpoint: r.get(3)?,
        model: r.get(4)?,
        key: r.get(5)?,
        stream: r.get(6)?,
        status: r.get(7)?,
        prompt_tokens: r.get(8)?,
        completion_tokens: r.get(9)?,
        cached_tokens: r.get(10)?,
        ttft_ms: r.get(11)?,
        latency_ms: r.get(12)?,
        decode_tps: r.get(13)?,
        finish_reason: r.get(14)?,
        images: r.get(15)?,
        thinking: r.get(16)?,
        engine_boot: r.get(17)?,
        engine_id: r.get(18)?,
        error: r.get(19)?,
        request_body: if with_bodies { r.get(20)? } else { None },
        response_body: if with_bodies { r.get(21)? } else { None },
    })
}

fn open_conn(path: &Path) -> anyhow::Result<Connection> {
    let c = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
    c.busy_timeout(Duration::from_secs(5))?;
    c.pragma_update(None, "journal_mode", "WAL")?;
    c.pragma_update(None, "synchronous", "NORMAL")?;
    Ok(c)
}

impl UsageStore {
    /// Open (creating it and its directory if needed) and start the writer thread.
    pub fn open(path: &Path, utc_offset_minutes: i32) -> anyhow::Result<UsageStore> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let w = open_conn(path)?;
        w.execute_batch(SCHEMA)
            .context("creating the usage schema")?;
        let read = open_conn(path)?;
        let (tx, rx) = mpsc::channel::<Op>();
        let stats = std::sync::Arc::new(Mutex::new(WriterStats::default()));
        let st = stats.clone();
        let off = std::sync::Arc::new(std::sync::atomic::AtomicI32::new(utc_offset_minutes));
        let off_w = off.clone();
        std::thread::Builder::new()
            .name("saber-usage-writer".into())
            .spawn(move || writer(w, &rx, &st, &off_w))
            .context("starting the usage writer")?;
        Ok(UsageStore {
            tx: Mutex::new(tx),
            read: Mutex::new(read),
            path: path.to_path_buf(),
            off_min: off,
            stats,
        })
    }

    #[must_use]
    pub fn utc_offset_minutes(&self) -> i32 {
        self.off_min.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Cut days and months at a new offset from now on (rows already rolled up keep their cut).
    pub fn set_utc_offset_minutes(&self, m: i32) {
        self.off_min
            .store(m.clamp(-840, 840), std::sync::atomic::Ordering::Relaxed);
    }

    /// Queue a write (never blocks).
    pub fn send(&self, op: Op) {
        let _ = self.tx.lock().unwrap_or_else(|e| e.into_inner()).send(op);
    }

    pub fn record(&self, row: UsageRow) {
        self.send(Op::Row(Box::new(row)));
    }

    /// Wait (blocking, bounded) until everything queued so far is committed.
    pub fn flush(&self) {
        let (t, r) = mpsc::channel();
        self.send(Op::Flush(t));
        let _ = r.recv_timeout(Duration::from_secs(30));
    }

    #[must_use]
    pub fn writer_stats(&self) -> WriterStats {
        self.stats.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn rd(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.read.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The poller's stored cursor (engine boot id, last id).
    pub fn cursor(&self) -> anyhow::Result<Option<(u64, u64)>> {
        let c = self.rd();
        let v: Option<String> = c
            .query_row("SELECT v FROM meta WHERE k = 'poll_cursor'", [], |r| {
                r.get(0)
            })
            .optional()?;
        Ok(v.and_then(|v| {
            let (b, a) = v.split_once(':')?;
            Some((b.parse().ok()?, a.parse().ok()?))
        }))
    }

    pub fn row(&self, uid: &str) -> anyhow::Result<Option<UsageRow>> {
        let c = self.rd();
        Ok(c.query_row(
            &format!("SELECT {ROW_COLS}, request_body, response_body FROM requests WHERE uid = ?1"),
            [uid],
            |r| row_from_sql(r, true),
        )
        .optional()?)
    }

    /// A filtered, sorted page of rows (bodies left out) and the total match count.
    pub fn rows(&self, f: &RowFilter) -> anyhow::Result<(i64, Vec<UsageRow>)> {
        let (wh, args) = f.where_clause();
        let c = self.rd();
        let total: i64 = c.query_row(
            &format!("SELECT COUNT(*) FROM requests{wh}"),
            rusqlite::params_from_iter(args.iter()),
            |r| r.get(0),
        )?;
        let sort = f
            .sort
            .as_deref()
            .filter(|s| {
                matches!(
                    *s,
                    "ts_ms"
                        | "latency_ms"
                        | "ttft_ms"
                        | "prompt_tokens"
                        | "completion_tokens"
                        | "cached_tokens"
                        | "decode_tps"
                        | "status"
                        | "model"
                        | "key"
                )
            })
            .unwrap_or("ts_ms");
        let dir = if f.asc { "ASC" } else { "DESC" };
        let sql = format!(
            "SELECT {ROW_COLS} FROM requests{wh} ORDER BY {sort} IS NULL, {sort} {dir}, uid {dir} \
             LIMIT {} OFFSET {}",
            f.limit.clamp(1, 100_000),
            f.offset.max(0)
        );
        let mut st = c.prepare(&sql)?;
        let rows = st
            .query_map(rusqlite::params_from_iter(args.iter()), |r| {
                row_from_sql(r, false)
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok((total, rows))
    }

    /// Raw row count (tests and status).
    pub fn row_count(&self) -> anyhow::Result<i64> {
        Ok(self
            .rd()
            .query_row("SELECT COUNT(*) FROM requests", [], |r| r.get(0))?)
    }

    /// Rollup cells of `period` with `from <= bucket < to`.
    pub fn cells(&self, period: Period, from: i64, to: i64) -> anyhow::Result<Vec<RollupCell>> {
        let c = self.rd();
        let mut st = c.prepare(
            "SELECT bucket, model, key, requests, errors, aborted, prompt, completion, cached, \
             tps_sum, tps_n, pps_sum, pps_n, ttft_sum, ttft_n, latency_sum, tps_hist, ttft_hist \
             FROM rollups WHERE period = ?1 AND bucket >= ?2 AND bucket < ?3 ORDER BY bucket",
        )?;
        let out = st
            .query_map(params![period.tag(), from, to], |r| {
                Ok(RollupCell {
                    bucket: r.get(0)?,
                    model: r.get(1)?,
                    key: r.get(2)?,
                    agg: Agg {
                        requests: r.get(3)?,
                        errors: r.get(4)?,
                        aborted: r.get(5)?,
                        prompt: r.get(6)?,
                        completion: r.get(7)?,
                        cached: r.get(8)?,
                        tps_sum: r.get(9)?,
                        tps_n: r.get(10)?,
                        pps_sum: r.get(11)?,
                        pps_n: r.get(12)?,
                        ttft_sum: r.get(13)?,
                        ttft_n: r.get(14)?,
                        latency_sum: r.get(15)?,
                        tps_hist: Hist::decode(&r.get::<_, String>(16)?),
                        ttft_hist: Hist::decode(&r.get::<_, String>(17)?),
                    },
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(out)
    }

    /// Energy (joules) by hour bucket, `from <= bucket < to`.
    pub fn energy(&self, from: i64, to: i64) -> anyhow::Result<Vec<(i64, f64)>> {
        let c = self.rd();
        let mut st = c.prepare(
            "SELECT bucket, joules FROM energy WHERE bucket >= ?1 AND bucket < ?2 ORDER BY bucket",
        )?;
        let out = st
            .query_map(params![from, to], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(out)
    }
}

/// `/api/usage/requests` filters.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct RowFilter {
    /// Unix ms, inclusive / exclusive.
    pub from: Option<i64>,
    pub to: Option<i64>,
    pub model: Option<String>,
    pub key: Option<String>,
    /// `ok` (< 400), `error` (>= 400 but not 499), `aborted` (499), or a status code.
    pub status: Option<String>,
    pub min_latency_ms: Option<f64>,
    pub max_latency_ms: Option<f64>,
    pub min_tokens: Option<i64>,
    pub max_tokens: Option<i64>,
    pub endpoint: Option<String>,
    /// `proxy` / `engine` (direct-to-engine traffic only).
    pub source: Option<String>,
    pub q: Option<String>,
    pub sort: Option<String>,
    pub asc: bool,
    pub limit: i64,
    pub offset: i64,
}

impl RowFilter {
    fn where_clause(&self) -> (String, Vec<rusqlite::types::Value>) {
        use rusqlite::types::Value as V;
        let mut w: Vec<String> = Vec::new();
        let mut a: Vec<V> = Vec::new();
        let mut push = |cond: &str, v: V, w: &mut Vec<String>| {
            a.push(v);
            w.push(cond.replace('?', &format!("?{}", a.len())));
        };
        if let Some(v) = self.from {
            push("ts_ms >= ?", V::Integer(v), &mut w);
        }
        if let Some(v) = self.to {
            push("ts_ms < ?", V::Integer(v), &mut w);
        }
        if let Some(v) = self.model.as_ref().filter(|s| !s.is_empty()) {
            push("model = ?", V::Text(v.clone()), &mut w);
        }
        if let Some(v) = self.key.as_ref().filter(|s| !s.is_empty()) {
            push("key = ?", V::Text(v.clone()), &mut w);
        }
        if let Some(v) = self.endpoint.as_ref().filter(|s| !s.is_empty()) {
            push("endpoint = ?", V::Text(v.clone()), &mut w);
        }
        match self.status.as_deref() {
            Some("ok") => w.push("status < 400".into()),
            Some("error") => w.push("status >= 400 AND status != 499".into()),
            Some("aborted") => w.push("status = 499".into()),
            Some(s) => {
                if let Ok(n) = s.parse::<i64>() {
                    push("status = ?", V::Integer(n), &mut w);
                }
            }
            None => {}
        }
        match self.source.as_deref() {
            Some("proxy") => w.push(format!("(source & {SRC_PROXY}) != 0")),
            Some("engine") => w.push(format!("(source & {SRC_PROXY}) = 0")),
            _ => {}
        }
        if let Some(v) = self.min_latency_ms {
            push("latency_ms >= ?", V::Real(v), &mut w);
        }
        if let Some(v) = self.max_latency_ms {
            push("latency_ms <= ?", V::Real(v), &mut w);
        }
        if let Some(v) = self.min_tokens {
            push(
                "COALESCE(prompt_tokens,0) + COALESCE(completion_tokens,0) >= ?",
                V::Integer(v),
                &mut w,
            );
        }
        if let Some(v) = self.max_tokens {
            push(
                "COALESCE(prompt_tokens,0) + COALESCE(completion_tokens,0) <= ?",
                V::Integer(v),
                &mut w,
            );
        }
        if let Some(v) = self.q.as_ref().filter(|s| !s.is_empty()) {
            a.push(V::Text(format!("%{v}%")));
            let n = a.len();
            // the same bound parameter for every LIKE
            w.push(format!(
                "(uid LIKE ?{n} OR model LIKE ?{n} OR key LIKE ?{n} OR finish_reason LIKE ?{n} \
                 OR error LIKE ?{n})"
            ));
        }
        if w.is_empty() {
            (String::new(), a)
        } else {
            (format!(" WHERE {}", w.join(" AND ")), a)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// the writer

fn writer(
    mut conn: Connection,
    rx: &mpsc::Receiver<Op>,
    stats: &Mutex<WriterStats>,
    off: &std::sync::atomic::AtomicI32,
) {
    const WINDOW: Duration = Duration::from_millis(150);
    while let Ok(first) = rx.recv() {
        let mut batch = vec![first];
        let deadline = Instant::now() + WINDOW;
        while batch.len() < 2000 {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            match rx.recv_timeout(left) {
                Ok(op) => {
                    let is_flush = matches!(op, Op::Flush(_));
                    batch.push(op);
                    if is_flush {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        let t0 = Instant::now();
        let mut flushes = Vec::new();
        let mut written = 0u64;
        let mut merged = 0u64;
        let mut pruned = 0u64;
        let off_min = off.load(std::sync::atomic::Ordering::Relaxed);
        let res = (|| -> anyhow::Result<()> {
            let tx = conn.transaction()?;
            for op in batch {
                match op {
                    Op::Row(r) => {
                        if upsert(&tx, *r, off_min)? {
                            merged += 1;
                        } else {
                            written += 1;
                        }
                    }
                    Op::Energy { ts_ms, joules } => {
                        tx.execute(
                            "INSERT INTO energy(bucket, joules) VALUES (?1, ?2) \
                             ON CONFLICT(bucket) DO UPDATE SET joules = joules + excluded.joules",
                            params![ts_ms.div_euclid(HOUR_MS) * HOUR_MS, joules],
                        )?;
                    }
                    Op::Cursor { boot, after } => {
                        tx.execute(
                            "INSERT INTO meta(k, v) VALUES ('poll_cursor', ?1) \
                             ON CONFLICT(k) DO UPDATE SET v = excluded.v",
                            [format!("{boot}:{after}")],
                        )?;
                    }
                    Op::Prune { before_ms } => {
                        pruned += tx
                            .execute("DELETE FROM requests WHERE ts_ms < ?1", [before_ms])?
                            as u64;
                    }
                    Op::Flush(t) => flushes.push(t),
                }
            }
            tx.commit()?;
            Ok(())
        })();
        {
            let mut s = stats.lock().unwrap_or_else(|e| e.into_inner());
            s.batches += 1;
            s.last_batch_ms = t0.elapsed().as_secs_f64() * 1e3;
            match res {
                Ok(()) => {
                    s.rows_written += written;
                    s.rows_merged += merged;
                    s.rows_pruned += pruned;
                }
                Err(e) => s.last_error = Some(format!("{e:#}")),
            }
        }
        for f in flushes {
            let _ = f.send(());
        }
    }
}

fn load_agg(tx: &rusqlite::Transaction<'_>, key: (&str, i64, &str, &str)) -> rusqlite::Result<Agg> {
    tx.query_row(
        "SELECT requests, errors, aborted, prompt, completion, cached, tps_sum, tps_n, pps_sum, \
         pps_n, ttft_sum, ttft_n, latency_sum, tps_hist, ttft_hist FROM rollups \
         WHERE period = ?1 AND bucket = ?2 AND model = ?3 AND key = ?4",
        params![key.0, key.1, key.2, key.3],
        |r| {
            Ok(Agg {
                requests: r.get(0)?,
                errors: r.get(1)?,
                aborted: r.get(2)?,
                prompt: r.get(3)?,
                completion: r.get(4)?,
                cached: r.get(5)?,
                tps_sum: r.get(6)?,
                tps_n: r.get(7)?,
                pps_sum: r.get(8)?,
                pps_n: r.get(9)?,
                ttft_sum: r.get(10)?,
                ttft_n: r.get(11)?,
                latency_sum: r.get(12)?,
                tps_hist: Hist::decode(&r.get::<_, String>(13)?),
                ttft_hist: Hist::decode(&r.get::<_, String>(14)?),
            })
        },
    )
    .optional()
    .map(Option::unwrap_or_default)
}

/// Add (`sign` 1) or remove (-1) a row's contribution to its three rollup cells.
fn apply_rollups(
    tx: &rusqlite::Transaction<'_>,
    r: &UsageRow,
    off_min: i32,
    sign: i64,
) -> rusqlite::Result<()> {
    let model = r.model.as_deref().unwrap_or("");
    let key = r.key.as_deref().unwrap_or("");
    for p in [Period::Minute, Period::Hour, Period::Day] {
        let b = p.bucket(r.ts_ms, off_min);
        let mut agg = load_agg(tx, (p.tag(), b, model, key))?;
        agg.add_row(r, sign);
        if agg.requests <= 0 {
            tx.execute(
                "DELETE FROM rollups WHERE period = ?1 AND bucket = ?2 AND model = ?3 AND key = ?4",
                params![p.tag(), b, model, key],
            )?;
            continue;
        }
        tx.execute(
            "INSERT OR REPLACE INTO rollups(period, bucket, model, key, requests, errors, aborted, \
             prompt, completion, cached, tps_sum, tps_n, pps_sum, pps_n, ttft_sum, ttft_n, \
             latency_sum, tps_hist, ttft_hist) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, \
             ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)",
            params![
                p.tag(),
                b,
                model,
                key,
                agg.requests,
                agg.errors,
                agg.aborted,
                agg.prompt,
                agg.completion,
                agg.cached,
                agg.tps_sum,
                agg.tps_n,
                agg.pps_sum,
                agg.pps_n,
                agg.ttft_sum,
                agg.ttft_n,
                agg.latency_sum,
                agg.tps_hist.encode(),
                agg.ttft_hist.encode(),
            ],
        )?;
    }
    Ok(())
}

/// Insert or merge one row; true when it merged into an existing row.
fn upsert(tx: &rusqlite::Transaction<'_>, row: UsageRow, off_min: i32) -> anyhow::Result<bool> {
    // an existing row: by uid, else by the engine's (boot, id)
    let existing: Option<(UsageRow, i32)> = tx
        .query_row(
            &format!(
                "SELECT {ROW_COLS}, request_body, response_body, day_off FROM requests \
                 WHERE uid = ?1 OR (engine_id IS NOT NULL AND engine_boot = ?2 AND engine_id = ?3) \
                 LIMIT 1"
            ),
            params![row.uid, row.engine_boot, row.engine_id],
            |r| Ok((row_from_sql(r, true)?, r.get::<_, i32>(22)?)),
        )
        .optional()?;
    let (new, was) = match existing {
        Some((old, old_off)) => {
            apply_rollups(tx, &old, old_off, -1)?;
            tx.execute("DELETE FROM requests WHERE uid = ?1", [&old.uid])?;
            (UsageRow::merged(old, row), true)
        }
        None => (row, false),
    };
    tx.execute(
        &format!(
            "INSERT INTO requests({ROW_COLS}, request_body, response_body, day_off) VALUES (?1, ?2, \
             ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, \
             ?21, ?22, ?23)"
        ),
        params![
            new.uid,
            new.ts_ms,
            new.source,
            new.endpoint,
            new.model,
            new.key,
            new.stream,
            new.status,
            new.prompt_tokens,
            new.completion_tokens,
            new.cached_tokens,
            new.ttft_ms,
            new.latency_ms,
            new.decode_tps,
            new.finish_reason,
            new.images,
            new.thinking,
            new.engine_boot,
            new.engine_id,
            new.error,
            new.request_body,
            new.response_body,
            off_min,
        ],
    )?;
    apply_rollups(tx, &new, off_min, 1)?;
    Ok(was)
}

// ---------------------------------------------------------------------------------------------
// queries over the rollups

/// The rollup period a range of `span_ms` is read from (minute cells up to 2 days, hours up to
/// 60 days, days beyond).
#[must_use]
pub fn period_for(span_ms: i64) -> Period {
    if span_ms <= 2 * DAY_MS {
        Period::Minute
    } else if span_ms <= 60 * DAY_MS {
        Period::Hour
    } else {
        Period::Day
    }
}

/// `/api/usage/summary`: totals, quantiles, per-model / per-key and a series over
/// `[from, to)`, all from rollups. `step_ms` = the series' bucket (a multiple of the period).
pub fn summary(
    store: &UsageStore,
    from: i64,
    to: i64,
    step_ms: Option<i64>,
) -> anyhow::Result<serde_json::Value> {
    let period = period_for(to - from);
    let off = store.utc_offset_minutes();
    let start = period.bucket(from, off);
    let cells = store.cells(period, start, to)?;
    let step = step_ms
        .unwrap_or_else(|| ((to - start) / 120).max(period.ms()))
        .max(period.ms());
    let step = step / period.ms() * period.ms();
    let mut total = Agg::default();
    let mut by_model: BTreeMap<String, Agg> = BTreeMap::new();
    let mut by_key: BTreeMap<String, Agg> = BTreeMap::new();
    let mut series: BTreeMap<i64, Agg> = BTreeMap::new();
    for c in &cells {
        total.merge(&c.agg);
        by_model.entry(c.model.clone()).or_default().merge(&c.agg);
        by_key.entry(c.key.clone()).or_default().merge(&c.agg);
        let b = start + (c.bucket - start).div_euclid(step) * step;
        series.entry(b).or_default().merge(&c.agg);
    }
    let named = |m: BTreeMap<String, Agg>| -> Vec<serde_json::Value> {
        let mut v: Vec<(String, Agg)> = m.into_iter().collect();
        v.sort_by(|a, b| {
            (b.1.prompt + b.1.completion)
                .cmp(&(a.1.prompt + a.1.completion))
                .then(a.0.cmp(&b.0))
        });
        v.into_iter()
            .map(|(name, a)| {
                let mut j = a.json();
                j["name"] = serde_json::Value::from(name);
                j
            })
            .collect()
    };
    let pts: Vec<serde_json::Value> = series
        .into_iter()
        .map(|(t, a)| {
            let mut j = a.json();
            j["t"] = serde_json::Value::from(t);
            j
        })
        .collect();
    let mut tot = total.json();
    tot["from"] = from.into();
    tot["to"] = to.into();
    tot["decode_tps_hist"] = hist_rows(&total.tps_hist, &TPS_BUCKETS);
    tot["ttft_hist"] = hist_rows(&total.ttft_hist, &TTFT_BUCKETS);
    Ok(serde_json::json!({
        "from": from,
        "to": to,
        "period": period.tag(),
        "step_ms": step,
        "totals": tot,
        "by_model": named(by_model),
        "by_key": named(by_key),
        "series": pts,
    }))
}

/// A histogram as `[{lo, hi, count}]` (occupied buckets only).
fn hist_rows(h: &Hist, b: &LogBuckets) -> serde_json::Value {
    h.0.iter()
        .filter(|(_, c)| **c > 0)
        .map(|(i, c)| {
            let i = usize::from(*i);
            let lo = if i == 0 { 0.0 } else { b.upper(i - 1) };
            serde_json::json!({"lo": lo, "hi": b.upper(i), "count": c})
        })
        .collect::<Vec<_>>()
        .into()
}

/// Lifetime totals per (model, key), from the day rollups (kept forever).
pub fn lifetime(store: &UsageStore) -> anyhow::Result<Vec<RollupCell>> {
    let cells = store.cells(Period::Day, i64::MIN, i64::MAX)?;
    let mut m: BTreeMap<(String, String), Agg> = BTreeMap::new();
    for c in cells {
        m.entry((c.model, c.key)).or_default().merge(&c.agg);
    }
    Ok(m.into_iter()
        .map(|((model, key), agg)| RollupCell {
            bucket: 0,
            model,
            key,
            agg,
        })
        .collect())
}

/// `/api/usage/monthly`: the last `months` months (from the day rollups + the energy table)
/// and the last `days` days for the calendar heatmap.
pub fn monthly(
    store: &UsageStore,
    now: i64,
    months: usize,
    days: usize,
) -> anyhow::Result<serde_json::Value> {
    let off = store.utc_offset_minutes();
    let cells = store.cells(Period::Day, i64::MIN, i64::MAX)?;
    let mut by_month: BTreeMap<String, Agg> = BTreeMap::new();
    let mut month_models: BTreeMap<String, BTreeMap<String, Agg>> = BTreeMap::new();
    let mut month_keys: BTreeMap<String, BTreeMap<String, Agg>> = BTreeMap::new();
    let mut by_day: BTreeMap<i64, Agg> = BTreeMap::new();
    let day_from = day_start(now, off) - (days as i64 - 1) * DAY_MS;
    for c in &cells {
        let m = month_of(c.bucket, off);
        by_month.entry(m.clone()).or_default().merge(&c.agg);
        month_models
            .entry(m.clone())
            .or_default()
            .entry(c.model.clone())
            .or_default()
            .merge(&c.agg);
        month_keys
            .entry(m)
            .or_default()
            .entry(c.key.clone())
            .or_default()
            .merge(&c.agg);
        if c.bucket >= day_from {
            by_day.entry(c.bucket).or_default().merge(&c.agg);
        }
    }
    let mut energy_month: BTreeMap<String, f64> = BTreeMap::new();
    let mut energy_day: BTreeMap<i64, f64> = BTreeMap::new();
    for (b, j) in store.energy(i64::MIN, i64::MAX)? {
        *energy_month.entry(month_of(b, off)).or_insert(0.0) += j;
        let d = day_start(b, off);
        if d >= day_from {
            *energy_day.entry(d).or_insert(0.0) += j;
        }
    }
    // the last `months` calendar months, oldest first, zero-filled
    let (mut y, mut mo, _) = civil((now + i64::from(off) * MINUTE_MS).div_euclid(DAY_MS));
    let mut names = Vec::new();
    for _ in 0..months.max(1) {
        names.push(format!("{y:04}-{mo:02}"));
        if mo == 1 {
            mo = 12;
            y -= 1;
        } else {
            mo -= 1;
        }
    }
    names.reverse();
    let top = |m: Option<&BTreeMap<String, Agg>>| -> Vec<serde_json::Value> {
        let mut v: Vec<(&String, &Agg)> = m.map(|m| m.iter().collect()).unwrap_or_default();
        v.sort_by_key(|x| std::cmp::Reverse(x.1.prompt + x.1.completion));
        v.into_iter()
            .take(8)
            .map(|(n, a)| {
                serde_json::json!({"name": n, "requests": a.requests,
                    "prompt_tokens": a.prompt, "completion_tokens": a.completion})
            })
            .collect()
    };
    let rows: Vec<serde_json::Value> = names
        .iter()
        .map(|n| {
            let a = by_month.get(n).cloned().unwrap_or_default();
            let mut j = a.json();
            j["month"] = n.clone().into();
            j["energy_kwh"] = energy_month.get(n).map(|j| j / 3.6e6).into();
            j["top_models"] = top(month_models.get(n)).into();
            j["top_keys"] = top(month_keys.get(n)).into();
            j
        })
        .collect();
    let day_rows: Vec<serde_json::Value> = (0..days as i64)
        .map(|i| {
            let d = day_from + i * DAY_MS;
            let a = by_day.get(&d).cloned().unwrap_or_default();
            serde_json::json!({
                "day": date_of(d, off),
                "t": d,
                "requests": a.requests,
                "errors": a.errors,
                "prompt_tokens": a.prompt,
                "completion_tokens": a.completion,
                "cached_tokens": a.cached,
                "energy_kwh": energy_day.get(&d).map(|j| j / 3.6e6),
            })
        })
        .collect();
    Ok(serde_json::json!({
        "utc_offset_minutes": off,
        "months": rows,
        "days": day_rows,
    }))
}

/// The console's Prometheus export of the long-term totals (`saber_console_*`).
pub fn prometheus(store: &UsageStore) -> anyhow::Result<String> {
    use std::fmt::Write as _;
    let esc = |s: &str| {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    };
    let life = lifetime(store)?;
    let mut o = String::new();
    let _ = writeln!(
        o,
        "# HELP saber_console_tokens_total Tokens served, from the console's usage store (all time)."
    );
    let _ = writeln!(o, "# TYPE saber_console_tokens_total counter");
    for c in &life {
        for (kind, v) in [
            ("prompt", c.agg.prompt),
            ("completion", c.agg.completion),
            ("cached", c.agg.cached),
        ] {
            let _ = writeln!(
                o,
                "saber_console_tokens_total{{kind=\"{kind}\",model=\"{}\",key=\"{}\"}} {v}",
                esc(&c.model),
                esc(&c.key)
            );
        }
    }
    let _ = writeln!(
        o,
        "# HELP saber_console_requests_total Requests recorded by the console's usage store."
    );
    let _ = writeln!(o, "# TYPE saber_console_requests_total counter");
    for c in &life {
        let _ = writeln!(
            o,
            "saber_console_requests_total{{model=\"{}\",key=\"{}\"}} {}",
            esc(&c.model),
            esc(&c.key),
            c.agg.requests
        );
    }
    let _ = writeln!(o, "# TYPE saber_console_request_errors_total counter");
    for c in &life {
        let _ = writeln!(
            o,
            "saber_console_request_errors_total{{model=\"{}\",key=\"{}\"}} {}",
            esc(&c.model),
            esc(&c.key),
            c.agg.errors
        );
    }
    let e: f64 = store
        .energy(i64::MIN, i64::MAX)?
        .iter()
        .map(|(_, j)| j)
        .sum();
    let _ = writeln!(o, "# TYPE saber_console_gpu_energy_joules_total counter");
    let _ = writeln!(o, "saber_console_gpu_energy_joules_total {e}");
    Ok(o)
}

/// CSV of rows (no bodies), RFC 4180 quoting.
#[must_use]
pub fn csv(rows: &[UsageRow]) -> String {
    use std::fmt::Write as _;
    let q = |s: &str| {
        if s.contains([',', '"', '\n', '\r']) {
            format!("\"{}\"", s.replace('"', "\"\""))
        } else {
            s.to_string()
        }
    };
    let o = |v: Option<String>| v.map(|s| q(&s)).unwrap_or_default();
    let mut out = String::from(
        "uid,ts_ms,time_utc,source,endpoint,model,key,stream,status,prompt_tokens,completion_tokens,\
         cached_tokens,ttft_ms,latency_ms,decode_tps,finish_reason,images,thinking,engine_boot,\
         engine_id,error\n",
    );
    for r in rows {
        let src = match (r.source & SRC_PROXY != 0, r.source & SRC_ENGINE != 0) {
            (true, true) => "proxy+engine",
            (true, false) => "proxy",
            _ => "engine",
        };
        let secs = r.ts_ms.div_euclid(1000);
        let (y, m, d) = civil(secs.div_euclid(86_400));
        let sod = secs.rem_euclid(86_400);
        let _ = writeln!(
            out,
            "{},{},{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z,{src},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            q(&r.uid),
            r.ts_ms,
            sod / 3600,
            sod % 3600 / 60,
            sod % 60,
            q(&r.endpoint),
            o(r.model.clone()),
            o(r.key.clone()),
            o(r.stream.map(|b| b.to_string())),
            o(r.status.map(|v| v.to_string())),
            o(r.prompt_tokens.map(|v| v.to_string())),
            o(r.completion_tokens.map(|v| v.to_string())),
            o(r.cached_tokens.map(|v| v.to_string())),
            o(r.ttft_ms.map(|v| v.to_string())),
            o(r.latency_ms.map(|v| v.to_string())),
            o(r.decode_tps.map(|v| v.to_string())),
            o(r.finish_reason.clone()),
            o(r.images.map(|v| v.to_string())),
            o(r.thinking.map(|b| b.to_string())),
            o(r.engine_boot.map(|v| v.to_string())),
            o(r.engine_id.map(|v| v.to_string())),
            o(r.error.clone()),
        );
    }
    out
}

// ---------------------------------------------------------------------------------------------
// the engine poller (direct-to-engine traffic)

/// A `/v1/requests` record as a usage row (`None` for a record without an id).
#[must_use]
pub fn row_from_engine(rec: &serde_json::Value, boot: u64) -> Option<UsageRow> {
    let id = rec["id"].as_i64()?;
    let boot = i64::try_from(boot).ok()?;
    let s = |k: &str| rec[k].as_str().map(str::to_string);
    let uid = match rec["client_ref"].as_str() {
        Some(r) if !r.is_empty() => r.to_string(),
        _ => format!("e:{boot}:{id}"),
    };
    Some(UsageRow {
        uid,
        ts_ms: rec["ts_ms"].as_i64().unwrap_or(0),
        source: SRC_ENGINE,
        endpoint: s("endpoint").unwrap_or_default(),
        model: s("model"),
        key: s("key"),
        stream: rec["stream"].as_bool(),
        status: rec["status"].as_i64(),
        prompt_tokens: rec["prompt_tokens"].as_i64(),
        completion_tokens: rec["completion_tokens"].as_i64(),
        cached_tokens: rec["cached_tokens"].as_i64(),
        ttft_ms: rec["ttft_ms"].as_f64(),
        latency_ms: rec["latency_ms"].as_f64(),
        decode_tps: rec["decode_tps"].as_f64(),
        finish_reason: s("finish_reason"),
        images: rec["images"].as_i64(),
        thinking: rec["thinking"].as_bool(),
        engine_boot: Some(boot),
        engine_id: Some(id),
        error: None,
        request_body: None,
        response_body: None,
    })
}

/// One poll page: queue its rows and the new cursor; returns (rows, next cursor, more pages).
pub fn ingest_page(
    store: &UsageStore,
    page: &serde_json::Value,
    cursor: Option<(u64, u64)>,
) -> (usize, (u64, u64), bool) {
    let boot = page["boot_id"].as_u64().unwrap_or(0);
    let mut after = match cursor {
        Some((b, a)) if b == boot => a,
        _ => 0,
    };
    let mut n = 0;
    if let Some(recs) = page["data"].as_array() {
        for r in recs {
            if let Some(row) = row_from_engine(r, boot) {
                after = after.max(u64::try_from(row.engine_id.unwrap_or(0)).unwrap_or(0));
                store.record(row);
                n += 1;
            }
        }
    }
    if n > 0 || cursor.is_none_or(|(b, _)| b != boot) {
        store.send(Op::Cursor { boot, after });
    }
    let newest = page["newest_id"].as_u64().unwrap_or(0);
    (n, (boot, after), newest > after)
}

/// The fallback poller: every `usage.poll_interval_s`, read the engine's `/v1/requests` past
/// the stored cursor (with the engine's Bearer key) and record each request exactly once.
pub async fn poll(
    store: std::sync::Arc<UsageStore>,
    sup: std::sync::Arc<crate::supervisor::Supervisor>,
) {
    let mut cursor = store.cursor().ok().flatten();
    loop {
        let interval = sup.config_now().usage.poll_interval_s.max(1);
        tokio::time::sleep(Duration::from_secs(interval)).await;
        let Some(port) = sup.live_port() else {
            continue;
        };
        let bearer = sup.engine_bearer();
        // drain every page that is ready
        for _ in 0..50 {
            // a cursor from another engine boot restarts at 0; the page's boot id says which
            let after = cursor.map_or(0, |(_, a)| a);
            let path = format!("/v1/requests?after={after}&limit=1000");
            let Ok((200, body)) =
                crate::httpc::get_auth(port, &path, Duration::from_secs(5), bearer.as_deref())
                    .await
            else {
                break;
            };
            let Ok(page) = serde_json::from_str::<serde_json::Value>(&body) else {
                break;
            };
            let boot = page["boot_id"].as_u64().unwrap_or(0);
            if cursor.is_some_and(|(b, _)| b != boot) && after != 0 {
                // a new engine boot: its ids restart at 1
                cursor = Some((boot, 0));
                continue;
            }
            let dropped = page["dropped"].as_u64().unwrap_or(0);
            if dropped > 0 {
                sup.logs.push(
                    "console",
                    format!(
                        "saber ui: usage poller fell {dropped} request(s) behind the engine's ring \
                         (direct-to-engine requests not recorded)"
                    ),
                );
            }
            let (_, next, more) = ingest_page(&store, &page, cursor);
            cursor = Some(next);
            if !more {
                break;
            }
        }
    }
}

/// The retention job: prune raw rows older than `usage.raw_retention_days`, hourly.
pub async fn retention(
    store: std::sync::Arc<UsageStore>,
    sup: std::sync::Arc<crate::supervisor::Supervisor>,
) {
    loop {
        let days = i64::from(sup.config_now().usage.raw_retention_days.max(1));
        store.send(Op::Prune {
            before_ms: crate::now_ms() as i64 - days * DAY_MS,
        });
        tokio::time::sleep(Duration::from_secs(3600)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(uid: &str, ts: i64, src: i64) -> UsageRow {
        UsageRow {
            uid: uid.into(),
            ts_ms: ts,
            source: src,
            endpoint: "/v1/chat/completions".into(),
            model: Some("m".into()),
            status: Some(200),
            ..UsageRow::default()
        }
    }

    #[test]
    fn buckets_and_quantiles() {
        let b = TTFT_BUCKETS;
        assert_eq!(b.index(0.5), 0);
        assert_eq!(b.index(1.0), 0);
        assert_eq!(b.index(1.1), 1);
        assert_eq!(b.index(2.0), 4);
        assert_eq!(b.index(1e12), b.n - 1);
        let mut h = Hist::default();
        for v in [10.0, 20.0, 40.0, 80.0] {
            h.add(b.index(v), 1);
        }
        let p50 = h.quantile(0.5, &b).unwrap();
        assert!((p50 / 20.0 - 1.0).abs() < 0.1, "{p50}");
        assert_eq!(Hist::decode(&h.encode()), h);
        h.add(b.index(10.0), -1);
        assert_eq!(h.count(), 3);
        assert_eq!(Hist::default().quantile(0.5, &b), None);
    }

    #[test]
    fn calendar_math() {
        assert_eq!(civil(0), (1970, 1, 1));
        assert_eq!(civil(20_406), (2025, 11, 14));
        assert_eq!(month_of(0, 0), "1970-01");
        // 1970-01-01 03:00 UTC is still Dec 31 at UTC-6
        assert_eq!(date_of(3 * HOUR_MS, -360), "1969-12-31");
        assert_eq!(day_start(3 * HOUR_MS, -360), -DAY_MS + 6 * HOUR_MS);
    }

    #[test]
    fn merge_keeps_each_sides_fields() {
        let mut p = row("c1", 1000, SRC_PROXY);
        p.ttft_ms = Some(50.0);
        p.key = Some("console-key".into());
        p.prompt_tokens = Some(1);
        let mut e = row("c1", 990, SRC_ENGINE);
        e.ttft_ms = Some(45.0);
        e.prompt_tokens = Some(12);
        e.completion_tokens = Some(7);
        e.key = Some("console".into());
        e.engine_id = Some(4);
        for m in [
            UsageRow::merged(p.clone(), e.clone()),
            UsageRow::merged(e.clone(), p.clone()),
        ] {
            assert_eq!(m.source, SRC_PROXY | SRC_ENGINE);
            assert_eq!(m.ts_ms, 1000);
            assert_eq!(m.ttft_ms, Some(50.0));
            assert_eq!(m.prompt_tokens, Some(12));
            assert_eq!(m.completion_tokens, Some(7));
            assert_eq!(m.key.as_deref(), Some("console-key"));
            assert_eq!(m.engine_id, Some(4));
        }
    }
}
