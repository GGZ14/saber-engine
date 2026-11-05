//! What the mirror writes and the replay writes beside it: one JSON object per line.
//!
//! Three kinds of line share a mirror log (`$SABER_SHARED_CACHE/shadow/<UTC date>.jsonl`):
//! `start` (one per mirror process), `exchange` (one per recorded request, written when its
//! response has ended) and `metrics` (the upstream's serving gauges and spec counters, polled on a
//! timer so the acceptance of a window can be read without touching the request path). A replay
//! file holds one `replay_meta` line and one `answer` per replayed exchange.

use std::io::{BufRead, Write};
use std::path::Path;

use anyhow::Context;
use serde::{Deserialize, Serialize};

/// Bytes, as text when they are UTF-8 and as hex when they are not. A chunk boundary can split a
/// multi-byte character, so this is decided per blob and the reader always works on the bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Blob {
    /// Valid UTF-8, stored as is.
    Text {
        /// The bytes, as a string.
        text: String,
    },
    /// Anything else, hex.
    Hex {
        /// The bytes, lowercase hex.
        hex: String,
    },
}

impl Blob {
    /// Wraps `bytes`.
    pub fn of(bytes: &[u8]) -> Self {
        match std::str::from_utf8(bytes) {
            Ok(s) => Self::Text { text: s.to_owned() },
            Err(_) => Self::Hex {
                hex: hex::encode(bytes),
            },
        }
    }

    /// The bytes back.
    pub fn bytes(&self) -> Vec<u8> {
        match self {
            Self::Text { text } => text.as_bytes().to_vec(),
            Self::Hex { hex } => hex::decode(hex).unwrap_or_default(),
        }
    }
}

/// One body chunk, with the time it arrived.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    /// Milliseconds since the request arrived (mirror) or was sent (replay).
    pub t_ms: f64,
    /// The bytes.
    pub data: Blob,
}

/// A response as one side saw it: status, headers and every body chunk with its time.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Response {
    /// HTTP status, `None` when the upstream could not be reached at all.
    pub status: Option<u16>,
    /// Response headers, in order (`content-type` is the one the reader uses).
    pub headers: Vec<(String, String)>,
    /// Every body chunk, in order.
    pub chunks: Vec<Chunk>,
    /// First body byte, ms after the request.
    pub t_first_byte_ms: Option<f64>,
    /// Body end, ms after the request.
    pub t_done_ms: Option<f64>,
    /// Whether the body ended normally (not a transport error, not a client that left).
    pub complete: bool,
    /// Transport error, if any.
    pub error: Option<String>,
}

impl Response {
    /// The body, reassembled.
    pub fn body(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for c in &self.chunks {
            out.extend_from_slice(&c.data.bytes());
        }
        out
    }
}

/// One recorded request and prod's response to it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Exchange {
    /// Mirror-local sequence number, from 1.
    pub id: u64,
    /// Arrival, Unix ms (wall).
    pub t_unix_ms: u64,
    /// Arrival, ms since the mirror started (monotonic): the replay's offsets.
    pub t_mono_ms: f64,
    /// `POST`.
    pub method: String,
    /// Path and query, as the client sent them.
    pub path: String,
    /// The request headers that matter to a replay (`content-type`, `accept`, `user-agent`, …).
    /// `authorization` is recorded as present or not, never its value.
    pub headers: Vec<(String, String)>,
    /// The request body, byte for byte.
    pub body: Blob,
    /// `"stream": true` in the body.
    pub stream: bool,
    /// Prod's response.
    pub response: Response,
    /// Bytes the upstream sent.
    pub upstream_bytes: u64,
    /// Bytes the mirror forwarded to the client. Equal to `upstream_bytes`, or the mirror changed
    /// what the client received (stop condition 6).
    pub forwarded_bytes: u64,
    /// FNV-1a 64 of the upstream bytes, hex.
    pub upstream_fnv: String,
    /// FNV-1a 64 of the forwarded bytes, hex.
    pub forwarded_fnv: String,
    /// The client went away before the body ended (not a mirror fault; the upstream saw it too).
    pub client_gone: bool,
}

/// The serving families a report reads, by name after the exposition namespace (`saber:` on
/// Saber; the prod server exports the same families under its own namespace).
pub const SERVED_FAMILIES: &[&str] = &[
    "spec_decode",
    "num_requests",
    "request_success_total",
    "prompt_tokens_total",
    "generation_tokens_total",
];

/// `Some(rest)` when an exposition line (or metric key) is `<namespace>:<rest>` with `rest` one
/// of [`SERVED_FAMILIES`], whatever the namespace.
pub fn served_family(line: &str) -> Option<&str> {
    let (ns, rest) = line.split_once(':')?;
    let ns_ok = !ns.is_empty() && ns.bytes().all(|b| b.is_ascii_lowercase() || b == b'_');
    (ns_ok && SERVED_FAMILIES.iter().any(|f| rest.starts_with(f))).then_some(rest)
}

/// The upstream's `/metrics`, filtered to the [`SERVED_FAMILIES`] a report reads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Unix ms.
    pub t_unix_ms: u64,
    /// ms since the mirror started.
    pub t_mono_ms: f64,
    /// The exposition lines kept ([`served_family`]: `*:spec_decode_*`, `*:num_requests_*`,
    /// `*:request_success_total`, `*:prompt_tokens_total`, `*:generation_tokens_total`).
    pub text: String,
}

/// A mirror log line.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LogEntry {
    /// A mirror process started.
    Start {
        /// Unix ms.
        t_unix_ms: u64,
        /// Where it listened.
        listen: String,
        /// What it forwarded to.
        upstream: String,
    },
    /// A recorded exchange.
    Exchange(Box<Exchange>),
    /// A poll of the upstream's `/metrics`.
    Metrics(MetricsSnapshot),
}

/// The replay's own facts, first line of a replay file.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReplayMeta {
    /// The endpoint replayed against.
    pub target: String,
    /// `recorded` (arrival offsets) or `sequential`.
    pub pace: String,
    /// Offsets divided by this (`recorded` only). Written into the report.
    pub speed: f64,
    /// Unix ms at the first send.
    pub started_unix_ms: u64,
    /// Exchanges replayed.
    pub replayed: u64,
    /// The target's whole `/metrics` before the first request.
    pub metrics_before: Option<String>,
    /// And after the last response.
    pub metrics_after: Option<String>,
    /// `power1_cap` of every card found, watts, read before and after.
    pub power_caps_w: Vec<(String, f64)>,
}

/// One replayed exchange's answer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Answer {
    /// The exchange's `id`.
    pub id: u64,
    /// When it was sent, ms after the replay started.
    pub t_sent_ms: f64,
    /// Saber's response.
    pub response: Response,
}

/// A replay file line.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReplayEntry {
    /// The replay's facts.
    ReplayMeta(ReplayMeta),
    /// One answer.
    Answer(Box<Answer>),
}

/// FNV-1a 64, the mirror's cheap byte-identity check.
#[derive(Clone, Copy, Debug)]
pub struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Fnv {
    /// Folds `bytes` in.
    pub fn update(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    /// The digest, hex.
    pub fn hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

/// Reads a mirror log. Lines that do not parse are an error naming the line: a log that cannot
/// be read is not a log that says nothing happened.
pub fn read_log(path: &Path) -> anyhow::Result<Vec<LogEntry>> {
    read_jsonl(path)
}

/// Reads a replay file.
pub fn read_replay(path: &Path) -> anyhow::Result<(ReplayMeta, Vec<Answer>)> {
    let mut meta = None;
    let mut answers = Vec::new();
    for e in read_jsonl::<ReplayEntry>(path)? {
        match e {
            ReplayEntry::ReplayMeta(m) => meta = Some(m),
            ReplayEntry::Answer(a) => answers.push(*a),
        }
    }
    Ok((
        meta.with_context(|| format!("{}: no replay_meta line", path.display()))?,
        answers,
    ))
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> anyhow::Result<Vec<T>> {
    let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in std::io::BufReader::new(f).lines().enumerate() {
        let line = line.with_context(|| format!("{}:{}", path.display(), i + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(
            serde_json::from_str(&line)
                .with_context(|| format!("{}:{}: not a shadow record", path.display(), i + 1))?,
        );
    }
    Ok(out)
}

/// Appends one line (one `write_all` of the whole line, the file opened `O_APPEND`).
pub fn append_line<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    let mut line = serde_json::to_vec(value)?;
    line.push(b'\n');
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening {} for append", path.display()))?;
    f.write_all(&line)?;
    Ok(())
}

/// Unix ms now.
pub fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// `YYYY-MM-DD` (UTC) of a Unix ms time: the day file a record goes to.
pub fn utc_date(unix_ms: u64) -> String {
    // Howard Hinnant's civil_from_days.
    let z = (unix_ms / 86_400_000) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blobs_round_trip_text_and_split_characters() {
        let text = "héllo".as_bytes();
        assert_eq!(Blob::of(text).bytes(), text);
        assert!(matches!(Blob::of(text), Blob::Text { .. }));
        // A chunk boundary inside `é` is not UTF-8 and must still come back byte for byte.
        let half = &text[..2];
        assert!(matches!(Blob::of(half), Blob::Hex { .. }));
        assert_eq!(Blob::of(half).bytes(), half);
        let json = serde_json::to_string(&Blob::of(half)).unwrap();
        let back: Blob = serde_json::from_str(&json).unwrap();
        assert_eq!(back.bytes(), half);
    }

    #[test]
    fn utc_dates() {
        assert_eq!(utc_date(0), "1970-01-01");
        assert_eq!(utc_date(1_790_138_632_000), "2026-09-23");
        assert_eq!(utc_date(951_782_400_000), "2000-02-29");
    }

    #[test]
    fn fnv_is_order_sensitive() {
        let mut a = Fnv::default();
        a.update(b"ab");
        let mut b = Fnv::default();
        b.update(b"ba");
        assert_ne!(a.hex(), b.hex());
        let mut c = Fnv::default();
        c.update(b"a");
        c.update(b"b");
        assert_eq!(a.hex(), c.hex(), "chunking does not change the digest");
    }
}
