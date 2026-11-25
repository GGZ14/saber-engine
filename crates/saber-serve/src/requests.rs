//! The recent-requests ring (`GET /v1/requests`) and the per-request tracker that feeds it and
//! the latency histograms.
//!
//! **No prompt or response body is ever stored here** -- token counts, timings and labels only.
//! A record is written ONCE, when its request ends (the tracker's drop), from instants the HTTP
//! handler took on its own task: one `Instant::now()` per engine event on the HTTP side, nothing
//! on the step thread.
//!
//! Ids are assigned at insertion under the ring's lock, so they increase by one in ring order and
//! a poller's `?after=<last id it saw>` never skips a record that is still in the ring; a poller
//! that fell further behind than the ring holds is told so (`dropped`).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};

use serde::Serialize;

use crate::engine::FinishReason;
use crate::hist::LocalHist;
use crate::metrics::Metrics;

/// The ring's default length (`--request-ring`).
pub const DEFAULT_RING: usize = 1024;
/// The most records one `/v1/requests` page returns.
pub const MAX_PAGE: usize = 1000;
/// A page's default length.
pub const DEFAULT_PAGE: usize = 100;

/// HTTP status recorded for a request whose client went away before it finished (nginx's).
pub const STATUS_CLIENT_CLOSED: u16 = 499;

/// One finished request, as `/v1/requests` reports it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RequestRecord {
    /// Monotone from 1, stable for the life of the process.
    pub id: u64,
    /// Arrival, unix milliseconds.
    pub ts_ms: u64,
    /// `/v1/chat/completions` or `/v1/completions`.
    pub endpoint: &'static str,
    pub model: String,
    /// The API key's label, when `--api-key` auth is on.
    pub key: Option<String>,
    pub stream: bool,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Prompt tokens served from the prefix cache, when the engine reports it (`null` = unknown).
    pub cached_tokens: Option<u64>,
    /// Arrival -> the first event that carried a token.
    pub ttft_ms: Option<f64>,
    /// Arrival -> the end.
    pub latency_ms: f64,
    /// Tokens after the first event over the span from the first event to the last.
    pub decode_tps: Option<f64>,
    pub finish_reason: Option<&'static str>,
    /// 200, 500 (the engine ended it with an error) or 499 (the client went away).
    pub status: u16,
    pub images: u32,
    /// `enable_thinking` as rendered (`null` for `/v1/completions`).
    pub thinking: Option<bool>,
    /// The `x-saber-console-ref` header the console's proxy set on the request
    /// (an opaque id, never content), so the console's usage store records a proxied request
    /// once. `null` for a client that called the engine directly.
    pub client_ref: Option<String>,
}

/// The header the console's `/v1` proxy tags a request with ([`RequestRecord::client_ref`]).
pub const CLIENT_REF_HEADER: &str = "x-saber-console-ref";

/// The header's value when it is a plausible console ref (1..=64 bytes of `[A-Za-z0-9_-]`);
/// anything else is ignored.
#[must_use]
pub fn client_ref(headers: &axum::http::HeaderMap) -> Option<String> {
    let v = headers.get(CLIENT_REF_HEADER)?.to_str().ok()?;
    let ok = !v.is_empty()
        && v.len() <= 64
        && v.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    ok.then(|| v.to_owned())
}

/// Totals since the process started (every request that reached the engine).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Lifetime {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_tokens: u64,
    pub images: u64,
    pub errors: u64,
    pub client_closed: u64,
}

#[derive(Debug)]
struct Inner {
    ring: VecDeque<RequestRecord>,
    next_id: u64,
    life: Lifetime,
}

/// The bounded ring.
#[derive(Debug)]
pub struct RequestRing {
    cap: usize,
    /// Unix ms the ring was built (once a process): ids restart at 1 with a new boot id, so a
    /// poller keys its cursor by (boot id, id).
    boot_id: u64,
    inner: Mutex<Inner>,
}

/// One `/v1/requests` page.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RequestsPage {
    pub requests: Vec<RequestRecord>,
    /// The id to pass as `after` next time (the last record returned, or `after` itself).
    pub next_after: u64,
    /// The oldest and newest ids the ring holds (0 when empty).
    pub oldest_id: u64,
    pub newest_id: u64,
    /// Records after `after` that the ring no longer holds (the poller fell behind).
    pub dropped: u64,
    pub capacity: usize,
    /// [`RequestRing::boot_id`].
    pub boot_id: u64,
}

impl RequestRing {
    #[must_use]
    pub fn new(cap: usize) -> Self {
        let cap = cap.max(1);
        let boot_id = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));
        Self {
            cap,
            boot_id,
            inner: Mutex::new(Inner {
                ring: VecDeque::with_capacity(cap.min(4096)),
                next_id: 1,
                life: Lifetime::default(),
            }),
        }
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    #[must_use]
    pub fn boot_id(&self) -> u64 {
        self.boot_id
    }

    /// Insert `rec` (its `id` is assigned here) and return the id.
    pub fn push(&self, mut rec: RequestRecord) -> u64 {
        let Ok(mut g) = self.inner.lock() else {
            return 0;
        };
        rec.id = g.next_id;
        g.next_id += 1;
        let l = &mut g.life;
        l.requests += 1;
        l.prompt_tokens += rec.prompt_tokens;
        l.completion_tokens += rec.completion_tokens;
        l.cached_tokens += rec.cached_tokens.unwrap_or(0);
        l.images += u64::from(rec.images);
        if rec.status >= 500 {
            l.errors += 1;
        } else if rec.status == STATUS_CLIENT_CLOSED {
            l.client_closed += 1;
        }
        if g.ring.len() == self.cap {
            g.ring.pop_front();
        }
        let id = rec.id;
        g.ring.push_back(rec);
        id
    }

    #[must_use]
    pub fn lifetime(&self) -> Lifetime {
        self.inner.lock().map(|g| g.life).unwrap_or_default()
    }

    /// Records with `id > after`, oldest first, at most `limit` (clamped to 1..=[`MAX_PAGE`]).
    #[must_use]
    pub fn page(&self, after: u64, limit: usize) -> RequestsPage {
        let limit = limit.clamp(1, MAX_PAGE);
        let Ok(g) = self.inner.lock() else {
            return RequestsPage {
                requests: Vec::new(),
                next_after: after,
                oldest_id: 0,
                newest_id: 0,
                dropped: 0,
                capacity: self.cap,
                boot_id: self.boot_id,
            };
        };
        let oldest = g.ring.front().map_or(0, |r| r.id);
        let newest = g.ring.back().map_or(0, |r| r.id);
        // ids are contiguous in the ring: the first record past `after` is at a computed index
        let start = if oldest == 0 || after < oldest {
            0
        } else {
            usize::try_from(after - oldest + 1).unwrap_or(usize::MAX)
        };
        let requests: Vec<RequestRecord> = g.ring.iter().skip(start).take(limit).cloned().collect();
        let dropped = if oldest > 0 && after + 1 < oldest {
            oldest - after - 1
        } else {
            0
        };
        RequestsPage {
            next_after: requests.last().map_or(after, |r| r.id),
            requests,
            oldest_id: oldest,
            newest_id: newest,
            dropped,
            capacity: self.cap,
            boot_id: self.boot_id,
        }
    }
}

/// What a tracker is told at the start of a request.
#[derive(Debug, Clone)]
pub struct TrackStart {
    pub endpoint: &'static str,
    pub model: String,
    pub key: Option<String>,
    pub stream: bool,
    pub prompt_tokens: u64,
    pub images: u32,
    pub thinking: Option<bool>,
    /// [`RequestRecord::client_ref`].
    pub client_ref: Option<String>,
}

/// One request's timing, recorded into the histograms and the ring when it is dropped.
///
/// Dropped without [`RequestTracker::finish`] = the client went away (499): a blocking
/// request's future dropped by axum, or a stream's generator that returned on a failed send.
#[derive(Debug)]
pub struct RequestTracker {
    metrics: Arc<Metrics>,
    ring: Arc<RequestRing>,
    start: Instant,
    ts_ms: u64,
    info: TrackStart,
    first: Option<Instant>,
    first_tokens: u64,
    last: Option<Instant>,
    itl: LocalHist,
    cached: Option<u64>,
    end: Option<(u64, Option<FinishReason>, u16)>,
    /// Tokens the engine delivered so far (what a request that never reached [`Self::finish`]
    /// reports).
    end_partial: u64,
}

impl RequestTracker {
    /// `start` is the request's arrival (the handler's entry).
    #[must_use]
    pub fn new(
        metrics: Arc<Metrics>,
        ring: Arc<RequestRing>,
        start: Instant,
        info: TrackStart,
    ) -> Self {
        let ts_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0))
            .saturating_sub(u64::try_from(start.elapsed().as_millis()).unwrap_or(0));
        let itl = metrics.itl_local();
        Self {
            metrics,
            ring,
            start,
            ts_ms,
            info,
            first: None,
            first_tokens: 0,
            last: None,
            itl,
            cached: None,
            end: None,
            end_partial: 0,
        }
    }

    /// An engine event carrying `tokens` new tokens (and, on its first, the cached-prefix count).
    pub fn on_event(&mut self, tokens: usize, cached: Option<u32>) {
        if self.cached.is_none() {
            self.cached = cached.map(u64::from);
        }
        if tokens == 0 {
            return;
        }
        self.end_partial += tokens as u64;
        let now = Instant::now();
        match self.last {
            None => {
                self.first = Some(now);
                self.first_tokens = tokens as u64;
            }
            Some(prev) => self.itl.observe(now.saturating_duration_since(prev)),
        }
        self.last = Some(now);
    }

    /// llama.cpp-style `timings` for the reply: wall-clock prefill of the uncached prompt tokens
    /// (arrival to first token, scheduling and cache reloads included) and effective decode rate. `None` before any token.
    #[must_use]
    pub fn timings(
        &self,
        prompt_tokens: usize,
        generated: u64,
    ) -> Option<crate::protocol::ResponseTimings> {
        let first = self.first?;
        let prompt_ms = first.saturating_duration_since(self.start).as_secs_f64() * 1000.0;
        let predicted_ms = self.last?.saturating_duration_since(first).as_secs_f64() * 1000.0;
        // Like llama.cpp, `prompt_n` counts only freshly evaluated tokens so the prefill rate
        // is not inflated by a cache-reused prefix; the reused part is reported as `cache_n`.
        let cache_n = self.cached.unwrap_or(0).min(prompt_tokens as u64);
        let prompt_n = prompt_tokens as u64 - cache_n;
        Some(crate::protocol::ResponseTimings {
            prompt_n,
            cache_n,
            predicted_n: generated,
            prompt_ms,
            predicted_ms,
            prompt_per_second: if prompt_ms > 0.0 && prompt_n > 0 {
                prompt_n as f64 * 1000.0 / prompt_ms
            } else {
                -1.0
            },
            predicted_per_second: if predicted_ms > 0.0 && generated > self.first_tokens {
                (generated - self.first_tokens) as f64 * 1000.0 / predicted_ms
            } else {
                -1.0
            },
        })
    }

    /// The request ended and its reply was assembled / its stream closed.
    pub fn finish(&mut self, completion_tokens: u64, reason: Option<FinishReason>, status: u16) {
        self.end = Some((completion_tokens, reason, status));
    }
}

impl Drop for RequestTracker {
    fn drop(&mut self) {
        let end_at = Instant::now();
        let (completion_tokens, reason, status) = self.end.unwrap_or((
            self.end_partial,
            Some(FinishReason::Abort),
            STATUS_CLIENT_CLOSED,
        ));
        let ttft = self.first.map(|f| f.saturating_duration_since(self.start));
        let latency = end_at.saturating_duration_since(self.start);
        let decode_tps = match (self.first, self.last) {
            (Some(f), Some(l)) if l > f && completion_tokens > self.first_tokens => Some(
                (completion_tokens - self.first_tokens) as f64
                    / l.saturating_duration_since(f).as_secs_f64(),
            ),
            _ => None,
        };
        self.metrics
            .observe_request(ttft, &self.itl, status == 200, latency);
        let ms = |d: std::time::Duration| (d.as_secs_f64() * 1e6).round() / 1e3;
        self.ring.push(RequestRecord {
            id: 0,
            ts_ms: self.ts_ms,
            endpoint: self.info.endpoint,
            model: std::mem::take(&mut self.info.model),
            key: self.info.key.take(),
            stream: self.info.stream,
            prompt_tokens: self.info.prompt_tokens,
            completion_tokens,
            cached_tokens: self.cached,
            ttft_ms: ttft.map(ms),
            latency_ms: ms(latency),
            decode_tps: decode_tps.map(|v| (v * 100.0).round() / 100.0),
            finish_reason: reason.map(FinishReason::as_str),
            status,
            images: self.info.images,
            thinking: self.info.thinking,
            client_ref: self.info.client_ref.take(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(prompt: u64, status: u16) -> RequestRecord {
        RequestRecord {
            id: 0,
            ts_ms: 1,
            endpoint: "/v1/chat/completions",
            model: "m".into(),
            key: None,
            stream: false,
            prompt_tokens: prompt,
            completion_tokens: 2,
            cached_tokens: Some(1),
            ttft_ms: None,
            latency_ms: 1.0,
            decode_tps: None,
            finish_reason: Some("stop"),
            status,
            images: 0,
            thinking: None,
            client_ref: None,
        }
    }

    #[test]
    fn ids_are_monotone_and_the_ring_wraps() {
        let r = RequestRing::new(3);
        for i in 1..=5 {
            assert_eq!(r.push(rec(i, 200)), i);
        }
        let p = r.page(0, 100);
        assert_eq!(
            p.requests.iter().map(|x| x.id).collect::<Vec<_>>(),
            vec![3, 4, 5],
            "the oldest two fell out"
        );
        assert_eq!((p.oldest_id, p.newest_id, p.next_after), (3, 5, 5));
        assert_eq!(p.dropped, 2, "ids 1 and 2 after 0 are gone");
        // lifetime totals count every request, not only those still held
        let l = r.lifetime();
        assert_eq!((l.requests, l.prompt_tokens, l.cached_tokens), (5, 15, 5));
    }

    #[test]
    fn after_and_limit_semantics() {
        let r = RequestRing::new(10);
        for i in 1..=6 {
            r.push(rec(i, 200));
        }
        let p = r.page(2, 2);
        assert_eq!(
            p.requests.iter().map(|x| x.id).collect::<Vec<_>>(),
            vec![3, 4]
        );
        assert_eq!((p.next_after, p.dropped), (4, 0));
        let p = r.page(p.next_after, 100);
        assert_eq!(
            p.requests.iter().map(|x| x.id).collect::<Vec<_>>(),
            vec![5, 6]
        );
        let p = r.page(6, 100);
        assert!(p.requests.is_empty());
        assert_eq!(p.next_after, 6, "an empty page keeps the cursor");
        let p = r.page(60, 100);
        assert!(
            p.requests.is_empty(),
            "a cursor past the newest is empty, not a wrap"
        );
        // limit 0 is clamped to 1
        assert_eq!(r.page(0, 0).requests.len(), 1);
    }

    #[test]
    fn client_ref_header_is_validated() {
        let mut h = axum::http::HeaderMap::new();
        assert_eq!(client_ref(&h), None);
        h.insert(CLIENT_REF_HEADER, "c18f-2a_x".parse().unwrap());
        assert_eq!(client_ref(&h).as_deref(), Some("c18f-2a_x"));
        h.insert(CLIENT_REF_HEADER, "has space".parse().unwrap());
        assert_eq!(client_ref(&h), None);
        h.insert(CLIENT_REF_HEADER, "x".repeat(65).parse().unwrap());
        assert_eq!(client_ref(&h), None);
    }

    #[test]
    fn statuses_feed_the_error_and_closed_totals() {
        let r = RequestRing::new(4);
        r.push(rec(1, 200));
        r.push(rec(1, 500));
        r.push(rec(1, STATUS_CLIENT_CLOSED));
        let l = r.lifetime();
        assert_eq!((l.errors, l.client_closed), (1, 1));
    }

    #[test]
    fn a_dropped_tracker_is_a_client_closed_record() {
        let m = Arc::new(Metrics::new("m"));
        let ring = Arc::new(RequestRing::new(8));
        let info = TrackStart {
            endpoint: "/v1/completions",
            model: "m".into(),
            key: Some("ci".into()),
            stream: true,
            prompt_tokens: 7,
            images: 0,
            thinking: None,
            client_ref: Some("c1-2".into()),
        };
        {
            let mut t = RequestTracker::new(
                Arc::clone(&m),
                Arc::clone(&ring),
                Instant::now(),
                info.clone(),
            );
            t.on_event(1, Some(3));
            t.on_event(2, None);
        }
        {
            let mut t =
                RequestTracker::new(Arc::clone(&m), Arc::clone(&ring), Instant::now(), info);
            t.on_event(1, None);
            t.finish(1, Some(FinishReason::Stop), 200);
        }
        let p = ring.page(0, 10);
        let (a, b) = (&p.requests[0], &p.requests[1]);
        assert_eq!(
            (a.status, a.finish_reason, a.completion_tokens),
            (499, Some("abort"), 3)
        );
        assert_eq!(a.cached_tokens, Some(3));
        assert_eq!(a.key.as_deref(), Some("ci"));
        assert_eq!(a.client_ref.as_deref(), Some("c1-2"));
        assert_eq!(p.boot_id, ring.boot_id());
        assert!(p.boot_id > 0);
        assert!(a.ttft_ms.is_some());
        assert_eq!((b.status, b.finish_reason), (200, Some("stop")));
        assert_eq!(b.decode_tps, None, "one event: no decode span");
        let s = m.latency_snapshots();
        assert_eq!(s.ttft.count, 2);
        assert_eq!(s.itl.count, 1);
        assert_eq!(s.e2e.count, 1, "e2e counts completed requests only");
    }
}
