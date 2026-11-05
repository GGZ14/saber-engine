//! The recording mirror (`docs/SHADOW.md` §1, mode R).
//!
//! A reverse proxy that forwards every request to the upstream **byte for byte** and, for the
//! two generation routes, records the exchange on the side. The rules it is built to, from the
//! runbook:
//!
//! * **transparent**: the request body is forwarded unchanged, the response status and headers
//!   are forwarded (hop-by-hop headers excepted, as any HTTP/1.1 proxy must), and a streamed body
//!   is forwarded chunk by chunk **as each chunk arrives** — never buffered — and copied on the
//!   side. The bytes forwarded and the bytes received are both hashed into the record, so the
//!   reader can prove, per request, that the client got exactly what the upstream sent;
//! * **a failure to log never fails the client**: records go to a writer thread over a channel;
//!   a write error is counted on `/shadow/health` and nothing else happens;
//! * **a client that leaves is a client that left**: the upstream stream is dropped with it, so
//!   the upstream sees the same disconnect it would have seen without the mirror;
//! * `/shadow/health` is the mirror's own, answered without touching the upstream.
//!
//! What is recorded: `POST /v1/chat/completions` and `POST /v1/completions`. Everything else
//! (`/metrics`, `/v1/models`, `/health`) is forwarded and not recorded.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Request, StatusCode};
use axum::response::{IntoResponse, Response as AxumResponse};
use axum::Router;
use bytes::Bytes;
use futures_util::StreamExt;

use crate::record::{
    append_line, unix_ms, utc_date, Blob, Chunk, Exchange, Fnv, LogEntry, MetricsSnapshot, Response,
};

/// The routes whose exchanges are recorded.
pub const RECORDED: &[&str] = &["/v1/chat/completions", "/v1/completions"];

/// Headers a proxy must not forward (RFC 9110 §7.6.1), plus the ones the HTTP stack recomputes.
const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-connection",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
];

/// Request headers worth keeping in the record for a replay.
const KEPT_REQUEST_HEADERS: &[&str] = &["content-type", "accept", "user-agent", "x-request-id"];

/// How the mirror is set up.
#[derive(Clone, Debug)]
pub struct MirrorConfig {
    /// The upstream's base URL, `http://127.0.0.1:8081`.
    pub upstream: String,
    /// Where the day files go (`$SABER_SHARED_CACHE/shadow`).
    pub log_dir: PathBuf,
    /// Poll the upstream's `/metrics` this often and log the serving families (`None`: never).
    /// Off the request path entirely.
    pub metrics_every: Option<Duration>,
    /// THE CONTROL, and never a serving setting: drop the body chunk with this index (0-based)
    /// from what the CLIENT receives while still recording it as received. It exists so the
    /// reader's transparency check (stop condition 6) is proven to fire on a mirror that changes
    /// a stream (`docs/SHADOW.md` §4 item 2).
    pub fault_drop_chunk: Option<usize>,
}

/// Counters `/shadow/health` reports.
#[derive(Debug, Default)]
pub struct MirrorCounters {
    /// Requests forwarded, any route.
    pub requests: AtomicU64,
    /// Exchanges recorded.
    pub recorded: AtomicU64,
    /// Records the writer failed to append.
    pub log_failures: AtomicU64,
    /// Requests the upstream could not be reached for (answered 502).
    pub upstream_errors: AtomicU64,
    /// Requests in flight right now.
    pub in_flight: AtomicU64,
    /// Streams whose client left before the end.
    pub client_gone: AtomicU64,
}

/// The mirror's state.
#[derive(Debug)]
pub struct Mirror {
    cfg: MirrorConfig,
    http: reqwest::Client,
    started: Instant,
    next_id: AtomicU64,
    /// Counters for `/shadow/health`.
    pub counters: Arc<MirrorCounters>,
    writer: std::sync::mpsc::Sender<LogEntry>,
}

impl Mirror {
    /// Starts the writer thread and returns the mirror. The log directory is created.
    ///
    /// # Errors
    /// The directory cannot be created or the HTTP client cannot be built.
    pub fn new(cfg: MirrorConfig) -> anyhow::Result<Arc<Self>> {
        std::fs::create_dir_all(&cfg.log_dir)?;
        let counters = Arc::new(MirrorCounters::default());
        let (tx, rx) = std::sync::mpsc::channel::<LogEntry>();
        let dir = cfg.log_dir.clone();
        let c = Arc::clone(&counters);
        std::thread::Builder::new()
            .name("shadow-writer".to_owned())
            .spawn(move || {
                for entry in rx {
                    let t = match &entry {
                        LogEntry::Exchange(e) => e.t_unix_ms,
                        LogEntry::Metrics(m) => m.t_unix_ms,
                        LogEntry::Start { t_unix_ms, .. } => *t_unix_ms,
                    };
                    let path = dir.join(format!("{}.jsonl", utc_date(t)));
                    if append_line(&path, &entry).is_err() {
                        c.log_failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })?;
        let http = reqwest::Client::builder()
            // No pooling surprises: a proxy that reuses a connection the upstream closed would
            // turn one request into a retry the client never sent.
            .pool_idle_timeout(Duration::from_secs(30))
            .build()?;
        Ok(Arc::new(Self {
            cfg,
            http,
            started: Instant::now(),
            next_id: AtomicU64::new(1),
            counters,
            writer: tx,
        }))
    }

    /// Writes the `start` line.
    pub fn log_start(&self, listen: &SocketAddr) {
        let _ = self.writer.send(LogEntry::Start {
            t_unix_ms: unix_ms(),
            listen: listen.to_string(),
            upstream: self.cfg.upstream.clone(),
        });
    }

    fn ms_since(&self, t: Instant) -> f64 {
        t.duration_since(self.started).as_secs_f64() * 1e3
    }

    /// The `/shadow/health` body.
    pub fn health(&self) -> serde_json::Value {
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        serde_json::json!({
            "upstream": self.cfg.upstream,
            "log_dir": self.cfg.log_dir.display().to_string(),
            "requests": g(&self.counters.requests),
            "recorded": g(&self.counters.recorded),
            "log_failures": g(&self.counters.log_failures),
            "upstream_errors": g(&self.counters.upstream_errors),
            "in_flight": g(&self.counters.in_flight),
            "client_gone": g(&self.counters.client_gone),
            "fault_drop_chunk": self.cfg.fault_drop_chunk,
        })
    }

    /// Polls the upstream's `/metrics` every `metrics_every`, logging the serving families.
    pub fn spawn_metrics_poller(self: &Arc<Self>) {
        let Some(every) = self.cfg.metrics_every else {
            return;
        };
        let me = Arc::clone(self);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(every);
            loop {
                tick.tick().await;
                let url = format!("{}/metrics", me.cfg.upstream);
                let Ok(r) = me.http.get(url).send().await else {
                    continue;
                };
                let Ok(text) = r.text().await else { continue };
                let kept: String = text
                    .lines()
                    .filter(|l| crate::record::served_family(l).is_some())
                    .map(|l| format!("{l}\n"))
                    .collect();
                let _ = me.writer.send(LogEntry::Metrics(MetricsSnapshot {
                    t_unix_ms: unix_ms(),
                    t_mono_ms: me.ms_since(Instant::now()),
                    text: kept,
                }));
            }
        });
    }
}

/// The mirror's router: `/shadow/health`, and everything else forwarded.
pub fn router(mirror: Arc<Mirror>) -> Router {
    Router::new()
        .route(
            "/shadow/health",
            axum::routing::get(
                |State(m): State<Arc<Mirror>>| async move { axum::Json(m.health()) },
            ),
        )
        .fallback(proxy)
        .with_state(mirror)
}

/// Serves the mirror on `listener` until `shutdown` resolves.
///
/// # Errors
/// The server failed.
pub async fn serve(
    mirror: Arc<Mirror>,
    listener: tokio::net::TcpListener,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    if let Ok(addr) = listener.local_addr() {
        mirror.log_start(&addr);
    }
    mirror.spawn_metrics_poller();
    axum::serve(listener, router(mirror))
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

fn forwardable(name: &HeaderName) -> bool {
    !HOP_BY_HOP.contains(&name.as_str())
}

struct InFlight(Arc<MirrorCounters>);

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

async fn proxy(State(m): State<Arc<Mirror>>, req: Request<Body>) -> AxumResponse {
    let arrived = Instant::now();
    let t_unix_ms = unix_ms();
    m.counters.requests.fetch_add(1, Ordering::Relaxed);
    m.counters.in_flight.fetch_add(1, Ordering::Relaxed);
    let guard = InFlight(Arc::clone(&m.counters));

    let (parts, body) = req.into_parts();
    let path = parts
        .uri
        .path_and_query()
        .map_or_else(|| parts.uri.path().to_owned(), |p| p.as_str().to_owned());
    let record = parts.method == axum::http::Method::POST && RECORDED.contains(&parts.uri.path());
    let body = match axum::body::to_bytes(body, 256 << 20).await {
        Ok(b) => b,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("shadow mirror: {e}")).into_response();
        }
    };

    let mut upstream_req = m
        .http
        .request(parts.method.clone(), format!("{}{}", m.cfg.upstream, path))
        .body(body.clone());
    for (name, value) in &parts.headers {
        if forwardable(name) && name != "content-length" {
            upstream_req = upstream_req.header(name, value);
        }
    }

    let mut exchange = record.then(|| Exchange {
        id: m.next_id.fetch_add(1, Ordering::Relaxed),
        t_unix_ms,
        t_mono_ms: m.ms_since(arrived),
        method: parts.method.to_string(),
        path: path.clone(),
        headers: kept_headers(&parts.headers),
        body: Blob::of(&body),
        stream: serde_json::from_slice::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v.get("stream").and_then(serde_json::Value::as_bool))
            .unwrap_or(false),
        response: Response::default(),
        upstream_bytes: 0,
        forwarded_bytes: 0,
        upstream_fnv: Fnv::default().hex(),
        forwarded_fnv: Fnv::default().hex(),
        client_gone: false,
    });

    let upstream = match upstream_req.send().await {
        Ok(r) => r,
        Err(e) => {
            m.counters.upstream_errors.fetch_add(1, Ordering::Relaxed);
            if let Some(mut x) = exchange.take() {
                x.response.error = Some(e.to_string());
                x.response.t_done_ms = Some(elapsed_ms(arrived));
                finish(&m, x);
            }
            return (
                StatusCode::BAD_GATEWAY,
                format!("shadow mirror: upstream: {e}"),
            )
                .into_response();
        }
    };

    let status = upstream.status();
    let mut headers = HeaderMap::new();
    for (name, value) in upstream.headers() {
        if forwardable(name) {
            headers.append(name.clone(), value.clone());
        }
    }
    if let Some(x) = exchange.as_mut() {
        x.response.status = Some(status.as_u16());
        x.response.headers = headers
            .iter()
            .map(|(n, v)| (n.to_string(), v.to_str().unwrap_or("").to_owned()))
            .collect();
    }

    // The body: every chunk forwarded the moment it arrives, copied on the side. The channel is
    // small on purpose — a slow client slows the upstream read, as it would without the mirror.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(8);
    let drop_chunk = m.cfg.fault_drop_chunk;
    let m2 = Arc::clone(&m);
    tokio::spawn(async move {
        let _guard = guard;
        let mut stream = upstream.bytes_stream();
        let (mut up_fnv, mut fw_fnv) = (Fnv::default(), Fnv::default());
        let (mut up_bytes, mut fw_bytes) = (0u64, 0u64);
        let mut index = 0usize;
        let mut chunks = Vec::new();
        let mut first = None;
        let mut complete = true;
        let mut error = None;
        let mut gone = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(bytes) => {
                    let t = elapsed_ms(arrived);
                    first.get_or_insert(t);
                    up_fnv.update(&bytes);
                    up_bytes += bytes.len() as u64;
                    if exchange.is_some() {
                        chunks.push(Chunk {
                            t_ms: t,
                            data: Blob::of(&bytes),
                        });
                    }
                    let dropped = drop_chunk == Some(index);
                    index += 1;
                    if dropped {
                        continue;
                    }
                    fw_fnv.update(&bytes);
                    fw_bytes += bytes.len() as u64;
                    if tx.send(Ok(bytes)).await.is_err() {
                        // The client left. Stop reading: dropping the upstream stream is the
                        // disconnect the upstream would have seen without the mirror.
                        gone = true;
                        complete = false;
                        m2.counters.client_gone.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                }
                Err(e) => {
                    complete = false;
                    error = Some(e.to_string());
                    let _ = tx
                        .send(Err(std::io::Error::other(format!("upstream: {e}"))))
                        .await;
                    break;
                }
            }
        }
        drop(tx);
        if let Some(mut x) = exchange {
            x.response.chunks = chunks;
            x.response.t_first_byte_ms = first;
            x.response.t_done_ms = Some(elapsed_ms(arrived));
            x.response.complete = complete;
            x.response.error = error;
            x.upstream_bytes = up_bytes;
            x.forwarded_bytes = fw_bytes;
            x.upstream_fnv = up_fnv.hex();
            x.forwarded_fnv = fw_fnv.hex();
            x.client_gone = gone;
            finish(&m2, x);
        }
    });

    let mut resp = AxumResponse::new(Body::from_stream(
        tokio_stream::wrappers::ReceiverStream::new(rx),
    ));
    *resp.status_mut() = status;
    *resp.headers_mut() = headers;
    if drop_chunk.is_some() {
        // The control changes the length; a length the body does not have would make the HTTP
        // stack refuse the response instead of delivering the altered stream the reader must
        // catch.
        resp.headers_mut()
            .remove(axum::http::header::CONTENT_LENGTH);
    }
    resp
}

fn finish(m: &Mirror, x: Exchange) {
    m.counters.recorded.fetch_add(1, Ordering::Relaxed);
    let _ = m.writer.send(LogEntry::Exchange(Box::new(x)));
}

fn elapsed_ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn kept_headers(h: &HeaderMap) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = h
        .iter()
        .filter(|(n, _)| KEPT_REQUEST_HEADERS.contains(&n.as_str()))
        .map(|(n, v)| (n.to_string(), v.to_str().unwrap_or("").to_owned()))
        .collect();
    if h.contains_key(axum::http::header::AUTHORIZATION) {
        out.push((
            "authorization".to_owned(),
            "<present, not recorded>".to_owned(),
        ));
    }
    out
}

/// A header value, for a caller building requests from a record.
pub fn header_value(s: &str) -> Option<HeaderValue> {
    HeaderValue::from_str(s).ok()
}
