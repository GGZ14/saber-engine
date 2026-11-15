//! The console's `/v1/*` reverse proxy.
//!
//! Every request is forwarded to the engine on the loopback, one HTTP/1 connection a request;
//! the response's frames are passed to the client AS THEY ARRIVE (no buffering: a streamed
//! reply's SSE events reach the client when the engine writes them). On the way through, the
//! proxy watches the frames of `/v1/chat/completions` and `/v1/completions` to record one usage
//! row per request: TTFT at the first SSE event that carries a token, the `usage` block's token
//! counts, the finish reason, the status (499 when the client went away first). The record is
//! queued to the usage store's writer thread, never written on the request path.
//!
//! The engine request carries `x-saber-console-ref: <the row's uid>`, which the engine keeps in
//! its `/v1/requests` ring, so the fallback poller merges the engine's view into the same row
//! instead of counting it twice.
//!
//! Auth: with `proxy.keys` configured a client presents one of them (its label rides the row;
//! each key is a salted hash checked in constant time, a revoked key is refused);
//! otherwise the admin token, when the console has one; otherwise the proxy is open (a loopback
//! bind). A wrong key counts toward the address's failed-auth lockout ([`crate::auth::Limiter`]). The engine sees the console's own engine key ([`crate::config::Config::engine_bearer`])
//! when one is configured, else (open mode only) the client's `Authorization` unchanged.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper_util::rt::TokioIo;
use serde_json::{json, Value};

use crate::config::ProxyConfig;
use crate::http::AppState;
use crate::usage::{UsageRow, UsageStore, SRC_PROXY};

/// The header the engine records ([`saber_serve`]'s `CLIENT_REF_HEADER`).
pub const REF_HEADER: &str = "x-saber-console-ref";
/// The largest request body the proxy accepts (images ride as base64).
pub const MAX_REQUEST_BYTES: usize = 256 * 1024 * 1024;
/// The most bytes of a non-streamed reply kept to read its `usage` block.
const MAX_COPY: usize = 8 * 1024 * 1024;
/// A single SSE line longer than this is not parsed (it is still passed through).
const MAX_LINE: usize = 1024 * 1024;

/// The proxy's shared state.
pub struct Proxy {
    seq: AtomicU64,
    prefix: String,
    pub usage: Option<Arc<UsageStore>>,
}

impl Proxy {
    #[must_use]
    pub fn new(usage: Option<Arc<UsageStore>>) -> Proxy {
        Proxy {
            seq: AtomicU64::new(0),
            // unique across console restarts: the start time, in hex
            prefix: format!("c{:x}", crate::now_ms()),
            usage,
        }
    }

    fn next_uid(&self) -> String {
        format!(
            "{}-{}",
            self.prefix,
            self.seq.fetch_add(1, Ordering::Relaxed) + 1
        )
    }
}

/// Who the client is, as far as the proxy's auth says.
#[derive(Debug, PartialEq, Eq)]
pub enum Caller {
    /// A configured proxy key (its label) or the admin token (`admin`).
    Label(String),
    /// Open mode (no proxy keys, no admin token).
    Open,
}

/// Authorize a presented Bearer key against the proxy config and the admin credential. Every
/// key is checked (no early exit), each in constant time; a revoked key never matches.
#[must_use]
pub fn authorize(
    cfg: &ProxyConfig,
    admin: &crate::auth::Admin,
    presented: Option<&str>,
) -> Option<Caller> {
    let p = presented.unwrap_or("");
    let mut hit = None;
    for k in &cfg.keys {
        if k.matches(p) && hit.is_none() {
            hit = Some(Caller::Label(k.label.clone()));
        }
    }
    if hit.is_some() {
        return hit;
    }
    if admin.is_set() {
        return admin.check(p).then(|| Caller::Label("admin".into()));
    }
    cfg.keys.is_empty().then_some(Caller::Open)
}

fn bearer_of(h: &HeaderMap) -> Option<&str> {
    h.get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim)
}

/// An OpenAI-style error envelope.
fn oai_error(status: StatusCode, msg: &str, kind: &str) -> Response {
    (
        status,
        axum::Json(json!({"error": {"message": msg, "type": kind, "code": status.as_u16()}})),
    )
        .into_response()
}

const HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];

/// What the proxy reads from a request body (nothing of it is kept unless `store_content`).
#[derive(Debug, Default, PartialEq)]
pub struct ReqInfo {
    pub model: Option<String>,
    pub stream: bool,
    pub thinking: Option<bool>,
    pub images: i64,
}

#[must_use]
pub fn parse_request(body: &[u8]) -> ReqInfo {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return ReqInfo::default();
    };
    let thinking = v["chat_template_kwargs"]["enable_thinking"]
        .as_bool()
        .or_else(|| v["enable_thinking"].as_bool());
    let mut images = 0;
    if let Some(msgs) = v["messages"].as_array() {
        for m in msgs {
            if let Some(parts) = m["content"].as_array() {
                images += parts
                    .iter()
                    .filter(|p| {
                        matches!(
                            p["type"].as_str(),
                            Some("image_url" | "input_image" | "image")
                        )
                    })
                    .count() as i64;
            }
        }
    }
    ReqInfo {
        model: v["model"].as_str().map(str::to_string),
        stream: v["stream"].as_bool().unwrap_or(false),
        thinking,
        images,
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…[truncated]", &s[..end])
}

/// Watches a reply's frames and finalizes its usage row (on the last frame, an upstream error,
/// or the drop when the client went away first).
pub struct Tap {
    store: Option<Arc<UsageStore>>,
    row: UsageRow,
    arrived: Instant,
    sse: bool,
    line: Vec<u8>,
    copy: Vec<u8>,
    store_content: bool,
    max_content: usize,
    content: String,
    reasoning: String,
    first_token: Option<Instant>,
    last_token: Option<Instant>,
    done: bool,
    /// The stream's `data: [DONE]` went through: a client that hangs up after it got the whole
    /// reply (some benchmark clients do) is not a 499.
    saw_done: bool,
}

impl Tap {
    fn new(
        store: Option<Arc<UsageStore>>,
        row: UsageRow,
        arrived: Instant,
        sse: bool,
        store_content: bool,
        max_content: usize,
    ) -> Tap {
        Tap {
            store,
            row,
            arrived,
            sse,
            line: Vec::new(),
            copy: Vec::new(),
            store_content,
            max_content,
            content: String::new(),
            reasoning: String::new(),
            first_token: None,
            last_token: None,
            done: false,
            saw_done: false,
        }
    }

    /// One frame on its way to the client.
    pub fn feed(&mut self, chunk: &[u8]) {
        if !self.sse {
            if self.copy.len() + chunk.len() <= MAX_COPY {
                self.copy.extend_from_slice(chunk);
            }
            return;
        }
        let mut rest = chunk;
        while let Some(i) = rest.iter().position(|&b| b == b'\n') {
            let (head, tail) = rest.split_at(i);
            if self.line.len() + head.len() <= MAX_LINE {
                self.line.extend_from_slice(head);
                let line = std::mem::take(&mut self.line);
                self.on_line(&line);
            } else {
                self.line.clear();
            }
            rest = &tail[1..];
        }
        if self.line.len() + rest.len() <= MAX_LINE {
            self.line.extend_from_slice(rest);
        } else {
            self.line.clear();
        }
    }

    fn on_line(&mut self, line: &[u8]) {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let Some(data) = line.strip_prefix(b"data:") else {
            return;
        };
        let data = data.trim_ascii();
        if data == b"[DONE]" {
            self.saw_done = true;
            return;
        }
        if data.is_empty() {
            return;
        }
        let Ok(v) = serde_json::from_slice::<Value>(data) else {
            return;
        };
        self.on_event(&v);
    }

    fn on_event(&mut self, v: &Value) {
        let mut token = false;
        if let Some(choices) = v["choices"].as_array() {
            for c in choices {
                let d = &c["delta"];
                for (k, is_reasoning) in [
                    ("content", false),
                    ("reasoning", true),
                    ("reasoning_content", true),
                ] {
                    if let Some(s) = d[k].as_str().filter(|s| !s.is_empty()) {
                        token = true;
                        if self.store_content {
                            let buf = if is_reasoning {
                                &mut self.reasoning
                            } else {
                                &mut self.content
                            };
                            if buf.len() < self.max_content {
                                buf.push_str(s);
                            }
                        }
                    }
                }
                if d["tool_calls"].as_array().is_some_and(|a| !a.is_empty()) {
                    token = true;
                }
                if let Some(s) = c["text"].as_str().filter(|s| !s.is_empty()) {
                    token = true;
                    if self.store_content && self.content.len() < self.max_content {
                        self.content.push_str(s);
                    }
                }
                if let Some(f) = c["finish_reason"].as_str() {
                    self.row.finish_reason = Some(f.to_string());
                }
            }
        }
        if token {
            let now = Instant::now();
            self.first_token.get_or_insert(now);
            self.last_token = Some(now);
        }
        self.read_common(v);
    }

    fn read_common(&mut self, v: &Value) {
        let u = &v["usage"];
        if u.is_object() {
            self.row.prompt_tokens = u["prompt_tokens"].as_i64().or(self.row.prompt_tokens);
            self.row.completion_tokens = u["completion_tokens"]
                .as_i64()
                .or(self.row.completion_tokens);
            self.row.cached_tokens = u["prompt_tokens_details"]["cached_tokens"]
                .as_i64()
                .or(self.row.cached_tokens);
        }
        if self.row.model.is_none() {
            self.row.model = v["model"].as_str().map(str::to_string);
        }
        if let Some(m) = v["error"]["message"].as_str() {
            self.row.error = Some(truncate(m, 500));
        }
    }

    /// The reply ended (`status` overrides the upstream status: 499 / 502).
    pub fn finish(&mut self, status: Option<u16>, error: Option<String>) {
        if self.done {
            return;
        }
        self.done = true;
        let end = Instant::now();
        if !self.sse && !self.copy.is_empty() {
            if let Ok(v) = serde_json::from_slice::<Value>(&self.copy) {
                if let Some(choices) = v["choices"].as_array() {
                    for c in choices {
                        if let Some(f) = c["finish_reason"].as_str() {
                            self.row.finish_reason = Some(f.to_string());
                        }
                        if self.store_content {
                            if let Some(s) = c["message"]["content"]
                                .as_str()
                                .or_else(|| c["text"].as_str())
                            {
                                self.content.push_str(s);
                            }
                            if let Some(s) = c["message"]["reasoning"]
                                .as_str()
                                .or_else(|| c["message"]["reasoning_content"].as_str())
                            {
                                self.reasoning.push_str(s);
                            }
                        }
                    }
                }
                self.read_common(&v);
            }
        }
        if let Some(s) = status {
            self.row.status = Some(i64::from(s));
        }
        if let Some(e) = error {
            self.row.error = Some(e);
        }
        self.row.latency_ms = Some(end.duration_since(self.arrived).as_secs_f64() * 1e3);
        self.row.ttft_ms = self
            .first_token
            .map(|f| f.duration_since(self.arrived).as_secs_f64() * 1e3);
        if let (Some(f), Some(l), Some(n)) = (
            self.first_token,
            self.last_token,
            self.row.completion_tokens,
        ) {
            let span = l.duration_since(f).as_secs_f64();
            if span > 0.0 && n > 1 {
                self.row.decode_tps = Some((n - 1) as f64 / span);
            }
        }
        if self.store_content {
            let mut out = String::new();
            if !self.reasoning.is_empty() {
                out.push_str("[reasoning]\n");
                out.push_str(&self.reasoning);
                out.push_str("\n[/reasoning]\n");
            }
            out.push_str(&self.content);
            self.row.response_body = Some(truncate(&out, self.max_content));
        }
        if let Some(s) = &self.store {
            s.record(std::mem::take(&mut self.row));
        }
    }

    /// The row as recorded so far (tests).
    #[must_use]
    pub fn row(&self) -> &UsageRow {
        &self.row
    }
}

impl Drop for Tap {
    fn drop(&mut self) {
        if self.saw_done {
            // the whole reply reached the client; it closed before the engine's end of body
            self.finish(None, None);
        } else {
            // the client went away before the last frame
            self.finish(Some(499), None);
        }
    }
}

/// `ANY /v1/*`.
pub async fn handle(State(st): State<AppState>, req: Request) -> Response {
    let arrived = Instant::now();
    let ts_ms = crate::now_ms() as i64;
    let cfg = st.sup.config_now();
    let presented = bearer_of(req.headers()).map(str::to_string);
    let ip = crate::http::client_ip(&req);
    if let Some(left) = st.auth.limiter.blocked(ip) {
        return crate::http::too_many(left, true);
    }
    let Some(caller) = authorize(&cfg.proxy, &st.auth.admin(), presented.as_deref()) else {
        if presented.as_deref().is_some_and(|p| !p.is_empty()) {
            st.auth.limiter.fail(ip);
        }
        return oai_error(
            StatusCode::UNAUTHORIZED,
            "a valid API key is required (Authorization: Bearer <key>)",
            "invalid_request_error",
        );
    };
    if presented.is_some() && caller != Caller::Open {
        st.auth.limiter.ok(ip);
    }
    let (parts, body) = req.into_parts();
    let path = parts.uri.path().to_string();
    let pq = parts
        .uri
        .path_and_query()
        .map_or_else(|| path.clone(), |p| p.as_str().to_string());
    let recorded = parts.method == Method::POST
        && matches!(path.as_str(), "/v1/chat/completions" | "/v1/completions");
    let bytes = match axum::body::to_bytes(body, MAX_REQUEST_BYTES).await {
        Ok(b) => b,
        Err(_) => {
            return oai_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request body too large",
                "invalid_request_error",
            )
        }
    };
    let store = if recorded && cfg.usage.enabled {
        st.proxy.usage.clone()
    } else {
        None
    };
    let uid = st.proxy.next_uid();
    let info = if recorded {
        parse_request(&bytes)
    } else {
        ReqInfo::default()
    };
    let label = match &caller {
        Caller::Label(l) => Some(l.clone()),
        Caller::Open => None,
    };
    let mut row = UsageRow {
        uid: uid.clone(),
        ts_ms,
        source: SRC_PROXY,
        endpoint: path.clone(),
        model: info.model.clone(),
        key: label,
        stream: Some(info.stream),
        images: Some(info.images),
        thinking: info.thinking,
        ..UsageRow::default()
    };
    if cfg.usage.store_content {
        row.request_body = Some(truncate(
            &String::from_utf8_lossy(&bytes),
            cfg.usage.content_max_bytes,
        ));
    }
    let fail = |row: UsageRow, status: StatusCode, msg: &str| {
        if let Some(s) = &store {
            let mut r = row;
            r.status = Some(i64::from(status.as_u16()));
            r.error = Some(msg.to_string());
            r.latency_ms = Some(arrived.elapsed().as_secs_f64() * 1e3);
            s.record(r);
        }
        oai_error(status, msg, "server_error")
    };
    let Some(port) = st.sup.live_port() else {
        return fail(
            row,
            StatusCode::SERVICE_UNAVAILABLE,
            "the engine is not running",
        );
    };

    // the upstream request
    let mut b = hyper::Request::builder()
        .method(parts.method.clone())
        .uri(&pq);
    for (k, v) in &parts.headers {
        let n = k.as_str();
        if HOP.contains(&n)
            || matches!(
                n,
                "host" | "authorization" | "content-length" | "accept-encoding" | REF_HEADER
            )
        {
            continue;
        }
        b = b.header(k, v);
    }
    b = b.header(header::HOST, format!("127.0.0.1:{port}"));
    let outbound_key = st.sup.engine_bearer().or(match caller {
        Caller::Open => presented,
        Caller::Label(_) => None,
    });
    if let Some(k) = outbound_key {
        if let Ok(v) = HeaderValue::from_str(&format!("Bearer {k}")) {
            b = b.header(header::AUTHORIZATION, v);
        }
    }
    if recorded {
        b = b.header(REF_HEADER, uid.as_str());
    }
    let Ok(up_req) = b.body(Full::new(bytes)) else {
        return fail(row, StatusCode::BAD_REQUEST, "malformed request");
    };
    let stream = match tokio::time::timeout(
        Duration::from_secs(5),
        tokio::net::TcpStream::connect(("127.0.0.1", port)),
    )
    .await
    {
        Ok(Ok(s)) => s,
        _ => return fail(row, StatusCode::BAD_GATEWAY, "the engine is not answering"),
    };
    let _ = stream.set_nodelay(true);
    let (mut sender, conn) = match hyper::client::conn::http1::handshake(TokioIo::new(stream)).await
    {
        Ok(x) => x,
        Err(e) => {
            return fail(
                row,
                StatusCode::BAD_GATEWAY,
                &format!("engine connection: {e}"),
            )
        }
    };
    tokio::spawn(async move {
        let _ = conn.await;
    });
    let resp = match sender.send_request(up_req).await {
        Ok(r) => r,
        Err(e) => {
            return fail(
                row,
                StatusCode::BAD_GATEWAY,
                &format!("engine request: {e}"),
            )
        }
    };
    let (rparts, incoming) = resp.into_parts();
    let mut out = Response::builder().status(rparts.status);
    for (k, v) in &rparts.headers {
        let n = k.as_str();
        if HOP.contains(&n) || (recorded && n == "content-length") {
            continue;
        }
        out = out.header(k, v);
    }
    if !recorded {
        return out
            .body(Body::new(incoming))
            .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response());
    }
    out = out.header(HeaderName::from_static(REF_HEADER), uid.as_str());
    let sse = rparts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/event-stream"));
    row.status = Some(i64::from(rparts.status.as_u16()));
    let tap = Tap::new(
        store,
        row,
        arrived,
        sse,
        cfg.usage.store_content,
        cfg.usage.content_max_bytes,
    );
    let body = futures_util::stream::unfold(
        (incoming, tap, false),
        |(mut inc, mut tap, ended)| async move {
            if ended {
                return None;
            }
            loop {
                match inc.frame().await {
                    None => {
                        tap.finish(None, None);
                        return None;
                    }
                    Some(Err(e)) => {
                        let msg = format!("engine stream: {e}");
                        tap.finish(Some(502), Some(msg));
                        return Some((Err(e), (inc, tap, true)));
                    }
                    Some(Ok(f)) => {
                        if let Ok(data) = f.into_data() {
                            tap.feed(&data);
                            return Some((Ok::<Bytes, hyper::Error>(data), (inc, tap, false)));
                        }
                    }
                }
            }
        },
    );
    out.body(Body::from_stream(body))
        .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProxyKey;

    #[test]
    fn auth_modes() {
        use crate::auth::{hash_secret, Admin};
        let none = Admin::None;
        let admin = Admin::Hash(hash_secret("admin-tok-0123456").unwrap());
        let mut c = ProxyConfig::default();
        assert_eq!(authorize(&c, &none, None), Some(Caller::Open));
        assert_eq!(authorize(&c, &none, Some("x")), Some(Caller::Open));
        assert_eq!(authorize(&c, &admin, None), None);
        assert_eq!(authorize(&c, &admin, Some("admin-tok-012345")), None);
        assert_eq!(
            authorize(&c, &admin, Some("admin-tok-0123456")),
            Some(Caller::Label("admin".into()))
        );
        c.keys.push(ProxyKey::new("laptop", "laptop-key-1"));
        // a hashed key (what a saved config holds)
        let mut h = ProxyKey::new("ci", "");
        h.hash = hash_secret("ci-key-123456").unwrap();
        c.keys.push(h);
        assert_eq!(authorize(&c, &none, None), None);
        assert_eq!(authorize(&c, &none, Some("nope")), None);
        assert_eq!(
            authorize(&c, &none, Some("laptop-key-1")),
            Some(Caller::Label("laptop".into()))
        );
        assert_eq!(
            authorize(&c, &none, Some("ci-key-123456")),
            Some(Caller::Label("ci".into()))
        );
        assert_eq!(
            authorize(&c, &admin, Some("admin-tok-0123456")),
            Some(Caller::Label("admin".into()))
        );
        // revoked: refused, and a revoked key is not an "open" proxy either
        c.keys[1].revoked = true;
        assert_eq!(authorize(&c, &none, Some("ci-key-123456")), None);
        c.keys[0].revoked = true;
        assert_eq!(
            authorize(&c, &none, None),
            None,
            "every key revoked = still closed"
        );
    }

    #[test]
    fn a_client_that_closes_after_done_is_not_a_499() {
        let dir = saber_test_scratch::ScratchDir::new("console-tap-done");
        let store = Arc::new(UsageStore::open(&dir.path().join("u.db"), 0).unwrap());
        let row = |uid: &str| UsageRow {
            uid: uid.into(),
            ts_ms: 1,
            status: Some(200),
            ..UsageRow::default()
        };
        // the whole stream, then the drop (no end-of-body seen)
        let mut t = Tap::new(
            Some(store.clone()),
            row("a"),
            Instant::now(),
            true,
            false,
            10,
        );
        t.feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n");
        drop(t);
        // the negative control: dropped before [DONE]
        let mut t = Tap::new(
            Some(store.clone()),
            row("b"),
            Instant::now(),
            true,
            false,
            10,
        );
        t.feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n");
        drop(t);
        store.flush();
        assert_eq!(store.row("a").unwrap().unwrap().status, Some(200));
        assert_eq!(store.row("b").unwrap().unwrap().status, Some(499));
    }

    #[test]
    fn request_fields() {
        let b = br#"{"model":"Q","stream":true,"chat_template_kwargs":{"enable_thinking":false},
            "messages":[{"role":"user","content":[{"type":"text","text":"hi"},
            {"type":"image_url","image_url":{"url":"data:x"}}]}]}"#;
        assert_eq!(
            parse_request(b),
            ReqInfo {
                model: Some("Q".into()),
                stream: true,
                thinking: Some(false),
                images: 1
            }
        );
        assert_eq!(parse_request(b"not json"), ReqInfo::default());
    }

    #[test]
    fn sse_tap_reads_tokens_usage_across_split_frames() {
        let mut t = Tap::new(None, UsageRow::default(), Instant::now(), true, true, 1000);
        let events = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning\":\"hm\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":3,",
            "\"prompt_tokens_details\":{\"cached_tokens\":4}}}\n\n",
            "data: [DONE]\n\n"
        );
        // split mid-line
        let (a, b) = events.as_bytes().split_at(70);
        t.feed(a);
        assert!(t.first_token.is_none(), "the role delta is not a token");
        t.feed(b);
        assert!(t.first_token.is_some());
        t.finish(None, None);
        let r = t.row();
        assert_eq!(
            (r.prompt_tokens, r.completion_tokens, r.cached_tokens),
            (Some(9), Some(3), Some(4))
        );
        assert_eq!(r.finish_reason.as_deref(), Some("stop"));
        assert_eq!(
            r.response_body.as_deref(),
            Some("[reasoning]\nhm\n[/reasoning]\nHello")
        );
        assert!(r.ttft_ms.is_some() && r.latency_ms.is_some());
    }

    #[test]
    fn json_tap_and_client_close() {
        let mut t = Tap::new(None, UsageRow::default(), Instant::now(), false, false, 10);
        t.feed(
            br#"{"model":"Q","choices":[{"message":{"content":"x"},"finish_reason":"length"}],"#,
        );
        t.feed(br#""usage":{"prompt_tokens":5,"completion_tokens":2}}"#);
        t.finish(None, None);
        assert_eq!(t.row().completion_tokens, Some(2));
        assert_eq!(t.row().finish_reason.as_deref(), Some("length"));
        assert_eq!(t.row().response_body, None, "content is off");
        let mut t = Tap::new(None, UsageRow::default(), Instant::now(), true, false, 10);
        t.row.status = Some(200);
        t.feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n");
        // the drop without a finish = the client went away
        drop(std::mem::replace(
            &mut t,
            Tap::new(None, UsageRow::default(), Instant::now(), true, false, 10),
        ));
        t.done = true;
    }
}
