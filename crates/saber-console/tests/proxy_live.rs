//! Live: the console's `/v1` proxy and fallback poller against an in-process
//! mock engine. Proves the SSE passthrough adds no buffering (the client reads the first token
//! BEFORE the mock is allowed to send the next frame), TTFT is taken at the first token frame,
//! the engine sees the console's engine key (engine auth on) and the ref header, the poller
//! merges the engine's record into the proxied row and records a direct request once, a client
//! that hangs up is a 499, the proxy's own keys and labels, and the `/api/usage/*` contract.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use saber_console::config::{Config, ProxyKey};
use saber_console::http::{router, AppState};
use saber_console::logring::LogRing;
use saber_console::proxy::Proxy;
use saber_console::scrape::MetricsStore;
use saber_console::supervisor::Supervisor;
use saber_console::usage::{self, UsageStore, SRC_ENGINE, SRC_PROXY};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Notify;

const ENGINE_KEY: &str = "engine-key-123";
const FIRST_TOKEN_DELAY: Duration = Duration::from_millis(300);

#[derive(Default)]
struct Mock {
    seen_auth: Mutex<Vec<Option<String>>>,
    seen_ref: Mutex<Vec<Option<String>>>,
    records: Mutex<Vec<Value>>,
    go: Notify,
}

fn hdr(h: &HeaderMap, k: &str) -> Option<String> {
    h.get(k).and_then(|v| v.to_str().ok()).map(str::to_string)
}

fn authed(m: &Mock, h: &HeaderMap) -> bool {
    let a = hdr(h, "authorization");
    m.seen_auth.lock().unwrap().push(a.clone());
    a.as_deref() == Some(&format!("Bearer {ENGINE_KEY}"))
}

async fn chat(State(m): State<Arc<Mock>>, h: HeaderMap, body: Bytes) -> Response {
    if !authed(&m, &h) {
        return (StatusCode::UNAUTHORIZED, "no key").into_response();
    }
    let r = hdr(&h, "x-saber-console-ref");
    m.seen_ref.lock().unwrap().push(r.clone());
    let req: Value = serde_json::from_slice(&body).unwrap();
    {
        let mut recs = m.records.lock().unwrap();
        let id = recs.len() as u64 + 1;
        recs.push(
            json!({"id": id, "ts_ms": 1, "endpoint": "/v1/chat/completions",
            "model": "mock-q", "key": "console", "stream": req["stream"], "prompt_tokens": 9,
            "completion_tokens": 3, "cached_tokens": 4, "ttft_ms": 299.0, "latency_ms": 310.0,
            "decode_tps": 50.0, "finish_reason": "stop", "status": 200, "images": 0,
            "thinking": false, "client_ref": r}),
        );
    }
    if req["stream"].as_bool() != Some(true) {
        return Json(json!({"id": "x", "model": "mock-q",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}}))
        .into_response();
    }
    let ev = |v: Value| Bytes::from(format!("data: {v}\n\n"));
    let s = futures_util::stream::unfold(0u8, move |step| {
        let m = m.clone();
        async move {
            let out = match step {
                0 => ev(json!({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})),
                1 => {
                    tokio::time::sleep(FIRST_TOKEN_DELAY).await;
                    ev(json!({"choices": [{"index": 0, "delta": {"content": "Hel"}}]}))
                }
                2 => {
                    // the client must have read "Hel" before this frame exists
                    m.go.notified().await;
                    ev(json!({"choices": [{"index": 0, "delta": {"content": "lo"},
                        "finish_reason": "stop"}]}))
                }
                3 => ev(json!({"choices": [], "usage": {"prompt_tokens": 9,
                    "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 4}}})),
                4 => Bytes::from_static(b"data: [DONE]\n\n"),
                _ => return None,
            };
            Some((Ok::<Bytes, std::io::Error>(out), step + 1))
        }
    });
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(s))
        .unwrap()
}

#[derive(serde::Deserialize)]
struct After {
    after: Option<u64>,
}

async fn requests(State(m): State<Arc<Mock>>, h: HeaderMap, Query(q): Query<After>) -> Response {
    if !authed(&m, &h) {
        return (StatusCode::UNAUTHORIZED, "no key").into_response();
    }
    let after = q.after.unwrap_or(0);
    let recs = m.records.lock().unwrap().clone();
    let data: Vec<&Value> = recs
        .iter()
        .filter(|r| r["id"].as_u64().unwrap() > after)
        .collect();
    Json(json!({"object": "list", "data": data, "boot_id": 777,
        "newest_id": recs.len(), "oldest_id": 1, "dropped": 0, "capacity": 1024}))
    .into_response()
}

async fn serve(app: Router) -> u16 {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = l.local_addr().unwrap().port();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    port
}

/// A raw HTTP/1.1 request; the reply is read as it arrives.
async fn send(port: u16, method: &str, path: &str, auth: Option<&str>, body: &str) -> TcpStream {
    let mut s = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    let a = auth.map_or(String::new(), |k| format!("Authorization: Bearer {k}\r\n"));
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{a}Content-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    s
}

async fn read_until(s: &mut TcpStream, buf: &mut Vec<u8>, needle: &str, within: Duration) -> bool {
    let deadline = Instant::now() + within;
    let mut chunk = [0u8; 4096];
    while !String::from_utf8_lossy(buf).contains(needle) {
        let left = deadline.saturating_duration_since(Instant::now());
        match tokio::time::timeout(left, s.read(&mut chunk)).await {
            Ok(Ok(0)) | Err(_) | Ok(Err(_)) => return false,
            Ok(Ok(n)) => buf.extend_from_slice(&chunk[..n]),
        }
    }
    true
}

async fn read_all(mut s: TcpStream) -> String {
    let mut b = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(10), s.read_to_end(&mut b)).await;
    String::from_utf8_lossy(&b).into_owned()
}

async fn get_json(port: u16, path: &str) -> Value {
    let (st, body) = saber_console::httpc::get(port, path, Duration::from_secs(10))
        .await
        .unwrap();
    assert_eq!(st, 200, "{path}: {body}");
    serde_json::from_str(&body).unwrap()
}

async fn wait_rows(
    store: &Arc<UsageStore>,
    pred: impl Fn(&[usage::UsageRow]) -> bool,
) -> Vec<usage::UsageRow> {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let s = store.clone();
        let rows = tokio::task::spawn_blocking(move || {
            s.flush();
            s.rows(&usage::RowFilter {
                limit: 1000,
                ..usage::RowFilter::default()
            })
            .unwrap()
            .1
        })
        .await
        .unwrap();
        if pred(&rows) || Instant::now() > deadline {
            return rows;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn proxy_poller_and_api_against_a_mock_engine() {
    let mock = Arc::new(Mock::default());
    let mock_port = serve(
        Router::new()
            .route("/v1/chat/completions", post(chat))
            .route("/v1/requests", get(requests))
            .route(
                "/v1/models",
                get(|| async { Json(json!({"data": [{"id": "mock-q"}]})) }),
            )
            .with_state(mock.clone()),
    )
    .await;

    let dir = saber_test_scratch::ScratchDir::new("console-proxy-live");
    let mut cfg = Config::mock(mock_port);
    cfg.proxy.engine_key = ENGINE_KEY.into();
    cfg.usage.poll_interval_s = 1;
    let store = Arc::new(UsageStore::open(&dir.path().join("usage.db"), 0).unwrap());
    let (sup, _port_rx) = Supervisor::new(
        cfg.clone(),
        dir.path().join("config.toml"),
        dir.path().to_path_buf(),
        dir.path().join("saber"),
        Arc::new(LogRing::new(100)),
    );
    sup.adopt_for_test(mock_port);
    let metrics = Arc::new(MetricsStore::new(100, 5));
    let state = AppState {
        sup: sup.clone(),
        console: Arc::new(saber_console::apply::Console::new(
            sup.clone(),
            metrics.clone(),
            &json!({}),
            0,
        )),
        metrics,
        schema: Arc::new(json!({})),
        auth: Arc::new(saber_console::auth::Auth::open()),
        loopback: true,
        proxy: Arc::new(Proxy::new(Some(store.clone()))),
        usage: Some(store.clone()),
    };
    let port = serve(router(state)).await;
    tokio::spawn(usage::poll(store.clone(), sup.clone()));

    // the console's engine key reaches the engine (engine auth on)
    let (st, _) = saber_console::httpc::get_auth(
        mock_port,
        "/v1/requests",
        Duration::from_secs(5),
        sup.engine_bearer().as_deref(),
    )
    .await
    .unwrap();
    assert_eq!(st, 200);

    // 1. a streamed chat: passthrough without buffering, TTFT at the first token frame
    let t0 = Instant::now();
    let body = r#"{"model":"mock-q","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
    let mut s = send(port, "POST", "/v1/chat/completions", None, body).await;
    let mut buf = Vec::new();
    assert!(
        read_until(&mut s, &mut buf, "Hel", Duration::from_secs(5)).await,
        "the first token frame did not reach the client before the next one was sent: the proxy \
         buffers ({})",
        String::from_utf8_lossy(&buf)
    );
    let first = t0.elapsed();
    assert!(first >= FIRST_TOKEN_DELAY, "{first:?}");
    let head = String::from_utf8_lossy(&buf).to_string();
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    assert!(
        head.to_ascii_lowercase()
            .contains("content-type: text/event-stream"),
        "{head}"
    );
    let uid = head
        .lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("x-saber-console-ref: ")
                .map(|v| v.trim().to_string())
        })
        .expect("the ref header on the reply");
    mock.go.notify_one();
    let rest = read_all(s).await;
    assert!(rest.contains("[DONE]"), "{rest}");
    assert_eq!(
        mock.seen_ref.lock().unwrap()[0].as_deref(),
        Some(uid.as_str())
    );
    assert!(mock
        .seen_auth
        .lock()
        .unwrap()
        .iter()
        .all(|a| a.as_deref() == Some("Bearer engine-key-123")));

    // 2. a non-streamed chat, and a direct-to-engine request the poller must find
    let body2 = r#"{"model":"mock-q","messages":[{"role":"user","content":"hi"}]}"#;
    let out = read_all(send(port, "POST", "/v1/chat/completions", None, body2).await).await;
    assert!(out.contains("\"usage\""), "{out}");
    mock.records
        .lock()
        .unwrap()
        .push(json!({"id": 3, "ts_ms": 5, "endpoint": "/v1/completions",
        "model": "mock-q", "key": "direct", "stream": false, "prompt_tokens": 4,
        "completion_tokens": 8, "status": 200, "finish_reason": "length", "client_ref": null}));

    let rows = wait_rows(&store, |r| {
        r.len() == 3
            && r.iter()
                .filter(|x| x.source == SRC_PROXY | SRC_ENGINE)
                .count()
                == 2
    })
    .await;
    assert_eq!(rows.len(), 3, "{rows:#?}");
    let streamed = rows.iter().find(|r| r.uid == uid).unwrap();
    assert_eq!(streamed.source, SRC_PROXY | SRC_ENGINE, "merged, once");
    assert_eq!(streamed.status, Some(200));
    assert_eq!(
        (
            streamed.prompt_tokens,
            streamed.completion_tokens,
            streamed.cached_tokens
        ),
        (Some(9), Some(3), Some(4))
    );
    let ttft = streamed.ttft_ms.unwrap();
    assert!(
        ttft >= FIRST_TOKEN_DELAY.as_secs_f64() * 1e3 && ttft < first.as_secs_f64() * 1e3 + 50.0,
        "the proxy's TTFT ({ttft} ms) is the first token frame's, not the role frame's"
    );
    assert_eq!(streamed.finish_reason.as_deref(), Some("stop"));
    assert_eq!(streamed.decode_tps, Some(50.0), "the engine's decode rate");
    let direct = rows.iter().find(|r| r.uid == "e:777:3").unwrap();
    assert_eq!(direct.source, SRC_ENGINE);
    assert_eq!(direct.key.as_deref(), Some("direct"));

    // 3. a client that hangs up mid-stream is a 499
    let mut s = send(port, "POST", "/v1/chat/completions", None, body).await;
    let mut buf = Vec::new();
    assert!(read_until(&mut s, &mut buf, "Hel", Duration::from_secs(5)).await);
    drop(s);
    let rows = wait_rows(&store, |r| r.iter().any(|x| x.status == Some(499))).await;
    let closed = rows
        .iter()
        .find(|r| r.status == Some(499))
        .expect("a 499 row");
    assert!(closed.ttft_ms.is_some());
    mock.go.notify_one(); // let the mock's generator finish

    // 4. the proxy's own keys: refusal without, the label with
    let mut cfg2 = cfg.clone();
    cfg2.proxy.keys = vec![ProxyKey::new("laptop", "laptop-key-1")];
    *sup.config.lock().unwrap() = cfg2;
    let out = read_all(send(port, "POST", "/v1/chat/completions", None, body2).await).await;
    assert!(out.starts_with("HTTP/1.1 401"), "{out}");
    let out = read_all(
        send(
            port,
            "POST",
            "/v1/chat/completions",
            Some("laptop-key-1"),
            body2,
        )
        .await,
    )
    .await;
    assert!(out.starts_with("HTTP/1.1 200"), "{out}");
    let rows = wait_rows(&store, |r| {
        r.iter().any(|x| x.key.as_deref() == Some("laptop"))
    })
    .await;
    assert!(rows.iter().any(|x| x.key.as_deref() == Some("laptop")));
    // a non-recorded route passes through (with the engine key)
    let out = read_all(send(port, "GET", "/v1/models", Some("laptop-key-1"), "").await).await;
    assert!(out.contains("mock-q"), "{out}");
    *sup.config.lock().unwrap() = cfg;

    // 5. the API contract
    let s = get_json(port, "/api/usage/summary?range=24h").await;
    for k in [
        "from", "to", "period", "step_ms", "totals", "by_model", "by_key", "series",
    ] {
        assert!(s.get(k).is_some(), "summary.{k}: {s}");
    }
    let n = s["totals"]["requests"].as_i64().unwrap();
    assert!(n >= 4, "{s}");
    let s = get_json(port, "/api/usage/summary?range=all").await;
    assert_eq!(s["period"], "d");
    let r = get_json(port, "/api/usage/requests?sort=latency_ms&limit=2").await;
    assert_eq!(r["rows"].as_array().unwrap().len(), 2);
    assert!(r["total"].as_i64().unwrap() >= 5);
    let one = get_json(port, &format!("/api/usage/requests/{uid}")).await;
    assert_eq!(one["uid"], uid.as_str());
    let (st, _) =
        saber_console::httpc::get(port, "/api/usage/requests/nope", Duration::from_secs(5))
            .await
            .unwrap();
    assert_eq!(st, 404);
    let m = get_json(port, "/api/usage/monthly?months=3&days=30").await;
    assert_eq!(m["months"].as_array().unwrap().len(), 3);
    assert_eq!(m["days"].as_array().unwrap().len(), 30);
    let (st, csv) =
        saber_console::httpc::get(port, "/api/usage/export?format=csv", Duration::from_secs(5))
            .await
            .unwrap();
    assert_eq!(st, 200);
    assert!(csv.starts_with("uid,ts_ms,"), "{csv}");
    let (_, js) = saber_console::httpc::get(
        port,
        "/api/usage/export?format=json&model=mock-q",
        Duration::from_secs(5),
    )
    .await
    .unwrap();
    assert!(serde_json::from_str::<Vec<Value>>(&js).unwrap().len() >= 3);
    let (st, prom) = saber_console::httpc::get(port, "/metrics", Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(st, 200);
    assert!(
        prom.contains("saber_console_tokens_total{kind=\"completion\",model=\"mock-q\""),
        "{prom}"
    );
    let st = get_json(port, "/api/usage/status").await;
    assert_eq!(st["enabled"], true);
    assert_eq!(st["settings"]["store_content"], false);
    let s = get_json(port, "/api/series?names=hw_&seconds=60").await;
    assert!(s.get("t").is_some() && s.get("series").is_some());
    let h = get_json(port, "/api/hw").await;
    assert!(h.get("gpus").is_some());
}

/// A relay that BUFFERS: it reads the engine's whole reply before it answers (what a proxy
/// that collected the body would do).
async fn buffering_relay(State(mock_port): State<u16>, body: Bytes) -> Response {
    let mut s = TcpStream::connect(("127.0.0.1", mock_port)).await.unwrap();
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {ENGINE_KEY}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    s.write_all(&body).await.unwrap();
    let mut all = Vec::new();
    s.read_to_end(&mut all).await.unwrap();
    let text = String::from_utf8_lossy(&all).into_owned();
    let payload = text.split_once("\r\n\r\n").map_or("", |x| x.1).to_string();
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from(payload))
        .unwrap()
}

/// The negative control of the passthrough check (structural rule 8): the SAME check that the
/// console's proxy passes ("the first token frame reaches the client before the engine may send
/// the next one") FAILS on a relay that buffers the reply.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_passthrough_check_fails_on_a_buffering_relay() {
    let mock = Arc::new(Mock::default());
    let mock_port = serve(
        Router::new()
            .route("/v1/chat/completions", post(chat))
            .with_state(mock.clone()),
    )
    .await;
    let relay = serve(
        Router::new()
            .route("/v1/chat/completions", post(buffering_relay))
            .with_state(mock_port),
    )
    .await;
    let body = r#"{"model":"mock-q","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
    let mut s = send(relay, "POST", "/v1/chat/completions", None, body).await;
    let mut buf = Vec::new();
    let unbuffered = read_until(&mut s, &mut buf, "Hel", Duration::from_secs(2)).await;
    assert!(
        !unbuffered,
        "the check must fail on a buffering relay (it read: {})",
        String::from_utf8_lossy(&buf)
    );
    // release the mock: the relay now answers everything at once
    mock.go.notify_one();
    let rest = read_all(s).await;
    assert!(rest.contains("Hel") && rest.contains("[DONE]"), "{rest}");
}
