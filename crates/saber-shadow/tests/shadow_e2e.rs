//! The mirror, the replay and the reader against the serving plane's own mock endpoint
//! (`saber-serve`'s `MockEngine` behind the real router), on real sockets (ws-v8).
//!
//! * `the_mirror_changes_nothing_the_client_receives` — a deterministic upstream: the bytes
//!   through the mirror are the bytes straight from the upstream, the headers that matter are
//!   kept, and a stream is forwarded as it arrives (a chunk held back 400 ms upstream is not
//!   waited for by the one before it);
//! * `the_mirror_records_everything_the_mock_sent` — the mock: every generation exchange is on
//!   the log with its request byte for byte and the client's bytes equal to the recorded ones;
//!   `/metrics` and `/v1/models` are forwarded and not recorded;
//! * `replay_reproduces_the_mocks_greedy_answers` — the log replayed against a second mock with
//!   the same scripts: every greedy answer identical, `usage` identical, no stop condition fired;
//! * the controls: a mirror that drops one chunk fires stop 6; a Saber that fails a request prod
//!   served fires stop 2; a Saber that answers one greedy request differently is reported as a
//!   greedy difference and fires nothing (identity is reported, not a stop).

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use saber_serve::detok::ByteCodec;
use saber_serve::engine::{FinishReason, MockEngine, MockScript, MockStep, ModelInfo, TokenId};
use saber_serve::state::{MinimalChatRenderer, ReasoningParserKind, ServerState, ToolParserKind};
use saber_shadow::mirror::{self, Mirror, MirrorConfig};
use saber_shadow::record::{read_log, Exchange, LogEntry};
use saber_shadow::replay::{replay, Pace, ReplayConfig};
use saber_shadow::report::{compare, ReaderOptions, Verdict};
use saber_test_scratch::ScratchDir;
use serde_json::{json, Value};

fn model_info() -> ModelInfo {
    ModelInfo {
        served_model_name: "Qwen3.8".to_owned(),
        aliases: Vec::new(),
        max_model_len: 4096,
        root: "/models/mock".to_owned(),
        eos_token_ids: vec![151_645],
    }
}

fn ids(s: &str) -> Vec<TokenId> {
    s.bytes().map(u32::from).collect()
}

/// The scripts both mocks run, in submission order.
fn scripts() -> Vec<MockScript> {
    vec![
        MockScript::in_steps_of(ids("Hello from the stream."), 4, FinishReason::Stop),
        MockScript::in_steps_of(ids("A plain answer."), 5, FinishReason::Stop),
        MockScript::in_steps_of(ids("completion text"), 3, FinishReason::Length),
        MockScript::in_steps_of(ids("sampled, not greedy"), 4, FinishReason::Stop),
    ]
}

/// The requests the client sends, in order. The third-from-last is malformed on purpose: the
/// upstream refuses it (a 4xx), which is recorded like any other exchange.
fn requests() -> Vec<(&'static str, Value)> {
    vec![
        (
            "/v1/chat/completions",
            json!({"model": "Qwen3.8", "messages": [{"role": "user", "content": "hi"}],
                   "temperature": 0, "stream": true, "stream_options": {"include_usage": true},
                   "max_tokens": 64}),
        ),
        (
            "/v1/chat/completions",
            json!({"model": "Qwen3.8", "messages": [{"role": "user", "content": "plain"}],
                   "temperature": 0, "max_tokens": 64}),
        ),
        (
            "/v1/completions",
            json!({"model": "Qwen3.8", "prompt": "complete me", "temperature": 0,
                   "max_tokens": 15}),
        ),
        (
            "/v1/chat/completions",
            json!({"model": "Qwen3.8", "messages": "not a list", "temperature": 0}),
        ),
        (
            "/v1/chat/completions",
            json!({"model": "Qwen3.8", "messages": [{"role": "user", "content": "draw"}],
                   "temperature": 0.8, "seed": 1, "max_tokens": 64}),
        ),
    ]
}

async fn spawn_mock(scripts: Vec<MockScript>) -> (String, Arc<MockEngine>) {
    let engine = Arc::new(MockEngine::new(model_info()).with_step_delay(Duration::from_millis(2)));
    for s in scripts {
        engine.push_script(s);
    }
    let state = ServerState::builder(
        engine.clone(),
        Arc::new(ByteCodec),
        Arc::new(MinimalChatRenderer),
    )
    .reasoning_parser(ReasoningParserKind::Qwen3)
    .tool_parser(ToolParserKind::Qwen3Coder)
    .enable_auto_tool_choice(true)
    .system_fingerprint("saber-test")
    .build();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        saber_serve::server::serve(state, listener).await.unwrap();
    });
    (format!("http://{addr}"), engine)
}

async fn spawn_mirror(
    upstream: &str,
    dir: &Path,
    drop_chunk: Option<usize>,
) -> (String, Arc<Mirror>) {
    let m = Mirror::new(MirrorConfig {
        upstream: upstream.to_owned(),
        log_dir: dir.to_owned(),
        metrics_every: None,
        fault_drop_chunk: drop_chunk,
    })
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let m2 = Arc::clone(&m);
    tokio::spawn(async move {
        mirror::serve(m2, listener, std::future::pending())
            .await
            .unwrap();
    });
    (format!("http://{addr}"), m)
}

/// A fresh log directory. Named by the process, a per-process SEQUENCE number and the clock: the
/// four `record_day` tests run beside each other, and a shared directory would hand one of them
/// the other's exchanges (ws-h42: the nanosecond clock alone is not a uniqueness proof).
fn scratch(name: &str) -> ScratchDir {
    // Under `target/`, removed on drop, never `/tmp` (FN51h); the helper adds the pid and a
    // per-process sequence number, the clock stays in the name as before.
    let d = ScratchDir::new(&format!("saber-shadow-{name}-{}", record_nonce()));
    assert!(
        std::fs::read_dir(&*d).unwrap().next().is_none(),
        "{}: a scratch directory is never shared",
        d.display()
    );
    d
}

fn record_nonce() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}

/// The log's exchanges, once `n` have been written (the writer is a thread; bounded wait).
///
/// The writer appends a line with one `write_all` while this reads the same file, and a regular
/// file gives a concurrent reader no atomicity: under load the read can land between a line's
/// first bytes and its newline and see a TORN last line, which `read_log` (rightly) refuses. A
/// snapshot that does not parse is therefore "not yet" and is read again until the deadline; one
/// that still does not parse at the deadline fails with the reader's own error (ws-h42). The law
/// is unchanged: every exchange must be logged and every line must parse.
async fn exchanges(dir: &Path, n: usize) -> Vec<LogEntry> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let mut entries = Vec::new();
        let mut torn = None;
        for f in std::fs::read_dir(dir).unwrap().filter_map(Result::ok) {
            match read_log(&f.path()) {
                Ok(e) => entries.extend(e),
                Err(e) => torn = Some(format!("{e:#}")),
            }
        }
        let got = entries
            .iter()
            .filter(|e| matches!(e, LogEntry::Exchange(_)))
            .count();
        if torn.is_none() && got >= n {
            return entries;
        }
        assert!(
            Instant::now() < deadline,
            "only {got} of {n} exchanges logged (last read error: {torn:?})"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn xs(entries: &[LogEntry]) -> Vec<Exchange> {
    let mut v: Vec<Exchange> = entries
        .iter()
        .filter_map(|e| match e {
            LogEntry::Exchange(x) => Some((**x).clone()),
            _ => None,
        })
        .collect();
    v.sort_by_key(|x| x.id);
    v
}

/// Sends every request through `base`, one at a time; returns (status, body bytes).
async fn drive(base: &str) -> Vec<(u16, Vec<u8>, Vec<u8>)> {
    let http = reqwest::Client::new();
    let mut out = Vec::new();
    for (path, body) in requests() {
        let raw = serde_json::to_vec(&body).unwrap();
        let r = http
            .post(format!("{base}{path}"))
            .header("content-type", "application/json")
            .body(raw.clone())
            .send()
            .await
            .unwrap();
        let status = r.status().as_u16();
        out.push((status, raw, r.bytes().await.unwrap().to_vec()));
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_mirror_changes_nothing_the_client_receives() {
    use axum::body::Body;
    use axum::response::Response;
    // A deterministic upstream: fixed bytes, a custom header, and a stream whose second chunk is
    // held back 400 ms.
    let app = axum::Router::new()
        .route(
            "/v1/completions",
            axum::routing::post(|body: axum::body::Bytes| async move {
                let mut r =
                    Response::new(Body::from([b"{\"echo\":".as_slice(), &body, b"}"].concat()));
                r.headers_mut()
                    .insert("content-type", "application/json".parse().unwrap());
                r.headers_mut()
                    .insert("x-upstream", "prod".parse().unwrap());
                r
            }),
        )
        .route(
            "/v1/chat/completions",
            axum::routing::post(|| async {
                let s = futures_util::stream::unfold(0u8, |i| async move {
                    match i {
                        0 => Some((
                            Ok::<_, std::io::Error>(bytes::Bytes::from_static(
                                b"data: {\"a\":1}\n\n",
                            )),
                            1,
                        )),
                        1 => {
                            tokio::time::sleep(Duration::from_millis(400)).await;
                            Some((
                                Ok(bytes::Bytes::from_static(
                                    b"data: {\"b\":\"\xc3\xa9\"}\n\ndata: [DONE]\n\n",
                                )),
                                2,
                            ))
                        }
                        _ => None,
                    }
                });
                let mut r = Response::new(Body::from_stream(s));
                r.headers_mut()
                    .insert("content-type", "text/event-stream".parse().unwrap());
                r
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let dir = scratch("transparent");
    let (base, m) = spawn_mirror(&upstream, &dir, None).await;
    let http = reqwest::Client::new();

    for (path, body) in [
        ("/v1/completions", b"{\"prompt\":\"x\",\"n\":1}".to_vec()),
        ("/v1/chat/completions", b"{\"stream\":true}".to_vec()),
    ] {
        let direct = http
            .post(format!("{upstream}{path}"))
            .body(body.clone())
            .send()
            .await
            .unwrap();
        let (ds, dh) = (direct.status(), direct.headers().clone());
        let direct = direct.bytes().await.unwrap();

        let t0 = Instant::now();
        let mut via = http
            .post(format!("{base}{path}"))
            .body(body.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(via.status(), ds);
        for h in ["content-type", "x-upstream"] {
            assert_eq!(via.headers().get(h), dh.get(h), "header {h}");
        }
        let mut got = Vec::new();
        let mut first_at = None;
        while let Some(c) = via.chunk().await.unwrap() {
            first_at.get_or_insert(t0.elapsed());
            got.extend_from_slice(&c);
        }
        assert_eq!(got, direct.to_vec(), "{path}: the bytes through the mirror");
        if path == "/v1/chat/completions" {
            // NOT BUFFERED: the first chunk arrives long before the upstream's second.
            let first = first_at.unwrap();
            assert!(
                first < Duration::from_millis(300),
                "the first chunk waited for the stream ({first:?})"
            );
            assert!(t0.elapsed() >= Duration::from_millis(400));
        }
    }
    let log = xs(&exchanges(&dir, 2).await);
    assert_eq!(log.len(), 2);
    for x in &log {
        assert_eq!(x.upstream_fnv, x.forwarded_fnv);
        assert!(x.response.complete);
    }
    // The split `é` of the stream came back byte for byte.
    assert!(String::from_utf8(log[1].response.body())
        .unwrap()
        .contains('é'));
    assert_eq!(m.health()["log_failures"], 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_mirror_records_everything_the_mock_sent() {
    let (prod, _engine) = spawn_mock(scripts()).await;
    let dir = scratch("records");
    let (base, m) = spawn_mirror(&prod, &dir, None).await;
    let sent = drive(&base).await;
    // Forwarded, not recorded.
    let http = reqwest::Client::new();
    let metrics = http.get(format!("{base}/metrics")).send().await.unwrap();
    assert_eq!(metrics.status(), 200);
    assert!(metrics
        .text()
        .await
        .unwrap()
        .contains("saber:num_requests_running"));
    let models = http.get(format!("{base}/v1/models")).send().await.unwrap();
    assert_eq!(models.status(), 200);

    let log = xs(&exchanges(&dir, sent.len()).await);
    assert_eq!(
        log.len(),
        sent.len(),
        "every generation exchange, nothing else"
    );
    for (x, (status, raw, body)) in log.iter().zip(&sent) {
        assert_eq!(
            x.body.bytes(),
            *raw,
            "#{}: the request body byte for byte",
            x.id
        );
        assert_eq!(x.response.status, Some(*status));
        assert_eq!(
            x.response.body(),
            *body,
            "#{}: the client got the recorded bytes",
            x.id
        );
        assert_eq!(x.upstream_fnv, x.forwarded_fnv);
        assert_eq!(x.upstream_bytes, body.len() as u64);
        assert!(x.response.complete && !x.client_gone);
        assert!(x
            .headers
            .iter()
            .any(|(k, v)| k == "content-type" && v == "application/json"));
    }
    assert!(log[0].stream && !log[1].stream);
    assert!(
        log[3]
            .response
            .status
            .is_some_and(|s| (400..500).contains(&s)),
        "the refusal is recorded too"
    );
    assert!(
        log[0].response.chunks.len() > 2,
        "a stream is recorded chunk by chunk ({} chunks)",
        log[0].response.chunks.len()
    );
    let h = m.health();
    assert_eq!(h["recorded"], sent.len() as u64);
    assert_eq!(h["requests"], sent.len() as u64 + 2);
    assert_eq!(h["log_failures"], 0);
    assert_eq!(h["in_flight"], 0);
}

async fn record_day(drop_chunk: Option<usize>) -> (Vec<LogEntry>, ScratchDir) {
    let (prod, _engine) = spawn_mock(scripts()).await;
    let dir = scratch("day");
    let (base, _m) = spawn_mirror(&prod, &dir, drop_chunk).await;
    let n = drive(&base).await.len();
    (exchanges(&dir, n).await, dir)
}

async fn replay_against(
    saber_scripts: Vec<MockScript>,
    log: &[LogEntry],
) -> saber_shadow::report::Report {
    let (saber, _engine) = spawn_mock(saber_scripts).await;
    let (meta, answers) = replay(
        &xs(log),
        &ReplayConfig {
            target: saber,
            pace: Pace::Sequential,
            power_cap_root: None,
            timeout: Duration::from_secs(30),
        },
    )
    .await
    .unwrap();
    assert_eq!(meta.replayed as usize, xs(log).len());
    assert!(meta.metrics_before.is_some() && meta.metrics_after.is_some());
    compare(
        log,
        &meta,
        &answers,
        &ReaderOptions {
            saber_log: Some("listening      http://127.0.0.1:8090/v1\nstopped        9 engine steps, card released\n".to_owned()),
            cap_w: 210.0,
            allowed_refusals: Vec::new(),
        },
    )
}

fn stop(r: &saber_shadow::report::Report, n: usize) -> &Verdict {
    &r.stops[n - 1].1
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn replay_reproduces_the_mocks_greedy_answers() {
    let (log, _dir) = record_day(None).await;
    let r = replay_against(scripts(), &log).await;
    println!("{}", r.render());
    assert_eq!(r.recorded, 5);
    assert_eq!(r.replayed, 5);
    assert_eq!(r.status_identical, 5);
    // Three greedy exchanges were served (the fourth greedy one is the 400); all identical.
    assert_eq!(r.greedy_compared, 3, "{}", r.render());
    assert_eq!(r.greedy_identical, 3, "{}", r.render());
    assert_eq!(r.both_served, 4);
    assert_eq!(r.content_identical, 4);
    assert_eq!(r.usage_compared, 4);
    assert_eq!(r.prompt_tokens_identical, 4);
    assert_eq!(r.completion_tokens_identical, 4);
    assert!(!r.fired(), "{}", r.render());
    assert_eq!(*stop(&r, 2), Verdict::Clear);
    assert_eq!(*stop(&r, 3), Verdict::Clear);
    assert_eq!(*stop(&r, 4), Verdict::Clear);
    assert_eq!(*stop(&r, 6), Verdict::Clear);
    // The mock has no census, and nobody read a power cap: said, not assumed.
    assert!(matches!(stop(&r, 1), Verdict::NotChecked(_)));
    assert!(matches!(stop(&r, 5), Verdict::NotChecked(_)));
    assert!(r.render().contains("no stop condition fired"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn control_a_mirror_that_drops_one_chunk_fires_stop_6() {
    let (log, _dir) = record_day(Some(1)).await;
    let r = replay_against(scripts(), &log).await;
    println!("{}", r.render());
    assert!(matches!(stop(&r, 6), Verdict::Fired(_)), "{}", r.render());
    assert!(r.fired());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn control_a_saber_that_fails_a_served_request_fires_stop_2() {
    let (log, _dir) = record_day(None).await;
    let mut s = scripts();
    s[1] = MockScript::new(vec![
        MockStep::new(ids("A pl")).failed("injected engine failure")
    ]);
    let r = replay_against(s, &log).await;
    println!("{}", r.render());
    match stop(&r, 2) {
        Verdict::Fired(d) => assert!(d.contains("#2"), "{d}"),
        other => panic!("stop 2 did not fire: {other:?}\n{}", r.render()),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn control_a_greedy_difference_is_reported_not_a_stop() {
    let (log, _dir) = record_day(None).await;
    let mut s = scripts();
    s[2] = MockScript::in_steps_of(ids("completion TEXT"), 3, FinishReason::Length);
    let r = replay_against(s, &log).await;
    println!("{}", r.render());
    assert_eq!(r.greedy_identical, 2);
    assert_eq!(r.greedy_differing, vec![3]);
    assert!(
        !r.fired(),
        "identity is reported, not a stop:\n{}",
        r.render()
    );
}
