//! Over HTTP against the mock engine: the latency histograms on `/metrics`,
//! `GET /v1/stats`, the `GET /v1/requests` ring (ids, `after`, no bodies), and the optional
//! Bearer API keys (off by default, open routes, bad keys, the label in the ring).

use std::sync::Arc;
use std::time::Duration;

use saber_serve::auth::ApiKeys;
use saber_serve::detok::ByteCodec;
use saber_serve::engine::{FinishReason, MockEngine, MockScript, ModelInfo};
use saber_serve::state::{MinimalChatRenderer, ServerState};
use serde_json::{json, Value};

const SECRET_PROMPT: &str = "the-secret-prompt-text";

fn script(text: &str) -> MockScript {
    MockScript::in_steps_of(text.bytes().map(u32::from), 2, FinishReason::Stop)
}

async fn spawn(engine: Arc<MockEngine>, keys: Option<ApiKeys>, ring: usize) -> String {
    let state = ServerState::builder(engine, Arc::new(ByteCodec), Arc::new(MinimalChatRenderer))
        .api_keys(keys)
        .request_ring(ring)
        .build();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move { saber_serve::server::serve(state, listener).await });
    format!("http://{addr}")
}

fn engine(step_ms: u64) -> Arc<MockEngine> {
    Arc::new(
        MockEngine::new(ModelInfo {
            served_model_name: "Qwen3.8".into(),
            ..ModelInfo::default()
        })
        .with_step_delay(Duration::from_millis(step_ms)),
    )
}

fn chat(stream: bool) -> Value {
    json!({
        "model": "Qwen3.8",
        "messages": [{"role": "user", "content": SECRET_PROMPT}],
        "stream": stream,
    })
}

async fn get_json(c: &reqwest::Client, url: &str) -> Value {
    let r = c.get(url).send().await.expect("get");
    assert_eq!(r.status(), 200, "{url}");
    r.json().await.expect("json")
}

/// The family's `_count` sample from an exposition.
fn hist_count(text: &str, name: &str) -> f64 {
    text.lines()
        .find(|l| l.starts_with(&format!("{name}_count{{")))
        .and_then(|l| l.rsplit(' ').next())
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no {name}_count in\n{text}"))
}

#[tokio::test]
async fn histograms_stats_and_the_request_ring() {
    let eng = engine(5);
    eng.set_cached_tokens(Some(4));
    for _ in 0..4 {
        eng.push_script(script("hello world"));
    }
    let base = spawn(Arc::clone(&eng), None, 3).await;
    let c = reqwest::Client::new();

    // before any request: every family exists, empty
    let m0 = c
        .get(format!("{base}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    for name in [
        "saber:time_to_first_token_seconds",
        "saber:inter_token_latency_seconds",
        "saber:e2e_request_latency_seconds",
        "saber:request_queue_time_seconds",
        "saber:step_time_seconds",
    ] {
        assert!(m0.contains(&format!("# TYPE {name} histogram")), "{name}");
        assert_eq!(hist_count(&m0, name), 0.0, "{name}");
    }

    // two blocking, two streaming
    for (i, stream) in [false, true, false, true].into_iter().enumerate() {
        // The last one arrives through the console's proxy (its ref header)
        let mut rq = c.post(format!("{base}/v1/chat/completions"));
        if i == 3 {
            rq = rq.header("x-saber-console-ref", "c18f00-4");
        }
        let r = rq.json(&chat(stream)).send().await.unwrap();
        assert_eq!(r.status(), 200);
        let _ = r.text().await.unwrap();
    }
    // a streaming tracker records when its generator ends, just after [DONE] reaches us
    tokio::time::sleep(Duration::from_millis(50)).await;

    let m = c
        .get(format!("{base}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(hist_count(&m, "saber:time_to_first_token_seconds"), 4.0);
    assert_eq!(hist_count(&m, "saber:e2e_request_latency_seconds"), 4.0);
    // 11 tokens at width 2 = 6 events a request = 5 gaps
    assert_eq!(hist_count(&m, "saber:inter_token_latency_seconds"), 20.0);
    assert_eq!(
        hist_count(&m, "saber:step_time_seconds"),
        24.0,
        "the mock's 6 steps x 4"
    );
    assert!(m.contains(
        "saber:time_to_first_token_seconds_bucket{engine=\"0\",model_name=\"Qwen3.8\",le=\"+Inf\"} 4.0"
    ));

    let s = get_json(&c, &format!("{base}/v1/stats")).await;
    assert_eq!(s["object"], "saber.stats");
    assert_eq!(s["state"], "ready");
    assert_eq!(s["model"]["served_model_name"], "Qwen3.8");
    assert_eq!(s["auth"], false);
    assert_eq!(s["counters"]["request_success_total"]["stop"], 4);
    assert_eq!(s["lifetime"]["requests"], 4);
    assert_eq!(s["lifetime"]["completion_tokens"], 44);
    assert_eq!(s["lifetime"]["cached_tokens"], 16);
    let ttft = &s["latency"]["ttft"];
    assert_eq!(ttft["count"], 4);
    let (p50, p99) = (
        ttft["p50_ms"].as_f64().unwrap(),
        ttft["p99_ms"].as_f64().unwrap(),
    );
    assert!(p50 > 0.0 && p50 <= p99, "{ttft}");
    let step = &s["latency"]["step_time"];
    assert_eq!(step["count"], 24);
    assert!(step["p99_ms"].as_f64().unwrap() >= 5.0, "{step}");
    assert!(s["uptime_s"].as_f64().is_some());

    // the ring (capacity 3): ids 2..4 held, 1 dropped
    let r = get_json(&c, &format!("{base}/v1/requests")).await;
    let ids: Vec<u64> = r["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x["id"].as_u64().unwrap())
        .collect();
    assert_eq!(ids, vec![2, 3, 4]);
    assert_eq!(
        (r["oldest_id"].as_u64(), r["newest_id"].as_u64()),
        (Some(2), Some(4))
    );
    assert_eq!(r["dropped"], 1);
    assert_eq!(r["next_after"], 4);
    assert!(r["boot_id"].as_u64().unwrap() > 1_700_000_000_000);
    assert_eq!(r["data"][2]["client_ref"], "c18f00-4");
    assert!(r["data"][0]["client_ref"].is_null());
    let rec = &r["data"][0];
    assert_eq!(rec["endpoint"], "/v1/chat/completions");
    assert_eq!(rec["model"], "Qwen3.8");
    assert_eq!(rec["stream"], true);
    assert_eq!(rec["completion_tokens"], 11);
    assert_eq!(rec["cached_tokens"], 4);
    assert_eq!(rec["finish_reason"], "stop");
    assert_eq!(rec["status"], 200);
    assert_eq!(rec["images"], 0);
    assert_eq!(rec["thinking"], true);
    assert!(rec["ttft_ms"].as_f64().unwrap() > 0.0);
    assert!(rec["latency_ms"].as_f64().unwrap() >= rec["ttft_ms"].as_f64().unwrap());
    assert!(rec["decode_tps"].as_f64().unwrap() > 0.0);
    assert!(rec["key"].is_null());
    assert!(rec["ts_ms"].as_u64().unwrap() > 1_700_000_000_000);
    // never a body
    let raw = c
        .get(format!("{base}/v1/requests"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        !raw.contains(SECRET_PROMPT) && !raw.contains("hello"),
        "{raw}"
    );

    // ?after / ?limit
    let r = get_json(&c, &format!("{base}/v1/requests?after=2&limit=1")).await;
    assert_eq!(r["data"].as_array().unwrap().len(), 1);
    assert_eq!(r["data"][0]["id"], 3);
    assert_eq!(r["next_after"], 3);
    let r = get_json(&c, &format!("{base}/v1/requests?after=4")).await;
    assert!(r["data"].as_array().unwrap().is_empty());
    assert_eq!(r["next_after"], 4);

    // /v1/completions lands in the ring too, with no thinking flag
    eng.push_script(script("abc"));
    let r = c
        .post(format!("{base}/v1/completions"))
        .json(&json!({"model": "Qwen3.8", "prompt": "x"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let r = get_json(&c, &format!("{base}/v1/requests?after=4")).await;
    assert_eq!(r["data"][0]["id"], 5);
    assert_eq!(r["data"][0]["endpoint"], "/v1/completions");
    assert!(r["data"][0]["thinking"].is_null());
}

#[tokio::test]
async fn an_engine_error_is_a_500_record() {
    let eng = engine(0);
    eng.push_script(MockScript::new(vec![
        saber_serve::engine::MockStep::new([104]),
        saber_serve::engine::MockStep::new([]).failed("restore failed: test"),
    ]));
    let base = spawn(Arc::clone(&eng), None, 8).await;
    let c = reqwest::Client::new();
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat(false))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 500);
    let r = get_json(&c, &format!("{base}/v1/requests")).await;
    assert_eq!(r["data"][0]["status"], 500);
    assert_eq!(r["data"][0]["finish_reason"], "abort");
    let s = get_json(&c, &format!("{base}/v1/stats")).await;
    assert_eq!(s["lifetime"]["errors"], 1);
    assert_eq!(
        s["latency"]["e2e"]["count"], 0,
        "e2e counts completed requests only"
    );
}

#[tokio::test]
async fn auth_is_off_by_default_and_on_with_keys() {
    // off: no header needed anywhere
    let eng = engine(0);
    let base = spawn(Arc::clone(&eng), None, 8).await;
    let c = reqwest::Client::new();
    for path in [
        "/v1/models",
        "/v1/stats",
        "/v1/requests",
        "/health",
        "/metrics",
    ] {
        let r = c.get(format!("{base}{path}")).send().await.unwrap();
        assert_eq!(r.status(), 200, "{path} with auth off");
    }

    // on: /v1 needs a key, /health + /metrics stay open
    let eng = engine(0);
    eng.push_script(script("ok"));
    let keys = ApiKeys::from_specs(
        ["ci=0123456789abcdef", "console=fedcba9876543210"],
        None,
        false,
    )
    .unwrap();
    let base = spawn(Arc::clone(&eng), keys, 8).await;
    for path in ["/health", "/metrics"] {
        let r = c.get(format!("{base}{path}")).send().await.unwrap();
        assert_eq!(r.status(), 200, "{path} is open");
    }
    for path in ["/v1/models", "/v1/stats", "/v1/requests"] {
        let r = c.get(format!("{base}{path}")).send().await.unwrap();
        assert_eq!(r.status(), 401, "{path} without a key");
        assert_eq!(
            r.headers()
                .get("www-authenticate")
                .map(|v| v.to_str().unwrap()),
            Some("Bearer")
        );
        let body: Value = r.json().await.unwrap();
        assert_eq!(body["error"]["code"], 401);
        let r = c
            .get(format!("{base}{path}"))
            .bearer_auth("wrong-key-000000")
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 401, "{path} with a bad key");
        let r = c
            .get(format!("{base}{path}"))
            .bearer_auth("0123456789abcdef")
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "{path} with a good key");
    }
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat(false))
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        401,
        "a completion without a key never reaches the engine"
    );
    assert!(eng.submitted().is_empty());
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth("fedcba9876543210")
        .header("x-saber-console-ref", "c18f00-7")
        .json(&chat(false))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    // A client with ANOTHER key that spoofs a console ref keeps no ref
    eng.push_script(script("ok"));
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth("0123456789abcdef")
        .header("x-saber-console-ref", "c18f00-8")
        .json(&chat(false))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let r: Value = c
        .get(format!("{base}/v1/requests"))
        .bearer_auth("0123456789abcdef")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        r["data"][0]["key"], "console",
        "the key's label rides into the ring"
    );
    assert_eq!(
        r["data"][0]["client_ref"], "c18f00-7",
        "the console's own key keeps its ref"
    );
    assert_eq!(r["data"][1]["key"], "ci");
    assert!(
        r["data"][1]["client_ref"].is_null(),
        "a spoofed ref from a non-console key is dropped: {r}"
    );
    let s: Value = c
        .get(format!("{base}/v1/stats"))
        .bearer_auth("0123456789abcdef")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(s["auth"], true);

    // --api-auth-all: the open routes close too
    let keys = ApiKeys::from_specs(["ci=0123456789abcdef"], None, true).unwrap();
    let base = spawn(engine(0), keys, 8).await;
    for path in ["/health", "/metrics"] {
        let r = c.get(format!("{base}{path}")).send().await.unwrap();
        assert_eq!(r.status(), 401, "{path} closed under --api-auth-all");
        let r = c
            .get(format!("{base}{path}"))
            .bearer_auth("0123456789abcdef")
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
    }
}

#[tokio::test]
async fn a_client_that_hangs_up_is_a_499_record() {
    let eng = engine(30);
    eng.push_script(script("a fairly long reply that takes a while"));
    let base = spawn(Arc::clone(&eng), None, 8).await;
    let c = reqwest::Client::new();
    let mut r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat(true))
        .send()
        .await
        .unwrap();
    // read the opening chunk and one step, then drop the response
    let _ = r.chunk().await.unwrap();
    let _ = r.chunk().await.unwrap();
    drop(r);
    let mut rec = Value::Null;
    for _ in 0..100 {
        tokio::time::sleep(Duration::from_millis(20)).await;
        let p = get_json(&c, &format!("{base}/v1/requests")).await;
        if let Some(x) = p["data"].as_array().and_then(|a| a.first()) {
            rec = x.clone();
            break;
        }
    }
    assert_eq!(rec["status"], 499, "{rec}");
    assert_eq!(rec["finish_reason"], "abort");
    let s = get_json(&c, &format!("{base}/v1/stats")).await;
    assert_eq!(s["lifetime"]["client_closed"], 1);
}
