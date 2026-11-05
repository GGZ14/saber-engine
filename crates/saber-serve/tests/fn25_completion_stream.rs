//! FN25: `/v1/completions` with `stream: true` -- the reference server's `completion_stream_generator`.
//!
//! FN17's conc-8 client (`tools/gates/fn17_api.py`: prompt ids, `stream: true`,
//! `stream_options.include_usage`, `return_token_ids`) got the plane's old 400 ("streaming is not
//! implemented for /v1/completions") on every boot. These tests pin the streamed shape against
//! prod's generator (read in `the reference image`, `completion/serving.py`) and the
//! stream against the blocking response of the same script (text, ids, usage).

use std::sync::Arc;

use saber_serve::detok::ByteCodec;
use saber_serve::engine::{
    FinishReason, MockEngine, MockScript, MockStep, ModelInfo, SampledLogprobs, TokenLogprob,
};
use saber_serve::state::{MinimalChatRenderer, ServerState};
use serde_json::{json, Value};

fn model_info() -> ModelInfo {
    ModelInfo {
        served_model_name: "Qwen3.8-Flash-Next".to_owned(),
        aliases: Vec::new(),
        max_model_len: 4096,
        root: "/models/Qwen3.8-Flash-Next-Quark-MXFP4".to_owned(),
        eos_token_ids: vec![255],
    }
}

fn bytes_of(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

async fn spawn(scripts: Vec<MockScript>) -> (String, Arc<MockEngine>) {
    let engine = Arc::new(MockEngine::new(model_info()));
    for s in scripts {
        engine.push_script(s);
    }
    let state = ServerState::builder(
        engine.clone(),
        Arc::new(ByteCodec),
        Arc::new(MinimalChatRenderer),
    )
    .system_fingerprint("saber-test")
    .build();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        saber_serve::server::serve(state, listener)
            .await
            .expect("serve")
    });
    (format!("http://{addr}"), engine)
}

/// The `data:` events of an SSE body, `[DONE]` as `None`.
fn events(body: &str) -> Vec<Option<Value>> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .map(str::trim)
        .map(|p| (p != "[DONE]").then(|| serde_json::from_str(p).expect("chunk json")))
        .collect()
}

async fn post(base: &str, body: Value) -> (u16, String) {
    let r = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .json(&body)
        .send()
        .await
        .expect("request");
    (r.status().as_u16(), r.text().await.expect("body"))
}

/// `fn17_api.py`'s exact request: every chunk one engine step, the prompt's ids on the first, the
/// step's ids on each, the usage chunk last (`choices: []`, the fingerprint), then `[DONE]`; the
/// text, ids and usage equal the blocking response's for the same script.
#[tokio::test]
async fn fn17s_streamed_request_is_served_as_the_reference_streams_it() {
    let answer = bytes_of(" the answer is 42.");
    let mut with_eos = answer.clone();
    with_eos.push(255);
    let script = || MockScript::in_steps_of(with_eos.clone(), 4, FinishReason::Stop);
    let (base, engine) = spawn(vec![script(), script()]).await;
    let prompt: Vec<u32> = vec![10, 20, 30, 40, 50, 60];
    let (status, body) = post(
        &base,
        json!({"model": "Qwen3.8-Flash-Next", "prompt": prompt, "max_tokens": 128,
               "temperature": 0.0, "stream": true, "return_token_ids": true,
               "stream_options": {"include_usage": true}}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let ev = events(&body);
    assert_eq!(ev.last(), Some(&None), "[DONE] ends the stream");
    let chunks: Vec<&Value> = ev.iter().flatten().collect();
    let (usage, content) = chunks.split_last().expect("chunks");
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(usage["object"], json!("text_completion"));
    assert_eq!(usage["system_fingerprint"], json!("saber-test"));
    assert_eq!(
        usage["usage"],
        json!({"prompt_tokens": 6, "total_tokens": 6 + with_eos.len(),
               "completion_tokens": with_eos.len()})
    );
    // one chunk per engine step (4 wide)
    assert_eq!(content.len(), with_eos.len().div_ceil(4));
    let mut text = String::new();
    let mut ids = Vec::new();
    for (k, c) in content.iter().enumerate() {
        assert_eq!(c["object"], json!("text_completion"));
        assert!(c.get("usage").is_none(), "content chunks carry no usage");
        // a usage chunk follows, so no content chunk carries the fingerprint
        assert!(c.get("system_fingerprint").is_none());
        let ch = &c["choices"][0];
        assert_eq!(ch["index"], json!(0));
        let want_prompt = if k == 0 { json!(prompt) } else { Value::Null };
        assert_eq!(ch["prompt_token_ids"], want_prompt, "chunk {k}");
        assert!(ch.get("logprobs").is_some_and(Value::is_null));
        text.push_str(ch["text"].as_str().expect("text"));
        ids.extend(
            ch["token_ids"]
                .as_array()
                .expect("token_ids")
                .iter()
                .map(|x| x.as_u64().expect("id") as u32),
        );
        let last = k + 1 == content.len();
        assert_eq!(
            ch["finish_reason"],
            if last { json!("stop") } else { Value::Null }
        );
    }
    assert_eq!(
        ids, with_eos,
        "the step ids, the eos included (prod's D5 law)"
    );
    assert_eq!(engine.submitted()[0].prompt_token_ids, prompt);

    // The blocking response of the same script: the same text, ids and usage.
    let (status, body) = post(
        &base,
        json!({"model": "Qwen3.8-Flash-Next", "prompt": prompt, "max_tokens": 128,
               "temperature": 0.0, "return_token_ids": true}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let b: Value = serde_json::from_str(&body).expect("json");
    assert_eq!(b["choices"][0]["text"], json!(text));
    assert_eq!(b["choices"][0]["token_ids"], json!(ids));
    assert_eq!(
        b["usage"]["completion_tokens"],
        usage["usage"]["completion_tokens"]
    );
}

/// Without `include_usage` the terminal chunk carries the fingerprint and no usage chunk follows;
/// without `return_token_ids` the id fields are `null`; `echo` puts the prompt text and its ids on
/// the first chunk.
#[tokio::test]
async fn the_streamed_variants_follow_the_reference_generator() {
    let out = bytes_of("abcdef");
    let (base, _engine) = spawn(vec![
        MockScript::in_steps_of(out.clone(), 3, FinishReason::Length),
        MockScript::in_steps_of(out.clone(), 3, FinishReason::Length),
    ])
    .await;
    let (status, body) = post(
        &base,
        json!({"prompt": "Hi", "max_tokens": 6, "stream": true}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let ev = events(&body);
    let chunks: Vec<&Value> = ev.iter().flatten().collect();
    assert_eq!(chunks.len(), 2, "two steps, no usage chunk");
    assert!(chunks[0].get("system_fingerprint").is_none());
    assert_eq!(chunks[1]["system_fingerprint"], json!("saber-test"));
    for c in &chunks {
        assert!(c["choices"][0]["token_ids"].is_null());
        assert!(c["choices"][0]["prompt_token_ids"].is_null());
    }
    assert_eq!(chunks[1]["choices"][0]["finish_reason"], json!("length"));

    let (status, body) = post(
        &base,
        json!({"prompt": "Hi", "max_tokens": 6, "stream": true, "echo": true}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let ev = events(&body);
    let chunks: Vec<&Value> = ev.iter().flatten().collect();
    let first = &chunks[0]["choices"][0];
    assert_eq!(first["text"], json!("Hiabc"));
    assert_eq!(first["prompt_token_ids"], json!(bytes_of("Hi")));
    assert_eq!(chunks[1]["choices"][0]["text"], json!("def"));
    assert!(chunks[1]["choices"][0]["prompt_token_ids"].is_null());
}

/// `logprobs`: one entry per token, the `text_offset`s continuing across chunks.
#[tokio::test]
async fn streamed_logprob_offsets_continue_across_chunks() {
    let lp = |id: u32| SampledLogprobs {
        sampled: TokenLogprob {
            token_id: id,
            logprob: -0.25,
            rank: Some(1),
        },
        top: vec![],
    };
    let out = bytes_of("abcdef");
    let steps: Vec<MockStep> = out
        .chunks(3)
        .enumerate()
        .map(|(k, c)| {
            let s =
                MockStep::new(c.iter().copied()).with_logprobs(c.iter().map(|&i| lp(i)).collect());
            if k == 1 {
                s.finish(FinishReason::Length)
            } else {
                s
            }
        })
        .collect();
    let (base, _engine) = spawn(vec![MockScript::new(steps)]).await;
    let (status, body) = post(
        &base,
        json!({"prompt": "Hi", "max_tokens": 6, "stream": true, "logprobs": 0}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let ev = events(&body);
    let chunks: Vec<&Value> = ev.iter().flatten().collect();
    let lps = |c: &Value| c["choices"][0]["logprobs"].clone();
    assert_eq!(lps(chunks[0])["text_offset"], json!([0, 1, 2]));
    assert_eq!(lps(chunks[1])["text_offset"], json!([3, 4, 5]));
    assert_eq!(lps(chunks[1])["tokens"], json!(["d", "e", "f"]));
    assert_eq!(
        lps(chunks[0])["token_logprobs"],
        json!([-0.25, -0.25, -0.25])
    );
}

/// `prompt_logprobs` on a stream is prod's `check_logprobs` 400, streamed or not the same body.
#[tokio::test]
async fn prompt_logprobs_on_a_stream_is_still_refused() {
    let (base, _engine) = spawn(vec![]).await;
    let (status, body) = post(
        &base,
        json!({"prompt": "Hi", "stream": true, "prompt_logprobs": 1}),
    )
    .await;
    assert_eq!(status, 400, "{body}");
    assert!(body.contains("`prompt_logprobs` are not available when `stream=True`."));
}
