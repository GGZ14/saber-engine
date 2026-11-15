//! The scheduler records the step-time and queue-wait histograms from each
//! step's `StepTiming` for an executor that keeps none, and `/metrics` + `/v1/stats` carry them.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{boot_ledger, decode_buckets, plan_with, GIB};
use saber_sched::{
    build_state, mock_model_info, spawn_step_loop, ArenaBytes, KvGeometry, Scheduler,
    SchedulerConfig, ScriptExecutor,
};
use saber_serve::detok::{ByteCodec, TextCodec};
use saber_serve::state::MinimalChatRenderer;
use serde_json::{json, Value};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_scheduler_feeds_step_and_queue_histograms() {
    let eos = 255;
    let codec = Arc::new(ByteCodec);
    let response: Vec<u32> = codec
        .encode("hello there, world")
        .expect("encode")
        .into_iter()
        .chain([eos])
        .collect();
    let plan = plan_with(&decode_buckets(), 2);
    let config = SchedulerConfig::from_plan(
        &plan,
        KvGeometry::PROD_FP8,
        ArenaBytes {
            kv_per_rank: 8 * GIB,
            gdn_state_per_rank: 8 * GIB,
        },
        262_144,
    )
    .expect("config");
    let model = mock_model_info("Qwen3.8", "/models/mock", eos, config.max_context_tokens);
    let sched = Arc::new(
        Scheduler::new(
            config,
            ScriptExecutor::new(vec![response], 3, eos).with_step_delay(Duration::from_millis(4)),
            model,
            boot_ledger(),
        )
        .expect("arenas fit"),
    );
    let steps = spawn_step_loop(Arc::clone(&sched));
    let state = build_state(Arc::clone(&sched), codec, Arc::new(MinimalChatRenderer));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let base = format!("http://{}", listener.local_addr().expect("addr"));
    tokio::spawn(async move {
        let _ = saber_serve::server::serve(state, listener).await;
    });

    let c = reqwest::Client::new();
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "Qwen3.8", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("post");
    assert_eq!(r.status(), 200);
    let _ = r.text().await;

    let s: Value = c
        .get(format!("{base}/v1/stats"))
        .send()
        .await
        .expect("stats")
        .json()
        .await
        .expect("json");
    let step = s["latency"]["step_time"]["count"]
        .as_u64()
        .expect("step count");
    assert!(step >= 3, "decode steps recorded: {s}");
    assert!(
        s["latency"]["step_time"]["p50_ms"].as_f64().expect("p50") >= 2.0,
        "the paced step's wall: {s}"
    );
    assert_eq!(s["latency"]["queue_wait"]["count"], 1, "one admission: {s}");
    let m = c
        .get(format!("{base}/metrics"))
        .send()
        .await
        .expect("metrics")
        .text()
        .await
        .expect("text");
    assert!(m.contains("# TYPE saber:step_time_seconds histogram"));
    assert!(m.contains(&format!(
        "saber:step_time_seconds_count{{engine=\"0\",model_name=\"Qwen3.8\"}} {step}.0"
    )));
    drop(steps);
}
