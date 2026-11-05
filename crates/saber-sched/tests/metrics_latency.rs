//! How long `/metrics` waits while the engine is mid-step.
//!
//! The step loop takes the engine's lock for a whole step. If it keeps that lock across the
//! **executor** call — the device work, ~22 ms in production — then every question the serving
//! plane asks the scheduler queues behind it: `/metrics` (`stats()`), `submit`, `abort`. And while
//! one of those waits, it is parking a tokio worker, so the stall is not confined to the scraper.
//!
//! That is not a hypothetical: the disconnect gate in `tests/e2e_http.rs` measured a **443 ms**
//! first read of the counters (`docs/LEDGER.md`, 2026-09-19), and under a full-workspace run the
//! same read overran a 15 s poll.
//!
//! This test is the number, measured the way the serving plane meets it: a real `/metrics` scrape
//! over HTTP, while a paced executor holds a step open. The bound it asserts is deliberately far
//! below one step — the point is that a scrape must not be *quantised to* the step time.

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{boot_ledger, decode_buckets, plan_with, GIB};
use saber_sched::{
    build_state, mock_model_info, spawn_step_loop, ArenaBytes, KvGeometry, Scheduler,
    SchedulerConfig, ScriptExecutor,
};
use saber_serve::detok::{ByteCodec, TextCodec};
use saber_serve::engine::{Engine, EngineRequest};
use saber_serve::sampling::SamplingParams;
use saber_serve::state::MinimalChatRenderer;

/// One engine step, as slow as a real one. Production is ~22 ms/step.
const STEP: Duration = Duration::from_millis(20);

/// Scrapes the measurement takes. The **state-bounded** assertion is that more of these complete
/// than steps run while they do: a step loop holding the engine lock across the executor call can
/// answer at most one scrape per step, so `SCRAPES > steps_during` is a property of the lock scope
/// and not of the clock. That matters because the wall-clock version of this test failed once under
/// a full 63-binary parallel run and passed alone (2026-09-19) — under that load a single scrape can
/// miss a 5 ms budget for reasons that have nothing to do with the lock.
const SCRAPES: usize = 30;

/// A loose wall-clock bound kept as a *second* statement, on the **median** rather than the maximum:
/// half a step, which a lock-holding implementation cannot meet (its median is ~`STEP`) and which
/// survives a loaded box, where one outlier scrape says nothing. The p50/p99/max are printed either
/// way, because the number is the thing worth reading.
const MEDIAN_BOUND: Duration = Duration::from_millis(10);

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_metrics_scrape_does_not_wait_for_the_executor() {
    let eos = 255;
    let codec = Arc::new(ByteCodec);
    let response: Vec<u32> = codec
        .encode(&"word ".repeat(4000))
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
            ScriptExecutor::new(vec![response], 5, eos).with_step_delay(STEP),
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

    // One long request, so the engine is stepping for the whole measurement.
    let submission = sched
        .submit(EngineRequest {
            prompt_token_ids: vec![1, 2, 3],
            params: SamplingParams {
                max_tokens: Some(4000),
                ..SamplingParams::default()
            },
        })
        .expect("submit");
    // Drain the events so the per-request channel does not grow without bound.
    tokio::spawn(async move {
        use futures_util::StreamExt;
        let mut events = submission.events;
        while events.next().await.is_some() {}
    });

    // Let it get going, then scrape while it is mid-step.
    tokio::time::sleep(STEP * 3).await;
    let client = reqwest::Client::new();
    let mut samples = Vec::new();
    let steps_before = sched.counters().steps;
    for _ in 0..SCRAPES {
        let t0 = Instant::now();
        let body = client
            .get(format!("{base}/metrics"))
            .send()
            .await
            .expect("scrape")
            .text()
            .await
            .expect("body");
        samples.push(t0.elapsed());
        assert!(body.contains("saber:num_requests_running"));
        tokio::time::sleep(Duration::from_millis(3)).await;
    }
    let steps_during = sched.counters().steps - steps_before;
    samples.sort();
    let p50 = samples[samples.len() / 2];
    let p99 = samples[samples.len() * 99 / 100];
    let max = *samples.last().expect("samples");
    eprintln!(
        "/metrics scrape during a {STEP:?} step: p50 {p50:?}, p99 {p99:?}, max {max:?} over {} \
         samples, while {steps_during} step(s) ran",
        samples.len()
    );

    // The engine really was stepping throughout — otherwise the measurement is of an idle server.
    let ran = sched.counters().steps;
    assert!(ran > 3, "only {ran} steps ran during the scrapes");
    drop(steps);

    // THE ASSERTION, and it carries no clock: a step loop that holds the engine lock across the
    // executor answers at most one scrape per step, so it cannot fit 30 scrapes into a handful of
    // steps however fast or slow the box is. Under load both sides slow down, and the ratio is what
    // survives.
    assert!(
        samples.len() > steps_during as usize,
        "{} scrapes completed while {steps_during} step(s) ran: a scrape is queueing behind the \
         executor, which is what holding the engine lock across the device call looks like",
        samples.len()
    );
    // And the median, as an order-of-magnitude statement rather than a budget on the worst sample.
    assert!(
        p50 < MEDIAN_BOUND,
        "the median /metrics scrape took {p50:?} during a {STEP:?} step (p99 {p99:?}, max {max:?}, \
         {steps_during} steps): half the scrapes are waiting for the executor"
    );
}
