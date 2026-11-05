//! The shape the serving plane actually deploys: `Arc<dyn Engine>` plus a step loop on a plain
//! `std::thread`.
//!
//! `ServerState` holds an `Arc<dyn Engine>`, so the scheduler has to be object-safe through that
//! trait; and `ARCHITECTURE.md` § Rust-specific hazards says the step loop is *not* async — "the
//! step loop wants fixed cadence, not throughput; a work-stealing scheduler adds variance to
//! exactly that path". These tests run it that way: tokio owns the request side, one ordinary
//! thread owns the steps, and the two meet at the event channel.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use common::{drain, request, scheduler, tokens_of};
use saber_sched::{Acceptance, RandomAcceptanceExecutor, Scheduler, StepOutcome};
use saber_serve::engine::{Engine, FinishReason};

/// Runs the step loop on a plain thread until the flag is set.
fn spawn_step_loop(
    sched: Arc<Scheduler<RandomAcceptanceExecutor>>,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<u64> {
    thread::spawn(move || {
        let mut steps = 0;
        while !stop.load(Ordering::Relaxed) {
            match sched.step() {
                StepOutcome::Idle => thread::sleep(Duration::from_millis(1)),
                StepOutcome::Ran(_) => steps += 1,
                StepOutcome::NoBucket(e) => panic!("no captured bucket: {e}"),
            }
        }
        steps
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_scheduler_serves_through_the_engine_trait_object() {
    let sched = Arc::new(scheduler(RandomAcceptanceExecutor::new(
        77,
        Acceptance::Fixed(2),
    )));
    // Exactly what `ServerState` holds.
    let engine: Arc<dyn Engine> = Arc::clone(&sched) as Arc<dyn Engine>;
    assert_eq!(engine.model().served_model_name, "Qwen3.8");
    assert_eq!(engine.stats().num_requests_running, 0);

    let stop = Arc::new(AtomicBool::new(false));
    let loop_handle = spawn_step_loop(Arc::clone(&sched), Arc::clone(&stop));

    let sub = engine.submit(request(&[1, 2, 3], 24)).expect("submit");
    let events = drain(sub).await;

    stop.store(true, Ordering::Relaxed);
    let steps = loop_handle.join().expect("step loop");

    assert_eq!(tokens_of(&events).len(), 24);
    assert_eq!(events.last().unwrap().finished, Some(FinishReason::Length));
    assert!(
        steps >= events.len() as u64,
        "{} events from {steps} steps: an event must never outnumber the steps that produced it",
        events.len()
    );
    assert_eq!(engine.stats().num_requests_running, 0);
    let stats = engine.stats();
    assert!(stats.spec_decode_num_drafts > 0);
    assert!(stats.spec_decode_num_accepted_tokens > 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn requests_join_the_batch_at_step_boundaries_while_the_loop_runs() {
    let sched = Arc::new(scheduler(RandomAcceptanceExecutor::new(
        5150,
        Acceptance::Bernoulli { p: 0.6 },
    )));
    let engine: Arc<dyn Engine> = Arc::clone(&sched) as Arc<dyn Engine>;
    let stop = Arc::new(AtomicBool::new(false));
    let loop_handle = spawn_step_loop(Arc::clone(&sched), Arc::clone(&stop));

    let mut tasks = Vec::new();
    for i in 0..6u32 {
        // Arrivals spread across steps: continuous batching, not a static batch.
        tokio::time::sleep(Duration::from_millis(2)).await;
        let sub = engine
            .submit(request(&[i, i + 1, i + 2], 10 + i as usize))
            .expect("submit");
        tasks.push(tokio::spawn(
            async move { (10 + i as usize, drain(sub).await) },
        ));
    }

    for t in tasks {
        let (want, events) = t.await.expect("task");
        assert_eq!(tokens_of(&events).len(), want);
        assert_eq!(
            events.iter().filter(|e| e.finished.is_some()).count(),
            1,
            "exactly one finished, under concurrency too"
        );
        for (i, e) in events.iter().enumerate() {
            assert_eq!(e.step_index, i as u32);
        }
    }

    stop.store(true, Ordering::Relaxed);
    loop_handle.join().expect("step loop");
    assert_eq!(engine.stats().num_requests_running, 0);
    assert_eq!(engine.stats().num_requests_waiting, 0);
    assert!(!sched.has_work());

    // Every allocation came back.
    let fresh = scheduler(RandomAcceptanceExecutor::new(0, Acceptance::Fixed(1)));
    assert_eq!(sched.arena_free(), fresh.arena_free());
    sched.with_executor(|e| {
        let (admits, releases) = e.admit_release_counts();
        assert_eq!(admits, 6);
        assert_eq!(releases, 6);
        assert_eq!(e.live_sequences(), 0);
    });
}
