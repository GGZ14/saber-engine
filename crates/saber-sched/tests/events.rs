//! `StepEvent` invariants, the stop rules, abort, and the counters `/metrics` reads.
//!
//! The serving plane turns one [`StepEvent`] into one SSE chunk, and BetterBench's `update p50` is
//! only ms/step because of it (`ARCHITECTURE.md` § 5. Serving plane). So "one event per step per
//! running request, exactly one of them carrying `finished`" is a contract, not a convenience.

mod common;

use common::{drain, request, scheduler, tokens_of, EOS};
use saber_sched::{
    Acceptance, MockExecutor, RandomAcceptanceExecutor, ScriptedStep, SeqScript, StepOutcome,
};
use saber_serve::engine::{Engine, FinishReason};

#[tokio::test]
async fn one_event_per_step_exactly_one_finished_step_index_monotone() {
    let mut exec = MockExecutor::new();
    exec.push_script(SeqScript::per_token([10, 11, 12, EOS]));
    let sched = scheduler(exec);

    let sub = sched.submit(request(&[1, 2, 3], 64)).expect("submit");
    let steps = sched.run_steps(100);
    assert_eq!(steps, 4, "four scripted steps, four engine steps");

    let events = drain(sub).await;
    assert_eq!(events.len(), 4, "one event per step");
    for (i, e) in events.iter().enumerate() {
        assert_eq!(e.step_index, i as u32, "step_index is monotone from 0");
        assert_eq!(e.new_tokens.len(), 1);
    }
    assert_eq!(
        events.iter().filter(|e| e.finished.is_some()).count(),
        1,
        "exactly one event carries finished"
    );
    assert_eq!(events[3].finished, Some(FinishReason::Stop));
    assert_eq!(tokens_of(&events), vec![10, 11, 12, EOS]);
}

#[tokio::test]
async fn eos_stops_and_the_eos_token_is_committed() {
    // The EOS id counts towards `usage`; it is the serving plane that refuses to detokenize it
    // (`ModelInfo::eos_token_ids`).
    let mut exec = MockExecutor::new();
    exec.push_script(SeqScript::new(vec![
        ScriptedStep::single(10),
        // A speculative step whose second token is the EOS: everything past it is dropped.
        // Three accepted draft tokens plus the one extra the step always commits.
        ScriptedStep::speculative([11, EOS, 12, 13], 5, 3),
    ]));
    let sched = scheduler(exec);
    let sub = sched.submit(request(&[7, 7], 64)).expect("submit");
    sched.run_steps(100);

    let events = drain(sub).await;
    assert_eq!(
        tokens_of(&events),
        vec![10, 11, EOS],
        "tokens past the EOS are discarded"
    );
    assert_eq!(events.last().unwrap().finished, Some(FinishReason::Stop));
}

#[tokio::test]
async fn max_tokens_finishes_with_length() {
    let mut exec = MockExecutor::new();
    exec.push_script(SeqScript::per_token([10, 11, 12, 13, 14]));
    let sched = scheduler(exec);
    let sub = sched.submit(request(&[7], 3)).expect("submit");
    sched.run_steps(100);

    let events = drain(sub).await;
    assert_eq!(tokens_of(&events), vec![10, 11, 12]);
    assert_eq!(events.last().unwrap().finished, Some(FinishReason::Length));
    assert_eq!(sched.counters().finished_length, 1);
    assert_eq!(sched.counters().finished_stop, 0);
}

#[tokio::test]
async fn a_speculative_step_commits_at_most_width_plus_one() {
    let sched = scheduler(RandomAcceptanceExecutor::new(0xFEED, Acceptance::Fixed(3)));
    let sub = sched.submit(request(&[1, 2, 3], 40)).expect("submit");
    sched.run_steps(200);
    let events = drain(sub).await;

    assert_eq!(events[0].new_tokens.len(), 1, "prefill commits one token");
    for e in &events[1..] {
        assert!(
            e.new_tokens.len() <= 6,
            "a step committed {} tokens, more than the widest captured width + 1",
            e.new_tokens.len()
        );
        assert!(!e.new_tokens.is_empty());
    }
    assert_eq!(
        events.iter().filter(|e| e.finished.is_some()).count(),
        1,
        "exactly one finished"
    );
    assert_eq!(events.last().unwrap().finished, Some(FinishReason::Length));
    assert_eq!(
        tokens_of(&events).len(),
        40,
        "max_tokens is exact under speculation"
    );
}

#[tokio::test]
async fn abort_frees_both_allocations_and_ends_the_stream() {
    let mut exec = MockExecutor::new();
    exec.push_script(SeqScript::per_token([10, 11, 12, 13, 14, 15, 16, 17]));
    let sched = scheduler(exec);
    let (free_kv_before, free_gdn_before) = sched.arena_free();

    let sub = sched.submit(request(&[1, 2, 3, 4], 64)).expect("submit");
    sched.run_steps(2);
    let (kv_running, gdn_running) = sched.arena_free();
    assert!(
        kv_running < free_kv_before,
        "KV blocks are held while running"
    );
    assert_eq!(
        gdn_running,
        free_gdn_before - 1,
        "one state slot set is held"
    );
    assert_eq!(sched.stats().num_requests_running, 1);

    sched.abort(sub.id);
    assert_eq!(
        sched.arena_free(),
        (free_kv_before, free_gdn_before),
        "abort frees the KV blocks AND the GDN slot set"
    );
    assert_eq!(sched.stats().num_requests_running, 0);
    assert_eq!(sched.counters().aborts_running, 1);
    sched.with_executor(|e| {
        let (admits, releases) = e.admit_release_counts();
        assert_eq!((admits, releases), (1, 1), "the executor saw the release");
        assert_eq!(e.live_sequences(), 0);
    });

    let events = drain(sub).await;
    assert_eq!(
        events.iter().filter(|e| e.finished.is_some()).count(),
        1,
        "an aborted request still carries exactly one finished event"
    );
    assert_eq!(events.last().unwrap().finished, Some(FinishReason::Abort));
    assert_eq!(sched.step(), StepOutcome::Idle, "nothing is left running");
}

#[tokio::test]
async fn abort_of_a_queued_request_never_allocates() {
    let mut exec = MockExecutor::new();
    exec.push_script(SeqScript::per_token([10, 11]));
    let sched = scheduler(exec);
    let before = sched.arena_free();

    let sub = sched.submit(request(&[1, 2], 8)).expect("submit");
    assert_eq!(sched.stats().num_requests_waiting, 1);
    sched.abort(sub.id);

    assert_eq!(sched.arena_free(), before);
    assert_eq!(sched.counters().aborts_waiting, 1);
    assert_eq!(sched.stats().num_requests_waiting, 0);
    assert_eq!(sched.step(), StepOutcome::Idle);
    sched.with_executor(|e| assert_eq!(e.admit_release_counts(), (0, 0)));

    let events = drain(sub).await;
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].finished, Some(FinishReason::Abort));
}

#[tokio::test]
async fn stats_and_spec_counters_are_the_metrics_names() {
    let mut exec = MockExecutor::new();
    exec.push_script(SeqScript::new(vec![
        ScriptedStep::single(10),
        ScriptedStep::speculative([11, 12, 13], 5, 2),
        ScriptedStep::speculative([14, EOS], 5, 1),
    ]));
    let sched = scheduler(exec);
    let sub = sched.submit(request(&[1], 64)).expect("submit");

    assert_eq!(sched.stats().num_requests_waiting, 1);
    assert_eq!(sched.stats().num_requests_running, 0);
    sched.run_steps(1);
    assert_eq!(sched.stats().num_requests_running, 1);
    assert_eq!(sched.stats().num_requests_waiting, 0);
    sched.run_steps(10);

    let stats = sched.stats();
    assert_eq!(stats.num_requests_running, 0);
    assert_eq!(
        stats.spec_decode_num_drafts, 2,
        "the prefill step drafts nothing"
    );
    assert_eq!(stats.spec_decode_num_draft_tokens, 10);
    assert_eq!(
        stats.spec_decode_num_accepted_tokens, 3,
        "accepted counts DRAFT tokens; the bonus token is not one"
    );

    let events = drain(sub).await;
    assert_eq!(tokens_of(&events), vec![10, 11, 12, 13, 14, EOS]);
}

#[tokio::test]
async fn a_prompt_longer_than_the_context_is_refused_at_submit() {
    let sched = scheduler(MockExecutor::new());
    let long = vec![1u32; 70_000];
    let err = sched
        .submit(request(&long, 16))
        .expect_err("65536 is the widest captured band");
    assert!(err.to_string().contains("longer than the model's"), "{err}");
    assert_eq!(sched.stats().num_requests_waiting, 0);
}
