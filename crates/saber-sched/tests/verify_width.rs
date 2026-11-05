//! Dynamic verify width through the whole scheduler — prod's rule, over captured widths.
//!
//! The policy itself is `saber-model::draft`, gated against prod's own code on 400 sequences /
//! 2541 steps; `tests/dflash_width_gate.rs` replays that section through *this* crate's wrapper.
//! What is tested here is the wiring: that the batch gate sees the running count, that each slot
//! verifies its own cap inside one captured shape, that the scheduled width is what the executor
//! reports back, and that a result which breaks the commit identity is counted rather than
//! believed.
//!
//! What is deliberately **not** asserted: that a narrower width is faster. That is a paired A/B on
//! the GPU (conc-8 +11–13%, `docs/LEDGER.md`), judged on tok/s — never on acceptance per draft,
//! which a degenerate repetition loop scores *higher* on.

mod common;

use common::{drain, request, scheduler, scheduler_by_batch, scheduler_ema, tokens_of};
use saber_model::draft::WidthPolicy;
use saber_sched::{
    Acceptance, BatchSlot, MockExecutor, RandomAcceptanceExecutor, RecordedBatch, ScriptedStep,
    SeqScript, SeqStep, StepBatch, StepExecutor, StepResult,
};
use saber_serve::engine::Engine;

/// The widths in the decode fixture's plan.
const WIDTHS: [u32; 3] = [1, 3, 5];

fn tail(batches: &[RecordedBatch], n: usize) -> Vec<&RecordedBatch> {
    let start = batches.len().saturating_sub(n);
    batches[start..].iter().collect()
}

#[test]
fn below_min_batch_every_step_verifies_the_full_width() {
    // Mismatch row 2. At concurrency 1 and 2 the decode GEMMs are weight-stream-bound, so
    // narrowing saves nothing and can only truncate a token that would have been accepted — the
    // measured conc-2 dip prod added MIN_BATCH for. The drafter here lands nothing at all, which
    // would collapse the width instantly if the gate were missing.
    for concurrency in [1usize, 2] {
        let sched = scheduler(RandomAcceptanceExecutor::new(11, Acceptance::Fixed(0)));
        for _ in 0..concurrency {
            sched.submit(request(&[1, 2, 3], 40)).expect("submit");
        }
        sched.run_steps(300);

        let batches = sched.with_executor(|e| e.batches().to_vec());
        for b in tail(&batches, 20) {
            assert_eq!(
                b.bucket.spec_width, 5,
                "concurrency {concurrency}: the cap must not bind below MIN_BATCH: {b:?}"
            );
            assert!(
                b.verify_widths.iter().all(|w| *w == 5),
                "concurrency {concurrency}: {b:?}"
            );
        }
    }
}

#[test]
fn at_min_batch_the_cap_binds_and_settles_where_the_drafter_lands() {
    // Three running requests, a drafter that lands exactly 2: cap = ceil(2) + 2 = 4, and the
    // narrowest captured shape that covers 4 is the width-5 bucket. **Every slot then verifies
    // 5**, not 4: the bucket launches five rows and five GDN state slots per sequence whatever a
    // slot asks for, so verifying four of them would pay for the fifth and throw its draft away
    // (`VerifyWidthPolicy::slot_in_bucket`). The narrowing the rounding-up absorbed is counted.
    let sched = scheduler_ema(RandomAcceptanceExecutor::new(42, Acceptance::Fixed(2)));
    for _ in 0..3 {
        sched.submit(request(&[1, 2, 3], 120)).expect("submit");
    }
    sched.run_steps(400);

    let batches = sched.with_executor(|e| e.batches().to_vec());
    let settled = tail(&batches, 20);
    for b in &settled {
        assert_eq!(b.bucket.spec_width, 5, "{b:?}");
        assert!(
            b.verify_widths.iter().all(|w| *w == 5),
            "every slot verifies the bucket it is paying for: {b:?}"
        );
    }
    assert!(
        sched.counters().slot_caps_absorbed > 0,
        "the cap asked for 4 and the ladder's nearest shape is 5: that is a counted absorption"
    );

    let spec = sched.spec_counters();
    assert!(
        (spec.acceptance_per_draft() - 2.0).abs() < 0.1,
        "{spec}" // Display is the prod-shaped one-liner
    );
    // The denominator is the scheduled width, which is now the bucket's: 5.
    let per_draft = spec.num_draft_tokens as f64 / spec.num_drafts as f64;
    assert!(
        per_draft > 4.9 && per_draft <= 5.0,
        "draft tokens per draft {per_draft}"
    );
}

#[test]
fn a_drafter_that_never_lands_stops_at_two_not_at_the_captured_width_1() {
    // Mismatch row 5. The plan captures width 1, but a cap is `ceil(ema) + margin 2`, so 2 is as
    // narrow as the policy can ask for and the shape that covers it is the width-3 bucket. (The
    // `min_width = 2` knob agrees with that floor and, at prod's margin, never binds on its own --
    // the width gate measures it as changing 0 of 2541 captured steps.)
    //
    // The BUCKET is where the saving is, and the bucket is 3. The slots verify 3, not 2: the
    // third row is launched either way.
    let sched = scheduler_ema(RandomAcceptanceExecutor::new(42, Acceptance::Fixed(0)));
    for _ in 0..3 {
        sched.submit(request(&[1, 2, 3], 60)).expect("submit");
    }
    sched.run_steps(400);

    let batches = sched.with_executor(|e| e.batches().to_vec());
    for b in tail(&batches, 20) {
        assert_eq!(b.bucket.spec_width, 3, "{b:?}");
        assert!(b.verify_widths.iter().all(|w| *w == 3), "{b:?}");
    }
    assert!(
        sched.counters().width_narrowed_steps > 0,
        "the dynamic width did something: steps ran below the ceiling"
    );
}

#[test]
fn full_acceptance_climbs_back_out_of_a_narrowed_width() {
    // Mismatch row 1. A request whose drafts stop landing is narrowed to the floor; when the
    // drafter starts landing everything again, `obs = accepted + (accepted >= drafted)` is what
    // puts the EMA *above* the width it is currently allowed, so the cap can climb back out of
    // its own shadow. Without the `+1` the only evidence a capped request ever sees is the width
    // it already allows.
    //
    // The climb-out survives the bucket-level law because `drafted` is the BUCKET's width: a
    // narrowed step that accepts all three of its rows observes four, which is the evidence for
    // five.
    let sched = scheduler_ema(RandomAcceptanceExecutor::new(7, Acceptance::Fixed(0)));
    for _ in 0..3 {
        sched.submit(request(&[4, 5, 6], 400)).expect("submit");
    }
    sched.run_steps(60);
    let narrowed = sched
        .with_executor(|e| e.batches().last().cloned())
        .unwrap();
    assert_eq!(
        narrowed.bucket.spec_width, 3,
        "narrowed first: {narrowed:?}"
    );
    assert!(narrowed.verify_widths.iter().all(|w| *w == 3));

    // The drafter recovers: every scheduled token lands.
    sched.with_executor(|e| e.set_acceptance(Acceptance::Fixed(9)));
    sched.run_steps(12);

    let back = sched
        .with_executor(|e| e.batches().last().cloned())
        .unwrap();
    assert_eq!(
        back.bucket.spec_width, 5,
        "the width must climb back to the ceiling: {back:?}"
    );
    assert!(back.verify_widths.iter().all(|w| *w == 5), "{back:?}");
}

#[tokio::test]
async fn every_slot_verifies_its_buckets_width_and_the_narrowing_is_counted() {
    // One bucket is one captured shape, and the widest sequence sets it. The two cold sequences'
    // own caps are 2, but the bucket launches five rows for them regardless, so they verify five
    // and their narrowing is recorded as `slot_caps_absorbed` instead of being taken.
    //
    // This is the ws-t1 correction. Before it the cold slots verified 2 rows of a 5-row bucket:
    // three rows launched per slot per step, three positions of draft discarded, and nothing
    // saved anywhere — a loss that could only appear as acceptance, never as step time.
    let mut exec = MockExecutor::new();
    let cold: Vec<ScriptedStep> = std::iter::once(ScriptedStep::single(100))
        .chain((0..60).map(|i| ScriptedStep::speculative([200 + i], 5, 0)))
        .collect();
    let hot: Vec<ScriptedStep> = std::iter::once(ScriptedStep::single(300))
        .chain((0..60).map(|i| {
            ScriptedStep::speculative(
                [
                    400 + i * 6,
                    401 + i * 6,
                    402 + i * 6,
                    403 + i * 6,
                    404 + i * 6,
                    405 + i * 6,
                ],
                5,
                5,
            )
        }))
        .collect();
    exec.push_script(SeqScript::new(cold.clone()));
    exec.push_script(SeqScript::new(cold));
    exec.push_script(SeqScript::new(hot));

    let sched = scheduler_ema(exec);
    let a = sched.submit(request(&[1], 60)).expect("submit");
    let b = sched.submit(request(&[2], 60)).expect("submit");
    let c = sched.submit(request(&[3], 300)).expect("submit");
    sched.run_steps(40);

    let batches = sched.with_executor(|e| e.batches().to_vec());
    let last = batches
        .iter()
        .rev()
        .find(|b| b.sequences == 3)
        .expect("a three-sequence step");
    assert_eq!(
        last.bucket.spec_width, 5,
        "the widest sequence sets the shape"
    );
    assert_eq!(
        last.verify_widths,
        vec![5, 5, 5],
        "every slot verifies the shape it is paying for: {last:?}"
    );
    assert!(
        sched.counters().slot_caps_absorbed >= 2,
        "the two cold sequences' caps of 2 were absorbed by the width-5 bucket, and counted"
    );

    for sub in [a, b, c] {
        sched.abort(sub.id);
        let _ = drain(sub).await;
    }
    assert_eq!(sched.counters().executor_protocol_errors, 0);
    assert_eq!(sched.counters().executor_token_count_errors, 0);
}

#[tokio::test]
async fn an_arriving_request_does_not_widen_the_batch_it_joins() {
    // A sequence on its FIRST step drafts nothing (`verify_width` 0) and uses one row of whatever
    // bucket the step runs in, so its (absent) acceptance EMA must not be part of choosing that
    // bucket. A cold EMA asks for the ceiling, so without the filter one arriving request pulls a
    // settled batch up to the widest captured shape — and at conc-8 over a corpus, a request
    // arrives every time one finishes.
    //
    // Three sequences whose drafts never land settle at the width-3 bucket; a fourth is then
    // submitted, and the step it joins must still be width 3.
    let sched = scheduler_ema(RandomAcceptanceExecutor::new(31, Acceptance::Fixed(0)));
    let mut subs = Vec::new();
    for _ in 0..3 {
        subs.push(sched.submit(request(&[1, 2, 3], 200)).expect("submit"));
    }
    sched.run_steps(120);
    let settled = sched
        .with_executor(|e| e.batches().last().cloned())
        .expect("a settled batch");
    assert_eq!(settled.bucket.spec_width, 3, "narrowed first: {settled:?}");

    subs.push(sched.submit(request(&[4, 5, 6], 200)).expect("submit"));
    sched.run_steps(6);
    let joined = sched
        .with_executor(|e| e.batches().iter().find(|b| b.sequences == 4).cloned())
        .expect("a four-sequence step");
    assert_eq!(
        joined.bucket.spec_width, 3,
        "the joiner drafts nothing and must not widen the shape: {joined:?}"
    );
    assert!(
        joined.verify_widths.contains(&0),
        "the joiner's own slot verifies nothing on its first step: {joined:?}"
    );

    // THE CONTROL, and it is the excluded input asked directly: the joiner's EMA at that step is
    // the fresh one, and a fresh EMA asks for the CEILING. So had it been part of the decision the
    // batch would have run width 5 instead of width 3 — which is the whole difference the filter
    // makes, stated as an arithmetic fact rather than as a second run.
    let policy = sched.config().width;
    assert_eq!(
        policy.slot_width(&policy.new_ema(), 4, policy.ceiling_width()),
        5,
        "a cold EMA asks for the ceiling; the filter is what keeps it out of the bucket choice"
    );
    assert_eq!(policy.ceiling_width(), 5);

    for sub in subs {
        sched.abort(sub.id);
        let _ = drain(sub).await;
    }
}

#[test]
fn the_per_position_histogram_is_kept() {
    // Mismatch row 9: `eval_drafter.py` reads the per-position curve to see WHERE acceptance
    // dies, and it is folded by saber-model's counters rather than re-derived here.
    let sched = scheduler(RandomAcceptanceExecutor::new(
        0x9111,
        Acceptance::Bernoulli { p: 0.6 },
    ));
    for _ in 0..3 {
        sched.submit(request(&[1, 2], 80)).expect("submit");
    }
    sched.run_steps(400);

    let spec = sched.spec_counters();
    assert!(spec.num_drafts > 50, "{spec}");
    assert_eq!(
        spec.accepted_per_pos.iter().sum::<u64>(),
        spec.num_accepted_tokens,
        "the histogram sums to the accepted total, over the draft positions and no more"
    );
    assert!(
        spec.accepted_per_pos.len() <= 5,
        "{:?}",
        spec.accepted_per_pos
    );
    for w in spec.accepted_per_pos.windows(2) {
        assert!(
            w[0] >= w[1],
            "acceptance is a prefix, so the curve cannot rise: {:?}",
            spec.accepted_per_pos
        );
    }
    // And the three totals the serving plane reads still agree with it.
    let stats = sched.stats();
    assert_eq!(stats.spec_decode_num_drafts, spec.num_drafts);
    assert_eq!(stats.spec_decode_num_draft_tokens, spec.num_draft_tokens);
    assert_eq!(
        stats.spec_decode_num_accepted_tokens,
        spec.num_accepted_tokens
    );
}

// ── Executors that break the contract ───────────────────────────────────────────

/// Reports the drafter's full block width instead of the scheduled one (mismatch row 10).
#[derive(Debug)]
struct WidthLiar {
    block: u32,
}

impl StepExecutor for WidthLiar {
    fn execute(&mut self, batch: &StepBatch<'_>) -> StepResult {
        StepResult::new(batch.slots.iter().map(|s: &BatchSlot| SeqStep {
            seq_id: s.seq_id,
            tokens: vec![7],
            draft_tokens: if s.first_step { 0 } else { self.block },
            accepted_tokens: 0,
            logprobs: None,
            prompt_logprobs: None,
        }))
    }
}

/// Commits more tokens than it accepted (mismatch row 8).
#[derive(Debug)]
struct CountLiar;

impl StepExecutor for CountLiar {
    fn execute(&mut self, batch: &StepBatch<'_>) -> StepResult {
        StepResult::new(batch.slots.iter().map(|s: &BatchSlot| SeqStep {
            seq_id: s.seq_id,
            // Three tokens for at most one accepted draft token: legal counts on their own, and
            // an impossible commit together.
            tokens: vec![7, 8, 9],
            draft_tokens: s.verify_width,
            accepted_tokens: s.verify_width.min(1),
            logprobs: None,
            prompt_logprobs: None,
        }))
    }
}

#[tokio::test]
async fn a_width_wider_than_the_schedule_is_a_counted_protocol_error() {
    // The scheduled width is the observation's denominator, so an executor that reports the
    // drafter's own block width would disable the climb-out silently. It is refused instead.
    let sched = scheduler(WidthLiar { block: 8 });
    let sub = sched.submit(request(&[1, 2], 40)).expect("submit");
    sched.run_steps(1); // prefill: verify_width 0, drafted 0, legal
    assert_eq!(sched.counters().executor_protocol_errors, 0);
    sched.run_steps(1); // decode: claims 8 against a scheduled 5
    assert_eq!(sched.counters().executor_protocol_errors, 1);
    assert_eq!(
        sched.stats().num_requests_running,
        0,
        "the request was aborted"
    );

    let events = drain(sub).await;
    assert_eq!(
        events.last().unwrap().finished,
        Some(saber_serve::engine::FinishReason::Abort)
    );
    assert_eq!(
        sched.spec_counters().num_drafts,
        0,
        "nothing was counted from it"
    );
}

#[tokio::test]
async fn a_commit_that_is_not_accepted_plus_one_is_a_counted_protocol_error() {
    let sched = scheduler(CountLiar);
    let sub = sched.submit(request(&[1, 2], 40)).expect("submit");
    sched.run_steps(1);
    assert_eq!(
        sched.counters().executor_token_count_errors,
        1,
        "3 tokens for 1 accepted breaks tokens == accepted + 1"
    );
    assert_eq!(
        sched.counters().executor_protocol_errors,
        0,
        "a different fault"
    );
    let events = drain(sub).await;
    assert_eq!(
        events.last().unwrap().finished,
        Some(saber_serve::engine::FinishReason::Abort)
    );
}

#[test]
fn seeded_bernoulli_acceptance_is_reproducible() {
    // A gate that cannot be re-run identically is not a gate.
    let run = || {
        let sched = scheduler(RandomAcceptanceExecutor::new(
            0x5EED,
            Acceptance::Bernoulli { p: 0.7 },
        ));
        for _ in 0..3 {
            sched.submit(request(&[9, 8, 7], 80)).expect("submit");
        }
        sched.run_steps(400);
        sched.with_executor(|e| e.batches().to_vec())
    };
    let a = run();
    let b = run();
    assert_eq!(a, b, "the seeded mock is not deterministic");

    let widths: Vec<u32> = a.iter().map(|x| x.bucket.spec_width).collect();
    assert!(widths.iter().all(|w| WIDTHS.contains(w)));
}

#[test]
fn the_knobs_are_prods_and_the_widths_are_the_plans() {
    let sched = scheduler(RandomAcceptanceExecutor::new(3, Acceptance::Fixed(4)));
    let policy = sched.config().width;
    assert_eq!(policy.widths, WIDTHS.to_vec(), "widths come from the plan");
    assert_eq!(
        policy.prod,
        WidthPolicy::PROD,
        "knobs come from saber-model's port of patch_dynwidth.py"
    );
    assert_eq!(policy.ceiling_width(), 5);
    assert_eq!(policy.floor_width(), 1);
}

#[tokio::test]
async fn a_single_stream_request_still_ends_where_it_should() {
    // MIN_BATCH means a single request runs at the full width for its whole life; the stop rules
    // are unaffected by that.
    let sched = scheduler(RandomAcceptanceExecutor::new(1, Acceptance::Fixed(3)));
    let sub = sched.submit(request(&[1, 2, 3], 37)).expect("submit");
    sched.run_steps(200);
    let events = drain(sub).await;
    assert_eq!(tokens_of(&events).len(), 37);
    assert_eq!(
        events.iter().filter(|e| e.finished.is_some()).count(),
        1,
        "exactly one finished"
    );
}

// =====================================================================================
// THE WIDTH AS A FUNCTION OF THE BATCH (ws-t2) — the default rule, through the scheduler
// =====================================================================================

#[test]
fn the_width_is_a_function_of_the_batch_and_not_of_the_acceptance() {
    // The default rule reads a table keyed on the running count and NOTHING else. The drafter
    // here lands nothing at all — the input that collapses the EMA cap to its floor in
    // `a_drafter_that_never_lands_stops_at_two_not_at_the_captured_width_1`, which is this test's
    // control — and the width does not move, because acceptance is not an input to it.
    for (concurrency, want) in [(1usize, 5u32), (2, 5), (3, 5), (4, 3), (6, 3)] {
        let sched = scheduler_by_batch(
            "1:5,4:3",
            RandomAcceptanceExecutor::new(11, Acceptance::Fixed(0)),
        );
        for _ in 0..concurrency {
            sched.submit(request(&[1, 2, 3], 40)).expect("submit");
        }
        sched.run_steps(300);
        let batches = sched.with_executor(|e| e.batches().to_vec());
        for b in tail(&batches, 20) {
            assert_eq!(
                b.bucket.spec_width, want,
                "concurrency {concurrency}: the table's own answer: {b:?}"
            );
            assert!(
                b.verify_widths.iter().all(|w| *w == want),
                "concurrency {concurrency}: every slot verifies the bucket: {b:?}"
            );
        }
    }
}

#[test]
fn a_table_cell_with_no_captured_shape_rounds_up_to_the_ceiling() {
    // The serving default is `1:8,4:5` and this fixture captures `1,3,5`: width 8 has no shape,
    // so a batch in the first row rounds UP to the ceiling. That is exactly what
    // `--engine-decode-widths 1,8` does with the conc-8 cell on a real boot, and it is why the
    // default is not a behaviour change until a plan captures both cells. Nothing is absorbed
    // here — a step that wanted a WIDER width than the plan has does not throw a draft away, it
    // runs the widest shape there is — and the fact that the table cannot move is a boot fact,
    // which `reachable_widths` states and the serve banner prints.
    let sched = scheduler(RandomAcceptanceExecutor::new(7, Acceptance::Fixed(2)));
    for _ in 0..8 {
        sched.submit(request(&[1, 2, 3], 60)).expect("submit");
    }
    sched.run_steps(400);
    let batches = sched.with_executor(|e| e.batches().to_vec());
    for b in tail(&batches, 10) {
        assert_eq!(b.bucket.spec_width, 5, "the ceiling, rounded up to: {b:?}");
    }
    assert_eq!(sched.config().width.reachable_widths(), vec![5]);
    assert!(
        !sched.config().width.can_narrow(),
        "one reachable width: the table has nowhere to go on this plan and the banner says INERT"
    );
    assert_eq!(
        sched.counters().slot_caps_absorbed,
        0,
        "rounding UP to the only shape there is throws nothing away"
    );
}

#[test]
fn a_table_cell_below_every_captured_width_is_absorbed_and_counted() {
    // The other direction, and the one that costs tokens: the table asks for 2 at batch 4 and the
    // narrowest captured shape at or above it is 3, so the step launches three rows per sequence
    // and every slot verifies all three. Truncating the slot to its own 2 would pay for the third
    // row and throw its draft away (`VerifyWidthPolicy::slot_in_bucket`), so the rounding stands
    // and the difference is COUNTED rather than hidden.
    let sched = scheduler_by_batch(
        "1:5,4:2",
        RandomAcceptanceExecutor::new(3, Acceptance::Fixed(1)),
    );
    for _ in 0..4 {
        sched.submit(request(&[1, 2, 3], 80)).expect("submit");
    }
    sched.run_steps(400);
    let batches = sched.with_executor(|e| e.batches().to_vec());
    let settled = tail(&batches, 10);
    for b in &settled {
        assert_eq!(
            b.bucket.spec_width, 3,
            "2 rounds up to the captured 3: {b:?}"
        );
        assert!(b.verify_widths.iter().all(|w| *w == 3), "{b:?}");
    }
    assert!(
        sched.counters().slot_caps_absorbed > 0,
        "a slot that asked for 2 inside a 3-wide bucket is the absorption this counter exists for"
    );
}

#[test]
fn an_arriving_request_cannot_move_the_shape_under_the_table() {
    // ws-t1's second prod/Saber mismatch, closed by construction rather than by a filter: under
    // the EMA rule a first-step sequence has no EMA and a fresh EMA asks for the ceiling, so one
    // arrival could pull a settled batch to the widest shape. Under the table the width is the
    // batch's, so an arrival moves the shape only by changing the BATCH — and here it does not,
    // because both counts sit in the same row of the table.
    let sched = scheduler_by_batch(
        "1:5,6:3",
        RandomAcceptanceExecutor::new(5, Acceptance::Fixed(4)),
    );
    for _ in 0..3 {
        sched.submit(request(&[1, 2, 3], 200)).expect("submit");
    }
    sched.run_steps(60);
    let before = sched.with_executor(|e| e.batches().to_vec());
    let settled: Vec<u32> = tail(&before, 5)
        .iter()
        .map(|b| b.bucket.spec_width)
        .collect();
    assert_eq!(settled, vec![5; settled.len()]);
    sched.submit(request(&[1, 2, 3], 200)).expect("submit");
    sched.run_steps(60);
    let after = sched.with_executor(|e| e.batches().to_vec());
    for b in tail(&after, 20) {
        assert_eq!(
            b.bucket.spec_width, 5,
            "a request joined and the width is still the table's: {b:?}"
        );
    }
}

/// Records, per decode step, whether any slot resumed from a commit wider than the step's block.
#[derive(Debug)]
struct FloorCheck {
    inner: RandomAcceptanceExecutor,
    violations: Vec<(u64, u32, u32)>,
}

impl StepExecutor for FloorCheck {
    fn execute(&mut self, batch: &StepBatch<'_>) -> StepResult {
        if batch.phase == saber_plan::Phase::Decode {
            for s in batch.slots.iter().filter(|s| !s.first_step) {
                let committed = s.input_tokens.len() as u32;
                if committed > batch.bucket.spec_width {
                    self.violations
                        .push((batch.step_index, committed, batch.bucket.spec_width));
                }
            }
        }
        self.inner.execute(batch)
    }
    fn admit(&mut self, seq: &saber_sched::AdmittedSeq) {
        self.inner.admit(seq);
    }
    fn release(&mut self, seq: saber_sched::SeqId) {
        self.inner.release(seq);
    }
}

/// **THE WIDTH FLOOR (ws-f38): a step is never narrower than what a slot committed last step.**
///
/// A decode step resumes each sequence from `state_slot + naccept - 1` inside ITS OWN `sidx`
/// block, so the block has to be at least as wide as the previous commit; the engine refuses a
/// step that is not by name and aborts the batch. Under the batch table `1:5,4:3` three settled
/// requests commit 5 at width 5, and a fourth arriving moves the table to 3: the step that
/// follows is raised back to 5 for the slots that committed more than 3, and counted. The
/// counter is the arm's own control: zero would mean the run never met the transition.
#[test]
fn the_width_is_never_narrower_than_a_slots_previous_commit() {
    let sched = scheduler_by_batch(
        "1:5,4:3",
        FloorCheck {
            inner: RandomAcceptanceExecutor::new(5, Acceptance::Fixed(4)),
            violations: Vec::new(),
        },
    );
    for _ in 0..3 {
        sched.submit(request(&[1, 2, 3], 200)).expect("submit");
    }
    sched.run_steps(40);
    sched.submit(request(&[1, 2, 3], 200)).expect("submit");
    sched.run_steps(80);
    let violations = sched.with_executor(|e| e.violations.clone());
    assert!(
        violations.is_empty(),
        "(step, committed, width) where a slot resumed past its block: {violations:?}"
    );
    assert!(
        sched.counters().width_floor_raised > 0,
        "the transition this arm is about never happened"
    );
}
