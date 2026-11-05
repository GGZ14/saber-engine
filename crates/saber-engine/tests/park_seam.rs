//! Parking through the real engine seam: a sequence leaves the device mid-generation and comes
//! back into **different** pages and a **different** state slot, and finishes the same stream.
//!
//! The scheduler has had a parking policy since ws-k5 and the tiers have been measured since K4,
//! but until now no executor could park: `StepExecutor::park_state` defaulted to `None`, the
//! scheduler counted `ExecutorCannotPark`, and every parking gate above ran on a mock. This is the
//! gate that says the engine can, and that it is the same session afterwards.
//!
//! What it is really asking. A parked session's pages and slot go back to the allocators, so the
//! restore lands somewhere else. If the engine's state were addressed rather than copied — a stale
//! block id, a slot index baked into a cache, a table rebuilt from the wrong session — the stream
//! would diverge at the first token after the restore and nowhere else. That is the 09-17
//! lazy-snapshot failure shape (`docs/LEDGER.md`): every single-shot gate read null and multi-turn
//! chat broke. So the instrument is a **continuation**, compared token for token against the run
//! that never left.

mod common;

use common::{
    drain, prompt_ids, request, scheduler_async_park, scheduler_async_pressure_policy,
    scheduler_two_slots, scheduler_with_damaged_park, scheduler_with_nvme, scheduler_with_park,
    scheduler_with_pressure_policy, tokens_of, Scratch, BATCHES, PREFILL,
};
use saber_engine::{DecodeSlot, ForwardRunner, NullForward, ParkImage, RealExecutor, RowRequest};
use saber_sched::{ParkError, SeqId, SeqState, StepExecutor};
use saber_serve::engine::{Engine, FinishReason, TokenId};

/// Drive the runner for `gen` tokens from a given admission, parking after `park_after` of them
/// and restoring into `into_blocks` — with `hold` extra sequences admitted in between so the state
/// slot the restore gets is **not** the one it gave back.
///
/// Returns the ids. With `park_after = None` nothing is parked, which is the control.
fn run_with_park(
    prompt: &[TokenId],
    gen: usize,
    blocks: &[u32],
    park_after: Option<usize>,
    into_blocks: &[u32],
) -> (Vec<TokenId>, Option<(u32, u32)>) {
    let mut f = NullForward::new(&[PREFILL], BATCHES, 6);
    let mut seq = SeqId(1);
    f.admit(seq, prompt.len() as u32, blocks).expect("admit");
    let mut out = Vec::new();
    let o = f
        .prefill(seq, prompt, 0, true, &RowRequest::greedy())
        .expect("prefill");
    out.push(o.token().expect("the last chunk samples"));
    let mut slots = None;

    for i in 1..gen {
        if park_after == Some(i) {
            let before_slot = f.session(seq).expect("live").0;
            let image = f.park(seq).expect("park");
            // The park reads; the release is what frees the pages and the slot, exactly as the
            // scheduler does it.
            f.release(seq);
            // Somebody else takes the slot that was just freed — the free list is a stack, so
            // without this the restore would land right back where it started and the gate would
            // be asserting nothing.
            f.admit(SeqId(90), 8, &[90, 91]).expect("an interloper");
            seq = SeqId(2);
            f.admit(seq, prompt.len() as u32, into_blocks)
                .expect("re-admit");
            f.restore(seq, &image).expect("restore");
            let after_slot = f.session(seq).expect("live").0;
            slots = Some((before_slot, after_slot));
        }
        let ctx = prompt.len() as u32 + out.len() as u32 - 1;
        let o = f
            .decode(&[DecodeSlot::single(seq, out[out.len() - 1], ctx)], 1, 1)
            .expect("decode");
        out.push(o.token().expect("a decode step samples"));
    }
    (out, slots)
}

#[test]
fn a_parked_session_resumes_into_different_pages_and_a_different_slot() {
    // The engine-level property, and the one the device holds too: the image is the CONTENTS of a
    // session's pages and slot, so where it lands is free. A slot index or a block id that had
    // leaked into the state would show up here as a divergence at the first token after the
    // restore and nowhere else — the 09-17 lazy-snapshot shape.
    let prompt = prompt_ids(234);
    let gen = 12usize;
    let from: Vec<u32> = (0..15).collect();
    let into: Vec<u32> = (40..55).collect();

    let (control, none) = run_with_park(&prompt, gen, &from, None, &[]);
    assert_eq!(control.len(), gen);
    assert!(none.is_none());

    let (parked, slots) = run_with_park(&prompt, gen, &from, Some(4), &into);
    let (before_slot, after_slot) = slots.expect("the run parked");
    assert_ne!(
        before_slot, after_slot,
        "the restore must land in a DIFFERENT state slot, or this gate tests nothing"
    );
    assert_eq!(
        parked, control,
        "a session that left the device and came back into other pages and another slot must \
         continue the stream it would have produced"
    );

    // CONTROL: the comparison has teeth only because a different session state gives different
    // ids. A stub that ignored its state would pass the line above.
    let (other, _) = run_with_park(&prompt_ids(233), gen, &from, None, &[]);
    assert_ne!(
        other, control,
        "a different prompt must not give the same ids"
    );
}

#[tokio::test]
async fn a_parked_session_through_the_scheduler_continues_its_stream() {
    // The same property through the whole seam: the scheduler's park/unpark, the shipped
    // `TieredParkStore`, `RealExecutor`'s hooks, the runner's image. Nothing else is in flight
    // here, so the free list hands the parked session its own slot back and it is the PAGES that
    // differ — the half the scheduler controls on its own. The different-slot half through the
    // scheduler is gated separately, with an interloper holding the old slot.
    let prompt = prompt_ids(234);
    let gen = 12usize;
    let (control, _) = run_with_park(&prompt, gen, &(0..15).collect::<Vec<u32>>(), None, &[]);

    let sched = scheduler_with_park(16);
    let sub = sched.submit(request(&prompt, gen)).expect("submit");
    sched.run_steps(4);
    let before = sched
        .with_executor(|e| e.forward_mut().session(SeqId(1)))
        .expect("admitted");

    sched.park(sub.id).expect("park");
    assert_eq!(sched.parked_ids(), vec![sub.id]);
    let counters = sched.with_executor(|e| e.counters());
    assert_eq!(counters.parks, 1, "the engine copied the session out");
    assert_eq!(counters.park_failures, 0);
    assert_eq!(
        sched.stats().num_requests_running,
        0,
        "its pages and its slot went back"
    );

    sched.unpark(sub.id).expect("unpark");
    sched.run_steps(500);
    let after = sched
        .with_executor(|e| e.forward_mut().session(SeqId(2)))
        .or_else(|| sched.with_executor(|e| e.forward_mut().session(SeqId(1))));
    let _ = (before, after);

    let events = drain(sub).await;
    assert_eq!(
        tokens_of(&events),
        control,
        "a session parked and restored through the scheduler must continue the same stream"
    );
    assert_eq!(events.last().unwrap().finished, Some(FinishReason::Length));

    let counters = sched.with_executor(|e| e.counters());
    assert_eq!(counters.restores, 1, "{}", counters.park_line());
    assert_eq!(counters.restore_failures, 0);
    assert!(counters.clean(), "{}", counters.line());
    assert_eq!(sched.counters().restore_failures, 0);
    assert_eq!(sched.park_counters().restored, 1);
}

#[tokio::test]
async fn a_corrupt_image_is_refused_on_the_live_path_and_the_session_stays_parked() {
    // `CLAUDE.md` rule 8, on the path that matters: not a unit test of the decoder, but a flipped
    // byte inside the scheduler's own restore, with the engine on the other side of it.
    let prompt = prompt_ids(64);
    let (sched, log) = scheduler_with_damaged_park();
    let sub = sched.submit(request(&prompt, 40)).expect("submit");
    sched.run_steps(3);
    sched.park(sub.id).expect("park");

    log.lock().expect("damage log").arm = true;
    sched.unpark(sub.id).expect("queued for restore");
    sched.run_steps(2);

    assert_eq!(sched.counters().restores, 0);
    assert_eq!(sched.counters().restore_failures, 1);
    assert_eq!(
        sched.parked_ids(),
        vec![sub.id],
        "a refused restore leaves the session parked, never half-restored"
    );
    let counters = sched.with_executor(|e| e.counters());
    assert_eq!(counters.restores, 0);
    assert_eq!(counters.restore_failures, 1, "{}", counters.park_line());
    assert_eq!(
        log.lock().expect("damage log").damaged,
        1,
        "the fault was injected exactly once"
    );

    sched.abort(sub.id);
    let events = drain(sub).await;
    assert_eq!(events.last().unwrap().finished, Some(FinishReason::Abort));
}

#[test]
fn a_runner_that_cannot_park_makes_the_scheduler_refuse_and_count() {
    // The behaviour before this branch, kept reachable: a forward with no way to move state is a
    // loud refusal, not an eviction whose state cannot be brought back.
    #[derive(Debug)]
    struct NoPark(NullForward);
    impl ForwardRunner for NoPark {
        fn facts(&self) -> &saber_engine::ForwardFacts {
            self.0.facts()
        }
        fn admit(
            &mut self,
            seq: SeqId,
            prompt_len: u32,
            kv_blocks: &[u32],
        ) -> Result<(), saber_engine::ForwardError> {
            self.0.admit(seq, prompt_len, kv_blocks)
        }
        fn release(&mut self, seq: SeqId) {
            self.0.release(seq);
        }
        fn prefill(
            &mut self,
            seq: SeqId,
            chunk: &[TokenId],
            done: u32,
            last: bool,
            _sampling: &RowRequest,
        ) -> Result<saber_engine::StepOutcome, saber_engine::ForwardError> {
            self.0
                .prefill(seq, chunk, done, last, &RowRequest::greedy())
        }
        fn decode(
            &mut self,
            slots: &[DecodeSlot],
            batch: u32,
            width: u32,
        ) -> Result<saber_engine::StepOutcome, saber_engine::ForwardError> {
            self.0.decode(slots, batch, width)
        }
        fn declared_launches(&self, rows: saber_engine::StepRows) -> Option<u32> {
            self.0.declared_launches(rows)
        }
        // can_park() defaults to false, which is the point.
    }

    let mut exec = RealExecutor::new(NoPark(NullForward::new(&[PREFILL], BATCHES, 2)));
    exec.admit(&saber_sched::AdmittedSeq {
        request_id: saber_serve::engine::RequestId(1),
        seq_id: SeqId(1),
        prompt_tokens: vec![1, 2, 3],
        kv_blocks: vec![0, 1],
        gdn_set: 0,
        gdn_slots: 1,
        choice_index: 0,
        shared_prefix_blocks: 0,
        cow_source: None,
    });
    assert!(
        exec.park_state(SeqId(1)).is_none(),
        "a runner that cannot park must answer None"
    );
    assert_eq!(exec.counters().parks, 0);
    assert_eq!(
        exec.counters().park_failures,
        0,
        "cannot-park is not a failure: it is the runner saying no before anything is attempted"
    );
}

#[test]
fn an_image_from_a_different_shape_is_refused_rather_than_written() {
    // The restore's own shape check: an image of the right length whose regions are not this
    // admission's is refused before anything is written.
    let mut f = NullForward::new(&[PREFILL], BATCHES, 4);
    f.admit(SeqId(1), 32, &[0, 1, 2]).expect("admit");
    let img = f.park(SeqId(1)).expect("park");
    assert_eq!(img.shape().0, 3, "three pages");

    f.admit(SeqId(2), 32, &[3, 4])
        .expect("admit the narrower one");
    let err = f
        .restore(SeqId(2), &img)
        .expect_err("a 3-page image into a 2-page admission");
    assert!(matches!(err, saber_engine::ForwardError::Image(_)), "{err}");

    // ...and the control: into an admission of the right shape it is accepted.
    f.admit(SeqId(3), 32, &[5, 6, 7]).expect("admit");
    f.restore(SeqId(3), &img).expect("the right shape restores");
}

#[test]
fn a_tampered_digest_list_is_refused_by_the_fold() {
    let mut f = NullForward::new(&[PREFILL], BATCHES, 4);
    f.admit(SeqId(1), 32, &[0, 1]).expect("admit");
    let mut img = f.park(SeqId(1)).expect("park");
    img.digests.chunks[0][0] ^= 0x01;
    f.admit(SeqId(2), 32, &[2, 3]).expect("admit");
    let err = f
        .restore(SeqId(2), &img)
        .expect_err("a chunk list that does not fold to the image digest");
    assert!(matches!(err, saber_engine::ForwardError::Image(_)), "{err}");
}

#[test]
fn the_image_survives_the_seqstate_round_trip() {
    let mut f = NullForward::new(&[PREFILL], BATCHES, 2);
    f.admit(SeqId(1), 64, &[0, 1, 2, 3]).expect("admit");
    let img = f.park(SeqId(1)).expect("park");
    let blob = SeqState(img.encode());
    assert_eq!(ParkImage::decode(&blob.0).expect("decode"), img);
}

/// A SESSION WHOSE PROMPT IS NOT IN YET IS NOT PARKED (ws-k17). The card run that first armed the
/// policy parked a pressure request between its admission and its prefill: the image carried what
/// the engine held (nothing), the restore marked the group prefilled, and its first decode step
/// died with `the batch says 714 token(s) are resident and the engine has 0`. The refusal is the
/// case; the same park one prefill later succeeding and the session finishing its own stream is the
/// control that says the refusal is about the prefill and nothing else.
#[tokio::test]
async fn a_session_still_prefilling_is_refused_and_parks_once_its_prompt_is_in() {
    let prompt = prompt_ids(234);
    let gen = 12usize;
    let (control, _) = run_with_park(&prompt, gen, &(0..15).collect::<Vec<u32>>(), None, &[]);

    let sched = scheduler_with_park(16);
    let a = sched.submit(request(&prompt_ids(97), 10)).expect("A");
    let b = sched.submit(request(&prompt, gen)).expect("B");
    // One step: both are admitted, and the step is ONE prefill chunk — A's.
    sched.step();
    assert_eq!(sched.stats().num_requests_running, 2);
    let err = sched.park(b.id).expect_err("B's prompt is not in");
    assert!(matches!(err, ParkError::StillPrefilling(_)), "{err}");
    assert_eq!(sched.counters().park_refusals, 1);
    assert_eq!(sched.counters().parks, 0);

    // The control: B's prefill runs, and the same park goes through and comes back whole.
    sched.run_steps(2);
    sched.park(b.id).expect("B's prompt is in now");
    sched.unpark(b.id).expect("unpark");
    sched.run_steps(500);
    assert_eq!(tokens_of(&drain(b).await), control);
    assert!(drain(a).await.last().unwrap().finished.is_some());
    assert_eq!(sched.counters().parks, 1);
}

/// The scheduler's own refusal, unchanged: no tier means no park.
#[test]
fn parking_without_a_tier_is_still_refused() {
    let sched = common::scheduler_no_park();
    let sub = sched.submit(request(&prompt_ids(16), 8)).expect("submit");
    sched.run_steps(1);
    assert!(matches!(sched.park(sub.id), Err(ParkError::NoTier)));
}

/// Every live session the runner holds, as `(seq id, state slot)`, seq order.
///
/// The scheduler hands each admission a fresh `SeqId`, so a restored session is not the id it was
/// parked under and a gate cannot name it in advance — it has to look.
fn live_slots(sched: &saber_sched::Scheduler<RealExecutor<NullForward>>) -> Vec<(u64, u32)> {
    sched.with_executor(|e| {
        (1..=16u64)
            .filter_map(|i| e.forward_mut().session(SeqId(i)).map(|s| (i, s.0)))
            .collect()
    })
}

#[tokio::test]
async fn a_parked_session_lands_in_a_different_slot_through_the_scheduler() {
    // The half that could not be gated through the scheduler before ws-l11. This engine used to
    // refuse every batched step, so the scheduler never had two sequences resident, the free list
    // is a stack, and the parked session got its own slot handed straight back — the assertion
    // would have been vacuous. With batch > 1 captured, somebody else can be holding that slot
    // when the restore asks for one, which is the arrangement the property is about.
    let prompt = prompt_ids(234);
    let gen = 12usize;
    let (control, _) = run_with_park(&prompt, gen, &(0..15).collect::<Vec<u32>>(), None, &[]);

    let sched = scheduler_with_park(16);
    let a = sched.submit(request(&prompt, gen)).expect("submit A");
    sched.run_steps(4);
    let before = live_slots(&sched);
    assert_eq!(before.len(), 1, "only A is resident: {before:?}");
    let a_slot = before[0].1;

    sched.park(a.id).expect("park A");
    assert_eq!(sched.parked_ids(), vec![a.id]);

    // The interloper. It is admitted while A is off the device, so it takes the slot A gave back.
    let b = sched
        .submit(request(&prompt_ids(97), 400))
        .expect("submit B");
    sched.run_steps(3);
    let held = live_slots(&sched);
    assert_eq!(held.len(), 1, "only B is resident: {held:?}");
    assert_eq!(
        held[0].1, a_slot,
        "the free list is a stack: B must take the slot A just freed, or this gate tests nothing"
    );

    sched.unpark(a.id).expect("unpark A");
    sched.run_steps(2);
    let both = live_slots(&sched);
    assert_eq!(both.len(), 2, "A came back beside B: {both:?}");
    assert!(
        both.iter().any(|(_, slot)| *slot != a_slot),
        "the restore must land in a DIFFERENT state slot: {both:?}"
    );

    sched.run_steps(2000);
    let events = drain(a).await;
    assert_eq!(
        tokens_of(&events),
        control,
        "a session restored into another slot, beside a sequence that took its old one, must \
         continue the stream it would have produced"
    );
    assert_eq!(events.last().unwrap().finished, Some(FinishReason::Length));

    let counters = sched.with_executor(|e| e.counters());
    assert_eq!(counters.parks, 1, "{}", counters.park_line());
    assert_eq!(counters.restores, 1, "{}", counters.park_line());
    assert!(counters.clean(), "{}", counters.line());

    sched.abort(b.id);
    let _ = drain(b).await;
}

#[tokio::test]
async fn the_pressure_policy_evicts_through_the_engine_and_the_stream_survives() {
    // `ParkPolicy::on_pressure` has been in the scheduler since ws-k5, but until this branch no
    // executor could park, so every firing of it ended in `ExecutorCannotPark` and the gate above
    // it ran on a mock. This is the policy driving the real engine: an admission refused for GDN
    // state evicts a resident session, the engine copies it off the device, the admission
    // succeeds, and the evicted session comes back on its own and finishes the stream it would
    // have produced.
    let prompt = prompt_ids(234);
    let gen = 12usize;
    let (control, _) = run_with_park(&prompt, gen, &(0..15).collect::<Vec<u32>>(), None, &[]);

    // Two resident sequences' worth of GDN state: a sequence costs `max_spec_width + 1` = 2 slots,
    // so four slots is two sequences and the third admission is refused for state, which is the
    // policy's trigger. `min_running` = 1 needs the two.
    let sched = scheduler_with_pressure_policy(4);
    let a = sched.submit(request(&prompt, gen)).expect("submit A");
    let b = sched
        .submit(request(&prompt_ids(97), 10))
        .expect("submit B");
    sched.run_steps(3);
    assert_eq!(
        sched.stats().num_requests_running,
        2,
        "both hold a state slot set"
    );
    assert_eq!(
        sched.counters().policy_parks,
        0,
        "nothing is under pressure yet"
    );

    // A is the deterministic victim: both committed a token this step, so the tie breaks on the
    // most work left, and A has the longer budget.
    let c = sched.submit(request(&prompt_ids(55), 8)).expect("submit C");
    sched.run_steps(2);
    assert_eq!(
        sched.counters().policy_parks,
        1,
        "the admission refused for state must have evicted one session"
    );
    assert_eq!(sched.counters().policy_park_misses, 0);
    assert_eq!(sched.parked_ids(), vec![a.id], "A is the victim");

    let counters = sched.with_executor(|e| e.counters());
    assert_eq!(
        counters.parks,
        1,
        "the ENGINE copied it off the device: {}",
        counters.park_line()
    );
    assert_eq!(counters.park_failures, 0);

    // Nobody sends an unpark: `auto_restore` brings the evicted session back when the space it was
    // evicted for comes free.
    sched.run_steps(4000);
    let events = drain(a).await;
    assert_eq!(
        tokens_of(&events),
        control,
        "a session the policy evicted must finish the stream it would have produced"
    );
    assert_eq!(events.last().unwrap().finished, Some(FinishReason::Length));

    let counters = sched.with_executor(|e| e.counters());
    assert_eq!(counters.restores, 1, "{}", counters.park_line());
    assert!(counters.clean(), "{}", counters.line());
    assert_eq!(sched.park_counters().restored, 1);

    for s in [b, c] {
        sched.abort(s.id);
        let _ = drain(s).await;
    }
}

/// THE SAME POLICY ON THE ASYNCHRONOUS PATH, which is the one `saber serve` takes on the card
/// (ws-k17). The first under-pressure card run with the policy actually armed read 25 x "eviction
/// policy could not park its victim: request 1 is not running": the park was issued, the group
/// stayed in the running set while its copy ran, every later step re-chose it and `park` refused
/// it — and when the copy landed, `finish_park` filed it as an EXPLICIT park, so `auto_restore`
/// never brought it back. The request hung until the client's timeout and held the server's
/// SIGTERM past its deadline. The victim must be parked once, waited on while it copies, marked as
/// the policy's, restored on its own, and finish its own stream.
#[tokio::test]
async fn the_pressure_policy_on_the_async_path_parks_once_and_brings_the_victim_back() {
    let prompt = prompt_ids(234);
    let gen = 12usize;
    let (control, _) = run_with_park(&prompt, gen, &(0..15).collect::<Vec<u32>>(), None, &[]);

    let sched = scheduler_async_pressure_policy(4, 3);
    let a = sched.submit(request(&prompt, gen)).expect("submit A");
    let b = sched
        .submit(request(&prompt_ids(97), 10))
        .expect("submit B");
    sched.run_steps(3);
    assert_eq!(sched.stats().num_requests_running, 2);

    let c = sched.submit(request(&prompt_ids(55), 8)).expect("submit C");
    // Enough steps for the three-poll copy to land, not enough for anything to finish.
    sched.run_steps(5);
    let sc = sched.counters();
    assert_eq!(sc.parks_started, 1, "one asynchronous park issued: {sc:?}");
    assert_eq!(
        sc.policy_parks, 1,
        "and counted as the policy's when it landed"
    );
    assert_eq!(
        sc.park_refusals, 0,
        "the victim is never re-chosen while it copies"
    );
    assert!(
        sc.policy_park_waits >= 1,
        "the admission waited on the copy"
    );
    assert_eq!(sched.parked_ids(), vec![a.id], "A is the victim");

    // Nobody sends an unpark.
    // `step` and not `run_steps`: a step whose only sequence is mid-restore is Idle, and
    // `run_steps` stops at an idle step where the serve plane's loop polls on.
    for _ in 0..4000 {
        sched.step();
    }
    let sc = sched.counters();
    assert!(
        sched.parked_ids().is_empty() && sched.stats().num_requests_running == 0,
        "everything finished, nothing left parked: parked {:?}, running {}, waiting {}, {sc:?}",
        sched.parked_ids(),
        sched.stats().num_requests_running,
        sched.stats().num_requests_waiting,
    );
    let events = drain(a).await;
    assert_eq!(
        tokens_of(&events),
        control,
        "the victim comes back on its own and finishes the stream it would have produced"
    );
    assert_eq!(events.last().unwrap().finished, Some(FinishReason::Length));
    let sc = sched.counters();
    assert_eq!(sc.policy_restores_queued, 1, "{sc:?}");
    assert_eq!(sc.restores, 1);
    assert_eq!(sc.park_refusals, 0);
    for s in [b, c] {
        assert!(drain(s).await.last().unwrap().finished.is_some());
    }
    assert!(sched.parked_ids().is_empty());
}

// =================================================================================================
// Parking off the step loop (workstream K9)
// =================================================================================================

#[tokio::test]
async fn a_park_in_flight_does_not_stall_the_other_sequences() {
    // The property the whole asynchronous path exists for. K8 measured a park at 642 ms and K9's
    // coalescing brought it to 217 ms — still seven decode steps. Off the step loop it should cost
    // the sequences that are NOT parking exactly nothing.
    //
    // The instrument is structural rather than a stopwatch: against a stub a step takes
    // microseconds, so a wall-clock bound would measure the test harness. What is asserted instead
    // is what a stopwatch would be standing in for — B commits one token on every step of the park,
    // A commits none while it is in flight, and A's pages and slot are still held until the copy is
    // done. A scheduler that waited for the park would show B missing tokens; one that freed early
    // would show the arena growing before the poll finished.
    const POLLS: u32 = 6;
    let prompt = prompt_ids(234);
    let gen = 24usize;
    let (control, _) = run_with_park(&prompt, gen, &(0..15).collect::<Vec<u32>>(), None, &[]);

    let sched = scheduler_async_park(16, POLLS);
    let a = sched.submit(request(&prompt, gen)).expect("submit A");
    let b = sched
        .submit(request(&prompt_ids(97), 400))
        .expect("submit B");
    sched.run_steps(6);
    assert_eq!(sched.stats().num_requests_running, 2, "both are resident");
    let (kv_free_before, gdn_free_before) = sched.arena_free();

    sched.park(a.id).expect("park A");
    assert_eq!(
        sched.counters().parks_started,
        1,
        "the park was ISSUED, not performed"
    );
    assert_eq!(sched.counters().parks, 0, "and it has not finished");
    assert_eq!(
        sched.arena_free(),
        (kv_free_before, gdn_free_before),
        "a park in flight still holds its pages and its slot: the copy is reading them"
    );
    assert!(
        sched.parked_ids().is_empty(),
        "and the session is not parked yet"
    );

    // Step through the transfer. B must not miss a single one.
    let counters_before = sched.with_executor(|e| e.counters());
    for _ in 0..POLLS {
        sched.run_steps(1);
    }
    let counters_after = sched.with_executor(|e| e.counters());
    assert_eq!(
        counters_after.decode_steps - counters_before.decode_steps,
        u64::from(POLLS),
        "every step of the park ran a decode for the sequences that were not parking"
    );
    assert_eq!(
        counters_after.parks,
        1,
        "...and the park finished inside them: {}",
        counters_after.park_line()
    );
    assert_eq!(sched.counters().parks, 1);
    assert_eq!(sched.parked_ids(), vec![a.id], "now it is parked");
    let (kv_free_after, gdn_free_after) = sched.arena_free();
    assert!(
        kv_free_after > kv_free_before && gdn_free_after > gdn_free_before,
        "the bytes are free exactly when they are free: {kv_free_before}->{kv_free_after} KV, \
         {gdn_free_before}->{gdn_free_after} GDN"
    );

    // And the session is still the same session.
    sched.unpark(a.id).expect("unpark A");
    sched.run_steps(4000);
    let events = drain(a).await;
    assert_eq!(
        tokens_of(&events),
        control,
        "a session parked off the step loop continues the stream it would have produced"
    );
    assert_eq!(events.last().unwrap().finished, Some(FinishReason::Length));
    let counters = sched.with_executor(|e| e.counters());
    assert_eq!(counters.restores, 1, "{}", counters.park_line());
    assert!(counters.clean(), "{}", counters.line());
    assert!(
        sched.counters().steps_during_transfer >= u64::from(POLLS),
        "the step loop ran while transfers were in flight"
    );

    sched.abort(b.id);
    let _ = drain(b).await;
}

#[tokio::test]
async fn a_park_and_a_restore_in_flight_across_the_same_steps_both_land() {
    // Two sessions moving in opposite directions over the same steps, with a third running
    // throughout. The engine holds ONE staging buffer, so the two transfers are serialised rather
    // than overlapped — the second is told `Busy` and asks again next step — and what this gate
    // says is that serialising them loses nothing: both complete, both streams are the ones they
    // would have produced, and the sequence that was doing neither never missed a step.
    const POLLS: u32 = 4;
    let p_a = prompt_ids(234);
    let p_c = prompt_ids(151);
    let gen = 24usize;
    let (control_a, _) = run_with_park(&p_a, gen, &(0..15).collect::<Vec<u32>>(), None, &[]);
    let (control_c, _) = run_with_park(&p_c, gen, &(0..10).collect::<Vec<u32>>(), None, &[]);

    let sched = scheduler_async_park(16, POLLS);
    let a = sched.submit(request(&p_a, gen)).expect("submit A");
    let c = sched.submit(request(&p_c, gen)).expect("submit C");
    let b = sched
        .submit(request(&prompt_ids(97), 4000))
        .expect("submit B");
    sched.run_steps(8);
    assert_eq!(sched.stats().num_requests_running, 3);

    // C goes off first and comes all the way back, so that its RESTORE is asked for while A's park
    // is still in flight.
    sched.park(c.id).expect("park C");
    sched.run_steps(POLLS as usize + 1);
    assert_eq!(sched.parked_ids(), vec![c.id]);

    let decode_before = sched.with_executor(|e| e.counters()).decode_steps;
    sched.park(a.id).expect("park A");
    // ...and immediately ask for C back. One buffer, so one of them waits.
    let _ = sched.unpark(c.id);
    sched.run_steps(4 * POLLS as usize + 8);
    let decode_after = sched.with_executor(|e| e.counters()).decode_steps;
    assert_eq!(
        decode_after - decode_before,
        (4 * POLLS + 8) as u64,
        "B kept stepping through both transfers"
    );

    // Both of them get where they were going.
    sched.run_steps(200);
    assert_eq!(
        sched.parked_ids(),
        vec![a.id],
        "A is parked and C is back: {:?}",
        sched.parked_ids()
    );
    sched.unpark(a.id).expect("unpark A");
    sched.run_steps(8000);

    let ev_a = drain(a).await;
    let ev_c = drain(c).await;
    assert_eq!(tokens_of(&ev_a), control_a, "A's stream survived");
    assert_eq!(tokens_of(&ev_c), control_c, "C's stream survived");
    let counters = sched.with_executor(|e| e.counters());
    assert_eq!(counters.parks, 2, "{}", counters.park_line());
    assert_eq!(counters.restores, 2, "{}", counters.park_line());
    assert!(counters.clean(), "{}", counters.line());

    sched.abort(b.id);
    let _ = drain(b).await;
}

// =================================================================================================
// The NVMe tier, end to end through the engine (workstream K10)
// =================================================================================================

#[tokio::test]
async fn a_session_spilled_to_nvme_and_promoted_back_continues_its_stream() {
    // The tier's demote/promote has been gated since K4 -- against a mock executor and a fake image.
    // This is the same rule with the **real engine** on one side and a real `O_DIRECT` file on the
    // other: a session parks to RAM, the next park spills it to the drive because RAM holds exactly
    // one, and the promote brings it back into DIFFERENT pages and a different slot.
    const POLLS: u32 = 3;
    let dir = Scratch::new("nvme-ids");
    let prompt_a = prompt_ids(234);
    let prompt_b = prompt_ids(151);
    let gen = 24usize;
    let (control_a, _) = run_with_park(&prompt_a, gen, &(0..15).collect::<Vec<u32>>(), None, &[]);
    let (control_b, _) = run_with_park(&prompt_b, gen, &(0..10).collect::<Vec<u32>>(), None, &[]);

    let sched = scheduler_with_nvme(16, POLLS, &dir.file("sessions.park"));
    let a = sched.submit(request(&prompt_a, gen)).expect("submit A");
    let b = sched.submit(request(&prompt_b, gen)).expect("submit B");
    let keep = sched
        .submit(request(&prompt_ids(97), 4000))
        .expect("submit the one that stays");
    sched.run_steps(8);
    assert_eq!(sched.stats().num_requests_running, 3);

    // A goes to RAM.
    sched.park(a.id).expect("park A");
    sched.run_steps(POLLS as usize + 2);
    let c = sched.park_counters();
    assert_eq!(c.parked_to_ram, 1, "A is in RAM: {c:?}");
    assert_eq!(c.demoted_to_nvme, 0);

    // B has nowhere to go but the drive -- so A spills and B takes the RAM slot.
    sched.park(b.id).expect("park B");
    sched.run_steps(POLLS as usize + 2);
    let c = sched.park_counters();
    assert_eq!(c.parked_to_ram, 2, "both were offered RAM: {c:?}");
    assert_eq!(
        c.demoted_to_nvme, 1,
        "one of them had to spill, because RAM holds one image: {c:?}"
    );
    assert_eq!(sched.parked_ids().len(), 2);

    // Both come back. One of them is read off the drive, which is the path under test.
    sched.unpark(a.id).expect("unpark A");
    sched.run_steps(400);
    sched.unpark(b.id).expect("unpark B");
    sched.run_steps(6000);

    let ev_a = drain(a).await;
    let ev_b = drain(b).await;
    assert_eq!(
        tokens_of(&ev_a),
        control_a,
        "the session that went to the drive must come back the same session"
    );
    assert_eq!(tokens_of(&ev_b), control_b);
    assert_eq!(ev_a.last().unwrap().finished, Some(FinishReason::Length));

    let c = sched.park_counters();
    assert_eq!(c.restored, 2, "{c:?}");
    assert_eq!(c.refused_corrupt, 0, "{c:?}");
    let counters = sched.with_executor(|e| e.counters());
    assert_eq!(counters.parks, 2, "{}", counters.park_line());
    assert_eq!(counters.restores, 2, "{}", counters.park_line());
    assert!(counters.clean(), "{}", counters.line());

    sched.abort(keep.id);
    let _ = drain(keep).await;
}

#[tokio::test]
async fn a_corrupt_parking_file_is_refused_on_the_live_path() {
    // `CLAUDE.md` rule 8 on the drive: not a unit test of the tier's digest, but a byte flipped in
    // the **file** while the session is spilled, with the real engine on the other side of the
    // restore. The tier's own digest covers host<->drive, so this is the one that must catch it --
    // and the session has to stay parked rather than come back as whatever the drive said.
    let dir = Scratch::new("nvme-corrupt");
    let path = dir.file("sessions.park");
    let prompt = prompt_ids(234);

    let sched = scheduler_with_nvme(16, 0, &path);
    let a = sched.submit(request(&prompt, 40)).expect("submit A");
    let b = sched
        .submit(request(&prompt_ids(151), 40))
        .expect("submit B");
    sched.run_steps(6);
    sched.park(a.id).expect("park A");
    sched.park(b.id).expect("park B");
    sched.run_steps(4);
    let c = sched.park_counters();
    assert_eq!(c.demoted_to_nvme, 1, "one is on the drive: {c:?}");

    // Flip a byte deep inside the file, past any header, while the image is sitting there.
    {
        use std::io::{Seek, SeekFrom, Write};
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("open the parking file");
        let len = f.metadata().expect("stat").len();
        assert!(len > 8192, "the parking file is {len} bytes");
        // Inside SLOT 0 -- the first spill goes there -- and past its header. The first cut flipped
        // at `len / 2`, which is in the second slot: the file is pre-allocated for three sessions
        // and only one is in it, so the flip landed in bytes nothing reads and both images restored
        // clean. A corruption gate that corrupts nothing passes for the wrong reason.
        f.seek(SeekFrom::Start(4096)).expect("seek");
        f.write_all(&[0xA5]).expect("flip a byte");
        f.sync_all().expect("sync");
    }

    // Whichever of the two is on the drive is now unrestorable; the other is fine. Ask for both.
    let _ = sched.unpark(a.id);
    sched.run_steps(6);
    let _ = sched.unpark(b.id);
    sched.run_steps(6);

    let c = sched.park_counters();
    assert_eq!(
        c.refused_corrupt, 1,
        "the flipped image must be refused by its digest, not restored: {c:?}"
    );
    assert_eq!(
        sched.parked_ids().len(),
        1,
        "and it must still be parked, never half-restored"
    );
    assert_eq!(
        sched.counters().restore_failures,
        1,
        "counted at the scheduler too"
    );
    // The engine never saw the bad bytes: the tier refused before they reached it.
    let counters = sched.with_executor(|e| e.counters());
    assert_eq!(
        counters.restore_failures,
        0,
        "the tier caught it, so the engine was never asked: {}",
        counters.park_line()
    );

    for s in [a, b] {
        sched.abort(s.id);
        let _ = drain(s).await;
    }
}

#[tokio::test]
async fn two_staging_slots_let_a_park_and_a_restore_overlap() {
    // The other half of the race gate. With ONE slot the two transfers take turns and at least one is
    // told `Busy` (counted, retried, and the gate above proves nothing is lost by it). With TWO they
    // have a slot each, so neither waits and `park_transfer_busy` stays at zero — which is the whole
    // observable difference the option buys, and what says whether it is worth its pinned bytes.
    const POLLS: u32 = 5;
    let p_a = prompt_ids(234);
    let p_c = prompt_ids(151);
    let gen = 24usize;
    let (control_a, _) = run_with_park(&p_a, gen, &(0..15).collect::<Vec<u32>>(), None, &[]);
    let (control_c, _) = run_with_park(&p_c, gen, &(0..10).collect::<Vec<u32>>(), None, &[]);

    for two in [false, true] {
        let sched = if two {
            scheduler_two_slots(16, POLLS)
        } else {
            scheduler_async_park(16, POLLS)
        };
        let a = sched.submit(request(&p_a, gen)).expect("submit A");
        let c = sched.submit(request(&p_c, gen)).expect("submit C");
        let b = sched
            .submit(request(&prompt_ids(97), 8000))
            .expect("submit B");
        sched.run_steps(8);

        // C off first, so its restore can be asked for while A's park is still going.
        sched.park(c.id).expect("park C");
        sched.run_steps(POLLS as usize + 1);
        sched.park(a.id).expect("park A");
        let _ = sched.unpark(c.id);
        sched.run_steps(4 * POLLS as usize + 8);

        let busy = sched.counters().park_transfer_busy;
        if two {
            assert_eq!(
                busy, 0,
                "with two slots neither transfer waits for the other"
            );
        } else {
            assert!(
                busy >= 1,
                "with one slot the second transfer must be told Busy at least once"
            );
        }

        // Either way both sessions end up where they were going and neither stream changes: the slot
        // count is a throughput knob, never a correctness one.
        sched.run_steps(400);
        sched.unpark(a.id).expect("unpark A");
        sched.run_steps(12_000);
        let ev_a = drain(a).await;
        let ev_c = drain(c).await;
        assert_eq!(tokens_of(&ev_a), control_a, "A's stream, two_slots={two}");
        assert_eq!(tokens_of(&ev_c), control_c, "C's stream, two_slots={two}");
        let counters = sched.with_executor(|e| e.counters());
        assert!(counters.clean(), "two_slots={two}: {}", counters.line());

        sched.abort(b.id);
        let _ = drain(b).await;
    }
}

/// THE BOUNDED RESTORE THROUGH THE ENGINE (ws-k18), on the serve plane's asynchronous path: every
/// restore refused by the executor (`fail_restores`, the card control's switch) is retried up to
/// `max_restore_attempts` and the victim's request then ENDS with a named error, its image
/// discarded on the executor; the neighbours finish. Before ws-k18 the failed async restore went
/// back as an explicit park that nothing asked for again, and the request never ended. The control
/// is the same scene with the switch off: the victim is restored and finishes its stream.
#[tokio::test]
async fn a_restore_refused_every_time_ends_the_request_with_its_reason() {
    let run = |fail: bool| {
        let sched = scheduler_async_pressure_policy(4, 3);
        sched.with_executor(|e| e.fail_restores(fail));
        let a = sched.submit(request(&prompt_ids(234), 12)).expect("A");
        let b = sched.submit(request(&prompt_ids(97), 10)).expect("B");
        sched.run_steps(3);
        let c = sched.submit(request(&prompt_ids(55), 8)).expect("C");
        for _ in 0..4000 {
            sched.step();
        }
        (sched, a, b, c)
    };

    let (sched, a, b, c) = run(true);
    let sc = sched.counters();
    assert_eq!(sc.policy_parks, 1, "{sc:?}");
    assert_eq!(sc.restores_given_up, 1, "{sc:?}");
    assert_eq!(sc.restores, 0);
    assert_eq!(
        sc.parked_discards, 1,
        "the given-up image is discarded on the executor"
    );
    assert!(sched.parked_ids().is_empty());
    let ec = sched.with_executor(|e| e.counters().clone());
    assert_eq!(
        ec.restores_refused_by_injection, 3,
        "three attempts, each refused"
    );
    assert_eq!(ec.parked_discards, 1);
    let events = drain(a).await;
    let last = events.last().expect("A ended");
    assert_eq!(last.finished, Some(FinishReason::Abort));
    assert!(
        last.error
            .as_deref()
            .is_some_and(|e| e.contains("after 3 attempt(s)")),
        "{:?}",
        last.error
    );
    for s in [b, c] {
        let ev = drain(s).await;
        assert!(ev.last().unwrap().finished.is_some());
        assert!(ev.iter().all(|e| e.error.is_none()));
    }

    // THE CONTROL: nothing refused, the victim restored, no error anywhere.
    let (sched, a, b, c) = run(false);
    let sc = sched.counters();
    assert_eq!(sc.restores_given_up, 0, "{sc:?}");
    assert_eq!(sc.restores, 1);
    assert_eq!(sc.parked_discards, 0);
    for s in [a, b, c] {
        let ev = drain(s).await;
        assert!(ev.iter().all(|e| e.error.is_none()));
        assert_ne!(ev.last().unwrap().finished, Some(FinishReason::Abort));
    }
}
