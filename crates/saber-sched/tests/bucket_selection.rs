//! Bucket selection through the whole scheduler: what shape actually got replayed.
//!
//! The unit tests in `src/buckets.rs` prove the selection function; these prove the scheduler only
//! ever hands the executor a shape the plan captured — which is the property that makes
//! "replay a captured graph" true rather than aspirational (`ARCHITECTURE.md` § 4. Execution plane,
//! and `CLAUDE.md` rule 7: launch count per bucket is declared statically and asserted at capture).

mod common;

use common::{decode_buckets, plan_with_capture, request, scheduler, scheduler_from_plan, GIB};
use saber_sched::{Acceptance, ArenaBytes, MockExecutor, RandomAcceptanceExecutor, SeqScript};
use saber_serve::engine::Engine;

#[test]
fn the_batch_rounds_up_and_the_spare_slots_are_padding() {
    let mut exec = MockExecutor::new();
    for _ in 0..3 {
        exec.push_script(SeqScript::per_token([10, 11, 12, 13]));
    }
    let sched = scheduler(exec);
    for _ in 0..3 {
        sched.submit(request(&[1, 2, 3], 64)).expect("submit");
    }
    sched.run_steps(2);

    let batches = sched.with_executor(|e| e.batches().to_vec());
    let first = &batches[0];
    assert_eq!(first.sequences, 3);
    assert_eq!(
        first.bucket.batch, 4,
        "3 running rounds up to the batch-4 bucket"
    );
    assert_eq!(
        first.padded_slots, 1,
        "the fourth slot is padding, and it is run"
    );
    assert!(first.bucket.captured);
}

#[test]
fn one_sequence_gets_the_smallest_bucket_of_all() {
    let mut exec = MockExecutor::new();
    exec.push_script(SeqScript::per_token([10, 11]));
    let sched = scheduler(exec);
    sched.submit(request(&[1, 2], 64)).expect("submit");
    sched.run_steps(1);

    let batches = sched.with_executor(|e| e.batches().to_vec());
    assert_eq!(batches[0].bucket.batch, 1);
    assert_eq!(
        batches[0].bucket.spec_width, 1,
        "a prefill step drafts nothing, so it does not pull in a wide bucket"
    );
    assert_eq!(
        batches[0].bucket.ctx_band, 8192,
        "the narrow band covers a 2-token prompt"
    );
    assert_eq!(
        batches[0].verify_widths,
        vec![0],
        "no draft on the prefill step"
    );
}

#[test]
fn every_replayed_shape_is_a_captured_plan_bucket() {
    // A mixed workload: arrivals at different steps, different lengths, speculation on.
    let sched = scheduler(RandomAcceptanceExecutor::new(
        0xC0FFEE,
        Acceptance::Fixed(2),
    ));
    let config = sched.config();
    sched.submit(request(&[1, 2, 3], 40)).expect("submit");
    sched.run_steps(3);
    for i in 0..5 {
        sched.submit(request(&[7; 12], 20 + i)).expect("submit");
    }
    sched.run_steps(200);

    let batches = sched.with_executor(|e| e.batches().to_vec());
    assert!(
        batches.len() > 10,
        "the workload actually ran: {}",
        batches.len()
    );
    for b in &batches {
        assert!(b.bucket.captured, "an eager shape was replayed: {b:?}");
        assert!(
            config.buckets.contains(&b.bucket),
            "a shape outside the plan was replayed: {b:?}"
        );
        assert!(
            b.sequences <= b.bucket.batch,
            "more live sequences than the captured batch: {b:?}"
        );
        assert_eq!(b.padded_slots, b.bucket.batch - b.sequences);
        for w in &b.verify_widths {
            assert!(
                *w <= b.bucket.spec_width,
                "a slot verified wider than the shape: {b:?}"
            );
        }
    }
}

#[test]
fn the_context_band_widens_with_the_context() {
    let sched = scheduler(RandomAcceptanceExecutor::new(5, Acceptance::Fixed(1)));
    // A prompt just under the narrow band: the first steps use it, and once the sequence crosses
    // 8192 tokens the next band takes over.
    let prompt = vec![3u32; 8_150];
    sched.submit(request(&prompt, 200)).expect("submit");
    sched.run_steps(200);

    let batches = sched.with_executor(|e| e.batches().to_vec());
    assert_eq!(batches[0].bucket.ctx_band, 8192);
    assert_eq!(
        batches.last().unwrap().bucket.ctx_band,
        65536,
        "the band is chosen per step from the live context"
    );
    // The band only ever widens for a growing sequence.
    let mut bands: Vec<u32> = batches.iter().map(|b| b.bucket.ctx_band).collect();
    bands.dedup();
    assert_eq!(bands, vec![8192, 65536]);
}

#[test]
fn an_eager_bucket_in_the_plan_is_never_replayed() {
    // The batch-1 bucket would win selection on every axis — but it is not captured, so the
    // scheduler drops it at construction and pads into the batch-4 shape instead.
    let plan = plan_with_capture(&[(1, 1, 8192, false), (4, 1, 8192, true)], 2);
    assert_eq!(plan.buckets().len(), 2, "the plan itself keeps both");

    let mut exec = MockExecutor::new();
    exec.push_script(SeqScript::per_token([10, 11, 12]));
    let sched = scheduler_from_plan(
        &plan,
        ArenaBytes {
            kv_per_rank: 2 * GIB,
            gdn_state_per_rank: 2 * GIB,
        },
        exec,
    );
    assert_eq!(
        sched.config().buckets.len(),
        1,
        "the eager bucket never reaches the scheduler"
    );

    sched.submit(request(&[1], 8)).expect("submit");
    sched.run_steps(3);
    let batches = sched.with_executor(|e| e.batches().to_vec());
    for b in &batches {
        assert!(b.bucket.captured);
        assert_eq!(b.bucket.batch, 4);
        assert_eq!(
            b.padded_slots, 3,
            "three padding slots rather than an uncaptured shape"
        );
    }
}

#[test]
fn the_plan_decides_the_ladder_not_the_scheduler() {
    // Same workload, two bucket sets: the shapes replayed are the plan's, not a built-in ladder.
    let full = scheduler(RandomAcceptanceExecutor::new(1, Acceptance::Fixed(2)));
    full.submit(request(&[1, 2], 12)).expect("submit");
    full.run_steps(50);
    let wide = full.with_executor(|e| e.batches().to_vec());
    assert!(wide.iter().any(|b| b.bucket.spec_width > 1));
    assert_eq!(full.config().buckets.len(), decode_buckets().len());

    let plan = plan_with_capture(&[(1, 1, 8192, true), (2, 1, 8192, true)], 2);
    let narrow_sched = scheduler_from_plan(
        &plan,
        ArenaBytes {
            kv_per_rank: 2 * GIB,
            gdn_state_per_rank: 2 * GIB,
        },
        RandomAcceptanceExecutor::new(1, Acceptance::Fixed(2)),
    );
    narrow_sched.submit(request(&[1, 2], 12)).expect("submit");
    narrow_sched.run_steps(50);
    let narrow = narrow_sched.with_executor(|e| e.batches().to_vec());
    assert!(
        narrow.iter().all(|b| b.bucket.spec_width == 1),
        "a plan with one width cannot produce a wider step"
    );
}
