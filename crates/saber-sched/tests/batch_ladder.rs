//! **B1: the batch ladder through the whole scheduler.** With a decode bucket at every batch
//! `1..=8` the step of `n` live sequences replays batch `n` and runs no padding slot; the
//! power-of-two set (`1,2,4,8`, the control) pads 3 -> 4 and 5 / 6 / 7 -> 8 exactly as the served
//! engine does today.

mod common;

use common::{request, scheduler_with, GIB};
use saber_sched::decode_law::{DECODE_BATCH_LADDER, PRE_B3_DECODE_BATCHES};
use saber_sched::{ArenaBytes, MockExecutor, SeqScript};
use saber_serve::engine::Engine;

fn buckets(batches: &[u32]) -> Vec<(u32, u32, u32)> {
    let mut v = Vec::new();
    for b in batches {
        for w in [1u32, 3, 5] {
            v.push((*b, w, 8192));
        }
    }
    v
}

/// `(bucket batch, padded slots)` of the widest decode step `n` concurrent requests produce.
fn widest_step(batches: &[u32], n: usize) -> (u32, u32, u32) {
    let mut exec = MockExecutor::new();
    for _ in 0..n {
        exec.push_script(SeqScript::per_token([10, 11, 12, 13, 14, 15]));
    }
    let sched = scheduler_with(
        &buckets(batches),
        2,
        ArenaBytes {
            kv_per_rank: 8 * GIB,
            gdn_state_per_rank: 8 * GIB,
        },
        exec,
    );
    for _ in 0..n {
        sched.submit(request(&[1, 2, 3], 6)).expect("submit");
    }
    sched.run_steps(40);
    let batches = sched.with_executor(|e| e.batches().to_vec());
    let widest = batches
        .iter()
        .filter(|b| b.sequences == n as u32)
        .max_by_key(|b| b.bucket.batch)
        .unwrap_or_else(|| panic!("{n} requests never ran together"));
    for b in &batches {
        assert_eq!(b.padded_slots, b.bucket.batch - b.sequences);
    }
    (widest.bucket.batch, widest.padded_slots, widest.sequences)
}

#[test]
fn the_ladder_runs_every_live_count_unpadded() {
    for n in 1..=8usize {
        let (batch, padded, live) = widest_step(DECODE_BATCH_LADDER, n);
        assert_eq!((batch, padded, live), (n as u32, 0, n as u32), "live {n}");
    }
}

/// The control: the power-of-two set pads the counts it does not hold.
#[test]
fn the_power_of_two_control_pads_three_five_six_and_seven() {
    let want = [
        (1, 0),
        (2, 0),
        (4, 1),
        (4, 0),
        (8, 3),
        (8, 2),
        (8, 1),
        (8, 0),
    ];
    for n in 1..=8usize {
        let (batch, padded, _) = widest_step(PRE_B3_DECODE_BATCHES, n);
        assert_eq!((batch, padded), want[n - 1], "live {n}");
    }
}
