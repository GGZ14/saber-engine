//! **RAGGED per-slot verify widths through the whole scheduler** (workstream C2).
//!
//! The unit tests in `src/width.rs` pin the per-slot rule against prod's `_ref_cap_spec_width`
//! and the ones in `src/buckets.rs` pin the band choice. These drive the scheduler over a plan that
//! captured row bands and check what the executor was actually handed: every slot at its own
//! width, the step in the smallest band that holds their sum (or the exact rectangle when every
//! slot asked one width), no width floor and no absorbed cap — and that `--rectangular-widths`
//! (ragged off) schedules byte for byte what a plan with no bands at all schedules.

mod common;

use common::{boot_ledger, config_with, decode_buckets, model_info, request, GIB};
use saber_plan::Bucket;
use saber_sched::{
    band_rows_for, Acceptance, ArenaBytes, RaggedCeiling, RandomAcceptanceExecutor, Scheduler,
    SchedulerConfig, WidthRule, DEFAULT_ROW_BANDS,
};
use saber_serve::engine::Engine;

fn arenas() -> ArenaBytes {
    ArenaBytes {
        kv_per_rank: 8 * GIB,
        gdn_state_per_rank: 8 * GIB,
    }
}

/// The fixture's plan (widths 1, 3, 5; batches 1, 2, 4, 8; two context bands) under the table
/// `1:5,4:3`, with or without the row bands a ragged engine captures (batch 8 x max width 5), and
/// with the ragged schedule on or off.
fn config(bands: bool, ragged: Option<RaggedCeiling>) -> SchedulerConfig {
    let mut c = config_with(&decode_buckets(), 2, arenas());
    c.width = c
        .width
        .clone()
        .with_rule(WidthRule::parse("batch=1:5,4:3").expect("table"))
        .with_ragged(ragged);
    if bands {
        for ctx in [8192u32, 65536] {
            for rows in band_rows_for(DEFAULT_ROW_BANDS, 8, 5) {
                c.buckets.push(Bucket::band(8, 5, rows, ctx, true));
            }
        }
    }
    c
}

fn run(
    c: SchedulerConfig,
    acceptance: Acceptance,
    requests: usize,
    steps: usize,
) -> Scheduler<RandomAcceptanceExecutor> {
    let sched = Scheduler::new(
        c,
        RandomAcceptanceExecutor::new(0xC2, acceptance),
        model_info(),
        boot_ledger(),
    )
    .expect("the arenas fit");
    // Different prompts, so each slot's seeded acceptance stream (and so its EMA) is its own.
    for i in 0..requests {
        sched
            .submit(request(&[7 + i as u32; 12], 40 + i))
            .expect("submit");
    }
    sched.run_steps(steps);
    sched
}

#[test]
fn a_ragged_batch_runs_each_slot_at_its_own_width_in_the_smallest_band() {
    // Every draft token has an even chance: the EMAs spread, the cap binds at 3+ running, and
    // the slots ask DIFFERENT widths. The BLOCK ceiling (5 rows = 4 drafts), because the
    // fixture's table cell at 4+ running is 3 rows = 2 drafts, which prod's cap (MIN 2) can never
    // narrow — under it every settled slot asks 3 and the steps are uniform.
    let c = config(true, Some(RaggedCeiling::Block));
    let buckets = c.buckets.clone();
    let sched = run(c, Acceptance::Bernoulli { p: 0.5 }, 8, 200);
    let batches = sched.with_executor(|e| e.batches().to_vec());
    let n = sched.counters();
    assert!(n.ragged_steps > 20, "{n:?}");
    let mut bands = 0;
    let mut mixed = 0;
    for b in batches
        .iter()
        .filter(|b| b.phase == saber_plan::Phase::Decode)
    {
        let rows: u32 = b.verify_widths.iter().map(|w| (*w).max(1)).sum();
        let running = b.sequences;
        // Every slot verifies 1 (first step) or the cap's `drafts + 1` (>= 3), never above the
        // plan's max width.
        let ceiling = 5;
        for w in &b.verify_widths {
            assert!(
                *w >= 1 && *w <= ceiling,
                "slot width {w} at ceiling {ceiling}: {b:?}"
            );
            assert!(*w != 2, "prod's cap keeps at least 2 drafts: {b:?}");
        }
        if b.verify_widths.windows(2).any(|p| p[0] != p[1]) {
            mixed += 1;
        }
        if b.bucket.is_band() {
            bands += 1;
            // The SMALLEST captured band that holds the step.
            let smallest = buckets
                .iter()
                .filter(|x| x.is_band() && x.ctx_band == b.bucket.ctx_band && x.tokens >= rows)
                .map(|x| x.tokens)
                .min()
                .expect("a band holds it");
            assert_eq!(b.bucket.tokens, smallest, "{b:?}");
            assert_eq!(b.bucket.batch, 8);
            assert_eq!(b.padded_slots, 8 - running);
        } else {
            // A rectangle only when every slot asked the SAME width, and exactly that width.
            assert!(
                b.verify_widths.iter().all(|w| *w == b.verify_widths[0]),
                "a mixed step in a rectangle: {b:?}"
            );
            assert_eq!(b.bucket.spec_width, b.verify_widths[0].max(1), "{b:?}");
        }
        assert!(rows <= b.bucket.m(), "{b:?}");
    }
    assert!(bands > 0 && mixed > 0, "bands {bands}, mixed steps {mixed}");
    assert_eq!(n.slot_caps_absorbed, 0, "nothing is absorbed in a band");
    assert_eq!(
        n.width_floor_raised, 0,
        "no floor under the ragged schedule"
    );
    assert!(n.ragged_slots_narrowed > 0, "the cap bound: {n:?}");
    assert!(n.ragged_live_rows <= n.ragged_step_rows);
    assert_eq!(n.no_bucket_steps, 0);
    assert_eq!(n.executor_protocol_errors, 0);
}

/// A single stream never meets the cap (`MIN_BATCH` 3): every step asks the table's cell, and a
/// uniform step runs in the exact rectangle rather than a batch-8 band.
#[test]
fn a_single_stream_runs_in_its_rectangle_under_the_ragged_schedule() {
    let sched = run(
        config(true, Some(RaggedCeiling::Table)),
        Acceptance::Fixed(1),
        1,
        30,
    );
    let batches = sched.with_executor(|e| e.batches().to_vec());
    for b in batches
        .iter()
        .filter(|b| b.phase == saber_plan::Phase::Decode)
    {
        assert!(!b.bucket.is_band(), "{b:?}");
        assert_eq!(b.bucket.batch, 1);
    }
    let n = sched.counters();
    assert_eq!(n.ragged_rect_steps, n.ragged_steps);
    assert!(n.ragged_steps > 0);
}

/// The CEILING SOURCE is the owner's A/B: `block` lets a settled slot rise to the plan's max width
/// at 4+ running, where `table` holds it at the table's 3.
#[test]
fn the_block_ceiling_lets_a_slot_verify_the_plans_max_width() {
    let full = Acceptance::Fixed(4);
    let table = run(config(true, Some(RaggedCeiling::Table)), full, 6, 80);
    let block = run(config(true, Some(RaggedCeiling::Block)), full, 6, 80);
    let widest = |s: &Scheduler<RandomAcceptanceExecutor>| {
        s.with_executor(|e| e.batches().to_vec())
            .iter()
            .filter(|b| b.sequences >= 4)
            .flat_map(|b| b.verify_widths.clone())
            .max()
            .unwrap_or(0)
    };
    assert_eq!(widest(&table), 3);
    assert_eq!(widest(&block), 5);
}

/// **C2 host test (g): `--rectangular-widths` IS today's schedule.** With the ragged schedule off
/// a plan that captured bands replays exactly the batches a plan without bands does — same
/// buckets, same widths, same order — so the bands leak into nothing.
#[test]
fn rectangular_widths_reproduce_the_schedule_of_a_plan_without_bands() {
    for acceptance in [Acceptance::Bernoulli { p: 0.6 }, Acceptance::Fixed(2)] {
        let with_bands = run(config(true, None), acceptance, 8, 150);
        let without = run(config(false, None), acceptance, 8, 150);
        let a = with_bands.with_executor(|e| e.batches().to_vec());
        let b = without.with_executor(|e| e.batches().to_vec());
        assert_eq!(a, b);
        assert!(a.iter().all(|x| !x.bucket.is_band()));
        assert_eq!(with_bands.counters().ragged_steps, 0);
        // NEGATIVE CONTROL: the same plan WITH the ragged schedule on is a different schedule.
        let ragged = run(config(true, Some(RaggedCeiling::Table)), acceptance, 8, 150);
        assert_ne!(ragged.with_executor(|e| e.batches().to_vec()), a);
    }
}
