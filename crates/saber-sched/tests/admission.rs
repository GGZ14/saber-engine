//! Admission: two allocators, one refusal, and the ~4.6k crossover between them.
//!
//! `ARCHITECTURE.md` § "What exploration corrected" item 3: 48 GDN layers x 48 value heads x
//! 128 x 128 x fp32 = **3.0 MiB/layer/sequence -> 144 MiB/sequence (72 MiB/rank)**, against
//! **16 KiB/token/rank** of fp8 KV. *"Crossover ~ 4.6k tokens; below it, state outweighs KV
//! ~10:1."* Everything below is that sentence as a test: the crossover is computed from the plan's
//! numbers, and with the **same** two arena sizes the allocator that refuses flips as the context
//! crosses it. The long-context case is the negative control the short-context case needs
//! (`CLAUDE.md` rule 8) — without it, "GDN refuses first" would just be a statement about arena
//! sizing.

mod common;

use common::{config_with, request, scheduler_with, wide_batch_buckets, GIB, MIB};
use saber_sched::{Acceptance, ArenaBytes, RandomAcceptanceExecutor, StepOutcome};
use saber_serve::engine::Engine;

/// Both arenas the same size, so the *only* thing that decides which one refuses is the shape of
/// the sequence.
fn equal_arenas() -> ArenaBytes {
    ArenaBytes {
        kv_per_rank: 2 * GIB,
        gdn_state_per_rank: 2 * GIB,
    }
}

#[test]
fn the_crossover_is_computed_from_the_plans_numbers() {
    let config = config_with(&wide_batch_buckets(), 2, equal_arenas());

    // 3.0 MiB/layer/slot x 48 GDN layers / tp 2 = 72 MiB per slot per rank.
    assert_eq!(config.gdn_bytes_per_slot, 72 * MIB);
    // 72 MiB / 16 KiB per token = 4608 tokens. The architecture's "~4.6k".
    assert_eq!(config.single_slot_crossover_tokens(), 4608);

    // A sequence reserves one slot per draft token plus one for the committed state, whether or
    // not it drafts that wide, so the crossover admission actually compares scales with the width.
    assert_eq!(config.gdn_slots_per_seq(), 2, "spec width 1 + 1");
    assert_eq!(config.gdn_bytes_per_seq(), 144 * MIB);
    assert_eq!(config.crossover_tokens(), 9216);

    // And the arena arithmetic the two tests below turn on.
    assert_eq!(config.kv.block_bytes(), 256 * 1024, "16 tokens x 16 KiB");
}

#[test]
fn below_the_crossover_the_state_arena_refuses_before_the_kv_arena() {
    // 512-token prompts: 578 reserved tokens is 37 KV blocks = 9.25 MiB, against 144 MiB of state.
    // State outweighs KV ~15:1 here, so the state arena is what runs out.
    let sched = scheduler_with(
        &wide_batch_buckets(),
        2,
        equal_arenas(),
        RandomAcceptanceExecutor::new(7, Acceptance::Fixed(1)),
    );
    let (kv_total, gdn_total) = sched.arena_free();
    assert_eq!((kv_total, gdn_total), (8192, 14), "2 GiB of each arena");

    let prompt = vec![5u32; 512];
    for _ in 0..20 {
        sched.submit(request(&prompt, 64)).expect("submit");
    }
    assert!(matches!(sched.step(), StepOutcome::Ran(_)));

    let stats = sched.stats();
    assert_eq!(stats.num_requests_running, 14, "one per state slot set");
    assert_eq!(stats.num_requests_waiting, 6);

    let counters = sched.admission_counters();
    assert_eq!(counters.admitted, 14);
    assert!(counters.refused_gdn >= 1, "{counters:?}");
    assert_eq!(counters.refused_kv, 0, "KV had room to spare: {counters:?}");
    assert_eq!(
        counters.kv_rollbacks, counters.refused_gdn,
        "a KV allocation that outlived a failed state allocation would leak the arena"
    );

    let (kv_free, gdn_free) = sched.arena_free();
    assert_eq!(gdn_free, 0, "the state arena is what is exhausted");
    assert_eq!(
        kv_free,
        kv_total - 14 * 37,
        "exactly the admitted sequences' blocks are held, and nothing else"
    );
    assert!(
        kv_free > 37 * 200,
        "KV could still hold hundreds more sequences at this context"
    );
}

#[test]
fn above_the_crossover_the_kv_arena_refuses_first() {
    // The negative control: same arenas, same bucket set, 16k prompts. 16450 reserved tokens is
    // 1029 blocks = 257 MiB of KV against the same 144 MiB of state, so the refusal flips.
    let sched = scheduler_with(
        &wide_batch_buckets(),
        2,
        equal_arenas(),
        RandomAcceptanceExecutor::new(7, Acceptance::Fixed(1)),
    );
    let prompt = vec![5u32; 16_384];
    for _ in 0..12 {
        sched.submit(request(&prompt, 64)).expect("submit");
    }
    assert!(matches!(sched.step(), StepOutcome::Ran(_)));

    assert_eq!(sched.stats().num_requests_running, 7);
    let counters = sched.admission_counters();
    assert_eq!(counters.admitted, 7);
    assert!(counters.refused_kv >= 1, "{counters:?}");
    assert_eq!(
        counters.refused_gdn, 0,
        "past the crossover the state arena is the one with room: {counters:?}"
    );
    assert_eq!(counters.kv_rollbacks, 0, "nothing to roll back");

    let (kv_free, gdn_free) = sched.arena_free();
    assert!(kv_free < 1029, "KV cannot fit another sequence");
    assert_eq!(gdn_free, 14 - 7, "half the state arena is still free");
}

#[test]
fn a_refused_request_waits_and_is_admitted_when_one_finishes() {
    // Continuous batching: the sequences that finish free both allocations in the same step, and
    // the next admission sees the space at the top of the next step.
    let sched = scheduler_with(
        &[(1, 1, 8192), (2, 1, 8192), (4, 1, 8192)],
        2,
        ArenaBytes {
            kv_per_rank: 2 * GIB,
            // Exactly two slot sets: 2 x 144 MiB.
            gdn_state_per_rank: 288 * MIB,
        },
        RandomAcceptanceExecutor::new(11, Acceptance::Fixed(1)),
    );
    assert_eq!(sched.arena_free().1, 2, "two state slot sets");

    let short = sched.submit(request(&[1, 2, 3], 4)).expect("submit");
    let _long = sched.submit(request(&[4, 5, 6], 64)).expect("submit");
    let third = sched.submit(request(&[7, 8, 9], 8)).expect("submit");

    sched.step();
    assert_eq!(sched.stats().num_requests_running, 2);
    assert_eq!(sched.stats().num_requests_waiting, 1, "the third waits");
    assert!(sched.admission_counters().refused_gdn >= 1);

    // Run until the short request is done; the third must then be running, not starved.
    sched.run_steps(6);
    assert!(
        sched.stats().num_requests_running >= 2,
        "the freed slot set was reused within the run"
    );
    assert_eq!(sched.stats().num_requests_waiting, 0);
    assert!(sched.counters().finished_length >= 1);
    drop((short, third));
}

#[test]
fn the_running_set_never_exceeds_the_widest_captured_batch() {
    // Plenty of arena, a deliberately small bucket ladder: admission stops at the widest captured
    // batch rather than replaying a shape that was never captured.
    let sched = scheduler_with(
        &[(1, 1, 8192), (2, 1, 8192)],
        2,
        ArenaBytes {
            kv_per_rank: 2 * GIB,
            gdn_state_per_rank: 2 * GIB,
        },
        RandomAcceptanceExecutor::new(3, Acceptance::Fixed(1)),
    );
    for _ in 0..5 {
        sched.submit(request(&[1, 2], 32)).expect("submit");
    }
    sched.step();
    assert_eq!(sched.stats().num_requests_running, 2, "max captured batch");
    assert_eq!(sched.stats().num_requests_waiting, 3);
    assert!(sched.admission_counters().refused_batch >= 1);
}

#[test]
fn over_committed_arenas_are_a_startup_error_not_an_oom_later() {
    use saber_mem::ledger::{Ledger, Pool};
    use saber_sched::{MockExecutor, Scheduler};

    let mut ledger = Ledger::new();
    ledger.declare(Pool::Vram(0), 8 * GIB).expect("rank 0");
    ledger.declare(Pool::Vram(1), 8 * GIB).expect("rank 1");
    ledger
        .reserve(Pool::Vram(0), "weights", 7 * GIB)
        .expect("weights");
    ledger
        .reserve(Pool::Vram(1), "weights", 7 * GIB)
        .expect("weights");

    let config = config_with(
        &wide_batch_buckets(),
        2,
        ArenaBytes {
            kv_per_rank: 2 * GIB,
            gdn_state_per_rank: 2 * GIB,
        },
    );
    let err = Scheduler::new(config, MockExecutor::new(), common::model_info(), ledger)
        .expect_err("4 GiB of arenas do not fit beside 7 GiB of weights in 8 GiB");
    let msg = err.to_string();
    assert!(msg.contains("over-commit"), "{msg}");
    assert!(msg.contains("kv_arena"), "{msg}");
}

#[test]
fn the_arenas_are_named_reservations_in_the_byte_ledger() {
    let sched = scheduler_with(
        &wide_batch_buckets(),
        2,
        equal_arenas(),
        RandomAcceptanceExecutor::new(1, Acceptance::Fixed(1)),
    );
    let snap = sched.ledger_snapshot();
    for rank in 0..2u32 {
        let pool = snap
            .pools
            .iter()
            .find(|p| p.label == format!("vram.rank{rank}"))
            .expect("rank pool");
        let names: Vec<&str> = pool.reservations.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains(&"kv_arena"), "{names:?}");
        assert!(names.contains(&"gdn_state_arena"), "{names:?}");
        assert_eq!(pool.used_bytes, 4 * GIB, "both arenas, on every rank");
    }
}

#[test]
fn a_kv_refusal_is_counted_once_per_request_and_the_gauges_read_the_reservations() {
    // LR-A: `refused_kv` ticks every step the head waits; the per-request counter and the
    // once-per-head log line tick once. The gauges read the arena and the running reservations.
    let sched = scheduler_with(
        &wide_batch_buckets(),
        2,
        equal_arenas(),
        RandomAcceptanceExecutor::new(7, Acceptance::Fixed(1)),
    );
    let prompt = vec![5u32; 16_384];
    for _ in 0..12 {
        sched.submit(request(&prompt, 64)).expect("submit");
    }
    for _ in 0..4 {
        assert!(matches!(sched.step(), StepOutcome::Ran(_)));
    }
    let counters = sched.admission_counters();
    assert_eq!(counters.admitted, 7);
    assert!(counters.refused_kv >= 4, "one per step: {counters:?}");
    let (snap, refused_requests) = sched.kv_admission();
    assert_eq!(refused_requests, 1, "the same head, refused at four steps");
    let (kv_free, _) = sched.arena_free();
    assert_eq!(snap.total_blocks, 8192);
    assert_eq!(snap.free_blocks, kv_free);
    assert_eq!(snap.reserved_tokens, 7 * (16_384 + 64));
    assert_eq!(snap.max_reserved_tokens, 16_384 + 64);
    assert!(snap.length_tokens >= 7 * 16_384 && snap.length_tokens < snap.reserved_tokens);

    let stats = sched.stats();
    let get = |name: &str| {
        stats
            .census
            .iter()
            .find(|s| s.def.name == name)
            .map(|s| s.value)
    };
    assert_eq!(get("saber:kv_blocks_total"), Some(8192));
    assert_eq!(get("saber:kv_blocks_free"), Some(u64::from(kv_free)));
    assert_eq!(get("saber:kv_reserved_tokens"), Some(7 * (16_384 + 64)));
    assert_eq!(get("saber:admission_refused_kv_requests_total"), Some(1));
}
