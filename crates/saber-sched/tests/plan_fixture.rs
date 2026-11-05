//! What the scheduler's fixture plan actually contains.
//!
//! These tests exist because the fixture broke silently once already: `saber-plan` grew the reference's
//! attention, GDN and all-reduce rows (workstream H2) and the fixture manifest still declared one
//! plugin, so every integration test in this crate died at `bind` with `UnknownPlugin`. The fix is
//! to derive the plugin list and the autotune records from the registry and from
//! `Resolved::requirements()`; these assertions are what make that derivation visible, so the next
//! divergence fails here with a name rather than 38 tests failing with a stack trace.
//!
//! They also re-check the two numbers the scheduler reads out of the plan rather than restating:
//! the GDN state bytes per slot, and the KV/state crossover computed from them.

mod common;

use common::{config_with, decode_buckets, plan_with, reference_plugins, wide_batch_buckets, MIB};
use saber_plan::{reference, ExecutionPlan};
use saber_sched::{ArenaBytes, KvGeometry, SchedulerConfig};

fn op_names(plan: &ExecutionPlan, batch: u32, spec_width: u32) -> Vec<String> {
    plan.buckets()
        .iter()
        .find(|b| b.bucket.batch == batch && b.bucket.spec_width == spec_width)
        .unwrap_or_else(|| panic!("no bucket batch {batch} width {spec_width}"))
        .ops
        .ops()
        .iter()
        .map(|o| o.op_name.clone())
        .collect()
}

#[test]
fn the_fixture_declares_every_plugin_the_registry_names() {
    let declared = reference_plugins();
    assert!(declared.iter().any(|p| p == "mxfp4"), "{declared:?}");
    assert!(declared.iter().any(|p| p == "saber"), "{declared:?}");

    // And nothing in the registry is left undeclared -- the UnknownPlugin bind error, pre-empted.
    let registry = reference::reference_registry();
    for imp in registry.implementations() {
        assert!(
            declared.contains(&imp.plugin),
            "implementation {} ships from undeclared plugin {}",
            imp.name,
            imp.plugin
        );
    }
}

#[test]
fn the_bound_plan_carries_the_whole_step_not_just_the_gemms() {
    let plan = plan_with(&decode_buckets(), 2);
    let ops = op_names(&plan, 8, 5);

    for expected in [
        "layer.gdn.in_proj_qkvz",
        "layer.mlp.down_proj",
        "lm_head",
        "layer.attn.paged_decode",
        // The all-reduce row is banded by message size (`.exact` below 128 KiB, `.wht` above);
        // a bucket carries exactly one of them.
        "layer.allreduce.",
    ] {
        assert!(
            ops.iter().any(|o| o == expected || o.starts_with(expected)),
            "{expected} missing from the bound bucket: {ops:?}"
        );
    }
}

#[test]
fn the_gdn_route_is_split_by_bucket_batch_and_both_halves_bind() {
    // TP=2: 24 value heads against the reference's fused-item limit, so batch 1 takes the fused kernel
    // (one launch) and batch 2 and up take the conv + recurrent pair (two launches). Which one
    // runs is a property of the bucket, and the fixture has to have a record for both.
    let plan = plan_with(&decode_buckets(), 2);

    let b1 = op_names(&plan, 1, 5);
    assert!(
        b1.iter().any(|o| o == "layer.gdn.decode_step.fused"),
        "batch 1 should bind the fused route: {b1:?}"
    );
    assert!(
        !b1.iter().any(|o| o == "layer.gdn.decode_step.pair"),
        "batch 1 must not bind the pair route: {b1:?}"
    );

    for batch in [2u32, 4, 8] {
        let ops = op_names(&plan, batch, 5);
        assert!(
            ops.iter().any(|o| o == "layer.gdn.decode_step.pair"),
            "batch {batch} should bind the pair route: {ops:?}"
        );
        assert!(
            !ops.iter().any(|o| o == "layer.gdn.decode_step.fused"),
            "batch {batch} must not bind the fused route: {ops:?}"
        );
    }

    // The launch budget follows: the pair is one more launch per GDN layer than the fused route
    // (48 layers), which is the number CLAUDE.md rule 7 asks to be declared per bucket.
    let launches = |batch: u32| {
        plan.buckets()
            .iter()
            .find(|b| {
                b.bucket.batch == batch && b.bucket.spec_width == 5 && b.bucket.ctx_band == 8192
            })
            .expect("bucket")
            .n_launches
    };
    assert_eq!(
        launches(2) - launches(1),
        48,
        "the pair route costs one extra launch per GDN layer"
    );
}

#[test]
fn a_kv_page_size_that_contradicts_the_bound_attention_route_is_refused() {
    // The plan binds paged attention at block_size 16; a scheduler handing that kernel 32-token
    // block tables would index at the wrong stride, so the configuration is refused at
    // construction rather than serving wrong KV.
    let plan = plan_with(&decode_buckets(), 2);
    let err = SchedulerConfig::from_plan(
        &plan,
        KvGeometry {
            block_tokens: 32,
            bytes_per_token_per_rank: 16 * 1024,
        },
        ArenaBytes {
            kv_per_rank: 2 * 1024 * MIB,
            gdn_state_per_rank: 2 * 1024 * MIB,
        },
        262_144,
    )
    .expect_err("32 != the plan's 16");
    let msg = err.to_string();
    assert!(msg.contains("block_size=16"), "{msg}");
    assert!(msg.contains("layer.attn.paged_decode"), "{msg}");

    // And the matching geometry is accepted, so the check is not simply always-on.
    SchedulerConfig::from_plan(
        &plan,
        KvGeometry::PROD_FP8,
        ArenaBytes {
            kv_per_rank: 2 * 1024 * MIB,
            gdn_state_per_rank: 2 * 1024 * MIB,
        },
        262_144,
    )
    .expect("prod geometry matches the plan");
}

#[test]
fn the_state_budget_and_the_crossover_still_derive_from_the_plan() {
    let arenas = ArenaBytes {
        kv_per_rank: 2 * 1024 * MIB,
        gdn_state_per_rank: 2 * 1024 * MIB,
    };
    let config = config_with(&wide_batch_buckets(), 2, arenas);

    // Derived, not restated: `resident_bytes / slots` from the planner's own GDN state budget.
    assert_eq!(
        config.gdn_bytes_per_slot,
        72 * MIB,
        "3.0 MiB x 48 layers / tp 2"
    );
    assert_eq!(
        config.single_slot_crossover_tokens(),
        4608,
        "the architecture's ~4.6k"
    );
    assert_eq!(config.crossover_tokens(), 9216, "2 slots at spec width 1");

    // The same derivation at TP=1, where the state is not sharded: one slot is the whole
    // 144 MiB/sequence, so against the same per-rank KV rate the crossover doubles.
    let tp1 = SchedulerConfig::from_plan(
        &plan_with(&[(1, 1, 8192), (2, 1, 8192)], 1),
        KvGeometry::PROD_FP8,
        arenas,
        262_144,
    )
    .expect("tp1 config");
    assert_eq!(tp1.gdn_bytes_per_slot, 144 * MIB);
    assert_eq!(tp1.single_slot_crossover_tokens(), 9216);
}
