//! **P27-W2-5: the GDN replay arm's PLAN law, on the host** (no card).
//!
//! Under `--engine-gdn-replay on` the engine's graph routes every decode bucket to the PAIR and
//! `saber_engine::gdn_replay::rewrite_plan` swaps every recurrent update for the replay form at the
//! plan's own VS and scalars. The laws: every GDN layer is rewritten (48 of 48), the replay symbol
//! carries the shipped symbol's VS (b1 vs2, b8 vs1 at TP=2), `naccept` is gone and `rp` / `stash` /
//! `err` are bound, the launch budget is unchanged; and the NEGATIVE CONTROLS refuse by name: the
//! fused update (b1 without pair-always) and the fp32 state (no replay form).

use saber_engine::gdn_replay::{rewrite_plan, stash_key, ERR_KEY};
use saber_exec::layer::{BlockPlan, Collectives, PlanOptions, Route, RuntimeParams, Source};
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_model::graph::{Bucket, Graph, ModelConfig};

fn graph(f16: bool, pair: bool) -> Graph {
    let g = Graph::qwen38_27b_at_width(ModelConfig::qwen38_27b(), 2, 8).expect("graph");
    let g = if f16 {
        g.with_gdn_state_f16().expect("f16")
    } else {
        g
    };
    if pair {
        g.with_gdn_pair_always()
    } else {
        g
    }
}

fn plan(g: &Graph, batch: u32) -> BlockPlan<'_> {
    let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
    BlockPlan::with_options(
        g,
        Bucket::Decode {
            batch,
            verify_width: 8,
        },
        &layers,
        RuntimeParams {
            max_ctx: 4096,
            gdn_slots: 1 + batch * 8,
        },
        &LaunchRules::prod_mxfp4().expect("rules"),
        &SaberRules::prod_saber().expect("saber rules"),
        PlanOptions::BODY
            .rank(0)
            .with_collectives(Collectives::Real)
            .with_batch_invariant_attn()
            .with_decode_fusion(),
    )
    .expect("the decode body plans")
}

fn recurrent_kernels(p: &BlockPlan<'_>) -> Vec<String> {
    p.steps
        .iter()
        .filter(|s| s.name().ends_with(".gdn.recurrent_update"))
        .filter_map(|s| match &s.route {
            Route::Kernel { descs, .. } => Some(descs[0].kernel.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn p27w2_every_decode_bucket_rewrites_all_48_gdn_layers_at_the_plans_own_vs() {
    let g = graph(true, true);
    for (batch, vs) in [(1u32, 2u32), (2, 2), (4, 1), (8, 1)] {
        let mut p = plan(&g, batch);
        let budget = p.expected_launches();
        let before = recurrent_kernels(&p);
        assert_eq!(before.len(), 48, "b{batch}: the pair at every GDN layer");
        assert!(before
            .iter()
            .all(|k| k == &format!("saber_gdn_recurrent_update_f16s_k128_v128_ab1_norm0_vs{vs}")));
        let layers = rewrite_plan(&mut p, 8).expect("the rewrite");
        assert_eq!(layers.len(), 48);
        assert_eq!(
            p.expected_launches(),
            budget,
            "b{batch}: one launch for one launch"
        );
        for s in p
            .steps
            .iter()
            .filter(|s| s.name().ends_with(".gdn.recurrent_update"))
        {
            let Route::Kernel { descs, .. } = &s.route else {
                panic!("a kernel route")
            };
            assert_eq!(
                descs[0].kernel,
                format!("p27w2_gdn_replay_f16s_k128_v128_ab1_vs{vs}")
            );
            assert!(
                s.source("naccept").is_none(),
                "naccept is the snapshot law's"
            );
            assert!(matches!(s.source("rp"), Some(Source::Index(k)) if k.ends_with(".rp")));
            let saber_model::graph::Site::Layer(l) = s.op.site else {
                panic!("a layer op")
            };
            assert!(
                matches!(s.source("stash"), Some(Source::State { key, .. }) if *key == stash_key(l))
            );
            assert!(matches!(s.source("err"), Some(Source::State { key, .. }) if key == ERR_KEY));
            // every buffer the descriptor names is bound
            for slot in &descs[0].args {
                if let saber_kernels::rules::ArgValue::Buffer { buffer } = &slot.value {
                    assert!(s.source(buffer).is_some(), "{buffer} unbound");
                }
            }
        }
    }
}

#[test]
fn p27w2_control_the_fused_update_and_the_fp32_state_are_refused_by_name() {
    // b1 at TP=2 runs the FUSED update without pair-always: no replay form -> refused
    let g = graph(true, false);
    let mut p = plan(&g, 1);
    let e = rewrite_plan(&mut p, 8).expect_err("the fused update has no replay form");
    assert!(e.contains("gdn_fused_update"), "{e}");
    // the fp32 state's recurrent symbol is not the replay form's source -> refused
    let g = graph(false, true);
    let mut p = plan(&g, 8);
    let e = rewrite_plan(&mut p, 8).expect_err("fp32 state");
    assert!(e.contains("not the f16"), "{e}");
}
