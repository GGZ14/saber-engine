//! **P27-16: the 27B's decode launch census by family** (TP=2, rank 0), on the host with no card.
//!
//! The served step's target body at `b1s8` and `b8s8` from the plan the engine builds
//! (`PlanOptions::BODY` + the rank + real collectives + batch-invariant attention), under the
//! served set (`--decode-fusion all`, rows 1-3 + 7) and under ws-h42's opt-in set (rows 4 + 5 on
//! top), by FAMILY (the op without its layer prefix) and kernel. `--nocapture` prints the table;
//! the laws are that each census sums to the plan's own budget and that the served set is the
//! 483-dispatch body (494 engine dispatches with the 11 head / selection / tap launches).

use std::collections::BTreeMap;

use saber_exec::layer::{BlockPlan, Collectives, PlanOptions, Route, RuntimeParams};
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_model::graph::{Bucket, Graph, ModelConfig};

fn plan(
    g: &Graph,
    batch: u32,
    width: u32,
    f: impl FnOnce(PlanOptions) -> PlanOptions,
) -> BlockPlan<'_> {
    let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
    let o = f(PlanOptions::BODY
        .rank(0)
        .with_collectives(Collectives::Real)
        .with_batch_invariant_attn());
    BlockPlan::with_options(
        g,
        Bucket::Decode {
            batch,
            verify_width: width,
        },
        &layers,
        RuntimeParams {
            max_ctx: 4096,
            gdn_slots: 1 + batch * width,
        },
        &LaunchRules::prod_mxfp4().expect("rules"),
        &SaberRules::prod_saber().expect("saber rules"),
        o,
    )
    .expect("the decode body plans")
}

fn census(p: &BlockPlan<'_>) -> BTreeMap<(String, String), u32> {
    let mut m = BTreeMap::new();
    for s in &p.steps {
        let name = s.name();
        let fam = name
            .strip_prefix("layer.")
            .and_then(|r| r.split_once('.'))
            .map_or(name.to_owned(), |(_, r)| r.to_owned());
        if let Route::Kernel { descs, .. } = &s.route {
            // An op's descriptors carry the op's whole count (`descs[0].n_launches`); split it
            // evenly over its kernels (the attention's decode + combine = 1 + 1).
            let each = descs[0].n_launches / u32::try_from(descs.len()).expect("a few descs");
            for d in descs {
                *m.entry((fam.clone(), d.kernel.clone())).or_insert(0) += each;
            }
        }
    }
    m
}

#[test]
fn p27_16_target_census_by_family() {
    let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("the TP=2 graph");
    let mut served = Vec::new();
    for (batch, width) in [(1u32, 8u32), (8, 8)] {
        for (tag, f) in [
            (
                "all",
                PlanOptions::with_decode_fusion as fn(PlanOptions) -> PlanOptions,
            ),
            ("h42", PlanOptions::with_h42_fusion),
        ] {
            let p = plan(&g, batch, width, f);
            let c = census(&p);
            let total: u32 = c.values().sum();
            println!(
                "P27-16 target b{batch}s{width} {tag}: {total} body launches (smq {}, gdn-nq {})",
                p.fused_gemm_smq, p.fused_gdn_nq
            );
            for ((fam, k), n) in &c {
                println!("  {fam:<28} {k:<64} {n:>4}");
            }
            assert_eq!(
                total,
                p.expected_launches(),
                "the census is the plan's budget"
            );
            if tag == "all" {
                served.push(total);
            }
        }
    }
    assert_eq!(served[0], 483, "the served b1s8 body");
}
