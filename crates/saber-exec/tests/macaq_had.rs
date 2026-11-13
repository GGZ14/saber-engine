//! the `--macaq-had` arm's plan law (host only). With the arm ON every target producer of a W4A8
//! GEMM's e4m3 input -- plain, tiled, the `_nq` epilogue and fusion-map row 3 -- is its `*_had32_*` twin
//! and binds `had_t`, with EXACTLY the launch count and arena of the arm-off plan (the transform rides the
//! existing producer: no new launch, no new buffer). With the arm OFF no descriptor names a twin.

use saber_exec::layer::{BlockPlan, Collectives, PlanOptions, Route, RuntimeParams, Source};
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_model::graph::{Bucket, Graph, ModelConfig};

fn plan<'g>(g: &'g Graph, bucket: Bucket, fuse: bool) -> BlockPlan<'g> {
    let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
    let collectives = if g.tp > 1 {
        Collectives::Real
    } else {
        Collectives::Identity
    };
    let o = PlanOptions::BODY
        .rank(0)
        .with_collectives(collectives)
        .with_batch_invariant_attn();
    let o = if fuse {
        o.with_decode_fusion()
    } else {
        o.without_fusion()
    };
    BlockPlan::with_options(
        g,
        bucket,
        &layers,
        RuntimeParams {
            max_ctx: 4096,
            gdn_slots: 1 + 8 * 8,
        },
        &LaunchRules::prod_mxfp4().expect("rules"),
        &SaberRules::prod_saber().expect("saber rules"),
        o,
    )
    .expect("the body plans")
}

fn kernels(p: &BlockPlan<'_>) -> Vec<(String, bool)> {
    p.steps
        .iter()
        .flat_map(|s| {
            let had = s
                .bindings
                .iter()
                .any(|(n, src)| n == "had_t" && matches!(src, Source::Weight { .. }));
            match &s.route {
                Route::Kernel { descs, .. } => descs
                    .iter()
                    .map(|d| (d.kernel.clone(), had))
                    .collect::<Vec<_>>(),
                _ => vec![],
            }
        })
        .collect()
}

fn is_producer(k: &str) -> bool {
    k.starts_with("add_rms_quant_")
        || k.starts_with("silu_mul_quant_")
        || k.starts_with("gdn_norm_quant_")
        || k.starts_with("attn_gate_quant_")
        || k.contains("_exact_nq")
        || k.starts_with("mxfp4_gemm_decode_arnq")
}

#[test]
fn the_arm_twins_every_producer_and_keeps_the_launch_law() {
    for tp in [2u32, 1] {
        let off = Graph::qwen38_27b(ModelConfig::qwen38_27b(), tp).expect("graph");
        let on = off.clone().with_macaq_had().expect("macaq graph");
        assert!(on.macaq_had() && !off.macaq_had());
        for (bucket, fuse) in [
            (
                Bucket::Decode {
                    batch: 1,
                    verify_width: 8,
                },
                true,
            ),
            (
                Bucket::Decode {
                    batch: 1,
                    verify_width: 8,
                },
                false,
            ),
            (
                Bucket::Decode {
                    batch: 8,
                    verify_width: 8,
                },
                true,
            ),
            (
                Bucket::Prefill {
                    tokens: 2048,
                    seqs: 1,
                },
                false,
            ),
            (
                Bucket::Prefill {
                    tokens: 96,
                    seqs: 1,
                },
                false,
            ),
        ] {
            let (p0, p1) = (plan(&off, bucket, fuse), plan(&on, bucket, fuse));
            assert_eq!(
                p0.expected_launches(),
                p1.expected_launches(),
                "{tp} {bucket:?}"
            );
            assert_eq!(
                p0.arena_bytes, p1.arena_bytes,
                "{tp} {bucket:?}: no new buffer"
            );
            let (k0, k1) = (kernels(&p0), kernels(&p1));
            assert_eq!(k0.len(), k1.len());
            let mut twins = 0;
            for ((a, _), (b, had)) in k0.iter().zip(&k1) {
                assert!(!a.contains("_had32"), "arm off names a twin: {a}");
                if is_producer(a) {
                    assert!(
                        b.contains("_had32") && *had,
                        "{tp} {bucket:?}: {a} -> {b} had_t {had}"
                    );
                    twins += 1;
                } else {
                    assert_eq!(a, b);
                }
            }
            assert!(twins >= 128, "{tp} {bucket:?}: {twins} twins");
        }
    }
}

#[test]
fn the_records_are_four_a_layer_and_shard_by_the_gemm_k() {
    for (tp, dim0) in [(2u32, [3072u64, 8704]), (1, [6144, 17408])] {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), tp)
            .unwrap()
            .with_macaq_had()
            .unwrap();
        let recs: Vec<_> = g
            .device_final_records()
            .unwrap()
            .into_iter()
            .filter(|r| r.name.contains(".macaq_had32."))
            .collect();
        assert_eq!(recs.len(), 256);
        for r in &recs {
            let k = r.shape[0];
            if r.name.ends_with(".in_t") || r.name.ends_with(".gu_t") {
                assert_eq!(k, 5120, "{}", r.name);
            } else {
                assert!(dim0.contains(&k), "{} {k}", r.name);
            }
        }
        assert!(g.with_macaq_had().is_err(), "twice is refused");
    }
}
