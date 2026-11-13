//! the form-c W8 promotions' plan law (host only). On the MACaQ-Hadamard graph, a Bc6-like set of
//! promotions (gdn.out_proj / attn.o_proj / mlp.down_proj) plans every promoted GEMM as ONE per-token W8A8
//! launch (`macaq_w8`) binding the producer's e4m3 rows, their per-token scale and the `<op>.w8` / `<op>.w8s`
//! records; its producer is the `*_had32_*_tiled0` twin (row-major) at EVERY bucket, prefill included; no
//! fused epilogue swallows a promoted GEMM (row 3 / arw stand down there, counted by the launch delta); the
//! non-promoted GEMMs are untouched. Bad sets are refused by name.

use std::collections::BTreeSet;

use saber_exec::layer::{BlockPlan, Collectives, PlanOptions, Route, RuntimeParams, Source};
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_model::graph::{macaq_w8_op, Bucket, Graph, KernelOp, ModelConfig};

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

/// A Bc6-shaped set: out on GDN layers, o on attention layers (every 4th is attention), down anywhere.
fn units(g: &Graph) -> BTreeSet<String> {
    let mut s = BTreeSet::new();
    for l in [0u32, 1, 2, 4, 5, 6] {
        s.insert(macaq_w8_op(l, "out").unwrap());
    }
    for l in [3u32, 7, 23, 63] {
        s.insert(macaq_w8_op(l, "o").unwrap());
    }
    for l in [3u32, 36, 54, 55, 63] {
        s.insert(macaq_w8_op(l, "down").unwrap());
    }
    assert!(
        s.iter().all(|o| g.ops.iter().any(|x| &x.name == o)),
        "{s:?}"
    );
    s
}

const BUCKETS: [(Bucket, bool); 5] = [
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
];

#[test]
fn every_promoted_gemm_is_one_per_token_w8a8_launch_on_a_row_major_twin() {
    for tp in [2u32, 1] {
        let had = Graph::qwen38_27b(ModelConfig::qwen38_27b(), tp)
            .unwrap()
            .with_macaq_had()
            .unwrap();
        let set = units(&had);
        let w8 = had.clone().with_macaq_w8(&set).expect("w8 graph");
        assert_eq!(w8.macaq_w8(), set);
        assert!(had.macaq_w8().is_empty());
        for (bucket, fuse) in BUCKETS {
            let (p0, p1) = (plan(&had, bucket, fuse), plan(&w8, bucket, fuse));
            let mut seen = BTreeSet::new();
            for (i, s) in p1.steps.iter().enumerate() {
                let Route::Kernel { descs, .. } = &s.route else {
                    continue;
                };
                if s.op.kernel == KernelOp::GemmW8A8 {
                    assert!(set.contains(&s.op.name), "{}", s.op.name);
                    assert_eq!(descs.len(), 1);
                    assert_eq!(descs[0].plugin, "macaq_w8", "{tp} {bucket:?}");
                    let w = |n: &str| {
                        s.bindings
                            .iter()
                            .find(|(b, _)| b == n)
                            .map(|(_, src)| src.clone())
                    };
                    assert!(
                        matches!(w("w"), Some(Source::Weight { ref key, layout: 7 }) if key == &format!("{}.w8", s.op.name))
                    );
                    assert!(
                        matches!(w("ws"), Some(Source::Weight { ref key, layout: 8 }) if key == &format!("{}.w8s", s.op.name))
                    );
                    assert!(matches!(w("a"), Some(Source::Arena { .. })));
                    assert!(matches!(w("as"), Some(Source::Arena { .. })));
                    seen.insert(s.op.name.clone());
                    // its producer: the step that writes `a` is a row-major had32 twin
                    let a = s.op.inputs[0].edge.clone();
                    let prod = p1.steps[..i]
                        .iter()
                        .rev()
                        .find(|p| p.op.outputs.iter().any(|e| e.name == a))
                        .expect("a producer step");
                    let Route::Kernel { descs: pd, .. } = &prod.route else {
                        panic!("producer route")
                    };
                    let k = &pd[0].kernel;
                    assert!(k.contains("_had32"), "{tp} {bucket:?} {k}");
                    assert!(
                        !k.contains("tiled1"),
                        "{tp} {bucket:?}: {} writes tiled ({k})",
                        prod.op.name
                    );
                } else {
                    for d in descs {
                        assert_ne!(d.plugin, "macaq_w8");
                    }
                }
            }
            // every promoted op appears as its own step (no fused epilogue took it)
            assert_eq!(seen, set, "{tp} {bucket:?}");
            // launches: a promoted GEMM that the W4 plan fused into row 3 (GEMM + reduce + norm in
            // one dispatch) now costs its unfused launches; nothing else moves
            let (l0, l1) = (p0.expected_launches(), p1.expected_launches());
            assert!(l1 >= l0, "{tp} {bucket:?}: {l0} -> {l1}");
            if !fuse || tp == 1 {
                assert_eq!(l0, l1, "{tp} {bucket:?}: unfused plans keep the count");
            }
        }
    }
}

#[test]
fn the_records_and_the_refusals() {
    for (tp, ks) in [(2u32, [3072u64, 8704]), (1, [6144, 17408])] {
        let had = Graph::qwen38_27b(ModelConfig::qwen38_27b(), tp)
            .unwrap()
            .with_macaq_had()
            .unwrap();
        let set = units(&had);
        let w8 = had.clone().with_macaq_w8(&set).unwrap();
        let recs = w8.device_final_records().unwrap();
        let mine: Vec<_> = recs
            .iter()
            .filter(|r| r.layout_id == 7 || r.layout_id == 8)
            .collect();
        assert_eq!(mine.len(), 2 * set.len());
        for r in &mine {
            let op = r.op.as_deref().unwrap();
            assert!(set.contains(op));
            if r.layout_id == 7 {
                assert_eq!(r.shape[0], 5120 / 16);
                assert!(ks.iter().any(|k| r.shape[1] == k * 16), "{:?}", r.shape);
            } else {
                assert_eq!(r.shape[0], 40);
                assert!(ks.iter().any(|k| r.shape[1] == k / 128), "{:?}", r.shape);
            }
        }
        for op in &set {
            assert!(
                !recs.iter().any(|r| r.name == format!("{op}.w")),
                "{op} keeps an MXFP4 record"
            );
        }
        assert_eq!(
            recs.len(),
            had.device_final_records().unwrap().len() - set.len()
        );
        // refusals by name
        let plain = Graph::qwen38_27b(ModelConfig::qwen38_27b(), tp).unwrap();
        let e = plain.with_macaq_w8(&set).unwrap_err().to_string();
        assert!(e.contains("macaq_had32"), "{e}");
        let bad: BTreeSet<String> = ["layer.0.gdn.in_proj_qkvz".to_owned()].into();
        assert!(had.clone().with_macaq_w8(&bad).is_err());
        let missing: BTreeSet<String> = ["layer.99.mlp.down_proj".to_owned()].into();
        assert!(had.clone().with_macaq_w8(&missing).is_err());
        assert!(w8.clone().with_macaq_w8(&set).is_err(), "twice is refused");
        // the empty set is the identity
        assert_eq!(
            had.clone()
                .with_macaq_w8(&BTreeSet::new())
                .unwrap()
                .macaq_w8(),
            BTreeSet::new()
        );
    }
}
