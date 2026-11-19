//! The W4A8-iu8 pack's plan law (host only). On the MACaQ-Hadamard graph, `with_macaq_iu8` with a
//! Bc3-shaped W8 set plans EVERY decoder linear as ONE iu8 launch (`macaq_iu8`: per-row int4, or per-row int8
//! at the W8 ops) binding the A-tiled int8 edge, its per-token scale and the `<op>.wi4|wi8` (layout 17 / 18)
//! and `.wi4s|wi8s` (layout 19) records; every producer of such an edge is an int8 twin (`*_hadi8_*`, tiled at
//! EVERY bucket, the `_nq` epilogue's twin where the reduce folds the norm); no MXFP4 GEMM and no e4m3 producer
//! remains in the body; unfused plans keep the had32 plan's launch count exactly; TP=2 and TP=1; the ab
//! (48 / 96 rows) shapes route. Bad graphs are refused by name.

use std::collections::BTreeSet;

use saber_exec::layer::{BlockPlan, Collectives, PlanOptions, Route, RuntimeParams, Source};
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_model::graph::{macaq_w8_op, Bucket, Graph, KernelOp, Layout, ModelConfig};

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

/// A Bc3-shaped W8 set (19 ops: out on GDN layers, o on attention layers, down anywhere).
fn bc3(g: &Graph) -> BTreeSet<String> {
    let mut s = BTreeSet::new();
    for l in [0u32, 1, 2, 4, 5, 6, 8] {
        s.insert(macaq_w8_op(l, "out").unwrap());
    }
    for l in [3u32, 7, 23, 63] {
        s.insert(macaq_w8_op(l, "o").unwrap());
    }
    for l in [3u32, 36, 54, 55, 60, 61, 62, 63] {
        s.insert(macaq_w8_op(l, "down").unwrap());
    }
    assert_eq!(s.len(), 19);
    assert!(
        s.iter().all(|o| g.ops.iter().any(|x| &x.name == o)),
        "{s:?}"
    );
    s
}

const BUCKETS: [(Bucket, bool); 6] = [
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
        Bucket::Decode {
            batch: 1,
            verify_width: 1,
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

const I8_PLUGINS: [&str; 3] = ["mxfp4_hadi8", "attn_glue_hadi8", "saber_ar_hadi8"];

#[test]
fn every_linear_is_one_iu8_launch_on_an_int8_twin_and_unfused_counts_hold() {
    for tp in [2u32, 1] {
        let had = Graph::qwen38_27b(ModelConfig::qwen38_27b(), tp)
            .unwrap()
            .with_macaq_had()
            .unwrap();
        let w8 = bc3(&had);
        let w4a8: BTreeSet<String> = had
            .ops
            .iter()
            .filter(|o| o.kernel == KernelOp::GemmW4A8 && o.name.starts_with("layer."))
            .map(|o| o.name.clone())
            .collect();
        let iu8 = had.clone().with_macaq_iu8(&w8).expect("iu8 graph");
        assert!(
            iu8.macaq_iu8() == w4a8,
            "every W4A8 decoder linear converts"
        );
        for (bucket, fuse) in BUCKETS {
            let (p0, p1) = (plan(&had, bucket, fuse), plan(&iu8, bucket, fuse));
            let mut seen = BTreeSet::new();
            let mut nq = 0;
            for (i, s) in p1.steps.iter().enumerate() {
                // a pair's second half is folded into the first's `macaq_iu8f` step
                if matches!(s.route, Route::Fused { .. })
                    && matches!(s.op.kernel, KernelOp::GemmI4A8 | KernelOp::GemmI8A8)
                {
                    assert!(
                        fuse,
                        "{tp} {bucket:?}: {} folded in an unfused plan",
                        s.op.name
                    );
                    seen.insert(s.op.name.clone());
                    continue;
                }
                let Route::Kernel { descs, .. } = &s.route else {
                    continue;
                };
                for d in descs {
                    assert!(
                        !matches!(
                            d.plugin.as_str(),
                            "mxfp4_had" | "attn_glue_had" | "saber_ar_had" | "mxfp4_quant"
                        ) && !d.plugin.starts_with("mxfp4_gemm"),
                        "{tp} {bucket:?}: {} still runs {} / {}",
                        s.op.name,
                        d.plugin,
                        d.kernel
                    );
                    if d.plugin == "saber_ar_hadi8" {
                        nq += 1;
                    }
                }
                if !matches!(s.op.kernel, KernelOp::GemmI4A8 | KernelOp::GemmI8A8) {
                    continue;
                }
                assert_eq!(descs.len(), 1);
                assert!(
                    descs[0].plugin == "macaq_iu8"
                        || (descs[0].plugin == "macaq_iu8f"
                            && (fuse
                                // the small-chunk band (mid decode body / small prefill tiles)
                                || descs[0].kernel.starts_with("macaq_iu8f_dec_")
                                || descs[0].kernel.starts_with("macaq_iu8f_pf_"))),
                    "{tp} {bucket:?}: {}",
                    descs[0].kernel
                );
                let is_w8 = s.op.kernel == KernelOp::GemmI8A8;
                assert_eq!(is_w8, w8.contains(&s.op.name), "{}", s.op.name);
                assert!(descs[0]
                    .kernel
                    .contains(if is_w8 { "_w8_" } else { "_w4_" }));
                let sfx = if is_w8 { "wi8" } else { "wi4" };
                let b = |n: &str| {
                    s.bindings
                        .iter()
                        .find(|(x, _)| x == n)
                        .map(|(_, src)| src.clone())
                };
                let wl = if is_w8 { 18 } else { 17 };
                assert!(
                    matches!(b("w"), Some(Source::Weight { ref key, layout }) if key == &format!("{}.{sfx}", s.op.name) && layout == wl),
                    "{:?}",
                    b("w")
                );
                assert!(
                    matches!(b("ws"), Some(Source::Weight { ref key, layout: 19 }) if key == &format!("{}.{sfx}s", s.op.name))
                );
                assert!(matches!(b("a"), Some(Source::Arena { .. })));
                assert!(matches!(b("as"), Some(Source::Arena { .. })));
                seen.insert(s.op.name.clone());
                // its producer: the step that writes `a` is an int8 twin, A-tiled
                let a = s.op.inputs[0].edge.clone();
                let j = p1.steps[..i]
                    .iter()
                    .rposition(|p| p.op.outputs.iter().any(|e| e.name == a))
                    .expect("a producer step");
                // a norm folded into the reduce before it (`_nq`) is carried by that reduce's step
                let carrier = p1.steps[..=j]
                    .iter()
                    .rev()
                    .find(|p| matches!(p.route, Route::Kernel { .. }))
                    .expect("a launching step");
                let prod = &p1.steps[j];
                let Route::Kernel { descs: pd, .. } = &carrier.route else {
                    unreachable!()
                };
                let last = pd.last().unwrap();
                // or row 3's iu8 body, which carries the reduce + the int8 `_nq`
                let row3 = last.plugin == "macaq_iu8f"
                    && last.kernel.starts_with("macaq_iu8f_arnq_hadi8_");
                assert!(
                    I8_PLUGINS.contains(&last.plugin.as_str()) || row3,
                    "{tp} {bucket:?}: {} -> {} writes `{a}` with {}",
                    prod.op.name,
                    s.op.name,
                    last.kernel
                );
                assert!(
                    last.kernel.ends_with("_tiled1") || last.plugin == "saber_ar_hadi8" || row3,
                    "{}",
                    last.kernel
                );
                let buf = p1.buffers.iter().find(|x| x.name == a);
                assert!(
                    buf.is_none_or(|x| x.layout == Layout::ATiledI8),
                    "{a}: {:?}",
                    buf.map(|x| x.layout)
                );
            }
            assert!(p1.buffers.iter().any(|x| x.layout == Layout::ATiledI8));
            assert!(!p1
                .buffers
                .iter()
                .any(|x| matches!(x.layout, Layout::ATiledFp8 | Layout::RowMajorFp8)));
            // every linear appears as its own step
            assert!(
                seen == w4a8,
                "{tp} {bucket:?}: missing {:?} extra {:?}",
                w4a8.difference(&seen).take(5).collect::<Vec<_>>(),
                seen.difference(&w4a8).take(5).collect::<Vec<_>>()
            );
            // every reduce + norm the had32 plan folded (`_nq`, or row 3's GEMM + reduce + norm) folds
            // into the int8 `_nq` twin here (row 3's GEMM is MXFP4: the iu8 GEMM stays its own launch)
            let count = |p: &BlockPlan<'_>, plug: &str| {
                p.steps
                    .iter()
                    .filter(|s| {
                        matches!(&s.route, Route::Kernel { descs, .. } if descs.iter().any(|d| d.plugin == plug))
                    })
                    .count()
            };
            let (nq0, r3) = (
                count(&p0, "saber_ar_had"),
                count(&p0, "mxfp4_gemm_decode_arnq_had"),
            );
            let count_op = |p: &BlockPlan<'_>, plug: &str, op: &str| {
                p.steps
                    .iter()
                    .filter(|s| {
                        matches!(&s.route, Route::Kernel { descs, .. } if descs.iter().any(|d| d.op == op && d.plugin.starts_with(plug)))
                    })
                    .count()
            };
            // row 3 and the decode pair fold on the `macaq_iu8f` bodies at every site the had32
            // plan folds them: the same sites, the same launch count
            let r3i = count_op(&p1, "macaq_iu8f", "gemm_decode_ar_nq");
            let (pairs0, pairs1) = (
                count_op(&p0, "mxfp4_gemm_decode", "gemm_decode_pair"),
                count_op(&p1, "macaq_iu8f", "gemm_decode_pair"),
            );
            assert_eq!(
                r3i, r3,
                "{tp} {bucket:?}: row 3 sites had32 {r3} -> iu8 {r3i}"
            );
            // the iu8 pair also fires where the MXFP4 pair refuses its halves' cells (TP=1 b8: 48)
            assert!(
                pairs1 >= pairs0,
                "{tp} {bucket:?}: pairs had32 {pairs0} -> iu8 {pairs1}"
            );
            if !fuse {
                assert_eq!(
                    r3 + pairs1,
                    0,
                    "{tp} {bucket:?}: an unfused plan folds nothing"
                );
            }
            assert_eq!(
                nq + r3i,
                nq0 + r3,
                "{tp} {bucket:?}: `_nq` {nq0} + row 3 {r3} -> int8 `_nq` {nq} + iu8 row 3 {r3i}"
            );
            let (l0, l1) = (p0.expected_launches(), p1.expected_launches());
            assert_eq!(
                l1 + (pairs1 - pairs0) as u32,
                l0,
                "{tp} {bucket:?}: fused {fuse}: the iu8 plan keeps the had32 count (less its extra pairs)"
            );
            eprintln!("  (row 3 sites {r3}, pairs {pairs1})");
            eprintln!(
                "iu8 plan tp{tp} {bucket:?} fuse {fuse}: launches had32 {l0} -> iu8 {l1}, `_nq` {nq0} + row 3 {r3} -> {nq} + {r3i}, arena {} -> {} B",
                p0.arena_bytes,
                p1.arena_bytes
            );
        }
    }
}

#[test]
fn the_records_shards_and_refusals() {
    for (tp, ab) in [(2u32, 48u64), (1, 96)] {
        let had = Graph::qwen38_27b(ModelConfig::qwen38_27b(), tp)
            .unwrap()
            .with_macaq_had()
            .unwrap();
        let w8 = bc3(&had);
        let iu8 = had.clone().with_macaq_iu8(&w8).unwrap();
        let recs = iu8.device_final_records().unwrap();
        let n_ops = iu8.macaq_iu8().len();
        for op in iu8.macaq_iu8() {
            let sfx = if w8.contains(&op) { "wi8" } else { "wi4" };
            let w = recs
                .iter()
                .find(|r| r.name == format!("{op}.{sfx}"))
                .unwrap();
            let s = recs
                .iter()
                .find(|r| r.name == format!("{op}.{sfx}s"))
                .unwrap();
            assert_eq!(w.layout_id, if sfx == "wi8" { 18 } else { 17 });
            assert_eq!(s.layout_id, 19);
            assert_eq!(s.shape[0] / 16, w.shape[0], "{op}");
            assert!(
                w.sources
                    .iter()
                    .all(|x| x.weight.ends_with(if sfx == "wi8" {
                        ".weight_i8"
                    } else {
                        ".weight_i4"
                    }) && x.scale.is_some()),
                "{op}"
            );
            let row_parallel = [".gdn.out_proj", ".attn.o_proj", ".mlp.down_proj"]
                .iter()
                .any(|r| op.ends_with(r));
            if row_parallel && tp == 2 {
                assert_eq!(w.shard.manifest_name(), "Dim1", "{op}");
                assert_eq!(s.shard.manifest_name(), "Replicated", "{op}");
            }
            assert!(
                !recs.iter().any(|r| r.name == format!("{op}.w")),
                "{op} keeps an MXFP4 record"
            );
            if op.ends_with(".gdn.in_proj_ba") {
                assert_eq!(s.shape, vec![ab], "{op}");
                assert_eq!(w.sources.len(), 2, "b then a");
            }
            if op.ends_with(".gdn.in_proj_qkvz") || op.ends_with(".mlp.gate_up") {
                assert_eq!(w.sources.len(), 2, "{op}");
            }
        }
        // MXFP4: 3 records an op (w, ws, wref) -> iu8: 2
        assert_eq!(
            recs.len(),
            had.device_final_records().unwrap().len() - n_ops
        );
        // refusals by name
        let plain = Graph::qwen38_27b(ModelConfig::qwen38_27b(), tp).unwrap();
        let e = plain.with_macaq_iu8(&w8).unwrap_err().to_string();
        assert!(e.contains("macaq_had32"), "{e}");
        let missing: BTreeSet<String> = ["layer.99.mlp.down_proj".to_owned()].into();
        assert!(had.clone().with_macaq_iu8(&missing).is_err());
        assert!(iu8.clone().with_macaq_iu8(&w8).is_err(), "twice is refused");
        let w8g = had.clone().with_macaq_w8(&w8).unwrap();
        assert!(
            w8g.with_macaq_iu8(&BTreeSet::new()).is_err(),
            "the e4m3 W8 promotions cannot mix in"
        );
        // an empty W8 set is a pure-W4 iu8 graph
        let w4only = had.clone().with_macaq_iu8(&BTreeSet::new()).unwrap();
        assert!(w4only.ops.iter().all(|o| o.kernel != KernelOp::GemmI8A8));
    }
}
