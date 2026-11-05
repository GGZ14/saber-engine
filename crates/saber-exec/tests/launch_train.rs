//! **THE DECODE LAUNCH TRAIN, counted on the host** (ws-h37).
//!
//! Every dispatch of the TP=2 decode body at the two serving buckets (`b1s8`, `b4s8`), by op, from
//! the plan the engine builds (`PlanOptions::BODY` + the rank + real collectives + batch-invariant
//! attention), with no card — and the node count the launch-train fusions delete from it (ws-h37's
//! `_nq` epilogue and decode pair, ws-h38's fusion-map row 3).

use std::collections::BTreeMap;

use saber_exec::layer::{BlockPlan, Collectives, PlanOptions, Route, RuntimeParams, Step};
use saber_kernels::rules::{LaunchDescriptor, LaunchRules};
use saber_kernels::saber::SaberRules;
use saber_model::graph::{Bucket, Graph, KernelOp, ModelConfig, Phase};

fn plan_at(g: &Graph, batch: u32, width: u32, fuse: bool) -> BlockPlan<'_> {
    plan_with(g, batch, width, |o| {
        if fuse {
            o.with_decode_fusion()
        } else {
            o.without_fusion()
        }
    })
}

/// [`plan_at`] with the fusion set chosen by the caller.
fn plan_with(
    g: &Graph,
    batch: u32,
    width: u32,
    fusions: impl FnOnce(PlanOptions) -> PlanOptions,
) -> BlockPlan<'_> {
    let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
    let collectives = if g.tp > 1 {
        Collectives::Real
    } else {
        Collectives::Identity
    };
    let o = fusions(
        PlanOptions::BODY
            .rank(0)
            .with_collectives(collectives)
            .with_batch_invariant_attn(),
    );
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

/// `(op name without its layer.{n}. prefix, kernel) -> (ops, launches)` over the whole body.
fn census(p: &BlockPlan<'_>) -> BTreeMap<(String, String), (u32, u32)> {
    let mut m = BTreeMap::new();
    for s in &p.steps {
        let name = s.name();
        let suffix = name
            .strip_prefix("layer.")
            .and_then(|r| r.split_once('.'))
            .map_or(name.to_owned(), |(_, r)| r.to_owned());
        let (kernel, n) = match &s.route {
            Route::Kernel { descs, .. } => (
                descs
                    .iter()
                    .map(|d| d.kernel.as_str())
                    .collect::<Vec<_>>()
                    .join("+"),
                descs[0].n_launches,
            ),
            Route::Fused { .. } => ("(fused)".to_owned(), 0),
            Route::Identity { .. } => ("(identity)".to_owned(), 0),
            Route::Supplied { .. } => ("(supplied)".to_owned(), 0),
        };
        let e = m.entry((suffix, kernel)).or_insert((0, 0));
        e.0 += 1;
        e.1 += n;
    }
    m
}

/// The census table itself (`--nocapture` prints it), and its one law: it sums to the plan's own
/// launch budget, so a row cannot be missing.
#[test]
fn the_decode_body_census_at_the_serving_buckets() {
    let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("the TP=2 graph");
    for fuse in [false, true] {
        for (batch, width) in [(1u32, 8u32), (4, 8)] {
            let p = plan_at(&g, batch, width, fuse);
            let c = census(&p);
            let total: u32 = c.values().map(|v| v.1).sum();
            println!(
                "b{batch}s{width} fused={fuse}: {} ops, {total} launches",
                p.steps.len()
            );
            for ((op, k), (n, l)) in &c {
                println!("  {op:<34} {k:<60} ops {n:>4} launches {l:>4}");
            }
            assert_eq!(
                total,
                p.expected_launches(),
                "the census is the plan's budget"
            );
        }
    }
}

/// The fusions delete exactly the nodes the census says they can. At `b1s8` every EXACT reduce
/// (a message is 80 KiB, under `quant_min_bytes`) is folded: 127 of them WITH the split decode
/// GEMM before them and the `add_rms_quant` after them (fusion-map row 3: 16 `o_proj`, 48 GDN
/// `out_proj`, 63 `down_proj`), and the embedding's reduce -- no GEMM before it -- as the plain
/// `_nq` epilogue; the last layer's `down_proj` reduce has no norm in the body and stays a plain
/// reduce. At `b4s8` (320 KiB, the rotated family) neither fires. The decode pair folds each GDN
/// layer's `in_proj_ba` into its `in_proj_qkvz` (48 at every decode bucket). The control is the
/// unfused plan: the same arena, the same ops, the whole budget.
#[test]
fn the_fusions_delete_the_nodes_the_census_names() {
    let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("the TP=2 graph");
    for (batch, width, nq, pairs, gemm_ar) in [
        (1u32, 8u32, 1u32, 48u32, 127u32),
        (4, 8, 0, 48, 0),
        (1, 5, 1, 48, 127),
    ] {
        let off = plan_at(&g, batch, width, false);
        let on = plan_at(&g, batch, width, true);
        assert_eq!(
            (off.fused_nq, off.fused_pairs, off.fused_gemm_ar),
            (0, 0, 0),
            "the control fused"
        );
        assert_eq!(
            (on.fused_nq, on.fused_pairs, on.fused_gemm_ar),
            (nq, pairs, gemm_ar),
            "b{batch}s{width}: the fusions folded a different set"
        );
        // ws-h38: the rope + append and the last-arriver combine, one per attention layer.
        let h38 = on.fused_rope_appends + on.fused_combines;
        // (Row 6, the attention combine, is out of the served set: measured slower.)
        assert_eq!((on.fused_rope_appends, on.fused_combines), (16, 0));
        let folded = nq + pairs + 2 * gemm_ar + h38;
        assert_eq!(on.fused_launches, folded);
        assert_eq!(on.expected_launches() + folded, off.expected_launches());
        assert_eq!(on.arena_bytes, off.arena_bytes, "a fusion moved a buffer");
        println!(
            "b{batch}s{width}: {} -> {} launches ({nq} nq, {pairs} pairs, {gemm_ar} gemm+ar+nq)",
            off.expected_launches(),
            on.expected_launches()
        );
        // Every pair carries both halves at their own unpaired split: its grid is the sum of the
        // two unpaired grids, and the op after it is folded.
        let blocks = |d: &LaunchDescriptor| d.grid[0] * d.grid[2];
        let mut seen = 0;
        for (i, s) in on.steps.iter().enumerate() {
            let Route::Kernel { descs, .. } = &s.route else {
                continue;
            };
            if descs[0].op != "gemm_decode_pair" {
                continue;
            }
            seen += 1;
            let (Route::Kernel { descs: a, .. }, Route::Kernel { descs: b, .. }) =
                (&off.steps[i].route, &off.steps[i + 1].route)
            else {
                panic!("the unfused plan has no two GEMMs at {}", s.name());
            };
            assert_eq!(
                descs[0].grid[0],
                blocks(&a[0]) + blocks(&b[0]),
                "{}",
                s.name()
            );
            assert!(matches!(on.steps[i + 1].route, Route::Fused { .. }));
        }
        assert_eq!(seen, pairs);
    }
}

/// The engine step adds 11 dispatches to the body (head 2, row selections 2, argmax 1, the 5 aux
/// taps, 1): ws-h37's census, `813 / 861` unfused.
const ENGINE_EXTRA: u32 = 11;

/// Fusion-map ROW 3 (ws-h38), counted: at `b1s8` it folds 127 `allreduce + add_rms_quant` pairs
/// into the split decode GEMM before them, so the body is 626 -> 499 dispatches on top of ws-h37's
/// two fusions and the engine step 637 -> 510 (not 509: the embedding's reduce has no GEMM before
/// it and keeps the plain `_nq` epilogue). Every folded pair is the two ops right after a
/// `gemm_decode_ar_nq` dispatch, both `Route::Fused` into it, and the dispatch is the unfused
/// GEMM's own cell with `_arnq_` in its name and the GEMM's own grid.
#[test]
fn row3_folds_every_exact_reduce_behind_a_split_decode_gemm() {
    let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("the TP=2 graph");
    let off = plan_at(&g, 1, 8, false);
    let h37 = plan_with(&g, 1, 8, PlanOptions::with_h37_fusion);
    let on = plan_at(&g, 1, 8, true);
    assert_eq!(off.expected_launches(), 802);
    assert_eq!(h37.expected_launches(), 626);
    assert_eq!(h37.fused_gemm_ar, 0, "ws-h37's set does not include row 3");
    // Row 7 (16) rides along in `--decode-fusion` (row 6 was measured slower and is out):
    // 626 - 127 - 16 = 483.
    let r3 = plan_with(&g, 1, 8, |o| o.with_h37_fusion().with_gemm_ar());
    assert_eq!(r3.expected_launches(), 499);
    assert_eq!(on.expected_launches(), 483);
    assert_eq!(
        (
            h37.expected_launches() + ENGINE_EXTRA,
            r3.expected_launches() + ENGINE_EXTRA,
            on.expected_launches() + ENGINE_EXTRA
        ),
        (637, 510, 494),
        "the engine's dispatches per step"
    );
    assert_eq!(on.arena_bytes, off.arena_bytes, "a fusion moved a buffer");
    let mut seen = 0u32;
    let mut sites: BTreeMap<String, u32> = BTreeMap::new();
    for (i, s) in on.steps.iter().enumerate() {
        let Route::Kernel { descs, .. } = &s.route else {
            continue;
        };
        if descs[0].op != "gemm_decode_ar_nq" {
            continue;
        }
        seen += 1;
        let d = &descs[0];
        assert_eq!(d.n_launches, 1);
        let Route::Kernel { descs: u, .. } = &off.steps[i].route else {
            panic!("the unfused plan has no GEMM at {}", s.name());
        };
        assert_eq!(
            d.kernel,
            u[0].kernel
                .replace("mxfp4_gemm_decode_", "mxfp4_gemm_decode_arnq_"),
            "{}",
            s.name()
        );
        assert_eq!(
            d.grid,
            u[0].grid,
            "{}: one dispatch, the GEMM's grid",
            s.name()
        );
        for (j, want) in [(1, KernelOp::AllReduce), (2, KernelOp::AddRmsQuant)] {
            let f = &on.steps[i + j];
            assert_eq!(f.op.kernel, want, "{}", f.name());
            assert!(
                matches!(&f.route, Route::Fused { into, .. } if into == s.name()),
                "{} must be fused into {}",
                f.name(),
                s.name()
            );
        }
        let site = s
            .name()
            .strip_prefix("layer.")
            .and_then(|r| r.split_once('.'))
            .map_or(s.name().to_owned(), |(_, r)| r.to_owned());
        *sites.entry(site).or_default() += 1;
    }
    assert_eq!(seen, 127);
    assert_eq!(
        sites.into_iter().collect::<Vec<_>>(),
        [
            ("attn.o_proj".to_owned(), 16),
            ("gdn.out_proj".to_owned(), 48),
            ("mlp.down_proj".to_owned(), 63),
        ]
    );
    // Every fused op names a step that DISPATCHES: the engine's aux tap follows the dispatch a
    // folded `input_layernorm` names (ws-h37 fdd62c1), and under row 3 that is the previous
    // layer's `down_proj`, which writes the norm's `res_out`.
    for s in &on.steps {
        if let Route::Fused { into, .. } = &s.route {
            let carrier = on
                .steps
                .iter()
                .find(|c| c.name() == into)
                .unwrap_or_else(|| panic!("{} is fused into a step that is not there", s.name()));
            assert!(
                matches!(carrier.route, Route::Kernel { .. }),
                "{} is fused into {}, which does not dispatch",
                s.name(),
                into
            );
        }
    }
    for l in 1..g.config.num_hidden_layers {
        let norm = format!("layer.{l}.input_layernorm");
        let s = on
            .steps
            .iter()
            .find(|s| s.name() == norm)
            .expect("the norm");
        let Route::Fused { into, .. } = &s.route else {
            panic!("{norm} is not fused at b1s8");
        };
        assert_eq!(*into, format!("layer.{}.mlp.down_proj", l - 1));
    }
}

/// Row 3 is a function of the bucket, not the model: at `b4s8` the message is 320 KiB, the
/// reduce is the rotated (wht6) family, and the plan with row 3 ON is ws-h37's plan exactly.
/// (Fusing a rotated reduce would need each 64-element quantisation group, its amax and its
/// rotation whole in one block, and a banded message's bands in order; not built.)
#[test]
fn row3_leaves_the_banded_bucket_alone() {
    let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("the TP=2 graph");
    let h37 = plan_with(&g, 4, 8, PlanOptions::with_h37_fusion);
    let on = plan_with(&g, 4, 8, |o| o.with_h37_fusion().with_gemm_ar());
    assert_eq!(on.fused_gemm_ar, 0);
    assert_eq!(on.expected_launches(), h37.expected_launches());
    assert_eq!(on.expected_launches(), 802);
    assert_eq!(on.expected_launches() + ENGINE_EXTRA, 813);
    // The whole `--decode-fusion` set there is rows 1-2 + row 7: 802 - 16 = 786.
    assert_eq!(plan_at(&g, 4, 8, true).expected_launches(), 786);
    let kernels = |p: &BlockPlan<'_>| -> Vec<String> {
        p.steps
            .iter()
            .flat_map(|s| s.route.symbols().into_iter().map(str::to_owned))
            .collect()
    };
    assert_eq!(kernels(&on), kernels(&h37));
}

/// Row 3 on its own (`with_gemm_ar`, the gate's `--fusion gemm-ar`) folds the same 127 triples
/// and nothing else: the embedding's reduce stays a plain reduce because `fuse` is off.
#[test]
fn row3_alone_folds_only_its_triples() {
    let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("the TP=2 graph");
    let off = plan_at(&g, 1, 8, false);
    let r3 = plan_with(&g, 1, 8, PlanOptions::with_gemm_ar);
    assert_eq!((r3.fused_nq, r3.fused_pairs, r3.fused_gemm_ar), (0, 0, 127));
    assert_eq!(r3.fused_launches, 254);
    assert_eq!(r3.expected_launches(), off.expected_launches() - 254);
    // ... and the unfused plan is the unfused plan.
    assert_eq!(off.fused_launches, 0);
    assert_eq!(off.expected_launches(), 802);
}

/// TP=1 has no reduce to fold: the plan with every fusion ON folds the pair, rows 6 and 7, and
/// nothing of row 3.
#[test]
fn row3_does_not_fire_at_tp1() {
    let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 1).expect("the TP=1 graph");
    let on = plan_at(&g, 1, 8, true);
    let h37 = plan_with(&g, 1, 8, |o| o.with_h37_fusion().with_rope_append());
    assert_eq!((on.fused_nq, on.fused_gemm_ar), (0, 0));
    assert_eq!((on.fused_rope_appends, on.fused_combines), (16, 0));
    assert_eq!(on.expected_launches(), h37.expected_launches());
}

/// A plan at `bucket` with exactly the options `opts` adds to the engine's body options.
fn plan_full(
    g: &Graph,
    bucket: Bucket,
    gdn_slots: u32,
    max_ctx: u32,
    opts: impl Fn(PlanOptions) -> PlanOptions,
) -> BlockPlan<'_> {
    let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
    let o = PlanOptions::BODY
        .rank(0)
        .with_collectives(Collectives::Real)
        .with_batch_invariant_attn();
    BlockPlan::with_options(
        g,
        bucket,
        &layers,
        RuntimeParams { max_ctx, gdn_slots },
        &LaunchRules::prod_mxfp4().expect("rules"),
        &SaberRules::prod_saber().expect("saber rules"),
        opts(o),
    )
    .expect("the body plans")
}

fn decode(batch: u32, width: u32) -> Bucket {
    Bucket::Decode {
        batch,
        verify_width: width,
    }
}

/// The kernels a step dispatches, `+`-joined, or its route's kind.
fn kernels_of(s: &Step<'_>) -> String {
    match &s.route {
        Route::Kernel { descs, .. } => descs
            .iter()
            .map(|d| d.kernel.as_str())
            .collect::<Vec<_>>()
            .join("+"),
        Route::Fused { into, .. } => format!("(fused into {into})"),
        Route::Identity { .. } => "(identity)".to_owned(),
        Route::Supplied { .. } => "(supplied)".to_owned(),
    }
}

/// **ws-h38: fusion-map rows 6 and 7, each alone.** `qk_norm_rope` + `kv_append` in one dispatch
/// (16 attention layers -> 16 nodes) and the decode attention's combine in its last-arriving split
/// block (16 more), at the two serving buckets and at the two narrower widths the engine also
/// serves. Each alone deletes exactly its 16 nodes and moves no buffer; the rope + append takes
/// the ROPE's grid (the decomposition the band picks: the per-head kernel at `b1s1`'s 14 pairs,
/// the wave kernel above 32) and the combine keeps the unfused decode's grid (the same split
/// count), which is > 1 at every one of these buckets.
#[test]
fn the_rope_append_and_the_attention_combine_fold_sixteen_each() {
    let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("the TP=2 graph");
    for (batch, width) in [(1u32, 8u32), (4, 8), (1, 5), (1, 1)] {
        let slots = 1 + batch * width;
        let off = plan_full(
            &g,
            decode(batch, width),
            slots,
            4096,
            PlanOptions::without_fusion,
        );
        let rope = plan_full(&g, decode(batch, width), slots, 4096, |o| {
            o.without_fusion().with_rope_append()
        });
        let comb = plan_full(&g, decode(batch, width), slots, 4096, |o| {
            o.without_fusion().with_attn_combine()
        });
        for (p, what) in [(&rope, "rope-append"), (&comb, "attn-combine")] {
            assert_eq!(p.arena_bytes, off.arena_bytes, "{what} moved a buffer");
            assert_eq!(p.steps.len(), off.steps.len());
            assert_eq!(
                p.expected_launches() + 16,
                off.expected_launches(),
                "b{batch}s{width} {what}"
            );
            assert_eq!(p.fused_launches, 16);
            assert_eq!((p.fused_nq, p.fused_pairs), (0, 0));
        }
        assert_eq!((rope.fused_rope_appends, rope.fused_combines), (16, 0));
        assert_eq!((comb.fused_rope_appends, comb.fused_combines), (0, 16));
        let (mut ropes, mut combs) = (0, 0);
        for (i, s) in off.steps.iter().enumerate() {
            let Route::Kernel { descs, .. } = &s.route else {
                continue;
            };
            if descs[0].op == "qk_norm_rope" {
                let Route::Kernel { descs: f, .. } = &rope.steps[i].route else {
                    panic!("{}: the rope is not a kernel in the fused plan", s.name());
                };
                assert_eq!(f[0].op, "qk_rope_append");
                assert_eq!((f[0].grid, f[0].block), (descs[0].grid, descs[0].block));
                let wave = descs[0].kernel.contains("_wg8_");
                assert_eq!(f[0].kernel.contains("_wg8_"), wave, "{}", f[0].kernel);
                assert!(
                    matches!(&rope.steps[i + 1].route,
                             Route::Fused { into, .. } if *into == s.name()),
                    "{}: the append after the rope is not folded into it",
                    s.name()
                );
                if ropes == 0 {
                    println!(
                        "b{batch}s{width}: {} + {} -> {}",
                        kernels_of(s),
                        kernels_of(&off.steps[i + 1]),
                        kernels_of(&rope.steps[i])
                    );
                }
                ropes += 1;
            }
            if descs[0].op == "attn_decode" {
                assert_eq!(descs.len(), 2, "the unfused decode attention is a pair");
                let Route::Kernel { descs: f, .. } = &comb.steps[i].route else {
                    panic!("{}: not a kernel in the fused plan", s.name());
                };
                assert_eq!(f.len(), 1);
                assert_eq!(f[0].op, "attn_decode_declc");
                assert_eq!((f[0].grid, f[0].block), (descs[0].grid, descs[0].block));
                assert!(f[0].grid[0] > 1, "splits {}", f[0].grid[0]);
                assert_eq!(f[0].n_launches, 1);
                if combs == 0 {
                    println!(
                        "b{batch}s{width}: {} -> {} ({} splits, grid {:?})",
                        kernels_of(s),
                        kernels_of(&comb.steps[i]),
                        f[0].grid[0],
                        f[0].grid
                    );
                }
                combs += 1;
            }
        }
        assert_eq!((ropes, combs), (16, 16));
        println!(
            "b{batch}s{width}: {} dispatches -> rope-append {} / attn-combine {}",
            off.expected_launches(),
            rope.expected_launches(),
            comb.expected_launches()
        );
    }
}

/// The fusions are DECODE fusions: a prefill chunk planned with them on is the plan it was --
/// every descriptor identical, nothing folded. And the split count the combine keys on is a real
/// condition: a decode plan whose context fits one tile resolves ONE split, so there is nothing
/// to merge and the pair stays (the negative control: the same options at 4096 fold 16).
#[test]
fn prefill_and_one_split_plans_are_left_alone() {
    let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("the TP=2 graph");
    let pre = g.bucket_for(2048, Phase::Prefill);
    let plain = plan_full(&g, pre, 2, 4096, PlanOptions::without_fusion);
    let fused = plan_full(&g, pre, 2, 4096, |o| {
        o.without_fusion().with_rope_append().with_attn_combine()
    });
    assert_eq!((fused.fused_rope_appends, fused.fused_combines), (0, 0));
    assert_eq!(fused.expected_launches(), plain.expected_launches());
    for (a, b) in plain.steps.iter().zip(&fused.steps) {
        assert_eq!(
            format!("{:?}", a.route),
            format!("{:?}", b.route),
            "{}",
            a.name()
        );
    }
    let short = plan_full(&g, decode(1, 8), 9, 16, |o| {
        o.without_fusion().with_attn_combine()
    });
    assert_eq!(short.fused_combines, 0, "one split has nothing to merge");
    let long = plan_full(&g, decode(1, 8), 9, 4096, |o| {
        o.without_fusion().with_attn_combine()
    });
    assert_eq!(long.fused_combines, 16, "the control");
}

/// Fusion-map ROW 4 (ws-h40): the `h40` set folds every layer's `silu_mul_quant` into the
/// gate_up decode GEMM before it -- 64 sites at both TP=2 serving buckets -- and removes exactly
/// those 64 launches from the `all` set's train. Each fused dispatch is the row-4 symbol at the
/// cell the gate_up GEMM takes alone. At TP=1 the producer's row (N = 17,408) takes the 512-thread
/// block, which the 256-thread decode block cannot carry, so row 4 does not fire there.
#[test]
fn row4_folds_every_layers_silu_mul_quant() {
    for tp in [2, 1] {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), tp).expect("the graph");
        for batch in [1, 4] {
            let all = plan_at(&g, batch, 8, true);
            let h40 = plan_with(&g, batch, 8, |o| o.with_h40_fusion());
            assert_eq!(
                all.fused_gemm_smq, 0,
                "tp{tp} b{batch}s8: `all` has no row 4"
            );
            let want = if tp == 2 { 64 } else { 0 };
            assert_eq!(h40.fused_gemm_smq, want, "tp{tp} b{batch}s8");
            assert_eq!(
                h40.expected_launches(),
                all.expected_launches() - want,
                "tp{tp} b{batch}s8"
            );
            let mut sites = 0;
            for s in &h40.steps {
                if let Route::Kernel { descs, .. } = &s.route {
                    if descs[0].op == "gemm_decode_smq" {
                        sites += 1;
                        assert!(descs[0]
                            .kernel
                            .starts_with("mxfp4_gemm_decode_smq_bk128_ks1_tm"));
                        assert_eq!(descs[0].grid[0], 136, "gate_up's n-tiles");
                    }
                }
            }
            assert_eq!(sites, want);
        }
    }
}

/// Fusion-map ROW 5 (ws-h42): the `h42` set folds every GDN layer's `gdn_norm_quant` into the
/// GDN fused update before it -- 48 sites -- wherever the decode GDN takes the fused update: one
/// sequence at TP=2 (`n_seqs x H = 24 <= fused_max_items = 32`). At four sequences the step takes
/// the conv + recurrent PAIR, and at TP=1 (`H = 48`) even one sequence does, so row 5 does not
/// fire there. Each fused dispatch is the row-5 symbol on the update's own grid. The row is
/// compiled for the f16 stored state (the serving default) only: on the f32 state graph it never
/// fires (the control below).
#[test]
fn row5_folds_every_gdn_layers_norm_where_the_update_is_fused() {
    let g32 = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("the graph");
    assert_eq!(
        plan_with(&g32, 1, 8, |o| o.with_h42_fusion()).fused_gdn_nq,
        0,
        "f32 state: no row-5 instantiation"
    );
    for tp in [2, 1] {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), tp)
            .expect("the graph")
            .with_gdn_state_f16()
            .expect("f16 state");
        for (batch, width) in [(1, 8), (1, 5), (4, 8)] {
            let h40 = plan_with(&g, batch, width, |o| o.with_h40_fusion());
            let h42 = plan_with(&g, batch, width, |o| o.with_h42_fusion());
            assert_eq!(
                h40.fused_gdn_nq, 0,
                "tp{tp} b{batch}s{width}: `h40` has no row 5"
            );
            let want = if tp == 2 && batch == 1 { 48 } else { 0 };
            assert_eq!(h42.fused_gdn_nq, want, "tp{tp} b{batch}s{width}");
            assert_eq!(h42.fused_gemm_smq, h40.fused_gemm_smq, "row 4 unchanged");
            assert_eq!(
                h42.expected_launches(),
                h40.expected_launches() - want,
                "tp{tp} b{batch}s{width}"
            );
            let mut sites = 0;
            for s in &h42.steps {
                if let Route::Kernel { descs, .. } = &s.route {
                    if descs[0].op == "gdn_fnq" {
                        sites += 1;
                        assert_eq!(descs[0].kernel, "saber_gdn_fnq_f16s_w4k128v128_ab1");
                        assert_eq!(descs[0].grid[0], 24, "1 sequence x 24 v heads");
                    }
                }
            }
            assert_eq!(sites, want);
        }
    }
}

/// CMP27AE: the GEMM + wht6 fold (`with_gemm_wht`, `--engine-ar-fuse`) carries the rotated reduce
/// after every N 5120 dwn4 decode GEMM -- out_proj (48) + o_proj (16) + down (64) = 128 sites a
/// step -- at the dwn4 buckets (b4s8 = 32 rows tm2, b8s8 = 64 tm4), and nowhere
/// else: b4s5 / b8s5 (20 / 40 rows: cells with no dwn4 twin), b2s8 (16 rows, tm1) and b1 (the
/// exact family) keep their launches, the dwn4 knob
/// off leaves nothing to fold, and the race check stands the fold down (the checked reduces).
#[test]
fn cmp27ae_the_gemm_wht_fold_carries_the_128_target_reduces_at_17_to_64_rows() {
    let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2)
        .expect("the graph")
        .with_gdn_state_f16()
        .expect("f16 state");
    let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
    let mut mx = LaunchRules::prod_mxfp4().expect("rules");
    mx.knobs.decode_pf = true;
    mx.knobs.decode_pf_tm2 = true;
    mx.knobs.decode_pf_dwn4 = true;
    let plan = |batch: u32, width: u32, mx: &LaunchRules, saber: &SaberRules, fold: bool| {
        let o = PlanOptions::BODY
            .rank(0)
            .with_collectives(Collectives::Real)
            .with_batch_invariant_attn()
            .with_decode_fusion();
        let o = if fold { o.with_gemm_wht() } else { o };
        BlockPlan::with_options(
            &g,
            Bucket::Decode {
                batch,
                verify_width: width,
            },
            &layers,
            RuntimeParams {
                max_ctx: 4096,
                gdn_slots: 1 + batch * width,
            },
            mx,
            saber,
            o,
        )
        .expect("the decode body plans")
    };
    let saber = SaberRules::prod_saber().expect("saber rules");
    for (batch, width, want) in [
        (4, 8, 128),
        (8, 8, 128),
        (4, 5, 0),
        (8, 5, 0),
        (2, 8, 0),
        (1, 8, 0),
    ] {
        let off = plan(batch, width, &mx, &saber, false);
        let on = plan(batch, width, &mx, &saber, true);
        assert_eq!(
            off.fused_gemm_wht, 0,
            "b{batch}s{width}: the control folds nothing"
        );
        assert_eq!(on.fused_gemm_wht, want, "b{batch}s{width}");
        assert_eq!(
            on.expected_launches(),
            off.expected_launches() - want,
            "b{batch}s{width}: one launch a folded reduce"
        );
        let mut sites = 0;
        for (i, s) in on.steps.iter().enumerate() {
            if let Route::Kernel { descs, .. } = &s.route {
                if descs[0].plugin == saber_kernels::rules::GEMM_ARW_PLUGIN {
                    sites += 1;
                    let next = &on.steps[i + 1];
                    assert!(matches!(next.route, Route::Fused { .. }), "{}", next.name());
                    assert_eq!(next.op.kernel, KernelOp::AllReduce);
                    let tm = if batch * width <= 32 { 2 } else { 4 };
                    assert_eq!(
                        descs[0].kernel,
                        format!("mxfp4_gemm_decode_arw_bk128_ks2_tm{tm}_dwn4")
                    );
                    assert_eq!(descs[0].grid, [80, 1, 2]);
                }
            }
        }
        assert_eq!(sites, want, "b{batch}s{width}");
    }
    // The dwn4 knob off: the GEMMs take the 8-wave cells, nothing is a fold site.
    let mut no_dwn4 = mx.clone();
    no_dwn4.knobs.decode_pf_dwn4 = false;
    assert_eq!(plan(8, 8, &no_dwn4, &saber, true).fused_gemm_wht, 0);
    // The race check stands the fold down: the checked reduce keeps its own launch.
    let mut checked = SaberRules::prod_saber().expect("saber rules");
    checked.ar.race_check = true;
    assert_eq!(plan(8, 8, &mx, &checked, true).fused_gemm_wht, 0);
}
