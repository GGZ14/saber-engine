//! **THE PREFILL BUCKET LADDER'S ROW LAW** (C5), on the host and with no card.
//!
//! A prefill chunk runs in the smallest planned bucket that holds it (`device::prefill_step`), and
//! every row-parallel op of that bucket runs at the BUCKET's M. On the Q3 conc-8 corpus the serving
//! launchers' `--engine-prefill-rows 256,2048` put 144 of 400 prompts (257–332 rows) into the
//! 2048-row bucket: 8x the GEMM rows they carry. A finer ladder is the fix, and this file is what a
//! ladder must satisfy for the change to move NOTHING but time:
//!
//! * every bucket of the ladder binds a PREFILL GEMM family (`mxfp4_gemm_folded*` at M <= 512,
//!   `mxfp4_gemm_atiled*` above) — never the split-K decode kernel, whose K partition is a
//!   different sum. The A-tiled and folded families are the same arithmetic per row (G1's ragged
//!   gate; the served-corpus ids against main are the card-side proof);
//! * the NEGATIVE CONTROL: a 64-row bucket binds the decode kernel (`decode_max_m = 64`), so a
//!   ladder that went that low would change a short prompt's numerics. This test fails if the
//!   boundary ever moves without the ladder moving with it;
//! * every bucket declares the same launch count (the body's launch train does not grow with M).

use saber_exec::layer::{BlockPlan, PlanOptions, Route, RuntimeParams};
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_model::graph::{Bucket, Graph, ModelConfig};

/// C5's measured ladder candidate (the serving launchers' `256,2048` plus the rungs between).
const LADDER: [u32; 11] = [96, 128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048];

fn gemm_kernels(g: &Graph, rows: u32) -> (Vec<String>, u32) {
    let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
    let p = BlockPlan::with_options(
        g,
        Bucket::Prefill {
            tokens: rows,
            seqs: 1,
        },
        &layers,
        RuntimeParams {
            max_ctx: 4096,
            gdn_slots: 65,
        },
        &LaunchRules::prod_mxfp4().expect("rules"),
        &SaberRules::prod_saber().expect("saber rules"),
        PlanOptions::BODY.with_batch_invariant_attn(),
    )
    .expect("a prefill body plan");
    let mut out = Vec::new();
    for s in &p.steps {
        if let Route::Kernel { descs, .. } = &s.route {
            for d in descs {
                if d.kernel.starts_with("mxfp4_gemm") {
                    out.push(d.kernel.clone());
                }
            }
        }
    }
    (out, p.declared_launches)
}

#[test]
fn every_ladder_bucket_binds_a_prefill_gemm_and_the_control_does_not() {
    for tp in [1u32, 2] {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), tp).expect("graph");
        let (_, base_launches) = gemm_kernels(&g, 2048);
        for rows in LADDER {
            let (k, launches) = gemm_kernels(&g, rows);
            assert!(!k.is_empty(), "tp{tp} p{rows}: no MXFP4 GEMM at all");
            for name in &k {
                assert!(
                    name.starts_with("mxfp4_gemm_folded") || name.starts_with("mxfp4_gemm_atiled"),
                    "tp{tp} p{rows} binds {name}: a ladder rung off the prefill GEMM family moves \
                     a prompt's numerics, not only its time"
                );
            }
            assert_eq!(
                launches, base_launches,
                "tp{tp} p{rows}: the body's launch train is per layer, not per row"
            );
        }
        // THE NEGATIVE CONTROL: at the decode kernel's M the plan leaves the prefill family.
        let (k, _) = gemm_kernels(&g, 64);
        assert!(
            k.iter().any(|n| n.starts_with("mxfp4_gemm_decode")),
            "tp{tp} p64 must bind the split-K decode GEMM (decode_max_m = 64); it bound {k:?}"
        );
    }
}

/// **C6: THE SMALL RUNGS ON THE DECODE GEMM FAMILY.** Under
/// `saber_engine::small_chunk::small_chunk_rules(.., 128, Exact)` the p96 and p128 plans bind the
/// decode kernel at ONE split for every layer linear (the folded kernel's summation order), with
/// the SAME arena (a one-split decode GEMM takes no workspace: the option costs no memory) and the
/// same launch count; p192 and up are the committed plans byte for byte. The band variant is the
/// control: it must bind at least one split, or the exact arm would not be a distinct claim.
#[test]
fn the_small_rungs_plan_the_decode_gemm_at_one_split_with_the_same_arena() {
    use saber_engine::small_chunk::{small_chunk_rules, SmallChunkClass};
    let base = LaunchRules::prod_mxfp4().expect("rules");
    let exact = small_chunk_rules(&base, 128, SmallChunkClass::Exact).expect("exact");
    let band = small_chunk_rules(&base, 128, SmallChunkClass::Band).expect("band");
    let plan = |g: &Graph, rows: u32, r: &LaunchRules| -> (Vec<String>, u32, u64) {
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let p = BlockPlan::with_options(
            g,
            Bucket::Prefill {
                tokens: rows,
                seqs: 1,
            },
            &layers,
            RuntimeParams {
                max_ctx: 4096,
                gdn_slots: 65,
            },
            r,
            &SaberRules::prod_saber().expect("saber rules"),
            PlanOptions::BODY.with_batch_invariant_attn(),
        )
        .expect("a prefill body plan");
        let mut out = Vec::new();
        for s in &p.steps {
            if let Route::Kernel { descs, .. } = &s.route {
                for d in descs {
                    if d.kernel.starts_with("mxfp4_gemm") {
                        out.push(d.kernel.clone());
                    }
                }
            }
        }
        (out, p.declared_launches, p.arena_bytes)
    };
    for tp in [1u32, 2] {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), tp).expect("graph");
        for rows in [96u32, 128] {
            let (kb, lb, ab) = plan(&g, rows, &base);
            let (ke, le, ae) = plan(&g, rows, &exact);
            let (kx, _, _) = plan(&g, rows, &band);
            assert_eq!(kb.len(), ke.len(), "tp{tp} p{rows}: the same linears");
            assert!(
                ke.iter()
                    .all(|k| k.starts_with("mxfp4_gemm_decode_bk") && k.contains("_ks1_")),
                "tp{tp} p{rows}: {ke:?}"
            );
            assert_eq!(le, lb, "tp{tp} p{rows}: launches");
            assert_eq!(ae, ab, "tp{tp} p{rows}: the arena (no split workspace)");
            assert!(
                kx.iter().any(|k| !k.contains("_ks1_")),
                "tp{tp} p{rows}: the band control never splits"
            );
        }
        for rows in [192u32, 256, 320] {
            assert_eq!(
                plan(&g, rows, &exact),
                plan(&g, rows, &base),
                "tp{tp} p{rows}"
            );
        }
    }
}

/// **G5: THE MID-M FOLDED TILES.** With the `folded_mt` knob on, the p192, p320 and p384 plans bind
/// the mid-M heights of the folded tile (`mxfp4_gemm_folded_mt_*`) for every folded linear whose N
/// has a row, each in the reference tile's summation class, with the SAME arena and launch count;
/// p96/p128 (the decode band here), p256 and p512 and the A-tiled rungs are the committed plans
/// byte for byte. Control: with the knob off (the committed table) no plan names the plugin.
#[test]
fn the_mid_rungs_plan_the_mid_m_folded_tiles_with_the_same_arena() {
    use saber_kernels::rules::summation_class;
    let base = LaunchRules::prod_mxfp4().expect("rules");
    let mut on = base.clone();
    on.knobs.folded_mt = true;
    let plan = |g: &Graph, rows: u32, r: &LaunchRules| -> (Vec<String>, u32, u64) {
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let p = BlockPlan::with_options(
            g,
            Bucket::Prefill {
                tokens: rows,
                seqs: 1,
            },
            &layers,
            RuntimeParams {
                max_ctx: 4096,
                gdn_slots: 65,
            },
            r,
            &SaberRules::prod_saber().expect("saber rules"),
            PlanOptions::BODY.with_batch_invariant_attn(),
        )
        .expect("a prefill body plan");
        let mut out = Vec::new();
        for s in &p.steps {
            if let Route::Kernel { descs, .. } = &s.route {
                for d in descs {
                    if d.kernel.starts_with("mxfp4_gemm") {
                        out.push(d.kernel.clone());
                    }
                }
            }
        }
        (out, p.declared_launches, p.arena_bytes)
    };
    for tp in [1u32, 2] {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), tp).expect("graph");
        for rows in [192u32, 320, 384] {
            let (kb, lb, ab) = plan(&g, rows, &base);
            let (km, lm, am) = plan(&g, rows, &on);
            assert_eq!(kb.len(), km.len(), "tp{tp} p{rows}: the same linears");
            assert!(
                kb.iter().all(|k| !k.starts_with("mxfp4_gemm_folded_mt")),
                "tp{tp} p{rows}: the committed table never binds the mid tiles"
            );
            let moved = kb.iter().zip(&km).filter(|(b, m)| b != m).count();
            assert!(moved > 0, "tp{tp} p{rows}: the knob moved nothing");
            for (b, m) in kb.iter().zip(&km) {
                assert_eq!(
                    summation_class(b),
                    summation_class(m),
                    "tp{tp} p{rows}: {b} -> {m}"
                );
            }
            assert_eq!(lm, lb, "tp{tp} p{rows}: launches");
            assert_eq!(am, ab, "tp{tp} p{rows}: the arena");
        }
        for rows in [96u32, 128, 256, 512, 768, 2048] {
            assert_eq!(plan(&g, rows, &on), plan(&g, rows, &base), "tp{tp} p{rows}");
        }
    }
}

/// **G6: THE LAYOUT LAW'S MID BAND AND THE SKINNY GATE.** With `atiled_mid` on (and G5's mid-M
/// folded tiles, the default, on in both arms), the p192..p512 plans write every fused producer's
/// edge in the fragment layout (`*_tiled1`, the glue's gate producer included) and bind the A-tiled
/// family for every layer linear reading one; with `skinny` on too the 48-wide gate takes the
/// skinny body at every prefill rung. Every GEMM stays in the fold class of the symbol it replaces
/// ([`saber_kernels::rules::fold_class`]), the arena and the launch count are unchanged, and the
/// rungs outside the band (p96 / p128, the decode kernel under C6's small-chunk rules; p768 /
/// p2048, already A-tiled) are byte-identical plans under `atiled_mid` alone. Control: the
/// committed table plans no tiled producer below 513 rows.
#[test]
fn the_mid_band_plans_tiled_producers_and_the_a_tiled_family_in_one_class() {
    use saber_kernels::rules::fold_class;
    let mut base = LaunchRules::prod_mxfp4().expect("rules");
    base.knobs.folded_mt = true;
    let mut on = base.clone();
    on.knobs.atiled_mid = true;
    let mut sk = on.clone();
    sk.knobs.skinny = true;
    // (every mxfp4 GEMM + producer + glue gate symbol, launches, arena bytes)
    let plan = |g: &Graph, rows: u32, r: &LaunchRules| -> (Vec<String>, u32, u64) {
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let p = BlockPlan::with_options(
            g,
            Bucket::Prefill {
                tokens: rows,
                seqs: 1,
            },
            &layers,
            RuntimeParams {
                max_ctx: 4096,
                gdn_slots: 65,
            },
            r,
            &SaberRules::prod_saber().expect("saber rules"),
            PlanOptions::BODY.with_batch_invariant_attn(),
        )
        .expect("a prefill body plan");
        let mut out = Vec::new();
        for s in &p.steps {
            if let Route::Kernel { descs, .. } = &s.route {
                for d in descs {
                    let k = &d.kernel;
                    if k.starts_with("mxfp4_gemm")
                        || k.contains("_quant_t")
                        || k.starts_with("attn_gate_quant")
                    {
                        out.push(k.clone());
                    }
                }
            }
        }
        (out, p.declared_launches, p.arena_bytes)
    };
    for tp in [1u32, 2] {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), tp).expect("graph");
        for rows in [192u32, 256, 320, 384, 512] {
            let (kb, lb, ab) = plan(&g, rows, &base);
            let (ko, lo, ao) = plan(&g, rows, &on);
            let (ks, ls, as_) = plan(&g, rows, &sk);
            assert_eq!(kb.len(), ko.len(), "tp{tp} p{rows}: the same ops");
            assert_eq!(kb.len(), ks.len(), "tp{tp} p{rows}: the same ops");
            assert!(
                kb.iter().all(|k| !k.ends_with("_tiled1")),
                "tp{tp} p{rows}: the committed law writes no tiled edge below 513"
            );
            for (b, o) in kb.iter().zip(&ko) {
                if b.starts_with("mxfp4_gemm") {
                    assert!(
                        o.starts_with("mxfp4_gemm_atiled"),
                        "tp{tp} p{rows}: {b} -> {o} is not the A-tiled family"
                    );
                    assert_eq!(fold_class(b), fold_class(o), "tp{tp} p{rows}: {b} -> {o}");
                    assert!(fold_class(o).is_some(), "tp{tp} p{rows}: {o}");
                } else {
                    assert!(
                        o.ends_with("_tiled1"),
                        "tp{tp} p{rows}: producer {b} -> {o}"
                    );
                }
            }
            let skinny = ks
                .iter()
                .filter(|k| k.starts_with("mxfp4_gemm_skinny_n"))
                .count();
            assert_eq!(
                skinny, 48,
                "tp{tp} p{rows}: one skinny gate per GDN layer ({ks:?})"
            );
            for (o, s) in ko.iter().zip(&ks) {
                assert_eq!(fold_class(o), fold_class(s), "tp{tp} p{rows}: {o} -> {s}");
            }
            assert_eq!((lo, ls), (lb, lb), "tp{tp} p{rows}: launches");
            assert_eq!((ao, as_), (ab, ab), "tp{tp} p{rows}: the arena");
        }
        for rows in [768u32, 2048] {
            assert_eq!(plan(&g, rows, &on), plan(&g, rows, &base), "tp{tp} p{rows}");
        }
    }
}
