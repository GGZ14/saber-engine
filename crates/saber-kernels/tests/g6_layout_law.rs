//! G6: the layout law below `a_tiled_min_m`, the A-tiled mid-M heights and the skinny-N body.
//!
//! Host-only: what the rules BIND. The bytes are gated on the card (`tools/capture/g6`, and
//! `step-tp2 --mid-chunk-atiled --skinny-gate` for the plan).

use saber_kernels::glue::GlueRules;
use saber_kernels::producer::AProducer;
use saber_kernels::rules::{
    fold_class, producer_writes_tiled, resolve, resolve_with_layout, summation_class, ALayout,
    ATiledLaw, LaunchRules, Op, Shape, MID_PLUGIN,
};

const PROD_RULES: &str = include_str!("../rules/mxfp4_gfx1201.toml");

fn committed() -> LaunchRules {
    LaunchRules::prod_mxfp4().expect("rules")
}

fn mid_on() -> LaunchRules {
    let mut r = committed();
    r.knobs.atiled_mid = true;
    r
}

/// The TP=2 and TP=1 layer linears of Qwen3.8-27B (N, K).
const LINEARS: [(u32, u32); 11] = [
    (17408, 5120),
    (5120, 8704),
    (8192, 5120),
    (7168, 5120),
    (5120, 3072),
    (48, 5120),
    (34816, 5120),
    (5120, 17408),
    (16384, 5120),
    (14336, 5120),
    (96, 5120),
];

/// The committed law is `a_tiled_min_m` alone; the knob adds the mid band and nothing else; and
/// `a_tiled_min_m = 0` (the sweepers' `ALayout::RowMajor`) still means row-major EVERYWHERE, mid
/// band included -- the control for "the band is additive to the old number, not a new number".
#[test]
fn the_layout_law_is_the_old_edge_plus_the_mid_band_under_its_knob() {
    let c = ATiledLaw::committed();
    assert_eq!(c, ATiledLaw::of(&committed()));
    assert_eq!(
        c.mid_min_m, None,
        "the committed table has the mid band OFF"
    );
    assert!(!c.tiled(512) && c.tiled(513) && c.tiled(8192));
    assert_eq!(c.edge(), 513);
    let on = ATiledLaw::of(&mid_on());
    assert_eq!(on.mid_min_m, Some(129));
    assert!(!on.tiled(128) && on.tiled(129) && on.tiled(320) && on.tiled(512) && on.tiled(513));
    assert_eq!(on.edge(), 129);
    let mut rm = mid_on();
    rm.knobs.a_tiled_min_m = 0;
    let rm = ATiledLaw::of(&rm);
    assert!(
        !rm.tiled(129) && !rm.tiled(4096),
        "a_tiled_min_m = 0 is row-major at every M"
    );
}

/// The mid band sits strictly between the decode band and `a_tiled_min_m`, and an `atiled_mid`
/// with no edge is refused rather than read as "no band".
#[test]
fn the_rules_refuse_a_mid_band_outside_its_place() {
    for (bad, why) in [
        ("atiled_mid_min_m = 129", "atiled_mid_min_m = 64"),
        ("atiled_mid_min_m = 129", "atiled_mid_min_m = 513"),
        ("atiled_mid_min_m = 129", "atiled_mid_min_m = 700"),
    ] {
        let t = PROD_RULES.replace(bad, why);
        assert!(LaunchRules::from_toml(&t).is_err(), "{why} must be refused");
    }
    let t = PROD_RULES
        .replace("atiled_mid = false", "atiled_mid = true")
        .replace("atiled_mid_min_m = 129", "atiled_mid_min_m = 0");
    assert!(
        LaunchRules::from_toml(&t).is_err(),
        "a mid band with no edge"
    );
    // and the committed band itself parses with the knob on
    let t = PROD_RULES.replace("atiled_mid = false", "atiled_mid = true");
    assert!(LaunchRules::from_toml(&t).is_ok());
}

/// Every producer, the glue's included, follows the law of the rules it is asked under: row-major
/// in the mid band on the committed table, tiled there with the knob on, identical elsewhere.
#[test]
fn every_producer_writes_the_mid_band_tiled_under_the_knob_and_only_under_it() {
    let glue = GlueRules::prod().expect("glue");
    let (base, on) = (committed(), mid_on());
    for p in [
        AProducer::AddRmsQuant,
        AProducer::SiluMulQuant,
        AProducer::GdnNormQuant,
        AProducer::AttnGateQuant,
    ] {
        for m in [129u32, 192, 256, 320, 384, 512] {
            assert!(!p.writes_tiled(m, &base, &glue), "{p:?} M={m} committed");
            assert!(p.writes_tiled(m, &on, &glue), "{p:?} M={m} mid band");
        }
        for m in [1u32, 64, 96, 128, 513, 2048, 8192] {
            assert_eq!(
                p.writes_tiled(m, &base, &glue),
                p.writes_tiled(m, &on, &glue),
                "{p:?} M={m} outside the band"
            );
        }
    }
    // The fused mxfp4 producers' kernels are the tiled1 instantiations there (they exist: the
    // reference compiles both stores for every block size).
    for m in [192u32, 320] {
        let d = resolve(Op::AddRmsQuant { eps: 1e-6 }, Shape::new(m, 0, 5120), &on).unwrap();
        assert!(d.kernel.ends_with("_tiled1"), "{}", d.kernel);
        assert_eq!(producer_writes_tiled(Op::SiluMulQuant, m, &on), Some(true));
    }
}

/// With the knob on, a mid-rung GEMM over a tiled edge takes the A-tiled family -- a
/// `[[atiled_mt]]` height where a row names one, else the reference's A-tiled tile -- and every
/// symbol is in the fold class of the one it replaces. A row-major edge, the rungs outside the
/// band and the committed table are unchanged. Negative control: the split decode cell is NOT in
/// the class.
#[test]
fn the_mid_rungs_take_the_a_tiled_family_in_one_fold_class() {
    let (base, on) = (committed(), mid_on());
    let mut moved = 0;
    for &(n, k) in &LINEARS {
        for m in [
            129u32, 160, 192, 200, 256, 257, 300, 320, 333, 384, 400, 512,
        ] {
            let s = Shape::new(m, n, k);
            let b = resolve(Op::gemm(), s, &base).unwrap();
            let o = resolve(Op::gemm(), s, &on).unwrap();
            assert!(
                o.plugin == MID_PLUGIN || o.kernel.starts_with("mxfp4_gemm_atiled_"),
                "M={m} N={n}: {} is not the A-tiled family",
                o.kernel
            );
            assert_eq!(
                fold_class(&b.kernel),
                Some("mxfp4-fold-kasc"),
                "M={m} N={n}: {}",
                b.kernel
            );
            assert_eq!(fold_class(&o.kernel), fold_class(&b.kernel), "M={m} N={n}");
            assert_eq!(o.kernarg_bytes, b.kernarg_bytes, "M={m} N={n}");
            moved += usize::from(o.kernel != b.kernel);
            // The edge that carries row-major bytes never reaches the A-tiled family.
            let rm = resolve_with_layout(Op::gemm(), s, &on, ALayout::RowMajor).unwrap();
            assert_eq!(rm.kernel, b.kernel, "M={m} N={n}: a row-major edge");
            // The grid covers the tile's own height.
            if o.plugin == MID_PLUGIN {
                let row = on
                    .atiled_mt
                    .iter()
                    .find(|r| o.kernel == r.kernel())
                    .expect("a row named it");
                assert_eq!(o.grid[1], m.div_ceil(row.rows()), "M={m} N={n}");
                assert_eq!(o.block, [row.wm * 64, 1, 1]);
            }
        }
        for m in [1u32, 5, 40, 64, 96, 128, 513, 1024, 2048, 8192] {
            let s = Shape::new(m, n, k);
            assert_eq!(
                resolve(Op::gemm(), s, &on).unwrap().kernel,
                resolve(Op::gemm(), s, &base).unwrap().kernel,
                "M={m} N={n}: outside the mid band"
            );
        }
    }
    assert!(moved > 100, "the knob moved {moved} cells");
    // The class is not everything: a SPLIT decode cell reassociates.
    let split = resolve(Op::gemm(), Shape::new(8, 48, 5120), &base).unwrap();
    assert!(split.kernel.contains("_ks4_"), "{}", split.kernel);
    assert_eq!(fold_class(&split.kernel), None);
    assert_eq!(
        summation_class("mxfp4_gemm_atiled_mt_tn2_lbk64_wperm1_t5w4p"),
        "mxfp4_gemm_atiled_tn2_lbk128_wperm1"
    );
}

/// The skinny body under its knob: the gate widths at every prefill rung, over the layout the edge
/// carries; never a split decode cell, never a wide N; the committed table binds none of it.
#[test]
fn the_skinny_body_takes_the_gate_and_nothing_else() {
    let base = committed();
    let mut sk = committed();
    sk.knobs.skinny = true;
    let mut both = sk.clone();
    both.knobs.atiled_mid = true;
    for n in [48u32, 96] {
        for m in [65u32, 96, 128, 129, 192, 320, 384, 512, 513, 768, 2048] {
            let s = Shape::new(m, n, 5120);
            let b = resolve(Op::gemm(), s, &base).unwrap();
            assert!(!b.kernel.starts_with("mxfp4_gemm_skinny"), "committed");
            let r = resolve(Op::gemm(), s, &sk).unwrap();
            let t = resolve(Op::gemm(), s, &both).unwrap();
            if m > base.knobs.decode_max_m {
                assert_eq!(r.plugin, MID_PLUGIN, "M={m} N={n}");
                assert!(r.kernel.starts_with("mxfp4_gemm_skinny_n"), "{}", r.kernel);
                let want = if m >= 513 { "t_" } else { "r_" };
                assert!(r.kernel.contains(want), "M={m}: {}", r.kernel);
                assert!(
                    t.kernel.contains(if m >= 129 { "t_" } else { "r_" }),
                    "{}",
                    t.kernel
                );
                assert_eq!(fold_class(&r.kernel), fold_class(&b.kernel), "M={m} N={n}");
            }
        }
        // The decode steps keep their split cell.
        for m in [1u32, 8, 40, 64] {
            let s = Shape::new(m, n, 5120);
            assert_eq!(
                resolve(Op::gemm(), s, &sk).unwrap().kernel,
                resolve(Op::gemm(), s, &base).unwrap().kernel
            );
        }
    }
    // No row above the timed 2048 rows: the 8192 rung keeps the A-tiled tile.
    let d = resolve(Op::gemm(), Shape::new(8192, 48, 5120), &sk).unwrap();
    assert!(!d.kernel.starts_with("mxfp4_gemm_skinny"), "{}", d.kernel);
    // A wide N never takes it.
    for &(n, k) in &LINEARS[..5] {
        let d = resolve(Op::gemm(), Shape::new(320, n, k), &sk).unwrap();
        assert!(!d.kernel.starts_with("mxfp4_gemm_skinny"), "N={n}");
    }
}
