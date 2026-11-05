//! The Paro producer rules, and their negative controls.
//!
//! Third plugin. The PARO tier is prod's ROLLBACK unit, so these producers are a serve path and
//! the same three things are checked as for the other two: `resolve` answers what the reference
//! launcher would have launched, the refusals are the entry points' own, and a case that must
//! fail is written down next to the case that must pass.

use saber_kernels::manifest::Manifest;
use saber_kernels::paro::{resolve, Op, ParoEwShape, ParoRules, ParoShape, Shape};
use saber_kernels::rules::{ArgValue, LaunchRules};
use saber_kernels::{prebuilt, Error};

fn rules() -> ParoRules {
    ParoRules::prod_paro().expect("prod paro rules parse")
}

const EPS: f32 = 1e-6;
const KROT: u32 = 2;

fn rms(m: u32, k: u32, p: u32) -> ParoShape {
    ParoShape {
        m,
        k,
        p,
        krot: KROT,
        eps_bits: EPS.to_bits(),
    }
}

fn ew(m: u32, n: u32) -> ParoEwShape {
    ParoEwShape {
        m,
        n,
        krot: KROT,
        ys: n as i64,
        eps_bits: EPS.to_bits(),
    }
}

// ---------------------------------------------------------------------------- rules file

#[test]
fn prod_rules_load_and_every_symbol_is_committed() {
    let r = rules();
    assert_eq!(r.name, "paro-gfx1201-paromx");
    assert_eq!(r.knobs.decode_max_m, 64);
    // The band edge is NOT a knob of this table (see `the_tiled_threshold_is_the_mxfp4_one`).
    assert!(r.tiled(513) && !r.tiled(512));
    assert!(
        !r.knobs.i8,
        "paromx serves int4; the int5 tier sets SABER_PQ_I8"
    );
    assert!(
        r.knobs.hw_cvt,
        "PQ_HW_CVT=1: prod exports no SABER_PQ_HIPCC_FLAGS"
    );
    let mut named: Vec<&str> = Vec::new();
    named.extend(r.add_rms_rot.iter().map(|x| x.symbol.as_str()));
    named.extend(r.ew_rot.iter().map(|x| x.symbol.as_str()));
    named.extend(r.add_rms_rot_tok.iter().map(|x| x.symbol.as_str()));
    named.extend(r.ew_rot_tok.iter().map(|x| x.symbol.as_str()));
    assert_eq!(named.len(), 36);
    let m = Manifest::load(prebuilt::dir().join("paro_producers.manifest.toml")).expect("manifest");
    assert_eq!(
        m.kernel.len(),
        36,
        "the blob exports exactly what the rules can name"
    );
    for s in named {
        assert!(
            m.kernel.iter().any(|k| k.name == s),
            "no committed kernel `{s}`"
        );
    }
}

/// The per-token producer writes the fragment layout the A-tiled GEMM reads, and the two take the
/// layout decision from the SAME threshold. In prod they live in two files and agree by
/// convention; here they agree because there is only one number
/// (`saber_kernels::rules::a_tiled_min_m`) and this table does not carry a copy of it.
#[test]
fn the_tiled_threshold_is_the_mxfp4_one() {
    let paro = rules();
    let mxfp4 = LaunchRules::prod_mxfp4().expect("mxfp4 rules");
    let edge = mxfp4.knobs.a_tiled_min_m;
    for m in [1u32, edge - 1, edge, edge + 1, 8192] {
        assert_eq!(
            paro.tiled(m),
            m >= edge,
            "producer and consumer must switch layout at the same M (M={m})"
        );
    }
}

/// NEGATIVE CONTROL for the single source: a paro table that states the band edge again is
/// refused BY NAME, rather than shadowing the one number. The drift test this replaces passed
/// whenever two copies agreed -- which is the state L15's `a_tiled_min_m = 99999` control was in
/// when it moved one of them and measured a different layout mismatch (P6).
#[test]
fn a_paro_table_that_restates_the_band_edge_is_refused() {
    let text = include_str!("../rules/paro_gfx1201.toml");
    assert!(
        !text.contains("a_tiled_min_m ="),
        "the committed paro table must not carry the band edge at all"
    );
    for v in [513u32, 99999] {
        let bad = text.replace(
            "decode_max_m = 64",
            &format!("decode_max_m = 64\na_tiled_min_m = {v}"),
        );
        assert_ne!(bad, text, "the mutation has to bite");
        let e = ParoRules::from_toml(&bad).expect_err("a restated band edge must not load");
        assert!(e.to_string().contains("a_tiled_min_m"), "{e}");
        assert!(e.to_string().contains("one source"), "{e}");
    }
}

/// NEGATIVE CONTROL: a rules file whose invariants disagree is refused at load rather than
/// mis-routing a shape at run time.
#[test]
fn rules_that_break_an_invariant_are_refused() {
    let text = include_str!("../rules/paro_gfx1201.toml");
    // The block is PQ_ROT_WAVES * 32 by construction.
    let bad = text.replace("rot_block = 256", "rot_block = 512");
    match ParoRules::from_toml(&bad) {
        Err(Error::Rule(m)) => assert!(m.contains("rot_block"), "{m}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
    // The tiled threshold has to be above the decode band, or a shape would be both. The
    // threshold is the MXFP4 rules' now, so the half of the pair this file owns is the one that
    // moves: a decode band that swallowed the band edge is refused.
    let bad = text.replace(
        "decode_max_m = 64",
        &format!("decode_max_m = {}", saber_kernels::rules::a_tiled_min_m()),
    );
    assert_ne!(bad, text, "the mutation has to bite");
    assert!(
        ParoRules::from_toml(&bad).is_err(),
        "a_tiled_min_m vs decode_max_m"
    );
}

// ---------------------------------------------------------------------------- the two ladders

#[test]
fn the_two_measured_ladders_are_tables() {
    let r = rules();
    // pq_rot_gpw: 1 at single-stream M where the serial rotation chain is the cost, 2 above.
    for (m, gpw) in [(1u32, 1u32), (40, 1), (41, 2), (2048, 2)] {
        assert_eq!(r.rot_gpw(m), gpw, "gpw at M={m}");
    }
    // pq_tok_waves(M, K), first match wins. The K row is the tokqt gate: K=8704 wants 16 at
    // large M where K=5120 wants 8.
    for (m, k, w) in [
        (1u32, 5120u32, 32u32),
        (16, 5120, 32),
        (17, 5120, 16),
        (64, 5120, 16),
        (65, 5120, 8),
        (65, 8704, 16),
        (2048, 5120, 8),
        (2048, 8704, 16),
    ] {
        assert_eq!(r.tok_waves(m, k), w, "waves at M={m} K={k}");
    }
    // `fused` and `_tiled`, both derived from M exactly as prod derives them.
    assert!(r.rot(64) && !r.rot(65));
    assert!(!r.tiled(512) && r.tiled(513));
}

// ---------------------------------------------------------------------------- per-group family

#[test]
fn the_per_group_producer_fuses_the_rotation_only_in_the_decode_band() {
    let r = rules();
    // M <= 64: the producer also rotates and per-group quantises.
    let d = resolve(Op::AddRmsRot, Shape::Rms(rms(8, 5120, 3)), &r).expect("resolve");
    assert_eq!(d.kernel, "paro_add_rms_rot_rot1_i80");
    // groups = 5120/128 = 40, gpw = 1 at M=8, rot_waves = 8 -> ceil(40/8) = 5; P is the z axis.
    assert_eq!(d.grid, [8, 5, 3]);
    assert_eq!(d.block, [256, 1, 1]);
    assert_eq!(d.shared_bytes, 0);
    assert_eq!(d.kernarg_bytes, 104);
    assert_eq!(d.n_launches, 1);
    assert!(
        d.workspaces.is_empty(),
        "every buffer is the layer's own tensor"
    );
    assert!(matches!(
        d.arg("a").map(|s| s.value.clone()),
        Some(ArgValue::Buffer { .. })
    ));
    assert_eq!(
        d.arg("gpw").map(|s| s.value.clone()),
        Some(ArgValue::I32 { ival: 1 })
    );

    // gpw = 2 above M=40 halves the second grid axis.
    let d = resolve(Op::AddRmsRot, Shape::Rms(rms(64, 5120, 1)), &r).unwrap();
    assert_eq!(d.grid, [64, 3, 1], "ceil(40 / (8*2))");
    assert_eq!(
        d.arg("gpw").map(|s| s.value.clone()),
        Some(ArgValue::I32 { ival: 2 })
    );

    // M > 64: the plain arm, one workgroup per row, and the three quant outputs are NOT written.
    let d = resolve(Op::AddRmsRot, Shape::Rms(rms(128, 5120, 3)), &r).unwrap();
    assert_eq!(d.kernel, "paro_add_rms_rot_rot0_i80");
    assert_eq!(d.grid, [128, 1, 1]);
    for n in ["a", "asg", "rs"] {
        assert!(
            matches!(
                d.arg(n).map(|s| s.value.clone()),
                Some(ArgValue::NullBuffer { .. })
            ),
            "`{n}` is not written on the ROT=false arm"
        );
    }
}

#[test]
fn the_elementwise_producer_has_one_symbol_per_consumer() {
    let r = rules();
    for (mode, n, blocks) in [(0u32, 5120u32, 3u32), (1, 5120, 3), (2, 3072, 2)] {
        let d = resolve(Op::EwRot { mode }, Shape::Ew(ew(64, n)), &r).expect("resolve");
        assert_eq!(d.kernel, format!("paro_ew_rot_m{mode}_rot1_i80"));
        // groups = n/128, gpw = 2 at M=64 -> ceil(groups / 16).
        assert_eq!(d.grid, [64, blocks, 1]);
        assert_eq!(d.block, [256, 1, 1]);
        assert_eq!(d.kernarg_bytes, 104);
        assert_eq!(d.n_launches, 1);
    }
    // The reference forces gpw = 1 on the plain arm, so the grid does NOT halve there.
    let d = resolve(Op::EwRot { mode: 0 }, Shape::Ew(ew(128, 5120)), &r).unwrap();
    assert_eq!(d.kernel, "paro_ew_rot_m0_rot0_i80");
    assert_eq!(
        d.grid,
        [128, 5, 1],
        "ceil(40 / (8*1)) -- gpw is 1 when not fused"
    );
    assert_eq!(
        d.arg("gpw").map(|s| s.value.clone()),
        Some(ArgValue::I32 { ival: 1 })
    );
    assert!(resolve(Op::EwRot { mode: 3 }, Shape::Ew(ew(8, 5120)), &r).is_err());
}

// ---------------------------------------------------------------------------- per-token family

#[test]
fn the_per_token_producer_picks_its_waves_and_its_layout_from_m() {
    let r = rules();
    // (M, K, waves, tiled). This is the tier paromx actually serves.
    for (m, k, w, tiled) in [
        (1u32, 5120u32, 32u32, false),
        (8, 5120, 32, false),
        (64, 5120, 16, false),
        (128, 5120, 8, false),
        (128, 8704, 16, false),
        (2048, 5120, 8, true),
        (2048, 8704, 16, true),
    ] {
        let d = resolve(Op::AddRmsRotTok, Shape::Rms(rms(m, k, 3)), &r).expect("resolve");
        assert_eq!(
            d.kernel,
            format!("paro_add_rms_rot_tok_w{w}_tiled{}", u8::from(tiled)),
            "M={m} K={k}"
        );
        assert_eq!(d.grid, [m, 3, 1], "one workgroup per (token, partition)");
        assert_eq!(d.block, [w * 32, 1, 1]);
        // The whole rotated row is staged in dynamic LDS.
        assert_eq!(d.shared_bytes, k * 2);
        assert_eq!(d.kernarg_bytes, 92);
        assert_eq!(d.n_launches, 1);
        assert!(d.workspaces.is_empty());
    }
    let d = resolve(Op::EwRotTok { mode: 2 }, Shape::Ew(ew(2048, 3072)), &r).unwrap();
    assert_eq!(d.kernel, "paro_ew_rot_tok_m2_w8_tiled1");
    assert_eq!(
        d.grid,
        [2048, 1, 1],
        "single partition: all three consumers are P = 1"
    );
    assert_eq!(d.shared_bytes, 3072 * 2);
    assert_eq!(d.kernarg_bytes, 92);
}

// ---------------------------------------------------------------------------- refusals

/// NEGATIVE CONTROLS: the conditions the reference entry points throw on, in their own words.
#[test]
fn the_entry_points_own_refusals_are_reproduced() {
    let r = rules();
    // K / N must be a multiple of PQ_GROUP.
    let e = resolve(Op::AddRmsRot, Shape::Rms(rms(8, 5000, 1)), &r).unwrap_err();
    assert!(
        format!("{e}").contains("K must be a multiple of 128"),
        "{e}"
    );
    let e = resolve(Op::EwRot { mode: 0 }, Shape::Ew(ew(8, 3000)), &r).unwrap_err();
    assert!(
        format!("{e}").contains("N must be a multiple of 128"),
        "{e}"
    );
    // krot in [1, PQ_KROT_MAX].
    for bad in [0u32, 9] {
        let mut s = rms(8, 5120, 1);
        s.krot = bad;
        let e = resolve(Op::AddRmsRot, Shape::Rms(s), &r).unwrap_err();
        assert!(format!("{e}").contains("krot out of range"), "{e}");
    }
    // The per-token launchers stage K*2 bytes of LDS and refuse above 60 KiB. 30848*2 = 61696.
    let e = resolve(Op::AddRmsRotTok, Shape::Rms(rms(8, 30848, 1)), &r).unwrap_err();
    assert!(format!("{e}").contains("LDS"), "{e}");
    // ... and the per-GROUP producer has no such limit: it never stages the row.
    assert!(resolve(Op::AddRmsRot, Shape::Rms(rms(8, 30848, 1)), &r).is_ok());
    // P must be at least one rotated copy.
    assert!(resolve(Op::AddRmsRot, Shape::Rms(rms(8, 5120, 0)), &r).is_err());
    // An op handed the wrong KIND of shape is refused rather than silently reinterpreted.
    assert!(resolve(Op::AddRmsRot, Shape::Ew(ew(8, 5120)), &r).is_err());
    assert!(resolve(Op::EwRot { mode: 0 }, Shape::Rms(rms(8, 5120, 1)), &r).is_err());
}

/// NEGATIVE CONTROL: `ROT=false` never carries `I8` — the reference's ladder is `fused && i8` /
/// `fused` / else, so there is no such row and asking for one must fail rather than silently
/// picking the i8=false arm. Exercised through a rules file with the int5 tier's knob.
#[test]
fn the_plain_arm_has_no_i8_instantiation() {
    let text = include_str!("../rules/paro_gfx1201.toml");
    // replacen(.., 1): `[knobs]` comes first, and `i8 = false` also names instantiation ROWS --
    // replacing those would rewrite the table rather than the tier's knob.
    let int5 =
        ParoRules::from_toml(&text.replacen("i8 = false", "i8 = true", 1)).expect("int5 knobs");
    // In the decode band the i8 arm exists...
    assert_eq!(
        resolve(Op::AddRmsRot, Shape::Rms(rms(8, 5120, 1)), &int5)
            .unwrap()
            .kernel,
        "paro_add_rms_rot_rot1_i81"
    );
    assert_eq!(
        resolve(Op::EwRot { mode: 1 }, Shape::Ew(ew(8, 5120)), &int5)
            .unwrap()
            .kernel,
        "paro_ew_rot_m1_rot1_i81"
    );
    // ... and above it the plain arm is taken with i8 off, which is what the reference does.
    assert_eq!(
        resolve(Op::AddRmsRot, Shape::Rms(rms(128, 5120, 1)), &int5)
            .unwrap()
            .kernel,
        "paro_add_rms_rot_rot0_i80"
    );
    // A rules file that DOES offer the impossible row is the control: with it removed, a lookup
    // for rot=false i8=true has nothing to find.
    let broken = text.replacen("i8 = false", "i8 = true", 1).replace(
        r#"[[add_rms_rot]]
rot = false
i8 = false
symbol = "paro_add_rms_rot_rot0_i80""#,
        r#"[[add_rms_rot]]
rot = false
i8 = true
symbol = "paro_add_rms_rot_rot0_i81""#,
    );
    let r = ParoRules::from_toml(&broken).expect("parses");
    let e = resolve(Op::AddRmsRot, Shape::Rms(rms(128, 5120, 1)), &r).unwrap_err();
    assert!(format!("{e}").contains("no rot=false i8=false"), "{e}");
}
