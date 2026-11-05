//! CMP27S: the wave-parallel wht6 body (`--engine-ar-wpar`), on the host -- what the knob
//! resolves, at which grid, with which kernargs, where it keeps the served body, and that the
//! code objects carry the symbols at the served ABI. The card half (bytes) is CMP27R's oracle and
//! the engine gate under the arm.

use saber_kernels::manifest::Manifest;
use saber_kernels::prebuilt;
use saber_kernels::saber::{
    self, ArPayload, ArShape, Op, SaberRules, Shape, AR_CHECK_SUFFIX, AR_WPAR_MAX_CHUNKS_PER_BLOCK,
    AR_WPAR_SOURCE, AR_WPAR_SYMBOL,
};
use saber_kernels::LaunchDescriptor;

fn rules(wpar: bool, check: bool) -> SaberRules {
    let mut r = SaberRules::prod_saber().expect("saber rules");
    r.ar.wpar = wpar;
    r.ar.race_check = check;
    r
}

fn ar(rows: u64, wpar: bool, check: bool) -> LaunchDescriptor {
    saber::resolve(
        Op::AllReduce {
            dtype: ArPayload::Bf16,
        },
        Shape::Ar(ArShape {
            n_elem: rows * 5120,
            elem_bytes: 2,
            world_size: 2,
        }),
        &rules(wpar, check),
    )
    .expect("the rotated reduce resolves")
}

fn manifest(plugin: &str) -> Manifest {
    Manifest::load(prebuilt::dir().join(format!("{plugin}.manifest.toml"))).expect("manifest")
}

fn chunks_per_block(d: &LaunchDescriptor) -> u32 {
    let n_chunks = match d.arg("n_chunks").map(|a| &a.value) {
        Some(saber_kernels::rules::ArgValue::I32 { ival }) => {
            u32::try_from(*ival).expect("n_chunks")
        }
        other => panic!("n_chunks: {other:?}"),
    };
    n_chunks.div_ceil(d.grid[0])
}

/// The knob is OFF in the committed rules; ON, the b2/b4/b8 messages (16/32/64 rows) take the
/// wave-parallel symbol at the SAME grid, block, kernargs and regions -- a symbol swap -- and the
/// control (knob off) keeps the served one.
#[test]
fn the_knob_swaps_the_symbol_at_the_served_grid() {
    assert!(!rules(false, false).ar.wpar, "off in the committed rules");
    for rows in [16u64, 32, 40, 64] {
        let served = ar(rows, false, false);
        let wpar = ar(rows, true, false);
        assert_eq!(served.kernel, AR_WPAR_SOURCE, "{rows} rows");
        assert_eq!(wpar.kernel, AR_WPAR_SYMBOL, "{rows} rows");
        assert_eq!(wpar.plugin, "saber_ar_wht_wpar");
        assert_eq!(wpar.args, served.args, "{rows} rows: one ABI");
        assert_eq!(wpar.kernarg_bytes, served.kernarg_bytes);
        assert_eq!(wpar.grid, served.grid, "{rows} rows: the served grid");
        assert_eq!(wpar.block, served.block);
        assert_eq!(wpar.workspaces.len(), served.workspaces.len());
        for (a, b) in wpar.workspaces.iter().zip(&served.workspaces) {
            assert_eq!((&a.name, a.kind, a.bytes), (&b.name, b.kind, b.bytes));
        }
        assert!(chunks_per_block(&wpar) <= AR_WPAR_MAX_CHUNKS_PER_BLOCK);
    }
    // The b8 message's grid is CMP27R's probe grid (29 blocks, 6 chunks a block).
    let b8 = ar(64, true, false);
    assert_eq!(b8.grid[0], 29);
    assert_eq!(chunks_per_block(&b8), 6);
}

/// A message whose chunks a block exceed the LDS word buffer keeps the served body (the kernel
/// would trap), and the exact family below quant_min_bytes is untouched.
#[test]
fn past_the_lds_bound_the_served_body_stays() {
    let big = ar(2048, true, false);
    assert!(chunks_per_block(&big) > AR_WPAR_MAX_CHUNKS_PER_BLOCK);
    assert_eq!(big.kernel, AR_WPAR_SOURCE);
    assert_eq!(big.plugin, "saber_ar_wht");
    let exact = ar(8, true, false);
    assert_eq!(
        exact.plugin, "saber_ar_exact",
        "8 rows ride the exact reduce"
    );
    // Every row count up to 2048 routes by the bound and nothing else.
    for rows in 13u64..=2048 {
        let d = ar(rows, true, false);
        if d.plugin == "saber_ar_exact" {
            continue;
        }
        let fits = chunks_per_block(&d) <= AR_WPAR_MAX_CHUNKS_PER_BLOCK;
        assert_eq!(d.kernel == AR_WPAR_SYMBOL, fits, "{rows} rows");
    }
}

/// Under the race check the wave-parallel body takes its own `_chk` twin (same image); the code
/// objects hold both symbols at the served kernarg size, and their LDS fits the declared buffer.
#[test]
fn the_check_twin_and_the_code_objects() {
    let c = ar(64, true, true);
    assert_eq!(c.kernel, format!("{AR_WPAR_SYMBOL}{AR_CHECK_SUFFIX}"));
    assert_eq!(c.plugin, "saber_ar_wht_wpar_chk");
    assert_eq!(c.args, ar(64, true, false).args);
    let served = manifest("saber_ar_wht")
        .kernel(AR_WPAR_SOURCE)
        .expect("served symbol")
        .kernarg_segment_size;
    for (plugin, sym) in [
        ("saber_ar_wht_wpar", AR_WPAR_SYMBOL.to_owned()),
        (
            "saber_ar_wht_wpar_chk",
            format!("{AR_WPAR_SYMBOL}{AR_CHECK_SUFFIX}"),
        ),
    ] {
        let m = manifest(plugin);
        let k = m
            .kernel(&sym)
            .unwrap_or_else(|| panic!("{sym} is not in {plugin}"));
        assert_eq!(k.kernarg_segment_size, served, "{sym}");
        assert_eq!(k.private_segment_fixed_size, 0, "{sym}: no scratch");
        // 8 chunks x 32 lanes x 12 words + the sequence word (+ the check flag in the twin).
        assert!(
            k.group_segment_fixed_size >= AR_WPAR_MAX_CHUNKS_PER_BLOCK * 32 * 12 * 4,
            "{sym}: LDS {}",
            k.group_segment_fixed_size
        );
    }
    // The device bound is the host's.
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("hip/saber_ar_wht_wpar.hip"),
    )
    .expect("source");
    assert!(src.contains(&format!(
        "#define SABER_WPAR_CPB_MAX {AR_WPAR_MAX_CHUNKS_PER_BLOCK}"
    )));
}

/// The split form (`--ar-form split`) derives from the wpar descriptor exactly as from the
/// served one (the same pair, the same carried arguments).
#[test]
fn the_split_form_derives_from_either_body() {
    let (sp, sr) = saber::ar_wht_split(&ar(64, false, false), 16).expect("served split");
    let (wp, wr) = saber::ar_wht_split(&ar(64, true, false), 16).expect("wpar split");
    assert_eq!((sp.kernel, sp.args), (wp.kernel, wp.args));
    assert_eq!((sr.kernel, sr.args), (wr.kernel, wr.args));
}

/// The boot line's band: none with the knob off; on, one contiguous run of row counts holding the
/// b2 / b4 / b8 messages, ending where the chunks a block pass the LDS bound.
#[test]
fn the_row_band_is_one_run_holding_the_decode_messages() {
    assert_eq!(
        saber::ar_wpar_row_band(&rules(false, false), 5120, 2048),
        None
    );
    let (lo, hi, n) = saber::ar_wpar_row_band(&rules(true, false), 5120, 2048).expect("a band");
    assert_eq!(n, hi - lo + 1, "one run");
    assert!(lo <= 16 && hi >= 64, "{lo}..={hi}");
    assert!(hi < 2048, "the widest prefill chunk keeps the served body");
    assert_eq!(
        saber::ar_wpar_row_band(&rules(true, true), 5120, 2048),
        Some((lo, hi, n)),
        "the checked twin takes the same band"
    );
}
