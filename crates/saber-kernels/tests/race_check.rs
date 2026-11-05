//! T4's race-check twins, on the host: what `--engine-race-check` resolves, what it refuses, and
//! that the twins are the served kernels' ABI in their own code objects.
//!
//! The card half (a checked run computes the served bytes; the checks fire on a real fault) is the
//! engine gate under `--race-check` and `ar_gate`; this file is what can be held without a GPU.

use saber_kernels::manifest::Manifest;
use saber_kernels::prebuilt;
use saber_kernels::rules::{gemm_decode_ar_nq, GEMM_AR_NQ_CHK_PLUGIN};
use saber_kernels::saber::{
    self, ArNqShape, ArPayload, ArShape, Op, SaberRules, Shape, AR_CHECKED_SYMBOLS,
    AR_CHECK_DIAG_OFFSET, AR_CHECK_DIAG_WORDS, AR_CHECK_RESERVE_BYTES, AR_CHECK_SEQ_BYTES,
    AR_CHECK_SUFFIX,
};
use saber_kernels::{LaunchDescriptor, LaunchRules, Op as GemmOp, Shape as GemmShape};

fn rules(check: bool) -> SaberRules {
    let mut r = SaberRules::prod_saber().expect("saber rules");
    r.ar.race_check = check;
    r
}

fn ar(rows: u64, check: bool) -> Result<LaunchDescriptor, saber_kernels::Error> {
    saber::resolve(
        Op::AllReduce {
            dtype: ArPayload::Bf16,
        },
        Shape::Ar(ArShape {
            n_elem: rows * 5120,
            elem_bytes: 2,
            world_size: 2,
        }),
        &rules(check),
    )
}

fn nq(check: bool) -> LaunchDescriptor {
    saber::resolve(
        Op::AllReduceExactNq,
        Shape::ArNq(ArNqShape {
            m_rows: 8,
            k_cols: 5120,
            eps_bits: 1e-6f32.to_bits(),
            world_size: 2,
        }),
        &rules(check),
    )
    .expect("the `_nq` epilogue resolves")
}

fn manifest(plugin: &str) -> Manifest {
    Manifest::load(prebuilt::dir().join(format!("{plugin}.manifest.toml"))).expect("manifest")
}

/// Every served reduce the engine resolves -- the exact family at a decode step, the rotated one at
/// a batch, the `_nq` epilogue -- becomes its `_chk` twin in the twin's own object, with the SAME
/// kernarg image (the twins take the served signature), and the control resolves the served one.
#[test]
fn the_check_resolves_every_served_reduce_to_its_twin_with_the_same_image() {
    for (rows, family) in [(8u64, "saber_ar_exact"), (64, "saber_ar_wht")] {
        let served = ar(rows, false).expect("served");
        let checked = ar(rows, true).expect("checked");
        assert_eq!(served.plugin, family, "{rows} rows");
        assert!(!served.kernel.ends_with(AR_CHECK_SUFFIX));
        assert_eq!(
            checked.kernel,
            format!("{}{AR_CHECK_SUFFIX}", served.kernel)
        );
        assert_eq!(checked.plugin, format!("{family}_chk"));
        assert_eq!(checked.args, served.args, "{rows} rows: one ABI");
        assert_eq!(checked.kernarg_bytes, served.kernarg_bytes);
        assert_eq!(checked.grid, served.grid);
        assert_eq!(checked.block, served.block);
        // The same regions, and only `ar_seq` grows: its tail holds the twin's diagnostic words.
        assert_eq!(checked.workspaces.len(), served.workspaces.len());
        for (c, w) in checked.workspaces.iter().zip(&served.workspaces) {
            assert_eq!((&c.name, c.kind), (&w.name, w.kind));
            if c.name == "ar_seq" {
                assert_eq!(w.bytes, 2048, "the served size is the rules file's");
                assert_eq!(c.bytes, AR_CHECK_SEQ_BYTES);
            } else {
                assert_eq!(c.bytes, w.bytes, "{}", c.name);
            }
        }
        let m = manifest(&checked.plugin);
        let k = m
            .kernel(&checked.kernel)
            .unwrap_or_else(|| panic!("{} is not in {}", checked.kernel, checked.plugin));
        assert_eq!(
            k.kernarg_segment_size,
            manifest(&served.plugin)
                .kernel(&served.kernel)
                .expect("served symbol")
                .kernarg_segment_size
        );
    }
    let (s, c) = (nq(false), nq(true));
    assert_eq!(s.kernel, "saber_ar_oneshot_2rank_exact_nq_ssqfma_bf16");
    assert_eq!(c.kernel, format!("{}{AR_CHECK_SUFFIX}", s.kernel));
    assert_eq!(c.plugin, "saber_ar_exact_chk");
    assert_eq!(c.args, s.args);
    assert!(manifest(&c.plugin).kernel(&c.kernel).is_some());
}

/// Every twin the rules can name is compiled, and every compiled twin is one the rules name.
#[test]
fn every_checked_symbol_has_a_compiled_twin() {
    for sym in AR_CHECKED_SYMBOLS {
        let twin = format!("{sym}{AR_CHECK_SUFFIX}");
        let plugin = saber::plugin_for(&twin).expect("a twin's plugin");
        assert!(
            manifest(plugin).kernel(&twin).is_some(),
            "{twin} is not a kernel of {plugin}"
        );
    }
    // Control: a served symbol does not map to a twin object.
    assert_eq!(
        saber::plugin_for("saber_ar_oneshot_2rank_exact_bf16").expect("served"),
        "saber_ar_exact"
    );
}

/// A reduce with no twin is a refusal BY NAME, never a quiet unchecked launch.
#[test]
fn a_reduce_with_no_twin_is_refused_by_name() {
    let d = |check| {
        saber::resolve(
            Op::AllReduceExact {
                dtype: ArPayload::Fp32,
            },
            Shape::Ar(ArShape {
                n_elem: 1024,
                elem_bytes: 4,
                world_size: 2,
            }),
            &rules(check),
        )
    };
    assert!(d(false).is_ok(), "control: the served fp32 reduce resolves");
    let e = d(true).expect_err("no fp32 twin");
    assert!(e.to_string().contains("no race-check twin"), "{e}");
}

/// A message that would reach the check words at the slot's tail is refused under the check, and
/// the same message resolves unchecked.
#[test]
fn a_message_that_reaches_the_check_words_is_refused() {
    let max = rules(false).ar.max_bytes;
    let big = saber::resolve(
        Op::AllReduceExact {
            dtype: ArPayload::Bf16,
        },
        Shape::Ar(ArShape {
            n_elem: (max - 16) / 2,
            elem_bytes: 2,
            world_size: 2,
        }),
        &rules(true),
    );
    let e = big.expect_err("a slot-filling message under the check");
    assert!(e.to_string().contains("check words"), "{e}");
    // Control: the largest prefill chunk the engine sends (2048 rows) is far below the reserve.
    assert!(ar(2048, true).is_ok());
    assert!(2048 * 5120 * 2 < max - AR_CHECK_RESERVE_BYTES);
}

/// Row 3 folds a CHECKED `_nq` into its checked twin, never into the served GEMM tail, and the
/// served `_nq` into the served one (the control).
#[test]
fn row3_folds_a_checked_epilogue_into_its_checked_twin() {
    let lr = LaunchRules::prod_mxfp4().expect("mxfp4 rules");
    let shape = GemmShape::new(8, 5120, 8704);
    let g = saber_kernels::rules::resolve(GemmOp::gemm(), shape, &lr).expect("gemm");
    let served = gemm_decode_ar_nq((&g, shape), &nq(false), &lr).expect("served row 3");
    let checked = gemm_decode_ar_nq((&g, shape), &nq(true), &lr).expect("checked row 3");
    assert_eq!(served.plugin, "mxfp4_gemm_decode_arnq");
    assert_eq!(checked.plugin, GEMM_AR_NQ_CHK_PLUGIN);
    assert_eq!(
        checked.kernel,
        format!("{}{AR_CHECK_SUFFIX}", served.kernel)
    );
    assert_eq!(checked.args, served.args);
    assert!(manifest(&checked.plugin).kernel(&checked.kernel).is_some());
}

/// The diagnostic words sit past the counters the served kernels index and inside the region, and
/// the device header states the same geometry the host reads.
#[test]
fn the_diag_words_and_the_reserve_agree_with_the_header() {
    let r = rules(false);
    // The served file is untouched (an edit to it would re-key every saber autotune record).
    assert_eq!(
        r.workspace.get("ar_seq").expect("ar_seq").capacity_bytes,
        2048
    );
    let seq = nq(true)
        .workspaces
        .iter()
        .find(|w| w.name == "ar_seq")
        .expect("ar_seq")
        .bytes;
    assert!(u64::from(r.ar.max_blocks) * 4 <= AR_CHECK_DIAG_OFFSET);
    assert!(AR_CHECK_DIAG_OFFSET + AR_CHECK_DIAG_WORDS as u64 * 4 <= seq);
    let h = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("hip/ar_race_check.h"),
    )
    .expect("header");
    let def = |name: &str| -> String {
        h.lines()
            .find_map(|l| l.strip_prefix(&format!("#define {name} ")))
            .unwrap_or_else(|| panic!("{name} not defined"))
            .split("//")
            .next()
            .unwrap_or_default()
            .trim()
            .to_owned()
    };
    assert_eq!(def("CHK_DIAG"), format!("{}u", AR_CHECK_DIAG_OFFSET / 4));
    assert_eq!(def("CHK_RESERVE_BYTES"), "(512L * 128L * 4L)");
    assert_eq!(AR_CHECK_RESERVE_BYTES, 512 * 128 * 4);
    assert_eq!(def("CHK_D_LAST_S"), "12");
    const { assert!(12 < AR_CHECK_DIAG_WORDS) };
}
