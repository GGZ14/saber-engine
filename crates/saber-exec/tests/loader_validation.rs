//! Load-time validation, on the host, with no GPU and no ROCm.
//!
//! The three checks are pure functions of (manifest, code object), which is exactly what makes
//! this testable here: the committed `prebuilt/gfx1201` blobs and their manifests are the positive
//! control, and a drifted copy of each is a negative control that is *known to fail*
//! (ARCHITECTURE.md, test tier 2 — "a gate without a negative control has never been shown to
//! detect anything").
//!
//! Each negative control also asserts **which** counter moved. A refusal that lands in the wrong
//! bucket is a refusal whose message will point at the wrong file the first time it fires in
//! anger.

use saber_exec::{implied_struct_align, validate_plugin, LoadCounters};
use saber_kernels::Manifest;

/// The smallest committed plugin, so the test reads ~50 KB rather than 1.3 MB.
const PLUGIN: &str = "mxfp4_gemm_base";

fn load(plugin: &str) -> (Manifest, Vec<u8>) {
    let dir = saber_kernels::prebuilt::dir();
    let m = Manifest::load(dir.join(format!("{plugin}.manifest.toml"))).expect("manifest parses");
    let bytes = std::fs::read(m.hsaco_path(&dir)).expect("blob is committed next to it");
    (m, bytes)
}

#[test]
fn every_committed_plugin_validates_and_counts_as_loaded() {
    let counters = LoadCounters::new();
    for (blob, _) in saber_kernels::prebuilt::ALL {
        let (m, bytes) = load(blob.trim_end_matches(".hsaco"));
        let facts = validate_plugin(&m, &bytes, &counters).unwrap_or_else(|e| {
            panic!("{blob} must validate against its own code object: {e}");
        });
        assert!(
            facts.target.ends_with("gfx1201"),
            "{blob}: target {}",
            facts.target
        );
        assert!(
            facts.wave_size == 32 || facts.wave_size == 64,
            "{blob}: wave size {}",
            facts.wave_size
        );
        assert!(facts.kernels > 0);
        // `validate_plugin` is the pure half; `loaded` is only counted once HIP has the module.
        assert_eq!(counters.snapshot().loaded, 0);
    }
    let c = counters.snapshot();
    assert_eq!(c.refused_kernarg, 0);
    assert_eq!(c.refused_wave, 0);
    assert_eq!(c.refused_metadata, 0);
}

#[test]
fn negative_control_a_drifted_kernarg_offset_is_refused_by_the_abi_rules() {
    let (mut m, bytes) = load(PLUGIN);
    // Drift ONE argument's offset by two bytes. Two, not four: it has to break natural alignment,
    // because that is the class of drift `saber_abi::check_kernarg` exists to catch, and a
    // 4-aligned drift would instead be caught downstream by the code-object comparison.
    let kernel = m.kernel[0].name.clone();
    let victim = m.kernel[0]
        .arg
        .iter()
        .position(|a| a.size == 8)
        .expect("this kernel takes pointers");
    let was = m.kernel[0].arg[victim].offset;
    m.kernel[0].arg[victim].offset = was + 2;

    let counters = LoadCounters::new();
    let err = validate_plugin(&m, &bytes, &counters).expect_err("a drifted offset must be refused");
    let text = err.to_string();
    assert!(
        text.contains(&kernel),
        "the refusal must name the kernel: {text}"
    );
    assert!(
        text.contains(&format!("arg[{victim}]")),
        "the refusal must name the field: {text}"
    );
    let c = counters.snapshot();
    assert_eq!(c.refused_kernarg, 1, "wrong counter moved: {}", c.line());
    assert_eq!(c.refused_wave, 0);
    assert_eq!(c.refused_metadata, 0);
    assert_eq!(c.loaded, 0);
}

#[test]
fn negative_control_an_aligned_offset_drift_is_still_refused_by_the_code_object() {
    // The other half of the same hazard: a drift that the ABI rules cannot see, because it is
    // still naturally aligned and still inside the segment. The code object's own metadata is what
    // catches it, and the counter says so.
    let (mut m, bytes) = load(PLUGIN);
    let victim = m.kernel[0]
        .arg
        .iter()
        .position(|a| a.size == 8)
        .expect("this kernel takes pointers");
    m.kernel[0].arg[victim].offset += 8;
    // Keep the table structurally legal so that check (b) has nothing to say about it.
    for a in m.kernel[0].arg.iter_mut().skip(victim + 1) {
        a.offset += 8;
    }
    m.kernel[0].kernarg_segment_size += 8;

    let counters = LoadCounters::new();
    let err = validate_plugin(&m, &bytes, &counters).expect_err("the code object disagrees");
    let c = counters.snapshot();
    assert_eq!(c.refused_metadata, 1, "wrong counter moved: {}", c.line());
    assert_eq!(c.refused_kernarg, 0);
    assert!(
        err.to_string().contains("kernarg_segment_size") || err.to_string().contains("offset"),
        "{err}"
    );
}

#[test]
fn negative_control_a_drifted_wave_size_is_refused_with_its_own_counter() {
    let (mut m, bytes) = load(PLUGIN);
    let kernel = m.kernel[0].name.clone();
    m.kernel[0].wavefront_size = if m.kernel[0].wavefront_size == 32 {
        64
    } else {
        32
    };

    let counters = LoadCounters::new();
    let err = validate_plugin(&m, &bytes, &counters).expect_err("wave size must be checked");
    let text = err.to_string();
    assert!(text.contains(&kernel), "{text}");
    assert!(text.contains("wavefront_size"), "{text}");
    let c = counters.snapshot();
    assert_eq!(c.refused_wave, 1, "wrong counter moved: {}", c.line());
    assert_eq!(c.refused_metadata, 0);
    assert_eq!(c.refused_kernarg, 0);
}

#[test]
fn negative_control_a_plugin_that_mixes_wave_sizes_is_two_code_objects() {
    // `-mwavefrontsize64` is translation-unit wide and the per-function attribute is silently
    // ignored (rdna4-wave64-wmma). A manifest claiming both in one plugin is describing something
    // that cannot exist.
    let (mut m, bytes) = load("mxfp4_quant");
    assert!(m.kernel.len() > 1, "this test needs a multi-kernel plugin");
    m.kernel[1].wavefront_size = 64;
    m.kernel[0].wavefront_size = 32;
    let counters = LoadCounters::new();
    let err = validate_plugin(&m, &bytes, &counters).expect_err("mixed wave sizes must be refused");
    assert!(err.to_string().contains("wavefront_size"), "{err}");
    assert_eq!(counters.snapshot().refused_wave, 1);
}

#[test]
fn negative_control_a_corrupt_blob_is_refused_before_anything_else_is_believed() {
    let (m, mut bytes) = load(PLUGIN);
    let n = bytes.len();
    bytes[n / 2] ^= 0xff;
    let counters = LoadCounters::new();
    let err = validate_plugin(&m, &bytes, &counters).expect_err("the sha256 must be checked");
    assert!(
        err.to_string().contains("sha256") || err.to_string().contains("parse"),
        "{err}"
    );
    assert_eq!(counters.snapshot().refused_metadata, 1);
}

#[test]
fn explicit_kernargs_must_fit_the_declared_segment() {
    let (mut m, bytes) = load(PLUGIN);
    let last = m.kernel[0].arg.len() - 1;
    m.kernel[0].kernarg_segment_size = m.kernel[0].arg[last].offset;
    let counters = LoadCounters::new();
    let err = validate_plugin(&m, &bytes, &counters).expect_err("segment too small");
    assert!(err.to_string().contains("kernarg_segment_size"), "{err}");
    assert_eq!(counters.snapshot().refused_kernarg, 1);
}

// --------------------------------------------------------------------------------------------
// By-value structs (ABI 0.3). The reference's paged-attention kernels take `const SABERArgs a`, which the
// code object reports as ONE `by_value` argument of 136 bytes. Before `SABER_ARG_STRUCT` the
// loader refused the plugin outright; these pin down that it now accepts exactly the layouts a C
// compiler could have produced, and nothing else.
// --------------------------------------------------------------------------------------------

/// The attention plugin, and the index of the kernel whose first argument is the struct.
fn attn_with_struct() -> (Manifest, Vec<u8>, usize) {
    let (m, bytes) = load("saber_attn");
    let i = m
        .kernel
        .iter()
        .position(|k| {
            k.arg
                .first()
                .is_some_and(|a| a.value_kind == "by_value" && a.size > 8)
        })
        .expect("both attention kernels take SABERArgs by value");
    (m, bytes, i)
}

#[test]
fn a_by_value_struct_argument_validates_and_is_136_bytes() {
    let (m, bytes, i) = attn_with_struct();
    let a = &m.kernel[i].arg[0];
    assert_eq!(a.size, 136, "sizeof(SABERArgs)");
    assert_eq!(a.offset, 0, "the struct is the first argument");
    assert_eq!(m.kernel[i].kernarg_segment_align, 8);
    let counters = LoadCounters::new();
    validate_plugin(&m, &bytes, &counters).expect("saber_attn validates");
    assert_eq!(counters.snapshot().refused_kernarg, 0);
}

#[test]
fn the_implied_struct_alignment_is_derived_from_the_blobs_own_numbers() {
    // sizeof is a multiple of alignof and no argument out-aligns the segment, so the largest power
    // of two dividing the size, capped at the segment's alignment, is the strongest claim the
    // metadata supports. 136 = 8 x 17 in an 8-aligned segment gives 8, SABERArgs' real alignment.
    assert_eq!(implied_struct_align(136, 8), 8);
    assert_eq!(implied_struct_align(136, 4), 4, "capped by the segment");
    assert_eq!(implied_struct_align(12, 16), 4);
    assert_eq!(implied_struct_align(64, 16), 16);
    // Degenerate inputs answer 1 rather than dividing by zero; `size == 0` is refused separately.
    assert_eq!(implied_struct_align(0, 8), 1);
    assert_eq!(implied_struct_align(136, 0), 1);
}

#[test]
fn negative_control_a_struct_at_a_misaligned_offset_is_refused() {
    let (mut m, bytes, i) = attn_with_struct();
    let kernel = m.kernel[i].name.clone();
    // An 8-aligned struct cannot begin at byte 4. Everything after it moves too, so the table
    // stays structurally legal and it is the ALIGNMENT that refuses.
    for a in m.kernel[i].arg.iter_mut() {
        a.offset += 4;
    }
    m.kernel[i].kernarg_segment_size += 8;
    let counters = LoadCounters::new();
    let err = validate_plugin(&m, &bytes, &counters).expect_err("a misaligned struct is refused");
    let text = err.to_string();
    assert!(text.contains(&kernel), "{text}");
    assert!(text.contains("arg[0]"), "{text}");
    assert_eq!(
        counters.snapshot().refused_kernarg,
        1,
        "wrong counter moved"
    );
}

#[test]
fn negative_control_a_struct_size_that_no_compiler_produces_is_refused() {
    for bad_size in [134u32, 0] {
        let (mut m, bytes, i) = attn_with_struct();
        m.kernel[i].arg[0].size = bad_size;
        let counters = LoadCounters::new();
        let err = validate_plugin(&m, &bytes, &counters)
            .err()
            .unwrap_or_else(|| panic!("a struct of {bad_size} bytes must be refused"));
        assert!(err.to_string().contains("arg[0]"), "{err}");
        assert_eq!(counters.snapshot().refused_kernarg, 1, "size {bad_size}");
    }
}

#[test]
fn negative_control_an_unknown_value_kind_is_still_refused() {
    // The struct support widened `by_value`, not `value_kind` as a whole: a kind this ABI has no
    // packing rule for must still refuse, or the widening would have deleted the check.
    let (mut m, bytes) = load(PLUGIN);
    m.kernel[0].arg[0].value_kind = "image".to_string();
    let counters = LoadCounters::new();
    let err = validate_plugin(&m, &bytes, &counters).expect_err("`image` is not packable here");
    assert!(err.to_string().contains("value_kind"), "{err}");
    assert_eq!(counters.snapshot().refused_kernarg, 1);
}
