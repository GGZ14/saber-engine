//! The manifest <-> code-object contract, and its negative controls.
//!
//! "A kernarg layout the host and the kernel disagree about is memory corruption, not an error"
//! (docs/ARCHITECTURE.md, Kernel plane). These tests check that the disagreement is *detected*,
//! which needs a case that is known to fail as much as it needs the happy path.

use saber_kernels::abi;
use saber_kernels::manifest::{load_prebuilt, Manifest};
use saber_kernels::{prebuilt, Error};

#[test]
fn every_committed_plugin_validates_against_its_blob() {
    let plugins = load_prebuilt().expect("prebuilt manifests validate");
    let total: usize = plugins.iter().map(|(m, _)| m.kernel.len()).sum();
    assert_eq!(plugins.len(), prebuilt::ALL.len());
    assert!(
        total >= 140,
        "only {total} kernels across {} plugins",
        plugins.len()
    );
    for (m, _) in &plugins {
        assert_eq!(m.arch, prebuilt::ARCH);
        assert!(!m.build_hash.is_empty() && !m.source_sha256.is_empty());
        for k in &m.kernel {
            assert_eq!(k.wavefront_size, 32, "{}: wave size", k.name);
            // gfx1201 has 64 KiB of LDS per workgroup; a kernel over it would fail at launch.
            assert!(
                k.group_segment_fixed_size <= 65536,
                "{}: LDS {}",
                k.name,
                k.group_segment_fixed_size
            );
            assert!(k.max_flat_workgroup_size >= k.arg.len() as u32);
        }
    }
}

#[test]
fn every_committed_kernel_matches_its_declared_signature() {
    for (m, _) in load_prebuilt().expect("prebuilt") {
        for k in &m.kernel {
            let sig = abi::signature_for(&k.name)
                .unwrap_or_else(|| panic!("{}: no declared signature", k.name));
            m.check_signature(&k.name, sig)
                .unwrap_or_else(|e| panic!("{}: {e}", k.name));
        }
    }
}

/// NEGATIVE CONTROL: a manifest that declares a different kernarg layout than the blob must be
/// refused. Without this the validation above proves nothing.
#[test]
fn a_kernarg_offset_that_drifted_is_refused() {
    let dir = prebuilt::dir();
    let mut m = Manifest::load(dir.join("mxfp4_gemm_decode.manifest.toml")).expect("load");
    let bytes = std::fs::read(m.hsaco_path(&dir)).expect("blob");
    m.validate_against_hsaco(&bytes)
        .expect("the committed pair agrees");

    // Move `astride` by 8 bytes -- exactly the class of drift that silently corrupts memory.
    let k = m
        .kernel
        .iter_mut()
        .find(|k| k.name.starts_with("mxfp4_gemm_decode"))
        .expect("kernel");
    let a = k.arg.last_mut().expect("astride");
    assert_eq!(a.name.as_deref(), Some("astride"));
    a.offset += 8;
    match m.validate_against_hsaco(&bytes) {
        Err(Error::Validate(v)) => {
            assert!(v.iter().any(|s| s.contains("offset")), "{v:?}");
        }
        other => panic!("a drifted kernarg offset was accepted: {other:?}"),
    }
}

/// NEGATIVE CONTROL: a manifest whose blob was rebuilt underneath it must be refused before any
/// field is read, because every field would then be describing a different build.
#[test]
fn a_blob_that_changed_under_the_manifest_is_refused() {
    let dir = prebuilt::dir();
    let m = Manifest::load(dir.join("mxfp4_quant.manifest.toml")).expect("load");
    let mut bytes = std::fs::read(m.hsaco_path(&dir)).expect("blob");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    match m.validate_against_hsaco(&bytes) {
        Err(Error::Validate(v)) => assert!(v[0].contains("sha256"), "{v:?}"),
        other => panic!("a modified blob was accepted: {other:?}"),
    }
}

/// NEGATIVE CONTROL: the parser must not accept a truncated code object as an empty one.
#[test]
fn a_truncated_code_object_is_an_error_not_an_empty_one() {
    let dir = prebuilt::dir();
    let bytes = std::fs::read(dir.join("mxfp4_gemm_base.hsaco")).expect("blob");
    let half = &bytes[..bytes.len() / 2];
    assert!(saber_kernels::codeobj::CodeObject::parse(half).is_err());
}

/// The decode kernel's LDS is (16*DTM + DWN*16) * (DBK + 8); the metadata must show the tile
/// growing with DTM and BK, which is the occupancy cliff the launch rules are built on.
#[test]
fn decode_lds_tracks_the_tile() {
    let dir = prebuilt::dir();
    let m = Manifest::load(dir.join("mxfp4_gemm_decode.manifest.toml")).expect("load");
    let lds = |n: &str| {
        m.kernel(n)
            .unwrap_or_else(|| panic!("{n}"))
            .group_segment_fixed_size
    };
    for (bk, tm) in [(64u32, 1u32), (64, 4), (128, 1), (128, 4)] {
        let name = format!("mxfp4_gemm_decode_bk{bk}_ks1_tm{tm}_wperm1_nt1");
        let want = (16 * tm + 8 * 16) * (bk + 8);
        // Plus the one int of `s_last` and whatever alignment the compiler adds.
        let got = lds(&name);
        assert!(
            got >= want && got <= want + 64,
            "{name}: LDS {got}, tile arithmetic says {want}"
        );
    }
    assert!(
        lds("mxfp4_gemm_decode_bk128_ks1_tm4_wperm1_nt1")
            > lds("mxfp4_gemm_decode_bk64_ks1_tm4_wperm1_nt1"),
        "BK=128 must cost more LDS than BK=64 at the same DTM"
    );
}

/// **A RULES ROW MAY NOT NAME A SYMBOL THE COMMITTED BLOB DOES NOT CARRY**, and this test exists
/// because one did.
///
/// `bgemm_bf16_mt` was bound to `dflash_bgemm_bf16_mtn16` when workstream S6 added the M-tiled
/// projection, and the code object was never rebuilt -- so from the commit that landed the rule
/// until the one that landed this test, `dflash_bgemm.hsaco` had three kernels and the rules named
/// a fourth. Every host test passed: `resolve` produces a descriptor out of the rules alone, and
/// the manifest tests walk the BLOB's kernels rather than the rules'. Only a launch could have
/// found it -- on the card, in a gate, a round later.
///
/// The check is over every dflash row, and the CONTROL is the second half: a symbol that no blob
/// carries must be reported as absent, or the lookup is vacuous.
#[test]
fn every_dflash_rules_symbol_is_a_kernel_of_its_own_plugin_blob() {
    use saber_kernels::dflash::DflashRules;
    let rules = DflashRules::prod_dflash().expect("the committed dflash rules");
    let plugins = load_prebuilt().expect("prebuilt manifests validate");
    let mut checked = 0;
    for row in &rules.kernel {
        let (m, _) = plugins
            .iter()
            .find(|(m, _)| m.plugin == row.plugin)
            .unwrap_or_else(|| panic!("{}: no committed blob for plugin {}", row.op, row.plugin));
        assert!(
            m.kernel(&row.symbol).is_some(),
            "{}: the rules bind `{}` but {}.hsaco carries {:?} -- the code object was not rebuilt",
            row.op,
            row.symbol,
            row.plugin,
            m.kernel.iter().map(|k| k.name.as_str()).collect::<Vec<_>>()
        );
        checked += 1;
    }
    assert!(checked >= 32, "only {checked} dflash rows checked");

    // THE CONTROL: the same lookup against a symbol nothing compiles must fail, so a green run
    // above is the blobs agreeing rather than `kernel()` answering yes to everything.
    let absent = plugins
        .iter()
        .any(|(m, _)| m.kernel("dflash_bgemm_bf16_mtn64_never_compiled").is_some());
    assert!(
        !absent,
        "the control symbol exists, so the check above is vacuous"
    );
}
