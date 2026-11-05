//! Shared fixtures for the integration tests.
//!
//! The cache builders here fabricate measurements. That is legal in a test and nowhere else: the
//! library deliberately ships no such helper, because a fabricated record is exactly what the
//! "error if not tuned" rule exists to prevent.

#![allow(dead_code)]

use saber_plan::{
    AutotuneCache, AutotuneRecord, Manifest, ModelOp, ModelSpec, Parallelism, Provenance,
    Requirement, Resolved, RunnerUp, VariantParams,
};

/// A manifest for the tiny model: one bucket, TP=1, one plugin.
pub const TINY_MANIFEST_TOML: &str = r#"
[model]
path = "/models/tiny"
tier_id = "tier:tiny-v1"

[device]
target_gfx = "gfx1201"
power_cap_w = 210
tp = 1

[[plugins]]
name = "mxfp4"
path = "plugins/gfx1201/mxfp4.hsaco"
build_hash = "b7f3a1c0"

[[buckets]]
batch = 2
spec_width = 2
ctx_band = 4096
"#;

/// The tiny manifest.
pub fn tiny_manifest() -> Manifest {
    Manifest::from_str(TINY_MANIFEST_TOML, "tiny.toml").expect("tiny manifest is valid")
}

/// Two ops, small shapes, TP=1 — enough to exercise the planner without a 400-row plan.
pub fn tiny_model() -> ModelSpec {
    ModelSpec {
        name: "tiny-2layer".to_string(),
        tp: 1,
        hidden: 256,
        layers: 2,
        gdn_layers: 2,
        attn_layers: 0,
        gdn_state_bytes_per_layer_per_slot: 1024 * 1024,
        ops: vec![
            ModelOp::new(
                "layer.mlp.down_proj",
                "gemm_w4a8",
                256,
                512,
                2,
                Parallelism::Row,
            )
            .key("w_dtype", "mxfp4")
            .key("a_dtype", "int8"),
            ModelOp::new("lm_head", "gemm_w4a8", 1024, 256, 1, Parallelism::Column)
                .key("w_dtype", "mxfp4")
                .key("a_dtype", "int8"),
        ],
    }
}

/// The provenance every fabricated record carries.
pub fn fixture_provenance() -> Provenance {
    Provenance::new("maintainer", "2026-09-16", "par_harness")
}

/// Fills every requirement with a record naming the first candidate.
pub fn tuned_cache(resolved: &Resolved<'_>) -> AutotuneCache {
    tuned_cache_with(resolved, |r| r.candidates[0].clone())
}

/// Fills every requirement with a record naming whatever `pick` returns.
pub fn tuned_cache_with<F>(resolved: &Resolved<'_>, pick: F) -> AutotuneCache
where
    F: Fn(&Requirement) -> String,
{
    let mut cache = AutotuneCache::new();
    for r in resolved.requirements() {
        let chosen = pick(&r);
        let runner_up = r
            .candidates
            .iter()
            .find(|c| **c != chosen)
            .map(|c| RunnerUp {
                impl_name: c.clone(),
                variant_id: 3,
                measured_us: 32.5,
            });
        cache.insert(
            r.key.clone(),
            AutotuneRecord {
                impl_name: chosen,
                variant: VariantParams::new(12).param("split_k", 4).param("tm", 4),
                measured_us: 31.2,
                samples_kept: 4,
                runner_up,
                unstable: None,
                provenance: fixture_provenance(),
            },
        );
    }
    cache
}

/// A temporary file path inside a fresh scratch directory under `target/` (never `/tmp`, FN51h),
/// unique to this process and tag. The directory -- and the file in it -- is removed when the
/// returned [`saber_test_scratch::ScratchDir`] drops, so hold it for as long as the path is used.
pub fn temp_path(tag: &str) -> (saber_test_scratch::ScratchDir, std::path::PathBuf) {
    let dir = saber_test_scratch::ScratchDir::new(&format!("saber-plan-{tag}"));
    let path = dir.join(format!("saber-plan-{tag}.json"));
    (dir, path)
}
