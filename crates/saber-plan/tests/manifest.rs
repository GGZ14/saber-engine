//! Manifest parsing, round-tripping and — mostly — the quality of its refusals.

use saber_plan::{reference, Manifest, ManifestError};

fn issues(err: &ManifestError) -> Vec<String> {
    match err {
        ManifestError::Invalid { issues, .. } => issues.clone(),
        other => panic!("expected a validation failure, got: {other}"),
    }
}

fn parse_err(toml: &str) -> ManifestError {
    Manifest::from_str(toml, "test.toml").expect_err("expected this manifest to be refused")
}

#[test]
fn example_manifest_parses_with_the_documented_values() {
    let m = Manifest::from_str(reference::EXAMPLE_MANIFEST_TOML, "example.toml").unwrap();
    assert_eq!(m.model.tier_id, "tier:qwen38-27b-nvfp4-mxfp4-dflash2-v22.3");
    assert_eq!(m.device.target_gfx, "gfx1201");
    assert_eq!(m.device.power_cap_w, 210);
    assert_eq!(m.device.tp, 2);
    // Three: the MXFP4 family, saber, and the attention glue the registry's three glue rows
    // are owned by. A registry row whose plugin no manifest declares is a refusal, which is
    // how the third one came to be here.
    assert_eq!(m.plugins.len(), 3);
    assert_eq!(m.plugin("saber").unwrap().build_hash, "5d91e2f4");
    assert_eq!(m.plugin("attn_glue").unwrap().build_hash, "1fb9de64");
    // Two decode buckets and two prefill ones: the example manifest documents both classes.
    assert_eq!(m.buckets.len(), 4);
    assert_eq!(
        m.buckets
            .iter()
            .filter(|b| b.phase == saber_plan::Phase::Prefill)
            .map(|b| b.tokens)
            .collect::<Vec<_>>(),
        vec![2048, 8192]
    );
    assert!(m.plan.capture);
    assert!(m.plan.overrides.is_empty());
    // Every bucket inherits [plan] capture.
    assert!(m.buckets.iter().all(|b| m.bucket_captured(b)));
}

#[test]
fn round_trips_through_toml() {
    let first = Manifest::from_str(reference::EXAMPLE_MANIFEST_TOML, "example.toml").unwrap();
    let text = first.to_toml_string();
    let second = Manifest::from_str(&text, "round-trip").unwrap();
    assert_eq!(first, second, "round-tripped manifest differs:\n{text}");
    assert_eq!(
        second.to_toml_string(),
        text,
        "serialization is not a fixed point"
    );
}

#[test]
fn round_trips_every_optional_field() {
    let toml = r#"
[model]
path = "/models/x"
tier_id = "tier:a"
drafter_tier_id = "tier:a"

[device]
target_gfx = "gfx1201"
power_cap_w = 210
tp = 2

[[plugins]]
name = "mxfp4"
path = "p.hsaco"
build_hash = "abc"

[[buckets]]
batch = 1
spec_width = 5
ctx_band = 8192
capture = false

[plan]
capture = true

[plan.overrides."layer.mlp.down_proj"]
impl = "gemm_w4a8_mxfp4_wperm"
variant_id = 9
"#;
    let first = Manifest::from_str(toml, "test.toml").unwrap();
    assert_eq!(first.model.drafter_tier_id.as_deref(), Some("tier:a"));
    assert_eq!(first.buckets[0].capture, Some(false));
    assert!(
        !first.bucket_captured(&first.buckets[0]),
        "a bucket overrides [plan] capture"
    );
    let over = &first.plan.overrides["layer.mlp.down_proj"];
    assert_eq!(over.impl_name.as_deref(), Some("gemm_w4a8_mxfp4_wperm"));
    assert_eq!(over.variant_id, Some(9));

    let text = first.to_toml_string();
    assert_eq!(
        Manifest::from_str(&text, "round-trip").unwrap(),
        first,
        "{text}"
    );
}

#[test]
fn tp_defaults_to_one_and_capture_defaults_to_true() {
    let toml = r#"
[model]
path = "/models/x"
tier_id = "tier:a"

[device]
target_gfx = "gfx1201"
power_cap_w = 210

[[plugins]]
name = "mxfp4"
path = "p.hsaco"
build_hash = "abc"

[[buckets]]
batch = 1
spec_width = 1
ctx_band = 1024
"#;
    let m = Manifest::from_str(toml, "test.toml").unwrap();
    assert_eq!(m.device.tp, 1);
    assert!(m.plan.capture);
}

#[test]
fn unknown_key_is_an_error_and_names_what_was_expected() {
    let toml = r#"
[model]
path = "/models/x"
tier_id = "tier:a"

[device]
target_gfx = "gfx1201"
powercap_w = 210
power_cap_w = 210

[[plugins]]
name = "mxfp4"
path = "p.hsaco"
build_hash = "abc"

[[buckets]]
batch = 1
spec_width = 1
ctx_band = 1024
"#;
    let err = parse_err(toml);
    let issues = issues(&err);
    assert_eq!(issues.len(), 1, "{issues:?}");
    assert_eq!(
        issues[0],
        "unknown key `device.powercap_w` (expected one of: power_cap_w, target_gfx, tp)"
    );
    // The rendered error names the origin and the count.
    let rendered = err.to_string();
    assert!(
        rendered.starts_with("test.toml: 1 problem(s):"),
        "{rendered}"
    );
}

#[test]
fn unknown_top_level_section_is_an_error() {
    let toml = r#"
[model]
path = "/models/x"
tier_id = "tier:a"

[device]
target_gfx = "gfx1201"
power_cap_w = 210

[[plugins]]
name = "mxfp4"
path = "p.hsaco"
build_hash = "abc"

[[buckets]]
batch = 1
spec_width = 1
ctx_band = 1024

[tiering]
enabled = true
"#;
    let issues = issues(&parse_err(toml));
    assert_eq!(
        issues,
        vec!["unknown key `tiering` (expected one of: buckets, device, model, plan, plugins)"]
    );
}

#[test]
fn every_missing_required_key_is_listed_at_once() {
    let toml = r#"
[model]
path = "/models/x"

[device]
tp = 2
"#;
    let issues = issues(&parse_err(toml));
    assert_eq!(
        issues,
        vec![
            "missing required key `model.tier_id`",
            "missing required key `device.target_gfx`",
            "missing required key `device.power_cap_w`",
            "missing required section `[[plugins]]`: a plan needs at least one entry",
            "missing required section `[[buckets]]`: a plan needs at least one entry",
        ],
        "one pass has to report every missing key, not the first"
    );
}

#[test]
fn power_cap_is_required_even_when_everything_else_is_present() {
    let toml = r#"
[model]
path = "/models/x"
tier_id = "tier:a"

[device]
target_gfx = "gfx1201"

[[plugins]]
name = "mxfp4"
path = "p.hsaco"
build_hash = "abc"

[[buckets]]
batch = 1
spec_width = 1
ctx_band = 1024
"#;
    assert_eq!(
        issues(&parse_err(toml)),
        vec!["missing required key `device.power_cap_w`"]
    );
}

#[test]
fn wrong_types_and_out_of_range_values_are_named_precisely() {
    let toml = r#"
[model]
path = 7
tier_id = "tier:a"

[device]
target_gfx = "gfx1201"
power_cap_w = "210"
tp = 99

[[plugins]]
name = "mxfp4"
path = "p.hsaco"
build_hash = ""

[[buckets]]
batch = 0
spec_width = 1
ctx_band = 1024
"#;
    let issues = issues(&parse_err(toml));
    assert!(
        issues.contains(&"`model.path` must be a string, found an integer".to_string()),
        "{issues:#?}"
    );
    assert!(
        issues.contains(&"`device.power_cap_w` must be an integer, found a string".to_string()),
        "{issues:#?}"
    );
    assert!(
        issues.contains(&"`device.tp` must be in 1..=8, found 99".to_string()),
        "{issues:#?}"
    );
    assert!(
        issues.contains(&"`plugins[0].build_hash` must not be empty".to_string()),
        "{issues:#?}"
    );
    assert!(
        issues.contains(&"`buckets[0].batch` must be in 1..=4096, found 0".to_string()),
        "{issues:#?}"
    );
}

#[test]
fn duplicate_plugins_and_buckets_are_refused() {
    let toml = r#"
[model]
path = "/models/x"
tier_id = "tier:a"

[device]
target_gfx = "gfx1201"
power_cap_w = 210

[[plugins]]
name = "mxfp4"
path = "a.hsaco"
build_hash = "abc"

[[plugins]]
name = "mxfp4"
path = "b.hsaco"
build_hash = "def"

[[buckets]]
batch = 1
spec_width = 5
ctx_band = 8192

[[buckets]]
batch = 1
spec_width = 5
ctx_band = 8192
"#;
    let issues = issues(&parse_err(toml));
    assert!(
        issues
            .iter()
            .any(|i| i.contains("duplicate plugin name `mxfp4`")),
        "{issues:#?}"
    );
    assert!(
        issues
            .iter()
            .any(|i| i.contains("duplicate bucket batch=1 spec_width=5 ctx_band=8192")),
        "{issues:#?}"
    );
}

#[test]
fn an_empty_bucket_list_is_refused() {
    let toml = r#"
buckets = []

[model]
path = "/models/x"
tier_id = "tier:a"

[device]
target_gfx = "gfx1201"
power_cap_w = 210

[[plugins]]
name = "mxfp4"
path = "p.hsaco"
build_hash = "abc"
"#;
    let issues = issues(&parse_err(toml));
    assert_eq!(
        issues,
        vec!["`[[buckets]]` is empty: a plan needs at least one entry"]
    );
}

#[test]
fn an_override_that_overrides_nothing_is_refused() {
    let toml = r#"
[model]
path = "/models/x"
tier_id = "tier:a"

[device]
target_gfx = "gfx1201"
power_cap_w = 210

[[plugins]]
name = "mxfp4"
path = "p.hsaco"
build_hash = "abc"

[[buckets]]
batch = 1
spec_width = 1
ctx_band = 1024

[plan.overrides.lm_head]
"#;
    let issues = issues(&parse_err(toml));
    assert_eq!(
        issues,
        vec!["`plan.overrides.lm_head` overrides nothing: set `impl`, `variant_id`, or both"]
    );
}

#[test]
fn an_unknown_override_key_is_refused() {
    let toml = r#"
[model]
path = "/models/x"
tier_id = "tier:a"

[device]
target_gfx = "gfx1201"
power_cap_w = 210

[[plugins]]
name = "mxfp4"
path = "p.hsaco"
build_hash = "abc"

[[buckets]]
batch = 1
spec_width = 1
ctx_band = 1024

[plan.overrides.lm_head]
kernel = "gemm_w4a8_mxfp4_decode"
"#;
    let issues = issues(&parse_err(toml));
    assert!(
        issues.contains(
            &"unknown key `plan.overrides.lm_head.kernel` (expected one of: impl, variant_id)"
                .to_string()
        ),
        "{issues:#?}"
    );
}

#[test]
fn broken_toml_is_a_syntax_error_not_a_validation_error() {
    let err = parse_err("[model\npath = 1");
    match err {
        ManifestError::Syntax { origin, .. } => assert_eq!(origin, "test.toml"),
        other => panic!("expected a syntax error, got {other}"),
    }
}

#[test]
fn plugin_build_digest_depends_on_every_plugin_hash() {
    let a = Manifest::from_str(reference::EXAMPLE_MANIFEST_TOML, "a").unwrap();
    let mut b = a.clone();
    b.plugins[1].build_hash = "5d91e2f5".to_string();
    assert_ne!(a.plugin_build_digest(), b.plugin_build_digest());

    // Declaration order does not change it; the hash does.
    let mut c = a.clone();
    c.plugins.swap(0, 1);
    assert_eq!(a.plugin_build_digest(), c.plugin_build_digest());
}

#[test]
fn missing_file_is_an_io_error() {
    let err = Manifest::from_path(std::path::Path::new("/nonexistent/saber.toml")).unwrap_err();
    assert!(matches!(err, ManifestError::Io { .. }), "{err}");
    assert!(err.to_string().contains("/nonexistent/saber.toml"), "{err}");
}
