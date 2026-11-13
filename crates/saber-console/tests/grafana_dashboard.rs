//! `tools/console/grafana-saber.json` is valid Grafana JSON and every metric
//! family it queries is one the engine or the console exports (read from the exporters' own
//! source, so a renamed family fails here, not in someone's Grafana).

use serde_json::Value;

const DASH: &str = include_str!("../../../tools/console/grafana-saber.json");
const EXPORTERS: &[&str] = &[
    include_str!("../../saber-serve/src/metrics.rs"),
    include_str!("../../saber-mem/src/expert_tier.rs"),
    include_str!("../../saber-mem/src/ple_tier.rs"),
    include_str!("../../saber-mem/src/elastic.rs"),
    include_str!("../../saber-mem/src/ledger.rs"),
    include_str!("../../saber-sched/src/alloc.rs"),
    include_str!("../../saber-sched/src/sched.rs"),
    include_str!("../../saber-cli/src/flashnext/worker.rs"),
    include_str!("../src/usage.rs"),
];

fn exprs(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::Object(m) => {
            for (k, x) in m {
                if k == "expr" {
                    if let Some(s) = x.as_str() {
                        out.push(s.to_string());
                    }
                } else {
                    exprs(x, out);
                }
            }
        }
        Value::Array(a) => a.iter().for_each(|x| exprs(x, out)),
        _ => {}
    }
}

/// `saber:...` / `saber_...` names in a PromQL expression.
fn families(e: &str) -> Vec<String> {
    let b = e.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 6 <= b.len() {
        let boundary = i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
        if boundary && (e[i..].starts_with("saber:") || e[i..].starts_with("saber_")) {
            let j = e[i..]
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == ':'))
                .map_or(e.len(), |n| i + n);
            out.push(e[i..j].to_string());
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

#[test]
fn the_grafana_dashboard_queries_only_exported_families() {
    let d: Value = serde_json::from_str(DASH).expect("valid JSON");
    assert_eq!(d["uid"], "saber-console");
    assert!(d["__inputs"][0]["pluginId"] == "prometheus");
    let mut all = Vec::new();
    exprs(&d["panels"], &mut all);
    assert!(all.len() >= 30, "{} queries", all.len());
    let mut fams: Vec<String> = all.iter().flat_map(|e| families(e)).collect();
    fams.sort();
    fams.dedup();
    assert!(fams.len() >= 25, "{fams:?}");
    let mut missing = Vec::new();
    for f in &fams {
        let base = f
            .strip_suffix("_bucket")
            .or_else(|| f.strip_suffix("_count"))
            .or_else(|| f.strip_suffix("_sum"))
            .unwrap_or(f);
        if !EXPORTERS.iter().any(|src| src.contains(base)) {
            missing.push(f.clone());
        }
    }
    assert!(
        missing.is_empty(),
        "queried but never exported: {missing:?}"
    );
    // the negative control: a made-up family is caught
    assert!(!EXPORTERS
        .iter()
        .any(|src| src.contains("saber:made_up_family_total")));
    // every panel has a grid position and every non-row panel a query
    for p in d["panels"].as_array().unwrap() {
        assert!(p["gridPos"]["w"].as_u64().is_some(), "{p}");
        if p["type"] != "row" {
            assert!(
                !p["targets"].as_array().unwrap().is_empty(),
                "{}",
                p["title"]
            );
        }
    }
}
