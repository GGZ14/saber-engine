//! `saber census-parity` (ws-v8): a `/metrics` scrape against the census a server printed at
//! shutdown, counter for counter, on a real boot.
//!
//! The host test (`saber-engine/tests/census_parity.rs`) proves the two agree on the stub with
//! faults injected; this is the same comparison over a card's own log, so a stage-3 serve phase
//! (`tools/capture/run_engine.sh`) can assert it with nothing but the scrape and the log it keeps.
//! Every `saber:` census def is looked up on both sides by the def's own pattern
//! (`saber_serve::census::find_in_census`); a family missing from either side is a failure.
//! `--ignore` names counters that move while a server idles between the scrape and the census
//! (`saber:sched_idle_steps_total`).

use std::path::Path;

use anyhow::Context;
use saber_sched::{AdmissionCounters, SchedCounters};
use saber_serve::census::{find_in_census, CensusDef};

/// Every census def a `saber serve --engine real` exposes.
pub fn all_defs() -> Vec<CensusDef> {
    let mut d = saber_engine::engine_defs();
    d.extend_from_slice(SchedCounters::CENSUS);
    d.extend_from_slice(AdmissionCounters::CENSUS);
    d
}

/// Compares; returns the mismatches (empty = parity).
pub fn compare(metrics: &str, census: &str, ignore: &[String]) -> Vec<String> {
    let live = |name: &str| -> Option<u64> {
        metrics.lines().find_map(|l| {
            let (k, v) = l.rsplit_once(' ')?;
            (k.split('{').next()? == name).then(|| v.parse::<f64>().ok().map(|x| x as u64))?
        })
    };
    let mut bad = Vec::new();
    for def in all_defs() {
        if ignore.iter().any(|i| i == def.name) {
            continue;
        }
        match (live(def.name), find_in_census(census, &def)) {
            (Some(a), Some(b)) if a == b => {}
            (a, b) => bad.push(format!(
                "{}: /metrics {a:?}, census {b:?} ({})",
                def.name, def.census
            )),
        }
    }
    bad
}

/// Runs the comparison; returns the number of mismatches.
///
/// # Errors
/// A file cannot be read.
pub fn run(metrics: &Path, census: &Path, ignore: &[String]) -> anyhow::Result<usize> {
    let m = std::fs::read_to_string(metrics).with_context(|| metrics.display().to_string())?;
    let c = std::fs::read_to_string(census).with_context(|| census.display().to_string())?;
    let bad = compare(&m, &c, ignore);
    let n = all_defs().len() - ignore.len().min(all_defs().len());
    if bad.is_empty() {
        println!("OK        /metrics equals the shutdown census, counter for counter ({n} saber: counters)");
    } else {
        for b in &bad {
            println!("FAIL      census parity: {b}");
        }
    }
    Ok(bad.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_and_its_control() {
        let ec = saber_engine::EngineCounters {
            eager_steps: 3,
            ..Default::default()
        };
        let census = saber_engine::shutdown_census(
            &SchedCounters::default(),
            &AdmissionCounters::default(),
            &ec,
            None,
        );
        let mut metrics = String::new();
        let mut vals = Vec::new();
        saber_engine::census::engine_values(&ec, &Default::default(), &mut |v| vals.push(v));
        SchedCounters::default().census_values(&mut |v| vals.push(v));
        AdmissionCounters::default().census_values(&mut |v| vals.push(v));
        for (d, v) in all_defs().iter().zip(&vals) {
            metrics.push_str(&format!(
                "{}{{engine=\"0\",model_name=\"Qwen3.8\"}} {v}.0\n",
                d.name
            ));
        }
        assert!(compare(&metrics, &census, &[]).is_empty());
        // CONTROL: one counter off by one is caught, by name; an ignored one is not.
        let moved = metrics.replace(
            "saber:engine_eager_steps_total{engine=\"0\",model_name=\"Qwen3.8\"} 3.0",
            "saber:engine_eager_steps_total{engine=\"0\",model_name=\"Qwen3.8\"} 4.0",
        );
        let bad = compare(&moved, &census, &[]);
        assert_eq!(bad.len(), 1, "{bad:?}");
        assert!(bad[0].starts_with("saber:engine_eager_steps_total"));
        assert!(compare(
            &moved,
            &census,
            &["saber:engine_eager_steps_total".to_owned()]
        )
        .is_empty());
    }
}
