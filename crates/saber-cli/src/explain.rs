//! `saber explain` — the whole configuration, resolved, printed, and diffable.
//!
//! One manifest replaces the 137 environment variables; this is what reads it back. The registry is
//! **built from the artifacts** (`prebuilt/gfx1201/*.manifest.toml` plus the launch rules) by
//! [`saber_plan::registry_from_prebuilt`], not transcribed, so wave size and LDS come out of the
//! code objects and the constraint bands come out of the same rules `resolve` enforces.
//!
//! With a cold autotune cache this prints the **untuned work list**, because that is the honest P0
//! state and because the planner has no cost model to fall back on:
//!
//! > No cost model. Constraint match → autotune lookup → **error if not tuned**.
//!
//! Once a cache covers every route, the same command prints the bound plan instead.

use std::path::PathBuf;

use std::collections::BTreeMap;

use saber_plan::{
    check_plugins_declared, explain, explain_missing, registry_from_prebuilt, AutotuneCache,
    DeviceProfile, Manifest, ModelSpec, Registry, Requirement, Resolved,
};

/// Arguments of `saber explain`.
pub struct ExplainArgs {
    /// The Saber manifest.
    pub manifest: PathBuf,
    /// The prebuilt plugin directory the registry is derived from.
    pub prebuilt: PathBuf,
    /// An autotune cache to bind against. Absent means a cold cache, which is the P0 state.
    pub autotune: Option<PathBuf>,
    /// Emit JSON instead of the human dump.
    pub json: bool,
}

/// Print the registry the artifacts produced: one row per implementation, with everything the
/// planner will decide on.
fn registry_table(registry: &Registry) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "registry       {} implementation(s) derived from the committed artifacts",
        registry.implementations().len()
    );
    for imp in registry.implementations() {
        let _ = writeln!(
            out,
            "  {:<26} op {:<16} plugin {:<20} {:<12} wave{:<3} lds {:>6} launches {}",
            imp.name,
            imp.op,
            imp.plugin,
            imp.promotion.state.as_str(),
            imp.wave_size,
            imp.lds_bytes,
            imp.launches.n_launches(4),
        );
        let _ = writeln!(
            out,
            "      {:<10} {}",
            if imp.capturable() {
                "capturable"
            } else {
                "EAGER-ONLY"
            },
            imp.shape
        );
        let constraints: Vec<String> = imp.constraints.iter().map(ToString::to_string).collect();
        let _ = writeln!(out, "      needs      {}", constraints.join(", "));
        for w in &imp.workspaces {
            let _ = writeln!(out, "      workspace  {} ({})", w.name, w.kind);
        }
    }
    out
}

/// The launch budget per bucket, **before** anything is tuned.
///
/// `CLAUDE.md` rule 7: "launch count per bucket is declared statically and asserted at capture".
/// Declared means exactly this — it is a property of the registry rows and the model's op counts,
/// so it exists the moment the plan resolves, months before a measurement picks between candidates.
/// `PlannedBucket::n_launches` prints the same number once a route is bound; this is what a cold
/// cache can still say.
///
/// Where an op's candidates disagree about their launch count the budget is a range, because a
/// single number would be a guess about which one the sweep will pick.
fn launch_budget(resolved: &Resolved<'_>, model: &ModelSpec, registry: &Registry) -> String {
    use std::fmt::Write as _;
    let mut per_bucket: BTreeMap<String, (u32, u32)> = BTreeMap::new();
    let mut rows: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for r in resolved.requirements() {
        let Requirement {
            bucket,
            op_name,
            candidates,
            ..
        } = &r;
        let count = model.op(op_name).map_or(0, |o| o.count);
        let each: Vec<u32> = candidates
            .iter()
            .filter_map(|c| registry.by_name(c))
            // `split_k` is a variant parameter and is only known once a record binds; 1 and 4 are
            // the two answers the split-K ladder can give, so they bracket the launch count.
            .map(|(_, imp)| (imp.launches.n_launches(1), imp.launches.n_launches(4)))
            .flat_map(|(lo, hi)| [lo, hi])
            .collect();
        let (lo, hi) = (
            each.iter().copied().min().unwrap_or(0),
            each.iter().copied().max().unwrap_or(0),
        );
        let e = per_bucket.entry(bucket.label()).or_insert((0, 0));
        e.0 += lo * count;
        e.1 += hi * count;
        let span = if lo == hi {
            format!("{lo}")
        } else {
            format!("{lo}-{hi}")
        };
        rows.entry(bucket.label())
            .or_default()
            .push(format!("{op_name} x{count} @ {span}"));
    }
    let mut out = String::new();
    let _ = writeln!(
        out,
        "launch budget  declared statically from the registry and the model's op counts, before \
         any measurement"
    );
    for (bucket, (lo, hi)) in &per_bucket {
        let total = if lo == hi {
            format!("{lo}")
        } else {
            format!("{lo}-{hi}")
        };
        let _ = writeln!(out, "  {bucket:<12} {total:>9} launches/step");
        if let Some(ops) = rows.get(bucket) {
            for chunk in ops.chunks(4) {
                let _ = writeln!(out, "               {}", chunk.join(", "));
            }
        }
    }
    out
}

/// Run `saber explain`.
pub fn run(args: &ExplainArgs) -> anyhow::Result<()> {
    let manifest = Manifest::from_path(&args.manifest)?;
    let registry = registry_from_prebuilt(&args.prebuilt)?;
    let device = DeviceProfile::gfx1201_reference();
    let model = ModelSpec::qwen38_27b_dense(manifest.device.tp)?;
    // A plan that resolves against a plugin the manifest does not declare would fail at load, not
    // at plan time. Refuse here, where the message can name the file. Scoped to the rows this
    // model can bind: the registry also carries the rollback tier's Paro producers, which
    // this deployment lists but never launches.
    check_plugins_declared(&registry, &manifest, &model)?;
    let resolved = saber_plan::resolve(&manifest, &device, &model, &registry)?;

    let cache = match &args.autotune {
        Some(p) => AutotuneCache::load(p)?,
        None => AutotuneCache::new(),
    };
    let missing = resolved.missing(&cache);

    if args.json {
        let value = if missing.is_empty() {
            let plan = resolved.bind(&cache)?;
            serde_json::json!({
                "manifest": args.manifest.display().to_string(),
                "prebuilt": args.prebuilt.display().to_string(),
                "state": "bound",
                "plan_hash": plan.plan_hash,
                "plan": plan.body,
            })
        } else {
            serde_json::json!({
                "manifest": args.manifest.display().to_string(),
                "prebuilt": args.prebuilt.display().to_string(),
                "state": "untuned",
                "plan_input_hash": resolved.input_hash(),
                "launch_budget": launch_budget(&resolved, &model, &registry),
                "registry": registry.implementations(),
                "missing": missing
                    .iter()
                    .map(|r| serde_json::json!({
                        "bucket": r.bucket,
                        "op_name": r.op_name,
                        "key": r.key,
                        "key_digest": r.key.digest(),
                        "candidates": r.candidates,
                    }))
                    .collect::<Vec<_>>(),
            })
        };
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(());
    }

    println!("manifest       {}", args.manifest.display());
    println!("prebuilt       {}", args.prebuilt.display());
    println!(
        "model          {}  tp={}  tier {}",
        model.name, model.tp, manifest.model.tier_id
    );
    print!("{}", registry_table(&registry));
    println!();
    print!("{}", launch_budget(&resolved, &model, &registry));
    println!();

    if missing.is_empty() {
        let plan = resolved.bind(&cache)?;
        print!("{}", explain(&plan));
    } else {
        // The expected P0 state: the routes exist and are `implemented`; nothing is measured, so
        // nothing can be bound, and the planner says what to measure rather than guessing.
        print!("{}", explain_missing(&resolved, &cache));
        println!();
        println!(
            "No route is bound: with a cold autotune cache the planner has nothing to choose with, \
             and it does not have a cost model to fall back on. Every row above is a measurement \
             the P0 sweep owes."
        );
    }
    Ok(())
}
