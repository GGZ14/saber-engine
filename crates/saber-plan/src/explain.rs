//! `explain` — the human-readable plan dump behind `saber explain`.
//!
//! It prints, for every bound route: the implementation and variant, its state on the promotion
//! ladder with provenance, its workspace kinds, its graph-safety bits, its static launch count, and
//! **why it was chosen** — the full constraint trace, including the candidates that were refused
//! and what refused them. Plus the plan hash, the tier id, and each bucket's GDN state against the
//! Infinity Cache.
//!
//! The output is deterministic: no timestamps, no map iteration order, no paths that vary by
//! machine beyond what the manifest itself declares. That is what makes it snapshot-testable, and
//! what will make `explain --diff` a two-line addition the first day two plans disagree.

use crate::autotune::AutotuneCache;
use crate::plan::{BoundOp, BucketOps, ExecutionPlan, Phase, PlannedBucket, Requirement, Resolved};
use std::fmt::Write as _;

/// The measurement noise floor on this box: 2.2%. A margin below it decided nothing.
pub const NOISE_FLOOR_PCT: f64 = 2.2;

/// The **absolute** floor the percentage band carries, in microseconds.
///
/// `LEDGER.md`, *P0 bit-exact gate → timing*: two 17 µs kernels landed on opposite sides of 1.000
/// between runs at ratios of 1.03, so "a kernel under ~25 µs is judged on the absolute delta". A
/// difference counts only when it clears both this and [`NOISE_FLOOR_PCT`].
pub const NOISE_FLOOR_US: f64 = 0.5;

/// Bytes as `1.50 MiB`, binary units, two decimals. Deterministic.
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [(&str, u64); 4] = [
        ("GiB", 1 << 30),
        ("MiB", 1 << 20),
        ("KiB", 1 << 10),
        ("B", 1),
    ];
    for (unit, scale) in UNITS {
        if bytes >= scale && scale > 1 {
            return format!("{:.2} {unit}", bytes as f64 / scale as f64);
        }
    }
    format!("{bytes} B")
}

/// The full plan dump.
pub fn explain(plan: &ExecutionPlan) -> String {
    let b = &plan.body;
    let mut out = String::new();
    let _ = writeln!(out, "saber plan");
    let _ = writeln!(out, "  tier           {}", b.tier_id);
    let _ = writeln!(
        out,
        "  drafter tier   {}",
        b.drafter_tier_id.as_deref().unwrap_or("(none bound)")
    );
    let _ = writeln!(out, "  model          {}  tp={}", b.model_name, b.tp);
    let _ = writeln!(
        out,
        "  device         {}  power cap {} W",
        b.gfx_target, b.power_cap_w
    );
    let _ = writeln!(
        out,
        "  plan hash      {}  (schema {})",
        plan.plan_hash, b.schema_version
    );
    let _ = writeln!(
        out,
        "  measured hash  {}  (the microseconds behind the bindings; NOT part of the plan hash)",
        plan.measurement_hash
    );
    let _ = writeln!(out, "  input hash     {}", b.plan_input_hash);
    let _ = writeln!(
        out,
        "  digests        manifest={} device={} model={} registry={} policy={}",
        short(&b.manifest_digest),
        short(&b.device_digest),
        short(&b.model_digest),
        short(&b.registry_fingerprint),
        if b.policy_digest.is_empty() {
            "(hand-built)"
        } else {
            short(&b.policy_digest)
        }
    );
    for p in &b.plugins {
        let _ = writeln!(
            out,
            "  plugin         {} @ {} (build {})",
            p.name, p.path, p.build_hash
        );
    }
    let _ = writeln!(
        out,
        "  totals         {} bucket(s), {} launches/step summed over buckets",
        b.buckets.len(),
        plan.total_launches()
    );
    // The two phases are separate budgets, not two points on one curve: a decode step and a
    // prefill chunk run different ops, so a summed total hides both. Print them side by side.
    for phase in [Phase::Decode, Phase::Prefill] {
        let of_phase: Vec<&PlannedBucket> = b
            .buckets
            .iter()
            .filter(|x| x.bucket.phase == phase)
            .collect();
        if of_phase.is_empty() {
            continue;
        }
        let cells: Vec<String> = of_phase
            .iter()
            .map(|x| format!("{}={}", x.bucket.label(), x.n_launches))
            .collect();
        let _ = writeln!(
            out,
            "  {:<14} {} bucket(s), launches/step {}",
            phase.as_str(),
            of_phase.len(),
            cells.join("  ")
        );
    }

    for bucket in &b.buckets {
        out.push('\n');
        explain_bucket(&mut out, bucket);
    }
    out
}

fn explain_bucket(out: &mut String, planned: &PlannedBucket) {
    let bucket = &planned.bucket;
    let shape = match bucket.phase {
        Phase::Decode => format!("batch {}  spec {}", bucket.batch, bucket.spec_width),
        Phase::Prefill => format!("tokens {}", bucket.tokens),
    };
    let _ = writeln!(
        out,
        "bucket {}  {}  {}  ctx_band {}  M={}  {}",
        bucket.label(),
        bucket.phase.as_str(),
        shape,
        bucket.ctx_band,
        planned.m,
        if bucket.captured { "CAPTURED" } else { "EAGER" }
    );
    let _ = writeln!(
        out,
        "  launches       {} per step (declared statically, asserted at capture)",
        planned.n_launches
    );
    let s = &planned.gdn_state;
    let _ = writeln!(
        out,
        "  gdn state      {} touched per layer = {:.1}% of the {} Infinity Cache",
        human_bytes(s.working_set_bytes),
        s.ic_fraction() * 100.0,
        human_bytes(s.ic_bytes)
    );
    let _ = writeln!(
        out,
        "                 {} resident ({} slot(s) x {} per layer)",
        human_bytes(s.resident_bytes),
        s.slots,
        human_bytes(s.bytes_per_slot_per_layer)
    );
    for w in &planned.workspaces {
        let _ = writeln!(
            out,
            "  workspace      {:<16} {:>12}  [{}] from {} request(s)",
            w.kind.as_str(),
            human_bytes(w.bytes),
            w.policy,
            w.contributors
        );
    }
    let ops = planned.ops.ops();
    if !bucket.captured {
        // What forced eager: either the manifest said so, or a route here could not be captured.
        let blockers: Vec<String> = ops
            .iter()
            .filter(|o| !o.graph_safe || o.may_sync)
            .map(|o| {
                format!(
                    "{} ({}{}{})",
                    o.impl_name,
                    if o.graph_safe { "" } else { "!graph_safe" },
                    if !o.graph_safe && o.may_sync {
                        ", "
                    } else {
                        ""
                    },
                    if o.may_sync { "may_sync" } else { "" }
                )
            })
            .collect();
        let _ = writeln!(
            out,
            "  eager because  {}",
            if blockers.is_empty() {
                "the manifest asks for it; every route here is capturable".to_string()
            } else {
                format!("these routes cannot be captured: {}", blockers.join(", "))
            }
        );
    }
    let calls: u32 = ops.iter().map(|o| o.occurrences).sum();
    let _ = writeln!(
        out,
        "  ops            {} row(s), {} call(s) per step, held {}",
        ops.len(),
        calls,
        match planned.ops {
            BucketOps::Captured(_) => "as captured ops (every one carries a GraphSafe proof)",
            BucketOps::Eager(_) => "eager (no capture proof required)",
        }
    );
    for op in ops {
        explain_op(out, op);
    }
}

fn explain_op(out: &mut String, op: &BoundOp) {
    let _ = writeln!(
        out,
        "    {}  x{}  {}  [{}]",
        op.op_name, op.occurrences, op.op, op.shape
    );
    let _ = writeln!(
        out,
        "      impl       {} (plugin {}, wave{})",
        op.impl_name, op.plugin, op.wave_size
    );
    // A tie says the margin decided nothing and names what did; it is printed on the same line as
    // the margin, because a margin of +0.4% with no explanation is how a reader concludes that a
    // sweep chose the faster kernel when it did not.
    let margin = match (op.margin_pct, op.tie_broken_by) {
        (Some(m), Some(t)) => format!(
            "   margin {m:+.1}% over the runner-up (inside the {NOISE_FLOOR_PCT}% / \
             {NOISE_FLOOR_US} us noise floor; tie broken by the {t})"
        ),
        (Some(m), None) if m < NOISE_FLOOR_PCT => format!(
            "   margin {m:+.1}% over the runner-up (below the {NOISE_FLOOR_PCT}% noise floor)"
        ),
        (Some(m), None) => format!("   margin {m:+.1}% over the runner-up"),
        (None, Some(t)) => format!("   tie broken by the {t}"),
        (None, None) => String::new(),
    };
    // A cell the two passes disagreed about is NOT a measurement, and the number beside it would
    // be read as one. Say so on its own line, above the variant.
    if let Some(u) = &op.unstable {
        let _ = writeln!(
            out,
            "      *** NOT REPRODUCIBLE *** pass 1 chose variant {} at {:.2} us, pass 2 chose {} \
             at {:.2} us; worst swing on one variant {:.1}%. The rules keep this cell.",
            u.first.variant_id,
            u.first.measured_us,
            u.second.variant_id,
            u.second.measured_us,
            u.worst_swing_pct
        );
    }
    match &op.not_measured {
        // Loud, and in the place a reader looks for the number: there is no number.
        Some(why) => {
            let _ = writeln!(
                out,
                "      variant    {}   *** NOT MEASURED *** ({why})",
                op.variant
            );
        }
        None => {
            let _ = writeln!(
                out,
                "      variant    {}   {:.2} us{}",
                op.variant, op.measured_us, margin
            );
        }
    }
    let _ = writeln!(out, "      state      {}", op.promotion);
    let _ = writeln!(
        out,
        "      capture    graph_safe={} may_sync={}   launches {} each, {} per step",
        yes_no(op.graph_safe),
        yes_no(op.may_sync),
        op.launches_each,
        op.launches_total
    );
    if op.workspaces.is_empty() {
        let _ = writeln!(out, "      workspace  none");
    } else {
        for w in &op.workspaces {
            let _ = writeln!(
                out,
                "      workspace  {:<16} {:>12}  \"{}\"",
                w.kind.as_str(),
                human_bytes(w.bytes),
                w.name
            );
            // What the bucket USES, beside what is allocated. The gap is not waste: the buffer is
            // sized once at construction for the worst shape the engine can schedule, because an
            // allocation inside a graph capture is not allowed and the address has to survive
            // replay. Printing only the allocation hides the split law that decides the rest.
            if let Some(used) = w.used_bytes {
                let pct = if w.bytes > 0 {
                    used as f64 / w.bytes as f64 * 100.0
                } else {
                    0.0
                };
                let _ = writeln!(
                    out,
                    "                 {:>28}  used at this bucket ({pct:.1}% of the allocation, \
                     which is sized once for the worst shape)",
                    human_bytes(used)
                );
            }
        }
    }
    let _ = writeln!(
        out,
        "      why        op `{}` at this geometry: {} candidate(s)",
        op.selection.op,
        op.selection.candidates.len()
    );
    for c in &op.selection.candidates {
        let _ = writeln!(
            out,
            "                   {} {:<32} {}",
            if c.ok { "+" } else { "-" },
            c.impl_name,
            c.reason
        );
    }
    let chosen = op.selection.chosen();
    let _ = writeln!(
        out,
        "                 {}",
        if chosen.len() > 1 {
            format!(
                "{} candidates matched; the tie was broken by measurement, not by a cost model",
                chosen.len()
            )
        } else {
            "one candidate matched; the variant came from the autotune cache".to_string()
        }
    );
}

/// What a cold cache owes this plan: every (bucket, op) that has no measurement.
///
/// This is what `saber explain` prints instead of a plan when resolution succeeds but binding
/// cannot — the tuning campaign's work list.
pub fn explain_missing(resolved: &Resolved<'_>, autotune: &AutotuneCache) -> String {
    let missing: Vec<Requirement> = resolved.missing(autotune);
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{} untuned route(s) at plan input hash {}",
        missing.len(),
        resolved.input_hash()
    );
    for r in &missing {
        let _ = writeln!(
            out,
            "  {}  {}  [{}]  candidates: {}",
            r.bucket.label(),
            r.op_name,
            r.key.shape,
            if r.candidates.is_empty() {
                "(none)".to_string()
            } else {
                r.candidates.join(", ")
            }
        );
    }
    out
}

fn yes_no(v: bool) -> &'static str {
    if v {
        "yes"
    } else {
        "no"
    }
}

fn short(hash: &str) -> &str {
    if hash.len() > 8 {
        &hash[..8]
    } else {
        hash
    }
}
