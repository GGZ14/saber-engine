//! LR-H (`saber flashnext-profile ringorder`): the staging ring's ORDER KEY priced offline.
//!
//! One rank's REAL tier (`--ring-experts` local experts: 512 = the tp2 expert law, every rank
//! holds every expert's half) replayed over a route trace's target steps. Every prefill chunk of
//! at least `--ring-min-rows` rows is a STAGED PASS: at the pass's (frozen) placements both keys
//! plan the ring (`StagePlan::plan_by`, 2 x `--ring-slots`), both plans are classified at the
//! pass's routes (`classify_seen`: ring / uva / staged unrouted / misordered), then the served
//! key notes the pass (`decay_prefill_seen` + `note_prefill_seen`) exactly as the engine's
//! tier_end does -- the replay IS the engine's host code, so its counterfactual is the engine's.
//! Reported per `--ring-bands` row band: passes, mean staged, and per key the mean staged
//! unrouted (the link bytes spent for nothing), routed-on-UVA and misordered, plus the waste cut
//! served vs profile. lr-c's gate: proceed to a hold only if the waste falls >= 40 % at 2-4.5k.

use super::{load_profile, params, windows, ProfileArgs};
use anyhow::Context;
use saber_mem::expert_tier::ExpertTier;
use saber_mem::route_trace::{replay_header, CountMover, InsertLaw, StepKind, TraceReader};
use saber_mem::stage_ring::{routed_seen, RingShape, StageCounts, StageKey, StagePlan};

/// One band's sums.
#[derive(Default, Clone, Copy)]
struct Band {
    passes: u64,
    rows: u64,
    prof: StageCounts,
    served: StageCounts,
}

#[allow(clippy::too_many_lines)]
pub(super) fn ringorder(a: &ProfileArgs) -> anyhow::Result<()> {
    let ws = windows(a, &a.only)?;
    let prof = load_profile(&a.profile)?;
    let law = InsertLaw::parse(a.max_inserts, &a.by_bucket).map_err(anyhow::Error::msg)?;
    let bands: Vec<(u32, u32)> = a
        .ring_bands
        .split(',')
        .map(|b| {
            let (lo, hi) = b.split_once('-').context("band lo-hi")?;
            Ok((lo.trim().parse()?, hi.trim().parse()?))
        })
        .collect::<anyhow::Result<_>>()?;
    let shape = RingShape {
        buffers: 2,
        slots: a.ring_slots,
    };
    let mut sums = vec![Band::default(); bands.len()];
    let mut all = Band::default();
    let (mut steps, mut pf, mut max_rows) = (0u64, 0u64, 0u32);
    for t in &a.trace {
        let mut r = TraceReader::open(t)?;
        let h = replay_header(0, r.layers, a.ring_experts, a.ring_stride);
        let mut p = params(a);
        p.ram_slots = a
            .ram
            .unwrap_or((r.layers * a.ring_experts).saturating_sub(a.vram));
        let mut tier = ExpertTier::new(&h, p, prof.as_ref(), 0, 1 << 40)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut mv = CountMover::default();
        while let Some(s) = r.next_step()? {
            if !s.kind.target() {
                continue;
            }
            if let Some(ws) = &ws {
                if !ws.iter().any(|w| w.holds(s.t)) {
                    continue;
                }
            }
            steps += 1;
            let is_pf = s.kind == StepKind::Prefill;
            if is_pf {
                pf += 1;
                max_rows = max_rows.max(s.rows);
            }
            tier.set_max_inserts(law.cap(if is_pf { 0 } else { s.bucket }));
            tier.begin_step();
            let staged = is_pf && s.rows >= a.ring_min_rows;
            let plans = staged.then(|| {
                (
                    StagePlan::plan_by(&tier, r.layers, shape, StageKey::Profile),
                    StagePlan::plan_by(&tier, r.layers, shape, StageKey::Served),
                )
            });
            if staged {
                tier.decay_prefill_seen();
            }
            let (mut cp, mut cs) = (StageCounts::default(), StageCounts::default());
            for l in 0..s.layers() {
                let lu = l as u32;
                let routed: Vec<u32> = s.layer(l).iter().map(|x| u32::from(x.e)).collect();
                if let Some((pp, sp)) = &plans {
                    let seen = routed_seen(&tier, &routed);
                    cp += pp.classify_seen(&tier, lu, &seen);
                    cs += sp.classify_seen(&tier, lu, &seen);
                    tier.note_prefill_seen(lu, &seen);
                }
                tier.resolve(lu, &routed, &mut mv)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            tier.refill(&mut mv).map_err(|e| anyhow::anyhow!("{e}"))?;
            tier.end_step(&mut mv).map_err(|e| anyhow::anyhow!("{e}"))?;
            if staged {
                let add = |b: &mut Band| {
                    b.passes += 1;
                    b.rows += u64::from(s.rows);
                    b.prof += cp;
                    b.served += cs;
                };
                add(&mut all);
                for (k, &(lo, hi)) in bands.iter().enumerate() {
                    if s.rows >= lo && s.rows <= hi {
                        add(&mut sums[k]);
                    }
                }
            }
        }
    }
    println!(
        "== LR-H ringorder{}: {} target steps, {} prefill chunks (widest {} rows), staged passes \
         (>= {} rows) {} | ring 2 x {} | {} local experts, vram {}",
        if a.label.is_empty() {
            String::new()
        } else {
            format!(" [{}]", a.label)
        },
        steps,
        pf,
        max_rows,
        a.ring_min_rows,
        all.passes,
        a.ring_slots,
        a.ring_experts,
        a.vram
    );
    println!(
        "band | passes | mean rows | staged | profile: unrouted uva misordered | served: unrouted \
         uva misordered | waste cut | misordered recovered"
    );
    let mut out = serde_json::Map::new();
    for (k, &(lo, hi)) in bands
        .iter()
        .enumerate()
        .chain([(bands.len(), &(0, u32::MAX))])
    {
        let (b, name) = if k == bands.len() {
            (all, "all".to_string())
        } else {
            (sums[k], format!("{lo}-{hi}"))
        };
        let n = b.passes.max(1) as f64;
        let m = |x: u64| x as f64 / n;
        let cut = if b.prof.wasted == 0 {
            0.0
        } else {
            100.0 * (1.0 - b.served.wasted as f64 / b.prof.wasted as f64)
        };
        // the order key's own ceiling: a swap removes one staged-unrouted record AND one UVA read,
        // so the waste can fall at most by `misordered` (lr-c's gate reads this share too)
        let rec = if b.prof.misordered == 0 {
            0.0
        } else {
            100.0 * (1.0 - b.served.misordered as f64 / b.prof.misordered as f64)
        };
        println!(
            "{name} | {} | {:.0} | {:.0} | {:.0} {:.0} {:.0} | {:.0} {:.0} {:.0} | {cut:.1} % | {rec:.1} %",
            b.passes,
            m(b.rows),
            m(b.prof.staged),
            m(b.prof.wasted),
            m(b.prof.uva),
            m(b.prof.misordered),
            m(b.served.wasted),
            m(b.served.uva),
            m(b.served.misordered),
        );
        out.insert(
            name,
            serde_json::json!({"passes": b.passes, "mean_rows": m(b.rows),
                "staged": m(b.prof.staged),
                "profile": {"unrouted": m(b.prof.wasted), "uva": m(b.prof.uva),
                    "ring": m(b.prof.ring), "misordered": m(b.prof.misordered)},
                "served": {"unrouted": m(b.served.wasted), "uva": m(b.served.uva),
                    "ring": m(b.served.ring), "misordered": m(b.served.misordered)},
                "waste_cut_pct": cut, "misordered_recovered_pct": rec}),
        );
    }
    if let Some(o) = &a.out {
        std::fs::create_dir_all(o)?;
        let name = if a.label.is_empty() {
            "ringorder".to_string()
        } else {
            a.label.clone()
        };
        std::fs::write(
            o.join(format!("{name}.json")),
            serde_json::to_string_pretty(&out)?,
        )?;
    }
    Ok(())
}
