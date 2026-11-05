//! FN103 (`saber flashnext-profile admcensus`): admission / insertion by EXPECTED EXPOSED COST.
//!
//! Both ranks' REAL tiers replayed in lockstep over a trace's target steps. A decode / verify
//! step's MoE layer ends at the slower rank (EP=2: the layer's reduce waits for both cards), and
//! a RAM read is link-bound and additive inside the block (FN84: 40-79 us a read at b1, FN47b: 88
//! us at b8, FN83: 102 us in a prefill chunk), so a layer's EXPOSED reads are `max(r0, r1)` and
//! the faster rank's `min(r0, r1)` are HIDDEN behind the slower rank's link. The census
//! attributes the slower rank's reads (a tie: half each) by reuse distance (steps since the
//! record was last routed on its rank) and by what today's admission did with the miss at the
//! step's end (inserted / held by the second chance / capped / a read-through layer), priced at
//! `--exposed-us` by bucket. `--admit-law` replays a per-layer insert-cap planner
//! (`ExpertTier::set_layer_caps`) beside today's law: `slow:HI:LO:THR` (a rank inserts HI a layer
//! where its EWMA read excess over the other rank is > THR, LO where < -THR, today's cap
//! between), `xexp:TOTAL:MAX` (the step's TOTAL inserts a rank shared by each layer's EWMA of the
//! rank's EXPOSED reads, at most MAX a layer), `miss:TOTAL:MAX` (shared by the EWMA of its reads).

use super::{load_profile, params, windows, ProfileArgs, PER_RANK, STRIDE};
use anyhow::{bail, Context};
use saber_mem::expert_tier::{ExpertTier, Placement};
use saber_mem::route_trace::{replay_header, CountMover, InsertLaw, StepKind, TraceReader};
use std::collections::BTreeMap;

const CLASSES: [&str; 5] = ["d1", "d2-8", "d9-64", "d65+", "first"];
const FATES: [&str; 4] = ["inserted", "sc-held", "capped", "read-thru"];

#[derive(Debug, Clone, Copy, PartialEq)]
enum Law {
    Today,
    Slow { hi: u32, lo: u32, thr: f64 },
    Xexp { total: u32, max: u32 },
    Miss { total: u32, max: u32 },
}

fn parse_law(s: &str) -> anyhow::Result<Law> {
    let f: Vec<&str> = s.split(':').collect();
    let n = |k: usize| -> anyhow::Result<f64> {
        Ok(f.get(k)
            .context("--admit-law field")?
            .trim()
            .parse::<f64>()?)
    };
    Ok(match f[0] {
        "today" => Law::Today,
        "slow" => Law::Slow {
            hi: n(1)? as u32,
            lo: n(2)? as u32,
            thr: n(3)?,
        },
        "xexp" => Law::Xexp {
            total: n(1)? as u32,
            max: n(2)? as u32,
        },
        "miss" => Law::Miss {
            total: n(1)? as u32,
            max: n(2)? as u32,
        },
        l => bail!("--admit-law {l} (today | slow:HI:LO:THR | xexp:TOTAL:MAX | miss:TOTAL:MAX)"),
    })
}

fn parse_us(s: &str) -> anyhow::Result<BTreeMap<u32, f64>> {
    s.split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| {
            let (b, v) = p.split_once(':').context("--exposed-us bucket:us")?;
            Ok((b.trim().parse()?, v.trim().parse()?))
        })
        .collect()
}

fn class_of(now: u64, last: u64) -> usize {
    if last == 0 {
        return 4;
    }
    match now - last {
        0 | 1 => 0,
        2..=8 => 1,
        9..=64 => 2,
        _ => 3,
    }
}

#[derive(Debug, Clone, Default)]
struct Acc {
    steps: f64,
    rows: f64,
    reads: [f64; 2],
    inserts: [f64; 2],
    exposed: f64,
    hidden: f64,
    /// Exposed reads by (class, fate), and every read by (class, fate) (both ranks).
    ex: [[f64; 4]; 5],
    all: [[f64; 4]; 5],
    /// Per layer: exposed reads, rank 0 the strictly slower, ties with reads.
    layer_ex: Vec<f64>,
    layer_r0_slow: Vec<f64>,
    /// Sign persistence: the slower rank == the one the EWMA predicted.
    pred_ok: f64,
    pred_n: f64,
}

/// One replayed miss: record index, reuse class, second-chance-held prediction, read-through
/// layer.
struct Miss {
    i: u32,
    cls: usize,
    held: bool,
    rt: bool,
}

/// The census + one admission law, both ranks in lockstep.
pub(super) fn admcensus(a: &ProfileArgs) -> anyhow::Result<()> {
    let ws = windows(a, &a.only)?;
    let prof = load_profile(&a.profile)?;
    let law = InsertLaw::parse(a.max_inserts, &a.by_bucket)
        .and_then(|l| l.with_second_chance(&a.second_chance))
        .map_err(anyhow::Error::msg)?;
    let admit = parse_law(&a.admit_law)?;
    let us = parse_us(&a.exposed_us)?;
    let alpha = a.admit_alpha;
    let mut r = TraceReader::open(&a.trace[0])?;
    let nl = r.layers as usize;
    let mut tiers = Vec::new();
    for rank in [0u32, 1] {
        let h = replay_header(rank, r.layers, PER_RANK, STRIDE);
        tiers.push(
            ExpertTier::new(&h, params(a), prof.as_ref(), 0, 1 << 40)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        );
    }
    let mut mv = [CountMover::default(), CountMover::default()];
    let recs = tiers[0].records() as usize;
    let mut last_route = [vec![0u64; recs], vec![0u64; recs]];
    let mut ew_d = [vec![0f64; nl], vec![0f64; nl]];
    let mut ew_x = [vec![0f64; nl], vec![0f64; nl]];
    let mut ew_m = [vec![0f64; nl], vec![0f64; nl]];
    let mut acc: BTreeMap<u32, Acc> = BTreeMap::new();
    let mut tot_res = [0f64; 2];
    let mut tot_ram = [0f64; 2];
    let thresh = a.read_through;
    while let Some(s) = r.next_step()? {
        if !s.kind.target() {
            continue;
        }
        if let Some(ws) = &ws {
            if !ws.iter().any(|w| w.holds(s.t)) {
                continue;
            }
        }
        let bucket = if s.kind == StepKind::Prefill {
            0
        } else {
            s.bucket
        };
        let w = u64::from(law.second_chance(bucket));
        for t in &mut tiers {
            t.set_max_inserts(law.cap(bucket));
            t.set_second_chance(law.second_chance(bucket));
            t.set_layer_caps(None);
            t.begin_step();
        }
        let now = tiers[0].steps() + 1;
        let mut misses: [Vec<Vec<Miss>>; 2] = [Vec::new(), Vec::new()];
        let mut reads = vec![[0u32; 2]; nl];
        let mut routed_local: [Vec<Vec<u32>>; 2] = [Vec::new(), Vec::new()];
        for (l, rd) in reads.iter_mut().enumerate() {
            let lu = l as u32;
            let routed: Vec<u32> = s.layer(l).iter().map(|x| u32::from(x.e)).collect();
            for rk in 0..2 {
                let t = &mut tiers[rk];
                let c = t
                    .resolve(lu, &routed, &mut mv[rk])
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let n = t.layer_slots(lu).1;
                let distinct = c.resident + c.ram + c.nvme;
                let rt = n > 0 && distinct as f64 > thresh * f64::from(n);
                let mut loc: Vec<u32> = routed.iter().filter_map(|&e| t.index(lu, e)).collect();
                loc.sort_unstable();
                loc.dedup();
                let mut ms = Vec::new();
                for &i in &loc {
                    let e = i % PER_RANK + rk as u32 * PER_RANK;
                    if matches!(t.placement(lu, e), Some(Placement::Ram(_))) {
                        let lr = t.last_ram_route(i);
                        ms.push(Miss {
                            i,
                            cls: class_of(now, last_route[rk][i as usize]),
                            held: w > 0 && (lr == 0 || now - lr > w),
                            rt,
                        });
                    }
                }
                rd[rk] = (c.ram + c.nvme) as u32;
                tot_res[rk] += c.resident as f64;
                tot_ram[rk] += (c.ram + c.nvme) as f64;
                misses[rk].push(ms);
                routed_local[rk].push(loc);
            }
        }
        let ac = acc.entry(bucket).or_insert_with(|| Acc {
            layer_ex: vec![0.0; nl],
            layer_r0_slow: vec![0.0; nl],
            ..Acc::default()
        });
        // the planner: EWMAs over decode steps (this step's reads included), then the caps
        if bucket > 0 {
            for l in 0..nl {
                let (r0, r1) = (f64::from(reads[l][0]), f64::from(reads[l][1]));
                if r0 != r1 {
                    let pred0 = ew_d[0][l] > 0.0;
                    ac.pred_n += 1.0;
                    if pred0 == (r0 > r1) {
                        ac.pred_ok += 1.0;
                    }
                }
                for rk in 0..2 {
                    let (me, ot) = if rk == 0 { (r0, r1) } else { (r1, r0) };
                    let xs = if me > ot {
                        me
                    } else if me == ot {
                        me / 2.0
                    } else {
                        0.0
                    };
                    ew_d[rk][l] = (1.0 - alpha) * ew_d[rk][l] + alpha * (me - ot);
                    ew_x[rk][l] = (1.0 - alpha) * ew_x[rk][l] + alpha * xs;
                    ew_m[rk][l] = (1.0 - alpha) * ew_m[rk][l] + alpha * me;
                }
            }
            let base = law.cap(bucket);
            for rk in 0..2 {
                let caps: Option<Vec<u32>> = match admit {
                    Law::Today => None,
                    Law::Slow { hi, lo, thr } => Some(
                        ew_d[rk]
                            .iter()
                            .map(|&d| {
                                if d > thr {
                                    hi
                                } else if d < -thr {
                                    lo
                                } else {
                                    base
                                }
                            })
                            .collect(),
                    ),
                    Law::Xexp { total, max } | Law::Miss { total, max } => {
                        let wv = if matches!(admit, Law::Xexp { .. }) {
                            &ew_x[rk]
                        } else {
                            &ew_m[rk]
                        };
                        let sum: f64 = wv.iter().sum();
                        Some(if sum <= 0.0 {
                            vec![base; nl]
                        } else {
                            wv.iter()
                                .map(|&x| {
                                    ((f64::from(total) * x / sum + 0.5).floor() as u32).min(max)
                                })
                                .collect()
                        })
                    }
                };
                if let Some(c) = caps {
                    let mx = c.iter().copied().max().unwrap_or(base).max(1);
                    tiers[rk].set_max_inserts(mx);
                    tiers[rk].set_layer_caps(Some(&c));
                }
            }
        }
        for rk in 0..2 {
            tiers[rk]
                .refill(&mut mv[rk])
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let rep = tiers[rk]
                .end_step(&mut mv[rk])
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            ac.inserts[rk] += rep.moves.inserts as f64;
        }
        ac.steps += 1.0;
        ac.rows += f64::from(s.rows);
        for l in 0..nl {
            let (r0, r1) = (reads[l][0], reads[l][1]);
            ac.exposed += f64::from(r0.max(r1));
            ac.hidden += f64::from(r0.min(r1));
            ac.layer_ex[l] += f64::from(r0.max(r1));
            if r0 > r1 {
                ac.layer_r0_slow[l] += 1.0;
            }
            let wt = if r0 > r1 {
                [1.0, 0.0]
            } else if r1 > r0 {
                [0.0, 1.0]
            } else {
                [0.5, 0.5]
            };
            for rk in 0..2 {
                ac.reads[rk] += f64::from(reads[l][rk]);
                for m in &misses[rk][l] {
                    let e = m.i % PER_RANK + rk as u32 * PER_RANK;
                    let fate =
                        if matches!(tiers[rk].placement(l as u32, e), Some(Placement::Vram(_))) {
                            0
                        } else if m.rt {
                            3
                        } else if m.held {
                            1
                        } else {
                            2
                        };
                    ac.all[m.cls][fate] += 1.0;
                    ac.ex[m.cls][fate] += wt[rk];
                }
            }
            for rk in 0..2 {
                for &i in &routed_local[rk][l] {
                    last_route[rk][i as usize] = now;
                }
            }
        }
    }
    let lab = &a.label;
    println!(
        "fnadmit {lab} law {:?} vram {} lambda {} cap {} sc '{}' rt {} alpha {alpha} exposed-us {:?}",
        admit, a.vram, a.lambda, a.max_inserts, a.second_chance, a.read_through, us
    );
    println!(
        "fnadmit {lab} ALL: VRAM r0 {:.3} % r1 {:.3} % (both {:.3} %)",
        100.0 * tot_res[0] / (tot_res[0] + tot_ram[0]).max(1.0),
        100.0 * tot_res[1] / (tot_res[1] + tot_ram[1]).max(1.0),
        100.0 * (tot_res[0] + tot_res[1])
            / (tot_res[0] + tot_res[1] + tot_ram[0] + tot_ram[1]).max(1.0)
    );
    let mut js = serde_json::Map::new();
    for (b, x) in &acc {
        let n = x.steps.max(1.0);
        let name = if *b == 0 {
            "prefill".to_string()
        } else {
            format!("b{b}")
        };
        let c = us.get(b).copied().unwrap_or(88.0);
        println!(
            "fnadmit {lab} {name:>7}: steps {:>7} rows {:.1} | RAM reads/step r0 {:.2} r1 {:.2} | EXPOSED (sum max) {:.2} = {:.3} ms | hidden (sum min) {:.2} | inserts/step r0 {:.2} r1 {:.2} | exposed/r0-read {:.2} | slower-rank predicted {:.1} %",
            x.steps,
            x.rows / n,
            x.reads[0] / n,
            x.reads[1] / n,
            x.exposed / n,
            x.exposed / n * c / 1e3,
            x.hidden / n,
            x.inserts[0] / n,
            x.inserts[1] / n,
            x.exposed / x.reads[0].max(1.0),
            100.0 * x.pred_ok / x.pred_n.max(1.0)
        );
        for (k, cl) in CLASSES.iter().enumerate() {
            let e: f64 = x.ex[k].iter().sum();
            let al: f64 = x.all[k].iter().sum();
            println!(
                "fnadmit {lab} {name:>7}   class {cl:>6}: exposed {:.2}/step ({:.3} ms) of {:.2} reads | exposed by fate {} ",
                e / n,
                e / n * c / 1e3,
                al / n,
                FATES
                    .iter()
                    .enumerate()
                    .map(|(f, nm)| format!("{nm} {:.2}", x.ex[k][f] / n))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        let bands: Vec<String> = (0..nl.div_ceil(8))
            .map(|g| {
                let lo = g * 8;
                let hi = (lo + 8).min(nl);
                let e: f64 = x.layer_ex[lo..hi].iter().sum::<f64>() / n;
                let s0: f64 = x.layer_r0_slow[lo..hi].iter().sum::<f64>() / n / (hi - lo) as f64;
                format!("L{lo}-{}: {e:.2} ({:.0} % r0-slow)", hi - 1, 100.0 * s0)
            })
            .collect();
        println!(
            "fnadmit {lab} {name:>7}   exposed by layer band: {}",
            bands.join(" | ")
        );
        js.insert(
            name,
            serde_json::json!({"steps": x.steps, "reads": [x.reads[0] / n, x.reads[1] / n],
                "exposed": x.exposed / n, "exposed_ms": x.exposed / n * c / 1e3, "hidden": x.hidden / n,
                "inserts": [x.inserts[0] / n, x.inserts[1] / n],
                "ex_by_class_fate": x.ex.iter().map(|r| r.iter().map(|v| v / n).collect::<Vec<_>>()).collect::<Vec<_>>(),
                "all_by_class_fate": x.all.iter().map(|r| r.iter().map(|v| v / n).collect::<Vec<_>>()).collect::<Vec<_>>(),
                "layer_exposed": x.layer_ex.iter().map(|v| v / n).collect::<Vec<_>>(),
                "pred_ok": x.pred_ok / x.pred_n.max(1.0)}),
        );
    }
    js.insert(
        "vram_pct".into(),
        serde_json::json!([
            100.0 * tot_res[0] / (tot_res[0] + tot_ram[0]).max(1.0),
            100.0 * tot_res[1] / (tot_res[1] + tot_ram[1]).max(1.0)
        ]),
    );
    if let Some(o) = &a.out {
        std::fs::create_dir_all(o)?;
        let name = if lab.is_empty() { "admcensus" } else { lab };
        std::fs::write(
            o.join(format!("{name}.json")),
            serde_json::to_string_pretty(&js)?,
        )?;
    }
    Ok(())
}
