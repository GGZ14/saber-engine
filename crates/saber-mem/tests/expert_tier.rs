//! FN7: the expert tier over a synthetic `experts.xtier` (the real format, small records), with
//! the byte-true host mover — every slot's bytes are checked against the sha256 table after every
//! step, every `xptr` entry a routed expert's kernel would read is checked to land on that
//! expert's record, and the forced-miss control moves every counter as predicted.

use std::collections::BTreeSet;

use saber_mem::dio::OpenMode;
use saber_mem::expert_mover::HostMover;
use saber_mem::expert_tier::{
    ExpertTier, PathCounts, Placement, RouteProfile, TierError, TierParams,
};
use saber_mem::xtier::{write_synthetic, SyntheticTier, XtierFile};
use saber_test_scratch::ScratchDir;

const LAYERS: u32 = 4;
const EXPERTS: u32 = 16; // per rank
const VRAM_BASE: u64 = 0x7f00_0000_0000;
const RAM_BASE: u64 = 0x5500_0000_0000;

fn synth(rank: u32, tag: &str) -> (ScratchDir, XtierFile) {
    let d = ScratchDir::new(&format!("saber-fn7-{tag}-r{rank}"));
    let p = d.join("experts.xtier");
    write_synthetic(
        &p,
        SyntheticTier {
            rank,
            layers: LAYERS,
            first_expert: EXPERTS * rank,
            experts: EXPERTS,
            hidden: 256,
            inter: 64,
        },
    )
    .unwrap();
    let f = XtierFile::open(&p, OpenMode::Buffered).unwrap();
    (d, f)
}

/// A tiny deterministic generator (no rand dependency).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    /// A skewed expert id in [0, 2 * EXPERTS): low ids are hot.
    fn expert(&mut self) -> u32 {
        let a = self.next() % (2 * u64::from(EXPERTS));
        let b = self.next() % (2 * u64::from(EXPERTS));
        a.min(b) as u32
    }
}

/// Routes for one step: per layer, `tokens × 10` global ids (both ranks' experts).
fn step_routes(g: &mut Lcg, tokens: usize) -> Vec<Vec<u32>> {
    (0..LAYERS)
        .map(|_| (0..tokens * 10).map(|_| g.expert()).collect())
        .collect()
}

/// Check every slot and every routed expert's xptr entries against the table.
fn check_bytes(tier: &ExpertTier, mv: &HostMover, xt: &XtierFile, routes: &[Vec<u32>]) {
    check_layer(tier, mv, xt, routes, None);
}

/// The slot checks, and the xptr checks of `only_layer` (all layers when `None`). A layer's
/// routed experts are dispatchable right after ITS resolve; a later layer's demand read may drop
/// an earlier layer's RAM record (its kernel already ran), so the xptr check is per layer.
fn check_layer(
    tier: &ExpertTier,
    mv: &HostMover,
    xt: &XtierFile,
    routes: &[Vec<u32>],
    only_layer: Option<usize>,
) {
    let h = xt.header();
    tier.check_invariants().unwrap();
    for s in 0..tier.params().vram_slots {
        if let Some(r) = tier.vram_record(s) {
            assert!(
                xt.verify(u64::from(r), mv.vram_slot(s)),
                "vram slot {s} != record {r}"
            );
        }
    }
    for s in 0..=tier.params().ram_slots {
        if let Some(r) = tier.ram_record(s) {
            assert!(
                xt.verify(u64::from(r), mv.ram_slot(s)),
                "ram slot {s} != record {r}"
            );
        }
    }
    let n = EXPERTS as usize;
    let (mut gu, mut dn) = (vec![0u64; 3 * (n + 1)], vec![0u64; 3 * (n + 1)]);
    for (l, rs) in routes.iter().enumerate() {
        if only_layer.is_some_and(|o| o != l) {
            continue;
        }
        tier.fill_xptr(l as u32, &mut gu, &mut dn);
        for &e in rs {
            let Some(i) = tier.index(l as u32, e) else {
                continue;
            };
            let loc = (e - h.first_expert) as usize;
            // Resolve the address back to a slot and check the slot holds record i.
            let a = gu[3 * loc];
            let (bytes, base) = if a >= VRAM_BASE {
                let s = (a - VRAM_BASE) / h.stride;
                (mv.vram_slot(s as u32), VRAM_BASE + s * h.stride)
            } else {
                let s = (a - RAM_BASE) / h.stride;
                (mv.ram_slot(s as u32), RAM_BASE + s * h.stride)
            };
            assert!(
                xt.verify(u64::from(i), bytes),
                "layer {l} expert {e}: xptr lands on another record"
            );
            for k in 0..3 {
                assert_eq!(gu[3 * loc + k], base + h.sub[k].0);
                assert_eq!(dn[3 * loc + k], base + h.sub[3 + k].0);
            }
        }
    }
}

fn run_steps(
    tier: &mut ExpertTier,
    mv: &mut HostMover,
    xt: &XtierFile,
    trace: &[Vec<Vec<u32>>],
    check: bool,
) -> Result<PathCounts, TierError> {
    let mut tot = PathCounts::default();
    for routes in trace {
        tier.begin_step();
        for (l, rs) in routes.iter().enumerate() {
            let c = tier.resolve(l as u32, rs, mv)?;
            tier.check_dispatch(l as u32, rs)?;
            if check {
                check_layer(tier, mv, xt, routes, Some(l));
            }
            tot.resident += c.resident;
            tot.ram += c.ram;
            tot.nvme += c.nvme;
            tot.prefetched += c.prefetched;
        }
        tier.end_step(mv)?;
        if check {
            check_layer(tier, mv, xt, &[], None);
        }
    }
    Ok(tot)
}

fn tier_and_mover<'x>(
    xt: &'x XtierFile,
    params: TierParams,
    profile: Option<&RouteProfile>,
) -> (ExpertTier, HostMover<'x>) {
    let tier = ExpertTier::new(xt.header(), params.clone(), profile, VRAM_BASE, RAM_BASE).unwrap();
    let mut mv = HostMover::new(xt, params.vram_slots, params.ram_slots, 4);
    mv.warm(&tier.warm_plan(), params.ram_slots).unwrap();
    (tier, mv)
}

fn params(vram: u32, ram: u32) -> TierParams {
    TierParams {
        vram_slots: vram,
        ram_slots: ram,
        min_layer_slots: 2,
        max_inserts: 3,
        ..TierParams::default()
    }
}

#[test]
fn every_byte_every_step_both_ranks_complete_and_demand() {
    for rank in [0, 1] {
        let (_p, xt) = synth(rank, "bytes");
        // Complete (VRAM 24 + RAM 40 = all 64 records) and demand (RAM 12: drops happen).
        for (vram, ram) in [(40, 24), (40, 12)] {
            let (mut tier, mut mv) = tier_and_mover(&xt, params(vram, ram), None);
            assert_eq!(tier.complete(), vram + ram >= LAYERS * EXPERTS);
            let mut g = Lcg(0x5eed + u64::from(rank));
            let trace: Vec<_> = (0..40).map(|s| step_routes(&mut g, 1 + s % 2)).collect();
            let tot = run_steps(&mut tier, &mut mv, &xt, &trace, true).unwrap();
            let (paths, moves) = tier.totals();
            assert_eq!(paths.routed(), tot.routed());
            assert!(moves.inserts > 0, "the LRU must move something");
            if tier.params().ram_slots >= LAYERS * EXPERTS - vram {
                assert_eq!(
                    paths.nvme, 0,
                    "complete tier: the NVMe is off the step path"
                );
                assert_eq!(mv.nvme_reads, 0);
            } else {
                assert!(
                    paths.nvme > 0 && moves.ram_drops > 0,
                    "demand tier must miss and drop"
                );
                assert_eq!(mv.nvme_reads, paths.nvme, "one read per demand miss");
            }
            assert_eq!(mv.copies, (moves.inserts, moves.demotions));
            let text = tier.render_prometheus();
            assert!(text.contains(&format!(
                "saber_xtier_routed_total{{rank=\"{rank}\",path=\"resident\"}} {}",
                paths.resident
            )));
        }
    }
}

#[test]
fn forced_miss_control_moves_every_counter_as_predicted() {
    let (_p, xt) = synth(0, "fm");
    let (mut tier, mut mv) = tier_and_mover(&xt, params(48, 64), None);
    let mut g = Lcg(77);
    let trace: Vec<_> = (0..30).map(|_| step_routes(&mut g, 1)).collect();
    // Warm pass: complete tier, no NVMe.
    let warm = run_steps(&mut tier, &mut mv, &xt, &trace, false).unwrap();
    assert_eq!(warm.nvme, 0);
    assert!(warm.resident > 0);
    // THE CONTROL: evict everything, replay. Prediction, computed from the routes alone: with RAM
    // large enough never to drop, every distinct (layer, expert) of this rank is read from NVMe
    // exactly once (its first touch) and never again; the first step has 0 resident / 0 ram.
    tier.evict_all();
    tier.check_invariants().unwrap();
    let mut seen = BTreeSet::new();
    let mut first_step_distinct = 0u64;
    for (s, routes) in trace.iter().enumerate() {
        for (l, rs) in routes.iter().enumerate() {
            for &e in rs {
                if let Some(i) = tier.index(l as u32, e) {
                    if seen.insert(i) && s == 0 {
                        first_step_distinct += 1;
                    }
                }
            }
        }
    }
    tier.begin_step();
    let mut first = PathCounts::default();
    for (l, rs) in trace[0].iter().enumerate() {
        let c = tier.resolve(l as u32, rs, &mut mv).unwrap();
        check_layer(&tier, &mv, &xt, &trace[0], Some(l));
        first.resident += c.resident;
        first.ram += c.ram;
        first.nvme += c.nvme;
    }
    tier.end_step(&mut mv).unwrap();
    check_bytes(&tier, &mv, &xt, &[]);
    assert_eq!(
        first.resident, 0,
        "forced miss: nothing resident on the first step"
    );
    assert_eq!(first.ram, 0);
    assert_eq!(first.nvme, first_step_distinct);
    let rest = run_steps(&mut tier, &mut mv, &xt, &trace[1..], true).unwrap();
    assert_eq!(
        first.nvme + rest.nvme,
        seen.len() as u64,
        "every distinct record read from NVMe exactly once"
    );
    assert_eq!(
        first.routed() + rest.routed(),
        warm.routed(),
        "same routes, same distinct count"
    );
    assert!(rest.resident > 0, "the LRU refills VRAM after the eviction");
    // And the control's control: a second replay with no eviction reads nothing from NVMe.
    let again = run_steps(&mut tier, &mut mv, &xt, &trace, false).unwrap();
    assert_eq!(again.nvme, 0);
}

#[test]
fn decisions_are_a_pure_function_of_the_routes() {
    let (_p, xt) = synth(1, "det");
    let mk = || tier_and_mover(&xt, params(20, 10), None);
    let (mut a, mut ma) = mk();
    let (mut b, mut mb) = mk();
    let mut g = Lcg(9);
    let trace: Vec<_> = (0..25).map(|_| step_routes(&mut g, 1)).collect();
    run_steps(&mut a, &mut ma, &xt, &trace, false).unwrap();
    run_steps(&mut b, &mut mb, &xt, &trace, false).unwrap();
    for l in 0..LAYERS {
        for e in EXPERTS..2 * EXPERTS {
            assert_eq!(a.placement(l, e), b.placement(l, e));
        }
    }
    assert_eq!(a.totals(), b.totals());
    assert_eq!(a.placement(0, 0), None, "rank 1 owns [16, 32)");
}

#[test]
fn read_through_and_the_insert_cap() {
    let (_p, xt) = synth(0, "rt");
    let p = TierParams {
        vram_slots: 16,
        ram_slots: 48,
        min_layer_slots: 4,
        max_inserts: 2,
        read_through: 0.5,
        ..TierParams::default()
    };
    let (mut tier, mut mv) = tier_and_mover(&xt, p, None);
    assert_eq!(tier.layer_slots(0), (0, 4));
    // Layer 0 routes 3 distinct experts not in VRAM (uniform warm: ids 0..4 resident): 3 > 0.5 * 4
    // -> read-through, no insert. Layer 1 routes 2 misses -> 2 inserts. Layer 2 routes 2 misses
    // plus... capped at max_inserts = 2 by construction of the case below.
    tier.begin_step();
    tier.resolve(0, &[5, 6, 7], &mut mv).unwrap();
    tier.resolve(1, &[8, 9], &mut mv).unwrap();
    let r = tier.end_step(&mut mv).unwrap();
    assert_eq!(r.moves.read_through_layers, 1);
    assert_eq!(r.moves.inserts, 2);
    assert_eq!(
        r.moves.demotions, 2,
        "the uniform warm start filled every slot"
    );
    assert!(matches!(tier.placement(1, 8), Some(Placement::Vram(_))));
    assert!(matches!(tier.placement(0, 5), Some(Placement::Ram(_))));
    // The cap: 2 of layer 2's misses are inserted even though 2 slots are evictable... route 2.
    let p2 = TierParams {
        vram_slots: 32,
        ram_slots: 32,
        min_layer_slots: 8,
        max_inserts: 1,
        read_through: 1.0,
        ..TierParams::default()
    };
    let (mut t2, mut m2) = tier_and_mover(&xt, p2, None);
    t2.begin_step();
    t2.resolve(2, &[10, 11, 12], &mut m2).unwrap();
    let r2 = t2.end_step(&mut m2).unwrap();
    assert_eq!(r2.moves.inserts, 1, "capped at max_inserts");
    assert!(
        matches!(t2.placement(2, 10), Some(Placement::Vram(_))),
        "lowest id first"
    );
    check_bytes(&t2, &m2, &xt, &[vec![], vec![], vec![10, 11, 12], vec![]]);
}

#[test]
fn profile_water_fill_and_warm_start() {
    let (_p, xt) = synth(1, "prof");
    // Global counts for 32 experts x 4 layers; layer 3 is the hot one for rank 1's experts.
    let mut counts = vec![vec![0u64; 32]; LAYERS as usize];
    for (l, row) in counts.iter_mut().enumerate() {
        for (e, c) in row.iter_mut().enumerate() {
            *c = if l == 3 {
                1000 - e as u64
            } else {
                (e as u64 * 7 + l as u64) % 13
            };
        }
    }
    let text = serde_json::json!({
        "format": "fn7-route-profile/1", "source": "test", "tokens": 5,
        "layers": 4, "experts": 32, "counts": counts,
    })
    .to_string();
    let prof = RouteProfile::from_json(&text).unwrap();
    let p = TierParams {
        vram_slots: 20,
        ram_slots: 8,
        min_layer_slots: 2,
        ..TierParams::default()
    };
    let tier = ExpertTier::new(xt.header(), p, Some(&prof), VRAM_BASE, RAM_BASE).unwrap();
    // Floors 2 x 4 = 8, the other 12 all go to layer 3 (its counts dominate): 2 + 12 = 14.
    assert_eq!(tier.layer_slots(3).1, 14);
    assert_eq!(tier.layer_slots(0).1, 2);
    assert!(
        matches!(tier.placement(3, 16), Some(Placement::Vram(_))),
        "the hottest of rank 1"
    );
    let (v, r, f) = tier.occupancy();
    assert_eq!((v, r, f), (20, 8, 64 - 28));
    assert!(!tier.complete());
    assert!(tier.boot_line().contains("demand"));
    // A malformed profile is refused by name.
    let bad = text.replace("fn7-route-profile/1", "other");
    assert!(RouteProfile::from_json(&bad).is_err());
}

/// One step of the prefetch case: layer 1's expert 15 (the hot one) every 4th step, and two
/// never-resident cold experts of layer 2 every step (ids 2..16 cycling, period 7 > the RAM's 4).
fn pf_routes(s: u32) -> Vec<Vec<u32>> {
    let mut routes = vec![vec![]; LAYERS as usize];
    routes[2] = vec![2 + (2 * s) % 14, 2 + (2 * s + 1) % 14];
    if s % 4 == 0 {
        routes[1] = vec![15];
    }
    routes
}

fn pf_run(prefetch: u32, xt: &XtierFile) -> (PathCounts, u64) {
    let p = TierParams {
        vram_slots: 8,
        ram_slots: 4,
        min_layer_slots: 2,
        max_inserts: 1,
        read_through: 0.0, // never insert: the RAM tier alone is under test
        prefetch_per_step: prefetch,
        ..TierParams::default()
    };
    let (mut tier, mut mv) = tier_and_mover(xt, p, None);
    let mut total = PathCounts::default();
    for s in 0..32u32 {
        let routes = pf_routes(s);
        tier.begin_step();
        for (l, rs) in routes.iter().enumerate() {
            let c = tier.resolve(l as u32, rs, &mut mv).unwrap();
            total.prefetched += c.prefetched;
            total.nvme += c.nvme;
            total.ram += c.ram;
        }
        tier.end_step(&mut mv).unwrap();
        check_bytes(&tier, &mv, xt, &[]);
    }
    (total, tier.totals().1.prefetch_reads)
}

#[test]
fn prefetch_brings_back_the_dropped_hot_expert() {
    let (_p, xt) = synth(0, "pf");
    let (with, reads) = pf_run(1, &xt);
    let (without, reads0) = pf_run(0, &xt);
    assert_eq!(
        (without.prefetched, reads0),
        (0, 0),
        "the control prefetches nothing"
    );
    assert!(
        with.prefetched >= 6,
        "the hot expert came back by prefetch: {with:?}"
    );
    assert!(reads >= with.prefetched);
    assert!(
        with.nvme < without.nvme,
        "prefetch {with:?} vs demand-only {without:?}"
    );
}

#[test]
fn ram_exhaustion_is_loud() {
    let (_p, xt) = synth(0, "ex");
    let (mut tier, mut mv) = tier_and_mover(&xt, params(8, 2), None);
    tier.begin_step();
    // Layer 3's experts 10..16 are all Fetch (uniform warm put 0,1 in VRAM and RAM holds 2).
    let e = tier.resolve(3, &[10, 11, 12, 13, 14], &mut mv).unwrap_err();
    assert!(matches!(e, TierError::RamExhausted { .. }), "{e}");
}

/// FN39: the LRFU admission margin -- a RAM route whose LRFU score does not beat the victim's by
/// the margin stays in RAM (the victim stays); once its score clears the margin it is promoted.
/// `admit_uses` gates on routed steps since the record entered RAM. Margin 0 / uses 0 = today.
#[test]
fn admission_margin_and_uses() {
    let (_p, xt) = synth(0, "fn39");
    let base = TierParams {
        vram_slots: 16,
        ram_slots: 48,
        min_layer_slots: 4,
        max_inserts: 2,
        read_through: 1.0,
        lrfu_lambda: 0.98,
        ..TierParams::default()
    };
    // Layer 0 has 4 VRAM slots, warm ids 0..3; routing 0, 1, 2 leaves id 3 (score 0) the victim,
    // and id 5 (RAM) is the miss: score 1.0 on its first route, 1.98 on its second.
    let run = |p: TierParams| {
        let (mut tier, mut mv) = tier_and_mover(&xt, p, None);
        let mut out = Vec::new();
        for _ in 0..2 {
            tier.begin_step();
            tier.resolve(0, &[0, 1, 2, 5], &mut mv).unwrap();
            let r = tier.end_step(&mut mv).unwrap();
            out.push((
                r.moves.inserts,
                r.moves.refused,
                matches!(tier.placement(0, 5), Some(Placement::Vram(_))),
            ));
            tier.check_invariants().unwrap();
        }
        check_bytes(&tier, &mv, &xt, &[vec![0, 1, 2, 5], vec![], vec![], vec![]]);
        out
    };
    // Today (margin 0, uses 0): promoted on the first miss.
    assert_eq!(run(base.clone()), vec![(1, 0, true), (0, 0, true)]);
    // Margin 1.5: 1.0 < 0 + 1.5 -> refused; 1.98 >= 1.5 -> promoted on the second step.
    let m = TierParams {
        admit_margin: 1.5,
        ..base.clone()
    };
    assert_eq!(run(m), vec![(0, 1, false), (1, 0, true)]);
    // A margin the score never reaches in two steps: never promoted.
    let m3 = TierParams {
        admit_margin: 3.0,
        ..base.clone()
    };
    assert_eq!(run(m3), vec![(0, 1, false), (0, 1, false)]);
    // Uses 2: the first RAM route is one use -> refused; the second -> promoted.
    let u = TierParams {
        admit_uses: 2,
        ..base.clone()
    };
    assert_eq!(run(u), vec![(0, 1, false), (1, 0, true)]);
    // Margin 0 is today's law byte for byte on a random trace.
    let mut g = Lcg(39);
    let trace: Vec<_> = (0..200).map(|_| step_routes(&mut g, 1)).collect();
    let go = |p: TierParams| {
        let (mut tier, mut mv) = tier_and_mover(&xt, p, None);
        let c = run_steps(&mut tier, &mut mv, &xt, &trace, false).unwrap();
        (c, tier.totals().1, tier.warm_plan())
    };
    let a = go(TierParams {
        read_through: 0.5,
        ..base.clone()
    });
    let b = go(TierParams {
        read_through: 0.5,
        admit_margin: 0.0,
        admit_uses: 0,
        ..base.clone()
    });
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    // And a positive margin moves fewer bytes over the link on the same trace.
    let c = go(TierParams {
        read_through: 0.5,
        admit_margin: 1.0,
        ..base
    });
    assert!(
        c.1.inserts < a.1.inserts,
        "{} vs {}",
        c.1.inserts,
        a.1.inserts
    );
}

/// FN54: the per-step promotion BYTE budget and second-chance insertion are placement only, are
/// today's law byte for byte when off, and move exactly the predicted counters when on.
#[test]
fn promote_budget_and_second_chance() {
    let (_p, xt) = synth(0, "fn54");
    let stride = xt.header().stride;
    let base = TierParams {
        vram_slots: 16,
        ram_slots: 48,
        min_layer_slots: 4,
        max_inserts: 2,
        read_through: 1.0,
        lrfu_lambda: 0.98,
        ..TierParams::default()
    };
    // Second chance: layer 0's VRAM holds ids 0..3; id 5 (RAM) is the miss.
    let sc = |w: u32, steps: &[bool]| {
        let (mut tier, mut mv) = tier_and_mover(&xt, base.clone(), None);
        let mut out = Vec::new();
        for &with5 in steps {
            tier.set_second_chance(w);
            tier.begin_step();
            let r: &[u32] = if with5 { &[0, 1, 2, 5] } else { &[0, 1, 2] };
            tier.resolve(0, r, &mut mv).unwrap();
            let rep = tier.end_step(&mut mv).unwrap();
            out.push((rep.moves.inserts, rep.moves.sc_held));
            tier.check_invariants().unwrap();
        }
        check_bytes(&tier, &mv, &xt, &[vec![0, 1, 2, 5], vec![], vec![], vec![]]);
        out
    };
    assert_eq!(sc(0, &[true, true]), vec![(1, 0), (0, 0)]);
    // W 8: the first miss marks, the second (1 step later) promotes
    assert_eq!(sc(8, &[true, true]), vec![(0, 1), (1, 0)]);
    // W 1: a gap of 2 steps is outside the window -> marked again, promoted on the next miss
    assert_eq!(
        sc(1, &[true, false, true, true]),
        vec![(0, 1), (0, 0), (0, 1), (1, 0)]
    );
    // The budget: every layer misses ids 5 and 6 (cap 2 a layer = 8 candidates); 3 strides of
    // budget = 3 promotions, shared from the step's rotating layer (step 1 -> layers 1, 2, 3).
    let bud = |b: Option<u64>| {
        let (mut tier, mut mv) = tier_and_mover(&xt, base.clone(), None);
        tier.set_promote_budget(b);
        tier.begin_step();
        for l in 0..LAYERS {
            // the uniform warm split: every layer's VRAM holds its ids 0..3
            tier.resolve(l, &[0, 1, 5, 6], &mut mv).unwrap();
        }
        let rep = tier.end_step(&mut mv).unwrap();
        tier.check_invariants().unwrap();
        let vram: Vec<usize> = (0..LAYERS)
            .map(|l| {
                [5, 6]
                    .iter()
                    .filter(|&&e| matches!(tier.placement(l, e), Some(Placement::Vram(_))))
                    .count()
            })
            .collect();
        check_bytes(&tier, &mv, &xt, &vec![vec![0, 1, 5, 6]; LAYERS as usize]);
        (
            rep.moves.inserts,
            rep.moves.budget_cut,
            rep.moves.bytes_h2d,
            vram,
        )
    };
    assert_eq!(bud(None), (8, 0, 8 * stride, vec![2, 2, 2, 2]));
    assert_eq!(
        bud(Some(3 * stride + 1)),
        (3, 5, 3 * stride, vec![0, 1, 1, 1])
    );
    assert_eq!(bud(Some(0)), (0, 8, 0, vec![0, 0, 0, 0]));
    // Off (no budget, window 0) is today's law byte for byte on a random trace.
    let mut g = Lcg(54);
    let trace: Vec<_> = (0..200).map(|_| step_routes(&mut g, 1)).collect();
    let go = |b: Option<u64>, w: u32| {
        let p = TierParams {
            read_through: 0.5,
            ..base.clone()
        };
        let (mut tier, mut mv) = tier_and_mover(&xt, p, None);
        tier.set_promote_budget(b);
        tier.set_second_chance(w);
        let c = run_steps(&mut tier, &mut mv, &xt, &trace, true).unwrap();
        (c, tier.totals().1)
    };
    let today = go(None, 0);
    assert_eq!(format!("{today:?}"), format!("{:?}", go(None, 0)));
}

/// FN103: the admission planner's per-layer insert caps -- `None` is today's law byte for byte,
/// a layer capped at 0 inserts nothing, a raised cap needs `max_inserts` raised too (the cap is a
/// minimum), and every capped placement keeps the bytes.
#[test]
fn layer_caps_shape_the_inserts_and_none_is_today() {
    let (_p, xt) = synth(0, "fn103");
    let base = TierParams {
        vram_slots: 16,
        ram_slots: 48,
        min_layer_slots: 4,
        max_inserts: 2,
        read_through: 1.0,
        lrfu_lambda: 0.98,
        ..TierParams::default()
    };
    let one = |caps: Option<&[u32]>, max: u32| {
        let (mut tier, mut mv) = tier_and_mover(&xt, base.clone(), None);
        tier.set_max_inserts(max);
        tier.set_layer_caps(caps);
        tier.begin_step();
        for l in 0..LAYERS {
            // every layer's VRAM holds its ids 0..3 (the uniform warm split); 5, 6, 7 miss (4 routes
            // of 4 slots: not a read-through layer)
            tier.resolve(l, &[0, 5, 6, 7], &mut mv).unwrap();
        }
        let rep = tier.end_step(&mut mv).unwrap();
        tier.check_invariants().unwrap();
        check_bytes(&tier, &mv, &xt, &vec![vec![0, 5, 6, 7]; LAYERS as usize]);
        let vram: Vec<usize> = (0..LAYERS)
            .map(|l| {
                [5, 6, 7]
                    .iter()
                    .filter(|&&e| matches!(tier.placement(l, e), Some(Placement::Vram(_))))
                    .count()
            })
            .collect();
        (rep.moves.inserts, vram)
    };
    assert_eq!(one(None, 2), (8, vec![2, 2, 2, 2]));
    assert_eq!(one(Some(&[0, 1, 2, 3]), 2), (5, vec![0, 1, 2, 2]));
    assert_eq!(one(Some(&[0, 1, 2, 3]), 3), (6, vec![0, 1, 2, 3]));
    assert_eq!(one(Some(&[]), 2), one(None, 2));
    // None on a random trace == a tier that never heard of the planner
    let mut g = Lcg(103);
    let trace: Vec<_> = (0..200).map(|_| step_routes(&mut g, 1)).collect();
    let go = |planner: bool| {
        let p = TierParams {
            read_through: 0.5,
            ..base.clone()
        };
        let (mut tier, mut mv) = tier_and_mover(&xt, p, None);
        if planner {
            tier.set_layer_caps(Some(&[9; LAYERS as usize]));
            tier.set_layer_caps(None);
        }
        let c = run_steps(&mut tier, &mut mv, &xt, &trace, true).unwrap();
        (c, tier.totals().1)
    };
    assert_eq!(format!("{:?}", go(false)), format!("{:?}", go(true)));
}
