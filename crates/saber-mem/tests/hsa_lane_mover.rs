//! FN63b: the HSA copy lane's SIGNAL LAW ([`SignalPlanner`]) over the host model of the lane
//! ([`FakeQueue::signal_lanes`]: every job runs in ANY order its dependencies allow -- across the
//! two SDMA engines and out of issue order, a stronger model than two in-order rings; a layer
//! mark sets its word and the consumer waits `word >= mark` like the command processor). Every
//! routed read after its gate lands on its record, every slot is its record after a drain, LAZY
//! and EAGER, with and without the hold law and FN17's deferred publication (the ALL mark); the
//! placements are the one stream's; FN8's one spare makes the law a chain and the spare ring
//! breaks it; the control: the planner without its slot dependencies lets a read see the wrong
//! bytes.

use saber_mem::device_mover::{DeviceMover, FakeQueue, MarkWord, SigWhat, SignalPlanner};
use saber_mem::dio::OpenMode;
use saber_mem::expert_tier::{ExpertTier, TierParams};
use saber_mem::xtier::{write_synthetic, SyntheticTier, XtierFile};
use saber_test_scratch::ScratchDir;

const LAYERS: u32 = 4;
const EXPERTS: u32 = 16;
const VRAM_BASE: u64 = 0x7f00_0000_0000;
const RAM_BASE: u64 = 0x5500_0000_0000;

fn synth(tag: &str) -> (ScratchDir, XtierFile) {
    let d = ScratchDir::new(&format!("saber-fn63b-{tag}"));
    let p = d.join("experts.xtier");
    write_synthetic(
        &p,
        SyntheticTier {
            rank: 0,
            layers: LAYERS,
            first_expert: 0,
            experts: EXPERTS,
            hidden: 256,
            inter: 64,
        },
    )
    .unwrap();
    let xt = XtierFile::open(&p, OpenMode::Buffered).unwrap();
    (d, xt)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn routes(&mut self, tokens: usize) -> Vec<Vec<u32>> {
        (0..LAYERS)
            .map(|_| {
                (0..tokens * 10)
                    .map(|_| {
                        let a = self.next() % (2 * u64::from(EXPERTS));
                        let b = self.next() % (2 * u64::from(EXPERTS));
                        a.min(b) as u32
                    })
                    .collect()
            })
            .collect()
    }
}

fn params(vram: u32, ram: u32, ring: u32) -> TierParams {
    TierParams {
        vram_slots: vram,
        ram_slots: ram,
        min_layer_slots: 2,
        max_inserts: 3,
        spare_ring: ring,
        ..TierParams::default()
    }
}

/// `seed` = None: the one in-order stream (the control law).
fn setup<'x>(
    xt: &'x XtierFile,
    p: &TierParams,
    seed: Option<u64>,
    eager: bool,
) -> (ExpertTier, DeviceMover<'x, FakeQueue>) {
    let tier = ExpertTier::new(xt.header(), p.clone(), None, VRAM_BASE, RAM_BASE).unwrap();
    let mut q = FakeQueue::new(
        xt.header().stride,
        p.vram_slots,
        ExpertTier::slab_slots(p) - 1,
    );
    q.warm(xt, &tier.warm_plan()).unwrap();
    if let Some(s) = seed {
        q = q.signal_lanes(s);
    }
    q.eager = eager;
    (tier, DeviceMover::new(xt, q, 4))
}

/// The bytes the kernel reads for a routed expert NOW (through the published xptr) match?
fn read_ok(tier: &ExpertTier, q: &FakeQueue, xt: &XtierFile, layer: u32, e: u32) -> bool {
    let h = xt.header();
    let n = EXPERTS as usize;
    let (mut gu, mut dn) = (vec![0u64; 3 * (n + 1)], vec![0u64; 3 * (n + 1)]);
    tier.fill_xptr(layer, &mut gu, &mut dn);
    let loc = (e - h.first_expert) as usize;
    let a = gu[3 * loc] - h.sub[0].0;
    let i = u64::from(tier.index(layer, e).unwrap());
    let bytes = if a >= VRAM_BASE {
        q.vram_slot(((a - VRAM_BASE) / h.stride) as u32)
    } else {
        q.ram_slot(((a - RAM_BASE) / h.stride) as u32)
    };
    xt.verify(i, bytes)
}

/// One step: resolve, gate, the kernel's reads, fence, end_step. Returns (reads, wrong reads).
fn step(
    tier: &mut ExpertTier,
    mv: &mut DeviceMover<'_, FakeQueue>,
    xt: &XtierFile,
    routes: &[Vec<u32>],
    hold: bool,
) -> (u64, u64) {
    tier.begin_step();
    let (mut reads, mut bad) = (0, 0);
    for (l, rs) in routes.iter().enumerate() {
        let l = l as u32;
        tier.resolve(l, rs, mv).unwrap();
        mv.gate(l, &()).unwrap();
        for &e in rs {
            if tier.index(l, e).is_none() {
                continue;
            }
            reads += 1;
            if !read_ok(tier, mv.queue(), xt, l, e) {
                bad += 1;
            }
        }
    }
    mv.set_hold(hold);
    mv.after(&()).unwrap();
    tier.end_step(mv).unwrap();
    if hold {
        mv.release(&()).unwrap();
    }
    (reads, bad)
}

fn check_all_slots(tier: &ExpertTier, q: &FakeQueue, xt: &XtierFile) {
    tier.check_invariants().unwrap();
    for s in 0..tier.params().vram_slots {
        if let Some(r) = tier.vram_record(s) {
            assert!(xt.verify(u64::from(r), q.vram_slot(s)), "vram slot {s}");
        }
    }
    for s in 0..tier.ram_slab_slots() {
        if let Some(r) = tier.ram_record(s) {
            assert!(xt.verify(u64::from(r), q.ram_slot(s)), "ram slot {s}");
        }
    }
}

#[test]
fn every_read_lands_on_its_record_over_the_signal_lane() {
    let (_scratch, xt) = synth("lands");
    let mut runs = 0;
    for (vram, ram) in [(40, 24), (40, 12)] {
        for ring in [0, 1, 4] {
            for eager in [false, true] {
                for hold in [false, true] {
                    for seed in [1u64, 0xfeed_beef, 77] {
                        let p = params(vram, ram, ring);
                        let (mut tier, mut mv) = setup(&xt, &p, Some(seed), eager);
                        let mut g = Lcg(0xfa63 + seed + u64::from(ram));
                        let (mut reads, mut bad) = (0, 0);
                        for s in 0..30 {
                            let (r, b) = step(&mut tier, &mut mv, &xt, &g.routes(1 + s % 3), hold);
                            reads += r;
                            bad += b;
                        }
                        assert!(reads > 0 && tier.totals().1.demotions > 0);
                        assert_eq!(
                            bad, 0,
                            "vram {vram} ram {ram} ring {ring} eager {eager} hold {hold} seed {seed}"
                        );
                        mv.drain().unwrap();
                        assert_eq!(mv.queue().in_flight(), 0);
                        check_all_slots(&tier, mv.queue(), &xt);
                        runs += 1;
                    }
                }
            }
        }
    }
    assert_eq!(runs, 2 * 3 * 2 * 2 * 3);
}

#[test]
fn the_placements_are_the_one_streams() {
    let (_scratch, xt) = synth("same");
    let run = |seed: Option<u64>, ring: u32| {
        let p = params(40, 24, ring);
        let (mut tier, mut mv) = setup(&xt, &p, seed, false);
        let mut g = Lcg(31);
        for s in 0..60 {
            step(&mut tier, &mut mv, &xt, &g.routes(1 + s % 3), false);
        }
        let (paths, moves) = tier.totals();
        (paths, moves, mv.counts.h2d, mv.counts.d2h)
    };
    let one = run(None, 0);
    assert_eq!(run(Some(7), 0), one);
    assert_eq!(run(Some(7), 4), one, "the spare ring moves no route's path");
}

#[test]
fn one_spare_is_a_chain_and_the_ring_breaks_it() {
    let (_scratch, xt) = synth("chain");
    let overlaps = |ring: u32| {
        let p = params(40, 24, ring);
        let (mut tier, mut mv) = setup(&xt, &p, Some(99), false);
        let mut g = Lcg(5);
        for s in 0..60 {
            step(&mut tier, &mut mv, &xt, &g.routes(1 + s % 3), true);
        }
        mv.drain().unwrap();
        mv.queue().sig.as_ref().unwrap().overlaps
    };
    assert_eq!(
        overlaps(0),
        0,
        "FN8's one spare: every D2H waits the H2D before it"
    );
    assert!(
        overlaps(4) > 0,
        "the ring lets a demotion pass an earlier promotion"
    );
}

#[test]
fn control_without_the_slot_dependencies_a_read_is_wrong() {
    let (_scratch, xt) = synth("control");
    let mut bad = 0;
    for seed in [3u64, 11, 0xabc] {
        let p = params(40, 24, 4);
        let (mut tier, mut mv) = setup(&xt, &p, Some(seed), false);
        mv.queue_mut().sig.as_mut().unwrap().planner.no_slot_deps = true;
        let mut g = Lcg(17 + seed);
        for s in 0..60 {
            bad += step(&mut tier, &mut mv, &xt, &g.routes(1 + s % 3), true).1;
        }
        mv.drain().unwrap();
        for s in 0..tier.params().vram_slots {
            if let Some(r) = tier.vram_record(s) {
                bad += u64::from(!xt.verify(u64::from(r), mv.queue().vram_slot(s)));
            }
        }
    }
    assert!(bad > 0, "the control never caught a wrong byte");
}

/// FN17's deferred publication over the lane: the fence's publication is the ALL mark (law 3).
#[test]
fn the_deferred_law_publishes_through_the_all_mark() {
    for eager in [false, true] {
        let (_scratch, xt) = synth(if eager { "defer-e" } else { "defer-l" });
        let p = TierParams {
            vram_slots: 40,
            ram_slots: 24,
            min_layer_slots: 2,
            max_inserts: 2,
            read_through: 1.0,
            defer_spares: 2 * 2 * LAYERS,
            ..TierParams::default()
        };
        let tier0 = ExpertTier::new(xt.header(), p.clone(), None, VRAM_BASE, RAM_BASE).unwrap();
        let mut q = FakeQueue::new(
            xt.header().stride,
            p.vram_slots,
            p.ram_slots + p.defer_spares,
        );
        q.warm(&xt, &tier0.warm_plan()).unwrap();
        q = q.signal_lanes(0x63b);
        q.eager = eager;
        let mut tier = tier0;
        let mut mv = DeviceMover::new(&xt, q, 4).with_defer(true);
        let mut g = Lcg(0x5eed);
        let stride = tier.stride();
        for s in 0..60 {
            tier.begin_step();
            let routes = g.routes(1 + s % 3);
            for (l, rs) in routes.iter().enumerate() {
                mv.gate(l as u32, &()).unwrap();
                let n = EXPERTS as usize;
                let (mut gu, mut dn) = (vec![0u64; 3 * (n + 1)], vec![0u64; 3 * (n + 1)]);
                tier.fill_xptr(l as u32, &mut gu, &mut dn);
                let off0 = tier.sub_offset(0);
                for &e in rs {
                    let Some(i) = tier.index(l as u32, e) else {
                        continue;
                    };
                    let i = u64::from(i);
                    let a = gu[3 * e as usize] - off0;
                    let q = mv.queue();
                    let bytes = if a >= VRAM_BASE {
                        q.vram_slot(((a - VRAM_BASE) / stride) as u32)
                    } else {
                        q.ram_slot(((a - RAM_BASE) / stride) as u32)
                    };
                    assert!(xt.verify(i, bytes), "step {s} layer {l}: a stale read");
                }
            }
            mv.after(&()).unwrap();
            for (l, rs) in routes.iter().enumerate() {
                tier.resolve(l as u32, rs, &mut mv).unwrap();
            }
            tier.end_step(&mut mv).unwrap();
        }
        assert!(mv.counts.publish_waits >= 60);
        mv.drain().unwrap();
        tier.settle();
        check_all_slots(&tier, mv.queue(), &xt);
    }
}

/// The planner alone: a layer's mark waits for its segment's copies and its previous mark (the
/// word only grows); the ALL mark covers every open ticket; a copy waits on the last toucher of
/// each of its slots.
#[test]
fn the_planner_names_the_slot_and_mark_dependencies() {
    use saber_mem::device_mover::CopyOp::{D2h, H2d, Landed};
    let mut p = SignalPlanner::default();
    let never = |_t: u64| false;
    let mut out = Vec::new();
    p.plan(
        &[
            D2h { vram: 3, ram: 9 },
            H2d { ram: 5, vram: 3 },
            Landed { layer: 0 },
            D2h { vram: 7, ram: 5 },
            H2d { ram: 2, vram: 7 },
            Landed { layer: 1 },
        ],
        &never,
        &mut out,
    );
    let deps: Vec<Vec<u64>> = out.iter().map(|j| j.deps.clone()).collect();
    assert_eq!(
        out.iter().map(|j| j.ticket).collect::<Vec<_>>(),
        [1, 2, 3, 4, 5, 6]
    );
    assert_eq!(deps[0], Vec::<u64>::new());
    assert_eq!(
        deps[1],
        [1],
        "the promotion into slot 3 waits its victim's D2H"
    );
    assert_eq!(deps[2], [1, 2], "layer 0's mark waits its segment");
    assert_eq!(
        deps[3],
        [2],
        "the D2H into RAM 5 waits the H2D that read it"
    );
    assert_eq!(deps[4], [4]);
    assert_eq!(deps[5], [4, 5]);
    assert_eq!(
        out[2].what,
        SigWhat::Mark {
            word: MarkWord::Layer(0),
            value: 1
        }
    );
    out.clear();
    p.plan(
        &[H2d { ram: 9, vram: 11 }, Landed { layer: 0 }],
        &never,
        &mut out,
    );
    assert_eq!(out[0].deps, [1], "RAM 9 was written by ticket 1");
    assert_eq!(out[1].deps, [3, 7], "layer 0's second mark waits its first");
    assert_eq!(
        out[1].what,
        SigWhat::Mark {
            word: MarkWord::Layer(0),
            value: 2
        }
    );
    let all = p.all(&never).unwrap();
    assert_eq!(all.deps, [6, 8], "the ALL mark waits the open marks");
    // everything landed: nothing open, no ALL mark needed
    assert!(p.all(&|_| true).is_none());
}
