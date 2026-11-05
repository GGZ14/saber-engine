//! FN54b: the TWO-QUEUE mover over the host model of two asynchronous copy lanes
//! ([`FakeQueue::two_queue`]: the promotions H2D + the layer marks on the urgent lane, the
//! demotions D2H on the background lane, each lane advanced in a seeded pseudo-random order the
//! slot law allows). Every routed read after its gate lands on its record, every slot is its
//! record after a drain, LAZY and EAGER, with and without the hold law; the placements are the
//! one stream's (same paths, same moves); with FN8's one spare the slot law is a CHAIN (no
//! demotion ever overlaps an earlier promotion) and the spare ring breaks it; the control: the
//! planner without its cross-lane waits lets a read see the wrong bytes.

use saber_mem::device_mover::{DeviceMover, FakeQueue};
use saber_mem::dio::OpenMode;
use saber_mem::expert_tier::{ExpertTier, TierParams};
use saber_mem::xtier::{write_synthetic, SyntheticTier, XtierFile};
use saber_test_scratch::ScratchDir;

const LAYERS: u32 = 4;
const EXPERTS: u32 = 16;
const VRAM_BASE: u64 = 0x7f00_0000_0000;
const RAM_BASE: u64 = 0x5500_0000_0000;

fn synth(tag: &str) -> (ScratchDir, XtierFile) {
    let d = ScratchDir::new(&format!("saber-fn54b-{tag}"));
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
        q = q.two_queue(s);
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
fn every_read_lands_on_its_record_over_two_lanes() {
    let (_scratch, xt) = synth("lands");
    let mut runs = 0;
    for (vram, ram) in [(40, 24), (40, 12)] {
        for ring in [0, 1, 4] {
            for eager in [false, true] {
                for hold in [false, true] {
                    for seed in [1u64, 0xfeed_beef] {
                        let p = params(vram, ram, ring);
                        let (mut tier, mut mv) = setup(&xt, &p, Some(seed), eager);
                        let mut g = Lcg(0xfa54 + seed + u64::from(ram));
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
    assert_eq!(runs, 2 * 3 * 2 * 2 * 2);
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
        mv.queue().lanes.as_ref().unwrap().overlaps
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
fn control_without_the_cross_lane_waits_a_read_is_wrong() {
    let (_scratch, xt) = synth("control");
    let mut bad = 0;
    for seed in [3u64, 11, 0xabc] {
        let p = params(40, 24, 4);
        let (mut tier, mut mv) = setup(&xt, &p, Some(seed), false);
        mv.queue_mut().lanes.as_mut().unwrap().planner.no_waits = true;
        let mut g = Lcg(17 + seed);
        for s in 0..60 {
            bad += step(&mut tier, &mut mv, &xt, &g.routes(1 + s % 3), true).1;
        }
        mv.drain().unwrap();
        // a slot whose bytes are wrong after the drain counts too
        for s in 0..tier.params().vram_slots {
            if let Some(r) = tier.vram_record(s) {
                bad += u64::from(!xt.verify(u64::from(r), mv.queue().vram_slot(s)));
            }
        }
    }
    assert!(bad > 0, "the control never caught a wrong byte");
}

/// FN54c: the batch-mark law -- groups of copies between marks: every read still lands, every
/// slot is its record after a drain at every group size, the default group plans fewer marks +
/// waits than one copy a group (FN54b's per-op form), and it still overlaps the two lanes.
#[test]
fn batch_marks_cut_the_calls_and_keep_the_bytes() {
    let (_scratch, xt) = synth("batch");
    let run = |group: usize, seed: u64| {
        let p = params(40, 24, 4);
        let (mut tier, mut mv) = setup(&xt, &p, Some(seed), false);
        mv.queue_mut().lanes.as_mut().unwrap().planner.group = group;
        let mut g = Lcg(23 + seed);
        let mut bad = 0;
        for s in 0..60 {
            bad += step(&mut tier, &mut mv, &xt, &g.routes(1 + s % 3), true).1;
        }
        mv.drain().unwrap();
        check_all_slots(&tier, mv.queue(), &xt);
        let fl = mv.queue().lanes.as_ref().unwrap();
        (
            bad,
            fl.planner.marks + fl.planner.waits,
            fl.overlaps,
            mv.counts.h2d + mv.counts.d2h,
        )
    };
    for seed in [5u64, 41, 0x77] {
        let (b1, calls1, _, copies) = run(1, seed);
        let (b2, calls2, ov2, copies2) = run(0, seed);
        let (b4, _, _, _) = run(4, seed);
        eprintln!("seed {seed}: copies {copies}, marks + waits group 1 {calls1}, group 2 {calls2}");
        assert_eq!((b1, b2, b4), (0, 0, 0), "a read saw the wrong bytes");
        assert_eq!(copies, copies2, "the group moves no copy");
        assert!(copies > 0);
        assert!(
            calls2 < calls1,
            "group 2 plans {calls2} marks + waits, group 1 {calls1}"
        );
        assert!(ov2 > 0, "the default group still overlaps the lanes");
    }
}
