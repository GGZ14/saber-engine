//! FN42: THE LENDING LAW of the expert tier -- slots `vram_slots ..` are LENDABLE memory another
//! client owns (the KV pool's free tail); lent, they join a layer's victim set as INCLUSIVE slots
//! (the record keeps its shadow RAM slot), so a reclaim is a placement flip with no copy. Over the
//! host model of the copy stream ([`FakeQueue`], lazy and eager): every read of every step lands
//! on its record with slots lent, reclaimed mid-run and lent again; registered-but-unlent slots
//! change nothing (lend 0 == today, placement for placement); lending raises the hit rate; a
//! reclaim issues no copy; the deferred law refuses to lend.

use saber_mem::device_mover::{DeviceMover, FakeQueue};
use saber_mem::dio::OpenMode;
use saber_mem::expert_tier::{ExpertTier, Placement, TierParams};
use saber_mem::xtier::{write_synthetic, SyntheticTier, XtierFile};
use saber_test_scratch::ScratchDir;

const LAYERS: u32 = 4;
const EXPERTS: u32 = 16;
const BASE: u32 = 24;
const LENDABLE: u32 = 16;
const VRAM_BASE: u64 = 0x7f00_0000_0000;
const RAM_BASE: u64 = 0x5500_0000_0000;

fn synth(tag: &str) -> (ScratchDir, XtierFile) {
    let d = ScratchDir::new(&format!("saber-fn42-{tag}"));
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
                let mut v: Vec<u32> = (0..tokens * 4)
                    .map(|_| {
                        let a = self.next() % u64::from(EXPERTS);
                        let b = self.next() % u64::from(EXPERTS);
                        a.min(b) as u32
                    })
                    .collect();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect()
    }
}

/// A COMPLETE tier: 24 static VRAM slots + 40 RAM = the 64 records (FN8's law, LRFU 0.98).
fn params() -> TierParams {
    TierParams {
        vram_slots: BASE,
        ram_slots: 64 - BASE,
        min_layer_slots: 2,
        max_inserts: 2,
        read_through: 1.0,
        lrfu_lambda: 0.98,
        ..TierParams::default()
    }
}

/// The tier and a fake queue whose VRAM arena also holds the lendable slots (their "owner's"
/// memory in the engine); `register` registers them (not lent).
fn setup<'x>(
    xt: &'x XtierFile,
    p: &TierParams,
    eager: bool,
    register: bool,
) -> (ExpertTier, DeviceMover<'x, FakeQueue>) {
    let mut tier = ExpertTier::new(xt.header(), p.clone(), None, VRAM_BASE, RAM_BASE).unwrap();
    if register {
        let s = xt.header().stride;
        let addrs: Vec<u64> = (0..u64::from(LENDABLE))
            .map(|i| VRAM_BASE + (u64::from(BASE) + i) * s)
            .collect();
        assert_eq!(tier.register_lendable(&addrs).unwrap(), BASE);
    }
    let mut q = FakeQueue::new(xt.header().stride, BASE + LENDABLE, p.ram_slots);
    q.warm(xt, &tier.warm_plan()).unwrap();
    q.eager = eager;
    (tier, DeviceMover::new(xt, q, 4))
}

fn tables(tier: &ExpertTier, layer: u32) -> Vec<u64> {
    let n = EXPERTS as usize;
    let (mut gu, mut dn) = (vec![0u64; 3 * (n + 1)], vec![0u64; 3 * (n + 1)]);
    tier.fill_xptr(layer, &mut gu, &mut dn);
    let off0 = tier.sub_offset(0);
    (0..n).map(|e| gu[3 * e] - off0).collect()
}

fn at(q: &FakeQueue, stride: u64, a: u64) -> &[u8] {
    if a >= VRAM_BASE {
        q.vram_slot(((a - VRAM_BASE) / stride) as u32)
    } else {
        q.ram_slot(((a - RAM_BASE) / stride) as u32)
    }
}

/// Every placed slot (static, lent, RAM incl. the lent slots' shadows) is its record.
fn check_all_slots(tier: &ExpertTier, q: &FakeQueue, xt: &XtierFile) {
    tier.check_invariants().unwrap();
    for s in 0..BASE + LENDABLE {
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

/// One step as the engine runs it (tables before, gates, reads checked, fence, resolve, end).
/// Returns (reads, stale reads, resident hits).
fn step(
    tier: &mut ExpertTier,
    mv: &mut DeviceMover<'_, FakeQueue>,
    xt: &XtierFile,
    routes: &[Vec<u32>],
) -> (u64, u64, u64) {
    tier.begin_step();
    let stride = tier.stride();
    let tabs: Vec<Vec<u64>> = (0..LAYERS).map(|l| tables(tier, l)).collect();
    let (mut reads, mut stale) = (0, 0);
    for (l, rs) in routes.iter().enumerate() {
        mv.gate(l as u32, &()).unwrap();
        for &e in rs {
            let i = u64::from(tier.index(l as u32, e).unwrap());
            reads += 1;
            if !xt.verify(i, at(mv.queue(), stride, tabs[l][e as usize])) {
                stale += 1;
            }
        }
    }
    mv.after(&()).unwrap();
    let mut hits = 0;
    for (l, rs) in routes.iter().enumerate() {
        hits += tier.resolve(l as u32, rs, mv).unwrap().resident;
    }
    tier.end_step(mv).unwrap();
    tier.check_invariants().unwrap();
    (reads, stale, hits)
}

#[test]
fn lent_slots_serve_every_read_through_lend_reclaim_and_relend() {
    for eager in [false, true] {
        let (_scratch, xt) = synth(if eager { "eager" } else { "lazy" });
        let (mut tier, mut mv) = setup(&xt, &params(), eager, true);
        let ids: Vec<u32> = (BASE..BASE + LENDABLE).collect();
        assert_eq!(tier.lend(&ids).unwrap(), LENDABLE as usize);
        assert_eq!(tier.lent(), (LENDABLE, LENDABLE));
        let mut g = Lcg(0x42);
        let mut reads = 0;
        for s in 0..120u32 {
            if s == 40 || s == 90 {
                // the owner needs the memory back: drain, then reclaim half (the high ids)
                mv.drain().unwrap();
                check_all_slots(&tier, mv.queue(), &xt);
                let d2h = mv.counts.d2h;
                let h2d = mv.counts.h2d;
                let dropped = tier.reclaim(&ids[8..]);
                assert_eq!(
                    (mv.counts.d2h, mv.counts.h2d),
                    (d2h, h2d),
                    "a reclaim copies nothing"
                );
                assert!(dropped <= 8);
                assert_eq!(tier.lent().0, 8);
                for &sl in &ids[8..] {
                    assert_eq!(tier.vram_record(sl), None);
                }
                check_all_slots(&tier, mv.queue(), &xt);
            }
            if s == 60 {
                assert_eq!(
                    tier.lend(&ids).unwrap(),
                    8,
                    "only the reclaimed slots are lent again"
                );
            }
            let (r, stale, _) = step(&mut tier, &mut mv, &xt, &g.routes(1 + (s % 3) as usize));
            assert_eq!(
                stale, 0,
                "step {s} (eager {eager}): a read landed on another record"
            );
            reads += r;
        }
        let (_, moves) = tier.totals();
        assert!(reads > 0 && moves.inserts > 0 && moves.lent_drops > 0);
        assert_eq!(mv.counts.h2d, moves.inserts);
        assert_eq!(
            mv.counts.d2h, moves.demotions,
            "a lent victim is never copied out"
        );
        mv.drain().unwrap();
        check_all_slots(&tier, mv.queue(), &xt);
        // everything back: the tier is today's static partition, complete
        tier.reclaim(&ids);
        assert_eq!(tier.lent().0, 0);
        assert!(tier.complete());
        check_all_slots(&tier, mv.queue(), &xt);
        assert!((0..64u32).all(|i| !matches!(
            tier.placement(i / EXPERTS, i % EXPERTS),
            Some(Placement::Vram(s)) if s >= BASE
        )));
    }
}

#[test]
fn registered_but_unlent_slots_are_todays_tier_placement_for_placement() {
    let (_scratch, xt) = synth("zero");
    let (mut a, mut ma) = setup(&xt, &params(), false, false);
    let (mut b, mut mb) = setup(&xt, &params(), false, true);
    let (mut ga, mut gb) = (Lcg(7), Lcg(7));
    for _ in 0..100 {
        step(&mut a, &mut ma, &xt, &ga.routes(2));
        step(&mut b, &mut mb, &xt, &gb.routes(2));
        for i in 0..64u32 {
            assert_eq!(
                a.placement(i / EXPERTS, i % EXPERTS),
                b.placement(i / EXPERTS, i % EXPERTS)
            );
        }
    }
    assert_eq!(a.totals(), b.totals());
    assert_eq!(
        (ma.counts.h2d, ma.counts.d2h),
        (mb.counts.h2d, mb.counts.d2h)
    );
}

#[test]
fn lending_raises_the_hit_rate_and_the_replay_arm_lends_at_boot() {
    let (_scratch, xt) = synth("hits");
    let run = |lend: bool| {
        let (mut t, mut m) = setup(&xt, &params(), false, true);
        if lend {
            let ids: Vec<u32> = (BASE..BASE + LENDABLE).collect();
            t.lend(&ids).unwrap();
        }
        let mut g = Lcg(99);
        let mut hits = 0;
        for _ in 0..200 {
            hits += step(&mut t, &mut m, &xt, &g.routes(1)).2;
        }
        hits
    };
    let (h0, h1) = (run(false), run(true));
    assert!(h1 > h0, "lent slots add hits: {h0} -> {h1}");
    // the replay's arm: `lend_boot` lends that many synthetic slots at construction
    let p = TierParams {
        lend_boot: LENDABLE,
        ..params()
    };
    let t = ExpertTier::new(xt.header(), p, None, VRAM_BASE, RAM_BASE).unwrap();
    assert_eq!(t.lent(), (LENDABLE, LENDABLE));
    t.check_invariants().unwrap();
}

#[test]
fn the_deferred_law_refuses_to_lend() {
    let (_scratch, xt) = synth("defer");
    let p = TierParams {
        defer_spares: 2 * 2 * LAYERS,
        ..params()
    };
    let mut t = ExpertTier::new(xt.header(), p, None, VRAM_BASE, RAM_BASE).unwrap();
    assert!(t.register_lendable(&[VRAM_BASE]).is_err());
    let mut t2 = ExpertTier::new(xt.header(), params(), None, VRAM_BASE, RAM_BASE).unwrap();
    assert!(t2.lend(&[BASE]).is_err(), "an unregistered slot is refused");
}
