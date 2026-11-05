//! FN42b: THE ARENA BORROW (client B of FN42's lending law) -- the wide prefill arena's tail is
//! the tier's LENDABLE slots: lent while no wide chunk runs (decode keeps them), RECLAIMED before
//! a wide chunk (a placement flip to the RAM shadows: the copy stream drained, no copy), the
//! arena then OWNS the bytes and scribbles over them, lent again after the chunk (EMPTY: the
//! insert law refills). Over the host model of the copy stream ([`FakeQueue`], lazy and eager):
//! no read of any step lands on another record through five borrow cycles with the arena's
//! garbage in the tail; a lend -> reclaim with no step between leaves the tier placement for
//! placement a tier that never lent; the ledger's `fn.arena.prefill` + `xtier.kv_lent` balance in
//! every state; the deferred and demotion-free laws refuse the borrow.

use saber_mem::device_mover::{DeviceMover, FakeQueue};
use saber_mem::dio::OpenMode;
use saber_mem::expert_tier::{Demote, ExpertTier, TierParams};
use saber_mem::flashnext_budget::{geometry, Class, Ep2Budget, KV_LENT};
use saber_mem::ledger::{Ledger, Pool};
use saber_mem::xtier::{write_synthetic, SyntheticTier, XtierFile};
use saber_test_scratch::ScratchDir;

const LAYERS: u32 = 4;
const EXPERTS: u32 = 16;
const BASE: u32 = 20;
const LENDABLE: u32 = 12;
const VRAM_BASE: u64 = 0x7f00_0000_0000;
const RAM_BASE: u64 = 0x5500_0000_0000;

fn synth(tag: &str) -> (ScratchDir, XtierFile) {
    let d = ScratchDir::new(&format!("saber-fn42b-{tag}"));
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
    /// A decode step's routes (skewed to the low experts).
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

/// A wide chunk routes every expert of every layer (FN38: ~every RAM record at c8192).
fn wide_routes() -> Vec<Vec<u32>> {
    (0..LAYERS).map(|_| (0..EXPERTS).collect()).collect()
}

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

/// The tier and a fake queue whose VRAM holds the static slots and then the arena's tail (the
/// lendable slots' memory); `register` registers the tail (not lent).
fn setup<'x>(
    xt: &'x XtierFile,
    eager: bool,
    register: bool,
) -> (ExpertTier, DeviceMover<'x, FakeQueue>, Vec<u32>) {
    let p = params();
    let mut tier = ExpertTier::new(xt.header(), p.clone(), None, VRAM_BASE, RAM_BASE).unwrap();
    let ids: Vec<u32> = if register {
        let s = xt.header().stride;
        let addrs: Vec<u64> = (0..u64::from(LENDABLE))
            .map(|i| VRAM_BASE + (u64::from(BASE) + i) * s)
            .collect();
        let first = tier.register_lendable(&addrs).unwrap();
        (first..first + LENDABLE).collect()
    } else {
        Vec::new()
    };
    let mut q = FakeQueue::new(xt.header().stride, BASE + LENDABLE, p.ram_slots);
    q.warm(xt, &tier.warm_plan()).unwrap();
    q.eager = eager;
    (tier, DeviceMover::new(xt, q, 4), ids)
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

/// One step as the engine runs it; returns (reads, stale reads, reads of a lendable slot).
fn step(
    tier: &mut ExpertTier,
    mv: &mut DeviceMover<'_, FakeQueue>,
    xt: &XtierFile,
    routes: &[Vec<u32>],
) -> (u64, u64, u64) {
    tier.begin_step();
    let stride = tier.stride();
    let tabs: Vec<Vec<u64>> = (0..LAYERS).map(|l| tables(tier, l)).collect();
    let tail = VRAM_BASE + u64::from(BASE) * stride;
    let (mut reads, mut stale, mut in_tail) = (0, 0, 0);
    for (l, rs) in routes.iter().enumerate() {
        mv.gate(l as u32, &()).unwrap();
        for &e in rs {
            let i = u64::from(tier.index(l as u32, e).unwrap());
            let a = tabs[l][e as usize];
            reads += 1;
            if (tail..tail + u64::from(LENDABLE) * stride).contains(&a) {
                in_tail += 1;
            }
            if !xt.verify(i, at(mv.queue(), stride, a)) {
                stale += 1;
            }
        }
    }
    mv.after(&()).unwrap();
    for (l, rs) in routes.iter().enumerate() {
        tier.resolve(l as u32, rs, mv).unwrap();
    }
    tier.end_step(mv).unwrap();
    tier.check_invariants().unwrap();
    (reads, stale, in_tail)
}

/// Every placed slot is its record.
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

#[test]
fn five_borrow_cycles_read_no_stale_byte_with_the_arenas_garbage_in_the_tail() {
    for eager in [false, true] {
        let (_scratch, xt) = synth(if eager { "eager" } else { "lazy" });
        let (mut tier, mut mv, ids) = setup(&xt, eager, true);
        // boot: the tail is LENT (decode keeps the slots)
        assert_eq!(tier.lend(&ids).unwrap(), LENDABLE as usize);
        let mut g = Lcg(0x42b);
        let (mut tail_reads, mut dropped) = (0, 0);
        for cycle in 0..5u8 {
            for s in 0..30 {
                let (_, stale, t) = step(&mut tier, &mut mv, &xt, &g.routes(1 + s % 3));
                assert_eq!(stale, 0, "cycle {cycle} decode step {s} (eager {eager})");
                tail_reads += t;
            }
            // a wide chunk is admitted: drain, reclaim (a flip, no copy), the arena owns the tail
            mv.drain().unwrap();
            let (h2d, d2h) = (mv.counts.h2d, mv.counts.d2h);
            dropped += tier.reclaim(&ids);
            assert_eq!(
                (mv.counts.h2d, mv.counts.d2h),
                (h2d, d2h),
                "a reclaim copies nothing"
            );
            assert_eq!(tier.lent().0, 0);
            check_all_slots(&tier, mv.queue(), &xt);
            for &s in &ids {
                mv.queue_mut().scribble(s, 0xA5 ^ cycle);
            }
            // the wide chunk's steps: every record routed, none read from the tail
            for _ in 0..2 {
                let (r, stale, t) = step(&mut tier, &mut mv, &xt, &wide_routes());
                assert_eq!(r, u64::from(LAYERS * EXPERTS));
                assert_eq!(
                    (stale, t),
                    (0, 0),
                    "cycle {cycle}: the wide chunk read the tail"
                );
            }
            // after the chunk: lent again, EMPTY -- the insert law refills them
            assert_eq!(tier.lend(&ids).unwrap(), LENDABLE as usize);
            for &s in &ids {
                assert_eq!(tier.vram_record(s), None, "a re-lent slot starts empty");
            }
        }
        assert!(tail_reads > 0, "the lent slots served reads");
        assert!(dropped > 0, "the reclaims flipped records");
        let (_, moves) = tier.totals();
        assert!(
            moves.lent_drops >= dropped as u64,
            "lent victims + the reclaims' flips"
        );
        assert_eq!(
            mv.counts.d2h, moves.demotions,
            "a lent victim is never copied out"
        );
        mv.drain().unwrap();
        check_all_slots(&tier, mv.queue(), &xt);
        tier.reclaim(&ids);
        assert!(tier.complete());
        check_all_slots(&tier, mv.queue(), &xt);
    }
}

#[test]
fn a_borrow_with_no_step_between_is_placement_for_placement_a_tier_that_never_lent() {
    let (_scratch, xt) = synth("nostep");
    let (mut a, mut ma, ids) = setup(&xt, false, true);
    let (mut b, mut mb, _) = setup(&xt, false, false);
    let (mut ga, mut gb) = (Lcg(7), Lcg(7));
    for s in 0..60 {
        if s % 20 == 5 {
            // lend then reclaim before any step: nothing moved
            ma.drain().unwrap();
            a.lend(&ids).unwrap();
            assert_eq!(a.reclaim(&ids), 0);
        }
        step(&mut a, &mut ma, &xt, &ga.routes(2));
        step(&mut b, &mut mb, &xt, &gb.routes(2));
        for i in 0..64u32 {
            assert_eq!(
                a.placement(i / EXPERTS, i % EXPERTS),
                b.placement(i / EXPERTS, i % EXPERTS),
                "step {s} record {i}"
            );
        }
    }
    assert_eq!(a.totals(), b.totals());
}

#[test]
fn the_borrows_ledger_balances_in_every_state() {
    let arena: u64 = 2_963 << 20;
    let tail = geometry::EXPERT_STRIDE * 846;
    let w = vec![Class {
        name: "weights".into(),
        bytes: 9 << 30,
    }];
    let rt = vec![
        Class {
            name: "fn.pools".into(),
            bytes: 1_583 << 20,
        },
        Class {
            name: "fn.arena.prefill".into(),
            bytes: arena,
        },
    ];
    let b = Ep2Budget::plan(
        0,
        30_500 << 20,
        w,
        rt,
        geometry::EXPERT_STRIDE,
        geometry::EXPERT_RECORDS,
    )
    .unwrap();
    let mut led = Ledger::new();
    b.declare(&mut led, 0).unwrap();
    let pool = Pool::Vram(0);
    let (used0, free0) = (led.used(pool), led.free(pool));
    // boot lent, a wide chunk reclaims, lent again, ... : the pool never moves
    for lent in [tail, 0, tail, 0, tail] {
        b.relend(&mut led, "fn.arena.prefill", arena, lent).unwrap();
        assert_eq!((led.used(pool), led.free(pool)), (used0, free0));
        let snap = format!("{:?}", led.snapshot());
        assert_eq!(snap.contains(KV_LENT), lent > 0, "{snap}");
    }
    assert!(
        b.relend(&mut led, "fn.arena.prefill", arena, arena + 1)
            .is_err(),
        "a lend past the arena refuses"
    );
}

#[test]
fn the_deferred_and_demotion_free_laws_refuse_the_borrow() {
    let (_scratch, xt) = synth("refuse");
    let defer = TierParams {
        defer_spares: 2 * 2 * LAYERS,
        ..params()
    };
    let mut t = ExpertTier::new(xt.header(), defer, None, VRAM_BASE, RAM_BASE).unwrap();
    assert!(t.register_lendable(&[VRAM_BASE]).is_err(), "--xtier-defer");
    let none = TierParams {
        demote: Demote::None,
        refill_spares: 2 * LAYERS,
        ..params()
    };
    if let Ok(mut t) = ExpertTier::new(xt.header(), none, None, VRAM_BASE, RAM_BASE) {
        assert!(
            t.register_lendable(&[VRAM_BASE]).is_err(),
            "--xtier-demote none"
        );
    }
}

/// FN44b: a lend hands the slots over EMPTY and a prefill chunk's layers are read-through, so
/// nothing filled them before a decode miss did (the harness's 2,048-row prompt ran with the
/// whole tail empty). `fill_lent` fills every empty lent slot with its layer's next profile /
/// LRFU record (one H2D each, behind the step's fence; the record keeps its RAM shadow): the
/// next wide-routed chunk reads the tail with no stale byte, through three cycles with the
/// arena's garbage in it, and a reclaim after a fill is still a flip (no copy).
#[test]
fn fn44b_the_fill_on_lend_reads_no_stale_byte_and_serves_the_next_chunk() {
    for eager in [false, true] {
        let (_scratch, xt) = synth(if eager { "fill-eager" } else { "fill-lazy" });
        let (mut tier, mut mv, ids) = setup(&xt, eager, true);
        assert_eq!(tier.lend(&ids).unwrap(), LENDABLE as usize);
        assert_eq!(tier.lent_empty(), LENDABLE);
        for cycle in 0..3u8 {
            // the fill: behind the fence (the engine's `after` at the pass's tier_begin)
            mv.after(&()).unwrap();
            let h2d = mv.counts.h2d;
            let n = tier.fill_lent(&mut mv).unwrap();
            assert_eq!(
                n, LENDABLE as usize,
                "cycle {cycle}: every lent slot filled"
            );
            assert_eq!(mv.counts.h2d - h2d, u64::from(LENDABLE), "one H2D a slot");
            assert_eq!(tier.lent_empty(), 0);
            tier.check_invariants().unwrap();
            // the next chunk: every record routed (read-through), the filled tail SERVES reads
            let (r, stale, t) = step(&mut tier, &mut mv, &xt, &wide_routes());
            assert_eq!(r, u64::from(LAYERS * EXPERTS));
            assert_eq!(stale, 0, "cycle {cycle}: a filled slot read stale bytes");
            assert_eq!(
                t,
                u64::from(LENDABLE),
                "cycle {cycle}: every filled slot served"
            );
            assert_eq!(tier.fill_lent(&mut mv).unwrap(), 0, "nothing left to fill");
            // a wide chunk: drain, reclaim (a flip, no copy), garbage in the tail, lend again
            mv.drain().unwrap();
            let (h2d, d2h) = (mv.counts.h2d, mv.counts.d2h);
            assert_eq!(tier.reclaim(&ids), LENDABLE as usize, "the fills flip back");
            assert_eq!(
                (mv.counts.h2d, mv.counts.d2h),
                (h2d, d2h),
                "a reclaim copies nothing"
            );
            check_all_slots(&tier, mv.queue(), &xt);
            for &s in &ids {
                mv.queue_mut().scribble(s, 0x5A ^ cycle);
            }
            assert_eq!(tier.lend(&ids).unwrap(), LENDABLE as usize);
        }
        mv.after(&()).unwrap();
        tier.fill_lent(&mut mv).unwrap();
        mv.drain().unwrap();
        check_all_slots(&tier, mv.queue(), &xt);
        let (_, moves) = tier.totals();
        assert_eq!(moves.lent_fills, 4 * u64::from(LENDABLE));
    }
}
