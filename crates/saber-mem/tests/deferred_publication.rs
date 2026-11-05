//! FN17: the DEFERRED PUBLICATION law of the expert tier over the host model of an asynchronous
//! copy stream ([`FakeQueue`]), at both extremes of the device: LAZY (a copy lands only when a
//! consumer waits behind it or the host drains) and EAGER (a copy lands the moment it is issued).
//! Under the law a step's copies go only to slots no placement points at (a promotion into a HOLE,
//! a demotion into a free RAM spare) and are published at the next `end_step`, after the
//! consumer's fence waited for them; so every read of every step lands on its record at BOTH
//! extremes, no layer gate ever waits, and every slot is its record after a drain. The controls:
//! the deferred copies without the fence's wait (a consumer skipping it) read stale bytes, and
//! FN8's law read with the placements of the step before its end (what a deferred publication
//! would do without holes) reads another record on the eager device.

use saber_mem::device_mover::{CopyQueue, DeviceMover, FakeQueue};
use saber_mem::dio::OpenMode;
use saber_mem::expert_tier::{ExpertTier, Placement, TierParams};
use saber_mem::xtier::{write_synthetic, SyntheticTier, XtierFile};
use saber_test_scratch::ScratchDir;

const LAYERS: u32 = 4;
const EXPERTS: u32 = 16;
const VRAM_BASE: u64 = 0x7f00_0000_0000;
const RAM_BASE: u64 = 0x5500_0000_0000;

fn synth(tag: &str) -> (ScratchDir, XtierFile) {
    let d = ScratchDir::new(&format!("saber-fn17-{tag}"));
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
                (0..tokens * 4)
                    .map(|_| {
                        let a = self.next() % u64::from(EXPERTS);
                        let b = self.next() % u64::from(EXPERTS);
                        a.min(b) as u32
                    })
                    .collect()
            })
            .collect()
    }
}

/// A COMPLETE tier (40 VRAM + 24 RAM = the 64 records), 2 inserts a layer, and the deferred law's
/// spares (2 x max_inserts x layers) when `defer`.
fn params(defer: bool) -> TierParams {
    TierParams {
        vram_slots: 40,
        ram_slots: 24,
        min_layer_slots: 2,
        max_inserts: 2,
        read_through: 1.0,
        defer_spares: if defer { 2 * 2 * LAYERS } else { 0 },
        ..TierParams::default()
    }
}

fn setup<'x>(
    xt: &'x XtierFile,
    p: &TierParams,
    eager: bool,
    arm: bool,
) -> (ExpertTier, DeviceMover<'x, FakeQueue>) {
    let tier = ExpertTier::new(xt.header(), p.clone(), None, VRAM_BASE, RAM_BASE).unwrap();
    let mut q = FakeQueue::new(
        xt.header().stride,
        p.vram_slots,
        p.ram_slots + p.defer_spares,
    );
    q.warm(xt, &tier.warm_plan()).unwrap();
    q.eager = eager;
    (tier, DeviceMover::new(xt, q, 4).with_defer(arm))
}

/// Every local expert's record base as the step's xptr tables publish it.
fn tables(tier: &ExpertTier, layer: u32) -> Vec<u64> {
    let n = EXPERTS as usize;
    let (mut gu, mut dn) = (vec![0u64; 3 * (n + 1)], vec![0u64; 3 * (n + 1)]);
    tier.fill_xptr(layer, &mut gu, &mut dn);
    let off0 = tier.sub_offset(0);
    (0..n).map(|e| gu[3 * e] - off0).collect()
}

/// The bytes at a record base NOW (VRAM slot or pinned RAM slot).
fn at(q: &FakeQueue, stride: u64, a: u64) -> &[u8] {
    if a >= VRAM_BASE {
        q.vram_slot(((a - VRAM_BASE) / stride) as u32)
    } else {
        q.ram_slot(((a - RAM_BASE) / stride) as u32)
    }
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

/// One step: the tables are filled BEFORE it (as the engine does: `tier_begin`), each layer's
/// gate, the kernel's reads of every routed expert (checked unless `skip_check`), the fence, the
/// routes resolved (the engine resolves from the route log after the step), `end_step`. Returns
/// (reads, stale reads).
fn step(
    tier: &mut ExpertTier,
    mv: &mut DeviceMover<'_, FakeQueue>,
    xt: &XtierFile,
    routes: &[Vec<u32>],
    skip_gate: bool,
) -> (u64, u64) {
    tier.begin_step();
    let stride = tier.stride();
    let tabs: Vec<Vec<u64>> = (0..LAYERS).map(|l| tables(tier, l)).collect();
    let (mut reads, mut stale) = (0, 0);
    for (l, rs) in routes.iter().enumerate() {
        if !skip_gate {
            mv.gate(l as u32, &()).unwrap();
        }
        for &e in rs {
            let i = u64::from(tier.index(l as u32, e).unwrap());
            reads += 1;
            if !xt.verify(i, at(mv.queue(), stride, tabs[l][e as usize])) {
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
    (reads, stale)
}

#[test]
fn deferred_reads_land_on_their_records_lazy_and_eager_and_no_gate_waits() {
    for eager in [false, true] {
        let (_scratch, xt) = synth(if eager { "eager" } else { "lazy" });
        let (mut tier, mut mv) = setup(&xt, &params(true), eager, true);
        let mut g = Lcg(0x5eed);
        let mut reads = 0;
        for s in 0..80 {
            let (r, stale) = step(&mut tier, &mut mv, &xt, &g.routes(1 + s % 3), false);
            assert_eq!(
                stale, 0,
                "step {s} (eager {eager}): a read landed on another record"
            );
            reads += r;
        }
        let (_, moves) = tier.totals();
        assert!(reads > 0 && moves.inserts > 0 && moves.demotions > 0 && moves.published > 0);
        assert_eq!(mv.counts.h2d, moves.inserts);
        assert_eq!(mv.counts.d2h, moves.demotions);
        assert_eq!(
            mv.counts.gate_waits, 0,
            "the deferred law never waits at a gate"
        );
        assert!(mv.counts.publish_waits >= 80);
        mv.drain().unwrap();
        tier.settle();
        check_all_slots(&tier, mv.queue(), &xt);
        assert_eq!(tier.pending_copies(), 0);
    }
}

#[test]
fn the_deferred_law_keeps_the_complete_tier_and_its_ram_accounting() {
    let (_scratch, xt) = synth("acct");
    let (mut tier, mut mv) = setup(&xt, &params(true), false, true);
    let mut g = Lcg(3);
    for s in 0..40 {
        step(&mut tier, &mut mv, &xt, &g.routes(1 + s % 2), false);
        assert!(tier.complete(), "a deferred step dropped a record");
    }
    // every record is placed exactly once (the invariants), and FN8's law is untouched when the
    // tier has no spares: the same routes publish at once
    let (mut t8, mut m8) = setup(&xt, &params(false), false, false);
    let mut g = Lcg(3);
    for s in 0..40 {
        let (_, stale) = step(&mut t8, &mut m8, &xt, &g.routes(1 + s % 2), false);
        assert_eq!(stale, 0);
    }
    assert_eq!(t8.totals().1.published, 0);
    assert!(m8.counts.gate_waits > 0, "FN8's law gates per layer");
}

#[test]
fn control_deferred_copies_without_the_fence_wait_read_stale_bytes() {
    // The tier defers (holes + spares) but the mover is NOT armed: its fence does not wait for the
    // copies, and a consumer that also skips the per-layer gate reads the promoted records' holes
    // before the copies land.
    let (_scratch, xt) = synth("ctl-nowait");
    let (mut tier, mut mv) = setup(&xt, &params(true), false, false);
    let mut g = Lcg(0x5eed);
    let mut stale = 0;
    for s in 0..80 {
        stale += step(&mut tier, &mut mv, &xt, &g.routes(1 + s % 3), true).1;
    }
    assert!(tier.totals().1.published > 0, "the control never published");
    assert!(
        stale > 0,
        "the control read no stale bytes: it proves nothing"
    );
}

#[test]
fn control_fn8_copies_read_with_the_last_steps_placements_hit_another_record_on_an_eager_device() {
    // What deferral WITHOUT holes would be: FN8's swap (the promotion written INTO the victim's
    // slot) while the next step still reads the placements of the step before -- on a device
    // whose copies land at once, the victim's readers get the promoted record.
    let (_scratch, xt) = synth("ctl-fn8");
    let (mut tier, mut mv) = setup(&xt, &params(false), true, false);
    let mut g = Lcg(0x5eed);
    let stride = tier.stride();
    let mut wrong = 0;
    for s in 0..80 {
        let routes = g.routes(1 + s % 3);
        let before: Vec<Vec<u64>> = (0..LAYERS).map(|l| tables(&tier, l)).collect();
        let idx: Vec<Vec<u32>> = (0..LAYERS)
            .map(|l| (0..EXPERTS).map(|e| tier.index(l, e).unwrap()).collect())
            .collect();
        step(&mut tier, &mut mv, &xt, &routes, false);
        // the old tables, read after the end_step's (eager) copies
        for l in 0..LAYERS as usize {
            for e in 0..EXPERTS as usize {
                if !xt.verify(u64::from(idx[l][e]), at(mv.queue(), stride, before[l][e])) {
                    wrong += 1;
                }
            }
        }
    }
    assert!(wrong > 0, "the control never caught an overwritten slot");
    // ... and the deferred law, read the same way, never does
    let (mut tier, mut mv) = setup(&xt, &params(true), true, true);
    let mut g = Lcg(0x5eed);
    for s in 0..80 {
        let routes = g.routes(1 + s % 3);
        let before: Vec<Vec<u64>> = (0..LAYERS).map(|l| tables(&tier, l)).collect();
        step(&mut tier, &mut mv, &xt, &routes, false);
        for l in 0..LAYERS {
            for e in 0..EXPERTS {
                let i = u64::from(tier.index(l, e).unwrap());
                let still = matches!(
                    tier.placement(l, e),
                    Some(Placement::Vram(_) | Placement::Ram(_))
                );
                assert!(still);
                assert!(
                    xt.verify(i, at(mv.queue(), stride, before[l as usize][e as usize]))
                        || tables(&tier, l)[e as usize] != before[l as usize][e as usize],
                    "step {s}: a slot the old placements point at was overwritten before its \
                     publication"
                );
            }
        }
    }
    let _ = CopyQueue::drain(mv.queue_mut());
}
