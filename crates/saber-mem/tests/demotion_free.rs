//! FN36: the DEMOTION-FREE law over the host model of an asynchronous copy stream
//! ([`FakeQueue`]), lazy and eager: a VRAM victim is re-read from NVMe into a refill pool slot
//! under the next step (no drain, no D2H copy), the swap is published at the next `end_step`
//! (the promotion's H2D, gated per layer as FN8's), every read of every step lands on its record,
//! the tier stays complete, and after a drain every slot (and every refilled pool slot) is its
//! record. The control: a consumer that skips the gates is refused (law 4b).

use saber_mem::device_mover::{DeviceMover, FakeQueue};
use saber_mem::dio::OpenMode;
use saber_mem::expert_tier::{Demote, ExpertTier, TierParams};
use saber_mem::xtier::{write_synthetic, SyntheticTier, XtierFile};
use saber_test_scratch::ScratchDir;

const LAYERS: u32 = 4;
const EXPERTS: u32 = 16;
const VRAM_BASE: u64 = 0x7f00_0000_0000;
const RAM_BASE: u64 = 0x5500_0000_0000;

fn synth(tag: &str) -> (ScratchDir, XtierFile) {
    let d = ScratchDir::new(&format!("saber-fn36-{tag}"));
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

/// A COMPLETE tier (40 VRAM + 24 RAM = the 64 records), 2 inserts a layer; `none` = FN36's law.
fn params(none: bool) -> TierParams {
    TierParams {
        vram_slots: 40,
        ram_slots: 24,
        min_layer_slots: 2,
        max_inserts: 2,
        read_through: 1.0,
        lrfu_lambda: 0.98,
        demote: if none { Demote::None } else { Demote::Ram },
        refill_spares: if none { 2 * 2 * LAYERS } else { 0 },
        ..TierParams::default()
    }
}

fn setup(xt: &XtierFile, none: bool, eager: bool) -> (ExpertTier, DeviceMover<'_, FakeQueue>) {
    let p = params(none);
    let tier = ExpertTier::new(xt.header(), p.clone(), None, VRAM_BASE, RAM_BASE).unwrap();
    let mut q = FakeQueue::new(
        xt.header().stride,
        p.vram_slots,
        p.ram_slots + p.refill_spares,
    );
    q.warm(xt, &tier.warm_plan()).unwrap();
    q.eager = eager;
    (tier, DeviceMover::new(xt, q, 4))
}

/// One step as the engine runs it: gates + reads (stale ones counted; `skip_gate` = the control),
/// the refills of the last end_step UNDER the step (`refill`; else end_step reads them late), the
/// fence, the routes, `end_step`.
fn step(
    tier: &mut ExpertTier,
    mv: &mut DeviceMover<'_, FakeQueue>,
    xt: &XtierFile,
    routes: &[Vec<u32>],
    refill: bool,
    skip_gate: bool,
) -> u64 {
    tier.begin_step();
    let stride = tier.stride();
    let tabs: Vec<Vec<u64>> = (0..LAYERS).map(|l| tables(tier, l)).collect();
    let mut stale = 0;
    for (l, rs) in routes.iter().enumerate() {
        if !skip_gate {
            mv.gate(l as u32, &()).unwrap();
        }
        for &e in rs {
            let i = u64::from(tier.index(l as u32, e).unwrap());
            if !xt.verify(i, at(mv.queue(), stride, tabs[l][e as usize])) {
                stale += 1;
            }
        }
    }
    if refill {
        let before = mv.queue().log.len();
        tier.refill(mv).unwrap();
        assert_eq!(
            mv.queue().log.len(),
            before,
            "a refill drained or issued a copy"
        );
    }
    mv.after(&()).unwrap();
    for (l, rs) in routes.iter().enumerate() {
        tier.resolve(l as u32, rs, mv).unwrap();
    }
    tier.end_step(mv).unwrap();
    tier.check_invariants().unwrap();
    assert!(
        tier.complete(),
        "the demotion-free law keeps the tier complete"
    );
    stale
}

/// 120 steps; [h2d, d2h, nvme reads, refills, refill_late, stale reads, resident hits].
fn run(none: bool, eager: bool, refill: bool, skip_gate: bool) -> [u64; 7] {
    let (_scratch, xt) = synth(&format!("n{none}-e{eager}-r{refill}-s{skip_gate}"));
    let (mut tier, mut mv) = setup(&xt, none, eager);
    let mut g = Lcg(0x5eed);
    let mut stale = 0;
    for s in 0..120u64 {
        let r = g.routes(1 + (s % 3) as usize);
        stale += step(&mut tier, &mut mv, &xt, &r, refill, skip_gate);
    }
    tier.refill(&mut mv).unwrap();
    mv.drain().unwrap();
    tier.settle();
    check_all_slots(&tier, mv.queue(), &xt);
    for (s, r) in tier.refill_targets() {
        assert!(
            xt.verify(u64::from(r), mv.queue().ram_slot(s)),
            "refill slot {s}"
        );
    }
    let (p, m) = tier.totals();
    let c = mv.counts;
    [
        c.h2d,
        c.d2h,
        c.nvme_reads,
        m.refills,
        m.refill_late,
        stale,
        p.resident,
    ]
}

#[test]
fn demotion_free_moves_no_victim_over_the_link_and_every_read_lands_lazy_and_eager() {
    for eager in [false, true] {
        let ram = run(false, eager, true, false);
        assert_eq!(ram[5], 0, "FN8's law: every read lands");
        assert!(ram[0] > 0 && ram[1] > 0 && ram[2] == 0);
        let none = run(true, eager, true, false);
        assert_eq!(none[5], 0, "eager {eager}: a read missed its record");
        assert_eq!(none[1], 0, "the demotion-free law copied a victim D2H");
        assert!(none[0] > 0 && none[3] > 0, "promotions and refills ran");
        assert_eq!(none[2], none[3], "every refill is one NVMe read");
        assert_eq!(none[4], 0, "every refill was read under a step");
        // the late path (no refill under the step) is the same law, read between steps
        let late = run(true, eager, false, false);
        assert_eq!(late[5], 0);
        assert_eq!(
            (late[0], late[1], late[3], late[6]),
            (none[0], none[1], none[3], none[6])
        );
        assert!(late[4] > 0, "end_step read the refills itself");
    }
}

#[test]
#[should_panic(expected = "law 4b")]
fn demotion_free_control_skipping_the_gates_is_refused() {
    // the lazy device: a published H2D lands only when a gate waits for it; a consumer that skips
    // the gates leaves the promoted record's old RAM slot in flight when the pool hands it to a
    // refill -- the fake queue refuses the host write (law 4b), before any stale byte is read
    run(true, false, true, true);
}
