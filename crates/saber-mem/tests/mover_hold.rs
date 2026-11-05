//! FN30c: the tier mover's HOLD law over the host model of an asynchronous copy stream
//! ([`FakeQueue`]), LAZY and EAGER. A step ending under `set_hold(true)` queues its copies on the
//! host: nothing is issued until `release` (the drafter round's end: a new fence, then the copies)
//! or the next step's first gate, so the device window between the step and the release sees no
//! copy; every read of every step lands on its record, the same copies run as under FN8's law, and
//! after a drain every slot is its record. The control: a consumer that skips the gates after a
//! held step reads stale bytes (the hold does not publish by itself).

use saber_mem::device_mover::{DeviceMover, FakeEvent, FakeQueue};
use saber_mem::dio::OpenMode;
use saber_mem::expert_tier::{ExpertTier, TierParams};
use saber_mem::xtier::{write_synthetic, SyntheticTier, XtierFile};
use saber_test_scratch::ScratchDir;

const LAYERS: u32 = 4;
const EXPERTS: u32 = 16;
const VRAM_BASE: u64 = 0x7f00_0000_0000;
const RAM_BASE: u64 = 0x5500_0000_0000;

fn synth(tag: &str) -> (ScratchDir, XtierFile) {
    let d = ScratchDir::new(&format!("saber-fn30c-{tag}"));
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

/// A COMPLETE tier (40 VRAM + 24 RAM = the 64 records), 2 inserts a layer, FN8's law.
fn params() -> TierParams {
    TierParams {
        vram_slots: 40,
        ram_slots: 24,
        min_layer_slots: 2,
        max_inserts: 2,
        read_through: 1.0,
        ..TierParams::default()
    }
}

fn setup(xt: &XtierFile, eager: bool) -> (ExpertTier, DeviceMover<'_, FakeQueue>) {
    let p = params();
    let tier = ExpertTier::new(xt.header(), p.clone(), None, VRAM_BASE, RAM_BASE).unwrap();
    let mut q = FakeQueue::new(xt.header().stride, p.vram_slots, p.ram_slots);
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

/// One step as the engine runs it: gates, reads (stale ones counted; `skip_gate` = the control),
/// the fence, the routes, `end_step` with the hold armed when `hold` (nothing may be issued), then,
/// when `drafter`, the release at the drafter round's end (a new fence first, then the copies).
fn step(
    tier: &mut ExpertTier,
    mv: &mut DeviceMover<'_, FakeQueue>,
    xt: &XtierFile,
    routes: &[Vec<u32>],
    hold: bool,
    drafter: bool,
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
    // the engine's order: the hold is armed BEFORE the step's fence (`tier_end`)
    mv.set_hold(hold);
    mv.after(&()).unwrap();
    for (l, rs) in routes.iter().enumerate() {
        tier.resolve(l as u32, rs, mv).unwrap();
    }
    let log0 = mv.queue().log.len();
    tier.end_step(mv).unwrap();
    tier.check_invariants().unwrap();
    if hold {
        assert_eq!(
            mv.queue().log.len(),
            log0,
            "a held step issued a copy before its release"
        );
    }
    if drafter {
        let h = mv.held();
        let log0 = mv.queue().log.len();
        assert_eq!(mv.release(&()).unwrap(), h);
        if h {
            let log = &mv.queue().log[log0..];
            assert_eq!(log[0], FakeEvent::After, "the release re-fences first");
            assert!(log[1..].iter().all(|e| matches!(e, FakeEvent::Issue(_))));
        }
    }
    stale
}

/// 80 steps; (h2d, d2h, releases, stale reads).
fn run(eager: bool, hold: bool, drafter_every: u64) -> (u64, u64, u64, u64) {
    let (_scratch, xt) = synth(&format!("e{eager}-h{hold}-d{drafter_every}"));
    let (mut tier, mut mv) = setup(&xt, eager);
    let mut g = Lcg(0x5eed);
    let mut stale = 0;
    for s in 0..80u64 {
        let drafter = drafter_every > 0 && s % drafter_every == 0;
        let r = g.routes(1 + (s % 3) as usize);
        stale += step(&mut tier, &mut mv, &xt, &r, hold, drafter, false);
    }
    mv.drain().unwrap();
    check_all_slots(&tier, mv.queue(), &xt);
    let c = mv.counts;
    (c.h2d, c.d2h, c.releases, stale)
}

#[test]
fn held_copies_never_issue_before_the_release_and_every_read_lands_lazy_and_eager() {
    for eager in [false, true] {
        let (h2d, d2h, rel, stale) = run(eager, false, 1);
        assert_eq!(stale, 0);
        assert!(h2d > 0 && d2h > 0);
        assert_eq!(rel, 0, "the control never holds");
        // released at the drafter round's end every step, every 3rd (else the next gate), never
        for every in [1, 3, 0] {
            let (h, d, r, s) = run(eager, true, every);
            assert_eq!(
                s, 0,
                "eager {eager} every {every}: a read missed its record"
            );
            assert_eq!((h, d), (h2d, d2h), "the same copies as FN8's law");
            if every > 0 {
                assert!(r > 0, "no release counted");
            }
        }
    }
}

#[test]
fn control_skipping_the_gates_after_a_held_step_reads_stale_bytes() {
    let (_scratch, xt) = synth("ctl-skip");
    let (mut tier, mut mv) = setup(&xt, false);
    let mut g = Lcg(0x5eed);
    let mut stale = 0;
    for s in 0..80u64 {
        let r = g.routes(1 + (s % 3) as usize);
        stale += step(&mut tier, &mut mv, &xt, &r, true, false, true);
    }
    assert!(
        stale > 0,
        "the control read no stale bytes: it proves nothing"
    );
}
