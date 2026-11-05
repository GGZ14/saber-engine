//! FN8: the device mover over a host model of an ASYNCHRONOUS copy stream ([`FakeQueue`]: issued
//! copies sit in flight until a consumer waits on a layer mark behind them or the host drains).
//! The tier runs FN7's policy over the synthetic `experts.xtier`; every routed expert's `xptr`
//! entry is read AFTER its layer's gate and must land on its record's bytes (sha256), every slot
//! is checked after a drain, and the controls show what the law prevents: a consumer that skips
//! the gate reads stale bytes, a copy queued before the fence is refused, a host write with copies
//! in flight is refused.

use saber_mem::device_mover::{CopyOp, CopyQueue, DeviceMover, FakeEvent, FakeQueue};
use saber_mem::dio::OpenMode;
use saber_mem::expert_tier::{ExpertTier, TierParams};
use saber_mem::xtier::{write_synthetic, SyntheticTier, XtierFile};
use saber_test_scratch::ScratchDir;

const LAYERS: u32 = 4;
const EXPERTS: u32 = 16;
const VRAM_BASE: u64 = 0x7f00_0000_0000;
const RAM_BASE: u64 = 0x5500_0000_0000;

fn synth(rank: u32, tag: &str) -> (ScratchDir, XtierFile) {
    let d = ScratchDir::new(&format!("saber-fn8-{tag}-r{rank}"));
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
    fn expert(&mut self) -> u32 {
        let a = self.next() % (2 * u64::from(EXPERTS));
        let b = self.next() % (2 * u64::from(EXPERTS));
        a.min(b) as u32
    }
}

fn step_routes(g: &mut Lcg, tokens: usize) -> Vec<Vec<u32>> {
    (0..LAYERS)
        .map(|_| (0..tokens * 10).map(|_| g.expert()).collect())
        .collect()
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

fn setup<'x>(xt: &'x XtierFile, p: &TierParams) -> (ExpertTier, DeviceMover<'x, FakeQueue>) {
    let tier = ExpertTier::new(xt.header(), p.clone(), None, VRAM_BASE, RAM_BASE).unwrap();
    let mut q = FakeQueue::new(xt.header().stride, p.vram_slots, p.ram_slots);
    q.warm(xt, &tier.warm_plan()).unwrap();
    (tier, DeviceMover::new(xt, q, 4))
}

/// The bytes the kernel reads for a routed expert of `layer` NOW, through the published xptr.
fn read_through_xptr<'q>(
    tier: &ExpertTier,
    q: &'q FakeQueue,
    xt: &XtierFile,
    layer: u32,
    e: u32,
) -> (u64, &'q [u8]) {
    let h = xt.header();
    let n = EXPERTS as usize;
    let (mut gu, mut dn) = (vec![0u64; 3 * (n + 1)], vec![0u64; 3 * (n + 1)]);
    tier.fill_xptr(layer, &mut gu, &mut dn);
    let loc = (e - h.first_expert) as usize;
    let a = gu[3 * loc] - h.sub[0].0;
    assert_eq!(dn[3 * loc], a + h.sub[3].0);
    let i = u64::from(tier.index(layer, e).unwrap());
    if a >= VRAM_BASE {
        (i, q.vram_slot(((a - VRAM_BASE) / h.stride) as u32))
    } else {
        (i, q.ram_slot(((a - RAM_BASE) / h.stride) as u32))
    }
}

fn check_all_slots(tier: &ExpertTier, q: &FakeQueue, xt: &XtierFile) {
    tier.check_invariants().unwrap();
    for s in 0..tier.params().vram_slots {
        if let Some(r) = tier.vram_record(s) {
            assert!(
                xt.verify(u64::from(r), q.vram_slot(s)),
                "vram slot {s} != record {r}"
            );
        }
    }
    for s in 0..=tier.params().ram_slots {
        if let Some(r) = tier.ram_record(s) {
            assert!(
                xt.verify(u64::from(r), q.ram_slot(s)),
                "ram slot {s} != record {r}"
            );
        }
    }
}

/// One step through the law: resolve, gate, the kernel's reads (checked), fence, end_step.
fn step(
    tier: &mut ExpertTier,
    mv: &mut DeviceMover<'_, FakeQueue>,
    xt: &XtierFile,
    routes: &[Vec<u32>],
) -> u64 {
    tier.begin_step();
    let mut reads = 0;
    for (l, rs) in routes.iter().enumerate() {
        let l = l as u32;
        tier.resolve(l, rs, mv).unwrap();
        tier.check_dispatch(l, rs).unwrap();
        mv.gate(l, &()).unwrap();
        for &e in rs {
            if tier.index(l, e).is_none() {
                continue;
            }
            let (i, bytes) = read_through_xptr(tier, mv.queue(), xt, l, e);
            assert!(
                xt.verify(i, bytes),
                "layer {l} expert {e}: the kernel reads another record"
            );
            reads += 1;
        }
    }
    mv.after(&()).unwrap();
    tier.end_step(mv).unwrap();
    reads
}

#[test]
fn every_routed_read_lands_on_its_record_through_the_async_queue() {
    for rank in [0, 1] {
        let (_scratch, xt) = synth(rank, "async");
        // Complete (every record in VRAM or RAM) and demand (NVMe reads + RAM drops).
        for (vram, ram) in [(40, 24), (40, 12)] {
            let p = params(vram, ram);
            let (mut tier, mut mv) = setup(&xt, &p);
            let mut g = Lcg(0xfee8 + u64::from(rank) + u64::from(ram));
            let mut reads = 0;
            for s in 0..60 {
                let routes = step_routes(&mut g, 1 + s % 3);
                reads += step(&mut tier, &mut mv, &xt, &routes);
            }
            let (paths, moves) = tier.totals();
            assert!(reads > 0 && moves.inserts > 0 && moves.demotions > 0);
            assert_eq!(mv.counts.h2d, moves.inserts);
            assert_eq!(mv.counts.d2h, moves.demotions);
            assert_eq!(mv.counts.nvme_reads, paths.nvme);
            assert!(
                mv.counts.gate_waits > 0,
                "copies were in flight at some gate"
            );
            mv.drain().unwrap();
            assert_eq!(mv.queue().in_flight(), 0);
            check_all_slots(&tier, mv.queue(), &xt);
        }
    }
}

#[test]
fn the_ordering_law_holds_in_the_issued_stream() {
    let (_scratch, xt) = synth(0, "order");
    let p = params(40, 24);
    let (mut tier, mut mv) = setup(&xt, &p);
    let mut g = Lcg(77);
    for s in 0..40 {
        step(&mut tier, &mut mv, &xt, &step_routes(&mut g, 1 + s % 2));
    }
    let log = &mv.queue().log;
    let ops: Vec<CopyOp> = log
        .iter()
        .filter_map(|e| match e {
            FakeEvent::Issue(op) => Some(*op),
            _ => None,
        })
        .collect();
    // Rule 2: every demotion is immediately followed by the promotion INTO the slot it emptied.
    let mut swaps = 0;
    for w in ops.windows(2) {
        if let CopyOp::D2h { vram, .. } = w[0] {
            assert_eq!(
                match w[1] {
                    CopyOp::H2d { vram: v, .. } => Some(v),
                    _ => None,
                },
                Some(vram),
                "a demotion out of slot {vram} not followed by the promotion into it"
            );
            swaps += 1;
        }
    }
    assert!(swaps > 0);
    // Rule 3: every copy of a layer's partition precedes that layer's mark, which closes its batch.
    let mut open: Vec<u32> = Vec::new();
    for op in &ops {
        match *op {
            CopyOp::H2d { vram, .. } | CopyOp::D2h { vram, .. } => open.push(vram),
            CopyOp::Landed { layer } => {
                let (b, n) = tier.layer_slots(layer);
                assert!(!open.is_empty(), "a mark with no copies");
                for v in open.drain(..) {
                    assert!(
                        (b..b + n).contains(&v),
                        "slot {v} copied under layer {layer}'s mark"
                    );
                }
            }
        }
    }
    assert!(open.is_empty(), "copies issued without a closing mark");
    // Rule 1: between two fences there is no copy issued before the fence of its step: every
    // Issue run is preceded (since the last Wait) by an After.
    let mut fenced = false;
    for e in log {
        match e {
            FakeEvent::After => fenced = true,
            FakeEvent::Wait(_) => fenced = false,
            FakeEvent::Issue(CopyOp::H2d { .. } | CopyOp::D2h { .. }) => {
                assert!(fenced, "a copy issued before the consumer fence")
            }
            _ => {}
        }
    }
}

#[test]
fn control_a_consumer_that_skips_the_gate_reads_stale_bytes() {
    let (_scratch, xt) = synth(0, "stale");
    let p = params(40, 24);
    let (mut tier, mut mv) = setup(&xt, &p);
    let mut g = Lcg(5);
    let mut caught = 0;
    for s in 0..40 {
        let routes = step_routes(&mut g, 1 + s % 2);
        let inserts_before = tier.totals().1.inserts;
        step(&mut tier, &mut mv, &xt, &routes);
        if tier.totals().1.inserts == inserts_before {
            continue;
        }
        // Copies of this end_step are in flight. Read every VRAM slot of an in-flight layer the
        // way a kernel would WITHOUT the gate: at least one must be stale (the promoted record
        // has not landed), and after the gate every one is its record.
        for l in 0..LAYERS {
            if !mv.in_flight(l) {
                continue;
            }
            let (b, n) = tier.layer_slots(l);
            let stale = (b..b + n)
                .filter(|&s| {
                    tier.vram_record(s)
                        .is_some_and(|r| !xt.verify(u64::from(r), mv.queue().vram_slot(s)))
                })
                .count();
            assert!(stale > 0, "layer {l}: copies in flight but nothing stale");
            caught += 1;
            mv.gate(l, &()).unwrap();
            for s in b..b + n {
                if let Some(r) = tier.vram_record(s) {
                    assert!(xt.verify(u64::from(r), mv.queue().vram_slot(s)));
                }
            }
        }
    }
    assert!(caught > 0, "the control never ran");
}

#[test]
fn control_a_copy_before_the_fence_is_refused() {
    let (_scratch, xt) = synth(0, "fence");
    let p = params(40, 24);
    let (mut tier, mut mv) = setup(&xt, &p);
    let mut g = Lcg(9);
    for _ in 0..40 {
        let routes = step_routes(&mut g, 1);
        tier.begin_step();
        for (l, rs) in routes.iter().enumerate() {
            tier.resolve(l as u32, rs, &mut mv).unwrap();
            mv.gate(l as u32, &()).unwrap();
        }
        // No `after`: the first promotion must be refused.
        match tier.end_step(&mut mv) {
            Ok(r) => assert_eq!(r.moves.inserts, 0, "a copy slipped past the fence"),
            Err(e) => {
                assert!(e.to_string().contains("fenced"), "{e}");
                return;
            }
        }
    }
    panic!("no step inserted: the control never ran");
}

#[test]
#[should_panic(expected = "copies in flight")]
fn control_a_host_write_with_copies_in_flight_is_refused() {
    let (_scratch, xt) = synth(0, "hostw");
    let p = params(40, 24);
    let (mut tier, mut mv) = setup(&xt, &p);
    let mut g = Lcg(11);
    for _ in 0..40 {
        step(&mut tier, &mut mv, &xt, &step_routes(&mut g, 1));
        if mv.queue().in_flight() > 0 {
            // What the mover itself never does (law 4: it drains first).
            let _ = mv.queue_mut().ram_mut();
        }
    }
}

/// FN71c: `--xtier-nvme-p2p on` -- the reads whose slot lives in the pool (the valve's bounce) go
/// drive -> pool, the rest drive -> the slab, in ONE `nvme_to_ram`; every slot holds its record
/// (sha256) and the which-path counters split exactly. Control: the same reads with no pool all
/// take the host route (0 pool reads) and land in the slab.
#[test]
fn pool_slots_are_read_drive_to_pool_and_the_rest_through_the_slab() {
    use saber_mem::expert_tier::ExpertMover as _;
    let (_d, xt) = synth(0, "pool");
    let st = xt.header().stride;
    let (ram_slots, lo, n) = (24u32, 16u32, 8u32);
    let reads: Vec<(u64, u32)> = vec![(3, 1), (17, 16), (40, 5), (41, 23), (63, 20), (9, 17)];
    let mut mv = DeviceMover::new(&xt, FakeQueue::new(st, 4, ram_slots).with_pool(lo, n), 4);
    mv.nvme_to_ram(&reads).unwrap();
    assert_eq!(mv.counts.nvme_pool_reads, 4);
    assert_eq!(mv.counts.nvme_reads, 2);
    for &(i, slot) in &reads {
        let bytes = if slot >= lo {
            mv.queue().pool_slot(slot - lo).to_vec()
        } else {
            mv.queue().ram_slot(slot).to_vec()
        };
        assert!(xt.verify(i, &bytes), "record {i} in slot {slot}");
        if slot >= lo {
            // the slab's copy of a pool slot was never written
            assert!(mv.queue().ram_slot(slot).iter().all(|&b| b == 0));
        }
    }
    // the control: no pool -> every read takes the host route into the slab
    let mut ctl = DeviceMover::new(&xt, FakeQueue::new(st, 4, ram_slots), 4);
    ctl.nvme_to_ram(&reads).unwrap();
    assert_eq!(ctl.counts.nvme_pool_reads, 0);
    assert_eq!(ctl.counts.nvme_reads, 6);
    for &(i, slot) in &reads {
        assert!(
            xt.verify(i, ctl.queue().ram_slot(slot)),
            "control record {i}"
        );
    }
}
