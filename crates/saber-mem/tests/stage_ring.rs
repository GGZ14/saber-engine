//! FN43: THE PREFILL STAGING RING's host law (`saber_mem::stage_ring`). On a synthetic tier over
//! the host model of the copy stream: the ring's memory is the tier's LENDABLE slots (FN42's
//! inclusive law) -- lent while decode runs, reclaimed by a flip for a staged pass, lent again
//! after it; a staged pass copies each layer's RAM records into its buffer and the retargeted
//! tables read the SAME bytes the UVA tables read (the staging byte law) at every layer; a record
//! past the ring's slots keeps its UVA entry (the fallback) and the which-path counts add up; a
//! read of the ring before its copies is the negative control (it differs).

use saber_mem::device_mover::{DeviceMover, FakeQueue};
use saber_mem::dio::OpenMode;
use saber_mem::expert_tier::{ExpertTier, Placement, TierParams};
use saber_mem::stage_ring::{RingShape, StageKey, StagePlan, PF_SEEN_DECAY};
use saber_mem::xtier::{write_synthetic, SyntheticTier, XtierFile};
use saber_test_scratch::ScratchDir;

const LAYERS: u32 = 4;
const EXPERTS: u32 = 16;
const BASE: u32 = 24;
const VRAM_BASE: u64 = 0x7f00_0000_0000;
const RAM_BASE: u64 = 0x5500_0000_0000;

fn synth(tag: &str) -> (ScratchDir, XtierFile) {
    let d = ScratchDir::new(&format!("saber-fn43-{tag}"));
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

/// The tier with `ring` lendable slots (the ring's memory, after the static slots in the fake
/// VRAM arena), registered and lent.
fn setup(xt: &XtierFile, ring: u32) -> (ExpertTier, DeviceMover<'_, FakeQueue>, Vec<u32>) {
    let p = params();
    let mut tier = ExpertTier::new(xt.header(), p.clone(), None, VRAM_BASE, RAM_BASE).unwrap();
    let s = xt.header().stride;
    let addrs: Vec<u64> = (0..u64::from(ring))
        .map(|i| VRAM_BASE + (u64::from(BASE) + i) * s)
        .collect();
    assert_eq!(tier.register_lendable(&addrs).unwrap(), BASE);
    let mut q = FakeQueue::new(s, BASE + ring, p.ram_slots);
    q.warm(xt, &tier.warm_plan()).unwrap();
    let ids: Vec<u32> = (BASE..BASE + ring).collect();
    assert_eq!(tier.lend(&ids).unwrap(), ring as usize);
    (tier, DeviceMover::new(xt, q, 4), ids)
}

fn fill(tier: &ExpertTier, l: u32) -> (Vec<u64>, Vec<u64>) {
    let n = 3 * (EXPERTS as usize + 1);
    let (mut gu, mut dn) = (vec![0u64; n], vec![0u64; n]);
    tier.fill_xptr(l, &mut gu, &mut dn);
    (gu, dn)
}

/// The record bytes an entry points at (VRAM arena incl. the ring's memory, or the RAM slab).
fn record_at(q: &FakeQueue, tier: &ExpertTier, a: u64, k: usize) -> Vec<u8> {
    let st = tier.stride();
    let base = a - tier.sub_offset(k);
    if base >= VRAM_BASE {
        q.vram_slot(((base - VRAM_BASE) / st) as u32).to_vec()
    } else {
        q.ram_slot(((base - RAM_BASE) / st) as u32).to_vec()
    }
}

/// A few decode steps with the ring's slots lent (the insert law fills them).
fn decode_steps(tier: &mut ExpertTier, mv: &mut DeviceMover<'_, FakeQueue>, seed: u64) {
    let mut x = seed;
    for _ in 0..30 {
        tier.begin_step();
        for l in 0..LAYERS {
            mv.gate(l, &()).unwrap();
        }
        mv.after(&()).unwrap();
        for l in 0..LAYERS {
            let mut v: Vec<u32> = (0..6)
                .map(|_| {
                    x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    ((x >> 33) % u64::from(EXPERTS)).min((x >> 40) % u64::from(EXPERTS)) as u32
                })
                .collect();
            v.sort_unstable();
            v.dedup();
            tier.resolve(l, &v, mv).unwrap();
        }
        tier.end_step(mv).unwrap();
    }
}

/// One staged pass as the engine runs it: drain, reclaim (a flip), plan, copies into the ring
/// (the lendable slots' memory), retargeted tables. Returns (the plan, per layer the UVA tables
/// and the ring tables).
#[allow(clippy::type_complexity)]
fn staged_pass(
    tier: &mut ExpertTier,
    mv: &mut DeviceMover<'_, FakeQueue>,
    ids: &[u32],
    shape: RingShape,
    copy: bool,
) -> (StagePlan, Vec<((Vec<u64>, Vec<u64>), (Vec<u64>, Vec<u64>))>) {
    staged_pass_by(tier, mv, ids, shape, copy, StageKey::Profile)
}

/// [`staged_pass`] under the ring's order key (LR-H).
#[allow(clippy::type_complexity)]
fn staged_pass_by(
    tier: &mut ExpertTier,
    mv: &mut DeviceMover<'_, FakeQueue>,
    ids: &[u32],
    shape: RingShape,
    copy: bool,
    key: StageKey,
) -> (StagePlan, Vec<((Vec<u64>, Vec<u64>), (Vec<u64>, Vec<u64>))>) {
    mv.drain().unwrap();
    let dropped = tier.reclaim(ids);
    DROPPED.with(|d| d.set(d.get() + dropped));
    assert_eq!(tier.lent().0, 0);
    tier.check_invariants().unwrap();
    tier.begin_step();
    let plan = StagePlan::plan_by(tier, LAYERS, shape, key);
    let st = tier.stride() as usize;
    let ring_base = VRAM_BASE + u64::from(BASE) * tier.stride();
    let mut out = Vec::new();
    for l in 0..LAYERS {
        if copy {
            let q = mv.queue_mut();
            for &(r, s) in &plan.layers[l as usize].copies {
                let src = q.ram_slot(r).to_vec();
                let d = (BASE + s) as usize * st;
                q.vram[d..d + st].copy_from_slice(&src);
            }
        }
        let uva = fill(tier, l);
        let mut ring = uva.clone();
        let n = plan.retarget(tier, l, ring_base, &mut ring.0, &mut ring.1);
        assert_eq!(n, plan.layers[l as usize].experts.len());
        // the table is read by the layer, then the next layer's copies reuse the other buffer:
        // check this layer NOW (a buffer of layer l + buffers overwrites it)
        let q = mv.queue();
        for e in 0..EXPERTS as usize {
            for k in 0..3 {
                let a = record_at(q, tier, uva.0[3 * e + k], k);
                let b = record_at(q, tier, ring.0[3 * e + k], k);
                let c = record_at(q, tier, uva.1[3 * e + k], 3 + k);
                let d = record_at(q, tier, ring.1[3 * e + k], 3 + k);
                if copy {
                    assert_eq!(a, b, "layer {l} expert {e} gate_up sub {k}: ring != UVA");
                    assert_eq!(c, d, "layer {l} expert {e} down sub {k}: ring != UVA");
                }
            }
        }
        // LR-H: the ring table's bytes NOW (the buffer is reused by layer l + buffers)
        RING_BYTES.with(|b| b.borrow_mut().push(deref(q, tier, &ring)));
        out.push((uva, ring));
    }
    (plan, out)
}

thread_local! {
    static DROPPED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// LR-H: per layer of the passes run, its retargeted table's dereferenced bytes at the layer.
    static RING_BYTES: std::cell::RefCell<Vec<Vec<Vec<u8>>>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[test]
fn staged_tables_read_the_uva_bytes_through_lend_reclaim_and_relend() {
    let (_scratch, xt) = synth("law");
    let shape = RingShape {
        buffers: 2,
        slots: EXPERTS,
    };
    let (mut tier, mut mv, ids) = setup(&xt, shape.total());
    for round in 0..3u64 {
        decode_steps(&mut tier, &mut mv, 0x43 + round);
        let (plan, tabs) = staged_pass(&mut tier, &mut mv, &ids, shape, true);
        // a whole buffer per layer: EVERY RAM record staged, none left on UVA
        for (l, ls) in plan.layers.iter().enumerate() {
            assert_eq!(ls.overflow, 0);
            let ram = (0..EXPERTS)
                .filter(|&e| matches!(tier.placement(l as u32, e), Some(Placement::Ram(_))))
                .count();
            assert_eq!(ls.copies.len(), ram, "layer {l}");
            // every staged entry moved into the ring; VRAM entries untouched
            let (uva, ring) = &tabs[l];
            for e in 0..EXPERTS as usize {
                let staged = ls.experts.contains(&(e as u32));
                assert_eq!(
                    uva.0[3 * e] != ring.0[3 * e],
                    staged,
                    "layer {l} expert {e}"
                );
            }
        }
        assert!(plan.copies() > 0);
        // the pass ends; the next decode step lends the ring again, EMPTY
        mv.drain().unwrap();
        assert_eq!(tier.lend(&ids).unwrap(), ids.len());
        tier.check_invariants().unwrap();
    }
    // the lent ring slots held records at every reclaim (the flip was exercised)
    assert!(DROPPED.with(std::cell::Cell::get) > 0);
}

#[test]
fn a_record_past_the_rings_slots_falls_back_to_uva_and_the_counts_add_up() {
    let (_scratch, xt) = synth("overflow");
    let shape = RingShape {
        buffers: 2,
        slots: 3,
    };
    let (mut tier, mut mv, ids) = setup(&xt, shape.total());
    decode_steps(&mut tier, &mut mv, 7);
    let (plan, tabs) = staged_pass(&mut tier, &mut mv, &ids, shape, true);
    assert!(
        plan.layers.iter().any(|s| s.overflow > 0),
        "no layer overflows"
    );
    assert!(plan.max_layer_ram() > 3);
    let routed: Vec<u32> = (0..EXPERTS).collect();
    for (l, ls) in plan.layers.iter().enumerate() {
        assert!(ls.copies.len() <= 3);
        // the staged prefix is the tier's order
        let order: Vec<u32> = tier.stage_order(l as u32).iter().map(|x| x.0).collect();
        assert_eq!(ls.experts[..], order[..ls.experts.len()]);
        let (uva, ring) = &tabs[l];
        for &(e, _) in &tier.stage_order(l as u32)[ls.experts.len()..] {
            let e = e as usize;
            assert_eq!(
                ring.0[3 * e],
                uva.0[3 * e],
                "an overflow record keeps its UVA entry"
            );
            assert!(ring.0[3 * e] < VRAM_BASE);
        }
        // every expert routed: ring + uva + vram = the layer's experts, nothing wasted
        let c = plan.classify(&tier, l as u32, &routed);
        assert_eq!(c.ring + c.uva + c.vram, u64::from(EXPERTS));
        assert_eq!(c.ring, ls.copies.len() as u64);
        assert_eq!(c.uva, u64::from(ls.overflow));
        assert_eq!(c.wasted, 0);
        // nothing routed: every staged record is waste
        let c0 = plan.classify(&tier, l as u32, &[]);
        assert_eq!((c0.ring, c0.uva, c0.wasted), (0, 0, ls.copies.len() as u64));
        // another rank's ids are skipped
        let c1 = plan.classify(&tier, l as u32, &[EXPERTS + 1, EXPERTS + 5]);
        assert_eq!(c1.ring + c1.uva + c1.vram, 0);
    }
}

#[test]
fn reading_the_ring_before_its_copies_differs_the_negative_control() {
    let (_scratch, xt) = synth("control");
    let shape = RingShape {
        buffers: 1,
        slots: EXPERTS,
    };
    let (mut tier, mut mv, ids) = setup(&xt, shape.total());
    decode_steps(&mut tier, &mut mv, 11);
    let (plan, tabs) = staged_pass(&mut tier, &mut mv, &ids, shape, false);
    let mut differ = 0;
    let q = mv.queue();
    for (l, (uva, ring)) in tabs.iter().enumerate() {
        for &e in &plan.layers[l].experts {
            let e = e as usize;
            if record_at(q, &tier, uva.0[3 * e], 0) != record_at(q, &tier, ring.0[3 * e], 0) {
                differ += 1;
            }
        }
    }
    assert!(plan.copies() > 0);
    assert!(
        differ > 0,
        "the control must read different bytes without the copies"
    );
}

#[test]
fn one_buffer_and_two_address_distinct_slots() {
    let s2 = RingShape {
        buffers: 2,
        slots: 5,
    };
    assert_eq!(s2.total(), 10);
    assert_eq!((s2.slot(0, 4), s2.slot(1, 0), s2.slot(2, 1)), (4, 5, 1));
    let s1 = RingShape {
        buffers: 1,
        slots: 5,
    };
    assert_eq!((s1.slot(0, 4), s1.slot(3, 2)), (4, 2));
}

/// FN43b: the WR arm's two lenders (FN42b's arena borrow + the ring), each registered as its own
/// lendable run, reclaimed and lent back in BOTH orders across decode steps: the borrow's reclaim
/// law (`lent` drops by exactly the reclaimed client's ids while the other's stay lent -- the
/// check FN43's hold refused at c8192 before 017627c0), no table reads a reclaimed slot, and the
/// invariants hold after every flip.
#[test]
fn two_lenders_reclaim_independently_the_wr_arms_law() {
    let (_scratch, xt) = synth("wr");
    let p = params();
    let mut tier = ExpertTier::new(xt.header(), p.clone(), None, VRAM_BASE, RAM_BASE).unwrap();
    let s = xt.header().stride;
    let (na, nr) = (4u32, 6u32);
    let run = |from: u32, n: u32| -> Vec<u64> {
        (0..u64::from(n))
            .map(|i| VRAM_BASE + (u64::from(from) + i) * s)
            .collect()
    };
    let arena_first = tier.register_lendable(&run(BASE, na)).unwrap();
    assert_eq!(arena_first, BASE);
    let ring_first = tier.register_lendable(&run(BASE + na, nr)).unwrap();
    assert_eq!(
        ring_first,
        BASE + na,
        "the second lender's ids follow the first's"
    );
    let arena: Vec<u32> = (arena_first..arena_first + na).collect();
    let ring: Vec<u32> = (ring_first..ring_first + nr).collect();
    let mut q = FakeQueue::new(s, BASE + na + nr, p.ram_slots);
    q.warm(&xt, &tier.warm_plan()).unwrap();
    tier.lend(&arena).unwrap();
    tier.lend(&ring).unwrap();
    assert_eq!(tier.lent(), (na + nr, na + nr));
    let mut mv = DeviceMover::new(&xt, q, 4);
    for (cycle, arena_first_order) in [true, false, true, false].into_iter().enumerate() {
        decode_steps(&mut tier, &mut mv, 17 + cycle as u64);
        mv.drain().unwrap();
        let order: [(&[u32], u32); 2] = if arena_first_order {
            [(&arena, na), (&ring, nr)]
        } else {
            [(&ring, nr), (&arena, na)]
        };
        for (ids, n) in order {
            let before = tier.lent().0;
            tier.reclaim(ids);
            // the worker's borrow_for law (FN43's fix): exactly this client's slots come back
            assert_eq!(tier.lent().0 + n, before, "cycle {cycle}");
            tier.check_invariants().unwrap();
            for l in 0..LAYERS {
                let (gu, _) = fill(&tier, l);
                for &a in gu.iter().filter(|&&a| a >= VRAM_BASE) {
                    let slot = ((a - VRAM_BASE) / s) as u32;
                    assert!(
                        !ids.contains(&slot),
                        "cycle {cycle}: a table still reads a reclaimed slot"
                    );
                }
            }
        }
        assert_eq!(tier.lent().0, 0);
        tier.lend(&ring).unwrap();
        tier.lend(&arena).unwrap();
        tier.check_invariants().unwrap();
        assert_eq!(tier.lent().0, na + nr);
    }
}

/// LR-H: the dereferenced record bytes of every (expert, sub) entry of one layer's tables.
fn deref(q: &FakeQueue, tier: &ExpertTier, t: &(Vec<u64>, Vec<u64>)) -> Vec<Vec<u8>> {
    let mut v = Vec::new();
    for e in 0..EXPERTS as usize {
        for k in 0..3 {
            v.push(record_at(q, tier, t.0[3 * e + k], k));
            v.push(record_at(q, tier, t.1[3 * e + k], 3 + k));
        }
    }
    v
}

/// LR-H: `--xtier-stage-order served` PERMUTES which RAM records take the ring's slots (a served
/// prefill history that disagrees with the warm profile moves the staged set), and the
/// retargeted tables of BOTH keys dereference to the UVA bytes -- identical bytes per (expert,
/// sub) entry, at every layer, through a lend / reclaim cycle; the default key is today's plan
/// exactly; the which-path counts price the swap (`misordered`).
#[test]
fn the_served_order_permutes_the_staged_set_and_every_table_reads_the_same_bytes() {
    let (_scratch, xt) = synth("served");
    let shape = RingShape {
        buffers: 2,
        slots: 3,
    };
    let (mut tier, mut mv, ids) = setup(&xt, shape.total());
    decode_steps(&mut tier, &mut mv, 0x1e4);
    // the served history: per layer the RAM records the profile order ranks LAST
    let mut hot: Vec<Vec<u32>> = Vec::new();
    for l in 0..LAYERS {
        let order: Vec<u32> = tier.stage_order(l).iter().map(|x| x.0).collect();
        hot.push(order.iter().rev().take(3).copied().collect());
    }
    for _ in 0..4 {
        tier.decay_prefill_seen();
        for l in 0..LAYERS {
            tier.note_prefill_routes(l, &hot[l as usize]);
        }
    }
    RING_BYTES.with(|b| b.borrow_mut().clear());
    let (p_plan, p_tabs) = staged_pass_by(&mut tier, &mut mv, &ids, shape, true, StageKey::Profile);
    // the default key is today's plan, byte for byte
    assert_eq!(p_plan, StagePlan::plan(&tier, LAYERS, shape));
    let p_bytes: Vec<Vec<Vec<u8>>> = RING_BYTES.with(|b| b.borrow_mut().drain(..).collect());
    assert_eq!(p_bytes.len(), LAYERS as usize);
    mv.drain().unwrap();
    assert_eq!(tier.lend(&ids).unwrap(), ids.len());
    let (s_plan, s_tabs) = staged_pass_by(&mut tier, &mut mv, &ids, shape, true, StageKey::Served);
    let s_bytes_all: Vec<Vec<Vec<u8>>> = RING_BYTES.with(|b| b.borrow_mut().drain(..).collect());
    let mut moved = 0;
    for l in 0..LAYERS {
        let (pl, sl) = (&p_plan.layers[l as usize], &s_plan.layers[l as usize]);
        // the same slots a layer, the same RAM set ordered differently
        assert_eq!(pl.copies.len(), sl.copies.len(), "layer {l}");
        assert_eq!(pl.overflow, sl.overflow, "layer {l}");
        let mut a: Vec<u32> = tier.stage_order(l).iter().map(|x| x.0).collect();
        let mut b: Vec<u32> = tier
            .stage_order_by(l, StageKey::Served)
            .iter()
            .map(|x| x.0)
            .collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "layer {l}: the orders are one permutation");
        if pl.experts != sl.experts {
            moved += 1;
        }
        // served stages the served-hot records first
        let want = hot[l as usize].len().min(sl.experts.len());
        for e in &sl.experts[..want] {
            assert!(
                hot[l as usize].contains(e),
                "layer {l}: {e} is not served-hot"
            );
        }
        // THE BYTE LAW: every (expert, sub) entry reads identical bytes under both keys
        // (each read at its layer, before layer l + buffers reused the buffer)
        let s_bytes = &s_bytes_all[l as usize];
        assert_eq!(
            &p_bytes[l as usize], s_bytes,
            "layer {l}: served != profile bytes"
        );
        assert_eq!(
            &deref(mv.queue(), &tier, &s_tabs[l as usize].0),
            s_bytes,
            "layer {l}: served ring != UVA bytes"
        );
        assert_eq!(
            deref(mv.queue(), &tier, &p_tabs[l as usize].0),
            deref(mv.queue(), &tier, &s_tabs[l as usize].0),
            "layer {l}: the UVA tables moved between the passes"
        );
        // the counts at the served-hot routes: served wastes no more than profile
        let pc = p_plan.classify(&tier, l, &hot[l as usize]);
        let sc = s_plan.classify(&tier, l, &hot[l as usize]);
        assert!(sc.wasted <= pc.wasted, "layer {l}");
        assert_eq!(
            pc.ring + pc.uva,
            sc.ring + sc.uva,
            "layer {l}: same routed RAM"
        );
        assert_eq!(pc.misordered, pc.wasted.min(pc.uva));
        assert_eq!(
            sc.misordered, 0,
            "layer {l}: served is the ideal order here"
        );
    }
    assert!(moved > 0, "the served key moved no layer's staged set");
}

/// LR-H: the served key's count -- one per staged pass that routed the record (repeats once,
/// another rank's ids skipped), decayed by `PF_SEEN_DECAY` a pass; with no served history the
/// served order IS the profile order (its tie-break chain is the profile key).
#[test]
fn the_served_key_decays_counts_once_a_pass_and_falls_back_to_the_profile() {
    let (_scratch, xt) = synth("decay");
    let (mut tier, mut mv, _ids) = setup(&xt, 6);
    decode_steps(&mut tier, &mut mv, 5);
    for l in 0..LAYERS {
        assert_eq!(
            tier.stage_order_by(l, StageKey::Served),
            tier.stage_order(l)
        );
        assert_eq!(
            tier.stage_order_by(l, StageKey::Profile),
            tier.stage_order(l)
        );
    }
    tier.decay_prefill_seen();
    tier.note_prefill_routes(1, &[2, 2, 7, EXPERTS + 3]);
    assert_eq!(tier.prefill_seen(1, 2), 1.0);
    assert_eq!(tier.prefill_seen(1, 7), 1.0);
    assert_eq!(tier.prefill_seen(1, 3), 0.0);
    assert_eq!(tier.prefill_seen(0, 2), 0.0);
    tier.decay_prefill_seen();
    tier.note_prefill_routes(1, &[2]);
    assert!((tier.prefill_seen(1, 2) - (1.0 + PF_SEEN_DECAY)).abs() < 1e-6);
    assert!((tier.prefill_seen(1, 7) - PF_SEEN_DECAY).abs() < 1e-6);
    // a RAM record with served history leads the served order of its layer
    let ram: Vec<u32> = tier.stage_order(1).iter().map(|x| x.0).collect();
    let last = *ram.last().unwrap();
    for _ in 0..3 {
        tier.decay_prefill_seen();
        tier.note_prefill_routes(1, &[last]);
    }
    assert_eq!(tier.stage_order_by(1, StageKey::Served)[0].0, last);
    assert_eq!(StageKey::parse("served"), Ok(StageKey::Served));
    assert_eq!(StageKey::parse("profile"), Ok(StageKey::Profile));
    assert!(StageKey::parse("lrfu").is_err());
    assert_eq!(StageKey::default(), StageKey::Profile);
}
