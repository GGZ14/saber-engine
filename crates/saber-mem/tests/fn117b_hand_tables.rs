//! FN117b (S2): THE NEXT STEP'S XPTR TABLES BUILT UNDER THE DRAFTER ROUND. The engine builds (and
//! uploads) the next verify step's expert-pointer tables right after the verify's tier end -- while
//! the drafter round runs on the card -- instead of at the next step's tier begin, and keeps them
//! iff [`ExpertTier::xptr_key`] at the begin equals the key it took at the build. Over the host
//! model of the copy stream (lazy and eager), a replay with the engine's between-step events
//! (EMPTY lends at decode, reclaims at an admission / a wide pass, lends again):
//!
//! * the early build calls no mutating tier method: the arm's placements are the control's,
//!   step for step (the lending order is untouched: lend / reclaim / begin / resolve / end run
//!   exactly where they run today);
//! * a kept build IS the begin's tables, byte for byte, and every read through it lands on its
//!   record (0 stale);
//! * THE NEGATIVE CONTROL: a reclaim between the build and the begin changes the key, and the
//!   build it voids would have read another record's bytes (stale) had it been kept.

use saber_mem::device_mover::{DeviceMover, FakeQueue};
use saber_mem::dio::OpenMode;
use saber_mem::expert_tier::{ExpertTier, TierParams, XptrKey};
use saber_mem::xtier::{write_synthetic, SyntheticTier, XtierFile};
use saber_test_scratch::ScratchDir;

const LAYERS: u32 = 4;
const EXPERTS: u32 = 16;
const BASE: u32 = 24;
const LENDABLE: u32 = 16;
const VRAM_BASE: u64 = 0x7f00_0000_0000;
const RAM_BASE: u64 = 0x5500_0000_0000;

fn synth(tag: &str) -> (ScratchDir, XtierFile) {
    let d = ScratchDir::new(&format!("saber-fn117b-{tag}"));
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

fn setup<'x>(xt: &'x XtierFile, eager: bool) -> (ExpertTier, DeviceMover<'x, FakeQueue>) {
    let p = params();
    let mut tier = ExpertTier::new(xt.header(), p.clone(), None, VRAM_BASE, RAM_BASE).unwrap();
    let s = xt.header().stride;
    let addrs: Vec<u64> = (0..u64::from(LENDABLE))
        .map(|i| VRAM_BASE + (u64::from(BASE) + i) * s)
        .collect();
    assert_eq!(tier.register_lendable(&addrs).unwrap(), BASE);
    let mut q = FakeQueue::new(xt.header().stride, BASE + LENDABLE, p.ram_slots);
    q.warm(xt, &tier.warm_plan()).unwrap();
    q.eager = eager;
    (tier, DeviceMover::new(xt, q, 4))
}

/// Every layer's two tables, as the engine fills them (`TierRank::tables`: the shared expert's
/// entry left at 0 here).
fn tables(tier: &ExpertTier) -> Vec<Vec<u64>> {
    let n = EXPERTS as usize;
    (0..LAYERS)
        .map(|l| {
            let (mut gu, mut dn) = (vec![0u64; 3 * (n + 1)], vec![0u64; 3 * (n + 1)]);
            tier.fill_xptr(l, &mut gu, &mut dn);
            gu.extend_from_slice(&dn);
            gu
        })
        .collect()
}

fn landed(q: &FakeQueue, stride: u64, a: u64) -> &[u8] {
    if a >= VRAM_BASE {
        q.vram_slot(((a - VRAM_BASE) / stride) as u32)
    } else {
        q.ram_slot(((a - RAM_BASE) / stride) as u32)
    }
}

/// Reads of one step through `tabs` (gate first, as the step's graph): (reads, stale).
fn reads(
    tier: &ExpertTier,
    mv: &mut DeviceMover<'_, FakeQueue>,
    xt: &XtierFile,
    tabs: &[Vec<u64>],
    routes: &[Vec<u32>],
) -> (u64, u64) {
    let stride = tier.stride();
    let off0 = tier.sub_offset(0);
    let (mut n, mut stale) = (0, 0);
    for (l, rs) in routes.iter().enumerate() {
        mv.gate(l as u32, &()).unwrap();
        for &e in rs {
            let i = u64::from(tier.index(l as u32, e).unwrap());
            n += 1;
            let a = tabs[l][3 * e as usize] - off0;
            if !xt.verify(i, landed(mv.queue(), stride, a)) {
                stale += 1;
            }
        }
    }
    (n, stale)
}

/// The negative control's reads: EVERY expert of every layer through `tabs` on `tier`'s queue
/// (the records a stale build would misread, whatever the step routes).
fn would_be_stale(
    tier: &ExpertTier,
    mv: &mut DeviceMover<'_, FakeQueue>,
    xt: &XtierFile,
    tabs: &[Vec<u64>],
) -> u64 {
    let all: Vec<Vec<u32>> = (0..LAYERS).map(|_| (0..EXPERTS).collect()).collect();
    reads(tier, mv, xt, tabs, &all).1
}

#[test]
fn the_tables_built_after_the_tier_end_are_the_next_begins_and_a_reclaim_voids_them() {
    for eager in [false, true] {
        let (_s, xt) = synth(if eager { "eager" } else { "lazy" });
        let (mut arm, mut ma) = setup(&xt, eager);
        let (mut ctl, mut mc) = setup(&xt, eager);
        let ids: Vec<u32> = (BASE..BASE + LENDABLE).collect();
        arm.lend(&ids).unwrap();
        ctl.lend(&ids).unwrap();
        let mut g = Lcg(0x117b ^ u64::from(eager));
        let mut pre: Option<(XptrKey, Vec<Vec<u64>>)> = None;
        let (mut kept, mut voided, mut moved, mut stale_if_kept, mut total) =
            (0u64, 0u64, 0u64, 0u64, 0u64);
        for s in 0..600u32 {
            let routes = g.routes(1 + (s % 4) as usize);
            // the between-step events (both tiers alike, where the engine runs them): an
            // admission / a wide pass reclaims half the lent slots (the copy stream drained
            // first); a later decode step lends them again, EMPTY
            if s % 53 == 20 {
                ma.drain().unwrap();
                mc.drain().unwrap();
                let (da, dc) = (arm.reclaim(&ids[8..]), ctl.reclaim(&ids[8..]));
                assert_eq!(da, dc);
                // the owner writes its memory back (the wide chunk's arena over the slots)
                for &sl in &ids[8..] {
                    ma.queue_mut().scribble(sl, 0xa5);
                    mc.queue_mut().scribble(sl, 0xa5);
                }
            }
            if s % 53 == 30 {
                assert_eq!(arm.lend(&ids).unwrap(), ctl.lend(&ids).unwrap());
            }
            // THE BEGIN: the control builds its tables here (today); the arm keeps its early
            // build iff the key holds
            arm.begin_step();
            ctl.begin_step();
            let truth = tables(&ctl);
            let key = arm.xptr_key();
            let used = match pre.take() {
                Some((k, t)) if k == key => {
                    assert_eq!(t, truth, "step {s}: a kept build differs from the begin's");
                    kept += 1;
                    t
                }
                Some((_, t)) => {
                    voided += 1;
                    if t != truth {
                        moved += 1;
                        // the negative control: kept anyway, the stale build reads wrong bytes
                        stale_if_kept += would_be_stale(&ctl, &mut mc, &xt, &t);
                    }
                    tables(&arm)
                }
                None => tables(&arm),
            };
            assert_eq!(used, truth);
            let (n, stale) = reads(&arm, &mut ma, &xt, &used, &routes);
            assert_eq!(
                stale, 0,
                "step {s} (eager {eager}): a read landed on another record"
            );
            reads(&ctl, &mut mc, &xt, &truth, &routes);
            total += n;
            ma.after(&()).unwrap();
            mc.after(&()).unwrap();
            for (l, rs) in routes.iter().enumerate() {
                arm.resolve(l as u32, rs, &mut ma).unwrap();
                ctl.resolve(l as u32, rs, &mut mc).unwrap();
            }
            arm.end_step(&mut ma).unwrap();
            ctl.end_step(&mut mc).unwrap();
            // the early build moved nothing: the arm's placement is the control's
            assert_eq!(arm.placements(), ctl.placements(), "step {s}");
            // THE ARM: the next step's tables right after the tier end (under the drafter round)
            pre = Some((arm.xptr_key(), tables(&arm)));
        }
        assert_eq!(arm.totals(), ctl.totals());
        assert!(total > 0);
        assert!(kept >= 580, "eager {eager}: {kept} of 600 builds kept");
        assert!(
            voided > 0 && moved > 0,
            "the reclaims void their step's build"
        );
        assert!(
            stale_if_kept > 0,
            "the negative control: a voided build that was kept would have read stale bytes"
        );
    }
}

#[test]
fn equal_keys_fill_equal_tables_and_a_lend_of_empty_slots_keeps_the_key() {
    let (_s, xt) = synth("key");
    let (mut t, mut m) = setup(&xt, false);
    let ids: Vec<u32> = (BASE..BASE + LENDABLE).collect();
    let mut g = Lcg(5);
    for s in 0..50u32 {
        t.begin_step();
        let routes = g.routes(2);
        m.after(&()).unwrap();
        for (l, rs) in routes.iter().enumerate() {
            t.resolve(l as u32, rs, &mut m).unwrap();
        }
        t.end_step(&mut m).unwrap();
        if s == 25 {
            let (k0, t0) = (t.xptr_key(), tables(&t));
            assert_eq!(t.lend(&ids).unwrap(), LENDABLE as usize);
            assert_eq!(t.xptr_key(), k0, "an EMPTY lend moves no placement");
            assert_eq!(tables(&t), t0);
        }
    }
    let k = t.xptr_key();
    let tb = tables(&t);
    t.begin_step();
    assert_eq!(
        t.xptr_key(),
        k,
        "a complete tier's begin moves no placement"
    );
    assert_eq!(tables(&t), tb);
}
