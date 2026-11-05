//! FN107 (X1): THE POOL SLOTS EXCLUSIVE -- lent slots their owner never reclaims (the P2P pool
//! after the boot fill) are made EXCLUSIVE: their records drop the RAM shadow, the RAM L2 is the
//! complement of the static AND exclusive slots (16 records smaller here), the freed RAM slots
//! take the records the smaller L2 left on the drive, and a victim leaving an exclusive slot is
//! WRITTEN BACK to RAM through the spare (FN8's swap). Over the host model of the copy stream
//! ([`FakeQueue`], lazy and eager): the tier is complete after the refill, every read of every step
//! lands on its record, write-backs happen and every written-back record re-reads identical from
//! RAM; a reclaim never drops an exclusive slot; only lent slots can be made exclusive.

use saber_mem::device_mover::{DeviceMover, FakeQueue};
use saber_mem::dio::OpenMode;
use saber_mem::expert_tier::{ExpertTier, Placement, TierParams};
use saber_mem::xtier::{write_synthetic, SyntheticTier, XtierFile};
use saber_test_scratch::ScratchDir;

const LAYERS: u32 = 4;
const EXPERTS: u32 = 16;
const BASE: u32 = 24;
const POOL: u32 = 16;
const VRAM_BASE: u64 = 0x7f00_0000_0000;
const RAM_BASE: u64 = 0x5500_0000_0000;

fn synth(tag: &str) -> (ScratchDir, XtierFile) {
    let d = ScratchDir::new(&format!("saber-fn107-{tag}"));
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

/// 24 static VRAM slots + 16 pool slots + an L2 of 64 - 24 - `excl` records.
fn params(excl: u32) -> TierParams {
    TierParams {
        vram_slots: BASE,
        ram_slots: 64 - BASE - excl,
        min_layer_slots: 2,
        max_inserts: 2,
        read_through: 1.0,
        lrfu_lambda: 0.98,
        ..TierParams::default()
    }
}

fn setup<'x>(
    xt: &'x XtierFile,
    p: &TierParams,
    eager: bool,
) -> (ExpertTier, DeviceMover<'x, FakeQueue>) {
    let mut tier = ExpertTier::new(xt.header(), p.clone(), None, VRAM_BASE, RAM_BASE).unwrap();
    let s = xt.header().stride;
    let addrs: Vec<u64> = (0..u64::from(POOL))
        .map(|i| VRAM_BASE + (u64::from(BASE) + i) * s)
        .collect();
    assert_eq!(tier.register_lendable(&addrs).unwrap(), BASE);
    let mut q = FakeQueue::new(xt.header().stride, BASE + POOL, p.ram_slots);
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

/// Every placed slot is its record (the placement byte law, `--fn-verify-slots`).
fn check_all_slots(tier: &ExpertTier, q: &FakeQueue, xt: &XtierFile) -> u64 {
    tier.check_invariants().unwrap();
    let mut n = 0;
    for s in 0..BASE + POOL {
        if let Some(r) = tier.vram_record(s) {
            assert!(xt.verify(u64::from(r), q.vram_slot(s)), "vram slot {s}");
            n += 1;
        }
    }
    for s in 0..tier.ram_slab_slots() {
        if let Some(r) = tier.ram_record(s) {
            assert!(xt.verify(u64::from(r), q.ram_slot(s)), "ram slot {s}");
            n += 1;
        }
    }
    n
}

fn step(
    tier: &mut ExpertTier,
    mv: &mut DeviceMover<'_, FakeQueue>,
    xt: &XtierFile,
    routes: &[Vec<u32>],
) -> u64 {
    tier.begin_step();
    let stride = tier.stride();
    let tabs: Vec<Vec<u64>> = (0..LAYERS).map(|l| tables(tier, l)).collect();
    let mut stale = 0;
    for (l, rs) in routes.iter().enumerate() {
        mv.gate(l as u32, &()).unwrap();
        for &e in rs {
            let i = u64::from(tier.index(l as u32, e).unwrap());
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
    stale
}

/// The boot as the worker runs it: lend + fill the pool (inclusive), make it exclusive, refill RAM.
fn boot_exclusive(xt: &XtierFile, eager: bool) -> (ExpertTier, DeviceMover<'_, FakeQueue>) {
    let (mut tier, mut mv) = setup(xt, &params(POOL), eager);
    let (_, ram, fetch) = tier.occupancy();
    assert_eq!(
        (ram, fetch),
        (u64::from(64 - BASE - POOL), u64::from(POOL)),
        "the smaller L2 leaves the pool's share on the drive at the warm"
    );
    let ids: Vec<u32> = (BASE..BASE + POOL).collect();
    assert_eq!(tier.lend(&ids).unwrap(), POOL as usize);
    mv.after(&()).unwrap();
    assert_eq!(tier.fill_lent(&mut mv).unwrap(), POOL as usize);
    mv.drain().unwrap();
    assert_eq!(tier.make_exclusive(&ids).unwrap(), POOL as usize);
    assert_eq!(tier.exclusive_slots(), POOL);
    assert!(
        !tier.complete(),
        "the dropped shadows leave the pool's share on the drive"
    );
    assert_eq!(tier.refill_ram(&mut mv).unwrap(), POOL as usize);
    assert!(tier.complete(), "the refill makes the tier complete again");
    assert_eq!(
        tier.occupancy(),
        (u64::from(BASE + POOL), u64::from(64 - BASE - POOL), 0)
    );
    (tier, mv)
}

#[test]
fn exclusive_pool_slots_write_back_and_every_byte_lands() {
    for eager in [false, true] {
        let (_scratch, xt) = synth(if eager { "eager" } else { "lazy" });
        let (mut tier, mut mv) = boot_exclusive(&xt, eager);
        assert_eq!(
            check_all_slots(&tier, mv.queue(), &xt),
            64,
            "64 records, no shadow"
        );
        let mut g = Lcg(0x107);
        for s in 0..200u32 {
            assert_eq!(
                step(&mut tier, &mut mv, &xt, &g.routes(1 + (s % 3) as usize)),
                0,
                "step {s} (eager {eager}): a read landed on another record"
            );
            if s % 50 == 49 {
                mv.drain().unwrap();
                // the written-back records re-read from RAM identical; still no shadow
                assert_eq!(check_all_slots(&tier, mv.queue(), &xt), 64);
                assert!(tier.complete());
            }
        }
        let (_, moves) = tier.totals();
        assert!(moves.excl_writebacks > 0, "pool victims were written back");
        assert_eq!(
            moves.lent_drops, 0,
            "an exclusive victim is never dropped to a shadow"
        );
        assert_eq!(mv.counts.d2h, moves.demotions, "every write-back is a D2H");
        assert!(moves.excl_writebacks <= moves.demotions);
        // a reclaim never drops an exclusive slot
        mv.drain().unwrap();
        let ids: Vec<u32> = (BASE..BASE + POOL).collect();
        let held = ids
            .iter()
            .filter(|&&s| tier.vram_record(s).is_some())
            .count();
        assert_eq!(tier.reclaim(&ids), 0);
        assert_eq!(
            ids.iter()
                .filter(|&&s| tier.vram_record(s).is_some())
                .count(),
            held
        );
        assert!(
            (0..64u32).all(|i| tier.placement(i / EXPERTS, i % EXPERTS) != Some(Placement::Fetch))
        );
    }
}

#[test]
fn only_lent_slots_become_exclusive_and_off_keeps_the_shadows() {
    let (_scratch, xt) = synth("neg");
    // negative control: an unlent (registered) slot cannot be made exclusive
    let (mut t, _m) = setup(&xt, &params(0), false);
    assert!(t.make_exclusive(&[BASE]).is_err());
    assert!(t.make_exclusive(&[0]).is_err(), "a static slot is not lent");
    // `off` (FN76's inclusive pool): the full L2, the pool's records keep their shadows
    let (mut t, mut m) = setup(&xt, &params(0), false);
    let ids: Vec<u32> = (BASE..BASE + POOL).collect();
    t.lend(&ids).unwrap();
    m.after(&()).unwrap();
    t.fill_lent(&mut m).unwrap();
    m.drain().unwrap();
    assert!(t.complete());
    let (v, r, _) = t.occupancy();
    assert_eq!(
        (v, r),
        (u64::from(BASE + POOL), u64::from(64 - BASE - POOL))
    );
    assert_eq!(t.exclusive_slots(), 0);
    assert!(
        t.reclaim(&ids) > 0,
        "an inclusive pool slot drops to its shadow"
    );
}
