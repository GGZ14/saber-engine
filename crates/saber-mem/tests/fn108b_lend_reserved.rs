//! FN108b: a LEND must not hand a layer more slots than it has RAM records left to fill them.
//!
//! The served regression (FN108b, LEDGER): FN108's fp8 HC mixers + FN107's exclusive pool left
//! rank 1 with 2 of its 820 pool slots EMPTY after the boot fill -- earlier lends (the arena top,
//! the sequence pools, the vision region) were still empty when the pool was lent, and `lend`
//! picked each pool slot's layer from the same RAM records those empty slots will consume, so one
//! layer got more lent slots than RAM records. `make_exclusive` then dropped 818 shadows, the RAM
//! refill read 818 drive records and 2 stayed on the drive: the tier was INCOMPLETE for the whole
//! run and every decode / verify step took the demand path (+5 ms a b1 verify step). The law:
//! `lend` skips, per layer, as many candidates as that layer already holds EMPTY lent slots.
//! NEGATIVE CONTROL: the two-call lend below leaves slots empty without the reservation (measured
//! on the unfixed tier: the assert on `fill_lent` fails).

use saber_mem::device_mover::{DeviceMover, FakeQueue};
use saber_mem::dio::OpenMode;
use saber_mem::expert_tier::{ExpertTier, TierParams};
use saber_mem::xtier::{write_synthetic, SyntheticTier, XtierFile};
use saber_test_scratch::ScratchDir;

const LAYERS: u32 = 4;
const EXPERTS: u32 = 16;
const BASE: u32 = 24;
const POOL: u32 = 16;
const VRAM_BASE: u64 = 0x7f00_0000_0000;
const RAM_BASE: u64 = 0x5500_0000_0000;

fn synth(tag: &str) -> (ScratchDir, XtierFile) {
    let d = ScratchDir::new(&format!("saber-fn108b-{tag}"));
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

#[test]
fn a_second_lend_leaves_no_lent_slot_empty_and_the_exclusive_tier_is_complete() {
    let (_scratch, xt) = synth("lend");
    let p = TierParams {
        vram_slots: BASE,
        ram_slots: 64 - BASE - POOL,
        min_layer_slots: 2,
        max_inserts: 2,
        read_through: 1.0,
        lrfu_lambda: 0.98,
        ..TierParams::default()
    };
    let mut tier = ExpertTier::new(xt.header(), p.clone(), None, VRAM_BASE, RAM_BASE).unwrap();
    let s = xt.header().stride;
    let addrs: Vec<u64> = (0..u64::from(POOL))
        .map(|i| VRAM_BASE + (u64::from(BASE) + i) * s)
        .collect();
    assert_eq!(tier.register_lendable(&addrs).unwrap(), BASE);
    let mut q = FakeQueue::new(s, BASE + POOL, p.ram_slots);
    q.warm(&xt, &tier.warm_plan()).unwrap();
    let mut mv = DeviceMover::new(&xt, q, 4);
    let (_, ram, fetch) = tier.occupancy();
    assert_eq!((ram, fetch), (u64::from(64 - BASE - POOL), u64::from(POOL)));
    // two owners lend before any fill (the arena top / the pool at the served boot)
    let a: Vec<u32> = (BASE..BASE + POOL / 2).collect();
    let b: Vec<u32> = (BASE + POOL / 2..BASE + POOL).collect();
    assert_eq!(tier.lend(&a).unwrap(), a.len());
    assert_eq!(tier.lend(&b).unwrap(), b.len());
    mv.after(&()).unwrap();
    assert_eq!(
        tier.fill_lent(&mut mv).unwrap(),
        POOL as usize,
        "every lent slot finds a RAM record of its layer"
    );
    assert_eq!(tier.lent_empty(), 0);
    mv.drain().unwrap();
    let ids: Vec<u32> = (BASE..BASE + POOL).collect();
    assert_eq!(tier.make_exclusive(&ids).unwrap(), POOL as usize);
    assert_eq!(tier.refill_ram(&mut mv).unwrap(), POOL as usize);
    assert!(tier.complete(), "no record left on the drive");
    tier.check_invariants().unwrap();
}
