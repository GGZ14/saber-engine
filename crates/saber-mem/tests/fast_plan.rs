//! FN59b: the cheap victim selection (`ExpertTier::set_fast_plan`) is placement-identical to the
//! full LRFU / LRU / policy sort -- two tiers in lockstep over seeded skewed traffic, every
//! step's placements, move counters and copy counts compared, for each victim law the engine can
//! arm (LRU, LRFU 0.98, LRFU + admission margin / window, reuse, lending, wide caps).

use saber_mem::expert_tier::{ExpertTier, TierParams, VictimKind, VictimPolicy};
use saber_mem::route_trace::{replay_header, CountMover};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// A skewed expert id in `0..n` (a few hot experts, a long tail).
    fn skewed(&mut self, n: u32) -> u32 {
        let a = (self.next() % u64::from(n)) as u32;
        let b = (self.next() % u64::from(n)) as u32;
        a.min(b)
    }
}

const LAYERS: u32 = 6;
const PER_RANK: u32 = 64;

fn lockstep(p: &TierParams, seed: u64, steps: u32, rows: &[u32]) {
    let h = replay_header(0, LAYERS, PER_RANK, 4096);
    let mut a = ExpertTier::new(&h, p.clone(), None, 0, 1 << 40).unwrap();
    let mut b = ExpertTier::new(&h, p.clone(), None, 0, 1 << 40).unwrap();
    b.set_fast_plan(true);
    let (mut ma, mut mb) = (CountMover::default(), CountMover::default());
    let mut rng = Rng(seed | 1);
    let mut moved = 0u64;
    for s in 0..steps {
        let r = rows[s as usize % rows.len()];
        a.begin_step();
        b.begin_step();
        for l in 0..LAYERS {
            let routed: Vec<u32> = (0..r * 8).map(|_| rng.skewed(PER_RANK)).collect();
            let ca = a.resolve(l, &routed, &mut ma).unwrap();
            let cb = b.resolve(l, &routed, &mut mb).unwrap();
            assert_eq!(ca, cb, "seed {seed} step {s} layer {l}");
        }
        a.refill(&mut ma).unwrap();
        b.refill(&mut mb).unwrap();
        let ra = a.end_step(&mut ma).unwrap();
        let rb = b.end_step(&mut mb).unwrap();
        assert_eq!(ra.moves, rb.moves, "seed {seed} step {s}: moves");
        assert!(
            a.placements() == b.placements(),
            "seed {seed} step {s}: placements"
        );
        moved += ra.moves.inserts;
    }
    assert_eq!((ma.h2d, ma.d2h), (mb.h2d, mb.d2h));
    assert!(moved > 0, "the traffic must promote (seed {seed})");
}

fn base() -> TierParams {
    TierParams {
        vram_slots: LAYERS * 40,
        ram_slots: LAYERS * 24,
        max_inserts: 2,
        lrfu_lambda: 0.98,
        min_layer_slots: 8,
        ..TierParams::default()
    }
}

#[test]
fn fast_plan_matches_lrfu_served_law() {
    for seed in 1..6 {
        lockstep(&base(), seed, 400, &[1, 1, 2, 8, 4, 1]);
    }
}

#[test]
fn fast_plan_matches_lru_and_wide_caps() {
    let mut p = base();
    p.lrfu_lambda = 0.0;
    for seed in 1..4 {
        lockstep(&p, seed, 300, &[1, 8]);
    }
    let mut p = base();
    p.max_inserts = 64; // every miss a candidate: the selection keeps many keys
    p.read_through = 1.0;
    for seed in 1..4 {
        lockstep(&p, seed, 300, &[1, 2, 8]);
    }
}

#[test]
fn fast_plan_matches_admission_and_policies() {
    let mut p = base();
    p.admit_margin = 0.5;
    p.admit_uses = 2;
    lockstep(&p, 7, 300, &[1, 8]);
    let mut p = base();
    p.admit_window = 16;
    lockstep(&p, 8, 300, &[1, 8]);
    let mut p = base();
    p.policy = VictimPolicy {
        kind: VictimKind::Reuse,
        ..VictimPolicy::default()
    };
    lockstep(&p, 9, 300, &[1, 4]);
    let mut p = base();
    p.lend_boot = 30;
    lockstep(&p, 10, 300, &[1, 8]);
    let mut p = base();
    p.second_chance = 4;
    lockstep(&p, 11, 300, &[1, 8]);
}
