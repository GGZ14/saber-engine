//! LR-G: lazy KV growth in the tier (`KvTier::lazy`, `--fn-kv-lazy`). The reservation is only
//! COUNTING: slots are taken when a block is opened, from the same free lists, so a lazy tier
//! driven through the same appends / opens / commits / gathers issues the SAME copies and the
//! SAME placements as the eager tier (the copy-order law K1-K4 untouched) -- the bytes cannot
//! move. Growth refuses loudly (`Admission`, nothing changed) past the slab, and the elastic
//! want counts written blocks.

use saber_mem::kv_tier::{KvTier, KvTierError, KvTierPlan, LAZY_STEP_BLOCKS};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

const BB: u64 = 16 * 1024;

fn plan(ram_slots: u64) -> KvTierPlan {
    // 2 layers, 16-token blocks, a 64-token window, 64-token chunks, 4 sequences, 1,024 ctx
    KvTierPlan::new_admitted(2, 16, 64, 64, 4, 1024, 160 * BB, ram_slots * BB).expect("plan")
}

/// Drive an eager and a lazy tier through one random schedule and compare every output.
fn differential(seed: u64) -> (u64, u64) {
    let p = plan(600);
    let mut eager = KvTier::new(p.clone());
    let mut lazy = KvTier::new(p).lazy().expect("empty");
    let mut rng = Rng(seed | 1);
    // per slot: (live, prompt, final length, len)
    let mut st: Vec<Option<(u32, u32, u32)>> = vec![None; 4];
    let (mut grows, mut steps) = (0u64, 0u64);
    for _ in 0..400 {
        steps += 1;
        let slot = rng.below(3) as u32;
        match st[slot as usize] {
            None => {
                let prompt = 16 + rng.below(200) as u32;
                let fin = prompt + 1 + rng.below(250) as u32;
                // eager: the whole request (+ the spec overshoot + the engine's extra 64);
                // lazy: the prompt + a 32-token headroom
                eager.admit(slot, fin + 6 + 64).expect("eager admit");
                lazy.admit(slot, prompt + 32 + 6 + 64).expect("lazy admit");
                st[slot as usize] = Some((prompt, fin, 0));
            }
            Some((prompt, fin, len)) => {
                let n = if len < prompt {
                    (prompt - len).min(64)
                } else {
                    (1 + rng.below(5) as u32).min(fin - len)
                };
                let before = lazy.max_tokens(slot).expect("live");
                for t in [&mut eager, &mut lazy] {
                    t.protect(slot, len < prompt).expect("protect");
                }
                let a = eager.append(slot, n).map_err(|e| format!("{e}"));
                let b = lazy.append(slot, n).map_err(|e| format!("{e}"));
                assert_eq!(a, b, "append");
                let ce = eager.commit(slot).expect("commit");
                let cl = lazy.commit(slot).expect("commit");
                assert_eq!(format!("{ce:?}"), format!("{cl:?}"), "the flushes");
                // the step's speculative rows + the pages ahead
                let open = len + n + 5 + 16;
                eager.open(slot, open).expect("open");
                lazy.open(slot, open).expect("open");
                if lazy.max_tokens(slot).expect("live") > before {
                    grows += 1;
                }
                let len = len + n;
                if len >= prompt && rng.below(3) == 0 {
                    let groups: Vec<u32> = (0..4)
                        .map(|_| rng.below(u64::from(len.div_ceil(4))) as u32)
                        .collect();
                    for layer in 0..2 {
                        let ge = eager.gather(slot, layer, &groups).expect("gather");
                        let gl = lazy.gather(slot, layer, &groups).expect("gather");
                        assert_eq!(format!("{ge:?}"), format!("{gl:?}"), "the gather");
                    }
                }
                for layer in 0..2 {
                    for b in 0..eager.blocks_of(slot, layer) {
                        assert_eq!(
                            eager.place(slot, layer, b).expect("place"),
                            lazy.place(slot, layer, b).expect("place"),
                            "the placement of {slot}/{layer}/{b}"
                        );
                    }
                }
                if len >= fin {
                    eager.finish(slot).expect("finish");
                    lazy.finish(slot).expect("finish");
                    st[slot as usize] = None;
                } else {
                    st[slot as usize] = Some((prompt, fin, len));
                }
            }
        }
        eager.check().expect("eager law");
        lazy.check().expect("lazy law");
        assert!(lazy.hot_want() <= eager.hot_want(), "the lazy want");
    }
    (grows, steps)
}

#[test]
fn lrg_a_lazy_tier_issues_the_eager_tiers_copies_and_placements() {
    let mut grows = 0;
    for seed in 1..=8u64 {
        grows += differential(seed * 0x9E37_79B9).0;
    }
    assert!(grows > 100, "the schedule grows lazily ({grows})");
}

#[test]
fn lrg_growth_past_the_slab_refuses_loudly_and_changes_nothing() {
    // 2 layers x 64 blocks a sequence at max ctx: a 140-slot slab holds one 1,024 + a little
    let p = plan(140);
    let mut t = KvTier::new(p).lazy().expect("empty");
    t.admit(0, 1024).expect("a max-context sequence");
    t.admit(1, 64).expect("a short one");
    t.check().expect("law");
    let before = t.max_tokens(1);
    let err = t.grow(1, 160).expect_err("past the slab");
    assert!(matches!(err, KvTierError::Admission { .. }), "{err}");
    assert_eq!(t.max_tokens(1), before);
    t.check().expect("unchanged");
    // an append that would need the refused growth is refused the same way (no partial open)
    t.append(1, 64).expect("inside the reservation");
    let err = t.append(1, 64).expect_err("past the slab");
    assert!(matches!(err, KvTierError::Admission { .. }), "{err}");
    t.check().expect("unchanged");
    // the long one leaves: the growth fits
    t.finish(0).expect("finish");
    t.append(1, 64).expect("grows now");
    assert_eq!(t.max_tokens(1), Some(128));
    t.finish(1).expect("finish");
    t.check().expect("balanced");
}

#[test]
fn lrg_an_eager_tier_still_refuses_past_its_reservation() {
    let mut t = KvTier::new(plan(600));
    t.admit(0, 64).expect("admit");
    t.append(0, 64).expect("inside");
    assert!(matches!(t.append(0, 1), Err(KvTierError::State(_))));
    // and `grow` is the explicit form either way
    t.grow(0, 80).expect("grow");
    t.append(0, 1).expect("inside again");
    t.check().expect("law");
}

#[test]
fn lrg_the_lazy_want_counts_written_blocks_and_the_floor_one_steps_growth() {
    let p = plan(600);
    let mut e = KvTier::new(p.clone());
    let mut l = KvTier::new(p).lazy().expect("empty");
    // a 900-token reservation with 40 tokens written
    e.admit(0, 900).expect("admit");
    l.admit(0, 900).expect("admit");
    for t in [&mut e, &mut l] {
        t.append(0, 40).expect("append");
    }
    let cb = 64 / 16 + 1;
    // eager: every reserved block + 1; lazy: the 3 opened blocks + 1
    assert_eq!(e.hot_want(), 2 * ((900 / 16 + 1 + 1) + cb));
    assert_eq!(l.hot_want(), 2 * (3 + 1 + cb));
    // the floor: the window's blocks either way at 900; a short lazy reservation gets the step's
    // growth beside it
    assert_eq!(e.hot_floor(), l.hot_floor());
    let mut e = KvTier::new(plan(600));
    let mut l = KvTier::new(plan(600)).lazy().expect("empty");
    e.admit(0, 32).expect("admit");
    l.admit(0, 32).expect("admit");
    assert_eq!(e.hot_floor(), 2 * (2 + cb));
    assert_eq!(
        l.hot_floor(),
        2 * (u64::from((2 + LAZY_STEP_BLOCKS).min(64 / 16 + 1)) + cb)
    );
}
