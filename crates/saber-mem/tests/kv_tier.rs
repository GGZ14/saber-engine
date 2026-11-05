//! FN45a: the tiered KV cache's laws on the host -- the placement byte law under random
//! prefill / decode / gather / evict / reclaim sequences, the window never evicted, the gather
//! lists exact, the ledger balance and the boot refusals, the replay's limits.

use std::collections::BTreeSet;

use saber_mem::device_mover::CopyOp;
use saber_mem::kv_tier::{
    prefix_keys, replay, Commit, KvTier, KvTierError, KvTierPlan, Place, GROUP_TOKENS,
    INDEX_TOKEN_BYTES, KV_HOT, KV_INDEX, KV_RAM, KV_SSD, KV_TOKEN_LAYER_BYTES, QSA_LAYERS,
};
use saber_mem::ledger::{Ledger, Pool};

const GIB: u64 = 1 << 30;

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

/// A host model of the two slabs: `tb` bytes per token (a model of the 1,024), copies applied in
/// issue order (the in-order stream; the step's compute waits for every issued copy).
struct Model {
    tb: usize,
    bt: usize,
    vram: Vec<Vec<u8>>,
    ram: Vec<Vec<u8>>,
    ssd: Vec<Vec<u8>>,
    /// The flat reference: `[seq][layer]` -> every written token's bytes.
    flat: Vec<Vec<Vec<u8>>>,
    gen: Vec<u64>,
    /// The negative control: drop every flush (a tier that forgot write-through).
    drop_flush: bool,
}

impl Model {
    fn new(p: &KvTierPlan, tb: usize) -> Model {
        let bt = p.block_tokens as usize;
        Model {
            tb,
            bt,
            vram: vec![vec![0xAA; bt * tb]; p.hot_slots as usize],
            ram: vec![vec![0x55; bt * tb]; p.ram_slots as usize],
            ssd: vec![vec![0x33; bt * tb]; p.ssd_slots as usize],
            flat: vec![vec![Vec::new(); p.layers as usize]; p.max_seqs as usize],
            gen: vec![0; p.max_seqs as usize],
            drop_flush: false,
        }
    }
    fn apply(&mut self, ops: &[CopyOp]) {
        for op in ops {
            match *op {
                CopyOp::H2d { ram, vram } => {
                    let r = self.ram[ram as usize].clone();
                    self.vram[vram as usize] = r;
                }
                CopyOp::D2h { .. } if self.drop_flush => {}
                CopyOp::D2h { vram, ram } => {
                    let v = self.vram[vram as usize].clone();
                    self.ram[ram as usize] = v;
                }
                CopyOp::Landed { .. } => {}
            }
        }
    }
    /// A commit: the flushes, then (after they landed) the SSD writes.
    fn commit(&mut self, c: &Commit) {
        self.apply(&c.flush);
        for &(r, d) in &c.ssd_writes {
            self.ssd[d as usize] = self.ram[r as usize].clone();
        }
    }
    /// A gather: the SSD reads land in RAM, then the copies.
    fn gather(&mut self, g: &saber_mem::kv_tier::Gather) {
        for &(d, r) in &g.ssd_reads {
            self.ram[r as usize] = self.ssd[d as usize].clone();
        }
        self.apply(&g.ops);
    }
    /// The step's compute writes rows `old..new` of `seq` into their (hot) blocks.
    fn write(&mut self, t: &KvTier, seq: u32, old: u32, new: u32) {
        for layer in 0..t.plan().layers {
            for pos in old..new {
                let blk = pos as usize / self.bt;
                let off = (pos as usize % self.bt) * self.tb;
                let mut bytes = vec![0u8; self.tb];
                for (i, b) in bytes.iter_mut().enumerate() {
                    *b = (self.gen[seq as usize]
                        .wrapping_mul(31)
                        .wrapping_add(u64::from(seq) * 7 + u64::from(layer) * 13)
                        .wrapping_add(u64::from(pos) * 3 + i as u64)
                        % 251) as u8;
                }
                let Place::Vram(v) = t.place(seq, layer, blk as u32).unwrap() else {
                    panic!("seq {seq} layer {layer} block {blk}: a written block is not hot");
                };
                self.vram[v as usize][off..off + self.tb].copy_from_slice(&bytes);
                self.flat[seq as usize][layer as usize].extend_from_slice(&bytes);
            }
        }
    }
    /// The byte law: every block of every live sequence, read through its placement, equals the
    /// flat reference (the written tokens of a partial block).
    fn law(&self, t: &KvTier) {
        for seq in 0..t.plan().max_seqs {
            let len = t.len(seq) as usize;
            if len == 0 {
                continue;
            }
            for layer in 0..t.plan().layers {
                let flat = &self.flat[seq as usize][layer as usize];
                assert_eq!(flat.len(), len * self.tb);
                for blk in 0..len.div_ceil(self.bt) {
                    let n = (len - blk * self.bt).min(self.bt) * self.tb;
                    let got = match t.place(seq, layer, blk as u32).unwrap() {
                        Place::Vram(v) => &self.vram[v as usize][..n],
                        Place::Ram(r) => &self.ram[r as usize][..n],
                        Place::Ssd(d) => &self.ssd[d as usize][..n],
                    };
                    let want = &flat[blk * self.bt * self.tb..blk * self.bt * self.tb + n];
                    assert_eq!(got, want, "seq {seq} layer {layer} block {blk}");
                }
            }
        }
    }
}

fn small_plan(hot_extra: u64, max_seqs: u32) -> KvTierPlan {
    // 3 layers, 16-token blocks, 32-token window, 64-token chunks, 512-token context, 2 tiers
    let bb = 16 * KV_TOKEN_LAYER_BYTES;
    let floor = 3 * (u64::from(max_seqs) * 3 + 5);
    KvTierPlan::new(
        3,
        16,
        32,
        64,
        max_seqs,
        512,
        (floor + hot_extra) * bb,
        u64::from(max_seqs) * 3 * 32 * bb,
        0,
    )
    .unwrap()
}

/// The same with an SSD tier: RAM at its floor + `ram_extra` (the floor = the hot floor + one
/// widest gather, which at 512 tokens is the 32-block context).
fn ssd_plan(hot_extra: u64, ram_extra: u64, max_seqs: u32) -> KvTierPlan {
    let bb = 16 * KV_TOKEN_LAYER_BYTES;
    let floor = 3 * (u64::from(max_seqs) * 3 + 5);
    KvTierPlan::new(
        3,
        16,
        32,
        64,
        max_seqs,
        512,
        (floor + hot_extra) * bb,
        (floor + 32 + ram_extra) * bb,
        u64::from(max_seqs) * 3 * 32 * bb,
    )
    .unwrap()
}

/// The random driver: prefill chunks, decode steps with random selections (QSA's shape: a few
/// "sink" groups, a recent band, random far groups), finishes and re-admissions.
fn drive(plan: KvTierPlan, seed: u64, rounds: usize) -> (KvTier, u64) {
    drive_ctl(plan, seed, rounds, false)
}

fn drive_ctl(plan: KvTierPlan, seed: u64, rounds: usize, drop_flush: bool) -> (KvTier, u64) {
    let mut t = KvTier::new(plan.clone());
    let mut m = Model::new(&plan, 8);
    m.drop_flush = drop_flush;
    let mut rng = Rng(seed);
    let mut checked = 0u64;
    for _ in 0..rounds {
        let seq = rng.below(u64::from(plan.max_seqs)) as u32;
        let len = t.len(seq);
        let live = len > 0;
        match rng.below(10) {
            0 if live => {
                t.finish(seq).unwrap();
                m.flat[seq as usize].iter_mut().for_each(Vec::clear);
                m.gen[seq as usize] += 1;
            }
            _ if !live => {
                // admit + prefill in chunks
                let max = 128 + rng.below(384) as u32;
                if t.admit(seq, max).is_err() {
                    continue;
                }
                let p0 = 1 + rng.below(u64::from(max / 2)) as u32;
                let mut done = 0;
                while done < p0 {
                    let n = plan.chunk_tokens.min(p0 - done);
                    t.append(seq, n).unwrap();
                    m.write(&t, seq, done, done + n);
                    let c = t.commit(seq).unwrap();
                    m.commit(&c);
                    done += n;
                }
            }
            _ => {
                let max = t.len(seq);
                let step = 1 + rng.below(4) as u32;
                if t.append(seq, step).is_err() {
                    continue; // at its admitted maximum
                }
                m.write(&t, seq, max, max + step);
                let len = t.len(seq);
                let visible = len / GROUP_TOKENS;
                for layer in 0..plan.layers {
                    let mut groups: Vec<u32> = Vec::new();
                    if visible > 0 {
                        for g in 0..visible.min(2) {
                            groups.push(g);
                        }
                        for _ in 0..(1 + rng.below(24)) {
                            groups.push(rng.below(u64::from(visible)) as u32);
                        }
                        for _ in 0..4 {
                            groups.push(visible - 1 - rng.below(u64::from(visible.min(8))) as u32);
                        }
                    }
                    let before: Vec<Place> = (0..len.div_ceil(plan.block_tokens))
                        .map(|b| t.place(seq, layer, b).unwrap())
                        .collect();
                    let g = t.gather(seq, layer, &groups).unwrap();
                    gather_is_exact(&t, seq, layer, &groups, &before, &g);
                    m.gather(&g);
                    // the attention reads every selected group through its placement
                    for &gr in &groups {
                        let b = gr / (plan.block_tokens / GROUP_TOKENS);
                        let p = t.place(seq, layer, b).unwrap();
                        assert!(
                            matches!(p, Place::Vram(_)) || g.uva.contains(&b),
                            "a selected block is neither hot nor a counted UVA read"
                        );
                    }
                    m.law(&t);
                    checked += 1;
                }
                let c = t.commit(seq).unwrap();
                assert!(c.flush.iter().all(|o| matches!(o, CopyOp::D2h { .. })));
                m.commit(&c);
            }
        }
        t.check().unwrap();
        m.law(&t);
    }
    (t, checked)
}

/// The gather list's law against the state before it.
fn gather_is_exact(
    t: &KvTier,
    seq: u32,
    layer: u32,
    groups: &[u32],
    before: &[Place],
    g: &saber_mem::kv_tier::Gather,
) {
    let gpb = t.plan().block_tokens / GROUP_TOKENS;
    let sel: BTreeSet<u32> = groups.iter().map(|g| g / gpb).collect();
    let was_ram: BTreeSet<u32> = sel
        .iter()
        .copied()
        .filter(|&b| !matches!(before[b as usize], Place::Vram(_)))
        .collect();
    // exactly the blocks that were on the SSD only are read first, each into its RAM slot
    let was_ssd: BTreeSet<u32> = sel
        .iter()
        .copied()
        .filter(|&b| matches!(before[b as usize], Place::Ssd(_)))
        .collect();
    assert_eq!(
        g.ssd_reads.len(),
        was_ssd.len(),
        "one SSD read per SSD-only selected block"
    );
    for &(d, r) in &g.ssd_reads {
        let b = (0..before.len() as u32)
            .find(|&b| before[b as usize] == Place::Ssd(d))
            .expect("an SSD read of an SSD-only block");
        assert!(was_ssd.contains(&b));
        let now = t.place(seq, layer, b).unwrap();
        assert!(now == Place::Ram(r) || matches!(now, Place::Vram(_)));
    }
    let ram_of = |b: u32| -> Option<u32> {
        match before[b as usize] {
            Place::Ram(r) => Some(r),
            Place::Ssd(d) => g.ssd_reads.iter().find(|x| x.0 == d).map(|x| x.1),
            Place::Vram(_) => None,
        }
    };
    assert_eq!(
        g.hits as usize,
        sel.len() - was_ram.len(),
        "hits = the selected hot blocks"
    );
    assert_eq!(
        g.copied as usize + g.uva.len(),
        was_ram.len(),
        "every absent block copied or UVA"
    );
    let mut copied = BTreeSet::new();
    for op in &g.ops {
        match *op {
            CopyOp::H2d { ram, vram } => {
                // the H2D reads exactly the absent block's RAM slot and lands where it is placed
                let b = was_ram
                    .iter()
                    .copied()
                    .find(|&b| ram_of(b) == Some(ram))
                    .expect("an H2D reads an absent block's RAM slot");
                assert!(was_ram.contains(&b), "an H2D for a block nobody selected");
                assert_eq!(t.place(seq, layer, b).unwrap(), Place::Vram(vram));
                assert!(copied.insert(b), "one H2D per block");
            }
            CopyOp::Landed { layer: l } => {
                assert_eq!(l, layer);
                assert_eq!(op, g.ops.last().unwrap(), "Landed closes the list");
            }
            CopyOp::D2h { .. } => panic!("a gather never flushes"),
        }
    }
    assert_eq!(copied.len(), g.copied as usize);
    assert_eq!(g.ops.is_empty(), g.copied == 0);
    for b in &g.uva {
        assert!(matches!(t.place(seq, layer, *b).unwrap(), Place::Ram(_)));
    }
    let _ = ram_of;
}

#[test]
fn placement_byte_law_under_random_prefill_decode_evict_reclaim() {
    for (seed, extra) in [(1u64, 0u64), (2, 12), (3, 60), (4, 400)] {
        let (t, checked) = drive(small_plan(extra, 4), seed, 3000);
        let c = t.counts();
        assert!(checked > 1000, "seed {seed}: {checked} gathers checked");
        assert!(c.d2h > 0, "seed {seed}: {c:?}");
        if extra <= 12 {
            assert!(
                c.evictions > 0 && c.h2d > 0,
                "a tight pool evicts and gathers: {c:?}"
            );
        }
    }
}

/// Append `n` tokens in plan-sized chunks, committing each.
fn fill(t: &mut KvTier, seq: u32, n: u32) {
    let c = t.plan().chunk_tokens;
    let mut done = 0;
    while done < n {
        let k = c.min(n - done);
        t.append(seq, k).unwrap();
        t.commit(seq).unwrap();
        done += k;
    }
}

#[test]
fn three_tier_placement_byte_law_and_ssd_reads() {
    // RAM at its floor: the warm LRU drops copies and the gathers read them back off the SSD
    for (seed, hot, ram) in [(11u64, 0u64, 0u64), (12, 12, 0), (13, 0, 40), (14, 60, 200)] {
        let (t, checked) = drive(ssd_plan(hot, ram, 4), seed, 3000);
        let c = t.counts();
        assert!(checked > 1000, "seed {seed}: {checked}");
        assert_eq!(
            c.ssd_writes, c.d2h,
            "seed {seed}: write-through to the lowest tier"
        );
        if ram == 0 {
            assert!(c.ram_drops > 0 && c.ssd_reads > 0, "seed {seed}: {c:?}");
        }
    }
}

#[test]
fn the_three_tier_byte_law_catches_a_lost_ssd_write() {
    let r = std::panic::catch_unwind(|| {
        let plan = ssd_plan(0, 0, 4);
        let mut t = KvTier::new(plan.clone());
        let mut m = Model::new(&plan, 8);
        t.admit(0, 512).unwrap();
        let mut done = 0;
        while done < 448 {
            t.append(0, 64).unwrap();
            m.write(&t, 0, done, done + 64);
            let mut c = t.commit(0).unwrap();
            c.ssd_writes.clear(); // the control: the drive never got the bytes
            m.commit(&c);
            done += 64;
        }
        for layer in 0..3 {
            let all: Vec<u32> = (0..t.len(0) / 4).collect();
            let g = t.gather(0, layer, &all).unwrap();
            m.gather(&g);
        }
        m.law(&t);
    });
    assert!(r.is_err(), "the byte law did not see the lost SSD write");
}

#[test]
fn the_prefix_keys_chain_so_a_longer_conversation_extends_its_turns() {
    let ids: Vec<u32> = (0..10_000).map(|i| (i * 7919) % 151_936).collect();
    let a = prefix_keys(b"pack-x", &ids[..4096], 2048);
    let b = prefix_keys(b"pack-x", &ids, 2048);
    assert_eq!(a.len(), 2);
    assert_eq!(b.len(), 4, "only complete chunks are keys");
    assert_eq!(
        &b[..2],
        &a[..],
        "a turn's keys are a prefix of the next turn's"
    );
    assert_eq!(b[3].0, 8192);
    let other = prefix_keys(b"pack-y", &ids, 2048);
    assert!(
        other.iter().zip(&b).all(|(x, y)| x.1 != y.1),
        "the model is in every key"
    );
    let mut ids2 = ids.clone();
    ids2[3000] ^= 1;
    let c = prefix_keys(b"pack-x", &ids2, 2048);
    assert_eq!(c[0], b[0]);
    assert!(
        c[1..].iter().zip(&b[1..]).all(|(x, y)| x.1 != y.1),
        "a changed token forks the chain"
    );
}

#[test]
fn the_byte_law_catches_a_tier_without_write_through() {
    // the negative control: the same driver with every flush dropped must fail the law
    let r = std::panic::catch_unwind(|| drive_ctl(small_plan(0, 4), 1, 3000, true));
    assert!(
        r.is_err(),
        "the byte law did not see the missing write-through"
    );
}

#[test]
fn a_starved_pool_falls_back_to_counted_uva_reads_and_keeps_the_window() {
    // 1 layer, 1 sequence: the floor = 3 window blocks + 5 append blocks = 8 slots
    let bb = 16 * KV_TOKEN_LAYER_BYTES;
    let plan = KvTierPlan::new(1, 16, 32, 64, 1, 512, 8 * bb, 32 * bb, 0).unwrap();
    let mut t = KvTier::new(plan);
    t.admit(0, 512).unwrap();
    fill(&mut t, 0, 400);
    t.append(0, 1).unwrap();
    let all: Vec<u32> = (0..t.len(0) / 4).collect();
    let g = t.gather(0, 0, &all).unwrap();
    // 401 tokens: blocks 0..=25, the window 23..=25 (3 slots); the selection = blocks 0..=24
    assert_eq!(g.hits + g.copied + g.uva.len() as u32, 25);
    assert_eq!(
        g.hits + g.copied,
        2 + 5,
        "the window's two + every other slot"
    );
    assert_eq!(g.uva.len(), 18);
    assert!(t.counts().uva == 18);
    t.commit(0).unwrap();
    t.check().unwrap();
    assert!(
        t.append(0, 65).is_err(),
        "an append wider than the chunk is refused"
    );
}

#[test]
fn the_window_is_never_evicted() {
    let plan = small_plan(0, 2);
    let mut t = KvTier::new(plan.clone());
    t.admit(0, 512).unwrap();
    t.admit(1, 512).unwrap();
    for seq in 0..2 {
        fill(&mut t, seq, 200);
    }
    // hammer sequence 1 with whole-context selections: sequence 0's window must stay hot
    for step in 0..100 {
        t.append(1, 1).unwrap();
        let vis = t.len(1) / 4;
        let all: Vec<u32> = (0..vis).collect();
        for layer in 0..plan.layers {
            t.gather(1, layer, &all).unwrap();
        }
        t.commit(1).unwrap();
        t.check().unwrap();
        for seq in 0..2 {
            let len = t.len(seq);
            let ws = len.saturating_sub(plan.window_tokens) / plan.block_tokens;
            for layer in 0..plan.layers {
                for b in ws..len.div_ceil(plan.block_tokens) {
                    assert!(
                        matches!(t.place(seq, layer, b).unwrap(), Place::Vram(_)),
                        "step {step}: seq {seq} layer {layer} window block {b} evicted"
                    );
                }
            }
        }
    }
}

#[test]
fn a_sequence_at_its_quota_recycles_its_own_blocks_and_a_reclaim_returns_every_slot() {
    let plan = small_plan(30, 2);
    assert_eq!(plan.quota, 30 / (3 * 2));
    let mut t = KvTier::new(plan.clone());
    let (v0, r0) = (t.vram_free(), t.ram_free());
    t.admit(0, 512).unwrap();
    t.admit(1, 512).unwrap();
    fill(&mut t, 1, 300);
    t.append(0, 64).unwrap();
    let flushes = t.commit(0).unwrap();
    assert!(flushes.ssd_writes.is_empty(), "two tiers write no SSD");
    assert_eq!(
        flushes.flush.len(),
        3 * 4,
        "one flush per completed block per layer"
    );
    fill(&mut t, 0, 336);
    // sequence 1 keeps what it holds now while sequence 0 (at its quota) walks its context
    let hot1 = |t: &KvTier| -> Vec<Place> {
        (0..3)
            .flat_map(|l| (0..300u32.div_ceil(16)).map(move |b| (l, b)))
            .map(|(l, b)| t.place(1, l, b).unwrap())
            .collect()
    };
    let mut rng = Rng(7);
    for _ in 0..30 {
        t.append(0, 1).unwrap();
        for layer in 0..3 {
            let vis = t.len(0) / 4;
            let g: Vec<u32> = (0..4).map(|_| rng.below(u64::from(vis)) as u32).collect();
            let at_quota = t.hot_outside(0, layer).unwrap() >= plan.quota;
            let before = hot1(&t);
            t.gather(0, layer, &g).unwrap();
            if at_quota {
                assert_eq!(
                    hot1(&t),
                    before,
                    "a sequence at its quota stole another's slot"
                );
            }
        }
        t.commit(0).unwrap();
        t.check().unwrap();
    }
    t.finish(0).unwrap();
    t.finish(1).unwrap();
    t.check().unwrap();
    assert_eq!((t.vram_free(), t.ram_free()), (v0, r0));
}

#[test]
fn a_full_quota_hits_every_reselected_block_and_a_zero_quota_copies_it_again() {
    let mut sel = Vec::new();
    // 64 rows over a 1,024-token prompt: the same 40 far groups every row
    let far: Vec<i32> = (0..40).map(|i| i * 5).collect();
    for _ in 0..64 {
        sel.push(far.clone());
    }
    let run = |hot_extra: u64| {
        let bb = 16 * KV_TOKEN_LAYER_BYTES;
        let plan = KvTierPlan::new(
            2,
            16,
            64,
            256,
            1,
            2048,
            (2 * (5 + 17) + hot_extra) * bb,
            2 * 128 * bb,
            0,
        )
        .unwrap();
        let mut t = KvTier::new(plan);
        replay(&mut t, 0, 1024, 64, 1, 1, |_, r| &sel[r as usize][..]).unwrap()
    };
    let full = run(2 * 64);
    let zero = run(0);
    // groups 5 apart: 40 distinct 16-token blocks, every step selects them all
    assert_eq!(full.selected, 2 * 40 * 64);
    assert_eq!(full.steps, 64);
    // with room the prefill leaves every block hot: nothing is ever copied
    assert_eq!((full.copied, full.uva), (0, 0));
    assert!((full.hit() - 1.0).abs() < 1e-12);
    // a floor-sized pool (22 slots a layer beside nothing) cannot hold 2 x 40: each layer keeps
    // what its first gather placed and reads the rest through UVA -- no cross-layer thrash
    assert!(zero.copied > 0 && zero.copied <= 2 * 40, "{zero:?}");
    assert!(zero.uva > 0 && zero.hits > 0, "{zero:?}");
    assert_eq!(zero.hits + zero.copied + zero.uva, zero.selected);
}

#[test]
fn the_plan_refuses_a_boot_whose_contexts_do_not_fit_ram_and_the_ledger_balances() {
    let bb = 64 * KV_TOKEN_LAYER_BYTES;
    // 8 x 64k at 13 layers = 6.5 GiB a rank of kv.ram
    let p = KvTierPlan::new(QSA_LAYERS, 64, 1024, 2048, 8, 65_536, 2 * GIB, 7 * GIB, 0).unwrap();
    assert_eq!(p.block_bytes(), bb);
    assert!(p.ram_bytes() >= 8 * 13 * 65_536 * KV_TOKEN_LAYER_BYTES);
    // 8 x 250k does not fit 20 GiB
    let e =
        KvTierPlan::new(QSA_LAYERS, 64, 1024, 2048, 8, 256_000, 2 * GIB, 20 * GIB, 0).unwrap_err();
    let KvTierError::Short { line, need, .. } = e else {
        panic!("{e}")
    };
    assert_eq!(line, KV_RAM);
    assert_eq!(need, 8 * 13 * 4000 * bb);
    // a hot pool below every window + one open append refuses
    let e =
        KvTierPlan::new(QSA_LAYERS, 64, 1024, 2048, 8, 65_536, 64 << 20, 7 * GIB, 0).unwrap_err();
    assert!(matches!(e, KvTierError::Short { line: KV_HOT, .. }), "{e}");
    // geometry
    assert!(KvTierPlan::new(13, 24, 1024, 2048, 8, 65_536, GIB, 7 * GIB, 0).is_err());
    assert!(KvTierPlan::new(13, 64, 1000, 2048, 8, 65_536, GIB, 7 * GIB, 0).is_err());

    let mut l = Ledger::new();
    l.declare(Pool::Vram(0), 32 * GIB).unwrap();
    l.declare(Pool::HostPinned, 30 * GIB).unwrap();
    p.declare(&mut l, 0).unwrap();
    assert_eq!(l.used(Pool::Vram(0)), p.hot_bytes() + p.index_bytes());
    assert_eq!(p.index_bytes(), 8 * 65_536 * INDEX_TOKEN_BYTES);
    assert_eq!(l.used(Pool::HostPinned), p.ram_bytes());
    assert_eq!(l.release(Pool::Vram(0), KV_HOT).unwrap(), p.hot_bytes());
    assert_eq!(l.release(Pool::Vram(0), KV_INDEX).unwrap(), p.index_bytes());
    assert_eq!(l.release(Pool::HostPinned, KV_RAM).unwrap(), p.ram_bytes());
    // a pinned pool too small for kv.ram refuses by name
    let mut l = Ledger::new();
    l.declare(Pool::Vram(0), 32 * GIB).unwrap();
    l.declare(Pool::HostPinned, GIB).unwrap();
    assert!(matches!(p.declare(&mut l, 0), Err(KvTierError::Ledger(_))));
    assert!(p.boot_line().contains("kv.ram"));
    // three tiers: 8 x 250k on the drive (26.6 GB a rank), RAM at a warm budget
    let p3 = KvTierPlan::new(
        QSA_LAYERS,
        16,
        1024,
        2048,
        8,
        256_000,
        3 * GIB,
        8 * GIB,
        26 * GIB,
    )
    .unwrap();
    assert!(p3.has_ssd());
    assert!(u64::from(p3.ssd_slots) * p3.block_bytes() >= 8 * 13 * 256_000 * 1024);
    let e = KvTierPlan::new(
        QSA_LAYERS,
        16,
        1024,
        2048,
        8,
        256_000,
        3 * GIB,
        8 * GIB,
        20 * GIB,
    )
    .unwrap_err();
    assert!(matches!(e, KvTierError::Short { line: KV_SSD, .. }), "{e}");
    let e = KvTierPlan::new(
        QSA_LAYERS,
        16,
        1024,
        2048,
        8,
        256_000,
        3 * GIB,
        64 << 20,
        26 * GIB,
    )
    .unwrap_err();
    assert!(
        matches!(e, KvTierError::Short { line: KV_RAM, .. }),
        "RAM below its floor: {e}"
    );
    let mut l = Ledger::new();
    l.declare(Pool::Vram(0), 32 * GIB).unwrap();
    l.declare(Pool::HostPinned, 30 * GIB).unwrap();
    l.declare(Pool::Nvme, 50 * GIB).unwrap();
    p3.declare(&mut l, 0).unwrap();
    assert_eq!(l.used(Pool::Nvme), p3.ssd_bytes());
    assert_eq!(l.used(Pool::HostPinned), p3.ram_bytes());
}

#[test]
fn admission_is_by_the_ram_reservation() {
    let bb = 16 * KV_TOKEN_LAYER_BYTES;
    // 4 x 512 tokens at 3 layers does not boot on room for 2 x 512
    let e = KvTierPlan::new(3, 16, 32, 64, 4, 512, 400 * bb, 2 * 3 * 32 * bb, 0).unwrap_err();
    assert!(matches!(e, KvTierError::Short { line: KV_RAM, .. }), "{e}");
    // a plan whose RAM was cut after the check (FN45b's mixed-length admission): the scheduler
    // admits by the reservation
    let mut plan = KvTierPlan::new(3, 16, 32, 64, 4, 256, 400 * bb, 4 * 3 * 16 * bb, 0).unwrap();
    plan.ram_slots = 2 * 3 * 16 + 3 * 8;
    let mut t = KvTier::new(plan);
    t.admit(0, 256).unwrap();
    t.admit(1, 256).unwrap();
    assert!(matches!(
        t.admit(2, 256),
        Err(KvTierError::Admission { want: 48, free: 24 })
    ));
    t.admit(2, 128).unwrap();
    assert!(t.admit(3, 1).is_err(), "the reservation is full");
    t.finish(1).unwrap();
    t.admit(3, 256).unwrap();
    assert!(t.admit(3, 16).is_err(), "already live");
    fill(&mut t, 3, 256);
    assert!(t.append(3, 1).is_err(), "past its admitted maximum");
    t.check().unwrap();
}

#[test]
fn the_tiered_runtime_classes_replace_the_flat_kv_lines() {
    use saber_mem::flashnext_budget::RuntimeParams;
    let rt = RuntimeParams::default();
    let p = KvTierPlan::new(QSA_LAYERS, 64, 1024, 2048, 8, 65_536, 2 * GIB, 7 * GIB, 0).unwrap();
    let flat = rt.classes();
    let tiered = rt.classes_tiered(&p);
    let names: Vec<&str> = tiered.iter().map(|c| c.name.as_str()).collect();
    assert!(!names.contains(&"fn.kv") && !names.contains(&"fn.indexer_keys"));
    assert_eq!(&names[..2], &[KV_HOT, KV_INDEX]);
    assert_eq!(tiered[0].bytes, p.hot_bytes());
    assert_eq!(tiered[1].bytes, p.index_bytes());
    assert_eq!(tiered.len(), flat.len());
    let rest = |cs: &[saber_mem::flashnext_budget::Class]| -> u64 {
        cs.iter()
            .filter(|c| {
                !c.name.starts_with("kv.") && c.name != "fn.kv" && c.name != "fn.indexer_keys"
            })
            .map(|c| c.bytes)
            .sum()
    };
    assert_eq!(rest(&flat), rest(&tiered), "every other class is untouched");
}

/// FN45b4: the PREFILL GATHER + protection driver: prompts prefill in chunks interleaved across
/// sequences (the serve plane's form), each chunk preceded by `gather_seq`; a sequence's first
/// decode step releases its protection; decode steps gather random selections. Laws: the byte law
/// every round, the bookkeeping law, a chunk reads every earlier block hot (or as a counted UVA
/// read), a prefill gather never evicts its own sequence's block, a decode gather never evicts a
/// protected sequence's block.
fn drive_prefill(plan: KvTierPlan, seed: u64, rounds: usize) -> (KvTier, u64, u64) {
    let mut t = KvTier::new(plan.clone());
    let mut m = Model::new(&plan, 8);
    let mut rng = Rng(seed);
    let ns = plan.max_seqs as usize;
    let mut target = vec![0u32; ns];
    let (mut chunks, mut uva) = (0u64, 0u64);
    let places = |t: &KvTier, s: u32| -> Vec<Place> {
        (0..plan.layers)
            .flat_map(|l| (0..t.blocks_of(s, l)).map(move |b| (l, b)))
            .map(|(l, b)| t.place(s, l, b).unwrap())
            .collect()
    };
    for _ in 0..rounds {
        let seq = rng.below(ns as u64) as u32;
        let live = t.is_live(seq);
        if live && rng.below(14) == 0 {
            t.finish(seq).unwrap();
            m.flat[seq as usize].iter_mut().for_each(Vec::clear);
            m.gen[seq as usize] += 1;
        } else if !live {
            let max = 128 + rng.below(384) as u32;
            if t.admit(seq, max).is_err() {
                continue;
            }
            t.protect(seq, true).unwrap();
            target[seq as usize] = 1 + rng.below(u64::from(max / 2)) as u32;
        } else if t.len(seq) < target[seq as usize] {
            // one prompt chunk: the gather, then the chunk's rows
            assert!(t.is_protected(seq));
            let before = places(&t, seq);
            let g = t.gather_seq(seq).unwrap();
            m.gather(&g);
            let after = places(&t, seq);
            for (i, (b, a)) in before.iter().zip(&after).enumerate() {
                if matches!(b, Place::Vram(_)) {
                    assert_eq!(a, b, "a prefill gather moved its own hot block {i}");
                }
            }
            for l in 0..plan.layers {
                for b in 0..t.blocks_of(seq, l) {
                    if !matches!(t.place(seq, l, b).unwrap(), Place::Vram(_)) {
                        assert!(g.uva.contains(&b), "an earlier block neither hot nor UVA");
                    }
                }
            }
            let mut copied = 0;
            for op in &g.ops {
                let CopyOp::H2d { .. } = op else {
                    panic!("a prefill gather copies H2D only (no Landed: stream order)");
                };
                copied += 1;
            }
            assert_eq!(copied, g.copied);
            assert_eq!(
                g.copied as usize + g.uva.len(),
                before
                    .iter()
                    .filter(|p| !matches!(p, Place::Vram(_)))
                    .count()
            );
            uva += g.uva.len() as u64;
            chunks += 1;
            let len = t.len(seq);
            let n = plan.chunk_tokens.min(target[seq as usize] - len);
            t.append(seq, n).unwrap();
            m.write(&t, seq, len, len + n);
            let c = t.commit(seq).unwrap();
            m.commit(&c);
        } else {
            t.protect(seq, false).unwrap();
            let len0 = t.len(seq);
            if t.append(seq, 1).is_err() {
                continue;
            }
            m.write(&t, seq, len0, len0 + 1);
            let visible = (len0 + 1) / GROUP_TOKENS;
            let others: Vec<(u32, Vec<Place>)> = (0..ns as u32)
                .filter(|&s| s != seq && t.is_live(s) && t.is_protected(s))
                .map(|s| (s, places(&t, s)))
                .collect();
            for layer in 0..plan.layers {
                let groups: Vec<u32> = (0..8)
                    .map(|_| rng.below(u64::from(visible.max(1))) as u32)
                    .filter(|&g| g < visible)
                    .collect();
                let g = t.gather(seq, layer, &groups).unwrap();
                m.gather(&g);
            }
            for (s, pl) in &others {
                let now = places(&t, *s);
                for (b, a) in pl.iter().zip(&now) {
                    if matches!(b, Place::Vram(_)) {
                        assert_eq!(
                            a, b,
                            "a decode gather evicted a prefilling sequence's block"
                        );
                    }
                }
            }
            let c = t.commit(seq).unwrap();
            m.commit(&c);
        }
        t.check().unwrap();
        m.law(&t);
    }
    (t, chunks, uva)
}

#[test]
fn the_prefill_gather_makes_every_earlier_block_hot_and_keeps_the_byte_law() {
    for (seed, extra) in [(11u64, 0u64), (12, 12), (13, 60), (14, 400)] {
        let (t, chunks, uva) = drive_prefill(small_plan(extra, 4), seed, 4000);
        let c = t.counts();
        assert!(chunks > 300, "seed {seed}: {chunks} chunks");
        if extra <= 12 {
            assert!(c.evictions > 0 && c.h2d > 0, "seed {seed}: {c:?}");
        }
        if extra >= 400 {
            // a pool holding every context: every chunk reads only hot blocks
            assert_eq!(uva, 0, "seed {seed}");
        }
    }
}

#[test]
fn a_prefilling_sequence_is_the_last_victim() {
    // A decodes (unprotected, cold blocks), B prefills (protected), C's prompt arrives
    let plan = small_plan(12, 4);
    let mut t = KvTier::new(plan.clone());
    for s in [0u32, 1] {
        t.admit(s, 512).unwrap();
    }
    t.protect(1, true).unwrap();
    fill(&mut t, 0, 128);
    fill(&mut t, 1, 128);
    let hot = |t: &KvTier, s: u32| -> u32 {
        (0..plan.layers).map(|l| t.hot_outside(s, l).unwrap()).sum()
    };
    let (a0, b0) = (hot(&t, 0), hot(&t, 1));
    assert!(a0 > 0 && b0 > 0, "{a0} {b0}");
    t.admit(2, 512).unwrap();
    t.protect(2, true).unwrap();
    fill(&mut t, 2, 128);
    let (a1, b1) = (hot(&t, 0), hot(&t, 1));
    assert!(a1 < a0, "A (decoding) evicted first: {a0} -> {a1}");
    assert_eq!(
        b1, b0,
        "B (prefilling) untouched while A had victims: {b0} -> {b1}"
    );
    let g = t.gather_seq(1).unwrap();
    assert!(
        g.uva.is_empty() && g.copied == 0,
        "B had nothing evicted: {g:?}"
    );
    // a decode gather of A never takes B's
    t.protect(0, false).unwrap();
    let before = hot(&t, 1);
    for l in 0..plan.layers {
        t.gather(0, l, &(0..32).collect::<Vec<_>>()).unwrap();
    }
    assert_eq!(hot(&t, 1), before);
    t.protect(1, false).unwrap();
    t.check().unwrap();
}

/// FN68: an fp8 plan is the bf16 plan at half the bytes: the same byte budgets hold twice the
/// blocks, the same block counts need half the bytes; the 8 x 250k RAM need halves (the boot the
/// bf16 plan refuses at 20 GiB fits at fp8); `new` == `new_dt(.., Bf16)`.
#[test]
fn fn68_an_fp8_plan_is_the_bf16_plan_at_half_the_bytes() {
    use saber_mem::kv_tier::KvDtype;
    assert_eq!(KvDtype::Bf16.token_layer_bytes(), KV_TOKEN_LAYER_BYTES);
    assert_eq!(KvDtype::Fp8.token_layer_bytes() * 2, KV_TOKEN_LAYER_BYTES);
    let b = KvTierPlan::new(QSA_LAYERS, 16, 1024, 2048, 8, 65_536, 2 * GIB, 7 * GIB, 0).unwrap();
    let b2 = KvTierPlan::new_dt(
        QSA_LAYERS,
        16,
        1024,
        2048,
        8,
        65_536,
        2 * GIB,
        7 * GIB,
        0,
        KvDtype::Bf16,
    )
    .unwrap();
    assert_eq!(b, b2);
    let f = KvTierPlan::new_dt(
        QSA_LAYERS,
        16,
        1024,
        2048,
        8,
        65_536,
        GIB,
        7 * GIB / 2,
        0,
        KvDtype::Fp8,
    )
    .unwrap();
    assert_eq!(f.block_bytes() * 2, b.block_bytes());
    assert_eq!(
        (f.hot_slots, f.ram_slots, f.quota),
        (b.hot_slots, b.ram_slots, b.quota)
    );
    assert_eq!(f.ram_bytes() * 2, b.ram_bytes());
    assert!(KvTierPlan::new(QSA_LAYERS, 16, 1024, 2048, 8, 256_000, 2 * GIB, 20 * GIB, 0).is_err());
    let f2 = KvTierPlan::new_dt(
        QSA_LAYERS,
        16,
        1024,
        2048,
        8,
        256_000,
        2 * GIB,
        20 * GIB,
        0,
        KvDtype::Fp8,
    )
    .unwrap();
    assert_eq!(f2.block_bytes(), 16 * 512);
    let a = KvTierPlan::new_admitted_dt(
        QSA_LAYERS,
        16,
        1024,
        2048,
        8,
        262_144,
        GIB,
        2 * GIB,
        KvDtype::Fp8,
    )
    .unwrap();
    assert_eq!(a.token_layer_bytes, 512);
}
