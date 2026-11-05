//! FN45b: the tier's DEVICE PAGE TABLE on the host -- the address law (one base, whole blocks,
//! the union's 2^27-page ceiling), the per-layer tables == the tier's placements after every
//! prefill / decode / gather / evict / finish (the incremental sync == a full rebuild), the
//! append slots on hot pages, the <= 8k law at the served geometry (today's 0.92 GiB as the hot
//! pool with an 8,192-token window holds every block of 8 x 8,192 hot: the flat run's reads), the
//! flush / gather runs and the upload spans.

use saber_mem::device_mover::CopyOp;
use saber_mem::kv_tier::{
    coalesce, CopyRun, KvAddr, KvPageTable, KvTier, KvTierError, KvTierPlan, Place, GROUP_TOKENS,
    KV_TOKEN_LAYER_BYTES, MAX_PAGE_INDEX, QSA_LAYERS,
};

const BB: u64 = 16 * KV_TOKEN_LAYER_BYTES;
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

/// Three regions in a plausible layout: the hot slab, the pad page after it, the RAM slab 64 GiB
/// above (a pinned slab's device address).
fn addr(p: &KvTierPlan) -> KvAddr {
    let hot = 0x7f00_0000_0000u64;
    let pad = hot + u64::from(p.hot_slots) * BB;
    let ram = hot + 64 * GIB;
    KvAddr::new(BB, hot, p.hot_slots, ram, p.ram_slots, pad).unwrap()
}

/// The table rebuilt from scratch (every row of every slot).
fn rebuilt(p: &KvTierPlan, t: &KvTier, slots: u32) -> KvPageTable {
    let mut f = KvPageTable::new(p, slots, addr(p)).unwrap();
    for s in 0..p.max_seqs {
        for l in 0..p.layers {
            f.rewrite(t, s, l).unwrap();
        }
    }
    f
}

/// Every live block's entry names its placement; every other entry is the pad page.
fn entries_are_placements(p: &KvTierPlan, t: &KvTier, pt: &KvPageTable, slots: u32) {
    let a = pt.addr();
    for s in 0..slots {
        for l in 0..p.layers {
            let n = if s < p.max_seqs { t.blocks_of(s, l) } else { 0 };
            for (b, &e) in pt.row(l, s).iter().enumerate() {
                if (b as u32) < n {
                    let want = t.place(s, l, b as u32).unwrap();
                    assert_eq!(a.resolve(e), Some(want), "seq {s} layer {l} block {b}");
                } else {
                    assert_eq!(
                        e, a.pad as i32,
                        "seq {s} layer {l} block {b} past the blocks"
                    );
                }
            }
        }
    }
}

#[test]
fn the_address_law_takes_one_base_whole_blocks_and_the_union_ceiling() {
    let hot = 0x7f00_0000_0000u64;
    let a = KvAddr::new(BB, hot, 100, hot - 10 * BB * 1000, 50, hot + 100 * BB).unwrap();
    assert_eq!(a.base, hot - 10 * BB * 1000);
    assert_eq!((a.ram, a.hot, a.pad), (0, 10_000, 10_100));
    for v in [0, 1, 99] {
        let pg = a.page(Place::Vram(v)).unwrap();
        assert_eq!(a.resolve(pg), Some(Place::Vram(v)));
    }
    for r in [0, 49] {
        let pg = a.page(Place::Ram(r)).unwrap();
        assert_eq!(a.resolve(pg), Some(Place::Ram(r)));
    }
    assert_eq!(a.resolve(a.pad as i32), None);
    assert!(
        a.page(Place::Ssd(3)).is_err(),
        "a kernel never reads the drive"
    );
    assert!(a.page(Place::Vram(100)).is_err());
    // a region not a whole number of blocks from the base
    let e = KvAddr::new(BB, hot, 100, hot + 4096 + 200 * BB, 50, hot + 100 * BB);
    assert!(matches!(e, Err(KvTierError::Geometry(m)) if m.contains("whole number")));
    // past the union's i32 `page * 16 + slot`: 2 TiB apart
    let far = hot + MAX_PAGE_INDEX * BB;
    let e = KvAddr::new(BB, hot, 100, far, 50, hot + 100 * BB);
    assert!(matches!(e, Err(KvTierError::Geometry(m)) if m.contains("2 TiB")));
    let near = hot + (MAX_PAGE_INDEX - 50) * BB;
    assert!(KvAddr::new(BB, hot, 100, near, 50, hot + 100 * BB).is_ok());
    // overlapping regions
    let e = KvAddr::new(BB, hot, 100, hot + 99 * BB, 50, hot + 300 * BB);
    assert!(matches!(e, Err(KvTierError::Geometry(m)) if m.contains("overlap")));
}

/// The random driver of FN45a's law with the device table synced after every operation.
fn drive_table(p: &KvTierPlan, seed: u64, rounds: usize) -> (u64, u64) {
    let slots = p.max_seqs + 1;
    let mut t = KvTier::new(p.clone());
    let mut pt = KvPageTable::new(p, slots, addr(p)).unwrap();
    let mut rng = Rng(seed);
    let (mut rows_synced, mut ram_entries) = (0u64, 0u64);
    for _ in 0..rounds {
        let seq = rng.below(u64::from(p.max_seqs)) as u32;
        let len = t.len(seq);
        match rng.below(10) {
            0 if len > 0 => t.finish(seq).unwrap(),
            _ if len == 0 => {
                let max = 128 + rng.below(384) as u32;
                if t.admit(seq, max).is_err() {
                    continue;
                }
                let p0 = 1 + rng.below(u64::from(max / 2)) as u32;
                let mut done = 0;
                while done < p0 {
                    let n = p.chunk_tokens.min(p0 - done);
                    t.append(seq, n).unwrap();
                    // the chunk's append slots are hot pages (the compute writes them)
                    rows_synced += pt.sync(&mut t).unwrap().len() as u64;
                    for l in 0..p.layers {
                        for pos in done..done + n {
                            let sl = pt.append_slot(l, seq, pos);
                            let pg = (sl / 16) as i32;
                            assert!(matches!(pt.addr().resolve(pg), Some(Place::Vram(_))));
                            assert_eq!(sl % 16, i64::from(pos % 16));
                        }
                    }
                    t.commit(seq).unwrap();
                    done += n;
                }
            }
            _ => {
                let step = 1 + rng.below(4) as u32;
                if t.append(seq, step).is_err() {
                    continue;
                }
                let len = t.len(seq);
                let visible = len / GROUP_TOKENS;
                for l in 0..p.layers {
                    let mut groups: Vec<u32> = (0..visible.min(2)).collect();
                    for _ in 0..(1 + rng.below(24)) {
                        groups.push(rng.below(u64::from(visible)) as u32);
                    }
                    t.gather(seq, l, &groups).unwrap();
                }
                t.commit(seq).unwrap();
            }
        }
        rows_synced += pt.sync(&mut t).unwrap().len() as u64;
        assert_eq!(
            pt,
            rebuilt(p, &t, slots),
            "the incremental table == a full rebuild"
        );
        entries_are_placements(p, &t, &pt, slots);
        ram_entries += (0..p.max_seqs)
            .flat_map(|s| (0..p.layers).map(move |l| (s, l)))
            .map(|(s, l)| {
                pt.row(l, s)
                    .iter()
                    .filter(|&&e| matches!(pt.addr().resolve(e), Some(Place::Ram(_))))
                    .count() as u64
            })
            .sum::<u64>();
    }
    (rows_synced, ram_entries)
}

#[test]
fn the_page_table_is_the_tiers_placements_after_every_operation() {
    // FN45a's small geometry: 3 layers, 32-token window, 64-token chunks, 512-token contexts
    for (extra, seqs) in [(0u64, 3u32), (24, 3), (200, 2)] {
        let floor = 3 * (u64::from(seqs) * 3 + 5);
        let p = KvTierPlan::new(
            3,
            16,
            32,
            64,
            seqs,
            512,
            (floor + extra) * BB,
            u64::from(seqs) * 3 * 32 * BB,
            0,
        )
        .unwrap();
        for seed in [1u64, 7, 0x5eed, 0xfeed_beef] {
            let (rows, ram) = drive_table(&p, seed, 400);
            assert!(rows > 0);
            if extra == 0 {
                assert!(
                    ram > 0,
                    "a starved pool reads RAM pages (the counted overflow)"
                );
            }
        }
    }
}

#[test]
fn the_served_geometry_holds_every_block_hot_at_8k() {
    // FN45b's boot: 13 QSA layers, 16-token pages, an 8,192-token window, 8,192-token chunks,
    // 8 x 65,536, the hot pool = today's flat pools (13 x 9 x 513 pages), RAM = every block
    let pools = u64::from(QSA_LAYERS) * 9 * 513 * BB;
    let ram = 8 * u64::from(QSA_LAYERS) * 4096 * BB;
    let p = KvTierPlan::new(QSA_LAYERS, 16, 8192, 8192, 8, 65_536, pools, ram, 0).unwrap();
    assert_eq!(
        ram, 6_979_321_856,
        "6.5 GiB a rank = 13.0 GiB pinned for 8 x 64k"
    );
    let mut t = KvTier::new(p.clone());
    let mut pt = KvPageTable::new(&p, 9, addr(&p)).unwrap();
    assert_eq!(pt.nbs(), 4097);
    for s in 0..8 {
        t.admit(s, 65_536).unwrap();
        for c in 0..4 {
            t.append(s, 2048).unwrap();
            pt.sync(&mut t).unwrap();
            let _ = c;
            t.commit(s).unwrap();
        }
    }
    // 8 x 8,192 tokens + decode steps up to the window's edge: every block hot, nothing evicted
    for _ in 0..15 {
        for s in 0..8 {
            assert!(t.len(s) < 8192 + 16);
            t.append(s, 1).unwrap();
            t.commit(s).unwrap();
        }
    }
    pt.sync(&mut t).unwrap();
    entries_are_placements(&p, &t, &pt, 9);
    assert_eq!(t.counts().evictions, 0);
    for l in 0..QSA_LAYERS {
        let mut seen = std::collections::BTreeSet::new();
        for s in 0..8 {
            for &e in &pt.row(l, s)[..t.blocks_of(s, l) as usize] {
                assert!(matches!(pt.addr().resolve(e), Some(Place::Vram(_))));
                assert!(
                    seen.insert(e),
                    "layer {l}: page {e} aliased across sequences"
                );
            }
        }
    }
    // past the window the oldest blocks leave VRAM only when a slot is needed: a 9th chunk on
    // every sequence evicts (the floor has no out-of-window quota) and those entries read RAM
    for s in 0..8 {
        t.append(s, 2048).unwrap();
        t.commit(s).unwrap();
    }
    let rows = pt.sync(&mut t).unwrap();
    assert!(!rows.is_empty());
    entries_are_placements(&p, &t, &pt, 9);
    assert!(t.counts().evictions > 0);
}

#[test]
fn a_prefill_flush_on_a_fresh_tier_is_one_run_and_runs_keep_every_copy() {
    let floor = 3 * (2 * 3 + 5);
    let p = KvTierPlan::new(3, 16, 32, 64, 2, 512, floor * BB, 2 * 3 * 32 * BB, 0).unwrap();
    let mut t = KvTier::new(p);
    t.admit(0, 512).unwrap();
    t.append(0, 64).unwrap();
    let c = t.commit(0).unwrap();
    assert_eq!(c.flush.len(), 12);
    let runs = coalesce(&c.flush);
    // the append takes consecutive slots layer after layer: ONE memcpy for the whole chunk
    assert_eq!(
        runs,
        vec![CopyRun {
            h2d: false,
            vram: 0,
            ram: 0,
            n: 12
        }]
    );
    // runs expand back to the ops, in order
    let back: Vec<CopyOp> = runs
        .iter()
        .flat_map(|r| {
            (0..r.n).map(move |i| CopyOp::D2h {
                vram: r.vram + i,
                ram: r.ram + i,
            })
        })
        .collect();
    assert_eq!(back, c.flush);
    // direction changes and gaps split; Landed marks drop
    let ops = [
        CopyOp::H2d { ram: 4, vram: 9 },
        CopyOp::H2d { ram: 5, vram: 10 },
        CopyOp::D2h { vram: 11, ram: 6 },
        CopyOp::H2d { ram: 8, vram: 12 },
        CopyOp::Landed { layer: 2 },
    ];
    assert_eq!(
        coalesce(&ops),
        vec![
            CopyRun {
                h2d: true,
                vram: 9,
                ram: 4,
                n: 2
            },
            CopyRun {
                h2d: false,
                vram: 11,
                ram: 6,
                n: 1
            },
            CopyRun {
                h2d: true,
                vram: 12,
                ram: 8,
                n: 1
            },
        ]
    );
}

#[test]
fn upload_spans_merge_adjacent_rows_and_the_table_refuses_a_foreign_page() {
    let floor = 3 * (2 * 3 + 5);
    let p = KvTierPlan::new(3, 16, 32, 64, 2, 512, floor * BB, 2 * 3 * 32 * BB, 0).unwrap();
    let pt = KvPageTable::new(&p, 3, addr(&p)).unwrap();
    let rb = u64::from(pt.nbs()) * 4;
    assert_eq!(pt.nbs(), 33);
    assert_eq!(pt.bytes(), 3 * 3 * rb);
    assert_eq!(pt.layer_offset(2), 2 * 3 * rb);
    let sp = pt.upload_spans(&[(0, 1), (0, 2), (1, 0), (2, 2)]);
    assert_eq!(sp, vec![(rb, 3 * rb), (8 * rb, rb)]);
    let p32 = KvTierPlan::new(3, 32, 32, 64, 2, 512, 1 << 30, 1 << 30, 0).unwrap();
    assert!(
        KvPageTable::new(&p32, 3, addr(&p)).is_err(),
        "the kernels' page is 16 tokens"
    );
    assert!(KvPageTable::new(&p, 1, addr(&p)).is_err());
}
