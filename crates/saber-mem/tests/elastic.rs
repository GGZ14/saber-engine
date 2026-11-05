//! FN51f: ELASTIC VRAM -- the KV hot pool and the expert tier lend ONE region. Over a byte-true
//! host model of the region's pages (the KV stamps every page it places a block in; every lent
//! expert slot holding a record stamps its pages after each step) and FN42's fake copy queue: a
//! mixed run (short sequences, then one long reservation, then short again) at BOTH page sizes
//! keeps every KV block's page its own (no alias), every expert read lands on its record, the
//! ownership law holds after every move, the region lends the idle window during the short
//! phase, grows the KV past the boot window during the long one and lends back after it -- one
//! segment a tick after the calm, reclaims at once; lending raises the expert hit rate.

use std::collections::BTreeMap;

use saber_mem::device_mover::{DeviceMover, FakeQueue};
use saber_mem::dio::OpenMode;
use saber_mem::elastic::{Elastic, ElasticGeom, Move, BAND_SEGS, CALM_TICKS, SEG_SLOTS};
use saber_mem::expert_tier::{ExpertTier, TierParams};
use saber_mem::kv_tier::{KvDtype, KvTier, KvTierPlan, Place};
use saber_mem::xtier::{write_synthetic, SyntheticTier, XtierFile};
use saber_test_scratch::ScratchDir;

const LAYERS: u32 = 4;
const EXPERTS: u32 = 16;
const BASE: u32 = 20;
const VRAM_BASE: u64 = 0x7f00_0000_0000;
const RAM_BASE: u64 = 0x5500_0000_0000;
// the KV: 2 layers, 16-token pages, a 64-token window, 64-token chunks, 4 sequences x 2,048
const KL: u32 = 2;
const WIN: u32 = 64;
const CHUNK: u32 = 64;
const CTX: u32 = 2048;

fn synth(tag: &str) -> (ScratchDir, XtierFile) {
    let d = ScratchDir::new(&format!("saber-fn51f-{tag}"));
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
    (d, XtierFile::open(&p, OpenMode::Buffered).unwrap())
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
    fn routes(&mut self) -> Vec<Vec<u32>> {
        (0..LAYERS)
            .map(|_| {
                let mut v: Vec<u32> = (0..6)
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

/// A page's last writer in the region model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pg {
    Free,
    Kv(u32, u32, u32),
    Expert(u32),
}

struct Run {
    hits: u64,
    routed: u64,
    lent_short: u32,
    cap_long: u32,
    lent_after: u32,
    boot_window: u32,
    moves: Vec<Move>,
    /// FN51f-c: phase 2 (short again, after the long request) hits / routed.
    hits2: u64,
    routed2: u64,
    /// FN51f-c: lends after the first reclaim, and the EMPTY slots each left in its segment.
    lends_after_reclaim: u32,
    empty_after_lend: u32,
}

fn run(dt: KvDtype, elastic: bool) -> Run {
    run_refill(dt, elastic, false)
}

/// `eager` = FN51f-c's fill on lend: a returning segment's slots take the hottest RAM records
/// in the same tick (`ExpertTier::fill_lent_hot`, the copy stream fenced after the KV's uses).
fn run_refill(dt: KvDtype, elastic: bool, eager: bool) -> Run {
    let (_d, xt) = synth(&format!("{}-{elastic}-{eager}", dt.name()));
    let stride = xt.header().stride;
    let page = 16 * dt.token_layer_bytes(); // the page at this dtype (16,384 B bf16, 8,192 B fp8)
                                            // the region: the boot window (every sequence's window + a chunk) + a borrow of 8 segments
    let slot_pages = stride.div_ceil(page) as u32;
    let boot = KL * (4 * (WIN / 16 + 1) + CHUNK / 16 + 1);
    let total = boot + 8 * SEG_SLOTS * slot_pages;
    let plan = KvTierPlan::new_admitted_dt(
        KL,
        16,
        WIN,
        CHUNK,
        4,
        CTX,
        u64::from(total) * 16 * dt.token_layer_bytes(),
        u64::from(KL) * u64::from(CTX / 16 + 64) * 4 * 16 * dt.token_layer_bytes(),
        dt,
    )
    .unwrap();
    assert_eq!(plan.hot_slots, total);
    let mut kv = KvTier::new(plan.clone());
    let keep = KL * (CHUNK / 16 + 1);
    let geom = ElasticGeom::new(total, page, stride, keep);
    assert_eq!(
        geom.seg_pages,
        SEG_SLOTS * slot_pages,
        "whole slots at either page size"
    );
    let nlend = geom.segs * SEG_SLOTS;
    let params = TierParams {
        vram_slots: BASE,
        ram_slots: 64 - BASE,
        min_layer_slots: 2,
        max_inserts: 2,
        read_through: 1.0,
        lrfu_lambda: 0.98,
        ..TierParams::default()
    };
    let mut tier = ExpertTier::new(xt.header(), params.clone(), None, VRAM_BASE, RAM_BASE).unwrap();
    // lendable slot (segment i, k) lives at the region's page offset (the KV pool at 0x9000..)
    let kv_base = 0x9000_0000_0000u64;
    let addrs: Vec<u64> = (0..geom.segs)
        .flat_map(|i| (0..SEG_SLOTS).map(move |k| (i, k)))
        .map(|(i, k)| kv_base + geom.slot_offset(i, k, page))
        .collect();
    let first = tier.register_lendable(&addrs).unwrap();
    let mut q = FakeQueue::new(stride, BASE + nlend, params.ram_slots);
    q.warm(&xt, &tier.warm_plan()).unwrap();
    let mut mv = DeviceMover::new(&xt, q, 4);
    let mut el = Elastic::new(geom, 0);
    if elastic {
        let n = Elastic::boot_lent(&geom, kv.hot_floor());
        kv.shrink_hot(geom.cap(n)).unwrap();
        let ids: Vec<u32> = (0..n).flat_map(|i| Elastic::seg_ids(first, i)).collect();
        tier.lend(&ids).unwrap();
        el = Elastic::new(geom, n);
    }
    let mut mem = vec![Pg::Free; total as usize];
    let mut rng = Lcg(0x51f0 + u64::from(elastic));
    let lent_ids = |tier: &ExpertTier| -> Vec<u32> {
        (first..first + nlend)
            .filter(|&s| tier.is_lent(s))
            .collect()
    };
    let mut out = Run {
        hits: 0,
        routed: 0,
        lent_short: 0,
        cap_long: 0,
        lent_after: 0,
        boot_window: boot,
        moves: Vec::new(),
        hits2: 0,
        routed2: 0,
        lends_after_reclaim: 0,
        empty_after_lend: 0,
    };
    let mut reclaimed = false;
    // phases: 0..40 short (4 x window-sized reservations), 40..120 one long (2,000), 120..400 short
    let mut live: BTreeMap<u32, u32> = BTreeMap::new(); // seq -> tokens
    for step in 0..400u32 {
        let phase = if step < 40 {
            0
        } else if step < 120 {
            1
        } else {
            2
        };
        // admissions / finishes (the scheduler's Zero / Free)
        if step == 0 || step == 120 {
            for s in live.keys().copied().collect::<Vec<_>>() {
                kv.finish(s).unwrap();
            }
            live.clear();
            for s in 0..4 {
                kv.admit(s, WIN).unwrap();
                live.insert(s, 0);
            }
        }
        if step == 40 {
            for s in live.keys().copied().collect::<Vec<_>>() {
                kv.finish(s).unwrap();
            }
            live.clear();
            kv.admit(0, 2000).unwrap();
            live.insert(0, 0);
        }
        // the elastic tick (between jobs: before the step's appends)
        if elastic {
            let need = el.need(kv.hot_floor(), kv.hot_want());
            let lent = el.lent;
            let m = el.tick(need, || {
                lent < geom.segs && kv.can_shrink_hot(geom.cap(lent + 1))
            });
            match m {
                Move::Hold => {}
                Move::Reclaim(n) => {
                    mv.drain().unwrap();
                    let ids: Vec<u32> = (el.lent - n..el.lent)
                        .flat_map(|i| Elastic::seg_ids(first, i))
                        .collect();
                    tier.reclaim(&ids);
                    kv.grow_hot(geom.cap(el.lent - n)).unwrap();
                    out.moves.push(m);
                    reclaimed = true;
                }
                Move::Lend => {
                    kv.shrink_hot(geom.cap(el.lent + 1)).unwrap();
                    el.lend_alias_check(el.lent, &kv).unwrap();
                    let ids = Elastic::seg_ids(first, el.lent);
                    tier.lend(&ids).unwrap();
                    if eager {
                        // the copy stream after every earlier use of the pages, then the fill
                        mv.after(&()).unwrap();
                        let n = tier.fill_lent_hot(&ids, &mut mv).unwrap();
                        assert!(n <= SEG_SLOTS as usize, "a tick moves one segment's worth");
                    }
                    if reclaimed {
                        out.lends_after_reclaim += 1;
                        out.empty_after_lend += ids
                            .iter()
                            .filter(|&&s| tier.vram_record(s).is_none())
                            .count() as u32;
                    }
                    out.moves.push(m);
                }
            }
            el.applied(m);
            el.owner_law(kv.hot_cap(), first, &lent_ids(&tier)).unwrap();
        }
        // the KV step: each live sequence grows (a chunk while prefilling the long one, else a
        // decode-like 16 tokens), commits, opens ahead -- every newly placed block stamps its page
        for (&s, len) in live.iter_mut() {
            let max = if phase == 1 { 2000 - 64 } else { WIN - 32 };
            let n = if phase == 1 && *len < 1600 { 64 } else { 16 };
            if *len + n <= max {
                kv.append(s, n).unwrap();
                *len += n;
                kv.commit(s).unwrap();
                kv.open(s, *len + 16).unwrap();
            }
        }
        for &s in live.keys() {
            for l in 0..KL {
                for b in 0..kv.blocks_of(s, l) {
                    if let Place::Vram(v) = kv.place(s, l, b).unwrap() {
                        let m = &mut mem[v as usize];
                        // a block placed hot since the last step: its page is written now
                        *m = Pg::Kv(s, l, b);
                    }
                }
            }
        }
        kv.check().unwrap();
        // the expert step (lent slots holding a record write their pages)
        tier.begin_step();
        let routes = rng.routes();
        mv.gate(0, &()).unwrap();
        mv.after(&()).unwrap();
        for (l, rs) in routes.iter().enumerate() {
            let r = tier.resolve(l as u32, rs, &mut mv).unwrap();
            if phase != 1 {
                out.hits += r.resident;
                out.routed += rs.len() as u64;
            }
            if phase == 2 {
                out.hits2 += r.resident;
                out.routed2 += rs.len() as u64;
            }
        }
        tier.end_step(&mut mv).unwrap();
        tier.check_invariants().unwrap();
        mv.drain().unwrap();
        for i in 0..geom.segs {
            for k in 0..SEG_SLOTS {
                let s = first + i * SEG_SLOTS + k;
                if tier.vram_record(s).is_some() {
                    let p0 = (geom.slot_offset(i, k, page) / page) as usize;
                    for m in &mut mem[p0..p0 + slot_pages as usize] {
                        *m = Pg::Expert(s);
                    }
                }
            }
        }
        // THE BYTE LAW over the region: every hot KV block still reads its own page
        for &s in live.keys() {
            for l in 0..KL {
                for b in 0..kv.blocks_of(s, l) {
                    if let Place::Vram(v) = kv.place(s, l, b).unwrap() {
                        assert!(v < kv.hot_cap(), "a KV block past the cap");
                        assert_eq!(
                            mem[v as usize],
                            Pg::Kv(s, l, b),
                            "step {step}: KV page {v} overwritten"
                        );
                    }
                }
            }
        }
        // every placed expert slot is its record
        for s in 0..BASE + nlend {
            if let Some(r) = tier.vram_record(s) {
                assert!(xt.verify(u64::from(r), mv.queue().vram_slot(s)), "slot {s}");
            }
        }
        match step {
            39 => out.lent_short = el.lent,
            119 => out.cap_long = kv.hot_cap(),
            399 => out.lent_after = el.lent,
            _ => {}
        }
    }
    out
}

#[test]
fn the_region_lends_the_idle_window_grows_for_a_long_request_and_lends_back_bit_exact() {
    for dt in [KvDtype::Bf16, KvDtype::Fp8] {
        let on = run(dt, true);
        let off = run(dt, false);
        // short phase: the idle window is lent (beyond the band) at boot
        assert!(
            on.lent_short > 2,
            "{dt:?}: lent {} in the short phase",
            on.lent_short
        );
        // long phase: the KV grew past the boot window (the long reservation borrows)
        assert!(
            on.cap_long > on.boot_window,
            "{dt:?}: cap {} vs boot window {}",
            on.cap_long,
            on.boot_window
        );
        assert!(on.moves.iter().any(|m| matches!(m, Move::Reclaim(_))));
        // after it the budget is lent back one segment a tick, down to the band (a reclaim fills
        // to the need exactly, a lend stops BAND_SEGS above it: the hysteresis)
        assert!(
            on.lent_after + BAND_SEGS >= on.lent_short && on.lent_after > 0,
            "{dt:?}: lent back {} vs {} ({:?})",
            on.lent_after,
            on.lent_short,
            on.moves
        );
        assert!(on.moves.iter().filter(|m| **m == Move::Lend).count() >= 2);
        // lending adds expert hits (the same routes)
        assert!(
            on.hits * off.routed > off.hits * on.routed,
            "{dt:?}: hits {}/{} vs off {}/{}",
            on.hits,
            on.routed,
            off.hits,
            off.routed
        );
        let _ = CALM_TICKS;
    }
}

/// FN51f-c: THE FILL ON LEND -- after a reclaim, every lend hands the experts a FULL segment
/// (each slot holds the hottest RAM record of its layer, bit-exact: the run's byte law checks
/// every placed slot against its record and every hot KV page against its block), and the short
/// phase after the long request hits at least as often as the lazy law's (today: the slots
/// arrive EMPTY and refill only through the insert cap). NEGATIVE CONTROL: the lazy run leaves
/// empty slots after its lends.
#[test]
fn a_lend_after_a_reclaim_leaves_no_empty_slot_and_the_short_phase_does_not_dip() {
    for dt in [KvDtype::Bf16, KvDtype::Fp8] {
        let eager = run_refill(dt, true, true);
        let lazy = run_refill(dt, true, false);
        assert!(
            eager.lends_after_reclaim > 0 && lazy.lends_after_reclaim > 0,
            "{dt:?}: no lend after the long request"
        );
        eprintln!(
            "FN51f-c {dt:?}: phase 2 hit eager {}/{} lazy {}/{}; lends after the reclaim {} \
             (empty slots left: eager {}, lazy {})",
            eager.hits2,
            eager.routed2,
            lazy.hits2,
            lazy.routed2,
            eager.lends_after_reclaim,
            eager.empty_after_lend,
            lazy.empty_after_lend
        );
        assert_eq!(
            eager.empty_after_lend, 0,
            "{dt:?}: eager lends left empty slots"
        );
        assert!(
            lazy.empty_after_lend > 0,
            "{dt:?}: the control (lazy) must hand slots over empty"
        );
        assert!(
            eager.hits2 * lazy.routed2 >= lazy.hits2 * eager.routed2,
            "{dt:?}: phase 2 hits {}/{} eager vs {}/{} lazy",
            eager.hits2,
            eager.routed2,
            lazy.hits2,
            lazy.routed2
        );
        assert_eq!(
            eager.moves, lazy.moves,
            "{dt:?}: placement only -- the region law is unchanged"
        );
    }
}

#[test]
fn a_window_block_in_the_next_segment_blocks_the_lend_and_a_shrink_refuses_without_change() {
    let plan = KvTierPlan::new_admitted_dt(
        1,
        16,
        32,
        32,
        2,
        512,
        64 * 16 * 1024,
        64 * 64 * 1024,
        KvDtype::Bf16,
    )
    .unwrap();
    let mut kv = KvTier::new(plan);
    kv.admit(0, 512).unwrap();
    // fill the pool so the top holds a window block
    for _ in 0..30 {
        kv.append(0, 16).unwrap();
        kv.commit(0).unwrap();
    }
    kv.open(0, 30 * 16 + 16).unwrap();
    let top = kv.plan().hot_slots;
    let hot: Vec<u32> = (0..kv.blocks_of(0, 0))
        .filter_map(|b| match kv.place(0, 0, b).unwrap() {
            Place::Vram(v) => Some(v),
            _ => None,
        })
        .collect();
    let lo = *hot.iter().max().unwrap();
    assert!(
        !kv.can_shrink_hot(lo),
        "the window's newest block sits at {lo}"
    );
    assert!(kv.shrink_hot(lo).is_err());
    assert_eq!(kv.hot_cap(), top, "a refused shrink changes nothing");
    kv.check().unwrap();
    // an evictable (out-of-window) block is evicted with no copy
    let ev = kv.counts().evictions;
    let free_top = (0..top).rev().take_while(|v| !hot.contains(v)).count() as u32;
    kv.shrink_hot(top - free_top).unwrap();
    assert_eq!(kv.counts().evictions, ev);
    kv.check().unwrap();
    kv.grow_hot(top).unwrap();
    kv.check().unwrap();
}

/// A small KV tier (1 layer, 16-token pages, 32-token window and chunks, 2 sequences x 2,048)
/// with `hot` slots, and a region geometry over it (8-page segments, 2 pages kept).
fn small_kv(hot: u32) -> (KvTier, ElasticGeom) {
    let plan = KvTierPlan::new_admitted_dt(
        1,
        16,
        32,
        32,
        2,
        2048,
        u64::from(hot) * 16 * 1024,
        2 * 2048 / 16 * 2 * 16 * 1024,
        KvDtype::Bf16,
    )
    .unwrap();
    let geom = ElasticGeom {
        seg_pages: 8,
        total: plan.hot_slots,
        segs: (plan.hot_slots - 2) / 8,
    };
    (KvTier::new(plan), geom)
}

/// The NEGATIVE CONTROL of the lend: a lend attempted while KV blocks still sit in the
/// segment's pages (no `shrink_hot` first) is REFUSED by the alias check -- the same segment
/// passes once the KV gave it up -- and in the byte model the skipped shrink is exactly what
/// overwrites a live KV page.
#[test]
fn a_lend_while_kv_blocks_sit_in_the_segment_is_refused_by_the_alias_check() {
    let (mut kv, geom) = small_kv(64);
    kv.admit(0, 2048).unwrap();
    // fill the whole pool: the top segment holds blocks (evictable ones: out of the window)
    for _ in 0..60 {
        kv.append(0, 16).unwrap();
        kv.commit(0).unwrap();
    }
    kv.open(0, 60 * 16 + 16).unwrap();
    let el = Elastic::new(geom, 0);
    let (held, _) = kv.hot_held_in(geom.seg_first_page(0), geom.total);
    assert!(held > 0, "the top segment holds KV blocks");
    let e = el.lend_alias_check(0, &kv).unwrap_err();
    assert!(e.contains("refused"), "{e}");
    // the byte model of the skipped shrink: the expert copy into the segment's first slot
    // writes the page a hot KV block still reads (what the check refuses)
    let lo = geom.seg_first_page(0);
    let alias = (0..kv.blocks_of(0, 0)).any(|b| {
        matches!(kv.place(0, 0, b).unwrap(), Place::Vram(v) if v >= lo && v < lo + geom.seg_pages)
    });
    assert!(
        alias,
        "the control differs from the case where the law runs"
    );
    // the law: shrink first (evicts the out-of-window blocks with no copy), then the check passes
    if kv.can_shrink_hot(geom.cap(1)) {
        kv.shrink_hot(geom.cap(1)).unwrap();
        el.lend_alias_check(0, &kv).unwrap();
    } else {
        // a window block in the top segment: the shrink is refused and so is the lend
        assert!(kv.shrink_hot(geom.cap(1)).is_err());
        assert!(el.lend_alias_check(0, &kv).is_err());
    }
    // a free page in the segment (the KV's free list) is refused too
    let (mut kv2, g2) = small_kv(64);
    assert!(Elastic::new(g2, 0).lend_alias_check(0, &kv2).is_err());
    kv2.shrink_hot(g2.cap(1)).unwrap();
    Elastic::new(g2, 0).lend_alias_check(0, &kv2).unwrap();
}

/// THE ADMISSION LAW of the borrow: a decode step has no KV gather, so hot slots reclaimed
/// mid-decode stay EMPTY (the out-of-window blocks keep reading RAM); reclaimed before the
/// prompt's first chunk, the prefill gather fills them -- why the engine ticks the law at the
/// admission (`Cmd::Zero` reserves; the tick runs before the first chunk's step law).
#[test]
fn slots_reclaimed_mid_decode_stay_empty_and_at_admission_the_gather_fills_them() {
    let ram_blocks = |kv: &KvTier| {
        (0..kv.blocks_of(0, 0))
            .filter(|&b| matches!(kv.place(0, 0, b).unwrap(), Place::Ram(_)))
            .count()
    };
    // one 1,024-token prompt in 32-token chunks (the gather before each), then 32 decode steps
    let run = |grow_at_admission: bool| {
        let (mut kv, geom) = small_kv(96);
        // boot: everything but the keep + 1 segment lent (cap 18 pages)
        let lent = geom.segs - 1;
        kv.shrink_hot(geom.cap(lent)).unwrap();
        kv.admit(0, 2048).unwrap();
        if grow_at_admission {
            kv.grow_hot(geom.total).unwrap();
        }
        for c in 0..32 {
            kv.protect(0, true).unwrap();
            let _ = kv.gather_seq(0).unwrap();
            kv.append(0, 32).unwrap();
            kv.commit(0).unwrap();
            kv.open(0, (c + 1) * 32 + 16).unwrap();
        }
        kv.protect(0, false).unwrap();
        if !grow_at_admission {
            // the reclaim lands mid-decode
            kv.grow_hot(geom.total).unwrap();
        }
        let before = ram_blocks(&kv);
        for t in 0..32 {
            kv.append(0, 1).unwrap();
            kv.commit(0).unwrap();
            kv.open(0, 1024 + t + 2).unwrap();
        }
        kv.check().unwrap();
        let (free, _) = (kv.hot_cap() - kv.hot_held_in(0, kv.hot_cap()).0, ());
        (before, ram_blocks(&kv), free)
    };
    let (mid_before, mid_after, mid_free) = run(false);
    let (adm_before, adm_after, _) = run(true);
    assert!(
        mid_after >= mid_before && mid_before > 0,
        "mid-decode: the reclaimed slots fill nothing ({mid_before} -> {mid_after} RAM blocks)"
    );
    assert!(
        mid_free > 32,
        "mid-decode: {mid_free} reclaimed slots stay empty"
    );
    assert!(
        adm_after < mid_after && adm_before < mid_before,
        "at admission the gather keeps the prompt hot: {adm_after} vs {mid_after} RAM blocks"
    );
}
