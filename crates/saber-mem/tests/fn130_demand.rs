//! FN130: THE DEMAND LAW of the elastic KV region (LEDGER FN130).
//!
//! FN128 h1 s4_A / h3: a served boot died with `kv tier: no vram slot for a block that must be
//! resident` on a 32k prompt whose first two 8,192-token pieces were RELOADED from the prefix
//! tree's drive tier back to back: a reload opens a whole piece's pages through the step law
//! (`KvDev::begin`) with no prefill chunk, so the elastic controller never ticked and the cap
//! stayed at the boot lend (9,580 of 120,940 pages). The model here is the tier half of
//! `kvwire::plan_step` (append to `p0 - SLACK`, commit, open to `end + AHEAD`, protected) over a
//! scaled geometry: a negative control that reproduces the fatal, the demand law that turns it
//! into an inline reclaim (no tick, no lend, counted), and the burst of several long prompts.

use saber_mem::elastic::{Elastic, ElasticGeom, Move};
use saber_mem::kv_tier::{KvDtype, KvTier, KvTierError, KvTierPlan};

/// `kvwire::SLACK` / `AHEAD`, scaled to the model's 16-token pages.
const SLACK: u32 = 8;
const AHEAD: u32 = 4;
const PIECE: u32 = 64;

/// 2 layers, 16-token pages, a 64-token window and chunk, 4 sequences x 2,048 tokens, 256 hot
/// pages; segments of 2 pages past the empty pool's floor (one chunk's blocks), the boot lend
/// down to the band (`Elastic::boot_lent` over the idle need) -- the served boot's shape.
fn boot() -> (KvTier, Elastic) {
    let plan = KvTierPlan::new_admitted_dt(
        2,
        16,
        64,
        64,
        4,
        2048,
        256 * 16 * 1024,
        4 * 2048 / 16 * 2 * 16 * 1024,
        KvDtype::Bf16,
    )
    .unwrap();
    let mut kv = KvTier::new(plan);
    let keep = kv.hot_floor() as u32;
    let geom = ElasticGeom {
        seg_pages: 2,
        total: kv.plan().hot_slots,
        segs: (kv.plan().hot_slots - keep) / 2,
    };
    let need = kv.hot_floor().max(kv.hot_want());
    let n = Elastic::boot_lent(&geom, need);
    kv.shrink_hot(geom.cap(n)).unwrap();
    (kv, Elastic::new(geom, n))
}

/// The tier half of `kvwire::plan_step` for one sequence's prefill-form step `p0..pend`.
fn step(kv: &mut KvTier, s: u32, p0: u32, pend: u32) -> Result<(), KvTierError> {
    kv.protect(s, true)?;
    let fin = p0.saturating_sub(SLACK);
    let mut len = kv.len(s);
    if fin > len {
        while len < fin {
            let n = (fin - len).min(kv.plan().chunk_tokens);
            kv.append(s, n)?;
            len += n;
        }
        kv.commit(s)?;
    }
    kv.open(s, pend + AHEAD)
}

/// `KvDev::shortfall` for one sequence's step.
fn short(kv: &KvTier, s: u32, pend: u32) -> u64 {
    let res = kv.max_tokens(s).unwrap();
    kv.open_shortfall(&[(s, (pend + AHEAD).min(res).div_ceil(kv.plan().block_tokens))])
}

/// The engine's demand site (`worker::elastic_demand`, `on`): the controller's need and the
/// step's shortfall, a reclaim applied at once (`grow_hot`), counted as an early one.
fn demand(kv: &mut KvTier, el: &mut Elastic, s: u32, pend: u32) -> Move {
    let need = el.need(kv.hot_floor(), kv.hot_want());
    let m = el.demand(need, short(kv, s, pend));
    if let Move::Reclaim(n) = m {
        kv.grow_hot(el.geom.cap(el.lent - n)).unwrap();
        el.applied_demand(m);
    }
    m
}

/// THE BOOT RACE (the controller has 0 ticks, a long prompt's pieces reload back to back): with
/// no demand site the second piece finds no VRAM slot (the FATAL, the negative control); with the
/// demand law the first reload reclaims what the reservation needs, inline, and every piece opens
/// -- no tick runs, no lend, the bookkeeping checks.
#[test]
fn fn130_a_reload_at_zero_ticks_reclaims_on_demand_and_the_control_dies() {
    // the control: the pre-FN130 law
    let (mut kv, el) = boot();
    let cap0 = kv.hot_cap();
    assert!(
        el.lent > 0 && cap0 < el.geom.total,
        "the boot lends the idle pool"
    );
    kv.admit(0, 1024).unwrap();
    let mut died = None;
    for p in 0..1024 / PIECE {
        if let Err(e) = step(&mut kv, 0, p * PIECE, (p + 1) * PIECE) {
            died = Some((p, e));
            break;
        }
    }
    let (p, e) = died.expect("the control must die: the cap holds fewer pages than two pieces");
    assert!(p >= 1, "one piece fits the boot cap");
    assert!(
        matches!(e, KvTierError::NoRoom { pool: "vram", .. }),
        "the FATAL's class: {e}"
    );

    // the law
    let (mut kv, mut el) = boot();
    kv.admit(0, 1024).unwrap();
    for p in 0..1024 / PIECE {
        let m = demand(&mut kv, &mut el, 0, (p + 1) * PIECE);
        if p == 0 {
            assert!(
                matches!(m, Move::Reclaim(n) if n > 0),
                "the first reload reclaims"
            );
        }
        step(&mut kv, 0, p * PIECE, (p + 1) * PIECE).unwrap();
        kv.check().unwrap();
    }
    let c = el.counts;
    assert_eq!(c.ticks, 0, "the demand law is not a tick");
    assert_eq!(c.lends, 0, "the demand law never lends");
    assert!(c.early_reclaims >= 1 && c.early_segs > 0);
    assert_eq!(c.reclaims, c.early_reclaims);
    assert_eq!(kv.hot_cap(), el.geom.cap(el.lent), "the owner law's cap");
}

/// THE BURST: three long prompts admitted at once (their reservations' want past the region:
/// the need is capped at the whole pool), their pieces interleaved; with the demand law every
/// piece opens (the region holds every window + one chunk), without it the second sequence's
/// first piece dies (the boot cap holds one piece).
#[test]
fn fn130_a_burst_of_long_prompts_reclaims_and_never_dies() {
    let run = |law: bool| -> Result<Elastic, KvTierError> {
        let (mut kv, mut el) = boot();
        for s in 0..3 {
            kv.admit(s, 2000).unwrap();
        }
        assert!(
            kv.hot_want() > u64::from(el.geom.total),
            "the burst wants past the region"
        );
        for p in 0..2000 / PIECE {
            for s in 0..3 {
                if law {
                    demand(&mut kv, &mut el, s, (p + 1) * PIECE);
                }
                step(&mut kv, s, p * PIECE, (p + 1) * PIECE)?;
                kv.check().unwrap();
            }
        }
        Ok(el)
    };
    assert!(
        matches!(run(false), Err(KvTierError::NoRoom { pool: "vram", .. })),
        "the control dies"
    );
    let el = run(true).unwrap();
    assert_eq!(el.lent, 0, "the burst takes the whole region back");
    assert!(el.counts.early_reclaims >= 1);
    assert_eq!(el.counts.lends, 0);
}

/// The pure law: a shortfall alone (the need under the cap) reclaims its segments; nothing short
/// holds without touching the calm count or the ticks; with nothing lent a demand holds (the
/// region is the KV's already); a demand never lends.
#[test]
fn fn130_the_demand_law_reclaims_by_need_or_shortfall_and_never_lends() {
    let geom = ElasticGeom {
        seg_pages: 10,
        total: 200,
        segs: 18,
    };
    let mut el = Elastic::new(geom, 10); // cap 100
    assert_eq!(el.demand(50, 0), Move::Hold);
    assert_eq!(el.demand(100, 0), Move::Hold);
    assert_eq!(el.demand(101, 0), Move::Reclaim(1));
    assert_eq!(el.demand(50, 25), Move::Reclaim(3), "the shortfall alone");
    assert_eq!(
        el.demand(150, 25),
        Move::Reclaim(5),
        "the larger of the two"
    );
    assert_eq!(
        el.demand(10_000, 0),
        Move::Reclaim(10),
        "capped by the lent count"
    );
    el.applied_demand(Move::Reclaim(10));
    assert_eq!(el.lent, 0);
    assert_eq!(
        el.demand(10_000, 999),
        Move::Hold,
        "nothing left to reclaim"
    );
    let c = el.counts;
    assert_eq!((c.ticks, c.lends), (0, 0));
    assert_eq!((c.early_reclaims, c.early_segs, c.demands), (1, 10, 7));
    // the step law's tick after a demand reclaim: a reclaim reset the calm count, so the
    // surplus is not lent back before CALM_TICKS ticks
    let mut el = Elastic::new(geom, 10);
    let m = el.demand(150, 0);
    el.applied_demand(m);
    assert_eq!(
        el.tick(0, || true),
        Move::Hold,
        "no lend on the first calm tick"
    );
}
