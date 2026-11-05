//! FN7: the Flash-Next EP=2 byte ledger pinned against FN1's numbers (LEDGER FN1 §3) and FN2's
//! named pack deltas (LEDGER FN2 §2), both ranks, the over-commit refusal, the host L2 default.

use saber_mem::flashnext_budget::{
    geometry, weights_from_plan, BudgetError, Ep2Budget, HostBudget, RuntimeParams, StateDtype,
};
use saber_mem::ledger::{Ledger, LedgerError, Pool};

const GIB: f64 = (1u64 << 30) as f64;

/// The pack's `plan.json` `ledger` rows, verbatim (qwen38-flashnext-quark-ep2, FN2).
fn plan_json() -> String {
    let mut rows = Vec::new();
    for r in ["rank0", "rank1"] {
        for (c, b) in [
            ("experts", 32_133_611_520u64),
            ("gdn", 2_086_509_312),
            ("attn.qo", 566_231_040),
            ("attn.kv", 31_457_280),
            ("embed", 635_699_200),
            ("lm_head", 635_699_200),
            ("mtp.experts", 2_516_582_400),
            ("mtp.dense", 62_914_560),
        ] {
            rows.push(serde_json::json!({"placement": r, "class": c, "bytes": b}));
        }
    }
    for (c, b) in [
        ("gdn", 9_216u64),
        ("indexer", 39_340_032),
        ("hc", 1_281_249_280),
        ("router", 126_074_880),
        ("shared_expert", 125_521_920),
        ("ple.dense", 65_679_640),
        ("mtp.dense", 55_307_776),
    ] {
        rows.push(serde_json::json!({"placement": "shared", "class": c, "bytes": b}));
    }
    serde_json::json!({ "ledger": rows }).to_string()
}

fn sum(cs: &[saber_mem::flashnext_budget::Class], f: impl Fn(&str) -> bool) -> u64 {
    cs.iter().filter(|c| f(&c.name)).map(|c| c.bytes).sum()
}

#[test]
fn weights_match_fn1_plus_fn2s_named_deltas() {
    for rank in [0, 1] {
        let (w, experts) = weights_from_plan(&plan_json(), rank).unwrap();
        // The expert class = 12,288 records x the 2,615,040 B payload (FN2: checkpoint + wref).
        assert_eq!(experts, geometry::EXPERT_RECORDS * 2_615_040);
        let is = |cls: &[&str]| {
            let v: Vec<String> = cls.iter().map(|s| s.to_string()).collect();
            move |n: &str| v.iter().any(|c| n.ends_with(&format!(".{c}")))
        };
        let dense = sum(
            &w,
            is(&[
                "gdn",
                "attn.qo",
                "attn.kv",
                "indexer",
                "hc",
                "router",
                "shared_expert",
                "ple.dense",
            ]),
        );
        let eh = sum(&w, is(&["embed", "lm_head"]));
        let mtp = sum(&w, is(&["mtp.experts", "mtp.dense"]));
        // FN2's deltas vs FN1: +8,064 (gdn f32 A_log/dt_bias + the replicated norm), +62,853,120
        // (the shared expert replicated + wref), +4,915,200 (the MTP bf16 shared expert replicated).
        let fn1_dense = (dense - 8_064 - 62_853_120) as f64 / GIB;
        let fn1_mtp = (mtp - 4_915_200) as f64 / GIB;
        // FN1's budget.txt categories give 3.9667 exactly (its LEDGER table's 3.965 is the sum of
        // rows rounded to 3 decimals).
        assert!((fn1_dense - 3.9667).abs() < 0.0001, "dense {fn1_dense}");
        assert!((eh as f64 / GIB - 1.184).abs() < 0.001, "embed+head");
        assert!((fn1_mtp - 2.450).abs() < 0.002, "mtp {fn1_mtp}");
        // The whole rank vs FN2's 37.590 GiB resident line.
        let total = (dense + eh + mtp + experts) as f64 / GIB;
        assert!((total - 37.590).abs() < 0.001, "per rank {total}");
    }
}

#[test]
fn the_default_plan_and_its_boot_line() {
    let (w, _) = weights_from_plan(&plan_json(), 0).unwrap();
    let rt = RuntimeParams::default().classes();
    // The runtime classes from FN1's geometry: KV 13,312 + indexer 832 B/token, GDN 27.0 MiB f16
    // and conv 1.05 MiB per sequence-slot, PLE conv 532,480 B per sequence (FN6's geometry: 9 + 4
    // spec rows in two ping-pong slots; FN7's 180 KiB was the 9-row one-slot minimum, FN9 fixed it).
    let get = |n: &str| rt.iter().find(|c| c.name == n).unwrap().bytes;
    assert_eq!(get("fn.kv"), 131_072 * 13_312);
    assert_eq!(
        get("fn.gdn_state") / 40,
        27 << 20,
        "27.0 MiB per slot at f16"
    );
    assert_eq!(get("fn.gdn_conv") / 40, 1_105_920, "1.05 MiB per slot");
    assert_eq!(
        get("fn.ple_conv") / 8,
        532_480,
        "532,480 B per sequence at MTP-4"
    );
    assert_eq!(
        geometry::ple_conv_bytes_per_seq(4),
        u64::from(saber_kernels_ple_conv_bytes()),
        "== FN6's PleConvState at MTP-4"
    );
    assert_eq!(
        get("fn.ple_conv") - 8 * geometry::PLE_CONV_BYTES_FN7,
        2_785_280,
        "+2.66 MiB per rank vs FN7's line"
    );
    let b = Ep2Budget::plan(
        0,
        30_500 << 20,
        w,
        rt,
        geometry::EXPERT_STRIDE,
        geometry::EXPERT_RECORDS,
    )
    .unwrap();
    assert_eq!(b.weight_bytes(), 8_228_275_736);
    assert_eq!(b.slots, 7_300, "{}", b.boot_line());
    assert_eq!(b.complement(), 4_988);
    let line = b.boot_line();
    assert!(line.contains("expert slots 7300 x 2617344 B"), "{line}");
    // f32 state costs 40 x 27 MiB = 1,080 MiB more = 432.7 strides: 433 slots at this remainder.
    let f32p = RuntimeParams {
        gdn_state: StateDtype::F32,
        ..RuntimeParams::default()
    };
    let b32 = Ep2Budget::plan(
        0,
        30_500 << 20,
        weights_from_plan(&plan_json(), 0).unwrap().0,
        f32p.classes(),
        geometry::EXPERT_STRIDE,
        geometry::EXPERT_RECORDS,
    )
    .unwrap();
    assert_eq!(b.slots - b32.slots, 433);
}

#[test]
fn both_ranks_declare_and_the_ledger_refuses_one_slot_more() {
    let mut l = Ledger::new();
    for rank in [0, 1] {
        let b = Ep2Budget::plan(
            rank,
            30_500 << 20,
            weights_from_plan(&plan_json(), rank).unwrap().0,
            RuntimeParams::default().classes(),
            geometry::EXPERT_STRIDE,
            geometry::EXPERT_RECORDS,
        )
        .unwrap();
        b.declare(&mut l, 0).unwrap();
        assert!(l.free(Pool::Vram(rank)).unwrap() < geometry::EXPERT_STRIDE);
        // THE CONTROL: one slot more than the remainder is an over-commit, refused by name.
        let mut l2 = Ledger::new();
        match b.declare(&mut l2, 1) {
            Err(BudgetError::Ledger(LedgerError::OverCommit { name, .. })) => {
                assert_eq!(name, "xtier.vram_slots");
            }
            other => panic!("rank {rank}: expected an over-commit, got {other:?}"),
        }
    }
    // A runtime that leaves no room is refused before any slot.
    let big = RuntimeParams {
        kv_tokens: 2_000_000,
        ..RuntimeParams::default()
    };
    let e = Ep2Budget::plan(
        0,
        30_500 << 20,
        weights_from_plan(&plan_json(), 0).unwrap().0,
        big.classes(),
        geometry::EXPERT_STRIDE,
        geometry::EXPERT_RECORDS,
    )
    .unwrap_err();
    assert!(matches!(e, BudgetError::NoRoom { .. }), "{e}");
}

#[test]
fn the_host_l2_default_covers_the_complement_on_this_box() {
    // This box: MemTotal 63,336,184 kB.
    let h = HostBudget::this_box(63_336_184 * 1024);
    let l2 = h.default_l2_slots(4_988, geometry::EXPERT_STRIDE);
    assert_eq!(
        l2, 4_988,
        "the complement fits: the NVMe is off the expert step path"
    );
    let mut l = Ledger::new();
    h.declare(&mut l, l2, geometry::EXPERT_STRIDE).unwrap();
    let line = h.boot_line(l2, 4_988, geometry::EXPERT_STRIDE);
    assert!(
        line.contains("covers") && line.contains("complete"),
        "{line}"
    );
    // A 32 GiB box does not: the default shrinks to what fits and the line says demand.
    let small = HostBudget::this_box(32 << 30);
    let l2s = small.default_l2_slots(4_988, geometry::EXPERT_STRIDE);
    assert!(l2s < 4_988);
    assert!(small
        .boot_line(l2s, 4_988, geometry::EXPERT_STRIDE)
        .contains("demand"));
    // And the over-commit control on the host pool.
    let mut l3 = Ledger::new();
    assert!(h.declare(&mut l3, 20_000, geometry::EXPERT_STRIDE).is_err());
}

/// FN8: the prefill staging ring is a ledger line of its own, taken out of the remainder before
/// the LRU's slots; the ring plus the LRU is exactly FN7's remainder, and one slot more is refused.
#[test]
fn the_stage_ring_is_its_own_ledger_line_out_of_the_remainder() {
    let b = Ep2Budget::plan(
        0,
        30_500 << 20,
        weights_from_plan(&plan_json(), 0).unwrap().0,
        RuntimeParams::default().classes(),
        geometry::EXPERT_STRIDE,
        geometry::EXPERT_RECORDS,
    )
    .unwrap();
    assert_eq!(b.slots, 7_300);
    let r = b.clone().with_stage_ring(256).unwrap();
    assert_eq!((r.slots, r.stage_slots), (7_300 - 256, 256));
    let mut l = Ledger::new();
    r.declare(&mut l, 0).unwrap();
    assert!(l.free(Pool::Vram(0)).unwrap() < geometry::EXPERT_STRIDE);
    assert!(
        r.boot_line().contains("stage ring 256 x 2617344 B"),
        "{}",
        r.boot_line()
    );
    let mut l2 = Ledger::new();
    assert!(matches!(
        r.declare(&mut l2, 1),
        Err(BudgetError::Ledger(LedgerError::OverCommit { .. }))
    ));
    assert!(
        b.with_stage_ring(7_300).is_err(),
        "a ring may not take every slot"
    );
}

/// FN6's conv-state bytes per sequence at MTP-4, spelled out from EXEC §13 (saber-mem does not link
/// saber-kernels): H = 9 + 4 rows x 10,240 bf16 x 2 slots.
fn saber_kernels_ple_conv_bytes() -> u32 {
    (9 + 4) * 10_240 * 2 * 2
}

/// FN42: the lending line moves bytes between the KV class and `xtier.kv_lent` and nothing else:
/// the pool's used / free never move, lend 0 leaves no line, a lend past the class is refused, and
/// the lent bytes are never also free (the tier cannot be declared a slot twice).
#[test]
fn the_kv_lending_line_balances_the_ledger_in_every_state() {
    use saber_mem::flashnext_budget::{Class, KV_LENT};
    let (w, _) = weights_from_plan(&plan_json(), 0).unwrap();
    let kv = 954_728_448u64;
    let rt = vec![
        Class {
            name: "fn.pools".into(),
            bytes: kv,
        },
        Class {
            name: "fn.arena.prefill".into(),
            bytes: 800 << 20,
        },
    ];
    let b = Ep2Budget::plan(
        0,
        30_500 << 20,
        w,
        rt,
        geometry::EXPERT_STRIDE,
        geometry::EXPERT_RECORDS,
    )
    .unwrap();
    let mut led = Ledger::new();
    b.declare(&mut led, 0).unwrap();
    let pool = Pool::Vram(0);
    let (used0, free0) = (led.used(pool), led.free(pool));
    for lent in [
        0,
        geometry::EXPERT_STRIDE * 12,
        geometry::EXPERT_STRIDE * 276,
        0,
    ] {
        b.relend(&mut led, "fn.pools", kv, lent).unwrap();
        assert_eq!(
            (led.used(pool), led.free(pool)),
            (used0, free0),
            "lent {lent}"
        );
        let snap = format!("{:?}", led.snapshot());
        assert_eq!(snap.contains(KV_LENT), lent > 0, "{snap}");
    }
    assert!(b.relend(&mut led, "fn.pools", kv, kv + 1).is_err());
    // the lent bytes are not free: one slot more than the remainder is still refused
    b.relend(&mut led, "fn.pools", kv, geometry::EXPERT_STRIDE * 100)
        .unwrap();
    assert!(matches!(
        led.reserve(pool, "xtier.extra", led.free(pool).unwrap() + 1),
        Err(LedgerError::OverCommit { .. })
    ));
}
