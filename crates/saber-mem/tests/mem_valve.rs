//! FN53b: the memory-plane valve -- (1) the controller's state machine on synthetic signals
//! (hysteresis, ordering, rate limit, the dead band, undo order); (2) the tier's valve law over
//! a synthetic `experts.xtier` with the byte-true host mover: a close drops the COLDEST records
//! layer-balanced and relocates the rest of the segment (every slot's bytes == its record), the
//! demand path serves the dropped records, a PREFILL pass's tail goes through the bounce (the L2
//! loses no record to it), and a reopen refills the tier to complete.

use saber_mem::dio::OpenMode;
use saber_mem::expert_mover::HostMover;
use saber_mem::expert_tier::{ExpertMover, ExpertTier, Placement, TierParams, VALVE_SEG};
use saber_mem::mem_valve::{
    mem_available_gib, resolve, vm_rss_gib, MemValve, Move, Signals, ValveParams,
};
use saber_mem::xtier::{write_synthetic, SyntheticTier, XtierFile};
use saber_test_scratch::ScratchDir;

fn sig(t_ms: u64, avail_gib: f64) -> Signals {
    Signals {
        t_ms,
        avail_gib,
        prefill_frac: 0.0,
        own_growth_gib: 0.0,
    }
}

/// Feed `avail` every 100 ms over `[t0, t1)`; return the moves with their times.
fn run(v: &mut MemValve, t0: u64, t1: u64, avail: f64) -> Vec<(u64, Move)> {
    (t0..t1)
        .step_by(100)
        .filter_map(|t| v.tick(sig(t, avail)).map(|m| (t, m)))
        .collect()
}

#[test]
fn the_valve_acts_only_after_a_sustained_window_and_one_move_a_window() {
    let mut v = MemValve::new(ValveParams::default());
    // a dip shorter than the 2 s window: nothing
    assert!(run(&mut v, 0, 1_900, 17.0).is_empty());
    assert!(run(&mut v, 1_900, 3_000, 19.0).is_empty());
    // sustained: the first demotion exactly at the window, then one each window (>= 1 s apart)
    let m = run(&mut v, 3_000, 9_100, 17.0);
    assert_eq!(
        m,
        vec![
            (5_000, Move::Demote(1)),
            (7_000, Move::Demote(2)),
            (9_000, Move::Demote(3))
        ]
    );
    assert_eq!(v.ledger().len(), 3);
    assert!(v.ledger()[0]
        .to_string()
        .starts_with("fnvalve t_ms 5000 move Demote(1)"));
}

#[test]
fn the_escalation_order_is_demote_then_park_then_refuse_then_saturates() {
    let p = ValveParams {
        park: true,
        refuse: true,
        act_ms: 0,
        ..ValveParams::default()
    };
    let mut v = MemValve::new(p);
    let m: Vec<Move> = run(&mut v, 0, 20_000, 10.0)
        .into_iter()
        .map(|x| x.1)
        .collect();
    assert_eq!(
        m,
        vec![
            Move::Demote(1),
            Move::Demote(2),
            Move::Demote(3),
            Move::Demote(4),
            Move::Park,
            Move::Refuse
        ]
    );
    // the rate limit: the moves are >= 1 s apart even with a zero window
    let t: Vec<u64> = v.ledger().iter().map(|l| l.at.t_ms).collect();
    assert!(t.windows(2).all(|w| w[1] - w[0] >= 1_000), "{t:?}");
    // without park / refuse the valve saturates after the band
    let mut w = MemValve::new(ValveParams {
        act_ms: 0,
        ..ValveParams::default()
    });
    assert_eq!(run(&mut w, 0, 60_000, 10.0).len(), 4);
}

#[test]
fn the_dead_band_never_moves_and_the_undo_runs_in_reverse_after_30_s() {
    let p = ValveParams {
        park: true,
        refuse: true,
        act_ms: 0,
        ..ValveParams::default()
    };
    let mut v = MemValve::new(p);
    run(&mut v, 0, 7_000, 10.0); // 4 demotions + park + refuse
    assert!(v.refusing() && v.parked() && v.closed() == 4);
    // between floor + 2 and floor + 4: nothing, for any time
    assert!(run(&mut v, 7_000, 200_000, 19.0).is_empty());
    // above floor + 4 but a dip resets the 30 s window
    assert!(run(&mut v, 200_000, 229_000, 20.5).is_empty());
    assert!(run(&mut v, 229_000, 229_200, 19.5).is_empty());
    let m = run(&mut v, 229_200, 229_200 + 6 * 30_000 + 100, 20.5);
    assert_eq!(
        m,
        vec![
            (259_200, Move::Admit),
            (289_200, Move::Unpark),
            (319_200, Move::Promote(3)),
            (349_200, Move::Promote(2)),
            (379_200, Move::Promote(1)),
            (409_200, Move::Promote(0)),
        ]
    );
    assert!(run(&mut v, 409_300, 500_000, 25.0).is_empty());
}

/// FN53c: the BOOT LAW -- the served 262k boot's idle MemAvailable (~17-19 GiB) sits AT FN53b's
/// constant trigger (18): the constant law demotes an idle boot (the control), the boot law never
/// does; a NEW job eating RAM below the boot floor minus the margin demotes; the engine's own RSS
/// growth is not pressure; the reopen sits above the trigger by more than a segment pair and below
/// the boot floor, so the tier reopens whole and a reopen does not re-trigger.
#[test]
fn the_trigger_is_the_boot_floor_minus_a_margin_and_an_idle_boot_never_moves() {
    let p = ValveParams::default();
    for (boot, trig, reopen) in [
        (18.0, 16.0, 17.5),
        (17.0, 16.0, 17.5),
        (19.0, 17.0, 18.5),
        (30.0, 28.0, 29.5),
        (15.0, 16.0, 17.5),
    ] {
        let th = resolve(&p, boot);
        assert!((th.trigger_gib - trig).abs() < 1e-9, "boot {boot}: {th}");
        assert!((th.reopen_gib - reopen).abs() < 1e-9, "boot {boot}: {th}");
        assert!(th.reopen_gib - th.trigger_gib >= p.reopen_gap_gib - 1e-9);
        assert!(th.to_string().starts_with("law boot: boot free"));
    }
    // the control: an idle 262k boot at 17.6 GiB for 10 minutes -- the constant law demotes
    let mut c = MemValve::new(p.clone());
    let m = run(&mut c, 0, 600_000, 17.6);
    assert!(!m.is_empty() && m[0].1 == Move::Demote(1), "{m:?}");
    // the boot law on the same boot: nothing in 10 minutes, idle or with its own growth
    let mut v = MemValve::at_boot(p.clone(), 17.6);
    assert!(run(&mut v, 0, 600_000, 17.6).is_empty());
    let grown: Vec<_> = (0..6_000u64)
        .filter_map(|k| {
            v.tick(Signals {
                t_ms: 600_000 + 100 * k,
                avail_gib: 14.6,
                prefill_frac: 0.0,
                own_growth_gib: 3.0,
            })
        })
        .collect();
    assert!(
        grown.is_empty(),
        "the engine's own growth moved the valve: {grown:?}"
    );
    // NEW pressure: a job takes 3 GiB -> effective 14.6 < 16 -> demote after the window
    let mut v = MemValve::at_boot(p.clone(), 17.6);
    let m = run(&mut v, 0, 9_000, 14.6);
    assert_eq!(m[0], (2_000, Move::Demote(1)));
    assert_eq!(m.len(), 4);
    // the job leaves: 4 segments demoted freed ~5 GiB on top of the boot floor -> reopen one at
    // a time, each reopen takes a segment pair (1.24 GiB) and never lands below the trigger
    let mut avail = 17.6 + 4.0 * 1.24;
    let mut t = 9_000;
    while v.closed() > 0 && t < 400_000 {
        if let Some(Move::Promote(_)) = v.tick(sig(t, avail)) {
            avail -= 1.24;
            assert!(avail > v.thresholds().trigger_gib);
        }
        t += 100;
    }
    assert_eq!(v.closed(), 0, "the tier reopened whole");
    assert!(run(&mut v, t, t + 600_000, avail).is_empty(), "no flap");
}

#[test]
fn vm_rss_parses_proc_status() {
    let st = "Name:\tsaber\nVmPeak:\t 1 kB\nVmRSS:\t36700160 kB\nThreads:\t9\n";
    assert!((vm_rss_gib(st).unwrap() - 35.0).abs() < 1e-9);
    assert!(vm_rss_gib("Name: x\n").is_none());
}

#[test]
fn mem_available_parses_proc_meminfo() {
    let t = "MemTotal:       61440000 kB\nMemFree:  100 kB\nMemAvailable:   16777216 kB\n";
    assert_eq!(mem_available_gib(t), Some(16.0));
    assert_eq!(mem_available_gib("MemTotal: 1 kB\n"), None);
}

// ---- the tier's valve law ----

const LAYERS: u32 = 8;
const EXPERTS: u32 = 64;
const VRAM_BASE: u64 = 0x7f00_0000_0000;
const RAM_BASE: u64 = 0x5500_0000_0000;

fn synth(tag: &str) -> (ScratchDir, XtierFile) {
    let d = ScratchDir::new(&format!("saber-fn53b-{tag}"));
    let p = d.join("experts.xtier");
    write_synthetic(
        &p,
        SyntheticTier {
            rank: 0,
            layers: LAYERS,
            first_expert: 0,
            experts: EXPERTS,
            hidden: 128,
            inter: 32,
        },
    )
    .unwrap();
    let f = XtierFile::open(&p, OpenMode::Buffered).unwrap();
    (d, f)
}

fn params() -> TierParams {
    TierParams {
        vram_slots: 128,
        ram_slots: 128,
        min_layer_slots: 4,
        max_inserts: 2,
        lrfu_lambda: 0.98,
        band_slots: 256,
        bounce_slots: 2 * 48,
        ..TierParams::default()
    }
}

fn slab(p: &TierParams) -> u32 {
    ExpertTier::slab_slots(p) + p.band_slots + p.bounce_slots
}

/// Every placed record's slot holds its bytes, and every xptr entry of a routed record lands
/// on a slot holding it.
fn check(tier: &ExpertTier, mv: &HostMover, xt: &XtierFile, routed: Option<(u32, &[u32])>) {
    tier.check_invariants().unwrap();
    let st = xt.header().stride;
    for i in 0..LAYERS * EXPERTS {
        match tier.placement(i / EXPERTS, i % EXPERTS).unwrap() {
            Placement::Vram(s) => assert!(xt.verify(u64::from(i), mv.vram_slot(s))),
            Placement::Ram(s) => assert!(xt.verify(u64::from(i), mv.ram_slot(s)), "rec {i}"),
            Placement::Fetch => {}
        }
    }
    if let Some((l, rs)) = routed {
        let n = EXPERTS as usize;
        let (mut gu, mut dn) = (vec![0u64; 3 * (n + 1)], vec![0u64; 3 * (n + 1)]);
        tier.fill_xptr(l, &mut gu, &mut dn);
        for &e in rs {
            let a = gu[3 * e as usize];
            let bytes = if a >= VRAM_BASE {
                mv.vram_slot(((a - VRAM_BASE) / st) as u32)
            } else {
                mv.ram_slot(((a - RAM_BASE) / st) as u32)
            };
            assert!(xt.verify(u64::from(l * EXPERTS + e), bytes), "L{l} e{e}");
        }
    }
}

fn copy_slots(mv: &mut HostMover, st: usize, relocs: &[(u32, u32)]) {
    for &(f, t) in relocs {
        let (f, t) = (f as usize * st, t as usize * st);
        mv.ram.copy_within(f..f + st, t);
    }
}

/// A decode-like step: each layer routes a skewed handful of experts.
fn decode_step(tier: &mut ExpertTier, mv: &mut HostMover, xt: &XtierFile, k: u64, div: u32) {
    tier.set_pass_prefill(false);
    tier.begin_step();
    for l in 0..LAYERS {
        let rs: Vec<u32> = (0..6)
            .map(|j| ((k * 7 + j * 13 + u64::from(l)) % u64::from(EXPERTS)) as u32 / div)
            .collect();
        tier.resolve(l, &rs, mv).unwrap();
        tier.check_dispatch(l, &rs).unwrap();
        check(tier, mv, xt, Some((l, &rs)));
    }
    tier.end_step(mv).unwrap();
    check(tier, mv, xt, None);
}

#[test]
fn a_close_drops_the_coldest_layer_balanced_and_a_reopen_completes_the_tier() {
    let (_d, xt) = synth("close");
    let p = params();
    let mut tier = ExpertTier::new(xt.header(), p.clone(), None, VRAM_BASE, RAM_BASE).unwrap();
    let mut mv = HostMover::new(&xt, p.vram_slots, slab(&p) - 1, 4);
    mv.warm(&tier.warm_plan(), p.ram_slots).unwrap();
    assert!(tier.complete());
    let st = xt.header().stride as usize;
    for k in 0..20 {
        decode_step(&mut tier, &mut mv, &xt, k, 2);
    }
    let (lo, n, open) = tier.band();
    assert_eq!((lo, n, open), (ExpertTier::slab_slots(&p), 256, 256));
    // the coldest: records never routed (odd local experts >= 32 never route above)
    let c = tier.valve_close().unwrap();
    copy_slots(&mut mv, st, &c.relocs);
    assert_eq!(c.dropped.len(), VALVE_SEG as usize);
    assert!(!tier.complete());
    let mut per = [0u32; LAYERS as usize];
    for &i in &c.dropped {
        per[(i / EXPERTS) as usize] += 1;
    }
    assert!(per.iter().all(|&x| x == 32), "layer-balanced: {per:?}");
    assert_eq!(tier.band().2, 0);
    check(&tier, &mv, &xt, None);
    // no placement points into the closed segment
    for i in 0..LAYERS * EXPERTS {
        if let Some(Placement::Ram(s)) = tier.placement(i / EXPERTS, i % EXPERTS) {
            assert!(s < lo || s >= lo + n, "record {i} in the closed band");
        }
    }
    // the demand path serves dropped records (decode misses enter the L2)
    for k in 20..60 {
        decode_step(&mut tier, &mut mv, &xt, k * 3 + 1, 1);
    }
    assert!(tier.totals().0.nvme > 0);
    // a second close of a closed band is refused by name
    assert!(tier.valve_close().is_err());
    // reopen: refill the L2 from the drive -> complete
    let reads = tier
        .valve_open(RAM_BASE + u64::from(lo) * st as u64)
        .unwrap();
    mv.nvme_to_ram(&reads).unwrap();
    assert!(tier.complete(), "every record placed after the reopen");
    check(&tier, &mv, &xt, None);
    decode_step(&mut tier, &mut mv, &xt, 99, 1);
    let vc = tier.valve_counts();
    assert_eq!((vc.closes, vc.opens, vc.dropped), (1, 1, 256));
}

#[test]
fn a_prefill_pass_reads_the_tail_through_the_bounce_and_the_l2_keeps_every_record() {
    let (_d, xt) = synth("bounce");
    let p = params();
    let mut tier = ExpertTier::new(xt.header(), p.clone(), None, VRAM_BASE, RAM_BASE).unwrap();
    let mut mv = HostMover::new(&xt, p.vram_slots, slab(&p) - 1, 4);
    mv.warm(&tier.warm_plan(), p.ram_slots).unwrap();
    let st = xt.header().stride as usize;
    let c = tier.valve_close().unwrap();
    copy_slots(&mut mv, st, &c.relocs);
    let before: Vec<Option<Placement>> = (0..LAYERS * EXPERTS)
        .map(|i| tier.placement(i / EXPERTS, i % EXPERTS))
        .collect();
    // a wide prefill chunk: every layer routes every expert; layer-ahead reads before resolve
    tier.set_pass_prefill(true);
    tier.begin_step();
    let all: Vec<u32> = (0..EXPERTS).collect();
    for l in 0..LAYERS {
        let got = tier.bounce_prefetch(l, &mut mv).unwrap();
        assert_eq!(got, 32, "layer {l}: its whole balanced tail");
        tier.resolve(l, &all, &mut mv).unwrap();
        tier.check_dispatch(l, &all).unwrap();
        check(&tier, &mv, &xt, Some((l, &all)));
    }
    tier.end_step(&mut mv).unwrap();
    let vc = tier.valve_counts();
    assert_eq!(
        (
            vc.bounce_prefetch,
            vc.bounce_hits,
            vc.bounce_demand,
            vc.bounce_full
        ),
        (256, 256, 0, 0)
    );
    // the L2 lost nothing to the chunk: every placement is what it was before the pass
    let after: Vec<Option<Placement>> = (0..LAYERS * EXPERTS)
        .map(|i| tier.placement(i / EXPERTS, i % EXPERTS))
        .collect();
    assert_eq!(before, after);
    assert_eq!(tier.totals().1.ram_drops, 0);
    // a short chunk (no prefetch) reads its tail on demand into the bounce
    tier.begin_step();
    tier.resolve(3, &all, &mut mv).unwrap();
    check(&tier, &mv, &xt, Some((3, &all)));
    tier.end_step(&mut mv).unwrap();
    assert_eq!(tier.valve_counts().bounce_demand, 32);
    // control: the same chunk WITHOUT the bypass (a decode pass) inserts into the L2 and drops
    tier.set_pass_prefill(false);
    tier.begin_step();
    for l in 0..LAYERS {
        tier.resolve(l, &all, &mut mv).unwrap();
    }
    tier.end_step(&mut mv).unwrap();
    assert!(tier.totals().1.ram_drops > 0, "the control thrashes the L2");
}

/// FN53c: THE DEVICE MISS FLAG's tier law -- under `set_pass_xmiss` a decode step's `Fetch`
/// routes are read into the layer's bounce half (`xmiss_slots`: each entry the record's bytes in
/// its slot, the tables patched as the kernel does), the routes count as served and STAY `Fetch`
/// (the open set is static: no L2 insert, no drop), every other placement untouched; a resident
/// record or more misses than a half holds are refused by name; the control (the same steps
/// without the flag = FN53b's demand path) inserts the misses into the L2.
#[test]
fn the_miss_flag_serves_fetch_routes_from_the_bounce_and_keeps_the_open_set_static() {
    let (_d, xt) = synth("xmiss");
    let p = params();
    let mut tier = ExpertTier::new(xt.header(), p.clone(), None, VRAM_BASE, RAM_BASE).unwrap();
    let mut mv = HostMover::new(&xt, p.vram_slots, slab(&p) - 1, 4);
    mv.warm(&tier.warm_plan(), p.ram_slots).unwrap();
    let st = xt.header().stride as usize;
    for k in 0..20 {
        decode_step(&mut tier, &mut mv, &xt, k, 2);
    }
    let c = tier.valve_close().unwrap();
    copy_slots(&mut mv, st, &c.relocs);
    let fetch = |t: &ExpertTier| -> Vec<u32> {
        (0..LAYERS * EXPERTS)
            .filter(|&i| t.placement(i / EXPERTS, i % EXPERTS) == Some(Placement::Fetch))
            .collect()
    };
    let open = fetch(&tier);
    assert_eq!(open.len(), VALVE_SEG as usize);
    let n = EXPERTS as usize;
    for k in 0..12u32 {
        tier.set_pass_prefill(false);
        tier.set_pass_xmiss(true);
        tier.begin_step();
        for l in 0..LAYERS {
            // three of the layer's open records + three resident ones, rotating
            let tail: Vec<u32> = open
                .iter()
                .filter(|&&i| i / EXPERTS == l)
                .map(|&i| i % EXPERTS)
                .collect();
            let mut rs: Vec<u32> = (0..3)
                .map(|j| tail[(k as usize * 3 + j) % tail.len()])
                .collect();
            rs.extend(
                (0..EXPERTS)
                    .filter(|e| !tail.contains(e))
                    .skip(k as usize)
                    .take(3),
            );
            let (mut gu, mut dn) = (vec![0u64; 3 * (n + 1)], vec![0u64; 3 * (n + 1)]);
            tier.fill_xptr(l, &mut gu, &mut dn);
            // the kernel's flag: the routed records with a 0 entry, in expert order, once each
            let mut miss: Vec<u32> = rs
                .iter()
                .copied()
                .filter(|&e| gu[3 * e as usize] == 0)
                .collect();
            miss.sort_unstable();
            miss.dedup();
            assert_eq!(miss.len(), 3, "L{l}");
            let slots = tier.xmiss_slots(l, &miss).unwrap();
            let reads: Vec<(u64, u32)> = slots
                .iter()
                .map(|s| (u64::from(s.record), s.slot))
                .collect();
            mv.nvme_to_ram(&reads).unwrap();
            for (s, &e) in slots.iter().zip(&miss) {
                assert_eq!(s.record, l * EXPERTS + e);
                for k3 in 0..3 {
                    gu[3 * e as usize + k3] = s.entries[k3];
                    dn[3 * e as usize + k3] = s.entries[3 + k3];
                }
            }
            for &e in &rs {
                let a = gu[3 * e as usize];
                let bytes = if a >= VRAM_BASE {
                    mv.vram_slot(((a - VRAM_BASE) / st as u64) as u32)
                } else {
                    mv.ram_slot(((a - RAM_BASE) / st as u64) as u32)
                };
                assert!(xt.verify(u64::from(l * EXPERTS + e), bytes), "L{l} e{e}");
            }
            let pc = tier.resolve(l, &rs, &mut mv).unwrap();
            assert_eq!(pc.nvme, 3);
        }
        tier.end_step(&mut mv).unwrap();
        assert_eq!(fetch(&tier), open, "step {k}: the open set is static");
        check(&tier, &mv, &xt, None);
    }
    let vc = tier.valve_counts();
    assert_eq!(
        (vc.xmiss_requests, vc.xmiss_records, vc.xmiss_routes),
        (96, 288, 288)
    );
    assert_eq!(tier.totals().1.ram_drops, 0);
    // refusals by name: a resident record; more misses than a bounce half holds
    let resident = (0..EXPERTS)
        .find(|&e| tier.placement(0, e) != Some(Placement::Fetch))
        .unwrap();
    assert!(tier.xmiss_slots(0, &[resident]).is_err());
    let many: Vec<u32> = (0..p.bounce_slots / 2 + 1).collect();
    assert!(tier.xmiss_slots(0, &many).is_err());
    // the flag never rides a prefill pass
    tier.set_pass_prefill(true);
    tier.set_pass_xmiss(true);
    tier.set_pass_prefill(false);
    // the control: FN53b's demand path on the same routes enters the L2
    for k in 0..12 {
        tier.set_pass_xmiss(false);
        decode_step(&mut tier, &mut mv, &xt, k * 3 + 1, 1);
    }
    assert!(fetch(&tier) != open, "the demand path moves the open set");
}
