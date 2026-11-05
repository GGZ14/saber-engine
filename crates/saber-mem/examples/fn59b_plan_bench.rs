//! FN59b host bench (no GPU): the expert tier's end-of-step PLANNING cost over a served route trace,
//! today's full victim sort vs the cheap selection (`set_fast_plan`), run in LOCKSTEP on the same
//! steps -- every step's placements and move counters must be identical (the host proof), and the
//! `end_step` host time of each is reported by bucket (counting mover: no copy is issued, so this
//! is the planning alone).
//!
//!   fn59b_plan_bench <trace.bin> <profile.json|-> <vram> <ram> <rank> [max_steps]

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use saber_mem::expert_tier::{ExpertTier, RouteProfile, TierParams};
use saber_mem::route_trace::{replay_header, CountMover, StepKind, TraceReader};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert!(
        a.len() >= 6,
        "usage: fn59b_plan_bench <trace> <profile|-> <vram> <ram> <rank> [max]"
    );
    let prof = if a[2] == "-" {
        None
    } else {
        Some(RouteProfile::from_json(&std::fs::read_to_string(&a[2]).unwrap()).unwrap())
    };
    let vram: u32 = a[3].parse().unwrap();
    let ram: u32 = a[4].parse().unwrap();
    let rank: u32 = a[5].parse().unwrap();
    let max: u64 = a.get(6).map_or(u64::MAX, |s| s.parse().unwrap());
    let mut r = TraceReader::open(Path::new(&a[1])).unwrap();
    let per_rank = r.experts / 2;
    let h = replay_header(rank, r.layers, per_rank, 1 << 20);
    // the served law (xtier_flags defaults): LRFU 0.98, cap 2, read-through 0.5, FN8's law
    let p = TierParams {
        vram_slots: vram,
        ram_slots: ram,
        max_inserts: 2,
        lrfu_lambda: 0.98,
        ..TierParams::default()
    };
    let mut t0 = ExpertTier::new(&h, p.clone(), prof.as_ref(), 0, 1 << 40).unwrap();
    let mut t1 = ExpertTier::new(&h, p, prof.as_ref(), 0, 1 << 40).unwrap();
    t1.set_fast_plan(true);
    let (mut m0, mut m1) = (CountMover::default(), CountMover::default());
    // bucket -> (steps, legacy ms, fast ms, inserts)
    let mut by: BTreeMap<u32, (u64, f64, f64, u64)> = BTreeMap::new();
    let mut steps = 0u64;
    while let Some(s) = r.next_step().unwrap() {
        if !s.kind.target() {
            continue;
        }
        if steps >= max {
            break;
        }
        let bucket = if s.kind == StepKind::Prefill {
            0
        } else {
            s.bucket
        };
        t0.begin_step();
        t1.begin_step();
        for l in 0..s.layers() {
            let routed: Vec<u32> = s.layer(l).iter().map(|x| u32::from(x.e)).collect();
            let c0 = t0.resolve(l as u32, &routed, &mut m0).unwrap();
            let c1 = t1.resolve(l as u32, &routed, &mut m1).unwrap();
            assert_eq!(c0, c1, "step {steps} layer {l}: paths differ");
        }
        let q = Instant::now();
        let r0 = t0.end_step(&mut m0).unwrap();
        let d0 = q.elapsed().as_secs_f64() * 1e3;
        let q = Instant::now();
        let r1 = t1.end_step(&mut m1).unwrap();
        let d1 = q.elapsed().as_secs_f64() * 1e3;
        assert_eq!(r0.moves, r1.moves, "step {steps}: moves differ");
        assert!(
            t0.placements() == t1.placements(),
            "step {steps}: placements differ"
        );
        let e = by.entry(bucket).or_default();
        e.0 += 1;
        e.1 += d0;
        e.2 += d1;
        e.3 += r0.moves.inserts;
        steps += 1;
    }
    assert_eq!((m0.h2d, m0.d2h), (m1.h2d, m1.d2h), "copy counts differ");
    println!(
        "fn59b-plan rank {rank} vram {vram} ram {ram}: {steps} steps, placements IDENTICAL every step"
    );
    for (b, (n, l, f, ins)) in by {
        let n = n as f64;
        println!(
            "  bucket {b}: {n} steps, end_step legacy {:.3} ms  fast {:.3} ms  ({:.1}x), inserts/step {:.2}",
            l / n,
            f / n,
            l / f.max(1e-9),
            ins as f64 / n
        );
    }
}
