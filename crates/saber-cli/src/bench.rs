//! `saber bench-steploop` — P0.5, the step loop with no model.
//!
//! Capture N tiny kernels into a graph, instantiate it, and replay it once per step request while
//! a tokio task feeds those requests through a bounded channel. The comparator is the traced host
//! chain: **8.5 ms at conc-8, ~3 ms single-stream, ≤1.5 ms of it exposed** (2026-09-04), at prod's
//! **1084 launches/step**.
//!
//! Buckets come from the manifest, so the M the dummy kernel runs at is the M the engine will
//! actually see (`batch × spec_width`).

use std::path::PathBuf;

use saber_exec::steploop::{self, BucketShape, StepLoopConfig, StepLoopRow};
use saber_hip::HipThread;
use saber_plan::Manifest;

/// Arguments of `saber bench-steploop`.
pub struct BenchArgs {
    /// The manifest whose buckets are measured.
    pub manifest: Option<PathBuf>,
    /// The committed code objects.
    pub prebuilt: PathBuf,
    /// Launches per captured graph.
    pub launches: Vec<u32>,
    /// Timed replays per row.
    pub steps: usize,
    /// Replays discarded before timing.
    pub warmup: usize,
    /// Bounded-channel depth between the tokio feeder and the step loop.
    pub channel_capacity: usize,
    /// Width of the dummy kernel.
    pub dummy_n: u32,
    /// HIP device ordinal within the visible set.
    pub device: i32,
}

/// Run the bench and print the table.
pub fn run(hip: &HipThread, args: &BenchArgs) -> anyhow::Result<()> {
    hip.set_device(args.device)?;

    let buckets = match &args.manifest {
        Some(path) => {
            let m = Manifest::from_path(path)?;
            m.buckets
                .iter()
                .map(|b| BucketShape {
                    batch: b.batch,
                    spec_width: b.spec_width,
                })
                .collect::<Vec<_>>()
        }
        None => StepLoopConfig::default().buckets,
    };
    let cfg = StepLoopConfig {
        launches_per_step: args.launches.clone(),
        buckets,
        steps: args.steps,
        warmup: args.warmup,
        channel_capacity: args.channel_capacity,
        dummy_n: args.dummy_n,
    };

    println!(
        "step loop      dummy kernel `silu_mul_quant` at width {}, one graph per (bucket, node \
         count)",
        cfg.dummy_n
    );
    println!(
        "               {} timed replays after {} warm-ups, bounded channel depth {}",
        cfg.steps, cfg.warmup, cfg.channel_capacity
    );
    println!("               capture mode ThreadLocal, on the HipThread; feeder is a tokio task");
    println!();

    let rows = steploop::bench(hip, &args.prebuilt, &cfg)?;

    println!("{}", StepLoopRow::header());
    for r in &rows {
        println!("{}", r.row());
        // Rule 7: the count is declared, and the capture asserted it. Say so per row rather than
        // hoping a reader trusts it.
        assert_eq!(
            r.launches_issued, r.launches,
            "launch budget: {} declared, {} captured",
            r.launches, r.launches_issued
        );
    }
    println!();
    println!(
        "launches/step asserted at capture for every row ({} rows, budgets {:?})",
        rows.len(),
        cfg.launches_per_step
    );
    if let Some(r) = rows.first() {
        println!(
            "one HipThread channel hop measured at {:.2} us this run; every `sub`/`step` figure \
             above contains one or two of them.",
            r.hip_thread_hop_us
        );
    }
    println!(
        "comparator: the traced prod host chain is 8.5 ms at conc-8 / ~3 ms single-stream, \
         <=1.5 ms exposed (2026-09-04), at 1084 launches/step."
    );
    Ok(())
}
