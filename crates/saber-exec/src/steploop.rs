//! P0.5 — the step loop with no model.
//!
//! > `saber-exec` capture/instantiate/replay of N dummy kernels at prod's bucket sizes on the
//! > `HipThread`; tokio HTTP thread issuing steps through a bounded channel. Measures the host
//! > chain in µs/step and validates `hipStreamCaptureModeThreadLocal` discipline before any real
//! > kernel can be blocked by it. Launch census wired here, from day one.
//! >
//! > — IMPLEMENTATION-PLAN.md, P0.5
//!
//! The number this produces is the comparator for the traced host chain: **8.5 ms at conc-8,
//! ~3 ms single-stream, of which ≤1.5 ms is exposed** (`async-scheduling-closed`, 2026-09-04).
//! Prod runs **1084 launches/step**, so `n = 1084` is the row that matters and the other two are
//! there to show the shape of the curve.
//!
//! # Shape of the harness
//!
//! * The **step loop is a plain `std::thread`** — the caller's. It wants fixed cadence, and a
//!   work-stealing scheduler adds variance to exactly that path.
//! * A **tokio task** feeds step requests through a **bounded** channel
//!   ([`tokio::sync::mpsc::channel`]), and the loop takes them with `blocking_recv`. Bounded is the
//!   point: an unbounded queue would measure the producer, not the loop.
//! * Every HIP call — capture, instantiate, replay, sync — runs on the [`HipThread`].
//!   [`saber_hip::StreamCaptureMode::ThreadLocal`] is then correct by construction rather than by
//!   convention, because there is no other thread to serialise against.
//!
//! # The dummy kernel
//!
//! One of the committed quant producers at a one-row problem (`silu_mul_quant`, M = the bucket's
//! `batch × spec_width`, N = 5120). It is a real, bit-exact-gated code object launched through the
//! real [`crate::Launcher`], so what is being timed is the engine's own submit path and not a
//! special case. No kernel source is added: `crates/saber-kernels` belongs to another workstream.
//!
//! # Capture cannot happen off the HIP thread
//!
//! Not by discipline — by type. [`saber_hip::Stream`] is `!Send`, so a stream cannot reach another
//! thread to be captured there, and the capture entry points take one:
//!
//! ```compile_fail
//! use saber_hip::{HipThread, StreamFlags, DEFAULT_HIP_SONAME};
//!
//! let hip = HipThread::spawn(DEFAULT_HIP_SONAME).unwrap();
//! let stream = hip.create_stream(StreamFlags::NonBlocking).unwrap();
//! // error[E0277]: `*mut ()` cannot be sent between threads safely
//! std::thread::spawn(move || {
//!     let _capture_me = stream;
//! });
//! ```
//!
//! The same program with a `Send` value in place of the stream compiles, which is the negative
//! control for the test above:
//!
//! ```
//! let addr = saber_hip::DeviceAddr::NULL;
//! std::thread::spawn(move || {
//!     let _ = addr;
//! });
//! ```

use std::time::{Duration, Instant};

use saber_hip::{EventFlags, HipThread, StreamCaptureMode, StreamFlags};
use saber_kernels::rules::{resolve, LaunchRules, Op, Shape};

use crate::counters::{LaunchCounters, LoadCounters};
use crate::error::{Error, Result};
use crate::launch::{ArgBindings, LaunchCtx, Launcher};
use crate::loader::PluginSet;

/// The plugin the dummy kernel comes from, and the op it resolves to.
const DUMMY_PLUGIN: &str = "mxfp4_quant";
/// Default N for the dummy `silu_mul_quant`: a production MLP width at TP=2, so the kernel is one
/// the bit-exact gate already covers. Lowering it (`--dummy-n 8`) is how the per-node cost is
/// separated from the dummy kernel's own work: the difference between the two runs is the kernel,
/// what is left is the node.
pub const DUMMY_N: u32 = 5120;

/// One bucket to measure: `batch × spec_width`, exactly as the manifest declares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BucketShape {
    /// Sequences in flight.
    pub batch: u32,
    /// Draft tokens verified per step.
    pub spec_width: u32,
}

impl BucketShape {
    /// Rows the GEMMs run at in this bucket, matching `saber_plan::Bucket::m`.
    #[must_use]
    pub fn m(self) -> u32 {
        self.batch * self.spec_width
    }

    /// Short stable label.
    #[must_use]
    pub fn label(self) -> String {
        format!("b{}s{}", self.batch, self.spec_width)
    }
}

/// What to measure.
#[derive(Debug, Clone)]
pub struct StepLoopConfig {
    /// Launches per captured graph. Prod is 1084/step.
    pub launches_per_step: Vec<u32>,
    /// Buckets, from the manifest.
    pub buckets: Vec<BucketShape>,
    /// Replays per row, after the warm-ups.
    pub steps: usize,
    /// Replays discarded before timing starts. The first-measured variant per shape reads ~17%
    /// inflated on this box; that rule is about autotune, but the reason (cold caches, cold clocks)
    /// applies to a replay just as much.
    pub warmup: usize,
    /// Bounded-channel depth between the tokio feeder and the step loop.
    pub channel_capacity: usize,
    /// Width of the dummy kernel. Must satisfy the `silu_mul_quant` rules (N % 8, N <= 20480).
    pub dummy_n: u32,
}

impl Default for StepLoopConfig {
    fn default() -> Self {
        StepLoopConfig {
            launches_per_step: vec![100, 500, 1084],
            buckets: vec![
                BucketShape {
                    batch: 1,
                    spec_width: 5,
                },
                BucketShape {
                    batch: 8,
                    spec_width: 5,
                },
            ],
            steps: 200,
            warmup: 20,
            channel_capacity: 8,
            dummy_n: DUMMY_N,
        }
    }
}

/// Order statistics of a sample, in microseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    /// Sample count.
    pub n: usize,
    /// Smallest.
    pub min_us: f64,
    /// Median.
    pub p50_us: f64,
    /// Arithmetic mean.
    pub mean_us: f64,
    /// 99th percentile (nearest rank).
    pub p99_us: f64,
}

impl Stats {
    /// Order statistics of `samples`. Empty input gives all zeros rather than a panic: an empty
    /// row is a reportable outcome, not a crash in a benchmark.
    #[must_use]
    pub fn of(samples: &[Duration]) -> Stats {
        if samples.is_empty() {
            return Stats {
                n: 0,
                min_us: 0.0,
                p50_us: 0.0,
                mean_us: 0.0,
                p99_us: 0.0,
            };
        }
        let mut us: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1e6).collect();
        us.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a duration"));
        let n = us.len();
        let sum: f64 = us.iter().sum();
        Stats {
            n,
            min_us: us[0],
            p50_us: us[n / 2],
            mean_us: sum / n as f64,
            p99_us: us[(n * 99 / 100).min(n - 1)],
        }
    }
}

/// One measured row.
#[derive(Debug, Clone)]
pub struct StepLoopRow {
    /// The bucket.
    pub bucket: BucketShape,
    /// Rows the dummy kernel ran at.
    pub m: u32,
    /// Width the dummy kernel ran at.
    pub n: u32,
    /// Launches captured into the graph, and therefore issued per replay.
    pub launches: u32,
    /// Dispatches the launcher actually issued while capturing — asserted equal to `launches`.
    pub launches_issued: u32,
    /// Time from taking a step request to `hipGraphLaunch` returning: the pure host submit cost.
    pub submit: Stats,
    /// Time from taking a step request to the replay having completed on the device.
    pub step: Stats,
    /// Time the loop spent blocked on the bounded channel. Near zero means the feeder kept up and
    /// the numbers above are the loop's own cost.
    pub recv: Stats,
    /// GPU time for the replay, from HIP events: what is *not* host chain.
    pub gpu_ms: f64,
    /// Round-trip cost of one `HipThread` channel hop, measured the same run with a trivial call.
    /// Every figure above contains one or two of these; this is what lets a reader subtract them.
    pub hip_thread_hop_us: f64,
}

impl StepLoopRow {
    /// Device microseconds per captured node: the replay's GPU time divided by the launch count.
    /// Includes the dummy kernel's own work, which is why `--dummy-n` exists.
    #[must_use]
    pub fn per_node_us(&self) -> f64 {
        if self.launches == 0 {
            return 0.0;
        }
        self.gpu_ms * 1000.0 / f64::from(self.launches)
    }

    /// One row of the report table.
    #[must_use]
    pub fn row(&self) -> String {
        format!(
            "{:>6} {:>4} {:>6} {:>6} {:>8.1} {:>8.1} {:>9.1} {:>9.1} {:>8.1} {:>8.3} {:>8.2} {:>7.2}",
            self.bucket.label(),
            self.m,
            self.n,
            self.launches,
            self.submit.p50_us,
            self.submit.mean_us,
            self.step.p50_us,
            self.step.mean_us,
            self.recv.p50_us,
            self.gpu_ms,
            self.per_node_us(),
            self.hip_thread_hop_us,
        )
    }

    /// The table header matching [`StepLoopRow::row`].
    #[must_use]
    pub fn header() -> String {
        format!(
            "{:>6} {:>4} {:>6} {:>6} {:>8} {:>8} {:>9} {:>9} {:>8} {:>8} {:>8} {:>7}",
            "bucket",
            "M",
            "width",
            "nodes",
            "sub p50",
            "sub mean",
            "step p50",
            "step mean",
            "recv p50",
            "gpu ms",
            "us/node",
            "hop us",
        )
    }
}

/// A step request. Deliberately trivial: P0.5 measures the chain, not the payload.
#[derive(Debug, Clone, Copy)]
struct StepRequest {
    /// Monotonic step index, so a dropped or reordered request is visible.
    seq: u64,
}

/// Capture, instantiate and replay the dummy graph for every (bucket, N) in `cfg`.
///
/// `hip` must already have the right device selected. Allocation stays under a megabyte per row:
/// prod owns the card.
pub fn bench(
    hip: &HipThread,
    prebuilt_dir: &std::path::Path,
    cfg: &StepLoopConfig,
) -> Result<Vec<StepLoopRow>> {
    let rules = LaunchRules::prod_mxfp4()?;
    let load_counters = LoadCounters::new();
    let plugins = PluginSet::load_named(hip, prebuilt_dir, &[DUMMY_PLUGIN], &load_counters)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);

    let hop_us = hip_thread_hop_us(hip)?;

    let mut rows = Vec::new();
    for &bucket in &cfg.buckets {
        let m = bucket.m();
        if m == 0 {
            return Err(Error::Refused(format!(
                "bucket {} has M=0; batch and spec_width are both at least 1",
                bucket.label()
            )));
        }
        let shape = Shape::new(m, cfg.dummy_n, 0);
        let desc = resolve(Op::SiluMulQuant, shape, &rules)?;

        // Buffers, allocated once per bucket and kept alive for every capture: an address baked
        // into a graph node must outlive every replay of that graph (invariant I1).
        let gu = hip.malloc(u64::from(m) * u64::from(cfg.dummy_n) * 2 * 2)?;
        let q = hip.malloc(u64::from(m) * u64::from(cfg.dummy_n))?;
        let scale = hip.malloc(u64::from(m) * 4)?;
        let args = ArgBindings::new(shape)
            .with("gu", gu.addr())
            .with("q", q.addr())
            .with("scale", scale.addr());

        for &n in &cfg.launches_per_step {
            let row = bench_one(
                hip,
                &ctx,
                &desc,
                &args,
                bucket,
                m,
                cfg.dummy_n,
                n,
                cfg,
                hop_us,
            )?;
            rows.push(row);
        }
        drop((gu, q, scale));
    }
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
fn bench_one(
    hip: &HipThread,
    ctx: &LaunchCtx<'_>,
    desc: &saber_kernels::LaunchDescriptor,
    args: &ArgBindings,
    bucket: BucketShape,
    m: u32,
    dummy_n: u32,
    n: u32,
    cfg: &StepLoopConfig,
    hop_us: f64,
) -> Result<StepLoopRow> {
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;
    let counters = LaunchCounters::new();
    let mut launcher = Launcher::new(format!("steploop/{}", bucket.label()), &counters);

    // One eager launch first. It warms whatever the launcher would otherwise allocate mid-capture
    // — an allocation inside a capture is a capture error, and a `hipMemset` inside one is worse.
    launcher.launch(ctx, desc, args, &stream)?;
    hip.sync_stream(&stream)?;

    let before = counters.snapshot().launched;
    hip.begin_capture(&stream, StreamCaptureMode::ThreadLocal)?;
    for _ in 0..n {
        launcher.launch(ctx, desc, args, &stream)?;
    }
    let graph = hip.end_capture(&stream)?;
    let issued = (counters.snapshot().launched - before) as u32;
    if issued != n {
        return Err(Error::launch(
            &desc.kernel,
            format!("captured {issued} dispatches for a declared budget of {n}"),
        ));
    }
    let exec = hip.instantiate(&graph)?;

    // ---- the bounded channel and its tokio feeder ------------------------------------------
    // One current-thread runtime is enough: the point is that the producer is a tokio task and the
    // consumer is this plain thread, not that the producer is parallel.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_time()
        .build()
        .map_err(|e| Error::Refused(format!("tokio runtime: {e}")))?;
    let total = cfg.warmup + cfg.steps;
    let (tx, mut rx) = tokio::sync::mpsc::channel::<StepRequest>(cfg.channel_capacity.max(1));
    rt.spawn(async move {
        for seq in 0..total as u64 {
            // `send` awaits when the channel is full: back-pressure is the property being kept.
            if tx.send(StepRequest { seq }).await.is_err() {
                break;
            }
        }
    });

    let start = hip.create_event(EventFlags::Default)?;
    let stop = hip.create_event(EventFlags::Default)?;
    let mut submit = Vec::with_capacity(cfg.steps);
    let mut step = Vec::with_capacity(cfg.steps);
    let mut recv = Vec::with_capacity(cfg.steps);
    let mut seen = 0u64;

    #[allow(clippy::explicit_counter_loop)] // `seen` is the FEEDER's sequence, checked below;
    // `i` is the local replay index. Collapsing them would delete the ordering check.
    for i in 0..total {
        let t_wait = Instant::now();
        let Some(req) = rx.blocking_recv() else {
            return Err(Error::Refused(format!(
                "the step feeder stopped after {i} of {total} requests"
            )));
        };
        let waited = t_wait.elapsed();
        if req.seq != seen {
            return Err(Error::Refused(format!(
                "step requests arrived out of order: expected {seen}, got {}",
                req.seq
            )));
        }
        seen += 1;

        let t0 = Instant::now();
        hip.graph_launch(&exec, &stream)?;
        let t1 = Instant::now();
        hip.sync_stream(&stream)?;
        let t2 = Instant::now();
        if i >= cfg.warmup {
            submit.push(t1 - t0);
            step.push(t2 - t0);
            recv.push(waited);
        }
    }
    drop(rx);
    rt.shutdown_timeout(Duration::from_millis(200));

    // One device-timed replay, so the report can say how much of `step` is not host chain.
    hip.record_event(&start, &stream)?;
    hip.graph_launch(&exec, &stream)?;
    hip.record_event(&stop, &stream)?;
    hip.sync_event(&stop)?;
    let gpu_ms = f64::from(hip.elapsed_ms(&start, &stop)?);

    Ok(StepLoopRow {
        bucket,
        m,
        n: dummy_n,
        launches: n,
        launches_issued: issued,
        submit: Stats::of(&submit),
        step: Stats::of(&step),
        recv: Stats::of(&recv),
        gpu_ms,
        hip_thread_hop_us: hop_us,
    })
}

/// Cost of one `HipThread` channel round trip, measured with the cheapest call there is.
///
/// Every figure in a row contains one or two of these, and they are an artefact of the current
/// `HipThread` API (one job per call), not of HIP. Measuring it is what keeps the report honest.
fn hip_thread_hop_us(hip: &HipThread) -> Result<f64> {
    for _ in 0..200 {
        hip.current_device()?;
    }
    let t0 = Instant::now();
    const N: u32 = 2000;
    for _ in 0..N {
        hip.current_device()?;
    }
    Ok(t0.elapsed().as_secs_f64() * 1e6 / f64::from(N))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_of_an_empty_sample_is_zeroed_not_a_panic() {
        let s = Stats::of(&[]);
        assert_eq!(s.n, 0);
        assert_eq!(s.p50_us, 0.0);
    }

    #[test]
    fn stats_are_order_statistics() {
        let d: Vec<Duration> = (1..=100).map(Duration::from_micros).collect();
        let s = Stats::of(&d);
        assert_eq!(s.n, 100);
        assert_eq!(s.min_us, 1.0);
        assert_eq!(s.p50_us, 51.0);
        assert!((s.mean_us - 50.5).abs() < 1e-9);
        assert_eq!(s.p99_us, 100.0);
    }

    #[test]
    fn the_default_config_measures_prods_launch_count() {
        let cfg = StepLoopConfig::default();
        assert!(
            cfg.launches_per_step.contains(&1084),
            "1084 launches/step is the number the plan compares against"
        );
        assert_eq!(cfg.buckets[0].m(), 5);
        assert_eq!(cfg.buckets[1].m(), 40);
    }

    #[test]
    fn a_bounded_channel_backpressures_the_feeder() {
        // The property the harness depends on: with capacity 2 and a consumer that never reads,
        // the third send does not complete. An unbounded channel would measure the producer.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        rt.block_on(async {
            let (tx, _rx) = tokio::sync::mpsc::channel::<u32>(2);
            assert!(tx.try_send(1).is_ok());
            assert!(tx.try_send(2).is_ok());
            assert!(tx.try_send(3).is_err(), "capacity 2 must refuse a third");
        });
    }
}
