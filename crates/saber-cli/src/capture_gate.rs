//! `saber capture-gate` — does a captured graph compute what eager launches compute?
//!
//! The bit-exact gate proved the *descriptors* are right. This proves the *capture* is: every case
//! from `.cache/capture` and `.cache/capture-saber` is run twice, once through
//! [`saber_exec::Launcher`] eagerly and once as a [`saber_exec::CapturedBucket`] that is captured
//! and replayed, and the two outputs are compared byte for byte.
//!
//! # Why eager is the reference here, and not the `.npy`
//!
//! The reference capture already gated the eager path at 72/72 and 57/57. Comparing the graph to
//! the `.npy` again would re-test that; comparing it to *this run's* eager output isolates the one
//! new variable — that recording a launch and replaying it is the same launch. When both differ
//! from the `.npy` the gate says so, because that is a different bug.
//!
//! # The sequence test is where the interesting failures live
//!
//! One case in one graph exercises almost nothing: a single node cannot alias a Scratch arena with
//! itself or fail to inherit a Persistent counter. So the gate also captures **as many cases as fit
//! the budget into ONE graph**, sharing one `Launcher` — which is what a real bucket does — and
//! replays it three times, comparing every output on every replay. Two bugs show up only here:
//!
//! * **aliased Scratch across ops**: the split-K partial slab is handed to every op that asks for
//!   one, so an op that reads it before writing it sees the previous op's bytes. Eagerly that is
//!   hidden by the order the gate happens to run in; in one graph it is the order the nodes run in.
//! * **a Persistent counter zeroed per pass**: the split-K block counter is zeroed once and
//!   inherited. A replay that re-zeroed it would pass on replay 1 and fail on replay 2 — which is
//!   exactly why the gate replays three times and compares each one.
//!
//! GPU rules (CLAUDE.md): device 1 only, under the card's lock (`~/saber/.cache/gpu.lock.<card>`, via `tools/capture/gpu_idle.sh`), with prod idle,
//! every allocation inside `--budget-mib`, and the power cap untouched.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use saber_exec::{
    ArgBindings, CapturableOp, CaptureCounters, CapturedBucket, GraphSafe, LaunchCounters,
    LaunchCtx, Launcher, LoadCounters, PluginSet,
};
use saber_hip::{DevicePtr, HipThread, Stream, StreamFlags};
use saber_kernels::rules::{ArgValue, LaunchRules, Shape};
use saber_kernels::saber::SaberRules;
use saber_kernels::LaunchDescriptor;

use crate::gate;
use crate::npy;

/// Arguments of `saber capture-gate`.
pub struct CaptureGateArgs {
    /// Capture directories to run. Each is read with the bit-exact gate's own case loader.
    pub data: Vec<PathBuf>,
    /// The committed code objects.
    pub prebuilt: PathBuf,
    /// Only run cases whose line contains this.
    pub only: Option<String>,
    /// Refuse a case whose device buffers exceed this, and stop adding to the sequence graph when
    /// the running total would.
    pub budget_mib: u64,
    /// A case joins the sequence graph only if its own footprint is at most this.
    ///
    /// The sequence test is about NODE COUNT and workspace sharing, not about buffer size: a
    /// 136 MiB prefill case contributes one node and eats a quarter of the budget. Capping the
    /// per-case footprint buys many more nodes inside the same VRAM, which is what the test is
    /// actually exercising.
    pub sequence_case_mib: u64,
    /// Replays of the sequence graph; each one is compared in full.
    pub replays: usize,
    /// Timed replays of the sequence graph, for the per-node figure.
    pub timed: usize,
    /// HIP device ordinal within the visible set.
    pub device: i32,
}

/// One buffer a launch writes, and how to put it back before the next run.
///
/// `restore` is what makes an IN-PLACE buffer correct. saber rewrites the conv-state cache and
/// the fp32 recurrent state in place, so the capture dumps them as both `.in.` and `.out.`;
/// zeroing such a buffer before a run deletes the input and the kernel computes something nobody
/// asked for. `Some(bytes)` means "re-upload this", `None` means "zero it".
struct Written {
    name: String,
    /// Index into the case's `owned` allocations.
    index: usize,
    /// What the reference capture says this buffer holds afterwards.
    want: Vec<u8>,
    /// The input to restore, for a buffer that is read and written.
    restore: Option<Vec<u8>>,
}

/// One case's buffers and what the reference says they should hold.
struct CaseRun {
    task: String,
    desc: LaunchDescriptor,
    bindings: ArgBindings,
    /// Owned device memory, kept alive for as long as the graph that names it.
    owned: Vec<DevicePtr>,
    /// Every buffer the launch writes.
    expects: Vec<Written>,
    bytes: u64,
}

impl CaseRun {
    /// Read every output back.
    fn read(&self, ctx: &LaunchCtx<'_>, stream: &Stream) -> anyhow::Result<Vec<Vec<u8>>> {
        let mut out = Vec::with_capacity(self.expects.len());
        for w in &self.expects {
            let mut got = vec![0u8; w.want.len()];
            ctx.hip
                .memcpy_d2h_blocking(&mut got, &self.owned[w.index], stream)?;
            out.push(got);
        }
        Ok(out)
    }

    /// Put every written buffer back to its starting state.
    ///
    /// A pure output is zeroed, so a kernel that writes nothing is a mismatch rather than a stale
    /// pass. An IN-PLACE buffer is re-uploaded, because zeroing it would delete the input — and
    /// because a replay that started from the previous replay's output would compute something
    /// different every time and call the difference a bug.
    fn restore(&self, ctx: &LaunchCtx<'_>, stream: &Stream) -> anyhow::Result<()> {
        for w in &self.expects {
            match &w.restore {
                Some(bytes) => ctx
                    .hip
                    .memcpy_h2d_blocking(&self.owned[w.index], bytes, stream)?,
                None => ctx.hip.memset(&self.owned[w.index], 0, stream)?,
            }
        }
        ctx.hip.sync_stream(stream)?;
        Ok(())
    }
}

/// Run the gate. Returns the number of failing comparisons.
pub fn run(hip: &HipThread, args: &CaptureGateArgs) -> anyhow::Result<usize> {
    hip.set_device(args.device)?;
    let mxfp4 = LaunchRules::prod_mxfp4()?;
    let saber = SaberRules::prod_saber()?;
    let head = saber_kernels::head::HeadRules::prod_head()?;
    let glue = saber_kernels::glue::GlueRules::prod()?;
    let load_counters = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &load_counters)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;
    let launch_counters = LaunchCounters::new();
    let capture_counters = CaptureCounters::new();

    println!(
        "capture-gate   {} plugin(s), {} symbol(s); {}",
        plugins.plugins().len(),
        functions.len(),
        load_counters.snapshot().line()
    );
    println!(
        "               device {}, {} MiB budget, {} replays of the sequence graph",
        args.device, args.budget_mib, args.replays
    );
    println!();

    let mut failed = 0usize;
    let mut sequence: Vec<CaseRun> = Vec::new();
    let mut sequence_bytes = 0u64;
    let budget = args.budget_mib << 20;

    println!(
        "{:<38} {:<44} {:>10} graph vs eager",
        "case", "kernel", "bytes"
    );
    for dir in &args.data {
        let shared = gate::read_shared(dir)?;
        for task in gate::read_cases(dir)? {
            if let Some(only) = &args.only {
                if !task.contains(only.as_str()) {
                    continue;
                }
            }
            let (desc, shape) = match gate::resolve_case(&task, &mxfp4, &saber, &head, &glue)? {
                Ok(pair) => pair,
                Err(e) => {
                    println!("{task:<38} {:<44} {:>10} REFUSED ({e})", "-", "-");
                    continue;
                }
            };
            // An op that declares more than one dispatch needs its whole descriptor sequence,
            // and this gate reads ONE descriptor per case line -- the GDN decode pair's recurrent
            // half is its own line. Capturing half an op would produce a node count that does not
            // match its budget, which is exactly the check `capture` is there to make, so the
            // honest answer is to skip it rather than to weaken the assertion.
            if desc.n_launches != 1 {
                println!(
                    "{task:<38} {:<44} {:>10} SKIP ({} dispatches; its partner is a separate case)",
                    desc.kernel, "-", desc.n_launches
                );
                continue;
            }
            // One launcher per case for the single-case comparison, so a case cannot inherit
            // another's Scratch contents and call it a pass.
            let mut solo = Launcher::new("capture-gate/solo", &launch_counters);
            let Some(case) = build_case(&ctx, &stream, dir, &shared, &task, desc, shape, budget)?
            else {
                println!(
                    "{task:<38} {:<44} {:>10} SKIP (no data or over budget)",
                    "-", "-"
                );
                continue;
            };

            match compare_one(&ctx, &mut solo, &stream, &case, &capture_counters) {
                Ok(Some(detail)) => {
                    println!(
                        "{:<38} {:<44} {:>10} MISMATCH {detail}",
                        case.task, case.desc.kernel, case.bytes
                    );
                    failed += 1;
                }
                Ok(None) => {
                    println!(
                        "{:<38} {:<44} {:>10} OK",
                        case.task, case.desc.kernel, case.bytes
                    );
                }
                Err(e) => {
                    println!(
                        "{:<38} {:<44} {:>10} ERROR {e}",
                        case.task, case.desc.kernel, case.bytes
                    );
                    failed += 1;
                }
            }

            // Keep it for the sequence graph while it fits. The sequence needs SEPARATE buffers
            // per case, so the budget is the running total rather than the per-case footprint --
            // and a single huge case is excluded outright, because one node is not worth a
            // quarter of the arena.
            if case.bytes <= (args.sequence_case_mib << 20) && sequence_bytes + case.bytes <= budget
            {
                sequence_bytes += case.bytes;
                sequence.push(case);
            }
        }
    }

    println!();
    failed += sequence_test(
        &ctx,
        &stream,
        &sequence,
        sequence_bytes,
        args,
        &launch_counters,
        &capture_counters,
    )?;

    println!();
    println!("load counters    {}", load_counters.snapshot().line());
    println!("launch counters  {}", launch_counters.snapshot().line());
    println!("capture counters {}", capture_counters.snapshot().line());
    Ok(failed)
}

/// Capture one case on its own and compare the replayed output to the eager one.
fn compare_one(
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    case: &CaseRun,
    counters: &std::sync::Arc<CaptureCounters>,
) -> anyhow::Result<Option<String>> {
    // ---- eager ------------------------------------------------------------------------------
    case.restore(ctx, stream)?;
    launcher.launch_op(ctx, &[&case.desc], &case.bindings, stream)?;
    ctx.hip.sync_stream(stream)?;
    let eager = case.read(ctx, stream)?;

    // Also say whether eager still matches the reference capture. If it does not, the graph
    // comparison below is measuring two unknowns.
    let drifted = eager
        .iter()
        .zip(&case.expects)
        .any(|(got, w)| *got != w.want);

    // ---- captured ---------------------------------------------------------------------------
    case.restore(ctx, stream)?;
    let proof =
        GraphSafe::from_flags(true, false).expect("the MXFP4 and saber rows are capturable");
    let ops = [CapturableOp::new(&case.desc, &case.bindings, proof)];
    let bucket = CapturedBucket::capture(ctx, launcher, &ops, stream, &case.task, counters)?;
    bucket.replay(ctx, stream)?;
    ctx.hip.sync_stream(stream)?;
    let replayed = case.read(ctx, stream)?;

    let mut bad = 0usize;
    let mut first = String::new();
    for (w, (a, b)) in case.expects.iter().zip(eager.iter().zip(&replayed)) {
        let n = a.iter().zip(b).filter(|(x, y)| x != y).count();
        if n > 0 && first.is_empty() {
            let at = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(0);
            first = format!("`{}` at {at}", w.name);
        }
        bad += n;
    }
    if bad > 0 {
        return Ok(Some(format!("{bad} bytes (first in {first})")));
    }
    if drifted {
        return Ok(Some(
            "graph == eager, but BOTH differ from the reference capture".to_string(),
        ));
    }
    if bucket.node_count() != 1 {
        return Ok(Some(format!(
            "one op captured into {} nodes",
            bucket.node_count()
        )));
    }
    Ok(None)
}

/// Capture the whole sequence into one graph and replay it several times.
#[allow(clippy::too_many_arguments)]
fn sequence_test(
    ctx: &LaunchCtx<'_>,
    stream: &Stream,
    sequence: &[CaseRun],
    bytes: u64,
    args: &CaptureGateArgs,
    launch_counters: &std::sync::Arc<LaunchCounters>,
    capture_counters: &std::sync::Arc<CaptureCounters>,
) -> anyhow::Result<usize> {
    if sequence.is_empty() {
        println!("sequence graph: nothing fit the budget");
        return Ok(0);
    }
    // ONE launcher for the whole sequence: the Scratch arena is aliased across every op and the
    // Persistent regions are shared, which is what a real bucket does and what this test is for.
    let mut launcher = Launcher::new("capture-gate/sequence", launch_counters);

    // Eager reference for every case, run as one sequence through the shared launcher so that the
    // graph is compared against the same aliasing it will reproduce.
    for case in sequence {
        case.restore(ctx, stream)?;
    }
    for case in sequence {
        launcher.launch_op(ctx, &[&case.desc], &case.bindings, stream)?;
    }
    ctx.hip.sync_stream(stream)?;
    let mut eager = Vec::with_capacity(sequence.len());
    for case in sequence {
        eager.push(case.read(ctx, stream)?);
    }

    let proof = GraphSafe::from_flags(true, false).expect("capturable");
    let ops: Vec<CapturableOp<'_>> = sequence
        .iter()
        .map(|c| CapturableOp::new(&c.desc, &c.bindings, proof))
        .collect();
    let declared: u32 = ops.iter().map(|o| o.descriptor().n_launches).sum();
    let bucket = CapturedBucket::capture(
        ctx,
        &mut launcher,
        &ops,
        stream,
        "sequence",
        capture_counters,
    )?;
    println!(
        "sequence graph:  {} case(s), {} node(s) for {} declared launch(es), {:.1} MiB of buffers",
        sequence.len(),
        bucket.node_count(),
        declared,
        bytes as f64 / (1024.0 * 1024.0)
    );

    let mut failed = 0usize;
    for pass in 1..=args.replays {
        for case in sequence {
            case.restore(ctx, stream)?;
        }
        bucket.replay(ctx, stream)?;
        ctx.hip.sync_stream(stream)?;
        let mut bad_cases = 0usize;
        let mut worst = String::new();
        for (case, want) in sequence.iter().zip(&eager) {
            let got = case.read(ctx, stream)?;
            if got != *want {
                if worst.is_empty() {
                    worst = case.task.clone();
                }
                bad_cases += 1;
            }
        }
        if bad_cases == 0 {
            println!(
                "  replay {pass}: all {} case(s) match eager",
                sequence.len()
            );
        } else {
            println!("  replay {pass}: {bad_cases} case(s) DIFFER (first: {worst})");
            failed += bad_cases;
        }
    }

    // ---- timing ------------------------------------------------------------------------------
    //
    // INTERLEAVED, because this box's measurement rule is paired A/B only: running every graph
    // pass and then every eager pass makes the two arms differ by clock ramp and cache state as
    // well as by the thing under test. Alternating them puts both arms in the same conditions and
    // the median of the per-pass samples is what is reported.
    if args.timed > 0 {
        for _ in 0..3 {
            bucket.replay(ctx, stream)?;
            for case in sequence {
                launcher.launch_op(ctx, &[&case.desc], &case.bindings, stream)?;
            }
        }
        ctx.hip.sync_stream(stream)?;

        let mut graph = Vec::with_capacity(args.timed);
        let mut eager = Vec::with_capacity(args.timed);
        for _ in 0..args.timed {
            let t = Instant::now();
            bucket.replay(ctx, stream)?;
            ctx.hip.sync_stream(stream)?;
            graph.push(t.elapsed().as_secs_f64() * 1e6);

            let t = Instant::now();
            for case in sequence {
                launcher.launch_op(ctx, &[&case.desc], &case.bindings, stream)?;
            }
            ctx.hip.sync_stream(stream)?;
            eager.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
            v[v.len() / 2]
        };
        let (g, e) = (median(&mut graph), median(&mut eager));
        let n = f64::from(declared);
        println!();
        println!(
            "timing ({} interleaved pairs, same buffers, same order, medians):",
            args.timed
        );
        println!("  graph replay   {g:>9.1} us  ({:>6.3} us/node)", g / n);
        println!("  eager launches {e:>9.1} us  ({:>6.3} us/launch)", e / n);
        println!(
            "  delta          {:>9.1} us  ({:>+6.3} us/launch, {:+.1}%)",
            e - g,
            (e - g) / n,
            100.0 * (e - g) / e
        );
        println!(
            "  P0.5 measured a captured node at 2.71-2.78 us with a near-null kernel. Eager \
             submission is ASYNCHRONOUS, so the host cost per launch (~4 us, one HipThread hop \
             plus dispatch) only becomes visible when it exceeds the GPU time per node; at \
             {:.1} us/node of real work here it is hidden, and the graph's win is small.",
            g / n
        );
    }
    Ok(failed)
}

/// Upload a case's inputs and allocate its outputs, or `None` when it does not fit or has no data.
#[allow(clippy::too_many_arguments)]
fn build_case(
    ctx: &LaunchCtx<'_>,
    stream: &Stream,
    data: &std::path::Path,
    shared: &BTreeMap<(String, String), String>,
    task: &str,
    desc: LaunchDescriptor,
    shape: Shape,
    budget: u64,
) -> anyhow::Result<Option<CaseRun>> {
    let cid = task.replace(' ', "_");
    let mut owned: Vec<DevicePtr> = Vec::new();
    let mut bindings = ArgBindings::new(shape);
    let mut expects = Vec::new();
    let mut seen: BTreeMap<String, ()> = BTreeMap::new();
    let mut total = 0u64;

    for slot in &desc.args {
        let ArgValue::Buffer { buffer } = &slot.value else {
            continue;
        };
        if seen.insert(buffer.clone(), ()).is_some() {
            continue;
        }
        let input = match npy::read_u8_opt(&data.join(format!("{cid}.in.{buffer}.npy")))? {
            Some(b) => Some(b),
            None => match shared.get(&(cid.clone(), buffer.clone())) {
                Some(f) => Some(npy::read_u8(&data.join(format!("{f}.npy")))?),
                None => None,
            },
        };
        let want = npy::read_u8_opt(&data.join(format!("{cid}.out.{buffer}.npy")))?;
        if input.is_none() && want.is_none() {
            if desc.workspaces.iter().any(|w| &w.name == buffer) {
                continue; // the launcher owns it, by kind
            }
            return Ok(None);
        }
        let bytes = input
            .as_ref()
            .map_or(0, Vec::len)
            .max(want.as_ref().map_or(0, Vec::len));
        total += bytes as u64;
        if total > budget {
            return Ok(None);
        }
        let p = ctx.hip.malloc(bytes as u64)?;
        match &input {
            Some(b) => ctx.hip.memcpy_h2d_blocking(&p, b, stream)?,
            None => ctx.hip.memset(&p, 0, stream)?,
        }
        bindings.bind(buffer, p.addr());
        if let Some(w) = want {
            // In place when the capture dumped the same buffer as BOTH an input and an output.
            expects.push(Written {
                name: buffer.clone(),
                index: owned.len(),
                want: w,
                restore: input.clone(),
            });
        }
        owned.push(p);
    }
    ctx.hip.sync_stream(stream)?;
    if expects.is_empty() {
        return Ok(None);
    }
    Ok(Some(CaseRun {
        task: task.to_owned(),
        desc,
        bindings,
        owned,
        expects,
        bytes: total,
    }))
}
