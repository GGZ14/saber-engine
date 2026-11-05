//! `saber chain-bench` — how much does a dependent node cost, over and above its kernel?
//!
//! `saber capture-gate` measured a graph saving 0.36 µs/launch over eager submission and read that
//! as "eager is asynchronous, so its host cost is hidden behind GPU work". That answers *which
//! submission path is cheaper* and says nothing about the number the architecture's thesis is
//! actually about:
//!
//! > The launch gap is **25.0% of the decode wall** (~5.6 ms of ~22.3) [...] 1084 launches/step ×
//! > ~2.1 µs of node cost is only **~2.3 ms**, so at least half the gap is *not* dispatch cost. It
//! > is drain/ramp between small dependent kernels plus the ≤1.5 ms of exposed host chain.
//! > **Retained PM4 replay (P6) removes only the dispatch component.**
//! >
//! > — ARCHITECTURE.md, *The thesis*
//!
//! So the question is: in a chain where node *n+1* reads what node *n* wrote, how much wall time
//! passes per node that is **not** the kernel running? That difference is dispatch **plus** drain,
//! and it is the ceiling on what any submission backend — graphs today, retained PM4 at P6 — can
//! ever recover.
//!
//! # The chain
//!
//! Not a synthetic loop: the op sequence of one real decoder layer at prod's decode bucket,
//! `Graph::ops_for_bucket(m = 5, Decode)` → `Decode { batch: 1, verify_width: 5 }`, restricted to
//! the ops that are single-rank and single-launch, repeated per layer with the weights shared:
//!
//! | # | op | kernel | shape |
//! |---|---|---|---|
//! | 1 | `input_layernorm` | `add_rms_quant_t256_tiled0` | m 5, k 5120 |
//! | 2 | `gdn.in_proj_qkvz` | `mxfp4_gemm_decode_bk128_ks2_tm1` | 5 × 8192 × 5120 |
//! | 3 | `gdn.in_proj_ba` | `mxfp4_gemm_decode_bk128_ks4_tm1` | 5 × 48 × 5120 |
//! | 4 | `gdn.norm` | `gdn_norm_quant_t256` | m 5, n 3072, zs 8192 |
//! | 5 | `gdn.out_proj` | `mxfp4_gemm_decode_bk128_ks4_tm1` | 5 × 5120 × 3072 |
//! | 6 | `post_attention_layernorm` | `add_rms_quant_t256_tiled0` | m 5, k 5120 |
//! | 7 | `mlp.gate_up` | `mxfp4_gemm_decode_bk128_ks1_tm1` | 5 × 17408 × 5120 |
//! | 8 | `mlp.act` | `silu_mul_quant_t256_tiled0` | m 5, n 8704 |
//! | 9 | `mlp.down_proj` | `mxfp4_gemm_decode_bk128_ks4_tm1` | 5 × 5120 × 8704 |
//!
//! **The data dependency is real, not just stream order.** Every node reads the buffer the
//! previous node wrote — the quant producers' `q`/`scale` are the following GEMM's `a`/`as`,
//! exactly as prod wires them — and node 9 writes the hidden state node 1 of the next layer
//! reads, closing the cycle. The shapes line up without padding because they are prod's own.
//!
//! What is left out, and why: the 129 all-reduces (two ranks and IPC scratch — `saber ar-gate`),
//! the 16 attention decodes and 48 GDN fused updates (real ops, but their inputs are paged caches
//! and recurrent state rather than an activation this chain can thread), and the 34 ops with no
//! ported kernel. The chain is 9 nodes/layer against the real layer's ~12.
//!
//! # The arms, and what each one isolates
//!
//! Interleaved and paired, because on this box unpaired ordering alone has already produced a 23×
//! error in exactly this measurement. Every arm is the same LENGTH, so the µs/node columns compare
//! without scaling, and each arm is run both ways — captured and replayed as a graph, and
//! submitted one descriptor at a time through the `Launcher`.
//!
//! The first version of this bench (ws-h7) measured a near-null control at **2.94 µs/node** and
//! could not say what that number was made of, because its cheapest chain still ran a real kernel
//! over real buffers. Three things are inside it:
//!
//! 1. the **command processor's** per-dispatch work, plus a graph node's share of replay;
//! 2. the **drain** between two dependent dispatches — waiting for the predecessor's writes to be
//!    visible, plus the successor's wave launch;
//! 3. the control **kernel's own execution**.
//!
//! The null plugin (`crates/saber-kernels/hip/noop.hip`, resolved through
//! `saber_kernels::noop`) is what separates them:
//!
//! | arm | chain | isolates |
//! |---|---|---|
//! | real chain | the 9-op decoder cycle above | (1)+(2)+(3) + real kernels |
//! | control | `silu_mul_quant` at m=1, n=8, one buffer set | (1)+(2)+(3) — ws-h7's 2.94 µs |
//! | noop | `saber_noop`: no arguments, no memory traffic | **(1) alone** |
//! | noop_touch | `saber_noop_touch`: one byte, read-modify-written | (1)+(2) at one byte |
//! | alternating | `saber_noop`, control, `saber_noop`, control, … | whether (2) follows the *predecessor* |
//!
//! So `noop` is term (1); `noop_touch − noop` is term (2) at the smallest footprint a dependency
//! can have; `control − noop_touch` is term (3) plus whatever (2) grows by between one byte and
//! eight elements. The alternating arm is the control on that last point: if a node's cost were
//! its own, its per-node wall would be the mean of the two chains it is made of.
//!
//! **Only term (1) is a submission cost.** A retained PM4 replay (P6) can attack it; nothing in
//! the submission path touches (2) or (3).
//!
//! A further arm — each kernel timed **alone** with events around a single launch — is reported for
//! its relative op costs and is NOT used for the decomposition. It over-states device time badly:
//! a lone launch on an idle queue pays a cold start the chain amortises, and summing those medians
//! over-predicts the real chain's wall by ~8%, i.e. it yields a *negative* overhead. That is how
//! the flaw announced itself, and it is why the control exists.
//!
//! GPU rules (CLAUDE.md): device 1, under the card's lock (`~/saber/.cache/gpu.lock.<card>`, via `tools/capture/gpu_idle.sh`), prod idle, every
//! allocation inside the budget, power cap untouched.

use std::time::Instant;

use saber_exec::{
    ArgBindings, CapturableOp, CaptureCounters, CapturedBucket, GraphSafe, LaunchCounters,
    LaunchCtx, Launcher, LoadCounters, PluginSet,
};
use saber_hip::{DevicePtr, EventFlags, HipThread, Stream, StreamFlags};
use saber_kernels::noop::{self, NoopOp, NoopRules};
use saber_kernels::rules::{resolve, LaunchRules, Op, Shape};
use saber_kernels::LaunchDescriptor;

/// The model's decode geometry at the b1s5 bucket, from `Graph::ops_for_bucket(5, Decode)`.
const M: u32 = 5;
const HIDDEN: u32 = 5120;
const QKVZ: u32 = 8192;
const BA: u32 = 48;
const GDN_OUT: u32 = 3072;
const GDN_ZS: i64 = 8192;
const MLP: u32 = 17408;
const MLP_HALF: u32 = 8704;
/// `rms_norm_eps` from the checkpoint config, as the graph resolves it.
const EPS: f32 = 1e-6;

/// Arguments of `saber chain-bench`.
pub struct ChainArgs {
    /// Layers of the 9-op cycle. 64 is the model's real depth.
    pub layers: u32,
    /// Timed interleaved pairs.
    pub passes: usize,
    /// Solo launches per distinct kernel for the kernel-only arm.
    pub solo: usize,
    /// Refuse if the chain's buffers would exceed this.
    pub budget_mib: u64,
    /// The committed code objects.
    pub prebuilt: std::path::PathBuf,
    /// HIP device ordinal within the visible set.
    pub device: i32,
    /// Skip the real decoder chain and measure only the null arms.
    ///
    /// The real chain costs ~96 MiB and ~27 ms a pass; the split of the per-node overhead does
    /// not use it. Dropping it is what makes the invariance-in-N sweep cheap enough to run at
    /// three node counts.
    pub nulls_only: bool,
}

/// Every buffer the chain threads through.
struct Buffers {
    /// Two hidden states, ping-ponged so node 9 writes what node 1 of the next layer reads.
    h0: DevicePtr,
    h1: DevicePtr,
    /// The residual the two norms carry.
    res: DevicePtr,
    /// Norm weights (one vector, shared: this measures dispatch, not numerics).
    norm_w: DevicePtr,
    gdn_w: DevicePtr,
    /// Quantised activations and their per-row scales.
    q_a: DevicePtr,
    s_a: DevicePtr,
    q_b: DevicePtr,
    s_b: DevicePtr,
    q_g: DevicePtr,
    s_g: DevicePtr,
    q_c: DevicePtr,
    s_c: DevicePtr,
    /// GEMM outputs.
    c_qkvz: DevicePtr,
    c_ba: DevicePtr,
    c_gu: DevicePtr,
    /// Weight triples, one per distinct GEMM, shared across every layer.
    w: Vec<(DevicePtr, DevicePtr, DevicePtr)>,
    bytes: u64,
}

/// One node of the chain: what to launch and what it reads and writes.
struct Node {
    label: &'static str,
    desc: LaunchDescriptor,
    bindings: ArgBindings,
}

/// One measured arm: a chain of nodes, captured once, measured as a graph and eagerly.
struct Arm {
    label: &'static str,
    /// What this arm's per-node number MEANS, printed beside it so the table reads alone.
    means: &'static str,
    nodes: Vec<Node>,
    bucket: CapturedBucket,
}

/// A node to be repeated: one descriptor and one set of bindings, cloned per node.
#[derive(Clone)]
struct Template {
    label: &'static str,
    desc: LaunchDescriptor,
    bindings: ArgBindings,
}

impl Template {
    fn node(&self) -> Node {
        Node {
            label: self.label,
            desc: self.desc.clone(),
            bindings: self.bindings.clone(),
        }
    }

    /// `n` copies: a chain, because every copy reads and writes the same buffers.
    fn chain(&self, n: u32) -> Vec<Node> {
        (0..n).map(|_| self.node()).collect()
    }
}

/// `a`, `b`, `a`, `b`, ... — the chain that asks whether a node's cost depends on what the node
/// BEFORE it touched.
fn alternate(a: &Template, b: &Template, n: u32) -> Vec<Node> {
    (0..n)
        .map(|i| if i % 2 == 0 { a.node() } else { b.node() })
        .collect()
}

/// Run the bench.
pub fn run(hip: &HipThread, args: &ChainArgs) -> anyhow::Result<()> {
    hip.set_device(args.device)?;
    // The shared idle check: VRAM first, /metrics second (`crate::gpu_idle`). This bench is timing
    // a per-node cost, so a card with something else resident on it does not just risk an OOM — it
    // changes the number.
    crate::gpu_idle::require_default(hip, args.device, 0)?;
    let rules = LaunchRules::prod_mxfp4()?;
    let noop_rules = NoopRules::prod_noop()?;
    let load_counters = LoadCounters::new();
    let plugins = PluginSet::load_named(
        hip,
        &args.prebuilt,
        &["mxfp4_gemm_decode", "mxfp4_quant", "noop"],
        &load_counters,
    )?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;
    let launch_counters = LaunchCounters::new();
    let capture_counters = CaptureCounters::new();

    // Every arm is the same LENGTH, so the per-node columns are comparable without scaling.
    let n_per_arm = args.layers * 9;
    let mut bytes = 0u64;
    let (real, real_bytes) = if args.nulls_only {
        (Vec::new(), 0)
    } else {
        build_chain(&ctx, &stream, &rules, args)?
    };
    bytes += real_bytes;
    let (control, control_bytes) = control_template(&ctx, &stream, &rules)?;
    bytes += control_bytes;
    let (noop, noop_bytes) = null_template(&ctx, &stream, &noop_rules, NoopOp::Noop)?;
    bytes += noop_bytes;
    let (touch, touch_bytes) = null_template(&ctx, &stream, &noop_rules, NoopOp::Touch)?;
    bytes += touch_bytes;
    if bytes > (args.budget_mib << 20) {
        anyhow::bail!(
            "the chains need {:.1} MiB, over the {} MiB budget",
            bytes as f64 / (1024.0 * 1024.0),
            args.budget_mib
        );
    }

    println!("chain-bench    what a dependent NODE costs, split into its terms");
    println!(
        "               {n_per_arm} node(s) per arm over {} layer(s), {:.1} MiB of buffers, \
         device {}",
        args.layers,
        bytes as f64 / (1024.0 * 1024.0),
        args.device
    );
    println!(
        "               {} interleaved pairs, {} solo launches per distinct kernel{}",
        args.passes,
        args.solo,
        if args.nulls_only {
            ", real chain SKIPPED (--nulls-only)"
        } else {
            ""
        }
    );
    println!();

    // ---- capture every arm ---------------------------------------------------------------------
    let mut launcher = Launcher::new("chain", &launch_counters);
    let proof = GraphSafe::from_flags(true, false).expect("every ported row here is capturable");
    let capture = |l: &mut Launcher,
                   label: &'static str,
                   means: &'static str,
                   nodes: Vec<Node>|
     -> anyhow::Result<Arm> {
        let ops: Vec<CapturableOp<'_>> = nodes
            .iter()
            .map(|n| CapturableOp::new(&n.desc, &n.bindings, proof))
            .collect();
        let bucket = CapturedBucket::capture(&ctx, l, &ops, &stream, label, &capture_counters)?;
        drop(ops);
        Ok(Arm {
            label,
            means,
            nodes,
            bucket,
        })
    };

    let mut arms: Vec<Arm> = Vec::new();
    if !args.nulls_only {
        arms.push(capture(
            &mut launcher,
            "real chain (9 ops/layer)",
            "(1)+(2)+(3)+kernels",
            real,
        )?);
    }
    arms.push(capture(
        &mut launcher,
        "control: silu m=1 n=8",
        "(1)+(2)+(3)",
        control.chain(n_per_arm),
    )?);
    arms.push(capture(
        &mut launcher,
        "noop: no args, no memory",
        "(1) alone",
        noop.chain(n_per_arm),
    )?);
    arms.push(capture(
        &mut launcher,
        "noop_touch: 1-byte RAW",
        "(1)+(2) at one byte",
        touch.chain(n_per_arm),
    )?);
    arms.push(capture(
        &mut launcher,
        "alternating noop/control",
        "does (2) follow the predecessor?",
        alternate(&noop, &control, n_per_arm),
    )?);
    for arm in &arms {
        println!(
            "captured       {:<26} {} node(s) for {} declared launch(es)",
            arm.label,
            arm.bucket.node_count(),
            arm.bucket.declared_launches()
        );
    }
    let n_nodes = f64::from(n_per_arm);

    // ---- kernel-only, per distinct kernel ------------------------------------------------------
    let start = hip.create_event(EventFlags::Default)?;
    let stop = hip.create_event(EventFlags::Default)?;
    let mut solo: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
    for node in arms.iter().flat_map(|a| a.nodes.iter()) {
        if solo.contains_key(&node.desc.kernel) {
            continue;
        }
        // Warm, then take the median of `solo` event-bracketed single launches. A lone launch's
        // event delta still contains its own launch latency, so this is a LOWER BOUND on the
        // kernel's device time and the overhead derived from it is a lower bound too.
        for _ in 0..8 {
            launcher.launch_op(&ctx, &[&node.desc], &node.bindings, &stream)?;
        }
        hip.sync_stream(&stream)?;
        let mut us = Vec::with_capacity(args.solo);
        for _ in 0..args.solo {
            hip.record_event(&start, &stream)?;
            launcher.launch_op(&ctx, &[&node.desc], &node.bindings, &stream)?;
            hip.record_event(&stop, &stream)?;
            hip.sync_event(&stop)?;
            us.push(f64::from(hip.elapsed_ms(&start, &stop)?) * 1000.0);
        }
        us.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        solo.insert(node.desc.kernel.clone(), us[us.len() / 2]);
    }

    // ---- every arm, graph and eager, interleaved -----------------------------------------------
    // One pass touches every arm in both modes, so clock ramp and cache state apply equally to
    // each. Paired A/B only: on this box, running the arms in blocks instead has already produced
    // a 23x error in exactly this measurement.
    let run_arm = |l: &mut Launcher, arm: &Arm, graph: bool| -> anyhow::Result<()> {
        if graph {
            arm.bucket.replay(&ctx, &stream)?;
        } else {
            for node in &arm.nodes {
                l.launch_op(&ctx, &[&node.desc], &node.bindings, &stream)?;
            }
        }
        Ok(())
    };
    for _ in 0..3 {
        for arm in &arms {
            run_arm(&mut launcher, arm, true)?;
            run_arm(&mut launcher, arm, false)?;
        }
    }
    hip.sync_stream(&stream)?;
    let mut samples: Vec<[Vec<f64>; 2]> = arms.iter().map(|_| Default::default()).collect();
    for _ in 0..args.passes {
        for (arm, sample) in arms.iter().zip(samples.iter_mut()) {
            for (mode, graph) in [true, false].into_iter().enumerate() {
                let t = Instant::now();
                run_arm(&mut launcher, arm, graph)?;
                hip.sync_stream(&stream)?;
                sample[mode].push(t.elapsed().as_secs_f64() * 1e6);
            }
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        v[v.len() / 2]
    };
    let mut g: Vec<f64> = Vec::new();
    let mut e: Vec<f64> = Vec::new();
    for s in &mut samples {
        g.push(median(&mut s[0]));
        e.push(median(&mut s[1]));
    }

    println!();
    println!("per-op solo medians (event-bracketed; a lower bound on device time):");
    let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for node in arms.iter().flat_map(|a| a.nodes.iter()) {
        if !seen.insert(node.label) {
            continue;
        }
        println!(
            "  {:<26} {:<46} {:>8.3} us",
            node.label, node.desc.kernel, solo[&node.desc.kernel]
        );
    }

    println!();
    println!(
        "{:<28} {:>10} {:>9} {:>10} {:>9}  means",
        "arm (dependent nodes)", "graph us", "us/node", "eager us", "us/node"
    );
    for (i, arm) in arms.iter().enumerate() {
        println!(
            "{:<28} {:>10.1} {:>9.3} {:>10.1} {:>9.3}  {}",
            arm.label,
            g[i],
            g[i] / n_nodes,
            e[i],
            e[i] / n_nodes,
            arm.means
        );
    }

    // ---- the split -----------------------------------------------------------------------------
    let find = |label: &str| -> usize {
        arms.iter()
            .position(|a| a.label.starts_with(label))
            .expect("the arm was built")
    };
    let (i_ctl, i_noop, i_touch, i_alt) = (
        find("control"),
        find("noop:"),
        find("noop_touch"),
        find("alternating"),
    );
    let per = |v: &[f64], i: usize| v[i] / n_nodes;
    let split = |v: &[f64], mode: &str| {
        let cp = per(v, i_noop);
        let drain1 = per(v, i_touch) - cp;
        let exec = per(v, i_ctl) - per(v, i_touch);
        println!();
        println!("the near-null control's per-node cost, split ({mode}):");
        println!("  (1) command processor + node    {cp:>8.3} us   the noop chain: no kernarg image, no memory");
        println!("  (2) drain at a 1-byte dependency{drain1:>8.3} us   noop_touch - noop");
        println!("  (3) control kernel execution    {exec:>8.3} us   control - noop_touch");
        println!("      ----------------------------{:->8}", "");
        println!(
            "      control, measured           {:>8.3} us   ({:.1}% of it is (1))",
            per(v, i_ctl),
            100.0 * cp / per(v, i_ctl)
        );
        cp
    };
    let cp_g = split(&g, "graph");
    let cp_e = split(&e, "eager");

    println!();
    println!("does the drain follow the PREDECESSOR's footprint?");
    for (mode, v) in [("graph", &g), ("eager", &e)] {
        let additive = (per(v, i_noop) + per(v, i_ctl)) / 2.0;
        let measured = per(v, i_alt);
        println!(
            "  {mode:<6} alternating {measured:>8.3} us/node vs {additive:>8.3} predicted if each \
             node's cost were its own ({:+.1}%)",
            100.0 * (measured - additive) / additive
        );
    }

    if !args.nulls_only {
        let i_real = find("real chain");
        println!();
        println!("decomposition of a REAL decoder node (graph):");
        let ovh = per(&g, i_ctl);
        println!(
            "  overhead (1)+(2)+(3)                {ovh:>8.3} us   <- the control's wall/node"
        );
        println!(
            "  kernel work (real - control)        {:>8.3} us",
            per(&g, i_real) - ovh
        );
        println!(
            "  overhead is {:.1}% of a real node's {:.2} us, of which the command processor is \
             {:.1}%",
            100.0 * ovh / per(&g, i_real),
            per(&g, i_real),
            100.0 * cp_g / per(&g, i_real)
        );
    }

    println!();
    println!("at prod's 1084 launches/step:");
    println!(
        "  control-node overhead   graph {:>5.2} ms   eager {:>5.2} ms",
        1084.0 * per(&g, i_ctl) / 1000.0,
        1084.0 * per(&e, i_ctl) / 1000.0
    );
    println!(
        "  of which term (1)       graph {:>5.2} ms   eager {:>5.2} ms   <- ALL that a retained \
         PM4 replay (P6) could address",
        1084.0 * cp_g / 1000.0,
        1084.0 * cp_e / 1000.0
    );
    println!(
        "  terms (2)+(3)           graph {:>5.2} ms                     <- drain and kernel: not \
         a submission cost, no backend removes it",
        1084.0 * (per(&g, i_ctl) - cp_g) / 1000.0
    );

    println!();
    println!("load counters    {}", load_counters.snapshot().line());
    println!("launch counters  {}", launch_counters.snapshot().line());
    println!("capture counters {}", capture_counters.snapshot().line());
    Ok(())
}

/// Allocate every buffer and build the 9-op cycle, `layers` times.
fn build_chain(
    ctx: &LaunchCtx<'_>,
    stream: &Stream,
    rules: &LaunchRules,
    args: &ChainArgs,
) -> anyhow::Result<(Vec<Node>, u64)> {
    let mut bytes = 0u64;
    let mut alloc = |n: u64| -> anyhow::Result<DevicePtr> {
        bytes += n;
        let p = ctx.hip.malloc(n.max(1))?;
        ctx.hip.memset(&p, 0, stream)?;
        Ok(p)
    };
    let m = u64::from(M);
    // The five distinct GEMM weight triples, allocated once and shared by every layer. This is
    // what keeps a 64-layer chain inside the budget: the weights are ~96 MiB, the activations are
    // kilobytes.
    let gemms = [
        (QKVZ, HIDDEN),
        (BA, HIDDEN),
        (HIDDEN, GDN_OUT),
        (MLP, HIDDEN),
        (HIDDEN, MLP_HALF),
    ];
    let mut w = Vec::new();
    for (n, k) in gemms {
        let (n64, k64) = (u64::from(n), u64::from(k));
        w.push((alloc(n64 * k64 / 2)?, alloc(k64 / 32 * n64)?, alloc(n64)?));
    }
    let b = Buffers {
        h0: alloc(m * u64::from(HIDDEN) * 2)?,
        h1: alloc(m * u64::from(HIDDEN) * 2)?,
        res: alloc(m * u64::from(HIDDEN) * 2)?,
        norm_w: alloc(u64::from(HIDDEN) * 2)?,
        gdn_w: alloc(128 * 2)?,
        q_a: alloc(m * u64::from(HIDDEN))?,
        s_a: alloc(m * 4)?,
        q_b: alloc(m * u64::from(HIDDEN))?,
        s_b: alloc(m * 4)?,
        q_g: alloc(m * u64::from(GDN_OUT))?,
        s_g: alloc(m * 4)?,
        q_c: alloc(m * u64::from(MLP_HALF))?,
        s_c: alloc(m * 4)?,
        c_qkvz: alloc(m * u64::from(QKVZ) * 2)?,
        c_ba: alloc(m * u64::from(BA) * 2)?,
        c_gu: alloc(m * u64::from(MLP) * 2)?,
        w,
        bytes,
    };
    ctx.hip.sync_stream(stream)?;

    let gemm = |n: u32, k: u32| -> anyhow::Result<LaunchDescriptor> {
        Ok(resolve(Op::gemm(), Shape::new(M, n, k), rules)?)
    };
    let norm = || -> anyhow::Result<LaunchDescriptor> {
        Ok(resolve(
            Op::AddRmsQuant { eps: EPS },
            Shape::new(M, 0, HIDDEN),
            rules,
        )?)
    };
    let mut nodes = Vec::new();
    for _ in 0..args.layers {
        // 1. input_layernorm: h0 + res -> q_a, s_a, res
        nodes.push(Node {
            label: "input_layernorm",
            desc: norm()?,
            bindings: ArgBindings::new(Shape::new(M, 0, HIDDEN))
                .with("y", b.h0.addr())
                .with("res", b.res.addr())
                .with("w", b.norm_w.addr())
                .with("q", b.q_a.addr())
                .with("scale", b.s_a.addr())
                .with("res_out", b.res.addr()),
        });
        // 2. gdn.in_proj_qkvz: q_a -> c_qkvz
        nodes.push(gemm_node(
            "gdn.in_proj_qkvz",
            gemm(QKVZ, HIDDEN)?,
            QKVZ,
            HIDDEN,
            &b,
            0,
            b.q_a.addr(),
            b.s_a.addr(),
            b.c_qkvz.addr(),
        ));
        // 3. gdn.in_proj_ba: q_a -> c_ba  (reads node 1's output, like the real graph)
        nodes.push(gemm_node(
            "gdn.in_proj_ba",
            gemm(BA, HIDDEN)?,
            BA,
            HIDDEN,
            &b,
            1,
            b.q_a.addr(),
            b.s_a.addr(),
            b.c_ba.addr(),
        ));
        // 4. gdn.norm: c_qkvz -> q_g, s_g
        nodes.push(Node {
            label: "gdn.norm",
            desc: resolve(
                Op::GdnNormQuant {
                    zs: GDN_ZS,
                    eps: EPS,
                },
                Shape::new(M, GDN_OUT, 0),
                rules,
            )?,
            bindings: ArgBindings::new(Shape::new(M, GDN_OUT, 0))
                .with("x", b.c_qkvz.addr())
                .with("z", b.c_qkvz.addr())
                .with("w", b.gdn_w.addr())
                .with("q", b.q_g.addr())
                .with("scale", b.s_g.addr()),
        });
        // 5. gdn.out_proj: q_g -> h1
        nodes.push(gemm_node(
            "gdn.out_proj",
            gemm(HIDDEN, GDN_OUT)?,
            HIDDEN,
            GDN_OUT,
            &b,
            2,
            b.q_g.addr(),
            b.s_g.addr(),
            b.h1.addr(),
        ));
        // 6. post_attention_layernorm: h1 + res -> q_b, s_b, res
        nodes.push(Node {
            label: "post_attention_layernorm",
            desc: norm()?,
            bindings: ArgBindings::new(Shape::new(M, 0, HIDDEN))
                .with("y", b.h1.addr())
                .with("res", b.res.addr())
                .with("w", b.norm_w.addr())
                .with("q", b.q_b.addr())
                .with("scale", b.s_b.addr())
                .with("res_out", b.res.addr()),
        });
        // 7. mlp.gate_up: q_b -> c_gu
        nodes.push(gemm_node(
            "mlp.gate_up",
            gemm(MLP, HIDDEN)?,
            MLP,
            HIDDEN,
            &b,
            3,
            b.q_b.addr(),
            b.s_b.addr(),
            b.c_gu.addr(),
        ));
        // 8. mlp.act: c_gu -> q_c, s_c
        nodes.push(Node {
            label: "mlp.act",
            desc: resolve(Op::SiluMulQuant, Shape::new(M, MLP_HALF, 0), rules)?,
            bindings: ArgBindings::new(Shape::new(M, MLP_HALF, 0))
                .with("gu", b.c_gu.addr())
                .with("q", b.q_c.addr())
                .with("scale", b.s_c.addr()),
        });
        // 9. mlp.down_proj: q_c -> h0, closing the cycle onto node 1's input.
        nodes.push(gemm_node(
            "mlp.down_proj",
            gemm(HIDDEN, MLP_HALF)?,
            HIDDEN,
            MLP_HALF,
            &b,
            4,
            b.q_c.addr(),
            b.s_c.addr(),
            b.h0.addr(),
        ));
    }
    // The buffers must outlive the graph that names their addresses (invariant I1), so they are
    // leaked deliberately rather than dropped at the end of this function.
    let total = b.bytes;
    std::mem::forget(b);
    Ok((nodes, total))
}

#[allow(clippy::too_many_arguments)]
fn gemm_node(
    label: &'static str,
    desc: LaunchDescriptor,
    n: u32,
    k: u32,
    b: &Buffers,
    weight: usize,
    a: saber_hip::DeviceAddr,
    scale: saber_hip::DeviceAddr,
    c: saber_hip::DeviceAddr,
) -> Node {
    let (w, ws, wref) = &b.w[weight];
    Node {
        label,
        desc,
        bindings: ArgBindings::new(Shape::new(M, n, k))
            .with("a", a)
            .with("as", scale)
            .with("w", w.addr())
            .with("ws", ws.addr())
            .with("wref", wref.addr())
            .with("c", c),
    }
}

/// The P0.5 dummy: `silu_mul_quant` at m=1, n=8. Chained, every node reads and writes the same
/// buffers, so each one waits for the last. This is ws-h7's near-null control, kept verbatim so
/// the 2.94 us it measured is the number being split here.
fn control_template(
    ctx: &LaunchCtx<'_>,
    stream: &Stream,
    rules: &LaunchRules,
) -> anyhow::Result<(Template, u64)> {
    const N: u32 = 8;
    let gu = ctx.hip.malloc(u64::from(N) * 2 * 2)?;
    let q = ctx.hip.malloc(u64::from(N))?;
    let scale = ctx.hip.malloc(4)?;
    ctx.hip.memset(&gu, 0, stream)?;
    ctx.hip.memset(&q, 0, stream)?;
    ctx.hip.memset(&scale, 0, stream)?;
    ctx.hip.sync_stream(stream)?;
    let desc = resolve(Op::SiluMulQuant, Shape::new(1, N, 0), rules)?;
    let bindings = ArgBindings::new(Shape::new(1, N, 0))
        .with("gu", gu.addr())
        .with("q", q.addr())
        .with("scale", scale.addr());
    let bytes = gu.len() + q.len() + scale.len();
    // The buffers must outlive the graph that names their addresses (invariant I1).
    std::mem::forget((gu, q, scale));
    Ok((
        Template {
            label: "control.silu",
            desc,
            bindings,
        },
        bytes,
    ))
}

/// The null plugin, as a template.
///
/// [`NoopOp::Noop`] allocates nothing and binds nothing: the descriptor has no argument slots, so
/// the kernarg image is zero bytes and the dispatch touches no memory at all. [`NoopOp::Touch`]
/// binds ONE byte, which every node of its chain read-modify-writes — the smallest RAW dependency
/// a chain can have, and the reason the two chains differ by the drain rather than by the kernel.
fn null_template(
    ctx: &LaunchCtx<'_>,
    stream: &Stream,
    rules: &NoopRules,
    op: NoopOp,
) -> anyhow::Result<(Template, u64)> {
    let desc = noop::resolve(op, rules)?;
    // No scalar slot reads it; the shape is here because bindings carry one, not because the
    // kernel has a geometry.
    let shape = Shape::new(1, 1, 1);
    match op {
        NoopOp::Noop => Ok((
            Template {
                label: "null.noop",
                desc,
                bindings: ArgBindings::new(shape),
            },
            0,
        )),
        NoopOp::Touch => {
            let p = ctx.hip.malloc(1)?;
            ctx.hip.memset(&p, 0, stream)?;
            ctx.hip.sync_stream(stream)?;
            let bindings = ArgBindings::new(shape).with("p", p.addr());
            let bytes = p.len();
            std::mem::forget(p);
            Ok((
                Template {
                    label: "null.noop_touch",
                    desc,
                    bindings,
                },
                bytes,
            ))
        }
    }
}
