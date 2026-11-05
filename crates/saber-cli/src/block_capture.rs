//! `saber block-capture` — the P1 block as ONE captured graph, and what that costs.
//!
//! `saber layer` proved the block bit-exact against prod's own kernels by launching it op by op.
//! This subcommand records the **same** 49 dispatches (decode, b1×s5) or 56 (prefill, M=128) into
//! one HIP graph, replays it, and byte-compares the result against both the reference capture and
//! the eager path that gate already certified. Three claims, in order:
//!
//! 1. **graph == eager == reference**, byte for byte, over every op output the capture dumped,
//!    every caller-owned state the block rewrites, and the block's own hidden state and residual.
//! 2. **The node count is the launch budget.** `hipGraphGetNodes` must report exactly
//!    `BlockPlan::expected_launches()` — 49 and 56 — which is the graph's own declared count minus
//!    the ten launches the identity all-reduce replaces.
//! 3. **A replay is not free of state.** The GDN conv and recurrent caches and the paged KV cache
//!    are rewritten in place, so every replay re-uploads them from the capture; and the six
//!    Persistent regions the block carries — five split-K block counters and the GDN fused
//!    update's cross-workgroup barrier — are proved to be prepared **before** the capture opened,
//!    by name, with no launch (`saber_exec::check_persistent_ready`).
//!
//! Every wait in this gate is a **deadline**, not a `hipStreamSynchronize`: `hipStreamQuery`
//! polled to 20 s, as `saber ar-gate` does. A kernel that waits for something that is not coming
//! then fails loud and releases the GPU lock instead of holding a card until someone notices.
//!
//! Then the number this was built for: 50 graph replays against 50 eager runs of the same block,
//! **paired and interleaved**, `hipEvent`s around the whole block, so the saving is per node
//! rather than per harness.
//!
//! Everything but the capture is deliberately the same shape as `saber layer`: the same
//! `block.json`, the same pack window, the same npy comparisons. It is a second reader of the same
//! reference, not a second reference.
//!
//! GPU rules (CLAUDE.md): one device under the card's lock (`~/saber/.cache/gpu.lock.<card>`, via `tools/capture/gpu_idle.sh`), inside `--budget-mib`,
//! power cap untouched. Prod down (`localhost:8080/metrics` refused) counts as idle and the report
//! says which regime it ran in.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use saber_exec::{
    bind_block, check_persistent_ready, BlockExecutor, BlockInputs, BlockPlan, BoundWeight,
    CaptureCounters, CapturedBlock, GraphSafe, LaunchCounters, LaunchCtx, Launcher, LoadCounters,
    PluginSet, Route, RuntimeParams, Source,
};
use saber_hip::{DevicePtr, EventFlags, HipThread, Stream, StreamFlags};
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_mem::dio::OpenMode;
use saber_mem::ledger::{Ledger, Pool};
use saber_mem::pack::PackReader;
use saber_mem::upload::{HipSink, PinnedRing, Upload, UploadConfig};
use saber_model::graph::{Bucket, Graph, ModelConfig};
use serde_json::Value;

use crate::npy;

/// Arguments of `saber block-capture`.
pub struct BlockCaptureArgs {
    /// Where `the internal reference harness` wrote `block.json` and its `.npy` files.
    pub data: PathBuf,
    /// The committed code objects and their manifests.
    pub prebuilt: PathBuf,
    /// The device-final tier pack: every weight the block binds comes out of it.
    pub pack: PathBuf,
    /// TP rank.
    pub rank: u32,
    /// Only run the case whose name contains this substring (`decode` / `prefill`).
    pub only: Option<String>,
    /// Replays to compare, each with fresh state.
    pub replays: usize,
    /// Timed pairs (one graph replay and one eager run each).
    pub iters: usize,
    /// Untimed pairs first.
    pub warmup: usize,
    /// Refuse to start if the block's device buffers would exceed this.
    pub budget_mib: u64,
    /// HIP device ordinal within the visible set.
    pub device: i32,
}

fn s<'a>(v: &'a Value, key: &str) -> anyhow::Result<&'a str> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("block.json: `{key}` missing or not a string"))
}

fn u(v: &Value, key: &str) -> anyhow::Result<u64> {
    v.get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow::anyhow!("block.json: `{key}` missing or not an integer"))
}

fn pack_file(pack: &Path, rank: u32) -> PathBuf {
    if pack.is_file() {
        pack.to_path_buf()
    } else {
        pack.join(format!("rank{rank}.pack"))
    }
}

/// The capture's dump name for a state buffer: `l3.attn.kv_cache` is `layer.3.attn.kv.in.kv`.
fn state_dump(name: &str) -> anyhow::Result<(String, &'static str)> {
    let (layer, tail) = name
        .strip_prefix('l')
        .and_then(|r| r.split_once('.'))
        .ok_or_else(|| anyhow::anyhow!("state `{name}` is not an `l<N>.…` name"))?;
    Ok(match tail {
        "gdn.conv_state" => (format!("layer.{layer}.gdn.state"), "cstate"),
        "gdn.recurrent_state" => (format!("layer.{layer}.gdn.state"), "state"),
        "attn.kv_cache" => (format!("layer.{layer}.attn.kv"), "kv"),
        other => anyhow::bail!("no capture dump is known for state `{other}`"),
    })
}

fn case_bucket(case: &Value) -> anyhow::Result<Bucket> {
    let batch = u(case, "batch")? as u32;
    let q_len = u(case, "q_len")? as u32;
    Ok(match s(case, "phase")? {
        "decode" => Bucket::Decode {
            batch,
            verify_width: q_len,
        },
        "prefill" => Bucket::Prefill {
            tokens: batch * q_len,
            seqs: batch,
        },
        other => anyhow::bail!("unknown phase `{other}`"),
    })
}

fn case_params(case: &Value) -> anyhow::Result<RuntimeParams> {
    Ok(RuntimeParams {
        max_ctx: u(case, "max_ctx")? as u32,
        gdn_slots: u(case, "gdn_slots")? as u32,
    })
}

/// One buffer this gate watches: where to read it on the device, and what it should be.
struct Probe {
    /// `<op>.<buffer>`, for the table.
    label: String,
    /// Where it lives.
    at: Where,
    /// The reference bytes.
    want: Vec<u8>,
}

enum Where {
    /// A byte range of the activation arena.
    Arena { offset: u64 },
    /// A byte range of a caller-owned state buffer, by index into the owned allocations.
    State {
        /// Which allocation.
        idx: usize,
        /// Where in it — a slot's offset, or 0 for the whole buffer.
        offset: u64,
    },
}

/// Every wait in this gate is bounded.
///
/// `hipStreamSynchronize` blocks forever, and the kernels in this block include two spin-waits:
/// the GDN fused update's cross-workgroup barrier and the split-K reduce's arrival counter. If one
/// of them is ever waited on for something that will not arrive, this returns instead of holding
/// the GPU lock until a human kills the container. Twenty seconds is ~2000x the block's own wall
/// time, so it cannot trip on a slow run.
const WAIT_DEADLINE: Duration = Duration::from_secs(20);

/// Poll `hipStreamQuery` to a deadline instead of blocking.
fn wait_for(ctx: &LaunchCtx<'_>, stream: &Stream, what: &str) -> anyhow::Result<()> {
    let start = Instant::now();
    loop {
        if ctx.hip.stream_ready(stream)? {
            return Ok(());
        }
        if start.elapsed() > WAIT_DEADLINE {
            anyhow::bail!(
                "`{what}` did not complete within {WAIT_DEADLINE:?}: the stream is still busy. \
                 Stopping so the process exits, the context is torn down and the GPU lock is \
                 released."
            );
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The first differing byte and how many differ.
fn diff(got: &[u8], want: &[u8]) -> Option<(usize, usize)> {
    let mut bad = 0usize;
    let mut first = 0usize;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if g != w {
            if bad == 0 {
                first = i;
            }
            bad += 1;
        }
    }
    (bad > 0).then_some((bad, first))
}

/// Read every probe off the device, in probe order.
fn snapshot(
    hip: &HipThread,
    stream: &Stream,
    exec: &BlockExecutor,
    owned: &[DevicePtr],
    probes: &[Probe],
) -> anyhow::Result<Vec<Vec<u8>>> {
    let mut out = Vec::with_capacity(probes.len());
    for p in probes {
        let mut got = vec![0u8; p.want.len()];
        match &p.at {
            Where::Arena { offset } => exec.download(hip, *offset, &mut got, stream)?,
            Where::State { idx, offset } => {
                hip.memcpy_d2h_at_blocking(&mut got, &owned[*idx], *offset, stream)?
            }
        }
        out.push(got);
    }
    Ok(out)
}

/// Compare a snapshot against the reference; returns the number of mismatching buffers.
fn compare(probes: &[Probe], got: &[Vec<u8>], what: &str, verbose: bool) -> usize {
    let mut bad = 0usize;
    for (p, g) in probes.iter().zip(got) {
        if let Some((n, first)) = diff(g, &p.want) {
            bad += 1;
            println!(
                "  {what}: {} MISMATCH {n} byte(s), first at {first}",
                p.label
            );
        } else if verbose {
            println!("  {what}: {} ok ({} bytes)", p.label, p.want.len());
        }
    }
    bad
}

/// Run the gate. Returns the number of failures.
pub fn run(hip: &HipThread, args: &BlockCaptureArgs) -> anyhow::Result<usize> {
    hip.set_device(args.device)?;
    let block: Value =
        serde_json::from_str(&std::fs::read_to_string(args.data.join("block.json"))?)?;
    let tp = u(&block, "tp")? as u32;
    let graph = Graph::qwen38_27b(ModelConfig::qwen38_27b(), tp)?;
    let layers: Vec<u32> = block
        .get("layers")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("block.json: no `layers`"))?
        .iter()
        .filter_map(|v| v.as_u64().map(|n| n as u32))
        .collect();
    let mxfp4 = LaunchRules::prod_mxfp4()?;
    let saber = SaberRules::prod_saber()?;

    let load_counters = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &load_counters)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;

    println!(
        "block-capture  layers {layers:?} of {} (tp {tp}), tier {}…",
        graph.config.num_hidden_layers,
        &s(&block, "tier")?[..12]
    );
    println!("reference      {}", args.data.display());
    println!(
        "plugins        {} loaded ({} symbols); {}",
        plugins.plugins().len(),
        functions.len(),
        load_counters.snapshot().line()
    );

    // ---- the weights, out of the device-final pack ------------------------------------------
    let mut budget = args.budget_mib << 20;
    let probe_case = block
        .get("cases")
        .and_then(|c| c.get("decode").or_else(|| c.get("prefill")))
        .ok_or_else(|| anyhow::anyhow!("block.json has no cases"))?;
    let probe_plan = BlockPlan::new(
        &graph,
        case_bucket(probe_case)?,
        &layers,
        case_params(probe_case)?,
        &mxfp4,
        &saber,
    )?;
    let want = probe_plan.weight_keys();
    let reader = PackReader::open(&pack_file(&args.pack, args.rank))?;
    let mut idx: Vec<usize> = Vec::new();
    let mut layout_of: BTreeMap<String, u32> = BTreeMap::new();
    for (i, rec) in reader.records().iter().enumerate() {
        let name = reader.name(i);
        if want.contains(name) {
            idx.push(i);
            layout_of.insert(name.to_owned(), rec.layout_id);
        }
    }
    if layout_of.len() != want.len() {
        anyhow::bail!(
            "{} record(s) the block binds are not in {}",
            want.len() - layout_of.len(),
            args.pack.display()
        );
    }
    let (lo, hi) = (idx[0], *idx.last().expect("non-empty"));
    let last = &reader.records()[hi];
    let span =
        saber_mem::pack::align_up(last.data_off + last.length) - reader.records()[lo].data_off;
    let cfg = UploadConfig {
        start_record: lo,
        limit_bytes: Some(span),
        mode: OpenMode::Direct,
        ..UploadConfig::default()
    };
    let upload = Upload::open(&args.pack, args.rank, cfg)?;
    let uplan = upload.plan();
    if uplan.arena_bytes > budget {
        anyhow::bail!(
            "the weight window is {} MiB and the budget is {} MiB",
            uplan.arena_bytes >> 20,
            budget >> 20
        );
    }
    budget -= uplan.arena_bytes;
    let mut ledger = Ledger::new();
    ledger.declare(Pool::Vram(args.rank), uplan.arena_bytes.max(1))?;
    ledger.declare(Pool::HostPinned, uplan.staging_bytes.max(1))?;
    let reserved = upload.reserve(&mut ledger)?;
    let mut staging = hip.host_malloc(reserved.staging_bytes())?;
    let ring = PinnedRing::new(&mut staging, &reserved)?;
    let mut sink = HipSink::new(hip, &reserved)?;
    let mut weights: BTreeMap<String, BoundWeight> = BTreeMap::new();
    upload.run(&mut sink, &ring, |r| {
        if let Some(layout) = layout_of.get(r.name) {
            weights.insert(r.name.to_string(), BoundWeight::new(r.addr, *layout));
        }
    })?;
    println!(
        "weights        {} records bound by name out of a {} MiB window of {}",
        weights.len(),
        span >> 20,
        args.pack.display()
    );

    let mut failures = 0usize;
    let cases = block
        .get("cases")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("block.json: no `cases`"))?;
    for (name, case) in cases {
        if let Some(only) = &args.only {
            if !name.contains(only.as_str()) {
                continue;
            }
        }
        failures += run_case(
            hip, &ctx, &stream, args, &graph, &layers, &mxfp4, &saber, &weights, name, case, budget,
        )?;
    }
    drop(sink);
    Ok(failures)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn run_case(
    hip: &HipThread,
    ctx: &LaunchCtx<'_>,
    stream: &Stream,
    args: &BlockCaptureArgs,
    graph: &Graph,
    layers: &[u32],
    mxfp4: &LaunchRules,
    saber: &SaberRules,
    weights: &BTreeMap<String, BoundWeight>,
    case_name: &str,
    case: &Value,
    budget: u64,
) -> anyhow::Result<usize> {
    let bucket = case_bucket(case)?;
    let mut plan = BlockPlan::new(graph, bucket, layers, case_params(case)?, mxfp4, saber)?;
    // ws-h30: `gdn_chunk_scan` reads `h0` and now writes `ht` at the slot the STEP points it at,
    // and `capture_layer.py` called the reference kernel with `h0 = state[:batch]` -- sequence 0 at offset 0. So
    // this replay is pointed at the reference's own base. A SERVING caller has a session pool and
    // calls `retarget_prefill_state`, which refuses slot 0 because the pool has a null slot there.
    if bucket.phase() == saber_model::graph::Phase::Prefill {
        plan.retarget_prefill_state_at_reference_base()?;
    }
    let plan = plan;
    let dir = args.data.join(case_name);
    println!();
    println!(
        "case           {case_name}: {:?}, M={}, {} steps, {} launches expected ({} declared − \
         {} identity)",
        bucket,
        bucket.m(),
        plan.steps.len(),
        plan.expected_launches(),
        plan.declared_launches,
        plan.identity_launches
    );

    // ---- everything the block reads that the arena does not own -------------------------------
    let mut budget = budget;
    let mut owned: Vec<DevicePtr> = Vec::new();
    let mut inputs = BlockInputs {
        weights: weights.clone(),
        ..BlockInputs::default()
    };
    // State: kept as (index, the reference bytes) so every replay can restore it.
    let mut states: Vec<(String, usize, Vec<u8>)> = Vec::new();
    for name in plan.state_keys() {
        let (op, buf) = state_dump(&name)?;
        let bytes = npy::read_u8(&dir.join(format!("{op}.in.{buf}.npy")))?;
        let n = bytes.len() as u64;
        if n > budget {
            anyhow::bail!("state `{name}` is {n} B and the budget is exhausted");
        }
        budget -= n;
        let p = hip.malloc(n.max(1))?;
        hip.memcpy_h2d_blocking(&p, &bytes, stream)?;
        inputs.states.insert(name.clone(), p.addr());
        states.push((name, owned.len(), bytes));
        owned.push(p);
    }
    for key in plan.index_keys() {
        let (op, buf) = key
            .rsplit_once('.')
            .ok_or_else(|| anyhow::anyhow!("index key `{key}` has no buffer part"))?;
        let bytes = npy::read_u8(&dir.join(format!("{op}.in.{buf}.npy")))?;
        let n = bytes.len() as u64;
        budget = budget.saturating_sub(n);
        let p = hip.malloc(n.max(1))?;
        hip.memcpy_h2d_blocking(&p, &bytes, stream)?;
        inputs.indices.insert(key.clone(), p.addr());
        owned.push(p);
    }

    // ---- the arena ----------------------------------------------------------------------------
    if plan.arena_bytes > budget {
        anyhow::bail!(
            "the activation arena is {} MiB and only {} MiB of budget is left",
            plan.arena_bytes >> 20,
            budget >> 20
        );
    }
    let mut exec = BlockExecutor::new(hip, &plan, 8usize << 20)?;

    // Everything a run starts from: the residual stream entering the block, and every output of
    // the ops with no code object. Collected once, applied before every run and every replay --
    // which is what makes "fresh state" a procedure rather than a hope.
    let mut seed: Vec<(u64, Vec<u8>)> = Vec::new();
    for edge in &plan.inputs {
        let b = plan
            .buffer(edge)
            .ok_or_else(|| anyhow::anyhow!("block input `{edge}` is not planned"))?;
        let bytes = npy::read_u8(&dir.join("block.in.hidden.npy"))?;
        if bytes.len() as u64 != b.bytes {
            anyhow::bail!(
                "block input `{edge}`: plan {} B, capture {}",
                b.bytes,
                bytes.len()
            );
        }
        seed.push((b.offset, bytes));
    }
    for step in &plan.steps {
        if !matches!(step.route, Route::Supplied { .. }) {
            continue;
        }
        for (buf, src) in &step.bindings {
            let Source::Arena { offset, bytes, .. } = src else {
                continue;
            };
            let host = npy::read_u8(&dir.join(format!("{}.out.{buf}.npy", step.name())))?;
            if host.len() as u64 != *bytes {
                anyhow::bail!("{}: supplied `{buf}` is {bytes} B in the plan", step.name());
            }
            seed.push((*offset, host));
        }
    }

    // ---- what this gate watches ---------------------------------------------------------------
    let files = case
        .get("files")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("case `{case_name}` has no `files`"))?;
    let mut probes: Vec<Probe> = Vec::new();
    for step in &plan.steps {
        if !matches!(step.route, Route::Kernel { .. }) {
            continue;
        }
        let Some(dumped) = files.get(step.name()).and_then(Value::as_object) else {
            continue;
        };
        for key in dumped.keys() {
            let Some(buf) = key.strip_prefix("out.") else {
                continue;
            };
            let Some(src) = step.source(buf) else {
                continue;
            };
            let want = npy::read_u8(&dir.join(format!("{}.{key}.npy", step.name())))?;
            match src {
                Source::Arena { offset, bytes, .. } if want.len() as u64 == *bytes => {
                    probes.push(Probe {
                        label: format!("{}.{buf}", step.name()),
                        at: Where::Arena { offset: *offset },
                        want,
                    });
                }
                Source::State { key: name, at } => {
                    let Some((_, idx, _)) = states.iter().find(|(n, _, _)| n == name) else {
                        continue;
                    };
                    // A SLOT binding watches that slot, not the whole pool: `chunk_scan` writes
                    // `ht` in place into one slot (ws-h30) and the reference dumps exactly one.
                    let offset = match at {
                        saber_exec::StateAt::Slot { offset, .. } => *offset,
                        _ => 0,
                    };
                    if offset + want.len() as u64 > owned[*idx].len() {
                        continue;
                    }
                    if matches!(at, saber_exec::StateAt::Pool)
                        && want.len() as u64 != owned[*idx].len()
                    {
                        continue;
                    }
                    probes.push(Probe {
                        label: format!("{}.{buf}", step.name()),
                        at: Where::State { idx: *idx, offset },
                        want,
                    });
                }
                _ => {}
            }
        }
    }
    // The block's own output.
    let last = *layers.last().expect("a block has layers");
    for (file, edge) in [
        ("block.out.hidden.npy", format!("l{last}.mlp.partial")),
        ("block.out.resid.npy", format!("l{last}.resid.b")),
    ] {
        let want = npy::read_u8(&dir.join(file))?;
        let b = plan
            .buffer(&edge)
            .ok_or_else(|| anyhow::anyhow!("`{edge}` is not planned"))?;
        probes.push(Probe {
            label: format!(
                "BLOCK.{}",
                file.trim_start_matches("block.out.")
                    .trim_end_matches(".npy")
            ),
            at: Where::Arena { offset: b.offset },
            want,
        });
    }
    println!(
        "watching       {} buffer(s): every dumped op output, every state the block rewrites, and \
         the block's hidden state and residual",
        probes.len()
    );

    // ---- one launcher for both paths ----------------------------------------------------------
    let counters = LaunchCounters::new();
    let capture_counters = CaptureCounters::new();
    let mut launcher = Launcher::new(format!("block.{case_name}"), &counters);

    // `reset` is the whole of "fresh state": zero the arena so a kernel that writes nothing shows
    // up as a mismatch rather than as the last run's answer, re-seed the inputs, and restore every
    // caller-owned state the block rewrites in place.
    let reset =
        |hip: &HipThread, exec: &mut BlockExecutor, owned: &[DevicePtr]| -> anyhow::Result<()> {
            exec.zero(hip, stream)?;
            for (offset, bytes) in &seed {
                exec.upload(hip, *offset, bytes, stream)?;
            }
            for (_, idx, bytes) in &states {
                hip.memcpy_h2d_blocking(&owned[*idx], bytes, stream)?;
            }
            wait_for(ctx, stream, "restoring the block's inputs and state")?;
            Ok(())
        };

    // ---- 1. the eager path, which `saber layer` already gates ---------------------------------
    reset(hip, &mut exec, &owned)?;
    let eager_run = exec.run(ctx, &mut launcher, stream, &plan, &inputs, false)?;
    wait_for(ctx, stream, "the eager block")?;
    let eager = snapshot(hip, stream, &exec, &owned, &probes)?;
    let eager_bad = compare(&probes, &eager, "eager", false);
    println!(
        "eager          {} launches, {} of {} buffer(s) match the reference",
        eager_run.launches,
        probes.len() - eager_bad,
        probes.len()
    );

    // ---- 2. capture ----------------------------------------------------------------------------
    reset(hip, &mut exec, &owned)?;
    // Every kernel in this block is a plain dispatch: the one op that would synchronise with
    // another rank is the all-reduce, and at rank 0 of a TP=2 tier with no peer it is
    // `Route::Identity` and launches nothing. `check_capturable` inside the capture asserts that
    // no descriptor declares a peer-visible workspace, so this proof is checked, not assumed.
    let proof = GraphSafe::from_flags(true, false).expect("graph_safe && !may_sync");
    let captured = CapturedBlock::capture(
        ctx,
        &mut launcher,
        &exec,
        &plan,
        &inputs,
        stream,
        format!("block.{case_name}"),
        proof,
        &capture_counters,
    )?;
    wait_for(ctx, stream, "the capture")?;
    let expected = plan.expected_launches() as usize;
    println!(
        "captured       {} node(s) for {} op(s); the plan's budget is {expected} — {}",
        captured.node_count(),
        captured.ops(),
        if captured.node_count() == expected {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    if captured.node_count() != expected {
        anyhow::bail!(
            "the graph has {} nodes and the block's launch budget is {expected}",
            captured.node_count()
        );
    }
    let persistent = launcher.persistent_regions();
    println!(
        "persistent     {} region(s) survive a replay: {}",
        persistent.len(),
        if persistent.is_empty() {
            "none".to_string()
        } else {
            persistent
                .iter()
                .map(|(k, b)| format!("{k} ({b} B)"))
                .collect::<Vec<_>>()
                .join(", ")
        }
    );

    // ---- 3. replay N times, fresh state each time ---------------------------------------------
    let mut failures = eager_bad;
    for r in 1..=args.replays {
        reset(hip, &mut exec, &owned)?;
        captured.replay(ctx, stream)?;
        wait_for(ctx, stream, &format!("graph replay {r}"))?;
        let got = snapshot(hip, stream, &exec, &owned, &probes)?;
        let vs_ref = compare(&probes, &got, &format!("replay {r} vs reference"), false);
        let mut vs_eager = 0usize;
        for (i, p) in probes.iter().enumerate() {
            if got[i] != eager[i] {
                vs_eager += 1;
                println!("  replay {r} vs eager: {} DIFFERS", p.label);
            }
        }
        println!(
            "replay {r}       {} of {} match the reference, {} of {} match the eager path \
             byte-for-byte",
            probes.len() - vs_ref,
            probes.len(),
            probes.len() - vs_eager,
            probes.len()
        );
        failures += vs_ref + vs_eager;
    }

    // ---- 4. NEGATIVE CONTROL: Persistent regions are prepared BEFORE the capture ------------
    // The property: a capture whose Persistent regions the launcher does not already own is
    // refused, by name, before a single node runs. Asked of a launcher that has prepared nothing,
    // over the very steps this block just captured -- the real function on the real descriptors,
    // with no launch.
    //
    // It is NOT tested by dirtying a counter and replaying. Two of the six regions here are
    // spin-waits: the split-K reduce's arrival counter and, worse, the GDN fused update's
    // cross-workgroup barrier, which is waited on until it arrives. A barrier that starts non-zero
    // never arrives. That was tried on 2026-09-19, wedged the card for fifty minutes and had to be
    // killed from outside -- it demonstrated nothing except that a spin-wait spins, and it is why
    // `Launcher` now offers only `zero_persistent`.
    let steps = bind_block(&exec, &plan, &inputs)?;
    if persistent.is_empty() {
        // Not a failure and not a pass: there is nothing to prepare, so there is nothing to
        // refuse. The prefill block is exactly this case — its GEMMs take the folded and A-tiled
        // kernels, which have no split-K counter, and the prefill GDN trio has no barrier — and a
        // control that "passed" here would be measuring an empty list.
        println!(
            "control        SKIPPED: this case declares no Persistent workspace at all, so the \
             refusal has nothing to name"
        );
    } else {
        let empty_counters = LaunchCounters::new();
        let unprepared = Launcher::new("control.unprepared", &empty_counters);
        match check_persistent_ready(&unprepared, &steps) {
            Err(e) => println!("control        a launcher that prepared nothing is REFUSED: {e}"),
            Ok(()) => {
                println!(
                    "control        FAILED: an unprepared launcher was accepted although this \
                     block declares {} Persistent region(s)",
                    persistent.len()
                );
                failures += 1;
            }
        }
        // And the launcher that DID prepare them passes the same check, which is the other half:
        // the refusal is about preparation, not about the workspaces existing.
        match check_persistent_ready(&launcher, &steps) {
            Ok(()) => println!(
                "control        the launcher that captured the block passes it: {} region(s) \
                 allocated and zeroed before the capture opened",
                persistent.len()
            ),
            Err(e) => {
                println!("control        FAILED: the prepared launcher was refused: {e}");
                failures += 1;
            }
        }
    }
    drop(steps);

    // ---- 5. timing: paired and interleaved ----------------------------------------------------
    let start = hip.create_event(EventFlags::Default)?;
    let end = hip.create_event(EventFlags::Default)?;
    let mut graph_us: Vec<f64> = Vec::with_capacity(args.iters);
    let mut eager_us: Vec<f64> = Vec::with_capacity(args.iters);
    // State drifts across these runs (the caches are rewritten in place and nothing is restored
    // between them, on purpose: a restore is a host copy and would be most of what was timed).
    // Only the wall time is read here; correctness was established above, with fresh state.
    for i in 0..(args.warmup + args.iters) {
        hip.record_event(&start, stream)?;
        captured.replay(ctx, stream)?;
        hip.record_event(&end, stream)?;
        // Bounded, like every other wait here: `hipEventSynchronize` on an event behind a kernel
        // that never finishes is the same hang as `hipStreamSynchronize`. The times themselves are
        // the events', so polling the stream costs the measurement nothing.
        wait_for(ctx, stream, "a timed graph replay")?;
        let g = f64::from(hip.elapsed_ms(&start, &end)?) * 1000.0;

        hip.record_event(&start, stream)?;
        exec.run(ctx, &mut launcher, stream, &plan, &inputs, false)?;
        hip.record_event(&end, stream)?;
        wait_for(ctx, stream, "a timed eager run")?;
        let e = f64::from(hip.elapsed_ms(&start, &end)?) * 1000.0;

        if i >= args.warmup {
            graph_us.push(g);
            eager_us.push(e);
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        v[v.len() / 2]
    };
    let g = median(&mut graph_us);
    let e = median(&mut eager_us);
    let nodes = f64::from(plan.expected_launches());
    println!();
    println!(
        "timing         {} interleaved pairs after {} warm-up pairs, hipEvents around the WHOLE \
         block, one stream",
        args.iters, args.warmup
    );
    println!(
        "               graph replay  {g:>9.1} µs/block   {:>6.3} µs/node",
        g / nodes
    );
    println!(
        "               eager run     {e:>9.1} µs/block   {:>6.3} µs/node",
        e / nodes
    );
    println!(
        "               saving        {:>9.1} µs/block   {:>6.3} µs/node ({:+.1}%)",
        e - g,
        (e - g) / nodes,
        100.0 * (e - g) / e
    );
    println!("               {}", counters.snapshot().line());
    println!("               {}", capture_counters.snapshot().line());

    // The graph names these addresses; it must go before they do.
    drop(captured);
    drop(exec);
    drop(owned);
    Ok(failures)
}
