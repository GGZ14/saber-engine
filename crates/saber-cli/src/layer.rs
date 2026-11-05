//! `saber layer` — the P1 decoder-block gate and its µs/block.
//!
//! `the internal reference harness` runs layers 0–4 of the real model through PROD's kernels on the
//! real weights and dumps every op's buffers. This subcommand builds the same block out of
//! `saber-model`'s [`Graph`], resolves every op through the engine's own `resolve`, runs it with
//! [`saber_exec::Launcher`] on one stream, and byte-compares what came out.
//!
//! What is compared, and what is not:
//!
//! * **Compared:** every output buffer of every op the engine launches, plus the block's final
//!   hidden state and residual. Both cases: decode at M = 5 (batch 1 × spec 5) and prefill at
//!   M = 128.
//! * **Uploaded, not compared:** `qk_norm_rope` and `attn_gate_quant`, which have no code object
//!   (`KernelOp::n_launches()` is `None` for both), and the index tables a scheduler would build.
//! * **Replaced by identity in BOTH chains:** the two-rank all-reduce. This is rank 0 of a TP=2
//!   tier with no peer, so every row-parallel output is a partial sum, and the launch budget says
//!   so rather than hiding it.
//!
//! Where the weights come from is a P1 finding in itself. The tier pack is v0 — its plan
//! descriptor carries `transforms: []` — so it holds the checkpoint's own NVFP4 / fp8-per-channel
//! bytes, not the MXFP4 fragment-order matrices the W4A8 kernel reads. Every record that IS
//! device-final (the three norms and the GDN conv filter) is bound straight out of a
//! `saber-mem::upload` arena by record name; the requantized matrices and the fp32 `A_log` /
//! `dt_bias` casts come from the capture's device-final dump, and the report prints the split.
//!
//! GPU rules (CLAUDE.md): prod serves on both cards. Run on HIP device 1 under
//! the card's lock (`~/saber/.cache/gpu.lock.<card>`, via `tools/capture/gpu_idle.sh`), with `saber:num_requests_running` at 0, inside `--budget-mib`,
//! and never touch the power cap.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use saber_exec::{
    BlockExecutor, BlockInputs, BlockPlan, BoundWeight, LaunchCounters, LaunchCtx, Launcher,
    LoadCounters, PluginSet, Route, RuntimeParams, Source,
};
use saber_hip::{DevicePtr, HipThread, Stream, StreamFlags};
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_mem::dio::OpenMode;
use saber_mem::ledger::{Ledger, Pool};
use saber_mem::pack::{LayoutId, PackReader};
use saber_mem::upload::{HipSink, PinnedRing, Upload, UploadConfig};
use saber_model::graph::{Bucket, Graph, ModelConfig};
use serde_json::Value;

use crate::npy;

/// Arguments of `saber layer`.
pub struct LayerArgs {
    /// Where `the internal reference harness` wrote `block.json` and its `.npy` files.
    pub data: PathBuf,
    /// The committed code objects and their manifests.
    pub prebuilt: PathBuf,
    /// The tier pack directory, for the records that are already device-final.
    pub pack: Option<PathBuf>,
    /// TP rank.
    pub rank: u32,
    /// Only run the case whose name contains this substring (`decode` / `prefill`).
    pub only: Option<String>,
    /// Print the provenance table: every weight record, its layout, and where it came from.
    pub provenance: bool,
    /// NEGATIVE CONTROL: bind this record at the wrong layout id, so the block must refuse it.
    pub corrupt_layout: Option<String>,
    /// Upload the WHOLE rank instead of the window the block needs. What a serving load does, and
    /// only possible with the GPU to ourselves: a rank of the device-final tier is 9.48 GB.
    pub full_rank: bool,
    /// Refuse to start if the block's device buffers would exceed this. Prod owns the card.
    pub budget_mib: u64,
    /// HIP device ordinal within the visible set.
    pub device: i32,
    /// Also replay the block and report µs/block.
    pub timing: bool,
    /// Timed replays.
    pub iters: usize,
    /// Replays discarded before timing starts.
    pub warmup: usize,
}

/// One compared buffer.
enum Verdict {
    Ok(usize),
    Mismatch {
        bytes: usize,
        bad: usize,
        first: usize,
    },
    Skipped(String),
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

/// Upload one host buffer into its own device allocation.
pub(crate) fn to_device(
    hip: &HipThread,
    stream: &Stream,
    bytes: &[u8],
    budget: &mut u64,
) -> anyhow::Result<DevicePtr> {
    let n = bytes.len() as u64;
    if n > *budget {
        anyhow::bail!(
            "the block's device buffers exceed the budget by {} MiB",
            (n - *budget) >> 20
        );
    }
    *budget -= n;
    let p = hip.malloc(n.max(1))?;
    hip.memcpy_h2d_blocking(&p, bytes, stream)?;
    Ok(p)
}

/// `<packdir>/rank<r>.pack`, or the file itself if that is what was passed.
pub(crate) fn pack_file(pack: &Path, rank: u32) -> PathBuf {
    if pack.is_dir() {
        pack.join(format!("rank{rank}.pack"))
    } else {
        pack.to_path_buf()
    }
}

fn hex12(h: &[u8; 32]) -> String {
    h.iter().take(6).map(|b| format!("{b:02x}")).collect()
}

/// Every weight record the block bound, with the layout it carries and — when the pack's manifest
/// sidecar is there — the checkpoint tensors it was built from.
///
/// This is the table that answers "did any of these bytes come from anywhere but the pack?". The
/// answer is structural rather than asserted: the only weight map the block is given is the one
/// filled from `RecordReady`, and there is no other source in this file any more.
fn print_provenance(pack: &Path, rows: &[(String, u32, u64, u64)]) -> anyhow::Result<()> {
    let manifest: Option<Value> = std::fs::read_to_string(pack.join("manifest.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok());
    let sources: BTreeMap<String, String> = manifest
        .as_ref()
        .and_then(|m| m.get("ranks"))
        .and_then(Value::as_array)
        .and_then(|r| r.first())
        .and_then(|r| r.get("records"))
        .and_then(Value::as_array)
        .map(|recs| {
            recs.iter()
                .filter_map(|r| {
                    let name = r.get("name")?.as_str()?.to_owned();
                    let src = r.get("sources")?.as_array()?;
                    Some((name, short_sources(src)))
                })
                .collect()
        })
        .unwrap_or_default();
    println!();
    println!(
        "{:<34} {:<20} {:>10} {:<18} built from",
        "record", "layout", "bytes", "device addr"
    );
    let mut by_layout: BTreeMap<u32, usize> = BTreeMap::new();
    for (name, layout, len, addr) in rows {
        *by_layout.entry(*layout).or_default() += 1;
        println!(
            "{:<34} {:<20} {len:>10} {:<#18x} {}",
            name,
            LayoutId::describe(*layout),
            addr,
            sources.get(name).map_or("-", String::as_str)
        );
    }
    println!(
        "{} weight records, all from the pack: {}",
        rows.len(),
        by_layout
            .iter()
            .map(|(l, n)| format!("{} x{n}", LayoutId::describe(*l)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(())
}

/// The source list a manifest record carries, shortened to the leaf names.
fn short_sources(src: &[Value]) -> String {
    let names: Vec<String> = src
        .iter()
        .filter_map(Value::as_str)
        .map(|n| {
            let short = n.rsplit_once(".layers.").map_or(n, |(_, rest)| rest);
            short.to_owned()
        })
        .collect();
    names.join(" + ")
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

/// Run the gate. Returns the number of mismatching buffers.
pub fn run(hip: &HipThread, args: &LayerArgs) -> anyhow::Result<usize> {
    hip.set_device(args.device)?;
    let block: Value = serde_json::from_str(
        &std::fs::read_to_string(args.data.join("block.json")).map_err(|e| {
            anyhow::anyhow!(
                "{}: {e} — run the internal reference harness first",
                args.data.join("block.json").display()
            )
        })?,
    )?;

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
        "block     layers {:?} of {} ({} tp), tier {}…",
        layers,
        graph.config.num_hidden_layers,
        tp,
        &s(&block, "tier")?[..12]
    );
    println!("reference {}", args.data.display());
    println!(
        "all_reduce {} — {}",
        s(&block, "all_reduce")?,
        s(&block, "all_reduce_note")?
    );
    println!(
        "plugins   {} loaded from {} ({} symbols); {}",
        plugins.plugins().len(),
        args.prebuilt.display(),
        functions.len(),
        load_counters.snapshot().line()
    );

    let mut budget = args.budget_mib << 20;

    // ---- the weights, ALL of them, out of the pack ------------------------------------------
    // Planned once at the decode bucket: the weight KEYS do not depend on M, and both cases read
    // the same records. Every one of them is a record in the device-final pack, at the layout id
    // the model's own plan gives it -- there is no second source of weights and no transform on
    // this path, which is the whole point of P2.
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

    let pack = args
        .pack
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--pack is required: every weight comes from the pack"))?;
    // Open the table first (no data read) to find the window that covers exactly the records this
    // block binds. The device-final pack is written in forward order, so "layers 0..N" IS a
    // contiguous window -- which is the property `docs/LEDGER.md` (workstream K) asked the builder
    // for and what makes a 9 GB pack loadable 500 MiB at a time.
    let reader = PackReader::open(&pack_file(&pack, args.rank))?;
    let mut idx: Vec<usize> = Vec::new();
    let mut layout_of: BTreeMap<String, u32> = BTreeMap::new();
    for (i, rec) in reader.records().iter().enumerate() {
        let name = reader.name(i);
        if want.contains(name) {
            idx.push(i);
            layout_of.insert(name.to_owned(), rec.layout_id);
        }
    }
    let missing: Vec<&String> = want
        .iter()
        .filter(|k| !layout_of.contains_key(*k))
        .collect();
    if !missing.is_empty() {
        anyhow::bail!(
            "{} record(s) the block binds are not in {}: {:?}{}. A raw (v0) pack cannot serve a \
             device-final plan -- get one with `tools/fetch_packs.sh`.",
            missing.len(),
            pack.display(),
            missing.iter().take(3).collect::<Vec<_>>(),
            if missing.len() > 3 { ", …" } else { "" }
        );
    }
    let (lo, hi) = if args.full_rank {
        (0, reader.records().len() - 1)
    } else {
        (idx[0], *idx.last().expect("non-empty"))
    };
    let last = &reader.records()[hi];
    let span =
        saber_mem::pack::align_up(last.data_off + last.length) - reader.records()[lo].data_off;
    println!(
        "pack      {} ({} records, plan {}…)",
        pack.display(),
        reader.records().len(),
        hex12(&reader.header().plan_hash)
    );
    if args.full_rank {
        println!(
            "window    the WHOLE rank: records {lo}..={hi} ({} of them, {} MiB) — what a serving \
             load does, and what the block's {} weight records are bound out of",
            hi - lo + 1,
            span >> 20,
            want.len()
        );
    } else {
        println!(
            "window    records {lo}..={hi} ({} of them, {} MiB) — the block's {} weight records \
             are contiguous because the pack is written in forward order",
            hi - lo + 1,
            span >> 20,
            want.len()
        );
    }

    let cfg = UploadConfig {
        start_record: lo,
        limit_bytes: Some(span),
        mode: OpenMode::Direct,
        ..UploadConfig::default()
    };
    let upload = Upload::open(&pack, args.rank, cfg)?;
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
    // The staging ring IS the pack stream's buffer pool (workstream K2): the drive reads into
    // these slots and the H2D sends from the same bytes.
    let mut staging = hip.host_malloc(reserved.staging_bytes())?;
    let ring = PinnedRing::new(&mut staging, &reserved)?;
    let mut sink = HipSink::new(hip, &reserved)?;
    let mut weights: BTreeMap<String, BoundWeight> = BTreeMap::new();
    let mut provenance: Vec<(String, u32, u64, u64)> = Vec::new();
    let mut landed_bytes = 0u64;
    let stats = upload.run(&mut sink, &ring, |r| {
        if let Some(layout) = layout_of.get(r.name) {
            weights.insert(r.name.to_string(), BoundWeight::new(r.addr, *layout));
            provenance.push((r.name.to_string(), *layout, r.len, r.addr.as_u64()));
            landed_bytes += r.len;
        }
    })?;
    println!(
        "weights   {} of {} records bound by name out of the arena, {} MiB, {:.2} GB/s verified \
         (sha256 per record, before the H2D)",
        weights.len(),
        want.len(),
        landed_bytes >> 20,
        stats.end_to_end_gbps()
    );
    if weights.len() != want.len() {
        anyhow::bail!(
            "{} of the block's records did not land in the window",
            want.len() - weights.len()
        );
    }
    // NEGATIVE CONTROL: bind one record at the wrong layout id on purpose. The bind must refuse
    // and name it; a gate nobody has watched fail is not a gate.
    if let Some(record) = &args.corrupt_layout {
        let Some(w) = weights.get_mut(record.as_str()) else {
            anyhow::bail!("--corrupt-layout `{record}` is not a record this block binds");
        };
        let was = w.layout_id;
        w.layout_id = w.layout_id.wrapping_add(1);
        println!(
            "CONTROL   `{record}` bound as layout {} instead of {was}: the block must refuse it",
            w.layout_id
        );
    }
    if args.provenance {
        print_provenance(&pack, &provenance)?;
    }

    // ---- the cases -------------------------------------------------------------------------
    let mut mismatches = 0usize;
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
        mismatches += run_case(
            hip, &ctx, &stream, args, &graph, &layers, &mxfp4, &saber, &weights, name, case, budget,
        )?;
    }
    // The sink owns the weight arena; the addresses bound above point into it.
    drop(sink);
    Ok(mismatches)
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

/// The per-call values the capture recorded, re-derived here and compared.
///
/// Not a formality: `kv_block_stride` and the conv window are what a paged kernel walks its cache
/// with, and a disagreement between the two halves of the gate would show up as a byte mismatch
/// with no explanation. Deriving them and checking turns that into a message.
fn check_derived(case: &Value, graph: &Graph, bucket: Bucket) -> anyhow::Result<()> {
    let cfg = &graph.config;
    let kv_heads = cfg.num_key_value_heads / graph.tp;
    let block_size = 16u64;
    let derived_block = u64::from(kv_heads) * block_size * 2 * u64::from(cfg.head_dim);
    let derived_head = block_size * 2 * u64::from(cfg.head_dim);
    let derived_blocks = u(case, "max_ctx")?.div_ceil(block_size);
    let derived_state_len =
        u64::from(cfg.linear_conv_kernel_dim - 1) + u64::from(bucket.q_len().saturating_sub(1));
    for (what, derived, recorded) in [
        (
            "kv_block_stride",
            derived_block,
            u(case, "kv_block_stride")?,
        ),
        ("kv_head_stride", derived_head, u(case, "kv_head_stride")?),
        ("kv_blocks", derived_blocks, u(case, "kv_blocks")?),
        (
            "gdn_state_len",
            if s(case, "phase")? == "decode" {
                derived_state_len
            } else {
                u64::from(cfg.linear_conv_kernel_dim - 1)
            },
            u(case, "gdn_state_len")?,
        ),
    ] {
        if derived != recorded {
            anyhow::bail!(
                "{what}: the engine derives {derived} from the graph and the bucket, the capture \
                 recorded {recorded}"
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn run_case(
    hip: &HipThread,
    ctx: &LaunchCtx<'_>,
    stream: &Stream,
    args: &LayerArgs,
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
    let params = case_params(case)?;
    check_derived(case, graph, bucket)?;
    let mut plan = BlockPlan::new(graph, bucket, layers, params, mxfp4, saber)?;
    // ws-h30: the reference called its kernel with `h0 = state[:batch]`, so this replay is pointed at
    // the capture's own base. A serving caller has a session pool and calls
    // `retarget_prefill_state`, which refuses slot 0 because the pool has a null slot there.
    if bucket.phase() == saber_model::graph::Phase::Prefill {
        plan.retarget_prefill_state_at_reference_base()?;
    }
    let plan = plan;
    let dir = args.data.join(case_name);

    println!();
    println!(
        "case      {case_name}: {:?}, M={}, {} ops ({} launched, {} identity, {} supplied)",
        bucket,
        bucket.m(),
        plan.steps.len(),
        plan.steps
            .iter()
            .filter(|s| matches!(s.route, Route::Kernel { .. }))
            .count(),
        plan.steps
            .iter()
            .filter(|s| matches!(s.route, Route::Identity { .. }))
            .count(),
        plan.supplied.values().sum::<u32>(),
    );
    let by_kind = plan.bytes_by_kind();
    println!(
        "arena     {} buffers, {} MiB ({}), one allocation, never reallocated (I1)",
        plan.buffers.len(),
        plan.arena_bytes >> 20,
        by_kind
            .iter()
            .map(|(k, v)| format!("{k} {} KiB", v >> 10))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "launches  {} declared by the graph for these ops, {} replaced by the identity \
         all-reduce, {} expected",
        plan.declared_launches,
        plan.identity_launches,
        plan.expected_launches()
    );

    let mut budget = budget;
    let mut owned: Vec<DevicePtr> = Vec::new();
    let mut inputs = BlockInputs {
        weights: weights.clone(),
        ..BlockInputs::default()
    };

    // ---- caller-owned state, from the capture ------------------------------------------------
    let mut state_ptrs: BTreeMap<String, usize> = BTreeMap::new();
    for name in plan.state_keys() {
        let (op, buf) = state_dump(&name)?;
        let bytes = npy::read_u8(&dir.join(format!("{op}.in.{buf}.npy")))?;
        let p = to_device(hip, stream, &bytes, &mut budget)?;
        inputs.states.insert(name.clone(), p.addr());
        state_ptrs.insert(name, owned.len());
        owned.push(p);
    }
    // ---- index tables a scheduler would build (P4) -------------------------------------------
    for key in plan.index_keys() {
        let (op, buf) = key
            .rsplit_once('.')
            .ok_or_else(|| anyhow::anyhow!("index key `{key}` has no buffer part"))?;
        let bytes = npy::read_u8(&dir.join(format!("{op}.in.{buf}.npy")))?;
        let p = to_device(hip, stream, &bytes, &mut budget)?;
        inputs.indices.insert(key.clone(), p.addr());
        owned.push(p);
    }

    // ---- the arena ---------------------------------------------------------------------------
    if plan.arena_bytes > budget {
        anyhow::bail!(
            "the activation arena is {} MiB and only {} MiB of budget is left",
            plan.arena_bytes >> 20,
            budget >> 20
        );
    }
    let staging = 8usize << 20;
    let mut exec = BlockExecutor::new(hip, &plan, staging)?;
    exec.zero(hip, stream)?;

    // Block inputs and the outputs of every op with no code object.
    let mut uploaded = 0usize;
    for edge in &plan.inputs {
        let b = plan
            .buffer(edge)
            .ok_or_else(|| anyhow::anyhow!("block input `{edge}` is not planned"))?;
        let bytes = npy::read_u8(&dir.join("block.in.hidden.npy"))?;
        if bytes.len() as u64 != b.bytes {
            anyhow::bail!(
                "block input `{edge}`: the plan says {} bytes, the capture dumped {}",
                b.bytes,
                bytes.len()
            );
        }
        exec.upload(hip, b.offset, &bytes, stream)?;
        uploaded += 1;
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
                anyhow::bail!(
                    "{}: supplied `{buf}` is {} bytes in the plan and {} in the capture",
                    step.name(),
                    bytes,
                    host.len()
                );
            }
            exec.upload(hip, *offset, &host, stream)?;
            uploaded += 1;
        }
    }
    hip.sync_stream(stream)?;
    println!(
        "uploaded  {uploaded} buffer(s): the residual stream entering layer {}, and every output \
         of the {} op(s) with no code object",
        layers[0],
        plan.supplied.values().sum::<u32>()
    );

    // ---- run ----------------------------------------------------------------------------------
    let counters = LaunchCounters::new();
    let mut launcher = Launcher::new(format!("block.{case_name}"), &counters);
    let run = exec.run(ctx, &mut launcher, stream, &plan, &inputs, false)?;
    hip.sync_stream(stream)?;

    // ---- compare --------------------------------------------------------------------------------
    println!();
    println!(
        "{:<34} {:<44} {:<12} {:>12} result",
        "op", "kernel", "buffer", "bytes"
    );
    let (mut pass, mut fail, mut skipped) = (0usize, 0usize, 0usize);
    let files = case
        .get("files")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("case `{case_name}` has no `files`"))?;
    for step in &plan.steps {
        let symbols = step.route.symbols().join(" + ");
        match &step.route {
            Route::Identity { .. } => {
                println!(
                    "{:<34} {:<44} {:<12} {:>12} IDENTITY (output edge aliases its input)",
                    step.name(),
                    "-",
                    "-",
                    "-"
                );
                continue;
            }
            Route::Supplied { .. } => {
                println!(
                    "{:<34} {:<44} {:<12} {:>12} SUPPLIED (no code object; uploaded)",
                    step.name(),
                    "-",
                    "-",
                    "-"
                );
                continue;
            }
            Route::Fused { into, .. } => {
                println!(
                    "{:<34} {:<44} {:<12} {:>12} FUSED into `{into}` (its dispatch does this op's work)",
                    step.name(),
                    "-",
                    "-",
                    "-"
                );
                continue;
            }
            Route::Kernel { .. } => {}
        }
        let Some(dumped) = files.get(step.name()).and_then(Value::as_object) else {
            println!(
                "{:<34} {:<44} {:<12} {:>12} SKIP (the capture has no dump for this op)",
                step.name(),
                symbols,
                "-",
                "-"
            );
            skipped += 1;
            continue;
        };
        for key in dumped.keys() {
            let Some(buf) = key.strip_prefix("out.") else {
                continue;
            };
            let want = npy::read_u8(&dir.join(format!("{}.{key}.npy", step.name())))?;
            let verdict = compare_buffer(
                hip,
                stream,
                &exec,
                &plan,
                &inputs,
                &owned,
                &state_ptrs,
                step,
                buf,
                &want,
            )?;
            match verdict {
                Verdict::Ok(bytes) => {
                    println!(
                        "{:<34} {:<44} {:<12} {bytes:>12} OK",
                        step.name(),
                        symbols,
                        buf
                    );
                    pass += 1;
                }
                Verdict::Mismatch { bytes, bad, first } => {
                    println!(
                        "{:<34} {:<44} {:<12} {bytes:>12} MISMATCH {bad} bytes (first at {first})",
                        step.name(),
                        symbols,
                        buf
                    );
                    fail += 1;
                }
                Verdict::Skipped(why) => {
                    println!(
                        "{:<34} {:<44} {:<12} {:>12} SKIP ({why})",
                        step.name(),
                        symbols,
                        buf,
                        "-"
                    );
                    skipped += 1;
                }
            }
        }
    }

    // The block's own output: the last layer's hidden state and residual.
    let last = *layers.last().expect("a block has layers");
    for (file, edge) in [
        ("block.out.hidden.npy", format!("l{last}.mlp.partial")),
        ("block.out.resid.npy", format!("l{last}.resid.b")),
    ] {
        let want = npy::read_u8(&dir.join(file))?;
        let b = plan
            .buffer(&edge)
            .ok_or_else(|| anyhow::anyhow!("`{edge}` is not planned"))?;
        let mut got = vec![0u8; want.len()];
        exec.download(hip, b.offset, &mut got, stream)?;
        match diff(&got, &want) {
            None => {
                println!(
                    "{:<34} {:<44} {:<12} {:>12} OK",
                    "BLOCK",
                    edge,
                    file.trim_end_matches(".npy")
                        .trim_start_matches("block.out."),
                    want.len()
                );
                pass += 1;
            }
            Some((bad, first)) => {
                println!(
                    "{:<34} {:<44} {:<12} {:>12} MISMATCH {bad} bytes (first at {first})",
                    "BLOCK",
                    edge,
                    file,
                    want.len()
                );
                fail += 1;
            }
        }
    }

    println!();
    println!("{case_name}: {pass} bit-exact, {fail} MISMATCH, {skipped} skipped");
    println!(
        "launches/block {} (graph budget {}), {}",
        run.launches,
        plan.expected_launches(),
        counters.snapshot().line()
    );

    // ---- timing ---------------------------------------------------------------------------------
    if args.timing {
        time_block(
            hip,
            ctx,
            stream,
            &mut exec,
            &plan,
            &inputs,
            &mut launcher,
            args,
            case_name,
        )?;
    }

    drop(exec);
    drop(owned);
    Ok(fail)
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

#[allow(clippy::too_many_arguments)]
fn compare_buffer(
    hip: &HipThread,
    stream: &Stream,
    exec: &BlockExecutor,
    plan: &BlockPlan<'_>,
    inputs: &BlockInputs,
    owned: &[DevicePtr],
    state_ptrs: &BTreeMap<String, usize>,
    step: &saber_exec::Step<'_>,
    buf: &str,
    want: &[u8],
) -> anyhow::Result<Verdict> {
    let Some(src) = step.source(buf) else {
        return Ok(Verdict::Skipped(format!(
            "`{buf}` is not a buffer this op binds"
        )));
    };
    let mut got = vec![0u8; want.len()];
    match src {
        Source::Arena {
            buffer,
            offset,
            bytes,
        } => {
            if want.len() as u64 != *bytes {
                return Ok(Verdict::Skipped(format!(
                    "the plan gives `{buffer}` {bytes} bytes here, the capture dumped {}",
                    want.len()
                )));
            }
            exec.download(hip, *offset, &mut got, stream)?;
        }
        Source::State { key: name, at } => {
            let idx = state_ptrs
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("state `{name}` was never uploaded"))?;
            let p = &owned[*idx];
            // A SLOT binding watches that slot, not the whole pool: since ws-h30 `chunk_scan`
            // writes `ht` in place into one slot and the reference dumps exactly one.
            let (offset, span) = match at {
                saber_exec::StateAt::Slot { offset, .. } => (*offset, want.len() as u64),
                _ => (0, p.len()),
            };
            if want.len() as u64 != span || offset + span > p.len() {
                return Ok(Verdict::Skipped(format!(
                    "state `{name}` is {} bytes on the device from {offset} and {} in the capture",
                    p.len(),
                    want.len()
                )));
            }
            hip.memcpy_d2h_at_blocking(&mut got, p, offset, stream)?;
        }
        Source::Weight { .. } | Source::Index(_) => {
            return Ok(Verdict::Skipped(
                "an input, not an output of this op".to_string(),
            ))
        }
    }
    let _ = (plan, inputs);
    Ok(match diff(&got, want) {
        None => Verdict::Ok(want.len()),
        Some((bad, first)) => Verdict::Mismatch {
            bytes: want.len(),
            bad,
            first,
        },
    })
}

/// Replay the block behind `hipEvent`s: µs/block, then per-op medians.
#[allow(clippy::too_many_arguments)]
fn time_block(
    hip: &HipThread,
    ctx: &LaunchCtx<'_>,
    stream: &Stream,
    exec: &mut BlockExecutor,
    plan: &BlockPlan<'_>,
    inputs: &BlockInputs,
    launcher: &mut Launcher,
    args: &LayerArgs,
    case_name: &str,
) -> anyhow::Result<()> {
    let start = hip.create_event(saber_hip::EventFlags::Default)?;
    let end = hip.create_event(saber_hip::EventFlags::Default)?;
    for _ in 0..args.warmup {
        exec.run(ctx, launcher, stream, plan, inputs, false)?;
    }
    hip.sync_stream(stream)?;
    let mut samples = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        hip.record_event(&start, stream)?;
        exec.run(ctx, launcher, stream, plan, inputs, false)?;
        hip.record_event(&end, stream)?;
        hip.sync_event(&end)?;
        samples.push((f64::from(hip.elapsed_ms(&start, &end)?) * 1000.0) as u64);
    }
    samples.sort_unstable();
    let p50 = samples[samples.len() / 2];
    let p05 = samples[samples.len() / 20];
    let p95 = samples[samples.len() * 19 / 20];

    // Per-op medians, on a second pass: the event pair around each op serialises the stream, so
    // the SUM of these is an upper bound on the block above, not a decomposition of it.
    let per_op_iters = args.iters.clamp(3, 10);
    let mut per_op: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for _ in 0..per_op_iters {
        let run = exec.run(ctx, launcher, stream, plan, inputs, true)?;
        for op in run.ops {
            if let Some(us) = op.micros {
                per_op.entry(op.name).or_default().push(us);
            }
        }
    }
    let mut rows: Vec<(String, u64)> = per_op
        .into_iter()
        .map(|(k, mut v)| {
            v.sort_unstable();
            let med = v[v.len() / 2];
            (k, med)
        })
        .collect();
    let serial: u64 = rows.iter().map(|(_, v)| v).sum();
    rows.sort_by_key(|(_, v)| std::cmp::Reverse(*v));

    println!();
    println!(
        "timing    {case_name}: {} replays after {} warm-ups, hipEvents around the WHOLE block, \
         one stream",
        args.iters, args.warmup
    );
    println!(
        "          µs/block p50 {p50} (p05 {p05}, p95 {p95}), {} launches/block, \
         {:.1} µs/launch",
        plan.expected_launches(),
        f64::from(u32::try_from(p50).unwrap_or(u32::MAX)) / f64::from(plan.expected_launches())
    );
    println!(
        "          µs/layer p50 {:.1} over {} layers ({} GDN + {} attention)",
        p50 as f64 / plan.layers.len() as f64,
        plan.layers.len(),
        plan.layers.len() - 1,
        1
    );
    println!(
        "          PROD IS SERVING ON BOTH CARDS while this runs, so the absolute microseconds \
         are under load and are not comparable across runs or harnesses."
    );
    println!(
        "          per-op medians of {per_op_iters} serialised replays (sum {serial} µs, an \
         UPPER bound on the block above -- the event pair drains the stream per op):"
    );
    for (name, us) in rows.iter().take(12) {
        println!("            {name:<40} {us:>7} µs");
    }
    if rows.len() > 12 {
        println!("            … {} more ops", rows.len() - 12);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pack_path_resolves_to_the_rank_file() {
        // A directory names the tier; a file names one rank of it. Both are accepted because the
        // upload takes the directory and the table read takes the file.
        assert_eq!(
            pack_file(Path::new("/packs/tier/rank1.pack"), 0),
            PathBuf::from("/packs/tier/rank1.pack"),
            "a file is used as given, whatever --rank says"
        );
    }

    #[test]
    fn the_provenance_table_shortens_a_source_to_its_layer_local_name() {
        let src = vec![
            Value::String("model.language_model.layers.4.mlp.gate_proj.weight_packed".into()),
            Value::String("model.language_model.layers.4.mlp.up_proj.weight_packed".into()),
        ];
        assert_eq!(
            short_sources(&src),
            "4.mlp.gate_proj.weight_packed + 4.mlp.up_proj.weight_packed"
        );
        assert_eq!(short_sources(&[]), "");
    }

    #[test]
    fn a_state_name_maps_to_the_capture_that_seeded_it() {
        assert_eq!(
            state_dump("l3.attn.kv_cache").unwrap(),
            ("layer.3.attn.kv".to_string(), "kv")
        );
        assert_eq!(
            state_dump("l12.gdn.recurrent_state").unwrap(),
            ("layer.12.gdn.state".to_string(), "state")
        );
        assert!(state_dump("nonsense").is_err());
    }
}
