//! `saber forward --tp 2` — one forward pass across BOTH cards, with the collectives real.
//!
//! The TP=1 pass (`crate::forward`) runs the whole model on one card and every `layer.allreduce`
//! is absent from its graph. This one runs the tensor-parallel model the tier was packed for: two
//! processes, one card each, each holding its own 8.83 GiB rank pack, and **129 all-reduces per
//! pass executed through the launcher's `IpcPeerVisible` workspaces** — the vocab-parallel
//! embedding's reduce plus two per decoder layer.
//!
//! # Why two processes
//!
//! `hipIpcOpenMemHandle` opens "an interprocess memory handle exported from **another process**"
//! and refuses one this process exported (`hipErrorInvalidContext`, found in ws-h5). So the parent
//! picks a free loopback port, spawns `--rank 0` and `--rank 1`, and the ranks meet over a
//! `TcpExchange`. Every handle, every digest and every barrier crosses that socket.
//!
//! # What each rank computes
//!
//! | stage | who | how |
//! |---|---|---|
//! | embedding | both | `saber_embed_bf16` over THIS rank's vocabulary window; ids it does not own write zeros |
//! | `embed_tokens.all_reduce` | both | real, in place over `hidden.0` — the sum is exactly the TP=1 embedding |
//! | 64 decoder layers | both | `BlockPlan` at TP=2 (shard widths), 128 collectives spliced in |
//! | row select | both | the last position's hidden state and residual |
//! | `model.norm`, `lm_head` | both | this rank's vocabulary shard of the logits |
//! | `logits.gather` | HOST | `KernelOp::LogitsGather` has no code object; the shards are swapped over the rendezvous socket |
//! | argmax | rank 0 | over the gathered full vocabulary |
//!
//! The gather and the argmax are host-side and the report says so on every run: the gather is an
//! all-GATHER, not the row selection the head plugin implements, and nothing in this tree
//! performs it on the device yet.
//!
//! # What is bit-exact, and what is bounded
//!
//! The all-reduce family is the plan's own band rule — `quant_min_bytes` = 128 KiB — so it is the
//! TOKEN COUNT that decides whether a pass is exact:
//!
//! * At **≤ 12 tokens** a message is `tokens × 5120 × 2 B` < 128 KiB and the exact family runs.
//!   The embedding's reduce is then bit-exact against a host reference, and that reference is the
//!   TP=1 embedding **by construction**: each token's row lives in exactly one rank's half of the
//!   table and the other rank contributes an exact zero, so `a + 0` is `a` in any rounding.
//! * Above it the rotated (wht6) family runs and the same comparison becomes a bounded one. The
//!   gate reports the relative RMS against the host reference and the band it took, because a
//!   lossy reduce that silently became the exact one — or the reverse — is the failure this band
//!   rule exists to prevent.
//!
//! GPU rules (CLAUDE.md): both cards, under BOTH card locks (`gpu.lock.0` then `gpu.lock.1`, in that order), **GPU idle** (prod is
//! down for this session; a refused `localhost:8080/metrics` is the idle check and the run prints
//! it), power cap untouched.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use saber_exec::layer::{
    BlockExecutor, BlockInputs, BlockPlan, BoundWeight, Collectives, PlanOptions, Route,
};
use saber_exec::tp2;
use saber_exec::{
    bind_steps, LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet, RankPeer,
    RuntimeParams, TcpExchange,
};
use saber_hip::{DevicePtr, EventFlags, HipThread, StreamFlags};
use saber_kernels::head::HeadRules;
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_mem::dio::OpenMode;
use saber_mem::ledger::{Ledger, Pool};
use saber_mem::pack::PackReader;
use saber_mem::upload::{HipSink, PinnedRing, Upload, UploadConfig};
use saber_model::graph::{Geometry, Graph, KernelOp, ModelConfig, Phase};

use crate::layer::{pack_file, to_device};

// ---- the session laws, which are `crate::forward`'s ------------------------------------------
//
// This file used to re-state them, with the note "if `forward.rs` ever exports them, this block
// should be deleted in the same commit". It did (ws-h15 moved them to `saber_exec::forward` and
// left `crate::forward::index_table` as the one-sequence-prefill adapter over them), and this
// block was not deleted — so the copy here stayed at the SIX keys a decoder block declared before
// the attention glue was bound, and the TP=2 pass refused `layer.11.attn.kv_append.k_scale` at
// startup for two workstreams. It also read `cache_idx = 0`, the kernels' NULL state slot, which
// no launch count can see. One law, one place; a key with no law is still a refusal.
use crate::forward::{i32_bytes, index_table};

/// bf16 bytes as f32, for the host references.
fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect()
}

/// The host's own all-reduce: `a + b` in f32, rounded back to bf16 the way the kernel does
/// (round-to-nearest-even), so "bit-exact" means bit-exact.
fn host_sum_bf16(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(a.len());
    for (x, y) in bf16_to_f32(a).into_iter().zip(bf16_to_f32(b)) {
        let s = x + y;
        let bits = s.to_bits();
        let lsb = (bits >> 16) & 1;
        let rounded = ((bits + 0x7fff + lsb) >> 16) as u16;
        out.extend_from_slice(&rounded.to_le_bytes());
    }
    out
}

/// Relative RMS of `got` against `want`, both bf16.
fn rel_rms(got: &[u8], want: &[u8]) -> f64 {
    let (g, w) = (bf16_to_f32(got), bf16_to_f32(want));
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (a, b) in g.iter().zip(&w) {
        num += f64::from(a - b) * f64::from(a - b);
        den += f64::from(*b) * f64::from(*b);
    }
    if den == 0.0 {
        return if num == 0.0 { 0.0 } else { f64::INFINITY };
    }
    (num / den).sqrt()
}

/// Raise the rules' exact/lossy threshold above every message this run can send, so that every
/// all-reduce resolves to the EXACT family.
///
/// This is the band rule stated once, as DATA, in the one place that owns it: `saber::resolve` picks
/// the family by comparing the message against `ar.quant_min_bytes` (`SaberRules::ar_wants_lossy`),
/// and nothing else in the plane consults the threshold. So a run that wants the exact family
/// everywhere raises the threshold rather than branching anywhere — no second law, and every
/// consumer of the band moves together (including `Planner::nq_fusion`, which takes the pair only
/// when the reduce resolves to `allreduce_exact`; see the caller's note).
///
/// The new threshold is the largest BAND plus one rather than `u64::MAX`, because that is the
/// true statement: this run sends nothing bigger, and a bigger message would still be rotated.
///
/// **The band, not the message** (H28): a reduce above `ar.max_bytes` is cut into row bands that
/// each fit the IPC scratch slot, so what the family rule sees is a band. Comparing the whole
/// message against the slot here would refuse `--ar-exact` at 14k tokens for a message no kernel
/// is ever handed.
///
/// # Errors
/// A single ROW wider than the slot, which is the one message the band law cannot cut and which
/// really does fall back to RCCL — a plan that silently reduced through a different transport is
/// not this experiment.
fn force_exact_all_reduce(
    saber: &mut SaberRules,
    max_rows: u32,
    hidden: u32,
) -> anyhow::Result<u64> {
    let bands = saber_kernels::saber::ar_bands(
        u64::from(max_rows),
        u64::from(hidden),
        2,
        saber.ar.max_bytes,
    )
    .map_err(|e| {
        anyhow::anyhow!(
            "--ar-exact: {e}; a message above the {} B IPC scratch slot that cannot be banded \
             falls back to RCCL and is not this plane's reduce at all",
            saber.ar.max_bytes
        )
    })?;
    let largest = bands
        .iter()
        .map(|b| b.rows * u64::from(hidden) * 2)
        .max()
        .unwrap_or(0);
    let was = saber.ar.quant_min_bytes;
    saber.ar.quant_min_bytes = largest + 1;
    Ok(was)
}

/// Which all-reduce family each reduce in a plan resolved to, by descriptor op id.
///
/// COUNTED, not assumed: `--ar-exact` changes one number in a rules table and the whole experiment
/// rests on that number having changed the 129 routes it is supposed to. A flag that silently
/// failed to bind would produce a plausible agreement table for the wrong arithmetic.
fn ar_families(plan: &BlockPlan<'_>) -> BTreeMap<String, u32> {
    let mut out: BTreeMap<String, u32> = BTreeMap::new();
    for step in &plan.steps {
        if step.op.kernel != KernelOp::AllReduce {
            continue;
        }
        let name = match &step.route {
            Route::Kernel { descs, .. } => descs
                .first()
                .map_or_else(|| "?".to_owned(), |d| d.op.clone()),
            Route::Identity { .. } => "identity".to_owned(),
            Route::Supplied { .. } => "supplied".to_owned(),
            Route::Fused { .. } => "fused".to_owned(),
        };
        *out.entry(name).or_default() += 1;
    }
    out
}

/// FNV-1a over bytes, for the cross-rank check on a dumped tensor. Not a security digest: it only
/// has to notice that two ranks hold different bytes.
fn fnv1a(b: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for byte in b {
        h ^= u64::from(*byte);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// bf16 bytes as `f32` bytes, little-endian. The conversion is exact, and it spares every reader
/// of a dump a bf16 dtype it may not have.
fn bf16_bytes_as_f32(b: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(b.len() * 2);
    for c in b.chunks_exact(2) {
        let bits = u32::from(u16::from_le_bytes([c[0], c[1]])) << 16;
        out.extend_from_slice(&f32::from_bits(bits).to_le_bytes());
    }
    out
}

/// Where the residual stream is readable after each decoder layer, as
/// `(step index to run through, edge, the layer whose output this is)`.
///
/// `layer.{i}.input_layernorm` writes `l{i}.resid.a = resid_in + hidden_in`, and those two are
/// layer `i-1`'s outputs — so that edge, read the moment the op retires, IS the hidden state
/// after layer `i-1`. The same quantity the reference server holds in its `residual` tensor once layer `i`'s
/// `input_layernorm` has run, which is what makes the two sides comparable without either of them
/// re-adding anything on the host.
///
/// Layer 0's own input (`hidden.0`, the reduced embedding) is included as layer `-1`: a dump that
/// starts at layer 0 cannot tell "the layers diverge" from "they were handed different inputs".
fn layer_dump_points(plan: &BlockPlan<'_>) -> Vec<(usize, String, i32)> {
    let mut out = Vec::new();
    for (i, step) in plan.steps.iter().enumerate() {
        let name = &step.op.name;
        let Some(rest) = name.strip_prefix("layer.") else {
            continue;
        };
        let Some((num, tail)) = rest.split_once('.') else {
            continue;
        };
        if tail != "input_layernorm" {
            continue;
        }
        let Ok(layer) = num.parse::<u32>() else {
            continue;
        };
        if layer == 0 {
            // Layer 0 adds no residual, so `l0.resid.a` is just the embedding: dump the reduced
            // embedding itself, at the step BEFORE this one, and call it layer -1.
            out.push((i.saturating_sub(1), "hidden.0".to_owned(), -1));
            continue;
        }
        out.push((i, format!("l{layer}.resid.a"), layer as i32 - 1));
    }
    out
}

/// The argv one rank is spawned with, **derived from `ForwardTp2Args`** rather than hand-listed.
///
/// A pure function, and separate from the spawn, because **this is where the TP=2 token gate was
/// lost once already**: the parent opens no HIP context and runs no pass, so every fact the ranks
/// need has to be here, and a flag that is parsed by the parent and not forwarded makes the ranks
/// run their DEFAULTS and exit 0. `--prompts 200 --compare … --both-arms` did exactly that until
/// ws-h23 — one prompt per rank, no comparison, a clean exit — and no test could see it because
/// the construction was inline in a `spawn` loop.
///
/// The `let ForwardTp2Args { .. }` below names **every field with no `..`**, so adding one to the
/// struct stops this compiling until the new field is either forwarded or named parent-only —
/// the difference between a list someone has to remember and a list the compiler keeps. Two tests
/// hold the rest: `the_ranks_are_spawned_with_every_flag_the_gate_sets` walks the struct's own
/// values and checks each reaches the child, and `every_non_default_flag_round_trips_to_the_child`
/// parses the argv back through the real clap definition, so a misspelled or mis-valued flag
/// fails too.
fn child_args(args: &ForwardTp2Args, rank_of: u32, port: u16) -> Vec<OsString> {
    let ForwardTp2Args {
        pack,
        prompt_ids,
        prompt_index,
        tokens,
        prompts,
        max_prompt,
        compare,
        both_arms,
        ar_exact,
        dump_layers,
        only_ids,
        require_complete,
        prebuilt,
        budget_mib,
        timing,
        iters,
        warmup,
        hip_lib,
        // PARENT-ONLY: the child is told its rank and the rendezvous port by this function's
        // arguments. `--rank-of`, NOT `--rank`: `--rank` is the PACK's rank on this subcommand,
        // and a child given only that would see no `--rank-of`, decide it was the parent, and
        // spawn two more. (It did, on 2026-09-19.)
        rank: _,
        port: _,
    } = args;

    let mut a: Vec<OsString> = Vec::new();
    // A free function rather than a closure: a closure capturing `a` mutably cannot coexist with
    // the direct `a.push` calls for the path arguments.
    fn flag(a: &mut Vec<OsString>, name: &str, value: &dyn std::fmt::Display) {
        a.push(OsString::from(name));
        a.push(OsString::from(value.to_string()));
    }
    a.push(OsString::from("forward"));
    flag(&mut a, "--tp", &2);
    flag(&mut a, "--rank-of", &rank_of);
    flag(&mut a, "--port", &port);
    a.push(OsString::from("--pack"));
    a.push(pack.clone().into_os_string());
    a.push(OsString::from("--prompt-ids"));
    a.push(prompt_ids.clone().into_os_string());
    flag(&mut a, "--prompt-index", prompt_index);
    flag(&mut a, "--tokens", tokens);
    a.push(OsString::from("--prebuilt"));
    a.push(prebuilt.clone().into_os_string());
    flag(&mut a, "--budget-mib", budget_mib);
    a.push(OsString::from("--hip-lib"));
    a.push(OsString::from(hip_lib));
    flag(&mut a, "--prompts", prompts);
    flag(&mut a, "--max-prompt", max_prompt);
    if let Some(reference) = compare {
        a.push(OsString::from("--compare"));
        a.push(reference.clone().into_os_string());
    }
    if *both_arms {
        a.push(OsString::from("--both-arms"));
    }
    if *ar_exact {
        a.push(OsString::from("--ar-exact"));
    }
    if let Some(dir) = dump_layers {
        a.push(OsString::from("--dump-layers"));
        a.push(dir.clone().into_os_string());
    }
    if !only_ids.is_empty() {
        a.push(OsString::from("--only-ids"));
        a.push(OsString::from(
            only_ids
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(","),
        ));
    }
    if *require_complete {
        a.push(OsString::from("--require-complete"));
    }
    if *timing {
        a.push(OsString::from("--timing"));
        flag(&mut a, "--iters", iters);
        flag(&mut a, "--warmup", warmup);
    }
    a
}

/// How long the parent waits for a rank before killing it.
const RANK_DEADLINE: Duration = Duration::from_secs(900);

/// One arm's answers: `(prompt id, prompt tokens, the greedy token)` per prompt.
/// The tokens one arm produced, in the shape [`crate::forward::compare`] reads. The last field is
/// this pass's own top-2 and is always `None` here: at TP=2 a rank's `logits.partial` is HALF the
/// vocabulary, so a top-2 of it is a top-2 of the wrong support. `--top2` is refused on this arm
/// rather than filled in with a shard's answer.
type ArmTokens = Vec<crate::forward::PassRow>;

/// Where one prompt's wall time went, in milliseconds.
///
/// Reported because the per-prompt loop re-does work a serving process would do once, and the
/// question "which part" has an answer that decides whether the launcher can be hoisted out of it:
/// `Launcher::ipc_key` carries the descriptor's SHAPE, so a hoisted launcher would allocate and
/// exchange a fresh peer-visible region for every distinct prompt length instead of reusing one.
#[derive(Clone, Copy, Default)]
struct PassCost {
    /// Both plans, the state caches, the index tables, both arenas.
    setup: f64,
    /// The peer-visible workspaces: allocate, export, open, zero, and agree the digests.
    ipc: f64,
    /// The pass itself, both ranks, between the two rendezvous.
    pass: f64,
    /// Everything, including the host gather and the argmax.
    total: f64,
    /// Norms this pass's plan actually folded into a reduce — a plan FACT, carried here because it
    /// is the only thing that makes `--both-arms` an experiment rather than the same run twice.
    /// It is 0 at every prompt length this command runs (see the band test below), and the report
    /// says so rather than leaving a reader to infer it from two identical tables.
    folded: u32,
    /// Reduces in this pass's plan that took the ROTATED family — 129 by default above 12 rows,
    /// and 0 under `--ar-exact`. Carried for the same reason as `folded`: the arm's arithmetic is
    /// a property of the plan and the report should not make a reader assume it.
    lossy_reduces: u32,
}

/// Which fusion arm a pass ran, for the report.
fn arm_name(fuse: bool) -> &'static str {
    if fuse {
        "the `_nq` fusion ON (the reduce carries the norm)"
    } else {
        "the `_nq` fusion OFF (the two-op path prod runs)"
    }
}

/// Arguments of `saber forward --tp 2`.
pub struct ForwardTp2Args {
    /// Prompts to run: 1 is the single pass, N makes this the token gate.
    pub prompts: usize,
    /// Skip corpus rows longer than this when `--prompts` selects them; it MUST match the
    /// reference capture's own limit or the two sides compare different lists.
    pub max_prompt: u32,
    /// A no-speculation reference to compare the greedy tokens against.
    pub compare: Option<PathBuf>,
    /// Run BOTH fusion arms and compare each against the reference. Note that the `_nq` fusion is
    /// unreachable at a prompt's row count unless `ar_exact` also raises the band (ws-h23).
    pub both_arms: bool,
    /// Force every all-reduce to the EXACT family by raising the rules' exact/lossy threshold
    /// above the largest message this run can send.
    ///
    /// The experiment it exists for: prod's TP=2 reference reduces LOSSILY above 128 KiB
    /// (`saber_ar_oneshot_2rank_wht6`, rel-RMS ~2%), and ws-h23 found that running Saber lossily too
    /// did not shrink the undecided tail — two independent rotations compound rather than cancel.
    /// With this on, Saber is lossless and prod is not, so the tail flip count says where the
    /// residual lives: near the TP=1 count or below means it is prod's reduce, unchanged means it
    /// is somewhere else.
    pub ar_exact: bool,
    /// Write the residual stream after every decoder layer, for each prompt, into this directory.
    ///
    /// The quantity is `l{i}.resid.a` — what `layer.{i}.input_layernorm` writes, which is the
    /// residual entering layer `i` plus the block output of layer `i-1`, i.e. **the hidden state
    /// after layer `i-1`** — and `head.resid` for the last layer. One row per prompt: the last
    /// position, the one the token comes from. It is dumped as `f32` so a reader needs no bf16
    /// dtype, and the conversion is exact.
    ///
    /// Slot reuse means every layer's `resid.a` is the SAME arena buffer, so the pass runs in
    /// slices and each tensor is read immediately after the op that wrote it. That is also why
    /// this is not free: 64 syncs per prompt.
    pub dump_layers: Option<PathBuf>,
    /// Run only these corpus row ids out of the selected list. The per-layer comparison is about
    /// four or five named prompts, not about a count.
    pub only_ids: Vec<u64>,
    /// Count a disagreement as a failure.
    pub require_complete: bool,
    /// The device-final tier pack DIRECTORY: each rank opens its own `rank<r>.pack`.
    pub pack: PathBuf,
    /// A JSON array of token ids, or a `.jsonl` corpus whose rows carry `token_ids`.
    pub prompt_ids: PathBuf,
    /// Row of the corpus to take.
    pub prompt_index: usize,
    /// Truncate the prompt to this many tokens (0 = the whole prompt).
    pub tokens: u32,
    /// Directory of committed `.hsaco` + `*.manifest.toml`.
    pub prebuilt: PathBuf,
    /// Refuse to start if a rank's device buffers would exceed this.
    pub budget_mib: u64,
    /// Replay the pass and report µs/forward.
    pub timing: bool,
    /// Timed replays.
    pub iters: usize,
    /// Replays discarded before timing starts.
    pub warmup: usize,
    /// This process's rank. `None` = the parent that spawns both.
    pub rank: Option<u32>,
    /// The loopback port the two ranks meet on.
    pub port: Option<u16>,
    /// The HIP runtime to load, passed through to the children.
    pub hip_lib: String,
}

/// Run the pass. Returns the number of failures.
pub fn run(hip: Option<&HipThread>, args: &ForwardTp2Args) -> anyhow::Result<usize> {
    if args.rank.is_none() && args.port.is_some() {
        // The shape a spawn loop takes: a child that was told where to meet but not who it is
        // would spawn two more children, each of which would do the same.
        anyhow::bail!(
            "--port without --rank-of: this process cannot tell whether it is the parent or a \
             rank. The parent picks the port; a rank is given both."
        );
    }
    match (args.rank, hip) {
        (Some(rank), Some(hip)) => rank_main(hip, rank, args),
        (Some(_), None) => anyhow::bail!("a rank process needs a HIP runtime"),
        (None, _) => spawn_ranks(args),
    }
}

/// The parent: pick a port, spawn both ranks, wait with a deadline.
fn spawn_ranks(args: &ForwardTp2Args) -> anyhow::Result<usize> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|e| anyhow::anyhow!("binding the rendezvous socket: {e}"))?;
    let port = listener.local_addr()?.port();
    drop(listener);

    println!("forward   TP=2: two processes, one card each, meeting on 127.0.0.1:{port}");
    println!(
        "          pack {} (rank0.pack on device 0, rank1.pack on device 1)",
        args.pack.display()
    );
    println!();

    let exe = std::env::current_exe()?;
    let mut children = Vec::new();
    for rank in [0u32, 1] {
        let mut cmd = std::process::Command::new(&exe);
        cmd.args(child_args(args, rank, port));
        children.push((rank, cmd.spawn()?));
    }

    let deadline = Instant::now() + RANK_DEADLINE;
    let mut failed = 0usize;
    for (rank, mut child) in children {
        loop {
            match child.try_wait()? {
                Some(status) => {
                    if !status.success() {
                        println!("rank {rank}: exited {status}");
                        failed += 1;
                    }
                    break;
                }
                None if Instant::now() > deadline => {
                    println!("rank {rank}: no result within {RANK_DEADLINE:?}; killing it");
                    let _ = child.kill();
                    let _ = child.wait();
                    failed += 1;
                    break;
                }
                None => std::thread::sleep(Duration::from_millis(50)),
            }
        }
    }
    Ok(failed)
}

/// One rank, in its own process.
#[allow(clippy::too_many_lines)]
fn rank_main(hip: &HipThread, rank: u32, args: &ForwardTp2Args) -> anyhow::Result<usize> {
    let port = args
        .port
        .ok_or_else(|| anyhow::anyhow!("--rank needs --port"))?;
    hip.set_device(rank as i32)?;
    let exchange: TcpExchange = if rank == 0 {
        let listener = std::net::TcpListener::bind(("127.0.0.1", port))
            .map_err(|e| anyhow::anyhow!("rank 0: binding port {port}: {e}"))?;
        TcpExchange::accept(&listener, Duration::from_secs(120))?
    } else {
        TcpExchange::connect(port, Duration::from_secs(120))?
    };
    let peer = Arc::new(RankPeer::new(Arc::new(exchange)));

    let graph = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2)?;
    let mxfp4 = LaunchRules::prod_mxfp4()?;
    let mut saber = SaberRules::prod_saber()?;
    let head_rules = HeadRules::prod_head()?;

    // The prompt LIST, not one prompt: `--prompts N` makes this the 200-prompt token gate at TP=2,
    // which is the only tensor parallelism where the `_nq` fusion exists at all (at TP=1 there are
    // no all-reduces to fuse into). Everything before this line happens once per PROCESS — the
    // rendezvous, the plugins, the 7.7 GiB of weights per rank — and everything inside `one_pass`
    // happens once per PROMPT, because the plans, the caches, the tables and both arenas are all
    // sized by the prompt's length.
    let prompts = crate::forward::read_prompt_list(
        &args.prompt_ids,
        args.prompt_index,
        args.prompts,
        args.max_prompt,
    )?;
    // Named rows, not the first N. `read_prompt_list` selects by position and the comparison is
    // about specific corpus ids, so the filter is applied here — and REFUSES an id the selection
    // does not contain rather than quietly running a shorter list.
    let prompts = if args.only_ids.is_empty() {
        prompts
    } else {
        let kept: Vec<(u64, Vec<i32>)> = prompts
            .iter()
            .filter(|(id, _)| args.only_ids.contains(id))
            .cloned()
            .collect();
        let missing: Vec<u64> = args
            .only_ids
            .iter()
            .filter(|id| !kept.iter().any(|(k, _)| k == *id))
            .copied()
            .collect();
        anyhow::ensure!(
            missing.is_empty(),
            "--only-ids names {missing:?}, which --prompts {} did not select from {}; raise \
             --prompts or check --max-prompt",
            args.prompts,
            args.prompt_ids.display()
        );
        kept
    };
    let single = prompts.len() == 1;

    // ---- the all-reduce band, if this run is the exact arm --------------------------------------
    //
    // AFTER the prompt list, because the threshold is a function of the LONGEST prompt: the flag
    // says "every message this run sends takes the exact family", and that sentence needs the
    // messages. One number in one table, and everything that reads the band follows it.
    if args.ar_exact {
        let longest = prompts
            .iter()
            .map(|(_, ids)| u32::try_from(ids.len()).unwrap_or(u32::MAX))
            .max()
            .unwrap_or(1);
        let was = force_exact_all_reduce(&mut saber, longest, graph.config.hidden_size)?;
        if rank == 0 {
            println!(
                "ar-exact  quant_min_bytes {was} -> {} B: every reduce this run issues takes the \
                 EXACT family (the longest prompt is {longest} tokens = {} B)",
                saber.ar.quant_min_bytes,
                u64::from(longest) * u64::from(graph.config.hidden_size) * 2
            );
        }
    }
    let saber = saber;

    // ---- once per PROCESS ----------------------------------------------------------------------
    //
    // The comment above has always said the plugins and the 7.7 GiB of weights live out here. They
    // did not: both were inside the per-prompt closure, so a 200-prompt gate re-loaded every code
    // object and re-opened, re-hashed and re-uploaded the whole rank window 400 times across the
    // two arms. Neither is a function of the prompt — a plan's weight SET is decided by which
    // sites it runs, not by its row count, which is the same reason `saber forward` probes it at
    // one length (ws-h18).
    let layers: Vec<u32> = (0..graph.config.num_hidden_layers).collect();
    let hidden = graph.config.hidden_size;
    let vocab_local = match graph
        .ops
        .iter()
        .find(|o| o.name == "lm_head")
        .map(|o| &o.geometry)
    {
        Some(Geometry::Gemm { n, .. }) => *n,
        _ => anyhow::bail!("the graph has no `lm_head` GEMM"),
    };

    let load_counters = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &load_counters)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;

    // ---- weights, from a PROBE plan at the first prompt's length --------------------------------
    let probe_n = u32::try_from(prompts[0].1.len())?;
    let probe_params = RuntimeParams {
        max_ctx: probe_n,
        gdn_slots: saber_exec::forward::FIRST_STATE_SLOT + 1,
    };
    let probe_body = BlockPlan::with_options(
        &graph,
        graph.bucket_for(probe_n, Phase::Prefill),
        &layers,
        probe_params,
        &mxfp4,
        &saber,
        PlanOptions::BODY
            .rank(rank)
            .with_collectives(Collectives::Real),
    )?;
    let mut want = probe_body.weight_keys();
    want.insert("model.language_model.embed_tokens.weight".to_string());
    want.insert("model.language_model.norm.weight".to_string());
    for k in ["lm_head.w", "lm_head.ws", "lm_head.wref"] {
        want.insert(k.to_string());
    }
    let reader = PackReader::open(&pack_file(&args.pack, rank))?;
    let mut layout_of: BTreeMap<String, u32> = BTreeMap::new();
    let mut idx: Vec<usize> = Vec::new();
    for (i, rec) in reader.records().iter().enumerate() {
        let name = reader.name(i);
        if want.contains(name) {
            idx.push(i);
            layout_of.insert(name.to_owned(), rec.layout_id);
        }
    }
    if layout_of.len() != want.len() {
        let missing: Vec<&String> = want
            .iter()
            .filter(|k| !layout_of.contains_key(*k))
            .collect();
        anyhow::bail!(
            "rank {rank}: {} record(s) are not in {}: {:?}",
            missing.len(),
            args.pack.display(),
            missing.iter().take(3).collect::<Vec<_>>()
        );
    }
    let (lo, hi) = (idx[0], *idx.last().expect("non-empty"));
    let last = &reader.records()[hi];
    let span =
        saber_mem::pack::align_up(last.data_off + last.length) - reader.records()[lo].data_off;
    let mut process_budget = args.budget_mib << 20;
    if span > process_budget {
        anyhow::bail!(
            "rank {rank}: the weight window is {} MiB and the budget is {} MiB",
            span >> 20,
            process_budget >> 20
        );
    }
    process_budget -= span;
    let upload = Upload::open(
        &args.pack,
        rank,
        UploadConfig {
            start_record: lo,
            limit_bytes: Some(span),
            mode: OpenMode::Direct,
            ..UploadConfig::default()
        },
    )?;
    let uplan = upload.plan();
    let mut ledger = Ledger::new();
    ledger.declare(Pool::Vram(rank), uplan.arena_bytes.max(1))?;
    ledger.declare(Pool::HostPinned, uplan.staging_bytes.max(1))?;
    let reserved = upload.reserve(&mut ledger)?;
    let mut staging = hip.host_malloc(reserved.staging_bytes())?;
    let ring = PinnedRing::new(&mut staging, &reserved)?;
    let mut sink = HipSink::new(hip, &reserved)?;
    let mut weights: BTreeMap<String, BoundWeight> = BTreeMap::new();
    let stats = upload.run(&mut sink, &ring, |r| {
        if let Some(layout) = layout_of.get(r.name) {
            weights.insert(r.name.to_string(), BoundWeight::new(r.addr, *layout));
        }
    })?;
    if rank == 0 {
        println!(
            "weights   {} records per rank, {} MiB, {:.2} GB/s verified — ONCE, before the \
             prompt loop",
            weights.len(),
            span >> 20,
            stats.end_to_end_gbps()
        );
    }
    let weights = weights;

    // One prompt, one pass, both ranks in lockstep. A FnMut closure rather than a function because
    // it borrows a dozen per-process things and a twelve-parameter helper would be worse; `fuse`
    // is the arm, so the same process can answer "does the `_nq` fusion change the token".
    let one_pass = |prompt_id: u64,
                    ids: &[i32],
                    fuse: bool,
                    verbose: bool|
     -> anyhow::Result<(i32, u32, PassCost)> {
        let n = u32::try_from(ids.len())?;
        let t_start = Instant::now();

        // The body and the head, both planned — which ws-h16 made possible at TP > 1: the embedding
        // takes THIS rank's vocabulary window, `logits.gather` is Supplied so the head plans at all,
        // and `layer.allreduce` resolves to the real reduce. The hand-resolved embedding, norm and
        // lm_head this command used to carry are gone with the splice.
        let params = RuntimeParams {
            max_ctx: n,
            gdn_slots: saber_exec::forward::FIRST_STATE_SLOT + 1,
        };
        let mut body = BlockPlan::with_options(
            &graph,
            graph.bucket_for(n, Phase::Prefill),
            &layers,
            params,
            &mxfp4,
            &saber,
            {
                // THE ARM. The `_nq` fusion exists only above TP=1, so this is the one place a token
                // gate can ask whether it changes the answer; `--both-arms` runs the loop twice.
                let o = PlanOptions::BODY
                    .rank(rank)
                    .with_collectives(Collectives::Real);
                if fuse {
                    o
                } else {
                    o.without_fusion()
                }
            },
        )?;
        // The same body with the `_nq` fusion off (ws-h17). The gate below has to OBSERVE the reduced
        // embedding, and the fused epilogue never writes it to memory — it norms and quantizes it out
        // of registers — so the one step the gate stops in front of is taken from this twin. Same
        // arena, same addresses (asserted), so it runs in the body's own executor.
        let mut body_unfused = BlockPlan::with_options(
            &graph,
            graph.bucket_for(n, Phase::Prefill),
            &layers,
            params,
            &mxfp4,
            &saber,
            PlanOptions::BODY
                .rank(rank)
                .with_collectives(Collectives::Real)
                .without_fusion(),
        )?;
        // The chunk's live rows, on BOTH arms and on BOTH ranks (ws-h27). A bucket of exactly the
        // prompt's length makes this the identity; stating it is what keeps this pass and the
        // engine's padded chunk on one law.
        body.retarget_prefill_rows(n, &saber)?;
        body_unfused.retarget_prefill_rows(n, &saber)?;
        // AND THE RECURRENT STATE AT ITS SLOT, on BOTH arms and BOTH ranks (ws-h30): the scan
        // indexes `h0`/`ht` by `blockIdx.z` and now writes `ht` in place, so a rank left at the
        // pool's base would scan the prompt through the null slot and publish into it.
        let state_slot = saber_exec::forward::FIRST_STATE_SLOT;
        body.retarget_prefill_state(state_slot)?;
        body_unfused.retarget_prefill_state(state_slot)?;
        anyhow::ensure!(
            body.arena_bytes == body_unfused.arena_bytes
                && body
                    .buffers
                    .iter()
                    .zip(&body_unfused.buffers)
                    .all(|(a, b)| a.name == b.name && a.offset == b.offset && a.bytes == b.bytes),
            "the fused and unfused bodies disagree about the arena"
        );
        // The head at the SAMPLED rows: `lm_head` at the chunk's row count would write a shard of
        // logits per token, of which the sampler reads one. `head_bucket` is where that rule
        // lives, so this site cannot drift from the body's bucket (`Bucket::sampled_rows`).
        let head = BlockPlan::with_options(
            &graph,
            graph.head_bucket(graph.bucket_for(n, Phase::Prefill)),
            &[],
            params,
            &mxfp4,
            &saber,
            PlanOptions::HEAD.rank(rank),
        )?;

        if rank == 0 && verbose {
            println!(
                "model     Qwen3.8-27B at TP=2: {} layers, hidden {hidden}, vocab {} per rank",
                graph.config.num_hidden_layers, vocab_local
            );
            println!("prompt    {n} tokens from {}", args.prompt_ids.display());
            println!(
                "body      {} ops over {} layers, arena {} MiB",
                body.steps.len(),
                layers.len(),
                body.arena_bytes >> 20
            );
        }

        // The weights are already resident (once per process, above). What is checked per prompt
        // is that THIS plan binds nothing the probe plan did not: a weight set that turned out to
        // be a function of the row count would otherwise bind a stale address rather than fail.
        let mut budget = process_budget;
        let unbound: Vec<String> = body
            .weight_keys()
            .into_iter()
            .chain(head.weight_keys())
            .filter(|k| !weights.contains_key(k))
            .collect();
        anyhow::ensure!(
            unbound.is_empty(),
            "rank {rank}: {} weight(s) this plan binds were not in the uploaded window \
             ({:?}); the window is probed at one prompt length and this one needs another",
            unbound.len(),
            unbound.iter().take(3).collect::<Vec<_>>()
        );

        // ---- session state and index tables --------------------------------------------------------
        let mut owned: Vec<DevicePtr> = Vec::new();
        let mut inputs = BlockInputs {
            weights: weights.clone(),
            ..BlockInputs::default()
        };
        let kv_blocks = n.div_ceil(16).max(1);
        for step in &body.steps {
            for st in &step.op.state {
                if inputs.states.contains_key(&st.name) {
                    continue;
                }
                let bytes = match st.bytes_per_slot {
                    Some(b) => b * u64::from(params.gdn_slots),
                    None => match step.op.geometry {
                        Geometry::Attn {
                            head_dim,
                            block_size,
                            kv_heads,
                            ..
                        } => {
                            u64::from(kv_blocks)
                                * u64::from(kv_heads)
                                * u64::from(block_size)
                                * 2
                                * u64::from(head_dim)
                        }
                        _ => anyhow::bail!("state `{}` states no size", st.name),
                    },
                };
                budget = budget
                    .checked_sub(bytes)
                    .ok_or_else(|| anyhow::anyhow!("state `{}` exceeds the budget", st.name))?;
                let p = hip.malloc(bytes.max(1))?;
                hip.memset(&p, 0, &stream)?;
                inputs.states.insert(st.name.clone(), p.addr());
                owned.push(p);
            }
        }
        for key in body.index_keys() {
            let bytes = index_table(&key, n, kv_blocks, ids)?;
            let p = to_device(hip, &stream, &bytes, &mut budget)?;
            inputs.indices.insert(key.clone(), p.addr());
            owned.push(p);
        }
        hip.sync_stream(&stream)?;

        // ---- the two arenas and the row selection ---------------------------------------------------
        if body.arena_bytes + head.arena_bytes > budget {
            anyhow::bail!(
                "rank {rank}: the arenas are {} MiB and {} MiB is left",
                (body.arena_bytes + head.arena_bytes) >> 20,
                budget >> 20
            );
        }
        budget -= body.arena_bytes + head.arena_bytes;
        let mut body_exec = BlockExecutor::new(hip, &body, 8usize << 20)?;
        let mut head_exec = BlockExecutor::new(hip, &head, 8usize << 20)?;
        body_exec.zero(hip, &stream)?;
        head_exec.zero(hip, &stream)?;
        // The head reads the LAST position of the sequence, so one index.
        let sel = to_device(hip, &stream, &i32_bytes(&[n as i32 - 1]), &mut budget)?;
        let row_gather = saber_exec::forward::Head::resolve(hidden, vocab_local, &head_rules)?;
        // The two edges the head reads and the body wrote, asked of the head plan so the names are the
        // graph's.
        let carried: Vec<String> = head.inputs.clone();
        if carried.len() != 2 {
            anyhow::bail!(
                "rank {rank}: the head reads {} edges from outside its plan ({carried:?}); this \
             command selects rows for exactly the hidden state and the residual",
                carried.len()
            );
        }
        let addr = |p: &BlockPlan<'_>,
                    base: saber_hip::DeviceAddr,
                    edge: &str|
         -> anyhow::Result<saber_hip::DeviceAddr> {
            let b = p
                .locate(edge)
                .ok_or_else(|| anyhow::anyhow!("`{edge}` is not a planned buffer"))?;
            base.offset(b.offset)
                .ok_or_else(|| anyhow::anyhow!("`{edge}` runs past its arena"))
        };

        // The band, counted on the plan this pass will actually run.
        let families = ar_families(&body);
        let lossy: u32 = families
            .iter()
            .filter(|(k, _)| k.as_str() == "allreduce_wht")
            .map(|(_, v)| *v)
            .sum();
        anyhow::ensure!(
            !args.ar_exact || lossy == 0,
            "rank {rank}: --ar-exact was asked and {lossy} reduce(s) still resolved to \
             `allreduce_wht` at {n} tokens; the band rewrite did not bind and this pass would \
             measure the arithmetic the flag exists to remove"
        );
        anyhow::ensure!(
            args.ar_exact || lossy > 0 || n <= 12,
            "rank {rank}: {n} tokens is above the rules' 128 KiB band and no reduce resolved to \
             `allreduce_wht`; the default arm is supposed to be the lossy one prod runs"
        );
        let t_setup = t_start.elapsed();
        let counters = LaunchCounters::new();
        let mut launcher =
            Launcher::new(format!("forward.tp2.r{rank}"), &counters).with_peer(Arc::clone(&peer));

        // ---- everything the collectives need, before anything launches -----------------------------
        let bound = bind_steps(&body_exec, &body.steps, &inputs)?;
        let bound_unfused = bind_steps(&body_exec, &body_unfused.steps, &inputs)?;
        for step in bound.iter().chain(&bound_unfused) {
            for desc in step.descs {
                launcher.prepare_workspaces(&ctx, desc, &step.args, &stream)?;
            }
        }
        launcher.reset_ipc_state(&ctx, &stream)?;
        peer.rendezvous("workspaces-up")?;
        let agreed = tp2::agree_on_collectives(&launcher, &peer, &bound)?;
        // The gate launches the unfused reduce, so its digest is agreed before it runs too.
        let agreed_unfused = tp2::agree_on_collectives(&launcher, &peer, &bound_unfused)?;
        let _ = agreed_unfused;
        let collectives = tp2::collective_steps(&bound).len();
        if rank == 0 && verbose {
            println!(
                "ipc       {} MiB peer-visible per rank, {collectives} collective(s) in the body, \
             {agreed} launch digest(s) agreed before anything launched",
                launcher.ipc_bytes() >> 20
            );
            if let Some(c) = tp2::collective_steps(&bound).first() {
                println!(
                    "          the family the rules picked: {}",
                    c.descs[0].kernel
                );
                // The band count is the H28 number: a message above the IPC scratch slot is cut
                // into row bands, one dispatch each, so `agreed` is the DISPATCH count and
                // `collectives` the op count. They are equal below 8,601 rows at hidden 5120.
                println!(
                    "          row bands per reduce: {} ({agreed} dispatch(es) over \
                     {collectives} reduce(s))",
                    c.descs.len()
                );
            }
            println!(
                "          the plan's own tally: {}",
                families
                    .iter()
                    .map(|(k, v)| format!("{k} x{v}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        drop(bound);
        let t_ipc = t_start.elapsed();

        // ---- one pass ---------------------------------------------------------------------------------
        let pass = |body_exec: &mut BlockExecutor,
                    head_exec: &mut BlockExecutor,
                    launcher: &mut Launcher|
         -> anyhow::Result<u32> {
            let mut total = body_exec
                .run_steps(&ctx, launcher, &stream, &body.steps, &inputs, false)?
                .launches;
            for edge in &carried {
                total += row_gather.select(
                    &ctx,
                    launcher,
                    &stream,
                    addr(&body, body_exec.arena_base(), edge)?,
                    sel.addr(),
                    addr(&head, head_exec.arena_base(), edge)?,
                )?;
            }
            total += head_exec
                .run_steps(&ctx, launcher, &stream, &head.steps, &inputs, false)?
                .launches;
            tp2::wait_bounded(&ctx, &stream, "the pass")?;
            Ok(total)
        };

        // The embedding-reduce gate is a once-per-RUN statement — the vocab-parallel contract does not
        // change from prompt to prompt — so a 200-prompt loop proves it on the first pass and then
        // spends its time on tokens instead of 200 host sums and 200 shard swaps.
        if verbose {
            // ---- the gate the collective earns: the embedding's reduce against a host sum ---------------
            //
            // The reduce is a step of the plan now, so the partial it consumes is only observable by
            // stopping in front of it: run the embedding, read `hidden.0`, exchange it, then run the ONE
            // all-reduce step and compare. The sum of the two ranks' partials IS the TP=1 embedding, by
            // construction — each token's row lives in exactly one rank's half of the table and the other
            // writes an exact zero — so below the rules' 128 KiB threshold, where the exact family runs,
            // this is a bit-exact check of the whole vocab-parallel contract.
            let ar_idx = body_unfused
                .steps
                .iter()
                .position(|s| s.op.kernel == KernelOp::AllReduce)
                .ok_or_else(|| {
                    anyhow::anyhow!("the body plans no all-reduce; is this a TP=1 graph?")
                })?;
            // `hidden.0` is the reduce's OUTPUT, which is an alias of its input — that alias is what makes
            // the reduce in place — and `locate` follows it (ws-h16), so a caller names the edge the graph
            // names.
            let h0 = body
                .locate("hidden.0")
                .ok_or_else(|| anyhow::anyhow!("`hidden.0` is not planned"))?;
            let (h0_off, h0_bytes) = (h0.offset, h0.bytes);
            peer.rendezvous("gate-start")?;
            body_exec.run_steps(
                &ctx,
                &mut launcher,
                &stream,
                &body_unfused.steps[..ar_idx],
                &inputs,
                false,
            )?;
            tp2::wait_bounded(&ctx, &stream, "the embedding")?;
            let mut partial = vec![0u8; usize::try_from(h0_bytes)?];
            body_exec.download(hip, h0_off, &mut partial, &stream)?;
            let theirs_partial = peer
                .exchange()
                .swap_bytes("embed.partial", &partial)
                .map_err(|e| anyhow::anyhow!("exchanging the embedding partials: {e}"))?;
            let want = host_sum_bf16(&partial, &theirs_partial);
            body_exec.run_steps(
                &ctx,
                &mut launcher,
                &stream,
                &body_unfused.steps[ar_idx..=ar_idx],
                &inputs,
                false,
            )?;
            tp2::wait_bounded(&ctx, &stream, "the embedding's reduce")?;
            let mut got = vec![0u8; usize::try_from(h0_bytes)?];
            body_exec.download(hip, h0_off, &mut got, &stream)?;
            let family = tp2::collective_steps(&bound_unfused)
                .first()
                .map_or_else(|| "?".to_string(), |c| c.descs[0].kernel.clone());
            let exact = family.contains("exact");
            let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            let rms = rel_rms(&got, &want);
            if rank == 0 {
                if exact {
                    println!(
                "gate      embedding + the EXACT reduce vs the host sum of both partials: {bad} of \
                 {} bytes differ, rel-RMS {rms:.2e} — {}",
                want.len(),
                if bad == 0 { "BIT-EXACT" } else { "MISMATCH" }
            );
                    if body.fused_nq > 0 {
                        println!(
                    "          (that one step is the UNFUSED twin of the reduce: the pass itself \
                     runs {} fused reduce(s), whose epilogue keeps the sum in registers and never \
                     writes it, so it cannot be read back. `saber step-tp2` gates the fused \
                     dispatch against the two-op plan byte for byte.)",
                    body.fused_nq
                );
                    }
                } else {
                    println!(
                "gate      embedding + the ROTATED reduce ({family}) vs the host sum: rel-RMS \
                 {rms:.4} ({bad} of {} bytes differ). Lossy BY DESIGN above the rules' 128 KiB \
                 threshold; run with fewer tokens for the bit-exact band.",
                want.len()
            );
                }
            }
            // Refusals, not tallies: the embedding gate runs once per RUN, so a counter that the
            // prompt loop never reads would be a silent pass.
            anyhow::ensure!(
                !(exact && bad != 0),
                "rank {rank}: the exact reduce differs from the host sum in {bad} byte(s)"
            );
            anyhow::ensure!(
                exact || matches!(rms.partial_cmp(&0.15), Some(std::cmp::Ordering::Less)),
                "rank {rank}: the rotated reduce is {rms:.4} off the host sum, over the 0.15 bound"
            );
        }
        peer.rendezvous("pass-start")?;
        let launched = if let Some(dir) = &args.dump_layers {
            // The body in SLICES, reading the residual stream the moment each layer's opening
            // norm has written it. Slot reuse makes every `resid.a` the same buffer, so "after
            // the pass" is too late: at the end it holds layer 63's value 64 times over.
            let points = layer_dump_points(&body);
            anyhow::ensure!(
                points.len() == graph.config.num_hidden_layers as usize,
                "rank {rank}: {} dump point(s) for {} layers; the plan's op names changed",
                points.len(),
                graph.config.num_hidden_layers
            );
            std::fs::create_dir_all(dir)?;
            let row = usize::try_from(n - 1)?;
            let mut total = 0u32;
            let mut cursor = 0usize;
            for (idx, edge, layer) in &points {
                let upto = *idx + 1;
                if upto > cursor {
                    total += body_exec
                        .run_steps(
                            &ctx,
                            &mut launcher,
                            &stream,
                            &body.steps[cursor..upto],
                            &inputs,
                            false,
                        )?
                        .launches;
                    cursor = upto;
                }
                tp2::wait_bounded(&ctx, &stream, "a dump slice")?;
                let b = body
                    .locate(edge)
                    .ok_or_else(|| anyhow::anyhow!("`{edge}` is not a planned buffer"))?;
                let width = usize::try_from(b.bytes)? / usize::try_from(n)?;
                let mut got = vec![0u8; width];
                body_exec.download(hip, b.offset + (row * width) as u64, &mut got, &stream)?;
                // Both ranks hold the reduced stream, so they must agree byte for byte; a dump
                // that silently took one rank's partial would look like a divergence at layer 0
                // and stay one for 64 layers.
                let mine = fnv1a(&got);
                let theirs = peer
                    .exchange()
                    .swap_digest(&format!("dump-{layer}"), mine)
                    .map_err(|e| anyhow::anyhow!("exchanging a dump digest: {e}"))?;
                anyhow::ensure!(
                    mine == theirs,
                    "rank {rank}: the two ranks disagree about the residual stream after layer \
                     {layer}; this is a reduced edge and they must hold the same bytes"
                );
                if rank == 0 {
                    std::fs::write(
                        dir.join(format!("p{prompt_id}_l{layer:+03}.f32")),
                        bf16_bytes_as_f32(&got),
                    )?;
                }
            }
            total += body_exec
                .run_steps(
                    &ctx,
                    &mut launcher,
                    &stream,
                    &body.steps[cursor..],
                    &inputs,
                    false,
                )?
                .launches;
            for edge in &carried {
                total += row_gather.select(
                    &ctx,
                    &mut launcher,
                    &stream,
                    addr(&body, body_exec.arena_base(), edge)?,
                    sel.addr(),
                    addr(&head, head_exec.arena_base(), edge)?,
                )?;
            }
            // The last layer's output is the head's own residual: the final norm adds layer 63's
            // block output to the stream exactly as a 65th `input_layernorm` would.
            total += head_exec
                .run_steps(
                    &ctx,
                    &mut launcher,
                    &stream,
                    &head.steps[..1],
                    &inputs,
                    false,
                )?
                .launches;
            tp2::wait_bounded(&ctx, &stream, "the head's norm")?;
            if rank == 0 {
                let b = head
                    .locate("head.resid")
                    .ok_or_else(|| anyhow::anyhow!("`head.resid` is not a planned buffer"))?;
                let mut got = vec![0u8; usize::try_from(b.bytes)?];
                head_exec.download(hip, b.offset, &mut got, &stream)?;
                let last = graph.config.num_hidden_layers - 1;
                std::fs::write(
                    dir.join(format!("p{prompt_id}_l{last:+03}.f32")),
                    bf16_bytes_as_f32(&got),
                )?;
            }
            total += head_exec
                .run_steps(
                    &ctx,
                    &mut launcher,
                    &stream,
                    &head.steps[1..],
                    &inputs,
                    false,
                )?
                .launches;
            tp2::wait_bounded(&ctx, &stream, "the pass")?;
            total
        } else {
            pass(&mut body_exec, &mut head_exec, &mut launcher)?
        };
        peer.rendezvous("pass-done")?;
        let t_pass = t_start.elapsed();
        let expected = body.expected_launches() + head.expected_launches() + 2;
        // A budget mismatch is a REFUSAL, not a tally: in a 200-prompt loop a counter nobody
        // reads is a silent pass, and the launch count is the one invariant every prompt shares.
        anyhow::ensure!(
            launched == expected,
            "rank {rank}: LAUNCH BUDGET {launched} issued, {expected} declared"
        );
        if rank == 0 && verbose {
            println!(
            "launches  {launched} per rank = {} the body declares (every layer, the embedding and \
             {collectives} reduces, LESS {} norms the `_nq` epilogue folds into them) + {} the \
             head declares + 2 row selections this command adds",
            body.expected_launches(),
            body.fused_launches,
            head.expected_launches()
        );
            println!(
            "          the graph's own whole-model budget at this bucket is {}, which counts each \
             folded norm as its own op; the differences are those {} and the 2 row selections it \
             does not model",
            graph
                .launch_count_lower_bound(graph.bucket_for(n, Phase::Prefill))
                .total,
            body.fused_launches
        );
        }

        // ---- the token: gather the two vocabulary shards on the HOST -------------------------------
        //
        // `logits.gather` is `Route::Supplied` (ws-h16): the plan has a buffer for it and no kernel, so
        // the shards are swapped over the rendezvous socket and the argmax is over the concatenation.
        let shard = head
            .locate("logits.partial")
            .ok_or_else(|| anyhow::anyhow!("the head plans no `logits.partial`"))?;
        let mut mine = vec![0u8; usize::try_from(shard.bytes)?];
        head_exec.download(hip, shard.offset, &mut mine, &stream)?;
        let others = peer
            .exchange()
            .swap_bytes("logits.shard", &mine)
            .map_err(|e| anyhow::anyhow!("gathering the logits: {e}"))?;
        let (lo_shard, hi_shard) = if rank == 0 {
            (&mine, &others)
        } else {
            (&others, &mine)
        };
        let full: Vec<f32> = bf16_to_f32(lo_shard)
            .into_iter()
            .chain(bf16_to_f32(hi_shard))
            .collect();
        let (tok, best) =
            full.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
                    if *v > bv {
                        (i, *v)
                    } else {
                        (bi, bv)
                    }
                });
        peer.agree("token", tok as u64)?;
        // The token is the model's only if every op in the BODY has a code object. That used to be
        // a printed caveat ("the attention layers have no qk_norm_rope … so this is NOT the
        // model's token"), left behind when the glue landed; a caveat nobody re-reads is how a
        // gate compares 200 tokens that were never the model's. It is a refusal now, and it is
        // stated in terms of what the plan says rather than what a comment remembers.
        let body_supplied: u32 = body.supplied.values().sum();
        anyhow::ensure!(
            body_supplied == 0,
            "rank {rank}: {body_supplied} op(s) in the body have no code object ({:?}); this pass \
             does not compute the model's token and nothing downstream may compare it",
            body.supplied.keys().collect::<Vec<_>>()
        );
        // The head's one Supplied op is the vocabulary all-GATHER, which this command performs on
        // the host 40 lines up — the head plugin's `row_gather` is a different op and no code
        // object here does the gather.
        let head_supplied: Vec<&str> = head.supplied.keys().copied().collect();
        anyhow::ensure!(
            head_supplied == vec!["logits_gather"],
            "rank {rank}: the head supplies {head_supplied:?}; this command performs exactly \
             `logits.gather` on the host and nothing else"
        );
        if rank == 0 && verbose {
            println!(
            "token     argmax = {tok} over {} logits gathered from both ranks (max {best:.4}); \
             both ranks agree",
            full.len()
        );
            println!(
            "          COMPLETE: every op in the body has a code object (the glue trio included); \
             the head's only Supplied op is `logits.gather`, the vocabulary all-gather this \
             command does on the host."
        );
        }

        // Timing is a single-prompt statement: a 200-prompt gate times the gate, not a step.
        if verbose {
            // ---- timing --------------------------------------------------------------------------------
            if args.timing {
                for _ in 0..args.warmup {
                    peer.rendezvous("warm")?;
                    pass(&mut body_exec, &mut head_exec, &mut launcher)?;
                }
                let start = hip.create_event(EventFlags::Default)?;
                let end = hip.create_event(EventFlags::Default)?;
                let mut us: Vec<f64> = Vec::with_capacity(args.iters);
                for i in 0..args.iters {
                    peer.rendezvous(&format!("t{i}"))?;
                    hip.record_event(&start, &stream)?;
                    pass(&mut body_exec, &mut head_exec, &mut launcher)?;
                    hip.record_event(&end, &stream)?;
                    tp2::wait_bounded(&ctx, &stream, "a timed pass")?;
                    us.push(f64::from(hip.elapsed_ms(&start, &end)?) * 1000.0);
                }
                us.sort_by(f64::total_cmp);
                let mine_us = us[us.len() / 2];
                let theirs = f64::from_bits(
                    peer.exchange()
                        .swap_digest("fwd-us", mine_us.to_bits())
                        .map_err(|e| anyhow::anyhow!("exchanging timings: {e}"))?,
                );
                if rank == 0 {
                    println!();
                    println!(
                    "time      {mine_us:.0} µs/forward at {n} tokens on rank 0, {theirs:.0} on rank 1 \
                     ({} replays, GPU idle)",
                    args.iters
                );
                }
            }
        }
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        let cost = PassCost {
            setup: ms(t_setup),
            ipc: ms(t_ipc - t_setup),
            pass: ms(t_pass - t_ipc),
            total: ms(t_start.elapsed()),
            folded: body.fused_nq,
            lossy_reduces: lossy,
        };
        Ok((i32::try_from(tok).unwrap_or(-1), launched, cost))
    };

    // ---- the loop, and the comparison ----------------------------------------------------------
    let mut outcome: Vec<(String, ArmTokens, u32)> = Vec::new();
    // One arm by default (whatever the plan's own default fusion is), both when the gate asks.
    let arms: Vec<bool> = if args.both_arms {
        vec![false, true]
    } else {
        vec![saber_exec::layer::PlanOptions::BODY.fuse]
    };
    for fuse in arms {
        let mut got: ArmTokens = Vec::with_capacity(prompts.len());
        let mut costs: Vec<PassCost> = Vec::with_capacity(prompts.len());
        let arm_start = Instant::now();
        for (i, (id, ids)) in prompts.iter().enumerate() {
            peer.rendezvous(&format!("prompt-{}-{i}", u8::from(fuse)))?;
            let (tok, _, cost) = one_pass(*id, ids, fuse, single)?;
            got.push((*id, ids.len(), tok, None));
            costs.push(cost);
            if !single && rank == 0 && (i + 1) % 25 == 0 {
                println!(
                    "   {}/{} prompts, {:.1} s elapsed ({})",
                    i + 1,
                    prompts.len(),
                    arm_start.elapsed().as_secs_f64(),
                    arm_name(fuse)
                );
            }
        }
        // What a prompt costs, and where. Everything but `pass` is per-prompt work a serving
        // process does once: the plans and the arenas are sized by the prompt's length, and the
        // peer-visible regions are keyed on the descriptor's shape, so both are re-done here.
        let folded: u32 = costs.iter().map(|c| c.folded).sum();
        let lossy: u32 = costs.iter().map(|c| c.lossy_reduces).sum();
        if rank == 0 && !single {
            let med = |f: fn(&PassCost) -> f64| {
                let mut v: Vec<f64> = costs.iter().map(f).collect();
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };
            println!(
                "per-prompt {:.0} ms median = setup {:.0} + ipc {:.0} + pass {:.0}; {:.1} s for \
                 {} prompt(s)",
                med(|c| c.total),
                med(|c| c.setup),
                med(|c| c.ipc),
                med(|c| c.pass),
                arm_start.elapsed().as_secs_f64(),
                prompts.len()
            );
        }
        if rank == 0 {
            println!(
                "arm       {} — {folded} norm(s) folded and {lossy} rotated reduce(s) over {} \
                 prompt(s)",
                arm_name(fuse),
                prompts.len()
            );
        }
        outcome.push((arm_name(fuse).to_string(), got, folded));
    }
    let mut failures = 0usize;
    // ONE rank reports. Both ranks agree the token on every pass (`peer.agree("token", ...)`), so
    // rank 1's comparison would be the same table printed a second time into the same inherited
    // stdout — and an agreement table printed twice is one a reader has to check for agreement.
    if let Some(reference) = &args.compare {
        if rank == 0 {
            for (arm, got, folded) in &outcome {
                println!();
                println!("arm       {arm} ({folded} norm(s) folded)");
                let bad = crate::forward::compare(reference, got)?;
                if bad > 0 {
                    println!("          {bad} prompt(s) disagreed with the reference in this arm");
                }
                failures += usize::from(args.require_complete && bad > 0);
            }
        }
    }

    // THE ARMS WERE THE SAME PLAN, said where the answer would otherwise be read off two
    // identical tables. `Planner::nq_fusion` takes the pair only when the rules resolve the reduce
    // to `allreduce_exact`, which at hidden 5120 in bf16 is 12 rows or fewer; every prompt in a
    // corpus is longer, so the fusion is not reachable from here at all. `docs/EXEC.md` sec.7.2: a
    // control must differ from the case in the way the case differs from the truth.
    if rank == 0 && args.both_arms && outcome.iter().all(|(_, _, folded)| *folded == 0) {
        println!();
        println!(
            "NOTE      neither arm folded a single norm, so the two tables above are ONE plan run \
             twice and they say nothing about the `_nq` fusion. The fusion needs the EXACT \
             all-reduce family, which the rules pick below 128 KiB = 12 rows at hidden 5120 bf16; \
             every prompt here is 35 tokens or more. The fusion's evidence is `saber ar-gate` \
             (`ar_nq_pair`) and `saber step-tp2` at a DECODE bucket, where the message is small \
             enough for it to exist."
        );
    }

    Ok(failures)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Every non-default flag reaches the child.
    ///
    /// The four gate flags -- `--prompts`, `--max-prompt`, `--compare`, `--both-arms` -- are the
    /// ones ws-h23 found the parent parsing and dropping, so the gate could not have reported a
    /// disagreement even with every index law in place. The exhaustive destructure in
    /// `child_args` makes a forgotten field a compile error; this makes a misspelled or mis-valued
    /// one a test failure, by parsing the argv back through the real clap definition.
    #[test]
    fn every_non_default_flag_round_trips_to_the_child() {
        let args = ForwardTp2Args {
            pack: PathBuf::from("/packs/tier"),
            prompt_ids: PathBuf::from("/tmp/corpus.jsonl"),
            prompt_index: 3,
            tokens: 234,
            prompts: 200,
            max_prompt: 4096,
            compare: Some(PathBuf::from("/ref/nospec.jsonl")),
            both_arms: true,
            ar_exact: true,
            dump_layers: Some(PathBuf::from("/tmp/layers")),
            only_ids: vec![54, 86],
            require_complete: true,
            prebuilt: PathBuf::from("/pre"),
            budget_mib: 26000,
            timing: true,
            iters: 13,
            warmup: 4,
            hip_lib: "/opt/rocm/lib/libamdhip64.so.7".to_owned(),
            rank: None,
            port: None,
        };

        let argv = child_args(&args, 1, 5555);
        // A bigger stack: `Cmd` is a large enum and clap builds it on the stack, which overflows
        // a test thread's default 2 MiB in a debug build.
        let cli = std::thread::Builder::new()
            .stack_size(16 << 20)
            .spawn(move || {
                crate::Cli::try_parse_from(std::iter::once(OsString::from("saber")).chain(argv))
            })
            .expect("spawning the parse thread")
            .join()
            .expect("the parse thread")
            .expect("the child argv parses with the real clap definition");

        let crate::Cmd::Forward {
            pack,
            prompt_ids,
            prompt_index,
            prompts,
            max_prompt,
            both_arms,
            ar_exact,
            dump_layers,
            only_ids,
            compare,
            tokens,
            prebuilt,
            budget_mib,
            timing,
            iters,
            warmup,
            require_complete,
            tp,
            tp_rank,
            port,
            hip_lib,
            rank: _,
            device: _,
            // `--top2` is TP=1 only and `child_args` never sends it: at two ranks a rank's
            // `logits.partial` is half the vocabulary. Named rather than `..` so adding a flag
            // to `Cmd::Forward` stays a compile error here.
            top2,
        } = cli.cmd
        else {
            panic!("the child argv named a different subcommand");
        };

        assert!(!top2, "--top2 is TP=1 only and must not reach a rank");
        assert_eq!(tp, 2, "the child must be told it is a TP=2 rank");
        assert_eq!(tp_rank, Some(1));
        assert_eq!(port, Some(5555));
        assert_eq!(pack, args.pack);
        assert_eq!(prompt_ids, args.prompt_ids);
        assert_eq!(prompt_index, args.prompt_index);
        assert_eq!(tokens, args.tokens);
        assert_eq!(prebuilt, args.prebuilt);
        assert_eq!(budget_mib, args.budget_mib);
        assert_eq!(hip_lib, args.hip_lib);
        // The gate's four, and the two arms.
        assert_eq!(prompts, args.prompts);
        assert_eq!(max_prompt, args.max_prompt);
        assert_eq!(compare, args.compare);
        assert_eq!(both_arms, args.both_arms);
        assert_eq!(ar_exact, args.ar_exact);
        assert_eq!(dump_layers, args.dump_layers);
        assert_eq!(only_ids, args.only_ids);
        assert_eq!(require_complete, args.require_complete);
        assert_eq!(timing, args.timing);
        assert_eq!(iters, args.iters);
        assert_eq!(warmup, args.warmup);
    }

    /// The negative control: defaults are not forwarded, and `--iters` rides on `--timing`.
    #[test]
    fn a_default_flag_is_not_forwarded() {
        let args = ForwardTp2Args {
            pack: PathBuf::from("/p"),
            prompt_ids: PathBuf::from("/i"),
            prompt_index: 0,
            tokens: 0,
            prompts: 1,
            max_prompt: 0,
            compare: None,
            both_arms: false,
            ar_exact: false,
            dump_layers: None,
            only_ids: Vec::new(),
            require_complete: false,
            prebuilt: PathBuf::from("/b"),
            budget_mib: 1,
            timing: false,
            iters: 7,
            warmup: 2,
            hip_lib: "x".to_owned(),
            rank: None,
            port: None,
        };
        let argv: Vec<String> = child_args(&args, 0, 1)
            .iter()
            .map(|o| o.to_string_lossy().into_owned())
            .collect();
        for absent in [
            "--compare",
            "--both-arms",
            "--ar-exact",
            "--dump-layers",
            "--only-ids",
            "--require-complete",
            "--timing",
            // `--iters` is only meaningful under `--timing`, and forwarding it alone would be a
            // flag the child parses and ignores -- the same shape of bug one level down.
            "--iters",
        ] {
            assert!(
                !argv.iter().any(|x| x == absent),
                "{absent} was forwarded although it is at its default: {argv:?}"
            );
        }
        assert!(argv.iter().any(|x| x == "--rank-of"));
    }

    /// The body plan this command runs, for one rank and one fusion arm. Pure: a plan is a
    /// function of `(graph, bucket, layers, params, rules, options)` and touches no device.
    fn body_plan<'g>(
        graph: &'g Graph,
        mxfp4: &LaunchRules,
        saber: &SaberRules,
        n: u32,
        rank: u32,
        fuse: bool,
        layers: &[u32],
    ) -> BlockPlan<'g> {
        let o = PlanOptions::BODY
            .rank(rank)
            .with_collectives(Collectives::Real);
        // BOTH arms explicit. `PlanOptions::BODY.fuse` is FALSE by default since ws-h19, so
        // "the fused arm" written as "the default" silently became the unfused arm and this
        // helper returned two identical plans -- which is the very failure the test below exists
        // to detect, arriving through the test's own scaffolding.
        let o = if fuse {
            o.with_fusion()
        } else {
            o.without_fusion()
        };
        BlockPlan::with_options(
            graph,
            graph.bucket_for(n, Phase::Prefill),
            layers,
            RuntimeParams {
                max_ctx: n,
                gdn_slots: saber_exec::forward::FIRST_STATE_SLOT + 1,
            },
            mxfp4,
            saber,
            o,
        )
        .expect("the TP=2 body plans")
    }

    /// **The negative control this command did not have.** Every index key the TP=2 body plan
    /// declares must have a law, on both ranks and in both fusion arms, at a length in each
    /// all-reduce band.
    ///
    /// Without it the first thing that answered "does `layer.11.attn.kv_append.k_scale` have a
    /// law here" was a two-card run under the GPU lock, minutes in and after both ranks had
    /// uploaded 7.7 GiB each. The laws live in `saber_exec::forward` and the TP=1 pass had been
    /// moved onto them; this file kept a six-key copy from before the attention glue was bound,
    /// and nothing compared the two.
    #[test]
    fn every_index_key_the_tp2_body_declares_has_a_law() {
        let graph = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("the TP=2 graph");
        let mxfp4 = LaunchRules::prod_mxfp4().expect("rules");
        let saber = SaberRules::prod_saber().expect("saber rules");
        let layers: Vec<u32> = (0..graph.config.num_hidden_layers).collect();
        let mut glue_seen = false;
        // 8 tokens is under the rules' 128 KiB message threshold and 234 is over it, so both
        // all-reduce families are planned; a family does not change the index keys, and a test
        // that assumes so is one nobody re-runs when it does.
        for n in [8u32, 234] {
            let ids: Vec<i32> = (0..n as i32).collect();
            let kv_blocks = n.div_ceil(16).max(1);
            for rank in [0u32, 1] {
                for fuse in [false, true] {
                    let body = body_plan(&graph, &mxfp4, &saber, n, rank, fuse, &layers);
                    let keys = body.index_keys();
                    assert!(
                        !keys.is_empty(),
                        "the body declares no index keys at all; this test would pass vacuously"
                    );
                    for key in &keys {
                        let table = index_table(key, n, kv_blocks, &ids).unwrap_or_else(|e| {
                            panic!("index `{key}` at n={n} rank={rank} fuse={fuse}: {e}")
                        });
                        assert!(
                            !table.is_empty(),
                            "index `{key}` built an EMPTY buffer, which a kernel reads as an offset"
                        );
                        glue_seen |= key.ends_with(".k_scale");
                    }
                }
            }
        }
        // The keys that were missing. Asserting they are DECLARED is what keeps the loop above
        // from passing because the plan stopped asking for them.
        assert!(
            glue_seen,
            "no `.k_scale` key in the TP=2 body plan: the attention glue is not bound, and this \
             test no longer covers the refusal it was written for"
        );
    }

    /// Everything a rank needs reaches the rank.
    ///
    /// The parent opens no HIP context and runs no pass, so a flag it parses and does not forward
    /// makes both ranks run their DEFAULTS and exit 0 — a gate that prints a clean table for a run
    /// that never happened. `--prompts 200 --compare … --both-arms` did exactly that until ws-h23.
    /// This walks the args struct's own values and asserts each appears in the child argv, so a
    /// NEW field is caught by the same test: add one to `ForwardTp2Args`, forget `child_args`, and
    /// the list below stops covering it — which is why the count is asserted too.
    #[test]
    fn the_ranks_are_spawned_with_every_flag_the_gate_sets() {
        let args = ForwardTp2Args {
            prompts: 200,
            max_prompt: 4096,
            compare: Some(PathBuf::from("/work/.cache/forward/nospec_ref.jsonl")),
            both_arms: true,
            ar_exact: true,
            dump_layers: Some(PathBuf::from("/work/.cache/layers-saber")),
            only_ids: vec![54, 86, 99, 11, 15],
            require_complete: true,
            pack: PathBuf::from("/packs/tier"),
            prompt_ids: PathBuf::from("/work/.cache/template_corpus.jsonl"),
            prompt_index: 3,
            tokens: 7,
            prebuilt: PathBuf::from("/work/prebuilt/gfx1201"),
            budget_mib: 24576,
            timing: true,
            iters: 11,
            warmup: 2,
            rank: None,
            port: None,
            hip_lib: "/opt/rocm/lib/libamdhip64.so.7".to_owned(),
        };
        let argv: Vec<String> = child_args(&args, 1, 41234)
            .into_iter()
            .map(|s| s.to_string_lossy().into_owned())
            .collect();
        let has = |flag: &str, value: &str| {
            let at = argv.iter().position(|a| a == flag);
            assert!(at.is_some(), "the child argv has no `{flag}`: {argv:?}");
            let at = at.expect("checked");
            assert_eq!(
                argv.get(at + 1).map(String::as_str),
                Some(value),
                "`{flag}` carries the wrong value"
            );
        };
        let flag = |flag: &str| {
            assert!(
                argv.iter().any(|a| a == flag),
                "the child argv has no `{flag}`"
            );
        };
        // The identity of the child, and the trap that made it spawn grandchildren.
        has("--rank-of", "1");
        has("--port", "41234");
        assert!(
            !argv.iter().any(|a| a == "--rank"),
            "`--rank` is the PACK's rank and a child given it without `--rank-of` decides it is \
             the parent and spawns two more"
        );
        // Every value the struct carries.
        has("--prompts", "200");
        has("--max-prompt", "4096");
        has("--compare", "/work/.cache/forward/nospec_ref.jsonl");
        has("--pack", "/packs/tier");
        has("--prompt-ids", "/work/.cache/template_corpus.jsonl");
        has("--prompt-index", "3");
        has("--tokens", "7");
        has("--prebuilt", "/work/prebuilt/gfx1201");
        has("--budget-mib", "24576");
        has("--hip-lib", "/opt/rocm/lib/libamdhip64.so.7");
        has("--dump-layers", "/work/.cache/layers-saber");
        has("--only-ids", "54,86,99,11,15");
        has("--iters", "11");
        has("--warmup", "2");
        flag("--both-arms");
        flag("--ar-exact");
        flag("--require-complete");
        flag("--timing");

        // And the switches are ABSENT when they are off, not passed as `false`.
        let off = ForwardTp2Args {
            both_arms: false,
            ar_exact: false,
            require_complete: false,
            timing: false,
            compare: None,
            dump_layers: None,
            only_ids: vec![],
            ..args
        };
        let argv: Vec<String> = child_args(&off, 0, 1)
            .into_iter()
            .map(|s| s.to_string_lossy().into_owned())
            .collect();
        for absent in [
            "--both-arms",
            "--ar-exact",
            "--require-complete",
            "--timing",
            "--compare",
            "--dump-layers",
            "--only-ids",
        ] {
            assert!(
                !argv.iter().any(|a| a == absent),
                "`{absent}` is set on a run that did not ask for it"
            );
        }
    }

    /// The dump points are one per layer, they name the edge that layer's stream lands in, and
    /// the edge is READABLE where the dump reads it.
    ///
    /// Three things can silently break the per-layer comparison and none of them fails a launch
    /// count: a renamed op (no dump point, and the run refuses — that is the easy one), an edge
    /// that is not a planned buffer (the dump would report a missing tensor), and reading the
    /// stream at the wrong STEP. The last is the one this pins: `l{i}.resid.a` is written by
    /// `layer.{i}.input_layernorm`, so the point for layer `i-1` must be that op's own index, and
    /// with slot reuse every one of these edges is the same arena buffer — read it one step late
    /// and it holds the next layer's value with nothing to say so.
    #[test]
    fn the_layer_dump_points_are_one_per_layer_and_name_a_readable_edge() {
        let graph = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("the TP=2 graph");
        let mxfp4 = LaunchRules::prod_mxfp4().expect("rules");
        let saber = SaberRules::prod_saber().expect("saber rules");
        let layers: Vec<u32> = (0..graph.config.num_hidden_layers).collect();
        let n = 234u32;
        let body = body_plan(&graph, &mxfp4, &saber, n, 0, false, &layers);
        let points = layer_dump_points(&body);

        assert_eq!(
            points.len(),
            graph.config.num_hidden_layers as usize,
            "one dump point per layer"
        );
        // Layer -1 is the reduced embedding; then 0..=62, each named `l{i+1}.resid.a`. Layer 63's
        // stream is the HEAD's `head.resid` and is not a body point.
        assert_eq!(points[0].2, -1);
        assert_eq!(points[0].1, "hidden.0");
        for (k, (idx, edge, layer)) in points.iter().enumerate().skip(1) {
            assert_eq!(*layer, k as i32 - 1, "the points are in layer order");
            assert_eq!(
                edge,
                &format!("l{}.resid.a", k),
                "the edge names its own layer"
            );
            // The point IS the op that writes the edge, not the one after it.
            assert_eq!(
                body.steps[*idx].op.name,
                format!("layer.{k}.input_layernorm"),
                "dump point {k} is not the op that writes {edge}"
            );
            assert!(
                body.locate(edge).is_some(),
                "{edge} is not a planned buffer, so the dump would read nothing"
            );
        }
        // And the last layer's stream, which the head carries.
        let head = BlockPlan::with_options(
            &graph,
            graph.head_bucket(graph.bucket_for(n, Phase::Prefill)),
            &[],
            RuntimeParams {
                max_ctx: n,
                gdn_slots: saber_exec::forward::FIRST_STATE_SLOT + 1,
            },
            &mxfp4,
            &saber,
            PlanOptions::HEAD.rank(0),
        )
        .expect("the head plans");
        assert!(
            head.locate("head.resid").is_some(),
            "`head.resid` is where layer 63's stream is readable"
        );
    }

    /// `--ar-exact` moves every reduce and nothing else, and it is the ONLY thing that makes
    /// `--both-arms` an experiment at a prompt's row count.
    ///
    /// Raising `quant_min_bytes` is one number in one table, so this asserts the two consequences
    /// separately: the 129 reduces change family, and `Planner::nq_fusion` — which takes the pair
    /// only when the reduce resolves to `allreduce_exact` — becomes reachable at 234 rows, where
    /// on prod's band it is not. Without the second half a `--both-arms --ar-exact` run would
    /// still be one plan run twice and the report would say the opposite.
    #[test]
    fn ar_exact_moves_every_reduce_and_unlocks_the_fusion() {
        let graph = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("the TP=2 graph");
        let mxfp4 = LaunchRules::prod_mxfp4().expect("rules");
        let layers: Vec<u32> = (0..graph.config.num_hidden_layers).collect();
        let n = 234u32;

        let prod = SaberRules::prod_saber().expect("saber rules");
        let lossy = body_plan(&graph, &mxfp4, &prod, n, 0, false, &layers);
        let f = ar_families(&lossy);
        assert_eq!(
            f.get("allreduce_wht").copied(),
            Some(129),
            "prod's band should rotate all 129 reduces at {n} rows: {f:?}"
        );
        assert_eq!(lossy.fused_nq, 0);

        let mut exact = SaberRules::prod_saber().expect("saber rules");
        let was = force_exact_all_reduce(&mut exact, n, graph.config.hidden_size)
            .expect("the band moves");
        assert_eq!(was, 131072, "the threshold this run raised");
        assert_eq!(exact.ar.quant_min_bytes, u64::from(n) * 5120 * 2 + 1);

        let off = body_plan(&graph, &mxfp4, &exact, n, 0, false, &layers);
        let g = ar_families(&off);
        assert_eq!(
            g.get("allreduce_wht"),
            None,
            "a rotated reduce survived: {g:?}"
        );
        assert_eq!(g.get("allreduce_exact").copied(), Some(129), "{g:?}");
        assert_eq!(
            off.fused_nq, 0,
            "the unfused arm folds nothing by construction"
        );

        let on = body_plan(&graph, &mxfp4, &exact, n, 0, true, &layers);
        let h = ar_families(&on);
        assert_eq!(h.get("allreduce_wht"), None, "{h:?}");
        assert!(
            on.fused_nq > 0,
            "the fusion is still unreachable with the exact band, so `--both-arms --ar-exact` \
             would be one plan run twice: {h:?}"
        );
        // The arms now DIFFER, which is what the ws-h23 run could not say.
        assert!(
            on.expected_launches() < off.expected_launches(),
            "the fused arm issues {} launches and the unfused {}",
            on.expected_launches(),
            off.expected_launches()
        );
    }

    /// A message above the slot is BANDED now, so `--ar-exact` raises the threshold to the
    /// largest band rather than refusing — and a row wider than the slot is still the refusal.
    #[test]
    fn ar_exact_raises_the_band_to_the_largest_band_and_refuses_an_uncuttable_row() {
        let mut saber = SaberRules::prod_saber().expect("saber rules");
        let one_band = u32::try_from(saber.ar.max_bytes / (5120 * 2)).expect("fits");
        // One row past a whole slot: two bands, each about half the message.
        let was = force_exact_all_reduce(&mut saber, one_band + 1, 5120).expect("banded");
        assert_eq!(was, 131072);
        let largest = u64::from(one_band + 1).div_ceil(2) * 5120 * 2;
        assert_eq!(saber.ar.quant_min_bytes, largest + 1);
        assert!(
            largest <= saber.ar.max_bytes,
            "{largest} B is a band, so it fits the slot"
        );

        // The control: a row that does not fit the slot at all cannot be cut and is refused.
        let mut wide = SaberRules::prod_saber().expect("saber rules");
        let hidden = u32::try_from(wide.ar.max_bytes / 2).expect("fits") + 16;
        let e = force_exact_all_reduce(&mut wide, 1, hidden).expect_err("one row over the slot");
        assert!(format!("{e}").contains("falls back to RCCL"), "{e}");
        assert_eq!(
            wide.ar.quant_min_bytes, 131072,
            "a refused rewrite must leave the band alone"
        );
    }

    /// **The `_nq` fusion cannot fire at a prompt length this command runs**, and a `--both-arms`
    /// comparison over a prompt corpus is therefore two runs of the SAME plan.
    ///
    /// `Planner::nq_fusion` takes the pair only when the rules resolve the reduce to
    /// `allreduce_exact`, which is the band below `quant_min_bytes` = 128 KiB. At hidden 5120 in
    /// bf16 a row is 10,240 bytes, so the exact family runs at **12 rows or fewer** and every
    /// longer step takes the rotated one, which the epilogue is not written for. The corpus this
    /// gate runs is 35 to 332 tokens.
    ///
    /// This is the control rule of `docs/EXEC.md` sec.7.2 — *a control must differ from the case in
    /// the way the case differs from the truth* — stated as a test, because the arms LOOK
    /// different (`PlanOptions::without_fusion` is real, `arm_name` prints two different
    /// sentences) and the plans are identical.
    #[test]
    fn the_nq_fusion_is_a_small_message_fusion_and_no_prompt_length_reaches_it() {
        let graph = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("the TP=2 graph");
        let mxfp4 = LaunchRules::prod_mxfp4().expect("rules");
        let saber = SaberRules::prod_saber().expect("saber rules");
        let layers: Vec<u32> = (0..graph.config.num_hidden_layers).collect();
        // 12 rows is 122.88 KiB and 13 is 133.12 KiB, either side of the rules' 128 KiB band.
        let folded = |n: u32| body_plan(&graph, &mxfp4, &saber, n, 0, true, &layers).fused_nq;
        assert!(
            folded(12) > 0,
            "the fusion does not fire even in the exact band; this test no longer describes it"
        );
        assert_eq!(folded(13), 0, "the fusion fired above the exact band");
        // Every length the token gate runs, and the ends of the corpus.
        for n in [35u32, 60, 132, 234, 332] {
            assert_eq!(
                folded(n),
                0,
                "the fusion folded a norm at {n} tokens: the `--both-arms` comparison is no longer \
                 two runs of one plan and this test's premise has changed"
            );
        }
        // And what keeps a plan from fusing by DEFAULT is not `fuse`, which is `true` in every
        // `PlanOptions` constant, but `collectives: Identity`, which `BODY` carries until a caller
        // says it has a peer. `docs/EXEC.md` sec.3 says the default is OFF; the code says the
        // fusion is unreachable without a peer, which is the same behaviour for one rank and a
        // different statement for two.
        let no_peer = BlockPlan::with_options(
            &graph,
            graph.bucket_for(12, Phase::Prefill),
            &layers,
            RuntimeParams {
                max_ctx: 12,
                gdn_slots: saber_exec::forward::FIRST_STATE_SLOT + 1,
            },
            &mxfp4,
            &saber,
            PlanOptions::BODY.with_fusion(),
        )
        .expect("a peerless body plans");
        // Opted in BY NAME. `fuse` defaults to FALSE since ws-h19 found the `_nq` fusion is not
        // bit-identical once attention is real (it had been gated on zeros), so this test -- whose
        // subject is what the fusion does with no peer to reduce with -- has to ask for it. A test
        // of a route that reads the default is a test of whatever the default happens to be.
        const {
            assert!(
                !PlanOptions::BODY.fuse,
                "`fuse` is off by default; this test opts in with `with_fusion()`"
            )
        };
        assert_eq!(
            no_peer.fused_nq, 0,
            "a plan with no peer fused a norm; there is no reduce to fuse into"
        );
        // And the two arms ARE the same plan there, which is the thing a reader of the report
        // must not have to infer.
        let on = body_plan(&graph, &mxfp4, &saber, 234, 0, true, &layers);
        let off = body_plan(&graph, &mxfp4, &saber, 234, 0, false, &layers);
        assert_eq!(on.expected_launches(), off.expected_launches());
        assert_eq!(on.arena_bytes, off.arena_bytes);
        assert_eq!(on.steps.len(), off.steps.len());
    }

    /// The two refusals the pass makes before it lets a token be compared, stated on the host so
    /// they are not first evaluated two cards and 7.7 GiB into a run. The body must supply
    /// NOTHING — a decoder block with an unbound op computes a value that is not the model's — and
    /// the head must supply exactly the vocabulary all-gather, which this command performs itself.
    #[test]
    fn the_body_supplies_nothing_and_the_head_supplies_only_the_gather() {
        let graph = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("the TP=2 graph");
        let mxfp4 = LaunchRules::prod_mxfp4().expect("rules");
        let saber = SaberRules::prod_saber().expect("saber rules");
        let layers: Vec<u32> = (0..graph.config.num_hidden_layers).collect();
        let n = 234u32;
        for rank in [0u32, 1] {
            for fuse in [false, true] {
                let body = body_plan(&graph, &mxfp4, &saber, n, rank, fuse, &layers);
                assert!(
                    body.supplied.is_empty(),
                    "the TP=2 body supplies {:?} at rank {rank} fuse {fuse}",
                    body.supplied
                );
            }
        }
        let head = BlockPlan::with_options(
            &graph,
            graph.head_bucket(graph.bucket_for(n, Phase::Prefill)),
            &[],
            RuntimeParams {
                max_ctx: n,
                gdn_slots: saber_exec::forward::FIRST_STATE_SLOT + 1,
            },
            &mxfp4,
            &saber,
            PlanOptions::HEAD.rank(0),
        )
        .expect("the head plans");
        // The spelling is the op ID, not the graph's dotted edge name. The pass asserted
        // `logits.gather` on its first run and refused its own head after 7.7 GiB had been
        // uploaded on both ranks.
        assert_eq!(
            head.supplied.keys().copied().collect::<Vec<&str>>(),
            vec!["logits_gather"]
        );
    }

    /// One law, one place, stated as a comparison rather than as a convention: the TP=1 and TP=2
    /// bodies declare the SAME index keys, because no index table is a function of the sharding.
    /// So the TP=1 pass exercising a key is the TP=2 pass exercising it, and a key that appears on
    /// one side only is a divergence to be looked at before a gate runs on two cards.
    #[test]
    fn the_two_tensor_parallelisms_declare_the_same_index_keys() {
        let mxfp4 = LaunchRules::prod_mxfp4().expect("rules");
        let saber = SaberRules::prod_saber().expect("saber rules");
        let n = 234u32;
        let g1 = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 1).expect("the TP=1 graph");
        let layers: Vec<u32> = (0..g1.config.num_hidden_layers).collect();
        let one = BlockPlan::with_options(
            &g1,
            g1.bucket_for(n, Phase::Prefill),
            &layers,
            RuntimeParams {
                max_ctx: n,
                gdn_slots: saber_exec::forward::FIRST_STATE_SLOT + 1,
            },
            &mxfp4,
            &saber,
            PlanOptions::BODY,
        )
        .expect("the TP=1 body plans");
        let g2 = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("the TP=2 graph");
        let two = body_plan(&g2, &mxfp4, &saber, n, 0, false, &layers);
        let (a, b) = (one.index_keys(), two.index_keys());
        let only_1: Vec<&String> = a.difference(&b).collect();
        let only_2: Vec<&String> = b.difference(&a).collect();
        assert!(
            only_1.is_empty() && only_2.is_empty(),
            "TP=1 declares {only_1:?} that TP=2 does not, and TP=2 declares {only_2:?} that TP=1 \
             does not"
        );
    }
}
