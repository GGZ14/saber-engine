//! `saber draft-step`: the DFlash2 drafter as a PLAN, resolved and — where prod's own state has
//! been captured — run and compared against prod's answer.
//!
//! Three things, in the order they can be checked:
//!
//! 1. **The plan.** [`saber_model::dflash_graph::DraftGraph`] declares every launch a decode step
//!    makes, the shape it runs at, and which are code objects. The budget is DECLARED there, not
//!    counted from a trace.
//! 2. **The routes.** Every covered op goes through `saber_kernels::dflash::resolve`, so the
//!    description and the committed rules have to agree: a shape the rules refuse is a description
//!    of a program that cannot run. The launch count this produces is compared against the budget,
//!    which is structural rule 7 applied to the drafter.
//! 3. **The selector, bit-exact against a LIVE serve.** With `--sel-data` the two kernels that
//!    produce the drafter's candidate scores run through the real [`saber_exec::Launcher`] on the
//!    inputs a running prod handed its own selector, and their output is byte-compared against what
//!    prod produced. This is an end-to-end comparison in the only sense available today: every
//!    rounding in those two launches is upstream of it.
//!
//! 4. **The proposal's context-KV store, bit-exact through the launcher.** With `--proposal` the
//!    three covered launches of `precompute_and_store_context_kv` run from resolved descriptors,
//!    in prod's own order and with the rope IN PLACE on the norm's output, and every output is
//!    byte-compared. It is the only part of a proposal reachable without the drafter's weights,
//!    because it runs before any of them is touched: its input is the TARGET's hidden states.
//!
//! WHAT IT CANNOT DO YET, and the reason has stopped being kernels. The plan declares **101
//! launches and 96 are covered**; the five that are not are the context store's fused KV
//! projection (`bgemm_bf16`'s own kernel, at an N where hipBLASLt changes algorithm and stops
//! agreeing with it — see `[bgemm] max_n`), one ATen copy (`permute` + `.contiguous()`, which
//! computes nothing and which a Saber store would fuse away), and the three ATen selections,
//! refused with a reason rather than a TODO: ATen's index order among ties is not a rule — it
//! depends on which of its algorithms the shape picks — so a kernel that matched it at one shape
//! would not at another, and ATen is faster here anyway. `--proposal` can be handed those three
//! from the capture, since they are exact index selections and every rounding is upstream of them.
//!
//! 5. **The drafter's PACK, against the plan that declares it.** With `--pack` every record
//!    [`saber_model::dflash_graph::DraftGraph::records`] declares is checked against the pack's own
//!    table — name, layout id, dtype, shape — and, with a card, the rank is uploaded and every
//!    launch whose operands are a pack weight and an activation is resolved, bound BY RECORD NAME
//!    and dispatched. That is the round's deliverable: the 96-record map, the preshuffled fp8 B,
//!    the packed int2 head and the rope table, all of them gated byte-for-byte against prod's own
//!    packer by `the internal reference harness` before they ever reach a card.
//!
//! WHAT IS STILL NOT A WHOLE PROPOSAL. The pack removes the blocker; it does not by itself make a
//! k = 7 draft block comparable. Two things are left and both are captures rather than code: the
//! five absent ops have to be handed in from prod's own step (they are exact selections and a
//! `.contiguous()`), and there is no capture of prod's PROPOSAL — its input hidden states, its
//! drafter KV cache and the draft block it produced — to compare against. Until there is, the
//! launches below run on zeroed activations and the command says so rather than printing a number
//! that looks like a comparison. `--explain` lists the uncovered launches with what each needs.

use std::collections::BTreeMap;
use std::path::PathBuf;

use saber_exec::{ArgBindings, LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet};
use saber_hip::{DeviceAddr, DevicePtr, HipThread, StreamFlags};
use saber_kernels::dflash::{DflashOp, DflashRules, DflashShape};
use saber_kernels::rules::{ArgValue, LaunchDescriptor, Shape};
use saber_mem::dio::OpenMode;
use saber_mem::ledger::{Ledger, Pool};
use saber_mem::pack::PackReader;
use saber_mem::upload::{HipSink, PinnedRing, Upload, UploadConfig};
use saber_model::dflash_graph::{Coverage, DflashConfig, DraftGraph, DraftRope};

use crate::npy;

/// Arguments for [`run`].
pub struct DraftStepArgs {
    /// Ranks the shapes are sharded for.
    pub tp: u32,
    /// `num_speculative_tokens`; the block is one more than this.
    pub spec: u32,
    /// Directory of committed `.hsaco` + `*.manifest.toml`.
    pub prebuilt: PathBuf,
    /// A live-serve selector capture (`the internal reference harness`). Without it, the
    /// command is the plan and the routes only.
    pub sel_data: Option<PathBuf>,
    /// A context-store capture (`the internal reference harness`).
    pub proposal: Option<PathBuf>,
    /// Print the uncovered launches with what each needs.
    pub explain: bool,
    /// Refuse to start if the device buffers would exceed this.
    pub budget_mib: u64,
    /// A drafter pack directory (`the internal reference harness --layout dflash`).
    pub pack: Option<PathBuf>,
    /// Which rank of that pack to check and load.
    pub pack_rank: u32,
    /// Upload the rank and dispatch the launches that read it. Without this, `--pack` is the
    /// RECORD gate alone, which is host-only and therefore runnable in CI — and the distinction
    /// has to be a flag rather than "did the caller pass `--hip-lib`", because that one has a
    /// default and a default cannot be asked whether it was meant.
    pub pack_launches: bool,
    /// Positions the pack's rope table covers; it must match the one it was built with, because
    /// the record's LENGTH is a function of it.
    pub rope_max_pos: u32,
    /// Timed passes over the pack's weight-reading launches.
    pub iters: u32,
    /// Run the proposal as a CHAIN rather than as seeded stages: every launch reads what the
    /// previous one wrote, and the only tensors taken from the capture are prod's own INPUTS.
    ///
    /// A separate flag rather than the default because it is a different claim and a weaker
    /// localiser: a chain says which BLOCK is wrong and a stage says which OP is, so a round that
    /// is debugging a kernel wants the stages and a round that is asking whether the drafter can
    /// propose wants this.
    pub no_seed: bool,
    /// Where the CHAIN writes its own intermediates, one directory per prompt, under the engine's
    /// arena-region names — the reference `saber engine --draft-replay --draft-chain-dump` holds
    /// the engine's plumbing against (ws-s7). Only `--no-seed` produces it: a staged replay's
    /// buffers are prod's own tensors and comparing those with the engine would be comparing the
    /// capture with itself.
    pub dump_chain: Option<PathBuf>,
    /// THE CONTROL FOR THE PROMPT'S CONTEXT (ws-s8): seed the chain's drafter KV cache with
    /// ZEROS instead of prod's `kv.before_ctx`.
    ///
    /// `kv.before_ctx` is prod's drafter cache as it stood BEFORE the captured step's store ran,
    /// and prod fills it from the FIRST prefill chunk onwards — `propose` runs after every target
    /// step, prefill included, and its `precompute_and_store_context_kv` writes one drafter K/V
    /// row per prompt token. Saber's engine proposes only on a DECODE step, so at its first decode
    /// step the same cache is exactly zeros. This flag is that state, and the block it produces is
    /// the measurement of what the prompt's context is worth. Only meaningful with `--no-seed`:
    /// a staged replay is handed prod's cache at every stage.
    pub empty_draft_cache: bool,
}

/// Build the plan, resolve it, and — with a capture — run the selector and compare.
///
/// # Errors
/// A rules refusal for a shape the description declares, a launch count that disagrees with the
/// declared budget, a missing capture file, or a HIP failure.
pub fn run(hip: Option<&HipThread>, args: &DraftStepArgs) -> anyhow::Result<usize> {
    let cfg = DflashConfig::qwen38_dflash2(args.tp, args.spec);
    // N6: the rerank's weight form is the PACK's (`head.rerank.w8` -> the fp8 record), so the
    // record gate checks the set the pack was built from.
    let rerank = match args.pack.as_ref() {
        Some(dir) => {
            let file = if dir.is_dir() {
                dir.join(format!("rank{}.pack", args.pack_rank))
            } else {
                dir.clone()
            };
            saber_engine::head_share::drafter_rerank_kind(&PackReader::open(&file)?)
        }
        None => saber_model::dflash_graph::RerankHead::Bf16,
    };
    let graph = DraftGraph::build(cfg.clone()).with_rerank_head(rerank);
    let rules = DflashRules::prod_dflash()?;

    // ---- 1. the plan --------------------------------------------------------------------------
    print!("{graph}");

    // ---- 2. the routes, and the launch count against the declared budget -----------------------
    println!("\n== routes through the kernel plane");
    let mut launches = 0u32;
    let mut refusals = 0usize;
    let mut by_symbol: BTreeMap<String, u32> = BTreeMap::new();
    for op in &graph.ops {
        let (Coverage::Dflash(kop), Some(shape)) = (op.coverage.clone(), op.shape) else {
            continue;
        };
        let shape = runtime_shape(kop, shape);
        match saber_kernels::dflash::resolve(kop, shape, &rules) {
            Ok(d) => {
                launches += d.n_launches * op.count;
                *by_symbol.entry(d.kernel.clone()).or_insert(0) += op.count;
            }
            Err(e) => {
                println!("  REFUSED {:<34} {e}", op.name);
                refusals += 1;
            }
        }
    }
    for (sym, n) in &by_symbol {
        println!("  {sym:<44} x{n}");
    }
    println!(
        "\n  declared budget {} launches, routes resolved {} launches{}",
        graph.budget(),
        launches,
        if refusals > 0 {
            format!(", {refusals} REFUSED")
        } else {
            String::new()
        }
    );
    if refusals > 0 {
        anyhow::bail!("{refusals} op(s) the committed rules refuse; the plan cannot run");
    }
    if launches != graph.budget() {
        anyhow::bail!(
            "launch count {launches} != declared budget {}: structural rule 7 for the drafter",
            graph.budget()
        );
    }

    if args.explain {
        println!("\n== not code objects yet (a proposal needs all of them)");
        for (name, why, n) in graph.uncovered() {
            println!("  {name:<34} x{n:<3} {why}");
        }
    }

    // ---- 3. the pack, against the plan that declares it ----------------------------------------
    let mut bad = 0usize;
    if let Some(dir) = args.pack.as_ref() {
        bad += pack_gate(hip, args, &graph, &rules, dir)?;
    }

    // ---- 4. the proposal, as far as the drafter's own state reaches --------------------------
    if let Some(dir) = args.proposal.as_ref() {
        let Some(hip) = hip else {
            anyhow::bail!("--proposal needs a device; pass --hip-lib");
        };
        // Two shapes of capture reach this flag and they answer different questions. A
        // CONTEXT-STORE capture (`ctx.json`, from `capture_dflash_ctx.py`) drives the three
        // launches of `precompute_and_store_context_kv` through the launcher, which is round 19's
        // gate. A PROPOSAL capture (`index.json`, from `proposal_capture.sh`) is one whole decode
        // step of prod's `DFlash2Speculator.propose` -- its hidden states, its index tables, its
        // KV cache, its candidates, its selector scores and its k=7 draft block -- and drives the
        // staged comparison in `draft_proposal`. Which one it is, is a fact about the directory,
        // so it is read rather than asked for on the command line.
        bad += if dir.join("index.json").exists() {
            crate::draft_proposal::run(hip, args, &cfg, &graph, &rules, dir)?
        } else {
            proposal(hip, args, &cfg, &graph, &rules, dir)?
        };
    }

    // ---- 5. the selector, against a live serve -------------------------------------------------
    let Some(dir) = args.sel_data.as_ref() else {
        if args.proposal.is_none() && args.pack.is_none() {
            println!("\n(no --sel-data: the plan and the routes only)");
        }
        return Ok(bad);
    };
    let Some(hip) = hip else {
        anyhow::bail!("--sel-data needs a device; pass --hip-lib");
    };
    println!("\n== the selector, through the launcher, against prod's own output");
    let man: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("selector.json"))?)?;
    let shape_of = |k: &str| -> anyhow::Result<Vec<usize>> {
        Ok(man[k]["shape"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("no shape for {k}"))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect())
    };
    let cand = shape_of("candidate_ids")?;
    let positions: u32 = cand[..cand.len() - 1].iter().product::<usize>() as u32;
    let l_len: u32 = if cand.len() >= 2 { cand[1] as u32 } else { 1 };
    let top_k = cand[cand.len() - 1] as u32;
    let rank = shape_of("projection_weight")?[0] as u32;
    let hidden = shape_of("projection_weight")?[1] as u32;
    println!("   captured: {positions} positions of {l_len}, top_k {top_k}, rank {rank}");

    let load = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &load)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let counters = LaunchCounters::new();
    let mut launcher = Launcher::new("draft-step", &counters);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;

    // The projection first -- its output is the scorer's `hidden` input -- then the scorer.
    let proj = saber_kernels::dflash::resolve(
        DflashOp::BgemmBf16,
        DflashShape::new(positions, rank, hidden),
        &rules,
    )?;
    let score = saber_kernels::dflash::resolve(
        DflashOp::SelectorScore,
        DflashShape::selector(positions, l_len, rank, top_k),
        &rules,
    )?;

    let mut owned: Vec<DevicePtr> = Vec::new();
    let mut total = 0u64;
    let mut buf: BTreeMap<&str, usize> = BTreeMap::new();
    // (descriptor arg name, capture file, bytes) -- the capture's names are the probe's, and the
    // mapping is stated here rather than by renaming files, so the probe stays readable.
    let inputs: &[(&str, &str)] = &[
        ("a", "sel.hidden_states"),
        ("w", "sel.proj_w"),
        ("pred_cb", "sel.pred_cb"),
        ("succ_cb", "sel.succ_cb"),
        ("cand", "sel.candidate_ids"),
        ("anchor", "sel.anchor_token_ids"),
        ("unary", "sel.unary_logits"),
    ];
    for (arg, file) in inputs {
        let bytes = npy::read_u8(&dir.join(format!("{file}.npy")))?;
        total += bytes.len() as u64;
        let p = ctx.hip.malloc(bytes.len() as u64)?;
        ctx.hip.memcpy_h2d_blocking(&p, &bytes, &stream)?;
        buf.insert(arg, owned.len());
        owned.push(p);
    }
    // `hidden` is the projection's output and the scorer's input; `out` is the answer.
    let hid_bytes = (positions as usize) * (rank as usize) * 2;
    let out_bytes = npy::read_u8(&dir.join("sel.out.npy"))?;
    total += (hid_bytes + out_bytes.len()) as u64;
    if total / (1024 * 1024) > args.budget_mib {
        anyhow::bail!(
            "{} MiB of device buffers exceeds --budget-mib {}",
            total / (1024 * 1024),
            args.budget_mib
        );
    }
    // Both start zeroed, so a kernel that fails to write a byte shows up as a mismatch rather than
    // as whatever the allocator handed back -- the gate's own convention.
    let hid = ctx.hip.malloc(hid_bytes as u64)?;
    ctx.hip.memset(&hid, 0, &stream)?;
    let out = ctx.hip.malloc(out_bytes.len() as u64)?;
    ctx.hip.memset(&out, 0, &stream)?;

    let bind =
        |desc: &LaunchDescriptor, extra: &[(&str, DeviceAddr)]| -> anyhow::Result<ArgBindings> {
            let mut b = ArgBindings::new(Shape::new(positions, rank, hidden));
            for slot in &desc.args {
                if let ArgValue::Buffer { buffer } = &slot.value {
                    if let Some((_, p)) = extra.iter().find(|(n, _)| n == buffer) {
                        b.bind(buffer, *p);
                    } else if let Some(i) = buf.get(buffer.as_str()) {
                        b.bind(buffer, owned[*i].addr());
                    } else {
                        anyhow::bail!("no capture for buffer `{buffer}`");
                    }
                }
            }
            Ok(b)
        };

    let b1 = bind(&proj, &[("c", hid.addr())])?;
    launcher.dispatch_one(&ctx, &proj, &b1, &stream)?;
    let b2 = bind(&score, &[("hidden", hid.addr()), ("out", out.addr())])?;
    launcher.dispatch_one(&ctx, &score, &b2, &stream)?;

    let mut got = vec![0u8; out_bytes.len()];
    ctx.hip.memcpy_d2h_blocking(&mut got, &out, &stream)?;
    let differ = got.iter().zip(&out_bytes).filter(|(a, b)| a != b).count();
    println!(
        "   2 launches; selector output {} of {} bytes{}",
        if differ == 0 {
            "identical".to_string()
        } else {
            format!("DIFFERS in {differ}")
        },
        out_bytes.len(),
        if differ == 0 {
            ""
        } else {
            "  <- NOT bit-exact"
        }
    );
    Ok(differ + bad)
}

/// The proposal's CONTEXT-KV sub-plan, run through the real [`Launcher`] in prod's own order and
/// byte-compared against prod's own C++ kernels.
///
/// This is the part of a proposal that the drafter's own state makes reachable today. The store
/// runs before any weight of the drafter is touched — it reads the TARGET's hidden states — so its
/// three covered launches can be replayed from a capture with nothing else loaded, and the rope
/// runs IN PLACE on the norm's output, which is what makes the third comparison a CHAIN rather
/// than a third independent kernel.
///
/// What is different from the gate that also compares these kernels: the gate launches them from
/// Python with hand-packed kernargs, and this launches the descriptor `dflash::resolve` produced —
/// so it also checks the grid, the block, the kernarg packing and the launcher's own binding. A
/// kernel can be bit-exact under a hand-written launch and unreachable through the plan.
///
/// # Errors
/// A missing capture file, a rules refusal, or a HIP failure.
fn proposal(
    hip: &HipThread,
    args: &DraftStepArgs,
    cfg: &DflashConfig,
    graph: &DraftGraph,
    rules: &DflashRules,
    dir: &std::path::Path,
) -> anyhow::Result<usize> {
    println!("\n== the proposal's context-KV store, through the launcher, in prod's order");
    let man: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("ctx.json"))?)?;
    let lens: Vec<u32> = man["ctx"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("the capture has no `ctx` lengths"))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u32)
        .collect();
    let h = cfg.hidden;
    let hd = cfg.head_dim;
    let nkv = cfg.kv_heads_local();
    for nm in man["symbols"].as_array().unwrap_or(&Vec::new()) {
        println!("   prod launched: {}", nm.as_str().unwrap_or(""));
    }

    let load = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &load)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let counters = LaunchCounters::new();
    let mut launcher = Launcher::new("draft-step-proposal", &counters);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;

    let mut differ = 0usize;
    let mut launches = 0u32;
    for len in lens {
        let tag = format!("c{len}");
        // Every buffer is uploaded from the capture; every OUTPUT starts zeroed, so a kernel that
        // fails to write a byte shows as a mismatch rather than as whatever the allocator held.
        let up = |name: &str| -> anyhow::Result<(DevicePtr, Vec<u8>)> {
            let bytes = npy::read_u8(&dir.join(format!("{name}.npy")))?;
            let p = ctx.hip.malloc(bytes.len() as u64)?;
            ctx.hip.memcpy_h2d_blocking(&p, &bytes, &stream)?;
            Ok((p, bytes))
        };
        let (hx, _) = up(&format!("{tag}.hidden_x"))?;
        let (hw, _) = up("norm.hidden_w")?;
        let (kx, _) = up(&format!("{tag}.k_x"))?;
        let (kw, _) = up("norm.k_w")?;
        let (pos, _) = up(&format!("{tag}.rope_pos"))?;
        let (cache, _) = up("rope.cache")?;
        let (_, ref_h) = up(&format!("{tag}.hidden_out"))?;
        let (_, ref_k) = up(&format!("{tag}.k_out"))?;
        let (_, ref_rope) = up(&format!("{tag}.rope_out"))?;
        let total_mib = (ref_h.len() + ref_k.len() + ref_rope.len()) as u64 * 3 / (1024 * 1024);
        if total_mib > args.budget_mib {
            anyhow::bail!(
                "{total_mib} MiB of device buffers exceeds --budget-mib {}",
                args.budget_mib
            );
        }
        let out_h = ctx.hip.malloc(ref_h.len() as u64)?;
        ctx.hip.memset(&out_h, 0, &stream)?;
        let out_k = ctx.hip.malloc(ref_k.len() as u64)?;
        ctx.hip.memset(&out_k, 0, &stream)?;

        let mut run = |op: DflashOp,
                       shape: DflashShape,
                       binds: &[(&str, DeviceAddr)]|
         -> anyhow::Result<()> {
            let d = saber_kernels::dflash::resolve(op, shape, rules)?;
            let mut b = ArgBindings::new(Shape::new(shape.m, shape.n, shape.k));
            for slot in &d.args {
                if let ArgValue::Buffer { buffer } = &slot.value {
                    let (_, p) = binds
                        .iter()
                        .find(|(n, _)| n == buffer)
                        .ok_or_else(|| anyhow::anyhow!("no capture for buffer `{buffer}`"))?;
                    b.bind(buffer, *p);
                }
            }
            launcher.dispatch_one(&ctx, &d, &b, &stream)?;
            Ok(())
        };

        // 1. `_project_context_kv`'s own norm over the context states, ONE group.
        run(
            DflashOp::CtxRmsNorm,
            DflashShape::ctx_norm(len, h, len),
            &[
                ("x", hx.addr()),
                ("weight", hw.addr()),
                ("out", out_h.addr()),
            ],
        )?;
        // 2. `_normalize_context_k`: ONE grouped launch over all five layers.
        let k_rows = cfg.layers * len * nkv;
        run(
            DflashOp::CtxRmsNorm,
            DflashShape::ctx_norm(k_rows, hd, len * nkv),
            &[
                ("x", kx.addr()),
                ("weight", kw.addr()),
                ("out", out_k.addr()),
            ],
        )?;
        let mut got_h = vec![0u8; ref_h.len()];
        let mut got_k = vec![0u8; ref_k.len()];
        ctx.hip.memcpy_d2h_blocking(&mut got_h, &out_h, &stream)?;
        ctx.hip.memcpy_d2h_blocking(&mut got_k, &out_k, &stream)?;
        // 3. the rope, IN PLACE on the norm's output — prod's `all_k_normed.view(L*num_ctx, kv)`.
        run(
            DflashOp::CtxRope,
            DflashShape::ctx_rope(cfg.layers * len, nkv, hd, hd),
            &[
                ("positions", pos.addr()),
                ("k", out_k.addr()),
                ("cos_sin", cache.addr()),
            ],
        )?;
        let mut got_r = vec![0u8; ref_rope.len()];
        ctx.hip.memcpy_d2h_blocking(&mut got_r, &out_k, &stream)?;
        launches += 3;

        for (name, got, want) in [
            ("hidden_norm", &got_h, &ref_h),
            ("k_norm", &got_k, &ref_k),
            ("k_norm -> rope", &got_r, &ref_rope),
        ] {
            let d = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
            differ += d;
            println!(
                "   num_ctx {len:>5}  {name:<16} {} of {} bytes",
                if d == 0 {
                    "identical".to_string()
                } else {
                    format!("DIFFERS in {d}")
                },
                want.len()
            );
        }
    }
    println!("   {launches} launches through the launcher, all from resolved descriptors");
    println!("\n== what the WHOLE k=7 proposal still needs, and it is not kernels");
    let uncovered: u32 = graph.uncovered().iter().map(|(_, _, n)| *n).sum();
    println!(
        "   Every launch after the context store reads a DRAFTER WEIGHT, and the drafter's\n   \
         checkpoint is not in a Saber pack: there is no record map, no preshuffled fp8 B, no\n   \
         packed int2 head and no drafter KV cache. `the internal reference harness` has no dflash\n   \
         arm. That is the gate's blocker, not a missing kernel -- {} of the plan's {} launches\n   \
         are covered and every covered route resolves; the {uncovered} that are not are named\n   \
         above and none of them is a weight loader.",
        graph.total_launches() - uncovered,
        graph.total_launches(),
    );
    Ok(differ)
}

/// The attention's `k` is a run-time sequence length rather than a model shape; give it one so the
/// route can be resolved at plan time.
/// The two fields of an attention shape that are properties of the STEP and not of the model: the
/// sequence length and the block table's row stride. A plan can declare neither, so a check that
/// resolves the plan's declarations asks the question at a representative step. `bt_stride` is
/// refused at zero by name (workstream F round 22), which is the point — the value has no knob and
/// no derivation — so leaving it unset here would read as a plan defect instead of as a caller
/// that has not been told.
fn runtime_shape(op: DflashOp, s: DflashShape) -> DflashShape {
    if matches!(op, DflashOp::AttnBf16Kv | DflashOp::AttnFp8Kv) {
        DflashShape::attn(s.m, s.n, 4096, s.r, s.block_size, 1)
    } else {
        s
    }
}

/// The DRAFTER PACK, against the plan that declares it — and then the weight-streaming half of a
/// proposal, from those bytes, through the real launcher.
///
/// Two phases, because they answer two different questions and only the second needs a card:
///
/// 1. **The record gate, on the host.** Every record
///    [`saber_model::dflash_graph::DraftGraph::records`] declares must be in the pack with the
///    layout id, dtype, shape and length the plan says. This is the check the round exists for: a
///    proposal cannot run from a pack whose names, layouts or shapes are a transform away from
///    what the kernels read, and none of those four is visible in a byte count.
/// 2. **The launches the pack alone makes reachable.** Every op whose inputs are (a weight, an
///    activation) — the 21 blockscale GEMMs, the 11 `bf16` projections, the int2 head and its
///    exact rerank — resolved through `dflash::resolve`, bound with its WEIGHT arguments taken
///    from the uploaded arena by record name and its activations from zeroed scratch, and
///    dispatched in forward order.
///
/// **What phase 2 is not.** It is not a proposal and it does not claim one: the activations are
/// zeros, the five absent ops are absent, and the conv / attention / KV / selector launches need
/// index tables a capture has to supply. What it measures is the thing the pack is FOR — whether
/// the weights it holds stream at prod's rate through the kernels that read them — and what it
/// proves is that every one of those kernels' weight kernargs binds by name out of the pack. The
/// census's own figure for the same set of launches is printed beside it so the number has a
/// comparator instead of a feeling.
fn pack_gate(
    hip: Option<&HipThread>,
    args: &DraftStepArgs,
    graph: &DraftGraph,
    rules: &DflashRules,
    dir: &std::path::Path,
) -> anyhow::Result<usize> {
    let rank = args.pack_rank;
    let path = dir.join(format!("rank{rank}.pack"));
    let reader = PackReader::open(&path)?;
    let want = graph.records(DraftRope::qwen38_dflash2(args.rope_max_pos));
    println!("\n== the drafter pack against the plan that declares it");
    println!(
        "   pack      {} ({} records, plan {})",
        path.display(),
        reader.records().len(),
        hex12(&reader.header().plan_hash)
    );

    let mut have: BTreeMap<&str, (&saber_mem::pack::PackRecord, usize)> = BTreeMap::new();
    for (i, rec) in reader.records().iter().enumerate() {
        have.insert(reader.name(i), (rec, i));
    }
    let mut bad = 0usize;
    let mut bytes = 0u64;
    let mut by_layout: BTreeMap<u32, u32> = BTreeMap::new();
    for w in &want {
        let Some((rec, _)) = have.get(w.name.as_str()) else {
            println!("   MISSING   {}", w.name);
            bad += 1;
            continue;
        };
        bytes += rec.length;
        *by_layout.entry(rec.layout_id).or_insert(0) += 1;
        if rec.layout_id != w.layout_id {
            println!(
                "   LAYOUT    {}: pack {}, plan {}",
                w.name,
                saber_mem::pack::LayoutId::describe(rec.layout_id),
                saber_mem::pack::LayoutId::describe(w.layout_id)
            );
            bad += 1;
        }
        if rec.dtype as u16 != dtype_tag(w.dtype) {
            println!(
                "   DTYPE     {}: pack {:?}, plan {}",
                w.name, rec.dtype, w.dtype
            );
            bad += 1;
        }
        if rec.shape != w.shape {
            println!(
                "   SHAPE     {}: pack {:?}, plan {:?}",
                w.name, rec.shape, w.shape
            );
            bad += 1;
        }
    }
    println!(
        "   records   {} of {} declared present, {} MiB, {bad} mismatch(es)",
        want.len() - bad.min(want.len()),
        want.len(),
        bytes >> 20
    );
    for (l, n) in &by_layout {
        println!(
            "     {:<22} {n:>3}",
            saber_mem::pack::LayoutId::describe(*l)
        );
    }
    // The plan's own record set is not a superset of the pack's by accident: a record the pack
    // carries that nothing declares is a builder that invented one, and it is named rather than
    // ignored.
    let declared: std::collections::BTreeSet<&str> = want.iter().map(|r| r.name.as_str()).collect();
    for (i, _) in reader.records().iter().enumerate() {
        let n = reader.name(i);
        if !declared.contains(n) {
            println!("   EXTRA     {n}: in the pack, not in the plan");
            bad += 1;
        }
    }
    if bad > 0 {
        anyhow::bail!("{bad} record(s) disagree with the plan; a proposal cannot bind this pack");
    }

    if !args.pack_launches {
        println!("   (the record gate only; --pack-launches uploads the rank and dispatches)");
        return Ok(0);
    }
    let Some(hip) = hip else {
        anyhow::bail!("--pack-launches needs a device; pass --hip-lib");
    };
    pack_launches(hip, args, graph, rules, dir, &want)
}

/// Phase 2: upload the rank and run every launch whose operands are a pack weight and an
/// activation, in forward order, timed.
fn pack_launches(
    hip: &HipThread,
    args: &DraftStepArgs,
    graph: &DraftGraph,
    rules: &DflashRules,
    dir: &std::path::Path,
    want: &[saber_model::dflash_graph::DraftRecord],
) -> anyhow::Result<usize> {
    let rank = args.pack_rank;
    let cfg = UploadConfig {
        mode: OpenMode::Direct,
        ..UploadConfig::default()
    };
    let upload = Upload::open(dir, rank, cfg)?;
    let uplan = upload.plan();
    if uplan.arena_bytes > args.budget_mib * (1 << 20) {
        anyhow::bail!(
            "the drafter's arena is {} MiB and --budget-mib is {}",
            uplan.arena_bytes >> 20,
            args.budget_mib
        );
    }
    let mut ledger = Ledger::new();
    ledger.declare(Pool::Vram(rank), uplan.arena_bytes.max(1))?;
    ledger.declare(Pool::HostPinned, uplan.staging_bytes.max(1))?;
    let reserved = upload.reserve(&mut ledger)?;
    let mut staging = hip.host_malloc(reserved.staging_bytes())?;
    let ring = PinnedRing::new(&mut staging, &reserved)?;
    let mut sink = HipSink::new(hip, &reserved)?;
    let mut addr: BTreeMap<String, DeviceAddr> = BTreeMap::new();
    let stats = upload.run(&mut sink, &ring, |r| {
        addr.insert(r.name.to_string(), r.addr);
    })?;
    println!(
        "   load      {} records, {} MiB into VRAM, {:.2} GB/s end to end (sha256 per record \
         before the H2D)",
        addr.len(),
        uplan.arena_bytes >> 20,
        stats.end_to_end_gbps()
    );

    // Which records each op reads, derived from the plan rather than from a table here: a record
    // names its op and its kernarg is its name's last segment.
    let mut weights_of: BTreeMap<&str, Vec<(&str, DeviceAddr)>> = BTreeMap::new();
    for r in want {
        let (Some(op), Some(a)) = (r.op.as_deref(), addr.get(&r.name)) else {
            continue;
        };
        weights_of.entry(op).or_default().push((r.kernarg(), *a));
    }

    let load = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &load)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let counters = LaunchCounters::new();
    let mut launcher = Launcher::new("draft-step-pack", &counters);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;

    // Scratch, sized from the SHAPE each op declares and the tile the kernel writes. Every one of
    // these is an over-allocation on purpose: a kernel that runs past a scratch buffer on a shared
    // card is a fault, and the cost of a megabyte here is nothing.
    let mut scratch: Vec<DevicePtr> = Vec::new();
    let mut alloc = |n: u64| -> anyhow::Result<DeviceAddr> {
        let p = ctx.hip.malloc(n.max(4096))?;
        ctx.hip.memset(&p, 0, &stream)?;
        let a = p.addr();
        scratch.push(p);
        Ok(a)
    };

    struct Planned {
        name: String,
        op: DflashOp,
        desc: LaunchDescriptor,
        binds: Vec<(String, DeviceAddr)>,
        shape: DflashShape,
    }
    let mut planned: Vec<Planned> = Vec::new();
    for op in &graph.ops {
        let (Coverage::Dflash(kop), Some(shape)) = (op.coverage.clone(), op.shape) else {
            continue;
        };
        if !matches!(
            kop,
            DflashOp::GemmBlockscale
                | DflashOp::BgemmBf16
                | DflashOp::DraftHeadInt2
                | DflashOp::RerankExact
        ) {
            continue;
        }
        let desc = saber_kernels::dflash::resolve(kop, shape, rules)?;
        let m = u64::from(shape.m.max(16));
        let n = u64::from(shape.n);
        let k = u64::from(shape.k);
        let r = u64::from(shape.r.max(1));
        let mut binds: Vec<(String, DeviceAddr)> = Vec::new();
        let ws = weights_of
            .get(op.name.as_str())
            .cloned()
            .unwrap_or_default();
        for slot in &desc.args {
            let ArgValue::Buffer { buffer } = &slot.value else {
                continue;
            };
            if let Some((_, a)) = ws.iter().find(|(nm, _)| nm == buffer) {
                binds.push((buffer.clone(), *a));
                continue;
            }
            let bytes = match (kop, buffer.as_str()) {
                (DflashOp::GemmBlockscale, "a") => m * k,
                (DflashOp::GemmBlockscale, "as") => m * k.div_ceil(128) * 4,
                (DflashOp::GemmBlockscale, "c") => m * n * 2,
                (DflashOp::BgemmBf16, "a") => m * k * 2,
                (DflashOp::BgemmBf16, "c") => m * n * 2,
                (DflashOp::DraftHeadInt2, "x") => m * k * 2,
                (DflashOp::DraftHeadInt2, "xs") => m * k.div_ceil(128) * 4,
                (DflashOp::DraftHeadInt2, "y") => m * n * 2,
                // `bmax`/`bidx` carry KCAND entries per 64-column block.
                (DflashOp::DraftHeadInt2, "bmax" | "bidx") => m * n.div_ceil(64) * 8 * 4,
                (DflashOp::RerankExact, "x") => m * k * 2,
                (DflashOp::RerankExact, "idx") => m * r * 4,
                // The plain variant writes `[M, R]` fp32; the `_scatter` one writes the head's own
                // `[M, N]`. Size for the larger so a rules change cannot walk off the end.
                (DflashOp::RerankExact, "out") => m * (r * 4).max(n * 2),
                _ => anyhow::bail!("{}: no scratch size for `{buffer}`", op.name),
            };
            binds.push((buffer.clone(), alloc(bytes)?));
        }
        planned.push(Planned {
            name: op.name.clone(),
            op: kop,
            desc,
            binds,
            shape,
        });
    }

    let issue = |launcher: &mut Launcher, p: &Planned| -> anyhow::Result<()> {
        let mut b = ArgBindings::new(Shape::new(p.shape.m, p.shape.n, p.shape.k));
        for (nm, a) in &p.binds {
            b.bind(nm, *a);
        }
        launcher.dispatch_one(&ctx, &p.desc, &b, &stream)?;
        Ok(())
    };

    // One pass to warm, then `--iters` timed passes; the median is reported because a first pass
    // reads every weight off a cold Infinity Cache and a mean would carry it.
    for p in &planned {
        issue(&mut launcher, p)?;
    }
    ctx.hip.sync_stream(&stream)?;
    let mut samples: Vec<f64> = Vec::new();
    for _ in 0..args.iters.max(1) {
        let t0 = std::time::Instant::now();
        for p in &planned {
            issue(&mut launcher, p)?;
        }
        ctx.hip.sync_stream(&stream)?;
        samples.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(f64::total_cmp);
    let med = samples[samples.len() / 2];
    let mut by_class: BTreeMap<&str, u32> = BTreeMap::new();
    for p in &planned {
        *by_class.entry(class_of(&p.desc.kernel)).or_insert(0) += 1;
    }
    println!(
        "   launches  {} weight-reading dispatches from the pack, in forward order",
        planned.len()
    );
    for (c, n) in &by_class {
        println!("     {c:<44} x{n}");
    }
    if args.explain {
        for p in &planned {
            println!(
                "     {:<40} {} of {} args bound from the pack",
                p.name,
                p.binds
                    .iter()
                    .filter(|(nm, _)| weights_of
                        .get(p.name.as_str())
                        .is_some_and(|w| w.iter().any(|(k, _)| k == nm)))
                    .count(),
                p.binds.len()
            );
        }
    }
    // The weight bytes this set actually READS, which is not the same as the bytes its records
    // hold and the difference is not small: the rerank GATHERS `m * r` rows of a 606 MiB head, so
    // charging it the whole record put the first run of this line at 817 GB/s — above the
    // measured 635 GB/s DRAM peak, which is the shape of an over-counted numerator (LEDGER,
    // "R9700 DRAM peak is 635 GB/s"). Counted per op from the SHAPE instead.
    let mut weight_bytes = 0u64;
    for p in &planned {
        let (m, n, k, r) = (
            u64::from(p.shape.m.max(16)),
            u64::from(p.shape.n),
            u64::from(p.shape.k),
            u64::from(p.shape.r.max(1)),
        );
        weight_bytes += match p.op {
            DflashOp::GemmBlockscale => n * k + n.div_ceil(128) * k.div_ceil(128) * 4,
            DflashOp::BgemmBf16 => n * k * 2,
            DflashOp::DraftHeadInt2 => n * k / 4 + 2 * n * k.div_ceil(128) * 2,
            DflashOp::RerankExact => m * r * k + r * 4,
            _ => 0,
        };
    }
    // ~1.2 GB against a 64 MB Infinity Cache: nineteen times over, so a repeated pass cannot be
    // reading the weights out of the IC. That is the trap the LEDGER's "a solo repeated launch
    // reads a <= 64 MB weight from the Infinity Cache" rule is about, and this is why it does not
    // bite here.
    println!(
        "   time      {med:.1} us for the set (median of {}), reading {} MiB of weights = \
         {:.0} GB/s of the card's measured 635",
        samples.len(),
        weight_bytes >> 20,
        weight_bytes as f64 / med * 1e-3,
    );
    println!(
        "   compare   prod's own census (rocprofv3 over a live dflash step, docs/LEDGER.md \
         2026-09-20) puts the same kernel classes at 1567 us for 21 blockscale GEMMs, 397.9 us \
         per draft-head launch, 507.8 us for 13 `Cijk_` projections and 11.2 us for the rerank. \
         A COMPARATOR, not a paired A/B: different instrument, different process."
    );
    println!(
        "   NOT A PROPOSAL: the activations are zeros and the conv / attention / KV / selector \
         launches are not in this set. What it says is that every weight kernarg binds by name \
         out of the pack and that the bytes stream."
    );
    drop(scratch);
    Ok(0)
}

/// The class name a launch is counted under, for the per-kernel line.
fn class_of(kernel: &str) -> &'static str {
    if kernel.starts_with("dflash_gemm_a8w8") {
        "dflash_gemm_a8w8_blockscale"
    } else if kernel.starts_with("dflash_bgemm") {
        "dflash_bgemm_bf16"
    } else if kernel.starts_with("dflash_draft_head") {
        "dflash_draft_head_int2"
    } else if kernel.starts_with("dflash_rerank") {
        "dflash_rerank_exact"
    } else {
        "other"
    }
}

/// The pack's dtype tag for a plan dtype. Two enums, one wire value, and the mapping is here
/// rather than a `From` impl because `saber-model` must not depend on `saber-mem`.
fn dtype_tag(d: saber_model::graph::Dtype) -> u16 {
    use saber_model::graph::Dtype;
    match d {
        Dtype::U8 => 0,
        Dtype::I8 => 1,
        Dtype::F16 => 8,
        Dtype::Bf16 => 9,
        Dtype::F32 => 10,
        Dtype::F8E4M3 => 12,
        Dtype::Fp4E2m1X2 => 14,
        Dtype::E8M0 => 16,
    }
}

/// First six bytes of a 32-byte hash, for a log line.
fn hex12(h: &[u8; 32]) -> String {
    h[..6].iter().map(|b| format!("{b:02x}")).collect()
}
