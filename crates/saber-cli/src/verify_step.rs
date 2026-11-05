//! `saber verify-step` — P5's TARGET side: a decode step whose five rows are DRAFT tokens.
//!
//! The `b1s5` bucket has been a real shape since ws-h11 and a real *speculative step* since this
//! command: five rows that are a draft, a per-row argmax that says which of them the model would
//! have written, an acceptance law over that, and the one property that makes speculation sound —
//!
//! > **at temperature 0, the committed sequence is the SPEC=0 sequence.** Speculation buys steps,
//! > never tokens.
//!
//! That property is a statement about STATE, and the state is where it can quietly fail: both GDN
//! decode kernels open with `si = sidx[n * stride + (naccept[n] - 1)]`, so `naccept` — the number of
//! rows the PREVIOUS step committed — picks which of that step's per-row snapshots this one resumes
//! from, and the conv cache is read at the same offset. Get it wrong and every dispatch still runs,
//! every shape still checks, a token still comes out, and the sequence's recurrence has forked.
//!
//! So the gate here is a chain: one verify step that drafts five rows and commits `a` of them must
//! leave exactly what `a` single-row steps over the same tokens leave.
//!
//! * **arm A** — two verify steps: `A1` drafts `X = [x0..x4]` from the seed state, `A2` drafts
//!   `Y = [y0..y4]` with `naccept = a`, so it resumes from `X`'s row `a - 1`;
//! * **arm B** — `a` single-row steps over `x0..x(a-1)`, then one more over `y0`. Each is
//!   `naccept = 1`, which is the only answer at `q = 1`.
//!
//! Compared, per GDN layer, bit for bit: the recurrent snapshot both arms leave in slot
//! `FIRST_STATE_SLOT`, the three conv-cache entries the next step would read (at offset `a - 1` out
//! of arm A's seven-wide window and offset 0 out of arm B's three-wide one), and — the arena half —
//! the logits row for `y0`. Then `a` runs from 1 to 5, and a NEGATIVE CONTROL runs `A2` with the
//! pre-speculation law (`naccept = live_rows`) and shows it diverges.
//!
//! # What is real here and what is not
//!
//! Since P4's glue landed, **at TP=1 every op in the plan has a code object**: `qk_norm_rope`,
//! `attn_gate_quant` and `kv_append` are bound, the paged cache is written, and the logits are the
//! model's. At TP > 1 exactly one op is not — `logits.gather`, the vocabulary all-gather the caller
//! performs — so a rank's logits are its own shard. **The report derives that from
//! `plan.supplied` rather than stating it**, which is the only way a caveat cannot outlive the
//! thing it describes.
//!
//! The plan is still built WITHOUT slot reuse, for one reason: the gate reads each attention
//! layer's own `.attn.core` row after the step, and a shared slot would hold the last layer's
//! values. The state comparison needs nothing of the kind — K/V, recurrent and conv buffers are per
//! layer and never shared — so an engine that runs with reuse ON is gated by the same laws, minus
//! the per-layer attention read.
//!
//! GPU rules (CLAUDE.md): the shared lock, the shared idle check, the power cap read and never
//! written; at TP=2 both cards, two processes, every host wait deadline-bounded.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use saber_exec::forward::{accept_prefix, Accepted, SeqFacts, StepIndex, FIRST_STATE_SLOT};
use saber_exec::layer::{
    BlockExecutor, BlockInputs, BlockPlan, BoundWeight, Collectives, PlanOptions,
};
use saber_exec::tp2;
use saber_exec::{
    bind_steps, LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet, RankPeer,
    RuntimeParams, TcpExchange,
};
use saber_hip::{DevicePtr, HipThread, StreamFlags};
use saber_kernels::head::HeadRules;
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_mem::dio::OpenMode;
use saber_mem::ledger::{Ledger, Pool};
use saber_mem::pack::PackReader;
use saber_mem::upload::{HipSink, PinnedRing, Upload, UploadConfig};
use saber_model::graph::{Geometry, Graph, ModelConfig, Phase};

use crate::layer::{pack_file, to_device};
use crate::step_capture::read_prompt_ids;

/// How long the parent waits for a rank before killing it.
const RANK_DEADLINE: Duration = Duration::from_secs(1800);

/// The draft ids after row 0, and the ones after the bonus token. Fixed values on purpose — a draft
/// is an INPUT to this gate, not a thing it measures, and a fixed one makes every run comparable —
/// but **`spec - 1` of them**, which they were not until ws-s2.
///
/// They were `[i32; 4]`, so the gate could only ever build a FIVE-row verify step whatever `--spec`
/// said. With the graph built at the default `VERIFY_WIDTH` too, the two agreed and nothing said
/// so; building the graph at `--spec` (ws-s2) made the disagreement reachable and `--spec 8` then
/// handed a step of eight rows five ids. `StepIndex`'s own law caught it rather than letting it
/// run — *"slot 0 accepted 8 of the previous step's rows, and this step's `sidx` block is 5 wide"*
/// — which is `saber_exec::forward::check_with` doing exactly what ws-h15 put it there for.
fn draft_tail(spec: u32, base: i32) -> Vec<i32> {
    (0..spec.saturating_sub(1) as i32)
        .map(|i| base + i)
        .collect()
}

/// Arguments of `saber verify-step`.
pub struct VerifyStepArgs {
    /// The device-final tier pack: the TP=1 pack at `--tp 1`, the TP=2 directory at `--tp 2`.
    pub pack: PathBuf,
    /// A JSON array of token ids, or a `.jsonl` corpus whose rows carry `token_ids`.
    pub prompt_ids: PathBuf,
    /// Row of the corpus to take.
    pub prompt_index: usize,
    /// Directory of committed `.hsaco` + `*.manifest.toml`.
    pub prebuilt: PathBuf,
    /// Context band the paged KV cache is sized for.
    pub ctx: u32,
    /// Draft width: rows of the verify step. 5 is the `b1s5` bucket.
    pub spec: u32,
    /// Refuse to start if the device buffers would exceed this.
    pub budget_mib: u64,
    /// HIP device, at TP=1.
    pub device: i32,
    /// Tensor parallelism: 1 (one process) or 2 (two).
    pub tp: u32,
    /// This process's rank at TP=2. `None` = the parent that spawns both.
    pub rank_of: Option<u32>,
    /// The loopback port the two ranks meet on.
    pub port: Option<u16>,
    /// The HIP runtime to load, passed through to the children.
    pub hip_lib: String,
    /// P27-W2-5: run arm A on the REPLAY form (`--engine-gdn-replay on`'s plan law) against the
    /// SNAPSHOT law on the same steps (arm B: the shipped verify step twice, then one shipped
    /// single-row step), both on the f16 stored state. Each arm gains a third step (one row over
    /// a fixed token `z`, which in arm A replays the bonus row) so arm A's base is the state after
    /// `y0` -- arm B's slot 1 after its second step. (The single-row CHAIN is not the reference on
    /// the f16 state: it rounds to f16 after every row, a verify step after its last.)
    pub gdn_replay: bool,
}

/// P27-W2-5: the fixed token of arm A's third step / arm B's last (replay mode only).
const REPLAY_Z: i32 = 3001;
/// P27-W2-5: the sequence's stash tag in the gate (any value; the header must carry it).
const REPLAY_TAG: u32 = 0x5a17;

/// Run it: one process at TP=1, a parent and two ranks at TP=2.
///
/// # Errors
/// Anything the pass refuses; the count of failed gates is the return value, not an error.
pub fn run(hip: Option<&HipThread>, args: &VerifyStepArgs) -> anyhow::Result<usize> {
    if args.rank_of.is_none() && args.port.is_some() {
        anyhow::bail!(
            "--port without --rank-of: this process cannot tell whether it is the parent or a \
             rank, and a parent that guessed would spawn two more children (ws-h10)."
        );
    }
    if args.spec < 2 {
        anyhow::bail!(
            "--spec {} is not a draft: a verify step has 2 or more rows",
            args.spec
        );
    }
    // A ROW COUNT IS NOT A BUCKET (ws-l14's plan-doc entry, commit 8aae434). `bucket_for(m, Decode)`
    // resolves `m` against the graph's OWN verify width: five rows is `b1s5` at width 5 and `b5s1`
    // at width 1, and both are five rows. This gate plans at `spec` rows and then compares ONE
    // drafted sequence against a chain of single-row steps, so a `spec` the graph does not read as
    // one sequence of `spec` rows would silently gate five independent recurrences and compare
    // nothing. Refused here, by name, before a card is touched.
    //
    // **The graph is built AT `--spec`** (ws-s2). It was built at the default `VERIFY_WIDTH` until
    // 2026-09-21, so `--spec 8` — prod's own SPEC=7 — read as `b8s1`, eight independent
    // recurrences, and was refused by the check below rather than run: the gate could only ever
    // ask its question at width 5. `Graph::qwen38_27b_at_width` is the same builder the engine
    // uses (`planned_widths`), and it REFUSES a width past the saber bound
    // (`attn_max_decode_rows / gqa` = 10) by name, so an illegal `--spec` is a message here rather
    // than an overrun conv slot 48 layers in.
    let probe = Graph::qwen38_27b_at_width(ModelConfig::qwen38_27b(), args.tp.max(1), args.spec)?;
    match probe.bucket_for(args.spec, Phase::Decode) {
        saber_model::graph::Bucket::Decode {
            batch: 1,
            verify_width,
        } if verify_width == args.spec => {}
        other => anyhow::bail!(
            "--spec {} reads as {other:?} in a graph built at width {}: this gate needs ONE \
             sequence of {} drafted rows",
            args.spec,
            probe.verify_width,
            args.spec,
        ),
    }
    match (args.tp, args.rank_of, hip) {
        (1, _, Some(hip)) => rank_main(hip, 0, None, args),
        (2, Some(rank), Some(hip)) => {
            let port = args
                .port
                .ok_or_else(|| anyhow::anyhow!("--rank-of needs --port"))?;
            let exchange: TcpExchange = if rank == 0 {
                let listener = std::net::TcpListener::bind(("127.0.0.1", port))
                    .map_err(|e| anyhow::anyhow!("rank 0: binding port {port}: {e}"))?;
                TcpExchange::accept(&listener, Duration::from_secs(300))?
            } else {
                TcpExchange::connect(port, Duration::from_secs(300))?
            };
            let peer = Arc::new(RankPeer::new(Arc::new(exchange)));
            rank_main(hip, rank, Some(peer), args)
        }
        (2, None, _) => spawn_ranks(args),
        (_, _, None) => anyhow::bail!("a rank process needs a HIP runtime"),
        (tp, _, _) => anyhow::bail!("--tp {tp}: this command runs at 1 or 2"),
    }
}

/// The parent at TP=2: a free port, two children, a deadline.
fn spawn_ranks(args: &VerifyStepArgs) -> anyhow::Result<usize> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|e| anyhow::anyhow!("binding the rendezvous socket: {e}"))?;
    let port = listener.local_addr()?.port();
    drop(listener);
    println!("verify    TP=2: two processes, one card each, on 127.0.0.1:{port}");
    println!("          pack {}", args.pack.display());
    println!();
    let exe = std::env::current_exe()?;
    let mut children = Vec::new();
    for rank in [0u32, 1] {
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("verify-step")
            .arg("--tp")
            .arg("2")
            .arg("--rank-of")
            .arg(rank.to_string())
            .arg("--port")
            .arg(port.to_string())
            .arg("--pack")
            .arg(&args.pack)
            .arg("--prompt-ids")
            .arg(&args.prompt_ids)
            .arg("--prompt-index")
            .arg(args.prompt_index.to_string())
            .arg("--prebuilt")
            .arg(&args.prebuilt)
            .arg("--ctx")
            .arg(args.ctx.to_string())
            .arg("--spec")
            .arg(args.spec.to_string())
            .arg("--budget-mib")
            .arg(args.budget_mib.to_string())
            .arg("--hip-lib")
            .arg(&args.hip_lib);
        if args.gdn_replay {
            cmd.arg("--gdn-replay");
        }
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

/// Every index table one step needs, built through the forward library's laws.
///
/// `accepted` is the number of rows the PREVIOUS step committed, which only `*.naccept` reads and
/// which is the whole point of this command: `sidx[accepted - 1]` is the snapshot the GDN kernels
/// resume from, and the conv cache is read at the same offset.
fn tables(
    plan: &BlockPlan<'_>,
    ids: &[i32],
    done_before: u32,
    pages: &[u32],
    max_blocks: u32,
    accepted: u32,
) -> anyhow::Result<BTreeMap<String, Vec<u8>>> {
    let seqs = [SeqFacts {
        state_slot: FIRST_STATE_SLOT,
        done_before,
        kv_blocks: pages,
        live_rows: ids.len() as u32,
        rope: None,
    }];
    let step = StepIndex {
        phase: Phase::Decode,
        rows: ids.len() as u32,
        max_blocks,
        seqs: &seqs,
        ids,
        band_width: None,
    };
    let acc = [accepted];
    step.check_with(Accepted::PerSeq(&acc))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut out = BTreeMap::new();
    for key in plan.index_keys() {
        let bytes = saber_exec::forward::index_table_with(&key, &step, Accepted::PerSeq(&acc))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        out.insert(key, bytes);
    }
    Ok(out)
}

/// What one arm leaves behind, per GDN layer: the snapshot the next step resumes from, the three
/// conv entries it reads, and the logits row for the last token.
struct Left {
    /// layer name -> the recurrent snapshot in slot [`FIRST_STATE_SLOT`].
    rec: BTreeMap<String, Vec<u8>>,
    /// layer name -> the three conv entries the next step reads, per channel.
    conv: BTreeMap<String, Vec<u8>>,
    /// layer name -> the WHOLE paged K/V cache. Compared in full on purpose: the only positions
    /// that may differ between the arms are the ones the next step's `done_before` excludes, and
    /// the byte count says exactly that.
    kv: BTreeMap<String, Vec<u8>>,
    /// attention layer -> the core attention output for the bonus row, which is the value that
    /// READS the accepted prefix of the cache.
    attn: BTreeMap<String, Vec<u8>>,
    /// The logits row for the last committed token.
    logits: Vec<u8>,
}

/// Bytes that differ between two byte strings of equal length, or `usize::MAX` on a length
/// mismatch — which is a failure of the gate's own bookkeeping and must not read as agreement.
fn differ(a: &[u8], b: &[u8]) -> usize {
    if a.len() != b.len() {
        return usize::MAX;
    }
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// One rank: load the model, plan both buckets, then run the chain gate for every acceptance.
#[allow(clippy::too_many_lines)]
fn rank_main(
    hip: &HipThread,
    rank: u32,
    peer: Option<Arc<RankPeer>>,
    args: &VerifyStepArgs,
) -> anyhow::Result<usize> {
    let device = if args.tp == 1 {
        args.device
    } else {
        rank as i32
    };
    hip.set_device(device)?;
    crate::gpu_idle::require_default(hip, device, 0)?;
    let graph = Graph::qwen38_27b_at_width(ModelConfig::qwen38_27b(), args.tp, args.spec)?;
    // P27-W2-5: the replay form is built for the f16 stored state, so BOTH arms run on it under
    // `--gdn-replay` (the chain arm B on the shipped snapshot kernels); the replay arm's graph runs
    // every bucket on the pair, as the engine's does.
    let graph = if args.gdn_replay {
        graph.with_gdn_state_f16()?
    } else {
        graph
    };
    let graph_r = if args.gdn_replay {
        Some(
            Graph::qwen38_27b_at_width(ModelConfig::qwen38_27b(), args.tp, args.spec)?
                .with_gdn_state_f16()?
                .with_gdn_pair_always(),
        )
    } else {
        None
    };
    let mxfp4 = LaunchRules::prod_mxfp4()?;
    let saber = SaberRules::prod_saber()?;
    let head_rules = HeadRules::prod_head()?;
    let _ = &head_rules;
    let ids = read_prompt_ids(&args.prompt_ids, args.prompt_index)?;
    let layers: Vec<u32> = (0..graph.config.num_hidden_layers).collect();
    let spec = args.spec;
    let collectives = if args.tp == 1 {
        Collectives::Identity
    } else {
        Collectives::Real
    };
    // Every layer-local edge gets its OWN buffer and the arena is zeroed before every step, so an
    // op with no code object contributes a deterministic zero to both arms instead of whatever the
    // slot last held. That is what makes a byte comparison across two different row counts mean
    // something while the attention glue is still P's.
    let opts = {
        let mut o = PlanOptions::FORWARD
            .rank(rank)
            .with_collectives(collectives);
        o.reuse_layer_slots = false;
        o
    };
    let params = RuntimeParams {
        max_ctx: args.ctx,
        gdn_slots: spec + 2,
    };
    let plan_of = |rows: u32| -> anyhow::Result<BlockPlan<'_>> {
        Ok(BlockPlan::with_options(
            &graph,
            graph.bucket_for(rows, Phase::Decode),
            &layers,
            params,
            &mxfp4,
            &saber,
            opts,
        )?)
    };
    let verify = plan_of(spec)?;
    let single = plan_of(1)?;
    // P27-W2-5: the replay arm's two plans, rewritten by the engine's own law.
    let replay_plans: Option<(BlockPlan<'_>, BlockPlan<'_>)> = match &graph_r {
        Some(gr) => {
            let plan_r = |rows: u32| -> anyhow::Result<BlockPlan<'_>> {
                let mut p = BlockPlan::with_options(
                    gr,
                    gr.bucket_for(rows, Phase::Decode),
                    &layers,
                    params,
                    &mxfp4,
                    &saber,
                    opts,
                )?;
                let l = saber_engine::gdn_replay::rewrite_plan(&mut p, spec)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                anyhow::ensure!(
                    !l.is_empty(),
                    "the replay rewrite found no recurrent update"
                );
                Ok(p)
            };
            Some((plan_r(spec)?, plan_r(1)?))
        }
        None => None,
    };

    let load_counters = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &load_counters)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;

    // ---- this rank's weights -------------------------------------------------------------------
    let mut want = verify.weight_keys();
    if let Some((vr, sr)) = &replay_plans {
        want.extend(vr.weight_keys());
        want.extend(sr.weight_keys());
    }
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
        anyhow::bail!(
            "rank {rank}: {} record(s) this step binds are not in {}",
            want.len() - layout_of.len(),
            args.pack.display()
        );
    }
    let (lo, hi) = (idx[0], *idx.last().expect("non-empty"));
    let last = &reader.records()[hi];
    let span =
        saber_mem::pack::align_up(last.data_off + last.length) - reader.records()[lo].data_off;
    let mut budget = args.budget_mib << 20;
    budget = budget
        .checked_sub(span)
        .ok_or_else(|| anyhow::anyhow!("the weight window is {} MiB", span >> 20))?;
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
    ledger.declare(
        Pool::Vram(u32::try_from(device).unwrap_or(0)),
        uplan.arena_bytes.max(1),
    )?;
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
    if rank == 0 {
        println!();
        println!(
            "model     Qwen3.8-27B at TP={}: {} layers, hidden {}, ctx band {}, draft width {spec}",
            args.tp, graph.config.num_hidden_layers, graph.config.hidden_size, args.ctx
        );
        // The banner is COMPUTED, not written: the glue landed (workstream P4), so the only op
        // without a code object is the TP > 1 vocabulary gather, which the caller performs. A
        // hard-coded "attention is partial" line would have gone on claiming otherwise.
        if verify.supplied.is_empty() {
            println!(
                "COMPLETE  every op in the plan has a code object — the attention glue \
                 (qk_norm_rope, attn_gate_quant, kv_append) is bound, so the K/V cache is WRITTEN \
                 and the logits are the model's. What the gate compares is therefore the whole \
                 step: the GDN recurrence, the conv cache, the paged K/V, each attention layer's \
                 output and the logits."
            );
        } else {
            println!(
                "PARTIAL   {} op(s) have no code object ({}). Everything else is bound, including \
                 the attention glue and the K/V append, so the K/V cache is written and compared.",
                verify.supplied.values().sum::<u32>(),
                verify
                    .supplied
                    .keys()
                    .copied()
                    .collect::<Vec<_>>()
                    .join(" + ")
            );
        }
        println!(
            "arena     no slot reuse, zeroed before every step: this gate reads each attention \
             layer's own `.attn.core` row after the step, and a shared slot would hold the LAST \
             layer's values instead. The state comparison (K/V, recurrent, conv) needs neither — \
             state buffers are per layer and never shared — so an engine that runs with reuse ON \
             is gated by the same laws, just without the per-layer attention read."
        );
        println!(
            "plans     verify {} ops / {} launches at M={spec}, single {} ops / {} launches at \
             M=1, arenas {} + {} MiB",
            verify.steps.len(),
            verify.expected_launches(),
            single.steps.len(),
            single.expected_launches(),
            verify.arena_bytes >> 20,
            single.arena_bytes >> 20
        );
    }

    // ---- state: ONE set of buffers, shared by both arms -----------------------------------------
    //
    // **THE CONV CACHE'S LAYOUT IS THE GRAPH'S, NOT THE BUCKET'S** (ws-l14, `gdn_shape`): one
    // allocation serves every bucket, so `cs_dim` is `conv_width - 1 + verify_width - 1` for the
    // GRAPH's verify width in BOTH plans, and a single-row step writes entries 0..conv_width-2 of
    // a seven-wide row rather than of a three-wide one. What is the bucket's own is the WALK
    // length (`state_len_max`), which is why the write-backs differ and the layout does not.
    //
    // This gate read arm B at a stride of three until ws-l17, which is every channel but the first
    // read off its row: `accept 1` reported `conv window 1961918/6881184` — 28.5% of the bytes,
    // one of the three entries in ~85% of channels — while the recurrent snapshot, the bonus row's
    // attention and the logits row were all 0 of their millions. A gate that reads a layout the
    // planner moved reports a defect in the planner's place.
    let w = graph.config.linear_conv_kernel_dim;
    let state_len = |q: u32| w - 1 + q.saturating_sub(1);
    let len_v = state_len(spec);
    // Both arms address the one allocation at the one layout. Named rather than inlined so the
    // report below can say so.
    let len_s = len_v;
    let slots = spec + 2;
    let kv_blocks_per_seq = args.ctx.div_ceil(16).max(1);
    let mut owned: Vec<DevicePtr> = Vec::new();
    let mut inputs = BlockInputs {
        weights: weights.clone(),
        ..BlockInputs::default()
    };
    // name -> (index into `owned`, bytes, per-slot bytes)
    let mut states: BTreeMap<String, (usize, u64, u64)> = BTreeMap::new();
    let mut conv_channels: u64 = 0;
    for step in &verify.steps {
        for st in &step.op.state {
            if states.contains_key(&st.name) {
                continue;
            }
            let (bytes, per_slot) = match st.bytes_per_slot {
                Some(b) if st.name.ends_with(".conv_state") => {
                    // `b` is the graph's q = 1 size: channels x (conv_width - 1) x 2.
                    let channels = b / (u64::from(w - 1) * 2);
                    conv_channels = channels;
                    let per = channels * u64::from(len_v) * 2;
                    (per * u64::from(slots), per)
                }
                Some(b) => (b * u64::from(slots), b),
                None => match step.op.geometry {
                    Geometry::Attn {
                        head_dim,
                        block_size,
                        kv_heads,
                        ..
                    } => {
                        let b = u64::from(kv_blocks_per_seq)
                            * u64::from(kv_heads)
                            * u64::from(block_size)
                            * 2
                            * u64::from(head_dim);
                        (b, b)
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
            states.insert(st.name.clone(), (owned.len(), bytes, per_slot));
            owned.push(p);
        }
    }
    // P27-W2-5: one stash pool a GDN layer (one slot set: the gate's one sequence) and the
    // refusal word, zeroed once; the stash headers carry the gate's tag.
    let mut replay_err: Option<usize> = None;
    if let Some((vr, _)) = &replay_plans {
        let heads = graph.config.linear_num_value_heads / args.tp.max(1);
        let per_layer = saber_kernels::saber::replay_stash_bytes_per_set(heads);
        let mut layers_r = std::collections::BTreeSet::new();
        for st in &vr.steps {
            if let saber_model::graph::Site::Layer(l) = st.op.site {
                if st.op.name.ends_with(".gdn.recurrent_update") {
                    layers_r.insert(l);
                }
            }
        }
        for l in layers_r {
            budget = budget
                .checked_sub(per_layer)
                .ok_or_else(|| anyhow::anyhow!("the replay stash exceeds the budget"))?;
            let p = hip.malloc(per_layer)?;
            hip.memset(&p, 0, &stream)?;
            inputs
                .states
                .insert(saber_engine::gdn_replay::stash_key(l), p.addr());
            owned.push(p);
        }
        let e = hip.malloc(256)?;
        hip.memset(&e, 0, &stream)?;
        inputs
            .states
            .insert(saber_engine::gdn_replay::ERR_KEY.to_owned(), e.addr());
        replay_err = Some(owned.len());
        owned.push(e);
    }
    hip.sync_stream(&stream)?;
    let rec_names: Vec<String> = states
        .keys()
        .filter(|n| n.ends_with(".recurrent_state"))
        .cloned()
        .collect();
    let conv_names: Vec<String> = states
        .keys()
        .filter(|n| n.ends_with(".conv_state"))
        .cloned()
        .collect();
    // The paged K/V caches, one per attention layer, and the bytes ONE position occupies in one:
    // a layer's buffer covers `kv_blocks_per_seq` pages of `KV_BLOCK` slots, so the per-position
    // size is the buffer divided by the slots it holds. That number is what makes the comparison
    // below exact rather than approximate — the arms may differ in `spec - 1` positions and in
    // nothing else.
    let kv_names: Vec<String> = states
        .keys()
        .filter(|n| !n.ends_with(".conv_state") && !n.ends_with(".recurrent_state"))
        .cloned()
        .collect();
    let slots_per_layer = u64::from(kv_blocks_per_seq) * u64::from(saber_exec::forward::KV_BLOCK);
    let kv_position_bytes: u64 = kv_names
        .first()
        .map(|n| states[n].1 / slots_per_layer.max(1))
        .unwrap_or(0);

    // ---- index tables: one device buffer per key per plan, rewritten before every step ----------
    let pages: Vec<u32> = (0..kv_blocks_per_seq).collect();
    let mut index_ptrs: Vec<BTreeMap<String, usize>> = Vec::new();
    let mut plan_rows: Vec<(&BlockPlan<'_>, u32)> = vec![(&verify, spec), (&single, 1u32)];
    if let Some((vr, sr)) = &replay_plans {
        plan_rows.push((vr, spec));
        plan_rows.push((sr, 1));
    }
    for (plan, rows) in plan_rows {
        let seed: Vec<i32> = (0..rows as i32).collect();
        // Sized at the FURTHEST position this run addresses, not at the first step's. One table's
        // length is a function of `done_before`: `.cos_sin` is the rope cache, "sized to the
        // furthest position this step addresses", and a shorter one is a bit-identical PREFIX of a
        // longer one. The steps below run at `done_before` up to `spec` (the second verify step
        // after `spec` committed rows), so a build at `2 * spec` covers every one of them and the
        // uploads that follow are shorter writes into a buffer that is already long enough.
        let t = tables(plan, &seed, 2 * spec, &pages, kv_blocks_per_seq, 1)?;
        let mut map = BTreeMap::new();
        for (key, bytes) in &t {
            let p = to_device(hip, &stream, bytes, &mut budget)?;
            map.insert(key.clone(), owned.len());
            owned.push(p);
        }
        index_ptrs.push(map);
    }
    hip.sync_stream(&stream)?;
    // The two plans' tables are different LENGTHS at the same key names — five ids against one —
    // so each has its own device buffers and `inputs` is pointed at one plan's set at a time.
    macro_rules! point_at {
        ($imap:expr) => {
            for (key, i) in &$imap {
                inputs.indices.insert(key.clone(), owned[*i].addr());
            }
        };
    }
    if verify.arena_bytes + single.arena_bytes > budget {
        anyhow::bail!(
            "the two arenas are {} MiB and {} MiB is left",
            (verify.arena_bytes + single.arena_bytes) >> 20,
            budget >> 20
        );
    }
    let mut exec_v = BlockExecutor::new(hip, &verify, 8usize << 20)?;
    let mut exec_s = BlockExecutor::new(hip, &single, 8usize << 20)?;
    let mut exec_r: Option<(BlockExecutor, BlockExecutor)> = match &replay_plans {
        Some((vr, sr)) => Some((
            BlockExecutor::new(hip, vr, 8usize << 20)?,
            BlockExecutor::new(hip, sr, 8usize << 20)?,
        )),
        None => None,
    };
    let counters = LaunchCounters::new();
    let mut launcher = Launcher::new(format!("verify.r{rank}"), &counters);
    if let Some(p) = &peer {
        launcher = launcher.with_peer(Arc::clone(p));
    }

    // ---- everything the collectives need, before anything launches ------------------------------
    if let Some(p) = &peer {
        let mut all: Vec<(usize, &BlockPlan<'_>, &BlockExecutor)> =
            vec![(0usize, &verify, &exec_v), (1usize, &single, &exec_s)];
        if let (Some((vr, sr)), Some((ev, es))) = (&replay_plans, &exec_r) {
            all.push((2, vr, ev));
            all.push((3, sr, es));
        }
        for &(which, plan, exec) in &all {
            point_at!(index_ptrs[which]);
            let bound = bind_steps(exec, &plan.steps, &inputs)?;
            for step in &bound {
                for desc in step.descs {
                    launcher.prepare_workspaces(&ctx, desc, &step.args, &stream)?;
                }
            }
        }
        launcher.reset_ipc_state(&ctx, &stream)?;
        p.rendezvous("workspaces-up")?;
        let mut agreed = 0usize;
        for &(which, plan, exec) in &all {
            point_at!(index_ptrs[which]);
            let bound = bind_steps(exec, &plan.steps, &inputs)?;
            agreed += tp2::agree_on_collectives(&launcher, p, &bound)?;
        }
        if rank == 0 {
            println!(
                "ipc       {} MiB peer-visible per rank, {agreed} launch digest(s) agreed with the \
                 peer before anything launched",
                launcher.ipc_bytes() >> 20
            );
        }
    }

    let vocab_local = match graph
        .ops
        .iter()
        .find(|o| o.name == "lm_head")
        .map(|o| &o.geometry)
    {
        Some(Geometry::Gemm { n, .. }) => *n,
        _ => anyhow::bail!("the graph has no `lm_head` GEMM"),
    };
    let logits_at = |plan: &BlockPlan<'_>| -> anyhow::Result<u64> {
        Ok(plan
            .locate("logits.partial")
            .ok_or_else(|| anyhow::anyhow!("`logits.partial` is not planned"))?
            .offset)
    };
    let (logits_v, logits_s) = (logits_at(&verify)?, logits_at(&single)?);
    let row_bytes = usize::try_from(u64::from(vocab_local) * 2)?;

    // ---- the seed: one single-row step, whose argmax is the draft's first token -----------------
    let mut failures = 0usize;
    let mut launched_wrong = 0usize;
    let mut step_no = 0usize;
    macro_rules! one_step {
        ($plan:expr, $exec:expr, $imap:expr, $ids:expr, $done:expr, $acc:expr, $what:expr) => {
            one_step!($plan, $exec, $imap, $ids, $done, $acc, $what, None)
        };
        ($plan:expr, $exec:expr, $imap:expr, $ids:expr, $done:expr, $acc:expr, $what:expr,
         $rp:expr) => {{
            let mut t = tables($plan, $ids, $done, &pages, kv_blocks_per_seq, $acc)?;
            // P27-W2-5: the session table the engine owns, over the shared law's neutral one.
            let rp: Option<(u32, u32, u32)> = $rp;
            if let Some(r) = rp {
                for (key, bytes) in &mut t {
                    if key.ends_with(".rp") {
                        *bytes = saber_engine::gdn_replay::rp_bytes(&[r], 1);
                    }
                }
            }
            for (key, bytes) in &t {
                let i = *$imap
                    .get(key)
                    .ok_or_else(|| anyhow::anyhow!("no device buffer for index `{key}`"))?;
                anyhow::ensure!(
                    bytes.len() as u64 <= owned[i].len(),
                    "index table `{key}` is {} bytes at this step and its buffer is {}: one table's \
                     length is a function of `done_before` (`.cos_sin`, the rope cache), so the \
                     buffers are sized at the furthest position the run addresses",
                    bytes.len(),
                    owned[i].len()
                );
                hip.memcpy_h2d_blocking(&owned[i], bytes, &stream)?;
                inputs.indices.insert(key.clone(), owned[i].addr());
            }
            $exec.zero(hip, &stream)?;
            hip.sync_stream(&stream)?;
            if let Some(p) = &peer {
                p.rendezvous(&format!("step-{step_no}"))?;
            }
            step_no += 1;
            let n = $exec
                .run_steps(&ctx, &mut launcher, &stream, &$plan.steps, &inputs, false)?
                .launches;
            tp2::wait_bounded(&ctx, &stream, $what)?;
            let want = $plan.expected_launches();
            if n != want {
                println!(
                    "rank {rank}: LAUNCH BUDGET {} issued {n}, declared {want}",
                    $what
                );
                launched_wrong += 1;
            }
            n
        }};
    }
    // A fresh sequence: nothing committed before it, so the first step reads the zeroed slot.
    one_step!(
        &single,
        exec_s,
        index_ptrs[1],
        &ids[..1],
        0,
        1,
        "the seed step"
    );
    let mut seed_row = vec![0u8; row_bytes];
    exec_s.download(hip, logits_s, &mut seed_row, &stream)?;
    let x0 = host_argmax(&[seed_row.clone()], &peer, rank, "seed")?[0];
    let tail = draft_tail(spec, 1001);
    let draft: Vec<i32> = std::iter::once(x0).chain(tail.iter().copied()).collect();
    if rank == 0 {
        // With the glue bound and nothing supplied, the seed step's argmax IS the model's token,
        // so the line says which it is rather than carrying a caveat that stopped being true.
        let whose = if verify.supplied.is_empty() {
            "the model's own token: every op in the plan has a code object"
        } else {
            "not the model's token: this rank holds half a vocabulary and the gather is the \
             caller's"
        };
        println!(
            "draft     row 0 = {x0}, the seed step's argmax ({whose}); rows 1..{} = the fixed ids \
             {tail:?}",
            spec - 1
        );
    }

    // ---- the chain gate, once per acceptance ----------------------------------------------------
    let zero_state = |hip: &HipThread| -> anyhow::Result<()> {
        for (i, _, _) in states.values() {
            hip.memset(&owned[*i], 0, &stream)?;
        }
        hip.sync_stream(&stream)?;
        Ok(())
    };
    // Slot 1 of a state buffer, read as the first two slots because a D2H copy starts at 0.
    let slot_of = |hip: &HipThread, name: &str, per_slot: u64| -> anyhow::Result<Vec<u8>> {
        let (i, _, _) = states[name];
        let two = usize::try_from(per_slot * 2)?;
        let mut buf = vec![0u8; two];
        hip.memcpy_d2h_blocking(&mut buf, &owned[i], &stream)?;
        Ok(buf.split_off(usize::try_from(per_slot)?))
    };
    if rank == 0 {
        println!();
        println!(
            "gate      one verify step that commits `a` rows must leave what `a` single-row steps \
             leave: per GDN layer the recurrent snapshot in slot {FIRST_STATE_SLOT} and the {} conv \
             entries the next step reads, plus the logits row for the last committed token",
            w - 1
        );
        println!(
            "          {} recurrent state(s) and {} conv cache(s) per rank, {} conv channels, \
             both arms' windows at entries 0..{} of the GRAPH's {len_v}-wide row (the layout is \
             the graph's, the WALK is the bucket's: A2 read its window at offset a-1 and rewrote \
             it from entry 0, and a single-row step writes entries 0..{} of the same row)",
            rec_names.len(),
            conv_names.len(),
            conv_channels,
            w - 2,
            w - 2
        );
    }
    let window = usize::try_from(u64::from(w - 1) * 2)?;
    #[allow(clippy::too_many_arguments)]
    let collect = |hip: &HipThread,
                   exec: &BlockExecutor,
                   exec_plan: &BlockPlan<'_>,
                   exec_rows: u32,
                   logits_off: u64,
                   row: u32,
                   len: u32,
                   off: u32|
     -> anyhow::Result<Left> {
        let mut rec = BTreeMap::new();
        for name in &rec_names {
            let (_, _, per) = states[name];
            rec.insert(name.clone(), slot_of(hip, name, per)?);
        }
        let mut conv = BTreeMap::new();
        for name in &conv_names {
            let per = conv_channels * u64::from(len) * 2;
            let slot = slot_of(hip, name, per)?;
            let mut win = Vec::with_capacity(usize::try_from(conv_channels)? * window);
            for c in 0..usize::try_from(conv_channels)? {
                let base = (c * len as usize + off as usize) * 2;
                win.extend_from_slice(&slot[base..base + window]);
            }
            conv.insert(name.clone(), win);
        }
        // The whole paged cache per attention layer. Cheap at this context band and it makes the
        // KV claim exact: what may differ is `spec - 1` positions, and the count proves it.
        let mut kv = BTreeMap::new();
        for name in &kv_names {
            let (i, bytes, _) = states[name];
            let mut buf = vec![0u8; usize::try_from(bytes)?];
            hip.memcpy_d2h_blocking(&mut buf, &owned[i], &stream)?;
            kv.insert(name.clone(), buf);
        }
        // The attention output for the bonus row — the value that actually READS the accepted
        // prefix of the cache, so it is where a wrong `done_before` shows up first.
        let mut attn = BTreeMap::new();
        for b in &exec_plan.buffers {
            if !b.name.ends_with(".attn.core") {
                continue;
            }
            let cols = b.bytes / u64::from(exec_rows);
            let mut buf = vec![0u8; usize::try_from(cols)?];
            exec.download(hip, b.offset + u64::from(row) * cols, &mut buf, &stream)?;
            attn.insert(b.name.clone(), buf);
        }
        let mut logits = vec![0u8; row_bytes];
        exec.download(
            hip,
            logits_off + u64::from(row) * u64::from(vocab_local) * 2,
            &mut logits,
            &stream,
        )?;
        Ok(Left {
            rec,
            conv,
            kv,
            attn,
            logits,
        })
    };

    let mut natural: Option<saber_exec::forward::Acceptance> = None;
    for a in 1..=spec {
        let (arm_a, arm_b, second) = if let (Some((vr, sr)), Some((ev, es))) =
            (&replay_plans, exec_r.as_mut())
        {
            // ---- P27-W2-5 arm A: the REPLAY form. A1 drafts X (nothing to replay), A2 drafts Y
            // replaying `a` rows of X into the base, A3 runs ONE row over a fixed `z` replaying the
            // bonus row y0 -- so the base after A3 is the state after y0, the chain's slot 1. ----
            zero_state(hip)?;
            let ie = replay_err.ok_or_else(|| anyhow::anyhow!("no refusal word"))?;
            hip.memset(&owned[ie], 0, &stream)?;
            one_step!(
                vr,
                ev,
                index_ptrs[2],
                &draft,
                0,
                1,
                "rA1",
                Some((0, REPLAY_TAG, 0))
            );
            let logits_vr = logits_at(vr)?;
            let mut rows: Vec<Vec<u8>> = Vec::with_capacity(spec as usize);
            for r in 0..spec {
                let mut buf = vec![0u8; row_bytes];
                ev.download(
                    hip,
                    logits_vr + u64::from(r) * u64::from(vocab_local) * 2,
                    &mut buf,
                    &stream,
                )?;
                rows.push(buf);
            }
            let per_row = host_argmax(&rows, &peer, rank, "rA1")?;
            if natural.is_none() {
                natural = Some(accept_prefix(&draft, &per_row)?);
            }
            let y0 = per_row[(a - 1) as usize];
            let second: Vec<i32> = std::iter::once(y0).chain(draft_tail(spec, 2001)).collect();
            one_step!(
                vr,
                ev,
                index_ptrs[2],
                &second,
                a,
                a,
                "rA2",
                Some((a, REPLAY_TAG, 1))
            );
            one_step!(
                sr,
                es,
                index_ptrs[3],
                &[REPLAY_Z],
                a + 1,
                1,
                "rA3",
                Some((1, REPLAY_TAG, 0))
            );
            let logits_sr = logits_at(sr)?;
            let arm_a = collect(hip, es, sr, 1, logits_sr, 0, len_s, 0)?;
            let mut word = vec![0u8; 4];
            hip.memcpy_d2h_blocking(&mut word, &owned[ie], &stream)?;
            let err = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            if err != 0 {
                println!("accept {a}  rank {rank}: the replay REFUSED (err {err:#x})");
                failures += 1;
            }
            // ---- arm B: the SNAPSHOT LAW on the same steps -- the shipped verify step twice
            // (A2 resumes from snapshot a - 1), then one shipped single-row step over z. On the f16
            // stored state this, not the single-row chain, is the replay's reference: the chain
            // rounds the state to f16 after EVERY row while a verify step carries it in fp32
            // across its rows, so the chain is a different (valid) summation at a >= 2. The
            // replay must reproduce the snapshot law's bytes, which is what it replaces. ----
            zero_state(hip)?;
            one_step!(&verify, exec_v, index_ptrs[0], &draft, 0, 1, "sA1");
            one_step!(&verify, exec_v, index_ptrs[0], &second, a, a, "sA2");
            let at_y0 = collect(hip, &exec_v, &verify, spec, logits_v, 0, len_v, 0)?.rec;
            one_step!(&single, exec_s, index_ptrs[1], &[REPLAY_Z], a + 1, 1, "sA3");
            let mut arm_b = collect(hip, &exec_s, &single, 1, logits_s, 0, len_s, 0)?;
            arm_b.rec = at_y0;
            // ---- the replay's own negative control, once: A2 replays `a - 1` rows (the
            // 2026-09-17 failure shape -- a base missing a committed row) ----
            if a == 2 {
                zero_state(hip)?;
                one_step!(
                    vr,
                    ev,
                    index_ptrs[2],
                    &draft,
                    0,
                    1,
                    "rC1",
                    Some((0, REPLAY_TAG, 0))
                );
                one_step!(
                    vr,
                    ev,
                    index_ptrs[2],
                    &second,
                    a,
                    a,
                    "rC2 (replays a - 1 rows)",
                    Some((a - 1, REPLAY_TAG, 1))
                );
                one_step!(
                    sr,
                    es,
                    index_ptrs[3],
                    &[REPLAY_Z],
                    a + 1,
                    1,
                    "rC3",
                    Some((1, REPLAY_TAG, 0))
                );
                let bad = collect(hip, es, sr, 1, logits_sr, 0, len_s, 0)?;
                let rec_diff: usize = bad.rec.iter().map(|(n, v)| differ(v, &arm_b.rec[n])).sum();
                let logit_diff = differ(&bad.logits, &arm_b.logits);
                let caught = rec_diff > 0 && logit_diff > 0;
                if rank == 0 {
                    println!(
                        "control   replay of a - 1 = {} row(s) instead of {a}: recurrent {rec_diff} \
                         bytes and the logits row {logit_diff} bytes differ from the chain -- {}",
                        a - 1,
                        if caught { "CAUGHT" } else { "NOT CAUGHT" }
                    );
                }
                if !caught {
                    failures += 1;
                }
            }
            (arm_a, arm_b, second)
        } else {
            // ---- arm A: two verify steps, the second resuming from row a-1 --------------------------
            zero_state(hip)?;
            one_step!(&verify, exec_v, index_ptrs[0], &draft, 0, 1, "A1");
            let mut rows: Vec<Vec<u8>> = Vec::with_capacity(spec as usize);
            for r in 0..spec {
                let mut buf = vec![0u8; row_bytes];
                exec_v.download(
                    hip,
                    logits_v + u64::from(r) * u64::from(vocab_local) * 2,
                    &mut buf,
                    &stream,
                )?;
                rows.push(buf);
            }
            let per_row = host_argmax(&rows, &peer, rank, "A1")?;
            if natural.is_none() {
                // The law lives in `saber_exec::forward` so that the engine's sampler and this gate
                // cannot drift apart (ws-l14 asked for it there, and its invariant is checked inside).
                natural = Some(accept_prefix(&draft, &per_row)?);
            }
            let y0 = per_row[(a - 1) as usize];
            let second: Vec<i32> = std::iter::once(y0).chain(draft_tail(spec, 2001)).collect();
            one_step!(&verify, exec_v, index_ptrs[0], &second, a, a, "A2");
            let arm_a = collect(hip, &exec_v, &verify, spec, logits_v, 0, len_v, 0)?;

            // ---- arm B: a single-row steps over the same tokens, then one more over y0 --------------
            zero_state(hip)?;
            for i in 0..a {
                one_step!(
                    &single,
                    exec_s,
                    index_ptrs[1],
                    &draft[i as usize..=i as usize],
                    i,
                    1,
                    "B"
                );
            }
            one_step!(&single, exec_s, index_ptrs[1], &second[..1], a, 1, "B-last");
            let arm_b = collect(hip, &exec_s, &single, 1, logits_s, 0, len_s, 0)?;
            (arm_a, arm_b, second)
        };

        // ---- compare ---------------------------------------------------------------------------
        let mut rec_bad = 0usize;
        let mut rec_bytes = 0usize;
        for (name, mine) in &arm_a.rec {
            let theirs = &arm_b.rec[name];
            rec_bad += differ(mine, theirs);
            rec_bytes += mine.len();
        }
        let mut conv_bad = 0usize;
        let mut conv_bytes = 0usize;
        for (name, mine) in &arm_a.conv {
            let theirs = &arm_b.conv[name];
            conv_bad += differ(mine, theirs);
            conv_bytes += mine.len();
        }
        // The attention output for the bonus row: the value that reads the accepted prefix of the
        // cache, so it is bit-exact only if the prefix is.
        let mut attn_bad = 0usize;
        let mut attn_bytes = 0usize;
        for (name, mine) in &arm_a.attn {
            attn_bad += differ(mine, &arm_b.attn[name]);
            attn_bytes += mine.len();
        }
        // The K/V caches, whole. The arms are ALLOWED to differ in exactly the positions the next
        // step's `done_before` excludes: arm A's second step drafted `spec` rows from position `a`
        // and only its first is committed, so positions `a+1 .. a+spec-1` hold draft K/V that the
        // chain never wrote. That is `spec - 1` positions per attention layer and NOTHING else —
        // which is a number, so it is checked as one rather than described.
        let mut kv_bad = 0usize;
        let mut kv_bytes = 0usize;
        for (name, mine) in &arm_a.kv {
            kv_bad += differ(mine, &arm_b.kv[name]);
            kv_bytes += mine.len();
        }
        let kv_allowed = usize::try_from(u64::from(spec - 1) * kv_position_bytes)? * arm_a.kv.len();
        let logit_bad = differ(&arm_a.logits, &arm_b.logits);
        let ok = rec_bad == 0
            && conv_bad == 0
            && logit_bad == 0
            && attn_bad == 0
            && kv_bad <= kv_allowed;
        if !ok {
            failures += 1;
        }
        if rank == 0 || !ok {
            println!(
                "accept {a}  rank {rank}: recurrent {rec_bad}/{rec_bytes}, conv window \
                 {conv_bad}/{conv_bytes}, attn(bonus row) {attn_bad}/{attn_bytes}, logits row \
                 {logit_bad}/{} — {}",
                arm_a.logits.len(),
                if ok { "BIT-EXACT" } else { "MISMATCH" }
            );
            println!(
                "          KV {kv_bad}/{kv_bytes} bytes differ, and at most {kv_allowed} may — \
                 the {} uncommitted draft position(s) per layer that arm A wrote and the chain did \
                 not, which the next step's `done_before = {a}` excludes and its own append \
                 overwrites",
                spec - 1
            );
        }
        if let Some(p) = &peer {
            p.agree(&format!("accept-{a}"), u64::from(ok))?;
        }

        // ---- the negative control, at one acceptance: the PRE-SPECULATION law -------------------
        //
        // `naccept = live_rows` is what every caller in the tree said before this branch, and at
        // SPEC=0 it is right. Here it sends the second step to the snapshot after ALL five drafted
        // rows instead of after the `a` committed ones, so the recurrence forks — with every
        // dispatch issued, every shape checked and a token still coming out. Run once, at an
        // acceptance where the two answers differ.
        if a == 1 {
            zero_state(hip)?;
            one_step!(&verify, exec_v, index_ptrs[0], &draft, 0, 1, "control-A1");
            one_step!(
                &verify,
                exec_v,
                index_ptrs[0],
                &second,
                a,
                spec,
                "control-A2 (naccept = live_rows, the old law)"
            );
            let bad_arm = collect(hip, &exec_v, &verify, spec, logits_v, 0, len_v, 0)?;
            let rec_diff: usize = bad_arm
                .rec
                .iter()
                .map(|(n, v)| differ(v, &arm_b.rec[n]))
                .sum();
            let conv_diff: usize = bad_arm
                .conv
                .iter()
                .map(|(n, v)| differ(v, &arm_b.conv[n]))
                .sum();
            let caught = rec_diff > 0 || conv_diff > 0;
            if rank == 0 {
                println!(
                    "control   naccept = {spec} (live rows) instead of {a}: recurrent \
                     {rec_diff}/{rec_bytes} and conv {conv_diff}/{conv_bytes} bytes differ from \
                     the chain — {}",
                    if caught {
                        "CAUGHT, so the gate above has teeth"
                    } else {
                        "NOT CAUGHT, which means the gate proves nothing"
                    }
                );
            }
            if !caught {
                failures += 1;
            }

            // ---- the KV control: `done_before` advanced by the DRAFT WIDTH, not the acceptance --
            //
            // The pre-speculation reading of a verify step is "all `spec` rows committed", which
            // moves the next step's `done_before` to `spec` instead of `a`. Every dispatch still
            // runs: the step writes its K/V `spec - a` positions further along, its rope positions
            // are shifted by the same amount, and its attention reads the REJECTED rows' K/V as
            // live context. This is the `a + spec` kept-live case, and it must not be silent.
            zero_state(hip)?;
            one_step!(
                &verify,
                exec_v,
                index_ptrs[0],
                &draft,
                0,
                1,
                "kv-control-A1"
            );
            one_step!(
                &verify,
                exec_v,
                index_ptrs[0],
                &second,
                spec,
                a,
                "kv-control-A2 (done_before = the draft width, so the rejected rows stay live)"
            );
            let kv_bad_arm = collect(hip, &exec_v, &verify, spec, logits_v, 0, len_v, 0)?;
            let attn_diff: usize = kv_bad_arm
                .attn
                .iter()
                .map(|(n, v)| differ(v, &arm_b.attn[n]))
                .sum();
            let logit_diff = differ(&kv_bad_arm.logits, &arm_b.logits);
            let kv_caught = attn_diff > 0 || logit_diff > 0;
            if rank == 0 {
                println!(
                    "control   done_before = {spec} (the draft width) instead of {a}: the rejected \
                     rows stay live, so attn(bonus row) {attn_diff}/{attn_bytes} and the logits \
                     row {logit_diff}/{} differ from the chain — {}",
                    kv_bad_arm.logits.len(),
                    if kv_caught {
                        "CAUGHT, so the KV half of the gate has teeth too"
                    } else {
                        "NOT CAUGHT, which means the KV comparison proves nothing"
                    }
                );
            }
            if !kv_caught {
                failures += 1;
            }
        }
    }

    if let Some(saber_exec::forward::Acceptance {
        accepted,
        committed,
    }) = natural
    {
        if rank == 0 {
            println!();
            // The acceptance a FIXED draft earns is a property of the draft, not a result: these
            // four ids were never going to be the model's continuation. What the line reports is
            // the law's shape, and it says so instead of dressing 1-of-5 up as a measurement.
            println!(
                "law       the acceptance this draft actually earns: {accepted} of {spec} row(s), \
                 committing {} token(s) = the accepted prefix plus one the model wrote itself \
                 ({committed:?}). The ids after row 0 are FIXED CONSTANTS, so this number is a \
                 property of the draft and not of the model — what is gated is the law: a prefix \
                 that stops at the first disagreement, and `committed = accepted + 1`.",
                committed.len()
            );
            println!(
                "          KV: the cache IS written (kv_append is bound), and the law is that a \
                 step which commits `a` rows advances the sequence by `a` and not by its draft \
                 width — arm A's second step declares seqused_k = a + {spec} while the chain's \
                 last declares a + 1, and both leave the next step at a + 1, which is why the \
                 only positions that may differ are the uncommitted draft rows above."
            );
        }
    }
    if launched_wrong > 0 {
        failures += launched_wrong;
    }
    if rank == 0 {
        println!();
        println!(
            "budget    every step issued exactly what its plan declares ({} at M={spec}, {} at \
             M=1) — {launched_wrong} exception(s)",
            verify.expected_launches(),
            single.expected_launches()
        );
    }
    println!("rank {rank}  {}", counters.snapshot().line());
    drop(exec_v);
    drop(exec_s);
    drop(owned);
    Ok(failures)
}

/// Per-row argmax on the host, over this rank's shard and — at TP=2 — the peer's.
///
/// The argmax has to be host-side above TP=1 in any case: `lm_head` is column-parallel, so a rank
/// holds half a vocabulary and half an argmax is not a token. Doing it on the host at TP=1 too keeps
/// one law for both.
fn host_argmax(
    rows: &[Vec<u8>],
    peer: &Option<Arc<RankPeer>>,
    rank: u32,
    what: &str,
) -> anyhow::Result<Vec<i32>> {
    let mut mine: Vec<u8> = Vec::new();
    for r in rows {
        mine.extend_from_slice(r);
    }
    let theirs = match peer {
        Some(p) => Some(
            p.exchange()
                .swap_bytes(&format!("logits-{what}"), &mine)
                .map_err(|e| anyhow::anyhow!("exchanging logits: {e}"))?,
        ),
        None => None,
    };
    let per_row = rows.first().map_or(0, Vec::len);
    let shard = (per_row / 2) as i32;
    let mut out = Vec::with_capacity(rows.len());
    for (r, row) in rows.iter().enumerate() {
        let mut best = (f32::NEG_INFINITY, 0i32);
        let mut scan = |bytes: &[u8], base: i32| {
            for (i, chunk) in bytes.chunks_exact(2).enumerate() {
                let bits = u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16;
                let v = f32::from_bits(bits);
                if v > best.0 {
                    best = (v, base + i32::try_from(i).unwrap_or(i32::MAX));
                }
            }
        };
        // Rank r owns vocabulary rows [r * shard, (r + 1) * shard), so WHICH half a buffer is
        // depends on whose it is: a rank that scanned its own shard at base 0 would report ids
        // from the wrong half of the vocabulary.
        scan(row, i32::try_from(rank).unwrap_or(0) * shard);
        if let Some(t) = &theirs {
            let base = i32::try_from(1 - rank.min(1)).unwrap_or(0) * shard;
            scan(&t[r * per_row..(r + 1) * per_row], base);
        }
        out.push(best.1);
    }
    Ok(out)
}
