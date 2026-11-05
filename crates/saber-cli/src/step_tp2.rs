//! `saber step-tp2` — the TP=2 decode step as ONE graph per rank, all-reduces included.
//!
//! This is the like-for-like number. ws-h11 captured the whole step at TP=1 and measured 33.8 ms at
//! b1s5 against prod's 18.7 — a comparison the reviewer correctly refused, because a TP=1 step
//! streams 15.4 GiB from one card while prod's 18.7 ms is TP=2 with half the weights per card. This
//! runs the step prod's own way: two processes, one card each, **129 all-reduces per rank inside the
//! captured graph**, and the same greedy, no-speculation, single-stream, GPU-idle conditions.
//!
//! # The all-reduce is a captured NODE
//!
//! ws-h9's rule was "a peer-visible workspace cannot be captured", and for one rank with no peer it
//! still is. It changes when **both** ranks capture: the reference's one-shot reduce is an ordinary
//! dispatch whose protocol state is a Persistent per-block sequence counter that each launch
//! advances, and whose `slot = seq & 1` alternates the double-buffered scratch. A replay advances it
//! exactly as a fresh launch does, so the graph is legal provided the two ranks replay the same
//! graph the same number of times in the same order — which one replay per rank per step is.
//!
//! What that costs in guarantees, and what this module pays:
//!
//! * every peer-visible region is allocated, exchanged and zeroed **before** either capture opens,
//!   and never resized afterwards (`Launcher` refuses a resize by name);
//! * the launch digest for every collective is compared on the host **before** the capture, because
//!   a graph cannot rendezvous — [`saber_exec::tp2::Protocol::Checked`] on the eager pass does that;
//! * every host wait is a bounded `hipStreamQuery` poll, and the kernel's own wait is bounded
//!   (`SABER_AR_SPIN_MAX`), so a rank that falls out of step yields a wrong answer a byte comparison
//!   catches rather than a hang that holds a card.
//!
//! [`saber_exec::PeerCapture::Lockstep`] is the name of that assertion, and it is passed exactly
//! here.
//!
//! # Stage (b), for when P's route lands
//!
//! The 129 reduces are spliced in by this module because `BlockPlan` routes `layer.allreduce` to
//! `Route::Identity` (ws-h10 wrote down the three API additions that would remove the splice). When
//! the planner resolves a real all-reduce, `collectives()` below and the interleaving in
//! `bound_step()` are the only two places to delete: the capture, the gates and the timing all work
//! off the bound sequence and do not care where it came from.
//!
//! GPU rules (CLAUDE.md): both cards under BOTH card locks (`gpu.lock.0` then `gpu.lock.1`, in that order), **GPU idle** — the
//! shared check (`crate::gpu_idle`) reads free VRAM first and prints it — power cap untouched.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use saber_exec::block_capture::{check_capturable_with, check_persistent_ready};
use saber_exec::layer::{
    BlockExecutor, BlockInputs, BlockPlan, BoundWeight, Collectives, PlanOptions,
};
use saber_exec::tp2;
use saber_exec::{
    bind_steps, ArgBindings, BoundStep, CapturableOp, CaptureCounters, CapturedBlock,
    CapturedBucket, GraphSafe, LaunchCounters, LaunchCtx, Launcher, LoadCounters, PeerCapture,
    PluginSet, RankPeer, RuntimeParams, TcpExchange,
};
use saber_hip::{DevicePtr, EventFlags, HipThread, Stream, StreamFlags};
use saber_kernels::head::HeadRules;
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_mem::dio::OpenMode;
use saber_mem::ledger::{Ledger, Pool};
use saber_mem::pack::PackReader;
use saber_mem::upload::{HipSink, PinnedRing, Upload, UploadConfig};
use saber_model::graph::{Geometry, Graph, ModelConfig, Phase};

use crate::layer::{pack_file, to_device};
use crate::step_capture::{family, index_table, read_prompt_ids, reset_state};

/// P-1b: prod at SPEC=0, greedy, one stream, GPU idle. **TP=2**, which is what makes this the
/// comparison and not a category error.
const PROD_MS_PER_STEP: f64 = 18.7;

/// How long the parent waits for a rank before killing it.
const RANK_DEADLINE: Duration = Duration::from_secs(1800);

/// Arguments of `saber step-tp2`.
pub struct StepTp2Args {
    /// Plan without slot reuse, so the fusion gate's two arms cannot share an arena buffer.
    ///
    /// The gate compares two DIFFERENT plans over one arena. With reuse on, a layer-local edge's
    /// buffer is shared across the block's layers, and the two plans do not write those slots in
    /// the same order — so a difference in the comparison is not necessarily a difference in the
    /// arithmetic. ws-h20 added this to tell those two apart, after `ar-gate`'s `ar_nq_pair` case
    /// proved the fused epilogue and the composition are bit-identical kernel-to-kernel.
    pub no_slot_reuse: bool,
    /// Which fusions the gate's fused arm plans: `both` / `all` (every launch-train fusion, what
    /// `--decode-fusion` serves), `h37` (the `_nq` epilogue and the decode pair, ws-h37's set),
    /// `nq`, `pair`, `gemm-ar` (fusion-map row 3 with the plain `_nq`, ws-h38), `rope-append` or
    /// `attn-combine` (fusion-map rows 7 and 6, ws-h38). See [`fusion_options`].
    pub fusion: String,
    /// The fusion gate's negative control: `none`, `pair-drop`, `nq-eps`, `gemmar-eps`,
    /// `gemmar-rows`, `rope-append-drop`, `combine-drop` (decisive), `pair-ks` or `nq-verbatim`
    /// (weak). See [`fusion_control`].
    pub fusion_control: String,
    /// Plan the head on every row of the chunk, in ONE combined plan, instead of the engine's
    /// body-plus-head split on the sampled rows.
    ///
    /// This command's original shape, kept because the figures it produced are in the ledger.
    /// It is not what the engine runs, and at a prefill bucket it costs a shard of logits per
    /// token that nobody reads (`docs/LEDGER.md`, workstreams N5 and N6).
    pub head_on_every_row: bool,
    /// The TP=2 device-final tier pack DIRECTORY: each rank opens its own `rank<r>.pack`.
    pub pack: PathBuf,
    /// A JSON array of token ids, or a `.jsonl` corpus whose rows carry `token_ids`.
    pub prompt_ids: PathBuf,
    /// Row of the corpus to take.
    pub prompt_index: usize,
    /// Directory of committed `.hsaco` + `*.manifest.toml`.
    pub prebuilt: PathBuf,
    /// Rows per step: at `spec_width` 5, 5 is b1s5 and 40 is b8s5.
    pub rows: Vec<u32>,
    /// **Decode ROW BANDS to run (C2)**, in rows: each is a `Bucket::DecodeBand` over
    /// [`StepTp2Args::ragged_widths`]'s slots at max width `spec_width`, its sequences feeding
    /// those widths packed by `cu` and the tail padded — the ragged step the scheduler runs, with
    /// the ragged attention pair, timed and (with `--per-op`) attributed like any bucket.
    pub ragged_rows: Vec<u32>,
    /// The per-slot widths every band here runs (C2): `Σ w_i` must fit each band. The default is a
    /// realistic conc-8 multiset under prod's cap, `5,5,5,3,5,2,5,4` (34 rows).
    pub ragged_widths: Vec<u32>,
    /// The graph's verify width — `1 + SPEC`. `--rows` resolves against it
    /// (`Graph::bucket_for`: `rows / width` is the batch), and the GDN conv slot and
    /// `gdn_shape`'s strides are derived from it, so it is a BUILD parameter of the graph and
    /// not a per-bucket choice ([`saber_model::graph::Graph::qwen38_27b_at_width`]).
    pub spec_width: u32,
    /// The GDN recurrent state STORED as f16 (C1, `--gdn-state-dtype f16`):
    /// [`Graph::with_gdn_state_f16`], forwarded to the child rank so both plan the same graph.
    pub gdn_state_f16: bool,
    /// Prefill buckets to run instead, by chunk length. Same law as `step-capture --prefill`
    /// (I15): a different op set at one sequence and no speculative width, everything else shared.
    pub prefill: Vec<u32>,
    /// Context band the paged KV cache is sized for.
    pub ctx: u32,
    /// Refuse to start if a rank's device buffers would exceed this.
    pub budget_mib: u64,
    /// Replays to byte-compare against the eager two-rank step.
    pub replays: usize,
    /// Timed graph/eager pairs.
    pub iters: usize,
    /// Untimed pairs first.
    pub warmup: usize,
    /// The measured DRAM peak, for the per-card weight-stream floor.
    pub dram_gbs: f64,
    /// The compute roof the W4A8 kernels upconvert into, for the non-linear table's `cmp` rows.
    pub peak_tflops: f64,
    /// After the whole-step timing, time every op as its own sub-graph IN PLACE, both ranks in
    /// lockstep, and print the non-linear table with each op's own floor beside it.
    pub per_op: bool,
    /// H36: with a prefill bucket, run the prefill all-reduce overlap at this slice row count
    /// (`saber_exec::overlap`) beside the sequential eager body -- byte-compared, then timed A B A B
    /// against the sequential body and the body without its reduces. 0 = off.
    pub ar_overlap: u32,
    /// P27-1d: the overlapped arm's reduce in the split form with this many push blocks; 0 = the
    /// one-shot (H36's arm, the control).
    pub ar_push_blocks: u32,
    /// With `--per-op`: print one detail line per link whose OP NAME contains this, naming the
    /// ops either side of it and timing the same op eagerly in the same position.
    pub detail: Option<String>,
    /// C6: plan every PREFILL bucket of at most this many rows on the decode GEMM family
    /// ([`saber_engine::small_chunk`]) in the gate's FUSED arm only, so the fusion gate compares
    /// that plan against the committed rules' folded plan over one arena and the timing and
    /// `--per-op` read the small-chunk plan. 0 = off.
    pub small_chunk_gemm: u32,
    /// C6: the small-chunk band law, `exact` (every shape at one split) or `band` (the
    /// reference's `M > 64` splits, the numerics-class arm and the exact gate's control).
    pub small_chunk_class: String,
    /// G5: plan every PREFILL bucket's fused arm with the folded GEMM at the mid-M tile heights
    /// (`[[folded_mt]]`, the `folded_mt` knob on); the unfused arm keeps the committed rules, so
    /// the fusion gate is the class comparison and `--per-op` reads the mid-tile plan.
    pub mid_chunk_gemm: bool,
    /// G6: plan every PREFILL bucket's fused arm under the layout law's mid band (`atiled_mid`
    /// on: the producers write the fragment layout from `atiled_mid_min_m` rows and the GEMMs
    /// take the A-tiled family, `[[atiled_mt]]`); the unfused arm keeps the committed rules.
    pub mid_chunk_atiled: bool,
    /// G6: the fused arm's 48-wide gate on the skinny-N body (`[[skinny]]`, the `skinny` knob).
    pub skinny_gate: bool,
    /// This process's rank. `None` = the parent that spawns both.
    pub rank: Option<u32>,
    /// The loopback port the two ranks meet on.
    pub port: Option<u16>,
    /// The HIP runtime to load, passed through to the children.
    pub hip_lib: String,
}

/// Every op as its own sub-graph, timed in its real position, with both ranks in lockstep.
///
/// `step-capture --per-op` does this on one card; the two-rank version needs one thing more. A
/// link that holds an all-reduce does not complete unless the other rank is inside the same
/// all-reduce, so every link is preceded by a rendezvous -- at capture time and before each
/// sample. Both ranks build the same plan from the same graph and shard it the same way, so link
/// `i` is the same op on both and the rendezvous names agree without exchanging them.
///
/// The instrument cost is measured, not assumed: the links' sum against the whole graph's wall
/// gives the per-link cost, and it is subtracted before any bandwidth is computed.
#[allow(clippy::too_many_arguments)]
fn per_op_links(
    hip: &HipThread,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    exec: &mut BlockExecutor,
    stream: &Stream,
    peer: &Arc<RankPeer>,
    bound: &[BoundStep],
    plan: &BlockPlan<'_>,
    inputs: &BlockInputs,
    capture_counters: &Arc<CaptureCounters>,
    args: &StepTp2Args,
    label: &str,
    rank: u32,
    whole_ms: f64,
    record_len: &BTreeMap<String, u64>,
    proof: GraphSafe,
) -> anyhow::Result<()> {
    // Agree on the count FIRST. Every rendezvous below is named by a link index, so two ranks
    // with different link counts would not fail -- they would wait for each other until the
    // 30-minute rank deadline, holding the GPU lock the whole time. This turns that into an
    // error before the first capture.
    peer.agree("per-op-links", bound.len() as u64)?;
    // `bind_steps` SKIPS a step whose route is not a kernel, and a caller may hand a SUBSET (the
    // reduce-ablated list below), so `bound[i]` is not `plan.steps[i]` and the two lengths need
    // not match. The mapping is therefore by NAME, which is unique in a plan -- an off-by-one
    // here would attribute one kernel's time to another op's geometry, silently, and only in
    // the numbers.
    let mut by_name: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, st) in plan.steps.iter().enumerate() {
        if matches!(st.route, saber_exec::Route::Kernel { .. })
            && by_name.insert(st.name(), i).is_some()
        {
            anyhow::bail!("two plan steps are called `{}`", st.name());
        }
    }
    let mut step_of: Vec<usize> = Vec::with_capacity(bound.len());
    for b in bound {
        let i = *by_name
            .get(b.name)
            .ok_or_else(|| anyhow::anyhow!("link `{}` is in no plan step", b.name))?;
        step_of.push(i);
    }
    let mut links: Vec<(usize, CapturedBlock)> = Vec::new();
    for i in 0..bound.len() {
        peer.rendezvous(&format!("link-cap-{i}"))?;
        let sub = CapturedBlock::capture_bound(
            ctx,
            launcher,
            &bound[i..=i],
            stream,
            format!("op.tp2.{label}.r{rank}.{i}"),
            proof,
            PeerCapture::Lockstep,
            capture_counters,
        )?;
        tp2::wait_bounded(ctx, stream, "an op link capture")?;
        links.push((i, sub));
    }
    let nodes: u32 = links.iter().map(|(_, s)| s.declared_launches()).sum();
    if rank == 0 {
        println!();
        println!(
            "per-op    {label}: {} link(s), {nodes} node(s) against the step's {} — {}",
            links.len(),
            plan.expected_launches(),
            if nodes == plan.expected_launches() {
                "MATCH"
            } else {
                "MISMATCH"
            }
        );
    }
    let start = hip.create_event(EventFlags::Default)?;
    let end = hip.create_event(EventFlags::Default)?;
    let mut per: Vec<Vec<f64>> = vec![Vec::with_capacity(args.iters); links.len()];
    for pass in 0..(args.warmup + args.iters) {
        for (j, (_, sub)) in links.iter().enumerate() {
            peer.rendezvous(&format!("link-{pass}-{j}"))?;
            hip.record_event(&start, stream)?;
            sub.replay(ctx, stream)?;
            hip.record_event(&end, stream)?;
            tp2::wait_bounded(ctx, stream, "an op link")?;
            if pass >= args.warmup {
                per[j].push(f64::from(hip.elapsed_ms(&start, &end)?) * 1000.0);
            }
        }
    }
    let raw: Vec<f64> = per
        .into_iter()
        .map(|mut v| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        })
        .collect();
    let sum_us: f64 = raw.iter().sum();
    // What one link costs that the whole graph does not: a graph launch and a host wait each.
    let overhead_us = ((sum_us / 1000.0 - whole_ms) / links.len() as f64 * 1000.0).max(0.0);
    let indices: Vec<usize> = links.iter().map(|(i, _)| step_of[*i]).collect();
    let per_us: Vec<f64> = raw.iter().map(|u| (u - overhead_us).max(0.0)).collect();

    // `--detail SUBSTR`: one line per matching link, with the ops either side of it and the same
    // op run EAGERLY in the same position.
    //
    // A median over 16 links of one family answers "how much", never "of what". F's own census of
    // `saber_attn_prefill` (one card, only `kv_heads` varied, GQA 6 held) has the kernel at 0.85-1.00x
    // per unit of work and its absolute time HALVING with the head count -- which is the opposite
    // of what this command's TP=2 table said, so the difference is not in the kernel and the
    // instrument has to be able to say where it is. Eager-vs-graph separates "the op is slow" from
    // "the link is", and the neighbours say what a link might be waiting for.
    if let Some(want) = args.detail.as_deref() {
        let mut rows: Vec<(usize, String, String, String, f64, f64)> = Vec::new();
        for (j, (b, _)) in links.iter().enumerate() {
            let step = &plan.steps[step_of[*b]];
            if !step.name().contains(want) {
                continue;
            }
            let prev = b.checked_sub(1).map_or_else(
                || String::from("-"),
                |k| plan.steps[step_of[k]].name().to_owned(),
            );
            let next = bound
                .get(b + 1)
                .map_or_else(|| String::from("-"), |n| n.name.to_owned());
            // The same op, in the same position, with no graph around it.
            let mut eager: Vec<f64> = Vec::with_capacity(args.iters);
            for pass in 0..(args.warmup + args.iters) {
                peer.rendezvous(&format!("eager-{pass}-{j}"))?;
                hip.record_event(&start, stream)?;
                exec.run_steps(
                    ctx,
                    launcher,
                    stream,
                    &plan.steps[step_of[*b]..=step_of[*b]],
                    inputs,
                    false,
                )?;
                hip.record_event(&end, stream)?;
                tp2::wait_bounded(ctx, stream, "an eager op link")?;
                if pass >= args.warmup {
                    eager.push(f64::from(hip.elapsed_ms(&start, &end)?) * 1000.0);
                }
            }
            eager.sort_by(f64::total_cmp);
            rows.push((
                *b,
                prev,
                step.name().to_owned(),
                next,
                raw[j],
                eager[eager.len() / 2],
            ));
        }
        if rank == 0 && !rows.is_empty() {
            println!();
            println!(
                "detail    {label}: every link whose op contains `{want}`, with the ops either \
                 side and the same op run eagerly in place"
            );
            println!(
                "          {:>4} {:<34} {:<34} {:<34} {:>10} {:>10}",
                "link", "previous op", "op", "next op", "graph µs", "eager µs"
            );
            for (b, prev, op, next, g, e) in &rows {
                println!("          {b:>4} {prev:<34} {op:<34} {next:<34} {g:>10.1} {e:>10.1}");
            }
        }
    }
    // The layer linears, so the non-linear table can say what fraction of the step it covers.
    let linears_ms: f64 = indices
        .iter()
        .zip(&per_us)
        .filter(|(i, _)| crate::step_capture::family(&plan.steps[**i]) == "GEMMs")
        .map(|(_, us)| us / 1000.0)
        .sum();
    if rank == 0 {
        println!(
            "          {} link(s) sum to {:.3} ms against the graph's {whole_ms:.3}: {:.2} µs \
             per link of instrument, removed below",
            links.len(),
            sum_us / 1000.0,
            overhead_us
        );
        println!();
        println!(
            "others    {label}: every op that is NOT a layer linear, in chain, per card, against \
             the floor its own geometry implies"
        );
        let rows = crate::step_capture::other_rows(plan, &indices, &per_us, record_len);
        crate::step_capture::print_others(
            &rows,
            args.dram_gbs,
            args.peak_tflops,
            linears_ms,
            whole_ms,
        );
        crate::step_capture::print_family_summary(plan, &indices, &per_us, whole_ms, label);
        // Per GEMM shape at THIS rank's shard, with TFLOP/s: the prefill GEMM question (ws-h34)
        // is asked of the TP=2 shapes the engine runs, not of TP=1's.
        let _ = crate::step_capture::print_gemm_shapes(
            plan,
            &indices,
            &per_us,
            record_len,
            args.dram_gbs,
            args.peak_tflops,
            label,
        );
    }
    Ok(())
}

/// Run it. The parent spawns two ranks; a rank runs the step.
pub fn run(hip: Option<&HipThread>, args: &StepTp2Args) -> anyhow::Result<usize> {
    if args.rank.is_none() && args.port.is_some() {
        anyhow::bail!(
            "--port without --rank-of: this process cannot tell whether it is the parent or a \
             rank, and a parent that guessed would spawn two more children (ws-h10)."
        );
    }
    match (args.rank, hip) {
        (Some(rank), Some(hip)) => rank_main(hip, rank, args),
        (Some(_), None) => anyhow::bail!("a rank process needs a HIP runtime"),
        (None, _) => spawn_ranks(args),
    }
}

impl StepTp2Args {
    /// This command's argv for one rank, **derived from this struct** rather than hand-listed.
    ///
    /// THE RANKS DO THE WORK. A flag the parent parses and does not forward is a flag that does
    /// nothing, silently, because the parent's own banner is printed by rank 0 and looks right.
    /// That has happened three times in this tree: `--prefill` on this command, the four gate
    /// flags on `forward --tp 2` (ws-h23), and `--head-on-every-row` on this one, which reported
    /// the wrong shape for two whole measurements before the arms were compared.
    ///
    /// The `let Self { .. }` below names **every field with no `..`**, so adding one to
    /// `StepTp2Args` stops this function compiling until the new field is either forwarded or
    /// named here as parent-only. That is the difference between a list someone has to remember
    /// and a list the compiler keeps.
    pub(crate) fn child_argv(&self, rank_of: u32, port: u16) -> Vec<OsString> {
        let Self {
            pack,
            prompt_ids,
            prompt_index,
            prebuilt,
            rows,
            ragged_rows,
            ragged_widths,
            spec_width,
            gdn_state_f16,
            prefill,
            ctx,
            budget_mib,
            replays,
            iters,
            warmup,
            dram_gbs,
            peak_tflops,
            per_op,
            ar_overlap,
            ar_push_blocks,
            detail,
            small_chunk_gemm,
            small_chunk_class,
            mid_chunk_gemm,
            mid_chunk_atiled,
            skinny_gate,
            no_slot_reuse,
            fusion,
            fusion_control,
            head_on_every_row,
            hip_lib,
            // PARENT-ONLY, and not an oversight: the child is told which rank it is and where to
            // meet by THIS function's arguments. Forwarding the parent's own `None`s would leave
            // the child deciding it is a parent and spawning two more of itself.
            rank: _,
            port: _,
        } = self;

        let mut a: Vec<OsString> = Vec::new();
        // A free function rather than a closure: a closure capturing `a` mutably cannot coexist
        // with the direct `a.push` calls for the path arguments.
        fn flag(a: &mut Vec<OsString>, name: &str, value: &dyn std::fmt::Display) {
            a.push(OsString::from(name));
            a.push(OsString::from(value.to_string()));
        }
        a.push(OsString::from("step-tp2"));
        flag(&mut a, "--rank-of", &rank_of);
        flag(&mut a, "--port", &port);
        a.push(OsString::from("--pack"));
        a.push(pack.clone().into_os_string());
        a.push(OsString::from("--prompt-ids"));
        a.push(prompt_ids.clone().into_os_string());
        flag(&mut a, "--prompt-index", prompt_index);
        a.push(OsString::from("--prebuilt"));
        a.push(prebuilt.clone().into_os_string());
        flag(&mut a, "--ctx", ctx);
        flag(&mut a, "--spec-width", spec_width);
        // By value (N8 made f16 the default): the child never inherits its parser's default.
        flag(
            &mut a,
            "--gdn-state-dtype",
            &if *gdn_state_f16 { "f16" } else { "f32" },
        );
        flag(&mut a, "--budget-mib", budget_mib);
        flag(&mut a, "--replays", replays);
        flag(&mut a, "--iters", iters);
        flag(&mut a, "--warmup", warmup);
        flag(&mut a, "--dram-gbs", dram_gbs);
        flag(&mut a, "--peak-tflops", peak_tflops);
        a.push(OsString::from("--hip-lib"));
        a.push(OsString::from(hip_lib));
        if *per_op {
            a.push(OsString::from("--per-op"));
        }
        if *ar_overlap > 0 {
            flag(&mut a, "--ar-overlap", ar_overlap);
        }
        if *ar_push_blocks > 0 {
            a.push(OsString::from("--ar-form"));
            a.push(OsString::from("split"));
            flag(&mut a, "--ar-push-blocks", ar_push_blocks);
        }
        // `--detail` only says anything under `--per-op`, but it is forwarded on its own terms:
        // the child is the process that runs the links, so a parent that kept the filter would
        // print the whole per-op table and none of the detail the caller asked for.
        if let Some(d) = detail {
            a.push(OsString::from("--detail"));
            a.push(OsString::from(d));
        }
        if *no_slot_reuse {
            a.push(OsString::from("--no-slot-reuse"));
        }
        // C6: both ranks plan the same small-chunk rungs, or the reduces' digests disagree.
        if *small_chunk_gemm > 0 {
            flag(&mut a, "--small-chunk-gemm", small_chunk_gemm);
            flag(&mut a, "--small-chunk-class", small_chunk_class);
        }
        // G5: both ranks plan the same folded tile heights.
        if *mid_chunk_gemm {
            a.push(OsString::from("--mid-chunk-gemm"));
        }
        // G6: the same layout law and the same gate body on both ranks.
        if *mid_chunk_atiled {
            a.push(OsString::from("--mid-chunk-atiled"));
        }
        if *skinny_gate {
            a.push(OsString::from("--skinny-gate"));
        }
        a.push(OsString::from("--fusion"));
        a.push(OsString::from(fusion));
        a.push(OsString::from("--fusion-control"));
        a.push(OsString::from(fusion_control));
        if *head_on_every_row {
            a.push(OsString::from("--head-on-every-row"));
        }
        for r in rows {
            flag(&mut a, "--rows", r);
        }
        // The BANDS (C2) and their width multiset: both ranks plan the same band or the reduces'
        // digests disagree before a launch.
        for r in ragged_rows {
            flag(&mut a, "--ragged-rows", r);
        }
        if !ragged_rows.is_empty() {
            for w in ragged_widths {
                flag(&mut a, "--ragged-widths", w);
            }
        }
        // The prefill buckets have to reach the child too, or the parent asks for them and the
        // ranks run the decode default (which is how the first TP=2 prefill run measured nothing).
        for t in prefill {
            flag(&mut a, "--prefill", t);
        }
        a
    }
}

/// The parent: a free port, two children, a deadline.
fn spawn_ranks(args: &StepTp2Args) -> anyhow::Result<usize> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|e| anyhow::anyhow!("binding the rendezvous socket: {e}"))?;
    let port = listener.local_addr()?.port();
    drop(listener);
    println!("step-tp2  TP=2 decode step: two processes, one card each, on 127.0.0.1:{port}");
    println!(
        "          pack {} (rank0 on device 0, rank1 on device 1)",
        args.pack.display()
    );
    println!();
    let exe = std::env::current_exe()?;
    let mut children = Vec::new();
    for rank in [0u32, 1] {
        let mut cmd = std::process::Command::new(&exe);
        cmd.args(args.child_argv(rank, port));
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

/// One rank: load this rank's half of the model, then run every bucket.
fn rank_main(hip: &HipThread, rank: u32, args: &StepTp2Args) -> anyhow::Result<usize> {
    let port = args
        .port
        .ok_or_else(|| anyhow::anyhow!("--rank-of needs --port"))?;
    hip.set_device(rank as i32)?;
    crate::gpu_idle::require_default(hip, rank as i32, 0)?;
    let exchange: TcpExchange = if rank == 0 {
        let listener = std::net::TcpListener::bind(("127.0.0.1", port))
            .map_err(|e| anyhow::anyhow!("rank 0: binding port {port}: {e}"))?;
        TcpExchange::accept(&listener, Duration::from_secs(300))?
    } else {
        TcpExchange::connect(port, Duration::from_secs(300))?
    };
    let peer = Arc::new(RankPeer::new(Arc::new(exchange)));

    let graph = Graph::qwen38_27b_at_width(ModelConfig::qwen38_27b(), 2, args.spec_width)?;
    let graph = if args.gdn_state_f16 {
        graph.with_gdn_state_f16()?
    } else {
        graph
    };
    let mxfp4 = LaunchRules::prod_mxfp4()?;
    let saber = SaberRules::prod_saber()?;
    let head_rules = HeadRules::prod_head()?;
    let ids = read_prompt_ids(&args.prompt_ids, args.prompt_index)?;
    let layers: Vec<u32> = (0..graph.config.num_hidden_layers).collect();

    let load_counters = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &load_counters)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;

    // ---- this rank's weights -------------------------------------------------------------------
    let probe = plan_for(
        &graph,
        &layers,
        // Whichever bucket was asked for: the op SET differs between the phases, the weight
        // records do not, and this probe exists to learn the records.
        *args.rows.first().or(args.prefill.first()).unwrap_or(&5),
        args.ctx,
        rank,
        if args.rows.is_empty() {
            Phase::Prefill
        } else {
            Phase::Decode
        },
        &mxfp4,
        &saber,
        true,
        &args.fusion,
        args.no_slot_reuse,
        // COMBINED, whichever shape the run uses: this names every weight the step binds, the
        // head's included. Asking the body alone would leave `lm_head.weight` out of the upload
        // and the head would then bind nothing.
        true,
        // A rectangle: a band binds the same weight records (C2).
        None,
    )?;
    let mut want = probe.weight_keys();
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
    let record_len: BTreeMap<String, u64> = reader
        .records()
        .iter()
        .enumerate()
        .map(|(i, r)| (reader.name(i).to_owned(), r.length))
        .collect();
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
            "model     Qwen3.8-27B at TP=2: {} layers, hidden {}, ctx band {}",
            graph.config.num_hidden_layers, graph.config.hidden_size, args.ctx
        );
        println!(
            "weights   {} records per rank, {} MiB, {:.2} GB/s verified",
            weights.len(),
            span >> 20,
            stats.end_to_end_gbps()
        );
    }

    let mut failures = 0usize;
    let buckets: Vec<(u32, Phase, bool)> = args
        .rows
        .iter()
        .map(|r| (*r, Phase::Decode, false))
        .chain(args.prefill.iter().map(|t| (*t, Phase::Prefill, false)))
        .chain(args.ragged_rows.iter().map(|r| (*r, Phase::Decode, true)))
        .collect();
    for (rows, phase, ragged) in buckets {
        failures += one_bucket(
            hip,
            &ctx,
            &stream,
            &peer,
            rank,
            args,
            &graph,
            &layers,
            &mxfp4,
            &saber,
            &head_rules,
            &weights,
            &record_len,
            rows,
            phase,
            ragged.then_some(args.ragged_widths.as_slice()),
            &ids,
            budget,
        )?;
    }
    drop(sink);
    Ok(failures)
}

/// Fill every arena buffer an op that a fusion FOLDED would have written with 0xFF (ws-h37).
///
/// Every `Route::Fused` step, whatever it is fused into: the `_nq` norm (into its reduce), the
/// pair's second GEMM, and both ops row 3 folds into a decode GEMM (ws-h38) -- the reduce, whose
/// output edge is the GEMM's `c` (dead in the fused arm, excluded by the gate), and the norm's
/// `q_out`, `scale_out` and `res_out`, which the GEMM's row owners must write.
fn poison_folded_outputs(
    hip: &HipThread,
    exec: &mut BlockExecutor,
    plan: &saber_exec::BlockPlan<'_>,
    stream: &Stream,
) -> anyhow::Result<u64> {
    let mut total = 0u64;
    for step in &plan.steps {
        // ws-h38: a step whose OWN dispatch is a fused kernel (the rope + append, the
        // last-arriver attention) must newly write its op's outputs too -- the attention's
        // combine output above all, which the unfused arm's second launch wrote.
        let fused_self = match &step.route {
            saber_exec::Route::Kernel { descs, .. } => {
                matches!(descs[0].op.as_str(), "qk_rope_append" | "attn_decode_declc")
            }
            _ => false,
        };
        if !fused_self && !matches!(step.route, saber_exec::Route::Fused { .. }) {
            continue;
        }
        let private = [format!("{}.k", step.op.name)];
        let extra = if fused_self && step.op.kernel == saber_model::graph::KernelOp::QkNormRope {
            &private[..]
        } else {
            &[]
        };
        for name in extra {
            if let Some(&bi) = plan.index.get(name) {
                let b = &plan.buffers[bi];
                exec.upload(
                    hip,
                    b.offset,
                    &vec![0xFFu8; usize::try_from(b.bytes)?],
                    stream,
                )?;
                total += b.bytes;
            }
        }
        for e in &step.op.outputs {
            for name in [e.name.clone(), format!("{}.scale", e.name)] {
                let Some(&bi) = plan.index.get(&name) else {
                    continue;
                };
                let b = &plan.buffers[bi];
                exec.upload(
                    hip,
                    b.offset,
                    &vec![0xFFu8; usize::try_from(b.bytes)?],
                    stream,
                )?;
                total += b.bytes;
            }
        }
    }
    Ok(total)
}

/// The fusion gate's NEGATIVE CONTROLS (ws-h37): a fused plan made WRONG in the one way a real
/// defect of each fusion would be, so the gate is shown able to read MISMATCH.
///
/// * `pair-ks` — every decode pair runs its second half at the FIRST half's split count: the
///   same kernel, the same buffers, a different partition of K and therefore a different fp32 sum
///   in `in_proj_ba` (ks4 -> ks2 at `b1s8`), which is exactly what a pair that did not carry each
///   half's own descriptor would compute.
/// * `nq-verbatim` — every `_nq` epilogue binds the reference's verbatim `_nq` (the contract-off sum of
///   squares) instead of `_ssqfma`, which is ws-h20's one e4m3 code in ~3.3M: a WEAK control (it
///   may read 0 at one step), reported as such. `pair-ks` is weak for the same reason: a split's
///   fp32 difference rarely survives the bf16 store.
/// * `pair-drop` — the pair's second half is never issued (`nb_b = 0`): DECISIVE only because the
///   folded outputs are poisoned before the fused arm ([`poison_folded_outputs`]).
/// * `nq-eps` — the fused epilogue's epsilon times ten: a different norm, DECISIVE.
/// * `gemmar-eps` (ws-h38) — row 3's epilogue (inside the decode GEMM) at ten times its epsilon:
///   a different norm in every row, DECISIVE.
/// * `smq-rows` (ws-h40) — row 4's GEMM told `ep_rows = M - 1`: the last row's `silu_mul_quant`
///   never runs, so its `q` / `scale` keep the poison, DECISIVE.
/// * `gemmar-rows` (ws-h38) — row 3's GEMM told `ep_rows = M - 1`: the last row gets no owner,
///   so its `q_out` / `scale_out` / `res_out` keep the poison, DECISIVE (and it proves the
///   gate reads the LAST row, where an owner-election off-by-one would land).
pub(crate) fn fusion_control(
    plan: &mut saber_exec::BlockPlan<'_>,
    control: &str,
) -> anyhow::Result<()> {
    use saber_kernels::rules::ArgValue;
    match control {
        "none" => return Ok(()),
        "pair-ks" | "nq-verbatim" | "pair-drop" | "nq-eps" | "gemmar-eps" | "gemmar-rows"
        | "rope-append-drop" | "combine-drop" | "smq-rows" | "fnq-rows" => {}
        other => anyhow::bail!(
            "--fusion-control {other}: none, pair-drop, nq-eps, gemmar-eps, gemmar-rows, pair-ks, \
             nq-verbatim, rope-append-drop, combine-drop, smq-rows or fnq-rows"
        ),
    }
    let mut touched = 0usize;
    for step in &mut plan.steps {
        let saber_exec::Route::Kernel { descs, .. } = &mut step.route else {
            continue;
        };
        let d = &mut descs[0];
        // ws-h38: the two row-6/7 controls swap the fused symbol for its rules-named control.
        if let Some(ctl) = h38_control_symbol(control, d)? {
            d.kernel = ctl;
            touched += 1;
        }
        if control == "nq-verbatim" && d.op == "allreduce_exact_nq" {
            d.kernel = "saber_ar_oneshot_2rank_exact_nq_bf16".to_owned();
            touched += 1;
        }
        if control == "nq-eps" && d.op == "allreduce_exact_nq" {
            for s in &mut d.args {
                if let ArgValue::F32 { fval } = &mut s.value {
                    *fval *= 10.0;
                }
            }
            touched += 1;
        }
        if control == "gemmar-eps" && d.op == "gemm_decode_ar_nq" {
            for s in &mut d.args {
                if let ArgValue::F32 { fval } = &mut s.value {
                    *fval *= 10.0;
                }
            }
            touched += 1;
        }
        if control == "fnq-rows" && d.op == "gdn_fnq" {
            // ws-h42, row 5: the norm's last row gets no owner, so its `q` / `scale` keep the
            // poison.
            let m = d
                .args
                .iter()
                .find_map(|s| match (&s.value, s.name.as_str()) {
                    (ArgValue::I32 { ival }, "ep_rows") => Some(*ival),
                    _ => None,
                })
                .ok_or_else(|| {
                    anyhow::anyhow!("{}: a row-5 descriptor without ep_rows", d.kernel)
                })?;
            anyhow::ensure!(m > 1, "{}: fnq-rows needs M > 1", d.kernel);
            for s in &mut d.args {
                if s.name == "ep_rows" {
                    s.value = ArgValue::I32 { ival: m - 1 };
                }
            }
            touched += 1;
        }
        if control == "smq-rows" && d.op == "gemm_decode_smq" {
            // ws-h40, row 4: the last row gets no owner, so its `q` / `scale` keep the poison.
            let m = d
                .args
                .iter()
                .find_map(|s| match (&s.value, s.name.as_str()) {
                    (ArgValue::I32 { ival }, "m") => Some(*ival),
                    _ => None,
                })
                .ok_or_else(|| anyhow::anyhow!("{}: a row-4 descriptor without m", d.kernel))?;
            anyhow::ensure!(m > 1, "{}: smq-rows needs M > 1", d.kernel);
            for s in &mut d.args {
                if s.name == "ep_rows" {
                    s.value = ArgValue::I32 { ival: m - 1 };
                }
            }
            touched += 1;
        }
        if control == "gemmar-rows" && d.op == "gemm_decode_ar_nq" {
            let m = d
                .args
                .iter()
                .find_map(|s| match (&s.value, s.name.as_str()) {
                    (ArgValue::I32 { ival }, "m") => Some(*ival),
                    _ => None,
                })
                .ok_or_else(|| anyhow::anyhow!("{}: a row-3 descriptor without m", d.kernel))?;
            anyhow::ensure!(m > 1, "{}: gemmar-rows needs M > 1", d.kernel);
            for s in &mut d.args {
                if s.name == "ep_rows" {
                    s.value = ArgValue::I32 { ival: m - 1 };
                }
            }
            touched += 1;
        }
        if control == "pair-drop" && d.op == "gemm_decode_pair" {
            // The second half's n-block count to zero: its blocks are never issued, so `in_proj_ba`
            // is never written. Without the poison below this arm reads BIT-EXACT (the unfused
            // arm's bytes are still in the buffer) -- which is the hole the poison closes.
            let nb = d
                .args
                .iter()
                .find_map(|s| match (&s.value, s.name.as_str()) {
                    (ArgValue::I32 { ival }, "nb") => Some(*ival),
                    _ => None,
                });
            let ks = d
                .args
                .iter()
                .find_map(|s| match (&s.value, s.name.as_str()) {
                    (ArgValue::I32 { ival }, "ks") => Some(*ival),
                    _ => None,
                });
            let (Some(nb), Some(ks)) = (nb, ks) else {
                anyhow::bail!("{}: a pair descriptor without nb / ks", d.kernel);
            };
            for s in &mut d.args {
                if s.name == "nb_b" {
                    s.value = ArgValue::I32 { ival: 0 };
                }
            }
            d.grid[0] = u32::try_from(nb * ks)?;
            touched += 1;
        }
        if control == "pair-ks" && d.op == "gemm_decode_pair" {
            let get = |d: &saber_kernels::rules::LaunchDescriptor, n: &str| {
                d.args.iter().find_map(|s| match (&s.value, s.name == n) {
                    (ArgValue::I32 { ival }, true) => Some(*ival),
                    _ => None,
                })
            };
            let (Some(ks), Some(nb), Some(nb_b)) = (get(d, "ks"), get(d, "nb"), get(d, "nb_b"))
            else {
                anyhow::bail!("{}: a pair descriptor without ks / nb / nb_b", d.kernel);
            };
            for s in &mut d.args {
                if s.name == "ks_b" {
                    s.value = ArgValue::I32 { ival: ks };
                }
            }
            d.grid[0] = u32::try_from(nb * ks + nb_b * ks)?;
            touched += 1;
        }
    }
    anyhow::ensure!(
        touched > 0,
        "--fusion-control {control}: the plan has nothing to perturb"
    );
    println!(
        "control   {control}: {touched} fused dispatch(es) perturbed — this arm MUST read MISMATCH"
    );
    Ok(())
}

/// ws-h38's two DECISIVE controls: the fused descriptor's symbol swapped for the control the rules
/// name beside it -- same grid, same kernarg, same buffers, one defect each.
///
/// * `rope-append-drop` -- `qk_rope_append_*` never writes the V half of the pages it appends
///   (`_dropv`). The gate poisons the step's KV slots before the fused arm
///   ([`poison_appended_slots`]), so the V bytes stay 0xFF and the STATE comparison must differ.
/// * `combine-drop` -- `saber_attn_declc_*`'s last arriver merges every segment but the last
///   (`_drop1`): a different softmax over a context of more than one segment, in every attention
///   output the step writes (the output is poisoned too, [`poison_folded_outputs`]).
fn h38_control_symbol(
    control: &str,
    d: &saber_kernels::rules::LaunchDescriptor,
) -> anyhow::Result<Option<String>> {
    match (control, d.op.as_str()) {
        ("rope-append-drop", "qk_rope_append") => {
            let rules = saber_kernels::glue::GlueRules::prod()?;
            let row = rules
                .rope_append
                .iter()
                .find(|r| r.interleaved == d.kernel)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "{}: no rope_append row carries a control for this symbol (only the \
                         interleaved layout has one)",
                        d.kernel
                    )
                })?;
            Ok(Some(row.control.clone()))
        }
        ("combine-drop", "attn_decode_declc") => {
            let rules = SaberRules::prod_saber()?;
            let row = rules
                .attn_decode_declc
                .iter()
                .find(|r| r.symbol == d.kernel)
                .ok_or_else(|| anyhow::anyhow!("{}: no attn_decode_declc row", d.kernel))?;
            Ok(Some(row.control.clone()))
        }
        _ => Ok(None),
    }
}

/// Fill the KV-cache bytes every FOLDED `kv_append` writes this step with 0xFF (ws-h38).
///
/// The pages are STATE, reset to zero between the gate's arms, and the append writes exactly the
/// `(slot, kv head)` rows its slot table names -- K then V, `2 * head_dim` bytes at
/// `blk * block_stride + off * page_stride + h * head_stride`. Everything else in the cache (the
/// resident context the attention reads) is left as it was: poisoning it would change what the
/// step computes, not what it must write. A fused kernel that skipped any of those rows leaves
/// 0xFF (NaN in e4m3) where the unfused arm wrote a code, which the state comparison reads.
#[allow(clippy::too_many_arguments)]
fn poison_appended_slots(
    hip: &HipThread,
    plan: &saber_exec::BlockPlan<'_>,
    owned: &[DevicePtr],
    state_ptrs: &[(String, usize, u64)],
    slot_table: &dyn Fn(&str) -> anyhow::Result<Vec<u8>>,
    stream: &Stream,
) -> anyhow::Result<u64> {
    let mut total = 0u64;
    for step in &plan.steps {
        if step.op.kernel != saber_model::graph::KernelOp::KvAppend
            || !matches!(step.route, saber_exec::Route::Fused { .. })
        {
            continue;
        }
        let Some((_, saber_kernels::glue::Shape::Kv(kv))) =
            saber_model::graph::glue_shape(step.op, plan.bucket.m())
        else {
            anyhow::bail!("{}: no kv_append shape", step.op.name);
        };
        let Some(saber_exec::Source::State { key, .. }) = step.source("kv") else {
            anyhow::bail!("{}: the folded append binds no KV state", step.op.name);
        };
        let Some(saber_exec::Source::Index(slots)) = step.source("slot") else {
            anyhow::bail!("{}: the folded append binds no slot table", step.op.name);
        };
        let Some((_, i, bytes)) = state_ptrs.iter().find(|(n, _, _)| n == key) else {
            anyhow::bail!("{}: state `{key}` was not allocated", step.op.name);
        };
        let row = vec![0xFFu8; usize::try_from(2 * kv.head_dim)?];
        for s in slot_table(slots)?.chunks_exact(8) {
            let s = i64::from_le_bytes(s.try_into()?);
            if s < 0 {
                continue;
            }
            let bs = i64::from(kv.block_size);
            let (blk, off) = (s / bs, s % bs);
            for h in 0..i64::from(kv.nkv) {
                let at = u64::try_from(
                    blk * kv.block_stride + off * kv.page_stride + h * kv.head_stride,
                )?;
                anyhow::ensure!(
                    at + row.len() as u64 <= *bytes,
                    "{}: slot {s} head {h} lies past the {bytes}-byte cache",
                    step.op.name
                );
                hip.memcpy_h2d_at_blocking(&owned[*i], at, &row, stream)?;
                total += row.len() as u64;
            }
        }
    }
    Ok(total)
}

/// The step's plan at `rows`: the LAYERS, one arena. The embedding and the head are resolved by
/// hand because `BlockPlan` refuses both above TP=1 (ws-h10 §"API additions").
#[allow(clippy::too_many_arguments)]
fn plan_for<'g>(
    graph: &'g Graph,
    layers: &[u32],
    rows: u32,
    ctx: u32,
    rank: u32,
    phase: Phase,
    mxfp4: &LaunchRules,
    saber: &SaberRules,
    fuse: bool,
    fusion: &str,
    no_reuse: bool,
    // `combined`: every site in one plan (the pre-N6 shape), or just the body.
    combined: bool,
    // A decode ROW BAND (C2): this many slots at the graph's max width, `rows` packed.
    band_batch: Option<u32>,
) -> anyhow::Result<BlockPlan<'g>> {
    let bucket = match band_batch {
        Some(batch) => saber_model::graph::Bucket::DecodeBand {
            batch,
            verify_width: graph.verify_width,
            rows,
        },
        None => graph.bucket_for(rows, phase),
    };
    let q = bucket.q_len().max(1);
    let batch = bucket.num_seqs();
    let mut plan = BlockPlan::with_options(
        graph,
        bucket,
        layers,
        RuntimeParams {
            max_ctx: ctx,
            // Slot 0 is the kernels' null slot and each sequence owns `q` consecutive snapshot
            // slots from the first real one, so a batch needs `1 + batch * q`.
            gdn_slots: crate::step_capture::gdn_slot_count(phase, batch, q),
        },
        mxfp4,
        saber,
        // Everything, at this rank, with the reduces REAL: the embedding's vocabulary window is the
        // rank's (ws-h16 ask 1), `logits.gather` is Supplied so the head plans at all (ask 2), and
        // `layer.allreduce` resolves to the family the rules' message-size threshold picks (ask 3).
        // This is what deleted the splice this command used to carry.
        {
            let mut o = if combined {
                // The pre-N6 shape: every site in one plan, the head on every row. `PlanOptions`
                // refuses that at a prefill chunk unless a caller says so, which is what makes it
                // a stated choice rather than an accident of which constant was reached for.
                PlanOptions::FORWARD.with_head_on_every_row()
            } else {
                // The engine's shape: the body here, the head in `plan_head` at the SAMPLED rows.
                PlanOptions::BODY
            }
            .rank(rank)
            .with_collectives(Collectives::Real);
            o.reuse_layer_slots = !no_reuse;
            // ws-h37: `o` alone was the UNFUSED plan here since ws-h19 made `fuse` default off,
            // so this gate's two arms were one plan and it read "0 of 129 reduces" at every
            // bucket. The fused arm now asks for its fusions by name.
            if fuse {
                fusion_options(o, fusion)?
            } else {
                o.without_fusion()
            }
        },
    )?;
    // The chunk's live rows, stated as the engine states them (ws-h27). Both ranks feed every row
    // of the bucket here, so the law is the identity and both ranks' descriptors are the ones this
    // command captured before — and a rank that stated it differently from its peer would answer a
    // different half of every reduce, which is why it is stated at all.
    if bucket.phase() == Phase::Prefill {
        plan.retarget_prefill_rows(q, saber)?;
        // AND THE RECURRENT STATE AT THE FIRST REAL SLOT, on both ranks (ws-h30).
        plan.retarget_prefill_state(saber_exec::forward::FIRST_STATE_SLOT)?;
    }
    Ok(plan)
}

/// The fused arm's fusion set, by name (`--fusion`). An unknown name is refused rather than
/// read as "everything": a typo would otherwise gate a different plan than the one asked for.
///
/// * `both` / `all` — [`PlanOptions::with_decode_fusion`], what `--decode-fusion` serves: the
///   `_nq` epilogue, the decode pair and fusion-map row 3 (ws-h38);
/// * `h37` — the `_nq` epilogue and the decode pair, ws-h37's set (row 3 off);
/// * `nq`, `pair` — one ws-h37 fusion alone;
/// * `gemm-ar` — row 3 with the plain `_nq` epilogue (which then only fires at the embedding's
///   reduce, the one exact reduce with no GEMM before it), the pair off.
/// * `gemm-smq` — row 4 alone (ws-h40); `h40` — `all` plus row 4.
/// * `gdn-nq` — row 5 alone (ws-h42); `h42` — `h40` plus row 5.
pub(crate) fn fusion_options(o: PlanOptions, fusion: &str) -> anyhow::Result<PlanOptions> {
    Ok(match fusion {
        "both" | "all" => o.with_decode_fusion(),
        "h37" => o.with_h37_fusion(),
        "nq" => o.with_fusion(),
        "pair" => o.with_gemm_pair(),
        "gemm-ar" => o.with_fusion().with_gemm_ar(),
        // ws-h38: fusion-map rows 7 and 6, each alone.
        "rope-append" => o.with_rope_append(),
        "attn-combine" => o.with_attn_combine(),
        // ws-h40: fusion-map row 4 alone, and the `h40` set (`all` + row 4).
        "gemm-smq" => o.with_gemm_smq(),
        "h40" => o.with_h40_fusion(),
        // ws-h42: fusion-map row 5 alone, and the `h42` set (`h40` + row 5).
        "gdn-nq" => o.with_gdn_nq(),
        "h42" => o.with_h42_fusion(),
        other => anyhow::bail!(
            "--fusion {other}: both, all, h37, h40, h42, nq, pair, gemm-ar, gemm-smq, gdn-nq, \
             rope-append or attn-combine"
        ),
    })
}

/// A ROW BAND's index table (C2): `widths.len()` sequences, sequence `n` feeding `widths[n]` of
/// the band's `rows` rows (packed by `cu`, the tail padded), each owning `q` (the max width)
/// consecutive state slots and its own pages, at the same resident context the rectangle's
/// tables use — the shared law with `band_width = Some(q)`.
fn band_index_table(
    key: &str,
    widths: &[u32],
    q: u32,
    rows: u32,
    kv_blocks_per_seq: u32,
    ctx: u32,
    ids: &[i32],
) -> anyhow::Result<Vec<u8>> {
    use saber_exec::forward::{SeqFacts, StepIndex, FIRST_STATE_SLOT};
    let batch = widths.len() as u32;
    let cycled: Vec<i32> = ids.iter().copied().cycle().take(rows as usize).collect();
    let pages: Vec<Vec<u32>> = (0..batch)
        .map(|n| (n * kv_blocks_per_seq..(n + 1) * kv_blocks_per_seq).collect())
        .collect();
    let seqs: Vec<SeqFacts<'_>> = widths
        .iter()
        .enumerate()
        .map(|(n, w)| SeqFacts {
            state_slot: FIRST_STATE_SLOT + n as u32 * q,
            done_before: ctx.saturating_sub(q),
            kv_blocks: &pages[n],
            live_rows: *w,
            rope: None,
        })
        .collect();
    let step = StepIndex {
        phase: Phase::Decode,
        rows,
        max_blocks: kv_blocks_per_seq,
        seqs: &seqs,
        ids: &cycled,
        band_width: Some(q),
    };
    step.check().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(saber_exec::forward::index_table(key, &step)?)
}

/// This rank's HEAD: the final norm, `lm_head` and the vocabulary gather, at the SAMPLED rows.
///
/// One per bucket, shared by both fusion arms: `_nq` fuses an all-reduce with its epilogue and
/// every all-reduce in this model is in a LAYER, so the head is identical either way.
///
/// At TP>1 `logits.gather` is `Supplied` -- this rank holds a vocabulary shard, and half a
/// vocabulary's winner is not a token. The argmax still runs, per rank, into this rank's own pick;
/// combining them is the caller's job and `saber forward --tp 2` is where that is done.
fn plan_head_tp2<'g>(
    graph: &'g Graph,
    body: &BlockPlan<'g>,
    ctx: u32,
    rank: u32,
    phase: Phase,
    mxfp4: &LaunchRules,
    saber: &SaberRules,
) -> anyhow::Result<BlockPlan<'g>> {
    let bucket = body.bucket;
    let q = bucket.q_len().max(1);
    Ok(BlockPlan::with_options(
        graph,
        graph.head_bucket(bucket),
        &[],
        RuntimeParams {
            max_ctx: ctx,
            gdn_slots: crate::step_capture::gdn_slot_count(phase, bucket.num_seqs(), q),
        },
        mxfp4,
        saber,
        PlanOptions::HEAD
            .rank(rank)
            .with_collectives(Collectives::Real),
    )?)
}

/// What one rank's step leaves behind, for the byte comparison: the arena — which now holds the
/// head's logits too, because the head is in the plan — and every state buffer.
type Snapshot = (Vec<u8>, Vec<Vec<u8>>);

/// One bucket on one rank: plan, seed, run eagerly, capture with the reduces inside, replay, time.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn one_bucket(
    hip: &HipThread,
    ctx: &LaunchCtx<'_>,
    stream: &Stream,
    peer: &Arc<RankPeer>,
    rank: u32,
    args: &StepTp2Args,
    graph: &Graph,
    layers: &[u32],
    mxfp4: &LaunchRules,
    saber: &SaberRules,
    head_rules: &HeadRules,
    weights: &BTreeMap<String, BoundWeight>,
    record_len: &BTreeMap<String, u64>,
    rows: u32,
    phase: Phase,
    // A decode ROW BAND's per-slot widths (C2), `None` for a rectangle or a chunk.
    band: Option<&[u32]>,
    ids: &[i32],
    budget: u64,
) -> anyhow::Result<usize> {
    let band_batch = band.map(|w| w.len() as u32);
    if let Some(w) = band {
        let fed: u32 = w.iter().sum();
        anyhow::ensure!(
            !w.is_empty() && fed <= rows && w.iter().all(|x| *x >= 1 && *x <= graph.verify_width),
            "--ragged-widths {w:?} ({fed} rows) do not fit a {rows}-row band at max width {}",
            graph.verify_width
        );
    }
    // C6: the fused arm of a small prefill rung runs the small-chunk rules; the unfused arm keeps
    // the committed ones, so the gate below IS the class comparison at this bucket.
    let small =
        if phase == Phase::Prefill && args.small_chunk_gemm > 0 && rows <= args.small_chunk_gemm {
            let class = match args.small_chunk_class.as_str() {
                "exact" => saber_engine::small_chunk::SmallChunkClass::Exact,
                "band" => saber_engine::small_chunk::SmallChunkClass::Band,
                other => anyhow::bail!("--small-chunk-class {other}: `exact` or `band`"),
            };
            let mut r =
                saber_engine::small_chunk::small_chunk_rules(mxfp4, args.small_chunk_gemm, class)
                    .map_err(|e| anyhow::anyhow!(e))?;
            // G6: the skinny gate replaces the one-split decode cell of the small rungs too.
            if args.skinny_gate {
                r.knobs.skinny = true;
                r.name = format!("{}+g6-skinny", r.name);
            }
            Some(r)
        } else {
            None
        };
    // G5: the fused arm of every other prefill bucket runs the committed rules with the mid-M
    // folded tiles bound; the unfused arm keeps the committed ones (the class comparison again).
    let mid = if phase == Phase::Prefill
        && (args.mid_chunk_gemm || args.mid_chunk_atiled || args.skinny_gate)
        && small.is_none()
    {
        let mut r = mxfp4.clone();
        if args.mid_chunk_gemm {
            r.knobs.folded_mt = true;
            r.name = format!("{}+g5-folded-mt", r.name);
        }
        // G6: the layout law's mid band and the skinny gate, on the fused arm only.
        if args.mid_chunk_atiled {
            r.knobs.atiled_mid = true;
            r.name = format!("{}+g6-atiled-mid", r.name);
        }
        if args.skinny_gate {
            r.knobs.skinny = true;
            r.name = format!("{}+g6-skinny", r.name);
        }
        Some(r)
    } else {
        None
    };
    if rank == 0 {
        if let Some(r) = &small {
            println!(
                "small     p{rows}: the fused arm is planned under `{}` (C6), the unfused arm \
                 under the committed rules: the fusion gate below is the class comparison",
                r.name
            );
        }
        if let Some(r) = &mid {
            println!(
                "midchunk  p{rows}: the fused arm is planned under `{}` (G5/G6), the unfused arm \
                 under the committed rules: the fusion gate below is the class comparison",
                r.name
            );
        }
    }
    let mut plan = plan_for(
        graph,
        layers,
        rows,
        args.ctx,
        rank,
        phase,
        small.as_ref().or(mid.as_ref()).unwrap_or(mxfp4),
        saber,
        true,
        &args.fusion,
        args.no_slot_reuse,
        args.head_on_every_row,
        band_batch,
    )?;
    fusion_control(&mut plan, &args.fusion_control)?;
    // The same plan with the `_nq` fusion off: the two-op sequence the fused one is gated
    // against. The arenas must be identical — a fusion moves a dispatch, not a buffer — and that
    // is asserted here rather than assumed, because the whole gate is a byte comparison of two
    // runs through ONE arena.
    let unfused = plan_for(
        graph,
        layers,
        rows,
        args.ctx,
        rank,
        phase,
        mxfp4,
        saber,
        false,
        &args.fusion,
        args.no_slot_reuse,
        args.head_on_every_row,
        band_batch,
    )?;
    anyhow::ensure!(
        plan.arena_bytes == unfused.arena_bytes
            && plan.buffers.len() == unfused.buffers.len()
            && plan
                .buffers
                .iter()
                .zip(&unfused.buffers)
                .all(|(a, b)| a.name == b.name && a.offset == b.offset && a.bytes == b.bytes),
        "the fused and unfused plans disagree about the arena; the gate below would compare two \
         different layouts"
    );
    // The head, at the SAMPLED rows, shared by both fusion arms. `--head-on-every-row` folds it
    // back into the body plan instead, which is this command's pre-N6 shape.
    let split = !args.head_on_every_row;
    let head_plan = if split {
        Some(plan_head_tp2(
            graph, &plan, args.ctx, rank, phase, mxfp4, saber,
        )?)
    } else {
        None
    };
    let all_plans: Vec<&BlockPlan<'_>> = std::iter::once(&plan).chain(head_plan.iter()).collect();
    let extras: u32 = if split {
        saber_exec::forward::Head::EXTRA_LAUNCHES
    } else {
        0
    };
    // Body + head + the three, which is what a step is. Both fusion arms share the head.
    let declared: u32 = plan.expected_launches()
        + head_plan.as_ref().map_or(0, BlockPlan::expected_launches)
        + extras;
    let arena_total: u64 = all_plans.iter().map(|p| p.arena_bytes).sum();
    // Over BOTH plans. At TP>1 `logits.gather` is `Supplied` and it lives in the HEAD plan, so a
    // banner that read the body alone would stop saying "this rank's shard only" the moment the
    // head moved out of it -- the caveat would disappear exactly when it became a separate plan.
    let mut supplied_ops: BTreeMap<&str, u32> = BTreeMap::new();
    for pl in &all_plans {
        for (op, n) in &pl.supplied {
            *supplied_ops.entry(op).or_default() += n;
        }
    }
    let bucket = plan.bucket;
    let q = bucket.q_len().max(1);
    let batch = bucket.num_seqs();
    let label = match band {
        Some(w) => format!(
            "b{batch}r{rows}(w{})",
            w.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
        ),
        None => format!("b{batch}s{q}"),
    };
    let kv_blocks_per_seq = args.ctx.div_ceil(16).max(1);
    if rank == 0 {
        println!();
        if let Some(h) = &head_plan {
            println!(
                "step      {label}: {:?}, M={rows}, body {} ops / {} launches ({} identity, {} \
                 fused), head {} ops / {} launches at {} row(s), + {extras} this command adds (2 \
                 row selections, 1 shard argmax) = {declared} declared, arena {} MiB",
                bucket,
                plan.steps.len(),
                plan.expected_launches(),
                plan.identity_launches,
                plan.fused_launches,
                h.steps.len(),
                h.expected_launches(),
                h.bucket.m(),
                arena_total >> 20
            );
        } else {
            println!(
                "step      {label}: {:?}, M={rows}, {} ops, {} launches declared ({} identity, {} \
                 fused), arena {} MiB   (ONE plan, head on every row -- not the engine's shape)",
                bucket,
                plan.steps.len(),
                plan.declared_launches,
                plan.identity_launches,
                plan.fused_launches,
                arena_total >> 20
            );
        }
        println!(
            "fusion    {} of {} reduce(s) carry the `_nq` epilogue (the exact reduce + the \
             residual add + the Gemma RMS norm + the per-token e4m3 quant in ONE dispatch) and \
             {} more are folded, with that epilogue, into the split decode GEMM before them \
             (row 3), so the step is {} nodes where the unfused plan is {}; {} decode GEMM \
             pair(s) fold `in_proj_ba` into `in_proj_qkvz`",
            plan.fused_nq,
            plan.steps
                .iter()
                .filter(|s| s.op.kernel == saber_model::graph::KernelOp::AllReduce)
                .count(),
            plan.fused_gemm_ar,
            plan.expected_launches(),
            unfused.expected_launches(),
            plan.fused_pairs
        );
        if plan.fused_gemm_smq > 0 {
            println!(
                "          ws-h40: {} `silu_mul_quant`(s) ride the gate_up decode GEMM before \
                 them (row 4: the last M arriving n-tiles own one row each)",
                plan.fused_gemm_smq
            );
        }
        if plan.fused_rope_appends + plan.fused_combines > 0 {
            let splits: Vec<u32> = plan
                .steps
                .iter()
                .filter_map(|s| match &s.route {
                    saber_exec::Route::Kernel { descs, .. }
                        if descs[0].op == "attn_decode_declc" =>
                    {
                        Some(descs[0].grid[0])
                    }
                    _ => None,
                })
                .collect();
            println!(
                "          ws-h38: {} `kv_append`(s) ride the `qk_norm_rope` before them, {} \
                 decode attention(s) merge in their last-arriving split block (splits {:?})",
                plan.fused_rope_appends,
                plan.fused_combines,
                splits.first()
            );
        }
        // GUARDED, and it was not before. This banner was unconditional, which was right only
        // while something was always unsupplied at TP=2 -- `logits_gather`, the vocabulary
        // all-gather. That op has a code object now, so the union over both plans is EMPTY and
        // the banner printed "0 op(s) have no code object ()" beside a complete step, telling a
        // reader to discount a number that is real. Silence is the honest output there.
        let supplied: u32 = supplied_ops.values().sum();
        if supplied > 0 {
            println!(
                "PARTIAL   {supplied} op(s) have no code object ({}), so the head is this rank's \
                 shard only. The launch structure, the collectives, the state traffic and the \
                 microseconds are real.",
                supplied_ops.keys().copied().collect::<Vec<_>>().join(" + ")
            );
        }
    }

    // ---- state, indices, arena -----------------------------------------------------------------
    let mut budget = budget;
    let mut owned: Vec<DevicePtr> = Vec::new();
    let mut inputs = BlockInputs {
        weights: weights.clone(),
        ..BlockInputs::default()
    };
    let mut state_ptrs: Vec<(String, usize, u64)> = Vec::new();
    let mut state_bytes = 0u64;
    for step in all_plans.iter().flat_map(|p| p.steps.iter()) {
        for st in &step.op.state {
            if inputs.states.contains_key(&st.name) {
                continue;
            }
            let bytes = match st.bytes_per_slot {
                Some(b) => b * u64::from(crate::step_capture::gdn_slot_count(phase, batch, q)),
                None => match step.op.geometry {
                    Geometry::Attn {
                        head_dim,
                        block_size,
                        kv_heads,
                        ..
                    } => {
                        u64::from(batch * kv_blocks_per_seq)
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
            state_bytes += bytes;
            let p = hip.malloc(bytes.max(1))?;
            hip.memset(&p, 0, stream)?;
            inputs.states.insert(st.name.clone(), p.addr());
            state_ptrs.push((st.name.clone(), owned.len(), bytes));
            owned.push(p);
        }
    }
    // This command is decode-only; the shared index law takes the phase explicitly since
    // I15 added a prefill mode to `step-capture`, and saying it here keeps the two callers
    // reading the same way.
    let mut index_keys: Vec<String> = all_plans.iter().flat_map(|p| p.index_keys()).collect();
    index_keys.sort();
    index_keys.dedup();
    for key in index_keys {
        let bytes = match band {
            Some(w) => band_index_table(&key, w, q, rows, kv_blocks_per_seq, args.ctx, ids)?,
            None => index_table(
                &key,
                Phase::Decode,
                batch,
                q,
                kv_blocks_per_seq,
                args.ctx,
                ids,
            )?,
        };
        let p = to_device(hip, stream, &bytes, &mut budget)?;
        inputs.indices.insert(key.clone(), p.addr());
        owned.push(p);
    }
    hip.sync_stream(stream)?;
    if arena_total > budget {
        anyhow::bail!(
            "the arenas are {} MiB and {} MiB is left",
            arena_total >> 20,
            budget >> 20
        );
    }
    let mut exec = BlockExecutor::new(hip, &plan, 8usize << 20)?;
    let mut head_exec = match &head_plan {
        Some(h) => Some(BlockExecutor::new(hip, h, 8usize << 20)?),
        None => None,
    };

    // ---- the three launches no plan models ---------------------------------------------------
    // The row selections and the argmax, the same three `saber-engine` issues between its body and
    // its head. The argmax is over THIS RANK's vocabulary shard: half a vocabulary's winner is not
    // a token, and combining the ranks' picks is the caller's job (`saber forward --tp 2`). It is
    // still real work the engine also issues, so it is in the graph and in the budget.
    let head_rows = head_plan.as_ref().map_or(0, |h| h.bucket.m());
    let sel_rows: Vec<i32> = match (split, phase) {
        (true, Phase::Prefill) => vec![i32::try_from(rows)? - 1],
        (true, Phase::Decode) => (0..i32::try_from(head_rows)?).collect(),
        (false, _) => Vec::new(),
    };
    let head_extra = match (&head_plan, head_exec.as_ref()) {
        (Some(hp), Some(_)) => {
            let hidden = graph.config.hidden_size;
            let vocab = match graph
                .ops
                .iter()
                .find(|o| o.name == "lm_head")
                .map(|o| &o.geometry)
            {
                Some(saber_model::graph::Geometry::Gemm { n, .. }) => *n,
                _ => anyhow::bail!("the graph has no `lm_head` GEMM"),
            };
            let gather = saber_kernels::head::resolve(
                saber_kernels::head::HeadOp::RowGather,
                saber_kernels::head::HeadShape {
                    rows: head_rows,
                    cols: hidden,
                    src_stride: i64::from(hidden),
                    vocab: (0, 0),
                },
                head_rules,
            )?;
            let argmax = saber_kernels::head::resolve(
                saber_kernels::head::HeadOp::Argmax,
                saber_kernels::head::HeadShape {
                    rows: head_rows,
                    cols: vocab,
                    src_stride: i64::from(vocab),
                    vocab: (0, 0),
                },
                head_rules,
            )?;
            let carried: Vec<String> = hp.inputs.clone();
            if carried.len() != 2 {
                anyhow::bail!(
                    "the head reads {} edges from outside its plan ({carried:?}); this command \
                     selects rows for exactly the hidden state and the residual",
                    carried.len()
                );
            }
            let sel = crate::layer::to_device(
                hip,
                stream,
                &saber_exec::forward::i32_bytes(&sel_rows),
                &mut budget,
            )?;
            let pick = hip.malloc(4 * u64::from(head_rows).max(1))?;
            hip.memset(&pick, 0, stream)?;
            owned.push(sel);
            owned.push(pick);
            Some(crate::step_capture::HeadExtra {
                gather,
                argmax,
                carried,
                sel: owned[owned.len() - 2].addr(),
                token: owned[owned.len() - 1].addr(),
            })
        }
        _ => None,
    };
    // Body + head + the three, which is what a step is. Both fusion arms share the head.
    if rank == 0 {
        println!(
            "state     {} caches, {} MiB per rank ({} KV blocks per sequence at ctx {}), {} index \
             tables",
            inputs.states.len(),
            state_bytes >> 20,
            kv_blocks_per_seq,
            args.ctx,
            inputs.indices.len()
        );
    }

    let counters = LaunchCounters::new();
    let capture_counters = CaptureCounters::new();
    let mut launcher =
        Launcher::new(format!("step.tp2.{label}.r{rank}"), &counters).with_peer(Arc::clone(peer));

    // ---- everything the collectives need, before anything launches -----------------------------
    //
    // The reduces are ordinary steps of the plan now (ws-h16), so they are bound with every other
    // op — and the two things a plan cannot carry happen here: the peer-visible regions are
    // allocated, exchanged and zeroed, and the launch digests are compared on the host.
    reset_state(hip, &mut exec, &owned, &state_ptrs, ctx, stream)?;
    let bound = bind_steps(&exec, &plan.steps, &inputs)?;
    let bound_unfused = bind_steps(&exec, &unfused.steps, &inputs)?;
    for step in bound.iter().chain(&bound_unfused) {
        for desc in step.descs {
            launcher.prepare_workspaces(ctx, desc, &step.args, stream)?;
        }
    }
    launcher.reset_ipc_state(ctx, stream)?;
    peer.rendezvous("workspaces-up")?;
    let agreed = tp2::agree_on_collectives(&launcher, peer, &bound)?;
    // The gate below launches the unfused sequence too, so ITS collectives are agreed as well:
    // a digest compared after the fact proves nothing.
    let agreed_unfused = tp2::agree_on_collectives(&launcher, peer, &bound_unfused)?;
    let collectives = tp2::collective_steps(&bound).len();
    if rank == 0 {
        println!(
            "ipc       {} MiB peer-visible per rank, {collectives} collective(s) per step, {agreed} \
             launch digest(s) agreed with the peer before anything launched ({agreed_unfused} \
             more for the unfused sequence the gate runs)",
            launcher.ipc_bytes() >> 20
        );
        if let Some(c) = tp2::collective_steps(&bound).first() {
            println!(
                "          the family the rules picked: {}",
                c.descs[0].kernel
            );
        }
    }

    // ---- the eager two-rank step, both plans ----------------------------------------------------
    let snapshot =
        |hip: &HipThread, exec: &BlockExecutor, owned: &[DevicePtr]| -> anyhow::Result<Snapshot> {
            let mut arena = vec![0u8; usize::try_from(plan.arena_bytes)?];
            exec.download(hip, 0, &mut arena, stream)?;
            let mut states = Vec::with_capacity(state_ptrs.len());
            for (_, i, bytes) in &state_ptrs {
                let mut b = vec![0u8; usize::try_from(*bytes)?];
                hip.memcpy_d2h_blocking(&mut b, &owned[*i], stream)?;
                states.push(b);
            }
            Ok((arena, states))
        };
    // The WHOLE step's arenas, body then head. The fusion gate above deliberately does NOT use
    // this: its two arms differ only in the body, the unfused arm never runs the head, and a
    // comparison that included a head only one arm computed would report a mismatch that is an
    // artifact of the gate's own design. The capture's reference does use it, because the graph
    // it is checking contains the head.
    let snapshot_all = |hip: &HipThread,
                        exec: &BlockExecutor,
                        head_exec: &Option<BlockExecutor>,
                        owned: &[DevicePtr]|
     -> anyhow::Result<Snapshot> {
        let (mut arena, states) = snapshot(hip, exec, owned)?;
        if let (Some(h), Some(hp)) = (head_exec.as_ref(), head_plan.as_ref()) {
            let mut harena = vec![0u8; usize::try_from(hp.arena_bytes)?];
            h.download(hip, 0, &mut harena, stream)?;
            arena.extend_from_slice(&harena);
        }
        Ok((arena, states))
    };
    let mut failures = 0usize;

    // The UNFUSED sequence first: it is the reference the fusion is gated against, and running it
    // first means the arena it leaves behind is overwritten by the fused run rather than the
    // other way round.
    peer.rendezvous("unfused-start")?;
    let unfused_launches = exec
        .run_steps(ctx, &mut launcher, stream, &unfused.steps, &inputs, false)?
        .launches;
    tp2::wait_bounded(ctx, stream, "the unfused eager step")?;
    peer.rendezvous("unfused-done")?;
    let want_unfused = unfused.expected_launches();
    if unfused_launches != want_unfused {
        println!(
            "rank {rank}: LAUNCH BUDGET unfused eager issued {unfused_launches}, expected \
             {want_unfused}"
        );
        failures += 1;
    }
    let (ref_arena, ref_states) = snapshot(hip, &exec, &owned)?;

    reset_state(hip, &mut exec, &owned, &state_ptrs, ctx, stream)?;
    // ws-h37: POISON what a fusion must newly write. The unfused arm ran first over this arena, so
    // a fused dispatch that never wrote a folded op's output would leave the REFERENCE's bytes in
    // place and the comparison below would read BIT-EXACT. Every output buffer of every op a
    // fusion folds (the norm's q / scale / res_out, the pair's second GEMM's C) is filled with
    // 0xFF -- NaN in bf16 and in e4m3 -- before the fused arm; `--fusion-control pair-drop` is
    // the arm that shows it matters.
    let poisoned = poison_folded_outputs(hip, &mut exec, &plan, stream)?;
    if rank == 0 && poisoned > 0 {
        println!(
            "poison    {poisoned} byte(s) of folded-op outputs set to 0xFF before the fused arm"
        );
    }
    // ws-h38: and the KV pages a FOLDED append must write, which live in STATE, not the arena.
    let slot_table = |key: &str| {
        index_table(
            key,
            Phase::Decode,
            batch,
            q,
            kv_blocks_per_seq,
            args.ctx,
            ids,
        )
    };
    let poisoned_kv = poison_appended_slots(hip, &plan, &owned, &state_ptrs, &slot_table, stream)?;
    if rank == 0 && poisoned_kv > 0 {
        println!(
            "poison    {poisoned_kv} byte(s) of the KV slots the folded appends write set to 0xFF \
             (state) before the fused arm"
        );
    }
    peer.rendezvous("eager-start")?;
    let eager_launches = exec
        .run_steps(ctx, &mut launcher, stream, &plan.steps, &inputs, false)?
        .launches;
    tp2::wait_bounded(ctx, stream, "the eager step")?;
    peer.rendezvous("eager-done")?;
    let expected = plan.expected_launches();
    if eager_launches != expected {
        println!("rank {rank}: LAUNCH BUDGET eager issued {eager_launches}, expected {expected}");
        failures += 1;
    }
    let (mut eager_arena, eager_states) = snapshot(hip, &exec, &owned)?;
    // G6: an fp8 activation edge the fused arm writes FRAGMENT-TILED (the layout law's mid band)
    // where the unfused arm writes it row-major holds the same bytes in another order. It is put
    // back through the inverse permutation before the byte comparison -- the permutation the
    // producer gate (`tools/capture/g6`) checks per kernel -- so the gate compares VALUES, and
    // every consumer's output (bf16, the state) is compared as it lies. Counted, never silent.
    let untiled = untile_mid_band_edges(&plan, &unfused, rows, &mut eager_arena)?;
    if untiled > 0 {
        println!(
            "layout    {untiled} fp8 edge(s) the fused arm writes fragment-tiled (G6 mid band) \
             compared through the inverse fragment permutation"
        );
    }

    // ---- the fusion gate: fused vs unfused, byte for byte ---------------------------------------
    //
    // One range per fused reduce is EXCLUDED, and it is the same range the fusion's own
    // precondition is about: the epilogue keeps the sum in registers and never writes it back, so
    // the reduce's output edge -- which aliases its input -- is left holding this rank's partial
    // instead of the sum. The planner only fuses when that edge has exactly ONE reader (the norm
    // it just folded in), so those bytes are dead the moment the dispatch ends. Everything else --
    // every quantized activation, every residual, the head's logits and all of the state -- is
    // compared in full.
    let mut dead: Vec<(usize, usize)> = Vec::new();
    for step in &plan.steps {
        let saber_exec::Route::Kernel { descs, .. } = &step.route else {
            continue;
        };
        // `_nq` names the reduce's input `in`; row 3 names it `c`, the GEMM's own output (the
        // same edge: the reduce reads exactly what the GEMM wrote, and leaves it the partial).
        let slot = match descs[0].op.as_str() {
            "allreduce_exact_nq" => "in",
            "gemm_decode_ar_nq" => "c",
            _ => continue,
        };
        if let Some(saber_exec::Source::Arena { offset, bytes, .. }) = step.source(slot) {
            dead.push((usize::try_from(*offset)?, usize::try_from(*bytes)?));
        }
    }
    dead.sort_unstable();
    dead.dedup();
    let mut live = vec![true; eager_arena.len()];
    for (off, len) in &dead {
        for b in &mut live[*off..(off + len).min(eager_arena.len())] {
            *b = false;
        }
    }
    let dead_bytes: usize = live.iter().filter(|b| !**b).count();
    let live_diff = eager_arena
        .iter()
        .zip(&ref_arena)
        .zip(&live)
        .filter(|((a, b), keep)| a != b && **keep)
        .count();
    let all_diff = eager_arena
        .iter()
        .zip(&ref_arena)
        .filter(|(a, b)| a != b)
        .count();
    let state_diff: usize = eager_states
        .iter()
        .zip(&ref_states)
        .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
        .sum();
    let fused_ok = live_diff == 0 && state_diff == 0;
    // WHERE it first differs, in plan order, and which op produced that buffer. A byte count says
    // the two arms disagree; the first divergent buffer says WHAT disagrees, which is the
    // difference between a finding and a mystery. Printed only on a mismatch.
    if !fused_ok {
        let mut named = 0usize;
        for b in &plan.buffers {
            let (from, to) = (
                usize::try_from(b.offset).unwrap_or(0),
                usize::try_from(b.offset + b.bytes).unwrap_or(0),
            );
            if to > eager_arena.len() || from >= to {
                continue;
            }
            let d = eager_arena[from..to]
                .iter()
                .zip(&ref_arena[from..to])
                .zip(&live[from..to])
                .filter(|((x, y), keep)| x != y && **keep)
                .count();
            if d == 0 {
                continue;
            }
            println!(
                "diverge   rank {rank}: `{}` ({} B, produced by `{}`) differs in {d} byte(s){}",
                b.name,
                b.bytes,
                b.producer,
                if named == 0 {
                    " — FIRST in plan order"
                } else {
                    ""
                }
            );
            named += 1;
            if named == 6 {
                println!("diverge   rank {rank}: ... and more, truncated at six");
                break;
            }
        }
        for (name, (mine, theirs)) in state_ptrs
            .iter()
            .map(|(n, _, _)| n)
            .zip(eager_states.iter().zip(&ref_states))
        {
            let d = mine.iter().zip(theirs).filter(|(x, y)| x != y).count();
            if d > 0 {
                println!(
                    "diverge   rank {rank}: state `{name}` differs in {d} of {} byte(s)",
                    mine.len()
                );
                break;
            }
        }
    }
    if plan.fused_launches > 0 {
        println!(
            "gate      the fused plan ({} `_nq` epilogue(s), {} decode pair(s), {} GEMM + reduce \
             + epilogue(s), control `{}`) vs the unfused plan, same arena, same seed: arena \
             {live_diff}/{} live bytes differ, state {state_diff}/{} — {}",
            plan.fused_nq,
            plan.fused_pairs,
            plan.fused_gemm_ar,
            args.fusion_control,
            eager_arena.len() - dead_bytes,
            ref_states.iter().map(Vec::len).sum::<usize>(),
            if fused_ok { "BIT-EXACT" } else { "MISMATCH" }
        );
        if plan.fused_rope_appends + plan.fused_combines > 0 {
            println!(
                "          (ws-h38 in this arm: {} rope + append(s), {} last-arriver combine(s))",
                plan.fused_rope_appends, plan.fused_combines
            );
        }
        println!(
            "          excluded: {} reduce buffer(s), {} KiB, which the fused kernel leaves \
             holding this rank's PARTIAL by design and which the fusion only fires when nothing \
             else reads ({all_diff} bytes differ across the whole arena, all of them there)",
            dead.len(),
            dead_bytes >> 10
        );
    } else {
        println!(
            "gate      no reduce qualifies for the `_nq` epilogue at this bucket (the message is \
             above the rules' threshold, so the family is the rotated kernel): the plan IS the \
             two-op plan, and it compares {all_diff}/{} arena and {state_diff} state bytes \
             against it",
            eager_arena.len()
        );
    }
    if !fused_ok || (plan.fused_launches == 0 && all_diff != 0) {
        failures += 1;
    }
    peer.agree("fusion-gate", u64::from(fused_ok))?;

    // ---- the WHOLE step, eagerly: body, row selections, head, argmax ---------------------------
    // The reference the replays are compared against. The fusion gate above ran the body alone,
    // twice; this runs the step the graph will contain.
    reset_state(hip, &mut exec, &owned, &state_ptrs, ctx, stream)?;
    if let Some(h) = head_exec.as_mut() {
        h.zero(hip, stream)?;
    }
    peer.rendezvous("whole-step-start")?;
    let whole_launches = crate::step_capture::run_whole_step(
        ctx,
        &mut launcher,
        stream,
        &mut exec,
        &plan,
        head_exec.as_mut(),
        head_plan.as_ref(),
        head_extra.as_ref(),
        &inputs,
    )?;
    tp2::wait_bounded(ctx, stream, "the whole eager step")?;
    peer.rendezvous("whole-step-done")?;
    if whole_launches != declared {
        println!(
            "rank {rank}: LAUNCH BUDGET the whole step issued {whole_launches}, the plans declare \
             {declared}"
        );
        failures += 1;
    }
    let (eager_arena, eager_states) = snapshot_all(hip, &exec, &head_exec, &owned)?;

    // ---- capture: the step AND its reduces, one graph per rank ---------------------------------
    reset_state(hip, &mut exec, &owned, &state_ptrs, ctx, stream)?;
    if let Some(h) = head_exec.as_mut() {
        h.zero(hip, stream)?;
    }
    let proof = GraphSafe::from_flags(true, false).expect("graph_safe && !may_sync");
    let bound = bind_steps(&exec, &plan.steps, &inputs)?;
    let head_bound = match (&head_plan, head_exec.as_ref()) {
        (Some(hp), Some(he)) => bind_steps(he, &hp.steps, &inputs)?,
        _ => Vec::new(),
    };
    check_capturable_with(&bound, PeerCapture::Lockstep)?;
    check_capturable_with(&head_bound, PeerCapture::Lockstep)?;
    // The extras' bindings have to outlive the op list that borrows them.
    let extra_args: Vec<(&saber_kernels::LaunchDescriptor, ArgBindings)> =
        match (&head_plan, head_exec.as_ref(), head_extra.as_ref()) {
            (Some(hp), Some(he), Some(x)) => {
                let (bb, hb) = (exec.arena_base(), he.arena_base());
                let mut v: Vec<(&saber_kernels::LaunchDescriptor, ArgBindings)> = Vec::new();
                for edge in &x.carried {
                    v.push((
                        &x.gather,
                        crate::step_capture::gather_args(&plan, bb, hp, hb, edge, x.sel)?,
                    ));
                }
                v.push((
                    &x.argmax,
                    crate::step_capture::argmax_args(hp, hb, x.token)?,
                ));
                v
            }
            _ => Vec::new(),
        };
    for st in bound.iter().chain(head_bound.iter()) {
        for desc in st.descs {
            launcher.prepare_workspaces(ctx, desc, &st.args, stream)?;
        }
    }
    for (desc, a) in &extra_args {
        launcher.prepare_workspaces(ctx, desc, a, stream)?;
    }
    check_persistent_ready(&launcher, &bound)?;
    check_persistent_ready(&launcher, &head_bound)?;
    let mut ops: Vec<CapturableOp<'_>> = Vec::new();
    for st in &bound {
        ops.push(CapturableOp::sequence(st.descs, &st.args, proof)?);
    }
    // ORDER IS THE STEP: body, selections, head, argmax.
    let n_gathers = extra_args.len().saturating_sub(1);
    for (desc, a) in extra_args.iter().take(n_gathers) {
        ops.push(CapturableOp::new(desc, a, proof));
    }
    for st in &head_bound {
        ops.push(CapturableOp::sequence(st.descs, &st.args, proof)?);
    }
    if let Some((desc, a)) = extra_args.last() {
        ops.push(CapturableOp::new(desc, a, proof));
    }
    peer.rendezvous("pre-capture")?;
    let captured = CapturedBucket::capture(
        ctx,
        &mut launcher,
        &ops,
        stream,
        format!("step.tp2.{label}.r{rank}"),
        &capture_counters,
    )?;
    tp2::wait_bounded(ctx, stream, "the capture")?;
    peer.rendezvous("post-capture")?;
    let n_ops = ops.len();
    let want_nodes = usize::try_from(declared)?;
    if rank == 0 {
        println!(
            "captured  {} node(s) for {} op(s) per rank; the plan's budget is {want_nodes} — {}",
            captured.node_count(),
            n_ops,
            if captured.node_count() == want_nodes {
                "MATCH"
            } else {
                "MISMATCH"
            }
        );
        println!("          of those, {collectives} are all-reduces INSIDE the graph");
    }
    if captured.node_count() != want_nodes {
        anyhow::bail!(
            "rank {rank}: {} nodes, not {want_nodes}",
            captured.node_count()
        );
    }

    // ---- replays, both ranks in lockstep --------------------------------------------------------
    for r in 1..=args.replays {
        reset_state(hip, &mut exec, &owned, &state_ptrs, ctx, stream)?;
        if let Some(h) = head_exec.as_mut() {
            h.zero(hip, stream)?;
        }
        peer.rendezvous(&format!("replay-{r}"))?;
        captured.replay(ctx, stream)?;
        tp2::wait_bounded(ctx, stream, &format!("graph replay {r}"))?;
        peer.rendezvous(&format!("replay-{r}-done"))?;
        // BOTH arenas: the graph contains the head, so a head that replayed differently has to be
        // visible here or the comparison is not of the thing that was captured.
        let (arena, states) = snapshot_all(hip, &exec, &head_exec, &owned)?;
        let a_bad = arena
            .iter()
            .zip(&eager_arena)
            .filter(|(a, b)| a != b)
            .count();
        let s_bad: usize = states
            .iter()
            .zip(&eager_states)
            .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
            .sum();
        let ok = a_bad == 0 && s_bad == 0;
        println!(
            "replay {r}  rank {rank}: arena {a_bad}/{} state {s_bad}/{} differ from the eager \
             step — {}",
            arena.len(),
            eager_states.iter().map(Vec::len).sum::<usize>(),
            if ok { "BIT-EXACT" } else { "MISMATCH" }
        );
        if !ok {
            failures += 1;
        }
        peer.agree(&format!("replay-{r}-verdict"), u64::from(ok))?;
    }

    // ---- timing: the pair, graph against eager, and the AR share -------------------------------
    // The AR share comes off the UNFUSED sequence: dropping a fused reduce would drop the norm
    // inside it too, and the difference would then be two families, not one.
    let no_ar: Vec<BoundStep<'_>> = bound_unfused
        .iter()
        .filter(|b| !b.descs[0].op.starts_with("allreduce"))
        .map(|b| BoundStep {
            name: b.name,
            descs: b.descs,
            args: b.args.clone(),
        })
        .collect();
    let bare = CapturedBlock::capture_bound(
        ctx,
        &mut launcher,
        &no_ar,
        stream,
        format!("step.tp2.{label}.r{rank}.no-ar"),
        proof,
        PeerCapture::Lockstep,
        &capture_counters,
    )?;
    // ... and the whole unfused step as its own graph, so the before/after is a PAIRED comparison
    // of two graphs on the same cards in the same loop, not two runs an hour apart.
    peer.rendezvous("pre-capture-unfused")?;
    let unfused_g = CapturedBlock::capture_bound(
        ctx,
        &mut launcher,
        &bound_unfused,
        stream,
        format!("step.tp2.{label}.r{rank}.unfused"),
        proof,
        PeerCapture::Lockstep,
        &capture_counters,
    )?;
    tp2::wait_bounded(ctx, stream, "the unfused capture")?;
    peer.rendezvous("post-capture-unfused")?;
    if rank == 0 {
        println!(
            "captured  {} node(s) for the unfused plan (its budget is {want_unfused}) — {}; the \
             fusion deletes {} node(s), {:.1}% of the step",
            unfused_g.node_count(),
            if unfused_g.node_count() == usize::try_from(want_unfused)? {
                "MATCH"
            } else {
                "MISMATCH"
            },
            want_unfused - expected,
            100.0 * f64::from(want_unfused - expected) / f64::from(want_unfused)
        );
    }
    anyhow::ensure!(
        unfused_g.node_count() == usize::try_from(want_unfused)?,
        "rank {rank}: the unfused graph is {} nodes, not {want_unfused}",
        unfused_g.node_count()
    );
    let start = hip.create_event(EventFlags::Default)?;
    let end = hip.create_event(EventFlags::Default)?;
    let mut g: Vec<f64> = Vec::with_capacity(args.iters);
    let mut e: Vec<f64> = Vec::with_capacity(args.iters);
    let mut n: Vec<f64> = Vec::with_capacity(args.iters);
    let mut u: Vec<f64> = Vec::with_capacity(args.iters);
    for i in 0..(args.warmup + args.iters) {
        peer.rendezvous(&format!("t{i}-g"))?;
        hip.record_event(&start, stream)?;
        captured.replay(ctx, stream)?;
        hip.record_event(&end, stream)?;
        tp2::wait_bounded(ctx, stream, "a timed replay")?;
        let gv = f64::from(hip.elapsed_ms(&start, &end)?);

        peer.rendezvous(&format!("t{i}-u"))?;
        hip.record_event(&start, stream)?;
        unfused_g.replay(ctx, stream)?;
        hip.record_event(&end, stream)?;
        tp2::wait_bounded(ctx, stream, "a timed unfused replay")?;
        let uv = f64::from(hip.elapsed_ms(&start, &end)?);

        peer.rendezvous(&format!("t{i}-e"))?;
        hip.record_event(&start, stream)?;
        // The WHOLE step, so the eager arm is the same work the graph replays.
        crate::step_capture::run_whole_step(
            ctx,
            &mut launcher,
            stream,
            &mut exec,
            &plan,
            head_exec.as_mut(),
            head_plan.as_ref(),
            head_extra.as_ref(),
            &inputs,
        )?;
        hip.record_event(&end, stream)?;
        tp2::wait_bounded(ctx, stream, "a timed eager step")?;
        let ev = f64::from(hip.elapsed_ms(&start, &end)?);

        peer.rendezvous(&format!("t{i}-n"))?;
        hip.record_event(&start, stream)?;
        bare.replay(ctx, stream)?;
        hip.record_event(&end, stream)?;
        tp2::wait_bounded(ctx, stream, "a timed step without the reduces")?;
        let nv = f64::from(hip.elapsed_ms(&start, &end)?);
        if i >= args.warmup {
            g.push(gv);
            e.push(ev);
            n.push(nv);
            u.push(uv);
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (gm, em, nm, um) = (med(&mut g), med(&mut e), med(&mut n), med(&mut u));
    let theirs = f64::from_bits(
        peer.exchange()
            .swap_digest("step-ms", gm.to_bits())
            .map_err(|e| anyhow::anyhow!("exchanging timings: {e}"))?,
    );
    // ---- H36: the prefill all-reduce overlap, byte-compared and timed in place ----------------
    //
    // The SAME bound body three ways, eagerly, on the cards the timing above just used: in plan
    // order on one stream (A), with every producer GEMM cut into row slices and each slice's
    // reduce on a second stream (B, `saber_exec::overlap`), and without its reduces (N, outputs
    // meaningless). A and B must leave the arena and every state buffer byte-identical; B - N
    // against A - N is the reduces' marginal cost before and after.
    if args.ar_overlap > 0 && phase == Phase::Prefill {
        let law = saber_exec::overlap::OverlapLaw {
            slice_rows: args.ar_overlap,
            min_rows: 2048,
        };
        let hidden = graph.config.hidden_size;
        let reduce = saber_kernels::saber::resolve_ar_banded(
            u64::from(law.slice_rows),
            u64::from(hidden),
            2,
            saber_kernels::saber::ArPayload::Bf16,
            2,
            saber,
            &saber_kernels::saber::Registry::new(saber),
        )?;
        // P27-1d: the overlapped arm in the split form (push + CP wait + reduce) when asked.
        let split = if args.ar_push_blocks > 0 {
            Some(saber_exec::overlap::SplitForm::derive(
                &reduce,
                args.ar_push_blocks,
            )?)
        } else {
            None
        };
        let sched = saber_exec::overlap::derive(&bound_unfused, rows, law, &reduce)?;
        let sched = match &split {
            Some(form) => sched.with_split(form)?,
            None => sched,
        };
        for pair in &sched.pairs {
            for (_, _, ra) in &pair.slices {
                for d in &reduce {
                    launcher.prepare_workspaces(ctx, d, ra, stream)?;
                }
                if let Some(form) = &split {
                    for d in [&form.push, &form.reduce] {
                        launcher.prepare_workspaces(ctx, d, ra, stream)?;
                    }
                }
            }
        }
        let side = hip.create_stream(saber_hip::StreamFlags::NonBlocking)?;
        let mut produced = Vec::new();
        for _ in 0..law.slices(rows) {
            produced.push(hip.create_event(EventFlags::DisableTiming)?);
        }
        let streams = saber_exec::overlap::OverlapStreams {
            side,
            produced,
            done: hip.create_event(EventFlags::DisableTiming)?,
        };
        let counters = saber_exec::overlap::OverlapCounters::default();
        if rank == 0 {
            println!();
            println!(
                "overlap   {label}: {} producer+reduce pair(s) cut into {} slice(s) of {} rows, {} \
                 reduce(s) left in place, +{} dispatches; slice reduce `{}`; form {}",
                sched.pairs.len(),
                law.slices(rows),
                law.slice_rows,
                sched.sequential,
                saber_exec::overlap::extra_launches(&sched),
                reduce.first().map_or("", |d| d.kernel.as_str()),
                split.as_ref().map_or_else(
                    || "oneshot".to_owned(),
                    |f| format!("split (push `{}` x {})", f.push.kernel, f.push.grid[0])
                )
            );
        }
        let run_seq = |launcher: &mut Launcher, steps: &[BoundStep<'_>]| -> anyhow::Result<()> {
            for st in steps {
                let refs: Vec<&saber_kernels::LaunchDescriptor> = st.descs.iter().collect();
                launcher.launch_op(ctx, &refs, &st.args, stream)?;
            }
            Ok(())
        };
        // BYTES FIRST: A, snapshot, B, snapshot, compare -- both ranks, from one reset state each.
        reset_state(hip, &mut exec, &owned, &state_ptrs, ctx, stream)?;
        peer.rendezvous("ov-a")?;
        run_seq(&mut launcher, &bound_unfused)?;
        tp2::wait_bounded(ctx, stream, "the sequential body")?;
        peer.rendezvous("ov-a-done")?;
        let (sa, ss) = snapshot_all(hip, &exec, &head_exec, &owned)?;
        reset_state(hip, &mut exec, &owned, &state_ptrs, ctx, stream)?;
        peer.rendezvous("ov-b")?;
        saber_exec::overlap::run(
            ctx,
            &mut launcher,
            stream,
            &streams,
            &bound_unfused,
            &sched,
            &counters,
        )?;
        tp2::wait_bounded(ctx, stream, "the overlapped body")?;
        peer.rendezvous("ov-b-done")?;
        let (oa, os) = snapshot_all(hip, &exec, &head_exec, &owned)?;
        let a_bad = sa.iter().zip(&oa).filter(|(x, y)| x != y).count();
        let s_bad: usize = ss
            .iter()
            .zip(&os)
            .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
            .sum();
        let ok = a_bad == 0 && s_bad == 0 && sa.len() == oa.len();
        println!(
            "overlap   rank {rank}: arena {a_bad}/{} state {s_bad}/{} differ from the sequential \
             body -- {}",
            sa.len(),
            ss.iter().map(Vec::len).sum::<usize>(),
            if ok { "BIT-EXACT" } else { "MISMATCH" }
        );
        if !ok {
            failures += 1;
        }
        peer.agree("ov-verdict", u64::from(ok))?;
        // THEN TIME: A B N interleaved, a rendezvous before every sample.
        let start = hip.create_event(EventFlags::Default)?;
        let end = hip.create_event(EventFlags::Default)?;
        let (mut ta, mut tb, mut tn) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..(args.warmup + args.iters) {
            for (arm, out) in [(0u8, &mut ta), (1, &mut tb), (2, &mut tn)] {
                peer.rendezvous(&format!("ov-t{i}-{arm}"))?;
                hip.record_event(&start, stream)?;
                match arm {
                    0 => run_seq(&mut launcher, &bound_unfused)?,
                    1 => {
                        saber_exec::overlap::run(
                            ctx,
                            &mut launcher,
                            stream,
                            &streams,
                            &bound_unfused,
                            &sched,
                            &counters,
                        )?;
                    }
                    _ => run_seq(&mut launcher, &no_ar)?,
                }
                hip.record_event(&end, stream)?;
                tp2::wait_bounded(ctx, stream, "a timed overlap arm")?;
                if i >= args.warmup {
                    out.push(f64::from(hip.elapsed_ms(&start, &end)?));
                }
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (am, bm, nm2) = (med(&mut ta), med(&mut tb), med(&mut tn));
        println!(
            "overlap   {label} rank {rank}: {} interleaved triples, eager body only, hipEvents on the \
             main stream: sequential {am:.3} ms, overlapped {bm:.3} ms ({:+.2} %), without the \
             reduces {nm2:.3} ms; the reduces' marginal cost {:.3} -> {:.3} ms; {}",
            args.iters,
            100.0 * (bm - am) / am,
            am - nm2,
            bm - nm2,
            counters.line()
        );
    }

    // ---- every op as its own sub-graph, both ranks in lockstep --------------------------------
    //
    // The same instrument `step-capture --per-op` uses, with one difference that is the whole
    // reason it needs writing again: a link that contains an all-reduce only completes if the
    // OTHER rank is replaying the same link, so every link is preceded by a rendezvous -- at
    // capture and at each sample. Both ranks build the same plan from the same graph, so link
    // `i` is the same op on both.
    if args.per_op {
        // SCOPE, stated because the instrument-cost derivation below depends on it: the links are
        // the BODY's ops. Under the split the graph also holds the head's two ops and the three
        // launches between them, so the links do not tile the graph exactly -- five nodes of
        // roughly 800 fall outside them and are absorbed into the per-link overhead the sum-vs-wall
        // comparison infers. That is small and it is not zero; `--head-on-every-row` puts every op
        // back in one plan if a per-op run needs the links to tile.
        if head_plan.is_some() && rank == 0 {
            println!(
                "per-op    the links are the body's {} op(s); the head's {} and the {extras} \
                 launches between the plans are in the GRAPH but not in the links, so the \
                 inferred per-link cost carries them",
                plan.steps.len(),
                head_plan.as_ref().map_or(0, |h| h.steps.len())
            );
        }
        per_op_links(
            hip,
            ctx,
            &mut launcher,
            &mut exec,
            stream,
            peer,
            &bound,
            &plan,
            &inputs,
            &capture_counters,
            args,
            &label,
            rank,
            gm,
            record_len,
            proof,
        )?;
        // The same pass with the REDUCES ABLATED, against `nm` (this step without them). If an op
        // that holds no collective is slower in the first pass than in this one, what it was
        // waiting for was not its own work -- which is the question F's census leaves open about
        // the TP=2 attention row.
        per_op_links(
            hip,
            ctx,
            &mut launcher,
            &mut exec,
            stream,
            peer,
            &no_ar,
            // `no_ar` is cut from the UNFUSED bound list (dropping a FUSED reduce would drop the
            // norm inside it too), so the plan it is read against is the unfused one -- the fused
            // plan has no `input_layernorm` op of its own to name.
            &unfused,
            &inputs,
            &capture_counters,
            args,
            &format!("{label} reduces-ablated"),
            rank,
            nm,
            record_len,
            proof,
        )?;
    }

    let mut gemm_keys: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for step in &plan.steps {
        if family(step) != "GEMMs" {
            continue;
        }
        for (_, src) in &step.bindings {
            if let saber_exec::Source::Weight { key, .. } = src {
                gemm_keys.insert(key.as_str());
            }
        }
    }
    let gemm_bytes: u64 = gemm_keys
        .iter()
        .filter_map(|k| record_len.get(*k).copied())
        .sum();
    let floor_ms = (gemm_bytes as f64 / (args.dram_gbs * 1e9)) * 1e3;
    if rank == 0 {
        let pair = gm.max(theirs);
        println!();
        println!(
            "time      {label}: {} interleaved triples, hipEvents around the WHOLE step, one \
             stream per rank, a rendezvous before every sample, GPU idle",
            args.iters
        );
        println!("          graph replay   rank 0 {gm:>8.3} ms   rank 1 {theirs:>8.3} ms   pair {pair:>8.3} ms/step");
        println!("          eager step     rank 0 {em:>8.3} ms                        (graph saves {:>6.3} ms, {:+.1}%)", em - gm, 100.0 * (em - gm) / em);
        println!(
            "          unfused graph  rank 0 {um:>8.3} ms  ({} node(s) more): the `_nq` fusion \
             saves {:>6.3} ms, {:+.1}%",
            want_unfused - expected,
            um - gm,
            100.0 * (gm - um) / um
        );
        println!(
            "          without the {collectives} reduces {nm:>8.3} ms: the all-reduces are \
             {:>6.3} ms, {:.1}% of the unfused step",
            um - nm,
            100.0 * (um - nm) / um
        );
        println!();
        // Same fact as the banner, not a sentence of its own: with the glue bound the only op
        // without a code object at TP=2 is the vocabulary gather, and claiming "attention is
        // PARTIAL" after that would discount a number that is real (I15).
        let caveat = if plan.supplied.is_empty() {
            "every op has a code object"
        } else {
            "with op(s) that have no code object, named above"
        };
        println!(
            "compare   P-1b: prod at SPEC=0, greedy, one stream, GPU idle, **TP=2** = \
             {PROD_MS_PER_STEP} ms/step. This step is {rows} rows to prod's 1, {caveat}, so the \
             ratio {:.2}x is a like-for-like reading: same tensor parallelism, same cards, same \
             conditions.",
            pair / PROD_MS_PER_STEP
        );
        println!(
            "floor     this rank's {} layer-GEMM record(s) are {} MiB; at {} GB/s that is \
             {floor_ms:.3} ms of weight stream per card, {:.1}% of the step",
            gemm_keys.len(),
            gemm_bytes >> 20,
            args.dram_gbs,
            100.0 * floor_ms / pair
        );
    }
    println!("rank {rank}  {}", counters.snapshot().line());
    if rank == 0 {
        println!("          {}", capture_counters.snapshot().line());
    }

    drop(captured);
    drop(unfused_g);
    drop(bare);
    drop(exec);
    drop(owned);
    Ok(failures)
}

/// G6: rewrite, in `arena` (the fused arm's snapshot), every buffer the fused plan declares
/// `ATiledFp8` and the unfused plan declares `RowMajorFp8` at the same offset into row-major
/// order, so the two arms compare by value. Returns how many buffers it rewrote. A buffer whose
/// offset or size differs between the plans is refused: the gate's premise is one arena.
fn untile_mid_band_edges(
    fused: &saber_exec::layer::BlockPlan<'_>,
    unfused: &saber_exec::layer::BlockPlan<'_>,
    rows: u32,
    arena: &mut [u8],
) -> anyhow::Result<usize> {
    use saber_model::graph::Layout;
    let mut n = 0;
    for b in &fused.buffers {
        if b.layout != Layout::ATiledFp8 {
            continue;
        }
        let Some(u) = unfused.buffers.iter().find(|u| u.name == b.name) else {
            continue;
        };
        if u.layout != Layout::RowMajorFp8 {
            continue;
        }
        anyhow::ensure!(
            u.offset == b.offset && u.bytes == b.bytes && rows % 16 == 0,
            "`{}`: tiled at {} + {} B, row-major at {} + {} B, {rows} rows: not one arena",
            b.name,
            b.offset,
            b.bytes,
            u.offset,
            u.bytes
        );
        let m = rows as usize;
        let (off, len) = (usize::try_from(b.offset)?, usize::try_from(b.bytes)?);
        // The edge's per-row fp32 scale vector carries the edge's layout tag too, and it is the
        // same bytes in both layouts (one float per row, in row order): nothing to permute.
        if len == 4 * m {
            continue;
        }
        let k = len / m;
        anyhow::ensure!(
            k % 16 == 0 && k * m == len,
            "`{}`: {len} B is not {m} rows",
            b.name
        );
        let tiled = arena[off..off + len].to_vec();
        let dst = &mut arena[off..off + len];
        for r in 0..m {
            for g in 0..k / 8 {
                let t = (((r >> 4) * (k / 16) + (g >> 1)) * 32 + (r & 15) + 16 * (g & 1)) * 8;
                dst[r * k + g * 8..r * k + g * 8 + 8].copy_from_slice(&tiled[t..t + 8]);
            }
        }
        n += 1;
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Every non-default flag reaches the child.
    ///
    /// The exhaustive `let Self { .. }` in `child_argv` makes a FORGOTTEN field a compile error;
    /// this makes a MISSPELLED or mis-valued one a test failure, by parsing the argv back through
    /// the real clap definition rather than through a second hand-written list. Between them there
    /// is no way left for a flag to be parsed by the parent and dropped on the floor -- which has
    /// happened three times in this tree (`--prefill` here, the four gate flags on
    /// `forward --tp 2`, and `--head-on-every-row` here again).
    ///
    /// EVERY field is set away from its default, deliberately: a round trip that passes because
    /// both sides are `false` proves nothing.
    #[test]
    fn every_non_default_flag_round_trips_to_the_child() {
        let args = StepTp2Args {
            pack: PathBuf::from("/packs/some-tier"),
            prompt_ids: PathBuf::from("/tmp/ids.json"),
            prompt_index: 7,
            prebuilt: PathBuf::from("/some/prebuilt"),
            rows: vec![5, 40],
            ragged_rows: vec![40, 48],
            ragged_widths: vec![5, 3, 8, 2, 5, 1, 4, 5],
            spec_width: 8,
            gdn_state_f16: true,
            prefill: vec![1024, 2048],
            ctx: 4096,
            budget_mib: 12345,
            replays: 9,
            iters: 11,
            warmup: 3,
            dram_gbs: 512.5,
            peak_tflops: 321.5,
            per_op: true,
            ar_overlap: 1024,
            ar_push_blocks: 16,
            detail: Some("all_reduce".to_owned()),
            small_chunk_gemm: 128,
            small_chunk_class: "band".to_owned(),
            mid_chunk_gemm: true,
            mid_chunk_atiled: true,
            skinny_gate: true,
            no_slot_reuse: true,
            fusion: "pair".to_owned(),
            fusion_control: "pair-ks".to_owned(),
            head_on_every_row: true,
            hip_lib: "/opt/rocm/lib/libamdhip64.so.7".to_owned(),
            // Parent-only; the child gets these from `child_argv`'s own arguments.
            rank: None,
            port: None,
        };

        let argv = args.child_argv(1, 4321);
        // On a thread with a bigger stack: `Cmd` is a large enum and clap builds it on the stack,
        // which overflows a test thread's default 2 MiB in a debug build. Nothing about the parse
        // itself is unusual -- this is the same `Cli::try_parse_from` `main` calls.
        let cli = std::thread::Builder::new()
            .stack_size(16 << 20)
            .spawn(move || {
                crate::Cli::try_parse_from(std::iter::once(OsString::from("saber")).chain(argv))
            })
            .expect("spawning the parse thread")
            .join()
            .expect("the parse thread")
            .expect("the child argv parses with the real clap definition");

        let crate::Cmd::StepTp2 {
            pack,
            prompt_ids,
            prompt_index,
            prebuilt,
            rows,
            ragged_rows,
            ragged_widths,
            spec_width,
            gdn_state_dtype,
            prefill,
            head_on_every_row,
            ctx,
            budget_mib,
            replays,
            iters,
            warmup,
            dram_gbs,
            peak_tflops,
            per_op,
            ar_overlap,
            ar_form,
            ar_push_blocks,
            detail,
            small_chunk_gemm,
            small_chunk_class,
            mid_chunk_gemm,
            mid_chunk_atiled,
            skinny_gate,
            no_slot_reuse,
            fusion,
            fusion_control,
            rank_of,
            port,
            hip_lib,
        } = cli.cmd
        else {
            panic!("the child argv named a different subcommand");
        };

        assert_eq!(pack, args.pack);
        assert_eq!(prompt_ids, args.prompt_ids);
        assert_eq!(prompt_index, args.prompt_index);
        assert_eq!(prebuilt, args.prebuilt);
        assert_eq!(rows, args.rows);
        assert_eq!(ragged_rows, args.ragged_rows);
        assert_eq!(ragged_widths, args.ragged_widths);
        assert_eq!(spec_width, args.spec_width);
        assert_eq!(gdn_state_dtype == "f16", args.gdn_state_f16);
        assert_eq!(prefill, args.prefill);
        assert_eq!(head_on_every_row, args.head_on_every_row);
        assert_eq!(fusion, args.fusion);
        assert_eq!(fusion_control, args.fusion_control);
        assert_eq!(ctx, args.ctx);
        assert_eq!(budget_mib, args.budget_mib);
        assert_eq!(replays, args.replays);
        assert_eq!(iters, args.iters);
        assert_eq!(warmup, args.warmup);
        assert!((dram_gbs - args.dram_gbs).abs() < 1e-9);
        assert!((peak_tflops - args.peak_tflops).abs() < 1e-9);
        assert_eq!(per_op, args.per_op);
        assert_eq!(ar_overlap, args.ar_overlap);
        assert_eq!(
            crate::ar_push_blocks_of(&ar_form, ar_push_blocks),
            args.ar_push_blocks
        );
        assert_eq!(detail, args.detail);
        assert_eq!(small_chunk_gemm, args.small_chunk_gemm);
        assert_eq!(small_chunk_class, args.small_chunk_class);
        assert_eq!(mid_chunk_gemm, args.mid_chunk_gemm);
        assert_eq!(mid_chunk_atiled, args.mid_chunk_atiled);
        assert_eq!(skinny_gate, args.skinny_gate);
        assert_eq!(no_slot_reuse, args.no_slot_reuse);
        assert_eq!(hip_lib, args.hip_lib);
        // The two the parent owns: the child is TOLD which rank it is and where to meet, and a
        // child that saw no `--rank-of` would decide it was a parent and spawn two more.
        assert_eq!(rank_of, Some(1));
        assert_eq!(port, Some(4321));
    }

    /// The b1s8 body both ranks plan for the fusion gate, on the host (ws-h38's pre-flight).
    fn gate_plan<'g>(graph: &'g Graph, fusion: &str, fuse: bool) -> BlockPlan<'g> {
        let layers: Vec<u32> = (0..graph.config.num_hidden_layers).collect();
        plan_for(
            graph,
            &layers,
            8,
            4096,
            0,
            Phase::Decode,
            &LaunchRules::prod_mxfp4().expect("rules"),
            &SaberRules::prod_saber().expect("saber rules"),
            fuse,
            fusion,
            true,
            false,
            None,
        )
        .expect("the gate's b1s8 plan")
    }

    /// `--fusion` by name: `both`/`all` is what `--decode-fusion` serves (row 3 included),
    /// `gemm-ar` is row 3 with the plain `_nq` (which is then the embedding's reduce alone),
    /// `h37` is ws-h37's set, and a typo is refused rather than read as "everything".
    #[test]
    fn the_fusion_names_plan_the_sets_they_name() {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
        let count = |p: &BlockPlan<'_>| (p.fused_nq, p.fused_pairs, p.fused_gemm_ar);
        assert_eq!(count(&gate_plan(&g, "both", true)), (1, 48, 127));
        assert_eq!(count(&gate_plan(&g, "all", true)), (1, 48, 127));
        assert_eq!(count(&gate_plan(&g, "gemm-ar", true)), (1, 0, 127));
        assert_eq!(count(&gate_plan(&g, "h37", true)), (128, 48, 0));
        assert_eq!(count(&gate_plan(&g, "nq", true)), (128, 0, 0));
        assert_eq!(count(&gate_plan(&g, "pair", true)), (0, 48, 0));
        assert_eq!(count(&gate_plan(&g, "gemm-ar", false)), (0, 0, 0));
        assert!(fusion_options(PlanOptions::BODY, "gemmar").is_err());
        let both = gate_plan(&g, "both", true);
        // One launch per folded op: the embedding's `_nq` norm, 48 pair halves, 127 x 2 row 3,
        // and row 7 (16); row 6 is out of the served set (measured slower).
        assert_eq!((both.fused_rope_appends, both.fused_combines), (16, 0));
        assert_eq!(
            gate_plan(&g, "both", false).expected_launches() - both.expected_launches(),
            1 + 48 + 2 * 127 + 16
        );
        let rope = gate_plan(&g, "rope-append", true);
        assert_eq!(
            (count(&rope), rope.fused_rope_appends, rope.fused_combines),
            ((0, 0, 0), 16, 0)
        );
        let comb = gate_plan(&g, "attn-combine", true);
        assert_eq!(
            (count(&comb), comb.fused_rope_appends, comb.fused_combines),
            ((0, 0, 0), 0, 16)
        );
    }

    /// The two DECISIVE row-3 controls perturb exactly the 127 row-3 dispatches and exactly the
    /// argument they name -- and nothing else in the plan (the `_nq` epilogue of the embedding's
    /// reduce keeps its epsilon). The folded ops are all `Fused`, so the poison covers them.
    #[test]
    fn the_row3_controls_perturb_what_they_name() {
        use saber_kernels::rules::ArgValue;
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
        let clean = gate_plan(&g, "gemm-ar", true);
        let arg = |d: &saber_kernels::rules::LaunchDescriptor, n: &str| {
            d.arg(n).map(|s| s.value.clone()).expect("the slot")
        };
        let descs = |p: &BlockPlan<'_>| -> Vec<saber_kernels::rules::LaunchDescriptor> {
            p.steps
                .iter()
                .filter_map(|s| match &s.route {
                    saber_exec::Route::Kernel { descs, .. } => Some(descs[0].clone()),
                    _ => None,
                })
                .collect()
        };
        for control in ["gemmar-eps", "gemmar-rows"] {
            let mut p = gate_plan(&g, "gemm-ar", true);
            fusion_control(&mut p, control).expect("the control applies");
            let mut touched = 0;
            for (a, b) in descs(&clean).iter().zip(descs(&p)) {
                if a.op != "gemm_decode_ar_nq" {
                    assert_eq!(*a, b, "{control} touched `{}`", a.kernel);
                    continue;
                }
                touched += 1;
                let (ArgValue::F32 { fval: e0 }, ArgValue::F32 { fval: e1 }) =
                    (arg(a, "eps"), arg(&b, "eps"))
                else {
                    panic!("eps is not an f32");
                };
                let rows = (arg(a, "ep_rows"), arg(&b, "ep_rows"));
                match control {
                    "gemmar-eps" => {
                        assert!((e1 - 10.0 * e0).abs() <= f32::EPSILON * e1.abs());
                        assert_eq!(rows.0, rows.1);
                    }
                    _ => {
                        assert_eq!(e0.to_bits(), e1.to_bits());
                        assert_eq!(rows, (ArgValue::I32 { ival: 8 }, ArgValue::I32 { ival: 7 }));
                    }
                }
            }
            assert_eq!(touched, 127, "{control}");
        }
        // A row-3 control on a plan with no row 3 has nothing to perturb: refused, not a pass.
        let mut h37 = gate_plan(&g, "h37", true);
        assert!(fusion_control(&mut h37, "gemmar-eps").is_err());
        assert!(fusion_control(&mut h37, "gemmar-row").is_err(), "a typo");
        // Every op row 3 folds is `Fused` (so the poison writes its outputs), and the norm's
        // three outputs are among them.
        let fused: Vec<&str> = clean
            .steps
            .iter()
            .filter(|s| matches!(s.route, saber_exec::Route::Fused { .. }))
            .map(|s| s.name())
            .collect();
        // 127 reduces + 127 norms into row 3, and the embedding's norm into its plain `_nq`.
        assert_eq!(fused.len(), 2 * 127 + 1);
        assert!(fused.contains(&"layer.3.post_attention_layernorm"));
        assert!(fused.contains(&"layer.3.attn.all_reduce"));
    }

    /// Row 3's names reach rank 1 intact.
    #[test]
    fn the_row3_gate_names_reach_the_child() {
        let args = StepTp2Args {
            pack: PathBuf::from("/p"),
            prompt_ids: PathBuf::from("/i"),
            prompt_index: 0,
            prebuilt: PathBuf::from("/b"),
            rows: vec![8],
            spec_width: 8,
            gdn_state_f16: false,
            prefill: Vec::new(),
            ctx: 4096,
            budget_mib: 1,
            replays: 1,
            iters: 1,
            warmup: 0,
            dram_gbs: 1.0,
            peak_tflops: 1.0,
            per_op: false,
            ar_overlap: 0,
            ar_push_blocks: 0,
            detail: None,
            small_chunk_gemm: 0,
            small_chunk_class: "exact".to_owned(),
            mid_chunk_gemm: false,
            mid_chunk_atiled: false,
            skinny_gate: false,
            no_slot_reuse: true,
            ragged_rows: Vec::new(),
            ragged_widths: Vec::new(),
            fusion: "gemm-ar".to_owned(),
            fusion_control: "gemmar-rows".to_owned(),
            head_on_every_row: false,
            hip_lib: "x".to_owned(),
            rank: None,
            port: None,
        };
        let argv = args.child_argv(1, 9);
        let cli = std::thread::Builder::new()
            .stack_size(16 << 20)
            .spawn(move || {
                crate::Cli::try_parse_from(std::iter::once(OsString::from("saber")).chain(argv))
            })
            .expect("spawning the parse thread")
            .join()
            .expect("the parse thread")
            .expect("the child argv parses");
        let crate::Cmd::StepTp2 {
            fusion,
            fusion_control,
            ..
        } = cli.cmd
        else {
            panic!("the child argv named a different subcommand");
        };
        assert_eq!(
            (fusion.as_str(), fusion_control.as_str()),
            ("gemm-ar", "gemmar-rows")
        );
    }

    /// ws-h38: every new gate arm and control reaches rank 1 as spelled. Both ranks plan the
    /// fused arm from their OWN argv, so a rank that planned `rope-append` beside one planning
    /// `both` would run different reduces -- the argv is the only way rank 1 learns the arm.
    #[test]
    fn the_h38_fusion_arms_and_controls_reach_rank_one() {
        for (fusion, control) in [
            ("rope-append", "none"),
            ("rope-append", "rope-append-drop"),
            ("attn-combine", "none"),
            ("attn-combine", "combine-drop"),
            ("both", "combine-drop"),
        ] {
            let args = StepTp2Args {
                pack: PathBuf::from("/p"),
                prompt_ids: PathBuf::from("/i"),
                prompt_index: 0,
                prebuilt: PathBuf::from("/b"),
                rows: vec![8, 32],
                spec_width: saber_model::graph::VERIFY_WIDTH,
                gdn_state_f16: false,
                prefill: Vec::new(),
                ctx: 4096,
                budget_mib: 1,
                replays: 1,
                iters: 1,
                warmup: 0,
                dram_gbs: 1.0,
                peak_tflops: 1.0,
                per_op: false,
                ar_overlap: 0,
                ar_push_blocks: 0,
                detail: None,
                small_chunk_gemm: 0,
                small_chunk_class: "exact".to_owned(),
                mid_chunk_gemm: false,
                mid_chunk_atiled: false,
                skinny_gate: false,
                no_slot_reuse: true,
                ragged_rows: Vec::new(),
                ragged_widths: Vec::new(),
                fusion: fusion.to_owned(),
                fusion_control: control.to_owned(),
                head_on_every_row: false,
                hip_lib: "x".to_owned(),
                rank: None,
                port: None,
            };
            let argv = args.child_argv(1, 4321);
            let cli = std::thread::Builder::new()
                .stack_size(16 << 20)
                .spawn(move || {
                    crate::Cli::try_parse_from(std::iter::once(OsString::from("saber")).chain(argv))
                })
                .expect("spawning the parse thread")
                .join()
                .expect("the parse thread")
                .expect("the child argv parses");
            let crate::Cmd::StepTp2 {
                fusion: f,
                fusion_control: c,
                rank_of,
                ..
            } = cli.cmd
            else {
                panic!("the child argv named a different subcommand");
            };
            assert_eq!(
                (f.as_str(), c.as_str(), rank_of),
                (fusion, control, Some(1))
            );
        }
    }

    /// The controls are names the rules own: each swaps the fused symbol for the row's control,
    /// and nothing else is touched (the unfused descriptors and the other fusion's are not).
    #[test]
    fn the_h38_controls_swap_to_the_rules_named_symbols() {
        let glue = saber_kernels::glue::GlueRules::prod().expect("glue rules");
        let saber = SaberRules::prod_saber().expect("saber rules");
        let desc = |op: &str, kernel: &str| saber_kernels::rules::LaunchDescriptor {
            op: op.to_owned(),
            plugin: String::new(),
            kernel: kernel.to_owned(),
            grid: [1, 1, 1],
            block: [32, 1, 1],
            shared_bytes: 0,
            kernarg_bytes: 0,
            args: Vec::new(),
            workspaces: Vec::new(),
            n_launches: 1,
        };
        for row in &glue.rope_append {
            let d = desc("qk_rope_append", &row.interleaved);
            let got = super::h38_control_symbol("rope-append-drop", &d).expect("resolves");
            assert_eq!(got.as_deref(), Some(row.control.as_str()));
            assert!(glue.plugin_for(&row.control).is_ok());
            // The other control leaves it alone.
            assert_eq!(
                super::h38_control_symbol("combine-drop", &d).expect("ok"),
                None
            );
            // The chunked layout has no control: refused by name, not silently kept.
            let d = desc("qk_rope_append", &row.chunked);
            assert!(super::h38_control_symbol("rope-append-drop", &d).is_err());
        }
        for row in &saber.attn_decode_declc {
            let d = desc("attn_decode_declc", &row.symbol);
            let got = super::h38_control_symbol("combine-drop", &d).expect("resolves");
            assert_eq!(got.as_deref(), Some(row.control.as_str()));
            assert!(row.control.ends_with("_drop1"));
        }
        // An unfused descriptor is never perturbed.
        let d = desc("attn_decode", "saber_attn_decode_h256_gqa6_fp8kv_nw3");
        assert_eq!(
            super::h38_control_symbol("combine-drop", &d).expect("ok"),
            None
        );
    }

    /// The negative control: a flag left at its default must NOT appear.
    ///
    /// Without this, `child_argv` could pass the test above by emitting every boolean
    /// unconditionally -- and a `--per-op` the caller never asked for would turn a one-minute
    /// measurement into a per-op sweep on two cards.
    #[test]
    fn a_default_flag_is_not_forwarded() {
        let args = StepTp2Args {
            pack: PathBuf::from("/p"),
            prompt_ids: PathBuf::from("/i"),
            prompt_index: 0,
            prebuilt: PathBuf::from("/b"),
            rows: Vec::new(),
            ragged_rows: Vec::new(),
            ragged_widths: vec![5, 5, 5, 3, 5, 2, 5, 4],
            spec_width: saber_model::graph::VERIFY_WIDTH,
            gdn_state_f16: false,
            prefill: Vec::new(),
            ctx: 8192,
            budget_mib: 1,
            replays: 1,
            iters: 1,
            warmup: 0,
            dram_gbs: 1.0,
            peak_tflops: 1.0,
            per_op: false,
            ar_overlap: 0,
            ar_push_blocks: 0,
            detail: None,
            small_chunk_gemm: 0,
            small_chunk_class: "exact".to_owned(),
            mid_chunk_gemm: false,
            mid_chunk_atiled: false,
            skinny_gate: false,
            no_slot_reuse: false,
            fusion: "both".to_owned(),
            fusion_control: "none".to_owned(),
            head_on_every_row: false,
            hip_lib: "x".to_owned(),
            rank: None,
            port: None,
        };
        let argv: Vec<String> = args
            .child_argv(0, 1)
            .iter()
            .map(|o| o.to_string_lossy().into_owned())
            .collect();
        for absent in [
            "--per-op",
            "--detail",
            "--no-slot-reuse",
            "--head-on-every-row",
            "--rows",
            "--ragged-rows",
            "--ragged-widths",
        ] {
            assert!(
                !argv.iter().any(|x| x == absent),
                "{absent} was forwarded although it is at its default: {argv:?}"
            );
        }
        assert!(argv.iter().any(|x| x == "--rank-of"));
    }
}

/// G6: the fusion gate's layout-aware comparison, on real plans and no card. The unfused (committed)
/// p192 plan's arena is filled with random bytes; the fused (mid band) arena is built from it by
/// TILING every edge the fused plan declares tiled (the producers' store permutation). After
/// `untile_mid_band_edges` the two arenas are byte-identical and the scale vectors are left alone.
/// Control: one flipped byte inside a tiled edge survives the untiling.
#[cfg(test)]
mod g6_untile_tests {
    use super::untile_mid_band_edges;
    use saber_exec::layer::{BlockPlan, PlanOptions, RuntimeParams};
    use saber_kernels::rules::LaunchRules;
    use saber_kernels::saber::SaberRules;
    use saber_model::graph::{Bucket, Graph, Layout, ModelConfig};

    #[test]
    fn untiling_the_mid_band_edges_recovers_the_row_major_arena_and_nothing_else() {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
        let base = LaunchRules::prod_mxfp4().expect("rules");
        let mut on = base.clone();
        on.knobs.atiled_mid = true;
        let rows = 192u32;
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let plan = |r: &LaunchRules| {
            BlockPlan::with_options(
                &g,
                Bucket::Prefill {
                    tokens: rows,
                    seqs: 1,
                },
                &layers,
                RuntimeParams {
                    max_ctx: 4096,
                    gdn_slots: 65,
                },
                r,
                &SaberRules::prod_saber().expect("saber"),
                PlanOptions::BODY.with_batch_invariant_attn(),
            )
            .expect("plan")
        };
        let (fused, unfused) = (plan(&on), plan(&base));
        assert_eq!(fused.arena_bytes, unfused.arena_bytes);
        let len = usize::try_from(unfused.arena_bytes).unwrap();
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let row_major: Vec<u8> = (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 24) as u8
            })
            .collect();
        let mut arena = row_major.clone();
        let m = rows as usize;
        let mut tiled_edges = 0;
        for b in &fused.buffers {
            let (off, n) = (b.offset as usize, b.bytes as usize);
            if b.layout != Layout::ATiledFp8 || n == 4 * m {
                continue;
            }
            let k = n / m;
            for r in 0..m {
                for gi in 0..k / 8 {
                    let t = (((r >> 4) * (k / 16) + (gi >> 1)) * 32 + (r & 15) + 16 * (gi & 1)) * 8;
                    arena[off + t..off + t + 8].copy_from_slice(
                        &row_major[off + r * k + gi * 8..off + r * k + gi * 8 + 8],
                    );
                }
            }
            tiled_edges += 1;
        }
        assert!(tiled_edges >= 5, "{tiled_edges} tiled fp8 edges at p192");
        assert_ne!(
            arena, row_major,
            "the tiled arena differs before the untiling"
        );
        let mut flipped = arena.clone();
        let n = untile_mid_band_edges(&fused, &unfused, rows, &mut arena).expect("untile");
        assert_eq!(n, tiled_edges);
        assert_eq!(arena, row_major, "untiling recovers every byte");
        // Control: a byte flipped inside a tiled edge is still there after the untiling.
        let e = fused
            .buffers
            .iter()
            .find(|b| b.layout == Layout::ATiledFp8 && b.bytes as usize != 4 * m)
            .unwrap();
        flipped[e.offset as usize + 777] ^= 0x40;
        untile_mid_band_edges(&fused, &unfused, rows, &mut flipped).expect("untile");
        assert_eq!(
            flipped
                .iter()
                .zip(&row_major)
                .filter(|(a, b)| a != b)
                .count(),
            1
        );
    }
}
