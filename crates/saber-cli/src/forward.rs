//! `saber forward` — one whole forward pass of the model, from token ids to a token id.
//!
//! P1 executed one decoder block. P2 made the pack carry device-final records so that block bound
//! every weight out of an arena. This is the pass those two were for: embedding → 64 layers →
//! final norm → row selection → `lm_head` → argmax, at TP = 1, on one card, with every weight
//! resident and nothing supplied from a capture.
//!
//! # Two plans, not one
//!
//! The body (embedding + layers) is planned at the prompt's row count with
//! [`PlanOptions::BODY`], which gives every layer-local edge ONE arena buffer shared by all 64
//! layers. Without that, a 2048-token chunk needs 27 GiB of activations and does not fit beside
//! 16.7 GiB of weights; with it, 0.6 GiB does.
//!
//! The head is planned separately at ONE row, because `lm_head` at the chunk's row count would
//! compute 2048 x 248320 logits — 970 MiB — of which the sampler reads one row. Selecting the
//! rows first is what prod's `LogitsProcessor` does, and it is why the head plugin has a
//! `row_gather`. Three launches sit between and after the two plans (two row selections and the
//! argmax); they are counted in this command's own launch budget because the graph does not model
//! them: its `logits.gather` is the TP > 1 vocabulary all-gather, a different op.
//!
//! # What this pass does NOT yet compute
//!
//! Three things the attention layers need have no code object on this branch — `qk_norm_rope`,
//! `attn_gate_quant`, and the append of K/V into the paged cache (which the graph does not model
//! as an op at all). The first two are `Route::Supplied` and simply do not launch; the third
//! leaves the KV cache as this command allocated it. So the 16 attention layers do not compute
//! their real output, and the token this prints is NOT the model's token. The command says so,
//! loudly, every run, and `--require-complete` turns the banner into a refusal.
//!
//! That is a statement about the kernel set, not about the engine: the launch count, the arena,
//! the weight binding, the slot reuse and the µs/forward are all real.
//!
//! GPU rules (CLAUDE.md): one HIP device, under the shared lock, never touching the power cap.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use saber_exec::layer::{
    BlockExecutor, BlockInputs, BlockPlan, BoundWeight, PlanOptions, Route, RuntimeParams,
};
use saber_exec::{ArgBindings, LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet};
use saber_hip::{DeviceAddr, DevicePtr, EventFlags, HipThread, StreamFlags};
use saber_kernels::head::{self, HeadOp, HeadRules, HeadShape};
use saber_kernels::rules::{LaunchRules, Shape};
use saber_kernels::saber::SaberRules;
use saber_mem::dio::OpenMode;
use saber_mem::ledger::{Ledger, Pool};
use saber_mem::pack::PackReader;
use saber_mem::upload::{HipSink, PinnedRing, Upload, UploadConfig};
use saber_model::graph::{Geometry, Graph, ModelConfig, Phase};

use crate::layer::{pack_file, to_device};

/// Arguments of `saber forward`.
pub struct ForwardArgs {
    /// The device-final tier pack (directory or `rank<r>.pack`). TP = 1.
    pub pack: PathBuf,
    /// TP rank of the pack.
    pub rank: u32,
    /// A JSON array of token ids, or a `.jsonl` corpus whose rows carry `token_ids`.
    pub prompt_ids: PathBuf,
    /// Row of the corpus to take, when `prompt_ids` is a corpus.
    pub prompt_index: usize,
    /// Truncate the prompt to this many tokens (0 = the whole prompt).
    pub tokens: u32,
    /// Run this many corpus rows in ONE process instead of one prompt.
    ///
    /// The weights load once per PROCESS and each prompt then costs only its own arenas, which is
    /// what makes a 200-prompt agreement gate minutes rather than hours.
    pub prompts: usize,
    /// Skip corpus rows longer than this when `--prompts` selects them.
    ///
    /// It must match the reference capture's own limit, or the two sides compare different lists.
    pub max_prompt: u32,
    /// A `tools/capture/capture_nospec.py` reference to compare the greedy tokens against.
    pub compare: Option<PathBuf>,
    /// Directory of committed `.hsaco` + `*.manifest.toml`.
    pub prebuilt: PathBuf,
    /// Refuse to start if the pass's device buffers would exceed this.
    pub budget_mib: u64,
    /// HIP device ordinal within the visible set.
    pub device: i32,
    /// Replay the pass and report µs/forward.
    pub timing: bool,
    /// Timed replays.
    pub iters: usize,
    /// Replays discarded before timing starts.
    pub warmup: usize,
    /// Refuse rather than print the partial-pass banner.
    pub require_complete: bool,
    /// Read this pass's own logit row back and report ITS top-2 ids and margin.
    ///
    /// The `--compare` table carries the REFERENCE's top1−top2 gap and nothing of ours, so a
    /// disagreement could be read only from prod's side: "prod was nearly indifferent" is not the
    /// same statement as "both engines were". Attributing a flip needs both margins, and this is
    /// the half that was missing (workstream P8, 2026-09-21).
    pub top2: bool,
}

/// This pass's own top-K at the sampled position, best first: `(id, logit)`.
///
/// The logits are `logits.partial` widened from bf16 exactly as the kernels widen them, so the
/// margin `logit1 − logit2` is the same quantity prod's `top5` logprob gap is: `log p1 − log p2`
/// is `z1 − z2` because the softmax normaliser cancels. It is quantised by bf16 — one ulp is
/// 0.0625 at 16 and 0.125 at 32 — which is the same order as the eighths of a nat prod's
/// completions endpoint reports, so neither side resolves a near-tie below ~0.1 nat.
pub(crate) type TopK = Vec<(i32, f32)>;

/// How many the readback keeps. Five, because that is what prod's completions endpoint returns
/// and the two tables are meant to be read side by side.
pub(crate) const TOP_K: usize = 5;

/// The top-K of one bf16 logit row, read as the head wrote it, best first.
///
/// Ties break on the LOWER id, which is what `head_argmax` does, so a row with two equal maxima
/// reports the token the pass actually committed first rather than the other one. NaNs are
/// skipped: an all-NaN row therefore comes back EMPTY, which is the caller's refusal.
pub(crate) fn topk_of_bf16_row(bytes: &[u8], k: usize) -> TopK {
    let mut best: TopK = Vec::with_capacity(k + 1);
    for (i, c) in bytes.chunks_exact(2).enumerate() {
        let v = f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16);
        if v.is_nan() {
            continue;
        }
        if best.len() == k && v <= best[k - 1].1 {
            continue;
        }
        let Ok(id) = i32::try_from(i) else { continue };
        // Strictly-greater keeps the first (lowest) id ahead of a later tie.
        let at = best.iter().position(|(_, u)| v > *u).unwrap_or(best.len());
        best.insert(at, (id, v));
        best.truncate(k);
    }
    best
}

/// The prompt's token ids, from a bare JSON array or from a corpus row.
///
/// `pub(crate)` for `saber engine`, which gates its own ids against this command's: the two paths
/// have to be given the same prompt in the same way for "identical to each other" to mean anything.
pub(crate) fn read_prompt_ids(path: &std::path::Path, index: usize) -> anyhow::Result<Vec<i32>> {
    let text =
        std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let ids: Vec<i64> = if text.trim_start().starts_with('[') {
        serde_json::from_str(&text)?
    } else {
        let line = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .nth(index)
            .ok_or_else(|| anyhow::anyhow!("{}: no row {index}", path.display()))?;
        let row: serde_json::Value = serde_json::from_str(line)?;
        row.get("token_ids")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{} row {index}: no `token_ids` array — this is the corpus \
                     the template corpus writer writes",
                    path.display()
                )
            })?
            .iter()
            .filter_map(serde_json::Value::as_i64)
            .collect()
    };
    if ids.is_empty() {
        anyhow::bail!("{}: the prompt is empty", path.display());
    }
    Ok(ids.into_iter().map(|v| v as i32).collect())
}

/// Every prompt this process will run, as `(corpus row id, token ids)`.
///
/// One by default. `--prompts N` takes the first N rows within `--max-prompt`, in file order,
/// which is exactly how `capture_nospec.py` selects them — both sides must choose the same list
/// the same way or the comparison is between two different sets.
pub(crate) fn read_prompts(args: &ForwardArgs) -> anyhow::Result<Vec<(u64, Vec<i32>)>> {
    let mut out = read_prompt_list(
        &args.prompt_ids,
        args.prompt_index,
        args.prompts,
        args.max_prompt,
    )?;
    if args.prompts <= 1 && args.tokens > 0 {
        for (_, ids) in &mut out {
            ids.truncate(args.tokens as usize);
        }
    }
    Ok(out)
}

/// The prompt list from the four facts it needs, so a caller with a different args struct — the
/// TP=2 pass has its own — reads the SAME list instead of a second implementation of it.
///
/// # Errors
/// The corpus missing, a row without `token_ids`, or no usable prompt.
pub(crate) fn read_prompt_list(
    prompt_ids: &std::path::Path,
    prompt_index: usize,
    prompts: usize,
    max_prompt: u32,
) -> anyhow::Result<Vec<(u64, Vec<i32>)>> {
    if prompts <= 1 {
        let ids = read_prompt_ids(prompt_ids, prompt_index)?;
        return Ok(vec![(prompt_index as u64, ids)]);
    }
    let text = std::fs::read_to_string(prompt_ids)
        .map_err(|e| anyhow::anyhow!("{}: {e}", prompt_ids.display()))?;
    let mut out = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let row: serde_json::Value = serde_json::from_str(line)?;
        let Some(arr) = row.get("token_ids").and_then(|v| v.as_array()) else {
            anyhow::bail!(
                "{}: a row with no `token_ids`; --prompts needs the template corpus",
                prompt_ids.display()
            );
        };
        if max_prompt > 0 && arr.len() > max_prompt as usize {
            continue;
        }
        out.push((
            row.get("id")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
            arr.iter()
                .filter_map(|v| v.as_i64().map(|x| x as i32))
                .collect(),
        ));
        if out.len() >= prompts {
            break;
        }
    }
    if out.is_empty() {
        anyhow::bail!("{}: no usable prompts", prompt_ids.display());
    }
    Ok(out)
}

/// Upper edges of the margin buckets the agreement table reports, in nats. The last bucket is
/// everything above the final edge. They are the ones the TP=1 token gate's table used
/// (`docs/LEDGER.md`, "The token gate RAN"), kept identical so the two runs can be read side by
/// side.
const MARGIN_BUCKETS: [f64; 5] = [0.05, 0.5, 1.0, 2.0, 4.0];

/// One reference prompt: the token the server chose, and the top-5 it chose from.
///
/// The logprobs come back keyed by token TEXT, not by id — the reference server's completions endpoint reports
/// strings — so the ids cannot be compared directly, and the margin is used to characterise a
/// disagreement rather than to decide it.
type RefRow = (Option<i64>, Vec<f64>);

/// One prompt's result: the corpus row id, its token count, the token the pass chose, and — with
/// `--top2` — the pass's own top-2 at that position.
pub(crate) type PassRow = (u64, usize, i32, Option<TopK>);

/// Compare this run's greedy tokens against a `capture_nospec.py` reference, row by row.
///
/// Returns the number that disagreed. The margin column is the reference's OWN top1 - top2 gap,
/// which is the quantity that says whether a disagreement is a near-tie two numerically different
/// passes can flip or a real divergence.
pub(crate) fn compare(reference: &std::path::Path, out: &[PassRow]) -> anyhow::Result<usize> {
    let text = std::fs::read_to_string(reference)
        .map_err(|e| anyhow::anyhow!("{}: {e}", reference.display()))?;
    let mut want: BTreeMap<u64, RefRow> = BTreeMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let row: serde_json::Value = serde_json::from_str(line)?;
        let id = row
            .get("id")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let mut lp: Vec<f64> = row
            .get("top5")
            .and_then(serde_json::Value::as_object)
            .map(|m| m.values().filter_map(serde_json::Value::as_f64).collect())
            .unwrap_or_default();
        lp.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        want.insert(
            id,
            (row.get("token_id").and_then(serde_json::Value::as_i64), lp),
        );
    }

    let (mut compared, mut agree, mut no_ref) = (0usize, 0usize, 0usize);
    let mut rows = Vec::new();
    let mut agree_decided = (0usize, 0usize);
    // Prompts and flips per margin bucket. The headline number answers "do the two sides agree";
    // this answers the only question that follows it, which is whether the disagreements sit where
    // two numerically different passes are ALLOWED to disagree. It was computed by hand from the
    // printed rows for the TP=1 gate, and a hand computation cannot see the prompts that did NOT
    // flip — the denominators are the whole point of the table.
    let mut buckets = [(0usize, 0usize); MARGIN_BUCKETS.len() + 1];
    for (id, n, tok, top2) in out {
        let Some((Some(reftok), lp)) = want.get(id) else {
            no_ref += 1;
            continue;
        };
        compared += 1;
        let gap = if lp.len() >= 2 {
            lp[0] - lp[1]
        } else {
            f64::NAN
        };
        let matched = i64::from(*tok) == *reftok;
        if matched {
            agree += 1;
        } else {
            rows.push((*id, *n, *tok, *reftok, gap, top2.clone()));
        }
        if !gap.is_nan() {
            let b = MARGIN_BUCKETS
                .iter()
                .position(|hi| gap < *hi)
                .unwrap_or(MARGIN_BUCKETS.len());
            buckets[b].0 += 1;
            buckets[b].1 += usize::from(!matched);
        }
        // The same count restricted to prompts the reference itself decides by more than a nat:
        // below that, two passes that differ numerically at all can disagree.
        if gap > 1.0 {
            agree_decided.1 += 1;
            agree_decided.0 += usize::from(matched);
        }
    }
    println!();
    println!(
        "agreement {agree}/{compared} greedy tokens match the no-speculation reference ({:.1}%)",
        100.0 * agree as f64 / compared.max(1) as f64
    );
    println!(
        "          {}/{} where the reference's own margin exceeds 1 nat ({:.1}%)",
        agree_decided.0,
        agree_decided.1,
        100.0 * agree_decided.0 as f64 / agree_decided.1.max(1) as f64
    );
    if no_ref > 0 {
        println!("          {no_ref} prompt(s) had no reference row and were not compared");
    }
    println!();
    println!(
        "{:<16} {:>8} {:>8} {:>8}",
        "ref top1-top2", "prompts", "flipped", "rate"
    );
    let mut lo = 0.0f64;
    for (i, (seen, flipped)) in buckets.iter().enumerate() {
        let label = match MARGIN_BUCKETS.get(i) {
            Some(hi) => format!("{lo:.2} - {hi:.2}"),
            None => format!("> {lo:.2}"),
        };
        if let Some(hi) = MARGIN_BUCKETS.get(i) {
            lo = *hi;
        }
        if *seen == 0 {
            continue;
        }
        println!(
            "{label:<16} {seen:>8} {flipped:>8} {:>7.0}%",
            100.0 * *flipped as f64 / *seen as f64
        );
    }
    if !rows.is_empty() {
        println!();
        // The last three columns only exist under `--top2`, and the header says which run this is:
        // a table with an empty margin column would read as a zero margin.
        let with_top2 = rows.iter().any(|r| r.5.is_some());
        if with_top2 {
            println!(
                "{:<8} {:>7} {:>10} {:>10} {:>12} {:>10} {:>12} {:>12}",
                "prompt",
                "tokens",
                "saber",
                "reference",
                "ref margin",
                "saber t2",
                "saber margin",
                "saber->ref"
            );
        } else {
            println!(
                "{:<8} {:>7} {:>10} {:>10} {:>12}",
                "prompt", "tokens", "saber", "reference", "ref margin"
            );
        }
        for (id, n, got, wanted, gap, top) in rows.iter().take(25) {
            match top {
                Some(k) if k.len() >= 2 => {
                    // How far BELOW our own top-1 we put the REFERENCE's token. When the two
                    // engines swap one pair this is the margin again; when they do not, it is the
                    // honest size of the disagreement and the margin column understates it.
                    let to_ref = k
                        .iter()
                        .find(|(i, _)| i64::from(*i) == *wanted)
                        .map_or_else(
                            || format!("<{:.4}", f64::from(k[0].1) - f64::from(k[k.len() - 1].1)),
                            |(_, v)| format!("{:.4}", f64::from(k[0].1) - f64::from(*v)),
                        );
                    println!(
                        "{id:<8} {n:>7} {got:>10} {wanted:>10} {gap:>12.4} {:>10} {:>12.4} \
                         {to_ref:>12}",
                        k[1].0,
                        f64::from(k[0].1) - f64::from(k[1].1)
                    );
                }
                _ => println!("{id:<8} {n:>7} {got:>10} {wanted:>10} {gap:>12.4}"),
            }
        }
        if rows.len() > 25 {
            println!("          … {} more", rows.len() - 25);
        }
    }
    Ok(rows.len())
}

/// `i32` host buffer as bytes — the shared encoder.
pub(crate) fn i32_bytes(v: &[i32]) -> Vec<u8> {
    saber_exec::forward::i32_bytes(v)
}

/// The index tables a whole-prompt forward pass needs, from the shared laws
/// (`saber_exec::forward`).
///
/// They used to be stated here. Three other callers stated them too, and the copies diverged — the
/// engine's knew that GDN state slot 0 is the kernels' null slot and that `naccept` may never be 0,
/// and this one did not (ws-h15). One law, one place, and a key with no law is still a refusal.
///
/// **`crate::forward_tp2` calls this one**, and did not until ws-h23. It carried its own six-key
/// `match` from before the attention glue was bound, so the TP=2 pass refused
/// `layer.11.attn.kv_append.k_scale` at startup while the TP=1 pass — which had been moved onto
/// the shared laws — ran the same ops. The session this describes is the same at either TP: one
/// sequence, its whole prompt in one prefill chunk, its pages allocated consecutively from 0.
/// Only the SHARDING differs above TP=1, and no index table is a function of it.
pub(crate) fn index_table(
    key: &str,
    n_tokens: u32,
    kv_blocks: u32,
    ids: &[i32],
) -> anyhow::Result<Vec<u8>> {
    use saber_exec::forward::{SeqFacts, StepIndex, FIRST_STATE_SLOT};
    let blocks: Vec<u32> = (0..kv_blocks).collect();
    let seqs = [SeqFacts {
        // The first real slot: 0 is the kernels' null slot, and a pass there updates no GDN state.
        state_slot: FIRST_STATE_SLOT,
        done_before: 0,
        kv_blocks: &blocks,
        live_rows: n_tokens,
        rope: None,
    }];
    let step = StepIndex {
        phase: saber_model::graph::Phase::Prefill,
        rows: n_tokens,
        max_blocks: kv_blocks,
        seqs: &seqs,
        ids,
        band_width: None,
    };
    step.check().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(saber_exec::forward::index_table(key, &step)?)
}

/// Run the pass. Returns 0, or 1 when `--require-complete` was asked of an incomplete kernel set.
#[allow(clippy::too_many_lines)]
pub fn run(hip: &HipThread, args: &ForwardArgs) -> anyhow::Result<usize> {
    hip.set_device(args.device)?;
    // The shared idle check: VRAM first, /metrics second (`crate::gpu_idle`). A whole forward pass
    // uploads the model, so a card with a loading server on it is the one case this has to catch —
    // and `/metrics` alone does not (docs/LEDGER.md, workstream H part 9).
    crate::gpu_idle::require_default(hip, args.device, 0)?;
    let graph = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 1)?;
    let mxfp4 = LaunchRules::prod_mxfp4()?;
    let saber = SaberRules::prod_saber()?;
    let head_rules = HeadRules::prod_head()?;

    // Every prompt this process will run. One by default; a corpus slice under `--prompts`,
    // which is what the agreement gate needs.
    let prompts = read_prompts(args)?;
    let single = prompts.len() == 1;
    // The weight KEYS do not depend on the prompt, so one plan at one length finds them all.
    let probe_n = u32::try_from(prompts[0].1.len())?;
    let layers: Vec<u32> = (0..graph.config.num_hidden_layers).collect();
    let params = RuntimeParams {
        max_ctx: probe_n,
        gdn_slots: saber_exec::forward::FIRST_STATE_SLOT + 1,
    };
    let plans = saber_exec::forward::Plans::new(
        &graph,
        graph.bucket_for(probe_n, Phase::Prefill),
        &layers,
        params,
        &mxfp4,
        &saber,
    )?;

    let load_counters = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &load_counters)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;

    let hidden = graph.config.hidden_size;
    let vocab = match graph
        .ops
        .iter()
        .find(|o| o.name == "lm_head")
        .map(|o| &o.geometry)
    {
        Some(Geometry::Gemm { n, .. }) => *n,
        _ => anyhow::bail!("the graph has no `lm_head` GEMM"),
    };
    println!(
        "model     Qwen3.8-27B at TP=1: {} layers, hidden {hidden}, vocab {vocab}",
        graph.config.num_hidden_layers
    );
    // ---- every weight, out of the pack -------------------------------------------------------
    // The weight KEYS come from the PROBE plans: they do not depend on the prompt's length, so
    // one plan at one length names every record the pass will bind.
    let mut want: BTreeSet<String> = plans.body.weight_keys();
    want.extend(plans.head.weight_keys());
    let reader = PackReader::open(&pack_file(&args.pack, args.rank))?;
    let mut layout_of: BTreeMap<String, u32> = BTreeMap::new();
    let mut idx: Vec<usize> = Vec::new();
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
            "{} record(s) this pass binds are not in {}: {:?}{}. A forward pass needs a TP=1 \
             device-final pack — get one with `tools/fetch_packs.sh`.",
            missing.len(),
            args.pack.display(),
            missing.iter().take(3).collect::<Vec<_>>(),
            if missing.len() > 3 { ", …" } else { "" }
        );
    }
    let (lo, hi) = (idx[0], *idx.last().expect("non-empty"));
    let last = &reader.records()[hi];
    let span =
        saber_mem::pack::align_up(last.data_off + last.length) - reader.records()[lo].data_off;
    let mut budget = args.budget_mib << 20;
    if span > budget {
        anyhow::bail!(
            "the weight window is {} MiB and the budget is {} MiB",
            span >> 20,
            budget >> 20
        );
    }
    budget -= span;

    let cfg = UploadConfig {
        start_record: lo,
        limit_bytes: Some(span),
        mode: OpenMode::Direct,
        ..UploadConfig::default()
    };
    let upload = Upload::open(&args.pack, args.rank, cfg)?;
    let uplan = upload.plan();
    let mut ledger = Ledger::new();
    ledger.declare(Pool::Vram(args.rank), uplan.arena_bytes.max(1))?;
    ledger.declare(Pool::HostPinned, uplan.staging_bytes.max(1))?;
    let reserved = upload.reserve(&mut ledger)?;
    let mut staging = hip.host_malloc(reserved.staging_bytes())?;
    let ring = PinnedRing::new(&mut staging, &reserved)?;
    let mut sink = HipSink::new(hip, &reserved)?;
    let mut weights: BTreeMap<String, BoundWeight> = BTreeMap::new();
    let mut landed = 0u64;
    let stats = upload.run(&mut sink, &ring, |r| {
        if let Some(layout) = layout_of.get(r.name) {
            weights.insert(r.name.to_string(), BoundWeight::new(r.addr, *layout));
            landed += r.len;
        }
    })?;
    println!(
        "weights   {} of {} records bound by name, {} MiB resident, {:.2} GB/s verified \
         (sha256 per record, before the H2D)",
        weights.len(),
        want.len(),
        landed >> 20,
        stats.end_to_end_gbps()
    );
    if weights.len() != want.len() {
        anyhow::bail!(
            "{} record(s) did not land in the window",
            want.len() - weights.len()
        );
    }

    // ---- the passes ---------------------------------------------------------------------------
    let mut out: Vec<PassRow> = Vec::with_capacity(prompts.len());
    for (i, (id, ids)) in prompts.iter().enumerate() {
        let (tok, _launched, top2) = one_pass(
            hip,
            &ctx,
            &stream,
            &graph,
            &mxfp4,
            &saber,
            &head_rules,
            &weights,
            ids,
            budget,
            args,
            single,
        )?;
        if !single && (i + 1) % 25 == 0 {
            println!("   {}/{} prompts", i + 1, prompts.len());
        }
        out.push((*id, ids.len(), tok, top2));
    }

    // `--top2` with no reference is the attribution run: one row per prompt, so a margin can be
    // read at a position nobody has a reference for (a reply's own tokens, say).
    if args.top2 && args.compare.is_none() {
        println!();
        println!(
            "{:<8} {:>7} {:>10} {:>12}  top-5 (id:logit)",
            "prompt", "tokens", "saber", "margin"
        );
        for (id, n, tok, top) in &out {
            match top {
                // The pass refuses when `top[0].0` and the device's argmax disagree, so the
                // `saber` column is the token the pass committed and the margin is beside it.
                Some(k) if k.len() >= 2 => println!(
                    "{id:<8} {n:>7} {tok:>10} {:>12.4}  {}",
                    f64::from(k[0].1) - f64::from(k[1].1),
                    k.iter()
                        .map(|(i, v)| format!("{i}:{v:.4}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                ),
                _ => println!("{id:<8} {n:>7} {tok:>10} {:>12}  -", "-"),
            }
        }
    }
    let bad = match &args.compare {
        Some(reference) => compare(reference, &out)?,
        None => 0,
    };
    drop(sink);
    Ok(bad + usize::from(args.require_complete && bad > 0))
}

/// One forward pass for one prompt, with the weights already resident.
///
/// Everything before this happens once per PROCESS — the idle check, the plugins, the pack window,
/// the 15.4 GiB of weights. Everything in it happens once per PROMPT, because the plans, the state
/// caches, the index tables and both arenas are all sized by the prompt's length. That split is
/// what makes a 200-prompt agreement gate possible: re-uploading the weights per prompt would be
/// two minutes each.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn one_pass(
    hip: &HipThread,
    ctx: &LaunchCtx<'_>,
    stream: &saber_hip::Stream,
    graph: &Graph,
    mxfp4: &LaunchRules,
    saber: &SaberRules,
    head_rules: &HeadRules,
    weights: &BTreeMap<String, BoundWeight>,
    ids: &[i32],
    mut budget: u64,
    args: &ForwardArgs,
    verbose: bool,
) -> anyhow::Result<(i32, u32, Option<TopK>)> {
    // From the graph, not passed in: both are properties of the model and `one_pass` already
    // has it, so a second parameter would be a second chance to disagree.
    let hidden = graph.config.hidden_size;
    let vocab = match graph
        .ops
        .iter()
        .find(|o| o.name == "lm_head")
        .map(|o| &o.geometry)
    {
        Some(Geometry::Gemm { n, .. }) => *n,
        _ => anyhow::bail!("the graph has no `lm_head` GEMM"),
    };
    // THIS prompt's ids, from the caller. Re-reading the file here would run row
    // `--prompt-index` for every iteration of a `--prompts` loop and silently compare 200
    // copies of one answer against 200 different references.
    let n = u32::try_from(ids.len())?;
    let layers: Vec<u32> = (0..graph.config.num_hidden_layers).collect();
    let params = RuntimeParams {
        max_ctx: n,
        // Slot 0 is the kernels' null slot and this pass takes the first real one, so the conv and
        // recurrent caches are sized for two: the reserved slot and the one live sequence. Sizing
        // for one and then pointing the kernels at slot 1 is a memory fault in `gdn_conv_prep`,
        // which is exactly what ws-h15 hit the first time it moved this pass onto the shared law.
        gdn_slots: saber_exec::forward::FIRST_STATE_SLOT + 1,
    };
    let mut body = BlockPlan::with_options(
        graph,
        graph.bucket_for(n, Phase::Prefill),
        &layers,
        params,
        mxfp4,
        saber,
        PlanOptions::BODY,
    )?;
    // THE CHUNK'S LIVE ROWS, stated the way the engine states them (ws-h27). This pass plans a
    // bucket of exactly the prompt's length, so the law is the identity and this pass runs what it
    // ran before -- which is the point: the engine's padded chunk now runs the same attention this
    // one does, instead of the same attention plus 1,814 rows of arena leftovers. A caller here
    // that ever pads (a bucket ladder, a chunked prompt) gets the engine's behaviour by calling
    // the same method with its own row count.
    body.retarget_prefill_rows(n, saber)?;
    // AND THE RECURRENT STATE AT ITS SLOT (ws-h30). `saber_gdn_chunk_scan` indexes `h0`/`ht` by
    // `blockIdx.z`, so the base it is handed has to be the sequence's own slot -- and it now
    // writes `ht` IN PLACE, so a plan left at the pool's base would publish the prompt's final
    // state into slot 0, the kernels' null slot. This pass runs the one sequence the index tables
    // above give `FIRST_STATE_SLOT`, so that is the number, from the same constant.
    body.retarget_prefill_state(saber_exec::forward::FIRST_STATE_SLOT)?;
    // NO layers: `PlanOptions::HEAD` selects the head SITE, and the layer list is what selects
    // the layer site. Passing the body's list here would plan the whole model a second time --
    // which it silently did, until the head reported one outside edge instead of two.
    let head = BlockPlan::with_options(
        graph,
        graph.bucket_for(1, Phase::Decode),
        &[],
        params,
        mxfp4,
        saber,
        PlanOptions::HEAD,
    )?;

    if verbose {
        println!(
            "prompt    {} tokens from {} (row {})",
            n,
            args.prompt_ids.display(),
            args.prompt_index
        );
        println!(
            "body      {} ops over {} layers, arena {} MiB ({} buffers shared across the layers)",
            body.steps.len(),
            layers.len(),
            body.arena_bytes >> 20,
            body.buffers.len()
        );
        println!(
            "head      {} ops at 1 row, arena {} MiB",
            head.steps.len(),
            head.arena_bytes >> 20
        );

        // ---- what has no kernel ------------------------------------------------------------------
        let supplied: BTreeMap<&str, u32> = body.supplied.iter().chain(head.supplied.iter()).fold(
            BTreeMap::new(),
            |mut m, (k, v)| {
                *m.entry(*k).or_default() += v;
                m
            },
        );
        let complete = supplied.is_empty();
        if !complete {
            println!();
            println!("PARTIAL   this pass does not compute the model's output. Missing kernels:");
            for (op, count) in &supplied {
                let why = body
                    .steps
                    .iter()
                    .find(|s| s.op.kernel.id() == *op)
                    .and_then(|s| match &s.route {
                        Route::Supplied { reason } => Some(*reason),
                        _ => None,
                    })
                    .unwrap_or("no route");
                println!("          {op} x{count}: {why}");
            }
            println!(
            "          and the K/V append into the paged cache, which the graph models as state \
             written by the attention op and no op writes."
        );
            println!(
                "          The 16 attention layers therefore produce a value that is not the \
             model's. Everything below — launches, arena, weights, µs — is real."
            );
            if args.require_complete {
                anyhow::bail!(
                    "--require-complete: {} op(s) have no code object on this branch",
                    supplied.values().sum::<u32>()
                );
            }
        }
    }
    // ---- session state: the caches a fresh sequence starts with ------------------------------
    let mut owned: Vec<DevicePtr> = Vec::new();
    let mut inputs = BlockInputs {
        weights: weights.clone(),
        ..BlockInputs::default()
    };
    let kv_blocks = n.div_ceil(16).max(1);
    let mut state_bytes = 0u64;
    for step in &body.steps {
        for st in &step.op.state {
            if inputs.states.contains_key(&st.name) {
                continue;
            }
            let bytes = match st.bytes_per_slot {
                Some(b) => b * u64::from(params.gdn_slots),
                // The paged K/V cache: the graph cannot state its size because a block allocator
                // owns it. One sequence's worth of blocks is what a forward pass needs.
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
                    _ => anyhow::bail!(
                        "state `{}` states no size and its op is not attention",
                        st.name
                    ),
                },
            };
            if bytes > budget {
                anyhow::bail!(
                    "state `{}` ({} MiB) exceeds the budget",
                    st.name,
                    bytes >> 20
                );
            }
            budget -= bytes;
            state_bytes += bytes;
            let p = hip.malloc(bytes.max(1))?;
            hip.memset(&p, 0, stream)?;
            inputs.states.insert(st.name.clone(), p.addr());
            owned.push(p);
        }
    }
    hip.sync_stream(stream)?;
    println!(
        "state     {} caches zeroed, {} MiB (conv + recurrent per GDN layer, {} KV blocks per \
         attention layer)",
        inputs.states.len(),
        state_bytes >> 20,
        kv_blocks
    );

    // ---- index tables a scheduler would build ------------------------------------------------
    for key in body.index_keys() {
        let bytes = index_table(&key, n, kv_blocks, ids)?;
        let p = to_device(hip, stream, &bytes, &mut budget)?;
        inputs.indices.insert(key.clone(), p.addr());
        owned.push(p);
    }
    println!(
        "indices   {} tables built for one sequence of {n} tokens",
        inputs.indices.len()
    );

    // ---- the two arenas ----------------------------------------------------------------------
    if body.arena_bytes + head.arena_bytes > budget {
        anyhow::bail!(
            "the arenas are {} MiB and only {} MiB of budget is left",
            (body.arena_bytes + head.arena_bytes) >> 20,
            budget >> 20
        );
    }
    budget -= body.arena_bytes + head.arena_bytes;
    let staging_bytes = 8usize << 20;
    let mut body_exec = BlockExecutor::new(hip, &body, staging_bytes)?;
    let mut head_exec = BlockExecutor::new(hip, &head, staging_bytes)?;
    body_exec.zero(hip, stream)?;
    head_exec.zero(hip, stream)?;

    // ---- the three launches the graph does not model -----------------------------------------
    // The row selection: `lm_head` is asked for the LAST position of the sequence, so one index.
    let sel = to_device(hip, stream, &i32_bytes(&[n as i32 - 1]), &mut budget)?;
    let token = hip.malloc(4)?;
    hip.memset(&token, 0, stream)?;
    let gather = head::resolve(
        HeadOp::RowGather,
        HeadShape {
            rows: 1,
            cols: hidden,
            src_stride: i64::from(hidden),
            vocab: (0, 0),
        },
        head_rules,
    )?;
    let argmax = head::resolve(
        HeadOp::Argmax,
        HeadShape {
            rows: 1,
            cols: vocab,
            src_stride: i64::from(vocab),
            vocab: (0, 0),
        },
        head_rules,
    )?;
    // The two edges the head reads and the body wrote. Asked of the head plan, so the names are
    // the graph's, never spelled out here.
    let carried: Vec<String> = head.inputs.clone();
    if carried.len() != 2 {
        anyhow::bail!(
            "the head reads {} edges from outside its plan ({carried:?}); this command selects \
             rows for exactly the hidden state and the residual",
            carried.len()
        );
    }
    let addr = |p: &BlockPlan<'_>, base: DeviceAddr, edge: &str| -> anyhow::Result<DeviceAddr> {
        let b = p
            .locate(edge)
            .ok_or_else(|| anyhow::anyhow!("`{edge}` is not a planned buffer"))?;
        base.offset(b.offset)
            .ok_or_else(|| anyhow::anyhow!("`{edge}` runs past its arena"))
    };

    let launch_counters = LaunchCounters::new();
    let mut launcher = Launcher::new("forward", &launch_counters);
    let once = |body_exec: &mut BlockExecutor,
                head_exec: &mut BlockExecutor,
                launcher: &mut Launcher|
     -> anyhow::Result<u32> {
        let mut total = body_exec
            .run_steps(ctx, launcher, stream, &body.steps, &inputs, false)?
            .launches;
        for edge in &carried {
            let mut a = ArgBindings::new(Shape::new(0, 0, 0));
            a.bind("src", addr(&body, body_exec.arena_base(), edge)?);
            a.bind("idx", sel.addr());
            a.bind("out", addr(&head, head_exec.arena_base(), edge)?);
            total += head_exec.launch_extra(ctx, launcher, stream, &gather, &a)?;
        }
        total += head_exec
            .run_steps(ctx, launcher, stream, &head.steps, &inputs, false)?
            .launches;
        let mut a = ArgBindings::new(Shape::new(0, 0, 0));
        a.bind(
            "logits",
            addr(&head, head_exec.arena_base(), "logits.partial")?,
        );
        a.bind("out", token.addr());
        total += head_exec.launch_extra(ctx, launcher, stream, &argmax, &a)?;
        Ok(total)
    };

    let launched = once(&mut body_exec, &mut head_exec, &mut launcher)?;
    hip.sync_stream(stream)?;
    let expected = body.expected_launches() + head.expected_launches() + 3;
    println!(
        "launches  {launched} issued; {} the graph declares for the body + {} for the head + 3 \
         this command adds (2 row selections, 1 argmax)",
        body.expected_launches(),
        head.expected_launches()
    );
    if launched != expected {
        anyhow::bail!("the pass issued {launched} dispatches, not the {expected} declared");
    }

    let mut got = [0u8; 4];
    hip.memcpy_d2h_blocking(&mut got, &token, stream)?;
    let tok = i32::from_le_bytes(got);
    if verbose {
        let complete = body.supplied.is_empty() && head.supplied.is_empty();
        println!(
            "token     argmax = {tok}{}",
            if complete {
                ""
            } else {
                "   (NOT the model's token: see the PARTIAL banner above)"
            }
        );
    }
    // The two failures the shared check tells apart: a bad id, and the kernel's
    // every-logit-was-NaN answer, which means the PASS is broken rather than the sampler.
    saber_exec::forward::check_token(tok, vocab)?;

    // ---- this pass's OWN top-2, when it was asked for ------------------------------------------
    // The same row the argmax kernel just read, widened on the host. It is read AFTER the token so
    // a `--top2` run cannot change what the pass committed, and it is asserted against the argmax:
    // a top-1 that disagrees with the device's own answer means the row moved between the two
    // reads, and reporting a margin off a different row is worse than reporting none.
    let top2 = if args.top2 {
        let b = head
            .locate("logits.partial")
            .ok_or_else(|| anyhow::anyhow!("`logits.partial` is not a planned buffer"))?;
        let mut bytes = vec![0u8; vocab as usize * 2];
        head_exec.download(hip, b.offset, &mut bytes, stream)?;
        let got = topk_of_bf16_row(&bytes, TOP_K);
        if got.len() < 2 {
            anyhow::bail!(
                "the logit row has {} finite value(s); the pass is broken, not the readback",
                got.len()
            );
        }
        if got[0].0 != tok {
            anyhow::bail!(
                "the host read {} as the top logit and the device's argmax says {tok}; the row \
                 moved between the two reads and this margin would be of neither",
                got[0].0
            );
        }
        if verbose {
            println!(
                "top{TOP_K}      {}; margin {:.4} nats",
                got.iter()
                    .map(|(i, v)| format!("{i}:{v:.4}"))
                    .collect::<Vec<_>>()
                    .join(" "),
                f64::from(got[0].1) - f64::from(got[1].1)
            );
        }
        Some(got)
    } else {
        None
    };

    // ---- timing -------------------------------------------------------------------------------
    if args.timing {
        for _ in 0..args.warmup {
            once(&mut body_exec, &mut head_exec, &mut launcher)?;
        }
        hip.sync_stream(stream)?;
        let start = hip.create_event(EventFlags::Default)?;
        let end = hip.create_event(EventFlags::Default)?;
        hip.record_event(&start, stream)?;
        for _ in 0..args.iters {
            once(&mut body_exec, &mut head_exec, &mut launcher)?;
        }
        hip.record_event(&end, stream)?;
        hip.sync_event(&end)?;
        let ms = f64::from(hip.elapsed_ms(&start, &end)?) / args.iters as f64;
        println!(
            "time      {:.0} µs/forward at {n} tokens ({} replays, GPU idle — prod is down, so \
             this is not a beside-prod number)",
            ms * 1000.0,
            args.iters
        );
        println!(
            "          {:.1} µs/layer, {:.2} µs/launch over {launched} dispatches",
            ms * 1000.0 / f64::from(graph.config.num_hidden_layers),
            ms * 1000.0 / f64::from(launched)
        );
    }
    println!("counters  {}", launch_counters.snapshot().line());

    drop(owned);
    Ok((tok, launched, top2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_json_array_and_a_corpus_row_both_read_as_a_prompt() {
        // Under `target/`, removed on drop -- never `/tmp` (FN51h).
        let dir = saber_test_scratch::ScratchDir::new("saber-forward-test");
        let a = dir.join("ids.json");
        std::fs::write(&a, "[1, 2, 3]").expect("write");
        assert_eq!(read_prompt_ids(&a, 0).expect("array"), vec![1, 2, 3]);

        let b = dir.join("corpus.jsonl");
        std::fs::write(
            &b,
            "{\"id\":0,\"token_ids\":[7,8]}\n{\"id\":1,\"token_ids\":[9]}\n",
        )
        .expect("write");
        assert_eq!(read_prompt_ids(&b, 1).expect("row 1"), vec![9]);
        assert!(read_prompt_ids(&b, 9).is_err(), "a row that is not there");
    }

    /// Every index table this pass builds is a fact about the SESSION, and each one has a
    /// different width and a different dtype. A wrong one is a kernel walking off a buffer.
    #[test]
    fn the_index_tables_are_the_ones_a_single_sequence_prefill_needs() {
        let ids = [11i32, 12, 13, 14];
        let t = |k: &str| index_table(k, 4, 1, &ids).expect(k);
        assert_eq!(t("embed_tokens.ids"), i32_bytes(&ids));
        assert_eq!(t("layer.0.gdn.conv_prep.cu"), i32_bytes(&[0, 4]));
        // Slot ONE, not zero. This assertion used to read `[0]`, and that was a real bug this
        // pass carried: `saber_gdn_conv_update` and `saber_gdn_recurrent_update` both bail on
        // `slot <= 0`, so every GDN layer of this pass launched and updated no state at all. The
        // shared law (ws-h15) numbers slots from `FIRST_STATE_SLOT`, and the engine's copy had
        // known it all along — which is the argument for there being one copy.
        assert_eq!(t("layer.0.gdn.conv_prep.cache_idx"), i32_bytes(&[1]));
        // A BYTE, not an int: the kernel reads `has_init[n]` as `uint8_t`.
        assert_eq!(t("layer.0.gdn.conv_prep.has_init"), vec![0u8]);
        assert_eq!(t("layer.3.attn.prefill.block_table"), i32_bytes(&[0]));
        assert_eq!(t("layer.3.attn.prefill.seqused_k"), i32_bytes(&[4]));
        // `sidx` now HAS a law (the shared one serves decode too): one snapshot slot per row,
        // from 1.
        assert_eq!(
            index_table("layer.0.gdn.decode.sidx", 4, 1, &ids).expect("sidx"),
            i32_bytes(&[1, 2, 3, 4])
        );
        // NEGATIVE CONTROL: an index with no law anywhere is still a refusal, not a zero-filled
        // buffer that a kernel then reads as an offset.
        assert!(index_table("layer.0.gdn.decode.whatever", 4, 1, &ids).is_err());
    }

    /// `--top2` reports the margin a flip is attributed with, so the three ways it could lie are
    /// each a case here: the wrong order, the wrong tie-break, and a row it cannot read.
    #[test]
    fn the_top2_of_a_logit_row_is_the_row_the_argmax_read() {
        let row = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                .collect()
        };
        // Ordinary row: id 2 wins by 2 nats over id 0. bf16 holds all four of these exactly.
        let k = topk_of_bf16_row(&row(&[3.0, 1.0, 5.0, -2.0]), TOP_K);
        assert_eq!(
            k.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![2, 0, 1, 3]
        );
        assert!((f64::from(k[0].1) - f64::from(k[1].1) - 2.0).abs() < 1e-6);
        // K binds: the whole point of the readback is that it is bounded, not a vocabulary sort.
        assert_eq!(topk_of_bf16_row(&row(&[3.0, 1.0, 5.0, -2.0]), 2).len(), 2);

        // A TIE breaks on the lower id, because that is what the argmax kernel commits. If this
        // went the other way a tied row would report the token the pass did NOT emit, and the
        // `k[0].0 != tok` refusal in the pass would fire on a correct run.
        let k = topk_of_bf16_row(&row(&[1.0, 7.0, 7.0]), TOP_K);
        assert_eq!((k[0].0, k[1].0), (1, 2));

        // NEGATIVE CONTROLS, both of which the caller must refuse on. A row of one logit has no
        // second, and an all-NaN row is the kernel's broken-pass answer.
        assert_eq!(topk_of_bf16_row(&row(&[1.0]), TOP_K).len(), 1);
        assert!(topk_of_bf16_row(&row(&[f32::NAN, f32::NAN]), TOP_K).is_empty());

        // The margin is the LOGIT gap, which is the logprob gap: the softmax normaliser cancels.
        // Held against a hand-computed log-softmax so the claim in the doc comment is tested and
        // not merely asserted.
        let z = [3.0f64, 1.0, 5.0, -2.0];
        let m = z.iter().fold(f64::MIN, |a, b| a.max(*b));
        let denom: f64 = z.iter().map(|x| (x - m).exp()).sum::<f64>().ln() + m;
        let lp: Vec<f64> = z.iter().map(|x| x - denom).collect();
        assert!((lp[2] - lp[0] - 2.0).abs() < 1e-12);
    }
}
