//! INK8: the Inkling 2 serve plane (`saber engine --pack <inkling2 pack> --tp 2 --ink-serve PORT`).
//!
//! The two-card process (one process, two HIP threads, the `ink_p2p_xchg` reduces + head gather)
//! behind the 27B's scheduler (`saber_sched::Scheduler`: admission, prompt chunks, stop strings,
//! `max_tokens`, the checkpoint's eos) and HTTP plane (`saber_serve::server`: `/v1/chat/completions`
//! and `/v1/completions`, SSE streaming, logprobs, `/metrics`), the dense sampler
//! (`Sampler::sample_dense` over the gathered row: greedy, temperature / top-k / top-p / min-p,
//! penalties, seeds keyed by output position). What is Inkling's own: THIS checkpoint's chat
//! template (`reasoning_effort` none..max, default 0.9 = the template's own; its `{% generation %}`
//! markers stripped exactly as the oracle `ink2_ref.py` renders), its tokenizer, eos 200006
//! (`<|content_model_end_sampling|>`); no generation-prompt header: the model writes
//! `<|message_model|>` itself (the oracle's law) and the special tokens are dropped from the text.
//! No reasoning / tool parser (the checkpoint's grammar is not one of Saber's parsers).
//!
//! The state: `--ink-serve-seqs` per-visit caches a rank (48 visits x {K, V pools, k / v conv
//! states} + 48 x {attn, mlp} sconv states), each sized to `--ink-serve-ctx` tokens; a step runs its
//! sequences one after another; a decode step replays ONE captured graph a cache (captured at the
//! cache's first decode; `--no-ink-serve-graph` = eager). Rank 0 decides and sends each command to
//! rank 1 (the same commands on the same state in the same order).

use std::collections::BTreeMap;

use anyhow::Context as _;
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::Duration;

use saber_mem::ledger::{Ledger, Pool};
use saber_plan::{Bucket, Phase};
use saber_sched::{
    ArenaBytes, KvGeometry, Scheduler, SchedulerConfig, SeqId, SlotSampling, StepBatch,
    StepExecutor, StepResult,
};
use saber_serve::engine::{ModelInfo, SampledLogprobs, TokenId};
use saber_serve::sampler::{
    EntropyNoise, NoiseSource, PenaltyContext, RowParams, Sampler, SeededNoise,
};
use serde::{Deserialize, Serialize};

use super::{Cache, Card, Forward, PackReader, Reg, Weights, A, D, VOCAB};

/// INK11: the device seam's row (`ink_seam_p2`): 64 values, 64 ids, lse, lse_t, 2 pad words.
const SEAM_K: usize = 64;
const SEAM_W: usize = 132;
const SEAM_NCH: u64 = 64;
use crate::inkling2::InkFlags;

/// The prompt chunk a prefill pass takes (rows; the pass / visit arenas hold it at the defaults).
pub const SERVE_CHUNK: u32 = 1024;

/// INK54: `--ink-p54` -> the decode-only steps after a chunk that ran beside decoders (`il<N>`;
/// 0 = the control). The exec plane reads the same flag's `wk` / `wke`.
pub fn p54_yield(v: &str) -> anyhow::Result<u32> {
    Ok(super::ink54_bits(v)?.yield_steps)
}

/// INK43: `--ink-p43` -> the served prefill chunk (rows): `ch<N>` (1,024..4,096, a multiple of
/// 64), else [`SERVE_CHUNK`] (the control).
#[cfg(test)]
pub fn p43_chunk(v: &str) -> anyhow::Result<u32> {
    Ok(p43_opts(v)?.0)
}

/// INK43: `--ink-p43` -> (the served prefill chunk, `am`): `am` = a one-slot drafter's prefill
/// append on the matrix cores (INK22's `pfmm`, <= 64 rows a GEMM, instead of 8-row GEMVs: the
/// drafter's context rows move, the target's never).
pub fn p43_opts(v: &str) -> anyhow::Result<(u32, bool)> {
    let (mut ch, mut am) = (SERVE_CHUNK, false);
    for it in v
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty() && *x != "none")
    {
        if it == "am" {
            am = true;
            continue;
        }
        match it.strip_prefix("ch").and_then(|n| n.parse::<u32>().ok()) {
            Some(n) if (1024..=4096).contains(&n) && n % 64 == 0 => ch = n,
            _ => {
                anyhow::bail!("--ink-p43 item {it:?}: ch<N> (N 1024..4096, a multiple of 64) | am")
            }
        }
    }
    Ok((ch, am))
}

/// INK43: the levers a plane runs: the two-card plane `--ink-p43` as given; the one-card plane
/// (INK41, gated before INK43) stands the DEFAULT down to the control (an explicit value is kept).
pub fn p43_plane(v: &str, two_card: bool) -> anyhow::Result<(u32, bool)> {
    if !two_card && v == crate::inkling2::DEFAULT_INK43 {
        return Ok((SERVE_CHUNK, false));
    }
    p43_opts(v)
}

/// INK43: the local KV ring for a `chunk`-row served prefill: at least the 1,024-row window + the
/// chunk (pages), `0` (full per-visit pools) kept.
pub fn p43_ring(ring: u64, chunk: u32) -> u64 {
    if ring == 0 {
        return 0;
    }
    ring.max((super::EXT + u64::from(chunk)).div_ceil(super::PAGE) * super::PAGE)
}

/// One command of the pair: rank 0 sends it, both ranks apply it.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum Cmd {
    /// Prefill `ids` into cache `slot` from position `pos0` (0 = a new sequence).
    Prefill {
        slot: u32,
        ids: Vec<u32>,
        pos0: u64,
    },
    /// INK9 task 4: one decode step over every decoding sequence: rows (slot, token, position),
    /// run as ONE forward padded to the bucket (1 / 2 / 4 / 8; pad rows on the pad slot).
    Decode {
        rows: Vec<(u32, u32, u64)>,
    },
    /// INK13: one sequence's verify chunk (`ids` = [the last committed token ; the drafts]) at
    /// positions `pos..`, on the captured graph of (slot, ids.len()), every conv stashed.
    Verify {
        slot: u32,
        ids: Vec<u32>,
        pos: u64,
    },
    /// INK13: roll the last verify's convs back to `keep` committed rows.
    Commit {
        keep: u64,
    },
    /// INK20 (`--ink-mtp-batch`): B sequences' verify chunks in ONE multi-sequence verify
    /// (rows (slot, [anchor ; drafts], position), every sequence the same width), on the captured
    /// graph of (B, width); every conv's history per sequence
    VerifyM {
        rows: Vec<(u32, Vec<u32>, u64)>,
    },
    /// INK20: roll the last multi verify's convs back to each sequence's keep
    CommitM {
        keeps: Vec<u32>,
    },
    /// INK23 (`--ink-prefix-cache`): copy slot `slot`'s state at `len` positions into prefix node
    /// `node` (+ the drafter's context rows on rank 0 when `draft`)
    PcStore {
        slot: u32,
        node: u64,
        len: u64,
        draft: bool,
    },
    /// INK23: copy prefix node `node` (`len` positions) into slot `slot` (+ the drafter's
    /// context on rank 0 when `draft`: its current slot then holds `len` rows)
    PcLoad {
        slot: u32,
        node: u64,
        len: u64,
        draft: bool,
    },
    Stop,
}

/// What the scheduler's step thread asks the engine thread.
enum Job {
    Admit(SeqId),
    /// INK23: an admitted sequence's whole prompt (the prefix cache's plan)
    Prompt(SeqId, Vec<TokenId>),
    Release(SeqId),
    Step(OwnedBatch, Sender<StepResult>),
}

struct OwnedSlot {
    seq: SeqId,
    verify_width: u32,
    input: Vec<TokenId>,
    chunk: Option<saber_sched::PrefillChunk>,
    sampling: SlotSampling,
}

struct OwnedBatch {
    phase: Phase,
    slots: Vec<OwnedSlot>,
}

struct InkExec {
    tx: Sender<Job>,
}

impl std::fmt::Debug for InkExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("InkExec")
    }
}

impl StepExecutor for InkExec {
    fn execute(&mut self, batch: &StepBatch<'_>) -> StepResult {
        let (rtx, rrx) = channel();
        let own = OwnedBatch {
            phase: batch.phase,
            slots: batch
                .slots
                .iter()
                .map(|s| OwnedSlot {
                    seq: s.seq_id,
                    verify_width: s.verify_width,
                    input: s.input_tokens.clone(),
                    chunk: s.chunk,
                    sampling: s.sampling.clone(),
                })
                .collect(),
        };
        if self.tx.send(Job::Step(own, rtx)).is_err() {
            return StepResult::default();
        }
        rrx.recv().unwrap_or_default()
    }
    fn admit(&mut self, seq: &saber_sched::AdmittedSeq) {
        let _ = self.tx.send(Job::Admit(seq.seq_id));
        let _ = self
            .tx
            .send(Job::Prompt(seq.seq_id, seq.prompt_tokens.clone()));
    }
    fn release(&mut self, seq: SeqId) {
        let _ = self.tx.send(Job::Release(seq));
    }
}

/// The checkpoint's template as the oracle renders it: `{% generation %}` markers dropped.
pub fn template_text(ckpt: &std::path::Path) -> anyhow::Result<String> {
    let p = ckpt.join("chat_template.jinja");
    let s = std::fs::read_to_string(&p)?;
    Ok(s.replace("{%- generation -%}", "")
        .replace("{%- endgeneration -%}", ""))
}

/// The checkpoint's eos ids (`generation_config.json`: 200006 = `<|content_model_end_sampling|>`);
/// this tokenizer has no `<|im_end|>`, so the 27B's fallback does not apply.
pub fn eos_ids(ckpt: &std::path::Path) -> anyhow::Result<Vec<TokenId>> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        ckpt.join("generation_config.json"),
    )?)?;
    let out: Vec<TokenId> = match &v["eos_token_id"] {
        serde_json::Value::Number(n) => n.as_u64().map(|x| vec![x as TokenId]).unwrap_or_default(),
        serde_json::Value::Array(a) => a
            .iter()
            .filter_map(|x| x.as_u64().map(|x| x as TokenId))
            .collect(),
        _ => Vec::new(),
    };
    anyhow::ensure!(
        !out.is_empty(),
        "generation_config.json names no eos_token_id"
    );
    Ok(out)
}

/// INK20: `--ink-mtp-batch B:K,...` parsed (a live count takes the largest entry <= it; K 0 =
/// plain), sorted by B.
pub fn batch_table(s: &str) -> anyhow::Result<Vec<(usize, u64)>> {
    let mut btable: Vec<(usize, u64)> = Vec::new();
    for e in s.split(',').filter(|e| !e.is_empty() && *e != "off") {
        let (b, k) = e
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("--ink-mtp-batch {e:?}: B:K"))?;
        let (b, k): (usize, u64) = (b.parse()?, k.parse()?);
        anyhow::ensure!(
            (2..=8).contains(&b) && k <= 7,
            "--ink-mtp-batch {e:?}: B 2..8, K 0..7"
        );
        btable.push((b, k));
    }
    btable.sort_unstable();
    Ok(btable)
}

/// INK33: `--ink-mtp-gate` parsed -- `prod:TAU:DMIN[:FIRST]` / `each:TAU:DMIN[:FIRST]` ->
/// (product?, TAU, DMIN, FIRST); empty / `off` = None (the fixed depth, the control). FIRST (0 =
/// none): a round that starts before the sequence's FIRST-th output verifies DMIN drafts only (the
/// first visible text rides a short round: the TTFT rule).
pub fn mtp_gate(s: &str) -> anyhow::Result<Option<(bool, f32, u64, u64)>> {
    if s.is_empty() || s == "off" {
        return Ok(None);
    }
    let p: Vec<&str> = s.split(':').collect();
    anyhow::ensure!(
        (p.len() == 3 || p.len() == 4) && matches!(p[0], "prod" | "each"),
        "--ink-mtp-gate {s:?}: prod:TAU:DMIN[:FIRST] | each:TAU:DMIN[:FIRST] | off"
    );
    let (tau, dmin): (f32, u64) = (p[1].parse()?, p[2].parse()?);
    let first: u64 = p.get(3).map_or(Ok(0), |x| x.parse())?;
    anyhow::ensure!(
        (0.0..=1.0).contains(&tau) && (1..=7).contains(&dmin),
        "--ink-mtp-gate {s:?}: TAU 0..1, DMIN 1..7"
    );
    Ok(Some((p[0] == "prod", tau, dmin, first)))
}

/// INK38: `--ink-mtp-bgate` parsed -- `prod|each:TAU:DMIN:AGG` (AGG max | min | mean) ->
/// (product?, TAU, DMIN, AGG); `off` / empty = None (the batched table's fixed depth, the control).
pub fn mtp_bgate(s: &str) -> anyhow::Result<Option<(bool, f32, u64, BAgg)>> {
    if s.is_empty() || s == "off" {
        return Ok(None);
    }
    let p: Vec<&str> = s.split(':').collect();
    anyhow::ensure!(
        p.len() == 4 && matches!(p[0], "prod" | "each"),
        "--ink-mtp-bgate {s:?}: prod:TAU:DMIN:AGG | each:TAU:DMIN:AGG | off"
    );
    let (tau, dmin): (f32, u64) = (p[1].parse()?, p[2].parse()?);
    anyhow::ensure!(
        (0.0..=1.0).contains(&tau) && (1..=7).contains(&dmin),
        "--ink-mtp-bgate {s:?}: TAU 0..1, DMIN 1..7"
    );
    let agg = match p[3] {
        "max" => BAgg::Max,
        "min" => BAgg::Min,
        "mean" => BAgg::Mean,
        o => anyhow::bail!("--ink-mtp-bgate {s:?}: AGG {o:?}: max | min | mean"),
    };
    Ok(Some((p[0] == "prod", tau, dmin, agg)))
}

/// INK38: how a batched round's one verify depth follows its sequences' gated depths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BAgg {
    Max,
    Min,
    /// the rounded-up mean
    Mean,
}

/// INK38: a batched round's verify depth from every sequence's gated depth.
pub fn bgate_depth(ds: &[u64], agg: BAgg) -> u64 {
    match agg {
        BAgg::Max => ds.iter().copied().max().unwrap_or(0),
        BAgg::Min => ds.iter().copied().min().unwrap_or(0),
        BAgg::Mean => (ds.iter().sum::<u64>()).div_ceil(ds.len().max(1) as u64),
    }
}

/// INK38: the deepest batched depth `nb` live sequences can verify in one multi verify
/// (`MV_ROWS` rows: nb x (K + 1) <= 32).
pub fn mv_depth_cap(nb: u64) -> u64 {
    (super::MV_ROWS / nb.max(1)).saturating_sub(1)
}

/// INK33: the gated depth -- the leading drafts whose running product (or own probability)
/// clears `tau`, clamped to `[dmin, p1.len()]`.
pub fn gated_depth(p1: &[f32], prod: bool, tau: f32, dmin: u64) -> u64 {
    let mut d = 0u64;
    let mut c = 1.0f32;
    for &p in p1 {
        c = if prod { c * p } else { p };
        if c < tau {
            break;
        }
        d += 1;
    }
    d.max(dmin).min(p1.len() as u64)
}

/// The scheduler's geometry: decode buckets 1..=seqs at width 1, one prefill rung of
/// [`SERVE_CHUNK`] rows, KV for `seqs` sequences of `max_ctx`. INK13: with the drafter
/// (`mtp` = K > 0) every batch also has a width-K bucket and the width rule is the batch table
/// `1:K,2:1` (width K at one running sequence, 1 above; the width-K buckets above batch 1 exist
/// only for the scheduler's floor: a step after a K + 1-token commit; the executor drafts only
/// when ONE sequence is live).
#[cfg(test)]
pub fn scheduler_config(max_ctx: u32, seqs: u32, mtp: u32) -> anyhow::Result<SchedulerConfig> {
    scheduler_config_b(max_ctx, seqs, mtp, None, mtp)
}

/// INK20: `batch` = `--ink-mtp-batch` on: the width cap at >= 2 live sequences is the batch
/// table's largest K (the executor's table decides the drafts a step, 0 = plain decode); INK33:
/// `Some(bk)` carries that K (it was `mtp` until INK33 -- identical at the default table, where
/// every K is `--ink-mtp`'s 2; a deeper `--ink-mtp` for one live sequence no longer widens the
/// batched steps' reservation).
#[cfg(test)]
pub fn scheduler_config_b(
    max_ctx: u32,
    seqs: u32,
    mtp: u32,
    batch: Option<u32>,
    // INK32 (`--ink-ngram on`): the width at ONE live sequence (a copy round verifies up to
    // `--ink-ngram-depth` drafts); = `mtp` without the arm
    w1: u32,
) -> anyhow::Result<SchedulerConfig> {
    scheduler_config_ch(max_ctx, seqs, mtp, batch, w1, SERVE_CHUNK)
}

/// INK43: [`scheduler_config_b`] with the served prefill chunk `chunk` (`--ink-p43 ch<N>`).
pub fn scheduler_config_ch(
    max_ctx: u32,
    seqs: u32,
    mtp: u32,
    batch: Option<u32>,
    w1: u32,
    chunk: u32,
) -> anyhow::Result<SchedulerConfig> {
    // 48 visits x (K + V) x 4 KV heads x 128 x bf16 a rank
    let kv = KvGeometry {
        block_tokens: 16,
        bytes_per_token_per_rank: 48 * 2 * 4 * 128 * 2,
    };
    let mut buckets: Vec<Bucket> = (1..=seqs)
        .map(|b| Bucket::decode(b, 1, max_ctx, true))
        .collect();
    if mtp > 1 {
        buckets.extend((1..=seqs).map(|b| Bucket::decode(b, mtp, max_ctx, true)));
    }
    let bw = batch.map(|k| k.min(mtp));
    if let Some(k) = bw.filter(|&k| k > 1 && k != mtp) {
        buckets.extend((1..=seqs).map(|b| Bucket::decode(b, k, max_ctx, true)));
    }
    let w1 = w1.max(mtp);
    if mtp > 0 && w1 > mtp {
        buckets.extend((1..=seqs).map(|b| Bucket::decode(b, w1, max_ctx, true)));
    }
    buckets.push(Bucket::prefill(chunk, max_ctx, true));
    let kv_tokens = u64::from(seqs) * u64::from(max_ctx);
    let mk = |gdn: u64| {
        SchedulerConfig::for_mock(
            buckets.clone(),
            2,
            1 << 20,
            kv,
            ArenaBytes {
                kv_per_rank: kv_tokens * kv.bytes_per_token_per_rank,
                gdn_state_per_rank: gdn,
            },
            max_ctx as usize,
            "inkling2:ink8".into(),
        )
    };
    let per_seq = mk(1 << 40)?.gdn_bytes_per_seq();
    let mut c = mk(u64::from(seqs) * per_seq)?;
    if mtp > 0 {
        let t = saber_sched::WidthByBatch::new([(1, w1), (2, bw.unwrap_or(1).max(1))])
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        c.width = c
            .width
            .clone()
            .with_rule(saber_sched::WidthRule::ByBatch(t));
    }
    Ok(c)
}

struct Started {
    jobs: Receiver<Job>,
    join: std::thread::JoinHandle<anyhow::Result<()>>,
}

fn start(f: &InkFlags, ckpt: &std::path::Path, port: u16, chunk: u32) -> anyhow::Result<Started> {
    use anyhow::Context;
    let tokenizer = Arc::new(
        saber_model::Tokenizer::from_file(ckpt.join("tokenizer.json"))
            .context("the checkpoint's tokenizer.json")?,
    );
    let out = f
        .ink_out
        .clone()
        .context("--ink-serve needs --ink-out (the template + the serve files)")?;
    std::fs::create_dir_all(&out)?;
    let tpath = out.join("chat_template.ink8.jinja");
    std::fs::write(&tpath, template_text(ckpt)?)?;
    let renderer = Arc::new(
        saber_serve::state::TemplateRenderer::from_file(&tpath)
            .with_context(|| format!("loading {}", tpath.display()))?,
    );
    let eos = eos_ids(ckpt)?;
    let sampling = crate::flashnext::serve::default_sampling(ckpt)?;
    let ctx = u32::try_from(f.ink_serve_ctx)?;
    let mtp = if f.ink_serve_graph { f.mtp() } else { 0 };
    let batch = if mtp > 0 && !f.ink_mtp_batch.is_empty() {
        // an empty table (`off`) keeps the pre-INK33 reservation (width `--ink-mtp` above one
        // live sequence) so the control's scheduler is unchanged
        let t = batch_table(&f.ink_mtp_batch)?;
        Some(u32::try_from(t.iter().map(|e| e.1).max().unwrap_or(mtp))?)
    } else {
        None
    };
    // INK32: a copy round at one live sequence verifies up to `--ink-ngram-depth` drafts
    let ng = super::ngram::NgramOpt::parse(
        &f.ink_ngram,
        f.ink_ngram_min,
        f.ink_ngram_max,
        f.ink_ngram_depth,
    )?;
    let w1 = match ng {
        Some(o) if mtp > 0 => u32::try_from(o.depth)?,
        _ => u32::try_from(mtp)?,
    };
    let mut config =
        scheduler_config_ch(ctx, f.ink_serve_seqs, u32::try_from(mtp)?, batch, w1, chunk)?;
    let yield_steps = p54_yield(&f.ink54)?;
    if yield_steps > 0 {
        config.prefill = config.prefill.clone().with_yield_steps(yield_steps);
        println!(
            "INK54 boot: --ink-p54 {:?}: {yield_steps} decode-only steps after a chunk beside decoders",
            f.ink54
        );
    }
    let model = ModelInfo {
        served_model_name: f.ink_served_name.clone(),
        aliases: Vec::new(),
        max_model_len: ctx as usize,
        root: ckpt.display().to_string(),
        eos_token_ids: eos.clone(),
    };
    let mut ledger = Ledger::new();
    for rank in 0..2 {
        ledger.declare(Pool::Vram(rank), 32 << 30)?;
    }
    ledger.declare(Pool::HostPinned, 4 << 30)?;
    ledger.declare(Pool::HostPageable, 48 << 30)?;
    ledger.declare(Pool::Nvme, 64 << 30)?;
    let (tx, jobs) = channel();
    let scheduler = Arc::new(Scheduler::new(config, InkExec { tx }, model, ledger)?);
    let parsers = saber_serve::state::ParserFlags {
        enable_auto_tool_choice: Some(false),
        tool_call_parser: Some(String::new()),
        reasoning_parser: Some(String::new()),
        ..Default::default()
    }
    .resolve()
    .map_err(anyhow::Error::msg)?;
    println!(
        "inkserve: template {} (the checkpoint's, generation markers stripped; reasoning_effort \
         none..max, default 0.9), eos {eos:?}, default sampling {sampling:?}, no tool / reasoning \
         parser, {} sequence(s) x {ctx} tokens, prefill chunk {chunk} (INK43 `--ink-p43 {}`), decode {}, drafter {}",
        tpath.display(),
        f.ink_serve_seqs,
        f.ink43,
        if f.ink_serve_graph {
            "one captured graph a bucket (1 / 2 / 4 / 8 rows)"
        } else {
            "eager"
        },
        if mtp == 0 {
            "off (--ink-mtp 0)".to_string()
        } else {
            format!("--ink-mtp {mtp} (one live sequence; width 1 above)")
        }
    );
    let steps = saber_sched::spawn_step_loop(Arc::clone(&scheduler));
    let state = saber_serve::state::ServerState::builder(
        Arc::clone(&scheduler) as Arc<dyn saber_serve::engine::Engine>,
        Arc::new(saber_serve::detok::ModelCodec::new(Arc::clone(&tokenizer))),
        renderer,
    )
    .parsers(&parsers)
    .default_sampling(sampling)
    .max_logprobs(20)
    .build();
    let join = std::thread::Builder::new().name("ink-http".into()).spawn(
        move || -> anyhow::Result<()> {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("building the tokio runtime")?;
            let addr = format!("0.0.0.0:{port}");
            runtime.block_on(async move {
                let listener = tokio::net::TcpListener::bind(&addr)
                    .await
                    .with_context(|| format!("binding {addr}"))?;
                println!(
                    "inkserve listening http://{}/v1  (SIGINT or SIGTERM to stop)",
                    listener.local_addr()?
                );
                saber_serve::server::serve_with_shutdown(
                    state,
                    listener,
                    crate::engine::shutdown_or_fatal(),
                )
                .await
                .context("serving")
            })?;
            match saber_sched::stop_with_deadline(scheduler, steps, Duration::from_secs(30)) {
                Ok(ran) => println!("inkserve stopped: {ran} engine steps"),
                Err(e) => println!("inkserve stuck at stop: {e}"),
            }
            Ok(())
        },
    )?;
    Ok(Started { jobs, join })
}

/// One rank's serve state.
struct Eng<'a, 'f, 'h, 'm> {
    c: &'a mut Card<'h, 'm>,
    fw: &'a Forward<'f>,
    pk: &'a PackReader,
    cache: Cache,
    /// one captured graph a bucket (index log2 of 1 / 2 / 4 / 8): (graph, exec, logits)
    graphs: Vec<Option<(saber_hip::Graph, saber_hip::GraphExec, u64)>>,
    emb_in: u64,
    use_graph: bool,
    rank: u32,
    pub launches_eager: u64,
    /// INK11: the seam's buffers (rank 0): per-chunk partials, the rows, the rows' temperatures;
    /// `seam` = run it after a decode step (`--ink-sampler seam|check`), `dense` = read the rows
    seam: bool,
    dense: bool,
    seam_part: u64,
    seam_out: u64,
    seam_temp: u64,
    /// the last decode step's logits (address, row stride f32) for a dense fallback
    last_lg: (u64, u64),
    /// INK13: the drafter (rank 0) + the captured verify chunks by (slot, rows) + the last
    /// verify's key (its rollback table)
    dr: Option<super::draft::Drafter>,
    vgs: BTreeMap<(u32, u64), super::draft::VGraph>,
    last_v: Option<(u32, u64)>,
    /// rank 0: tap the next prefill / verify into the drafter's planes
    tap: bool,
    /// INK16: `--ink-first-token seam` (and the seam sampler is on)
    first_seam: bool,
    /// INK20 (`--ink-mtp-batch`): the multi verify graphs by (B, width), the last one's key,
    /// its embeddings buffer; `btap` = every prefill / decode taps the drafter (rank 0)
    mvgs: BTreeMap<(u64, u64), super::draft::VGraph>,
    last_m: Option<(u64, u64)>,
    emb_m: u64,
    btap: bool,
    /// INK22 (`--ink-batch-append`): 0 = `each`, 1 = `graph`, 2 = `check`; the served context
    /// (a row appends only with 8 rows of room); the last plain step's appended rows
    bapp: u8,
    bctx: u64,
    app_rows: Vec<Option<(usize, u64, bool)>>,
    /// INK23: the prefix nodes (`--ink-prefix-cache`), both ranks
    pc: Option<super::pcache::Store>,
    /// INK26: `--ink-prefix-cache-verify`
    pc_verify: bool,
    /// INK27: `--ink-prefix-cache-sync` (a two-rank barrier after every node copy)
    pc_sync: bool,
    /// INK43: `--ink-p43-trace` -- every prefill phase drained by a stream sync and timed (ms):
    /// the forward, the seam / last-row read, the drafter append of the LAST applied chunk
    p43_tr: bool,
    p43_t: [f64; 3],
}

/// INK11: what a decode step hands the sampler (rank 0).
#[derive(Default)]
struct DecOut {
    dense: Vec<u8>,
    seam: Vec<u8>,
}

impl Eng<'_, '_, '_, '_> {
    /// Apply one command; rank 0 gets the logits row (`VOCAB` f32), rank 1 nothing.
    /// Apply one command; rank 0 gets the logits rows (`VOCAB` f32 each: one for a prefill, one
    /// a live row for a decode step), rank 1 nothing.
    fn apply(&mut self, cmd: &Cmd) -> anyhow::Result<Vec<u8>> {
        Ok(self.apply2(cmd, &[])?.dense)
    }

    /// INK11: apply with the decode rows' temperatures (rank 0's seam) -> the dense rows and/or
    /// the seam rows.
    fn apply2(&mut self, cmd: &Cmd, temps: &[f32]) -> anyhow::Result<DecOut> {
        match *cmd {
            Cmd::Prefill {
                slot,
                ref ids,
                pos0,
            } => {
                let fwd = Forward {
                    pos0,
                    graph: false,
                    slot: u64::from(slot),
                    batch: false,
                    sched: self.fw.sched.clone(),
                    ..*self.fw
                };
                // INK13: rank 0 taps the chunk into the drafter's planes and appends its rows
                let tap = self.tap && self.dr.is_some();
                if tap {
                    let d = self.dr.as_ref().context("dr")?;
                    self.c.tapdev = Some((d.taps, d.tcap * D * 4));
                }
                // INK43 trace: the phases' host clock, each drained (a diagnostic)
                let t43 = std::time::Instant::now();
                let mut t43f = t43;
                // INK16: `--ink-first-token seam` -- the chunk's last row through the device seam
                // (rank 0; rank 1 synchronises), the dense row only when the seam refuses
                let (dense, seam) = if self.first_seam {
                    let r = super::pass_lg(
                        self.c,
                        &fwd,
                        self.pk,
                        Some(&self.cache),
                        ids,
                        ids.len() as u64 - 1,
                    );
                    self.c.tapdev = None;
                    let (lg, ld) = r?;
                    if self.p43_tr {
                        self.c.hip.sync_stream(&self.c.stream)?;
                        t43f = std::time::Instant::now();
                    }
                    if self.rank == 0 {
                        let v = temps.first().copied().unwrap_or(1.0);
                        let mut tb = Vec::with_capacity(32);
                        for _ in 0..8 {
                            tb.extend_from_slice(&(if v > 0.0 { v } else { 1.0 }).to_le_bytes());
                        }
                        self.c.write(self.seam_temp, &tb)?;
                    }
                    let o = self.outs(lg, ld, 1)?;
                    (o.dense, o.seam)
                } else {
                    let r = super::pass(
                        self.c,
                        &fwd,
                        self.pk,
                        Some(&self.cache),
                        ids,
                        ids.len() as u64 - 1,
                    );
                    self.c.tapdev = None;
                    t43f = std::time::Instant::now();
                    (r?, Vec::new())
                };
                let t43o = std::time::Instant::now();
                if tap {
                    let d = self.dr.as_mut().context("dr")?;
                    if pos0 == 0 {
                        d.reset_ctx();
                    }
                    d.append(self.c, ids.len() as u64, pos0)?;
                }
                if self.p43_tr {
                    self.c.hip.sync_stream(&self.c.stream)?;
                    let ms =
                        |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
                    let t43a = std::time::Instant::now();
                    self.p43_t = [ms(t43, t43f), ms(t43f, t43o), ms(t43o, t43a)];
                }
                Ok(DecOut { dense, seam })
            }
            Cmd::Verify { slot, ref ids, pos } => {
                let t = ids.len() as u64;
                anyhow::ensure!((2..=8).contains(&t), "a verify of {t} rows");
                if self.seam && self.rank == 0 {
                    let mut tb = Vec::with_capacity(32);
                    for i in 0..8 {
                        let v = temps.get(i).copied().unwrap_or(1.0);
                        tb.extend_from_slice(&(if v > 0.0 { v } else { 1.0 }).to_le_bytes());
                    }
                    self.c.write(self.seam_temp, &tb)?;
                }
                let key = (slot, t);
                if !self.vgs.contains_key(&key) {
                    let taps = if self.rank == 0 {
                        self.dr.as_ref().map(|d| (d.taps, d.tcap * D * 4))
                    } else {
                        None
                    };
                    let vg = super::capture_verify(
                        self.c,
                        self.fw,
                        &self.cache,
                        self.emb_in,
                        t,
                        u64::from(slot),
                        taps,
                    )?;
                    println!(
                        "inkserve rank{}: slot {slot} verify T {t} graph captured ({} launches, {} nodes, {} convs logged)",
                        self.rank, vg.launches, vg.nodes, vg.n
                    );
                    self.vgs.insert(key, vg);
                }
                let vg = &self.vgs[&key];
                super::replay_verify(
                    self.c,
                    self.pk,
                    &self.cache,
                    self.emb_in,
                    vg,
                    ids,
                    pos,
                    u64::from(slot),
                )?;
                let (lg, ld) = (vg.lg, vg.ld);
                self.last_v = Some(key);
                self.outs(lg, ld, t as usize)
            }
            Cmd::Commit { keep } => {
                let key = self.last_v.take().context("a commit without a verify")?;
                let vg = &self.vgs[&key];
                self.c.commit_tab(vg, keep)?;
                Ok(DecOut::default())
            }
            Cmd::Decode { ref rows } => {
                let n = rows.len();
                anyhow::ensure!((1..=8).contains(&n), "a decode step of {n} rows");
                let bucket = n.next_power_of_two();
                let bi = bucket.trailing_zeros() as usize;
                let pad = u32::try_from(self.cache.seqs)?;
                let mut emb = Vec::with_capacity(bucket * D as usize * 4);
                let mut rs = Vec::with_capacity(bucket * 4);
                let mut rp = Vec::with_capacity(bucket * 4);
                let toks: Vec<u32> = rows.iter().map(|r| r.1).collect();
                emb.extend(self.pk.embed_rows(&toks)?);
                emb.resize(bucket * D as usize * 4, 0);
                for i in 0..bucket {
                    let (sl, pos) = rows.get(i).map_or((pad, 0), |r| (r.0, r.2));
                    rs.extend_from_slice(&i32::try_from(sl)?.to_le_bytes());
                    rp.extend_from_slice(&i32::try_from(pos)?.to_le_bytes());
                }
                self.c.write(self.emb_in, &emb)?;
                self.c.write(self.cache.rseq_b, &rs)?;
                self.c.write(self.cache.rpos_b, &rp)?;
                if self.seam && self.rank == 0 {
                    let mut tb = Vec::with_capacity(32);
                    for i in 0..8 {
                        let t = temps.get(i).copied().unwrap_or(1.0);
                        tb.extend_from_slice(&(if t > 0.0 { t } else { 1.0 }).to_le_bytes());
                    }
                    self.c.write(self.seam_temp, &tb)?;
                }
                let fwd = Forward {
                    pos0: 1,
                    graph: false,
                    slot: 0,
                    batch: true,
                    sched: self.fw.sched.clone(),
                    ..*self.fw
                };
                let t = bucket as u64;
                if !self.use_graph {
                    let l0 = self.c.k.launches;
                    self.c.reset(Reg::Pass)?;
                    let (lg, ld) = fwd.run(self.c, Some(&self.cache), self.emb_in, t, 0)?;
                    self.launches_eager = self.c.k.launches - l0;
                    return self.outs(lg, ld, n);
                }
                // INK22 (`graph` / `check`): the rows the step's own graph appends to the drafter
                self.app_rows.clear();
                if self.btap && self.bapp > 0 {
                    if let Some(d) = self.dr.as_ref() {
                        self.app_rows = rows
                            .iter()
                            .map(|r| {
                                let (sl, pos) = (r.0 as usize, r.2);
                                (d.nctx_of(sl) == pos && pos + 8 <= self.bctx)
                                    .then_some((sl, pos, true))
                            })
                            .collect();
                        d.write_atab(self.c, &self.app_rows)?;
                    }
                }
                if self.graphs[bi].is_none() {
                    self.c.hip.sync_stream(&self.c.stream)?;
                    // INK20: under `--ink-mtp-batch` a decode step taps the drafter (rank 0) so
                    // every sequence's context follows its committed rows
                    if self.btap {
                        if let Some(d) = self.dr.as_ref() {
                            self.c.tapdev = Some((d.taps, d.tcap * D * 4));
                        }
                    }
                    self.c
                        .hip
                        .begin_capture(&self.c.stream, saber_hip::StreamCaptureMode::ThreadLocal)?;
                    self.c.reset(Reg::Pass)?;
                    let cap = fwd.run(self.c, Some(&self.cache), self.emb_in, t, 0);
                    self.c.tapdev = None;
                    let capa = match (self.btap && self.bapp > 0, self.dr.as_ref()) {
                        (true, Some(d)) => d.append_batch(self.c, t),
                        _ => Ok(()),
                    };
                    let g = self.c.hip.end_capture(&self.c.stream)?;
                    let (lg, _) = cap?;
                    capa?;
                    let e = self.c.hip.instantiate(&g)?;
                    println!(
                        "inkserve rank{}: b{bucket} decode graph captured ({} nodes)",
                        self.rank,
                        self.c.hip.graph_node_count(&g)?
                    );
                    self.graphs[bi] = Some((g, e, lg));
                }
                let (_, e, lg) = self.graphs[bi].as_ref().expect("captured");
                let lg = *lg;
                self.c.hip.graph_launch(e, &self.c.stream)?;
                // the logits row stride: the gathered pair (2 halves) or the one card's whole head
                let ld: u64 = if self.fw.peer.is_some() {
                    2 * self.fw.w.head[0].rows()
                } else {
                    self.fw.w.head.iter().map(|h| h.shape[0]).sum()
                };
                self.outs(lg, ld, n)
            }
            Cmd::VerifyM { ref rows } => {
                let b = rows.len() as u64;
                let w = rows.first().map_or(0, |r| r.1.len()) as u64;
                anyhow::ensure!(
                    rows.iter().all(|r| r.1.len() as u64 == w)
                        && (2..=8).contains(&w)
                        && b * w <= super::MV_ROWS,
                    "a multi verify of uneven / {w}-row sequences"
                );
                if self.seam && self.rank == 0 {
                    let tb: Vec<u8> = (0..super::MV_ROWS as usize)
                        .flat_map(|i| {
                            let v = temps.get(i).copied().unwrap_or(1.0);
                            (if v > 0.0 { v } else { 1.0 }).to_le_bytes()
                        })
                        .collect();
                    self.c.write(self.seam_temp, &tb)?;
                }
                let key = (b, w);
                if !self.mvgs.contains_key(&key) {
                    let taps = if self.rank == 0 {
                        self.dr.as_ref().map(|d| (d.taps, d.tcap * D * 4))
                    } else {
                        None
                    };
                    let vg = super::capture_mverify(
                        self.c,
                        self.fw,
                        &self.cache,
                        self.emb_m,
                        b,
                        w,
                        taps,
                    )?;
                    println!(
                        "inkserve rank{}: multi verify {b} x {w} graph captured ({} launches, {} nodes)",
                        self.rank, vg.launches, vg.nodes
                    );
                    self.mvgs.insert(key, vg);
                }
                let ids: Vec<u32> = rows.iter().flat_map(|r| r.1.iter().copied()).collect();
                let words = super::mverify_words(
                    &rows
                        .iter()
                        .map(|r| (u64::from(r.0), r.2))
                        .collect::<Vec<_>>(),
                    w,
                )?;
                self.c.write(self.emb_m, &self.pk.embed_rows(&ids)?)?;
                self.c.write(self.cache.rseq_m, &words)?;
                let vg = &self.mvgs[&key];
                self.c.hip.graph_launch(&vg.e, &self.c.stream)?;
                let (lg, ld) = (vg.lg, vg.ld);
                self.last_m = Some(key);
                self.outs(lg, ld, (b * w) as usize)
            }
            Cmd::CommitM { ref keeps } => {
                let key = self
                    .last_m
                    .take()
                    .context("a commit without a multi verify")?;
                let vg = &self.mvgs[&key];
                super::commit_mverify(self.c, &self.cache, vg, key.1, keeps)?;
                Ok(DecOut::default())
            }
            Cmd::PcStore {
                slot,
                node,
                len,
                draft,
            }
            | Cmd::PcLoad {
                slot,
                node,
                len,
                draft,
            } => {
                let out = matches!(cmd, Cmd::PcStore { .. });
                let pc = self
                    .pc
                    .as_mut()
                    .context("a prefix node with --ink-prefix-cache 0")?;
                let dr = if draft && self.rank == 0 {
                    self.dr.as_ref().context("dr")?.ctx_regions(len)
                } else {
                    Vec::new()
                };
                let cps = pc.copies(&self.cache, u64::from(slot), node, len, out, &dr)?;
                super::pcache::run(self.c, &cps)?;
                if self.pc_verify {
                    let (bad, moved) = super::pcache::verify(self.c, pc, &cps, out)?;
                    if bad > 0 || moved > 0 {
                        println!(
                            "INK26 pcverify rank{} MISS: {} slot {slot} node {node} len {len}: {bad} copies differ, {moved} node regions changed since the store",
                            self.rank,
                            if out { "store" } else { "load" }
                        );
                    }
                }
                if !out && draft && self.rank == 0 {
                    self.dr.as_mut().context("dr")?.set_nctx(len);
                }
                if self.pc_sync {
                    // INK27: both ranks' copies (and everything before them) complete before
                    // either rank queues the next forward
                    let pr = self
                        .fw
                        .peer
                        .context("--ink-prefix-cache-sync needs the pair")?;
                    pr.barrier(self.c, 0x1027_0000 | u64::from(out))?;
                }
                Ok(DecOut::default())
            }
            Cmd::Stop => Ok(DecOut::default()),
        }
    }

    /// INK11: after a decode forward (queued, not yet synchronised): rank 0 runs the seam and / or
    /// reads the dense rows.
    fn outs(&mut self, lg: u64, ld: u64, n: usize) -> anyhow::Result<DecOut> {
        self.last_lg = (lg, ld);
        let mut o = DecOut::default();
        if self.rank == 0 && self.seam {
            self.c.launch(
                "ink_seam_p1",
                [SEAM_NCH, n as u64, 1],
                256,
                &[
                    A::P(lg),
                    A::L(i64::try_from(ld)?),
                    A::I(i32::try_from(VOCAB)?),
                    A::P(self.seam_temp),
                    A::P(self.seam_part),
                ],
            )?;
            self.c.launch(
                "ink_seam_p2",
                [n as u64, 1, 1],
                1024,
                &[A::P(self.seam_part), A::P(self.seam_out)],
            )?;
            o.seam = self.c.read(self.seam_out, n * SEAM_W * 4)?;
        } else {
            self.c.hip.sync_stream(&self.c.stream)?;
        }
        if self.dense {
            o.dense = self.rows(lg, ld, n)?;
        }
        Ok(o)
    }

    /// INK11: one dense row of the last decode step (a seam fallback).
    fn dense_row(&self, k: usize) -> anyhow::Result<Vec<f32>> {
        let (lg, ld) = self.last_lg;
        Ok(super::f32s(
            &self.c.read(lg + k as u64 * ld * 4, VOCAB * 4)?,
        ))
    }

    /// Rank 0: the first `n` logits rows (row stride `ld` f32), VOCAB each.
    fn rows(&self, lg: u64, ld: u64, n: usize) -> anyhow::Result<Vec<u8>> {
        if self.rank != 0 {
            return Ok(Vec::new());
        }
        // INK10 task 3: ONE read of the n rows' span (row stride ld), then the rows sliced out
        let span = (n as u64 - 1) * ld * 4 + VOCAB as u64 * 4;
        let all = self.c.read(lg, span as usize)?;
        let mut out = Vec::with_capacity(n * VOCAB * 4);
        for r in 0..n {
            let o = r * ld as usize * 4;
            out.extend_from_slice(&all[o..o + VOCAB * 4]);
        }
        Ok(out)
    }
}

fn sample_row(
    sampler: &Sampler,
    row: &mut [f32],
    s: &SlotSampling,
) -> (TokenId, Option<SampledLogprobs>) {
    let params = RowParams::resolve(&s.params, VOCAB);
    let ctx = match &s.penalties {
        Some(h) => PenaltyContext {
            prompt: &h.prompt,
            output: &h.output,
        },
        None => PenaltyContext::default(),
    };
    let mut noise: Box<dyn NoiseSource> = match s.params.seed {
        Some(seed) => Box::new(SeededNoise::new(seed, s.position, s.choice_index)),
        None => Box::new(EntropyNoise::new()),
    };
    match sampler.sample_dense(row, &params, &ctx, noise.as_mut()) {
        Ok(x) => (x.token_id, x.logprobs),
        Err(e) => {
            eprintln!("inkserve: sampler refused ({e:?}); committing the argmax");
            (super::argmax(row), None)
        }
    }
}

/// INK11: the seam row `k` of `ink_seam_p2`'s output as the sampler's [`DeviceRow`].
fn seam_row(b: &[u8], k: usize) -> saber_serve::sampler::DeviceRow {
    let w = |i: usize| {
        let o = (k * SEAM_W + i) * 4;
        [b[o], b[o + 1], b[o + 2], b[o + 3]]
    };
    saber_serve::sampler::DeviceRow {
        top_values: (0..SEAM_K).map(|i| f32::from_le_bytes(w(i))).collect(),
        top_ids: (0..SEAM_K)
            .map(|i| u32::from_le_bytes(w(SEAM_K + i)) as TokenId)
            .collect(),
        logsumexp: f32::from_le_bytes(w(2 * SEAM_K)),
        logsumexp_at_temperature: Some(f32::from_le_bytes(w(2 * SEAM_K + 1))),
        penalised: Vec::new(),
        vocab_size: VOCAB,
    }
}

/// INK11: the temperature the seam's `lse_t` is taken at (the resolved row's; 1 when greedy).
fn seam_temperature(s: &SlotSampling) -> f32 {
    let p = RowParams::resolve(&s.params, VOCAB);
    if p.is_greedy() || p.temperature <= 0.0 {
        1.0
    } else {
        p.temperature
    }
}

/// INK11: sample a row from the seam; `None` = the seam cannot decide it (penalties, or the
/// sampler refused by name) -> the dense row.
fn sample_seam(
    sampler: &Sampler,
    row: &saber_serve::sampler::DeviceRow,
    s: &SlotSampling,
) -> Option<(TokenId, Option<SampledLogprobs>)> {
    let params = RowParams::resolve(&s.params, VOCAB);
    if !params.no_penalties() {
        return None;
    }
    let mut noise: Box<dyn NoiseSource> = match s.params.seed {
        Some(seed) => Box::new(SeededNoise::new(seed, s.position, s.choice_index)),
        None => Box::new(EntropyNoise::new()),
    };
    sampler
        .sample(row, &params, &PenaltyContext::default(), noise.as_mut())
        .ok()
        .map(|x| (x.token_id, x.logprobs))
}

/// Both ranks' serve loop (called on each rank's thread after its weights are on its card).
pub(super) fn rank_serve(
    c: &mut Card<'_, '_>,
    fw: &Forward<'_>,
    pk: &PackReader,
    w: &Weights,
    f: &InkFlags,
    _link: u64,
    draft_dir: &std::path::Path,
) -> anyhow::Result<()> {
    use anyhow::{bail, ensure, Context};
    // INK41: the one-card plane has no peer (rank 0 alone, no command channel, no reduces)
    let peer = fw.peer;
    let rank = peer.map_or(0, |p| p.rank);
    ensure!(
        peer.is_none_or(|p| p.dev),
        "--ink-serve runs on the fused reduce (--ink-reduce p2p)"
    );
    let n = f.ink_serve_seqs as usize;
    ensure!((1..=8).contains(&n), "--ink-serve-seqs 1..8");
    // INK43: `--ink-p43 ch<N>` -- the served chunk; the local ring holds the window + the chunk
    let (chunk, p43_am) = p43_plane(&f.ink43, peer.is_some())?;
    if peer.is_none() && f.ink43 == crate::inkling2::DEFAULT_INK43 && !f.ink43.is_empty() {
        println!(
            "inkserve: INK43 `--ink-p43 {}` STANDS DOWN on the one-card plane (gated on two cards)",
            f.ink43
        );
    }
    let ring = p43_ring(f.ink_local_ring, chunk);
    let cache = Cache::new(c, w, &fw.sched, f.ink_serve_ctx, ring, n as u64)?;
    let emb_in = c.alloc_zero(8 * D * 4)?;
    let (free, total) = c.hip.mem_info()?;
    println!(
        "inkserve rank{}: {n} sequence slot(s) of {} tokens (+ the pad slot; local ring {} rows), {:.2} of {:.2} GiB used",
        rank,
        f.ink_serve_ctx,
        cache.ring_pages * super::PAGE,
        (total - free) as f64 / f64::from(1 << 30),
        total as f64 / f64::from(1 << 30)
    );
    let mut eng = Eng {
        c,
        fw,
        pk,
        cache,
        graphs: (0..4).map(|_| None).collect(),
        emb_in,
        use_graph: f.ink_serve_graph,
        rank,
        launches_eager: 0,
        seam: false,
        dense: true,
        seam_part: 0,
        seam_out: 0,
        seam_temp: 0,
        last_lg: (0, 0),
        dr: None,
        vgs: BTreeMap::new(),
        last_v: None,
        tap: false,
        first_seam: false,
        mvgs: BTreeMap::new(),
        last_m: None,
        emb_m: 0,
        btap: false,
        bapp: 0,
        bctx: f.ink_serve_ctx,
        app_rows: Vec::new(),
        pc: None,
        pc_verify: f.ink_prefix_cache_verify,
        pc_sync: f.ink_prefix_cache_sync,
        p43_tr: f.ink_p43_trace,
        p43_t: [0.0; 3],
    };
    // INK13: the drafter (`--ink-mtp K`): rank 0 holds its weights + one owner sequence's
    // context; both ranks hold the verify history buffer (192 convs x [3 + 8 rows] a verify)
    // INK33: 1..7 (the block drafter's 7 mask rows; the verify chunk's 8 rows); 2 = the default
    ensure!(f.mtp() <= 7, "--ink-mtp {}: 0 (off) or 1..7", f.mtp());
    // the drafter runs on the captured verify: the eager control (`--no-ink-serve-graph`) stands
    // it down by name
    let mtp_k = if eng.use_graph { f.mtp() } else { 0 };
    if f.mtp() > 0 && mtp_k == 0 && rank == 0 {
        println!(
            "inkserve: --ink-mtp {} STANDS DOWN under --no-ink-serve-graph (eager)",
            f.mtp()
        );
    }
    // INK20: `--ink-mtp-batch B:K,...` (a live count takes the largest entry <= it; K 0 = plain)
    let btable = batch_table(&f.ink_mtp_batch)?;
    // INK22: the batched arm needs >= 2 sequence slots (one slot = INK13's owner form)
    let batch = mtp_k > 0 && !btable.is_empty() && n >= 2;
    // INK21: the plain-decode rules (`--ink-batch-plain`)
    let (mut r_first, mut r_rel) = (0u64, false);
    for r in f.ink_batch_plain.split(',').filter(|e| !e.is_empty()) {
        match r {
            "rel" => r_rel = true,
            "none" => {}
            x if x.starts_with("first") && x[5..].parse::<u64>().is_ok_and(|n| n >= 1) => {
                r_first = x[5..].parse()?;
            }
            o => bail!("--ink-batch-plain {o:?}: first<N> | rel | none, joined by ,"),
        }
    }
    // INK22: the plain-step append form (`--ink-batch-append`)
    let mut bapp = 0u8;
    let mut pfmm = false;
    for r in f.ink_batch_append.split(',').filter(|e| !e.is_empty()) {
        match r {
            "each" => bapp = 0,
            "graph" => bapp = 1,
            "check" => bapp = 2,
            "pfmm" => pfmm = true,
            o => bail!("--ink-batch-append {o:?}: each | graph | check [,pfmm]"),
        }
    }
    if pfmm && f.ink_draft_mm != "wmma" {
        pfmm = false;
        if rank == 0 {
            println!("inkserve: --ink-batch-append pfmm STANDS DOWN (--ink-draft-mm base)");
        }
    }
    let (mut n_app_rows, mut n_app_checked, mut n_app_mis) = (0u64, 0u64, 0u64);
    ensure!(
        matches!(f.ink_batch_ahead.as_str(), "on" | "full" | "off"),
        "--ink-batch-ahead {:?}: on | full | off",
        f.ink_batch_ahead
    );
    let mut n_ahead_skip = 0u64;
    let mut released = false;
    let mut n_rel_plain = 0u64;
    if rank == 0 {
        // INK26: the rollback by flag (the pre-flip serve defaults)
        println!(
            "inkserve: INK26 defaults; control: INK22's `{}`, + INK25's: `{}`",
            crate::inkling2::INK22_CONTROL,
            crate::inkling2::INK26_CONTROL
        );
        println!(
            "inkserve: INK29 prefill levers `{}`; control: `{}`",
            crate::inkling2::DEFAULT_INK29,
            crate::inkling2::INK29_CONTROL
        );
        // CONS40: the prefill stack's defaults and their rollback
        println!(
            "inkserve: CONS40 prefill wm floor {} + p35 `{}` p36 `{}` p37 `{}` p40 `{}`; control: `{}`",
            crate::inkling2::DEFAULT_INK_PREFILL_ATTN_FLOOR,
            crate::inkling2::DEFAULT_INK35,
            crate::inkling2::DEFAULT_INK36,
            crate::inkling2::DEFAULT_INK37,
            crate::inkling2::DEFAULT_INK40,
            crate::inkling2::CONS40_CONTROL
        );
        println!(
            "inkserve: INK43 served prefill `{}` (default `{}`); control: `{}`",
            f.ink43,
            crate::inkling2::DEFAULT_INK43,
            crate::inkling2::INK43_CONTROL
        );
        println!(
            "inkserve: INK39 multi-verify levers `{}`; control: `{}`",
            f.ink_p39,
            crate::inkling2::INK39_CONTROL
        );
    }
    if !btable.is_empty() && !batch && rank == 0 {
        println!(
            "inkserve: --ink-mtp-batch STANDS DOWN ({})",
            if mtp_k == 0 {
                "no drafter"
            } else {
                "one sequence slot"
            }
        );
    }
    // INK33: the confidence-gated depth at one live sequence (`--ink-mtp-gate`, an ARM)
    let gate = if mtp_k > 0 {
        mtp_gate(f.mtp_gate_spec())?
    } else {
        None
    };
    if let Some((prod, tau, dmin, first)) = gate {
        ensure!(
            f.draft_vocab() > 0,
            "--ink-mtp-gate needs the pruned draft vocabulary (--ink-draft-vocab N)"
        );
        if rank == 0 {
            println!(
                "INK33 mtp gate: draft at k {mtp_k}, verify the leading drafts with {} top-1 \
                 probability >= {tau}, at least {dmin} (exactly {dmin} before output {first}; \
                 INK34 control: {})",
                if prod { "running-product" } else { "own" },
                crate::inkling2::INK34_MTP_CONTROL
            );
        }
    }
    // INK38: the confidence-gated depth of a batched round (`--ink-mtp-bgate`, an ARM)
    let bgate = if batch {
        mtp_bgate(&f.ink_mtp_bgate)?
    } else {
        None
    };
    if let Some((prod, tau, dmin, agg)) = bgate {
        ensure!(
            f.draft_vocab() > 0,
            "--ink-mtp-bgate needs the pruned draft vocabulary (--ink-draft-vocab N)"
        );
        if rank == 0 {
            println!(
                "INK38 mtp bgate: batched rounds draft at the table's K, each sequence's depth = \
                 the leading drafts with {} top-1 probability >= {tau}, at least {dmin}; the round \
                 verifies their {agg:?} (control: --ink-mtp-bgate off)",
                if prod { "running-product" } else { "own" }
            );
        }
    } else if !f.ink_mtp_bgate.is_empty() && f.ink_mtp_bgate != "off" && rank == 0 {
        println!("inkserve: --ink-mtp-bgate STANDS DOWN (no batched speculation)");
    }
    let mut bg_stat = [0u64; 8];
    let mut bg_rows = 0u64;
    let mut g_stat = [0u64; 8];
    let bk = |live: usize| -> u64 {
        btable
            .iter()
            .rev()
            .find(|e| e.0 <= live)
            .map_or(0, |e| e.1.min(mtp_k))
    };
    if mtp_k > 0 {
        let hcap = 3 * (3 + 8) * D * 4 * 48 * 2 * if batch { 6 } else { 1 };
        let buf = eng.c.alloc_zero(hcap)?;
        let tab = eng.c.alloc_zero(3 * 8 * 512)?;
        eng.c.hist_spare = Some(super::draft::Hist {
            buf,
            cap: hcap,
            tab,
            ..Default::default()
        });
        if rank == 0 {
            eng.dr = Some(super::draft::Drafter::load(
                eng.c,
                pk,
                draft_dir,
                &w.head[0],
                f.ink_serve_ctx,
                u64::from(chunk),
                &{
                    let mut o = super::draft::DraftOpt::of(
                        f.draft_vocab(),
                        &f.ink_draft_gemv,
                        f.draft_layers(),
                    )?
                    .with_mm(&f.ink_draft_mm)?;
                    if batch {
                        o.slots = n as u64;
                        o.pfmm = pfmm;
                    } else if p43_am {
                        // INK43 `am`: the one-slot drafter's prefill append on the matrix cores
                        ensure!(
                            f.ink_draft_mm == "wmma",
                            "--ink-p43 am needs --ink-draft-mm wmma"
                        );
                        o.pfmm = true;
                    }
                    o.conf = gate.is_some();
                    o.mconf = bgate.is_some();
                    o.ks = super::ink49_bits(&f.ink49)?;
                    o.df2 = f.dflash2()?;
                    o
                },
            )?);
            let (free, total) = eng.c.hip.mem_info()?;
            if batch {
                println!("INK20 mtp batch: table {btable:?}, {n} drafter context slots");
            }
            println!(
                "inkserve rank0: drafter loaded (k {mtp_k}), {:.2} of {:.2} GiB used",
                (total - free) as f64 / f64::from(1 << 30),
                total as f64 / f64::from(1 << 30)
            );
        }
    }
    // INK23: the prefix nodes (`--ink-prefix-cache N`), both ranks (rank 0 + the drafter's part)
    if f.ink_prefix_cache > 0 {
        let caps: Vec<u64> = eng
            .dr
            .as_ref()
            .map(|d| d.ctx_regions(d.cap).iter().map(|r| r.1).collect())
            .unwrap_or_default();
        let st =
            super::pcache::Store::new(eng.c, &eng.cache, &fw.sched, f.ink_prefix_cache, &caps)?;
        let (free, total) = eng.c.hip.mem_info()?;
        println!(
            "INK23 pcache rank{}: {} prefix nodes, {:.1} MiB ({} drafter regions), {:.2} of {:.2} GiB used",
            rank,
            st.n,
            st.bytes as f64 / f64::from(1 << 20),
            caps.len(),
            (total - free) as f64 / f64::from(1 << 30),
            total as f64 / f64::from(1 << 30)
        );
        eng.pc = Some(st);
    }
    if batch {
        eng.emb_m = eng.c.alloc_zero(super::MV_ROWS * D * 4)?;
        eng.btap = rank == 0;
        eng.bapp = bapp;
    }
    // INK11: the decode sampler (`--ink-sampler`)
    match f.ink_sampler.as_str() {
        "dense" => {}
        "seam" | "check" => {
            eng.seam = true;
            eng.dense = f.ink_sampler == "check";
            if rank == 0 {
                let r = super::MV_ROWS;
                eng.seam_part = eng.c.alloc_zero(r * SEAM_NCH * SEAM_W as u64 * 4)?;
                eng.seam_out = eng.c.alloc_zero(r * SEAM_W as u64 * 4)?;
                eng.seam_temp = eng.c.alloc_zero(r * 4)?;
            }
        }
        o => bail!("--ink-sampler {o:?}: seam | dense (the control) | check"),
    }
    if rank == 1 {
        eng.dense = false;
    }
    match f.ink_first_token.as_str() {
        "seam" => eng.first_seam = eng.seam,
        "dense" => {}
        o => bail!("--ink-first-token {o:?}: seam | dense (the control)"),
    }
    let (mut n_first_seam, mut n_first_dense) = (0u64, 0u64);
    let (mut n_seam, mut n_fallback, mut n_checked, mut n_mismatch) = (0u64, 0u64, 0u64, 0u64);
    if let Some(peer) = peer.filter(|p| p.rank == 1) {
        loop {
            let b = peer.rows_rx.recv().context("rank 0 gone")?;
            let cmd: Cmd = serde_json::from_slice(&b)?;
            if matches!(cmd, Cmd::Stop) {
                break;
            }
            eng.apply(&cmd)?;
        }
        if let Some(pc) = eng.pc.as_ref().filter(|_| eng.pc_verify) {
            println!(
                "INK26 pcverify rank1: ops {}, copies differing {}, node regions changed {}, regions checked {}",
                pc.vstat[0], pc.vstat[1], pc.vstat[2], pc.vstat[3]
            );
        }
        eng.graphs.clear();
        return Ok(());
    }
    let port = f.ink_serve.context("--ink-serve")?;
    let ckpt = pk.ckpt.clone();
    let st = start(f, &ckpt, port, chunk)?;
    let sampler = Sampler::new();
    let mut slot_of: BTreeMap<SeqId, (u32, u64)> = BTreeMap::new();
    let mut free_slots: Vec<u32> = (0..n as u32).rev().collect();
    let send = |cmd: &Cmd| -> anyhow::Result<()> {
        match peer {
            Some(pr) => pr
                .rows_tx
                .send(serde_json::to_vec(cmd)?)
                .context("rank 1 gone"),
            None => Ok(()),
        }
    };
    let mut steps = 0u64;
    // INK13: the drafter's owner (the one sequence whose every committed row the drafter's
    // context holds) and the round queued for it at the last commit (position, anchor, k)
    let eos = eos_ids(&ckpt)?;
    let ctx = f.ink_serve_ctx;
    let mut owner: Option<SeqId> = None;
    let mut pending: Option<(u64, u32, u64)> = None;
    // INK20: the batched round queued at the last multi commit ((slot, position, anchor) a
    // sequence, k) + the counters by live count (rounds, tokens, drafted)
    #[allow(clippy::type_complexity)]
    let mut pending_m: Option<(Vec<(u32, u64, u32)>, u64)> = None;
    let mut mb_stat: BTreeMap<usize, [u64; 3]> = BTreeMap::new();
    let mut mb_time: Vec<[f64; 4]> = Vec::new();
    let (mut m_rounds, mut m_tokens, mut m_drafted, mut m_lost) = (0u64, 0u64, 0u64, 0u64);
    // INK13: a round's host-clock phases (draft wait, verify + seam, walk + commit + append +
    // the next round queued, the gap since the previous round's end)
    let mut m_time: Vec<[f64; 4]> = Vec::new();
    let mut last_end: Option<std::time::Instant> = None;
    // INK16: a prefill chunk's host-clock phases (rows, the chunk's apply = embed + forward +
    // drafter tap / append + the last row's read, the first token's host sample)
    let mut p_time: Vec<[f64; 3]> = Vec::new();
    // INK43 (`--ink-p43-trace`): every prompt's prefill timeline (admit, its chunks), the last
    // reply's instant (the host gap before the next step)
    let p43_on = f.ink_p43_trace;
    let mut p43_req: BTreeMap<SeqId, (std::time::Instant, String)> = BTreeMap::new();
    let mut p43_reply: Option<std::time::Instant> = None;
    let mut p43_out: Vec<String> = Vec::new();
    // INK22: the server-clock TTFT decomposition (admit -> first chunk -> last chunk's reply ->
    // the 3rd output's reply: the first visible text) and every step's host clock by kind
    #[allow(clippy::type_complexity)]
    let mut seq_n: BTreeMap<
        SeqId,
        (
            std::time::Instant,
            u64,
            Option<std::time::Instant>,
            Option<std::time::Instant>,
        ),
    > = BTreeMap::new();
    let mut ttft_rec: Vec<[f64; 4]> = Vec::new();
    let mut step_rec: Vec<(u8, usize, f64)> = Vec::new();
    // INK23: the prefix cache's node table + each admitted prompt and its plan (rank 0)
    let pc_on = f.ink_prefix_cache > 0;
    let mut pcx = super::pcache::Index::new(f.ink_prefix_cache);
    let pc_user = if pc_on {
        super::pcache::user_token(&ckpt)?
    } else {
        None
    };
    if pc_on {
        println!(
            "INK23 pcache: {} nodes, cut before the last user turn (token {pc_user:?}), min {} tokens, stored at its second sighting",
            f.ink_prefix_cache, f.ink_prefix_min
        );
    }
    let mut pc_prompt: BTreeMap<SeqId, (Vec<TokenId>, super::pcache::Plan)> = BTreeMap::new();
    // INK26 (`--ink-prefix-cache-verify`): each request's schedule -- prompt length + hash, the
    // reload, and the (step kind, rows) of every step that emitted its outputs
    let mut trace: BTreeMap<SeqId, String> = BTreeMap::new();
    let tr_on = f.ink_prefix_cache_verify || f.ink_serve_trace;
    // INK32 (`--ink-ngram on`): every live sequence's tokens (prompt + committed outputs) and
    // their min-gram index; counters [copy rounds, tokens, drafted, rounds whose drafter pass was
    // not queued, batched rows drafted by copy, their tokens]
    let ng = if mtp_k > 0 {
        super::ngram::NgramOpt::parse(
            &f.ink_ngram,
            f.ink_ngram_min,
            f.ink_ngram_max,
            f.ink_ngram_depth,
        )?
    } else {
        None
    };
    if let Some(o) = ng {
        println!(
            "INK32 ngram: copy drafting ON (min {} max {} depth {}; batched rounds {})",
            o.min,
            o.max,
            o.depth,
            if f.ink_ngram_batch { "on" } else { "off" }
        );
    }
    let mut hist: BTreeMap<SeqId, super::ngram::History> = BTreeMap::new();
    let mut ng_stat = [0u64; 6];
    // a copy proposal for history `h`: Some(drafts) when the match qualifies
    let ng_pick = |h: Option<&super::ngram::History>, cap: usize| -> Option<Vec<u32>> {
        let o = ng?;
        let (m, mut d) = h?.propose(o.max, o.depth);
        if m < o.min || d.is_empty() {
            return None;
        }
        d.truncate(super::ngram::depth_for(&o, m).min(cap));
        (!d.is_empty()).then_some(d)
    };
    let res: anyhow::Result<()> = (|| loop {
        let job = match st.jobs.recv_timeout(Duration::from_millis(500)) {
            Ok(j) => j,
            Err(RecvTimeoutError::Timeout) => {
                if st.join.is_finished() {
                    return Ok(());
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => return Ok(()),
        };
        match job {
            Job::Admit(seq) => {
                let Some(s) = free_slots.pop() else {
                    bail!("inkserve: {seq} admitted with every cache busy");
                };
                slot_of.insert(seq, (s, 0));
                seq_n.insert(seq, (std::time::Instant::now(), 0, None, None));
                if p43_on {
                    p43_req.insert(seq, (std::time::Instant::now(), String::new()));
                }
            }
            Job::Prompt(seq, ids) => {
                if tr_on {
                    use std::hash::{Hash as _, Hasher as _};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    ids.hash(&mut h);
                    trace.insert(
                        seq,
                        format!(
                            "prompt {} h {:016x} live {} |",
                            ids.len(),
                            h.finish(),
                            slot_of.len()
                        ),
                    );
                }
                if let Some(o) = ng {
                    let mut h = super::ngram::History::new(o.min);
                    h.push(&ids);
                    hist.insert(seq, h);
                }
                if pc_on {
                    let pl = pcx.admit(&ids, pc_user, f.ink_prefix_min);
                    pc_prompt.insert(seq, (ids, pl));
                }
            }
            Job::Release(seq) => {
                p43_req.remove(&seq);
                if let Some(t) = trace.remove(&seq) {
                    println!("INK26 trace {seq}: {t}");
                }
                seq_n.remove(&seq);
                hist.remove(&seq);
                pc_prompt.remove(&seq);
                if let Some((s, _)) = slot_of.remove(&seq) {
                    free_slots.push(s);
                    released = true;
                }
                if owner == Some(seq) {
                    owner = None;
                    pending = None;
                }
            }
            Job::Step(b, reply) => {
                let ts0 = std::time::Instant::now();
                // INK22: (sequence, outputs this step), the chunks that ended a prompt, the kind
                // (0 prefill, 1 plain, 2 one-sequence round, 3 batched round) and its rows
                let mut emitted: Vec<(SeqId, u64)> = Vec::new();
                let mut pf_last: Vec<SeqId> = Vec::new();
                let mut skind: (u8, usize) = (1, 0);
                // INK32: the sequences whose outputs this step were already added to `hist`
                let mut ng_pushed: Vec<SeqId> = Vec::new();
                let mut out = Vec::with_capacity(b.slots.len());
                // INK9 task 4: the decoding rows of a step run as ONE batched forward; prefill
                // chunks one after another (a chunk is one sequence's rows)
                let mut dec: Vec<(usize, u32, u32, u64)> = Vec::new();
                for (i, sl) in b.slots.iter().enumerate() {
                    let (slot, pos) = *slot_of
                        .get(&sl.seq)
                        .with_context(|| format!("{} not admitted", sl.seq))?;
                    let fed = if sl.chunk.is_some() {
                        sl.input.len() as u64
                    } else {
                        1
                    };
                    ensure!(
                        pos + fed <= f.ink_serve_ctx,
                        "{}: past the cache's {} tokens",
                        sl.seq,
                        f.ink_serve_ctx
                    );
                    out.push(crate::flashnext::serve::seq_step(sl.seq, None, None));
                    match (b.phase, sl.chunk) {
                        (Phase::Prefill, Some(ch)) => {
                            ensure!(
                                u64::from(ch.tokens_done_before) == pos,
                                "{}: chunk after {} tokens, the cache holds {pos}",
                                sl.seq,
                                ch.tokens_done_before
                            );
                            // INK13: a prefill taps the drafter when its sequence is the only
                            // live one (it becomes the owner) or is the owner's next chunk
                            eng.tap = mtp_k > 0
                                && (batch
                                    || slot_of.len() == 1 && (pos == 0 || owner == Some(sl.seq)));
                            if batch {
                                // INK20: every sequence's own drafter context
                                if let Some(d) = eng.dr.as_mut() {
                                    d.select(slot as usize)?;
                                }
                            }
                            let tp0 = std::time::Instant::now();
                            skind = (0, sl.input.len());
                            if pos == 0 {
                                if let Some(e) = seq_n.get_mut(&sl.seq) {
                                    e.2 = Some(tp0);
                                }
                            }
                            let ftemp = seam_temperature(&sl.sampling);
                            // INK23 (`--ink-prefix-cache`): the chunk [pos, end) after the plan --
                            // a reload at the prompt's first chunk (the node's rows need no
                            // forward), the chunk split at the prompt's cut and the node stored
                            // there (a cut never moves a byte: INK23 h1's chunk law)
                            let end = pos + sl.input.len() as u64;
                            let mut lo = pos;
                            let mut store_at = None;
                            if let Some((ids_all, pl)) = pc_prompt.get_mut(&sl.seq) {
                                if let Some((nd, p)) = pl.load {
                                    if pos == 0 {
                                        let has = pcx.nodes[nd as usize]
                                            .as_ref()
                                            .is_some_and(|m| m.draft);
                                        if eng.tap && !has {
                                            // the drafter would hold this sequence: a node
                                            // without its rows is not reloaded (cold)
                                            pcx.skipped_nodraft += 1;
                                            pl.load = None;
                                        } else {
                                            if batch {
                                                if let Some(d) = eng.dr.as_mut() {
                                                    d.select(slot as usize)?;
                                                }
                                            }
                                            let cm = Cmd::PcLoad {
                                                slot,
                                                node: nd,
                                                len: p,
                                                draft: eng.tap,
                                            };
                                            send(&cm)?;
                                            eng.apply(&cm)?;
                                            if let Some(t) = trace.get_mut(&sl.seq) {
                                                t.push_str(&format!(" load {p} tap {}", eng.tap));
                                            }
                                            pcx.touch(nd);
                                            pcx.loads += 1;
                                            pcx.load_rows += p;
                                        }
                                    }
                                    if let Some((_, p)) = pl.load {
                                        lo = lo.max(p).min(end);
                                    }
                                }
                                if let Some(p) = pl.store {
                                    if p >= lo && p <= end && p < ids_all.len() as u64 {
                                        store_at = Some((p, ids_all[..p as usize].to_vec()));
                                    }
                                }
                            }
                            let mut cuts = vec![lo];
                            if let Some((p, _)) = &store_at {
                                if *p > lo && *p < end {
                                    cuts.push(*p);
                                }
                            }
                            cuts.push(end);
                            let mut row: anyhow::Result<DecOut> = Ok(DecOut::default());
                            let mut ran = false;
                            let do_store = |eng: &mut Eng<'_, '_, '_, '_>,
                                            pcx: &mut super::pcache::Index,
                                            p: u64,
                                            ids: Vec<u32>|
                             -> anyhow::Result<()> {
                                let draft = eng.tap
                                    && eng
                                        .dr
                                        .as_ref()
                                        .is_some_and(|d| d.nctx_of(slot as usize) == p);
                                let nd = pcx.victim();
                                let cm = Cmd::PcStore {
                                    slot,
                                    node: nd,
                                    len: p,
                                    draft,
                                };
                                send(&cm)?;
                                eng.apply(&cm)?;
                                pcx.put(nd, ids, draft);
                                Ok(())
                            };
                            if let Some((p, ids)) = store_at.clone() {
                                if p == lo {
                                    do_store(&mut eng, &mut pcx, p, ids)?;
                                    store_at = None;
                                }
                            }
                            for wd in cuts.windows(2) {
                                let (s0, e0) = (wd[0], wd[1]);
                                if s0 >= e0 {
                                    continue;
                                }
                                let cmd = Cmd::Prefill {
                                    slot,
                                    ids: sl.input[(s0 - pos) as usize..(e0 - pos) as usize]
                                        .to_vec(),
                                    pos0: s0,
                                };
                                send(&cmd)?;
                                row = eng.apply2(&cmd, &[ftemp]);
                                ran = true;
                                if row.is_err() {
                                    break;
                                }
                                if let Some((p, ids)) = store_at.clone() {
                                    if p == e0 {
                                        do_store(&mut eng, &mut pcx, p, ids)?;
                                        store_at = None;
                                    }
                                }
                            }
                            ensure!(
                                ran || !ch.last,
                                "{}: a last chunk with no row to compute",
                                sl.seq
                            );
                            let tp1 = std::time::Instant::now();
                            if let Some((a, t)) = p43_req.get_mut(&sl.seq) {
                                let ms = |x: std::time::Instant, y: std::time::Instant| {
                                    (y.max(x) - x).as_secs_f64() * 1e3
                                };
                                let gap = p43_reply.map_or(f64::NAN, |r| ms(r, ts0));
                                t.push_str(&format!(
                                    " | pos {pos} rows {} t0 {:.2} gap {gap:.2} pre {:.2} fwd {:.2} out {:.2} app {:.2} chunk {:.2}",
                                    end - lo,
                                    ms(*a, ts0),
                                    ms(ts0, tp0),
                                    eng.p43_t[0],
                                    eng.p43_t[1],
                                    eng.p43_t[2],
                                    ms(tp0, tp1)
                                ));
                            }
                            let tapped = eng.tap;
                            eng.tap = false;
                            let row = row?;
                            if tapped {
                                owner = Some(sl.seq);
                                pending = None;
                            } else if owner == Some(sl.seq) {
                                owner = None;
                                pending = None;
                            }
                            slot_of.insert(sl.seq, (slot, pos + sl.input.len() as u64));
                            let mut ts = 0.0;
                            if ch.last {
                                let picked = if eng.first_seam {
                                    sample_seam(&sampler, &seam_row(&row.seam, 0), &sl.sampling)
                                } else {
                                    None
                                };
                                let (tok, lp) = if let Some(p) = picked {
                                    n_first_seam += 1;
                                    p
                                } else {
                                    n_first_dense += 1;
                                    let mut r = if row.dense.is_empty() {
                                        eng.dense_row(0)?
                                    } else {
                                        super::f32s(&row.dense)
                                    };
                                    sample_row(&sampler, &mut r, &sl.sampling)
                                };
                                out[i] = crate::flashnext::serve::seq_step(sl.seq, Some(tok), lp);
                                emitted.push((sl.seq, 1));
                                pf_last.push(sl.seq);
                                ts = tp1.elapsed().as_secs_f64() * 1e3;
                                if let Some((a, t)) = p43_req.remove(&sl.seq) {
                                    p43_out.push(format!(
                                        "INK43 req {} prompt {} admit->sampled {:.2} sample {ts:.2}{t}",
                                        sl.seq,
                                        end,
                                        a.elapsed().as_secs_f64() * 1e3
                                    ));
                                }
                            }
                            p_time.push([(end - lo) as f64, (tp1 - tp0).as_secs_f64() * 1e3, ts]);
                        }
                        _ => {
                            // a decode row feeds the LAST of the previous step's committed
                            // tokens (`pending_input` carries the whole commit)
                            let t = *sl.input.last().context("a decode row with no token")?;
                            dec.push((i, slot, t, pos));
                        }
                    }
                }
                // INK13: the drafter's round + the verify, when ONE sequence is live and it is the
                // owner (greedy or sampled without penalties: every row is the target's own draw
                // at its output position -- the drawn-row walk; a draft survives only where the
                // draw equals it, so the committed tokens are the non-speculative stream's)
                let spec = mtp_k > 0 && dec.len() == 1 && slot_of.len() == 1 && {
                    let (i, _, _, pos) = dec[0];
                    let sl = &b.slots[i];
                    (batch || owner == Some(sl.seq))
                        && sl.verify_width >= 1
                        && RowParams::resolve(&sl.sampling.params, VOCAB).no_penalties()
                        && pos + 8 <= ctx
                        && eng
                            .dr
                            .as_ref()
                            .is_some_and(|d| d.nctx_of(dec[0].1 as usize) == pos)
                };
                if spec {
                    let (i, slot, tok, pos) = dec[0];
                    dec.clear();
                    let sl = &b.slots[i];
                    let kk = mtp_k.min(u64::from(sl.verify_width));
                    // INK32: a copy proposal replaces the drafter's round (its round is not
                    // queued: the drafter's context still takes every verified row below)
                    let copy = ng_pick(hist.get(&sl.seq), sl.verify_width as usize);
                    if copy.is_none() {
                        let d = eng.dr.as_mut().context("dr")?;
                        if batch {
                            d.select(slot as usize)?;
                        }
                        if pending != Some((pos, tok, kk)) {
                            d.issue(eng.c, pk, tok, kk)?;
                        }
                    } else if batch {
                        eng.dr.as_mut().context("dr")?.select(slot as usize)?;
                    }
                    pending = None;
                    let t0 = std::time::Instant::now();
                    // INK32: a copy proposal is verified whole; INK33: otherwise the gate
                    // verifies the confident prefix of the k-deep draft
                    let is_copy = copy.is_some();
                    let drafts = match (copy, gate) {
                        (Some(c), _) => c,
                        (None, Some((prod, tau, dmin, first))) => {
                            let (mut d, p1) =
                                eng.dr.as_ref().context("dr")?.drafts_conf(eng.c, kk)?;
                            let mut g = gated_depth(&p1, prod, tau, dmin.min(kk));
                            if sl.sampling.position < first {
                                g = dmin.min(kk);
                            }
                            d.truncate(g as usize);
                            g_stat[g as usize] += 1;
                            d
                        }
                        (None, None) => eng.dr.as_ref().context("dr")?.drafts(eng.c, kk)?,
                    };
                    let kd = drafts.len() as u64;
                    let t1 = std::time::Instant::now();
                    let mut ids = vec![tok];
                    ids.extend_from_slice(&drafts);
                    let cmd = Cmd::Verify {
                        slot,
                        ids: ids.clone(),
                        pos,
                    };
                    send(&cmd)?;
                    let temp = seam_temperature(&sl.sampling);
                    let outs = eng.apply2(&cmd, &vec![temp; ids.len()])?;
                    let t2 = std::time::Instant::now();
                    let (mut newt, mut lps) = (Vec::new(), Vec::new());
                    for a in 0..ids.len() {
                        let mut ss = sl.sampling.clone();
                        ss.position += a as u64;
                        let pick = if eng.seam {
                            if let Some(p) = sample_seam(&sampler, &seam_row(&outs.seam, a), &ss) {
                                n_seam += 1;
                                p
                            } else {
                                n_fallback += 1;
                                let mut r = eng.dense_row(a)?;
                                sample_row(&sampler, &mut r, &ss)
                            }
                        } else {
                            let mut r =
                                super::f32s(&outs.dense[a * VOCAB * 4..(a + 1) * VOCAB * 4]);
                            sample_row(&sampler, &mut r, &ss)
                        };
                        newt.push(pick.0);
                        lps.push(pick.1);
                        if !(a + 1 < ids.len() && pick.0 == ids[a + 1] && !eos.contains(&pick.0)) {
                            break;
                        }
                    }
                    let keep = newt.len() as u64;
                    // INK31: the first three verify rounds of a traced request -- the round's
                    // input ids (the anchor + the drafts) and each picked row's logprob bits: a
                    // reloaded request whose drafts differ from the control's names the drafter's
                    // state; identical drafts with different bits name the target's
                    if tr_on {
                        if let Some(t) = trace.get_mut(&sl.seq) {
                            if t.matches(" V@").count() < 3 {
                                let idl: Vec<String> = ids.iter().map(|x| x.to_string()).collect();
                                let lpl: Vec<String> = newt
                                    .iter()
                                    .zip(&lps)
                                    .map(|(n, x)| match x {
                                        Some(l) => {
                                            format!("{n}:{:08x}", l.sampled.logprob.to_bits())
                                        }
                                        None => format!("{n}:-"),
                                    })
                                    .collect();
                                t.push_str(&format!(
                                    " V@{pos}[{}]k{keep}<{}>",
                                    idl.join(","),
                                    lpl.join(",")
                                ));
                            }
                        }
                    }
                    let cm = Cmd::Commit { keep };
                    send(&cm)?;
                    eng.apply(&cm)?;
                    let np = pos + keep;
                    let anchor = *newt.last().context("walk")?;
                    // INK32: the history takes the round's tokens now (the next round's
                    // proposal decides whether the drafter's round is queued)
                    let mut next_copy = false;
                    if let Some(h) = hist.get_mut(&sl.seq) {
                        h.push(&newt);
                        ng_pushed.push(sl.seq);
                        next_copy = ng_pick(Some(&*h), 7).is_some();
                    }
                    if is_copy {
                        ng_stat[0] += 1;
                        ng_stat[1] += keep;
                        ng_stat[2] += kd;
                    }
                    let kq = mtp_k.min(u64::from(sl.verify_width));
                    {
                        let d = eng.dr.as_mut().context("dr")?;
                        d.append(eng.c, keep, pos)?;
                        // the next round, queued now (it runs while the scheduler commits)
                        if next_copy {
                            ng_stat[3] += 1;
                        } else if !eos.contains(&anchor) && np + 8 <= ctx {
                            d.issue(eng.c, pk, anchor, kq)?;
                            pending = Some((np, anchor, kq));
                        }
                    }
                    let t3 = std::time::Instant::now();
                    let ms =
                        |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
                    m_time.push([
                        ms(t0, t1),
                        ms(t1, t2),
                        ms(t2, t3),
                        last_end.map_or(0.0, |e| ms(e, t0)),
                    ]);
                    last_end = Some(std::time::Instant::now());
                    m_rounds += 1;
                    m_tokens += keep;
                    emitted.push((sl.seq, keep));
                    skind = (2, 1);
                    m_drafted += kd;
                    slot_of.insert(sl.seq, (slot, np));
                    let lp = if lps.iter().all(Option::is_some) {
                        Some(lps.into_iter().flatten().collect())
                    } else {
                        None
                    };
                    out[i] = crate::flashnext::serve::seq_step_k(
                        sl.seq,
                        newt,
                        u32::try_from(kd)?,
                        u32::try_from(keep - 1)?,
                        lp,
                    );
                }
                // INK20 (`--ink-mtp-batch`): every decoding sequence's round in ONE batched drafter
                // pass + ONE multi-sequence verify, when the table gives the live count drafts
                // and every sequence can speculate (in sync, no penalties, room)
                // a step that carries a prefill chunk decodes plainly (the chunk's first token
                // is replied at the step's end: a speculative round there costs its TTFT)
                let has_prefill = b.slots.iter().any(|sl| sl.chunk.is_some());
                // INK21 (`rel`): the step after a sequence left decodes plainly (the next
                // request, arriving now, joins at the next step boundary: one ~28 ms step)
                let rel_plain = r_rel && released && batch && dec.len() >= 2;
                if rel_plain {
                    n_rel_plain += 1;
                }
                let mk = if batch && dec.len() >= 2 && !has_prefill && !rel_plain {
                    bk(dec.len())
                } else {
                    0
                };
                let mspec = mk > 0
                    && dec.iter().all(|&(i, slot, _, pos)| {
                        let sl = &b.slots[i];
                        // a sequence's first output rows decode plainly (its first visible
                        // text rides the next step: a round there costs the TTFT)
                        sl.sampling.position >= r_first
                            && sl.verify_width >= 1
                            && RowParams::resolve(&sl.sampling.params, VOCAB).no_penalties()
                            && pos + 8 <= ctx
                            && eng
                                .dr
                                .as_ref()
                                .is_some_and(|d| d.nctx_of(slot as usize) == pos)
                    });
                if mspec {
                    let kk = dec
                        .iter()
                        .map(|&(i, ..)| u64::from(b.slots[i].verify_width))
                        .min()
                        .unwrap_or(0)
                        .min(mk)
                        // INK38: K up to 7 -- the round fits one multi verify (<= 32 rows)
                        .min(mv_depth_cap(dec.len() as u64));
                    let nb = dec.len();
                    let sig: Vec<(u32, u64, u32)> = dec
                        .iter()
                        .map(|&(_, slot, tok, pos)| (slot, pos, tok))
                        .collect();
                    let t0 = std::time::Instant::now();
                    let drafts = {
                        let d = eng.dr.as_mut().context("dr")?;
                        if pending_m.as_ref() != Some(&(sig.clone(), kk)) {
                            let rows: Vec<(usize, u32)> = dec
                                .iter()
                                .map(|&(_, slot, tok, _)| (slot as usize, tok))
                                .collect();
                            d.issue_multi(eng.c, pk, &rows, kk)?;
                        }
                        pending_m = None;
                        pending = None;
                        if let Some((prod, tau, dmin, agg)) = bgate {
                            // INK38: every sequence's gated depth, ONE depth for the round
                            let dc = d.drafts_multi_conf(eng.c, nb as u64, kk)?;
                            let ds: Vec<u64> = dc
                                .iter()
                                .map(|(_, p1)| gated_depth(p1, prod, tau, dmin))
                                .collect();
                            let v = bgate_depth(&ds, agg).clamp(1, kk);
                            bg_stat[v as usize] += 1;
                            dc.into_iter()
                                .map(|(t, _)| t.into_iter().take(v as usize).collect::<Vec<u32>>())
                                .collect::<Vec<_>>()
                        } else {
                            d.drafts_multi(eng.c, nb as u64, kk)?
                        }
                    };
                    // INK38: the verified depth (the gate may cut the drafted K)
                    let kd = drafts.first().map_or(kk, |v| v.len() as u64);
                    let w = kd + 1;
                    bg_rows += nb as u64 * w;
                    // INK32: a sequence with a copy proposal verifies it in its K rows (the rest
                    // of the K, if the copy is shorter, keep the drafter's tail)
                    let mut drafts = drafts;
                    if ng.is_some() && f.ink_ngram_batch {
                        for (q, &(i, ..)) in dec.iter().enumerate() {
                            if let Some(c) = ng_pick(hist.get(&b.slots[i].seq), kk as usize) {
                                let dd = &mut drafts[q];
                                ng_stat[4] += 1;
                                for (a, t) in c.into_iter().enumerate() {
                                    if a < dd.len() {
                                        dd[a] = t;
                                    }
                                }
                            }
                        }
                    }
                    let t1 = std::time::Instant::now();
                    let vrows: Vec<(u32, Vec<u32>, u64)> = dec
                        .iter()
                        .zip(&drafts)
                        .map(|(&(_, slot, tok, pos), dd)| {
                            let mut v = vec![tok];
                            v.extend_from_slice(dd);
                            (slot, v, pos)
                        })
                        .collect();
                    let temps: Vec<f32> = dec
                        .iter()
                        .flat_map(|&(i, ..)| {
                            std::iter::repeat_n(seam_temperature(&b.slots[i].sampling), w as usize)
                        })
                        .collect();
                    let cmd = Cmd::VerifyM {
                        rows: vrows.clone(),
                    };
                    send(&cmd)?;
                    let outs = eng.apply2(&cmd, &temps)?;
                    let t2 = std::time::Instant::now();
                    let mut keeps = Vec::with_capacity(nb);
                    let mut results = Vec::with_capacity(nb);
                    for (q, &(i, ..)) in dec.iter().enumerate() {
                        let sl = &b.slots[i];
                        let ids = &vrows[q].1;
                        let (mut newt, mut lps) = (Vec::new(), Vec::new());
                        for a in 0..ids.len() {
                            let row = q * w as usize + a;
                            let mut ss = sl.sampling.clone();
                            ss.position += a as u64;
                            let pick = if eng.seam {
                                if let Some(p) =
                                    sample_seam(&sampler, &seam_row(&outs.seam, row), &ss)
                                {
                                    n_seam += 1;
                                    p
                                } else {
                                    n_fallback += 1;
                                    let mut r = eng.dense_row(row)?;
                                    sample_row(&sampler, &mut r, &ss)
                                }
                            } else {
                                let mut r = super::f32s(
                                    &outs.dense[row * VOCAB * 4..(row + 1) * VOCAB * 4],
                                );
                                sample_row(&sampler, &mut r, &ss)
                            };
                            newt.push(pick.0);
                            lps.push(pick.1);
                            if !(a + 1 < ids.len()
                                && pick.0 == ids[a + 1]
                                && !eos.contains(&pick.0))
                            {
                                break;
                            }
                        }
                        keeps.push(u32::try_from(newt.len())?);
                        results.push((newt, lps));
                    }
                    let cm = Cmd::CommitM {
                        keeps: keeps.clone(),
                    };
                    send(&cm)?;
                    eng.apply(&cm)?;
                    let mut next: Vec<(usize, u32)> = Vec::new();
                    let mut nsig: Vec<(u32, u64, u32)> = Vec::new();
                    {
                        let d = eng.dr.as_mut().context("dr")?;
                        let ap: Vec<(usize, u64, u64, u64)> = dec
                            .iter()
                            .enumerate()
                            .map(|(q, &(_, slot, _, pos))| {
                                (slot as usize, u64::from(keeps[q]), pos, q as u64 * w)
                            })
                            .collect();
                        if eng.bapp > 0 {
                            // INK22: the round's kept rows (rows q * w + a, a < keep) in ONE
                            // captured append (`check`: + INK20's eager append, compared)
                            let mut tab: Vec<Option<(usize, u64, bool)>> =
                                vec![None; nb * w as usize];
                            for &(sl, t, p, r0) in &ap {
                                for a in 0..t {
                                    tab[(r0 + a) as usize] = Some((sl, p + a, a + 1 == t));
                                }
                            }
                            d.write_atab(eng.c, &tab)?;
                            d.append_batch_graph(eng.c, nb as u64 * w)?;
                            if eng.bapp == 2 {
                                let mut got = Vec::new();
                                for &(sl, t, p, _) in &ap {
                                    for a in 0..t {
                                        got.push(d.ctx_rows(eng.c, sl, p + a)?);
                                    }
                                }
                                d.append_multi(eng.c, &ap)?;
                                let mut gi = 0;
                                for &(sl, t, p, _) in &ap {
                                    for a in 0..t {
                                        n_app_checked += 1;
                                        if d.ctx_rows(eng.c, sl, p + a)? != got[gi] {
                                            n_app_mis += 1;
                                            eprintln!(
                                                "INK22 append check: round slot {sl} position {} DIFFERS",
                                                p + a
                                            );
                                        }
                                        gi += 1;
                                    }
                                }
                            } else {
                                for &(sl, t, p, _) in &ap {
                                    for a in 0..t {
                                        d.appended(sl, p + a)?;
                                    }
                                }
                            }
                            n_app_rows += ap.iter().map(|e| e.1).sum::<u64>();
                        } else {
                            d.append_multi(eng.c, &ap)?;
                        }
                        for (q, &(_, slot, _, pos)) in dec.iter().enumerate() {
                            let np = pos + u64::from(keeps[q]);
                            let anchor = *results[q].0.last().context("walk")?;
                            if !eos.contains(&anchor) && np + 8 <= ctx {
                                next.push((slot as usize, anchor));
                                nsig.push((slot, np, anchor));
                            }
                        }
                        // the next round, queued now when every sequence continues (INK22:
                        // `--ink-batch-ahead full` only while no slot is free, `off` never)
                        let ahead = match f.ink_batch_ahead.as_str() {
                            "off" => false,
                            "full" => free_slots.is_empty(),
                            _ => true,
                        };
                        if next.len() == nb && !ahead {
                            n_ahead_skip += 1;
                        }
                        if next.len() == nb && ahead {
                            d.issue_multi(eng.c, pk, &next, kk)?;
                            pending_m = Some((nsig, kk));
                        }
                    }
                    skind = (3, nb);
                    let t3 = std::time::Instant::now();
                    let ms =
                        |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
                    mb_time.push([ms(t0, t1), ms(t1, t2), ms(t2, t3), nb as f64]);
                    let e = mb_stat.entry(nb).or_insert([0; 3]);
                    e[0] += 1;
                    for (q, &(i, slot, _, pos)) in dec.iter().enumerate() {
                        let sl = &b.slots[i];
                        let (newt, lps) = std::mem::take(&mut results[q]);
                        let keep = u64::from(keeps[q]);
                        emitted.push((sl.seq, keep));
                        e[1] += keep;
                        e[2] += kd;
                        slot_of.insert(sl.seq, (slot, pos + keep));
                        let lp = if lps.iter().all(Option::is_some) {
                            Some(lps.into_iter().flatten().collect())
                        } else {
                            None
                        };
                        out[i] = crate::flashnext::serve::seq_step_k(
                            sl.seq,
                            newt,
                            u32::try_from(kd)?,
                            u32::try_from(keep - 1)?,
                            lp,
                        );
                    }
                    dec.clear();
                }
                if let (true, Some(o)) = (batch, owner) {
                    let _ = o;
                    owner = None;
                }
                if let Some(o) = owner {
                    if dec.iter().any(|&(i, ..)| b.slots[i].seq == o) {
                        // the owner decoded without the drafter: its context is behind
                        owner = None;
                        pending = None;
                        m_lost += 1;
                    }
                }
                if !dec.is_empty() {
                    let cmd = Cmd::Decode {
                        rows: dec.iter().map(|d| (d.1, d.2, d.3)).collect(),
                    };
                    send(&cmd)?;
                    let temps: Vec<f32> = dec
                        .iter()
                        .map(|&(i, ..)| seam_temperature(&b.slots[i].sampling))
                        .collect();
                    let outs = eng.apply2(&cmd, &temps)?;
                    let rows = outs.dense;
                    // INK10 task 3: the rows' samplers in parallel (one scoped thread a row; each
                    // row's draw depends on its own logits + its own (seed, position) noise only,
                    // so the tokens are the serial loop's)
                    let picks: Vec<(TokenId, Option<SampledLogprobs>)> = if eng.seam {
                        // INK11: the seam (64 candidates a row); a row it cannot decide reads its
                        // dense row; `check` samples both and compares the deterministic rows
                        let mut v = Vec::with_capacity(dec.len());
                        for (k, &(i, ..)) in dec.iter().enumerate() {
                            let sl = &b.slots[i].sampling;
                            let got = sample_seam(&sampler, &seam_row(&outs.seam, k), sl);
                            let pick = match got {
                                Some(p) => {
                                    n_seam += 1;
                                    p
                                }
                                None => {
                                    n_fallback += 1;
                                    let mut r = eng.dense_row(k)?;
                                    sample_row(&sampler, &mut r, sl)
                                }
                            };
                            if eng.dense {
                                let det = sl.params.seed.is_some()
                                    || RowParams::resolve(&sl.params, VOCAB).is_greedy();
                                if det {
                                    let mut r =
                                        super::f32s(&rows[k * VOCAB * 4..(k + 1) * VOCAB * 4]);
                                    let d = sample_row(&sampler, &mut r, sl);
                                    n_checked += 1;
                                    if d.0 != pick.0 {
                                        n_mismatch += 1;
                                        eprintln!(
                                            "inkserve: seam {} vs dense {} ({})",
                                            pick.0, d.0, b.slots[i].seq
                                        );
                                    }
                                }
                            }
                            v.push(pick);
                        }
                        v
                    } else if dec.len() > 1 {
                        std::thread::scope(|sc| {
                            let hs: Vec<_> = dec
                                .iter()
                                .enumerate()
                                .map(|(k, &(i, ..))| {
                                    let (rows, sampler, sl) = (&rows, &sampler, &b.slots[i]);
                                    sc.spawn(move || {
                                        let mut r =
                                            super::f32s(&rows[k * VOCAB * 4..(k + 1) * VOCAB * 4]);
                                        sample_row(sampler, &mut r, &sl.sampling)
                                    })
                                })
                                .collect();
                            hs.into_iter()
                                .map(|h| h.join().expect("a sampler thread panicked"))
                                .collect()
                        })
                    } else {
                        let mut r = super::f32s(&rows[..VOCAB * 4]);
                        vec![sample_row(&sampler, &mut r, &b.slots[dec[0].0].sampling)]
                    };
                    if batch && eng.bapp > 0 {
                        // INK22: the step's graph appended its rows; `check` re-runs INK20's
                        // eager append over them and compares every row
                        let rows_ = std::mem::take(&mut eng.app_rows);
                        let d = eng.dr.as_mut().context("dr")?;
                        if eng.bapp == 2 {
                            let mut ap = Vec::new();
                            let mut got = Vec::new();
                            for (k, r) in rows_.iter().enumerate() {
                                if let Some((sl, pos, _)) = *r {
                                    got.push(d.ctx_rows(eng.c, sl, pos)?);
                                    ap.push((sl, 1, pos, k as u64));
                                }
                            }
                            if !ap.is_empty() {
                                d.append_multi(eng.c, &ap)?;
                            }
                            for (q, &(sl, _, pos, _)) in ap.iter().enumerate() {
                                n_app_checked += 1;
                                if d.ctx_rows(eng.c, sl, pos)? != got[q] {
                                    n_app_mis += 1;
                                    eprintln!(
                                        "INK22 append check: slot {sl} position {pos} DIFFERS"
                                    );
                                }
                            }
                            n_app_rows += ap.len() as u64;
                        } else {
                            for (sl, pos, _) in rows_.into_iter().flatten() {
                                d.appended(sl, pos)?;
                                n_app_rows += 1;
                            }
                        }
                        pending_m = None;
                        pending = None;
                    } else if batch {
                        // INK20: the decode step tapped the drafter: every in-sync context
                        // takes its committed row
                        let d = eng.dr.as_mut().context("dr")?;
                        let ap: Vec<(usize, u64, u64, u64)> = dec
                            .iter()
                            .enumerate()
                            .filter(|(_, &(_, slot, _, pos))| {
                                d.nctx_of(slot as usize) == pos && pos + 8 <= ctx
                            })
                            .map(|(k, &(_, slot, _, pos))| (slot as usize, 1, pos, k as u64))
                            .collect();
                        if !ap.is_empty() {
                            d.append_multi(eng.c, &ap)?;
                        }
                        pending_m = None;
                        pending = None;
                    }
                    for (k, &(i, slot, _, pos)) in dec.iter().enumerate() {
                        let sl = &b.slots[i];
                        slot_of.insert(sl.seq, (slot, pos + 1));
                        let (tok, lp) = picks[k].clone();
                        out[i] = crate::flashnext::serve::seq_step(sl.seq, Some(tok), lp);
                        emitted.push((sl.seq, 1));
                    }
                    skind = (1, dec.len());
                }
                steps += 1;
                released = false;
                let te = std::time::Instant::now();
                let ms = |a: std::time::Instant| (te - a).as_secs_f64() * 1e3;
                step_rec.push((skind.0, skind.1, ms(ts0)));
                for q in pf_last {
                    if let Some(e) = seq_n.get_mut(&q) {
                        e.3 = Some(te);
                    }
                }
                for &(q, k) in &emitted {
                    if let Some(t) = trace.get_mut(&q) {
                        t.push_str(&format!(" {}x{}:{k}", skind.0, skind.1));
                    }
                }
                for (q, k) in emitted {
                    if let Some(e) = seq_n.get_mut(&q) {
                        let was = e.1;
                        e.1 += k;
                        if was < 3 && e.1 >= 3 {
                            let st = e.2.map_or(f64::NAN, |t| (t - e.0).as_secs_f64() * 1e3);
                            let pe = e.3.map_or(f64::NAN, |t| (t - e.0).as_secs_f64() * 1e3);
                            ttft_rec.push([st, pe, ms(e.0), e.3.map_or(f64::NAN, ms)]);
                        }
                    }
                }
                if !hist.is_empty() {
                    for o in &out {
                        if !o.tokens.is_empty() && !ng_pushed.contains(&o.seq_id) {
                            if let Some(h) = hist.get_mut(&o.seq_id) {
                                h.push(&o.tokens);
                            }
                        }
                    }
                }
                let tr0 = std::time::Instant::now();
                let _ = reply.send(StepResult::new(out));
                if p43_on {
                    let now = std::time::Instant::now();
                    for l in p43_out.drain(..) {
                        // `reply` = the step's end -> the reply handed to the scheduler (ms)
                        println!("{l} | reply {:.3}", (now - tr0).as_secs_f64() * 1e3);
                    }
                    p43_reply = Some(now);
                }
            }
        }
    })();
    let _ = send(&Cmd::Stop);
    if let Some(pc) = eng.pc.as_ref().filter(|_| eng.pc_verify) {
        println!(
            "INK26 pcverify rank0: ops {}, copies differing {}, node regions changed {}, regions checked {}",
            pc.vstat[0], pc.vstat[1], pc.vstat[2], pc.vstat[3]
        );
    }
    eng.graphs.clear();
    if let Err(e) = &res {
        eprintln!("inkserve: engine loop failed: {e:#}");
        saber_serve::fatal::set(0, format!("inkserve engine loop: {e:#}"));
    }
    drop(st.jobs);
    let j = st
        .join
        .join()
        .map_err(|_| anyhow::anyhow!("the HTTP thread panicked"))?;
    println!("inkserve rank0: {steps} engine steps");
    if batch {
        let mut cols: Vec<Vec<f64>> = (0..3)
            .map(|j| {
                let mut v: Vec<f64> = mb_time
                    .iter()
                    .filter(|r| r[3] >= 8.0)
                    .map(|r| r[j])
                    .collect();
                v.sort_by(f64::total_cmp);
                v
            })
            .collect();
        let m = |v: &mut Vec<f64>| v.get(v.len() / 2).copied().unwrap_or(f64::NAN);
        println!(
            "INK20 mtp batch: rules {} (rel plain steps {n_rel_plain}); by live count [rounds, tokens, drafted] {mb_stat:?}; at 8 live p50 (host clock) draft wait {:.2} verify + seam {:.2} walk + commit + append + queue {:.2} ms",
            f.ink_batch_plain,
            m(&mut cols[0]),
            m(&mut cols[1]),
            m(&mut cols[2])
        );
    }
    if ng.is_some() {
        println!(
            "INK32 ngram: copy rounds {} tokens {} ({:.3} a round), drafted {}, drafter rounds not queued {}, batched rows by copy {}",
            ng_stat[0],
            ng_stat[1],
            ng_stat[1] as f64 / ng_stat[0].max(1) as f64,
            ng_stat[2],
            ng_stat[3],
            ng_stat[4]
        );
    }
    if mtp_k > 0 {
        println!(
            "INK13 mtp k {mtp_k}: rounds {m_rounds} tokens {m_tokens} ({:.3} a round), drafted {m_drafted}, owner lost {m_lost}",
            m_tokens as f64 / m_rounds.max(1) as f64
        );
        if gate.is_some() {
            println!("INK33 mtp gate: rounds by verified depth 0..7 {g_stat:?}");
        }
        if bgate.is_some() {
            println!(
                "INK38 mtp bgate: batched rounds by verified depth 0..7 {bg_stat:?}, {bg_rows} verify rows"
            );
        }
        let mut cols: Vec<Vec<f64>> = (0..4)
            .map(|j| m_time.iter().map(|r| r[j]).collect())
            .collect();
        let p50 = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
        };
        let q: Vec<f64> = cols.iter_mut().map(p50).collect();
        println!(
            "INK13 mtp round p50 (host clock): draft wait {:.2} verify + seam {:.2} walk + commit + append + queue {:.2} gap {:.2} ms",
            q[0], q[1], q[2], q[3]
        );
        // INK30 (update p99): the slow rounds' composition -- a round whose four terms sum past
        // the p50 sum + 5 ms, each term's mean over those rounds vs its p50 (host clock)
        let tq = |v: &Vec<f64>, f: f64| {
            v.get(((v.len() as f64 * f) as usize).min(v.len().saturating_sub(1)))
                .copied()
                .unwrap_or(f64::NAN)
        };
        let tot50: f64 = q.iter().sum();
        let slow: Vec<&[f64; 4]> = m_time
            .iter()
            .filter(|r| r.iter().sum::<f64>() > tot50 + 5.0)
            .collect();
        let mean = |j: usize| slow.iter().map(|r| r[j]).sum::<f64>() / slow.len().max(1) as f64;
        println!(
            "INK30 mtp round tails: p99 draft {:.2} verify {:.2} post {:.2} gap {:.2}; max {:.2} / {:.2} / {:.2} / {:.2}; slow rounds (> p50 sum + 5 ms) {} of {}: mean draft {:.2} verify {:.2} post {:.2} gap {:.2} ms",
            tq(&cols[0], 0.99), tq(&cols[1], 0.99), tq(&cols[2], 0.99), tq(&cols[3], 0.99),
            tq(&cols[0], 1.0), tq(&cols[1], 1.0), tq(&cols[2], 1.0), tq(&cols[3], 1.0),
            slow.len(), m_time.len(), mean(0), mean(1), mean(2), mean(3)
        );
    }
    {
        let band = |lo: f64, hi: f64| -> String {
            let mut a: Vec<f64> = p_time
                .iter()
                .filter(|r| r[0] >= lo && r[0] < hi)
                .map(|r| r[1])
                .collect();
            let mut b: Vec<f64> = p_time
                .iter()
                .filter(|r| r[0] >= lo && r[0] < hi)
                .map(|r| r[2])
                .collect();
            a.sort_by(f64::total_cmp);
            b.sort_by(f64::total_cmp);
            let m = |v: &Vec<f64>| v.get(v.len() / 2).copied().unwrap_or(f64::NAN);
            format!(
                "rows [{lo}, {hi}) n {} chunk {:.2} sample {:.2}",
                a.len(),
                m(&a),
                m(&b)
            )
        };
        println!(
            "INK16 prefill p50 (host clock): {}; {}; {}; {}; cl launches {} pf fallbacks {} push reduces {}; first token {} seam {n_first_seam} dense {n_first_dense}",
            band(0.0, 64.0),
            band(64.0, 128.0),
            band(128.0, 256.0),
            band(256.0, 1e9),
            eng.c.cl_n[0],
            eng.c.cl_n[1],
            eng.c.cl_n[2],
            f.ink_first_token
        );
        println!(
            "INK25 prefill trunk {}: iu8 GEMMs {} quant launches {} law misses {}",
            f.ink_prefill_trunk, eng.c.i8_n[0], eng.c.i8_n[1], eng.c.i8_n[2]
        );
        println!(
            "INK39 mv levers {:?}: merged projection GEMMs {} push reduces {}",
            f.ink_p39, eng.c.p39_n[0], eng.c.p39_n[1]
        );
        println!(
            "INK29 prefill levers {:?}: kx {} bw {} rs {} fq {} quant arms {} subset skips {}",
            f.ink29,
            eng.c.p29_n[0],
            eng.c.p29_n[1],
            eng.c.p29_n[2],
            eng.c.p29_n[3],
            eng.c.p29_n[4],
            eng.c.p29_n[5]
        );
    }
    {
        let q = |v: &mut Vec<f64>, p: f64| -> f64 {
            v.retain(|x| x.is_finite());
            v.sort_by(f64::total_cmp);
            if v.is_empty() {
                f64::NAN
            } else {
                v[((v.len() - 1) as f64 * p).round() as usize]
            }
        };
        let col = |j: usize| -> Vec<f64> { ttft_rec.iter().map(|r| r[j]).collect() };
        let (mut a, mut b_, mut c_, mut d_) = (col(0), col(1), col(2), col(3));
        println!(
            "INK22 ttft (server clock, admit -> 3rd output reply) n {}: p50 {:.1} p90 {:.1} p99 {:.1} ms; admit -> first chunk p50 {:.1} p90 {:.1}; admit -> last chunk reply p50 {:.1}; last chunk reply -> 3rd output p50 {:.1} p90 {:.1}",
            ttft_rec.len(),
            q(&mut c_.clone(), 0.5),
            q(&mut c_.clone(), 0.9),
            q(&mut c_, 0.99),
            q(&mut a.clone(), 0.5),
            q(&mut a, 0.9),
            q(&mut b_, 0.5),
            q(&mut d_.clone(), 0.5),
            q(&mut d_, 0.9),
        );
        let mut by: BTreeMap<(u8, usize), Vec<f64>> = BTreeMap::new();
        for &(k, n, t) in &step_rec {
            let n = if k == 0 {
                n.next_power_of_two().max(64)
            } else {
                n
            };
            by.entry((k, n)).or_default().push(t);
        }
        let parts: Vec<String> = by
            .iter_mut()
            .map(|(&(k, n), v)| {
                let kn = ["prefill<=", "plain b", "round1 b", "roundM b"][k as usize];
                format!("{kn}{n} n {} p50 {:.2}", v.len(), q(v, 0.5))
            })
            .collect();
        println!("INK22 steps (host clock a step): {}", parts.join("; "));
        if batch {
            println!(
                "INK22 batch append {}: rows {n_app_rows}, checked {n_app_checked}, differing {n_app_mis}; ahead {} (rounds not queued {n_ahead_skip})",
                f.ink_batch_append, f.ink_batch_ahead
            );
        }
    }
    if pc_on {
        println!(
            "INK23 pcache: stores {}, loads {}, rows reloaded {}, skipped (node without drafter rows) {}, nodes held {}, first sightings {}",
            pcx.stores,
            pcx.loads,
            pcx.load_rows,
            pcx.skipped_nodraft,
            pcx.nodes.iter().flatten().count(),
            pcx.first_sightings
        );
    }
    println!(
        "INK11 sampler {}: seam rows {n_seam}, dense fallbacks {n_fallback}, checked {n_checked}, \
         mismatches {n_mismatch}",
        f.ink_sampler
    );
    res.and(j)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ink54_the_interleave_arm_parses_and_sets_the_yield() {
        assert_eq!(p54_yield(crate::inkling2::DEFAULT_INK54).unwrap(), 0);
        assert_eq!(p54_yield("il8").unwrap(), 8);
        assert!(p54_yield("il0").is_err());
        assert!(p54_yield("il").is_err());
        assert!(p54_yield("x").is_err());
        assert_eq!(p54_yield("wk,il4").unwrap(), 4);
        let c = scheduler_config_ch(40960, 8, 7, Some(2), 7, 2048).unwrap();
        assert_eq!(c.prefill.yield_steps, 0);
        assert_eq!(c.prefill.clone().with_yield_steps(8).yield_steps, 8);
    }

    #[test]
    fn ink43_the_chunk_lever_parses_and_the_ring_holds_the_window_plus_the_chunk() {
        assert_eq!(
            p43_opts(crate::inkling2::DEFAULT_INK43).unwrap(),
            (2048, true)
        );
        assert_eq!(
            crate::inkling2::INK43_CONTROL
                .strip_prefix("--ink-p43 ")
                .map(|v| p43_opts(v).unwrap()),
            Some((SERVE_CHUNK, false))
        );
        assert_eq!(p43_chunk("none").unwrap(), SERVE_CHUNK);
        assert_eq!(p43_chunk("ch2048").unwrap(), 2048);
        assert_eq!(p43_chunk("ch1536").unwrap(), 1536);
        assert!(p43_chunk("ch1000").is_err());
        assert!(p43_chunk("ch8192").is_err());
        assert!(p43_chunk("hg").is_err());
        assert_eq!(p43_opts("am").unwrap(), (SERVE_CHUNK, true));
        assert_eq!(p43_opts("ch2048,am").unwrap(), (2048, true));
        assert_eq!(p43_opts("").unwrap(), (SERVE_CHUNK, false));
        let d = crate::inkling2::DEFAULT_INK43;
        assert_eq!(p43_plane(d, true).unwrap(), (2048, true));
        assert_eq!(p43_plane(d, false).unwrap(), (SERVE_CHUNK, false));
        assert_eq!(p43_plane("am", false).unwrap(), (SERVE_CHUNK, true));
        assert_eq!(p43_ring(2048, SERVE_CHUNK), 2048);
        assert_eq!(p43_ring(2048, 2048), 3072);
        assert_eq!(p43_ring(2048, 1536), 2560);
        assert_eq!(p43_ring(0, 2048), 0);
        let c = scheduler_config_ch(16896, 1, 7, None, 7, 2048).unwrap();
        assert_eq!(c.prefill.max_chunk(), 2048);
        assert_eq!(c.prefill.chunk_for(16728), Some((2048, 2048)));
        assert_eq!(c.prefill.chunk_for(344), Some((344, 2048)));
    }

    #[test]
    fn ink8_the_scheduler_geometry_admits_the_served_sequences() {
        let c = scheduler_config(4096, 2, 0).unwrap();
        assert!(c.prefill.sizes.contains(&SERVE_CHUNK));
    }

    /// INK13: `--ink-mtp K` adds a width-K bucket a batch and the batch table `1:K,2:1`: width K
    /// at one running sequence, 1 above; off keeps the width-1 buckets only.
    #[test]
    fn ink13_the_drafter_widths_follow_the_live_count() {
        let c = scheduler_config(4096, 8, 2).unwrap();
        let none: Vec<&saber_sched::AcceptanceEma> = Vec::new();
        assert_eq!(c.width.step_width(1, none.clone()), 2);
        assert_eq!(c.width.step_width(2, none.clone()), 1);
        assert_eq!(c.width.step_width(8, none.clone()), 1);
        assert_eq!(c.width.bucket_width(3), 2);
        let off = scheduler_config(4096, 8, 0).unwrap();
        assert_eq!(off.width.step_width(1, none), 1);
        assert_eq!(off.max_spec_width(), 1);
    }

    /// INK33: a deeper `--ink-mtp K` (up to 7) widens the ONE-live-sequence round only; the
    /// batched steps keep the batch table's K (the default `2:2,4:2,8:2` + `--ink-mtp 2` = the
    /// pre-INK33 geometry), and `off` keeps the pre-INK33 reservation.
    #[test]
    fn ink33_a_deeper_single_depth_leaves_the_batched_width_alone() {
        let none: Vec<&saber_sched::AcceptanceEma> = Vec::new();
        let d = scheduler_config_b(4096, 8, 2, Some(2), 2).unwrap();
        assert_eq!(d.width.step_width(1, none.clone()), 2);
        assert_eq!(d.width.step_width(8, none.clone()), 2);
        let k7 = scheduler_config_b(4096, 8, 7, Some(2), 7).unwrap();
        assert_eq!(k7.width.step_width(1, none.clone()), 7);
        assert_eq!(k7.width.step_width(2, none.clone()), 2);
        assert_eq!(k7.width.step_width(8, none.clone()), 2);
        assert_eq!(k7.max_spec_width(), 7);
        let solo = scheduler_config_b(4096, 8, 7, None, 7).unwrap();
        assert_eq!(solo.width.step_width(2, none), 1);
        assert_eq!(
            batch_table("2:2,4:2,8:2").unwrap(),
            vec![(2, 2), (4, 2), (8, 2)]
        );
        assert!(batch_table("off").unwrap().is_empty());
        // INK38: K up to 7 (the round clamps to 32 multi-verify rows at run time)
        assert!(batch_table("2:4").is_ok() && batch_table("2:8").is_err());
    }

    /// INK33: the gate's flag and depth rule (empty = off = the control).
    #[test]
    fn ink33_the_gate_verifies_the_confident_prefix() {
        assert_eq!(mtp_gate("").unwrap(), None);
        assert_eq!(mtp_gate("off").unwrap(), None);
        assert_eq!(mtp_gate("prod:0.5:2").unwrap(), Some((true, 0.5, 2, 0)));
        assert_eq!(mtp_gate("each:0.8:1").unwrap(), Some((false, 0.8, 1, 0)));
        assert_eq!(mtp_gate("prod:0.6:2:3").unwrap(), Some((true, 0.6, 2, 3)));
        assert!(mtp_gate("prod:0.6:2:x").is_err());
        assert!(mtp_gate("prod:0.5").is_err());
        assert!(mtp_gate("max:0.5:2").is_err());
        assert!(mtp_gate("prod:1.5:2").is_err());
    }

    #[test]
    fn ink38_the_batched_depth_reaches_7_within_32_rows_and_the_bgate_is_an_arm() {
        assert_eq!(
            batch_table("2:7,4:5,6:4,8:3").unwrap(),
            vec![(2, 7), (4, 5), (6, 4), (8, 3)]
        );
        assert!(batch_table("2:8").is_err());
        assert_eq!(
            (
                mv_depth_cap(2),
                mv_depth_cap(4),
                mv_depth_cap(5),
                mv_depth_cap(7),
                mv_depth_cap(8)
            ),
            (15, 7, 5, 3, 3)
        );
        assert_eq!(mtp_bgate("off").unwrap(), None);
        assert_eq!(
            mtp_bgate("prod:0.6:2:max").unwrap(),
            Some((true, 0.6, 2, BAgg::Max))
        );
        assert!(mtp_bgate("prod:0.6:2").is_err());
        assert!(mtp_bgate("prod:0.6:2:avg").is_err());
        assert_eq!(bgate_depth(&[2, 5, 3], BAgg::Max), 5);
        assert_eq!(bgate_depth(&[2, 5, 3], BAgg::Min), 2);
        assert_eq!(bgate_depth(&[2, 5, 3], BAgg::Mean), 4);
        assert_eq!(InkFlags::default().ink_mtp_bgate, "off");
        let p = [0.9f32, 0.8, 0.9, 0.3, 0.99, 0.99, 0.99];
        // running product 0.9, 0.72, 0.648, 0.194 -> 3 leading at 0.5
        assert_eq!(gated_depth(&p, true, 0.5, 2), 3);
        // own probability: 0.9, 0.8, 0.9 clear 0.5, 0.3 does not
        assert_eq!(gated_depth(&p, false, 0.5, 1), 3);
        // the floor and the cap
        assert_eq!(gated_depth(&[0.1, 0.9], true, 0.5, 2), 2);
        assert_eq!(gated_depth(&[0.99, 0.99], true, 0.5, 1), 2);
        assert_eq!(gated_depth(&[0.1, 0.9], true, 0.5, 1), 1);
    }

    /// INK13: `--ink-mtp 2` is the Inkling serve default; `--ink-mtp 0` = the control.
    #[test]
    fn ink16_the_first_token_on_the_seam_and_the_column_lane_gemm_are_the_defaults() {
        let f = crate::inkling2::InkFlags::default();
        assert_eq!(f.ink_first_token, "seam");
        assert_eq!(f.ink_prefill_gemm, "cl");
        assert_eq!(
            super::super::prefill_gemm(&f.ink_prefill_gemm).unwrap(),
            (true, true, false)
        );
    }

    /// INK18: the drafter's pruned vocabulary + split-K GEMV + 8-bit layers are the defaults;
    /// `--ink-draft-vocab 0 --ink-draft-gemv base --ink-draft-layers bf16` = the control.
    #[test]
    fn ink22_arm_c_is_the_serve_default_and_every_flag_has_its_control() {
        let f = crate::inkling2::InkFlags::default();
        assert_eq!(f.ink_mtp_batch, "2:2,4:2,8:2");
        assert_eq!(f.ink_draft_mm, "wmma");
        assert_eq!(f.ink_fuse, "all17,cl8,r1");
        assert_eq!(f.ink_batch_plain, "rel,first3");
        assert_eq!(f.ink_batch_append, "graph,pfmm");
        let o = super::super::draft::DraftOpt::of(0, "sk", "w8").unwrap();
        assert!(o.clone().with_mm("wmma").unwrap().wmma);
        assert!(!o.with_mm("base").unwrap().wmma);
        assert_eq!(
            super::super::fuse_bits("all17").unwrap() & super::super::FUSE_CL8,
            0
        );
    }

    /// INK26: the user's flips (INK22's arm + INK25's iu8 trunk) are the defaults and
    /// `INK26_CONTROL` parses back to every pre-flip value (the rollback by flag); the prefix
    /// cache stays OFF.
    #[test]
    fn ink26_the_flipped_defaults_and_the_control_flags_roll_every_one_back() {
        use clap::Parser;
        #[derive(clap::Parser)]
        struct T {
            #[command(flatten)]
            f: crate::inkling2::InkFlags,
        }
        let d = T::parse_from(["t"]).f;
        assert_eq!(d, crate::inkling2::InkFlags::default());
        assert_eq!(d.ink_prefill_trunk, "iu8");
        assert_eq!(d.ink_prefix_cache, 0);
        let argv: Vec<&str> = std::iter::once("t")
            .chain(crate::inkling2::INK26_CONTROL.split_whitespace())
            .collect();
        let c = T::parse_from(argv).f;
        assert_eq!(c.ink_mtp_batch, "off");
        assert_eq!(c.ink_draft_mm, "base");
        assert_eq!(c.ink_fuse, "all17");
        assert_eq!(c.ink_batch_plain, "first2");
        assert_eq!(c.ink_batch_append, "each");
        assert_eq!(c.ink_prefill_trunk, "f32");
        assert!(!super::super::prefill_trunk(&c.ink_prefill_trunk).unwrap());
    }

    /// INK29: the prefill levers are the default and `INK29_CONTROL` rolls them back.
    #[test]
    fn ink29_the_prefill_levers_are_the_default_and_the_control_rolls_them_back() {
        use clap::Parser;
        #[derive(clap::Parser)]
        struct T {
            #[command(flatten)]
            f: crate::inkling2::InkFlags,
        }
        let d = T::parse_from(["t"]).f;
        let b = super::super::ink29_bits(&d.ink29).unwrap();
        assert_eq!(
            b,
            super::super::P29_KX
                | super::super::P29_RS
                | super::super::P29_FQ
                | super::super::P29_MT
                | super::super::P29_MZ
                | super::super::P29_B16
                | super::super::P29_BW
        );
        let argv: Vec<&str> = std::iter::once("t")
            .chain(crate::inkling2::INK29_CONTROL.split_whitespace())
            .collect();
        let c = T::parse_from(argv).f;
        assert_eq!(super::super::ink29_bits(&c.ink29).unwrap(), 0);
    }

    /// INK39: `cat` is the default, `INK39_CONTROL` rolls it back, `push` / `all` parse as arms.
    #[test]
    fn ink39_cat_is_the_default_and_the_control_rolls_it_back() {
        use clap::Parser;
        #[derive(clap::Parser)]
        struct T {
            #[command(flatten)]
            f: crate::inkling2::InkFlags,
        }
        let d = T::parse_from(["t"]).f;
        assert_eq!(
            super::super::ink39_bits(&d.ink_p39).unwrap(),
            super::super::P39_CAT
        );
        assert_eq!(
            super::super::ink39_bits(&crate::inkling2::InkFlags::default().ink_p39).unwrap(),
            super::super::P39_CAT
        );
        let argv: Vec<&str> = std::iter::once("t")
            .chain(crate::inkling2::INK39_CONTROL.split_whitespace())
            .collect();
        let c = T::parse_from(argv).f;
        assert_eq!(super::super::ink39_bits(&c.ink_p39).unwrap(), 0);
        assert_eq!(
            super::super::ink39_bits("all").unwrap(),
            super::super::P39_CAT | super::super::P39_PUSH
        );
        assert!(super::super::ink39_bits("fused").is_err());
    }

    #[test]
    fn ink18_the_drafter_defaults() {
        let f = crate::inkling2::InkFlags::default();
        let o =
            super::super::draft::DraftOpt::of(f.draft_vocab(), &f.ink_draft_gemv, f.draft_layers())
                .unwrap();
        assert_eq!((o.vocab, o.sk, o.w8, o.np_max), (32_768, true, true, 4));
        let c = super::super::draft::DraftOpt::of(0, "base", "bf16").unwrap();
        assert_eq!((c.vocab, c.sk, c.w8), (0, false, false));
        assert!(super::super::draft::DraftOpt::of(512, "sk", "w8").is_err());
        assert!(super::super::draft::DraftOpt::of(0, "split", "w8").is_err());
        assert!(super::super::draft::DraftOpt::of(0, "sk", "int4").is_err());
    }

    #[test]
    fn ink13_the_drafter_at_k2_is_the_served_default_and_0_is_the_control() {
        assert_eq!(crate::inkling2::DEFAULT_INK_MTP_UNGATED, 2);
        let off = InkFlags {
            ink_mtp_gate: "off".into(),
            ..InkFlags::default()
        };
        assert_eq!((off.mtp(), off.mtp_gate_spec()), (2, "off"));
        assert_eq!(mtp_gate(off.mtp_gate_spec()).unwrap(), None);
    }

    #[test]
    fn ink34_the_gated_seven_deep_draft_is_the_served_default_and_its_control_is_k2_ungated() {
        let d = InkFlags::default();
        assert_eq!((d.mtp(), d.mtp_gate_spec()), (7, "prod:0.6:2:3"));
        assert_eq!(
            mtp_gate(d.mtp_gate_spec()).unwrap(),
            Some((true, 0.6, 2, 3))
        );
        // an explicit depth keeps its own gate setting (auto = off)
        let k2 = InkFlags {
            ink_mtp: Some(2),
            ..InkFlags::default()
        };
        assert_eq!((k2.mtp(), k2.mtp_gate_spec()), (2, ""));
        // the full head cannot carry the gate: the ungated depth
        let fh = InkFlags {
            ink_draft_vocab: Some(0),
            ..InkFlags::default()
        };
        assert_eq!((fh.mtp(), fh.mtp_gate_spec()), (2, ""));
        assert_eq!(
            crate::inkling2::INK34_MTP_CONTROL,
            "--ink-mtp 2 --ink-mtp-gate off"
        );
        // the batched table is unchanged
        assert_eq!(crate::inkling2::DEFAULT_INK_MTP_BATCH, "2:2,4:2,8:2");
    }

    /// The template renders the oracle's prompt ids (host-only; skips without the checkpoint).
    #[test]
    fn ink8_the_template_renders_the_oracle_prompts() {
        let ckpt = std::path::Path::new("/mnt/nvme/models/Inkling-2-Mini-nvfp4");
        let pj = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../saber-cache/ws-ink1/prompts.jsonl"
        ));
        if !ckpt.is_dir() || !pj.is_file() {
            return;
        }
        let tok = saber_model::Tokenizer::from_file(ckpt.join("tokenizer.json")).unwrap();
        assert_eq!(eos_ids(ckpt).unwrap(), vec![200_006]);
        assert_eq!(
            tok.encode("<|content_model_end_sampling|>").unwrap(),
            vec![200_006]
        );
        let dir = saber_test_scratch::ScratchDir::new("saber-ink8-tpl");
        let tp = dir.path().join("t.jinja");
        std::fs::write(&tp, template_text(ckpt).unwrap()).unwrap();
        let r = saber_model::Renderer::from_file(&tp).unwrap();
        let mut n = 0;
        for line in std::fs::read_to_string(pj).unwrap().lines() {
            let d: serde_json::Value = serde_json::from_str(line).unwrap();
            if d["kind"] == "raw" {
                continue;
            }
            let mut extra = serde_json::Map::new();
            extra.insert("reasoning_effort".into(), d["reasoning_effort"].clone());
            let req = saber_model::ChatRequest {
                messages: d["messages"].as_array().unwrap().clone(),
                tools: None,
                documents: None,
                add_generation_prompt: true,
                extra,
            };
            let text = r.render(&req).unwrap();
            assert_eq!(text, d["text"].as_str().unwrap(), "{}", d["id"]);
            let ids = tok.encode(&text).unwrap();
            let want: Vec<u32> = d["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(ids, want, "{}", d["id"]);
            n += 1;
        }
        drop(dir);
        assert!(n > 0);
    }
}
