//! The Flash-Next serve plane (FN14, plan §8 F4 part 2): the 27B's OpenAI plane over the
//! Flash-Next engine. `saber engine --pack <flash-next pack> --tp 2 --fn-serve PORT` boots the
//! worker-process TP=2 engine exactly as the harness does, then -- on rank 0 -- puts the 27B's
//! scheduler (`saber_sched::Scheduler`: admission, decode buckets 1/2/4/8, one SSE update per
//! step, stop strings, max_tokens), the 27B's sampler (the top-64 seam + the log-sum-exps, prod's
//! order of operations, seeded by output position) and the 27B's HTTP plane
//! (`saber_serve::server`: chat/completions, SSE, logprobs, `/metrics`, the `qwen3_coder` tool
//! parser, the `qwen3` reasoning parser) in front of it. What is Flash-Next's own: THIS
//! checkpoint's chat template (`chat_template.jinja`, `reasoning_effort` xhigh/medium/low; the
//! top-level `reasoning_effort` field is forwarded to it), its tokenizer, its
//! `generation_config.json` (eos ids, the default sampling: temperature 1.0 / top_k 20 / top_p
//! 0.95), and the memory plane's exposition on `/metrics` (`saber_xtier_*`, `saber_ple_*`).
//!
//! The engine thread owns every HIP handle; the scheduler's step thread talks to it through
//! [`FnExec`] (a channel, like the 27B's `HipForward`). Rank 1 follows rank 0's [`Cmd`] stream
//! over the pair's exchange -- the same commands, applied to the same state, in the same order.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

use anyhow::{ensure, Context};
use saber_mem::ledger::{Ledger, Pool};
use saber_plan::{Bucket, Phase};
use saber_sched::{
    ArenaBytes, KvGeometry, MixedPrefill, PrefillChunk, Scheduler, SchedulerConfig, SeqId, SeqStep,
    SlotSampling, StepBatch, StepExecutor, StepResult,
};
use saber_serve::engine::{ModelInfo, SampledLogprobs, TokenId};
use saber_serve::sampler::{DeviceRow, PenaltyContext, RowParams, Sampler};
use saber_serve::sampling::DefaultSampling;
use serde::{Deserialize, Serialize};

use super::run::MAX_ROWS;
use super::worker::{RowOut, RowReq, MAX_SEQS};

/// The Flash-Next vocabulary (the head's rows).
pub const VOCAB: usize = super::exec::VOCAB as usize;

/// One command of the pair's serve protocol: rank 0 decides, sends it, and both ranks apply it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Cmd {
    /// A sequence starts in pool slot `slot` (every region zeroed).
    Zero {
        slot: u32,
        /// FN45c2: the scheduler's KV reservation in tokens (admission by request length under
        /// the drive tier; 0 = the harness: admitted at the max context by its first step).
        #[serde(default)]
        tokens: u32,
        /// FN55b (`--fn-vision`): the sequence's image facts (its M-RoPE table + the tower run
        /// its rows scatter from); absent on a text request (the pre-FN55b frame, byte for byte).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        img: Option<super::vision::ImgFacts>,
    },
    /// FN55b: run the tower over one request's images on both ranks (the pixel rows follow as
    /// the `fn.pix` exchange: `[patches, 1536]` f32 little-endian).
    Encode { grids: Vec<[u32; 3]> },
    /// Prefill `ids` into `slot` (continuing its state); pick the last row with `req`. `last` = the
    /// prompt's last chunk (FN16: a non-last chunk's drafter rows run now, its last row stashed).
    Prefill {
        slot: u32,
        ids: Vec<u32>,
        req: RowReq,
        #[serde(default = "yes")]
        last: bool,
        /// FN119f (`--fn-first-step dev`): the prompt's first token is walked on the card behind
        /// its last chunk (this row's device block + the next round's chain length); `None` =
        /// today's host draw.
        #[serde(default)]
        ah: Option<super::dstep::AheadCmd>,
    },
    /// `slot`'s prompt is in: normalize its state to the batch's parities and place it at the
    /// ready set's end `to` (through the spare slot when `to` holds a prefilling sequence).
    Ready { slot: u32, to: u32 },
    /// A ready sequence left from `from`; the last ready one moves into its hole.
    Leave { from: u32 },
    /// A prefilling sequence left from `slot`.
    Free { slot: u32 },
    /// One decode step over the ready set: the fed token and the pick request per ready slot, at
    /// verify width `w` (FN16: 1 = SPEC 0's step; `w` > 1 = the committed token + the slot's
    /// `w - 1` drafts per sequence, every row picked with the slot's request).
    Decode {
        toks: Vec<u32>,
        reqs: Vec<RowReq>,
        #[serde(default = "one")]
        w: u32,
        /// FN119 (`--fn-device-step ahead`): the step is issued ahead (the device walk + the
        /// drafter round behind the verify); `None` = today's order.
        #[serde(default)]
        ah: Option<super::dstep::AheadCmd>,
        /// LR-N (`--spec-couple on`): each ready slot's coupled-drafting row
        /// ([`super::worker::CoupleRow`]); empty = off (the pre-LR-N frame, byte for byte).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        cp: Vec<super::couple::CoupleRow>,
    },
    /// FN16 (MTP): the last decode step's commit per ready slot -- `acc` drafts accepted and the
    /// last committed token (row `acc`'s decision: the argmax, or F40's draw) -- then the drafter
    /// (F37: pass 1 over the verify rows, then the chain to the next step's width).
    /// FN92: `kn` = the leader's chain length for the next step (the adaptive depth arm at b1;
    /// `None` = each rank's own table law, the frame before FN92).
    Commit {
        acc: Vec<u32>,
        last: Vec<u32>,
        #[serde(default)]
        kn: Option<u32>,
        /// FN119 (`ahead`): the round issued ahead is void (the device walk refused or the chain
        /// length moved): both ranks drain it and run today's round on these commits.
        #[serde(default)]
        redo: bool,
        /// FN119d (`next`): the drafts rank 0 read back for the step this commits when that step
        /// was issued early (each rank reads its own chain's ids): rank 1 checks its own equal,
        /// fail loud. `None` = not an early step.
        #[serde(default)]
        dchk: Option<Vec<Vec<u32>>>,
    },
    /// FN16 (MTP): the drafter's first pass over `slot`'s last prompt chunk (ready at `slot`), the
    /// last row fed the sampled first token `tok`.
    First {
        slot: u32,
        tok: u32,
        /// FN119f: the card's walk of the first token was refused: the round queued behind the
        /// chunk is drained and run again on the host's token.
        #[serde(default)]
        redo: bool,
    },
    /// Both halves of these rows of the last pick's logits (the sampler's dense fallback).
    Dense {
        rows: Vec<u32>,
        /// FN119 (`ahead`): read the rows from the pinned backup the device wrote for the walk's
        /// refused rows (the round ahead reused the arena that held the logits).
        #[serde(default)]
        stage: bool,
    },
    /// FN53b (`--fn-mem-valve on`): both ranks close / reopen valve band segments, between
    /// steps, until `closed` segments are closed (placement only).
    Valve { closed: u32 },
    /// Nothing (rank 0's idle heartbeat: the pair's exchange times out after 300 s of silence).
    Noop,
    /// FN55c (`--fn-vision lazy`): the tower leaves both ranks after the idle time; its region is
    /// lent back to the expert tier.
    VisionRelease,
    /// The serve plane is done.
    Stop,
}

fn yes() -> bool {
    true
}
fn one() -> u32 {
    1
}

/// What the scheduler's step thread asks the engine thread.
#[derive(Debug)]
pub enum Job {
    /// FN45c2: + the scheduler's reservation in tokens (its KV blocks x 16).
    Admit(SeqId, u32),
    /// FN55b: [`Job::Admit`] of an image sequence (its facts).
    AdmitImg(SeqId, u32, super::vision::ImgFacts),
    /// FN55b: the tower over a request's images (grids, pixel rows); the reply is the tower run's
    /// epoch. FN125: (epoch, the deferred run's ticket) -- one of the two is 0.
    Encode(Vec<[u32; 3]>, Vec<f32>, Sender<Result<(u64, u64), String>>),
    Release(SeqId),
    Step(OwnedBatch, Sender<StepResult>),
    /// FN17: a MIXED step (the scheduler's packed form, `--no-fn-mixed` = the control): the decode
    /// half and the prompt chunks packed beside it, run as ONE engine job with one reply -- the
    /// very commands the two unmixed steps would send (the decode step, then each chunk step, in
    /// that order), so every row's bytes are the unmixed rows' (disjoint sequences).
    Mixed(
        OwnedBatch,
        Vec<OwnedBatch>,
        Sender<(StepResult, Vec<StepResult>)>,
    ),
}

/// A [`StepBatch`] the engine thread can own.
#[derive(Debug, Clone)]
pub struct OwnedBatch {
    pub phase: Phase,
    pub slots: Vec<OwnedSlot>,
}

/// One slot of an [`OwnedBatch`].
#[derive(Debug, Clone)]
pub struct OwnedSlot {
    pub seq: SeqId,
    pub input: Vec<TokenId>,
    pub chunk: Option<PrefillChunk>,
    pub sampling: SlotSampling,
    /// Rows the scheduler scheduled this slot (FN16: 0 on a sequence's first step -- it may not
    /// draft --, else the bucket's width; the engine reports `<=` this many drafts).
    pub verify_width: u32,
}

/// FN55b / FN125: an encoded request's facts until its admission.
#[derive(Debug, Clone)]
struct Pending {
    grids: Vec<[u32; 3]>,
    epoch: u64,
    defer: u64,
    hashes: Vec<u64>,
}

/// FN125: each image's content hash (`ImageBatch::content_hash_of`, the prefix key's law), the
/// images hashed on their own threads (a request's ~5 screenshots are ~100 MB of pixels).
#[must_use]
pub fn image_hashes(images: &saber_serve::engine::ImageBatch) -> Vec<u64> {
    let n = images.grids.len();
    if n <= 1 {
        return images.content_hashes();
    }
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..n)
            .map(|k| s.spawn(move || images.content_hash_of(k)))
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("an image hash thread"))
            .collect()
    })
}

/// The scheduler's executor: a channel to the engine thread (which owns every HIP handle).
#[derive(Debug)]
pub struct FnExec {
    tx: Sender<Job>,
    plane: Arc<Mutex<String>>,
    /// FN55b: the tower is resident (`--fn-vision`).
    vision: bool,
    /// FN125 (`--fn-image-cache on` with a cache): each image's content hash rides its facts.
    image_keys: bool,
    hash125: Arc<Mutex<HashStats>>,
    /// FN55b: each encoded request's grids and tower run, read at its admission (bounded: the
    /// scheduler holds the tower for one request until its prompt is in). FN125: + the deferred
    /// run's ticket (0 = the tower ran) and the images' content hashes (empty off the arm).
    pending: std::collections::BTreeMap<saber_serve::engine::RequestId, Pending>,
    /// FN75: the served-step ledger (the scheduler's half lands here at every commit).
    ledger: Arc<Mutex<super::stepledger::StepLedger>>,
}

/// A [`StepBatch`] as the engine thread's own.
fn own(batch: &StepBatch<'_>) -> OwnedBatch {
    OwnedBatch {
        phase: batch.phase,
        slots: batch
            .slots
            .iter()
            .map(|s| OwnedSlot {
                seq: s.seq_id,
                input: s.input_tokens.clone(),
                chunk: s.chunk,
                sampling: s.sampling.clone(),
                verify_width: s.verify_width,
            })
            .collect(),
    }
}

impl StepExecutor for FnExec {
    fn execute(&mut self, batch: &StepBatch<'_>) -> StepResult {
        let (rtx, rrx) = channel();
        if self.tx.send(Job::Step(own(batch), rtx)).is_err() {
            return StepResult::default();
        }
        rrx.recv().unwrap_or_default()
    }
    fn execute_mixed(
        &mut self,
        decode: &StepBatch<'_>,
        chunk: &StepBatch<'_>,
    ) -> (StepResult, StepResult) {
        let (d, mut c) = self.execute_packed(decode, std::slice::from_ref(chunk));
        (d, c.pop().unwrap_or_default())
    }
    fn execute_packed(
        &mut self,
        decode: &StepBatch<'_>,
        chunks: &[StepBatch<'_>],
    ) -> (StepResult, Vec<StepResult>) {
        let (rtx, rrx) = channel();
        let empty = || {
            (
                StepResult::default(),
                chunks.iter().map(|_| StepResult::default()).collect(),
            )
        };
        if self
            .tx
            .send(Job::Mixed(
                own(decode),
                chunks.iter().map(own).collect(),
                rtx,
            ))
            .is_err()
        {
            return empty();
        }
        rrx.recv().unwrap_or_else(|_| empty())
    }
    fn admit(&mut self, seq: &saber_sched::AdmittedSeq) {
        let tokens = u32::try_from(seq.kv_blocks.len() * 16).unwrap_or(u32::MAX);
        // FN55b: an image request's facts from its EXPANDED prompt and its grids (a layout the
        // grids do not account for is admitted as nothing: its first step finds no sequence and
        // the scheduler aborts it -- loud, never an image served as text)
        if let Some(p) = self.pending.get(&seq.request_id) {
            let ids: Vec<u32> = seq.prompt_tokens.clone();
            match super::vision::ImgFacts::keyed(&ids, &p.grids, p.epoch, &p.hashes, p.defer) {
                Ok(f) => {
                    let _ = self.tx.send(Job::AdmitImg(seq.seq_id, tokens, f));
                }
                Err(e) => eprintln!("fnserve: {} refused: {e}", seq.seq_id),
            }
            return;
        }
        let _ = self.tx.send(Job::Admit(seq.seq_id, tokens));
    }
    fn serves_images(&self) -> bool {
        self.vision
    }
    fn encode_images(
        &mut self,
        request: saber_serve::engine::RequestId,
        images: &saber_serve::engine::ImageBatch,
    ) -> Result<(), String> {
        if !self.vision {
            return Err(saber_serve::engine::TEXT_ONLY_REFUSAL.to_owned());
        }
        let (rtx, rrx) = channel();
        self.tx
            .send(Job::Encode(
                images.grids.clone(),
                images.pixel_values.clone(),
                rtx,
            ))
            .map_err(|_| "the engine thread is gone".to_owned())?;
        // FN125: the content hashes while the engine thread takes the pixels (the arm only)
        let hashes = if self.image_keys {
            let t0 = std::time::Instant::now();
            let h = image_hashes(images);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if let Ok(mut s) = self.hash125.lock() {
                s.requests += 1;
                s.images += h.len() as u64;
                s.bytes += images.pixel_values.len() as u64 * 4;
                s.ms += ms;
                s.last_ms = ms;
            }
            h
        } else {
            Vec::new()
        };
        let (epoch, defer) = rrx
            .recv()
            .map_err(|_| "the engine thread is gone".to_owned())??;
        // the scheduler encodes one request at a time and admits it next: older entries are
        // requests that were aborted before admission
        while self.pending.len() >= 64 {
            self.pending.pop_first();
        }
        self.pending.insert(
            request,
            Pending {
                grids: images.grids.clone(),
                epoch,
                defer,
                hashes,
            },
        );
        Ok(())
    }
    fn release(&mut self, seq: SeqId) {
        let _ = self.tx.send(Job::Release(seq));
    }
    fn memory_plane(&self) -> Option<Arc<Mutex<String>>> {
        Some(Arc::clone(&self.plane))
    }
    fn step_done(&mut self, t: &saber_sched::StepTiming) {
        if let Ok(mut l) = self.ledger.lock() {
            l.push_sched(*t);
        }
    }
}

/// The serve plane on rank 0's other threads (the scheduler's step loop + the HTTP runtime).
pub struct Started {
    pub jobs: Receiver<Job>,
    pub join: std::thread::JoinHandle<anyhow::Result<()>>,
    pub plane: Arc<Mutex<String>>,
    /// FN75: the served-step ledger (the engine's half is pushed by rank 0's serve loop).
    pub ledger: Arc<Mutex<super::stepledger::StepLedger>>,
    /// FN125: the image content hashing's cost (the serve plane's half of `saber_fn125_*`).
    pub hash125: Arc<Mutex<HashStats>>,
}

/// FN125: what hashing the requests' images cost on the scheduler thread.
#[derive(Debug, Clone, Copy, Default)]
pub struct HashStats {
    pub requests: u64,
    pub images: u64,
    pub bytes: u64,
    pub ms: f64,
    pub last_ms: f64,
}

impl HashStats {
    /// `/metrics`.
    #[must_use]
    pub fn render(&self) -> String {
        format!(
            "# TYPE saber_fn125_hash_requests_total counter\nsaber_fn125_hash_requests_total {}\n\
             # TYPE saber_fn125_hash_bytes_total counter\nsaber_fn125_hash_bytes_total {}\n\
             # TYPE saber_fn125_hash_ms_total counter\nsaber_fn125_hash_ms_total {:.3}\n\
             # TYPE saber_fn125_hash_ms_last gauge\nsaber_fn125_hash_ms_last {:.3}\n",
            self.requests, self.bytes, self.ms, self.last_ms
        )
    }
}

/// FN126: the boot line of the image budget: the processor's pixel ceiling FOLLOWS the booted
/// tower arena (prod's smart-resize at `arena x 256` px; an image past it is downscaled, never
/// refused) and a request's images run through the tower in groups that fit (only the largest
/// single image must fit the arena).
#[must_use]
pub fn budget_line(arena_patches: u32) -> String {
    let prod = saber_vision_input::ProcessorConfig::QWEN38;
    let c = prod.clamped_to_arena(arena_patches);
    format!(
        "fnserve: FN126 images: pixel budget {} px = the tower arena's {arena_patches} patches x \
         256 (prod's {} px = {} patches; {}) -- an image past it is DOWNSCALED by prod's \
         smart-resize (counted saber_fn126_downscaled_total), a request's images run through the \
         tower in per-image groups that fit (only the largest single image must fit; \
         --fn-vision-max-patches = more detail, not more acceptance)",
        c.max_pixels,
        prod.max_pixels,
        prod.max_pixels / 256,
        if c.max_pixels < prod.max_pixels {
            "CLAMPED"
        } else {
            "prod's own"
        }
    )
}

/// The checkpoint's default sampling (`generation_config.json`), the keys the reference server's
/// `--generation-config auto` reads.
pub fn default_sampling(ckpt: &Path) -> anyhow::Result<DefaultSampling> {
    let p = ckpt.join("generation_config.json");
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&p).with_context(|| p.display().to_string())?,
    )?;
    let g = |k: &str, d: f64| v.get(k).and_then(serde_json::Value::as_f64).unwrap_or(d);
    Ok(DefaultSampling {
        temperature: g("temperature", 1.0) as f32,
        top_p: g("top_p", 1.0) as f32,
        top_k: g("top_k", 0.0) as i32,
        min_p: g("min_p", 0.0) as f32,
        repetition_penalty: g("repetition_penalty", 1.0) as f32,
    })
}

/// The scheduler's geometry for the Flash-Next pools: decode buckets 1/2/4/8 at `width` (1 = SPEC 0;
/// FN16: under `--fn-mtp` every bucket is captured at the widest width the law can ask for, so
/// every slot is scheduled that many rows and the ENGINE runs the law's width for the step --
/// the executor reports the drafts it verified, `<=` the slot's), one prefill rung of 2,048 rows
/// (a Flash-Next chunk runs at its own M: every GEMM band is row-local), KV blocks for `MAX_SEQS`
/// sequences of `max_ctx`, one GDN set per sequence. FN17 (`mixed`): the scheduler's PACKED mixed
/// form -- a decode step carries every queued prompt's next chunk that fits whole beside it (the
/// 27B's C10/C11 law: never a cut chunk; a 1,024-row rung is added so two or more chunks pack in
/// class) -- run by the engine in the serial form (each chunk at its own M, the decode half at its
/// bucket: the unmixed bytes).
///
/// FN38 (`wide` = `--fn-max-chunk`, `busy` = [`super::run::FnFlags::busy_chunk`]): a wide rung
/// above the 2,048 one when `wide` > 2,048 (the chunk runs at its own M, so the rung is a cap, not
/// a padding); `busy` > 0 caps a chunk at that many rows while some sequence decodes.
pub fn scheduler_config(
    max_ctx: u32,
    width: u32,
    mixed: bool,
    wide: u32,
    busy: u32,
    kv_tokens: Option<u64>,
) -> anyhow::Result<SchedulerConfig> {
    // FN45c2: under the drive tier the KV budget is the RAM slab's ACTIVE tokens (admission by
    // request length: a request whose prompt + max_tokens does not fit waits in the queue);
    // otherwise every slot at the max context (the slab holds 8 x it). FN51: the same law on the
    // two tiers past the pinned-RAM ceiling (admission by reservation)
    let admit = kv_tokens;
    if let Some(t) = admit {
        // a request the context allows must fit the slab ALONE (else it could never run)
        let one = u64::from(max_ctx) + u64::from(width.max(1)) + 1;
        ensure!(
            t >= one,
            "admission by reservation: the KV slab admits {t} tokens a rank, under one \
             {max_ctx}-token request ({one} with its verify rows) -- the ceiling is --fn-max-ctx \
             {}, or a larger --fn-kv-ram-gib",
            (t.saturating_sub(u64::from(width.max(1)) + 1) / 256) * 256
        );
    }
    let kv_tokens = kv_tokens.unwrap_or(u64::from(MAX_SEQS) * u64::from(max_ctx));
    let kv = KvGeometry {
        block_tokens: 16,
        bytes_per_token_per_rank: 13_312,
    };
    let mut buckets: Vec<Bucket> = super::worker::BUCKETS
        .iter()
        .map(|&b| Bucket::decode(b, width.max(1), max_ctx, true))
        .collect();
    if mixed {
        buckets.push(Bucket::prefill(MIXED_PACK_RUNG, max_ctx, true));
    }
    buckets.push(Bucket::prefill(MAX_ROWS, max_ctx, true));
    if wide > MAX_ROWS {
        buckets.push(Bucket::prefill(wide, max_ctx, true));
    }
    let gdn_slot = 36 * 24 * 128 * 128 * 2;
    let n = u64::from(MAX_SEQS);
    let probe = SchedulerConfig::for_mock(
        buckets.clone(),
        2,
        gdn_slot,
        kv,
        ArenaBytes {
            kv_per_rank: kv_tokens * kv.bytes_per_token_per_rank,
            gdn_state_per_rank: 1 << 40,
        },
        max_ctx as usize,
        "flashnext:fn14".into(),
    )?;
    let per_seq = probe.gdn_bytes_per_seq();
    let mut c = SchedulerConfig::for_mock(
        buckets,
        2,
        gdn_slot,
        kv,
        ArenaBytes {
            kv_per_rank: kv_tokens * kv.bytes_per_token_per_rank,
            gdn_state_per_rank: n * per_seq,
        },
        max_ctx as usize,
        "flashnext:fn14".into(),
    )?;
    if mixed {
        c.mixed = MixedPrefill::packed(MixedPrefill::PROD_MAX_BATCHED_TOKENS);
    }
    c.prefill = c.prefill.clone().with_busy_max(busy);
    if admit.is_some() && max_ctx > super::kvwire::DEFAULT_FN_KV_MAX_CTX {
        // FN51: a request without max_tokens reserves what it did at the 10,240 default
        c.implicit_context = Some(super::kvwire::DEFAULT_FN_KV_MAX_CTX as usize);
        c.implicit_headroom = IMPLICIT_HEADROOM;
    }
    Ok(c)
}

/// FN51: tokens a request WITHOUT `max_tokens` may generate past a prompt longer than the
/// pre-FN51 served context (10,240) under admission by reservation (its reservation = prompt +
/// this; an explicit `max_tokens` is clamped to the context as before).
pub const IMPLICIT_HEADROOM: usize = 16_384;

/// FN17: the rung below the 2,048 chunk that lets the packing law put two or more prompts' chunks
/// (+ the decode rows) in one mixed step (the law packs while they sit below `LARGE_MIN_M`).
pub const MIXED_PACK_RUNG: u32 = 1024;

/// Rank 0: build the scheduler + the HTTP plane on their own threads and hand back the job
/// channel the engine thread serves.
#[allow(clippy::too_many_arguments)]
pub fn start(
    port: u16,
    ckpt: &Path,
    template: Option<&PathBuf>,
    served: &str,
    max_ctx: u32,
    thinking: bool,
    width: u32,
    mixed: bool,
    wide: u32,
    busy: u32,
    yield_steps: u32,
    kv_lazy: u32,
    kv_tokens: Option<u64>,
    vision: Option<u32>,
    image_keys: bool,
    warm: Option<(
        Vec<saber_serve::warm::WarmPrompt>,
        saber_serve::warm::CacheProbe,
    )>,
    step_trace: Option<&Path>,
) -> anyhow::Result<Started> {
    let tokenizer = Arc::new(
        saber_model::Tokenizer::from_file(ckpt.join("tokenizer.json"))
            .context("the checkpoint's tokenizer.json")?,
    );
    let tpath = template
        .cloned()
        .unwrap_or_else(|| ckpt.join("chat_template.jinja"));
    let renderer = Arc::new(
        saber_serve::state::TemplateRenderer::from_file(&tpath)
            .with_context(|| format!("loading {}", tpath.display()))?,
    );
    let eos = crate::serve::eos_ids(&tokenizer, ckpt)?;
    let sampling = default_sampling(ckpt)?;
    let mut config = scheduler_config(max_ctx, width, mixed, wide, busy, kv_tokens)?;
    // LR-B (`--fn-prefill-yield`, an arm): decode steps between two prefill chunks
    config.prefill = config.prefill.clone().with_yield_steps(yield_steps);
    // LR-G (`--fn-kv-lazy`, an arm): reserve prompt + min(max_tokens, N), grow, preempt; a lazy
    // admission leaves N tokens free for the running sequences' growth
    config.admit.lazy_headroom = kv_lazy;
    config.admit.lazy_margin = kv_lazy;
    let rungs = config.prefill.sizes.clone();
    let model = ModelInfo {
        served_model_name: served.to_owned(),
        aliases: Vec::new(),
        max_model_len: max_ctx as usize,
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
    let plane = Arc::new(Mutex::new(String::new()));
    let steps75 = Arc::new(Mutex::new(
        super::stepledger::StepLedger::new(step_trace)
            .context("--fn-step-trace: creating the trace")?,
    ));
    let hash125 = Arc::new(Mutex::new(HashStats::default()));
    let exec = FnExec {
        tx,
        plane: Arc::clone(&plane),
        vision: vision.is_some(),
        image_keys,
        hash125: Arc::clone(&hash125),
        pending: std::collections::BTreeMap::new(),
        ledger: Arc::clone(&steps75),
    };
    let scheduler = Arc::new(Scheduler::new(config, exec, model, ledger)?);
    let parsers = saber_serve::state::ParserFlags::default()
        .resolve()
        .map_err(anyhow::Error::msg)?;
    println!(
        "fnserve: template {} (reasoning_effort xhigh|medium|low), tokenizer {}, eos {eos:?}, \
         default sampling {sampling:?}, parsers tool {:?} reasoning {:?}, buckets 1/2/4/8 x width {width}, \
         prefill rungs {:?} (beside decode: {}), max ctx {max_ctx}, {MAX_SEQS} sequences, thinking by default {thinking}, \
         mixed step {}, decode steps between chunks {yield_steps}",
        tpath.display(),
        ckpt.join("tokenizer.json").display(),
        parsers.tool_call_parser_name,
        parsers.reasoning_parser_name,
        rungs,
        if busy > 0 {
            format!("<= {busy} rows")
        } else {
            "any rung".to_string()
        },
        if mixed {
            "PACKED (--fn-mixed; serial execution: every row's bytes are the unmixed rows')"
        } else {
            "off (every chunk a step of its own; --fn-mixed = the packed arm)"
        },
    );
    let steps = saber_sched::spawn_step_loop(Arc::clone(&scheduler));
    let state = saber_serve::state::ServerState::builder(
        Arc::clone(&scheduler) as Arc<dyn saber_serve::engine::Engine>,
        Arc::new(saber_serve::detok::ModelCodec::new(Arc::clone(&tokenizer))),
        renderer,
    )
    .parsers(&parsers)
    .default_sampling(sampling)
    .default_enable_thinking(thinking)
    .max_logprobs(20)
    // FN25: the reference 0.29's tool-call `arguments` law (the template iterates `arguments|items`; the
    // OpenAI wire sends a JSON string) -- the plane FN2's template gate is measured against.
    .tool_arguments(saber_serve::state::ToolArguments::Objects);
    // FN28: the conversation stays `AsSent` (the builder default) -- the reference 0.29's
    // `_parse_chat_message_content` rebuild is the same code as 0.27.1's, but aligning the
    // Flash-Next plane to it is the user's call (LEDGER FN28).
    // FN55b: image parts when the tower is resident (`--fn-vision`; the reference server's default limit 999);
    // a text-only boot's state is the pre-FN55b one
    let state = match vision {
        Some(p) => state.vision(p, 999),
        None => state,
    }
    .build();
    // FN126: the image budget follows the booted tower arena
    if let Some(p) = vision {
        println!("{}", budget_line(p));
    }
    let join = std::thread::Builder::new().name("fn-http".into()).spawn(
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
                    "fnserve listening http://{}/v1  (SIGINT or SIGTERM to stop)",
                    listener.local_addr()?
                );
                // FN66: the warm prompts run beside the listener (requests queue behind them)
                if let Some((prompts, probe)) = warm {
                    tokio::spawn(saber_serve::warm::run(Arc::clone(&state), prompts, probe));
                }
                saber_serve::server::serve_with_shutdown(
                    state,
                    listener,
                    // FN73: SIGINT/SIGTERM, or the fatal (a failed rank)
                    crate::engine::shutdown_or_fatal(),
                )
                .await
                .context("serving")
            })?;
            match saber_sched::stop_with_deadline(
                scheduler,
                steps,
                std::time::Duration::from_secs(30),
            ) {
                Ok(ran) => println!("fnserve stopped: {ran} engine steps"),
                Err(e) => println!("fnserve stuck at stop: {e}"),
            }
            Ok(())
        },
    )?;
    Ok(Started {
        jobs,
        join,
        plane,
        ledger: steps75,
        hash125,
    })
}

/// The seam a merged pick hands the 27B's sampler.
#[must_use]
pub fn device_row(o: &RowOut, penalty_ids: &[TokenId]) -> DeviceRow {
    DeviceRow {
        top_values: o.top.iter().map(|x| x.1).collect(),
        top_ids: o.top.iter().map(|x| x.0).collect(),
        logsumexp: o.lse,
        logsumexp_at_temperature: o.lse_t,
        penalised: penalty_ids
            .iter()
            .zip(&o.probe)
            .map(|(&i, &v)| (i, v))
            .collect(),
        vocab_size: VOCAB,
    }
}

/// What rank 0 asks of a row's pick for this slot's sampling (penalty logits, top_p's mass).
#[must_use]
pub fn row_req(s: &SlotSampling) -> (RowParams, Vec<TokenId>, RowReq) {
    let params = RowParams::resolve(&s.params, VOCAB);
    let ctx = penalty_ctx(s);
    let ids = params.penalty_ids(&ctx);
    let temp = if !params.is_greedy() && params.top_p < 1.0 {
        params.temperature
    } else {
        0.0
    };
    let lse = params.logprobs.is_some() || !params.is_greedy();
    (
        params,
        ids.clone(),
        RowReq {
            probes: ids,
            temp,
            lse,
            tk: 0,
        },
    )
}

fn penalty_ctx(s: &SlotSampling) -> PenaltyContext<'_> {
    match &s.penalties {
        Some(h) => PenaltyContext {
            prompt: &h.prompt,
            output: &h.output,
        },
        None => PenaltyContext::default(),
    }
}

/// Sample one row: from the seam (`dense` None) or from the whole row. `None` = the seam refused
/// (the caller reads the dense row and asks again); a dense refusal commits the seam's argmax.
#[must_use]
pub fn sample(
    sampler: &Sampler,
    o: &RowOut,
    dense: Option<Vec<f32>>,
    params: &RowParams,
    pen: &[TokenId],
    s: &SlotSampling,
) -> Option<(TokenId, Option<SampledLogprobs>)> {
    let ctx = penalty_ctx(s);
    let mut noise: Box<dyn saber_serve::sampler::NoiseSource> = match s.params.seed {
        Some(seed) => Box::new(saber_serve::sampler::SeededNoise::new(
            seed,
            s.position,
            s.choice_index,
        )),
        None => Box::new(saber_serve::sampler::EntropyNoise::new()),
    };
    match dense {
        Some(mut d) => Some(
            sampler
                .sample_dense(&mut d, params, &ctx, noise.as_mut())
                .map_or((o.top[0].0, None), |x| (x.token_id, x.logprobs)),
        ),
        None => sampler
            .sample(&device_row(o, pen), params, &ctx, noise.as_mut())
            .ok()
            .map(|x| (x.token_id, x.logprobs)),
    }
}

/// A verify step's result for one sequence (FN16): the committed tokens (the accepted drafts +
/// one), the drafts verified and accepted, one logprobs entry per token when asked.
#[must_use]
pub fn seq_step_k(
    seq: SeqId,
    tokens: Vec<TokenId>,
    drafted: u32,
    accepted: u32,
    lp: Option<Vec<SampledLogprobs>>,
) -> SeqStep {
    SeqStep {
        seq_id: seq,
        tokens,
        draft_tokens: drafted,
        accepted_tokens: accepted,
        logprobs: lp,
        prompt_logprobs: None,
    }
}

/// A step's result for one sequence.
#[must_use]
pub fn seq_step(seq: SeqId, tok: Option<TokenId>, lp: Option<SampledLogprobs>) -> SeqStep {
    match tok {
        Some(t) => SeqStep::single(seq, t).with_logprobs(lp.map(|x| vec![x])),
        None => SeqStep {
            seq_id: seq,
            tokens: Vec::new(),
            draft_tokens: 0,
            accepted_tokens: 0,
            logprobs: None,
            prompt_logprobs: None,
        },
    }
}

/// Encode / decode a [`Cmd`] for the pair's exchange.
pub fn enc(c: &Cmd) -> anyhow::Result<Vec<u8>> {
    Ok(serde_json::to_vec(c)?)
}
pub fn dec(b: &[u8]) -> anyhow::Result<Cmd> {
    serde_json::from_slice(b).context("a serve command")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FN125: the threaded hashes are the plane's own (`content_hashes`), one per image, and two
    /// images differing in one pixel differ.
    #[test]
    fn fn125_the_threaded_image_hashes_are_the_planes() {
        let mut b = saber_serve::engine::ImageBatch {
            grids: vec![[1, 4, 8], [1, 2, 2], [1, 4, 4]],
            pixel_values: (0..(32 + 4 + 16) * 1536)
                .map(|i| (i % 977) as f32)
                .collect(),
        };
        assert_eq!(image_hashes(&b), b.content_hashes());
        let h0 = image_hashes(&b);
        b.pixel_values[32 * 1536 + 5] += 1.0;
        let h1 = image_hashes(&b);
        assert_eq!(h0[0], h1[0]);
        assert_ne!(h0[1], h1[1]);
        assert_eq!(h0[2], h1[2]);
    }

    #[test]
    fn the_command_protocol_round_trips() {
        let c = Cmd::Decode {
            toks: vec![1, 2, 3],
            reqs: vec![RowReq {
                probes: vec![7, 8],
                temp: 0.7,
                lse: true,
                tk: 20,
            }],
            w: 4,
            ah: Some(super::super::dstep::AheadCmd {
                blocks: vec![[1, 2, 3, 4, 5, 6, 7, 8]],
                kn: 3,
            }),
            cp: vec![[9, 10, 11, 0x3fb6_db6e, 20]],
        };
        assert_eq!(dec(&enc(&c).unwrap()).unwrap(), c);
        assert_eq!(dec(&enc(&Cmd::Stop).unwrap()).unwrap(), Cmd::Stop);
        for c in [
            Cmd::Commit {
                acc: vec![0, 3],
                last: vec![9, 10],
                kn: None,
                redo: false,
                dchk: None,
            },
            Cmd::Commit {
                acc: vec![2],
                last: vec![11],
                kn: Some(4),
                redo: true,
                dchk: Some(vec![vec![1, 2, 3]]),
            },
            Cmd::First {
                slot: 2,
                tok: 77,
                redo: false,
            },
            Cmd::Prefill {
                slot: 1,
                ids: vec![5, 6],
                req: RowReq::default(),
                last: false,
                ah: None,
            },
        ] {
            assert_eq!(dec(&enc(&c).unwrap()).unwrap(), c);
        }
        // a SPEC-0 frame without the FN16 fields reads as width 1 / the last chunk
        let old = r#"{"Decode":{"toks":[1],"reqs":[{"probes":[],"temp":0.0,"lse":false}]}}"#;
        assert!(matches!(
            dec(old.as_bytes()).unwrap(),
            Cmd::Decode { w: 1, .. }
        ));
    }

    #[test]
    fn the_scheduler_geometry_is_buckets_1_2_4_8_at_width_1_and_one_prefill_rung() {
        let m = scheduler_config(4096, 4, false, MAX_ROWS, 0, None).unwrap();
        assert!(m.buckets.iter().all(|b| b.spec_width == 4));
        assert_eq!(m.max_spec_width(), 4);
        let c = scheduler_config(4096, 1, false, MAX_ROWS, 0, None).unwrap();
        let b: Vec<u32> = c.buckets.iter().map(|b| b.batch).collect();
        assert_eq!(b, vec![1, 2, 4, 8]);
        assert!(c.buckets.iter().all(|b| b.spec_width == 1));
        assert_eq!(c.prefill.sizes, vec![MAX_ROWS]);
        assert_eq!(c.max_batch(), 8);
        // FN17: the mixed form packs (the 27B's law) with a 1,024 rung below the chunk
        let x = scheduler_config(4096, 1, true, MAX_ROWS, 0, None).unwrap();
        assert!(x.mixed.enabled && x.mixed.packs());
        assert_eq!(x.prefill.sizes, vec![MIXED_PACK_RUNG, MAX_ROWS]);
        assert_eq!(
            x.prefill.chunk_for(3000),
            Some((MAX_ROWS, MAX_ROWS)),
            "no cut moves"
        );
        assert_eq!(x.prefill.chunk_for(952), Some((952, MIXED_PACK_RUNG)));
        assert!(!c.mixed.enabled);
    }

    #[test]
    fn fn38_the_wide_rungs_cap_idle_chunks_and_packing_never_cuts() {
        use saber_sched::mixed_law::{chunk_sequence_holds, pack};
        for wide in [4096u32, 8192] {
            // unmixed, idle-only wide rung
            let c = scheduler_config(16384, 1, false, wide, MAX_ROWS, None).unwrap();
            assert_eq!(c.prefill.sizes, vec![MAX_ROWS, wide]);
            assert_eq!(c.prefill.max_chunk(), wide);
            let idle = c.prefill.room(false);
            let busy = c.prefill.room(true);
            assert_eq!(c.prefill.chunk_for(12000u32.min(idle)), Some((wide, wide)));
            assert_eq!(
                c.prefill.chunk_for(12000u32.min(busy)),
                Some((MAX_ROWS, MAX_ROWS))
            );
            // a prompt that fits the wide rung is ONE chunk at its own M (the rung is a cap)
            assert_eq!(c.prefill.chunk_for(3000), Some((3000, wide)));
            assert_eq!(
                c.prefill.cut(wide * 2 + 5),
                vec![(wide, wide), (wide, wide), (5, MAX_ROWS)]
            );
            // `always`: no cap beside decode
            let a = scheduler_config(16384, 1, false, wide, 0, None).unwrap();
            assert_eq!(a.prefill.room(true), u32::MAX);
            // mixed (packed): beside decode the chunks keep the 1,024 / 2,048 rungs; the packing
            // law carries the unpacked (capped) chunks WHOLE (C10/C11: never a cut chunk)
            let x = scheduler_config(16384, 1, true, wide, MAX_ROWS, None).unwrap();
            assert_eq!(x.prefill.sizes, vec![MIXED_PACK_RUNG, MAX_ROWS, wide]);
            let unpacked: Vec<u32> = [12000u32, 952, 3000]
                .iter()
                .map(|&n| x.prefill.chunk_for(n.min(x.prefill.room(true))).unwrap().0)
                .collect();
            assert_eq!(unpacked, vec![MAX_ROWS, 952, MAX_ROWS]);
            let p = pack(
                &unpacked,
                8,
                &x.prefill.sizes,
                x.mixed.max_batched_tokens,
                x.mixed.max_chunks,
            );
            for (k, r) in p.rows.iter().enumerate() {
                assert_eq!(*r, unpacked[k], "packed chunk {k} is the unpacked one");
            }
            // a 12,000-token prompt fed beside decode throughout: 2,048 x 5 + 1,760, contiguous
            let mut fed = Vec::new();
            let (mut done, n) = (0u32, 12000u32);
            while done < n {
                let (r, _) = x
                    .prefill
                    .chunk_for((n - done).min(x.prefill.room(true)))
                    .unwrap();
                fed.push((done, r));
                done += r;
            }
            let cut: Vec<u32> = fed.iter().map(|f| f.1).collect();
            assert_eq!(cut, vec![2048, 2048, 2048, 2048, 2048, 1760]);
            chunk_sequence_holds(&cut, &fed).unwrap();
        }
    }

    #[test]
    fn a_greedy_row_is_the_seam_argmax_and_top_p_asks_for_the_tempered_mass() {
        let o = RowOut {
            top: (0..super::super::worker::SEAM_K as u32)
                .map(|i| (i + 10, 5.0 - i as f32 * 0.01))
                .collect(),
            lse: 7.0,
            lse_t: Some(7.5),
            probe: Vec::new(),
            sha: None,
        };
        let mut s = SlotSampling::greedy();
        let (p, pen, rq) = row_req(&s);
        assert_eq!(rq.temp, 0.0);
        let (t, _) = sample(&Sampler::new(), &o, None, &p, &pen, &s).unwrap();
        assert_eq!(t, 10);
        s.params.temperature = 0.7;
        s.params.top_p = 0.95;
        s.params.top_k = 20;
        let (_, _, rq) = row_req(&s);
        assert!((rq.temp - 0.7).abs() < 1e-6);
    }
}
