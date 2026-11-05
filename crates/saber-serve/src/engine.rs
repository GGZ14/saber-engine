//! The boundary between the serving plane and the engine.
//!
//! `saber-exec` / `saber-sched` implement [`Engine`]; this crate only consumes it.
//! The unit of the contract is the **engine step**, not the token: with DFlash2 block
//! speculative decoding one step commits 1..=8 tokens, and production's SSE stream —
//! the thing BetterBench times as `update p50` — emits exactly one update per step.
//! Making the step the unit here is what keeps that property from being an accident
//! of the HTTP layer (see `docs/ARCHITECTURE.md` § "5. Serving plane").
//!
//! [`MockEngine`] replays a scripted sequence of steps, so every test in this crate
//! runs on the host with no GPU, no HIP and no model.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_core::Stream;

use crate::sampling::SamplingParams;

/// A vocabulary id. Same width as the tokenizer's, and as `saber-model`'s.
pub type TokenId = u32;

/// Identifies one in-flight generation inside the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(pub u64);

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Why generation stopped. The engine owns `Stop` (an EOS or stop token id) and
/// `Length`; the serving plane owns stop *strings* and `ToolCalls`, because it is the
/// side that detokenizes and parses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    Abort,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::Abort => "abort",
        }
    }
}

/// One entry of a logprob distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    pub token_id: TokenId,
    pub logprob: f32,
    /// 1-based rank within the distribution, as the reference server reports it.
    pub rank: Option<u32>,
}

/// The logprobs for one sampled token: the token itself plus the requested top-k.
#[derive(Debug, Clone, PartialEq)]
pub struct SampledLogprobs {
    pub sampled: TokenLogprob,
    pub top: Vec<TokenLogprob>,
}

/// `prompt_logprobs` for one prompt position. `None` for position 0, which has no
/// conditional distribution — that is also how the reference server reports it, and `ppl.py` /
/// `kld.py` depend on the shape.
pub type PromptPositionLogprobs = Option<Vec<TokenLogprob>>;

/// **One engine step, for one choice.** With speculative decoding `new_tokens`
/// carries 1..=8 tokens; at SPEC=0 it carries exactly one. The serving plane turns
/// each of these into exactly one SSE chunk.
///
/// With `n > 1` a request is one **sequence group**: the prompt is prefilled once and
/// the branches decode independently, so one request's stream carries `n` interleaved
/// choices and `choice_index` is what routes a delta to `choices[i]`. Each choice
/// carries its own `step_index` (from 0) and exactly one `finished`; the request is
/// over when every choice has had one.
#[derive(Debug, Clone, PartialEq)]
pub struct StepEvent {
    pub request_id: RequestId,
    /// Which of the request's `n` choices this step belongs to, `0..n`.
    pub choice_index: u32,
    /// 0-based index of this step **within this choice**.
    pub step_index: u32,
    pub new_tokens: Vec<TokenId>,
    /// `Some` on the last event of a request, and only there.
    pub finished: Option<FinishReason>,
    /// One entry per token in `new_tokens`, when logprobs were requested.
    pub logprobs: Option<Vec<SampledLogprobs>>,
    /// Prompt logprobs (the reference server extension). Delivered once, on the first step.
    pub prompt_logprobs: Option<Vec<PromptPositionLogprobs>>,
    /// Why the ENGINE ended this choice, when it ended it for a reason the client must be told
    /// (ws-k18: a parked session whose restore failed `max_restore_attempts` times). Only ever
    /// `Some` on the `finished` event, beside `FinishReason::Abort`; the serving plane turns it into
    /// an OpenAI error body (a 500 for a non-streaming request, an `error` event before `[DONE]`
    /// on a stream), which is what prod does with an engine exception.
    pub error: Option<String>,
}

/// What the serving plane hands the engine.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineRequest {
    pub prompt_token_ids: Vec<TokenId>,
    pub params: SamplingParams,
}

/// A request's images, processed (D3): prod's `pixel_values` and `image_grid_thw`, in the order
/// the `<|image_pad|>` runs appear in the prompt. The prompt ids already carry each image's run
/// expanded to its token count.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageBatch {
    /// `image_grid_thw`, one per image.
    pub grids: Vec<[u32; 3]>,
    /// `[sum of patches, 1536]` f32, the images concatenated in order.
    pub pixel_values: Vec<f32>,
}

impl ImageBatch {
    /// Total patches.
    pub fn patches(&self) -> u64 {
        self.grids
            .iter()
            .map(|g| u64::from(g[0]) * u64::from(g[1]) * u64::from(g[2]))
            .sum()
    }

    /// Total image tokens (patches / 4).
    pub fn tokens(&self) -> u64 {
        self.patches() / 4
    }

    /// One content hash per image (D4): FNV-1a over the grid and the processed `pixel_values`'
    /// f32 bits -- what a prefix cache must fold into the key of every block an image's tokens
    /// touch (`saber_sched::prefix_key`), so two different images with the same pad ids never
    /// share a block.
    pub fn content_hashes(&self) -> Vec<u64> {
        (0..self.grids.len())
            .map(|k| self.content_hash_of(k))
            .collect()
    }

    /// Image `k`'s content hash ([`Self::content_hashes`]' law for one image: a caller may hash
    /// the images on their own threads, FN125).
    pub fn content_hash_of(&self, k: usize) -> u64 {
        let rows = |g: &[u32; 3]| (g[0] * g[1] * g[2]) as usize * 1536;
        let at: usize = self.grids[..k].iter().map(rows).sum();
        let g = &self.grids[k];
        let n = rows(g);
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |v: u32| {
            for b in v.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for v in g {
            eat(*v);
        }
        for p in self.pixel_values.get(at..at + n).unwrap_or(&[]) {
            eat(p.to_bits());
        }
        h
    }
}

/// The refusal an engine with no vision tower gives an image, by name.
pub const TEXT_ONLY_REFUSAL: &str = "this server serves TEXT ONLY: start it with `--vision` (the \
     tower and its arena are not resident) to send image parts (parameter=image)";

/// A stream of the steps of **one** request. `StepEvent::request_id` is still carried
/// so a scheduler can fan one global step out to many of these.
pub type StepStream = Pin<Box<dyn Stream<Item = StepEvent> + Send>>;

/// The result of [`Engine::submit`].
pub struct Submission {
    pub id: RequestId,
    pub events: StepStream,
}

impl std::fmt::Debug for Submission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Submission")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

/// Engine-side counters `/metrics` exports. Token counts are *not* here: the serving
/// plane counts those itself, because it is what computes `usage`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EngineStats {
    pub num_requests_running: u64,
    pub num_requests_waiting: u64,
    pub spec_decode_num_drafts: u64,
    pub spec_decode_num_draft_tokens: u64,
    pub spec_decode_num_accepted_tokens: u64,
    /// `saber:spec_decode_num_accepted_tokens_per_pos_total[i]` — drafts whose accepted
    /// prefix reached past position `i`, one entry per draft position.
    ///
    /// Prod exports this (`upstream/v1/spec_decode/metrics.py`, `pos_labelnames = labelnames +
    /// ["position"]`) and Saber's exposition did not, although
    /// [`saber_model::draft::SpecDecodeCounters`] has folded it since the counters were
    /// written. The three totals say acceptance *is* 0.9 per draft; only this vector says
    /// **where** it dies, and a `/metrics` scrape is the only view of it a harness outside
    /// the process has. It is a `Vec` rather than a fixed width because the width is the
    /// run's verify width, which the serving plane does not know.
    pub spec_decode_accepted_per_pos: Vec<u64>,
    /// The engine's, the drafter's and the scheduler's which-path counters (ws-v8), each with the
    /// shutdown census line it mirrors ([`crate::census`]). Empty for an engine that keeps none
    /// (the mock); rendered as the `saber:` family.
    pub census: Vec<crate::census::CensusSample>,
    /// The Flash-Next memory plane's own exposition (FN8): `saber_xtier_*` (per rank, per layer,
    /// by path) and `saber_ple_*`, rendered by `saber_mem::expert_tier::ExpertTier::render_prometheus`
    /// and `saber_mem::ple_tier::PleTier::render_prometheus` and appended verbatim. Empty for an
    /// engine without the tier (every 27B engine).
    pub memory_plane: String,
}

/// Static facts about the loaded model that `/v1/models` and the request validator
/// need.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    /// The name clients send and responses echo (prod: `Qwen3.8`).
    pub served_model_name: String,
    /// Extra aliases `/v1/models` lists and requests may use.
    pub aliases: Vec<String>,
    /// `--max-model-len`.
    pub max_model_len: usize,
    /// The checkpoint path, reported as `root` by `/v1/models`.
    pub root: String,
    /// Token ids that end generation; they count towards `usage` but are never
    /// detokenized into the response text.
    pub eos_token_ids: Vec<TokenId>,
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            served_model_name: "Qwen3.8".to_owned(),
            aliases: Vec::new(),
            max_model_len: 262_144,
            root: String::new(),
            eos_token_ids: Vec::new(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("engine is at capacity")]
    AtCapacity,
    #[error("engine is shutting down")]
    ShuttingDown,
    #[error("prompt is {got} tokens, longer than the model's {max}")]
    PromptTooLong { got: usize, max: usize },
    #[error("engine error: {0}")]
    Other(String),
    /// A request this engine cannot serve AS ASKED, refused before anything runs: a 400 naming
    /// the parameter, never a 500 and never a silently different answer (rule 5). `n > 1` with a
    /// resident drafter and `prompt_logprobs` on an engine that cannot produce them are the two.
    #[error("{message}")]
    Unsupported {
        /// The request field the refusal is about.
        param: &'static str,
        /// reference-server-style text, `(parameter=…)` suffix included.
        message: String,
    },
}

/// The engine, as the serving plane sees it.
///
/// Implementors: one call to [`Engine::submit`] starts a generation and returns a
/// stream that yields **one [`StepEvent`] per engine step** until exactly one event
/// carries `finished`. Dropping the stream is not an abort — call [`Engine::abort`].
pub trait Engine: Send + Sync + std::fmt::Debug + 'static {
    fn submit(&self, request: EngineRequest) -> Result<Submission, EngineError>;
    /// A request with images (D3). The default is the text-only engine's refusal, by name.
    fn submit_with_images(
        &self,
        request: EngineRequest,
        images: ImageBatch,
    ) -> Result<Submission, EngineError> {
        let _ = (request, images);
        Err(EngineError::Unsupported {
            param: "image",
            message: TEXT_ONLY_REFUSAL.to_owned(),
        })
    }
    fn abort(&self, id: RequestId);
    fn stats(&self) -> EngineStats;
    fn model(&self) -> &ModelInfo;
}

// ── MockEngine ──────────────────────────────────────────────────────────────────

/// One scripted step.
#[derive(Debug, Clone, PartialEq)]
pub struct MockStep {
    pub tokens: Vec<TokenId>,
    pub finished: Option<FinishReason>,
    /// How many tokens the drafter proposed, for the spec-decode metrics.
    pub draft_tokens: u32,
    pub logprobs: Option<Vec<SampledLogprobs>>,
    /// The engine's reason for ending the choice (`StepEvent::error`), for the error path's tests.
    pub error: Option<String>,
}

impl MockStep {
    /// A step that commits `tokens` and keeps going.
    pub fn new(tokens: impl IntoIterator<Item = TokenId>) -> Self {
        Self {
            tokens: tokens.into_iter().collect(),
            finished: None,
            draft_tokens: 0,
            logprobs: None,
            error: None,
        }
    }

    /// End the choice with `Abort` and the engine's named reason (ws-k18).
    pub fn failed(mut self, message: impl Into<String>) -> Self {
        self.finished = Some(FinishReason::Abort);
        self.error = Some(message.into());
        self
    }

    pub fn finish(mut self, reason: FinishReason) -> Self {
        self.finished = Some(reason);
        self
    }

    pub fn drafted(mut self, draft_tokens: u32) -> Self {
        self.draft_tokens = draft_tokens;
        self
    }

    pub fn with_logprobs(mut self, logprobs: Vec<SampledLogprobs>) -> Self {
        self.logprobs = Some(logprobs);
        self
    }
}

/// A scripted generation: the steps one request will produce.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MockScript {
    pub steps: Vec<MockStep>,
    pub prompt_logprobs: Option<Vec<PromptPositionLogprobs>>,
}

impl MockScript {
    pub fn new(steps: Vec<MockStep>) -> Self {
        Self {
            steps,
            prompt_logprobs: None,
        }
    }

    /// Every token as its own step — the SPEC=0 shape.
    pub fn per_token(tokens: impl IntoIterator<Item = TokenId>, finish: FinishReason) -> Self {
        let tokens: Vec<TokenId> = tokens.into_iter().collect();
        let last = tokens.len().saturating_sub(1);
        Self::new(
            tokens
                .into_iter()
                .enumerate()
                .map(|(i, t)| {
                    let step = MockStep::new([t]).drafted(0);
                    if i == last {
                        step.finish(finish)
                    } else {
                        step
                    }
                })
                .collect(),
        )
    }

    /// Fixed-width steps — the speculative shape BetterBench classifies as `batched`.
    pub fn in_steps_of(
        tokens: impl IntoIterator<Item = TokenId>,
        width: usize,
        finish: FinishReason,
    ) -> Self {
        let tokens: Vec<TokenId> = tokens.into_iter().collect();
        let chunks: Vec<Vec<TokenId>> = tokens.chunks(width.max(1)).map(<[u32]>::to_vec).collect();
        let last = chunks.len().saturating_sub(1);
        Self::new(
            chunks
                .into_iter()
                .enumerate()
                .map(|(i, chunk)| {
                    let step = MockStep::new(chunk).drafted(width as u32);
                    if i == last {
                        step.finish(finish)
                    } else {
                        step
                    }
                })
                .collect(),
        )
    }
}

#[derive(Debug, Default)]
struct MockInner {
    scripts: VecDeque<MockScript>,
    running: u64,
    waiting: u64,
    drafts: u64,
    draft_tokens: u64,
    accepted_tokens: u64,
    accepted_per_pos: Vec<u64>,
    aborted: Vec<RequestId>,
    submitted: Vec<EngineRequest>,
}

/// An [`Engine`] that replays scripts. Used by every test in this crate.
#[derive(Debug)]
pub struct MockEngine {
    inner: Arc<Mutex<MockInner>>,
    next_id: AtomicU64,
    step_delay: Duration,
    model: ModelInfo,
    /// Every image batch submitted, with the prompt it came with (D3's plane tests).
    images: Mutex<Vec<(Vec<TokenId>, ImageBatch)>>,
}

impl MockEngine {
    pub fn new(model: ModelInfo) -> Self {
        Self {
            inner: Arc::new(Mutex::new(MockInner::default())),
            next_id: AtomicU64::new(1),
            step_delay: Duration::ZERO,
            model,
            images: Mutex::new(Vec::new()),
        }
    }

    /// The image batches submitted so far, with their prompts.
    pub fn submitted_images(&self) -> Vec<(Vec<TokenId>, ImageBatch)> {
        self.images.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Wall-clock gap between steps, so a test can observe update pacing.
    #[must_use]
    pub fn with_step_delay(mut self, delay: Duration) -> Self {
        self.step_delay = delay;
        self
    }

    /// Queue a script. Scripts are consumed in order, **one per choice**: a submit with
    /// `params.n == 3` takes the next three, because the engine owns the fan-out now
    /// (the serving plane submits one request, not `n`).
    pub fn push_script(&self, script: MockScript) {
        self.inner
            .lock()
            .expect("mock lock")
            .scripts
            .push_back(script);
    }

    /// Reports a non-zero queue depth, for a `/metrics` test.
    pub fn set_waiting(&self, waiting: u64) {
        self.inner.lock().expect("mock lock").waiting = waiting;
    }

    /// Every request handed to the engine, in order — lets a test assert that the
    /// sampling parameters a client sent actually reached the engine.
    pub fn submitted(&self) -> Vec<EngineRequest> {
        self.inner.lock().expect("mock lock").submitted.clone()
    }

    pub fn aborted(&self) -> Vec<RequestId> {
        self.inner.lock().expect("mock lock").aborted.clone()
    }
}

impl Engine for MockEngine {
    /// The mock serves images: it records the batch and replays its script as for text.
    fn submit_with_images(
        &self,
        request: EngineRequest,
        images: ImageBatch,
    ) -> Result<Submission, EngineError> {
        if let Ok(mut g) = self.images.lock() {
            g.push((request.prompt_token_ids.clone(), images));
        }
        self.submit(request)
    }

    fn submit(&self, request: EngineRequest) -> Result<Submission, EngineError> {
        if request.prompt_token_ids.len() > self.model.max_model_len {
            return Err(EngineError::PromptTooLong {
                got: request.prompt_token_ids.len(),
                max: self.model.max_model_len,
            });
        }
        let id = RequestId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let n = request.params.n.max(1);
        let scripts = {
            let mut inner = self.inner.lock().expect("mock lock");
            inner.submitted.push(request);
            inner.running += 1;
            let mut scripts = Vec::with_capacity(n);
            for _ in 0..n {
                // Fail loud: a submit with no script is a test that forgot to set one up.
                match inner.scripts.pop_front() {
                    Some(script) => scripts.push(script),
                    None => {
                        inner.running -= 1;
                        return Err(EngineError::Other(
                            "MockEngine: no script queued for this request".to_owned(),
                        ));
                    }
                }
            }
            scripts
        };

        let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(64);
        // One task per choice, all writing into the request's one stream: the request is
        // running until the last of them is done, which is what group finish means.
        let live = Arc::new(AtomicU64::new(n as u64));
        for (choice_index, script) in scripts.into_iter().enumerate() {
            let inner = Arc::clone(&self.inner);
            let delay = self.step_delay;
            let tx = tx.clone();
            let live = Arc::clone(&live);
            tokio::spawn(async move {
                let mut prompt_logprobs = script.prompt_logprobs;
                for (step_index, step) in script.steps.into_iter().enumerate() {
                    if !delay.is_zero() {
                        tokio::time::sleep(delay).await;
                    }
                    {
                        let mut guard = inner.lock().expect("mock lock");
                        if guard.aborted.contains(&id) {
                            if live.fetch_sub(1, Ordering::AcqRel) == 1 {
                                guard.running = guard.running.saturating_sub(1);
                            }
                            return;
                        }
                        guard.drafts += 1;
                        guard.draft_tokens += u64::from(step.draft_tokens);
                        guard.accepted_tokens += step.tokens.len() as u64;
                        // The per-position histogram, in prod's definition: entry `i` counts
                        // drafts whose accepted prefix reached past position `i`. The mock runs
                        // no verify, so the accepted prefix it can honestly report is the commit
                        // minus the one bonus token every verified step commits, capped at the
                        // width it drafted — which is exactly the shape `MockScript::in_steps_of`
                        // builds (width `w`, `w` tokens committed, `w - 1` of them draft).
                        let accepted = (step.tokens.len().saturating_sub(1) as u32)
                            .min(step.draft_tokens) as usize;
                        if guard.accepted_per_pos.len() < step.draft_tokens as usize {
                            guard.accepted_per_pos.resize(step.draft_tokens as usize, 0);
                        }
                        for slot in guard.accepted_per_pos.iter_mut().take(accepted) {
                            *slot += 1;
                        }
                    }
                    let event = StepEvent {
                        request_id: id,
                        choice_index: choice_index as u32,
                        step_index: step_index as u32,
                        new_tokens: step.tokens,
                        finished: step.finished,
                        logprobs: step.logprobs,
                        prompt_logprobs: prompt_logprobs.take(),
                        error: step.error,
                    };
                    if tx.send(event).await.is_err() {
                        break;
                    }
                }
                if live.fetch_sub(1, Ordering::AcqRel) == 1 {
                    inner.lock().expect("mock lock").running -= 1;
                }
            });
        }
        drop(tx);

        Ok(Submission {
            id,
            events: Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)),
        })
    }

    fn abort(&self, id: RequestId) {
        self.inner.lock().expect("mock lock").aborted.push(id);
    }

    fn stats(&self) -> EngineStats {
        let inner = self.inner.lock().expect("mock lock");
        EngineStats {
            num_requests_running: inner.running,
            num_requests_waiting: inner.waiting,
            spec_decode_num_drafts: inner.drafts,
            spec_decode_num_draft_tokens: inner.draft_tokens,
            spec_decode_num_accepted_tokens: inner.accepted_tokens,
            spec_decode_accepted_per_pos: inner.accepted_per_pos.clone(),
            census: Vec::new(),
            memory_plane: String::new(),
        }
    }

    fn model(&self) -> &ModelInfo {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    #[tokio::test]
    async fn mock_replays_one_event_per_step() {
        let engine = MockEngine::new(ModelInfo::default());
        engine.push_script(MockScript::in_steps_of(
            [1, 2, 3, 4, 5, 6, 7],
            3,
            FinishReason::Stop,
        ));
        let submission = engine
            .submit(EngineRequest {
                prompt_token_ids: vec![9, 9],
                params: SamplingParams::default(),
            })
            .expect("submit");
        let events: Vec<StepEvent> = submission.events.collect().await;
        assert_eq!(events.len(), 3, "7 tokens at width 3 is 3 steps");
        assert_eq!(events[0].new_tokens, vec![1, 2, 3]);
        assert_eq!(events[2].new_tokens, vec![7]);
        assert_eq!(events[2].finished, Some(FinishReason::Stop));
        assert!(events[..2].iter().all(|e| e.finished.is_none()));
        let stats = engine.stats();
        assert_eq!(stats.spec_decode_num_drafts, 3);
        assert_eq!(stats.spec_decode_num_accepted_tokens, 7);
    }

    #[tokio::test]
    async fn submit_without_a_script_fails_loud() {
        let engine = MockEngine::new(ModelInfo::default());
        let err = engine
            .submit(EngineRequest {
                prompt_token_ids: vec![1],
                params: SamplingParams::default(),
            })
            .expect_err("no script queued");
        assert!(err.to_string().contains("no script queued"));
        assert_eq!(engine.stats().num_requests_running, 0);
    }
}
