//! The one seam between the scheduler and the GPU: [`StepExecutor`].
//!
//! Everything above this trait is pure logic and runs on the host with no HIP, no plan binding and
//! no model — which is what lets the whole of `saber-sched` be tested at P3-minus-the-GPU. Below it
//! is `saber-exec`: bucket capture, graph replay, the drafter and the sampler.
//!
//! The contract, stated so the exec crate can be written against it without reading this crate:
//!
//! * The scheduler hands over a [`StepBatch`] naming **one plan bucket** (batch × spec width × ctx
//!   band) and the live sequences **in slot order**. `slots.len() <= bucket.batch`; the remaining
//!   [`StepBatch::padded_slots`] are the padding that keeps the replayed shape a captured one, and
//!   the executor is expected to run them (a captured graph has a fixed M) and discard the results.
//! * One [`SeqStep`] comes back per live slot, in the same order and with the same [`SeqId`]. Its
//!   `tokens` are the tokens the step **commits**, and their count is not a range but an identity:
//!   **`tokens.len() == accepted_tokens + 1`**, always — the accepted prefix plus the recovery or
//!   bonus token (`docs/abi/DFLASH2-SEMANTICS.md` § 1.4). The scheduler checks it and counts a
//!   violation; an executor whose two numbers disagree desynchronises the metrics from the stream.
//! * **At prefill the identity is different, and it is the other half of the contract.** A slot
//!   with a [`PrefillChunk`] commits **no** token unless the chunk is the last one, in which case
//!   it commits exactly **one** — the prompt's first sampled token. `draft_tokens` and
//!   `accepted_tokens` are zero throughout: there is nothing to speculate on a prompt.
//! * `draft_tokens` is the **scheduled** width (`<= BatchSlot::verify_width`), and
//!   `accepted_tokens` counts **draft** tokens only; the bonus token is not an accepted draft
//!   token, and counting it inflates acceptance per draft by exactly 1.0 — at prod's 2.248, a 45%
//!   overstatement. These are the two `saber:spec_decode_*` numerators `eval_drafter.py` reads.
//! * [`StepExecutor::admit`] and [`StepExecutor::release`] bracket a sequence's device state. A
//!   slot set handed back by `release` is reused by a later sequence, so whatever state lives in it
//!   is initialised at `admit` and never read across the boundary. The multi-turn gate in
//!   `tests/multi_turn.rs` is what fails when it is.
//! * The two parking hooks default to "this executor cannot park", which is the fail-loud default:
//!   the scheduler refuses the park and counts it rather than silently evicting state it cannot
//!   bring back.

use std::fmt;
use std::sync::Arc;

use saber_model::draft::Verified;
use saber_plan::{Bucket, Phase};
use saber_serve::engine::{PromptPositionLogprobs, RequestId, SampledLogprobs, TokenId};
use saber_serve::sampling::SamplingParams;

/// Identifies one sequence inside the scheduler. Unique for the life of the process; a request is
/// one sequence today (`n > 1` is not implemented — see the crate docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeqId(pub u64);

impl fmt::Display for SeqId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "seq{}", self.0)
    }
}

/// A sequence that has just been given its KV blocks and its GDN state slot set.
///
/// Both allocations exist before this is called and stay valid until
/// [`StepExecutor::release`]; the scheduler never reallocates either one mid-sequence, which is
/// what "admission refuses, and then it never OOMs" means in practice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmittedSeq {
    /// The request this sequence belongs to.
    pub request_id: RequestId,
    /// Which of the request's `n` choices this sequence is. A request is a **sequence group**:
    /// one prompt, `n` branches that share its KV pages and decode independently.
    pub choice_index: u32,
    /// The sequence.
    pub seq_id: SeqId,
    /// The prompt, in full. The first step feeds it.
    pub prompt_tokens: Vec<TokenId>,
    /// The paged KV block table, in sequence order.
    pub kv_blocks: Vec<u32>,
    /// How many leading blocks of `kv_blocks` are **already populated**, because they are the
    /// group leader's prompt pages shared with this branch.
    ///
    /// This is where "the prefill runs once" is cashed in: a branch with
    /// `shared_prefix_blocks > 0` does not need those positions computed again, only sampled
    /// from. The host cannot do that for the executor — it is the executor that decides what to
    /// launch — so the scheduler states it here rather than pretending to have saved it.
    pub shared_prefix_blocks: u32,
    /// The **copy-on-write boundary**: the leader's partially-filled block whose contents this
    /// branch must copy into its first private block before writing there. `None` when the prompt
    /// ends on a block boundary, or for a request with `n == 1`.
    pub cow_source: Option<u32>,
    /// Index of the GDN state slot set (a *set*, because speculation needs one slot per draft
    /// token plus one for the committed state).
    pub gdn_set: u32,
    /// How many state slots that set holds: `max spec width + 1`.
    pub gdn_slots: u32,
}

/// One chunk of a prompt, as the prefill step sees it.
///
/// Prefill is **chunked**: a prompt longer than the largest captured prefill bucket is walked in
/// pieces, each one a step of its own, with the KV pages and the GDN state carried from chunk to
/// chunk. The two numbers here are what an executor needs to place a chunk in a sequence it is
/// half-way through:
///
/// * `prompt_tokens` — how many tokens *this* step feeds (also `input_tokens.len()`, named
///   separately because it is the bucket's `M` and the launch geometry reads it as such);
/// * `tokens_done_before` — how many of the prompt were already written into this sequence's KV
///   by earlier chunks, which is where this chunk starts.
///
/// A chunk that is not the last **samples nothing**: the step commits no token, and the scheduler
/// emits no event for it. The last chunk yields the sequence's first token, exactly as an
/// unchunked prefill does — which is why a client sees nothing at all until the prompt is done and
/// why `TTFT` covers every prefill step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillChunk {
    /// Tokens this step feeds.
    pub prompt_tokens: u32,
    /// Tokens of the prompt already resident in this sequence's KV.
    pub tokens_done_before: u32,
    /// Whether this is the last chunk, and therefore the one that samples the first token.
    pub last: bool,
    /// The prompt token right AFTER this chunk, when the chunk is not the last (V7).
    ///
    /// `prompt_logprobs` scores each prompt row against the NEXT prompt token, so a chunk's last
    /// row is scored against a token the chunk does not feed. `None` on the last chunk: its last
    /// row is the one that samples, and the reference server reports no prompt logprob for it.
    pub next_prompt_token: Option<TokenId>,
}

/// The token history a penalised row reads, as the scheduler holds it.
///
/// Built only when a row's penalties actually bind — prod's own defaults carry none — because it
/// is the one part of a sampling request whose size is the sequence's length rather than a
/// constant. `prompt` contributes to the repetition mask only; `output` contributes to all three
/// penalties, and to the frequency penalty **by count**.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PenaltyHistory {
    /// The prompt's tokens.
    pub prompt: Vec<TokenId>,
    /// Everything this sequence has committed, in order.
    pub output: Vec<TokenId>,
}

/// What a row needs to be sampled, as the SCHEDULER knows it.
///
/// The scheduler owns the policy (the request's parameters, the sequence's history, which of the
/// request's `n` branches this row is) and the executor owns the device (the logits, the seam
/// kernel, the dense fallback). This is the seam between those two facts, and it rides on the
/// slot because it is per ROW: one step can hold a greedy row, a sampled row and a penalised row
/// at once, and each is a different launch-time decision.
#[derive(Clone, Debug, PartialEq)]
pub struct SlotSampling {
    /// The request's parameters, unresolved: the executor resolves them against the vocabulary it
    /// knows (`SamplingParams` is what the client sent; `RowParams` is what the reference server's
    /// `__post_init__` and the input batch make of it, and the vocabulary is the engine's fact).
    pub params: SamplingParams,
    /// Which of the request's `n` choices this row is. Part of the noise key, so two branches of
    /// one seeded request draw differently and reproducibly.
    pub choice_index: u32,
    /// This sequence's own step index. Informational since ws-f40 (the sample dump carries it);
    /// it is the pre-F40 noise key and survives as the executor's negative control only, because
    /// a park, a crowd or a different verify width moves step boundaries.
    pub step_index: u64,
    /// The OUTPUT POSITION of the first token this step decides: how many tokens this sequence has
    /// committed so far (0 on the step that samples the prompt's first token). Row `j` of a verify
    /// step decides position `position + j`. With the seed and `choice_index` this is the noise key
    /// (ws-f40): a function of the decision alone, so a seeded request draws the same stream
    /// whatever step, width, batch, park or fork it is decided in.
    pub position: u64,
    /// The tokens the penalties read, when they bind. `None` is the common case and means the
    /// sampler does not touch a logit before the argmax.
    pub penalties: Option<Arc<PenaltyHistory>>,
}

impl SlotSampling {
    /// The greedy row: temperature 0, no history, choice 0. What every step did before the
    /// sampler was wired in, and what every temperature-0 gate still runs.
    #[must_use]
    pub fn greedy() -> Self {
        Self {
            params: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            choice_index: 0,
            step_index: 0,
            position: 0,
            penalties: None,
        }
    }

    /// Whether this row is decided by the argmax alone — the byte-identical path.
    #[must_use]
    pub fn is_greedy(&self) -> bool {
        f64::from(self.params.temperature) < saber_serve::sampler::SAMPLING_EPS
    }
}

impl Default for SlotSampling {
    fn default() -> Self {
        Self::greedy()
    }
}

/// One live sequence's place in a step.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchSlot {
    /// The request.
    pub request_id: RequestId,
    /// The sequence.
    pub seq_id: SeqId,
    /// Slot index inside the captured batch, `0..bucket.batch`.
    pub slot: u32,
    /// The KV block table.
    pub kv_blocks: Vec<u32>,
    /// The GDN state slot set.
    pub gdn_set: u32,
    /// Draft tokens **this sequence** verifies this step, `0..=bucket.spec_width`. The cap comes
    /// from the sequence's acceptance EMA (see [`crate::width`]); the bucket's width is the
    /// captured shape and is an upper bound, not the value to draft at.
    pub verify_width: u32,
    /// Tokens to feed: the whole prompt on the first step, then the previous step's committed
    /// tokens.
    pub input_tokens: Vec<TokenId>,
    /// Tokens already in this sequence's KV before the step.
    pub context_len: u32,
    /// Whether this is the sequence's first (prefill) step.
    pub first_step: bool,
    /// The prompt chunk this step feeds, when the step is a **prefill** step. `None` at decode.
    ///
    /// One field rather than two loose numbers and a flag: a slot either is a prefill chunk or is
    /// not, and a decode slot carrying a `prompt_tokens` of zero is a shape that invites being
    /// read as "a prefill of nothing".
    pub chunk: Option<PrefillChunk>,
    /// How this slot's next token is to be chosen. [`SlotSampling::greedy`] is the argmax, which
    /// is what every step committed before the sampler was wired in.
    pub sampling: SlotSampling,
}

/// FN75: one step's clock and shape, as the scheduler saw it -- handed to
/// [`StepExecutor::step_done`] after the commit (the served-step ledger). Every instant is the
/// step thread's own `Instant::now()`; nothing here is used for a decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepTiming {
    /// The previous step's end (`None` on the first step).
    pub prev_end: Option<std::time::Instant>,
    /// Whether any request was running or queued when the previous step ended (a gap with
    /// nothing to do is the client's, not the server's).
    pub work_at_prev_end: bool,
    /// Idle polls the step loop made since the previous step's end.
    pub idle_polls: u64,
    /// The step's entry (the stepper's lock held, before admission).
    pub start: std::time::Instant,
    /// The executor call's start (admission + batch built).
    pub exec_start: std::time::Instant,
    /// The executor call's return.
    pub exec_end: std::time::Instant,
    /// The commit's end.
    pub end: std::time::Instant,
    /// The step's phase (a mixed / packed step is `Decode` with `chunks > 0`).
    pub phase: Phase,
    /// Decode rows (live sequences) of a decode step; 0 on a prefill step.
    pub decode_rows: u32,
    /// The bucket's verify width.
    pub width: u32,
    /// Prompt rows the step's chunks carry (all of them).
    pub chunk_rows: u32,
    /// Prompt chunks the step carries.
    pub chunks: u32,
    /// Decoders that waited out a prefill step run alone.
    pub waiting_decoders: u32,
    /// Sequences running at the step (admission included).
    pub running: u32,
    /// Tokens the step committed.
    pub committed_tokens: u32,
    /// Sequences the step finished.
    pub finished: u32,
    /// FN157 (the admit-slack census): requests still queued after this step's admission.
    pub waiting: u32,
    /// FN157: requests this step's admission took (fresh requests; a restored park is not one).
    pub admitted: u32,
    /// FN157: each admitted request's arrival -> this step's entry, microseconds (the first 8;
    /// 0 past `admitted`, and for a request that arrived after the entry).
    pub admit_lag_us: [u32; 8],
}

/// One engine step, as the executor sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct StepBatch<'a> {
    /// Which half of the step this is. A step is **one phase**: prefill chunks and decode steps
    /// run different ops and different GDN state geometry, so a batch that mixed them would be a
    /// batch with no captured shape. The scheduler refuses to build one
    /// ([`crate::buckets::NoBucket::MixedPhase`]).
    pub phase: Phase,
    /// The plan bucket being replayed. Always a captured one.
    pub bucket: Bucket,
    /// Engine-wide step counter, monotone from 0. Not the per-request `step_index`.
    pub step_index: u64,
    /// Live sequences, in slot order.
    pub slots: &'a [BatchSlot],
    /// `bucket.batch - slots.len()`: slots the captured shape runs and the scheduler discards.
    pub padded_slots: u32,
}

impl StepBatch<'_> {
    /// Rows the GEMMs run at in this bucket — the captured M, padding included.
    pub fn m(&self) -> u32 {
        self.bucket.m()
    }

    /// The widest verify width any live slot asked for.
    pub fn max_verify_width(&self) -> u32 {
        self.slots.iter().map(|s| s.verify_width).max().unwrap_or(0)
    }
}

/// What [`StepExecutor::fork`] hands back for a branch (ws-f39).
#[derive(Clone, Debug, PartialEq)]
pub struct Forked {
    /// The branch's first token: its own draw from the leader's row.
    pub token: TokenId,
    /// What that draw REPORTS, when the request asked for `logprobs` — the same pipeline, the same
    /// row and the same schema as the leader's first token (ws-f39). `None` when nobody asked,
    /// or when an executor has no distribution to report (a mock).
    pub logprobs: Option<SampledLogprobs>,
}

impl Forked {
    /// A first token with nothing to report.
    pub fn token(token: TokenId) -> Self {
        Self {
            token,
            logprobs: None,
        }
    }
}

/// What one sequence produced in one step.
#[derive(Clone, Debug, PartialEq)]
pub struct SeqStep {
    /// Must equal the slot's sequence id; a mismatch is a protocol error and the scheduler counts
    /// it and aborts the request rather than committing tokens to the wrong stream.
    pub seq_id: SeqId,
    /// Tokens committed by this step: the accepted prefix plus **exactly one** more — the
    /// recovery token at the rejected position, or the bonus token when every draft token
    /// survived. `tokens.len() == accepted_tokens + 1`, always, and the scheduler counts a
    /// violation rather than trusting either number
    /// (`docs/abi/DFLASH2-SEMANTICS.md` § 1.4: `_insert_resampled_kernel` increments
    /// `num_sampled` unconditionally).
    pub tokens: Vec<TokenId>,
    /// Draft tokens **scheduled** — the width after the verify-width cap, which is what
    /// `saber:spec_decode_num_draft_tokens_total` sums (`len(scheduled_spec_token_ids)`), *not*
    /// the drafter's own block width. It must be `<= BatchSlot::verify_width`: the scheduler
    /// asked for that many, and a drafter that produced fewer reports fewer.
    ///
    /// Reporting the drafter's full block width instead would silently disable the climb-out in
    /// the acceptance EMA — `accepted >= drafted` would then need a full *block* acceptance
    /// rather than a full *scheduled* one (mismatch row 10).
    pub draft_tokens: u32,
    /// Draft tokens accepted, **excluding** the bonus token
    /// (`saber:spec_decode_num_accepted_tokens_total`). Acceptance is a prefix: it stops at the
    /// first rejection and never resumes.
    pub accepted_tokens: u32,
    /// One entry per token in `tokens`, when the request asked for logprobs. `None` otherwise —
    /// and `None` rather than an empty vector, because "the client did not ask" and "the sampler
    /// had nothing to report" are different facts and only one of them is normal.
    pub logprobs: Option<Vec<SampledLogprobs>>,
    /// `prompt_logprobs` for the prompt positions THIS prefill chunk scored (V7): one entry per
    /// row the chunk scored, which is every row of a chunk that is not the last and every row but
    /// the last of the one that is. `None` when the request did not ask — and a request that did
    /// ask and gets `None` back is refused loudly by the scheduler, never answered with a `null`.
    pub prompt_logprobs: Option<Vec<PromptPositionLogprobs>>,
}

impl SeqStep {
    /// The SPEC=0 shape: one token, no draft.
    pub fn single(seq_id: SeqId, token: TokenId) -> Self {
        Self {
            seq_id,
            tokens: vec![token],
            draft_tokens: 0,
            accepted_tokens: 0,
            logprobs: None,
            prompt_logprobs: None,
        }
    }

    /// The same step, carrying the distribution the sampler reported for each of its tokens.
    #[must_use]
    pub fn with_logprobs(mut self, logprobs: Option<Vec<SampledLogprobs>>) -> Self {
        self.logprobs = logprobs;
        self
    }

    /// A speculative step: `accepted` of `drafted` scheduled draft tokens survived, plus the one
    /// extra token every verified step commits.
    pub fn speculative(
        seq_id: SeqId,
        tokens: impl IntoIterator<Item = TokenId>,
        drafted: u32,
        accepted: u32,
    ) -> Self {
        Self {
            seq_id,
            tokens: tokens.into_iter().collect(),
            draft_tokens: drafted,
            accepted_tokens: accepted,
            logprobs: None,
            prompt_logprobs: None,
        }
    }

    /// The same step as `saber-model`'s verifier reports it.
    ///
    /// The spec-decode counters — including the per-position acceptance histogram
    /// `eval_drafter.py` reads — are folded by [`saber_model::draft::SpecDecodeCounters`], which
    /// is gated against prod's own kernel. This is the view it takes, so the scheduler never
    /// re-derives "drafts whose accepted prefix reached past position i".
    pub fn as_verified(&self) -> Verified {
        Verified {
            accepted_len: self.accepted_tokens,
            tokens: self.tokens.iter().map(|t| *t as i32).collect(),
            rejected_at: (self.accepted_tokens < self.draft_tokens).then_some(self.accepted_tokens),
        }
    }
}

/// The result of one step: one entry per live slot, in slot order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepResult {
    /// Per-sequence results, in the batch's slot order.
    pub seqs: Vec<SeqStep>,
}

impl StepResult {
    /// A result built from an iterator of per-sequence steps.
    pub fn new(seqs: impl IntoIterator<Item = SeqStep>) -> Self {
        Self {
            seqs: seqs.into_iter().collect(),
        }
    }
}

/// An opaque image of one sequence's device state — KV **and** GDN state as one unit.
///
/// `ARCHITECTURE.md` § Memory plane: *"Evict an idle session's KV and its 72 MiB/rank GDN state as
/// one unit — they are useless apart."* The scheduler never looks inside; it only moves it between
/// the executor and a [`crate::park::Parkable`] tier and checks that what comes back is what went
/// in.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SeqState(pub Vec<u8>);

impl SeqState {
    /// Bytes in the image — what the tier bills to the ledger.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the image is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// What an executor did when asked to start a transfer off the step loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferStart {
    /// It has no asynchronous path; use the blocking [`StepExecutor::park_state`] /
    /// [`StepExecutor::restore_state`] instead. The honest default.
    Blocking,
    /// In flight. The scheduler stops stepping the sequence and polls.
    Started,
    /// This executor cannot move session state at all — a loud refusal, never a silent eviction.
    CannotPark,
    /// It already has a transfer in flight and holds one staging buffer, so this one has to wait.
    /// **Not a failure**: the caller leaves the group where it is and asks again next step.
    Busy,
    /// It tried and failed before anything was in flight.
    Failed(String),
}

/// Where an in-flight park has got to.
#[derive(Debug)]
pub enum ParkPoll {
    /// Still copying. The sequence keeps its pages and its slot.
    Pending,
    /// The bytes are off the device. Now — and only now — the allocations may be freed.
    Done(SeqState),
    /// It failed. The sequence is put back in the running set untouched.
    Failed(String),
}

/// Where an in-flight restore has got to.
#[derive(Debug)]
pub enum RestorePoll {
    /// Still copying. The sequence is not stepped.
    Pending,
    /// Every byte is back in its pages and its slot; the sequence is schedulable.
    Done,
    /// It failed, and the image comes **back** so the session can stay parked.
    ///
    /// The executor owns the bytes while a restore is in flight — it is handed them by value, which
    /// is what keeps a 403 MiB image from being copied across the engine channel — so a failure has
    /// to return them. The ticket was spent when the restore was asked for, and these are the only
    /// copy left; `None` means even that is gone and the session is lost, which is logged as such.
    Failed(String, Option<SeqState>),
}

/// What an executor did when asked to start a **restore**, which it takes ownership of.
#[derive(Debug)]
pub enum RestoreStart {
    /// In flight. The scheduler stops stepping the sequence and polls.
    Started,
    /// No asynchronous path; the caller uses the blocking `restore_state`. The image comes back.
    Blocking(SeqState),
    /// A transfer is already in flight. **Not a failure**: retry next step, image returned.
    Busy(SeqState),
    /// Refused outright, image returned.
    Refused(String, Option<SeqState>),
}

/// The GPU side of one engine step.
///
/// `saber-exec` implements this at P3. Everything the scheduler needs from the device is here, and
/// nothing else: no allocation, no residency, no lifetime.
pub trait StepExecutor: Send + fmt::Debug + 'static {
    /// Run one step of `batch` and return one [`SeqStep`] per live slot, in slot order.
    fn execute(&mut self, batch: &StepBatch<'_>) -> StepResult;

    /// **A mixed step (C4)**: one decode step that also carries ONE prefill chunk of another
    /// sequence (`crate::MixedPrefill`). `decode` is the step's decode half, exactly the batch the
    /// unmixed schedule would have run; `chunk` is a one-slot [`Phase::Prefill`] batch, exactly the
    /// chunk step it would have run first. Returns `(decode result, chunk result)`, each under its
    /// own batch's contract.
    ///
    /// **The default is the bit-exact form, and it is also the law any other form must equal**:
    /// the chunk's forward and then the decode half's, back to back in one call — the same
    /// kernels over the same rows as the two unmixed steps, so the decode rows commit the ids
    /// the same slots commit unmixed and the chunk writes the KV pages and GDN state a standalone
    /// chunk writes. The two halves touch DISJOINT sequences (a prefilling sequence is never a
    /// row of a decode half), so neither can see the other's state. What the step saves is the
    /// step boundary between them, not the chunk's forward; a form that shares the GEMMs (the
    /// decode rows riding in the chunk bucket's padding, `docs/EXEC.md` § the mixed step) changes
    /// the decode rows' kernel and is gated by end-task quality instead.
    ///
    /// **The chunk runs FIRST.** Its rows are the fused form's rows `0..chunk` — where the prefill
    /// kernels' row-0-based descriptors already point — and the decode rows follow it; running
    /// the sequential form in the same order keeps one law for both.
    fn execute_mixed(
        &mut self,
        decode: &StepBatch<'_>,
        chunk: &StepBatch<'_>,
    ) -> (StepResult, StepResult) {
        let c = self.execute(chunk);
        let d = self.execute(decode);
        (d, c)
    }

    /// **C10: a PACKED mixed step** — one decode step carrying SEVERAL queued prompts' chunks
    /// (`crate::MixedPrefill::packs`, the packing law `crate::mixed_law::pack`). `chunks` are
    /// one-slot [`Phase::Prefill`] batches in queue order, each exactly the chunk step the
    /// unpacked schedule would have run. Returns the decode result and one result per chunk, in
    /// order.
    ///
    /// The default is the serial form (every chunk, in order, then the decode half) — the
    /// bit-exact law a fused form must equal row for row on its chunk rows.
    fn execute_packed(
        &mut self,
        decode: &StepBatch<'_>,
        chunks: &[StepBatch<'_>],
    ) -> (StepResult, Vec<StepResult>) {
        let cs: Vec<StepResult> = chunks.iter().map(|c| self.execute(c)).collect();
        let d = self.execute(decode);
        (d, cs)
    }

    /// A sequence was admitted: its KV blocks and GDN slot set now exist and its state must be
    /// initialised from the prompt. Called once, before the sequence's first step.
    fn admit(&mut self, _seq: &AdmittedSeq) {}

    /// D3: whether this executor has a vision tower resident. `false` (the default) makes the
    /// scheduler refuse an image request by name at submit.
    fn serves_images(&self) -> bool {
        false
    }

    /// D3: run the vision tower over one request's images, before the request is admitted. The
    /// rows stay in the tower's arena (D4 scatters them into the prompt's embedding rows). An
    /// `Err` ends the request with that reason. The default is the text-only refusal.
    fn encode_images(
        &mut self,
        _request: saber_serve::engine::RequestId,
        _images: &saber_serve::engine::ImageBatch,
    ) -> Result<(), String> {
        Err(saber_serve::engine::TEXT_ONLY_REFUSAL.to_owned())
    }

    /// A sequence's allocations are being freed (finish, abort or park). Its slots are handed to
    /// the next sequence, so nothing may be read out of them after this returns.
    fn release(&mut self, _seq: SeqId) {}

    /// The executor's live census (ws-v8): the counters its shutdown lines print, published into
    /// a lock-free mirror the scrape reads (`saber_serve::census`). Asked once, at construction;
    /// `None` (the default) is an executor with no census of its own.
    fn census_mirror(&self) -> Option<std::sync::Arc<saber_serve::census::CensusMirror>> {
        None
    }

    /// The executor's memory-plane exposition (FN14): a shared text the executor refreshes (the
    /// Flash-Next tier's `saber_xtier_*` / `saber_ple_*` families) and `/metrics` appends verbatim.
    /// Asked once, at construction; `None` (the default, every 27B executor) appends nothing.
    fn memory_plane(&self) -> Option<std::sync::Arc<std::sync::Mutex<String>>> {
        None
    }

    /// FN75: one step's scheduler-side clock and shape (the served-step ledger), called with the
    /// executor lock held at the end of every step's commit. The default records nothing.
    fn step_done(&mut self, _t: &StepTiming) {}

    /// Publish this executor's counters into [`StepExecutor::census_mirror`]. The scheduler calls
    /// it with the executor lock held — at the end of every step's commit, on an idle poll, and
    /// after an abort retired a sequence — and it must be a no-op (one relaxed load) until a
    /// scrape has armed the mirror.
    fn publish_census(&mut self) {}

    /// A sequence's allocations are being freed because it was PARKED: its image is on the tier
    /// and will be restored into another sequence. Everything [`StepExecutor::release`] frees is
    /// freed; what an executor keeps FOR THE IMAGE must survive it. At TP=2 that is the follower's
    /// half of the image, which a plain release tells it to drop — so a serve-plane park at two
    /// ranks restored into "this image has no half on the other rank" (ws-k17). The default is a
    /// plain release, which is right for every executor that keeps nothing outside the image.
    fn release_parked(&mut self, seq: SeqId) {
        self.release(seq);
    }

    /// The image parked under `seq` was DROPPED without a restore — the request was aborted while
    /// parked, or the scheduler gave up on its restore — so whatever this executor keeps FOR that
    /// image goes now (ws-k18). The pair of [`StepExecutor::release_parked`]: that call kept it for
    /// a restore, and this is the call that says no restore is coming. At TP=2 it is the
    /// follower's half, which before this stayed held until the follower exited. The default
    /// holds nothing outside the image, so it has nothing to drop.
    fn discard_parked(&mut self, _seq: SeqId) {}

    /// Copy a sequence's KV + GDN state out, for parking. `None` means this executor cannot park,
    /// which the scheduler treats as a refusal (counted, never a silent eviction).
    ///
    /// **Blocking.** An executor whose park is a device transfer should implement
    /// [`StepExecutor::park_begin`] instead and leave this at its default: a park of a full context
    /// is hundreds of milliseconds (workstream K8 measured 225 ms, K9 101 ms after coalescing) and
    /// the step loop cannot be inside it.
    fn park_state(&mut self, _seq: SeqId) -> Option<SeqState> {
        None
    }

    /// Put a parked image back into a freshly admitted sequence's slots. `false` means the restore
    /// failed and the sequence stays parked. **Blocking**, as above.
    fn restore_state(&mut self, _seq: &AdmittedSeq, _state: &SeqState) -> bool {
        false
    }

    /// Whether this executor moves session state **off the step loop**.
    ///
    /// Decided once, per executor, rather than per sequence: a sequence group parks whole, and a
    /// group with some branches copying and some copied synchronously would be a group whose
    /// allocations are freed at two different times. `false` is the honest default and keeps every
    /// mock on the blocking seam.
    fn parks_async(&self) -> bool {
        false
    }

    /// Whether a session transfer is already in flight.
    ///
    /// Asked **before** anything is taken off the parking tier: a restore that discovered it was
    /// busy after consuming its tickets could not be retried, because the images would already be
    /// off the tier and the tickets spent.
    fn transfer_in_flight(&self) -> bool {
        false
    }

    /// Start copying a sequence's state off the device and **return without waiting**.
    ///
    /// The default is [`TransferStart::Blocking`], which tells the scheduler to use
    /// [`StepExecutor::park_state`] as before — every mock and every test executor keeps working
    /// unchanged. An executor that answers [`TransferStart::Started`] promises two things: the
    /// sequence's pages and slot are **still being read**, so they must not be freed until
    /// [`StepExecutor::park_poll`] says the copy is done; and the poll is cheap and never blocks.
    fn park_begin(&mut self, _seq: SeqId) -> TransferStart {
        TransferStart::Blocking
    }

    /// Has the park started for `seq` finished? Must not block.
    fn park_poll(&mut self, _seq: SeqId) -> ParkPoll {
        ParkPoll::Failed("this executor has no asynchronous park".to_owned())
    }

    /// Start putting an image back into a freshly admitted sequence and **return without waiting**.
    ///
    /// The sequence's host-side bookkeeping is already the restored one when this is called; what
    /// the scheduler withholds until [`StepExecutor::restore_poll`] is done is *stepping* it.
    ///
    /// Takes the image **by value**, and every answer that is not `Started` gives it back. That is
    /// what lets an engine move the bytes to its own thread instead of copying them.
    fn restore_begin(&mut self, _seq: &AdmittedSeq, state: SeqState) -> RestoreStart {
        RestoreStart::Blocking(state)
    }

    /// Has the restore started for `seq` finished? Must not block.
    fn restore_poll(&mut self, _seq: SeqId) -> RestorePoll {
        RestorePoll::Failed("this executor has no asynchronous restore".to_owned(), None)
    }

    /// Whether this executor can serve a **sequence group**: `n` branches over one prompt, where
    /// branch `i > 0` is admitted with the leader's KV pages already populated
    /// ([`AdmittedSeq::shared_prefix_blocks`]) and owes a copy of the partial one
    /// ([`AdmittedSeq::cow_source`]).
    ///
    /// The default is `false`, which is the fail-loud one: a scheduler asked for `n > 1` on an
    /// executor that cannot branch refuses the request rather than serving one choice and calling
    /// it three.
    fn can_fork(&self) -> bool {
        false
    }

    /// The widest `prompt_logprobs` this executor can answer, or `None` when it produces none.
    ///
    /// Asked once, at construction. `None` is the fail-loud default: a request that asks for
    /// prompt logprobs from an executor that cannot produce them is a 400 at submit, never a 200
    /// with `"prompt_logprobs": null` (Q2 found the real engine doing exactly that; `ppl.py` then
    /// dies on a `TypeError` two layers away from the cause).
    fn max_prompt_logprobs(&self) -> Option<u32> {
        None
    }

    /// Create branch `branch` from `leader`, **after the leader's last prefill chunk has sampled**,
    /// and return the branch's own first token.
    ///
    /// This is the device half of a sequence group, and it is called INSTEAD of
    /// [`StepExecutor::admit`] for a branch: the scheduler defers that admission until the moment
    /// the leader samples, because that is the only moment at which all three of the branch's
    /// starting facts exist at once.
    ///
    /// **The fork law, in three parts** (`docs/LEDGER.md`, V3's design and V4's build):
    ///
    /// 1. **The prompt's whole KV pages are the leader's, shared read-only for ever.** The page
    ///    list is per sequence, the leading [`AdmittedSeq::shared_prefix_blocks`] of it are the
    ///    leader's own pages, and every branch appends only into pages it owns — so there is no
    ///    copy-on-write after this call and nothing to track. The **one** partial page
    ///    ([`AdmittedSeq::cow_source`]) is copied into the branch's first private block, because
    ///    it is the one page both would write into.
    /// 2. **The GDN state slot set is copied**, device to device. Recurrent state is not
    ///    addressable per token, so it cannot be shared the way a page can; the leader's set is
    ///    the prompt's final state and the branch continues the delta rule from a copy of it.
    /// 3. **The branch's first token is a second draw from the LEADER'S OWN prefill row**, with
    ///    the branch's own noise key `(seed, position 0, choice)` (ws-f40). That is what makes `n`
    ///    branches of one prompt cost one prefill, and it is why the fork happens at the sampled
    ///    step rather than at admission: the row exists only there.
    ///
    /// `None` is a refusal — the branch is then aborted, loudly, rather than started from a state
    /// nothing copied. An executor that answers `Some` has admitted the branch and owes a
    /// [`StepExecutor::release`] for it like any other sequence.
    fn fork(&mut self, _leader: SeqId, _branch: &AdmittedSeq) -> Option<Forked> {
        None
    }

    /// FN52: whether this executor snapshots and restores prefix state
    /// ([`crate::prefix_cache`]). `false` (the default) makes `--enable-prefix-caching on` a
    /// refusal at construction rather than a cache that silently never hits.
    fn prefix_cache_supported(&self) -> bool {
        false
    }

    /// FN52: `seq`'s prefill has just reached `tokens` (a whole-block boundary): snapshot the
    /// state a resumed prefill needs there — everything but the KV pages, which the scheduler
    /// retains in place — under `entry`. Returns the host bytes the snapshot holds (per rank).
    fn prefix_save(&mut self, _seq: SeqId, _entry: u32, _tokens: u32) -> Result<u64, String> {
        Err("this executor has no prefix cache".to_owned())
    }

    /// FN52: `seq` was just admitted ADOPTING `entry`'s pages as its first blocks: restore the
    /// entry's snapshot into its slots, so its prefill resumes at `tokens`.
    fn prefix_load(&mut self, _seq: &AdmittedSeq, _entry: u32, _tokens: u32) -> Result<(), String> {
        Err("this executor has no prefix cache".to_owned())
    }

    /// FN52: `entry` was evicted; its snapshot goes.
    fn prefix_drop(&mut self, _entry: u32) {}

    /// FN52, the verify arm: `seq` prefilled the same `tokens` FRESH (with a cut there); compare
    /// its state and its first `tokens / block` pages with `entry`'s snapshot and `entry_blocks`.
    /// `Ok(true)` = byte-identical.
    fn prefix_verify(
        &mut self,
        _seq: SeqId,
        _entry: u32,
        _tokens: u32,
        _entry_blocks: &[u32],
    ) -> Result<bool, String> {
        Err("this executor has no prefix cache".to_owned())
    }
}
