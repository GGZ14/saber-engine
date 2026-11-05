//! Two executors that stand in for the GPU, so the whole scheduler is testable on the host.
//!
//! * [`MockExecutor`] replays a script per sequence — the same shape as `saber-serve`'s
//!   `MockEngine`, one layer lower. Used where a test needs an exact token stream and an exact
//!   acceptance pattern.
//! * [`RandomAcceptanceExecutor`] is deterministic given its seed: tokens are a hash chain over the
//!   prompt and what has been generated, and acceptance is either a fixed count or a seeded
//!   Bernoulli draw. Used where a test needs *plausible* behaviour that is still reproducible —
//!   the verify-width convergence gate and the multi-turn gate.
//!
//! Both model device state the way the device has it: **per slot**, not per sequence. The state
//! lives in a table indexed by the GDN slot-set index and is initialised at
//! [`StepExecutor::admit`]. That is what gives the multi-turn gate teeth — a scheduler that
//! recycled a slot without re-admitting would read the previous session's state, exactly the class
//! of bug the 09-17 lazy-snapshot negative result is (`docs/LEDGER.md`).
//! [`RandomAcceptanceExecutor::leaky`] turns that re-initialisation off and is the negative control
//! the gate ships with (`CLAUDE.md` rule 8).

use std::collections::{BTreeMap, VecDeque};

use saber_plan::{Bucket, Phase};
use saber_serve::engine::{PromptPositionLogprobs, TokenId};

use crate::executor::{
    AdmittedSeq, Forked, PrefillChunk, SeqId, SeqState, SeqStep, StepBatch, StepExecutor,
    StepResult,
};

/// What a step executed, recorded so a test can assert on the shapes that were replayed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedBatch {
    /// Which half of the step.
    pub phase: Phase,
    /// The bucket.
    pub bucket: Bucket,
    /// Engine-wide step index.
    pub step_index: u64,
    /// Live sequences.
    pub sequences: u32,
    /// Padding slots.
    pub padded_slots: u32,
    /// Per-slot verify widths, in slot order.
    pub verify_widths: Vec<u32>,
    /// Per-slot prompt chunks, in slot order. `None` at decode.
    pub chunks: Vec<Option<PrefillChunk>>,
}

pub(crate) fn record(batch: &StepBatch<'_>) -> RecordedBatch {
    RecordedBatch {
        phase: batch.phase,
        bucket: batch.bucket,
        step_index: batch.step_index,
        sequences: batch.slots.len() as u32,
        padded_slots: batch.padded_slots,
        verify_widths: batch.slots.iter().map(|s| s.verify_width).collect(),
        chunks: batch.slots.iter().map(|s| s.chunk).collect(),
    }
}

// ── MockExecutor ────────────────────────────────────────────────────────────────

/// One scripted step of one sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptedStep {
    /// Tokens the step commits.
    pub tokens: Vec<TokenId>,
    /// Draft tokens proposed.
    pub draft_tokens: u32,
    /// Draft tokens accepted (bonus token excluded).
    pub accepted_tokens: u32,
}

impl ScriptedStep {
    /// One token, no draft.
    pub fn single(token: TokenId) -> Self {
        Self {
            tokens: vec![token],
            draft_tokens: 0,
            accepted_tokens: 0,
        }
    }

    /// `accepted` accepted draft tokens of `drafted`, plus the bonus token.
    pub fn speculative(
        tokens: impl IntoIterator<Item = TokenId>,
        drafted: u32,
        accepted: u32,
    ) -> Self {
        Self {
            tokens: tokens.into_iter().collect(),
            draft_tokens: drafted,
            accepted_tokens: accepted,
        }
    }
}

/// One sequence's script.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SeqScript {
    /// The steps, in order. The first is the prefill step.
    pub steps: Vec<ScriptedStep>,
}

impl SeqScript {
    /// A script from steps.
    pub fn new(steps: Vec<ScriptedStep>) -> Self {
        Self { steps }
    }

    /// One token per step — the SPEC=0 shape.
    pub fn per_token(tokens: impl IntoIterator<Item = TokenId>) -> Self {
        Self::new(tokens.into_iter().map(ScriptedStep::single).collect())
    }
}

/// An executor that replays one script per admitted sequence, in admission order.
#[derive(Debug, Default)]
pub struct MockExecutor {
    scripts: VecDeque<SeqScript>,
    live: BTreeMap<SeqId, (SeqScript, usize)>,
    batches: Vec<RecordedBatch>,
    admits: u64,
    releases: u64,
}

impl MockExecutor {
    /// An executor with no scripts yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queues a script. The n-th sequence admitted replays the n-th script.
    pub fn push_script(&mut self, script: SeqScript) {
        self.scripts.push_back(script);
    }

    /// Every batch executed, in order.
    pub fn batches(&self) -> &[RecordedBatch] {
        &self.batches
    }

    /// Sequences admitted and released. They must be equal once nothing is running.
    pub fn admit_release_counts(&self) -> (u64, u64) {
        (self.admits, self.releases)
    }

    /// Sequences whose state the executor still holds.
    pub fn live_sequences(&self) -> usize {
        self.live.len()
    }
}

impl StepExecutor for MockExecutor {
    fn execute(&mut self, batch: &StepBatch<'_>) -> StepResult {
        self.batches.push(record(batch));
        let mut seqs = Vec::with_capacity(batch.slots.len());
        for slot in batch.slots {
            // A prefill chunk that is not the last samples nothing and consumes no script step:
            // the script describes what a sequence *says*, and a prompt chunk says nothing.
            if let Some(chunk) = slot.chunk {
                if !chunk.last {
                    seqs.push(SeqStep {
                        seq_id: slot.seq_id,
                        tokens: Vec::new(),
                        draft_tokens: 0,
                        accepted_tokens: 0,
                        logprobs: None,
                        prompt_logprobs: synthetic_prompt_logprobs(slot),
                    });
                    continue;
                }
            }
            let (script, cursor) = self
                .live
                .get_mut(&slot.seq_id)
                .expect("MockExecutor: step for a sequence that was never admitted");
            let step = script.steps.get(*cursor).unwrap_or_else(|| {
                panic!(
                    "MockExecutor: {} ran past its {}-step script; the scheduler kept it alive \
                     longer than the test expected",
                    slot.seq_id,
                    script.steps.len()
                )
            });
            *cursor += 1;
            assert_eq!(
                step.tokens.len(),
                step.accepted_tokens as usize + 1,
                "MockExecutor: scripted step for {} commits {} tokens for {} accepted; a verified \
                 step commits the prefix plus exactly one",
                slot.seq_id,
                step.tokens.len(),
                step.accepted_tokens
            );
            // The script proposes; the scheduler schedules. A step reports the **scheduled**
            // width, and a narrower schedule verifies a shorter prefix -- which is exactly what a
            // real executor does when the width policy narrows the block under it.
            let drafted = step.draft_tokens.min(slot.verify_width);
            let accepted = step.accepted_tokens.min(drafted);
            seqs.push(SeqStep {
                seq_id: slot.seq_id,
                tokens: step.tokens[..accepted as usize + 1].to_vec(),
                draft_tokens: drafted,
                accepted_tokens: accepted,
                logprobs: None,
                prompt_logprobs: synthetic_prompt_logprobs(slot),
            });
        }
        StepResult { seqs }
    }

    fn admit(&mut self, seq: &AdmittedSeq) {
        let script = self
            .scripts
            .pop_front()
            .expect("MockExecutor: no script queued for this sequence");
        self.live.insert(seq.seq_id, (script, 0));
        self.admits += 1;
    }

    fn release(&mut self, seq: SeqId) {
        self.live.remove(&seq);
        self.releases += 1;
    }

    /// A scripted branch is a scripted sequence: it takes the next script, exactly as a fresh
    /// admission does. That is the whole of what a host-side mock can do about a fork, and it is
    /// what makes `n = 3` comparable to three single submits token for token.
    fn can_fork(&self) -> bool {
        true
    }

    fn max_prompt_logprobs(&self) -> Option<u32> {
        Some(SYNTHETIC_PROMPT_K as u32)
    }

    /// Admit the branch and hand back the FIRST step of its script — which is the step a branch
    /// no longer runs, because it never prefills. The script is unchanged, so a group of `n` is
    /// still `n` single submits token for token.
    fn fork(&mut self, _leader: SeqId, branch: &AdmittedSeq) -> Option<Forked> {
        self.admit(branch);
        let (script, cursor) = self.live.get_mut(&branch.seq_id)?;
        let step = script.steps.get(*cursor)?;
        assert_eq!(
            step.tokens.len(),
            1,
            "MockExecutor: a fork takes the prefill step's one token, and {}'s script opens with \
             {} of them",
            branch.seq_id,
            step.tokens.len()
        );
        *cursor += 1;
        step.tokens.first().copied().map(Forked::token)
    }
}

// ── RandomAcceptanceExecutor ────────────────────────────────────────────────────

/// How many draft tokens the mock accepts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Acceptance {
    /// Always this many (capped by the width actually verified). The verify-width policy's fixed
    /// point is then exact and a convergence test can assert equality.
    Fixed(u32),
    /// Each draft token is accepted with probability `p`, drawn from the seeded stream; the first
    /// rejection ends the draft, as a real verify does.
    Bernoulli {
        /// Per-token acceptance probability.
        p: f64,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SlotState {
    /// The hash chain: the whole of this mock's "device state".
    hash: u64,
    /// Tokens generated into this slot.
    generated: u64,
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A deterministic executor: same seed and same prompt, same tokens, every time.
#[derive(Debug)]
pub struct RandomAcceptanceExecutor {
    seed: u64,
    acceptance: Acceptance,
    vocab_lo: u32,
    vocab_span: u32,
    eos_token: Option<TokenId>,
    eos_after: Option<u64>,
    /// State per **slot set**, which is how the device holds it.
    slots: BTreeMap<u32, SlotState>,
    seq_slot: BTreeMap<SeqId, u32>,
    leaky: bool,
    batches: Vec<RecordedBatch>,
    admits: u64,
    releases: u64,
    /// Every `discard_parked` the scheduler sent, in order (ws-k18).
    discarded_parked: Vec<SeqId>,
}

impl RandomAcceptanceExecutor {
    /// A new executor.
    pub fn new(seed: u64, acceptance: Acceptance) -> Self {
        Self {
            seed,
            acceptance,
            vocab_lo: 1000,
            vocab_span: 4096,
            eos_token: None,
            eos_after: None,
            slots: BTreeMap::new(),
            seq_slot: BTreeMap::new(),
            leaky: false,
            batches: Vec::new(),
            admits: 0,
            releases: 0,
            discarded_parked: Vec::new(),
        }
    }

    /// Emits `token` as the `n`-th generated token, so a test can exercise the stop path.
    #[must_use]
    pub fn with_eos(mut self, token: TokenId, after: u64) -> Self {
        self.eos_token = Some(token);
        self.eos_after = Some(after);
        self
    }

    /// Changes what the drafter lands mid-run, so a test can watch the width policy follow it
    /// down and back up.
    pub fn set_acceptance(&mut self, acceptance: Acceptance) {
        self.acceptance = acceptance;
    }

    /// **The negative control.** A leaky executor does not re-initialise a slot's state at admit,
    /// so a recycled slot carries the previous session's state. The multi-turn gate must fail with
    /// this and pass without it; a gate that passes either way is measuring nothing.
    #[must_use]
    pub fn leaky(mut self) -> Self {
        self.leaky = true;
        self
    }

    /// Every batch executed.
    pub fn batches(&self) -> &[RecordedBatch] {
        &self.batches
    }

    /// Admits and releases seen.
    pub fn admit_release_counts(&self) -> (u64, u64) {
        (self.admits, self.releases)
    }

    /// Sequences whose slot mapping the executor still holds.
    pub fn live_sequences(&self) -> usize {
        self.seq_slot.len()
    }

    /// The parked names the scheduler told this executor to drop, in order.
    pub fn discarded_parked(&self) -> &[SeqId] {
        &self.discarded_parked
    }

    fn slot_of(&self, seq: SeqId) -> u32 {
        *self
            .seq_slot
            .get(&seq)
            .unwrap_or_else(|| panic!("RandomAcceptanceExecutor: {seq} was never admitted"))
    }

    fn next_token(state: &mut SlotState, lo: u32, span: u32) -> TokenId {
        state.hash = splitmix64(state.hash);
        state.generated += 1;
        lo + (state.hash % u64::from(span)) as u32
    }

    fn accepted_for(&self, state: &mut SlotState, width: u32) -> u32 {
        match self.acceptance {
            Acceptance::Fixed(k) => k.min(width),
            Acceptance::Bernoulli { p } => {
                let mut accepted = 0;
                for _ in 0..width {
                    state.hash = splitmix64(state.hash ^ 0xA5A5_A5A5_5A5A_5A5A);
                    let draw = (state.hash >> 11) as f64 / (1u64 << 53) as f64;
                    if draw < p {
                        accepted += 1;
                    } else {
                        break;
                    }
                }
                accepted
            }
        }
    }
}

impl StepExecutor for RandomAcceptanceExecutor {
    fn execute(&mut self, batch: &StepBatch<'_>) -> StepResult {
        self.batches.push(record(batch));
        let (lo, span) = (self.vocab_lo, self.vocab_span);
        let eos = self.eos_token;
        let eos_after = self.eos_after;
        let mut seqs = Vec::with_capacity(batch.slots.len());
        for slot in batch.slots {
            let set = self.slot_of(slot.seq_id);
            if let Some(chunk) = slot.chunk {
                if !chunk.last {
                    // The prompt is already in this slot's state: `admit` seeded the hash chain
                    // with every one of its tokens. A chunk therefore changes nothing here, which
                    // is exactly the property the chunking has to have -- a prompt cut three ways
                    // and a prompt fed whole must leave the sequence in the same place.
                    seqs.push(SeqStep {
                        seq_id: slot.seq_id,
                        tokens: Vec::new(),
                        draft_tokens: 0,
                        accepted_tokens: 0,
                        logprobs: None,
                        prompt_logprobs: None,
                    });
                    continue;
                }
            }
            let mut state = *self.slots.get(&set).expect("admitted slots have state");
            let accepted = self.accepted_for(&mut state, slot.verify_width);
            // Exactly `accepted + 1` tokens, always: the accepted prefix plus the recovery or
            // bonus token. An EOS *replaces* a token rather than shortening the step -- stopping
            // at it is the scheduler's job, and an executor that returned a short block would be
            // breaking the identity the scheduler now counts.
            let mut tokens = Vec::with_capacity(accepted as usize + 1);
            for _ in 0..=accepted {
                let mut t = Self::next_token(&mut state, lo, span);
                if let (Some(eos), Some(after)) = (eos, eos_after) {
                    if state.generated >= after {
                        t = eos;
                    }
                }
                tokens.push(t);
            }
            self.slots.insert(set, state);
            seqs.push(SeqStep {
                seq_id: slot.seq_id,
                tokens,
                draft_tokens: slot.verify_width,
                accepted_tokens: accepted,
                logprobs: None,
                prompt_logprobs: None,
            });
        }
        StepResult { seqs }
    }

    fn admit(&mut self, seq: &AdmittedSeq) {
        let mut hash = self.seed;
        for t in &seq.prompt_tokens {
            hash = splitmix64(hash ^ u64::from(*t));
        }
        self.seq_slot.insert(seq.seq_id, seq.gdn_set);
        let fresh = SlotState { hash, generated: 0 };
        if self.leaky {
            // Negative control: whatever the slot held stays.
            self.slots.entry(seq.gdn_set).or_insert(fresh);
        } else {
            self.slots.insert(seq.gdn_set, fresh);
        }
        self.admits += 1;
    }

    fn release(&mut self, seq: SeqId) {
        // The slot's bytes are *not* cleared: device memory is not zeroed on free, and pretending
        // otherwise would hide the bug the multi-turn gate looks for.
        self.seq_slot.remove(&seq);
        self.releases += 1;
    }

    fn discard_parked(&mut self, seq: SeqId) {
        self.discarded_parked.push(seq);
    }

    fn park_state(&mut self, seq: SeqId) -> Option<SeqState> {
        let set = *self.seq_slot.get(&seq)?;
        let st = self.slots.get(&set)?;
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(&st.hash.to_le_bytes());
        bytes.extend_from_slice(&st.generated.to_le_bytes());
        Some(SeqState(bytes))
    }

    fn can_fork(&self) -> bool {
        true
    }

    /// The mock's own version of the fork's device half: the branch's slot set is a **copy of the
    /// leader's**, and its first token is drawn from that copy.
    ///
    /// This executor is deterministic in its state, so both branches then produce the same token —
    /// which is what a temperature-0 fork does on the card too, and is the control the real one is
    /// gated against. What it cannot model is a second DRAW from one row; that needs a
    /// distribution, and the only executor with one is the real engine.
    fn fork(&mut self, leader: SeqId, branch: &AdmittedSeq) -> Option<Forked> {
        let leader_set = *self.seq_slot.get(&leader)?;
        let mut state = *self.slots.get(&leader_set)?;
        let (lo, span) = (self.vocab_lo, self.vocab_span);
        let token = {
            let mut t = Self::next_token(&mut state, lo, span);
            if let (Some(eos), Some(after)) = (self.eos_token, self.eos_after) {
                if state.generated >= after {
                    t = eos;
                }
            }
            t
        };
        self.seq_slot.insert(branch.seq_id, branch.gdn_set);
        self.slots.insert(branch.gdn_set, state);
        self.admits += 1;
        Some(Forked::token(token))
    }

    fn restore_state(&mut self, seq: &AdmittedSeq, state: &SeqState) -> bool {
        if state.0.len() != 16 {
            return false;
        }
        let hash = u64::from_le_bytes(state.0[0..8].try_into().expect("8 bytes"));
        let generated = u64::from_le_bytes(state.0[8..16].try_into().expect("8 bytes"));
        self.slots
            .insert(seq.gdn_set, SlotState { hash, generated });
        true
    }
}

/// The seam width the host mocks claim, the device's own `DEVICE_ROW_K`.
pub const SYNTHETIC_PROMPT_K: usize = 64;

/// A deterministic prompt-logprob answer for a prefill slot that asked for one, built through the
/// SAME law the real engine uses (`saber_serve::prompt_logprobs::position`) over a synthetic seam.
///
/// The row for target `t` puts `t` inside the top-K at position `t % 4` when that is below 3, and
/// outside it (counted rank 70) otherwise, so both of the law's branches — the dict's duplicate
/// key and the target that is not in the top-k — run on the host. `None` for a slot that is not a
/// prefill chunk or whose request did not ask.
#[must_use]
pub fn synthetic_prompt_logprobs(
    slot: &crate::executor::BatchSlot,
) -> Option<Vec<PromptPositionLogprobs>> {
    let chunk = slot.chunk?;
    let num = slot.sampling.params.prompt_logprobs?.max(0) as usize;
    let mut targets: Vec<TokenId> = slot.input_tokens.iter().skip(1).copied().collect();
    if let Some(next) = chunk.next_prompt_token {
        targets.push(next);
    }
    Some(
        targets
            .into_iter()
            .map(|t| {
                let at = (t % 4) as usize;
                let mut ids: Vec<TokenId> = (0..SYNTHETIC_PROMPT_K as u32)
                    .map(|j| 100_000 + j)
                    .collect();
                let (rank, logit) = if at < 3 && !ids.contains(&t) {
                    ids[at] = t;
                    (at as u32 + 1, 10.0 - at as f32 * 0.25)
                } else {
                    (70, -3.0)
                };
                let row = saber_serve::sampler::DeviceRow {
                    top_values: (0..SYNTHETIC_PROMPT_K)
                        .map(|j| 10.0 - j as f32 * 0.25)
                        .collect(),
                    top_ids: ids,
                    logsumexp: 12.0,
                    logsumexp_at_temperature: Some(12.0),
                    penalised: vec![(t, logit)],
                    vocab_size: 248_320,
                };
                saber_serve::prompt_logprobs::position(&row, t, rank, num).ok()
            })
            .collect(),
    )
}
