//! THE ANCHOR IS THE LAW: a block made from a token the step did not commit is not a proposal.
//!
//! **Since ws-f37 the serving order is prod's** — the step stashes its proposal and the executor
//! runs it after the host sampler, from the COMMITTED token (the second half of this file) — and
//! the in-step order below is the pairing control (`--propose-in-step`). Its law, which follows,
//! is unchanged: under that order a block anchored on a replaced argmax is dropped and counted.
//!
//! The device proposes INSIDE the decode step, out of the ARGMAX of the row the step is about to
//! commit (`saber_engine::device`: `let anchor = *verdict.committed.last()`), because a DFlash2
//! proposal reads the target's hidden states for that step's rows and they live in the target's
//! arena only until the next step overwrites them. The host sampler then replaces that same token
//! for a row with a temperature — so above temperature 0 the block is, on a measurable fraction of
//! steps, the continuation of a token the sequence never had.
//!
//! Prod has no such step: `execute_model` samples and THEN calls `speculator.propose`, so prod's
//! anchor is always the committed token. Measured against prod on the 160 held-out prompts
//! (ws-f35, off ws-v6's and ws-s11's own captures): at temperature 0.7 Saber's FIRST drafted token
//! survives 0.683 of the time against prod's 0.800, while positions 1–6 sit at 0.96–0.99 of prod
//! and are unchanged from the temperature-0 arm. The deficit is at position 0 and nowhere else,
//! which is the signature of an anchor that moved — such a block is drafting a different text and
//! is filtered out at the first token.
//!
//! What is gated here is the LAW and its CONTROL, on the host, with no card:
//!
//!   * temperature 0 offers every block and marks none stale (the argmax IS the committed token);
//!   * a temperature whose draw moves the anchor drops the block and counts it;
//!   * `keep_stale_anchor_blocks(true)` — the engine as it was before ws-f35 — offers it, and then
//!     tallies what it earned, which is the arm that says what the drop costs.

use std::sync::Arc;

use saber_engine::{
    Decider, DecodeSlot, EngineCounters, ForwardError, ForwardFacts, ForwardRunner, NullForward,
    Proposal, RealExecutor, RowDecider, RowRequest, StepOutcome, StepRows,
};
use saber_plan::{Bucket, Phase};
use saber_sched::{
    AdmittedSeq, BatchSlot, PrefillChunk, SeqId, SlotSampling, StepBatch, StepExecutor,
};
use saber_serve::engine::{RequestId, TokenId};
use saber_serve::sampler::DeviceRow;
use saber_serve::sampling::SamplingParams;

const PREFILL: u32 = 2048;
const BATCHES: &[u32] = &[1, 2, 4];
const PROMPT: usize = 12;
/// The block the stub proposes. Its ids are irrelevant — what is gated is whether it is OFFERED.
const BLOCK: &[TokenId] = &[7001, 7002, 7003];

/// [`NullForward`] with a drafter tier bolted on: every step comes back with one block per slot,
/// which is the shape a pass with `--draft-block` returns and which `NullForward` alone never has.
#[derive(Debug)]
struct Proposing(NullForward);

impl ForwardRunner for Proposing {
    fn facts(&self) -> &ForwardFacts {
        self.0.facts()
    }
    fn admit(
        &mut self,
        seq: SeqId,
        prompt_len: u32,
        kv_blocks: &[u32],
    ) -> Result<(), ForwardError> {
        self.0.admit(seq, prompt_len, kv_blocks)
    }
    fn release(&mut self, seq: SeqId) {
        self.0.release(seq);
    }
    fn prefill(
        &mut self,
        seq: SeqId,
        chunk: &[TokenId],
        done: u32,
        last: bool,
        sampling: &RowRequest,
    ) -> Result<StepOutcome, ForwardError> {
        let mut out = self.0.prefill(seq, chunk, done, last, sampling)?;
        if last {
            out.drafts = vec![BLOCK.to_vec()];
        }
        Ok(out)
    }
    fn decode(
        &mut self,
        slots: &[DecodeSlot],
        batch: u32,
        width: u32,
    ) -> Result<StepOutcome, ForwardError> {
        let mut out = self.0.decode(slots, batch, width)?;
        out.drafts = vec![BLOCK.to_vec(); slots.len()];
        Ok(out)
    }
    fn declared_launches(&self, rows: StepRows) -> Option<u32> {
        self.0.declared_launches(rows)
    }
    fn dense_row(&mut self, row: u32) -> Result<Vec<f32>, ForwardError> {
        self.0.dense_row(row)
    }
}

fn prompt() -> Vec<TokenId> {
    (0..PROMPT as TokenId).map(|i| 1000 + i).collect()
}

fn executor() -> RealExecutor<Proposing> {
    let mut exec = RealExecutor::new(Proposing(NullForward::new(&[PREFILL], BATCHES, 8)));
    exec.admit(&AdmittedSeq {
        request_id: RequestId(1),
        choice_index: 0,
        seq_id: SeqId(1),
        prompt_tokens: prompt(),
        kv_blocks: (0..16).collect(),
        shared_prefix_blocks: 0,
        cow_source: None,
        gdn_set: 1,
        gdn_slots: 2,
    });
    exec
}

fn sampling(params: SamplingParams, step: u64) -> SlotSampling {
    SlotSampling {
        params,
        choice_index: 0,
        step_index: step,
        position: step,
        penalties: None,
    }
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    }
}

/// A temperature wide enough that the draw is routinely not the argmax, seeded so it is the SAME
/// draw every run. The stub's row is a peak, thirty-one runners-up and a flat floor, so a top-k of
/// 20 at temperature 2 spreads the mass over the runners-up.
fn hot(seed: u64) -> SamplingParams {
    SamplingParams {
        temperature: 2.0,
        top_p: 1.0,
        top_k: 20,
        seed: Some(seed),
        ..SamplingParams::default()
    }
}

fn prefill_slot(s: SlotSampling) -> BatchSlot {
    BatchSlot {
        request_id: RequestId(1),
        seq_id: SeqId(1),
        slot: 0,
        kv_blocks: (0..16).collect(),
        gdn_set: 1,
        verify_width: 0,
        input_tokens: prompt(),
        context_len: 0,
        first_step: true,
        chunk: Some(PrefillChunk {
            prompt_tokens: PROMPT as u32,
            tokens_done_before: 0,
            last: true,
            next_prompt_token: None,
        }),
        sampling: s,
    }
}

fn decode_slot(input: &[TokenId], context: u32, s: SlotSampling) -> BatchSlot {
    BatchSlot {
        request_id: RequestId(1),
        seq_id: SeqId(1),
        slot: 0,
        kv_blocks: (0..16).collect(),
        gdn_set: 1,
        verify_width: 0,
        input_tokens: input.to_vec(),
        context_len: context,
        first_step: false,
        chunk: None,
        sampling: s,
    }
}

/// One prefill and `steps` decode steps at the same parameters. Returns the ids and the counters.
fn generate(
    params: &SamplingParams,
    steps: usize,
    keep_stale: bool,
) -> (Vec<TokenId>, EngineCounters) {
    let mut exec = executor();
    exec.keep_stale_anchor_blocks(keep_stale);
    let mut ids: Vec<TokenId> = Vec::new();
    let slots = vec![prefill_slot(sampling(params.clone(), 0))];
    let out = exec.execute(&StepBatch {
        phase: Phase::Prefill,
        bucket: Bucket::prefill(PREFILL, 8192, true),
        step_index: 0,
        slots: &slots,
        padded_slots: 0,
    });
    ids.extend(out.seqs[0].tokens.iter().copied());
    let mut context = PROMPT as u32;
    for step in 1..=steps as u64 {
        let last = vec![*ids.last().expect("a token")];
        let slots = vec![decode_slot(&last, context, sampling(params.clone(), step))];
        let out = exec.execute(&StepBatch {
            phase: Phase::Decode,
            bucket: Bucket::decode(1, 1, 8192, true),
            step_index: step,
            slots: &slots,
            padded_slots: 0,
        });
        ids.extend(out.seqs[0].tokens.iter().copied());
        context += out.seqs[0].tokens.len() as u32;
    }
    (ids, exec.counters())
}

/// **THE LAW AT TEMPERATURE 0, which is where every gate in this tree lives.** A greedy row commits
/// the argmax the drafter was anchored on, so nothing is ever stale and nothing is ever dropped —
/// the law is inert on the temperature-0 arm by construction, and this is what says so.
#[test]
fn a_greedy_row_never_has_a_stale_anchor() {
    let (_, c) = generate(&greedy(), 8, false);
    assert_eq!(c.pass_proposals, 9, "the prefill's block and eight more");
    assert_eq!(c.pass_proposals_stale, 0, "a greedy row commits the argmax");
    assert_eq!(c.stale_anchor_drafts, 0);
    assert_eq!(c.greedy_rows, 9);
    assert_eq!(c.sampled_rows, 0);
    assert!(c.clean());
}

/// A seeded temperature moves the anchor on some steps, and every one of those blocks is DROPPED.
#[test]
fn a_draw_that_moves_the_anchor_drops_the_block() {
    let (_, c) = generate(&hot(1234), 16, false);
    assert_eq!(c.sampled_rows, 17, "every row went through the seam");
    assert!(
        c.pass_proposals_stale > 0,
        "at temperature 2 the draw must move the anchor on at least one of 17 rows; \
         it moved on {} (offered {})",
        c.pass_proposals_stale,
        c.pass_proposals
    );
    assert_eq!(
        c.pass_proposals + c.pass_proposals_stale,
        17,
        "every step proposed exactly once: {} offered + {} dropped",
        c.pass_proposals,
        c.pass_proposals_stale
    );
    assert_eq!(
        c.stale_anchor_drafts, 0,
        "a dropped block is never verified, so it can earn nothing"
    );
    assert_eq!(c.stale_anchor_accepted, 0);
    assert!(c.clean());
}

/// **THE CONTROL, and it must differ from the case in the way the case differs from the truth.**
///
/// `keep_stale_anchor_blocks(true)` is this engine before ws-f35: the same steps, the same draws,
/// the same blocks — and the stale ones offered anyway. The two arms must disagree by exactly the
/// stale count, and the control's own tally must then say what those blocks were verified for.
#[test]
fn the_control_offers_the_stale_blocks_and_says_what_they_earned() {
    let (ids_law, law) = generate(&hot(1234), 16, false);
    let (ids_keep, keep) = generate(&hot(1234), 16, true);
    assert_eq!(
        ids_law, ids_keep,
        "the LAW MUST NOT MOVE A TOKEN: the drafter proposes, the target verifies, and a proposal \
         that is not made can only cost speculation, never change what is committed"
    );
    assert_eq!(keep.pass_proposals_stale, 0, "the control drops nothing");
    assert_eq!(
        keep.pass_proposals,
        law.pass_proposals + law.pass_proposals_stale,
        "the control offers exactly the blocks the law dropped, and no others"
    );
    // Every stale block the control offered is verified by the NEXT step and tallied there — so
    // the tally is the stale count, less the last step's block, which is still sitting in the
    // mailbox when the run ends. That off-by-the-last is the tally being taken where the
    // acceptance is rather than where the offer is, and it is asserted rather than rounded away.
    assert!(
        keep.stale_anchor_drafts == law.pass_proposals_stale
            || keep.stale_anchor_drafts + 1 == law.pass_proposals_stale,
        "the control tallied {} of the {} stale blocks it offered; at most one — the last step's, \
         never verified — may be outstanding",
        keep.stale_anchor_drafts,
        law.pass_proposals_stale
    );
    assert!(
        keep.stale_anchor_drafts > 0,
        "a control that tallies nothing has not been exercised"
    );
    assert!(keep.clean());
}

/// A dropped block leaves the mailbox EMPTY, which is the cost side of the law and is counted on
/// the drafter rather than hidden: the next step runs narrow instead of verifying a guess about a
/// text the sequence is not writing.
#[test]
fn the_drop_is_visible_as_an_empty_mailbox_and_not_as_a_silent_narrow_step() {
    let (_, law) = generate(&hot(1234), 16, false);
    let (_, keep) = generate(&hot(1234), 16, true);
    assert!(law.pass_proposals < keep.pass_proposals);
    assert_eq!(
        law.pass_proposals_stale,
        keep.pass_proposals - law.pass_proposals,
        "the difference between the two arms is the stale count and nothing else"
    );
}

// =================================================================================================
// ws-f37: THE PROPOSAL AFTER THE COMMIT — prod's order, and the law above becomes an invariant.
// =================================================================================================
//
// The step no longer proposes from the argmax: it STASHES the proposal's inputs and hands the slots
// back on `StepOutcome::proposal_pending`, and the executor runs `propose_after_commit` with the
// tokens the host sampler COMMITTED. What is gated here, on the host:
//
//   * the anchor is the committed token on EVERY slot at a temperature, with a sampler that is
//     forced to differ from the argmax on every row (the seam's argmax is moved to the bottom);
//   * at temperature 0 the two orders are BIT-IDENTICAL: the same ids and the same blocks, because
//     a greedy row commits the argmax the in-step order anchored on;
//   * THE CONTROL: a pass that feeds its drafter the stashed argmax instead of the handed anchor
//     (the in-step bug, injected) is caught on every such block as a DEFECT, refused by name,
//     never offered, and the run is unclean;
//   * the executor collects every stash before its next call into the pass.

/// The block a stub drafter makes from an anchor: a pure function of it, so two orders that feed
/// the same anchor make the same block and a moved anchor makes a different one.
fn block_of(anchor: TokenId) -> Vec<TokenId> {
    vec![anchor + 1, anchor + 2, anchor + 3]
}

/// A seam with its argmax moved to the bottom (values kept descending), so a sampler at `top_k`
/// 20 can never draw it.
fn without_argmax(row: &DeviceRow) -> DeviceRow {
    let mut row = row.clone();
    let id = row.top_ids.remove(0);
    row.top_values.remove(0);
    let floor = row.top_values.last().copied().unwrap_or(0.0) - 100.0;
    row.top_ids.push(id);
    row.top_values.push(floor);
    row
}

/// The walk's decider, handed every seam without its argmax.
#[derive(Debug)]
struct StripArgmax(Arc<dyn RowDecider>);

impl RowDecider for StripArgmax {
    fn decide(&self, j: usize, seam: Option<&DeviceRow>) -> Option<TokenId> {
        let row = seam.map(without_argmax);
        self.0.decide(j, row.as_ref())
    }
    fn ends_at(&self, j: usize, token: TokenId) -> bool {
        self.0.ends_at(j, token)
    }
}

/// [`NullForward`] with a drafter tier that proposes in EITHER order, and three switches.
#[derive(Debug)]
struct Ordered {
    inner: NullForward,
    /// The in-step order (`--propose-in-step`): blocks on the outcome, from the argmax.
    in_step: bool,
    /// THE CONTROL: feed the drafter the stashed ARGMAX instead of the anchor it is handed.
    feed_argmax: bool,
    /// Take the argmax out of every seam row, so a sampled row NEVER commits it.
    never_argmax: bool,
    /// `(sequence, argmax)` per slot of the step that stashed.
    stash: Option<Vec<(SeqId, TokenId)>>,
    /// `(handed anchor, argmax)` for every slot collected after the commit.
    handed: Vec<(TokenId, TokenId)>,
    /// Every block this pass made, in order, in either order.
    blocks: Vec<Vec<TokenId>>,
    /// Stashes the executor did not collect before its next call.
    uncollected: u64,
}

impl Ordered {
    fn new(in_step: bool) -> Self {
        Self {
            inner: NullForward::new(&[PREFILL], BATCHES, 8),
            in_step,
            feed_argmax: false,
            never_argmax: false,
            stash: None,
            handed: Vec::new(),
            blocks: Vec::new(),
            uncollected: 0,
        }
    }

    /// What the device does with a stash nobody collected: drop it, and count it.
    fn abandon(&mut self) {
        if self.stash.take().is_some() {
            self.uncollected += 1;
        }
    }

    /// Move the argmax to the bottom of the seam, keeping the values in descending order: the
    /// sampler at `top_k` 20 can then never draw it.
    fn strip_argmax(&self, out: &mut StepOutcome) {
        if !self.never_argmax {
            return;
        }
        for c in &mut out.commits {
            if let Some(Some(row)) = c.rows.last_mut() {
                *row = without_argmax(row);
            }
        }
    }

    /// Each slot's ARGMAX, read off its last seam row before [`Self::strip_argmax`] moves it (the
    /// seam's first id IS the argmax), or its committed token when the row carries no seam.
    fn row_argmaxes(out: &StepOutcome) -> Vec<TokenId> {
        out.commits
            .iter()
            .filter_map(|c| match c.rows.last() {
                Some(Some(row)) if !row.top_ids.is_empty() => Some(row.top_ids[0]),
                _ => c.tokens.last().copied(),
            })
            .collect()
    }

    /// Either make the blocks now (in-step) or stash the argmax for the collection.
    ///
    /// The in-step block is anchored where the DEVICE anchors it — `verdict.committed.last()` —
    /// which since ws-f40 is the drawn token on a decode step (the walk decided it inside the
    /// step) and still the argmax on a prefill chunk (the prompt's first token is drawn after it).
    fn propose_or_stash(&mut self, out: &mut StepOutcome, row_argmax: Vec<TokenId>) {
        let argmaxes: Vec<(SeqId, TokenId)> =
            out.commits.iter().map(|c| c.seq).zip(row_argmax).collect();
        if argmaxes.is_empty() {
            return;
        }
        if self.in_step {
            out.drafts = out
                .commits
                .iter()
                .filter_map(|c| c.tokens.last().map(|a| block_of(*a)))
                .collect();
            self.blocks.extend(out.drafts.iter().cloned());
        } else {
            out.proposal_pending = argmaxes.iter().map(|(s, _)| *s).collect();
            self.stash = Some(argmaxes);
        }
    }
}

impl ForwardRunner for Ordered {
    fn facts(&self) -> &ForwardFacts {
        self.inner.facts()
    }
    fn admit(
        &mut self,
        seq: SeqId,
        prompt_len: u32,
        kv_blocks: &[u32],
    ) -> Result<(), ForwardError> {
        self.abandon();
        self.inner.admit(seq, prompt_len, kv_blocks)
    }
    fn release(&mut self, seq: SeqId) {
        self.abandon();
        self.inner.release(seq);
    }
    fn prefill(
        &mut self,
        seq: SeqId,
        chunk: &[TokenId],
        done: u32,
        last: bool,
        sampling: &RowRequest,
    ) -> Result<StepOutcome, ForwardError> {
        self.abandon();
        let mut out = self.inner.prefill(seq, chunk, done, last, sampling)?;
        let argmax = Self::row_argmaxes(&out);
        self.strip_argmax(&mut out);
        if last {
            self.propose_or_stash(&mut out, argmax);
        }
        Ok(out)
    }
    fn decode(
        &mut self,
        slots: &[DecodeSlot],
        batch: u32,
        width: u32,
    ) -> Result<StepOutcome, ForwardError> {
        self.abandon();
        // A decode step's sampled rows are decided INSIDE the step (ws-f40), so the argmax is
        // taken out of the seam the walk sees, not only out of the rows handed back.
        let slots: Vec<DecodeSlot> = slots
            .iter()
            .map(|s| {
                let mut s = s.clone();
                if self.never_argmax {
                    s.decider = s
                        .decider
                        .take()
                        .map(|d| Decider(Arc::new(StripArgmax(d.0)) as Arc<dyn RowDecider>));
                }
                s
            })
            .collect();
        let mut out = self.inner.decode(&slots, batch, width)?;
        let argmax = Self::row_argmaxes(&out);
        self.strip_argmax(&mut out);
        self.propose_or_stash(&mut out, argmax);
        Ok(out)
    }
    fn propose_after_commit(
        &mut self,
        anchors: &[(SeqId, TokenId)],
    ) -> Result<Vec<Proposal>, ForwardError> {
        let stash = self
            .stash
            .take()
            .ok_or_else(|| ForwardError::Device("nothing stashed".to_owned()))?;
        if !stash.iter().map(|s| s.0).eq(anchors.iter().map(|a| a.0)) {
            return Err(ForwardError::Device("anchors for other slots".to_owned()));
        }
        let mut out = Vec::with_capacity(stash.len());
        for ((seq, argmax), (_, anchor)) in stash.iter().zip(anchors) {
            self.handed.push((*anchor, *argmax));
            let fed = if self.feed_argmax { *argmax } else { *anchor };
            let block = block_of(fed);
            self.blocks.push(block.clone());
            out.push(Proposal {
                seq: *seq,
                anchor: fed,
                block,
            });
        }
        Ok(out)
    }
    fn declared_launches(&self, rows: StepRows) -> Option<u32> {
        self.inner.declared_launches(rows)
    }
    fn dense_row(&mut self, row: u32) -> Result<Vec<f32>, ForwardError> {
        self.inner.dense_row(row)
    }
}

/// What [`generate_batch`] hands back.
struct Batched {
    /// Every sequence's ids.
    ids: Vec<Vec<TokenId>>,
    /// The committed LAST token of every slot of every step, prompts first, in the order the pass
    /// is handed anchors.
    committed: Vec<TokenId>,
    counters: EngineCounters,
    forward: Ordered,
}

/// `n` sequences prefilled one at a time, then `steps` decode steps of ALL of them at once in the
/// batch-4 bucket.
fn generate_batch(forward: Ordered, n: usize, params: &SamplingParams, steps: usize) -> Batched {
    let mut exec = RealExecutor::new(forward);
    let prompt_of = |i: usize| -> Vec<TokenId> {
        (0..PROMPT as TokenId)
            .map(|j| 1000 + 100 * i as TokenId + j)
            .collect()
    };
    let blocks_of = |i: usize| -> Vec<u32> { (i as u32 * 16..i as u32 * 16 + 16).collect() };
    let mut ids: Vec<Vec<TokenId>> = vec![Vec::new(); n];
    let mut committed: Vec<TokenId> = Vec::new();
    for i in 0..n {
        exec.admit(&AdmittedSeq {
            request_id: RequestId(i as u64 + 1),
            choice_index: 0,
            seq_id: SeqId(i as u64 + 1),
            prompt_tokens: prompt_of(i),
            kv_blocks: blocks_of(i),
            shared_prefix_blocks: 0,
            cow_source: None,
            gdn_set: i as u32 + 1,
            gdn_slots: 2,
        });
    }
    for (i, seq_ids) in ids.iter_mut().enumerate() {
        let slots = vec![BatchSlot {
            request_id: RequestId(i as u64 + 1),
            seq_id: SeqId(i as u64 + 1),
            slot: 0,
            kv_blocks: blocks_of(i),
            gdn_set: i as u32 + 1,
            verify_width: 0,
            input_tokens: prompt_of(i),
            context_len: 0,
            first_step: true,
            chunk: Some(PrefillChunk {
                prompt_tokens: PROMPT as u32,
                tokens_done_before: 0,
                last: true,
                next_prompt_token: None,
            }),
            sampling: sampling(params.clone(), 0),
        }];
        let out = exec.execute(&StepBatch {
            phase: Phase::Prefill,
            bucket: Bucket::prefill(PREFILL, 8192, true),
            step_index: 0,
            slots: &slots,
            padded_slots: 0,
        });
        seq_ids.extend(out.seqs[0].tokens.iter().copied());
        committed.extend(out.seqs[0].tokens.last().copied());
    }
    let mut context: Vec<u32> = vec![PROMPT as u32; n];
    for step in 1..=steps as u64 {
        let slots: Vec<BatchSlot> = (0..n)
            .map(|i| BatchSlot {
                request_id: RequestId(i as u64 + 1),
                seq_id: SeqId(i as u64 + 1),
                slot: i as u32,
                kv_blocks: blocks_of(i),
                gdn_set: i as u32 + 1,
                verify_width: 0,
                input_tokens: vec![*ids[i].last().expect("a token")],
                context_len: context[i],
                first_step: false,
                chunk: None,
                sampling: sampling(params.clone(), step),
            })
            .collect();
        let out = exec.execute(&StepBatch {
            phase: Phase::Decode,
            bucket: Bucket::decode(4, 1, 8192, true),
            step_index: step,
            slots: &slots,
            padded_slots: 4 - n as u32,
        });
        for (i, q) in out.seqs.iter().enumerate() {
            ids[i].extend(q.tokens.iter().copied());
            context[i] += q.tokens.len() as u32;
            committed.extend(q.tokens.last().copied());
        }
    }
    let counters = exec.counters();
    let forward = std::mem::replace(exec.forward_mut(), Ordered::new(false));
    Batched {
        ids,
        committed,
        counters,
        forward,
    }
}

/// **THE ANCHOR IS THE COMMITTED TOKEN ON EVERY SLOT**, with a sampler that never commits the
/// argmax: every one of the pass's anchors is the token the slot committed, and not one of them is
/// the argmax the in-step order would have anchored on.
#[test]
fn after_the_commit_every_slot_is_anchored_on_the_token_it_committed() {
    let mut f = Ordered::new(false);
    f.never_argmax = true;
    let r = generate_batch(f, 3, &hot(1234), 12);
    let handed: Vec<TokenId> = r.forward.handed.iter().map(|h| h.0).collect();
    assert_eq!(
        handed, r.committed,
        "the anchors handed back are the committed tokens, slot by slot, step by step"
    );
    assert!(
        r.forward.handed.iter().all(|(a, argmax)| a != argmax),
        "the control on the sampler: it must differ from the argmax on EVERY row, or this test \
         could not tell the two orders apart"
    );
    let c = &r.counters;
    assert_eq!(
        r.forward.handed.len(),
        3 + 3 * 12,
        "three prompts, twelve steps of three"
    );
    assert_eq!(c.proposals_after_commit, 39);
    assert_eq!(c.pass_proposals, 39, "every block offered");
    assert_eq!(c.stale_anchor_defects, 0);
    assert_eq!(
        c.pass_proposals_stale, 0,
        "the in-step law has nothing to drop: no block is stale"
    );
    assert_eq!(c.sampled_rows, 39);
    assert_eq!(
        r.forward.uncollected, 0,
        "the executor collected every stash"
    );
    assert!(c.clean());
}

/// **THE CONTROL: THE IN-STEP BUG, INJECTED.** A pass that feeds its drafter the argmax it stashed
/// rather than the anchor it was handed makes exactly the blocks the old order made. Every one of
/// them is a DEFECT now — refused by name, never offered — and the run is unclean. The ids are the
/// case's: a proposal cannot move a committed token.
#[test]
fn an_injected_stale_anchor_is_a_defect_refused_by_name() {
    let mut case = Ordered::new(false);
    case.never_argmax = true;
    let case = generate_batch(case, 3, &hot(1234), 12);
    let mut ctl = Ordered::new(false);
    ctl.never_argmax = true;
    ctl.feed_argmax = true;
    let ctl = generate_batch(ctl, 3, &hot(1234), 12);
    assert_eq!(case.ids, ctl.ids, "a refused block moves no token");
    let c = &ctl.counters;
    assert_eq!(c.proposals_after_commit, 39);
    assert_eq!(
        c.stale_anchor_defects, 39,
        "every block is anchored on the argmax"
    );
    assert_eq!(c.pass_proposals, 0, "and not one of them is offered");
    assert!(
        !c.clean(),
        "a stale anchor after the commit is a defect, not a rate"
    );
    assert!(
        c.proposal_line().contains("39 stale-anchor DEFECT"),
        "{}",
        c.proposal_line()
    );
    assert!(case.counters.clean());
}

/// **AT TEMPERATURE 0 THE TWO ORDERS ARE BIT-IDENTICAL**: the same ids and the same blocks, in
/// the same order. A greedy row commits the argmax, so the anchor the in-step order used and the
/// one the commit hands back are one token, and nothing else about the proposal moved.
#[test]
fn at_temperature_0_the_two_orders_propose_the_same_blocks_and_commit_the_same_ids() {
    let inside = generate_batch(Ordered::new(true), 3, &greedy(), 12);
    let after = generate_batch(Ordered::new(false), 3, &greedy(), 12);
    assert_eq!(inside.ids, after.ids, "the same ids");
    assert_eq!(inside.committed, after.committed);
    assert_eq!(
        inside.forward.blocks, after.forward.blocks,
        "the same blocks, in the same order"
    );
    assert_eq!(inside.forward.blocks.len(), 39);
    assert_eq!(
        inside.counters.pass_proposals,
        after.counters.pass_proposals
    );
    assert_eq!(inside.counters.pass_proposals_stale, 0);
    assert_eq!(
        inside.counters.proposals_after_commit, 0,
        "the in-step order collects nothing"
    );
    assert_eq!(after.counters.proposals_after_commit, 39);
    assert_eq!(after.counters.stale_anchor_defects, 0);
    assert_eq!(after.counters.greedy_rows, 39);
    assert!(inside.counters.clean() && after.counters.clean());
    // The control that the comparison can fail: a SAMPLED run of the in-step order anchors on
    // argmaxes the sampler replaced, so its blocks are not the after-commit order's — and its own
    // law drops every one of them.
    let mut hot_in = Ordered::new(true);
    hot_in.never_argmax = true;
    let mut hot_after = Ordered::new(false);
    hot_after.never_argmax = true;
    let hot_in = generate_batch(hot_in, 3, &hot(1234), 12);
    let hot_after = generate_batch(hot_after, 3, &hot(1234), 12);
    assert_ne!(hot_in.forward.blocks, hot_after.forward.blocks);
    // Since ws-f40 a decode step's sampled token is decided INSIDE the step (the drawn walk), so
    // the in-step order's decode blocks are anchored on the committed draw and none is stale; the
    // prompt's first token is still drawn after its prefill chunk, and those three blocks are.
    assert_eq!(
        hot_in.counters.pass_proposals_stale, 3,
        "the in-step order drops exactly the three prefill blocks: every decode block is anchored \
         on the walk's own draw"
    );
}

/// A pass that stashed for OTHER slots than the step committed is not collected: counted, the
/// step's token stands, and the run is unclean — and the pass sees its stash abandoned at the
/// executor's next call, which is where the device counts it.
#[test]
fn a_stash_that_names_other_slots_is_not_collected() {
    #[derive(Debug)]
    struct Mislabel(Ordered);
    impl ForwardRunner for Mislabel {
        fn facts(&self) -> &ForwardFacts {
            self.0.facts()
        }
        fn admit(&mut self, s: SeqId, p: u32, k: &[u32]) -> Result<(), ForwardError> {
            self.0.admit(s, p, k)
        }
        fn release(&mut self, s: SeqId) {
            self.0.release(s);
        }
        fn prefill(
            &mut self,
            s: SeqId,
            c: &[TokenId],
            d: u32,
            l: bool,
            r: &RowRequest,
        ) -> Result<StepOutcome, ForwardError> {
            let mut out = self.0.prefill(s, c, d, l, r)?;
            for p in &mut out.proposal_pending {
                *p = SeqId(p.0 + 100);
            }
            Ok(out)
        }
        fn decode(
            &mut self,
            s: &[DecodeSlot],
            b: u32,
            w: u32,
        ) -> Result<StepOutcome, ForwardError> {
            self.0.decode(s, b, w)
        }
        fn propose_after_commit(
            &mut self,
            a: &[(SeqId, TokenId)],
        ) -> Result<Vec<Proposal>, ForwardError> {
            self.0.propose_after_commit(a)
        }
        fn declared_launches(&self, r: StepRows) -> Option<u32> {
            self.0.declared_launches(r)
        }
    }
    let mut exec = RealExecutor::new(Mislabel(Ordered::new(false)));
    exec.admit(&AdmittedSeq {
        request_id: RequestId(1),
        choice_index: 0,
        seq_id: SeqId(1),
        prompt_tokens: prompt(),
        kv_blocks: (0..16).collect(),
        shared_prefix_blocks: 0,
        cow_source: None,
        gdn_set: 1,
        gdn_slots: 2,
    });
    let slots = vec![prefill_slot(sampling(greedy(), 0))];
    let out = exec.execute(&StepBatch {
        phase: Phase::Prefill,
        bucket: Bucket::prefill(PREFILL, 8192, true),
        step_index: 0,
        slots: &slots,
        padded_slots: 0,
    });
    assert_eq!(out.seqs[0].tokens.len(), 1, "the token stands");
    let c = exec.counters();
    assert_eq!(c.proposal_collect_failures, 1);
    assert_eq!(c.pass_proposals, 0);
    assert!(!c.clean());
    exec.release(SeqId(1));
    assert_eq!(exec.forward_mut().0.uncollected, 1);
}
