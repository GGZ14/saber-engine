//! **`top_k = 1` AT WIDTH 8 IN A BATCH OF 4 IS THE ARGMAX CHAIN** — and the three ways it could not
//! be, each one a control that fails (workstream F39).
//!
//! F38's serve arm found `top_k 1` at temperature 0.7 matching the temperature-0 answer on 12 of
//! 24 branches, every divergence one token "inserted" mid-stream, on the leader as often as on a
//! branch. A `top_k 1` row is greedy except where the mask keeps more than one id, so an inserted
//! token is one of four things, and this file separates them on the host, where every row's argmax
//! is known:
//!
//! * **(a) the sampler drew from the wrong row** — the `DeviceRow` handed to it for a width > 1
//!   slot is not the row that chose the slot's last token (`i * width + accepted`);
//! * **(b) the acceptance committed a token the target's argmax did not choose** — a verify-width
//!   law under sampling;
//! * **(c) a stream artefact** — asked of the SERVE plane over HTTP, not here: F38's divergences
//!   were `/v1/completions` NON-streaming replies, which have no SSE to duplicate a chunk in;
//! * **(d) an exact tie at the top of the row**, which the `top_k` mask keeps whole (prod's
//!   `apply_top_k_only` is `logits < the k-th largest`), so the draw may commit the tied id that
//!   temperature 0 — lowest id first — does not. `top_ties` / `top_tie_draws` count it.
//!
//! The CASE is the executor exactly as it serves: four slots in one `b4s8` step, a drafter whose
//! scripts are the temperature-0 chain (so blocks are accepted at every depth), every slot sampled
//! at 0.7 with `top_k 1`. Its ids must be the chain, token for token, with zero tie draws. Each
//! CONTROL tampers with what the pass hands back in exactly one of the ways above and must
//! diverge — and only (d)'s divergence is counted as a tie draw, which is what lets the card's
//! divergences be attributed rather than argued about.

use std::sync::{Arc, Mutex};

use saber_engine::{
    Decider, DecodeSlot, EngineCounters, ForwardError, ForwardFacts, ForwardRunner, NullForward,
    RealExecutor, RecordedDrafter, RowDecider, RowRequest, StepOutcome, StepRows,
};
use saber_plan::{Bucket, Phase};
use saber_sched::{
    AdmittedSeq, BatchSlot, PrefillChunk, SeqId, SlotSampling, StepBatch, StepExecutor,
};
use saber_serve::engine::{RequestId, TokenId};
use saber_serve::sampler::DeviceRow;
use saber_serve::sampling::SamplingParams;

const PREFILL: u32 = 2048;
const BATCH: u32 = 4;
const WIDTH: u32 = 8;
const PROMPT: usize = 12;
/// Tokens each sequence must reach.
const TOKENS: usize = 48;

/// What a control does to the step the pass returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tamper {
    /// The case: nothing.
    None,
    /// (a) The sampled row is the slot's FIRST row instead of the row that chose its last token.
    WrongRow,
    /// (b) An accepted prefix token is replaced by an id the target's argmax did not choose.
    CommitNonArgmax,
    /// (d) The sampled row's runner-up is raised to TIE the argmax (only where its id is higher,
    /// so the argmax — lowest id first — is still the token the device committed).
    TieAtTheTop,
}

/// The walk's decider under a tamper: (a) every row after the first is handed the FIRST row's
/// seam; (d) every seam has its runner-up raised to tie the argmax (only where the runner-up's id
/// is higher, so the argmax — lowest id first — is still the device's token).
#[derive(Debug)]
struct TamperedSeams {
    inner: Arc<dyn RowDecider>,
    tamper: Tamper,
    row0: Mutex<Option<DeviceRow>>,
}

impl RowDecider for TamperedSeams {
    fn decide(&self, j: usize, seam: Option<&DeviceRow>) -> Option<TokenId> {
        let row = match self.tamper {
            Tamper::WrongRow => {
                let mut first = self.row0.lock().expect("row 0");
                if j == 0 {
                    *first = seam.cloned();
                }
                first.clone()
            }
            Tamper::TieAtTheTop => seam.map(|r| {
                let mut r = r.clone();
                if r.top_ids.len() > 1 && r.top_ids[1] > r.top_ids[0] {
                    r.top_values[1] = r.top_values[0];
                }
                r
            }),
            _ => seam.cloned(),
        };
        self.inner.decide(j, row.as_ref())
    }
    fn ends_at(&self, j: usize, token: TokenId) -> bool {
        self.inner.ends_at(j, token)
    }
}

#[derive(Debug)]
struct Tampered {
    inner: NullForward,
    tamper: Tamper,
}

impl ForwardRunner for Tampered {
    fn facts(&self) -> &ForwardFacts {
        self.inner.facts()
    }
    fn admit(
        &mut self,
        seq: SeqId,
        prompt_len: u32,
        kv_blocks: &[u32],
    ) -> Result<(), ForwardError> {
        self.inner.admit(seq, prompt_len, kv_blocks)
    }
    fn release(&mut self, seq: SeqId) {
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
        self.inner.prefill(seq, chunk, done, last, sampling)
    }
    fn decode(
        &mut self,
        slots: &[DecodeSlot],
        batch: u32,
        width: u32,
    ) -> Result<StepOutcome, ForwardError> {
        // Since ws-f40 a sampled slot's rows are decided INSIDE the step by the walk, so a tamper
        // of what the SAMPLER sees is applied to the seams the walk is handed, not to the rows the
        // step hands back afterwards (which are only reported).
        let slots: Vec<DecodeSlot> = slots
            .iter()
            .map(|s| {
                let mut s = s.clone();
                if matches!(self.tamper, Tamper::WrongRow | Tamper::TieAtTheTop) {
                    s.decider = s.decider.take().map(|d| {
                        Decider(Arc::new(TamperedSeams {
                            inner: d.0,
                            tamper: self.tamper,
                            row0: Mutex::new(None),
                        }) as Arc<dyn RowDecider>)
                    });
                }
                s
            })
            .collect();
        let mut out = self.inner.decode(&slots, batch, width)?;
        for c in &mut out.commits {
            let n = c.tokens.len();
            match self.tamper {
                Tamper::None => {}
                Tamper::WrongRow => {
                    if n > 1 {
                        let first = c.rows[0].clone();
                        c.rows[n - 1] = first;
                    }
                }
                Tamper::CommitNonArgmax => {
                    if n > 1 {
                        c.tokens[0] = c.tokens[0].wrapping_add(1).max(1);
                    }
                }
                Tamper::TieAtTheTop => {
                    if let Some(Some(row)) = c.rows.last_mut() {
                        if row.top_ids.len() > 1 && row.top_ids[1] > row.top_ids[0] {
                            row.top_values[1] = row.top_values[0];
                        }
                    }
                }
            }
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

fn prompt(seq: u64) -> Vec<TokenId> {
    (0..PROMPT as TokenId)
        .map(|i| 1000 + i + 97 * seq as TokenId)
        .collect()
}

fn admitted(seq: u64) -> AdmittedSeq {
    AdmittedSeq {
        request_id: RequestId(seq),
        choice_index: 0,
        seq_id: SeqId(seq),
        prompt_tokens: prompt(seq),
        kv_blocks: (0..16).map(|b| b + 16 * seq as u32).collect(),
        shared_prefix_blocks: 0,
        cow_source: None,
        gdn_set: seq as u32,
        gdn_slots: 2,
    }
}

fn sampling(params: &SamplingParams, step: u64) -> SlotSampling {
    SlotSampling {
        params: params.clone(),
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

/// The served-path shape of a greedy-equivalent request: temperature 0.7, `top_k 1`, seeded so a
/// run is reproducible (the law holds for every key).
fn top_k_one(seed: u64) -> SamplingParams {
    SamplingParams {
        temperature: 0.7,
        top_k: 1,
        seed: Some(seed),
        ..SamplingParams::default()
    }
}

fn prefill_slot(seq: u64, s: SlotSampling) -> BatchSlot {
    BatchSlot {
        request_id: RequestId(seq),
        seq_id: SeqId(seq),
        slot: 0,
        kv_blocks: admitted(seq).kv_blocks,
        gdn_set: seq as u32,
        verify_width: 0,
        input_tokens: prompt(seq),
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

/// Run `seqs` to `TOKENS` tokens each. Width 1 / batch 1 per sequence when `width == 1` (the
/// reference chain), otherwise all of them together in `b4s8` steps.
fn run<R: ForwardRunner>(
    exec: &mut RealExecutor<R>,
    seqs: &[u64],
    params: &dyn Fn(u64) -> SamplingParams,
    width: u32,
) -> (Vec<Vec<TokenId>>, u32) {
    let mut ids: Vec<Vec<TokenId>> = Vec::new();
    for &seq in seqs {
        exec.admit(&admitted(seq));
        let out = exec.execute(&StepBatch {
            phase: Phase::Prefill,
            bucket: Bucket::prefill(PREFILL, 8192, true),
            step_index: 0,
            slots: &[prefill_slot(seq, sampling(&params(seq), 0))],
            padded_slots: 0,
        });
        ids.push(out.seqs[0].tokens.clone());
    }
    let mut widest = 0u32;
    let mut step = 1u64;
    let groups: Vec<Vec<usize>> = if width == 1 {
        (0..seqs.len()).map(|i| vec![i]).collect()
    } else {
        vec![(0..seqs.len()).collect()]
    };
    for group in groups {
        while group.iter().any(|&i| ids[i].len() < TOKENS) {
            let slots: Vec<BatchSlot> = group
                .iter()
                .map(|&i| {
                    let seq = seqs[i];
                    BatchSlot {
                        verify_width: width - 1,
                        input_tokens: vec![*ids[i].last().expect("a token")],
                        context_len: (PROMPT + ids[i].len() - 1) as u32,
                        first_step: false,
                        chunk: None,
                        sampling: sampling(&params(seq), step),
                        ..prefill_slot(seq, sampling(&params(seq), step))
                    }
                })
                .collect();
            let batch = if width == 1 { 1 } else { BATCH };
            let out = exec.execute(&StepBatch {
                phase: Phase::Decode,
                bucket: Bucket::decode(batch, width, 8192, true),
                step_index: step,
                slots: &slots,
                padded_slots: batch - slots.len() as u32,
            });
            for (&i, s) in group.iter().zip(&out.seqs) {
                assert!(
                    !s.tokens.is_empty(),
                    "step {step}: seq {} committed nothing",
                    seqs[i]
                );
                widest = widest.max(s.tokens.len() as u32);
                ids[i].extend_from_slice(&s.tokens);
            }
            step += 1;
            assert!(step < 400, "the run is not advancing");
        }
    }
    for v in &mut ids {
        v.truncate(TOKENS);
    }
    (ids, widest)
}

const SEQS: &[u64] = &[1, 2, 3, 4];

/// The temperature-0 chain each sequence gets alone, at width 1 — every row's argmax.
fn chain() -> Vec<Vec<TokenId>> {
    let mut exec = RealExecutor::new(NullForward::with_widths(
        &[PREFILL],
        &[1, 4],
        &[1, WIDTH],
        8,
    ));
    run(&mut exec, SEQS, &|_| greedy(), 1).0
}

/// The case or a control: four sequences at `top_k 1` / 0.7, one `b4s8` step at a time, the
/// drafter scripted with the chain so blocks are accepted (`truthful` of every 7).
fn served(tamper: Tamper, truthful: u32) -> (Vec<Vec<TokenId>>, u32, EngineCounters) {
    let reference = chain();
    let mut drafter = RecordedDrafter::new(WIDTH - 1, truthful);
    for (i, &seq) in SEQS.iter().enumerate() {
        // The script starts AFTER the first token (the prefill's), which is the drafter's anchor.
        drafter.record(SeqId(seq), reference[i][1..].to_vec());
    }
    let inner = NullForward::with_widths(&[PREFILL], &[1, 4], &[1, WIDTH], 8);
    let mut exec = RealExecutor::with_drafter(Tampered { inner, tamper }, Box::new(drafter));
    let (ids, widest) = run(&mut exec, SEQS, &|seq| top_k_one(9000 + seq), WIDTH);
    (ids, widest, exec.counters())
}

fn first_divergence(a: &[TokenId], b: &[TokenId]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// THE CASE. Width 8, batch 4, `top_k 1` at 0.7: the ids are the argmax chain, token for token,
/// at full acceptance and at partial acceptance, and no tie was drawn.
#[test]
fn top_k_one_at_width_eight_in_a_batch_of_four_is_the_argmax_chain() {
    let reference = chain();
    for truthful in [7, 3] {
        let (ids, widest, c) = served(Tamper::None, truthful);
        assert!(
            widest > 1,
            "truthful {truthful}: nothing above width 1 was committed"
        );
        assert!(
            c.accepted_tokens > 0,
            "truthful {truthful}: the drafter was never accepted"
        );
        assert!(
            c.sampled_rows > 0,
            "truthful {truthful}: no row was sampled"
        );
        assert_eq!(c.greedy_rows, 0, "every row asked for a temperature");
        for (i, (got, want)) in ids.iter().zip(&reference).enumerate() {
            assert_eq!(
                first_divergence(got, want),
                None,
                "truthful {truthful}: seq {} left the argmax chain",
                SEQS[i]
            );
        }
        assert_eq!(c.top_ties, 0, "the stub's rows have no ties");
        assert_eq!(c.top_tie_draws, 0);
        assert!(c.clean(), "{}", c.line());
    }
}

/// (a) THE WRONG ROW: a pass that hands the sampler the slot's first row instead of the row that
/// chose its last token commits that row's argmax — the chain's token one or more positions back.
/// It diverges, and it is not a tie.
#[test]
fn control_a_the_wrong_row_leaves_the_chain_and_is_not_a_tie() {
    let reference = chain();
    let (ids, _, c) = served(Tamper::WrongRow, 7);
    let diverged = ids
        .iter()
        .zip(&reference)
        .filter(|(g, w)| first_divergence(g, w).is_some())
        .count();
    assert!(diverged > 0, "a wrong row went unnoticed");
    assert_eq!(c.top_tie_draws, 0, "a wrong row is not a tie");
}

/// (b) THE ACCEPTANCE: a pass that commits an id the target's argmax did not choose in the accepted
/// prefix. It diverges, and it is not a tie.
#[test]
fn control_b_a_committed_non_argmax_leaves_the_chain_and_is_not_a_tie() {
    let reference = chain();
    let (ids, _, c) = served(Tamper::CommitNonArgmax, 7);
    let diverged = ids
        .iter()
        .zip(&reference)
        .filter(|(g, w)| first_divergence(g, w).is_some())
        .count();
    assert!(diverged > 0, "a committed non-argmax went unnoticed");
    assert_eq!(c.top_tie_draws, 0, "a committed non-argmax is not a tie");
}

/// (d) THE TIE: the same rows with the runner-up raised to tie the argmax. Temperature 0 would
/// still commit the chain (the argmax is the lower id); `top_k 1` keeps both and the draw commits
/// the higher one on a share of the rows — and EVERY such divergence is a counted tie draw: the
/// first token at which each diverged sequence leaves the chain was drawn off a tie.
#[test]
fn control_d_a_tie_at_the_top_is_the_only_divergence_counted_as_a_tie_draw() {
    let reference = chain();
    let (ids, _, c) = served(Tamper::TieAtTheTop, 7);
    let diverged = ids
        .iter()
        .zip(&reference)
        .filter(|(g, w)| first_divergence(g, w).is_some())
        .count();
    assert!(c.top_ties > 0, "the tamper made no tie");
    assert!(
        diverged > 0 && c.top_tie_draws > 0,
        "a tie at the top of every sampled row never moved a draw: {} ties, {} tie draws",
        c.top_ties,
        c.top_tie_draws
    );
    assert!(
        c.top_tie_draws >= diverged as u64,
        "{diverged} sequences left the chain on only {} tie draws",
        c.top_tie_draws
    );
}
