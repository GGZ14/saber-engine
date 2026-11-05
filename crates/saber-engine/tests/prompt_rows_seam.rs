//! `prompt_logprobs` through the engine seam, on the host (V7).
//!
//! What this gates is the WIRING between the scheduler's prefill slot and the pass's
//! `prompt_rows`: that a chunk's rows are scored against the NEXT prompt token each (the chunk's
//! last row against the token the scheduler names past the chunk, and no row at all for the
//! prompt's final token, which samples), that the ranks come from the seam when it decides them and
//! from a dense read when it does not, and that a request that does not ask pays nothing — no call,
//! no counter, the same token.

use saber_engine::{NullForward, RealExecutor};
use saber_plan::{Bucket, Phase};
use saber_sched::{
    AdmittedSeq, BatchSlot, PrefillChunk, SeqId, SlotSampling, StepBatch, StepExecutor, StepResult,
};
use saber_serve::engine::{RequestId, TokenId};
use saber_serve::sampling::SamplingParams;

const PREFILL: u32 = 2048;

fn prompt() -> Vec<TokenId> {
    (0..20).map(|i| 1000 + i).collect()
}

fn executor() -> RealExecutor<NullForward> {
    let mut e = RealExecutor::new(NullForward::new(&[PREFILL], &[1, 2, 4], 8));
    e.admit(&AdmittedSeq {
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
    e
}

fn chunk(
    exec: &mut RealExecutor<NullForward>,
    range: std::ops::Range<usize>,
    last: bool,
    plp: Option<i32>,
) -> StepResult {
    let p = prompt();
    let slots = vec![BatchSlot {
        request_id: RequestId(1),
        seq_id: SeqId(1),
        slot: 0,
        kv_blocks: (0..16).collect(),
        gdn_set: 1,
        verify_width: 0,
        input_tokens: p[range.clone()].to_vec(),
        context_len: range.start as u32,
        first_step: range.start == 0,
        chunk: Some(PrefillChunk {
            prompt_tokens: range.len() as u32,
            tokens_done_before: range.start as u32,
            last,
            next_prompt_token: if last { None } else { Some(p[range.end]) },
        }),
        sampling: SlotSampling {
            params: SamplingParams {
                temperature: 0.0,
                prompt_logprobs: plp,
                ..SamplingParams::default()
            },
            ..SlotSampling::greedy()
        },
    }];
    exec.execute(&StepBatch {
        phase: Phase::Prefill,
        bucket: Bucket::prefill(PREFILL, 8192, true),
        step_index: 0,
        slots: &slots,
        padded_slots: 0,
    })
}

#[test]
fn two_chunks_score_every_prompt_row_but_the_last_against_the_next_token() {
    let mut exec = executor();
    let a = chunk(&mut exec, 0..8, false, Some(2));
    let b = chunk(&mut exec, 8..20, true, Some(2));
    let p = prompt();
    let first = a.seqs[0].prompt_logprobs.as_ref().expect("chunk 1 scored");
    let second = b.seqs[0].prompt_logprobs.as_ref().expect("chunk 2 scored");
    assert_eq!(first.len(), 8, "every row of a chunk that is not the last");
    assert_eq!(
        second.len(),
        11,
        "every row but the last of the one that is"
    );
    for (i, pos) in first.iter().chain(second).enumerate() {
        let e = pos.as_ref().expect("a distribution");
        assert_eq!(
            e[0].token_id,
            p[i + 1],
            "row {i} is scored against the next token"
        );
        assert!(e.len() == 2 || e.len() == 3, "k or k + 1 entries");
        assert!(e[0].logprob <= 0.0 && e[0].rank.is_some());
    }
    let c = exec.counters();
    assert_eq!(c.prompt_logprob_chunks, 2);
    assert_eq!(c.prompt_logprob_rows, 19);
    assert_eq!(c.prompt_logprob_refusals, 0);
    // The stub's prompt tokens are background ids, tied with most of the vocabulary: the seam
    // cannot count their rank and every one was counted densely.
    assert_eq!(exec.forward_mut().dense_prompt_ranks, 19);
}

#[test]
fn a_request_that_does_not_ask_pays_nothing() {
    let mut with = executor();
    let a = chunk(&mut with, 0..20, true, Some(1));
    let mut without = executor();
    let b = chunk(&mut without, 0..20, true, None);
    assert!(b.seqs[0].prompt_logprobs.is_none());
    assert_eq!(a.seqs[0].tokens, b.seqs[0].tokens, "the same first token");
    let c = without.counters();
    assert_eq!(
        (c.prompt_logprob_chunks, c.prompt_logprob_rows),
        (0, 0),
        "{}",
        c.prompt_line()
    );
    assert_eq!(without.forward_mut().dense_prompt_ranks, 0);
}

/// NEGATIVE CONTROL: the rows are the LAST chunk's, of that chunk's sequence, and no more of them
/// than it fed. Anything else is refused by name; the executor counts it and answers `None`, which
/// the scheduler turns into an error rather than a `null`.
#[test]
fn rows_asked_after_another_step_are_refused() {
    let mut exec = executor();
    let _ = chunk(&mut exec, 0..20, true, None);
    use saber_engine::forward::ForwardRunner;
    let f = exec.forward_mut();
    assert!(
        f.prompt_rows(SeqId(1), &[1001]).is_ok(),
        "right after the chunk"
    );
    assert!(
        f.prompt_rows(SeqId(2), &[1001]).is_err(),
        "another sequence's rows are not these"
    );
    assert!(
        f.prompt_rows(SeqId(1), &[1; 21]).is_err(),
        "more rows than the chunk fed"
    );
}
