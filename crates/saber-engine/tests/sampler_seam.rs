//! The sampler, wired into the step loop — on the host, with the device taken out.
//!
//! `saber_serve::sampler` is gated against the reference server's own V1 sampler over fixed logit rows (workstream
//! V1, 184/184). What this file gates is the WIRING: that a request's parameters reach the row, that
//! the row the head wrote is the row the sampler decides, that the token it draws is the token the
//! step commits, and — the one that matters most — that **temperature 0 is still the argmax and
//! nothing else runs**.
//!
//! Every branch declares a counter (`CLAUDE.md` rule 5) and every test here reads them, because
//! "the right answer by the wrong path" is exactly what a sampler seam fails as: a row answered
//! from the argmax when it asked to be drawn looks like a perfectly good token.
//!
//! [`NullForward`] is the device stub. Its "logit row" is not a model — a peak, thirty-one
//! runners-up and a flat floor — but it has the SHAPE of one, which is what lets the seam, its
//! refusal, the dense fallback and the noise all be exercised with no card.

use saber_engine::{EngineCounters, NullForward, RealExecutor};
use saber_plan::{Bucket, Phase};
use saber_sched::{
    AdmittedSeq, BatchSlot, PenaltyHistory, PrefillChunk, SeqId, SlotSampling, StepBatch,
    StepExecutor, StepResult,
};
use saber_serve::engine::{RequestId, TokenId};
use saber_serve::sampling::SamplingParams;
use std::sync::Arc;

const PREFILL: u32 = 2048;
const BATCHES: &[u32] = &[1, 2, 4];
const PROMPT: usize = 12;

fn executor() -> RealExecutor<NullForward> {
    RealExecutor::new(NullForward::new(&[PREFILL], BATCHES, 8))
}

fn prompt() -> Vec<TokenId> {
    (0..PROMPT as TokenId).map(|i| 1000 + i).collect()
}

fn admit(exec: &mut RealExecutor<NullForward>, seq: u64) {
    exec.admit(&AdmittedSeq {
        request_id: RequestId(seq),
        choice_index: 0,
        seq_id: SeqId(seq),
        prompt_tokens: prompt(),
        kv_blocks: (0..16).collect(),
        shared_prefix_blocks: 0,
        cow_source: None,
        gdn_set: seq as u32,
        gdn_slots: 2,
    });
}

/// The parameters a request sends, in the shape the scheduler hands them over.
fn sampling(params: SamplingParams, step: u64, choice: u32) -> SlotSampling {
    SlotSampling {
        params,
        choice_index: choice,
        step_index: step,
        position: step,
        penalties: None,
    }
}

fn greedy_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    }
}

/// Prod's own serving defaults, which is what an unparameterised request is sampled at.
fn prod_params(seed: Option<u64>) -> SamplingParams {
    SamplingParams {
        temperature: 0.7,
        top_p: 0.95,
        top_k: 20,
        seed,
        ..SamplingParams::default()
    }
}

fn prefill_slot(seq: u64, sampling: SlotSampling) -> BatchSlot {
    BatchSlot {
        request_id: RequestId(seq),
        seq_id: SeqId(seq),
        slot: 0,
        kv_blocks: (0..16).collect(),
        gdn_set: seq as u32,
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
        sampling,
    }
}

fn decode_slot(seq: u64, input: &[TokenId], context: u32, sampling: SlotSampling) -> BatchSlot {
    BatchSlot {
        request_id: RequestId(seq),
        seq_id: SeqId(seq),
        slot: 0,
        kv_blocks: (0..16).collect(),
        gdn_set: seq as u32,
        verify_width: 0,
        input_tokens: input.to_vec(),
        context_len: context,
        first_step: false,
        chunk: None,
        sampling,
    }
}

fn run_prefill(exec: &mut RealExecutor<NullForward>, seq: u64, s: SlotSampling) -> StepResult {
    let slots = vec![prefill_slot(seq, s)];
    exec.execute(&StepBatch {
        phase: Phase::Prefill,
        bucket: Bucket::prefill(PREFILL, 8192, true),
        step_index: 0,
        slots: &slots,
        padded_slots: 0,
    })
}

fn run_decode(
    exec: &mut RealExecutor<NullForward>,
    seq: u64,
    input: &[TokenId],
    context: u32,
    step: u64,
    s: SlotSampling,
) -> StepResult {
    let slots = vec![decode_slot(seq, input, context, s)];
    exec.execute(&StepBatch {
        phase: Phase::Decode,
        bucket: Bucket::decode(1, 1, 8192, true),
        step_index: step,
        slots: &slots,
        padded_slots: 0,
    })
}

/// One whole generation: a prefill and `n` decode steps, with the same parameters throughout.
fn generate(params: &SamplingParams, steps: usize, choice: u32) -> (Vec<TokenId>, EngineCounters) {
    let mut exec = executor();
    admit(&mut exec, 1);
    let mut ids: Vec<TokenId> = Vec::new();
    let out = run_prefill(&mut exec, 1, sampling(params.clone(), 0, choice));
    ids.extend(out.seqs[0].tokens.iter().copied());
    let mut context = PROMPT as u32;
    for step in 1..=steps as u64 {
        let last = vec![*ids.last().expect("a token")];
        let out = run_decode(
            &mut exec,
            1,
            &last,
            context,
            step,
            sampling(params.clone(), step, choice),
        );
        ids.extend(out.seqs[0].tokens.iter().copied());
        context += out.seqs[0].tokens.len() as u32;
    }
    (ids, exec.counters())
}

/// **The property the temperature-0 gates in this tree depend on.** Nothing below the argmax runs,
/// and the counter says so.
#[test]
fn temperature_zero_is_the_argmax_and_nothing_else_runs() {
    let (a, counters) = generate(&greedy_params(), 8, 0);
    let (b, _) = generate(&greedy_params(), 8, 0);
    assert_eq!(a, b, "a greedy generation is a function of its prompt");
    assert_eq!(counters.greedy_rows, 9, "the prefill's row and eight more");
    assert_eq!(counters.sampled_rows, 0);
    assert_eq!(counters.dense_rows, 0);
    assert_eq!(counters.seam_refusals, 0);
    assert_eq!(counters.sampler_fallbacks_to_argmax, 0);
    assert!(counters.clean());
}

/// A temperature makes the SEAM decide the token — reproducibly, at a fixed seed, and differently
/// from the argmax.
#[test]
fn a_seeded_request_draws_from_the_seam_and_repeats_itself() {
    let (greedy, _) = generate(&greedy_params(), 8, 0);
    let (a, counters) = generate(&prod_params(Some(42)), 8, 0);
    let (b, _) = generate(&prod_params(Some(42)), 8, 0);
    assert_eq!(a, b, "the same seed draws the same stream");
    assert_ne!(
        a, greedy,
        "prod's own defaults over a peaked row must not reproduce the argmax exactly"
    );
    assert_eq!(counters.sampled_rows, 9, "every row came from the seam");
    assert_eq!(counters.greedy_rows, 0);
    assert_eq!(
        counters.dense_rows, 0,
        "prod's defaults truncate well inside a 64-wide seam"
    );
    assert_eq!(counters.seam_refusals, 0);
    assert_eq!(counters.sampler_fallbacks_to_argmax, 0);
    assert!(counters.clean());
}

/// The noise is keyed by `(seed, position, choice)` (ws-f40), so the branches of one `n > 1`
/// request are different draws from the same distribution rather than `n` copies of one draw.
#[test]
fn two_branches_of_one_seeded_request_draw_differently() {
    let (first, _) = generate(&prod_params(Some(7)), 8, 0);
    let (second, _) = generate(&prod_params(Some(7)), 8, 1);
    assert_ne!(
        first, second,
        "two branches of one seeded request must not be the same stream"
    );
    // And the second branch is itself reproducible, which is what makes `n > 1` debuggable.
    let (again, _) = generate(&prod_params(Some(7)), 8, 1);
    assert_eq!(second, again);
}

/// A request with NO truncation samples from the whole vocabulary, and a 64-wide seam cannot serve
/// it. The refusal is by name, the dense row answers it, and both are counted.
#[test]
fn an_untruncated_request_is_refused_by_the_seam_and_read_densely() {
    let params = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(3),
        ..SamplingParams::default()
    };
    let (ids, counters) = generate(&params, 4, 0);
    assert_eq!(ids.len(), 5);
    assert_eq!(
        counters.seam_refusals, 5,
        "every row of an untruncated request is refused by the seam"
    );
    assert_eq!(counters.dense_rows, 5, "and every one is read densely");
    assert_eq!(counters.sampled_rows, 0);
    assert_eq!(
        counters.sampler_fallbacks_to_argmax, 0,
        "a refusal that fell back to the argmax would be serving a request greedily that asked \
         not to be"
    );
    assert!(counters.clean());
}

/// The seam's penalty gather is bounded, and a history that moves more ids than it carries is read
/// densely rather than sampled from a gather that would have missed one.
#[test]
fn a_penalty_list_wider_than_the_seam_is_read_densely() {
    let mut exec = executor();
    admit(&mut exec, 1);
    // A thousand distinct ids in the history, against a seam that carries 256.
    let history = Arc::new(PenaltyHistory {
        prompt: (0..1000u32).map(|i| 5000 + i).collect(),
        output: Vec::new(),
    });
    let params = SamplingParams {
        temperature: 0.7,
        top_p: 0.95,
        top_k: 20,
        repetition_penalty: 1.2,
        seed: Some(11),
        ..SamplingParams::default()
    };
    let out = run_prefill(
        &mut exec,
        1,
        SlotSampling {
            params,
            choice_index: 0,
            step_index: 0,
            position: 0,
            penalties: Some(history),
        },
    );
    assert_eq!(out.seqs[0].tokens.len(), 1);
    let counters = exec.counters();
    assert_eq!(
        counters.penalty_lists_too_wide, 1,
        "the list is wider than the seam and the row says so"
    );
    assert_eq!(counters.dense_rows, 1, "so the row is read densely");
    assert_eq!(counters.sampler_fallbacks_to_argmax, 0);
}

/// A penalised row whose list FITS is sampled through the seam, with the penalty logits gathered
/// on the device.
#[test]
fn a_penalised_row_inside_the_seam_is_sampled_through_it() {
    let mut exec = executor();
    admit(&mut exec, 1);
    let history = Arc::new(PenaltyHistory {
        prompt: prompt(),
        output: vec![7, 7, 9],
    });
    let params = SamplingParams {
        temperature: 0.7,
        top_p: 0.95,
        top_k: 20,
        frequency_penalty: 0.5,
        presence_penalty: 0.3,
        seed: Some(5),
        ..SamplingParams::default()
    };
    let out = run_prefill(
        &mut exec,
        1,
        SlotSampling {
            params,
            choice_index: 0,
            step_index: 0,
            position: 0,
            penalties: Some(history),
        },
    );
    assert_eq!(out.seqs[0].tokens.len(), 1);
    let counters = exec.counters();
    assert_eq!(counters.penalty_lists_too_wide, 0);
    assert_eq!(counters.sampled_rows, 1);
    assert_eq!(counters.dense_rows, 0);
    assert_eq!(counters.seam_refusals, 0);
}

/// `logprobs` come back for the committed token, in prod's schema: the sampled token first,
/// whether or not it is in the top-k.
#[test]
fn logprobs_come_back_for_the_committed_token() {
    let mut exec = executor();
    admit(&mut exec, 1);
    let params = SamplingParams {
        logprobs: Some(3),
        ..prod_params(Some(2))
    };
    let out = run_prefill(&mut exec, 1, sampling(params, 0, 0));
    let step = &out.seqs[0];
    let lp = step.logprobs.as_ref().expect("logprobs were asked for");
    assert_eq!(lp.len(), 1, "one entry per committed token");
    assert_eq!(
        lp[0].sampled.token_id, step.tokens[0],
        "the sampled entry is the token that was committed"
    );
    assert!(
        lp[0].sampled.logprob <= 0.0 && lp[0].sampled.logprob.is_finite(),
        "a logprob is a log probability"
    );
    assert!(
        lp[0].top.len() <= 3 && !lp[0].top.iter().any(|t| t.token_id == step.tokens[0]),
        "the sampled token is first and is not repeated in the top-k"
    );
    // The rank is 1-based and counts the token itself, so it exists and is at least 1 whenever the
    // seam was wide enough to count it — which at prod's defaults over this row it always is.
    assert!(
        lp[0].sampled.rank.is_some_and(|r| r >= 1),
        "a rank the seam could not count is reported as None, and this row is not that case"
    );
}

/// A greedy request asks for no logprobs and gets none — and, more to the point, it does not pay
/// for a seam it is not going to read.
#[test]
fn a_greedy_row_asks_the_head_for_nothing_beyond_the_argmax() {
    let mut exec = executor();
    admit(&mut exec, 1);
    let out = run_prefill(&mut exec, 1, sampling(greedy_params(), 0, 0));
    assert!(out.seqs[0].logprobs.is_none());
    let counters = exec.counters();
    assert_eq!(counters.greedy_rows, 1);
    assert_eq!(counters.sampled_rows, 0);
}

/// The dump is a DIAGNOSTIC, and a diagnostic whose format drifts is worse than none: the checker
/// that replays it (`saber sample-check`) parses these lines. One line per SAMPLED row, the
/// committed token on it, and nothing at all for a greedy row.
#[test]
fn the_sample_dump_writes_one_line_per_sampled_row() {
    let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target"))
        .join("saber-sampler-seam");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join(format!("dump-{}.jsonl", std::process::id()));

    let mut exec = executor();
    exec.dump_samples_to(&path).expect("the dump opens");
    admit(&mut exec, 1);
    let params = prod_params(Some(99));
    let out = run_prefill(&mut exec, 1, sampling(params.clone(), 0, 0));
    let first = out.seqs[0].tokens[0];
    let out = run_decode(
        &mut exec,
        1,
        &[first],
        PROMPT as u32,
        1,
        sampling(params, 1, 0),
    );
    let second = out.seqs[0].tokens[0];
    // A greedy row in the middle of the same executor writes nothing.
    let _ = run_decode(
        &mut exec,
        1,
        &[second],
        PROMPT as u32 + 1,
        2,
        sampling(greedy_params(), 2, 0),
    );
    drop(exec);

    let text = std::fs::read_to_string(&path).expect("the dump is there");
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(
        lines.len(),
        2,
        "two sampled rows, and the greedy one is not one"
    );
    assert!(
        lines[0].contains(&format!("\"token\":{first}")),
        "{}",
        lines[0]
    );
    assert!(
        lines[1].contains(&format!("\"token\":{second}")),
        "{}",
        lines[1]
    );
    for line in &lines {
        for field in [
            "\"seq\":",
            "\"choice\":",
            "\"step\":",
            "\"seed\":99",
            "\"temperature\":",
            "\"top_p\":",
            "\"top_k\":",
            "\"vocab\":",
            "\"path\":\"seam\"",
            "\"lse\":",
            "\"lse_t\":",
            "\"top_ids\":[",
            "\"top_values\":[",
        ] {
            assert!(line.contains(field), "{field} missing from {line}");
        }
    }
    std::fs::remove_file(&path).expect("clean up");
}

// ── logprobs for a verify step's accepted prefix (V3) ───────────────────────────────────────

/// A GREEDY request that asks for logprobs gets them. The token is still the argmax's — that is
/// what `greedy_rows` counts and what every temperature-0 gate in this tree runs — but `logprobs`
/// is a REPORT and not a sampling knob, and prod answers it for a greedy request too. Before V3
/// the seam was not asked for at all on a greedy row and the reply carried none.
#[test]
fn a_greedy_request_that_asks_for_logprobs_gets_them_and_the_argmax() {
    let params = SamplingParams {
        logprobs: Some(3),
        ..greedy_params()
    };
    let mut with = executor();
    admit(&mut with, 1);
    let out = run_prefill(&mut with, 1, sampling(params, 0, 0));
    let step = &out.seqs[0];
    let lp = step
        .logprobs
        .as_ref()
        .expect("a greedy request asked for logprobs");
    assert_eq!(lp.len(), 1);
    assert_eq!(lp[0].sampled.token_id, step.tokens[0]);
    let c = with.counters();
    assert_eq!(c.greedy_rows, 1, "the row is still greedy");
    assert_eq!(c.sampled_rows, 0, "and nothing drew from it");

    // The token is the one a request that asked for NO logprobs gets: reporting must not sample.
    let mut without = executor();
    admit(&mut without, 1);
    let plain = run_prefill(&mut without, 1, sampling(greedy_params(), 0, 0));
    assert_eq!(plain.seqs[0].tokens, step.tokens);
    assert!(plain.seqs[0].logprobs.is_none(), "none asked, none given");
}

/// The same generation at width 1 and at width 5 with an ORACLE drafter — `RecordedDrafter`
/// scripted with the width-1 run's own tokens, so every draft is accepted and a step commits five.
///
/// Two claims, and the second is the one that could not be made before V3:
///
/// 1. **logprobs come back for EVERY committed token**, not just the last. The seam kernel runs on
///    every verify row, so the accepted prefix has a distribution of its own; before this it was
///    counted away in `logprobs_skipped_on_verify` and the client got none at all;
/// 2. **they are the width-1 run's own numbers**, token for token and bit for bit. A verify step
///    is a batched re-derivation of the same rows, so a logprob that moved with the width would
///    mean the distribution a client is shown depends on how much the drafter guessed.
///
/// Greedy, because that is the only way the two runs are the same sequence: a verify step's
/// accepted prefix is compared against the model's ARGMAX, so a run that drew would diverge from
/// one that accepted, at the first token the draw did not agree with. The claim is about the
/// REPORT, and greedy is where the report is comparable.
#[test]
fn logprobs_come_back_for_every_committed_token_at_width_five() {
    let params = SamplingParams {
        logprobs: Some(3),
        ..greedy_params()
    };
    // ---- the width-1 run: the reference, one token and one entry per step -------------------
    let mut narrow = executor();
    admit(&mut narrow, 1);
    let first = run_prefill(&mut narrow, 1, sampling(params.clone(), 0, 0));
    let mut ids: Vec<TokenId> = first.seqs[0].tokens.clone();
    let mut want: Vec<saber_serve::engine::SampledLogprobs> = first.seqs[0]
        .logprobs
        .clone()
        .expect("the prefill reports logprobs");
    for step in 0..12u64 {
        let ctx = (PROMPT + ids.len() - 1) as u32;
        let out = run_decode(
            &mut narrow,
            1,
            &[*ids.last().expect("a token")],
            ctx,
            step,
            sampling(params.clone(), step + 1, 0),
        );
        let s = &out.seqs[0];
        assert_eq!(s.tokens.len(), 1, "width 1 commits one token a step");
        want.extend(s.logprobs.clone().expect("logprobs at width 1"));
        ids.extend_from_slice(&s.tokens);
    }

    // ---- the width-5 run, with that continuation as the drafter's script --------------------
    let mut oracle = saber_engine::RecordedDrafter::new(5, 5);
    oracle.record(SeqId(1), ids[1..].to_vec());
    // `b1s5` is a PLANNED shape of this stub, not the default `[1]`: a row count is not a bucket.
    let mut wide = RealExecutor::with_drafter(
        NullForward::with_widths(&[PREFILL], BATCHES, &[1, 5], 8),
        Box::new(oracle),
    );
    admit(&mut wide, 1);
    let out = run_prefill(&mut wide, 1, sampling(params.clone(), 0, 0));
    let mut got: Vec<saber_serve::engine::SampledLogprobs> = out.seqs[0]
        .logprobs
        .clone()
        .expect("logprobs at the prompt");
    let mut wide_ids: Vec<TokenId> = out.seqs[0].tokens.clone();
    assert_eq!(wide_ids, ids[..1], "the same prompt, the same first token");

    let mut steps = 0usize;
    let mut widest = 0usize;
    while wide_ids.len() < ids.len() {
        let ctx = (PROMPT + wide_ids.len() - 1) as u32;
        let slots = vec![BatchSlot {
            verify_width: 5,
            ..decode_slot(
                1,
                &[*wide_ids.last().expect("a token")],
                ctx,
                sampling(params.clone(), steps as u64 + 1, 0),
            )
        }];
        let out = wide.execute(&StepBatch {
            phase: Phase::Decode,
            bucket: Bucket::decode(1, 5, 8192, true),
            step_index: steps as u64,
            slots: &slots,
            padded_slots: 0,
        });
        let s = &out.seqs[0];
        widest = widest.max(s.tokens.len());
        let lp = s
            .logprobs
            .clone()
            .expect("logprobs for every committed token");
        assert_eq!(
            lp.len(),
            s.tokens.len(),
            "one logprob entry per COMMITTED token, which is the claim"
        );
        for (entry, token) in lp.iter().zip(&s.tokens) {
            assert_eq!(
                entry.sampled.token_id, *token,
                "the entry names the token it belongs to"
            );
        }
        got.extend(lp);
        wide_ids.extend_from_slice(&s.tokens);
        steps += 1;
        assert!(steps < 32, "the wide run is not advancing");
    }
    assert!(
        widest > 1,
        "the oracle drafter accepted nothing, so nothing above width 1 was tested"
    );
    assert!(steps < 12, "width 5 must take fewer steps than width 1 did");

    // ---- the same ids, and the same numbers for them ----------------------------------------
    assert_eq!(&wide_ids[..ids.len()], &ids[..], "the same ids");
    assert!(got.len() >= want.len(), "an entry per token, both runs");
    for (i, (a, b)) in want.iter().zip(&got).enumerate() {
        assert_eq!(a.sampled.token_id, b.sampled.token_id, "token {i}");
        assert_eq!(
            a.sampled.logprob.to_bits(),
            b.sampled.logprob.to_bits(),
            "token {i}: the logprob moved with the verify width"
        );
        assert_eq!(a.sampled.rank, b.sampled.rank, "token {i}: the rank moved");
        assert_eq!(
            a.top.len(),
            b.top.len(),
            "token {i}: the top-k length moved"
        );
        for (x, y) in a.top.iter().zip(&b.top) {
            assert_eq!(x.token_id, y.token_id, "token {i}: a top-k id moved");
            assert_eq!(
                x.logprob.to_bits(),
                y.logprob.to_bits(),
                "token {i}: a top-k logprob moved"
            );
        }
    }
    assert_eq!(
        wide.counters().logprobs_skipped_on_verify,
        0,
        "not one step declined to report its prefix"
    );
}
