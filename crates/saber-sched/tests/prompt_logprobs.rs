//! `prompt_logprobs` through the scheduler (V7): one answer per prompt token, assembled across
//! prefill chunks and delivered once, with the first token — or a loud refusal, never a `null`.
//!
//! Before V7 the scheduler hardcoded `prompt_logprobs: None` on every event, so a request that
//! asked got HTTP 200 with `"prompt_logprobs": null` from the real engine and `ppl.py` died on a
//! `TypeError` (Q2). The contract now:
//!
//! * a request that asks gets `[None, p_1, .., p_{n-1}]` on the event that carries its first
//!   token, `p_i` keyed first by prompt token `i`, however many chunks the prompt took;
//! * a request that does not ask sees exactly what it saw before — no field, the same tokens and
//!   the same batches;
//! * an executor that cannot produce them, a width it cannot produce, a plan with no prefill
//!   bucket, or a chunk the executor did not score, is refused by name.

mod common;

use std::sync::Arc;

use common::{boot_ledger, decode_buckets, plan_with, plan_with_prefill, GIB};
use saber_sched::{
    AdmittedSeq, ArenaBytes, KvGeometry, Scheduler, SchedulerConfig, ScriptExecutor, SeqId,
    StepBatch, StepExecutor, StepResult,
};
use saber_serve::detok::{ByteCodec, TextCodec};
use saber_serve::engine::{Engine, EngineError, EngineRequest, StepEvent, TokenId};
use saber_serve::sampling::SamplingParams;

const EOS: TokenId = 255;
const PREFILL: [(u32, u32); 2] = [(2048, 65536), (8192, 65536)];

fn answer() -> Vec<TokenId> {
    let mut v = ByteCodec.encode("after the prompt").expect("encode");
    v.push(EOS);
    v
}

fn config(prefill: &[(u32, u32)]) -> SchedulerConfig {
    let decode: Vec<(u32, u32, u32)> = decode_buckets()
        .into_iter()
        .filter(|(_, _, band)| *band == 65536)
        .collect();
    let plan = if prefill.is_empty() {
        plan_with(&decode, 2)
    } else {
        plan_with_prefill(&decode, prefill, 2)
    };
    SchedulerConfig::from_plan(
        &plan,
        KvGeometry::PROD_FP8,
        ArenaBytes {
            kv_per_rank: 8 * GIB,
            gdn_state_per_rank: 8 * GIB,
        },
        262_144,
    )
    .expect("config")
}

fn sched<E: StepExecutor>(prefill: &[(u32, u32)], exec: E) -> Arc<Scheduler<E>> {
    let config = config(prefill);
    let model =
        saber_sched::mock_model_info("Qwen3.8", "/models/mock", EOS, config.max_context_tokens);
    Arc::new(Scheduler::new(config, exec, model, boot_ledger()).expect("arenas fit"))
}

fn script() -> ScriptExecutor {
    ScriptExecutor::new(vec![answer()], 5, EOS)
}

fn prompt(n: usize) -> Vec<TokenId> {
    (0..n).map(|i| (i % 200) as TokenId + 1).collect()
}

fn request(n: usize, prompt_logprobs: Option<i32>) -> EngineRequest {
    EngineRequest {
        prompt_token_ids: prompt(n),
        params: SamplingParams {
            max_tokens: Some(8),
            prompt_logprobs,
            ..SamplingParams::default()
        },
    }
}

async fn stream(submission: saber_serve::engine::Submission) -> Vec<StepEvent> {
    use futures_util::StreamExt;
    submission.events.collect().await
}

fn tokens_of(events: &[StepEvent]) -> Vec<TokenId> {
    events.iter().flat_map(|e| e.new_tokens.clone()).collect()
}

#[tokio::test]
async fn a_chunked_prompt_gets_one_answer_per_token_on_its_first_event() {
    // 12000 tokens is two chunks (8192 + 3808): the answer is assembled across both, and the row
    // that ends the first chunk is scored against the first token of the second.
    let s = sched(&PREFILL, script());
    let sub = s.submit(request(12_000, Some(3))).expect("submit");
    s.run_steps(400);
    let events = stream(sub).await;
    let carrying: Vec<&StepEvent> = events
        .iter()
        .filter(|e| e.prompt_logprobs.is_some())
        .collect();
    assert_eq!(carrying.len(), 1, "delivered once");
    assert_eq!(carrying[0].step_index, 0, "with the first token");
    let plp = carrying[0].prompt_logprobs.as_ref().unwrap();
    let p = prompt(12_000);
    assert_eq!(plp.len(), p.len(), "one entry per prompt token");
    assert!(plp[0].is_none(), "position 0 has no distribution");
    for (i, pos) in plp.iter().enumerate().skip(1) {
        let entries = pos
            .as_ref()
            .unwrap_or_else(|| panic!("position {i} is empty"));
        assert_eq!(
            entries[0].token_id, p[i],
            "position {i}: the prompt token comes first"
        );
        assert!(
            entries.len() == 3 || entries.len() == 4,
            "k or k + 1 entries"
        );
    }
    // The chunk boundary: position 8192 was scored by the FIRST chunk's last row.
    assert_eq!(plp[8192].as_ref().unwrap()[0].token_id, p[8192]);
    let chunks: Vec<_> = s.with_executor(|e| {
        e.batches()
            .iter()
            .flat_map(|b| b.chunks.clone())
            .flatten()
            .collect()
    });
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].next_prompt_token, Some(p[8192]));
    assert_eq!(chunks[1].next_prompt_token, None);
}

#[tokio::test]
async fn a_request_that_does_not_ask_sees_what_it_saw_before() {
    let with = sched(&PREFILL, script());
    let sub = with.submit(request(3000, Some(1))).expect("submit");
    with.run_steps(200);
    let a = stream(sub).await;

    let without = sched(&PREFILL, script());
    let sub = without.submit(request(3000, None)).expect("submit");
    without.run_steps(200);
    let b = stream(sub).await;

    assert!(b.iter().all(|e| e.prompt_logprobs.is_none()), "no field");
    assert_eq!(tokens_of(&a), tokens_of(&b), "the same tokens");
    let shapes = |s: &Arc<Scheduler<ScriptExecutor>>| {
        s.with_executor(|e| {
            e.batches()
                .iter()
                .map(|b| (b.phase, b.bucket, b.sequences))
                .collect::<Vec<_>>()
        })
    };
    assert_eq!(shapes(&with), shapes(&without), "the same steps");
}

/// An executor that forwards everything to a script and says what it is told to about prompt
/// logprobs — the two refusals' negative controls.
#[derive(Debug)]
struct Wrap {
    inner: ScriptExecutor,
    max: Option<u32>,
    /// Drop the prompt logprobs the inner executor produced: the protocol fault.
    drop_rows: bool,
}

impl StepExecutor for Wrap {
    fn execute(&mut self, batch: &StepBatch<'_>) -> StepResult {
        let mut r = self.inner.execute(batch);
        if self.drop_rows {
            for s in &mut r.seqs {
                s.prompt_logprobs = None;
            }
        }
        r
    }
    fn admit(&mut self, seq: &AdmittedSeq) {
        self.inner.admit(seq);
    }
    fn release(&mut self, seq: SeqId) {
        self.inner.release(seq);
    }
    fn max_prompt_logprobs(&self) -> Option<u32> {
        self.max
    }
}

fn wrap(max: Option<u32>, drop_rows: bool) -> Wrap {
    Wrap {
        inner: script(),
        max,
        drop_rows,
    }
}

fn unsupported(r: Result<saber_serve::engine::Submission, EngineError>) -> (String, String) {
    match r {
        Err(EngineError::Unsupported { param, message }) => (param.to_owned(), message),
        Err(e) => panic!("refused, but not as Unsupported: {e}"),
        Ok(_) => panic!("served"),
    }
}

#[test]
fn an_engine_that_cannot_is_a_400_never_a_null() {
    let s = sched(&PREFILL, wrap(None, false));
    let (param, message) = unsupported(s.submit(request(100, Some(1))));
    assert_eq!(param, "prompt_logprobs");
    assert!(
        message.contains("does not produce prompt logprobs"),
        "{message}"
    );
    // The HTTP status a client sees.
    let api: saber_serve::protocol::ApiError = EngineError::Unsupported {
        param: "prompt_logprobs",
        message,
    }
    .into();
    assert_eq!(api.body.code, 400);
    // The same engine serves a request that does not ask.
    assert!(s.submit(request(100, None)).is_ok());
}

#[test]
fn wider_than_the_seam_or_the_whole_vocabulary_is_refused() {
    let s = sched(&PREFILL, wrap(Some(64), false));
    let (_, m) = unsupported(s.submit(request(100, Some(65))));
    assert!(m.contains("at most 64"), "{m}");
    let (_, m) = unsupported(s.submit(request(100, Some(-1))));
    assert!(m.contains("at most 64"), "{m}");
    assert!(s.submit(request(100, Some(64))).is_ok());
    assert!(
        s.submit(request(100, Some(0))).is_ok(),
        "0 is the target alone"
    );
}

#[test]
fn a_plan_with_no_prefill_bucket_is_refused() {
    let s = sched(&[], script());
    let (_, m) = unsupported(s.submit(request(100, Some(1))));
    assert!(m.contains("no prefill bucket"), "{m}");
}

/// NEGATIVE CONTROL: an executor that ran the chunk and did not score it. The request ends with
/// an ERROR naming the reason — the 200-with-`null` Q2 found is exactly what must not come back.
#[tokio::test]
async fn a_chunk_the_executor_did_not_score_ends_the_request_loudly() {
    let s = sched(&PREFILL, wrap(Some(64), true));
    let sub = s.submit(request(100, Some(1))).expect("submit");
    s.run_steps(50);
    let events = stream(sub).await;
    assert!(events.iter().all(|e| e.prompt_logprobs.is_none()));
    let last = events.last().expect("an event");
    assert_eq!(
        last.finished,
        Some(saber_serve::engine::FinishReason::Abort)
    );
    let error = last
        .error
        .as_deref()
        .expect("an error, not a silent finish");
    assert!(error.contains("prompt_logprobs"), "{error}");
    assert!(tokens_of(&events).is_empty(), "no token was served");
}

/// `n > 1` on an executor that cannot branch a group is a 400 naming `n` — it was a 500.
#[test]
fn n_above_one_without_a_fork_is_a_400_naming_n() {
    let s = sched(&PREFILL, wrap(Some(64), false));
    let mut r = request(100, None);
    r.params.n = 2;
    r.params.temperature = 0.7;
    let (param, message) = unsupported(s.submit(r));
    assert_eq!(param, "n");
    assert!(message.contains("(parameter=n, value=2)"), "{message}");
}

/// A group wider than the batch could never be admitted whole — its branches fork together at
/// the leader's last prefill chunk — so on an executor that CAN fork it is a 400 naming `n`, in
/// The reference server's own words for an `n` above its limit (ws-f38; it was a 500). The widest group that fits
/// is served, which is the control.
#[test]
fn n_above_the_batch_is_a_400_in_the_reference_words_and_the_widest_group_is_served() {
    let s = sched(&PREFILL, script());
    let max = s.with_executor(|e| e.can_fork());
    assert!(max, "the script executor forks");
    let widest = config(&PREFILL).max_batch() as usize;
    let mut r = request(100, None);
    r.params.n = widest + 1;
    r.params.temperature = 0.7;
    let (param, message) = unsupported(s.submit(r));
    assert_eq!(param, "n");
    assert!(
        message.starts_with(&format!("n must be at most {widest}, got {}.", widest + 1)),
        "{message}"
    );
    let mut ok = request(100, None);
    ok.params.n = widest;
    ok.params.temperature = 0.7;
    assert!(
        s.submit(ok).is_ok(),
        "a group as wide as the batch is served"
    );
}
