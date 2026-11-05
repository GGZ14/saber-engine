//! What the scheduler owes the sampler, and what it does with what comes back.
//!
//! The scheduler chooses no token — the executor does — but it owns everything the choice is made
//! FROM: the request's parameters, which of the request's `n` branches a row is, how many steps
//! that branch has taken (the noise key), and the tokens the penalties read. This file gates that
//! those four reach the executor per ROW, and that the distribution the executor reports comes back
//! out as `StepEvent::logprobs` **cut to the tokens the stop rule actually committed**.
//!
//! A recording executor rather than a mock model: what is being tested is a hand-off, and the
//! cheapest true statement about a hand-off is what arrived.

mod common;

use std::sync::{Arc, Mutex};

use common::{drain, scheduler};
use saber_sched::{SeqStep, SlotSampling, StepBatch, StepExecutor, StepResult};
use saber_serve::engine::{Engine, EngineRequest, SampledLogprobs, TokenId, TokenLogprob};
use saber_serve::sampling::SamplingParams;

/// Records every slot's sampling request, and reports a distribution for every token it commits.
///
/// ONE token per step and never an EOS: the run is bounded by `max_tokens`, so the number of steps
/// is the test's to choose rather than the executor's.
#[derive(Debug, Default)]
struct Recorder {
    seen: Arc<Mutex<Vec<SlotSampling>>>,
    token: TokenId,
    logprobs: bool,
}

impl StepExecutor for Recorder {
    fn execute(&mut self, batch: &StepBatch<'_>) -> StepResult {
        let mut seqs = Vec::with_capacity(batch.slots.len());
        for slot in batch.slots {
            self.seen
                .lock()
                .expect("recorder")
                .push(slot.sampling.clone());
            let tokens: Vec<TokenId> = match slot.chunk {
                Some(c) if !c.last => Vec::new(),
                _ => vec![self.token],
            };
            let logprobs = self.logprobs.then(|| {
                tokens
                    .iter()
                    .map(|t| SampledLogprobs {
                        sampled: TokenLogprob {
                            token_id: *t,
                            logprob: -0.5,
                            rank: Some(1),
                        },
                        top: Vec::new(),
                    })
                    .collect()
            });
            seqs.push(SeqStep {
                seq_id: slot.seq_id,
                tokens: tokens.clone(),
                draft_tokens: 0,
                accepted_tokens: 0,
                logprobs,
                prompt_logprobs: None,
            });
        }
        StepResult { seqs }
    }
}

fn request(prompt: &[TokenId], params: SamplingParams) -> EngineRequest {
    EngineRequest {
        prompt_token_ids: prompt.to_vec(),
        params,
    }
}

#[tokio::test]
async fn a_requests_sampling_parameters_reach_every_row_of_its_step() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sched = scheduler(Recorder {
        seen: Arc::clone(&seen),
        token: 11,
        logprobs: false,
    });
    let params = SamplingParams {
        temperature: 0.7,
        top_p: 0.95,
        top_k: 20,
        seed: Some(4242),
        max_tokens: Some(8),
        // Penalties bind, so the history has to come with the row.
        frequency_penalty: 0.5,
        ..SamplingParams::default()
    };
    let sub = sched.submit(request(&[1, 2, 3], params)).expect("submit");
    sched.run_steps(8);
    let _ = drain(sub).await;

    let seen = seen.lock().expect("recorder").clone();
    assert!(!seen.is_empty(), "the step loop ran");
    let first = &seen[0];
    assert_eq!(first.params.temperature, 0.7);
    assert_eq!(first.params.top_p, 0.95);
    assert_eq!(first.params.top_k, 20);
    assert_eq!(first.params.seed, Some(4242));
    assert_eq!(first.choice_index, 0);
    assert!(!first.is_greedy());
    let penalties = first
        .penalties
        .as_ref()
        .expect("a penalised row carries the tokens its penalties read");
    assert_eq!(penalties.prompt, vec![1, 2, 3]);
    assert!(
        penalties.output.is_empty(),
        "nothing has been generated at the prefill step"
    );
    // The noise key advances with the sequence's own steps, so two steps of one request draw
    // differently and a re-run of the same request draws the same.
    let steps: Vec<u64> = seen.iter().map(|s| s.step_index).collect();
    assert!(
        steps.windows(2).any(|w| w[1] > w[0]),
        "the step index advances: {steps:?}"
    );
    // And the history grows with what was committed.
    let last = seen.last().expect("a row");
    let last_penalties = last.penalties.as_ref().expect("still penalised");
    assert!(
        !last_penalties.output.is_empty(),
        "by the second step the output is part of what the penalties read"
    );
}

#[tokio::test]
async fn a_greedy_request_says_so_on_every_row() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sched = scheduler(Recorder {
        seen: Arc::clone(&seen),
        token: 11,
        logprobs: false,
    });
    let params = SamplingParams {
        temperature: 0.0,
        max_tokens: Some(4),
        ..SamplingParams::default()
    };
    let sub = sched.submit(request(&[1, 2, 3], params)).expect("submit");
    sched.run_steps(6);
    let _ = drain(sub).await;
    let seen = seen.lock().expect("recorder").clone();
    assert!(seen.iter().all(SlotSampling::is_greedy));
    assert!(
        seen.iter().all(|s| s.penalties.is_none()),
        "no penalties, no history: the one part of a row whose size is the sequence's length"
    );
}

#[tokio::test]
async fn the_distribution_comes_back_cut_to_the_tokens_that_committed() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sched = scheduler(Recorder {
        seen: Arc::clone(&seen),
        token: 11,
        logprobs: true,
    });
    let params = SamplingParams {
        temperature: 0.7,
        logprobs: Some(2),
        max_tokens: Some(4),
        ..SamplingParams::default()
    };
    let sub = sched.submit(request(&[1, 2, 3], params)).expect("submit");
    sched.run_steps(8);
    let events = drain(sub).await;

    for e in &events {
        let lp = e
            .logprobs
            .as_ref()
            .expect("the executor reported a distribution for every step");
        assert_eq!(
            lp.len(),
            e.new_tokens.len(),
            "one entry per COMMITTED token, not per token the executor produced"
        );
        for (entry, token) in lp.iter().zip(&e.new_tokens) {
            assert_eq!(entry.sampled.token_id, *token);
        }
    }
    let stopped = events.last().expect("an event");
    assert_eq!(
        stopped.finished,
        Some(saber_serve::engine::FinishReason::Length),
        "the budget is what ends this stream"
    );
    assert_eq!(
        events.iter().map(|e| e.new_tokens.len()).sum::<usize>(),
        4,
        "max_tokens is the bound and every one of those tokens has a distribution"
    );
}
