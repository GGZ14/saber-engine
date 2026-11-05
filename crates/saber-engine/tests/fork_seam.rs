//! `n > 1`: the fork, on the host, with the device taken out (workstream V4).
//!
//! A sequence group is one prompt and `n` branches. The prefill runs **once**, on the leader, and
//! at the moment its last chunk samples every other branch is created from it:
//!
//! 1. the prompt's whole KV pages are the leader's and are shared **read-only for ever** — each
//!    branch appends only into pages it owns, so there is no copy-on-write after the fork;
//! 2. the one **partial** page is copied, because it is the one page two branches would both
//!    write into;
//! 3. the leader's **GDN state slot set** is copied device to device, because recurrent state is
//!    not addressable per token and cannot be shared the way a page can;
//! 4. the branch's **first token is a second draw from the leader's own prefill row**, with the
//!    branch's own noise key `(seed, position, choice)` (ws-f40).
//!
//! What this file gates is (1)–(4) as *laws*, with `NullForward` as the device: its session state
//! is one hash, so a branch that did not get a copy of the leader's produces a different stream
//! and the comparison sees it; its logit row is synthetic but has the shape of one, so the draw,
//! the key and the refusals are all exercised with no card. What it cannot gate is the bytes:
//! `saber d2d-gate` and `tools/gates/fork_gates.sh` do that on a card.
//!
//! Every test here reads the counters, because "the right answer by the wrong path" is exactly how
//! a fork fails: a branch that quietly re-prefilled, or one answered from the argmax when it asked
//! to be drawn, produces perfectly plausible tokens.

mod common;

use std::collections::BTreeMap;

use common::{drain, prompt_ids, PREFILL};
use saber_engine::{NullForward, RealExecutor};
use saber_plan::{Bucket, Phase};
use saber_sched::{
    AdmittedSeq, BatchSlot, PrefillChunk, SeqId, SlotSampling, StepBatch, StepExecutor,
};
use saber_serve::engine::{Engine, EngineRequest, RequestId, StepEvent, TokenId};
use saber_serve::sampling::SamplingParams;

const PROMPT: usize = 40;
const BLOCK: usize = 16;

/// Prod's own serving defaults, which is what an unparameterised request is sampled at.
fn prod_params(seed: Option<u64>, n: usize) -> SamplingParams {
    SamplingParams {
        temperature: 0.7,
        top_p: 0.95,
        top_k: 20,
        seed,
        n,
        ..SamplingParams::default()
    }
}

fn greedy_params(n: usize) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        n,
        ..SamplingParams::default()
    }
}

// ── the executor seam, driven directly ──────────────────────────────────────────────────────────

fn executor() -> RealExecutor<NullForward> {
    RealExecutor::new(NullForward::new(&[PREFILL], &[1, 2, 4], 8))
}

/// The leader's pages: `PROMPT` tokens is two whole 16-token blocks and part of a third.
fn leader_blocks() -> Vec<u32> {
    (0..8).collect()
}

/// A branch's: the leader's two WHOLE pages, then its own.
fn branch_blocks() -> Vec<u32> {
    let mut v: Vec<u32> = leader_blocks()[..2].to_vec();
    v.extend(20..26);
    v
}

fn admitted(seq: u64, choice: u32, blocks: Vec<u32>, shared: u32, cow: Option<u32>) -> AdmittedSeq {
    AdmittedSeq {
        request_id: RequestId(1),
        choice_index: choice,
        seq_id: SeqId(seq),
        prompt_tokens: prompt_ids(PROMPT),
        kv_blocks: blocks,
        shared_prefix_blocks: shared,
        cow_source: cow,
        gdn_set: seq as u32,
        gdn_slots: 2,
    }
}

/// Feed the leader its whole prompt and return the token it sampled.
fn prefill_leader(exec: &mut RealExecutor<NullForward>, params: &SamplingParams) -> TokenId {
    let sampling = SlotSampling {
        params: params.clone(),
        choice_index: 0,
        step_index: 0,
        position: 0,
        penalties: None,
    };
    let slots = vec![BatchSlot {
        request_id: RequestId(1),
        seq_id: SeqId(1),
        slot: 0,
        kv_blocks: leader_blocks(),
        gdn_set: 1,
        verify_width: 0,
        input_tokens: prompt_ids(PROMPT),
        context_len: 0,
        first_step: true,
        chunk: Some(PrefillChunk {
            prompt_tokens: PROMPT as u32,
            tokens_done_before: 0,
            last: true,
            next_prompt_token: None,
        }),
        sampling,
    }];
    let out = exec.execute(&StepBatch {
        phase: Phase::Prefill,
        bucket: Bucket::prefill(PREFILL, 8192, true),
        step_index: 0,
        slots: &slots,
        padded_slots: 0,
    });
    out.seqs[0].tokens[0]
}

/// **The fork law's fourth part, on its own**: a branch's first token is the draw the leader's own
/// row gives at that branch's key, and nothing else about the branch is in it.
///
/// The control is the leader's own prefill re-run AT THAT CHOICE INDEX. It is the same row and the
/// same parameters, so if the two agree, the only thing the fork put into the key is
/// `choice_index` — which is exactly the claim. A fork that keyed by the branch's sequence id, or
/// that forgot the choice and re-used the leader's key, fails it in one direction or the other.
#[test]
fn a_branch_draws_the_leaders_row_at_its_own_key() {
    let params = prod_params(Some(90_210), 4);
    let mut exec = executor();
    exec.admit(&admitted(1, 0, leader_blocks(), 0, None));
    let leader_token = prefill_leader(&mut exec, &params);

    let mut forked: Vec<TokenId> = Vec::new();
    for choice in 1..4u32 {
        let branch = admitted(
            10 + u64::from(choice),
            choice,
            branch_blocks(),
            2,
            Some(leader_blocks()[2]),
        );
        let t = exec
            .fork(SeqId(1), &branch)
            .unwrap_or_else(|| panic!("choice {choice} was refused"))
            .token;
        forked.push(t);
    }

    // The control: the same row, drawn at each choice, by a prefill of its own.
    let mut control: Vec<TokenId> = Vec::new();
    for choice in 1..4u32 {
        let mut solo = executor();
        solo.admit(&admitted(1, choice, leader_blocks(), 0, None));
        let sampling = SlotSampling {
            params: params.clone(),
            choice_index: choice,
            step_index: 0,
            position: 0,
            penalties: None,
        };
        let slots = vec![BatchSlot {
            request_id: RequestId(1),
            seq_id: SeqId(1),
            slot: 0,
            kv_blocks: leader_blocks(),
            gdn_set: 1,
            verify_width: 0,
            input_tokens: prompt_ids(PROMPT),
            context_len: 0,
            first_step: true,
            chunk: Some(PrefillChunk {
                prompt_tokens: PROMPT as u32,
                tokens_done_before: 0,
                last: true,
                next_prompt_token: None,
            }),
            sampling,
        }];
        let out = solo.execute(&StepBatch {
            phase: Phase::Prefill,
            bucket: Bucket::prefill(PREFILL, 8192, true),
            step_index: 0,
            slots: &slots,
            padded_slots: 0,
        });
        control.push(out.seqs[0].tokens[0]);
    }
    assert_eq!(
        forked, control,
        "each branch's first token must be the LEADER's row drawn at that branch's key"
    );

    let counters = exec.counters();
    assert_eq!(counters.forks, 3, "three branches forked");
    assert_eq!(counters.fork_refusals, 0);
    assert_eq!(
        counters.sampled_rows, 4,
        "the leader's row and one draw per branch, all from the seam"
    );
    assert_eq!(counters.dense_rows, 0);
    assert_eq!(counters.greedy_rows, 0);
    assert!(counters.clean(), "{}", counters.line());

    // And the branches are three different draws of one distribution, not three copies of the
    // leader's token. (A NEGATIVE CONTROL that is allowed to be weak: two draws of a peaked row
    // may legitimately agree, so this asserts that they are not ALL the leader's.)
    assert!(
        forked.iter().any(|t| *t != leader_token),
        "every branch reproduced the leader's token: {forked:?} against {leader_token}"
    );
}

/// **The control: at temperature 0 a group is one answer `n` times.** The kernel skips a greedy
/// row, so there is no row to draw from and the branch takes the leader's own argmax — which is
/// why `n = 2` greedy must return two identical branches, and why `greedy_rows` counts them.
#[test]
fn a_greedy_fork_is_the_leaders_own_argmax() {
    let params = greedy_params(2);
    let mut exec = executor();
    exec.admit(&admitted(1, 0, leader_blocks(), 0, None));
    let leader_token = prefill_leader(&mut exec, &params);
    let branch = admitted(11, 1, branch_blocks(), 2, Some(leader_blocks()[2]));
    let t = exec.fork(SeqId(1), &branch).expect("the greedy fork").token;
    assert_eq!(
        t, leader_token,
        "temperature 0 is the argmax on every branch"
    );
    let counters = exec.counters();
    assert_eq!(counters.forks, 1);
    assert_eq!(counters.sampled_rows, 0, "nothing below the argmax ran");
    assert_eq!(counters.dense_rows, 0);
    assert!(counters.clean(), "{}", counters.line());
}

/// The refusals, each by name and each leaving the engine as it was.
#[test]
fn a_fork_refuses_what_it_cannot_do_and_the_branch_is_not_admitted() {
    let params = prod_params(Some(7), 2);
    let mut exec = executor();
    exec.admit(&admitted(1, 0, leader_blocks(), 0, None));
    let _ = prefill_leader(&mut exec, &params);

    // A leader nobody kept a row for.
    assert!(
        exec.fork(SeqId(99), &admitted(21, 1, branch_blocks(), 2, None))
            .is_none(),
        "a fork from a sequence that never sampled"
    );
    // A shared prefix that is not the leader's pages: the pages a branch shares are the leader's
    // BY REFERENCE, and a branch that named others would read a prompt nothing wrote.
    let wrong: Vec<u32> = (30..38).collect();
    assert!(
        exec.fork(SeqId(1), &admitted(22, 1, wrong, 2, None))
            .is_none(),
        "a shared prefix that is not the leader's own pages"
    );
    // A copy-on-write source the leader does not hold at that position.
    assert!(
        exec.fork(SeqId(1), &admitted(23, 1, branch_blocks(), 2, Some(7)))
            .is_none(),
        "a copy-on-write source that is not the leader's partial page"
    );
    let counters = exec.counters();
    assert_eq!(counters.forks, 0);
    assert_eq!(counters.fork_refusals, 3);
    assert!(
        !counters.clean(),
        "a refused fork is a choice the client will not get, and that is not a clean run"
    );
}

// ── the whole group, through the scheduler ──────────────────────────────────────────────────────

fn group_request(prompt: &[TokenId], params: SamplingParams, max_tokens: usize) -> EngineRequest {
    EngineRequest {
        prompt_token_ids: prompt.to_vec(),
        params: SamplingParams {
            max_tokens: Some(max_tokens),
            ..params
        },
    }
}

/// Every choice's token stream, keyed by `choice_index`, with the per-choice step indices checked.
fn choices(events: &[StepEvent]) -> BTreeMap<u32, Vec<TokenId>> {
    let mut out: BTreeMap<u32, Vec<TokenId>> = BTreeMap::new();
    let mut steps: BTreeMap<u32, u32> = BTreeMap::new();
    for e in events {
        out.entry(e.choice_index)
            .or_default()
            .extend(e.new_tokens.iter().copied());
        let next = steps.entry(e.choice_index).or_default();
        assert_eq!(e.step_index, *next, "choice {} step index", e.choice_index);
        *next += 1;
    }
    out
}

/// **The prefill runs once.** `n = 4` over a prompt that takes one chunk is ONE prefill step, and
/// the branches' first tokens arrive in that same engine step — because they are draws from the
/// row it produced, not steps of their own.
///
/// The counters are the proof, not the token count: `prefill_steps` is the engine's own, and a
/// branch that quietly re-prefilled would raise it to four while producing exactly the same text.
#[tokio::test]
async fn a_group_prefills_once_and_its_branches_are_four_draws_of_one_row() {
    let sched = common::scheduler_no_park();
    let prompt = prompt_ids(PROMPT);
    let sub = sched
        .submit(group_request(&prompt, prod_params(Some(4242), 4), 24))
        .expect("submit");
    sched.run_steps(400);
    let events = drain(sub).await;
    let by_choice = choices(&events);
    assert_eq!(by_choice.len(), 4, "four choices");
    for (choice, ids) in &by_choice {
        assert!(!ids.is_empty(), "choice {choice} produced nothing");
    }

    let c = sched.counters();
    assert_eq!(
        (
            c.forks_pending,
            c.forks,
            c.fork_failures,
            c.fork_branches_orphaned
        ),
        (3, 3, 0, 0),
        "three branches, deferred and then forked: {c:?}"
    );
    let e = sched.with_executor(|e| e.counters());
    assert_eq!(
        e.prefill_steps,
        1,
        "ONE prefill for four branches: {}",
        e.line()
    );
    assert_eq!(e.forks, 3);
    assert_eq!(e.fork_refusals, 0);
    assert!(e.clean(), "{}", e.line());

    // The four streams are four draws, so at least two of them differ. (Weak on purpose: two
    // draws of a peaked row may agree, and a gate that demanded four distinct streams would be
    // asserting a property of the noise rather than of the fork.)
    let streams: Vec<&Vec<TokenId>> = by_choice.values().collect();
    assert!(
        streams.iter().any(|s| **s != *streams[0]),
        "every branch produced the same text: the choice index is not in the key"
    );
}

/// **A seeded group is reproducible, branch by branch.** Two submissions of the same request give
/// the same `n` streams in the same order — which is what makes `seed` mean anything at `n > 1`.
#[tokio::test]
async fn a_seeded_group_repeats_itself_branch_for_branch() {
    let prompt = prompt_ids(PROMPT);
    let mut runs = Vec::new();
    for _ in 0..2 {
        let sched = common::scheduler_no_park();
        let sub = sched
            .submit(group_request(&prompt, prod_params(Some(90_210), 2), 16))
            .expect("submit");
        sched.run_steps(400);
        runs.push(choices(&drain(sub).await));
    }
    assert_eq!(runs[0], runs[1], "the same seed gives the same n streams");
}

/// **The control: `n = 2` at temperature 0 is two identical greedy branches.**
///
/// It is run through the scheduler rather than over HTTP on purpose: prod's `_verify_greedy_sampling`
/// refuses `n > 1` with temperature 0 at the API (`n must be 1 when using greedy sampling`), and
/// `validate()` reproduces that — so this control can only be asked of the engine underneath it.
#[tokio::test]
async fn a_greedy_group_is_one_answer_twice() {
    let sched = common::scheduler_no_park();
    let prompt = prompt_ids(PROMPT);
    let sub = sched
        .submit(group_request(&prompt, greedy_params(2), 16))
        .expect("submit");
    sched.run_steps(400);
    let by_choice = choices(&drain(sub).await);
    assert_eq!(by_choice.len(), 2);
    assert_eq!(
        by_choice[&0], by_choice[&1],
        "temperature 0 is the argmax, and two branches of one prompt are one answer twice"
    );
    assert!(!by_choice[&0].is_empty());

    // And it is the answer the same prompt gets with `n = 1`: a group must not move the greedy
    // stream at all.
    let solo = common::scheduler_no_park();
    let sub = solo
        .submit(group_request(&prompt, greedy_params(1), 16))
        .expect("submit");
    solo.run_steps(400);
    let alone = choices(&drain(sub).await);
    assert_eq!(
        by_choice[&0], alone[&0],
        "a greedy branch is the same stream the prompt gets alone"
    );
}

/// **The KV census: the prompt is billed ONCE and each branch's tail is billed to it.**
///
/// `arena_free()` is the allocator's own count, which is what admission asks — the same number
/// K13's capacity census reads. A group of three over a 40-token prompt holds two shared pages
/// plus three tails; three separate requests would hold three copies of the prompt.
#[tokio::test]
async fn the_group_bills_the_shared_pages_once_and_each_branch_its_own() {
    let sched = common::scheduler_no_park();
    let free_before = sched.arena_free();
    let prompt = prompt_ids(PROMPT);
    let whole_blocks = (PROMPT / BLOCK) as u32;
    let sub = sched
        .submit(group_request(&prompt, greedy_params(3), 48))
        .expect("submit");
    // Admission, the leader's prefill and the fork.
    sched.run_steps(2);
    let (kv_now, state_now) = sched.arena_free();
    let kv_used = free_before.0 - kv_now;
    let state_used = free_before.1 - state_now;
    assert_eq!(state_used, 3, "one GDN slot set per branch, always");
    // A sequence reserves `prompt + max_tokens` rounded up to whole blocks; a branch references
    // the prompt's WHOLE blocks and pays for the rest. So the arithmetic is exact and the saving
    // is `(n - 1) * shared`, which is the whole of what a group buys in KV.
    let per_seq = ((PROMPT + 48).div_ceil(BLOCK)) as u32;
    let alone = 3 * per_seq;
    assert_eq!(
        kv_used,
        alone - 2 * whole_blocks,
        "three branches over a {whole_blocks}-block prompt: {per_seq} blocks each alone \
         ({alone}), less the {whole_blocks} shared pages the two branches do not pay for"
    );
    assert!(
        kv_used < alone,
        "{kv_used} blocks against {alone} for three single submits: the prompt is not being shared"
    );
    sched.run_steps(400);
    let _ = drain(sub).await;
    assert_eq!(
        sched.arena_free(),
        free_before,
        "every shared block came back once the last branch released it"
    );
}

/// **The use-after-free control.** One branch finishes and its tail pages go back to the free
/// list; a NEW session is then admitted into them and runs. The surviving branch's ids must not
/// move — which is the property a shared prefix freed too early, or a partial page copied by
/// reference instead of by value, breaks.
///
/// It is an ids assertion, and the control that gives it teeth is the pair of runs: the same group
/// with and without the intruder. A run that only checked "the survivor produced something" would
/// pass with the survivor reading the intruder's pages.
#[tokio::test]
async fn a_released_branchs_pages_are_reused_without_moving_the_survivors_tokens() {
    let prompt = prompt_ids(PROMPT);
    // The two branches ask for different lengths, so one of them finishes first and hands its
    // tail back while the other is still decoding.
    let short = 4usize;
    let long = 40usize;

    // A: the group alone.
    let a = {
        let sched = common::scheduler_no_park();
        let sub = sched
            .submit(group_request(&prompt, greedy_params(2), long))
            .expect("submit");
        sched.run_steps(600);
        choices(&drain(sub).await)
    };

    // B: the same group, with a second request admitted into the pages the first branch gives
    // back. The intruder's prompt is DIFFERENT, so anything it writes into a reused page is
    // visible in the survivor's stream if the pages are not really private.
    let b = {
        let sched = common::scheduler_no_park();
        let sub = sched
            .submit(group_request(&prompt, greedy_params(2), long))
            .expect("submit");
        // Let the group get going, then stop one branch early by aborting nothing — instead admit
        // an intruder once some pages are back. Running a short request alongside is what frees
        // pages mid-flight.
        sched.run_steps(6);
        let intruder = sched
            .submit(group_request(
                &prompt_ids(PROMPT + 3),
                greedy_params(1),
                short,
            ))
            .expect("intruder");
        sched.run_steps(600);
        let _ = drain(intruder).await;
        choices(&drain(sub).await)
    };

    assert_eq!(
        a, b,
        "another session running in the pages this group freed changed its tokens"
    );
    assert_eq!(a.len(), 2);
    assert!(a[&0].len() > short, "the survivor ran past the intruder");
}

/// Run the leader's prompt at `choice` and return what the step committed and reported.
fn prefill_step_at(
    exec: &mut RealExecutor<NullForward>,
    params: &SamplingParams,
    choice: u32,
) -> (TokenId, Option<Vec<saber_serve::engine::SampledLogprobs>>) {
    let sampling = SlotSampling {
        params: params.clone(),
        choice_index: choice,
        step_index: 0,
        position: 0,
        penalties: None,
    };
    let slots = vec![BatchSlot {
        request_id: RequestId(1),
        seq_id: SeqId(1),
        slot: 0,
        kv_blocks: leader_blocks(),
        gdn_set: 1,
        verify_width: 0,
        input_tokens: prompt_ids(PROMPT),
        context_len: 0,
        first_step: true,
        chunk: Some(PrefillChunk {
            prompt_tokens: PROMPT as u32,
            tokens_done_before: 0,
            last: true,
            next_prompt_token: None,
        }),
        sampling,
    }];
    let out = exec.execute(&StepBatch {
        phase: Phase::Prefill,
        bucket: Bucket::prefill(PREFILL, 8192, true),
        step_index: 0,
        slots: &slots,
        padded_slots: 0,
    });
    (out.seqs[0].tokens[0], out.seqs[0].logprobs.clone())
}

/// **A BRANCH'S FIRST TOKEN CARRIES ITS LOGPROBS** (ws-f39). The fork draws a branch's first token
/// from the LEADER's row, and that draw reports exactly what a request of its own would: the
/// sampled token's logprob and the row's top-k, off the same row — so the top-k list is the
/// leader's own, number for number, and the sampled entry names the branch's token.
///
/// The control is the same prompt prefilled on its own at the branch's choice index: the same row
/// and the same key, so its report is the report the fork must hand back, entry for entry. Before
/// ws-f39 the fork's report was dropped (`fork_first_token_logprobs_absent`) and this failed at the
/// first `expect`.
#[test]
fn a_branchs_first_token_reports_the_logprobs_of_the_leaders_row() {
    let params = SamplingParams {
        logprobs: Some(3),
        ..prod_params(Some(4242), 4)
    };
    let mut exec = executor();
    exec.admit(&admitted(1, 0, leader_blocks(), 0, None));
    let (leader_token, leader_lp) = prefill_step_at(&mut exec, &params, 0);
    let leader_lp = leader_lp.expect("the leader's first token reports its logprobs");
    assert_eq!(leader_lp.len(), 1);
    assert_eq!(leader_lp[0].sampled.token_id, leader_token);

    for choice in 1..4u32 {
        let branch = admitted(
            10 + u64::from(choice),
            choice,
            branch_blocks(),
            2,
            Some(leader_blocks()[2]),
        );
        let forked = exec
            .fork(SeqId(1), &branch)
            .unwrap_or_else(|| panic!("choice {choice} was refused"));
        let lp = forked.logprobs.clone().unwrap_or_else(|| {
            panic!("choice {choice}: the fork dropped its first token's logprobs")
        });
        assert_eq!(
            lp.sampled.token_id, forked.token,
            "choice {choice}: the report names the token the branch committed"
        );
        // The same ROW: every id both reports carry has the same number, bit for bit (prod's
        // schema puts the sampled token first, so the two lists differ only by which token that is).
        let row = |l: &saber_serve::engine::SampledLogprobs| -> BTreeMap<TokenId, u32> {
            std::iter::once(&l.sampled)
                .chain(&l.top)
                .map(|t| (t.token_id, t.logprob.to_bits()))
                .collect()
        };
        let (ours, theirs) = (row(&lp), row(&leader_lp[0]));
        let common: Vec<&TokenId> = ours.keys().filter(|k| theirs.contains_key(k)).collect();
        assert!(
            !common.is_empty(),
            "choice {choice}: no id in common with the leader's report"
        );
        for id in common {
            assert_eq!(
                ours[id], theirs[id],
                "choice {choice}: id {id} moved off the leader's row"
            );
        }
        // The control: a prompt of its own at this choice reports the same entry, bit for bit.
        let mut solo = executor();
        solo.admit(&admitted(1, choice, leader_blocks(), 0, None));
        let (solo_token, solo_lp) = prefill_step_at(&mut solo, &params, choice);
        let solo_lp = solo_lp.expect("the control reports");
        assert_eq!(solo_token, forked.token, "choice {choice}: the same draw");
        assert_eq!(solo_lp[0], lp, "choice {choice}: the same report");
    }
    // A request that asked for nothing gets nothing, and pays nothing.
    let quiet = prod_params(Some(4242), 2);
    let mut exec = executor();
    exec.admit(&admitted(1, 0, leader_blocks(), 0, None));
    let _ = prefill_step_at(&mut exec, &quiet, 0);
    let forked = exec
        .fork(
            SeqId(1),
            &admitted(11, 1, branch_blocks(), 2, Some(leader_blocks()[2])),
        )
        .expect("forked");
    assert!(forked.logprobs.is_none(), "no logprobs were asked for");
}

/// The same law through the SCHEDULER: a group of `n = 2` that asks for `logprobs` gets one entry
/// per token on EVERY choice, the branch's first included, and the scheduler's
/// `fork_first_token_logprobs_absent` stays at zero.
#[tokio::test]
async fn every_choice_of_a_group_reports_one_logprob_entry_per_token() {
    let sched = common::scheduler_no_park();
    let params = SamplingParams {
        logprobs: Some(2),
        ..prod_params(Some(7), 2)
    };
    let sub = sched
        .submit(group_request(&prompt_ids(PROMPT), params, 6))
        .expect("submit");
    sched.run_steps(400);
    let events = drain(sub).await;
    let mut tokens: BTreeMap<u32, usize> = BTreeMap::new();
    let mut entries: BTreeMap<u32, usize> = BTreeMap::new();
    for e in &events {
        *tokens.entry(e.choice_index).or_default() += e.new_tokens.len();
        *entries.entry(e.choice_index).or_default() +=
            e.logprobs.as_ref().map_or(0, std::vec::Vec::len);
    }
    assert_eq!(tokens.len(), 2, "two choices");
    assert_eq!(
        tokens, entries,
        "one logprob entry per token, on every choice"
    );
    assert_eq!(sched.counters().fork_first_token_logprobs_absent, 0);
    assert_eq!(sched.counters().forks, 1);
}
