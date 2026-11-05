//! **A SEEDED REQUEST IS A FUNCTION OF ITS SEED, NOT OF ITS SCHEDULE** (workstream F40).
//!
//! F39's park leg found a seeded n = 2 group parked mid-stream differing from the same group run
//! alone (5 of 6 groups matched), and F38 saw a seeded reference move when three other groups ran
//! beside it. The draw was keyed by `(seed, the sequence's STEP, choice)` and only the LAST row of
//! a verify step was drawn — the accepted prefix was the argmax — so anything that moves a step
//! boundary (a park and the drafter's re-proposal after it, a crowd, a different verify width)
//! moved the stream.
//!
//! The law since F40: every token of a sampled request is a DRAW keyed by
//! `(seed, its absolute output position, choice)` over its own row (`SeededNoise`, counter-based),
//! and a verify step accepts a draft token iff it IS that position's draw
//! (`saber_engine::drawn_rows`). The committed stream is then the width-1 stream, draw for draw,
//! whatever else happens. This file drives the executor directly over [`NullForward`] — whose rows
//! are a fixed function of the prefix — and asserts it token for token:
//!
//!   (a) alone, at width 1 (the reference) and at width 8 with a drafter;
//!   (b) in a crowd of 3 (batch 4, width 8, the others sampled and greedy);
//!   (c) parked after 6 steps and restored into other pages and another slot, the drafter's
//!       mailbox empty on the first step back (the engine drafter's re-proposal);
//!   (d) at verify widths 1 / 5 / 8;
//!   (e) as choice 0 of an n = 2 group (the leader, decoded beside its branch).
//!
//! CONTROLS: `RealExecutor::step_keyed_draws(true)` — the engine before F40 — fails (c) against its
//! own unparked run and (d) against its own width-1 run. Temperature 0 is untouched (no draw, the
//! same ids and the same counters under the law and the control), the dumped-row law (V3) holds
//! for every committed token, and the tie counters count each committed draw exactly once.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use saber_engine::{
    Decider, DecodeSlot, Drafter, EngineCounters, ForwardError, ForwardFacts, ForwardRunner,
    NullForward, RealExecutor, RecordedDrafter, RowDecider, RowRequest, StepOutcome, StepRows,
};
use saber_plan::{Bucket, Phase};
use saber_sched::{
    AdmittedSeq, BatchSlot, PrefillChunk, SeqId, SlotSampling, StepBatch, StepExecutor,
};
use saber_serve::engine::{RequestId, TokenId};
use saber_serve::sampler::{DeviceRow, PenaltyContext, RowParams, Sampler, SeededNoise};
use saber_serve::sampling::SamplingParams;

const PREFILL: u32 = 2048;
const BATCHES: &[u32] = &[1, 2, 4];
const WIDTHS: &[u32] = &[1, 5, 8];
const PROMPT: usize = 12;
/// Tokens every stream is run to.
const TOKENS: usize = 48;
/// The drafter's own depth: a width-8 bucket verifies 7 guesses.
const DEPTH: u32 = 7;

// ── the drafter ─────────────────────────────────────────────────────────────────────────────────

/// A drafter that proposes a REFERENCE stream's next tokens, wrong at every position `p` with
/// `p % 5 == 3` — so steps accept 0..7 guesses and step boundaries fall everywhere. Its scripts can
/// be added while the executor owns it (a restored sequence arrives under a new id), and a
/// sequence can be told its mailbox is empty for one step, which is what the engine's own drafter
/// does after a restore: it re-proposes, and the first step back verifies nothing.
#[derive(Clone, Debug, Default)]
struct Oracle(Arc<Mutex<OracleState>>);

#[derive(Debug, Default)]
struct OracleState {
    /// Per sequence: the reference stream and how much of it this sequence has committed.
    scripts: BTreeMap<SeqId, (Vec<TokenId>, usize)>,
    empty_once: BTreeSet<SeqId>,
}

impl Oracle {
    fn script(&self, seq: SeqId, reference: &[TokenId], committed: usize, empty_once: bool) {
        let mut st = self.0.lock().expect("oracle");
        st.scripts.insert(seq, (reference.to_vec(), committed));
        if empty_once {
            st.empty_once.insert(seq);
        }
    }
}

impl Drafter for Oracle {
    fn name(&self) -> &'static str {
        "oracle"
    }
    fn max_width(&self) -> u32 {
        DEPTH
    }
    fn propose(&mut self, seq: SeqId, _last: TokenId, width: u32) -> Vec<TokenId> {
        let mut st = self.0.lock().expect("oracle");
        if st.empty_once.remove(&seq) {
            return Vec::new();
        }
        let Some((reference, at)) = st.scripts.get(&seq) else {
            return Vec::new();
        };
        (0..width.min(DEPTH) as usize)
            .map(|i| {
                let p = at + i;
                if p % 5 == 3 {
                    RecordedDrafter::WRONG
                } else {
                    reference.get(p).copied().unwrap_or(RecordedDrafter::WRONG)
                }
            })
            .collect()
    }
    fn commit(&mut self, seq: SeqId, committed: &[TokenId]) {
        if let Some((_, at)) = self.0.lock().expect("oracle").scripts.get_mut(&seq) {
            *at += committed.len();
        }
    }
    fn release(&mut self, seq: SeqId) {
        self.0.lock().expect("oracle").scripts.remove(&seq);
    }
}

// ── one request, as the scheduler would hold it ─────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Req {
    seq: u64,
    choice: u32,
    prompt: u64,
    params: SamplingParams,
    /// Everything committed, in order: `ids.len()` is the position the next step decides first.
    ids: Vec<TokenId>,
    /// What the last step committed — the next step's input, and its `committed_before`.
    last_commit: Vec<TokenId>,
    /// The sequence's own step index (the pre-F40 key; the control reads it).
    steps: u64,
}

fn prompt_of(p: u64) -> Vec<TokenId> {
    (0..PROMPT as TokenId)
        .map(|i| 1000 + i + 97 * p as TokenId)
        .collect()
}

fn blocks_of(seq: u64) -> Vec<u32> {
    (0..16).map(|b| b + 16 * seq as u32).collect()
}

fn admitted(r: &Req) -> AdmittedSeq {
    AdmittedSeq {
        request_id: RequestId(r.seq),
        choice_index: r.choice,
        seq_id: SeqId(r.seq),
        prompt_tokens: prompt_of(r.prompt),
        kv_blocks: blocks_of(r.seq),
        shared_prefix_blocks: 0,
        cow_source: None,
        gdn_set: r.seq as u32,
        gdn_slots: 2,
    }
}

fn sampling(r: &Req) -> SlotSampling {
    SlotSampling {
        params: r.params.clone(),
        choice_index: r.choice,
        step_index: r.steps,
        position: r.ids.len() as u64,
        penalties: None,
    }
}

fn seeded(seed: u64) -> SamplingParams {
    SamplingParams {
        temperature: 0.7,
        top_p: 0.95,
        top_k: 20,
        seed: Some(seed),
        ..SamplingParams::default()
    }
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    }
}

fn req(seq: u64, prompt: u64, params: SamplingParams) -> Req {
    Req {
        seq,
        choice: 0,
        prompt,
        params,
        ids: Vec::new(),
        last_commit: Vec::new(),
        steps: 0,
    }
}

type Exec<R> = RealExecutor<R>;

fn executor(oracle: &Oracle) -> Exec<NullForward> {
    RealExecutor::with_drafter(
        NullForward::with_widths(&[PREFILL], BATCHES, WIDTHS, 8),
        Box::new(oracle.clone()),
    )
}

fn prefill<R: ForwardRunner>(exec: &mut Exec<R>, r: &mut Req) {
    exec.admit(&admitted(r));
    let slots = vec![BatchSlot {
        request_id: RequestId(r.seq),
        seq_id: SeqId(r.seq),
        slot: 0,
        kv_blocks: blocks_of(r.seq),
        gdn_set: r.seq as u32,
        verify_width: 0,
        input_tokens: prompt_of(r.prompt),
        context_len: 0,
        first_step: true,
        chunk: Some(PrefillChunk {
            prompt_tokens: PROMPT as u32,
            tokens_done_before: 0,
            last: true,
            next_prompt_token: None,
        }),
        sampling: sampling(r),
    }];
    let out = exec.execute(&StepBatch {
        phase: Phase::Prefill,
        bucket: Bucket::prefill(PREFILL, 8192, true),
        step_index: 0,
        slots: &slots,
        padded_slots: 0,
    });
    let t = out.seqs[0].tokens.clone();
    assert_eq!(t.len(), 1, "the last prefill chunk commits one token");
    r.ids.extend_from_slice(&t);
    r.last_commit = t;
    r.steps += 1;
}

/// One decode step of every request in `reqs`, together, at `width`.
fn decode<R: ForwardRunner>(exec: &mut Exec<R>, reqs: &mut [&mut Req], width: u32, step: u64) {
    let batch = *BATCHES
        .iter()
        .find(|b| **b as usize >= reqs.len())
        .expect("a planned batch");
    let slots: Vec<BatchSlot> = reqs
        .iter()
        .map(|r| BatchSlot {
            request_id: RequestId(r.seq),
            seq_id: SeqId(r.seq),
            slot: 0,
            kv_blocks: blocks_of(r.seq),
            gdn_set: r.seq as u32,
            verify_width: width - 1,
            input_tokens: r.last_commit.clone(),
            context_len: (PROMPT + r.ids.len() - 1) as u32,
            first_step: false,
            chunk: None,
            sampling: sampling(r),
        })
        .collect();
    let out = exec.execute(&StepBatch {
        phase: Phase::Decode,
        bucket: Bucket::decode(batch, width, 8192, true),
        step_index: step,
        slots: &slots,
        padded_slots: batch - slots.len() as u32,
    });
    for (r, s) in reqs.iter_mut().zip(&out.seqs) {
        assert!(!s.tokens.is_empty(), "seq {} committed nothing", r.seq);
        r.ids.extend_from_slice(&s.tokens);
        r.last_commit = s.tokens.clone();
        r.steps += 1;
    }
}

/// Run every request to `TOKENS` together at `width`, the drafter scripted from `scripts`.
fn run_together<R: ForwardRunner>(
    exec: &mut Exec<R>,
    oracle: &Oracle,
    reqs: &mut [&mut Req],
    width: u32,
    scripts: &BTreeMap<u64, Vec<TokenId>>,
) {
    for r in reqs.iter_mut() {
        prefill(exec, r);
        if let Some(s) = scripts.get(&r.seq) {
            oracle.script(SeqId(r.seq), s, r.ids.len(), false);
        }
    }
    let mut step = 1;
    while reqs.iter().any(|r| r.ids.len() < TOKENS) {
        let mut live: Vec<&mut Req> = reqs
            .iter_mut()
            .filter(|r| r.ids.len() < TOKENS)
            .map(|r| &mut **r)
            .collect();
        decode(exec, &mut live, width, step);
        step += 1;
        assert!(step < 500, "not advancing");
    }
    for r in reqs.iter_mut() {
        r.ids.truncate(TOKENS);
    }
}

/// The request alone at `width`, its drafter scripted from `script` (none at width 1).
fn alone(params: &SamplingParams, width: u32, script: Option<&[TokenId]>, control: bool) -> Req {
    let oracle = Oracle::default();
    let mut exec = executor(&oracle);
    exec.step_keyed_draws(control);
    let mut r = req(1, 1, params.clone());
    let scripts: BTreeMap<u64, Vec<TokenId>> =
        script.map(|s| (1, s.to_vec())).into_iter().collect();
    run_together(&mut exec, &oracle, &mut [&mut r], width, &scripts);
    assert!(exec.counters().clean(), "{}", exec.counters().line());
    r
}

/// THE REFERENCE: the request alone at width 1 — one token, one step, one draw.
fn reference(params: &SamplingParams) -> Vec<TokenId> {
    alone(params, 1, None, false).ids
}

fn first_divergence(a: &[TokenId], b: &[TokenId]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

const SEED: u64 = 90_210;

// ── (a) + (d): alone, at every width ────────────────────────────────────────────────────────────

/// (a) and (d): the request alone at widths 1 / 5 / 8, the drafter scripted from the reference so
/// it is accepted at every depth, is the reference token for token — and at widths 5 and 8 it took
/// fewer steps, so speculation bought steps and not tokens.
#[test]
fn a_seeded_request_is_the_same_stream_at_widths_1_5_and_8() {
    let params = seeded(SEED);
    let want = reference(&params);
    assert_eq!(want.len(), TOKENS);
    for width in [1u32, 5, 8] {
        let got = alone(&params, width, Some(&want), false);
        assert_eq!(
            first_divergence(&got.ids, &want),
            None,
            "width {width}: the seeded stream left the width-1 stream"
        );
        if width > 1 {
            assert!(
                got.steps < want.len() as u64,
                "width {width}: {} steps for {} tokens — nothing was accepted, so the test \
                 proved nothing about the verify rows",
                got.steps,
                TOKENS
            );
        }
    }
    // The stream is a DRAW, not the argmax chain: the invariance is not the greedy theorem again.
    assert_ne!(
        want,
        reference(&greedy()),
        "the seeded stream is the argmax chain; this test would be the temperature-0 theorem"
    );
}

/// (d)'s CONTROL: the engine before F40 — argmax acceptance, the last row drawn, keyed by the
/// step — at width 8 is NOT its own width-1 stream.
#[test]
fn control_the_step_keyed_law_moves_with_the_verify_width() {
    let params = seeded(SEED);
    let want = alone(&params, 1, None, true).ids;
    assert_eq!(
        want,
        reference(&params),
        "at width 1 one step is one token, so the step key IS the position key"
    );
    let got = alone(&params, 8, Some(&want), true);
    assert!(
        first_divergence(&got.ids, &want).is_some(),
        "the step-keyed control at width 8 reproduced the width-1 stream: the control has no teeth"
    );
}

// ── (b): in a crowd ─────────────────────────────────────────────────────────────────────────────

/// (b) the request in a crowd of three others (batch 4, width 8): another seeded request, an
/// unseeded one and a greedy one, all scripted, all stepping together. Its stream is the reference.
#[test]
fn a_seeded_request_in_a_crowd_of_three_is_the_same_stream() {
    let params = seeded(SEED);
    let want = reference(&params);
    let oracle = Oracle::default();
    let mut exec = executor(&oracle);
    let mut me = req(1, 1, params.clone());
    let mut a = req(2, 2, seeded(7));
    let mut b = req(
        3,
        3,
        SamplingParams {
            seed: None,
            ..seeded(0)
        },
    );
    let mut c = req(4, 4, greedy());
    let scripts: BTreeMap<u64, Vec<TokenId>> = [
        (1, want.clone()),
        (2, reference(&seeded(7))),
        (4, reference(&greedy())),
    ]
    .into_iter()
    .collect();
    run_together(
        &mut exec,
        &oracle,
        &mut [&mut me, &mut a, &mut b, &mut c],
        8,
        &scripts,
    );
    assert_eq!(first_divergence(&me.ids, &want), None, "in a crowd of 3");
    assert!(me.steps < TOKENS as u64, "nothing was accepted");
    let cs = exec.counters();
    assert!(
        cs.greedy_rows > 0 && cs.sampled_rows > 0,
        "{}",
        cs.sampler_line()
    );
    assert!(cs.clean(), "{}", cs.line());
}

// ── (c): parked and restored ────────────────────────────────────────────────────────────────────

/// The request at width 8, parked after `park_after` decode steps and restored into other pages
/// and another state slot (an interloper holds the old one), the drafter's mailbox EMPTY on the
/// first step back. `park_after = None` is the unparked run.
fn parked(
    params: &SamplingParams,
    script: &[TokenId],
    park_after: Option<u64>,
    control: bool,
) -> Req {
    let oracle = Oracle::default();
    let mut exec = executor(&oracle);
    exec.step_keyed_draws(control);
    let mut r = req(1, 1, params.clone());
    prefill(&mut exec, &mut r);
    oracle.script(SeqId(1), script, r.ids.len(), false);
    let mut step = 1u64;
    while r.ids.len() < TOKENS {
        if park_after == Some(step - 1) {
            let state = exec.park_state(SeqId(r.seq)).expect("the stub parks");
            exec.release_parked(SeqId(r.seq));
            // Somebody else takes the freed slot, so the restore lands elsewhere.
            exec.admit(&admitted(&req(90, 9, greedy())));
            r.seq = 2;
            exec.admit(&admitted(&r));
            assert!(exec.restore_state(&admitted(&r), &state), "restore");
            oracle.script(SeqId(r.seq), script, r.ids.len(), true);
        }
        decode(&mut exec, &mut [&mut r], 8, step);
        step += 1;
        assert!(step < 500);
    }
    r.ids.truncate(TOKENS);
    let c = exec.counters();
    assert!(c.clean(), "{}", c.line());
    if park_after.is_some() {
        assert_eq!((c.parks, c.restores), (1, 1));
    }
    r
}

/// (c) parked after 6 steps and restored, the first step back verifying nothing: the reference.
#[test]
fn a_seeded_request_parked_and_restored_after_6_steps_is_the_same_stream() {
    let params = seeded(SEED);
    let want = reference(&params);
    let unparked = parked(&params, &want, None, false);
    let got = parked(&params, &want, Some(6), false);
    assert_eq!(first_divergence(&unparked.ids, &want), None, "unparked");
    assert_eq!(
        first_divergence(&got.ids, &want),
        None,
        "parked after 6 steps"
    );
    assert!(got.ids.len() == TOKENS && got.steps > unparked.steps.min(7));
}

/// (c)'s CONTROL: under the pre-F40 law the SAME park moves the stream against the SAME request
/// unparked — the empty mailbox after the restore shifts every later step boundary, and the step
/// key moves with them. This is F39's "5 of 6 groups", on the host.
#[test]
fn control_the_step_keyed_law_moves_with_a_park() {
    let params = seeded(SEED);
    // The drafter scripted from the control's OWN width-8 stream, so it is accepted under the
    // control's own (argmax) law and the step boundaries the park moves are real ones. Under the
    // control the stream depends on the drafter, so "its own stream" is a FIXPOINT: script, run,
    // re-script from what ran, until the run is the script.
    let mut own = reference(&params);
    for _ in 0..64 {
        let next = parked(&params, &own, None, true).ids;
        if next == own {
            break;
        }
        own = next;
    }
    let unparked = parked(&params, &own, None, true);
    assert_eq!(
        unparked.ids, own,
        "the control's fixpoint (it repeats itself)"
    );
    assert!(unparked.steps < TOKENS as u64, "nothing was accepted");
    let got = parked(&params, &own, Some(6), true);
    assert!(
        first_divergence(&got.ids, &unparked.ids).is_some(),
        "the step-keyed control survived a park: the control has no teeth"
    );
}

// ── (e): choice 0 of an n = 2 group ─────────────────────────────────────────────────────────────

/// (e) the request as choice 0 of an n = 2 group: the leader prefills, a branch is forked off its
/// row, and the two decode together at width 8. The leader is the reference; the branch is a
/// different draw (choice 1), and is itself its own reference at width 1 (the same group, alone,
/// with no drafter).
#[test]
fn a_seeded_request_as_choice_0_of_a_group_of_two_is_the_same_stream() {
    let params = seeded(SEED);
    let want = reference(&params);
    let group = |width: u32, script: Option<(&[TokenId], &[TokenId])>| -> (Req, Req) {
        let oracle = Oracle::default();
        let mut exec = executor(&oracle);
        let mut leader = req(1, 1, params.clone());
        prefill(&mut exec, &mut leader);
        let mut branch = Req {
            seq: 2,
            choice: 1,
            ..leader.clone()
        };
        let mut adm = admitted(&branch);
        adm.shared_prefix_blocks = 0;
        let first = exec.fork(SeqId(1), &adm).expect("the stub forks").token;
        branch.ids = vec![first];
        branch.last_commit = vec![first];
        if let Some((l, b)) = script {
            oracle.script(SeqId(1), l, 1, false);
            oracle.script(SeqId(2), b, 1, false);
        }
        let mut step = 1;
        while leader.ids.len() < TOKENS || branch.ids.len() < TOKENS {
            let mut live: Vec<&mut Req> = [&mut leader, &mut branch]
                .into_iter()
                .filter(|r| r.ids.len() < TOKENS)
                .collect();
            decode(&mut exec, &mut live, width, step);
            step += 1;
            assert!(step < 500);
        }
        leader.ids.truncate(TOKENS);
        branch.ids.truncate(TOKENS);
        assert!(exec.counters().clean(), "{}", exec.counters().line());
        (leader, branch)
    };
    let (l1, b1) = group(1, None);
    assert_eq!(
        first_divergence(&l1.ids, &want),
        None,
        "choice 0 at width 1"
    );
    assert_ne!(l1.ids, b1.ids, "the two branches are draws, not copies");
    let (l8, b8) = group(8, Some((&l1.ids, &b1.ids)));
    assert_eq!(
        first_divergence(&l8.ids, &want),
        None,
        "choice 0 at width 8"
    );
    assert_eq!(
        first_divergence(&b8.ids, &b1.ids),
        None,
        "choice 1 at width 8 is choice 1 at width 1"
    );
    assert!(l8.steps < TOKENS as u64 && b8.steps < TOKENS as u64);
}

// ── temperature 0 is untouched ──────────────────────────────────────────────────────────────────

/// Temperature 0 draws nothing: under the law and under the control the ids, the acceptance and
/// every sampler counter are the same, and no row was drawn.
#[test]
fn temperature_0_is_bit_identical_and_draws_nothing() {
    let want = reference(&greedy());
    let run = |control: bool| -> (Req, EngineCounters) {
        let oracle = Oracle::default();
        let mut exec = executor(&oracle);
        exec.step_keyed_draws(control);
        let mut r = req(1, 1, greedy());
        let scripts = [(1, want.clone())].into_iter().collect();
        run_together(&mut exec, &oracle, &mut [&mut r], 8, &scripts);
        (r, exec.counters())
    };
    let (law, lc) = run(false);
    let (ctl, cc) = run(true);
    assert_eq!(law.ids, want);
    assert_eq!(ctl.ids, want);
    assert_eq!(law.steps, ctl.steps, "the same steps");
    assert_eq!(
        lc.accepted_tokens, cc.accepted_tokens,
        "the same acceptance"
    );
    for c in [&lc, &cc] {
        assert_eq!(c.sampled_rows, 0, "no row was drawn: {}", c.sampler_line());
        assert_eq!(c.step_keyed_rows, 0);
        assert_eq!((c.top_ties, c.top_tie_draws), (0, 0));
        assert!(c.greedy_rows > 0);
    }
}

// ── the dumped-row law (V3) and the tie counters ────────────────────────────────────────────────

/// One dumped row, as the executor writes it.
struct DumpedRow {
    seq: u64,
    position: u64,
    choice: u32,
    seed: Option<u64>,
    token: TokenId,
    row: DeviceRow,
    params: RowParams,
}

fn read_dump(path: &std::path::Path) -> Vec<DumpedRow> {
    let text = std::fs::read_to_string(path).expect("the dump");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).expect("a JSON line");
            let ids: Vec<TokenId> = v["top_ids"]
                .as_array()
                .expect("ids")
                .iter()
                .map(|x| x.as_u64().expect("id") as TokenId)
                .collect();
            let values: Vec<f32> = v["top_values"]
                .as_array()
                .expect("values")
                .iter()
                .map(|x| x.as_f64().expect("value") as f32)
                .collect();
            let vocab = v["vocab"].as_u64().expect("vocab") as usize;
            let p = SamplingParams {
                temperature: v["temperature"].as_f64().expect("t") as f32,
                top_p: v["top_p"].as_f64().expect("p") as f32,
                top_k: v["top_k"].as_i64().expect("k") as i32,
                min_p: v["min_p"].as_f64().expect("min_p") as f32,
                ..SamplingParams::default()
            };
            DumpedRow {
                seq: v["seq"].as_u64().expect("seq"),
                position: v["position"].as_u64().expect("position"),
                choice: v["choice"].as_u64().expect("choice") as u32,
                seed: v["seed"].as_u64(),
                token: v["token"].as_u64().expect("token") as TokenId,
                row: DeviceRow {
                    top_values: values,
                    top_ids: ids,
                    logsumexp: v["lse"].as_f64().map_or(f32::NAN, |x| x as f32),
                    logsumexp_at_temperature: v["lse_t"].as_f64().map(|x| x as f32),
                    penalised: Vec::new(),
                    vocab_size: vocab,
                },
                params: RowParams::resolve(&p, vocab),
            }
        })
        .collect()
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).expect("tmp");
    dir.join(name)
}

/// V3's DUMPED-ROW LAW at width 8: every committed token of a sampled request has a dumped row,
/// in position order, and replaying the shipped sampler over that row at `(seed, position,
/// choice)` gives the committed token — the accepted prefix included, which before F40 was the
/// argmax and had no line at all.
#[test]
fn the_dumped_row_law_holds_for_every_committed_token_at_width_8() {
    let params = seeded(SEED);
    let want = reference(&params);
    let oracle = Oracle::default();
    let mut exec = executor(&oracle);
    let path = scratch("f40-dump.jsonl");
    exec.dump_samples_to(&path).expect("dump");
    let mut r = req(1, 1, params.clone());
    let scripts = [(1, want.clone())].into_iter().collect();
    run_together(&mut exec, &oracle, &mut [&mut r], 8, &scripts);
    drop(exec); // flush the dump
    let rows = read_dump(&path);
    let mine: Vec<&DumpedRow> = rows.iter().filter(|d| d.seq == 1).collect();
    assert!(
        mine.len() >= TOKENS,
        "{} dumped rows for {TOKENS} tokens",
        mine.len()
    );
    let positions: Vec<u64> = mine.iter().map(|d| d.position).collect();
    assert_eq!(
        positions,
        (0..mine.len() as u64).collect::<Vec<_>>(),
        "one row per committed token, in position order"
    );
    let dumped: Vec<TokenId> = mine.iter().map(|d| d.token).take(TOKENS).collect();
    assert_eq!(dumped, r.ids, "the stream is the dumped rows' tokens");
    let sampler = Sampler::new();
    for d in &mine {
        let seed = d.seed.expect("seeded");
        let out = sampler
            .sample(
                &d.row,
                &d.params,
                &PenaltyContext::default(),
                &mut SeededNoise::new(seed, d.position, d.choice),
            )
            .expect("the dumped row replays");
        assert_eq!(out.token_id, d.token, "position {}", d.position);
    }
    // THE CONTROL: the same rows replayed at the STEP-shaped key (position shifted by one) do not
    // all give the committed token, so the replay above is not vacuous.
    let moved = mine
        .iter()
        .filter(|d| {
            sampler
                .sample(
                    &d.row,
                    &d.params,
                    &PenaltyContext::default(),
                    &mut SeededNoise::new(d.seed.unwrap_or(0), d.position + 1, d.choice),
                )
                .map(|o| o.token_id)
                .ok()
                != Some(d.token)
        })
        .count();
    assert!(
        moved > 0,
        "every row gives its token at any key: the replay proves nothing"
    );
    let _ = std::fs::remove_file(&path);
}

/// A pass whose seams all carry a TIE at the top (the runner-up raised to the argmax, where its id
/// is higher), handed to the walk as the device would hand them.
#[derive(Debug)]
struct Tied {
    inner: NullForward,
}

#[derive(Debug)]
struct TiedSeams(Arc<dyn RowDecider>);

fn tie(r: &DeviceRow) -> DeviceRow {
    let mut r = r.clone();
    if r.top_ids.len() > 1 && r.top_ids[1] > r.top_ids[0] {
        r.top_values[1] = r.top_values[0];
    }
    r
}

impl RowDecider for TiedSeams {
    fn decide(&self, j: usize, seam: Option<&DeviceRow>) -> Option<TokenId> {
        let row = seam.map(tie);
        self.0.decide(j, row.as_ref())
    }
    fn ends_at(&self, j: usize, token: TokenId) -> bool {
        self.0.ends_at(j, token)
    }
}

impl ForwardRunner for Tied {
    fn facts(&self) -> &ForwardFacts {
        self.inner.facts()
    }
    fn admit(&mut self, seq: SeqId, prompt_len: u32, kv: &[u32]) -> Result<(), ForwardError> {
        self.inner.admit(seq, prompt_len, kv)
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
        let mut out = self.inner.prefill(seq, chunk, done, last, sampling)?;
        for c in &mut out.commits {
            for r in c.rows.iter_mut().flatten() {
                *r = tie(r);
            }
        }
        Ok(out)
    }
    fn decode(
        &mut self,
        slots: &[DecodeSlot],
        batch: u32,
        width: u32,
    ) -> Result<StepOutcome, ForwardError> {
        let slots: Vec<DecodeSlot> = slots
            .iter()
            .map(|s| {
                let mut s = s.clone();
                s.decider = s
                    .decider
                    .take()
                    .map(|d| Decider(Arc::new(TiedSeams(d.0)) as Arc<dyn RowDecider>));
                s
            })
            .collect();
        let mut out = self.inner.decode(&slots, batch, width)?;
        for c in &mut out.commits {
            for r in c.rows.iter_mut().flatten() {
                *r = tie(r);
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

/// THE TIE COUNTERS COUNT DRAWS, ONCE EACH (ws-f40, F39's conc-8 nit). `top_k 1` over rows that
/// all tie at the top, at width 1 and at width 8 with the drafter accepting: the two streams are
/// the same (the invariance holds through ties), and at BOTH widths `top_ties` is the number of
/// committed tokens and `top_tie_draws` the number whose token is not the tie's lowest id — counted
/// independently off the dump. Before F40 a width-8 step drew only its last row, so a tie resolved
/// in an accepted prefix was never a draw and never counted.
#[test]
fn the_tie_counters_are_the_committed_draws_that_resolved_a_tie_at_every_width() {
    let params = SamplingParams {
        top_k: 1,
        ..seeded(SEED)
    };
    let tied_run = |width: u32, script: Option<&[TokenId]>, name: &str| {
        let oracle = Oracle::default();
        let mut exec = RealExecutor::with_drafter(
            Tied {
                inner: NullForward::with_widths(&[PREFILL], BATCHES, WIDTHS, 8),
            },
            Box::new(oracle.clone()),
        );
        let path = scratch(name);
        exec.dump_samples_to(&path).expect("dump");
        let mut r = req(1, 1, params.clone());
        let scripts: BTreeMap<u64, Vec<TokenId>> =
            script.map(|s| (1, s.to_vec())).into_iter().collect();
        run_together(&mut exec, &oracle, &mut [&mut r], width, &scripts);
        let c = exec.counters();
        drop(exec);
        let rows = read_dump(&path);
        let _ = std::fs::remove_file(&path);
        (r, c, rows)
    };
    let (w1, c1, rows1) = tied_run(1, None, "f40-tie-w1.jsonl");
    let (w8, c8, rows8) = tied_run(8, Some(&w1.ids), "f40-tie-w8.jsonl");
    assert_eq!(
        first_divergence(&w8.ids, &w1.ids),
        None,
        "ties: width 8 = width 1"
    );
    assert!(w8.steps < TOKENS as u64, "nothing was accepted at width 8");
    for (c, rows, w) in [(&c1, &rows1, 1), (&c8, &rows8, 8)] {
        let ties = rows
            .iter()
            .filter(|d| d.row.top_values.len() > 1 && d.row.top_values[1] == d.row.top_values[0])
            .count() as u64;
        let off = rows
            .iter()
            .filter(|d| {
                let top = d.row.top_values[0];
                let lowest = d
                    .row
                    .top_ids
                    .iter()
                    .zip(&d.row.top_values)
                    .filter(|(_, v)| **v == top)
                    .map(|(i, _)| *i)
                    .min();
                d.row.top_values.len() > 1 && d.row.top_values[1] == top && Some(d.token) != lowest
            })
            .count() as u64;
        assert_eq!(
            rows.len() as u64,
            c.sampled_rows,
            "width {w}: one dumped row per draw"
        );
        assert!(
            ties > 0 && off > 0,
            "width {w}: the tamper made no tie draw"
        );
        assert_eq!(
            c.top_ties, ties,
            "width {w}: ties at the top = tied committed rows"
        );
        assert_eq!(
            c.top_tie_draws, off,
            "width {w}: tie draws = committed tokens that are not the tie's lowest id"
        );
    }
    assert!(
        c8.top_tie_draws > 0 && c8.top_ties >= c8.top_tie_draws,
        "{}",
        c8.sampler_line()
    );
}

/// THE DEVICE FEED after a draw (ws-f40). The device leaves the ARGMAX of a step's last committed
/// row in its ids buffer; a sampled walk may commit a different token there, and the next step must
/// then UPLOAD its input rather than feed what the device holds. The pass records the argmax, not
/// the committed token, as what it left — so every step after a draw that is not its row's argmax
/// is host-fed. Counted off the dump: one row per token, `top_ids[0]` the row's argmax.
#[test]
fn a_draw_that_is_not_the_argmax_is_never_fed_from_the_device() {
    let params = seeded(SEED);
    let oracle = Oracle::default();
    let mut exec = executor(&oracle);
    let path = scratch("f40-feed.jsonl");
    exec.dump_samples_to(&path).expect("dump");
    let mut r = req(1, 1, params);
    run_together(&mut exec, &oracle, &mut [&mut r], 1, &BTreeMap::new());
    let c = exec.counters();
    drop(exec);
    let rows = read_dump(&path);
    let _ = std::fs::remove_file(&path);
    // Every committed token but the last is some later step's input.
    let off_argmax = rows[..TOKENS - 1]
        .iter()
        .filter(|d| d.token != d.row.top_ids[0])
        .count() as u64;
    let on_argmax = (TOKENS as u64 - 1) - off_argmax;
    assert!(
        off_argmax > 0 && on_argmax > 0,
        "{off_argmax} / {on_argmax}"
    );
    assert!(
        c.host_fed_steps >= off_argmax,
        "{} host-fed step(s) for {off_argmax} input(s) the device never wrote",
        c.host_fed_steps
    );
    assert!(
        c.device_fed_steps <= on_argmax,
        "{} device-fed step(s), but only {on_argmax} input(s) were the argmax the device wrote",
        c.device_fed_steps
    );
}

// ── (f) the walk verifies the whole block, the request's end included (ws-f41) ──────────────────

/// Run one seeded request alone at width 8 until it has at least `until` tokens, its drafter
/// scripted from `script`; the executor's counters and the raw (untruncated) commits come back.
fn run_until(params: &SamplingParams, script: &[TokenId], until: usize) -> (Req, EngineCounters) {
    let oracle = Oracle::default();
    let mut exec = executor(&oracle);
    let mut r = req(1, 1, params.clone());
    prefill(&mut exec, &mut r);
    oracle.script(SeqId(r.seq), script, r.ids.len(), false);
    let mut step = 1;
    while r.ids.len() < until {
        decode(&mut exec, &mut [&mut r], 8, step);
        step += 1;
        assert!(step < 500, "not advancing");
    }
    assert!(exec.counters().clean(), "{}", exec.counters().line());
    (r, exec.counters())
}

/// PROD VERIFIES THE WHOLE BLOCK and counts what it accepts before the scheduler cuts at
/// `max_tokens` (the reference server's rejection sampler knows nothing of the request's end), and so does Saber's
/// greedy law. F40's drawn walk stopped at the request's last position and committed the matched
/// draft there as the RECOVERY token, so every sampled request that ended inside a verify block
/// lost that accepted draft and every draft behind it — at 64 tokens per request that was the
/// larger half of Q3's "sampled gap" (LEDGER F41). The law now: `max_tokens` does not move the walk.
/// The same seeded request with and without `max_tokens` commits the same tokens at every step and
/// counts the same acceptance; the step that crosses the end commits PAST it (the scheduler drops
/// those, exactly as it drops a greedy step's), and what is served is the width-1 reference.
///
/// Control: the pre-F41 walk, which stopped at `max_tokens - 1`, commits exactly `LIMIT` tokens
/// here and counts fewer accepted — the `overshoot` and counter assertions are what it fails.
#[test]
fn max_tokens_does_not_move_the_walk() {
    const LIMIT: usize = 40;
    let unbounded = seeded(SEED);
    let bounded = SamplingParams {
        max_tokens: Some(LIMIT),
        ..seeded(SEED)
    };
    let want = reference(&unbounded);
    let (a, ca) = run_until(&bounded, &want, LIMIT);
    let (b, cb) = run_until(&unbounded, &want, LIMIT);
    assert_eq!(a.ids, b.ids, "max_tokens moved the walk");
    assert_eq!(
        (ca.draft_tokens, ca.accepted_tokens),
        (cb.draft_tokens, cb.accepted_tokens),
        "max_tokens moved the acceptance count"
    );
    let overshoot = a.ids.len() - LIMIT;
    assert!(
        overshoot > 0,
        "the step that crossed position {LIMIT} stopped at the end: the walk still reads max_tokens"
    );
    assert_eq!(
        first_divergence(&a.ids[..LIMIT], &want[..LIMIT]),
        None,
        "the served prefix is not the width-1 reference"
    );
    eprintln!(
        "max_tokens {LIMIT}: {} committed ({overshoot} past the end, dropped by the scheduler), \
         {} accepted of {} drafted — the same with no max_tokens",
        a.ids.len(),
        ca.accepted_tokens,
        ca.draft_tokens
    );
}
