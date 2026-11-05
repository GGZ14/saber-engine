//! **P27-3: the prompt's drafter store, deferred past the first token** (`--drafter-store`).
//!
//! On the control (`on-path`, today's law) a prompt's last chunk stores the drafter's context and
//! the executor collects the first proposal BEFORE it hands the chunk's token back. Under
//! `deferred` the runner leaves the last chunk's store in the stash with the first proposal
//! (`ForwardRunner::defers_prompt_store`), the executor hands the token back first, and collects
//! the stash at the head of its NEXT call into the runner. These laws hold on the host, with a
//! stub that logs every runner call:
//!
//! * the store + first proposal runs EXACTLY ONCE per prompt in both modes;
//! * under `deferred` it runs AFTER the chunk's call returned and BEFORE any other runner call —
//!   a decode step, an admit, a release (the stash law: a runner that is called for anything else
//!   first abandons it);
//! * the first decode step drafts the SAME block in both modes (the mailbox is filled before the
//!   step asks), so a step never verifies before the store lands, and the ids are the control's;
//! * the control: with `defers_prompt_store` false, the collection is inside the chunk's call.

use saber_engine::{
    EngineDrafter, ForwardError, ForwardFacts, ForwardRunner, NullForward, Proposal, RealExecutor,
    RowRequest, StepOutcome, StepRows,
};
use saber_plan::{Bucket, Phase};
use saber_sched::{
    AdmittedSeq, BatchSlot, PrefillChunk, SeqId, SlotSampling, StepBatch, StepExecutor,
};
use saber_serve::engine::{RequestId, TokenId};
use saber_serve::sampling::SamplingParams;

const PREFILL: u32 = 2048;
const PROMPT: usize = 12;
const WIDTH: u32 = 4;

/// One runner call, in the order the executor made them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Call {
    Admit,
    Release,
    Prefill,
    Decode,
    /// The stash collected: the prompt's (deferred) store + its first proposal, or a decode
    /// step's proposal.
    Collect,
}

/// [`NullForward`] whose steps stash a proposal (prod's order), with the P27-3 switch.
#[derive(Debug)]
struct Logged {
    inner: NullForward,
    defer: bool,
    stash: Option<Vec<SeqId>>,
    /// Whether the stash is a PROMPT's (its store rides in it under `defer`).
    stash_is_prompt: bool,
    calls: Vec<Call>,
    /// Prompt stores run (the stash of a last chunk collected).
    prompt_stores: u32,
    /// Stashes dropped because another call came first.
    abandoned: u32,
}

impl Logged {
    fn new(defer: bool) -> Self {
        Self {
            inner: NullForward::with_widths(&[PREFILL], &[1, 2, 4], &[1, WIDTH], 8),
            defer,
            stash: None,
            stash_is_prompt: false,
            calls: Vec::new(),
            prompt_stores: 0,
            abandoned: 0,
        }
    }
    fn abandon(&mut self) {
        if self.stash.take().is_some() {
            self.abandoned += 1;
        }
    }
}

fn block_of(anchor: TokenId) -> Vec<TokenId> {
    (1..WIDTH).map(|j| anchor + j).collect()
}

impl ForwardRunner for Logged {
    fn facts(&self) -> &ForwardFacts {
        self.inner.facts()
    }
    fn defers_prompt_store(&self) -> bool {
        self.defer
    }
    fn admit(&mut self, seq: SeqId, prompt_len: u32, kv: &[u32]) -> Result<(), ForwardError> {
        self.abandon();
        self.calls.push(Call::Admit);
        self.inner.admit(seq, prompt_len, kv)
    }
    fn release(&mut self, seq: SeqId) {
        self.abandon();
        self.calls.push(Call::Release);
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
        self.calls.push(Call::Prefill);
        let mut out = self.inner.prefill(seq, chunk, done, last, sampling)?;
        if last {
            out.proposal_pending = vec![seq];
            self.stash = Some(vec![seq]);
            self.stash_is_prompt = true;
        }
        Ok(out)
    }
    fn decode(
        &mut self,
        slots: &[saber_engine::DecodeSlot],
        batch: u32,
        width: u32,
    ) -> Result<StepOutcome, ForwardError> {
        self.abandon();
        self.calls.push(Call::Decode);
        let mut out = self.inner.decode(slots, batch, width)?;
        out.proposal_pending = out.commits.iter().map(|c| c.seq).collect();
        self.stash = Some(out.proposal_pending.clone());
        self.stash_is_prompt = false;
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
        if !stash.iter().copied().eq(anchors.iter().map(|a| a.0)) {
            return Err(ForwardError::Device("anchors for other slots".to_owned()));
        }
        self.calls.push(Call::Collect);
        if self.stash_is_prompt {
            self.prompt_stores += 1;
        }
        Ok(anchors
            .iter()
            .map(|(seq, a)| Proposal {
                seq: *seq,
                anchor: *a,
                block: block_of(*a),
            })
            .collect())
    }
    fn declared_launches(&self, rows: StepRows) -> Option<u32> {
        self.inner.declared_launches(rows)
    }
    fn dense_row(&mut self, row: u32) -> Result<Vec<f32>, ForwardError> {
        self.inner.dense_row(row)
    }
}

fn sampling(step: u64) -> SlotSampling {
    SlotSampling {
        params: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
        choice_index: 0,
        step_index: step,
        position: step,
        penalties: None,
    }
}

fn prompt() -> Vec<TokenId> {
    (0..PROMPT as TokenId).map(|j| 1000 + j).collect()
}

fn admitted() -> AdmittedSeq {
    AdmittedSeq {
        request_id: RequestId(1),
        choice_index: 0,
        seq_id: SeqId(1),
        prompt_tokens: prompt(),
        kv_blocks: (0..16).collect(),
        shared_prefix_blocks: 0,
        cow_source: None,
        gdn_set: 1,
        gdn_slots: 2,
    }
}

fn prefill(exec: &mut RealExecutor<Logged>) -> Vec<TokenId> {
    let slots = vec![BatchSlot {
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
        sampling: sampling(0),
    }];
    let out = exec.execute(&StepBatch {
        phase: Phase::Prefill,
        bucket: Bucket::prefill(PREFILL, 8192, true),
        step_index: 0,
        slots: &slots,
        padded_slots: 0,
    });
    out.seqs[0].tokens.clone()
}

/// One decode step at width [`WIDTH`]: `(tokens, drafted)`.
fn decode(
    exec: &mut RealExecutor<Logged>,
    last: TokenId,
    ctx: u32,
    step: u64,
) -> (Vec<TokenId>, u32) {
    let slots = vec![BatchSlot {
        request_id: RequestId(1),
        seq_id: SeqId(1),
        slot: 0,
        kv_blocks: (0..16).collect(),
        gdn_set: 1,
        verify_width: WIDTH,
        input_tokens: vec![last],
        context_len: ctx,
        first_step: false,
        chunk: None,
        sampling: sampling(step),
    }];
    let out = exec.execute(&StepBatch {
        phase: Phase::Decode,
        bucket: Bucket::decode(1, WIDTH, 8192, true),
        step_index: step,
        slots: &slots,
        padded_slots: 0,
    });
    (out.seqs[0].tokens.clone(), out.seqs[0].draft_tokens)
}

struct Run {
    ids: Vec<TokenId>,
    drafted: Vec<u32>,
    /// The runner calls made by the prefill's own `execute`.
    prefill_calls: Vec<Call>,
    /// Whether the store was owed when the prefill's `execute` returned.
    owed_after_prefill: bool,
    calls: Vec<Call>,
    prompt_stores: u32,
    abandoned: u32,
    settles: u64,
}

fn run(defer: bool, steps: u64) -> Run {
    let mut exec =
        RealExecutor::with_drafter(Logged::new(defer), Box::new(EngineDrafter::new(WIDTH - 1)));
    exec.admit(&admitted());
    let before = exec.forward_mut().calls.len();
    let mut ids = prefill(&mut exec);
    let prefill_calls = exec.forward_mut().calls[before..].to_vec();
    let owed_after_prefill = exec.store_owed();
    let mut drafted = Vec::new();
    let mut ctx = PROMPT as u32;
    for step in 1..=steps {
        let (t, d) = decode(&mut exec, *ids.last().expect("a token"), ctx, step);
        ctx += t.len() as u32;
        ids.extend(t);
        drafted.push(d);
    }
    let settles = exec.deferred_store_settles();
    let f = exec.forward_mut();
    Run {
        ids,
        drafted,
        prefill_calls,
        owed_after_prefill,
        calls: f.calls.clone(),
        prompt_stores: f.prompt_stores,
        abandoned: f.abandoned,
        settles,
    }
}

/// The control: the store + first proposal is collected INSIDE the chunk's call.
#[test]
fn on_path_collects_the_prompt_store_inside_the_chunk() {
    let r = run(false, 4);
    assert_eq!(r.prefill_calls, vec![Call::Prefill, Call::Collect]);
    assert!(!r.owed_after_prefill);
    assert_eq!(r.prompt_stores, 1, "the prompt's store runs exactly once");
    assert_eq!(r.settles, 0, "the control settles nothing late");
    assert_eq!(r.abandoned, 0);
}

/// The arm: the chunk's call returns with the store OWED, and the next call into the runner is the
/// collection — before the decode step runs and before it asks the mailbox for a block.
#[test]
fn deferred_hands_the_token_back_first_and_stores_before_the_next_runner_call() {
    let r = run(true, 4);
    assert_eq!(
        r.prefill_calls,
        vec![Call::Prefill],
        "the token went back before the store ran"
    );
    assert!(
        r.owed_after_prefill,
        "the store is owed when the chunk returns"
    );
    let first_decode = r
        .calls
        .iter()
        .position(|c| *c == Call::Decode)
        .expect("a decode step ran");
    assert_eq!(
        r.calls[first_decode - 1],
        Call::Collect,
        "the owed store landed immediately before the first decode step"
    );
    assert_eq!(r.prompt_stores, 1, "the prompt's store runs exactly once");
    assert_eq!(r.settles, 1);
    assert_eq!(r.abandoned, 0, "no stash was ever dropped");
}

/// The byte law on the host: the same ids and the same drafted rows on EVERY step, the first
/// decode step's block included — a step never drafts from a store that has not landed.
#[test]
fn deferred_drafts_the_same_first_block_and_commits_the_same_ids() {
    let a = run(false, 6);
    let b = run(true, 6);
    assert_eq!(a.ids, b.ids, "the target's ids are the control's");
    assert_eq!(
        a.drafted, b.drafted,
        "every step drafted the control's rows"
    );
    assert_eq!(
        b.drafted[0],
        WIDTH - 1,
        "the first decode step verified the prompt's block, not an empty mailbox"
    );
}

/// The stash law beyond decode: an admit or a release that comes first still finds the store
/// collected before it (never abandoned).
#[test]
fn an_owed_store_is_collected_before_an_admit_or_a_release() {
    for release in [false, true] {
        let mut exec =
            RealExecutor::with_drafter(Logged::new(true), Box::new(EngineDrafter::new(WIDTH - 1)));
        exec.admit(&admitted());
        let _ = prefill(&mut exec);
        assert!(exec.store_owed());
        if release {
            exec.release(SeqId(1));
        } else {
            let mut other = admitted();
            other.seq_id = SeqId(2);
            other.request_id = RequestId(2);
            other.gdn_set = 2;
            other.kv_blocks = (16..32).collect();
            exec.admit(&other);
        }
        assert!(!exec.store_owed());
        let f = exec.forward_mut();
        let n = f.calls.len();
        assert_eq!(f.calls[n - 2], Call::Collect, "{:?}", f.calls);
        assert_eq!(f.prompt_stores, 1);
        assert_eq!(f.abandoned, 0);
    }
}

/// The arm defers only an UNMIXED prefill: a mixed call runs its chunk after the decode half in the
/// same call (a packed call several chunks), so their collections stay in place — the runner holds
/// one stash at a time. Hold 1's gate found exactly that abandonment in the packed arms (4 slots)
/// before this law existed.
#[test]
fn a_mixed_call_collects_its_chunks_store_in_place() {
    let mut exec =
        RealExecutor::with_drafter(Logged::new(true), Box::new(EngineDrafter::new(WIDTH - 1)));
    exec.admit(&admitted());
    let first = prefill(&mut exec);
    let (t, _) = decode(&mut exec, *first.last().expect("a token"), PROMPT as u32, 1);
    let mut other = admitted();
    other.seq_id = SeqId(2);
    other.request_id = RequestId(2);
    other.gdn_set = 2;
    other.kv_blocks = (16..32).collect();
    exec.admit(&other);
    let dslots = vec![BatchSlot {
        request_id: RequestId(1),
        seq_id: SeqId(1),
        slot: 0,
        kv_blocks: (0..16).collect(),
        gdn_set: 1,
        verify_width: WIDTH,
        input_tokens: vec![*t.last().expect("a token")],
        context_len: PROMPT as u32 + t.len() as u32,
        first_step: false,
        chunk: None,
        sampling: sampling(2),
    }];
    let cslots = vec![BatchSlot {
        request_id: RequestId(2),
        seq_id: SeqId(2),
        slot: 0,
        kv_blocks: (16..32).collect(),
        gdn_set: 2,
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
        sampling: sampling(0),
    }];
    let _ = exec.execute_mixed(
        &StepBatch {
            phase: Phase::Decode,
            bucket: Bucket::decode(1, WIDTH, 8192, true),
            step_index: 2,
            slots: &dslots,
            padded_slots: 0,
        },
        &StepBatch {
            phase: Phase::Prefill,
            bucket: Bucket::prefill(PREFILL, 8192, true),
            step_index: 2,
            slots: &cslots,
            padded_slots: 0,
        },
    );
    assert!(
        !exec.store_owed(),
        "a mixed call's chunk is collected in place"
    );
    let f = exec.forward_mut();
    assert_eq!(f.prompt_stores, 2, "both prompts stored exactly once");
    assert_eq!(f.abandoned, 0, "no stash was ever dropped: {:?}", f.calls);
}
