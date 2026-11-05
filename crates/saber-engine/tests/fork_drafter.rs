//! `n > 1` UNDER THE DRAFTER, on the host (workstream F38).
//!
//! V4 built the fork — one prefill, the prompt's whole KV pages shared read-only, the partial page
//! and the GDN slot set copied, each branch's first token a draw from the leader's row with its own
//! key — and refused it whenever a DFlash2 drafter was resident, because a branch then had no
//! drafter context. F38 gives it one, in three parts, and this file gates each as a LAW with the
//! device taken out:
//!
//! * **(a) its own drafter session** — a context slot that is not the leader's and a page run with
//!   no page in common with the leader's, holding a COPY of the leader's written pages
//!   (`DraftTier::fork`, the one law the device and this stub both call);
//! * **(b) its own first block** — the leader's prompt proposal was anchored on the LEADER's first
//!   token, and F37 makes a block anchored on a token the sequence did not commit a defect; so the
//!   pass proposes again for the branch, over the same context rows, from ITS token
//!   (`ForwardRunner::propose_fork`), and the executor offers it only if the echoed anchor is that
//!   token;
//! * **(c) park, restore and release of the branch's drafter run** — the same paths any session's
//!   drafter half takes, now reached by a branch.
//!
//! The stub below is `NullForward` with a DRAFTER bolted on whose pages are host memory: one u64
//! per drafter KV position, written by the "context store" from the tokens a step fed and by the
//! "backbone" from the anchor, and a block that is a function of the pages AND the anchor. So a
//! branch whose pages were not the leader's, or whose block was made from another anchor, proposes
//! a DIFFERENT block — which is what the comparisons below see. It mirrors the engine thread's
//! bookkeeping rules exactly as `device.rs` states them (a stash abandoned by anything but its
//! collection, the prompt proposal's inputs kept only until a forward runs), and it is checked
//! against a single request of the same prompt, which runs none of the fork's code.
//!
//! What it cannot gate is the bytes on a card; `tools/gates/f38_fork_card.sh` does that.

mod common;

use std::collections::BTreeMap;

use common::{drain, prompt_ids, PREFILL};
use saber_engine::draft_tier::{branch_is_distinct, DraftSession, DraftTier};
use saber_engine::{
    DecodeSlot, EngineDrafter, ForwardError, ForwardFacts, ForwardRunner, NullForward, ParkImage,
    Proposal, RealExecutor, RowRequest, StepOutcome, StepRows,
};
use saber_mem::ledger::{Ledger, Pool};
use saber_mem::park::{ImageDigests, RegionKind, RegionSpec};
use saber_plan::Bucket;
use saber_sched::{
    ArenaBytes, KvGeometry, ParkPolicy, Scheduler, SchedulerConfig, SeqId, TieredParkConfig,
    TieredParkStore,
};
use saber_serve::engine::{Engine, EngineRequest, ModelInfo, StepEvent, TokenId};
use saber_serve::sampling::SamplingParams;

/// A prompt of 40 tokens: two whole 16-token target pages and part of a third, and three drafter
/// pages once the first block of 8 is written past it.
const PROMPT: usize = 40;
/// The drafter's block (`1 + num_speculative_tokens`), as the serving profile runs it.
const BLOCK: u32 = 8;
/// Tokens a stub block proposes.
const DRAFT: usize = 3;
const MAX_SEQS: u32 = 4;

fn mix(a: u64, b: u64) -> u64 {
    let mut x = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^ (x >> 29)
}

/// What a step stashed for the proposal after the commit.
#[derive(Clone, Debug)]
enum Stash {
    /// A prompt's last chunk: its LAST TILE of context rows, `first ..`. The tiles before it
    /// were stored in the step, as the device stores them.
    Prefill {
        seq: SeqId,
        first: u32,
        rows: Vec<TokenId>,
    },
    /// A decode step: per slot, the first position and the tokens its committed rows FED.
    Decode {
        reqs: Vec<(SeqId, u32, Vec<TokenId>)>,
    },
}

impl Stash {
    fn seqs(&self) -> Vec<SeqId> {
        match self {
            Stash::Prefill { seq, .. } => vec![*seq],
            Stash::Decode { reqs } => reqs.iter().map(|r| r.0).collect(),
        }
    }
}

/// The prompt proposal's inputs, kept after its collection for the branches (the device's
/// `ForkTail`).
#[derive(Clone, Debug)]
struct Tail {
    leader: SeqId,
    first: u32,
    rows: Vec<TokenId>,
}

/// Context rows per tile, as the device's store tiles a prompt: the proposal re-stores only the
/// LAST one, so everything before it reaches a branch through the fork's copy or not at all.
const TILE: usize = 8;

/// The host drafter: a real `DraftTier` for the sessions and the fork law, host pages for K/V.
#[derive(Debug)]
struct Draft {
    tier: DraftTier,
    /// Tokens per drafter page.
    page: u32,
    /// Pages per session's run.
    per: u32,
    /// One u64 per position of every page in the pool.
    pool: Vec<u64>,
    free_runs: Vec<u32>,
    run_of: BTreeMap<SeqId, u32>,
    prompt_of: BTreeMap<SeqId, Vec<TokenId>>,
    stash: Option<Stash>,
    tail: Option<Tail>,
    /// Slots whose stash was abandoned (the device's `DraftCounters::uncollected`).
    uncollected: u64,
    /// Every proposal made, in order: `(sequence, anchor fed, block)`.
    made: Vec<(SeqId, TokenId, Vec<TokenId>)>,
    /// Branches forked: `(leader, branch)`.
    forks: Vec<(SeqId, SeqId)>,
    /// Drafter runs imaged by a park, by the sequence parked, and published by a restore.
    parked: Vec<(SeqId, Vec<u64>)>,
    restored: Vec<(SeqId, Vec<u64>)>,
    /// THE CONTROL: hand the tier the LEADER's own session as the branch's.
    reuse_leader_session: bool,
    /// THE CONTROL: feed a branch's first block the LEADER's anchor instead of its own.
    feed_leader_anchor: bool,
    /// THE CONTROL: zero the branch's run instead of copying the leader's pages into it.
    skip_copy: bool,
}

impl Draft {
    fn new() -> Self {
        let tier = DraftTier::plan(1, BLOCK, 4096, MAX_SEQS).expect("a tier");
        let page = tier.kv_cache().block_size;
        let per = tier.kv_blocks() / MAX_SEQS;
        Draft {
            pool: vec![0; (tier.kv_blocks() * page) as usize],
            tier,
            page,
            per,
            free_runs: (0..MAX_SEQS).rev().collect(),
            run_of: BTreeMap::new(),
            prompt_of: BTreeMap::new(),
            stash: None,
            tail: None,
            uncollected: 0,
            made: Vec::new(),
            forks: Vec::new(),
            parked: Vec::new(),
            restored: Vec::new(),
            reuse_leader_session: false,
            feed_leader_anchor: false,
            skip_copy: false,
        }
    }

    fn pages(&self, run: u32) -> Vec<u32> {
        (run * self.per..(run + 1) * self.per).collect()
    }

    fn span(&self, seq: SeqId) -> std::ops::Range<usize> {
        let first = self.tier.session(seq).expect("a drafter session").kv_blocks[0];
        let at = (first * self.page) as usize;
        at..at + (self.per * self.page) as usize
    }

    /// The whole run a session's drafter holds.
    fn run(&self, seq: SeqId) -> Vec<u64> {
        self.pool[self.span(seq)].to_vec()
    }

    fn admit(&mut self, seq: SeqId) -> Result<(), ForwardError> {
        let run = self.free_runs.pop().ok_or(ForwardError::AtCapacity {
            live: self.run_of.len() as u32,
            max: MAX_SEQS,
        })?;
        let pages = self.pages(run);
        if let Err(e) = self.tier.admit(seq, &pages) {
            self.free_runs.push(run);
            return Err(e);
        }
        self.run_of.insert(seq, run);
        let span = self.span(seq);
        self.pool[span].fill(0);
        self.prompt_of.insert(seq, Vec::new());
        Ok(())
    }

    fn release(&mut self, seq: SeqId) {
        self.tier.release(seq);
        if let Some(run) = self.run_of.remove(&seq) {
            self.free_runs.push(run);
        }
        self.prompt_of.remove(&seq);
    }

    /// The device's two rules at the engine thread's door: a stash outlives only its collection
    /// and the dense fallback, the fork tail only the fork's own requests.
    fn door(&mut self, forward: bool, keeps_stash: bool) {
        if !keeps_stash {
            if let Some(s) = self.stash.take() {
                self.uncollected += s.seqs().len() as u64;
            }
        }
        if forward {
            self.tail = None;
        }
    }

    /// The proposal: the context store writes the fed rows, the backbone writes the block's
    /// positions from the anchor, and the block is a function of every position it reads.
    fn propose(&mut self, seq: SeqId, first: u32, fed: &[TokenId], anchor: TokenId) -> Proposal {
        let span = self.span(seq);
        let base = span.start;
        for (i, t) in fed.iter().enumerate() {
            let pos = first as usize + i;
            self.pool[base + pos] = mix(u64::from(*t), pos as u64);
        }
        let end = first as usize + fed.len();
        for off in 0..BLOCK as usize {
            let fill = if off == 0 { u64::from(anchor) } else { 0xB10C };
            self.pool[base + end + off] = mix(fill, (end + off) as u64);
        }
        let digest = self.pool[base..base + end + BLOCK as usize]
            .iter()
            .fold(0u64, |a, v| mix(a, *v));
        let block: Vec<TokenId> = (0..DRAFT as u64)
            .map(|i| 5000 + (mix(digest ^ u64::from(anchor), i) % 1000) as TokenId)
            .collect();
        self.made.push((seq, anchor, block.clone()));
        Proposal { seq, anchor, block }
    }
}

/// `NullForward` with the host drafter.
#[derive(Debug)]
struct Drafted {
    null: NullForward,
    d: Draft,
}

impl Drafted {
    fn new() -> Self {
        Drafted {
            null: NullForward::new(&[PREFILL], &[1, 2, 4], MAX_SEQS),
            d: Draft::new(),
        }
    }
}

fn image(layout: Vec<RegionSpec>, bytes: Vec<u8>, resident: u32, committed: u32) -> ParkImage {
    let total = bytes.len() as u64;
    let chunks = saber_mem::park::chunk_digests(&bytes, 2);
    ParkImage {
        resident,
        committed,
        digests: ImageDigests {
            layout,
            bytes: total,
            chunk_bytes: saber_mem::park::DIGEST_CHUNK as u32,
            sha256: saber_mem::park::fold_chunk_digests(total as usize, &chunks),
            chunks,
        },
        bytes,
    }
}

impl ForwardRunner for Drafted {
    fn facts(&self) -> &ForwardFacts {
        self.null.facts()
    }
    fn admit(
        &mut self,
        seq: SeqId,
        prompt_len: u32,
        kv_blocks: &[u32],
    ) -> Result<(), ForwardError> {
        self.d.door(false, false);
        self.null.admit(seq, prompt_len, kv_blocks)?;
        if let Err(e) = self.d.admit(seq) {
            self.null.release(seq);
            return Err(e);
        }
        Ok(())
    }
    fn release(&mut self, seq: SeqId) {
        self.d.door(false, false);
        self.d.release(seq);
        self.null.release(seq);
    }
    fn fork(
        &mut self,
        leader: SeqId,
        branch: SeqId,
        kv_blocks: &[u32],
        shared_prefix: u32,
        cow_source: Option<u32>,
    ) -> Result<(), ForwardError> {
        self.d.door(false, false);
        self.null
            .fork(leader, branch, kv_blocks, shared_prefix, cow_source)?;
        // The drafter half, through the SAME law the engine thread calls.
        let d = &mut self.d;
        let prompt = d.prompt_of.get(&leader).cloned().unwrap_or_default();
        let resident = prompt.len() as u32 + BLOCK;
        let forked = if d.reuse_leader_session {
            let lead = d.tier.session(leader).cloned().expect("the leader");
            d.tier.fork(leader, branch, &lead.kv_blocks, resident)
        } else {
            match d.free_runs.pop() {
                None => Err(ForwardError::AtCapacity {
                    live: d.run_of.len() as u32,
                    max: MAX_SEQS,
                }),
                Some(run) => {
                    let pages = d.pages(run);
                    let r = d.tier.fork(leader, branch, &pages, resident);
                    if r.is_ok() {
                        d.run_of.insert(branch, run);
                    } else {
                        d.free_runs.push(run);
                    }
                    r
                }
            }
        };
        let copy = match forked {
            Ok(c) => c,
            Err(e) => {
                self.null.release(branch);
                return Err(e);
            }
        };
        // The copy plan applied to host pages: the leader's first `copy` pages, then zeros.
        let (p, per) = (d.page as usize, d.per as usize);
        let from = copy.from_first as usize * p;
        let to = copy.to_first as usize * p;
        let n = copy.copy as usize * p;
        let src: Vec<u64> = if d.skip_copy {
            vec![0; n]
        } else {
            d.pool[from..from + n].to_vec()
        };
        d.pool[to..to + n].copy_from_slice(&src);
        d.pool[to + n..to + per * p].fill(0);
        d.prompt_of.insert(branch, prompt);
        d.forks.push((leader, branch));
        Ok(())
    }
    fn can_fork(&self) -> bool {
        true
    }
    fn propose_fork(
        &mut self,
        leader: SeqId,
        branch: SeqId,
        anchor: TokenId,
    ) -> Result<Option<Proposal>, ForwardError> {
        self.d.door(false, false);
        let tail = self.d.tail.clone().ok_or_else(|| {
            ForwardError::Device(format!("{branch}: no prompt proposal is being kept"))
        })?;
        if tail.leader != leader {
            return Err(ForwardError::Device(format!(
                "{branch}: the kept proposal is {}'s",
                tail.leader
            )));
        }
        let fed_anchor = if self.d.feed_leader_anchor {
            self.d
                .made
                .iter()
                .find(|m| m.0 == leader)
                .map_or(anchor, |m| m.1)
        } else {
            anchor
        };
        Ok(Some(
            self.d.propose(branch, tail.first, &tail.rows, fed_anchor),
        ))
    }
    fn prefill(
        &mut self,
        seq: SeqId,
        chunk: &[TokenId],
        done: u32,
        last: bool,
        sampling: &RowRequest,
    ) -> Result<StepOutcome, ForwardError> {
        self.d.door(true, false);
        let mut out = self.null.prefill(seq, chunk, done, last, sampling)?;
        let prompt = self.d.prompt_of.entry(seq).or_default();
        prompt.extend_from_slice(chunk);
        // Every row but the last chunk's last tile is stored IN the step.
        let keep = if last { TILE.min(chunk.len()) } else { 0 };
        let stored = chunk.len() - keep;
        let base = self.d.span(seq).start;
        for (i, t) in chunk[..stored].iter().enumerate() {
            let pos = done as usize + i;
            self.d.pool[base + pos] = mix(u64::from(*t), pos as u64);
        }
        if last {
            self.d.stash = Some(Stash::Prefill {
                seq,
                first: done + stored as u32,
                rows: chunk[stored..].to_vec(),
            });
            out.proposal_pending = vec![seq];
        }
        Ok(out)
    }
    fn decode(
        &mut self,
        slots: &[DecodeSlot],
        batch: u32,
        width: u32,
    ) -> Result<StepOutcome, ForwardError> {
        self.d.door(true, false);
        let mut out = self.null.decode(slots, batch, width)?;
        let reqs = slots
            .iter()
            .zip(&out.commits)
            .map(|(s, c)| {
                let mut fed = vec![s.input];
                fed.extend_from_slice(&s.draft);
                fed.truncate(c.tokens.len());
                (s.seq, s.context_len, fed)
            })
            .collect::<Vec<_>>();
        out.proposal_pending = reqs.iter().map(|r| r.0).collect();
        self.d.stash = Some(Stash::Decode { reqs });
        Ok(out)
    }
    fn propose_after_commit(
        &mut self,
        anchors: &[(SeqId, TokenId)],
    ) -> Result<Vec<Proposal>, ForwardError> {
        let Some(stash) = self.d.stash.take() else {
            return Err(ForwardError::Device("nothing is pending".to_owned()));
        };
        if !stash.seqs().iter().copied().eq(anchors.iter().map(|a| a.0)) {
            self.d.uncollected += stash.seqs().len() as u64;
            return Err(ForwardError::Device("anchors for other slots".to_owned()));
        }
        Ok(match stash {
            Stash::Prefill { seq, first, rows } => {
                let p = self.d.propose(seq, first, &rows, anchors[0].1);
                self.d.tail = Some(Tail {
                    leader: seq,
                    first,
                    rows,
                });
                vec![p]
            }
            Stash::Decode { reqs } => reqs
                .into_iter()
                .zip(anchors)
                .map(|((seq, first, fed), (_, a))| self.d.propose(seq, first, &fed, *a))
                .collect(),
        })
    }
    fn declared_launches(&self, rows: StepRows) -> Option<u32> {
        self.null.declared_launches(rows)
    }
    fn dense_row(&mut self, row: u32) -> Result<Vec<f32>, ForwardError> {
        self.d.door(false, true);
        self.null.dense_row(row)
    }
    fn can_park(&self) -> bool {
        self.null.can_park()
    }
    /// The target's image with the drafter's run appended as one `DraftKv` region, as the
    /// device's image carries it.
    fn park(&mut self, seq: SeqId) -> Result<ParkImage, ForwardError> {
        self.d.door(true, false);
        let target = self.null.park(seq)?;
        let run = self.d.run(seq);
        self.d.parked.push((seq, run.clone()));
        let mut layout = target.digests.layout.clone();
        let mut bytes = target.bytes.clone();
        let draft: Vec<u8> = run.iter().flat_map(|v| v.to_le_bytes()).collect();
        layout.push(RegionSpec {
            kind: RegionKind::DraftKv,
            len: draft.len() as u64,
        });
        bytes.extend_from_slice(&draft);
        Ok(image(layout, bytes, target.resident, target.committed))
    }
    fn restore(&mut self, seq: SeqId, image_in: &ParkImage) -> Result<(), ForwardError> {
        self.d.door(true, false);
        let mut layout = image_in.digests.layout.clone();
        let last = layout
            .pop()
            .ok_or_else(|| ForwardError::Image("empty".to_owned()))?;
        if last.kind != RegionKind::DraftKv {
            return Err(ForwardError::Image(
                "an image with no drafter half, for a session with a drafter".to_owned(),
            ));
        }
        let cut = image_in.bytes.len() - last.len as usize;
        let target = image(
            layout,
            image_in.bytes[..cut].to_vec(),
            image_in.resident,
            image_in.committed,
        );
        self.null.restore(seq, &target)?;
        let run: Vec<u64> = image_in.bytes[cut..]
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect();
        let span = self.d.span(seq);
        if run.len() != span.len() {
            return Err(ForwardError::Image(
                "a drafter run of another length".to_owned(),
            ));
        }
        self.d.pool[span].copy_from_slice(&run);
        self.d.restored.push((seq, run));
        Ok(())
    }
}

// ── the scheduler over it ───────────────────────────────────────────────────────────────────────

fn ledger() -> Ledger {
    const GIB: u64 = 1 << 30;
    let mut l = Ledger::new();
    l.declare(Pool::Vram(0), 32 * GIB).expect("rank 0");
    l.declare(Pool::HostPinned, 4 * GIB).expect("pinned");
    l.declare(Pool::HostPageable, 48 * GIB).expect("pageable");
    l.declare(Pool::Nvme, 16 * GIB).expect("nvme");
    l
}

fn scheduler(forward: Drafted) -> Scheduler<RealExecutor<Drafted>> {
    let f = forward.facts();
    let kv = KvGeometry {
        block_tokens: 16,
        bytes_per_token_per_rank: f.kv_block_bytes / 16,
    };
    let gdn = f.gdn_bytes_per_slot;
    let pool = f.kv_pool_blocks;
    let mut buckets: Vec<Bucket> = [1u32, 2, 4]
        .iter()
        .map(|b| Bucket::decode(*b, 1, 8192, true))
        .collect();
    buckets.push(Bucket::prefill(PREFILL, 8192, true));
    let config = SchedulerConfig::for_mock(
        buckets,
        1,
        gdn,
        kv,
        ArenaBytes {
            kv_per_rank: u64::from(pool) * kv.block_bytes(),
            gdn_state_per_rank: 16 * gdn,
        },
        65_536,
        "test:fork-drafter".to_owned(),
    )
    .expect("config")
    .with_park_policy(ParkPolicy::default());
    let exec = RealExecutor::with_drafter(forward, Box::new(EngineDrafter::new(DRAFT as u32)));
    Scheduler::new(
        config,
        exec,
        ModelInfo {
            served_model_name: "Qwen3.8".to_owned(),
            aliases: Vec::new(),
            max_model_len: 65_536,
            root: "/packs/test".to_owned(),
            eos_token_ids: vec![151_645],
        },
        ledger(),
    )
    .expect("arenas fit")
    .with_park_tier(Box::new(
        TieredParkStore::host(&TieredParkConfig::ram(1 << 20, 16 << 20).hashing_on(2))
            .expect("ram store"),
    ))
}

fn params(n: usize, temperature: f32, seed: Option<u64>, max_tokens: usize) -> SamplingParams {
    SamplingParams {
        n,
        temperature,
        top_p: 0.95,
        top_k: 20,
        seed,
        max_tokens: Some(max_tokens),
        ..SamplingParams::default()
    }
}

fn choices(events: &[StepEvent]) -> BTreeMap<u32, Vec<TokenId>> {
    let mut out: BTreeMap<u32, Vec<TokenId>> = BTreeMap::new();
    for e in events {
        out.entry(e.choice_index)
            .or_default()
            .extend(e.new_tokens.iter().copied());
    }
    out
}

/// One request through a fresh scheduler, optionally parked and restored mid-stream. Returns the
/// streams by choice, the stub's drafter record and the executor's counters.
async fn run(
    p: SamplingParams,
    park_after: Option<usize>,
    setup: impl FnOnce(&mut Draft),
) -> (
    BTreeMap<u32, Vec<TokenId>>,
    Draft,
    saber_engine::EngineCounters,
) {
    let sched = scheduler(Drafted::new());
    sched.with_executor(|e| setup(&mut e.forward_mut().d));
    let sub = sched
        .submit(EngineRequest {
            prompt_token_ids: prompt_ids(PROMPT),
            params: p,
        })
        .expect("submit");
    if let Some(steps) = park_after {
        sched.run_steps(steps);
        sched.park(sub.id).expect("park");
        sched.run_steps(3);
        sched.unpark(sub.id).expect("unpark");
    }
    sched.run_steps(2000);
    let events = drain(sub).await;
    let counters = sched.with_executor(|e| e.counters());
    let d = sched.with_executor(|e| std::mem::replace(&mut e.forward_mut().d, Draft::new()));
    (choices(&events), d, counters)
}

/// The blocks proposed for one sequence, in order.
fn blocks_of(d: &Draft, seq: SeqId) -> Vec<(TokenId, Vec<TokenId>)> {
    d.made
        .iter()
        .filter(|m| m.0 == seq)
        .map(|m| (m.1, m.2.clone()))
        .collect()
}

/// **(a) + (b) at temperature 0: a branch is its own session, its pages are the leader's bytes,
/// and it proposes exactly what an independent request of the same prompt proposes.**
///
/// The independent request runs none of the fork's code — its drafter pages come from its own
/// context store — so "the branch's first block is the solo request's first block" can only hold
/// if the copy put the leader's prompt context into the branch's run AND the branch's block was
/// made from the branch's token. Every block after it too: the branch's later context stores land
/// on the copied prefix.
#[tokio::test]
async fn at_temperature_0_a_branch_is_an_independent_request_down_to_its_drafter_blocks() {
    let (group, d, c) = run(params(2, 0.0, None, 24), None, |_| {}).await;
    let (solo, ds, _) = run(params(1, 0.0, None, 24), None, |_| {}).await;
    assert_eq!(group.len(), 2);
    assert_eq!(group[&0], solo[&0], "the leader is the n = 1 stream");
    assert_eq!(group[&1], solo[&0], "and so is the branch");

    let (leader, branch) = d.forks[0];
    assert_eq!(d.forks.len(), 1);
    // The branch's blocks, every one, are the solo request's — the first included.
    let solo_blocks = blocks_of(&ds, SeqId(1));
    assert_eq!(blocks_of(&d, leader), solo_blocks);
    assert_eq!(blocks_of(&d, branch), solo_blocks);
    assert!(
        solo_blocks.len() > 3,
        "a block per step: {}",
        solo_blocks.len()
    );

    // The counters: every proposal collected, the branch's first block made and offered, no
    // stale anchor, no refusal.
    assert_eq!(d.uncollected, 0, "the uncollected-proposal counter stays 0");
    assert_eq!(c.forks, 1);
    assert_eq!(c.fork_blocks, 1);
    assert_eq!(c.fork_block_failures, 0);
    assert_eq!(c.stale_anchor_defects, 0);
    assert!(c.clean(), "{} / {}", c.line(), c.fork_line());
}

/// The same at the moment of the fork, where the pages can be read before either sequence moves
/// on: a DIFFERENT slot and a disjoint run, holding the leader's bytes page for page.
#[test]
fn at_the_fork_the_branchs_run_is_a_copy_of_the_leaders_in_pages_of_its_own() {
    use saber_plan::Phase;
    use saber_sched::{
        AdmittedSeq, BatchSlot, PrefillChunk, SlotSampling, StepBatch, StepExecutor,
    };
    use saber_serve::engine::RequestId;

    let mut exec =
        RealExecutor::with_drafter(Drafted::new(), Box::new(EngineDrafter::new(DRAFT as u32)));
    let admitted =
        |seq: u64, choice: u32, blocks: Vec<u32>, shared: u32, cow: Option<u32>| AdmittedSeq {
            request_id: RequestId(1),
            choice_index: choice,
            seq_id: SeqId(seq),
            prompt_tokens: prompt_ids(PROMPT),
            kv_blocks: blocks,
            shared_prefix_blocks: shared,
            cow_source: cow,
            gdn_set: seq as u32,
            gdn_slots: 2,
        };
    exec.admit(&admitted(1, 0, (0..8).collect(), 0, None));
    let slots = vec![BatchSlot {
        request_id: RequestId(1),
        seq_id: SeqId(1),
        slot: 0,
        kv_blocks: (0..8).collect(),
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
        sampling: SlotSampling {
            params: params(2, 0.0, None, 8),
            choice_index: 0,
            step_index: 0,
            position: 0,
            penalties: None,
        },
    }];
    let out = exec.execute(&StepBatch {
        phase: Phase::Prefill,
        bucket: Bucket::prefill(PREFILL, 8192, true),
        step_index: 0,
        slots: &slots,
        padded_slots: 0,
    });
    let leader_token = out.seqs[0].tokens[0];
    let mut branch_blocks: Vec<u32> = (0..2).collect();
    branch_blocks.extend(20..26);
    let token = exec
        .fork(SeqId(1), &admitted(2, 1, branch_blocks, 2, Some(2)))
        .expect("forked")
        .token;
    assert_eq!(token, leader_token, "temperature 0: one answer twice");

    let d = &exec.forward_mut().d;
    let lead = d.tier.session(SeqId(1)).expect("leader").clone();
    let br = d.tier.session(SeqId(2)).expect("branch").clone();
    assert_ne!(
        br.slot, lead.slot,
        "the branch's drafter context slot is its own"
    );
    assert!(
        br.kv_blocks.iter().all(|p| !lead.kv_blocks.contains(p)),
        "no drafter page in common"
    );
    assert_eq!(
        d.run(SeqId(2)),
        d.run(SeqId(1)),
        "the branch's pages are the leader's bytes"
    );
    // The branch's first anchor is ITS committed token.
    let first = blocks_of(d, SeqId(2));
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].0, token);
    assert_eq!(d.uncollected, 0);
    let c = exec.counters();
    assert_eq!((c.forks, c.fork_blocks, c.fork_block_failures), (1, 1, 0));
    assert!(c.clean(), "{}", c.fork_line());
}

/// **(b) above temperature 0: the branch's first anchor is ITS draw, not the leader's.**
///
/// A seed and a temperature under which the two branches' first tokens differ: the block the
/// branch is offered was fed its own token (the echo), the prompt positions of its run are the
/// leader's and the block positions are not, and not one anchor fed anywhere is a token its
/// sequence did not commit (`stale_anchor_defects` 0 over the whole run).
#[tokio::test]
async fn above_temperature_0_each_branch_proposes_from_its_own_first_token() {
    let mut found = None;
    for seed in 1..200u64 {
        let (group, d, c) = run(params(2, 1.5, Some(seed), 12), None, |_| {}).await;
        if group[&0][0] != group[&1][0] {
            found = Some((group, d, c));
            break;
        }
    }
    let (group, d, c) = found.expect("a seed whose two first draws differ");
    let (leader, branch) = d.forks[0];
    let lead_first = blocks_of(&d, leader)[0].0;
    let branch_first = blocks_of(&d, branch)[0].0;
    assert_eq!(lead_first, group[&0][0], "the leader's anchor is its token");
    assert_eq!(
        branch_first, group[&1][0],
        "the branch's anchor is ITS token"
    );
    assert_ne!(lead_first, branch_first);
    assert_eq!(c.stale_anchor_defects, 0, "{}", c.proposal_line());
    assert_eq!(c.fork_blocks, 1);
    assert_eq!(d.uncollected, 0);
    assert!(c.clean(), "{}", c.fork_line());
}

/// **CONTROL for (b)**: a pass that feeds the branch's first block the LEADER's anchor — the
/// stale block F37's law exists for — is caught at the fork as a defect, never offered, and the
/// run is unclean. The case above differs from this in exactly the anchor.
#[tokio::test]
async fn control_a_branch_block_fed_the_leaders_anchor_is_a_stale_anchor_defect() {
    let mut caught = false;
    for seed in 1..200u64 {
        let (group, _, c) = run(params(2, 1.5, Some(seed), 12), None, |d| {
            d.feed_leader_anchor = true;
        })
        .await;
        if group[&0][0] != group[&1][0] {
            assert_eq!(c.stale_anchor_defects, 1, "{}", c.proposal_line());
            assert!(!c.clean(), "a stale block is an unclean run");
            caught = true;
            break;
        }
    }
    assert!(caught, "a seed whose two first draws differ");
}

/// **CONTROL for (a)**: a branch handed its LEADER's drafter session is refused by name — the
/// fork fails, the branch is aborted, and the refusal is counted. The law itself, with the slot
/// alone reused, is refused by name too.
#[tokio::test]
async fn control_a_branch_that_reuses_the_leaders_drafter_session_is_refused_by_name() {
    let (group, _, c) = run(params(2, 0.0, None, 8), None, |d| {
        d.reuse_leader_session = true;
    })
    .await;
    assert_eq!(c.fork_refusals, 1, "{}", c.fork_line());
    assert!(!c.clean());
    assert!(
        group.get(&1).is_none_or(|v| v.is_empty()),
        "no branch was served"
    );

    let lead = DraftSession {
        slot: 3,
        kv_blocks: (0..4).collect(),
        done: 40,
        rope: None,
    };
    let same_slot = DraftSession {
        slot: 3,
        kv_blocks: (4..8).collect(),
        done: 40,
        rope: None,
    };
    let err = branch_is_distinct(SeqId(2), &lead, &same_slot).expect_err("the leader's slot");
    assert!(
        err.to_string()
            .contains("REUSES its leader's drafter context slot"),
        "{err}"
    );
}

/// **CONTROL for the copy**: a branch whose run is ZEROED instead of given the leader's pages —
/// the fork as it would be with an empty drafter context, which is why V4 refused it — proposes
/// blocks that are NOT the independent request's, from the first one on, while its token stream
/// (which `draft::Greedy` protects) does not move. The case above differs from this in exactly
/// the copy; this is what an ids-only gate cannot see.
#[tokio::test]
async fn control_a_branch_without_the_leaders_pages_proposes_other_blocks() {
    let (group, d, c) = run(params(2, 0.0, None, 24), None, |d| d.skip_copy = true).await;
    let (solo, ds, _) = run(params(1, 0.0, None, 24), None, |_| {}).await;
    assert_eq!(group[&1], solo[&0], "the ids cannot see it");
    let (_, branch) = d.forks[0];
    let solo_blocks = blocks_of(&ds, SeqId(1));
    let branch_blocks = blocks_of(&d, branch);
    assert_ne!(
        branch_blocks[0], solo_blocks[0],
        "the first block reads the missing context"
    );
    assert!(
        branch_blocks.iter().zip(&solo_blocks).all(|(a, b)| a != b),
        "and so does every block after it"
    );
    assert!(
        c.clean(),
        "nothing counts it — which is why the case compares blocks"
    );
}

/// **(c) park and restore of a branch round-trip its drafter pages, and the stream does not
/// move.** The group is parked mid-stream (both branches — a group is a victim as one unit),
/// restored into fresh sessions, and finishes; every run the park imaged is the run the restore
/// published, and every stream and every block after it is the unparked run's.
#[tokio::test]
async fn park_and_restore_of_a_forked_branch_round_trips_its_drafter_pages() {
    for (temperature, seed) in [(0.0f32, None), (0.7, Some(4242u64))] {
        let (plain, dp, _) = run(params(2, temperature, seed, 24), None, |_| {}).await;
        let (parked, d, c) = run(params(2, temperature, seed, 24), Some(6), |_| {}).await;
        assert_eq!(
            plain, parked,
            "temperature {temperature}: a park moved a stream"
        );
        assert_eq!(d.parked.len(), 2, "both branches imaged");
        assert_eq!(d.restored.len(), 2, "both restored");
        for (i, (seq, run)) in d.parked.iter().enumerate() {
            assert!(
                run.iter().any(|v| *v != 0),
                "{seq}: a run with context in it"
            );
            assert_eq!(
                &d.restored[i].1, run,
                "{seq}: the restore published what the park imaged"
            );
        }
        // The blocks proposed after the restore are the unparked run's, choice for choice: the
        // restored drafter context is the one the stream had.
        let tail = |d: &Draft| -> Vec<Vec<TokenId>> {
            d.made.iter().rev().take(4).map(|m| m.2.clone()).collect()
        };
        assert_eq!(tail(&d), tail(&dp), "temperature {temperature}");
        assert_eq!(c.parks, 2);
        assert_eq!(c.restores, 2);
        assert_eq!(d.uncollected, 0);
        assert!(c.clean(), "{} / {}", c.line(), c.park_line());
    }
}

/// `n = 4`: three branches, three sessions of their own, three first blocks, one prefill.
#[tokio::test]
async fn four_branches_are_three_forks_three_first_blocks_and_one_prefill() {
    let (group, d, c) = run(params(4, 0.0, None, 12), None, |_| {}).await;
    let (solo, _, _) = run(params(1, 0.0, None, 12), None, |_| {}).await;
    assert_eq!(group.len(), 4);
    for ids in group.values() {
        assert_eq!(ids, &solo[&0]);
    }
    assert_eq!(c.prefill_steps, 1);
    assert_eq!((c.forks, c.fork_blocks), (3, 3));
    let slots: std::collections::BTreeSet<SeqId> = d.forks.iter().map(|f| f.1).collect();
    assert_eq!(slots.len(), 3);
    assert_eq!(d.uncollected, 0);
    assert!(c.clean(), "{}", c.fork_line());
}

/// **A group parks and restores through the ASYNCHRONOUS seam with ONE staging slot** (ws-f38).
///
/// Every serving engine parks off the step loop with one staging slot, and until this round a
/// group of `n > 1` could not be parked there at all: the second branch's `park_begin` answered
/// `Busy` (the first branch held the slot), the scheduler refused the whole group as
/// `TransferBusy` — leaving the first branch's copy in flight — and a restore did the same on its
/// second branch. The card's first fork run read `TransferBusy` on every group. The branches now
/// QUEUE behind each other (`Transfer::ParkQueued` / `RestoreQueued`) and the group completes when
/// every branch has; the streams are the unparked group's, at `n = 2` and `n = 4`.
#[tokio::test]
async fn a_group_parks_and_restores_through_one_async_staging_slot() {
    for n in [2usize, 4] {
        let mut runs = Vec::new();
        for park in [false, true] {
            let sched = common::scheduler_async_park(64, 3);
            let sub = sched
                .submit(EngineRequest {
                    prompt_token_ids: prompt_ids(PROMPT),
                    params: params(n, 0.7, Some(99), 24),
                })
                .expect("submit");
            if park {
                sched.run_steps(4);
                sched.park(sub.id).expect("the group's park starts");
                // A step with nothing steppable is idle and `run_steps` returns; each call still
                // polls the transfers once, which is what moves a queued branch along.
                for _ in 0..200 {
                    if !sched.parked_ids().is_empty() {
                        break;
                    }
                    sched.run_steps(1);
                }
                assert_eq!(
                    sched.parked_ids(),
                    vec![sub.id],
                    "n = {n}: the whole group parked"
                );
                sched.unpark(sub.id).expect("unpark");
            }
            for _ in 0..2000 {
                if !sched.has_work() {
                    break;
                }
                sched.run_steps(8);
            }
            let events = drain(sub).await;
            let c = sched.with_executor(|e| e.counters());
            runs.push((choices(&events), c));
        }
        let (plain, _) = &runs[0];
        let (parked, c) = &runs[1];
        assert_eq!(plain.len(), n);
        assert_eq!(
            parked, plain,
            "n = {n}: a parked and restored group is the unparked group"
        );
        assert_eq!(c.parks, n as u64, "n = {n}: {}", c.park_line());
        assert_eq!(c.restores, n as u64, "n = {n}: {}", c.park_line());
        assert!(c.clean(), "{} / {}", c.line(), c.park_line());
    }
}
