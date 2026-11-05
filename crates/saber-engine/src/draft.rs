//! Speculation through the seam: who proposes the draft tokens, and what survives verification.
//!
//! The engine has verified one token per step since P3. A verify step runs `w = k + 1` rows per
//! sequence instead of one — row 0 feeds the token the last step committed and row `i` feeds draft
//! token `i - 1` — and commits somewhere between 1 and `w` tokens depending on how many of the
//! drafter's guesses the model agreed with. Everything about that is here except the rows
//! themselves, which are [`crate::device`]'s.
//!
//! # Why the drafter is a trait with mocks in it
//!
//! ws-f is porting the DFlash2 drafter's kernels and they are not ready. Waiting for them would mean
//! discovering the scheduler's, the engine's and the serve plane's speculation bugs *after* the
//! kernels land, with two unfinished things to blame. So the drafter is a **trait**, and this module
//! ships two implementations whose acceptance is known in advance:
//!
//! * [`FixedDrafter`] proposes the same token `k` times. The model almost never agrees, so
//!   acceptance is 0 and every step commits exactly one token — the SPEC=0 answer, reached through
//!   the whole verify machinery.
//! * [`RecordedDrafter`] proposes the tokens a recorded SPEC=0 run actually produced, for its first
//!   `truthful` positions, and a token that cannot match after that. Acceptance is therefore
//!   **exactly `truthful`**, chosen per run anywhere in `0..=k`, which is what makes "the ids equal
//!   the SPEC=0 run's ids at every acceptance rate" a gate rather than a hope.
//!
//! Both are wrong in the way a real drafter is wrong — they propose tokens the model may reject —
//! and neither can make the engine commit a token the model did not produce, because [`Greedy`] is
//! what decides and the drafter never gets a vote.
//!
//! # The acceptance law, and why identity with SPEC=0 is a theorem rather than a test result
//!
//! At temperature 0 the law is a prefix comparison: `accepted` is the first position where the
//! model's row disagrees with the draft, and the step commits the model's rows `0..=accepted`.
//! Every committed token is therefore a token the MODEL produced at a position whose whole prefix
//! the model also produced, so the committed sequence is the greedy sequence — whatever the drafter
//! guessed. Speculation at temperature 0 buys steps, never tokens.
//!
//! That is why `tokens.len() == accepted + 1` always, which is the contract `saber_sched::SeqStep`
//! states and counts: the accepted prefix plus **one more** — the recovery token at the rejected
//! position, or the bonus token when every draft token survived. the reference server's
//! `_insert_resampled_kernel` increments `num_sampled` unconditionally for the same reason
//! (`docs/abi/DFLASH2-SEMANTICS.md` § 1.4).
//!
//! # What this module is NOT
//!
//! It is not a sampler. Above temperature 0 the law is the rejection sampler
//! (accept with probability `min(1, p/q)`, resample from `max(0, p - q)` normalised), which needs
//! the draft distribution and not just the draft token, and the engine's head returns an argmax.
//! [`Greedy`] says so by name rather than by producing a plausible answer, and
//! [`Acceptance`] is a trait so that the day a distribution is available the law is a new
//! implementation and not an edit to this one.
//!
//! Since ws-f40 a SAMPLED slot runs through [`Greedy`] too, over different targets: each row's
//! target is the host sampler's draw at that row's output position (`crate::forward::drawn_rows`),
//! not the argmax. With a greedy (one-hot) drafter that is the rejection sampler in distribution —
//! accept with probability `p(draft)`, else a draw from `p` without the draft — and, unlike it, a
//! function of `(seed, position, choice)` alone (`docs/EXEC.md`, "the sampler's draw law").
//!
//! **The law itself is `saber_exec::forward::accept_prefix`** since ws-h18: [`Greedy`] is a thin
//! call into it plus the translation between two conventions (that one counts the row-0 input as
//! part of the draft; this one counts only the guesses, because `saber_sched::SeqStep`'s tokens are
//! the NEW ones). The trait stays because the engine has to be able to name which law ran, and
//! because a temperature > 0 sampler is then a second implementation rather than an edit to the
//! first. These tests stay too: they pin THIS convention, which is the one the scheduler acts on.

use std::collections::BTreeMap;

use saber_exec::forward;
use saber_sched::SeqId;
use saber_serve::engine::TokenId;

/// Why a verify step's rows could not be turned into committed tokens.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AcceptError {
    /// The rows and the draft do not describe one step.
    #[error(
        "a verify step of {draft} draft token(s) produces {want} row(s) and this one produced \
         {rows}: row 0 feeds the last committed token and row i feeds draft token i-1"
    )]
    Rows {
        /// Draft tokens proposed.
        draft: usize,
        /// Rows that implies.
        want: usize,
        /// Rows given.
        rows: usize,
    },
    /// No rows at all: a step that sampled nothing cannot commit anything.
    #[error("a verify step with no rows: there is no token to commit")]
    Empty,
    /// The shared law refused. Its own invariants, in its own words.
    #[error("the acceptance law: {0}")]
    Law(String),
}

/// What one sequence's verify step committed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verdict {
    /// Draft tokens accepted, **excluding** the bonus token. A prefix: it stops at the first
    /// rejection and never resumes.
    pub accepted: u32,
    /// The tokens this step commits, in order. Always `accepted + 1` of them.
    pub committed: Vec<TokenId>,
}

impl Verdict {
    /// The contract `saber_sched::SeqStep` states, checked here so a law that broke it is caught
    /// where it broke rather than where it was counted.
    fn checked(accepted: u32, committed: Vec<TokenId>) -> Verdict {
        debug_assert_eq!(
            committed.len(),
            accepted as usize + 1,
            "a verified step commits the accepted prefix plus exactly one more"
        );
        Verdict {
            accepted,
            committed,
        }
    }
}

/// How a verify step's rows become committed tokens.
pub trait Acceptance: Send + std::fmt::Debug {
    /// A name for the report and for the which-path counter.
    fn name(&self) -> &'static str;

    /// Decide. `rows` is the model's token at each of the `draft.len() + 1` rows, in row order.
    ///
    /// # Errors
    /// A row count that does not match the draft, or no rows at all.
    fn accept(&self, draft: &[TokenId], rows: &[TokenId]) -> Result<Verdict, AcceptError>;
}

/// Temperature 0: accept the longest prefix on which the model and the draft agree.
#[derive(Clone, Copy, Debug, Default)]
pub struct Greedy;

impl Acceptance for Greedy {
    fn name(&self) -> &'static str {
        "greedy"
    }

    /// The shared law, translated across the one place the two conventions differ.
    ///
    /// `saber_exec::forward::accept_prefix` counts the **row 0 input** as part of the draft: its
    /// `draft` is `[input, d0, d1, …]`, its `accepted` starts at 1 (row 0 is accepted by
    /// construction), and its `committed` begins with that input. This trait's convention is
    /// `saber_sched::SeqStep`'s — `draft` is the guesses only, `accepted` counts guesses that
    /// survived, and `tokens` are the NEW tokens, because the input was committed by the step
    /// before and committing it twice would duplicate it on the stream.
    ///
    /// So the translation is exactly: prepend the input going in, subtract one from `accepted` and
    /// drop the first token coming back. It is written out rather than absorbed because a seam
    /// between two off-by-one conventions is where an off-by-one lives, and both are named here.
    fn accept(&self, draft: &[TokenId], rows: &[TokenId]) -> Result<Verdict, AcceptError> {
        if rows.is_empty() {
            return Err(AcceptError::Empty);
        }
        if rows.len() != draft.len() + 1 {
            return Err(AcceptError::Rows {
                draft: draft.len(),
                want: draft.len() + 1,
                rows: rows.len(),
            });
        }
        // The input token's own value is irrelevant to the law — it is only ever `draft[0]`, which
        // `accept_prefix` accepts by construction and never compares — so any id will do, and 0
        // says "this slot is not a guess" rather than pretending to be one.
        let mut theirs: Vec<i32> = Vec::with_capacity(rows.len());
        theirs.push(0);
        theirs.extend(draft.iter().map(|t| *t as i32));
        let answers: Vec<i32> = rows.iter().map(|t| *t as i32).collect();
        let a = forward::accept_prefix(&theirs, &answers)
            .map_err(|e| AcceptError::Law(e.to_string()))?;
        // Their `accepted` counts the input; ours counts guesses. Their `committed[0]` IS the input.
        let accepted = (a.accepted - 1) as u32;
        let committed: Vec<TokenId> = a.committed[1..].iter().map(|t| *t as TokenId).collect();
        Ok(Verdict::checked(accepted, committed))
    }
}

/// What proposes the draft tokens a verify step checks.
///
/// The engine calls [`Drafter::propose`] once per sequence per step, runs the verify, and then tells
/// the drafter what actually committed so a stateful one can advance. A drafter that proposes fewer
/// tokens than asked is reporting a narrower width, which is legal and counted — the scheduler's
/// `draft_tokens` is the **scheduled** width and its acceptance EMA is computed against that.
pub trait Drafter: Send + std::fmt::Debug {
    /// A name for the report and for the which-path counter.
    fn name(&self) -> &'static str;

    /// The widest proposal this drafter can make, whatever the scheduler asks for.
    fn max_width(&self) -> u32;

    /// Up to `width` draft tokens for `seq`, whose last committed token is `last`.
    fn propose(&mut self, seq: SeqId, last: TokenId, width: u32) -> Vec<TokenId>;

    /// What the step actually committed, in order. Called once per sequence per step, after the
    /// verify, including when nothing was accepted.
    fn commit(&mut self, _seq: SeqId, _committed: &[TokenId]) {}

    /// A proposal the **pass itself** made for this sequence's next step
    /// ([`crate::forward::StepOutcome::drafts`]).
    ///
    /// Ignored by default, which is the honest answer for a drafter that proposes out of its own
    /// state: a mock has no use for what the device thought. [`EngineDrafter`] is the one
    /// implementation that takes it, and the split exists because of WHEN a real DFlash2 proposal
    /// can run. It reads the target's hidden states at the five tapped layers for the rows the step
    /// just committed, and those live in the target's arena only until the next step overwrites
    /// them — so the proposal has to happen inside the step that produced them, on the HIP thread,
    /// and the host-side drafter can only be a mailbox for what it produced. A `propose` that went
    /// to the device between two engine calls would be reading whatever step ran in between, and a
    /// prefill for another sequence is exactly such a step.
    ///
    /// Called once per sequence per step, after [`Drafter::commit`], and only when the pass
    /// proposed anything at all.
    fn offer(&mut self, _seq: SeqId, _draft: &[TokenId]) {}

    /// Drop a sequence's drafting state. Its slots go to the next sequence.
    fn release(&mut self, _seq: SeqId) {}

    /// The sequence's whole prompt, once, when it is admitted (FN69: the prompt-lookup source
    /// indexes it). Ignored by default.
    fn prompt(&mut self, _seq: SeqId, _ids: &[TokenId]) {}

    /// The token a decode step feeds for `seq` (its last committed token, the host sampler's draw
    /// on a sampled row), every decode step, before [`Drafter::propose`] and whether or not the
    /// step drafts. Ignored by default (FN69: the lookup's history follows it).
    fn input(&mut self, _seq: SeqId, _last: TokenId) {}

    /// FN69: the prompt-lookup source's counters, when this drafter has one.
    fn lookup_counts(&self) -> Option<crate::lookup::LookupCounts> {
        None
    }
}

/// No speculation: the SPEC=0 path, through the same seam.
///
/// Not the same thing as "no drafter": the engine still runs a one-row step, still applies the
/// acceptance law to a one-row verdict, and still reports `draft_tokens = 0`. That is what makes
/// SPEC=0 a special case of the verify path rather than a separate one nobody exercises.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoDrafter;

impl Drafter for NoDrafter {
    fn name(&self) -> &'static str {
        "none"
    }

    fn max_width(&self) -> u32 {
        0
    }

    fn propose(&mut self, _seq: SeqId, _last: TokenId, _width: u32) -> Vec<TokenId> {
        Vec::new()
    }
}

/// The same token, `k` times: a drafter whose guesses the model will almost certainly reject.
///
/// Its acceptance is not *guaranteed* to be 0 — the model may genuinely want this token next, and
/// with a partial forward pass it sometimes does — which is exactly why the gate asserts the ids
/// against the SPEC=0 run rather than asserting an acceptance rate. What it does guarantee is that
/// the rejection path runs on every step it is wrong on: the recovery token, the KV positions the
/// next step overwrites, and the snapshot the next step continues from.
#[derive(Clone, Copy, Debug)]
pub struct FixedDrafter {
    /// The token proposed at every position.
    pub token: TokenId,
    /// The widest proposal.
    pub width: u32,
}

impl FixedDrafter {
    /// `width` copies of `token`, every step.
    #[must_use]
    pub fn new(token: TokenId, width: u32) -> Self {
        FixedDrafter { token, width }
    }
}

impl Drafter for FixedDrafter {
    fn name(&self) -> &'static str {
        "fixed"
    }

    fn max_width(&self) -> u32 {
        self.width
    }

    fn propose(&mut self, _seq: SeqId, _last: TokenId, width: u32) -> Vec<TokenId> {
        vec![self.token; width.min(self.width) as usize]
    }
}

/// The tokens a recorded SPEC=0 run produced — for the first `truthful` positions.
///
/// This is the mock that makes acceptance a **dial**. Position `i < truthful` gets the token the
/// model is about to produce, so the model agrees; position `i >= truthful` gets
/// [`RecordedDrafter::WRONG`], which no vocabulary contains, so the model disagrees and acceptance
/// stops there. Set `truthful` to the width and every step accepts everything; set it to 0 and every
/// step accepts nothing while still paying for `k + 1` rows. Both ends, and everything between, must
/// produce the recorded run's ids — that is the gate.
///
/// The script is consumed as the sequence advances: [`Drafter::commit`] moves each sequence's cursor
/// by however many tokens the step committed, so the drafter stays aligned whatever the acceptance.
#[derive(Clone, Debug)]
pub struct RecordedDrafter {
    script: BTreeMap<SeqId, Vec<TokenId>>,
    cursor: BTreeMap<SeqId, usize>,
    truthful: u32,
    width: u32,
}

impl RecordedDrafter {
    /// A token id no vocabulary contains, so the model can never agree with it.
    ///
    /// `u32::MAX` rather than something plausible: a "wrong" token that happened to be the right one
    /// would raise acceptance above `truthful` and make the dial lie.
    pub const WRONG: TokenId = TokenId::MAX;

    /// `truthful` of every `width` proposals are the recorded run's next tokens.
    #[must_use]
    pub fn new(width: u32, truthful: u32) -> Self {
        RecordedDrafter {
            script: BTreeMap::new(),
            cursor: BTreeMap::new(),
            truthful: truthful.min(width),
            width,
        }
    }

    /// Give a sequence the continuation a recorded run produced for it, from its current position.
    pub fn record(&mut self, seq: SeqId, tokens: Vec<TokenId>) {
        self.script.insert(seq, tokens);
        self.cursor.insert(seq, 0);
    }

    /// Where this sequence's script has got to.
    #[must_use]
    pub fn position(&self, seq: SeqId) -> usize {
        self.cursor.get(&seq).copied().unwrap_or(0)
    }

    /// The acceptance this drafter is dialled to.
    #[must_use]
    pub fn truthful(&self) -> u32 {
        self.truthful
    }
}

impl Drafter for RecordedDrafter {
    fn name(&self) -> &'static str {
        "recorded"
    }

    fn max_width(&self) -> u32 {
        self.width
    }

    fn propose(&mut self, seq: SeqId, _last: TokenId, width: u32) -> Vec<TokenId> {
        let want = width.min(self.width) as usize;
        let at = self.position(seq);
        let script = self.script.get(&seq);
        (0..want)
            .map(|i| {
                if (i as u32) < self.truthful {
                    // The token the recorded run produced at this position. Running off the end of
                    // the script is not an error — it means the recording is shorter than this run,
                    // and a token the model will reject is the honest answer.
                    script
                        .and_then(|s| s.get(at + i))
                        .copied()
                        .unwrap_or(Self::WRONG)
                } else {
                    Self::WRONG
                }
            })
            .collect()
    }

    fn commit(&mut self, seq: SeqId, committed: &[TokenId]) {
        *self.cursor.entry(seq).or_insert(0) += committed.len();
    }

    fn release(&mut self, seq: SeqId) {
        self.script.remove(&seq);
        self.cursor.remove(&seq);
    }
}

/// The drafter that is **the pass's own**: a mailbox for what the device proposed.
///
/// It proposes nothing itself. Each step, the pass hands back one proposal per live slot
/// ([`crate::forward::StepOutcome::drafts`]), [`crate::executor::RealExecutor`] offers each to its
/// drafter, and this stores it; the next step's `propose` hands that block straight back. Two
/// consequences worth naming, because both are properties of the SHAPE and not of the drafter:
///
/// * **A sequence's first decode step has no offer yet**, so it proposes nothing and runs one row
///   inside a wider bucket — padded, exactly as a slot below the bucket's width already is. This is
///   counted (`empty`), not hidden: a mailbox that is empty every step is a drafter that never ran,
///   and the count is the difference between that and a drafter that is one step behind.
/// * **A proposal is consumed once.** Re-using one would mean drafting from a history the model has
///   moved past, which the acceptance law would reject — correctly, but silently and at the cost of
///   the whole block. Taking it means a missing offer shows up as a narrow step instead.
///
/// It can never change what is committed: [`Greedy`] decides, and every committed token is the
/// model's own at a position whose prefix the model also produced.
#[derive(Clone, Debug, Default)]
pub struct EngineDrafter {
    inbox: BTreeMap<SeqId, Vec<TokenId>>,
    width: u32,
    /// Proposals taken from the inbox.
    taken: u64,
    /// `propose` calls that found nothing waiting.
    empty: u64,
}

impl EngineDrafter {
    /// A mailbox for a pass whose proposals are at most `width` tokens wide.
    #[must_use]
    pub fn new(width: u32) -> Self {
        EngineDrafter {
            inbox: BTreeMap::new(),
            width,
            taken: 0,
            empty: 0,
        }
    }

    /// Proposals taken, and `propose` calls that found nothing waiting.
    #[must_use]
    pub fn counts(&self) -> (u64, u64) {
        (self.taken, self.empty)
    }
}

impl Drafter for EngineDrafter {
    fn name(&self) -> &'static str {
        "engine"
    }

    fn max_width(&self) -> u32 {
        self.width
    }

    fn propose(&mut self, seq: SeqId, _last: TokenId, width: u32) -> Vec<TokenId> {
        match self.inbox.remove(&seq) {
            Some(mut d) => {
                self.taken += 1;
                d.truncate(width.min(self.width) as usize);
                d
            }
            None => {
                self.empty += 1;
                Vec::new()
            }
        }
    }

    fn offer(&mut self, seq: SeqId, draft: &[TokenId]) {
        self.inbox.insert(seq, draft.to_vec());
    }

    fn release(&mut self, seq: SeqId) {
        self.inbox.remove(&seq);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: SeqId = SeqId(1);

    /// The contract, on every shape: the committed run is the accepted prefix plus exactly one.
    #[test]
    fn a_verified_step_always_commits_the_prefix_plus_one() {
        let law = Greedy;
        for (draft, rows) in [
            (vec![], vec![9u32]),
            (vec![1u32], vec![1, 9]),
            (vec![1u32], vec![2, 9]),
            (vec![1u32, 2, 3, 4], vec![1, 2, 3, 4, 9]),
            (vec![1u32, 2, 3, 4], vec![1, 2, 9, 0, 0]),
        ] {
            let v = law.accept(&draft, &rows).expect("a legal step");
            assert_eq!(v.committed.len(), v.accepted as usize + 1, "{draft:?}");
            assert!(v.accepted as usize <= draft.len());
            // Everything accepted was the draft's, and the one extra is the model's.
            assert_eq!(
                &v.committed[..v.accepted as usize],
                &draft[..v.accepted as usize]
            );
            assert_eq!(v.committed[v.accepted as usize], rows[v.accepted as usize]);
        }
    }

    /// Acceptance stops at the first disagreement and never resumes, even when the draft agrees
    /// again afterwards. A count rather than a prefix would commit a token whose own prefix the
    /// model never produced.
    #[test]
    fn acceptance_is_a_prefix_and_does_not_resume() {
        let v = Greedy
            .accept(&[1, 2, 3, 4], &[1, 9, 3, 4, 7])
            .expect("a legal step");
        assert_eq!(v.accepted, 1);
        assert_eq!(v.committed, vec![1, 9]);
    }

    /// Everything accepted: the bonus token is the model's row `k`.
    #[test]
    fn a_fully_accepted_step_commits_the_bonus_token() {
        let v = Greedy
            .accept(&[1, 2, 3], &[1, 2, 3, 42])
            .expect("a legal step");
        assert_eq!(v.accepted, 3);
        assert_eq!(v.committed, vec![1, 2, 3, 42]);
    }

    /// Nothing accepted: one token, the model's own, exactly as SPEC=0 would have produced.
    #[test]
    fn a_fully_rejected_step_commits_one_token() {
        let v = Greedy
            .accept(&[7, 7, 7], &[1, 0, 0, 0])
            .expect("a legal step");
        assert_eq!(v.accepted, 0);
        assert_eq!(v.committed, vec![1]);
    }

    /// **The theorem, as a test.** Whatever the drafter guesses, the tokens a verified run commits
    /// are the tokens a SPEC=0 run commits — because every committed token is the model's own at a
    /// position whose prefix the model also produced. Speculation at temperature 0 buys steps, not
    /// tokens, and this is the property the GPU gate asserts against the real pass.
    #[test]
    fn a_verified_run_commits_the_spec0_run_whatever_the_draft() {
        // A deterministic "model": the next token is a function of the last one. In u64 because a
        // draft may contain `RecordedDrafter::WRONG`, and a model that overflowed on the sentinel
        // would be a model that cannot be asked the question the law exists to ask.
        let next = |t: TokenId| -> TokenId { ((u64::from(t) * 7 + 3) % 1000) as TokenId };
        let spec0: Vec<TokenId> = {
            let mut out = Vec::new();
            let mut t = 1;
            for _ in 0..40 {
                t = next(t);
                out.push(t);
            }
            out
        };
        for width in 1u32..=5 {
            for truthful in 0..=width {
                let mut drafter = RecordedDrafter::new(width, truthful);
                drafter.record(S, spec0.clone());
                let law = Greedy;
                let mut committed: Vec<TokenId> = Vec::new();
                let mut last: TokenId = 1;
                let mut steps = 0usize;
                while committed.len() < spec0.len() {
                    let at = drafter.position(S);
                    let draft = drafter.propose(S, last, width.saturating_sub(1));
                    // The rows the model would produce: row 0 from `last`, row i from draft[i-1].
                    let mut rows = vec![next(last)];
                    for d in &draft {
                        rows.push(next(*d));
                    }
                    let v = law.accept(&draft, &rows).expect("a legal step");
                    // The dial is an upper bound on acceptance: a position past `truthful` gets the
                    // sentinel and cannot be accepted, and a position inside it can still be
                    // rejected once the script runs out.
                    assert!(
                        v.accepted <= truthful,
                        "accepted {} > dial {truthful}",
                        v.accepted
                    );
                    // And a LOWER bound, which is the assertion that matters: while the script still
                    // has `truthful` tokens to give, every truthful position IS the model's own next
                    // token, so acceptance is the dial EXACTLY. Written `<=` alone until 2026-09-20,
                    // and that is the hole the GPU arm fell through: a pass where only row 0 answers
                    // correctly commits row 0's answer every step, which is the SPEC=0 token, so the
                    // ids match while acceptance quietly collapses to ~0. The ids cannot see it. This
                    // can, and so can its device twin.
                    // `width` here is the BUCKET width, so only `width - 1` positions are ever
                    // proposed: a dial above that is a dial the shape cannot reach.
                    let reachable = truthful.min(width - 1);
                    if at + reachable as usize <= spec0.len() {
                        assert_eq!(
                            v.accepted, reachable,
                            "width {width}, dial {truthful}, cursor {at}: a truthful draft the \
                             model agrees with must be accepted"
                        );
                    }
                    drafter.commit(S, &v.committed);
                    last = *v.committed.last().expect("one token at least");
                    committed.extend(v.committed);
                    steps += 1;
                    assert!(steps <= spec0.len() + 1, "a step that commits nothing");
                }
                committed.truncate(spec0.len());
                assert_eq!(committed, spec0, "width {width}, truthful {truthful}");
            }
        }
    }

    /// The dial: `truthful` of `k` guesses are right, so acceptance is exactly `truthful`.
    #[test]
    fn the_recorded_drafter_accepts_exactly_what_it_was_dialled_to() {
        let script: Vec<TokenId> = (100..140).collect();
        for k in 1u32..=4 {
            for truthful in 0..=k {
                let mut d = RecordedDrafter::new(k, truthful);
                d.record(S, script.clone());
                let draft = d.propose(S, 0, k);
                assert_eq!(draft.len(), k as usize);
                // Rows: the model produces the script, so row i is script[i].
                let rows: Vec<TokenId> = script[..=k as usize].to_vec();
                let v = Greedy.accept(&draft, &rows).expect("a legal step");
                assert_eq!(v.accepted, truthful, "k {k}, truthful {truthful}");
                assert_eq!(v.committed.len(), truthful as usize + 1);
            }
        }
    }

    /// The fixed drafter proposes one token, and the recorded one advances by what committed — not
    /// by what it proposed.
    #[test]
    fn the_mocks_report_themselves_honestly() {
        let mut f = FixedDrafter::new(77, 4);
        assert_eq!(f.name(), "fixed");
        assert_eq!(f.propose(S, 0, 9), vec![77, 77, 77, 77]);
        assert_eq!(f.propose(S, 0, 2), vec![77, 77]);

        let mut n = NoDrafter;
        assert_eq!(n.max_width(), 0);
        assert!(n.propose(S, 0, 5).is_empty());

        let mut r = RecordedDrafter::new(3, 3);
        r.record(S, (0..10).collect());
        assert_eq!(r.propose(S, 0, 3), vec![0, 1, 2]);
        r.commit(S, &[0, 1]); // two committed, not three proposed
        assert_eq!(r.position(S), 2);
        assert_eq!(r.propose(S, 1, 3), vec![2, 3, 4]);
        r.release(S);
        assert_eq!(r.position(S), 0);
        assert_eq!(r.propose(S, 0, 2), vec![RecordedDrafter::WRONG; 2]);
    }

    /// NEGATIVE CONTROL: a row count that does not match the draft. A verify step of `k` drafts
    /// produces `k + 1` rows, and anything else means the engine ran a shape the law was not
    /// handed — which would otherwise be read as a rejection at the first missing row and commit a
    /// token from the wrong position.
    #[test]
    fn a_row_count_that_is_not_the_draft_plus_one_is_refused() {
        assert_eq!(
            Greedy.accept(&[1, 2, 3], &[1, 2, 3]),
            Err(AcceptError::Rows {
                draft: 3,
                want: 4,
                rows: 3
            })
        );
        assert!(Greedy.accept(&[1], &[1, 2, 3]).is_err());
        assert_eq!(Greedy.accept(&[], &[]), Err(AcceptError::Empty));
        assert_eq!(Greedy.accept(&[1, 2], &[]), Err(AcceptError::Empty));
    }

    /// The mailbox: what the pass proposed is what the next step drafts, once, and then it is gone.
    ///
    /// The "once" is the assertion that matters. A mailbox that re-served a stale block would draft
    /// from a history the model has moved past — rejected by the law, so the ids stay right and the
    /// acceptance quietly collapses, which is the failure mode ws-l14 named as the one the ids
    /// cannot see.
    #[test]
    fn the_engine_drafter_serves_each_proposal_once_and_counts_the_gaps() {
        let mut d = EngineDrafter::new(7);
        // Nothing offered yet: a sequence's first decode step proposes nothing and says so.
        assert!(d.propose(S, 1, 7).is_empty());
        assert_eq!(d.counts(), (0, 1));
        d.offer(S, &[10, 11, 12]);
        assert_eq!(d.propose(S, 1, 7), vec![10, 11, 12]);
        assert_eq!(d.counts(), (1, 1));
        // Consumed: the next step is narrow rather than drafting a block the model has moved past.
        assert!(d.propose(S, 1, 7).is_empty());
        assert_eq!(d.counts(), (1, 2));
        // The SCHEDULER's width still binds, and so does the mailbox's own.
        d.offer(S, &[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(d.propose(S, 1, 3), vec![1, 2, 3]);
        let mut narrow = EngineDrafter::new(2);
        narrow.offer(S, &[1, 2, 3, 4]);
        assert_eq!(narrow.propose(S, 1, 9), vec![1, 2]);
        // Released with the session.
        d.offer(S, &[9]);
        d.release(S);
        assert!(d.propose(S, 1, 7).is_empty());
    }

    /// NEGATIVE CONTROL for the dial itself: `WRONG` must be a token the model cannot produce, or
    /// `truthful` would be a lower bound on acceptance rather than the value of it.
    #[test]
    fn the_wrong_token_is_outside_every_vocabulary() {
        // The largest vocabulary in this tree is 248,320 (`Qwen3.8-27B` at TP=1). A compile-time
        // assertion, because the sentinel is a constant and a run-time check of it would only ever
        // fail on a build that had already shipped.
        const { assert!(RecordedDrafter::WRONG > 1_000_000) };
        let mut d = RecordedDrafter::new(2, 0);
        d.record(S, vec![5, 6, 7]);
        // Dialled to 0: every proposal is the sentinel, so nothing can be accepted.
        assert_eq!(d.propose(S, 0, 2), vec![RecordedDrafter::WRONG; 2]);
        let v = Greedy
            .accept(&[RecordedDrafter::WRONG; 2], &[5, 6, 7])
            .expect("legal");
        assert_eq!(v.accepted, 0);
        assert_eq!(v.committed, vec![5]);
    }
}
