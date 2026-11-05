//! The forward pass, as a library: the two-plan split, the session index laws, the head's extra
//! launches, the weight window, and the token check.
//!
//! Four callers re-derived all of this independently — `saber forward`, `saber step-capture`,
//! `saber step-tp2` and `saber-engine`'s device thread — and the copies had already diverged in a
//! way that matters: the engine's decode tables know that **GDN state slot 0 is the kernels' null
//! slot** and that `naccept` may never be 0, and the two CLI copies did not. A step pointed at slot
//! 0 launches every dispatch, returns a token and updates no GDN state in any of the 48 layers; a
//! `naccept` of 0 makes both decode kernels read one `i32` before their table. Neither is a crash
//! and neither shows up in a launch count.
//!
//! So this module is the union of the four, with the strictest law winning, and it is where a law
//! gets fixed once:
//!
//! * [`Session`] + [`index_table`] — the session bookkeeping the graph does not model, for **N**
//!   sequences of `q_len` rows rather than the engine's one, because a decode bucket is `b8s5` as
//!   often as `b1s5`.
//! * [`Plans`] — the body-at-chunk-rows / head-at-one-row split, which exists because `lm_head` at a
//!   2048-token chunk would write 970 MiB of logits the sampler reads one row of.
//! * [`Head`] — the row gathers and the argmax the graph does not model, and [`check_token`], which
//!   is the difference between "the pass produced a token" and "the pass produced 248320, the
//!   kernel's every-logit-was-NaN answer".
//! * [`weight_window`] — which contiguous span of a pack's record table covers the records a plan
//!   binds. A pure function over `(name, offset, length)` triples, deliberately: `saber-exec` does
//!   not depend on `saber-mem`, and a law does not need a `PackReader` to be stated.
//!
//! Nothing here touches a device except [`Head::run`], which launches the four descriptors it
//! resolved.

use std::collections::{BTreeMap, BTreeSet};

use saber_hip::{DeviceAddr, HipThread, Stream};
use saber_kernels::head::{self, HeadOp, HeadRules, HeadShape};
use saber_kernels::rules::{LaunchRules, Shape};
use saber_kernels::saber::SaberRules;
use saber_kernels::LaunchDescriptor;
use saber_model::graph::{Bucket, Geometry, Graph, Phase};

use crate::error::{Error, Result};
use crate::launch::{ArgBindings, LaunchCtx, Launcher};
use crate::layer::{BlockExecutor, BlockPlan, PlanOptions, RuntimeParams};

/// The first GDN state slot a LIVE sequence may use.
///
/// **Slot 0 is the kernels' null slot**: `saber_gdn_conv_update` bails on `slot <= 0` and
/// `saber_gdn_recurrent_update` on `si <= 0` — their NULL_BLOCK_ID / padding path. A live sequence
/// pointed there runs every dispatch and updates no state, which no launch count can see. So live
/// sequences take slots from 1 and a PADDING slot is exactly 0, which is what makes padding
/// harmless.
pub const FIRST_STATE_SLOT: u32 = 1;

/// Why an index table could not be built.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum IndexError {
    /// An index key with no law here.
    #[error("no law for index table `{0}`: refusing to hand a kernel a zero-filled buffer")]
    Unknown(String),
    /// The step's numbers do not fit the table.
    #[error("index table `{key}` cannot be built: {detail}")]
    Shape {
        /// The key.
        key: String,
        /// What is wrong.
        detail: String,
    },
    /// The step itself is not one a set of tables can be built for.
    #[error("{0}")]
    Step(String),
}

/// One sequence's place in a step. A **padding** slot is `live_rows == 0` with `state_slot == 0`.
///
/// Padding exists because a captured graph has a fixed `M`: a step at the `b8s1` bucket runs eight
/// rows whether or not eight sequences are running. What makes those rows harmless is not that
/// nobody looks at their output — it is that every table points them at something that cannot touch
/// a live session: the null state slot, a zero `cu` span, and `seqused_k = 0`, which both attention
/// kernels open by testing (`if (ctx <= 0) return;`).
#[derive(Clone, Copy, Debug)]
pub struct SeqFacts<'a> {
    /// GDN state slot. **1 or more for a live sequence**; exactly 0 for padding.
    pub state_slot: u32,
    /// Tokens of this sequence already resident in its KV before this step.
    pub done_before: u32,
    /// Pages the sequence holds, in sequence order. Empty for padding.
    pub kv_blocks: &'a [u32],
    /// Rows this sequence feeds in this step. 0 for padding.
    pub live_rows: u32,
    /// Where this sequence's tokens sit for the ROPE (D4): `None` is a text-only sequence, whose
    /// token `i` is at `(i, i, i)`; `Some` carries the M-RoPE table of a prompt with images and
    /// the delta every token past it is placed by. The KV slot and the mask are the index either
    /// way.
    pub rope: Option<&'a crate::mrope::MropePositions>,
}

impl SeqFacts<'_> {
    /// A padding slot: no rows, the null state slot, no pages.
    #[must_use]
    pub fn padding() -> SeqFacts<'static> {
        SeqFacts {
            state_slot: 0,
            done_before: 0,
            kv_blocks: &[],
            live_rows: 0,
            rope: None,
        }
    }

    /// Whether this slot carries a sequence.
    #[must_use]
    pub fn live(&self) -> bool {
        self.live_rows > 0
    }
}

/// Everything the index tables of one step are a projection of.
///
/// Rows per sequence is `rows / seqs.len()`: 1 at SPEC=0, the verify width under speculation. It is
/// derived rather than carried so that a caller cannot state a width its own row count contradicts.
#[derive(Clone, Copy, Debug)]
pub struct StepIndex<'a> {
    /// Which half of the step.
    pub phase: Phase,
    /// Rows the plan runs — the **planned** count, padding included, because that is what the
    /// embedding gathers and what the GEMMs run at.
    pub rows: u32,
    /// Block-table width the plans were built with. Shorter rows are padded up to it.
    pub max_blocks: u32,
    /// One entry per slot of the bucket's batch, live slots first.
    pub seqs: &'a [SeqFacts<'a>],
    /// The step's input ids, `rows` of them (padding included).
    pub ids: &'a [i32],
    /// **A decode ROW BAND's max verify width (workstream C2), or `None` for a rectangle.**
    ///
    /// `None` is every step before C2: `q_len = rows / seqs`, and slot `i` owns rows
    /// `[i * q, (i + 1) * q)`. `Some(w)` is a band: `q_len` is `w` (the plan's MAX width — the
    /// `sidx` stride, the conv-cache stride, every GDN kernel's `maxq`, the ragged attention's
    /// scratch stride), each live sequence feeds its OWN `live_rows` in `1..=w`, and the rows are
    /// PACKED by the `cu` prefix: sequence `i` owns `[cu[i], cu[i + 1])` and the rows past
    /// `cu[seqs]` are the band's padding (`slot` −1, `pos` 0). The laws that are per SLOT (`cu`,
    /// `cache_idx`, `naccept`, `seqused_k`, `block_table`, `has_init`) and `sidx` (per slot, `w`
    /// entries each) read the same in both; only the per-ROW tables (`pos`, `slot`) and the
    /// caller's `ids` are laid out by `cu` rather than by `i * q`.
    ///
    /// The two layouts agree wherever every live sequence feeds `q` rows — every rectangular
    /// decode step and every full prefill chunk — which is why a band of equal widths is the
    /// rectangle byte for byte (`a_band_of_equal_widths_is_the_rectangle_table_for_table`).
    pub band_width: Option<u32>,
}

impl StepIndex<'_> {
    /// Rows the live sequences feed, summed.
    #[must_use]
    pub fn live_rows(&self) -> u32 {
        self.seqs.iter().map(|s| s.live_rows).sum()
    }

    /// Live sequences in this step.
    #[must_use]
    pub fn live_seqs(&self) -> usize {
        self.seqs.iter().filter(|s| s.live()).count()
    }

    /// Rows each sequence feeds: the verify width, derived from the plan's row count — or, for a
    /// row band (C2), the band's MAX width ([`Self::band_width`]).
    #[must_use]
    pub fn q_len(&self) -> u32 {
        if self.seqs.is_empty() {
            return 0;
        }
        if let Some(w) = self.band_width {
            return w.max(1);
        }
        (self.rows / self.seqs.len() as u32).max(1)
    }

    /// Which (slot, row-within-slot) each of the step's rows belongs to, `None` for a row no
    /// sequence feeds.
    ///
    /// A rectangle: slot `i` owns rows `[i * q, (i + 1) * q)`, live below its `live_rows` —
    /// `seqs * q` entries, the layout every per-row table had before C2. A row band: PACKED by the
    /// `cu` prefix over live rows, `rows` entries, the tail past `cu[seqs]` unowned.
    fn row_owners(&self) -> Vec<Option<(usize, u32)>> {
        let q = self.q_len();
        if self.band_width.is_some() {
            let mut out: Vec<Option<(usize, u32)>> = Vec::with_capacity(self.rows as usize);
            for (i, seq) in self.seqs.iter().enumerate() {
                for t in 0..seq.live_rows {
                    out.push(Some((i, t)));
                }
            }
            out.resize(self.rows.max(out.len() as u32) as usize, None);
            return out;
        }
        let mut out = Vec::with_capacity(self.seqs.len() * q as usize);
        for (i, seq) in self.seqs.iter().enumerate() {
            for t in 0..q {
                out.push((seq.live() && t < seq.live_rows).then_some((i, t)));
            }
        }
        out
    }

    /// The GDN state slot a PREFILL step's `chunk_scan` must be pointed at: the FIRST live
    /// sequence's.
    ///
    /// `saber_gdn_chunk_scan` indexes `h0`/`ht` by `blockIdx.z`, the sequence's index in the step,
    /// so the base it is handed is this slot and slot `i` of the step is `slot + i` — which is
    /// why [`StepIndex::check`] refuses a prefill step whose live slots are not consecutive.
    /// `None` for a step with no live sequence, and for a decode step, which finds every slot it
    /// touches through `sidx` instead.
    ///
    /// `saber_exec::layer::BlockPlan::retarget_prefill_state` is what consumes it.
    #[must_use]
    pub fn prefill_state_slot(&self) -> Option<u32> {
        if self.phase != Phase::Prefill {
            return None;
        }
        self.seqs.iter().find(|s| s.live()).map(|s| s.state_slot)
    }

    /// Recurrent snapshot slots this step needs: `q_len` consecutive ones per live sequence, from
    /// each sequence's own `state_slot`.
    #[must_use]
    pub fn state_slots_needed(&self) -> u32 {
        self.seqs
            .iter()
            .filter(|s| s.live())
            .map(|s| s.state_slot + self.q_len())
            .max()
            .unwrap_or(FIRST_STATE_SLOT)
    }

    /// Everything that must hold before a single table is built.
    ///
    /// Checked once per step rather than per table: these are the conditions under which the laws
    /// are correct, and a step that violates one would otherwise produce tables a kernel reads
    /// without complaint.
    ///
    /// # Errors
    /// No slots at all, an id count that is not one per row, a live sequence at the null slot, a
    /// padding slot that is not at it, more live rows than the step runs, or pages wider than the
    /// plans' table.
    pub fn check(&self) -> std::result::Result<(), IndexError> {
        self.check_with(Accepted::AllRows)
    }

    /// [`StepIndex::check`] plus the acceptance law: a live sequence's `naccept` is `1..=q_len`,
    /// because it indexes this step's own `sidx` block — a count above the width would read past
    /// the sequence's slots and into the next one's.
    ///
    /// # Errors
    /// Everything [`StepIndex::check`] refuses, plus an acceptance of 0 or one wider than `q_len`.
    pub fn check_with(&self, accepted: Accepted<'_>) -> std::result::Result<(), IndexError> {
        let bad = |detail: String| Err(IndexError::Step(detail));
        // Only an EXPLICIT acceptance is checked against the block width. `AllRows` is the law
        // every pre-speculation caller states, floored at 1 and derived from the step's own rows,
        // and tightening it here would refuse steps that are legal today for a reason that has
        // nothing to do with acceptance.
        if let Accepted::PerSeq(counts) = accepted {
            let q = self.q_len();
            for (i, s) in self.seqs.iter().enumerate() {
                if !s.live() {
                    continue;
                }
                let a = counts.get(i).copied().unwrap_or(0);
                if a == 0 || a > q {
                    return bad(format!(
                        "slot {i} accepted {a} of the previous step's rows, and this step's \
                         `sidx` block is {q} wide: `naccept - 1` indexes it, so 0 reads before \
                         the block and {q} or more reads into the next sequence's slots"
                    ));
                }
            }
        }
        if self.seqs.is_empty() {
            return bad("a step with no slots".to_owned());
        }
        if self.ids.len() as u64 != u64::from(self.rows) {
            return bad(format!(
                "a step of {} row(s) was given {} id(s); the embedding gathers one id per row",
                self.rows,
                self.ids.len()
            ));
        }
        if self.live_rows() > self.rows {
            return bad(format!(
                "{} live row(s) in a {}-row step: a step is padded UP into its bucket, never \
                 truncated",
                self.live_rows(),
                self.rows
            ));
        }
        // A ROW BAND (C2): decode only, every live sequence at `1..=w` rows, and a band no wider
        // than the rectangle it is carved from (the GDN tables are `seqs x w` and the ragged
        // attention's partials `seqs x w` rows).
        if let Some(w) = self.band_width {
            if self.phase != Phase::Decode {
                return bad("a row band at prefill: a band is a DECODE step's shape".to_owned());
            }
            if w == 0 {
                return bad("a row band of max width 0".to_owned());
            }
            if u64::from(self.rows) > self.seqs.len() as u64 * u64::from(w) {
                return bad(format!(
                    "a {}-row band over {} slot(s) at max width {w}: a band is clipped to \
                     batch x max width",
                    self.rows,
                    self.seqs.len()
                ));
            }
            if let Some((i, s)) = self.seqs.iter().enumerate().find(|(_, s)| s.live_rows > w) {
                return bad(format!(
                    "slot {i} feeds {} row(s) in a band of max width {w}: its `sidx` block and \
                     the attention's scratch stride are {w} wide",
                    s.live_rows
                ));
            }
        }
        // SEVERAL sequences' chunks in one prefill step are legal now
        // (`Bucket::prefill_batch`), and the condition is narrower than "equal chunks": it is
        // about where `cu` says a sequence starts.
        //
        // Slot `i` owns rows `[i*q, (i+1)*q)`. `cu` is the cumulative LIVE-row prefix, so with
        // ONE live sequence the two agree however short the chunk is -- the sequence starts at
        // row 0 either way, and its padded tail is masked by `slot = -1` and `seqused_k = 0`.
        // That is the padded single-chunk case and it stays legal.
        //
        // With TWO live sequences it stops agreeing the moment the first is short: rows are laid
        // out at `i*q` but `cu` would say the second sequence starts at `live_rows[0]`, one
        // chunk's padding earlier. Every kernel that walks sequences by `cu` would read the first
        // sequence's padding as the second's first tokens. So a multi-sequence prefill step
        // requires every live chunk to be FULL.
        if self.phase == Phase::Prefill && self.live_seqs() > 1 {
            let q = self.q_len();
            for (i, seq) in self.seqs.iter().enumerate() {
                if seq.live() && seq.live_rows != q {
                    return bad(format!(
                        "prefill slot {i} feeds {} of its {q} row(s) in a step with {} live \
                         sequence(s): a chunk may be padded only when it is the step's ONLY one, \
                         because `cu` would otherwise place the next sequence a chunk's padding \
                         early",
                        seq.live_rows,
                        self.live_seqs()
                    ));
                }
            }
        }
        // AND THE SLOTS THEMSELVES MUST BE CONSECUTIVE at prefill (ws-h30). `saber_gdn_chunk_scan`
        // is the one GDN kernel that does not find a session's state through an index table: it
        // indexes `h0`/`ht` by `blockIdx.z`, so the step hands it ONE base and sequence `i` is
        // `base + i` slots along. With one live sequence that is free; with two it is a condition,
        // and the alternative to checking it is a second sequence updating a slot that belongs to
        // some other session while every dispatch issues and every launch count matches.
        if self.phase == Phase::Prefill {
            let live: Vec<(usize, u32)> = self
                .seqs
                .iter()
                .enumerate()
                .filter(|(_, s)| s.live())
                .map(|(i, s)| (i, s.state_slot))
                .collect();
            if let Some((_, base)) = live.first().copied() {
                for (n, (i, slot)) in live.iter().copied().enumerate() {
                    let want = base + n as u32;
                    if slot != want {
                        return bad(format!(
                            "prefill slot {i} is live sequence {n} of this step and holds GDN \
                             state slot {slot}, not {want}: `saber_gdn_chunk_scan` indexes the \
                             recurrent state by `blockIdx.z`, so a step's live sequences must \
                             hold CONSECUTIVE slots from {base}"
                        ));
                    }
                }
            }
        }
        // Live slots come first, so a live slot after a padding one would put the `cu` prefix and
        // the row order out of step with each other.
        let mut seen_padding = false;
        for (i, s) in self.seqs.iter().enumerate() {
            if !s.live() {
                seen_padding = true;
                if s.state_slot != 0 {
                    return bad(format!(
                        "padding slot {i} points at GDN state slot {}: padding must use slot 0, \
                         the kernels' null slot, or it updates a live session's state",
                        s.state_slot
                    ));
                }
                continue;
            }
            if seen_padding {
                return bad(format!(
                    "slot {i} is live and follows a padding slot; live slots come first because \
                     `cu` is a prefix over them"
                ));
            }
            if s.state_slot < FIRST_STATE_SLOT {
                return bad(format!(
                    "live slot {i} is on GDN state slot 0, the kernels' null slot: \
                     `saber_gdn_conv_update` and `saber_gdn_recurrent_update` both bail on `slot <= 0`, \
                     so the step would launch every dispatch and update no state at all"
                ));
            }
            if s.kv_blocks.is_empty() {
                return bad(format!("live slot {i} holds no KV pages"));
            }
            if s.kv_blocks.len() as u64 > u64::from(self.max_blocks) {
                return bad(format!(
                    "live slot {i} holds {} pages and the plans' block table is {} wide",
                    s.kv_blocks.len(),
                    self.max_blocks
                ));
            }
        }
        Ok(())
    }
}

fn as_i32(v: u32) -> i32 {
    i32::try_from(v).unwrap_or(i32::MAX)
}

/// `i32` host buffer as little-endian bytes.
#[must_use]
pub fn i32_bytes(v: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// What a verify step committed: the accepted prefix, plus one token the model wrote itself.
///
/// Built only through [`accept_prefix`], which is what makes the invariant a fact rather than a
/// convention: `committed.len() == accepted + 1`, always. ws-l14 asserts the same thing at its own
/// `Verdict` constructor for the same reason — a law that breaks the contract should fail where it
/// broke, not where the scheduler counts tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Acceptance {
    /// Rows of the draft that survived, `1..=draft.len()`. Row 0 is the token the previous step
    /// already committed, so it is accepted by construction and the count is never 0.
    pub accepted: usize,
    /// The tokens this step commits: `accepted` of them from the draft, then one more the model
    /// produced — the recovery token at the rejected position, or the bonus token when the whole
    /// draft survived.
    pub committed: Vec<i32>,
}

/// The acceptance law at temperature 0.
///
/// `draft[0]` is the token the previous step already committed; `rows[i]` is what the model produces
/// GIVEN `draft[..=i]`, so `draft[i + 1]` survives only if it equals `rows[i]`. Acceptance is a
/// **prefix**: it stops at the first disagreement and does not resume when a later draft token
/// happens to agree again, because that agreement answers a question about a prefix the model never
/// wrote.
///
/// This is the law behind the property the whole scheme rests on: **at temperature 0 the committed
/// sequence is the SPEC=0 sequence**, because every committed token is the model's own at a position
/// whose entire prefix the model also produced. Speculation buys steps, never tokens.
///
/// # Which convention this is
///
/// `draft` includes the row-0 **input** token — the one the previous step already committed — so
/// `accepted` starts at 1 and `committed[0]` IS that input. That is the convention a STATE gate
/// needs: row 0 is a row, it is fed to the kernels, and a chain comparison has to run it. A
/// scheduler wants the other one (`draft` = the guesses only, `accepted` = guesses that survived,
/// tokens = the NEW ones, because committing the input twice duplicates it on the stream), and
/// `saber_engine::draft::Greedy` translates at its call site with both conventions named there —
/// a seam between two off-by-one conventions is exactly where an off-by-one lives.
///
/// # Errors
/// An empty draft, one row short of the draft, or the invariant `committed = accepted + 1` failing —
/// which would be this function's own bug and is checked rather than assumed.
pub fn accept_prefix(draft: &[i32], rows: &[i32]) -> Result<Acceptance> {
    if draft.is_empty() {
        return Err(Error::Refused(
            "an empty draft: row 0 is the token the previous step committed, so there is always \
             one"
            .to_owned(),
        ));
    }
    if rows.len() < draft.len() {
        return Err(Error::Refused(format!(
            "{} draft row(s) but only {} argmax row(s): the model answers every row it was given",
            draft.len(),
            rows.len()
        )));
    }
    let mut accepted = 1usize;
    while accepted < draft.len() && draft[accepted] == rows[accepted - 1] {
        accepted += 1;
    }
    let mut committed: Vec<i32> = draft[..accepted].to_vec();
    committed.push(rows[accepted - 1]);
    if committed.len() != accepted + 1 {
        return Err(Error::Refused(format!(
            "the acceptance law committed {} token(s) for {accepted} accepted row(s); it is always \
             the prefix plus one",
            committed.len()
        )));
    }
    Ok(Acceptance {
        accepted,
        committed,
    })
}

/// How many of the PREVIOUS step's rows were accepted, per sequence.
///
/// This is the one fact a speculative step carries that a SPEC=0 step does not, and it is a fact
/// about the step BEFORE this one. Both decode kernels open with
/// `si = sidx[n * stride + (naccept[n] - 1)]`: `naccept` selects which of the previous step's
/// per-row snapshots this step resumes from, and the conv cache is read at the same offset. A
/// verify step that drafted `q` rows and had `a` of them accepted therefore hands the next step
/// `a` — not `q`, and never 0.
///
/// [`Accepted::AllRows`] is the law every caller had before speculation: the previous step's rows
/// were all accepted, which at `q = 1` is the only possible answer and is what SPEC=0 reads.
///
/// **The table CLAMPS and [`StepIndex::check_with`] REFUSES**, which makes `check_with` mandatory
/// rather than advisory: `index_table_with` floors a count at 1 and pads a short slice with 1,
/// because a table builder that returned a `Result` per key would be unusable — so a caller that
/// skipped the check would get a legal-LOOKING `naccept` pointing at the wrong snapshot, which is
/// the original bug wearing a different hat (ws-l14 found this reading the two side by side).
#[derive(Clone, Copy, Debug)]
pub enum Accepted<'a> {
    /// Every row of the previous step was accepted (`naccept = live_rows`, floored at 1).
    AllRows,
    /// One count per slot, in `seqs` order: `1..=q` for a live sequence, ignored for padding.
    PerSeq(&'a [u32]),
}

impl Accepted<'_> {
    /// This slot's `naccept`, never 0.
    fn for_seq(&self, i: usize, seq: &SeqFacts<'_>) -> u32 {
        match self {
            Accepted::AllRows => seq.live_rows.max(1),
            Accepted::PerSeq(a) => a.get(i).copied().unwrap_or(1).max(1),
        }
    }
}

/// What the step's attention descriptors carry as their `q_len` kernarg.
///
/// The one fact `seqused_k` needs that the tables cannot see, and it is the same kind of fact as
/// [`Accepted`]: a property of the PLAN this step is about to run, not of the sequences in it.
/// `saber_attn_prefill`'s mask is right-aligned on `q_len`, so `seqused_k` and the kernarg are one
/// number stated twice and a step that moves one without the other is a wrong answer with every
/// launch count intact (ws-l15 in one direction, ws-l16 in the other).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnQLen {
    /// Each live sequence's own row count — what a plan that was never retargeted runs when its
    /// chunks fill their bucket, and what every DECODE step means (a verify step's live slots all
    /// feed the bucket's width).
    Live,
    /// The count a retargeted prefill plan's descriptors carry
    /// ([`crate::layer::BlockPlan::prefill_q_len`]). Below a live sequence's rows it is a refusal.
    Exact(u32),
}

/// One index table for a step, or a refusal.
///
/// The laws, one per key family:
///
/// * `*.ids` — the step's ids, one per row.
/// * `*.cu` — the cumulative-rows prefix over slots, `seqs + 1` entries, counting **live** rows so a
///   padded bucket's tail is skipped.
/// * `*.cache_idx` — one conv-state slot per slot: the sequence's own, 0 for padding.
/// * `*.sidx` — one recurrent snapshot slot per ROW, `q_len` consecutive ones from each live
///   sequence's `state_slot` and 0 for padding. At `q_len = 1` that is one entry per sequence, which
///   is what SPEC=0 reads.
/// * `*.naccept` — how many of the **previous** step's rows were accepted ([`Accepted`]), which is
///   a sequence's live rows when nothing was rejected, and **never 0**: both decode kernels index
///   `sidx[naccept[n] - 1]`, so a zero reads one `i32` before the table — including for padding,
///   which is why a padding slot reads 1 and is skipped by its null state slot instead.
/// * `*.has_init` — a **byte** per slot (`uint8_t has_init[n]`), zero on a sequence's first chunk.
/// * `*.block_table` — each sequence's pages, padded to `max_blocks` with that sequence's **own**
///   last page. Padding with 0 would point a stray read at page 0, which belongs to another session.
/// * `*.seqused_k` — keys the attention may read: `done_before` plus the rows the attention treats
///   as live ([`AttnQLen`], the same number as its `q_len` kernarg because the prefill mask is
///   right-aligned on it), and **0 for padding**, which is how both attention kernels skip a
///   padded slot (`if (ctx <= 0) return;`).
///
/// # Errors
/// A key with no law here.
pub fn index_table(key: &str, s: &StepIndex<'_>) -> std::result::Result<Vec<u8>, IndexError> {
    index_table_with(key, s, Accepted::AllRows)
}

/// [`index_table`] with the previous step's acceptance, which only `*.naccept` reads.
///
/// # Errors
/// A key with no law here.
pub fn index_table_with(
    key: &str,
    s: &StepIndex<'_>,
    accepted: Accepted<'_>,
) -> std::result::Result<Vec<u8>, IndexError> {
    index_table_for(key, s, accepted, AttnQLen::Live)
}

/// [`index_table_with`] for a step whose prefill attention has been retargeted
/// ([`crate::layer::BlockPlan::retarget_prefill_rows`]), which only `*.seqused_k` reads.
///
/// # Errors
/// A key with no law here, or an [`AttnQLen::Exact`] below a live sequence's rows.
pub fn index_table_for(
    key: &str,
    s: &StepIndex<'_>,
    accepted: Accepted<'_>,
    attn: AttnQLen,
) -> std::result::Result<Vec<u8>, IndexError> {
    let n = s.seqs.len();
    let q = s.q_len();
    if key.ends_with(".ids") {
        return Ok(i32_bytes(s.ids));
    }
    if key.ends_with(".cu") {
        let mut cu = Vec::with_capacity(n + 1);
        let mut acc = 0u32;
        cu.push(0i32);
        for seq in s.seqs {
            acc += seq.live_rows;
            cu.push(as_i32(acc));
        }
        return Ok(i32_bytes(&cu));
    }
    if key.ends_with(".cache_idx") {
        return Ok(i32_bytes(
            &s.seqs
                .iter()
                .map(|q| as_i32(q.state_slot))
                .collect::<Vec<i32>>(),
        ));
    }
    if key.ends_with(".sidx") {
        // One slot per ROW: the recurrent kernel reads `sidx[n * sidx_stride + t]` for every
        // candidate `t`, so a sequence that verifies `q` rows needs `q` consecutive slots.
        let mut out: Vec<i32> = Vec::with_capacity(n * q as usize);
        for seq in s.seqs {
            for t in 0..q {
                out.push(if seq.live() {
                    as_i32(seq.state_slot + t)
                } else {
                    0
                });
            }
        }
        return Ok(i32_bytes(&out));
    }
    if key.ends_with(".naccept") {
        // The PREVIOUS step's accepted count, and NOT zero for any slot: `it = naccept[n] - 1`
        // indexes `sidx`, and a zero reads before the table. A padding slot is skipped by its null
        // state slot, not by this.
        return Ok(i32_bytes(
            &s.seqs
                .iter()
                .enumerate()
                .map(|(i, q)| as_i32(accepted.for_seq(i, q)))
                .collect::<Vec<i32>>(),
        ));
    }
    // ---- P27-15d: the tree verify's tables, by the CHAIN law -----------------------------------
    //
    // A chain IS a tree (every row's parent is the row before it), so each tree table has a law
    // here: the tables a chain of `q` rows would bind. They size the blob (`BlobLayout::plan`)
    // and are the reference the engine's real trees are held against
    // (`saber_engine::spec_tree_step::TreeStepTables`, a chain plan == these bytes); a real tree
    // step OVERRIDES them with its plan's tables.
    if key.ends_with(".anc") {
        // One u16 ancestor word a ROW: row t of a chain sees rows 0..=t of its sequence.
        let mut words: Vec<u16> = Vec::new();
        for owner in s.row_owners() {
            words.push(match owner {
                Some((_, t)) if t < 16 => ((2u32 << t) - 1) as u16,
                _ => 0,
            });
        }
        return Ok(words.iter().flat_map(|w| w.to_le_bytes()).collect());
    }
    if key.ends_with(".tsrc") {
        // [rows][4] i32: row t's three older conv taps (0..2 = the committed window, 3 + r = row
        // r); a chain's row t reads the sliding window [t, t + 1, t + 2].
        let mut out: Vec<i32> = Vec::new();
        for owner in s.row_owners() {
            let t = owner.map_or(0, |(_, t)| as_i32(t));
            if owner.is_some() {
                out.extend_from_slice(&[t, t + 1, t + 2, 0]);
            } else {
                out.extend_from_slice(&[0; 4]);
            }
        }
        return Ok(i32_bytes(&out));
    }
    if key.ends_with(".hsrc") {
        // [slots][4] i32: the committed window's cache entries; after a chain step of `naccept`
        // committed tokens they are `naccept - 1 ..= naccept + 1` (the shipped conv's offset).
        let mut out: Vec<i32> = Vec::with_capacity(n * 4);
        for (i, seq) in s.seqs.iter().enumerate() {
            if seq.live() {
                let o = as_i32(accepted.for_seq(i, seq).max(1)) - 1;
                out.extend_from_slice(&[o, o + 1, o + 2, 0]);
            } else {
                out.extend_from_slice(&[0; 4]);
            }
        }
        return Ok(i32_bytes(&out));
    }
    if key.ends_with(".rp") {
        // P27-W2-5: [slots][3] i32 = {rows to replay, stash tag, stash parity} -- host-owned facts
        // of the SESSION, not of the step, so the shared law writes the neutral table (nothing to
        // replay) and the engine overwrites it with its own record (`device::stage_replay`).
        return Ok(i32_bytes(&vec![0i32; 3 * n]));
    }
    if key.ends_with(".acc") {
        // [slots] i32: the previous tree's accepted row (the ADV prologue); -1 = nothing pending.
        return Ok(i32_bytes(&vec![-1i32; n]));
    }
    if key.ends_with(".tslot") {
        // [slots] i32: the ONE committed recurrent slot a tree reads and advances in place -- after
        // a chain step the snapshot its last committed token wrote (`sidx[naccept - 1]`); 0 (the
        // kernels' null slot) for a padding slot.
        return Ok(i32_bytes(
            &s.seqs
                .iter()
                .enumerate()
                .map(|(i, seq)| {
                    if seq.live() {
                        as_i32(seq.state_slot + accepted.for_seq(i, seq).max(1) - 1)
                    } else {
                        0
                    }
                })
                .collect::<Vec<i32>>(),
        ));
    }
    if key.ends_with(".has_init") {
        // BYTES, not ints.
        return Ok(s.seqs.iter().map(|q| u8::from(q.done_before > 0)).collect());
    }
    if key.ends_with(".block_table") {
        let mut table: Vec<i32> = Vec::with_capacity(n * s.max_blocks as usize);
        for q in s.seqs {
            let last = q.kv_blocks.last().copied().unwrap_or(0);
            let mut row: Vec<i32> = q.kv_blocks.iter().map(|b| as_i32(*b)).collect();
            row.resize(s.max_blocks as usize, as_i32(last));
            table.extend(row);
        }
        return Ok(i32_bytes(&table));
    }
    if key.ends_with(".seqused_k") {
        // `done_before + the rows the attention treats as live`, and the second half of that sum
        // is the kernel's own `q_len` kernarg. The two are ONE number and this is the place they
        // are held together.
        //
        // **Why it is a sum and not a context length.** `saber_attn_prefill`'s mask is
        // RIGHT-ALIGNED:
        //
        //     klimit = ctx - q_len + qpos          // saber_attn.hip, the prefill leg
        //
        // — the kernel takes the `q_len` query rows to be the LAST `q_len` positions of a
        // `ctx`-key sequence. So `ctx` has to be `done_before + q_len` for the SAME `q_len` the
        // descriptor carries, and then `klimit = done_before + qpos` is the position that row's
        // token actually holds. Reporting `done_before + live` against a descriptor carrying the
        // BUCKET's rows masked every live row short by the padding — a 234-token prompt in a
        // 2048-row chunk was masked to nothing, which is what every garbage multi-turn reply was
        // (ws-l15). Reporting `done_before + bucket` against a descriptor carrying the CHUNK's
        // rows is the same mistake mirrored: it would hand the padded lanes keys past the ones the
        // sequence has written.
        //
        // [`AttnQLen`] is how a step says which number its descriptors carry. `Live` — the answer
        // for every caller whose chunks fill their bucket, and for every decode step — is the
        // sequence's own live rows; `Exact(n)` is a plan retargeted to `n`
        // (`BlockPlan::retarget_prefill_rows`, ws-h27), and it refuses an `n` below a live
        // sequence's rows, which is ws-l15's mask made unplannable.
        //
        // A padding SLOT still reports 0, which is how both attention kernels skip it
        // (`if (ctx <= 0) return;`).
        let mut out: Vec<i32> = Vec::with_capacity(n);
        for seq in s.seqs {
            if !seq.live() {
                out.push(0);
                continue;
            }
            let q_len = match attn {
                AttnQLen::Live => seq.live_rows,
                AttnQLen::Exact(n) => {
                    if n < seq.live_rows {
                        return Err(IndexError::Shape {
                            key: key.to_string(),
                            detail: format!(
                                "the step's attention carries q_len {n} and this sequence feeds \
                                 {} row(s): the mask is right-aligned on `q_len`, so every live \
                                 row would be cut short by the difference",
                                seq.live_rows
                            ),
                        });
                    }
                    n
                }
            };
            out.push(as_i32(seq.done_before + q_len));
        }
        return Ok(i32_bytes(&out));
    }
    // ---- the attention glue (workstream F, plugin 4) ---------------------------------------
    //
    // Four tables that exist because `qk_norm_rope` and `kv_append` do. They are SESSION facts
    // like every table above: where each sequence already is, and where its K/V goes.
    if key.ends_with(".cos_sin") {
        // The rope table, bf16 `[max_pos, rotary_dim]` rows of `[cos | sin]`. A config constant,
        // not a checkpoint tensor — prod computes it at load for the same reason.
        //
        // Sized by the CONTEXT BAND the plan was built for, never by where the sequences have got
        // to. Sizing it from `done_before + live_rows` makes the buffer GROW as a session advances,
        // and a caller that allocated it once from step 1 then gets a truncated table: the rope
        // values for every position past the first step's are simply absent, so the tokens are
        // wrong and a replayed graph (fixed sizes) and the eager walk disagree. `saber engine`
        // caught exactly that — "the graph and the eager path disagree at token 1" plus two
        // batched-identity failures — on the day this table was added.
        let max_pos = (s.max_blocks * KV_BLOCK).max(s.rows).max(1);
        return Ok(rope_cache(max_pos, ROTARY_DIM, ROPE_THETA));
    }
    if key.ends_with(".pos") {
        // mrope positions, `[3, rows]` int64. Row `t` of a TEXT-ONLY sequence sits at
        // `done_before + t` on all three axes, which is the case where the interleaved-vs-chunked
        // choice cannot be observed. A sequence with images (D4) reads its prompt rows out of its
        // M-RoPE table and places every row past the prompt at `index + delta` on all three axes
        // (`crate::mrope`). A padding slot contributes position 0, which its zero `cu` span and
        // `seqused_k = 0` make inert.
        // One entry per ROW, in the step's row layout ([`StepIndex::row_owners`]: `i * q` for a
        // rectangle, packed by `cu` for a row band).
        let owners = s.row_owners();
        if s.seqs.iter().any(|q| q.rope.is_some()) {
            let mut axes: [Vec<i64>; 3] = std::array::from_fn(|_| Vec::with_capacity(owners.len()));
            for owner in &owners {
                let p = match owner.map(|(i, t)| (&s.seqs[i], t)) {
                    Some((seq, t)) => match seq.rope {
                        Some(r) => r.at(seq.done_before + t),
                        None => [i64::from(seq.done_before + t); 3],
                    },
                    None => [0; 3],
                };
                for (a, v) in axes.iter_mut().zip(p) {
                    a.push(v);
                }
            }
            let [t_axis, h_axis, w_axis] = axes;
            let mut v = t_axis;
            v.extend_from_slice(&h_axis);
            v.extend_from_slice(&w_axis);
            return Ok(i64_bytes(&v));
        }
        let mut base: Vec<i64> = Vec::with_capacity(owners.len());
        for owner in &owners {
            // Position 0 for every row the step does not feed -- a padding SLOT, the padded tail
            // of a live sequence's span, and a row band's tail past `cu[seqs]` alike, the same
            // cases the `slot` law masks. The tail matters for a reason of its own: `qk_norm_rope`
            // indexes `cos_sin` by this value with no clamp, so a session near the top of the band
            // feeding a short chunk in a 256- or 2048-row bucket would read
            // `done_before + rows - 1`, past the end of a table sized for the band. Inert for what
            // is computed -- nothing reads a padded row's q or k, because `cu` stops at the live
            // rows -- and not inert for where the kernel reads from.
            base.push(match owner {
                Some((i, t)) => i64::from(s.seqs[*i].done_before + t),
                None => 0,
            });
        }
        let mut v = Vec::with_capacity(base.len() * 3);
        for _ in 0..3 {
            v.extend_from_slice(&base);
        }
        return Ok(i64_bytes(&v));
    }
    if key.ends_with(".slot") {
        // Where each row's K/V lands in the paged cache: page `done_before / block` of THIS
        // sequence's own run, at offset `done_before % block`. A padding row is **-1**, which is
        // the kernel's own early-out ("the token is padding and nothing is written") — the null
        // state slot cannot help here, because `kv_append` does not read one.
        let owners = s.row_owners();
        let mut slots: Vec<i64> = Vec::with_capacity(owners.len());
        for owner in &owners {
            // -1 for every row the step does not FEED, which is three cases: a padding SLOT (no
            // sequence at all), the padded tail of a live sequence's row span, and a row band's
            // tail past `cu[seqs]` (C2; the owner comes from [`StepIndex::row_owners`], which lays
            // a band out by `cu` and a rectangle at `i * q`). The second is the prefill bucket: a
            // 25-token prompt runs in the planned 256-row shape with `live_rows = 25`, and rows
            // 25..256 are zeros the GEMMs multiply and nothing reads -- `cu` stops at 25 and
            // `seqused_k` is 25, so attention never sees them. Giving those rows a real slot
            // would append 231 rows of garbage K/V over positions the sequence has not reached
            // yet, so the NEXT chunk's attention reads its own future; it also asks for pages past
            // the end of a list sized for the prompt, which is how this was found ("a sequence at
            // row 192 holds 12 page(s)").
            let Some((i, t)) = *owner else {
                slots.push(-1);
                continue;
            };
            let seq = &s.seqs[i];
            let at = seq.done_before + t;
            // A REFUSAL, never page 0. `kv_append` needs the page that holds ABSOLUTE position
            // `at`, so `kv_blocks` has to be the session's whole list, indexable by position --
            // and today's callers do not all supply that: `saber engine` hands 3 pages for a
            // session at position 8192, because before `kv_append` existed the only positional
            // reader was `block_table`, which pads a short row with its last page and so never
            // noticed. Falling back to page 0 would write a stranger's K/V into whichever session
            // owns page 0, which is the corruption the batched-identity gate exists to find;
            // naming it here is what turned that gate's "stream 0 differs from its solo ids" into
            // a sentence about pages.
            let Some(page) = seq.kv_blocks.get((at / KV_BLOCK) as usize).copied() else {
                return Err(IndexError::Step(format!(
                    "a sequence at row {at} holds {} page(s): `kv_append` has nowhere to write it",
                    seq.kv_blocks.len()
                )));
            };
            slots.push(i64::from(page * KV_BLOCK + at % KV_BLOCK));
        }
        return Ok(i64_bytes(&slots));
    }
    if key.ends_with(".k_scale") || key.ends_with(".v_scale") {
        // fp32 scalars. The checkpoint's calibrated value is 1.0 (`docs/LEDGER.md`: "nothing was
        // clipping at scale 1.0") and the device-final pack carries them as **bf16** records,
        // which `kv_append` cannot read — it takes `const float*`. They come from here until the
        // pack gives them the `f32_from_bf16` cast `A_log` and `dt_bias` already get.
        return Ok(1.0f32.to_le_bytes().to_vec());
    }
    Err(IndexError::Unknown(key.to_string()))
}

/// Rotated dims per head: `partial_rotary_factor` 0.25 of `head_dim` 256, which is what
/// `get_rope` computes.
pub const ROTARY_DIM: u32 = 64;

/// `rope_theta` from the checkpoint's `text_config`.
pub const ROPE_THETA: f64 = 10_000_000.0;

/// Paged K/V block size. `--kv-cache-dtype fp8` with 16 slots per page.
pub const KV_BLOCK: u32 = 16;

/// `i64` host buffer as bytes. The glue kernels take `const long*` for both the mrope positions
/// and the slot map, matching the reference tensors they were gated against.
#[must_use]
pub fn i64_bytes(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The rotary cos/sin cache `get_rope` builds, as **bf16** rows of
/// `[cos(rotary_dim/2) | sin(rotary_dim/2)]`.
///
/// Computed rather than read from the pack because it is not a checkpoint tensor: it is a
/// function of `rope_theta`, `rotary_dim` and how many positions are addressable.
#[must_use]
pub fn rope_cache(max_pos: u32, rotary_dim: u32, theta: f64) -> Vec<u8> {
    let half = (rotary_dim / 2) as usize;
    let inv: Vec<f64> = (0..half)
        .map(|i| 1.0 / theta.powf((2 * i) as f64 / f64::from(rotary_dim)))
        .collect();
    let mut out = Vec::with_capacity(max_pos as usize * rotary_dim as usize * 2);
    let bf16 = |x: f64| ((x as f32).to_bits() >> 16) as u16;
    for p in 0..max_pos {
        let row: Vec<f64> = inv.iter().map(|f| f64::from(p) * f).collect();
        for f in &row {
            out.extend_from_slice(&bf16(f.cos()).to_le_bytes());
        }
        for f in &row {
            out.extend_from_slice(&bf16(f.sin()).to_le_bytes());
        }
    }
    out
}

/// The two plans a forward pass needs, and why they are two.
///
/// The body is planned at the chunk's row count with [`PlanOptions::BODY`], which gives every
/// layer-local edge ONE arena buffer shared by all 64 layers — without it a 2048-token chunk needs
/// 27 GiB of activations. The head is planned separately at the **sampled** rows
/// ([`Bucket::sampled_rows`]), because `lm_head` at the chunk's row count would compute
/// 2048 × 248320 logits, 970 MiB, of which the sampler reads one row. Selecting the rows first is
/// what prod's `LogitsProcessor` does and why the head plugin has a `row_gather`.
///
/// The head's row count is **derived from the bucket, not passed in**: it was a parameter until
/// 2026-09-20, every caller passed a literal `1`, and a literal `1` is right only for a
/// single-sequence chunk. [`Graph::head_bucket`] answers it for any bucket.
pub struct Plans<'g> {
    /// Embedding + every layer, one arena.
    pub body: BlockPlan<'g>,
    /// The final norm and `lm_head`, at `head_rows`.
    pub head: BlockPlan<'g>,
}

impl<'g> Plans<'g> {
    /// Plan the body at `bucket` and the head at that bucket's sampled rows.
    ///
    /// # Errors
    /// Either plan refusing: a geometry the kernel plane will not serve, or a site this planner
    /// cannot plan at this TP (the embedding and the vocabulary gather above TP=1).
    pub fn new(
        graph: &'g Graph,
        bucket: Bucket,
        layers: &[u32],
        params: RuntimeParams,
        mxfp4: &LaunchRules,
        saber: &SaberRules,
    ) -> Result<Plans<'g>> {
        let body = BlockPlan::with_options(
            graph,
            bucket,
            layers,
            params,
            mxfp4,
            saber,
            PlanOptions::BODY,
        )?;
        // NO layers: `PlanOptions::HEAD` selects the head SITE, and the layer list is what selects
        // the layer site. Passing the body's list plans the whole model a second time — which it
        // silently did until the head reported one outside edge instead of two.
        let head = BlockPlan::with_options(
            graph,
            graph.head_bucket(bucket),
            &[],
            params,
            mxfp4,
            saber,
            PlanOptions::HEAD,
        )?;
        Ok(Plans { body, head })
    }

    /// Launches both plans declare, plus the [`Head`]'s three.
    #[must_use]
    pub fn expected_launches(&self) -> u32 {
        self.body.expected_launches() + self.head.expected_launches() + Head::EXTRA_LAUNCHES
    }
}

/// The launches the graph does not model: two row selections and the argmax.
///
/// The graph's `logits.gather` is the TP > 1 vocabulary all-gather, a different op; these three are
/// the sampler's own, and they are counted in a pass's launch budget here so that no caller has to
/// remember the number.
pub struct Head {
    /// One row of `hidden` out of the body's arena, per carried edge.
    pub gather: LaunchDescriptor,
    /// Greedy sampling over one row of logits.
    pub argmax: LaunchDescriptor,
}

impl Head {
    /// Two row selections and one argmax.
    pub const EXTRA_LAUNCHES: u32 = 3;

    /// Resolve both descriptors for a model of `hidden` columns and `vocab` logits.
    ///
    /// # Errors
    /// The head rules refusing a width.
    pub fn resolve(hidden: u32, vocab: u32, rules: &HeadRules) -> Result<Head> {
        let gather = head::resolve(
            HeadOp::RowGather,
            HeadShape {
                rows: 1,
                cols: hidden,
                src_stride: i64::from(hidden),
                vocab: (0, 0),
            },
            rules,
        )?;
        let argmax = head::resolve(
            HeadOp::Argmax,
            HeadShape {
                rows: 1,
                cols: vocab,
                src_stride: i64::from(vocab),
                vocab: (0, 0),
            },
            rules,
        )?;
        Ok(Head { gather, argmax })
    }

    /// Select `row` of `src` into `dst`, one edge at a time.
    ///
    /// # Errors
    /// The launch failing.
    pub fn select(
        &self,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        stream: &Stream,
        src: DeviceAddr,
        idx: DeviceAddr,
        dst: DeviceAddr,
    ) -> Result<u32> {
        let mut a = ArgBindings::new(Shape::new(0, 0, 0));
        a.bind("src", src);
        a.bind("idx", idx);
        a.bind("out", dst);
        Ok(launcher
            .launch_op(ctx, &[&self.gather], &a, stream)?
            .launches)
    }

    /// Argmax over one row of logits into `out`.
    ///
    /// # Errors
    /// The launch failing.
    pub fn sample(
        &self,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        stream: &Stream,
        logits: DeviceAddr,
        out: DeviceAddr,
    ) -> Result<u32> {
        let mut a = ArgBindings::new(Shape::new(0, 0, 0));
        a.bind("logits", logits);
        a.bind("out", out);
        Ok(launcher
            .launch_op(ctx, &[&self.argmax], &a, stream)?
            .launches)
    }
}

/// Is this a token, or is it the kernel telling us the pass produced no finite logit?
///
/// The argmax kernel answers `vocab` when every logit was NaN, which is a different failure from a
/// token out of range and has to be reported differently: one means the forward pass is broken,
/// the other means the sampler is.
///
/// # Errors
/// A negative id, one past the vocabulary, or exactly `vocab`.
pub fn check_token(tok: i32, vocab: u32) -> Result<u32> {
    if tok < 0 {
        return Err(Error::Refused(format!(
            "the argmax returned {tok}, which is not a token id"
        )));
    }
    let t = u32::try_from(tok).unwrap_or(u32::MAX);
    if t == vocab {
        return Err(Error::Refused(format!(
            "the argmax returned {vocab} — the kernel's 'every logit was NaN' answer. The forward \
             pass produced no finite logit."
        )));
    }
    if t > vocab {
        return Err(Error::Refused(format!(
            "the argmax returned {t}, which is outside [0, {vocab}]"
        )));
    }
    Ok(t)
}

/// The vocabulary the head's GEMM declares, per rank.
///
/// # Errors
/// A graph with no `lm_head` GEMM.
pub fn vocab_of(graph: &Graph) -> Result<u32> {
    match graph
        .ops
        .iter()
        .find(|o| o.name == "lm_head")
        .map(|o| &o.geometry)
    {
        Some(Geometry::Gemm { n, .. }) => Ok(*n),
        _ => Err(Error::Refused("the graph has no `lm_head` GEMM".to_owned())),
    }
}

/// The contiguous span of a pack that covers the records a plan binds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Window {
    /// Index of the first record in the span.
    pub first: usize,
    /// Index of the last record in the span.
    pub last: usize,
    /// Bytes from the first record's offset to the end of the last, aligned up.
    pub span: u64,
    /// Every wanted record's layout id, by name.
    pub layout_of: BTreeMap<String, u32>,
    /// Wanted records the pack does not have.
    pub missing: Vec<String>,
}

/// One record of a pack's table, as this law needs it.
#[derive(Debug, Clone, Copy)]
pub struct Record<'a> {
    /// Record name.
    pub name: &'a str,
    /// Byte offset of its data in the file.
    pub offset: u64,
    /// Data length.
    pub length: u64,
    /// `docs/abi/PACK.md` § Layout ids.
    pub layout_id: u32,
}

/// Which span of `records` covers `want`, and what is missing.
///
/// A device-final pack is written in forward order, so "the records this plan binds" is a
/// **contiguous** window — the property that makes a 9 GB pack loadable 500 MiB at a time. This
/// states it as a law over the record table rather than over a `PackReader`, because `saber-exec`
/// does not depend on `saber-mem` and a law does not need a reader to be true.
///
/// `align` is the pack's data alignment; the span is rounded up to it, as the loader reads.
///
/// # Errors
/// None of the wanted records being present at all, which is a different failure from some of them
/// missing: it means this is the wrong pack, not an incomplete one.
pub fn weight_window(
    records: &[Record<'_>],
    want: &BTreeSet<String>,
    align: u64,
) -> Result<Window> {
    let mut layout_of = BTreeMap::new();
    let mut idx: Vec<usize> = Vec::new();
    for (i, r) in records.iter().enumerate() {
        if want.contains(r.name) {
            idx.push(i);
            layout_of.insert(r.name.to_owned(), r.layout_id);
        }
    }
    if idx.is_empty() {
        return Err(Error::Refused(format!(
            "none of the {} record(s) this plan binds are in a pack of {} record(s): this is the \
             wrong pack, not an incomplete one",
            want.len(),
            records.len()
        )));
    }
    let missing: Vec<String> = want
        .iter()
        .filter(|k| !layout_of.contains_key(*k))
        .cloned()
        .collect();
    let (first, last) = (idx[0], *idx.last().expect("non-empty"));
    let end = records[last].offset + records[last].length;
    let aligned = if align > 1 {
        end.div_ceil(align) * align
    } else {
        end
    };
    Ok(Window {
        first,
        last,
        span: aligned - records[first].offset,
        layout_of,
        missing,
    })
}

/// Download one buffer of the body arena and read it as a token id.
///
/// # Errors
/// The copy failing, or [`check_token`].
pub fn read_token(
    hip: &HipThread,
    exec: &BlockExecutor,
    offset: u64,
    vocab: u32,
    stream: &Stream,
) -> Result<u32> {
    let mut got = [0u8; 4];
    exec.download(hip, offset, &mut got, stream)?;
    check_token(i32::from_le_bytes(got), vocab)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn live<'a>(slot: u32, done: u32, blocks: &'a [u32], rows: u32) -> SeqFacts<'a> {
        SeqFacts {
            state_slot: slot,
            done_before: done,
            kv_blocks: blocks,
            live_rows: rows,
            rope: None,
        }
    }

    /// A fresh single-sequence prefill: the case three of the four callers started from.
    /// The rope table is a CONSTANT of the plan, not of the step.
    ///
    /// Sizing it from `done_before` made the buffer grow as a session advanced, so a caller that
    /// allocated once from step 1 read a truncated table from step 2 on. `saber engine` reported
    /// it as "the graph and the eager path disagree at token 1"; this says it directly.
    #[test]
    fn the_rope_table_does_not_change_size_as_a_session_advances() {
        let pages: Vec<u32> = (0..4).collect();
        let at = |done: u32| {
            let seqs = vec![SeqFacts {
                state_slot: FIRST_STATE_SLOT,
                done_before: done,
                kv_blocks: &pages,
                live_rows: 1,
                rope: None,
            }];
            let s = StepIndex {
                phase: Phase::Decode,
                rows: 1,
                max_blocks: 4,
                seqs: &seqs,
                ids: &[7],
                band_width: None,
            };
            index_table("layer.3.attn.qk_norm_rope.cos_sin", &s)
                .expect("cos_sin")
                .len()
        };
        let first = at(0);
        assert_eq!(first, 4 * KV_BLOCK as usize * ROTARY_DIM as usize * 2);
        for done in [1u32, 17, 63] {
            assert_eq!(at(done), first, "the table grew at done_before={done}");
        }
    }
    /// ws-h30: a prefill step's live sequences hold CONSECUTIVE GDN slots, and the step says
    /// which one its `chunk_scan` starts at.
    ///
    /// `saber_gdn_chunk_scan` is the one GDN kernel that does not find a session's state through an
    /// index table: it reads `h0` and writes `ht` at `(blockIdx.z * H + hv) * VD * KD`, so the
    /// step hands it ONE base and sequence `i` is `i` slots along from it. Every other slotted
    /// law here is per-slot lookup and does not care where the slots are; this one is the
    /// exception, so it is the one that has to be checked.
    ///
    /// The control is the pair that ws-n7 built for `cu`: two full chunks at consecutive slots are
    /// legal, the SAME two chunks at slots that skip one are refused by name, and a single short
    /// chunk — which has no second sequence to be consecutive with — is still legal.
    #[test]
    fn a_prefill_steps_live_sequences_hold_consecutive_state_slots() {
        let q = KV_BLOCK;
        let rows = 2 * q;
        let ids = vec![0i32; rows as usize];
        let a = [3u32];
        let b = [7u32];
        let pair = |slot_a: u32, slot_b: u32| -> Vec<SeqFacts<'_>> {
            vec![live(slot_a, 0, &a, q), live(slot_b, 0, &b, q)]
        };

        let ok_seqs = pair(FIRST_STATE_SLOT, FIRST_STATE_SLOT + 1);
        let s = StepIndex {
            phase: Phase::Prefill,
            rows,
            max_blocks: 2,
            seqs: &ok_seqs,
            ids: &ids,
            band_width: None,
        };
        s.check().expect("consecutive slots are legal");
        assert_eq!(
            s.prefill_state_slot(),
            Some(FIRST_STATE_SLOT),
            "the base the scan is handed is the FIRST live sequence's slot"
        );

        // NEGATIVE CONTROL: the same two chunks, one slot apart the wrong way. Every other table
        // this step builds is unchanged — `cu`, the pages and the K/V slots are all per-sequence
        // lookups — so nothing but this check stands between it and sequence 1 scanning through a
        // slot that belongs to whoever holds `FIRST_STATE_SLOT + 1`.
        let skipped = pair(FIRST_STATE_SLOT, FIRST_STATE_SLOT + 2);
        let bad = StepIndex {
            phase: Phase::Prefill,
            rows,
            max_blocks: 2,
            seqs: &skipped,
            ids: &ids,
            band_width: None,
        };
        let e = bad.check().expect_err("a gap between the slots");
        let text = format!("{e}");
        assert!(text.contains("blockIdx.z"), "{text}");
        assert!(text.contains("CONSECUTIVE"), "{text}");

        // And the reverse order, which is the same fault with the same cause.
        let swapped = pair(FIRST_STATE_SLOT + 1, FIRST_STATE_SLOT);
        assert!(StepIndex {
            phase: Phase::Prefill,
            rows,
            max_blocks: 2,
            seqs: &swapped,
            ids: &ids,
            band_width: None,
        }
        .check()
        .is_err());

        // THE CONTROL THAT THE RULE IS NOT OVER-APPLIED: one short chunk, alone, is still legal —
        // it is the padded single-chunk case every serve prefill runs.
        let one = vec![live(FIRST_STATE_SLOT + 4, 0, &a, 5)];
        let short = StepIndex {
            phase: Phase::Prefill,
            rows: q,
            max_blocks: 2,
            seqs: &one,
            ids: &ids[..q as usize],
            band_width: None,
        };
        short.check().expect("a padded lone chunk stays legal");
        assert_eq!(short.prefill_state_slot(), Some(FIRST_STATE_SLOT + 4));

        // A DECODE step has no such base: every slot it touches comes out of `sidx`, and the
        // `q` consecutive snapshots per sequence are strided by the verify width, not by 1.
        let dec = vec![
            live(FIRST_STATE_SLOT, 8, &a, 1),
            live(FIRST_STATE_SLOT + 5, 8, &b, 1),
        ];
        let d = StepIndex {
            phase: Phase::Decode,
            rows: 2,
            max_blocks: 2,
            seqs: &dec,
            ids: &ids[..2],
            band_width: None,
        };
        d.check().expect("decode slots are strided by the width");
        assert_eq!(d.prefill_state_slot(), None);
    }

    /// The two halves of the `slot` law that a padded prefill bucket exercises.
    ///
    /// A prompt runs in the PLANNED row count, not its own: 25 tokens in a 256-row prefill shape
    /// with `live_rows = 25`. Rows past the prompt are padding inside a live sequence's span, and
    /// they must write nothing -- `-1`, the kernel's early-out -- or they append garbage K/V over
    /// positions the sequence has not reached and the next chunk reads its own future.
    ///
    /// Two sequences' chunks in ONE prefill step: the `cu` prefix, the spans, and the two ways
    /// it is refused.
    ///
    /// This is the L9 batching shape. The step is sized by TOTAL rows and each sequence owns a
    /// span of `rows / slots`; `cu` is what tells a kernel where one chunk ends and the next
    /// begins, and it has to agree with that layout.
    #[test]
    fn two_chunks_in_one_prefill_step_and_what_makes_it_illegal() {
        let q = KV_BLOCK; // 16 rows per chunk, two chunks
        let rows = 2 * q;
        let ids = vec![0i32; rows as usize];
        let a = [3u32];
        let b = [7u32];
        let seqs = vec![
            live(FIRST_STATE_SLOT, 0, &a, q),
            live(FIRST_STATE_SLOT + 1, 0, &b, q),
        ];
        let s = StepIndex {
            phase: Phase::Prefill,
            rows,
            max_blocks: 2,
            seqs: &seqs,
            ids: &ids,
            band_width: None,
        };
        s.check().expect("two full chunks in one step is legal");
        assert_eq!(s.q_len(), q, "each sequence feeds its own chunk");
        assert_eq!(s.live_seqs(), 2);
        assert_eq!(s.live_rows(), rows);

        // `cu` is the prefix over the chunks, and with full chunks it agrees with the row layout:
        // sequence 1's rows start at `q`, which is exactly where its span starts.
        let cu = index_table("layer.3.gdn.cu", &s).expect("cu");
        assert_eq!(cu, i32_bytes(&[0, q as i32, rows as i32]));

        // Each sequence's K/V goes to ITS OWN pages, which is the thing a shared step must not
        // get wrong: rows 0..16 to page 3, rows 16..32 to page 7.
        let slot = index_table("layer.3.attn.kv_append.slot", &s).expect("slot");
        let want: Vec<i64> = (0..rows as i64)
            .map(|at| {
                let page = if at < i64::from(q) { 3i64 } else { 7 };
                page * i64::from(KV_BLOCK) + at % i64::from(KV_BLOCK)
            })
            .collect();
        assert_eq!(slot, i64_bytes(&want));

        // And each gets its own recurrent slots, `q` consecutive from its own base.
        let sidx = index_table("layer.3.gdn.sidx", &s).expect("sidx");
        let mut want_sidx: Vec<i32> = (0..q).map(|t| (FIRST_STATE_SLOT + t) as i32).collect();
        want_sidx.extend((0..q).map(|t| (FIRST_STATE_SLOT + 1 + t) as i32));
        assert_eq!(sidx, i32_bytes(&want_sidx));

        // NEGATIVE CONTROL 1: a SHORT chunk beside another live one. `cu` would put sequence 1 a
        // chunk's padding early, so every kernel walking by `cu` would read padding as tokens.
        let short = vec![
            live(FIRST_STATE_SLOT, 0, &a, q - 3),
            live(FIRST_STATE_SLOT + 1, 0, &b, q),
        ];
        let bad_step = StepIndex {
            phase: Phase::Prefill,
            rows,
            max_blocks: 2,
            seqs: &short,
            ids: &ids,
            band_width: None,
        };
        let err = bad_step
            .check()
            .expect_err("a short chunk beside another is refused");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("padded only when it is the step's ONLY one"),
            "{msg}"
        );

        // NEGATIVE CONTROL 2: the SAME short chunk alone is still legal, because with one live
        // sequence `cu` and the layout agree however short it is. If this ever starts failing,
        // the multi-sequence rule has been over-applied to the single-sequence case.
        let alone = vec![live(FIRST_STATE_SLOT, 0, &a, q - 3)];
        StepIndex {
            phase: Phase::Prefill,
            rows: q,
            max_blocks: 2,
            seqs: &alone,
            ids: &ids[..q as usize],
            band_width: None,
        }
        .check()
        .expect("a padded single chunk is still legal");
    }

    /// The negative control is the other half: a row the sequence DOES feed, with no page to hold
    /// it, is a refusal and never page 0 (which would be a stranger's K/V).
    #[test]
    fn a_padded_prefill_bucket_writes_only_the_rows_it_feeds() {
        let ids = vec![0i32; 2 * KV_BLOCK as usize];
        let rows = 2 * KV_BLOCK;
        let pages = [3u32, 4];
        let step = |pages: &[u32], live_rows: u32| {
            let seqs = vec![live(FIRST_STATE_SLOT, 0, pages, live_rows)];
            let s = StepIndex {
                phase: Phase::Prefill,
                rows,
                max_blocks: 4,
                seqs: &seqs,
                ids: &ids,
                band_width: None,
            };
            s.check().expect("a legal step");
            index_table("layer.3.attn.kv_append.slot", &s)
        };
        // 25 real rows of a 32-row bucket: the last 7 write nothing.
        let got = step(&pages, 25).expect("a padded prefill");
        let mut want: Vec<i64> = (0..25i64)
            .map(|at| {
                i64::from(pages[(at / i64::from(KV_BLOCK)) as usize]) * i64::from(KV_BLOCK)
                    + at % i64::from(KV_BLOCK)
            })
            .collect();
        want.resize(rows as usize, -1);
        assert_eq!(got, i64_bytes(&want));
        // `pos` masks the same tail: the rope kernel indexes `cos_sin` by it WITHOUT a clamp, so a
        // padded row carrying `done_before + t` reads past a table sized for the band.
        let seqs = vec![live(FIRST_STATE_SLOT, 0, &pages, 25)];
        let s = StepIndex {
            phase: Phase::Prefill,
            rows,
            max_blocks: 4,
            seqs: &seqs,
            ids: &ids,
            band_width: None,
        };
        let pos = index_table("layer.3.attn.qk_norm_rope.pos", &s).expect("pos");
        let mut one: Vec<i64> = (0..25i64).collect();
        one.resize(rows as usize, 0);
        let want_pos: Vec<i64> = one.iter().chain(&one).chain(&one).copied().collect();
        assert_eq!(pos, i64_bytes(&want_pos));
        // NEGATIVE CONTROL: rows it really does feed, and one page to put them in.
        let err = step(&[0], rows).expect_err("one page for two pages of fed rows");
        assert!(
            matches!(&err, IndexError::Step(m) if m.contains("nowhere to write")),
            "{err:?}"
        );
    }

    /// D4: the `.pos` law with images. A text-only sequence's table is BYTE-IDENTICAL whether it
    /// carries no rope (`None`, the engine before D4) or a text-only M-RoPE table (delta 0); an
    /// image sequence's rows are its table's three axes; decode rows past the prompt are
    /// `index + delta` on every axis; padding stays 0.
    #[test]
    fn the_pos_table_reads_the_mrope_table_and_is_unchanged_for_text() {
        let pages = [0u32, 1, 2, 3];
        let ids = [7i32; 32];
        let table =
            |rope: Option<&crate::mrope::MropePositions>, done: u32, rows: u32, live_rows: u32| {
                let seqs = vec![SeqFacts {
                    rope,
                    ..live(FIRST_STATE_SLOT, done, &pages, live_rows)
                }];
                let s = StepIndex {
                    phase: Phase::Prefill,
                    rows,
                    max_blocks: 4,
                    seqs: &seqs,
                    ids: &ids[..rows as usize],
                    band_width: None,
                };
                index_table("layer.3.attn.qk_norm_rope.pos", &s).expect("pos")
            };
        // Text only: `None` and a delta-0 table are the same bytes.
        let text = crate::mrope::for_prompt(&[1i32, 2, 3, 4, 5, 6, 7, 8], 9, &[]).unwrap();
        assert_eq!(table(None, 0, 32, 25), table(Some(&text), 0, 32, 25));
        assert_eq!(table(None, 30, 8, 5), table(Some(&text), 30, 8, 5));
        // An image: 2 text, a (1, 4, 6) grid = 6 pads, 2 text -> delta -3.
        let mut prompt = vec![1i32, 2];
        prompt.extend([9; 6]);
        prompt.extend([3, 4]);
        let img = crate::mrope::for_prompt(&prompt, 9, &[[1, 4, 6]]).unwrap();
        let got = table(Some(&img), 0, 16, 10);
        let mut want = Vec::new();
        for axis in [&img.t, &img.h, &img.w] {
            let mut a = axis.clone();
            a.resize(16, 0);
            want.extend(a);
        }
        assert_eq!(got, i64_bytes(&want));
        // A decode step of 4 rows at token 10: every axis `index + delta`.
        let got = table(Some(&img), 10, 4, 4);
        let one: Vec<i64> = (10..14).map(|i| i + img.delta).collect();
        let want: Vec<i64> = one.iter().chain(&one).chain(&one).copied().collect();
        assert_eq!(got, i64_bytes(&want));
        // The control: the index law is NOT this image's table.
        assert_ne!(table(None, 0, 16, 10), table(Some(&img), 0, 16, 10));
    }

    #[test]
    fn the_tables_are_the_ones_a_fresh_prefill_needs() {
        let ids = [11i32, 12, 13, 14];
        let blocks = [0u32, 1];
        let seqs = [live(1, 0, &blocks, 4)];
        let s = StepIndex {
            phase: Phase::Prefill,
            rows: 4,
            max_blocks: 4,
            seqs: &seqs,
            ids: &ids,
            band_width: None,
        };
        s.check().expect("a legal step");
        let t = |k: &str| index_table(k, &s).expect(k);
        assert_eq!(t("embed_tokens.ids"), i32_bytes(&ids));
        assert_eq!(t("layer.0.gdn.conv_prep.cu"), i32_bytes(&[0, 4]));
        assert_eq!(t("layer.0.gdn.conv_prep.cache_idx"), i32_bytes(&[1]));
        assert_eq!(t("layer.0.gdn.conv_prep.has_init"), vec![0u8]);
        assert_eq!(t("layer.3.attn.prefill.seqused_k"), i32_bytes(&[4]));
        // Padded to the plans' width with this sequence's OWN last page, never 0.
        assert_eq!(
            t("layer.3.attn.prefill.block_table"),
            i32_bytes(&[0, 1, 1, 1])
        );
        // NEGATIVE CONTROL: a key with no law is a refusal, not a zero buffer.
        assert_eq!(
            index_table("layer.0.gdn.decode.whatever", &s),
            Err(IndexError::Unknown("layer.0.gdn.decode.whatever".into()))
        );
    }

    /// A decode step that VERIFIES: eight sequences of five rows each, which is where the engine's
    /// SPEC=0 laws and the CLI's spec-decode laws had to meet. `sidx` is per ROW and `naccept` is
    /// the sequence's live rows, both of which reduce to the SPEC=0 answer at one row.
    #[test]
    fn a_verify_step_widens_sidx_per_row_and_naccept_per_sequence() {
        let ids: Vec<i32> = (0..40).collect();
        let pages: Vec<u32> = (0..2).collect();
        let seqs: Vec<SeqFacts<'_>> = (0..8).map(|n| live(1 + n * 5, 8192, &pages, 5)).collect();
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 40,
            max_blocks: 2,
            seqs: &seqs,
            ids: &ids,
            band_width: None,
        };
        s.check().expect("a legal verify step");
        assert_eq!(s.q_len(), 5);
        assert_eq!(s.live_seqs(), 8);
        let t = |k: &str| index_table(k, &s).expect(k);
        assert_eq!(
            t("l0.gdn.conv_update.cu"),
            i32_bytes(&[0, 5, 10, 15, 20, 25, 30, 35, 40])
        );
        // One conv slot per sequence — each sequence's own base.
        assert_eq!(
            t("l0.gdn.conv_update.cache_idx"),
            i32_bytes(&[1, 6, 11, 16, 21, 26, 31, 36])
        );
        // FIVE snapshot slots per sequence, consecutive from its base, never 0.
        let sidx = t("l0.gdn.recurrent_update.sidx");
        assert_eq!(sidx.len(), 8 * 5 * 4);
        assert_eq!(&sidx[..4 * 6], &i32_bytes(&[1, 2, 3, 4, 5, 6])[..]);
        assert_eq!(s.state_slots_needed(), 41);
        // naccept is the sequence's live rows, which is the verify width here and 1 at SPEC=0.
        assert_eq!(t("l0.gdn.conv_update.naccept"), i32_bytes(&[5; 8]));
        assert_eq!(t("l3.attn.decode.seqused_k"), i32_bytes(&[8197; 8]));
    }

    /// ws-h18: acceptance. `naccept` is the PREVIOUS step's accepted count, so the same verify
    /// step's tables differ only in that one number, and it selects which of the previous step's
    /// per-row snapshots this step resumes from (`sidx[naccept - 1]`).
    #[test]
    fn naccept_is_the_previous_steps_acceptance_and_selects_the_snapshot_to_resume_from() {
        let ids: Vec<i32> = (0..5).collect();
        let pages: Vec<u32> = (0..2).collect();
        let seqs = [live(1, 8192, &pages, 5)];
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 5,
            max_blocks: 2,
            seqs: &seqs,
            ids: &ids,
            band_width: None,
        };
        // The slots this step writes, one per row, whatever was accepted before it.
        let sidx = index_table("l0.gdn.recurrent_update.sidx", &s).expect("sidx");
        assert_eq!(sidx, i32_bytes(&[1, 2, 3, 4, 5]));
        for a in 1..=5u32 {
            let acc = [a];
            s.check_with(Accepted::PerSeq(&acc)).expect("a legal step");
            let n = index_table_with("l0.gdn.conv_update.naccept", &s, Accepted::PerSeq(&acc))
                .expect("naccept");
            assert_eq!(n, i32_bytes(&[as_i32(a)]));
            // What the kernel reads: `sidx[naccept - 1]` is the slot the previous step wrote row
            // `a - 1` into, so accepting `a` rows resumes from the state after row `a - 1`.
            let slot = i32::from_le_bytes(
                sidx[(a as usize - 1) * 4..a as usize * 4]
                    .try_into()
                    .expect("4 bytes"),
            );
            assert_eq!(slot, as_i32(a), "accepting {a} resumes from slot {a}");
        }
        // The default is still "everything was accepted", which at q = 1 is the only answer.
        assert_eq!(
            index_table("l0.gdn.conv_update.naccept", &s).expect("naccept"),
            i32_bytes(&[5])
        );
        // NEGATIVE CONTROLS: 0 reads before the block, 6 reads into the next sequence's slots.
        for bad in [0u32, 6] {
            let acc = [bad];
            assert!(
                s.check_with(Accepted::PerSeq(&acc)).is_err(),
                "an acceptance of {bad} must be refused"
            );
        }
    }

    /// The acceptance law: a PREFIX that stops at the first disagreement, and one token more than
    /// it accepted. The "resumes later" case is the one worth pinning — a draft that agrees again
    /// after a rejection is answering a question about a prefix the model never wrote.
    #[test]
    fn acceptance_is_a_prefix_and_commits_one_token_more_than_it_accepts() {
        // Nothing after row 0 survives: the model wanted 99 where the draft said 11.
        let a = accept_prefix(&[7, 11], &[99, 98]).expect("law");
        assert_eq!(a.accepted, 1);
        assert_eq!(a.committed, vec![7, 99]);
        // The whole draft survives, so the extra token is the BONUS.
        let a = accept_prefix(&[7, 20, 30], &[20, 30, 40]).expect("law");
        assert_eq!(a.accepted, 3);
        assert_eq!(a.committed, vec![7, 20, 30, 40]);
        // Agreement AFTER a rejection does not count: row 2 matches, and it is still 1 accepted.
        let a = accept_prefix(&[7, 11, 30, 40], &[99, 30, 40, 50]).expect("law");
        assert_eq!(a.accepted, 1);
        assert_eq!(a.committed, vec![7, 99]);
        // `committed = accepted + 1`, for every draft and every pattern of agreement.
        for k in 1..6usize {
            for mask in 0..(1u32 << k) {
                let draft: Vec<i32> = (0..=k as i32).collect();
                let rows: Vec<i32> = (0..=k)
                    .map(|i| {
                        if i < k && mask >> i & 1 == 1 {
                            draft[i + 1]
                        } else {
                            9000 + i as i32
                        }
                    })
                    .collect();
                let a = accept_prefix(&draft, &rows).expect("law");
                assert_eq!(a.committed.len(), a.accepted + 1);
                assert!(a.accepted >= 1 && a.accepted <= draft.len());
            }
        }
        // NEGATIVE CONTROLS: no draft, and fewer answers than rows.
        assert!(accept_prefix(&[], &[1]).is_err());
        assert!(accept_prefix(&[1, 2, 3], &[1]).is_err());
    }

    /// A PADDED prefill chunk's `seqused_k` and its `q_len` kernarg are ONE number, and the law is
    /// what that number is: the chunk's live rows (ws-h27), or the bucket's if the plan was never
    /// retargeted (the pre-h27 pair, kept as the control).
    ///
    /// `saber_attn.hip`'s prefill leg masks query row `qpos` at `ctx - q_len + qpos`. So the mask a
    /// live row gets is `done_before + live - q_len_kernarg + i`, and it is the position that
    /// row's token actually holds exactly when `ctx = done_before + q_len_kernarg`. This test
    /// walks both readings and asserts the KLIMIT rather than the table, because the table is only
    /// interesting through the kernel's arithmetic.
    #[test]
    fn a_prefill_chunks_key_count_and_its_q_len_kernarg_are_one_number() {
        // 16 pages: enough for the 234 live rows this chunk writes.
        let pages: Vec<u32> = (0..16).collect();
        // 234 live rows of a fresh prompt in the planned 256-row shape.
        let padded = [live(1, 0, &pages, 234)];
        let ids = vec![7i32; 256];
        let s = StepIndex {
            phase: Phase::Prefill,
            rows: 256,
            max_blocks: 16,
            seqs: &padded,
            ids: &ids,
            band_width: None,
        };
        s.check().expect("a padded single-chunk prefill is legal");
        // THE LAW: the chunk's rows, which is what a retargeted plan's descriptors carry.
        assert_eq!(
            index_table("x.seqused_k", &s).expect("seqused_k"),
            i32_bytes(&[234])
        );
        // ...and the same number said the other way, for a plan that says so explicitly.
        assert_eq!(
            index_table_for("x.seqused_k", &s, Accepted::AllRows, AttnQLen::Exact(234))
                .expect("seqused_k"),
            i32_bytes(&[234])
        );
        // The kernel's own arithmetic, under both readings: `klimit(i) = ctx - q_len + i`.
        let klimit =
            |q_len: u32, ctx: u32, i: u32| i64::from(ctx) - i64::from(q_len) + i64::from(i);
        for i in 0..234u32 {
            assert_eq!(klimit(234, 234, i), i64::from(i), "the live rows' own keys");
        }
        // THE CONTROL — the plan that was never retargeted. `AttnQLen::Exact(256)` is the
        // pre-ws-h27 pair, and it is still self-consistent for the LIVE rows (which is why ws-l15
        // fixed the garbage): what it adds is 22 rows the kernel treats as live, whose mask runs
        // to 255 over keys the sequence has not written.
        assert_eq!(
            index_table_for("x.seqused_k", &s, Accepted::AllRows, AttnQLen::Exact(256))
                .expect("seqused_k"),
            i32_bytes(&[256])
        );
        for i in 0..234u32 {
            assert_eq!(
                klimit(256, 256, i),
                i64::from(i),
                "still right for a live row"
            );
        }
        assert_eq!(
            klimit(256, 256, 255),
            255,
            "and a padded row reaches 21 keys past the 234 the sequence wrote"
        );
        // A REFUSAL, not a clamp: a `q_len` BELOW a live sequence's rows is ws-l15's mask, which
        // cut every live row short by the padding and produced a loop from token 0.
        assert!(
            index_table_for("x.seqused_k", &s, Accepted::AllRows, AttnQLen::Exact(100)).is_err()
        );
        // And the rows themselves still say where the live ones are: `cu` stops at 234 and the
        // padded tail writes no K/V.
        assert_eq!(index_table("x.cu", &s).expect("cu"), i32_bytes(&[0, 234]));
        let slots = index_table("x.slot", &s).expect("slot");
        let last: Vec<i64> = slots[slots.len() - 8..]
            .chunks(8)
            .map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect();
        assert_eq!(last, vec![-1], "the padded tail writes nothing");

        // CONTROL: the same chunk, FULL. Every reading coincides here — which is exactly why a
        // bucket-sized prompt has always been right and a short one has not.
        let full = [live(1, 0, &pages, 256)];
        let f = StepIndex {
            phase: Phase::Prefill,
            rows: 256,
            max_blocks: 16,
            seqs: &full,
            ids: &ids,
            band_width: None,
        };
        assert_eq!(
            index_table("x.seqused_k", &f).expect("seqused_k"),
            i32_bytes(&[256])
        );
        assert_eq!(
            index_table_for("x.seqused_k", &f, Accepted::AllRows, AttnQLen::Exact(256))
                .expect("seqused_k"),
            i32_bytes(&[256])
        );
        // NEGATIVE CONTROL: a padding SLOT still reports 0, or both attention kernels would run it.
        let padded_slot = [live(1, 0, &pages, 4), SeqFacts::padding()];
        let ids2 = vec![7i32; 8];
        let p = StepIndex {
            phase: Phase::Decode,
            rows: 8,
            max_blocks: 4,
            seqs: &padded_slot,
            ids: &ids2,
            band_width: None,
        };
        assert_eq!(
            index_table("x.seqused_k", &p).expect("seqused_k"),
            i32_bytes(&[4, 0])
        );
    }

    /// SEVERAL sequences' chunks in one prefill step (ws-n7) under the live-rows law: every live
    /// chunk is FULL there, so the one `q_len` kernarg is every sequence's row count and
    /// `seqused_k` is each sequence's own `done_before` plus it.
    ///
    /// This is the case the law has to be checked at rather than argued about, because the kernarg
    /// is ALSO the q/out row stride (`(seq * q_len + qrow) * q_heads`): a short chunk beside
    /// another live one would move sequence 1's rows. `StepIndex::check` refuses that shape, and
    /// the control below is that refusal.
    #[test]
    fn a_batched_prefill_steps_chunks_are_full_so_one_q_len_serves_every_sequence() {
        let a: Vec<u32> = (0..8).collect();
        let b: Vec<u32> = (8..16).collect();
        let seqs = [live(1, 0, &a, 64), live(2, 128, &b, 64)];
        let ids = vec![7i32; 128];
        let s = StepIndex {
            phase: Phase::Prefill,
            rows: 128,
            max_blocks: 8,
            seqs: &seqs,
            ids: &ids,
            band_width: None,
        };
        s.check().expect("two FULL chunks in one step are legal");
        // Each sequence's own `done_before`, plus the rows it feeds — which is the same number for
        // both, and therefore a legal `q_len` kernarg for the step.
        assert_eq!(
            index_table("x.seqused_k", &s).expect("seqused_k"),
            i32_bytes(&[64, 192])
        );
        assert_eq!(
            index_table_for("x.seqused_k", &s, Accepted::AllRows, AttnQLen::Exact(64))
                .expect("seqused_k"),
            i32_bytes(&[64, 192])
        );
        assert_eq!(
            index_table("x.cu", &s).expect("cu"),
            i32_bytes(&[0, 64, 128])
        );
        // THE CONTROL: the same step with sequence 0 short. `cu` would put sequence 1 a chunk's
        // padding early and one `q_len` could not be both sequences' row count, so the step is
        // refused before a table is built — and the SAME short chunk alone stays legal.
        let short = [live(1, 0, &a, 40), live(2, 128, &b, 64)];
        let bad = StepIndex {
            phase: Phase::Prefill,
            rows: 128,
            max_blocks: 8,
            seqs: &short,
            ids: &ids,
            band_width: None,
        };
        assert!(bad.check().is_err());
        let alone = [live(1, 0, &a, 40)];
        let ok = StepIndex {
            phase: Phase::Prefill,
            rows: 128,
            max_blocks: 8,
            seqs: &alone,
            ids: &ids,
            band_width: None,
        };
        ok.check().expect("a padded chunk alone is still legal");
        assert_eq!(
            index_table("x.seqused_k", &ok).expect("seqused_k"),
            i32_bytes(&[40])
        );
    }

    /// Padding: the null state slot, a zero `cu` span and `seqused_k = 0` — and `naccept` still 1,
    /// because a zero there reads before the table for every slot, padding included.
    ///
    /// The bucket is `b4s2`: four slots, one live, and the step's rows are `4 x 2`. It used to say
    /// `rows: 4` with a live slot feeding two of them, which is a shape the engine refuses (a
    /// verify step's every live slot feeds the bucket's width) and which only `seqused_k`'s law
    /// ever noticed.
    #[test]
    fn a_padded_bucket_points_its_spare_slots_at_nothing() {
        let ids = [7i32, 8, 0, 0, 0, 0, 0, 0];
        let pages = [2u32];
        let seqs = [
            live(1, 16, &pages, 2),
            SeqFacts::padding(),
            SeqFacts::padding(),
            SeqFacts::padding(),
        ];
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 8,
            max_blocks: 1,
            seqs: &seqs,
            ids: &ids,
            band_width: None,
        };
        s.check().expect("one live sequence in a four-slot bucket");
        let t = |k: &str| index_table(k, &s).expect(k);
        assert_eq!(t("x.cu"), i32_bytes(&[0, 2, 2, 2, 2]));
        assert_eq!(t("x.seqused_k"), i32_bytes(&[18, 0, 0, 0]));
        assert_eq!(t("x.cache_idx"), i32_bytes(&[1, 0, 0, 0]));
        assert_eq!(t("x.naccept"), i32_bytes(&[2, 1, 1, 1]));
    }

    /// NEGATIVE CONTROLS: a live sequence at the null slot and a padding slot that is not at it are
    /// both refused, by name, with the reason.
    #[test]
    fn the_null_slot_rules_are_enforced_both_ways() {
        let ids = [1i32, 2];
        let pages = [0u32];
        let bad_live = [live(0, 0, &pages, 2)];
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 2,
            max_blocks: 1,
            seqs: &bad_live,
            ids: &ids,
            band_width: None,
        };
        match s.check() {
            Err(IndexError::Step(m)) => assert!(m.contains("null slot"), "{m}"),
            other => panic!("expected a refusal, got {other:?}"),
        }
        let mut pad = SeqFacts::padding();
        pad.state_slot = 3;
        let bad_pad = [live(1, 0, &pages, 2), pad];
        let s2 = StepIndex {
            phase: Phase::Decode,
            rows: 2,
            max_blocks: 1,
            seqs: &bad_pad,
            ids: &ids,
            band_width: None,
        };
        match s2.check() {
            Err(IndexError::Step(m)) => assert!(m.contains("padding"), "{m}"),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    /// The window law: contiguous, aligned, and a wrong pack is a different error from a partial one.
    #[test]
    fn the_weight_window_is_the_contiguous_span_that_covers_what_is_wanted() {
        let recs = [
            Record {
                name: "a",
                offset: 0,
                length: 100,
                layout_id: 0,
            },
            Record {
                name: "b",
                offset: 4096,
                length: 100,
                layout_id: 1,
            },
            Record {
                name: "c",
                offset: 8192,
                length: 100,
                layout_id: 2,
            },
        ];
        let want: BTreeSet<String> = ["b".to_string(), "c".to_string()].into_iter().collect();
        let w = weight_window(&recs, &want, 4096).expect("window");
        assert_eq!((w.first, w.last), (1, 2));
        assert_eq!(w.span, 12288 - 4096);
        assert_eq!(w.layout_of.get("c"), Some(&2));
        assert!(w.missing.is_empty());

        let want2: BTreeSet<String> = ["b".to_string(), "zzz".to_string()].into_iter().collect();
        assert_eq!(
            weight_window(&recs, &want2, 1).expect("window").missing,
            vec!["zzz".to_string()]
        );

        // NEGATIVE CONTROL: none of them present is the WRONG PACK, a different failure.
        let want3: BTreeSet<String> = ["nope".to_string()].into_iter().collect();
        let e = weight_window(&recs, &want3, 1).expect_err("the wrong pack");
        assert!(format!("{e}").contains("wrong pack"), "{e}");
    }

    /// The token check tells the two failures apart, which is the whole reason it exists.
    #[test]
    fn the_token_check_separates_a_nan_pass_from_a_bad_id() {
        assert_eq!(check_token(7, 100).expect("a token"), 7);
        let nan = check_token(100, 100).expect_err("the NaN answer");
        assert!(format!("{nan}").contains("no finite logit"), "{nan}");
        let oob = check_token(101, 100).expect_err("out of range");
        assert!(format!("{oob}").contains("outside"), "{oob}");
        assert!(check_token(-1, 100).is_err());
    }

    // ---------------------------------------------------------------- the row band (C2)

    fn i32s(b: &[u8]) -> Vec<i32> {
        b.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn i64s(b: &[u8]) -> Vec<i64> {
        b.chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()
    }

    /// **A ROW BAND's index blob, against hand-written expectations** (C2 host test e).
    ///
    /// Four slots at max width 8 in a 16-row band: A feeds 5 rows, B 3, C 1, the fourth is a
    /// padding slot. Every per-SLOT law reads as it does in a rectangle; `sidx` keeps its stride at
    /// the MAX width with all eight entries of a live run filled (a band's `naccept` can exceed
    /// this step's rows: B committed 8 last step and verifies 3 now); the per-ROW tables are
    /// packed by `cu`, and the band's seven tail rows are `slot` −1 / `pos` 0.
    #[test]
    fn a_row_band_lays_its_rows_out_by_cu_and_its_tables_by_hand() {
        let pa: Vec<u32> = vec![0, 1];
        let pb: Vec<u32> = vec![2, 3, 4];
        let pc: Vec<u32> = vec![5];
        let seqs = [
            live(1, 20, &pa, 5),
            live(9, 33, &pb, 3),
            live(17, 15, &pc, 1),
            SeqFacts::padding(),
        ];
        let ids: Vec<i32> = (100..116).collect();
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 16,
            max_blocks: 3,
            seqs: &seqs,
            ids: &ids,
            band_width: Some(8),
        };
        let acc = [3u32, 8, 1, 1];
        s.check_with(Accepted::PerSeq(&acc))
            .expect("a legal band: naccept 8 > this step's 3 rows is legal at max width 8");
        assert_eq!(s.q_len(), 8, "a band's q_len is its MAX width");
        assert_eq!(s.live_rows(), 9);
        let t = |k: &str| index_table_for(k, &s, Accepted::PerSeq(&acc), AttnQLen::Live).expect(k);
        assert_eq!(i32s(&t("l0.gdn.conv_update.cu")), vec![0, 5, 8, 9, 9]);
        assert_eq!(i32s(&t("l0.gdn.conv_update.cache_idx")), vec![1, 9, 17, 0]);
        let mut sidx: Vec<i32> = (1..=8).collect();
        sidx.extend(9..=16);
        sidx.extend(17..=24);
        sidx.extend([0; 8]);
        assert_eq!(
            i32s(&t("l0.gdn.recurrent_update.sidx")),
            sidx,
            "stride 8, every entry of a live run filled (B's run too, though it feeds 3 rows)"
        );
        assert_eq!(
            i32s(&t("l0.gdn.conv_update.naccept")),
            vec![3, 8, 1, 1],
            "naccept is the previous step's commit, never 0, 1 for padding"
        );
        assert_eq!(
            i32s(&t("l3.attn.decode.seqused_k")),
            vec![25, 36, 16, 0],
            "done_before + this step's OWN rows (the ragged mask: klimit = ctx - ql + qrow), 0 \
             for padding"
        );
        assert_eq!(s.state_slots_needed(), 25);
        // K/V lands at each row's own position in its own pages: A at 20..24 (page 1 of [0, 1]),
        // B at 33..35 (page 4 of [2, 3, 4]), C at 15 (page 5, the first of its list).
        let mut slot: Vec<i64> = (20..25).map(|p| 16 + (p % 16)).collect();
        slot.extend([4 * 16 + 1, 4 * 16 + 2, 4 * 16 + 3]);
        slot.push(5 * 16 + 15);
        slot.extend([-1; 7]);
        assert_eq!(i64s(&t("l3.attn.kv_append.slot")), slot);
        let mut pos: Vec<i64> = (20..25).collect();
        pos.extend([33, 34, 35, 15]);
        pos.extend([0; 7]);
        let pos3: Vec<i64> = pos.iter().chain(&pos).chain(&pos).copied().collect();
        assert_eq!(i64s(&t("l3.attn.qk_norm_rope.pos")), pos3);
        // The block table is the rectangle's: per slot, padded with the slot's own last page.
        assert_eq!(
            i32s(&t("l3.attn.decode.block_table")),
            vec![0, 1, 1, 2, 3, 4, 5, 5, 5, 0, 0, 0]
        );
        assert_eq!(i32s(&t("l0.embed.ids")), ids);
    }

    /// NEGATIVE CONTROLS for the band's own conditions: a sequence wider than the band's max
    /// width, a band wider than `seqs x w`, a band at prefill, and an acceptance past the max
    /// width are each refused by name.
    #[test]
    fn a_row_band_refuses_what_its_tables_cannot_hold() {
        let pages: Vec<u32> = vec![0, 1];
        let ids = [0i32; 16];
        let step = |seqs: &[SeqFacts<'_>], rows: u32, w: u32, phase: Phase| {
            StepIndex {
                phase,
                rows,
                max_blocks: 2,
                seqs,
                ids: &ids[..rows as usize],
                band_width: Some(w),
            }
            .check()
            .map_err(|e| e.to_string())
        };
        let ok = [live(1, 4, &pages, 5), live(6, 4, &pages, 3)];
        assert!(step(&ok, 8, 5, Phase::Decode).is_ok(), "the case itself");
        let wide = [live(1, 4, &pages, 6), live(7, 4, &pages, 2)];
        let e = step(&wide, 8, 5, Phase::Decode).unwrap_err();
        assert!(e.contains("max width 5"), "{e}");
        let e = step(&ok, 11, 5, Phase::Decode).unwrap_err();
        assert!(e.contains("clipped to batch x max width"), "{e}");
        let e = step(&ok, 8, 5, Phase::Prefill).unwrap_err();
        assert!(e.contains("row band at prefill"), "{e}");
        let s = StepIndex {
            phase: Phase::Decode,
            rows: 8,
            max_blocks: 2,
            seqs: &ok,
            ids: &ids[..8],
            band_width: Some(5),
        };
        assert!(s.check_with(Accepted::PerSeq(&[5, 5])).is_ok());
        assert!(
            s.check_with(Accepted::PerSeq(&[6, 1])).is_err(),
            "naccept 6 reads past a 5-wide sidx block"
        );
    }

    /// **A band of equal widths IS the rectangle, table for table** — which is what makes the
    /// row layout law one law: `row_owners` packs by `cu`, the rectangle strides by `q`, and the
    /// two agree wherever every live slot feeds `q` rows (padding slots included).
    #[test]
    fn a_band_of_equal_widths_is_the_rectangle_table_for_table() {
        let pages: Vec<u32> = (0..3).collect();
        let seqs = [
            live(1, 40, &pages, 5),
            live(6, 33, &pages, 5),
            live(11, 17, &pages, 5),
            SeqFacts::padding(),
        ];
        let ids: Vec<i32> = (0..20).collect();
        let rect = StepIndex {
            phase: Phase::Decode,
            rows: 20,
            max_blocks: 3,
            seqs: &seqs,
            ids: &ids,
            band_width: None,
        };
        let band = StepIndex {
            band_width: Some(5),
            ..rect
        };
        let acc = [2u32, 5, 1, 1];
        for key in [
            "x.ids",
            "x.cu",
            "x.cache_idx",
            "x.sidx",
            "x.naccept",
            "x.has_init",
            "x.block_table",
            "x.seqused_k",
            "x.pos",
            "x.slot",
            "x.cos_sin",
        ] {
            let a = index_table_for(key, &rect, Accepted::PerSeq(&acc), AttnQLen::Live);
            let b = index_table_for(key, &band, Accepted::PerSeq(&acc), AttnQLen::Live);
            assert_eq!(a.expect(key), b.expect(key), "{key}");
        }
        // NEGATIVE CONTROL: a band whose slots feed DIFFERENT widths is not the rectangle — the
        // per-row tables move with `cu` (or this test compares two identical things).
        let ragged = [
            live(1, 40, &pages, 5),
            live(6, 33, &pages, 2),
            live(11, 17, &pages, 5),
            SeqFacts::padding(),
        ];
        let b2 = StepIndex {
            seqs: &ragged,
            ..band
        };
        let r2 = StepIndex {
            seqs: &ragged,
            band_width: None,
            ..band
        };
        let slot_b = index_table_for("x.slot", &b2, Accepted::AllRows, AttnQLen::Live).unwrap();
        let slot_r = index_table_for("x.slot", &r2, Accepted::AllRows, AttnQLen::Live).unwrap();
        assert_ne!(slot_b, slot_r);
        // The rectangle leaves slot 1's short span's tail at -1 IN PLACE (rows 7..10); the band
        // packs slot 2's rows straight after slot 1's two.
        assert_eq!(i64s(&slot_r)[7], -1);
        assert_eq!(i64s(&slot_b)[7], 17);
    }

    /// The GDN decode kernels' index arithmetic, as a host model (`saber_gdn_fused_update.hip`'s
    /// conv phase and both kernels' `si = sidx[n * stride + naccept - 1]` / `so = sidx[n * stride
    /// + t]`), run over a band's tables for several steps at CHANGING per-slot widths.
    struct GdnModel {
        /// The conv cache: `[slot][layout_len]` entries (one channel is enough for the law).
        conv: Vec<Vec<i64>>,
        /// Which (step, row) last wrote each recurrent snapshot slot.
        snap: Vec<Option<(usize, u32)>>,
        /// The last step's writers of each packed OUTPUT row (the conv's q/k/v and the
        /// recurrent update's `o`, both `bos + t` for `t < slen`).
        writers: Vec<Vec<usize>>,
    }

    const CONV_W: u32 = 4;

    impl GdnModel {
        fn new(slots: u32, max_w: u32) -> Self {
            let layout_len = (CONV_W - 1 + max_w - 1) as usize;
            GdnModel {
                conv: vec![vec![0; layout_len]; slots as usize],
                snap: vec![None; slots as usize],
                writers: Vec::new(),
            }
        }

        /// One step. `x` is the step's packed rows (the conv inputs); `len_from_cu` false is the
        /// NEGATIVE CONTROL that reads every sequence at `maxq` rows instead of its own. Returns
        /// the window each sequence's conv read and the snapshot each resumed from.
        #[allow(clippy::type_complexity)]
        fn step(
            &mut self,
            step_no: usize,
            s: &StepIndex<'_>,
            acc: &[u32],
            x: &[i64],
            len_from_cu: bool,
        ) -> Vec<Option<(Vec<i64>, Option<(usize, u32)>)>> {
            let q = s.q_len();
            let cu = i32s(&index_table("x.cu", s).unwrap());
            let sidx = i32s(&index_table("x.sidx", s).unwrap());
            let nacc = i32s(&index_table_with("x.naccept", s, Accepted::PerSeq(acc)).unwrap());
            // The kernargs a band plan publishes: stride = ci_stride = sidx_stride = q (the max
            // width), maxq = q, state_len_max = conv_width - 1 + maxq - 1.
            let (stride, maxq) = (q as usize, q as i32);
            let state_len_max = (CONV_W - 1) as i32 + maxq - 1;
            let hist_n = (CONV_W - 1) as usize;
            self.writers = vec![Vec::new(); x.len()];
            let mut out = Vec::new();
            for n in 0..s.seqs.len() {
                let bos = cu[n] as usize;
                let slen = if len_from_cu { cu[n + 1] - cu[n] } else { maxq };
                if cu[n + 1] - cu[n] <= 0 {
                    out.push(None);
                    continue;
                }
                let slot = sidx[n * stride] as usize; // cache_idx = sidx[:, 0] at ci_stride q
                let off = (nacc[n] - 1) as usize;
                let hist: Vec<i64> = (0..hist_n).map(|i| self.conv[slot][off + i]).collect();
                let slen_eff = state_len_max - (maxq - slen);
                for i in 0..slen_eff {
                    let v = if i + slen < slen_eff {
                        hist[(i + 1) as usize]
                    } else {
                        let ti = (i - (CONV_W as i32 - 2)) as usize;
                        x.get(bos + ti).copied().unwrap_or(-7)
                    };
                    self.conv[slot][i as usize] = v;
                }
                // The recurrent resume and the snapshot writes.
                let si = sidx[n * stride + off] as usize;
                let resumed = self.snap[si];
                for t in 0..slen as usize {
                    let so = sidx[n * stride + t] as usize;
                    self.snap[so] = Some((step_no, t as u32));
                    if let Some(w) = self.writers.get_mut(bos + t) {
                        w.push(n);
                    }
                }
                out.push(Some((hist, resumed)));
            }
            out
        }
    }

    /// **THE GDN LAW AT RAGGED WIDTHS** (C2): with `sidx`'s stride at the MAX width and every
    /// entry of a live run filled, each sequence's conv window is the last three inputs it
    /// COMMITTED and its recurrent state resumes from the snapshot its last committed row wrote —
    /// whatever width it verified last step and whatever width it verifies now, beside
    /// neighbours at other widths, and with a zero-row padding slot on the null slot touching
    /// nothing. This is why the ws-f38 floor (a slot never narrower than its last commit) is not
    /// needed in a band: `naccept - 1` indexes a `w`-wide block that is always whole.
    ///
    /// NEGATIVE CONTROLS: (1) a `sidx` that fills only this step's live rows (0 beyond) resumes a
    /// sequence whose last commit exceeds its new width from the null slot; (2) a walk that takes
    /// every sequence at `maxq` rows instead of its own `cu` span writes its neighbour's output
    /// rows. (Its conv WINDOW survives — the next read stays inside the sequence's own entries,
    /// `off + 2 <= naccept + 1 <= w + 1` — which is why the length law is gated on the rows.)
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn the_band_resumes_every_sequence_from_what_it_committed() {
        let max_w = 8u32;
        let pages: Vec<u32> = (0..4).collect();
        // (width, accepted) per step per sequence; slot 2 is padding throughout.
        let plan: [[(u32, u32); 2]; 4] = [
            [(8, 7), (3, 2)],
            [(2, 1), (8, 0)], // seq 0 committed 8 last step and verifies 2 now
            [(5, 4), (1, 0)], // seq 1 committed 1 and verifies 1
            [(1, 0), (6, 5)],
        ];
        let bases = [1u32, 1 + max_w];
        let slots_total = 1 + 2 * max_w;
        let run = |control_sidx: bool, len_from_cu: bool| -> std::result::Result<(), String> {
            let mut m = GdnModel::new(slots_total, max_w);
            // What each sequence has consumed, in order: the conv inputs of its committed rows.
            let mut consumed: [Vec<i64>; 2] = [Vec::new(), Vec::new()];
            // Which (step, row) wrote the snapshot of each sequence's last committed row.
            let mut last_row: [Option<(usize, u32)>; 2] = [None, None];
            let mut nacc = [1u32, 1];
            let mut done = [0u32, 0];
            for (step_no, cells) in plan.iter().enumerate() {
                let seqs: Vec<SeqFacts<'_>> = vec![
                    live(bases[0], done[0], &pages, cells[0].0),
                    live(bases[1], done[1], &pages, cells[1].0),
                    SeqFacts::padding(),
                ];
                let rows: u32 = cells.iter().map(|c| c.0).sum::<u32>() + 2; // two tail rows
                let ids = vec![0i32; rows as usize];
                let s = StepIndex {
                    phase: Phase::Decode,
                    rows,
                    max_blocks: 4,
                    seqs: &seqs,
                    ids: &ids,
                    band_width: Some(max_w),
                };
                let acc = [nacc[0], nacc[1], 1];
                s.check_with(Accepted::PerSeq(&acc))
                    .map_err(|e| e.to_string())?;
                // Distinct conv inputs per packed row: seq, step, row.
                let mut x: Vec<i64> = Vec::new();
                for (n, c) in cells.iter().enumerate() {
                    for t in 0..c.0 {
                        x.push(1_000 * (n as i64 + 1) + 10 * step_no as i64 + i64::from(t));
                    }
                }
                x.extend([-5, -5]);
                let got = if control_sidx {
                    // The control table: only this step's live rows filled.
                    m.step_with_live_only_sidx(step_no, &s, &acc, &x)
                } else {
                    m.step(step_no, &s, &acc, &x, len_from_cu)
                };
                for n in 0..2 {
                    let (window, resumed) = got[n].clone().ok_or("a live sequence skipped")?;
                    let mut want: Vec<i64> = vec![0; 3];
                    want.extend(consumed[n].iter().copied());
                    let want = want[want.len() - 3..].to_vec();
                    if window != want {
                        return Err(format!(
                            "step {step_no} seq {n}: conv window {window:?}, committed {want:?}"
                        ));
                    }
                    if resumed != last_row[n] {
                        return Err(format!(
                            "step {step_no} seq {n}: resumed from {resumed:?}, last commit wrote \
                             {:?}",
                            last_row[n]
                        ));
                    }
                }
                assert!(
                    got[2].is_none(),
                    "the padding slot is skipped by its zero span"
                );
                // Every packed row is written by its OWN sequence and nobody else, and the band's
                // tail rows by nobody (the control's `maxq` walk writes its neighbour's rows).
                if !control_sidx {
                    let cu = i32s(&index_table("x.cu", &s).unwrap());
                    for (r, w) in m.writers.iter().enumerate() {
                        let owner =
                            (0..seqs.len()).find(|n| (cu[*n]..cu[n + 1]).contains(&(r as i32)));
                        let want: Vec<usize> = owner.into_iter().collect();
                        if *w != want {
                            return Err(format!(
                                "step {step_no}: packed row {r} written by {w:?}, owned by {want:?}"
                            ));
                        }
                    }
                }
                // Commit: row 0 plus the accepted drafts are consumed.
                let mut at = 0usize;
                for (n, c) in cells.iter().enumerate() {
                    let (w, a) = *c;
                    consumed[n].extend(x[at..at + a as usize + 1].iter().copied());
                    last_row[n] = Some((step_no, a));
                    nacc[n] = a + 1;
                    done[n] += a + 1;
                    at += w as usize;
                }
                // The null slot is never written by a live sequence.
                if m.snap[0].is_some() && !control_sidx {
                    return Err("a live sequence wrote the null slot".to_owned());
                }
            }
            Ok(())
        };
        run(false, true).expect("the band's GDN law");
        let e = run(true, true).expect_err("the live-rows-only sidx control must fail");
        assert!(e.contains("resumed from"), "{e}");
        let e = run(false, false).expect_err("the maxq-length control must fail");
        assert!(e.contains("written by"), "{e}");
    }

    impl GdnModel {
        /// The control: the same step with `sidx` zeroed past each sequence's live rows.
        #[allow(clippy::type_complexity)]
        fn step_with_live_only_sidx(
            &mut self,
            step_no: usize,
            s: &StepIndex<'_>,
            acc: &[u32],
            x: &[i64],
        ) -> Vec<Option<(Vec<i64>, Option<(usize, u32)>)>> {
            let q = s.q_len() as usize;
            let cu = i32s(&index_table("x.cu", s).unwrap());
            let mut sidx = i32s(&index_table("x.sidx", s).unwrap());
            for n in 0..s.seqs.len() {
                let live_rows = (cu[n + 1] - cu[n]) as usize;
                for t in live_rows..q {
                    sidx[n * q + t] = 0;
                }
            }
            let nacc = i32s(&index_table_with("x.naccept", s, Accepted::PerSeq(acc)).unwrap());
            let mut out = Vec::new();
            for n in 0..s.seqs.len() {
                let slen = cu[n + 1] - cu[n];
                if slen <= 0 {
                    out.push(None);
                    continue;
                }
                let off = (nacc[n] - 1) as usize;
                let si = sidx[n * q + off] as usize;
                let resumed = self.snap[si];
                // The conv half is not what this control is about: take it from the real model.
                let slot = sidx[n * q] as usize;
                let hist: Vec<i64> = (0..3).map(|i| self.conv[slot][off + i]).collect();
                let state_len_max = 3 + q as i32 - 1;
                let slen_eff = state_len_max - (q as i32 - slen);
                for i in 0..slen_eff {
                    let v = if i + slen < slen_eff {
                        hist[(i + 1) as usize]
                    } else {
                        x[cu[n] as usize + (i - 2) as usize]
                    };
                    self.conv[slot][i as usize] = v;
                }
                for t in 0..slen as usize {
                    let so = sidx[n * q + t] as usize;
                    self.snap[so] = Some((step_no, t as u32));
                }
                out.push(Some((hist, resumed)));
            }
            out
        }
    }
}
