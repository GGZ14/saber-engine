//! The verify width — a table keyed on the BATCH by default, prod's acceptance-EMA cap on request,
//! and in both cases only over the plan's captured widths.
//!
//! # Which rule picks the width, and why the default is the table (ws-t2)
//!
//! Two rules live here and exactly one of them runs a step ([`WidthRule`]).
//!
//! * [`WidthByBatch`] — **the default**. The step's width is a function of how many sequences are
//!   running, read out of a table ([`WidthByBatch::prod_like`] = `1:8,4:5`): width 8 at batch 1–3,
//!   width 5 at batch 4 and above. It is a MEASUREMENT in a table (`CLAUDE.md` rule 6) and the
//!   cells are the two the orchestrator's paired BetterBench took on `main` `0b8a39f`, out of two
//!   separate boots: static width 8 reads **158.0 t/s single-stream / 416.2 conc-8** and static
//!   width 5 reads **131.7 / 427.7**. The table is what gets both numbers out of ONE boot —
//!   158.0 / 427.7 — because the single-stream cell and the conc-8 cell are different batches and
//!   a static width has to choose.
//! * [`WidthRule::AcceptanceEma`] — prod's `patch_dynwidth.py` cap, described below, **available
//!   and off by default**. It is off because it was measured and it LOST: 153.0 / 405.1 on the
//!   same paired BetterBench (below static 8 on both halves), and 260.0 conc-8 against static 8's
//!   273.1 on ws-t1's own re-sweep. The reason is structural and is written out in
//!   [`Self::slot_in_bucket`]: prod's cap is a per-request decision in a RAGGED batch, Saber
//!   replays one rectangular captured shape per step, and a rectangular bucket must cover its
//!   WIDEST slot — at eight streams there is almost always one sequence whose EMA still wants the
//!   ceiling. A rule that was measured to lose is kept, not deleted: it is one enum arm, its
//!   arithmetic is still gated against prod's, and `--width-rule ema` is how the next measurement
//!   asks for it.
//!
//! **Both rules ask for a width the plan CAPTURED** ([`VerifyWidthPolicy::bucket_width`] rounds up
//! to one), so a table cell of 5 on a plan whose widths are `1,8` serves 8 and says so at boot.
//! That is what keeps `--engine-decode-widths 1,8` (static 8) and `1,5` (static 5) reading exactly
//! as they did before this table existed, and it is why the table is not a behaviour change until
//! a boot captures both widths.
//!
//! # Where the EMA rule lives
//!
//! The *policy* is not written here. `saber-model::draft` ports `patch_dynwidth.py` and is gated
//! against prod's own code on 400 captured sequences / 2541 steps
//! (`docs/abi/DFLASH2-SEMANTICS.md` § 8), so this module holds [`draft::WidthPolicy`] and
//! [`draft::AcceptanceEma`] and adds exactly one thing on top: **a width Saber asks for must be a
//! width the plan captured**, because a width with no captured shape has no graph to replay.
//! That is row 7 of the mismatch table — a deliberate difference, recorded so it is not read as
//! drift.
//!
//! # The rule, as prod runs it
//!
//! Live knobs on `qwen_prod_nvfp4` (`SABER_DYNAMIC_WIDTH=1`): `ALPHA=0.35`, `MARGIN=2`,
//! `MIN=2`, `MIN_BATCH=3` — [`draft::WidthPolicy::PROD`], which is where this crate reads them
//! from rather than restating them.
//!
//! **Observe**, on every step that drafted:
//!
//! ```text
//!     obs = accepted + (accepted >= drafted)        // drafted = the SCHEDULED width
//!     ema = obs                  if there is no ema yet
//!         = 0.35*obs + 0.65*ema  otherwise
//! ```
//!
//! The `+1` is the climb-out, and it is the part that cannot be recovered from later: a request
//! that accepted its whole capped width observes *one above it*, so the EMA can climb back out of
//! its own shadow. Without it a cap at 3 never sees evidence for 5, because the only widths it
//! ever observes are the ones it already allows, and the cap is monotonically self-confirming.
//!
//! **Cap**, before scheduling:
//!
//! ```text
//!     if nothing was drafted:        leave it alone
//!     if running < MIN_BATCH (3):    leave it alone
//!     if there is no ema yet:        leave it alone   // cold start verifies full width
//!     w = max(ceil(ema) + 2, 2)
//!     if w < width:                  narrow to w      // it only ever shortens
//! ```
//!
//! Three properties are policy rather than bookkeeping:
//!
//! * **The batch gate is on the cap, never the observation.** History stays warm at every
//!   concurrency, so a cap applies the instant a batch forms. Below `MIN_BATCH` the batch's M sits
//!   in the weight-stream-bound flat zone where narrowing saves nothing and can only throw away a
//!   token that would have been accepted — the measured conc-2 dip.
//! * **Cold start verifies full width.** A new request is probed wide and pulled down, never the
//!   other way round — and it is probed, not *seeded*: there is no prior to blend the first real
//!   observation with.
//! * **It only ever shortens.** The cap cannot ask for a wider draft than the drafter produced.
//!
//! One consequence for the **plan**, because the batch gate and the captured bucket set meet here:
//! a batch that can run below `MIN_BATCH` — batches 1 and 2, i.e. every single-stream request —
//! verifies the *widest* width, so the plan has to capture `(batch, max spec width)` for those
//! batches or bucket selection has no shape to replay and refuses loudly
//! ([`crate::buckets::NoBucket::Combination`]).
//!
//! # Why the lever is the verify width at all
//!
//! DFlash2's block pass emits every position in one graphed pass, so the *draft* cost does not
//! vary with depth and the only per-step lever is how many positions the target is asked to
//! verify (`docs/LEDGER.md`, "Dynamic verify width": conc-8 **+11–13%**, single-stream unchanged).
//! Judge it on tok/s, never on accepted-per-draft: a configuration that degenerates into a
//! repetition loop scores higher acceptance *and* higher tok/s at once.

use saber_model::draft;

/// One sequence's acceptance EMA, in accepted **draft** tokens per draft — prod's, re-exported so
/// there is one implementation of the observation rule in the workspace.
pub use saber_model::draft::AcceptanceEma;

/// Why a `batch -> verify width` table is not usable.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WidthTableError {
    /// No rows at all: a table that answers nothing is not a policy.
    #[error("an empty batch -> width table: a step of any batch would have no width")]
    Empty,
    /// A row whose width is 0. A step verifies at least one row.
    #[error("batch {batch} maps to width 0; a step verifies at least one row")]
    ZeroWidth {
        /// The offending key.
        batch: u32,
    },
    /// A row whose batch is 0. The table is read at `running >= 1`.
    #[error("a row keyed on batch 0; the table is read at one running sequence or more")]
    ZeroBatch,
    /// The same batch twice, with two answers. Which one wins is not something to guess.
    #[error("batch {batch} appears twice ({a} and {b}); a table with two answers is not a table")]
    Duplicate {
        /// The repeated key.
        batch: u32,
        /// The first answer.
        a: u32,
        /// The second.
        b: u32,
    },
    /// The text is not `batch:width[,batch:width…]`.
    #[error("`{text}` is not a batch -> width table (expected `1:8,4:5`): {detail}")]
    Syntax {
        /// What was passed.
        text: String,
        /// Which part of it failed.
        detail: String,
    },
}

/// The verify width as a function of the step's BATCH — a measured table (`CLAUDE.md` rule 6).
///
/// # The law, which is prod's own carry-forward
///
/// The row that applies to a step of `n` running sequences is the one with the **greatest key at
/// or below `n`**, and a batch below the first key takes the first row — `_batch_ceil` in
/// `ref/ref_draft.py`, the shape prod's own `SABER_DRAFT_SCHEDULE`
/// (`1:8,2:7,4:6,8:5,16:4`) is written in. That knob is not the verify width and never was (it
/// caps V1's serial MTP forwards and `SPEC_METHOD=dflash` never reads it — ws-t1); what is
/// borrowed here is the TABLE and its carry-forward, not its numbers.
///
/// # Why a table and not a curve
///
/// Each cell is one measured configuration, so the table can only ever say what was measured. The
/// default's two rows are the two cells the orchestrator's paired BetterBench took on `main`
/// `0b8a39f` (static 8: 158.0 single / 416.2 conc-8; static 5: 131.7 / 427.7). **Batch 2 and
/// batch 3 are NOT measured cells** — they carry the batch-1 answer, which is the conservative
/// direction, because narrowing a batch that is still in the weight-stream-bound flat zone can
/// only throw away a token that would have been accepted (prod's own `MIN_BATCH = 3` says the
/// same thing about the same region).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WidthByBatch {
    /// `(batch, width)` ascending by batch, deduplicated at construction.
    rows: Vec<(u32, u32)>,
}

impl WidthByBatch {
    /// The default table: width 8 at batch 1–3, width 5 at batch 4 and above.
    pub const PROD_LIKE: &'static [(u32, u32)] = &[(1, 8), (4, 5)];

    /// [`Self::PROD_LIKE`] as a table.
    #[must_use]
    pub fn prod_like() -> Self {
        Self {
            rows: Self::PROD_LIKE.to_vec(),
        }
    }

    /// A table from its rows.
    ///
    /// # Errors
    /// An empty table, a zero batch or width, or the same batch twice.
    pub fn new(rows: impl IntoIterator<Item = (u32, u32)>) -> Result<Self, WidthTableError> {
        let mut rows: Vec<(u32, u32)> = rows.into_iter().collect();
        if rows.is_empty() {
            return Err(WidthTableError::Empty);
        }
        rows.sort_unstable();
        for w in &rows {
            if w.0 == 0 {
                return Err(WidthTableError::ZeroBatch);
            }
            if w.1 == 0 {
                return Err(WidthTableError::ZeroWidth { batch: w.0 });
            }
        }
        if let Some((a, b)) = rows
            .windows(2)
            .find(|p| p[0].0 == p[1].0)
            .map(|p| (p[0], p[1]))
        {
            if a.1 != b.1 {
                return Err(WidthTableError::Duplicate {
                    batch: a.0,
                    a: a.1,
                    b: b.1,
                });
            }
        }
        rows.dedup();
        Ok(Self { rows })
    }

    /// Parse `1:8,4:5`.
    ///
    /// # Errors
    /// Anything [`Self::new`] refuses, and any token that is not `batch:width`.
    pub fn parse(text: &str) -> Result<Self, WidthTableError> {
        let mut rows = Vec::new();
        for tok in text.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            let (b, w) = tok.split_once(':').ok_or_else(|| WidthTableError::Syntax {
                text: text.to_owned(),
                detail: format!("`{tok}` has no `:`"),
            })?;
            let parse = |s: &str, what: &str| -> Result<u32, WidthTableError> {
                s.trim()
                    .parse::<u32>()
                    .map_err(|e| WidthTableError::Syntax {
                        text: text.to_owned(),
                        detail: format!("{what} `{s}`: {e}"),
                    })
            };
            rows.push((parse(b, "batch")?, parse(w, "width")?));
        }
        Self::new(rows)
    }

    /// The width this table asks for at `batch` running sequences.
    #[must_use]
    pub fn width_for(&self, batch: u32) -> u32 {
        let mut w = self.rows[0].1;
        for &(b, v) in &self.rows {
            if batch >= b {
                w = v;
            } else {
                break;
            }
        }
        w.max(1)
    }

    /// The rows, ascending by batch.
    #[must_use]
    pub fn rows(&self) -> &[(u32, u32)] {
        &self.rows
    }

    /// The distinct widths the table can ask for, ascending.
    #[must_use]
    pub fn widths(&self) -> Vec<u32> {
        let mut out: Vec<u32> = self.rows.iter().map(|r| r.1).collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

impl std::fmt::Display for WidthByBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for (b, w) in &self.rows {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "{b}:{w}")?;
            first = false;
        }
        Ok(())
    }
}

/// Which rule picks a step's verify width. See this module's header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WidthRule {
    /// The width is a function of the BATCH, read out of a measured table. The default.
    ByBatch(WidthByBatch),
    /// prod's acceptance-EMA cap (`patch_dynwidth.py`). Available, and off by default because it
    /// was measured against the table and the static widths and lost at both concurrencies.
    AcceptanceEma,
}

impl Default for WidthRule {
    fn default() -> Self {
        WidthRule::ByBatch(WidthByBatch::prod_like())
    }
}

impl WidthRule {
    /// The rule named by `batch` / `ema`, with an optional table after `batch:` (`batch=1:8,4:5`).
    ///
    /// # Errors
    /// An unknown name, or a table the parser refuses.
    pub fn parse(text: &str) -> Result<Self, WidthTableError> {
        let t = text.trim();
        match t.split_once('=') {
            Some(("batch", table)) => Ok(WidthRule::ByBatch(WidthByBatch::parse(table)?)),
            None if t.eq_ignore_ascii_case("batch") => Ok(WidthRule::default()),
            None if t.eq_ignore_ascii_case("ema") => Ok(WidthRule::AcceptanceEma),
            _ => Err(WidthTableError::Syntax {
                text: text.to_owned(),
                detail: "expected `batch`, `batch=1:8,4:5` or `ema`".to_owned(),
            }),
        }
    }

    /// The table this rule reads, or `None` for the EMA cap.
    #[must_use]
    pub fn table(&self) -> Option<&WidthByBatch> {
        match self {
            WidthRule::ByBatch(t) => Some(t),
            WidthRule::AcceptanceEma => None,
        }
    }
}

impl std::fmt::Display for WidthRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WidthRule::ByBatch(t) => write!(f, "batch={t}"),
            WidthRule::AcceptanceEma => write!(f, "ema"),
        }
    }
}

/// **Where a RAGGED step's per-slot ceiling comes from** (workstream C2, `--ragged-ceiling`).
///
/// A ragged step gives every running slot its OWN verify width: prod's carry-forward
/// acceptance-EMA cap ([`VerifyWidthPolicy::slot_cap`], `patch_dynwidth.py`'s
/// `_ref_cap_spec_width`), which only ever NARROWS a slot below a ceiling. This is the
/// ceiling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RaggedCeiling {
    /// **The default ceiling of a ragged schedule** (the schedule itself is opt-in since C3,
    /// `--ragged-widths`): the batch table's cell for this step's running count
    /// ([`WidthByBatch::width_for`], `1:8,4:5`), so the cap narrows below what the rectangular
    /// schedule would have verified and never above it. Under [`WidthRule::AcceptanceEma`] (no
    /// table) this reads as [`RaggedCeiling::Block`].
    #[default]
    Table,
    /// The plan's MAX width — prod's own ceiling (every request's block is `num_speculative_tokens`
    /// long before the cap). C3's paired BetterBench at the N8 defaults: 501.3 conc-8 against the
    /// table's 495.9 and the rectangles' 501.8 — not a win past the MDE, so not a default.
    Block,
}

impl RaggedCeiling {
    /// `table` or `block`.
    ///
    /// # Errors
    /// Anything else, by name.
    pub fn parse(text: &str) -> Result<Self, WidthTableError> {
        match text.trim() {
            "table" => Ok(RaggedCeiling::Table),
            "block" => Ok(RaggedCeiling::Block),
            other => Err(WidthTableError::Syntax {
                text: other.to_owned(),
                detail: "expected `table` or `block`".to_owned(),
            }),
        }
    }
}

impl std::fmt::Display for RaggedCeiling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            RaggedCeiling::Table => "table",
            RaggedCeiling::Block => "block",
        })
    }
}

/// One slot's rows in a ragged step, and whether prod's cap narrowed it (C2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaggedSlot {
    /// Rows this slot verifies: `1 + draft tokens`, `1` on a sequence's first decode step.
    pub rows: u32,
    /// Whether the acceptance-EMA cap put it below the step's ceiling.
    pub narrowed: bool,
}

/// The verify-width policy: which rule runs, prod's knobs, and the plan's captured widths.
#[derive(Clone, Debug, PartialEq)]
pub struct VerifyWidthPolicy {
    /// Speculative widths the plan captured, ascending. The only widths this scheduler may ask
    /// for, because a width with no captured shape has no graph.
    pub widths: Vec<u32>,
    /// `alpha` / `margin` / `min_width` / `min_batch`, from `saber-model`'s port of
    /// `patch_dynwidth.py`. Defaults to [`draft::WidthPolicy::PROD`]. Read only by
    /// [`WidthRule::AcceptanceEma`].
    pub prod: draft::WidthPolicy,
    /// Which rule picks the step's width. [`WidthRule::ByBatch`] by default (ws-t2).
    pub rule: WidthRule,
    /// **RAGGED per-slot widths (C2)**: `Some(ceiling)` gives every slot of a decode step its own
    /// width ([`VerifyWidthPolicy::ragged_slot`]) and runs the step in a ROW BAND; `None` is the
    /// rectangular schedule (`--rectangular-widths`, the pairing control), under which `rule`
    /// alone decides. The scheduler takes the ragged path only when the plan captured bands.
    pub ragged: Option<RaggedCeiling>,
}

/// What one slot of a chosen bucket verifies, and what its own acceptance EMA would have asked
/// for. They differ whenever the captured ladder could not offer a width as narrow as the cap;
/// see [`VerifyWidthPolicy::slot_in_bucket`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotWidth {
    /// Rows this slot verifies — the bucket's width, because the bucket launches them anyway.
    pub verify: u32,
    /// The width this slot's own policy asked for — the batch table's answer for this step's
    /// batch, or this sequence's own EMA cap when the EMA rule is on. Never above `verify`.
    pub own_cap: u32,
}

impl SlotWidth {
    /// Whether the rectangular bucket absorbed a narrowing prod's ragged batch would have taken.
    #[must_use]
    pub fn absorbed(&self) -> bool {
        self.own_cap < self.verify
    }
}

impl VerifyWidthPolicy {
    /// The policy over a plan's captured widths, with prod's knobs and the default rule (the
    /// batch table).
    pub fn from_widths(mut widths: Vec<u32>) -> Self {
        widths.sort_unstable();
        widths.dedup();
        Self {
            widths,
            prod: draft::WidthPolicy::PROD,
            rule: WidthRule::default(),
            ragged: None,
        }
    }

    /// The same policy with ragged per-slot widths on (`Some`) or off (C2).
    #[must_use]
    pub fn with_ragged(mut self, ragged: Option<RaggedCeiling>) -> Self {
        self.ragged = ragged;
        self
    }

    /// **A ragged step's ceiling** at `running` sequences (C2): the table's cell
    /// ([`RaggedCeiling::Table`]) or the plan's max width ([`RaggedCeiling::Block`]), never above
    /// the widest captured width. The table's cell is taken EXACTLY — a band captures any width
    /// multiset, so there is nothing to round up to.
    #[must_use]
    pub fn ragged_ceiling(&self, running: u32) -> u32 {
        let block = self.ceiling_width();
        match (self.ragged.unwrap_or_default(), self.rule.table()) {
            (RaggedCeiling::Table, Some(t)) => t.width_for(running).min(block),
            _ => block,
        }
        .max(1)
    }

    /// **One slot's rows in a ragged step** (C2): prod's `_ref_cap_spec_width`, per request.
    ///
    /// The cap lives in DRAFT-TOKEN space, as prod's does: with `ceiling` rows the slot's block is
    /// `ceiling - 1` draft tokens, `w = max(ceil(ema) + MARGIN, MIN)` and, when `w` is below the
    /// block, `del spec_token_ids[w:]` — the unverified tail is DROPPED and the slot verifies
    /// `w + 1` rows. A cold EMA or a batch below `MIN_BATCH` keeps the whole block. A sequence on
    /// its FIRST decode step drafts nothing and is one row (prod's request with no
    /// `spec_token_ids` contributes one row to its ragged `M`).
    #[must_use]
    pub fn ragged_slot(&self, ema: &AcceptanceEma, running: u32, first_step: bool) -> RaggedSlot {
        if first_step {
            return RaggedSlot {
                rows: 1,
                narrowed: false,
            };
        }
        let ceiling = self.ragged_ceiling(running);
        match self.prod.cap(ema, running, ceiling - 1) {
            Some(drafts) => RaggedSlot {
                rows: drafts + 1,
                narrowed: true,
            },
            None => RaggedSlot {
                rows: ceiling,
                narrowed: false,
            },
        }
    }

    /// The same policy under another rule.
    #[must_use]
    pub fn with_rule(mut self, rule: WidthRule) -> Self {
        self.rule = rule;
        self
    }

    /// The table this policy reads, or `None` when the EMA cap is running.
    #[must_use]
    pub fn table(&self) -> Option<&WidthByBatch> {
        self.rule.table()
    }

    /// The widest captured width — what a cold or un-gated step verifies.
    pub fn ceiling_width(&self) -> u32 {
        self.widths.last().copied().unwrap_or(1)
    }

    /// The narrowest captured width.
    pub fn floor_width(&self) -> u32 {
        self.widths.first().copied().unwrap_or(1)
    }

    /// A fresh per-sequence EMA: no value, no prior.
    pub fn new_ema(&self) -> AcceptanceEma {
        AcceptanceEma::new()
    }

    /// The cap for one sequence drafting `block_width` tokens with `running` requests in flight,
    /// or `None` when the policy does not bind this step (nothing drafted, below `MIN_BATCH`,
    /// cold start, or a cap that would not shorten).
    ///
    /// This is [`draft::WidthPolicy::cap`] unchanged; it is wrapped rather than reimplemented so
    /// the gated rule and the scheduled rule cannot drift. **It is the EMA rule's answer whatever
    /// rule is running** — under [`WidthRule::ByBatch`] the EMAs are still observed and this is
    /// what they would have asked for, which is what makes the two rules comparable out of one
    /// run rather than one boot apiece.
    pub fn slot_cap(&self, ema: &AcceptanceEma, running: u32, block_width: u32) -> Option<u32> {
        self.prod.cap(ema, running, block_width)
    }

    /// What one sequence verifies under the RUNNING rule: the batch table's answer, or the EMA's
    /// cap, never wider than `block_width`.
    pub fn slot_width(&self, ema: &AcceptanceEma, running: u32, block_width: u32) -> u32 {
        match &self.rule {
            WidthRule::ByBatch(t) => t.width_for(running).min(block_width),
            WidthRule::AcceptanceEma => self
                .slot_cap(ema, running, block_width)
                .unwrap_or(block_width),
        }
    }

    /// What one slot verifies **inside the bucket this step already chose**, and the cap it would
    /// have asked for on its own.
    ///
    /// # Why that is two numbers here and one number in prod
    ///
    /// prod's cap deletes entries from `request.spec_token_ids` (`del …[w:]`, `patch_dynwidth.py`)
    /// and the reference server's decode batch is **ragged**: a request capped to `w` removes `block − w` rows
    /// from the step's `M`, and `block − w` GDN state slots with them. There the cap is a
    /// slot-level decision because the slot is what the batch is made of.
    ///
    /// Saber replays ONE CAPTURED SHAPE per step. `bucket.batch × bucket.spec_width` rows are
    /// launched and the same number of GDN state slots is written whatever any slot asks for
    /// (`Bucket::gdn_state_slots`). A slot that verifies fewer rows than its bucket therefore
    /// pays for every one of them and can only **accept fewer tokens**: the rows past its cap are
    /// padded with a token the model cannot want, and the draft the drafter already produced for
    /// those positions is thrown away. Truncating a slot below its own bucket is pure loss, in the
    /// one direction that never shows up as a step-time regression.
    ///
    /// So the cap picks the **bucket** ([`Self::step_width`]) and every slot in it verifies the
    /// bucket's width. `own_cap` is carried out so the absorption is a COUNTER rather than an
    /// invisible policy difference (`CLAUDE.md` rule 5).
    ///
    /// Under [`WidthRule::ByBatch`] every slot asks for the same number — the table's — so the
    /// counter reads zero unless the plan captured no width as narrow as the table's cell and the
    /// rounding up put the step in a wider bucket. That is the same fact the boot banner's
    /// `reachable` line states, counted per step.
    pub fn slot_in_bucket(
        &self,
        ema: &AcceptanceEma,
        running: u32,
        bucket_width: u32,
    ) -> SlotWidth {
        SlotWidth {
            verify: bucket_width,
            own_cap: self.slot_width(ema, running, bucket_width),
        }
    }

    /// The captured widths the RUNNING rule can actually select, ascending.
    ///
    /// Under [`WidthRule::ByBatch`] that is the table's own widths, each rounded up to a captured
    /// one: a table of `1:8,4:5` on a plan whose widths are `1,8` reaches `[8]` and is inert,
    /// which is a boot-time fact and is printed as one.
    ///
    /// Under [`WidthRule::AcceptanceEma`] the cap is `max(ceil(ema) + margin, min_width)` and is
    /// then rounded UP to a captured width, so the narrowest bucket it can ever reach is
    /// `bucket_width(max(min_width, margin))`. A ladder that offers only 1 and the ceiling reaches
    /// the ceiling only, and the dynamic width is then inert however warm the EMAs are.
    pub fn reachable_widths(&self) -> Vec<u32> {
        let ceiling = self.ceiling_width();
        let mut out: Vec<u32> = match &self.rule {
            WidthRule::ByBatch(t) => t
                .widths()
                .into_iter()
                .map(|w| self.bucket_width(w))
                .collect(),
            WidthRule::AcceptanceEma => {
                let floor_cap = self.prod.min_width.max(self.prod.margin).max(1);
                let mut v: Vec<u32> = (floor_cap..=ceiling.max(floor_cap))
                    .map(|cap| self.bucket_width(cap))
                    .collect();
                v.push(ceiling);
                v
            }
        };
        out.sort_unstable();
        out.dedup();
        out
    }

    /// `true` when the cap has a narrower captured width to fall back to.
    ///
    /// A serve whose plan captured only `1` and the ceiling is a serve where the EMA buys nothing,
    /// and that is worth a line in the boot banner rather than a throughput number nobody can
    /// explain.
    pub fn can_narrow(&self) -> bool {
        self.reachable_widths().len() > 1
    }

    /// The captured width a step of `cap` runs in: the smallest available width that covers it.
    ///
    /// Rounding **up** is what keeps the cap from becoming a truncation the drafter did not
    /// intend — a bucket narrower than a sequence's cap would throw away tokens the policy asked
    /// to verify.
    pub fn bucket_width(&self, cap: u32) -> u32 {
        self.widths
            .iter()
            .copied()
            .find(|w| *w >= cap)
            .unwrap_or_else(|| self.ceiling_width())
    }

    /// The captured width this step runs, under the rule that is on.
    ///
    /// Under [`WidthRule::ByBatch`] it is the table's answer for `running`, rounded up to a
    /// captured width — the EMAs are not read, which is the whole of the rule: the width is a
    /// function of the batch and of nothing else, so it does not move mid-stream and one arriving
    /// request cannot pull a settled batch to another shape.
    ///
    /// Under [`WidthRule::AcceptanceEma`] it is the widest sequence's cap, rounded up, because a
    /// rectangular bucket has to cover its widest slot.
    ///
    /// `running` is the batch, counted the way prod counts `len(self.running)` — and it is the
    /// batch the `MIN_BATCH` gate is tested against on the EMA path.
    pub fn step_width<'a>(
        &self,
        running: u32,
        emas: impl IntoIterator<Item = &'a AcceptanceEma>,
    ) -> u32 {
        let ceiling = self.ceiling_width();
        match &self.rule {
            WidthRule::ByBatch(t) => self.bucket_width(t.width_for(running)),
            WidthRule::AcceptanceEma => {
                let widest = emas
                    .into_iter()
                    .map(|ema| self.slot_width(ema, running, ceiling))
                    .max()
                    .unwrap_or(ceiling);
                self.bucket_width(widest)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The EMA rule over a `1,3,5` ladder. **Named**, because it is no longer the default: every
    /// test below that asks what an acceptance EMA does has to say that it wants that rule, and a
    /// test that forgets gets the batch table's answer and fails.
    fn policy() -> VerifyWidthPolicy {
        VerifyWidthPolicy::from_widths(vec![1, 3, 5]).with_rule(WidthRule::AcceptanceEma)
    }

    /// prod's `_ref_cap_spec_width` (`patch_dynwidth.py`), transcribed: the number of draft
    /// tokens a request KEEPS out of a `spec_len`-token block.
    ///
    /// ```text
    /// if running < MIN_BATCH or ema is None: return          # keep all
    /// w = max(ceil(ema) + MARGIN, MIN_W)
    /// if w < len(spec_token_ids): del spec_token_ids[w:]
    /// ```
    fn prod_keeps(ema: Option<f64>, running: u32, spec_len: u32) -> u32 {
        let Some(e) = ema else { return spec_len };
        if running < 3 {
            return spec_len;
        }
        let w = (e.ceil() as u32 + 2).max(2);
        if w < spec_len {
            w
        } else {
            spec_len
        }
    }

    /// **C2 host test (f): the ragged per-slot rule IS prod's cap**, over a grid of EMA values,
    /// running counts and both ceiling sources, with the first-step and cold-start answers.
    #[test]
    fn a_ragged_slot_verifies_prods_capped_block_plus_one() {
        let table = WidthByBatch::prod_like();
        let base = VerifyWidthPolicy::from_widths(vec![1, 5, 8]);
        for ceiling_src in [RaggedCeiling::Table, RaggedCeiling::Block] {
            let p = base.clone().with_ragged(Some(ceiling_src));
            for running in 1..=8u32 {
                let ceiling = match ceiling_src {
                    RaggedCeiling::Table => table.width_for(running),
                    RaggedCeiling::Block => 8,
                };
                assert_eq!(
                    p.ragged_ceiling(running),
                    ceiling,
                    "{ceiling_src} r{running}"
                );
                // Cold: the whole block, never narrowed.
                let cold = p.new_ema();
                assert_eq!(
                    p.ragged_slot(&cold, running, false),
                    RaggedSlot {
                        rows: ceiling,
                        narrowed: false
                    }
                );
                // First step: one row, whatever the EMA.
                assert_eq!(p.ragged_slot(&cold, running, true).rows, 1);
                for (accepted, drafted, steps) in [
                    (0u32, 4u32, 20),
                    (1, 4, 20),
                    (2, 7, 20),
                    (4, 4, 30),
                    (7, 7, 30),
                    (3, 7, 5),
                ] {
                    let mut ema = p.new_ema();
                    for _ in 0..steps {
                        ema.observe(&p.prod, accepted, drafted);
                    }
                    let got = p.ragged_slot(&ema, running, false);
                    let keeps = prod_keeps(ema.value(), running, ceiling - 1);
                    assert_eq!(
                        got.rows,
                        keeps + 1,
                        "{ceiling_src} running {running} ema {:?}",
                        ema.value()
                    );
                    assert_eq!(got.narrowed, keeps < ceiling - 1);
                    assert!(got.rows <= ceiling && got.rows >= 1);
                }
            }
        }
        // Below MIN_BATCH nothing binds even a starved EMA; at MIN_BATCH it drops to MIN + 1 rows.
        let p = base.clone().with_ragged(Some(RaggedCeiling::Block));
        let mut starved = p.new_ema();
        for _ in 0..20 {
            starved.observe(&p.prod, 0, 7);
        }
        assert_eq!(p.ragged_slot(&starved, 2, false).rows, 8);
        assert_eq!(
            p.ragged_slot(&starved, 3, false).rows,
            3,
            "MIN 2 drafts + the input row"
        );
        // Under the EMA RULE there is no table, so the table ceiling reads as the block.
        let ema_rule = base
            .with_rule(WidthRule::AcceptanceEma)
            .with_ragged(Some(RaggedCeiling::Table));
        assert_eq!(ema_rule.ragged_ceiling(8), 8);
        // NEGATIVE CONTROL: prod's cap read in ROW space (w rows, not w drafts + 1) is a
        // different rule and this comparison must see it.
        let mut ema = p.new_ema();
        for _ in 0..20 {
            ema.observe(&p.prod, 1, 7);
        }
        let rows_space = p.prod.cap(&ema, 8, 8).unwrap_or(8);
        assert_ne!(rows_space, p.ragged_slot(&ema, 8, false).rows);
        assert_eq!(RaggedCeiling::parse("table"), Ok(RaggedCeiling::Table));
        assert_eq!(RaggedCeiling::parse("block"), Ok(RaggedCeiling::Block));
        assert!(RaggedCeiling::parse("rows").is_err());
    }

    /// A batch big enough for the cap to apply at all.
    const BATCH: u32 = 3;

    #[test]
    fn the_knobs_are_prods_not_this_crates() {
        let p = policy();
        assert_eq!(p.prod, draft::WidthPolicy::PROD);
        assert_eq!(p.prod.alpha, 0.35);
        assert_eq!(p.prod.margin, 2);
        assert_eq!(p.prod.min_width, 2);
        assert_eq!(p.prod.min_batch, 3);
    }

    #[test]
    fn a_cold_sequence_verifies_the_full_width() {
        let p = policy();
        let ema = p.new_ema();
        assert_eq!(ema.value(), None, "no prior is seeded");
        assert_eq!(p.slot_cap(&ema, BATCH, 5), None, "cold start does not bind");
        assert_eq!(p.slot_width(&ema, BATCH, 5), 5);
        assert_eq!(p.step_width(BATCH, [&ema]), 5);
    }

    #[test]
    fn below_min_batch_the_cap_does_not_bind() {
        // Row 2 of the mismatch table: at concurrency 1 and 2 the decode GEMMs are
        // weight-stream-bound, so narrowing saves nothing and can only truncate a token that
        // would have been accepted.
        let p = policy();
        let mut ema = p.new_ema();
        for _ in 0..20 {
            ema.observe(&p.prod, 0, 5);
        }
        assert!(ema.value().unwrap() < 0.1);
        for running in [1u32, 2] {
            assert_eq!(p.slot_cap(&ema, running, 5), None, "running={running}");
            assert_eq!(p.slot_width(&ema, running, 5), 5);
            assert_eq!(p.step_width(running, [&ema]), 5);
        }
        assert_eq!(p.slot_cap(&ema, 3, 5), Some(2), "and it binds at 3");
    }

    #[test]
    fn the_cap_bottoms_out_at_two_and_never_asks_for_the_captured_width_1() {
        // Row 5. The plan captures width 1, but a cap is `ceil(ema) + margin 2`, so the narrowest
        // width the policy can ever ask for is 2 — and the bucket that covers 2 is width 3. The
        // `min_width = 2` knob agrees with that and is, at prod's margin, unreachable: the
        // dflash width gate measures it as changing 0 of 2541 captured steps.
        let p = policy();
        let mut ema = p.new_ema();
        for _ in 0..20 {
            ema.observe(&p.prod, 0, 5);
        }
        assert_eq!(p.slot_cap(&ema, BATCH, 5), Some(2));
        assert_eq!(p.bucket_width(2), 3);
        assert_eq!(p.step_width(BATCH, [&ema]), 3);
    }

    #[test]
    fn full_acceptance_observes_one_above_the_width() {
        // Row 1, the climb-out, at the arithmetic level: an EMA fed nothing but full acceptance
        // at width 3 settles at 4, not at 3, which is what lets the cap climb back past a width
        // it has already narrowed to. Without the `+1` this converges to exactly 3 and the cap is
        // self-confirming forever.
        let p = policy();
        let mut ema = p.new_ema();
        for _ in 0..40 {
            ema.observe(&p.prod, 3, 3);
        }
        let v = ema.value().expect("observed");
        assert!((v - 4.0).abs() < 1e-9, "ema {v}");
        assert_eq!(p.slot_cap(&ema, BATCH, 8), Some(6), "ceil(4) + 2");
    }

    #[test]
    fn a_partial_acceptance_does_not_get_the_extra_token() {
        let p = policy();
        let mut ema = p.new_ema();
        ema.observe(&p.prod, 2, 5);
        assert_eq!(
            ema.value(),
            Some(2.0),
            "2 accepted of 5 is an observation of 2"
        );
        let mut full = p.new_ema();
        full.observe(&p.prod, 5, 5);
        assert_eq!(full.value(), Some(6.0), "5 of 5 is an observation of 6");
    }

    #[test]
    fn width_0_steps_are_not_observations() {
        let p = policy();
        let mut ema = p.new_ema();
        ema.observe(&p.prod, 3, 5);
        let before = ema;
        ema.observe(&p.prod, 0, 0);
        assert_eq!(
            ema, before,
            "a non-drafting step says nothing about the drafter"
        );
    }

    #[test]
    fn the_cap_only_ever_shortens() {
        let p = policy();
        let mut ema = p.new_ema();
        for _ in 0..20 {
            ema.observe(&p.prod, 5, 5);
        }
        // ceil(6) + 2 = 8, which is wider than the block: the policy leaves it alone rather than
        // asking for a draft the drafter did not produce.
        assert_eq!(p.slot_cap(&ema, BATCH, 5), None);
        assert_eq!(p.slot_width(&ema, BATCH, 5), 5);
    }

    #[test]
    fn the_step_width_is_the_widest_cap_rounded_up_to_a_captured_width() {
        let p = policy();
        let mut cold = p.new_ema();
        let mut warm = p.new_ema();
        for _ in 0..20 {
            cold.observe(&p.prod, 0, 5);
            warm.observe(&p.prod, 2, 5);
        }
        assert_eq!(p.slot_cap(&cold, BATCH, 5), Some(2));
        assert_eq!(p.slot_cap(&warm, BATCH, 5), Some(4));
        assert_eq!(p.step_width(BATCH, [&cold, &warm]), 5, "4 rounds up to 5");
        assert_eq!(p.step_width(BATCH, [&cold]), 3, "2 rounds up to 3");
    }

    #[test]
    fn a_slot_verifies_its_buckets_width_and_the_absorption_is_counted() {
        // The law: the cap picks the bucket, the bucket is what every slot verifies. A slot whose
        // own cap is narrower is `absorbed` — the rows are launched either way, so verifying
        // fewer of them would only throw accepted tokens away.
        let p = policy();
        let mut cold = p.new_ema();
        for _ in 0..20 {
            cold.observe(&p.prod, 0, 5);
        }
        assert_eq!(p.slot_cap(&cold, BATCH, 5), Some(2), "its own cap is 2");
        // …and 2 rounds up to the captured width 3, which is the bucket this step runs.
        let bucket = p.step_width(BATCH, [&cold]);
        assert_eq!(bucket, 3);
        let slot = p.slot_in_bucket(&cold, BATCH, bucket);
        assert_eq!(slot.verify, 3, "it verifies the bucket, not its own 2");
        assert_eq!(slot.own_cap, 2);
        assert!(slot.absorbed(), "the rounding up is counted, not hidden");

        // The CONTROL, and it is the defect this replaced: with only `1` and the ceiling captured
        // there is no narrower bucket, so the bucket stays 5 while the slot's own cap is 2. The
        // old rule verified 2 rows of a 5-row bucket — three rows launched, three rows of draft
        // discarded, nothing saved.
        let flat = VerifyWidthPolicy::from_widths(vec![1, 5]).with_rule(WidthRule::AcceptanceEma);
        assert_eq!(flat.step_width(BATCH, [&cold]), 5, "nothing to narrow to");
        let slot = flat.slot_in_bucket(&cold, BATCH, 5);
        assert_eq!(slot.verify, 5);
        assert_eq!(
            slot.own_cap, 2,
            "the old rule's answer, kept as the counter"
        );
        assert!(slot.absorbed());
    }

    #[test]
    fn a_ladder_of_one_and_the_ceiling_cannot_narrow() {
        // Row of the serve banner: `--engine-decode-widths 1,8` gives the EMA nothing to reach.
        // The cap bottoms out at `max(min_width 2, margin 2)` = 2, and 2 rounds up to 8.
        let flat = VerifyWidthPolicy::from_widths(vec![1, 8]).with_rule(WidthRule::AcceptanceEma);
        assert_eq!(flat.reachable_widths(), vec![8]);
        assert!(!flat.can_narrow());

        // Add the widths prod's cap actually lands on and it can.
        let ladder =
            VerifyWidthPolicy::from_widths(vec![1, 3, 5, 8]).with_rule(WidthRule::AcceptanceEma);
        assert_eq!(ladder.reachable_widths(), vec![3, 5, 8]);
        assert!(ladder.can_narrow());
        // Width 1 is NOT reachable by the cap — `min_width`/`margin` are both 2 — which is why it
        // is absent above and why a plan that captures only `1` and `8` has one usable width.
        assert!(!ladder.reachable_widths().contains(&1));
    }

    // ---- the width as a function of the BATCH (ws-t2) ------------------------------------------

    #[test]
    fn the_default_rule_is_the_batch_table_and_its_cells_are_the_measured_ones() {
        // The default is the table, not the EMA cap: the cap was measured against it and lost
        // (153.0/405.1 against static 8's 158.0/416.2 on the paired BetterBench).
        let p = VerifyWidthPolicy::from_widths(vec![1, 5, 8]);
        assert_eq!(p.rule, WidthRule::ByBatch(WidthByBatch::prod_like()));
        let t = p.table().expect("the default rule reads a table");
        assert_eq!(t.rows(), &[(1, 8), (4, 5)]);
        // Single-stream takes static 8's cell and conc-8 takes static 5's, out of ONE plan.
        assert_eq!(p.step_width(1, []), 8);
        assert_eq!(p.step_width(2, []), 8);
        assert_eq!(p.step_width(8, []), 5);
    }

    #[test]
    fn the_carry_forward_is_prods_batch_ceil() {
        // `_batch_ceil` (ref_draft.py): the greatest key at or below the batch, and the
        // first row for anything below the first key.
        let t = WidthByBatch::parse("2:7,4:6,8:5,16:4").expect("a table");
        assert_eq!(
            t.width_for(1),
            7,
            "below the first key it takes the first row"
        );
        assert_eq!(t.width_for(2), 7);
        assert_eq!(t.width_for(3), 7);
        assert_eq!(t.width_for(4), 6);
        assert_eq!(t.width_for(7), 6);
        assert_eq!(t.width_for(8), 5);
        assert_eq!(t.width_for(15), 5);
        assert_eq!(t.width_for(16), 4);
        assert_eq!(t.width_for(1024), 4);
        // Batch 2 and 3 are NOT measured cells in the default and carry batch 1's answer.
        let d = WidthByBatch::prod_like();
        assert_eq!((d.width_for(2), d.width_for(3)), (8, 8));
        assert_eq!(d.to_string(), "1:8,4:5");
    }

    #[test]
    fn a_table_cell_is_rounded_up_to_a_captured_width_and_says_so() {
        // THE CONTROL that keeps this from being a behaviour change on the plans this tree has
        // been measuring with: `--engine-decode-widths 1,8` captures no width 5, so the table's
        // conc-8 cell rounds up to 8 and the boot serves exactly what static 8 served.
        let static8 = VerifyWidthPolicy::from_widths(vec![1, 8]);
        assert_eq!(static8.step_width(8, []), 8);
        assert_eq!(static8.reachable_widths(), vec![8]);
        assert!(!static8.can_narrow(), "one reachable width is inert");
        // And `1,5` is static 5 at every batch, because 8 has nothing to round up to.
        let static5 = VerifyWidthPolicy::from_widths(vec![1, 5]);
        assert_eq!(static5.step_width(1, []), 5);
        assert_eq!(static5.step_width(8, []), 5);
        assert_eq!(static5.reachable_widths(), vec![5]);
        // Both cells captured: the table reaches both and the banner says so.
        let table = VerifyWidthPolicy::from_widths(vec![1, 5, 8]);
        assert_eq!(table.reachable_widths(), vec![5, 8]);
        assert!(table.can_narrow());
    }

    #[test]
    fn the_table_does_not_read_the_emas_and_the_ema_rule_does() {
        // The point of the table: the width does not move mid-stream. A cold sequence and a
        // settled one get the same shape at the same batch, so one arriving request cannot pull
        // the step to another bucket (ws-t1's second coupling, closed here by construction).
        let table = VerifyWidthPolicy::from_widths(vec![1, 3, 5, 8]);
        let ema_rule = table.clone().with_rule(WidthRule::AcceptanceEma);
        let mut cold = table.new_ema();
        for _ in 0..20 {
            cold.observe(&table.prod, 0, 8);
        }
        let fresh = table.new_ema();
        assert_eq!(table.step_width(8, [&cold]), 5);
        assert_eq!(
            table.step_width(8, [&cold, &fresh]),
            5,
            "a fresh EMA changes nothing"
        );
        // The EMA rule, same ladder, same EMAs: the cold one narrows to 3 and the fresh one pulls
        // the whole step back to the ceiling. That difference IS the measurement.
        assert_eq!(ema_rule.step_width(8, [&cold]), 3);
        assert_eq!(ema_rule.step_width(8, [&cold, &fresh]), 8);
    }

    #[test]
    fn under_the_table_a_slot_asks_for_the_tables_width_and_the_rounding_is_counted() {
        let p = VerifyWidthPolicy::from_widths(vec![1, 8]);
        let ema = p.new_ema();
        // The table asks for 5 at batch 8; the plan's narrowest width at or above it is 8.
        let bucket = p.step_width(8, [&ema]);
        assert_eq!(bucket, 8);
        let slot = p.slot_in_bucket(&ema, 8, bucket);
        assert_eq!(slot.verify, 8);
        assert_eq!(slot.own_cap, 5, "what the table asked for");
        assert!(
            slot.absorbed(),
            "the rounding up is a counter, not a silence"
        );
        // With width 5 captured there is nothing to absorb.
        let both = VerifyWidthPolicy::from_widths(vec![1, 5, 8]);
        let bucket = both.step_width(8, [&ema]);
        assert_eq!(bucket, 5);
        let slot = both.slot_in_bucket(&ema, 8, bucket);
        assert_eq!((slot.verify, slot.own_cap), (5, 5));
        assert!(!slot.absorbed());
    }

    #[test]
    fn a_table_that_answers_nothing_or_twice_is_refused() {
        assert_eq!(WidthByBatch::new([]), Err(WidthTableError::Empty));
        assert_eq!(
            WidthByBatch::new([(0, 8)]),
            Err(WidthTableError::ZeroBatch),
            "the table is read at one running sequence or more"
        );
        assert_eq!(
            WidthByBatch::new([(1, 0)]),
            Err(WidthTableError::ZeroWidth { batch: 1 })
        );
        assert_eq!(
            WidthByBatch::new([(4, 5), (4, 8)]),
            Err(WidthTableError::Duplicate {
                batch: 4,
                a: 5,
                b: 8
            })
        );
        assert!(
            WidthByBatch::new([(4, 5), (4, 5)]).is_ok(),
            "the same row twice is one row"
        );
        assert!(matches!(
            WidthByBatch::parse("1-8"),
            Err(WidthTableError::Syntax { .. })
        ));
        assert!(matches!(
            WidthByBatch::parse("1:x"),
            Err(WidthTableError::Syntax { .. })
        ));
    }

    #[test]
    fn the_rule_is_named_on_a_command_line_and_round_trips() {
        assert_eq!(
            WidthRule::parse("batch").expect("batch"),
            WidthRule::default()
        );
        assert_eq!(
            WidthRule::parse("ema").expect("ema"),
            WidthRule::AcceptanceEma
        );
        let custom = WidthRule::parse("batch=1:8,3:5").expect("a table");
        assert_eq!(custom.table().expect("table").width_for(3), 5);
        assert_eq!(custom.to_string(), "batch=1:8,3:5");
        assert_eq!(
            WidthRule::parse(&WidthRule::default().to_string()).expect("round trip"),
            WidthRule::default()
        );
        assert!(WidthRule::parse("dynamic").is_err());
    }

    #[test]
    fn the_knobs_are_data_not_a_ladder() {
        let mut p = policy();
        let mut ema = p.new_ema();
        for _ in 0..20 {
            ema.observe(&p.prod, 2, 5);
        }
        assert_eq!(p.slot_cap(&ema, BATCH, 8), Some(4));
        p.prod.margin = 4;
        assert_eq!(p.slot_cap(&ema, BATCH, 8), Some(6), "one field, no branch");
    }
}
