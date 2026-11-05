//! Provenance and the promotion ladder.
//!
//! `ARCHITECTURE.md` § *The promotion ladder*: capability and certification are different things,
//! and conflating them is how unproven paths reach production. Every implementation, variant and
//! bound route carries a state, and `explain` prints it. Nothing is promoted by being newer.

use serde::{Deserialize, Serialize};
use std::fmt;

/// How a sweep decided between variants that the noise floor cannot tell apart.
///
/// Below the floor the measurement decided nothing, so something else has to — deterministically,
/// or the same sweep run twice binds two different kernels and every artefact keyed on the plan
/// changes with the weather. `None` on a record means the margin cleared the floor and the
/// measurement is what chose.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TieBreak {
    /// The reference launcher's own choice was inside the tied set, so the tie went to it. The
    /// engine then agrees with production wherever the measurement has no opinion, which is the
    /// weakest claim that is still useful.
    ReferenceLauncher,
    /// No tied variant was the reference launcher's, so the lowest `variant_id` won: arbitrary,
    /// but total and stable across runs and machines.
    LowestVariantId,
}

impl TieBreak {
    /// The phrase `explain` prints.
    pub fn as_str(self) -> &'static str {
        match self {
            TieBreak::ReferenceLauncher => "reference launcher",
            TieBreak::LowestVariantId => "lowest variant id",
        }
    }
}

impl fmt::Display for TieBreak {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Who measured a number, when, and with which harness.
///
/// The measurement rules in `CLAUDE.md` make all three mandatory: *"every number carries harness,
/// SPEC, power cap, date"*. The power cap lives in the autotune key rather than here, because it
/// is part of the identity of the record, not a description of it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Provenance {
    /// Who produced the record (a person or a CI job name).
    pub who: String,
    /// When, as an ISO-8601 date (`2026-09-18`). A string, so this crate stays dependency-free.
    pub when: String,
    /// The harness that produced the number. Never compare across harnesses.
    pub harness: String,
    /// What broke the tie, when the winner's margin was inside the noise floor. `None` means the
    /// measurement decided on its own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tie_broken_by: Option<TieBreak>,
    /// Why this record carries **no measurement**, when it carries none.
    ///
    /// A route that cannot be timed by the harness that tuned the plan — the two-rank all-reduce
    /// in a single-process sweep is the case this exists for — still has to be bound, because the
    /// planner has no cost model and a missing key is a hard refusal. It is bound to the only
    /// variant the rules permit, with `measured_us` **NaN** and this field set. What it must never
    /// carry is a number: a fabricated microsecond would be indistinguishable from a measured one
    /// the day someone compares two plans.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub not_measured: Option<String>,
}

impl Provenance {
    /// Builds a provenance record for a margin that decided itself.
    pub fn new(who: &str, when: &str, harness: &str) -> Self {
        Self {
            who: who.to_string(),
            when: when.to_string(),
            harness: harness.to_string(),
            tie_broken_by: None,
            not_measured: None,
        }
    }

    /// The same record, marked as carrying no measurement and saying why.
    #[must_use]
    pub fn not_measured(mut self, why: &str) -> Self {
        self.not_measured = Some(why.to_string());
        self
    }

    /// The same record, noting what resolved a below-floor tie.
    #[must_use]
    pub fn with_tie_break(mut self, tie: Option<TieBreak>) -> Self {
        self.tie_broken_by = tie;
        self
    }
}

impl fmt::Display for Provenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "who={} when={} harness={}",
            self.who, self.when, self.harness
        )?;
        if let Some(tie) = self.tie_broken_by {
            write!(f, " tie_broken_by={tie}")?;
        }
        if let Some(why) = &self.not_measured {
            write!(f, " NOT MEASURED: {why}")?;
        }
        Ok(())
    }
}

/// The four rungs of the promotion ladder, in increasing order of authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromotionState {
    /// Exists, loads, produces output.
    Implemented,
    /// Has a provenance record: who, when, at what power cap, against which harness.
    Measured,
    /// Wins a paired A/B and passes the gates.
    Default,
    /// Bit-exact or end-task-equivalent against the reference path, recorded.
    Certified,
}

impl PromotionState {
    /// The lower-case name `explain` prints.
    pub fn as_str(self) -> &'static str {
        match self {
            PromotionState::Implemented => "implemented",
            PromotionState::Measured => "measured",
            PromotionState::Default => "default",
            PromotionState::Certified => "certified",
        }
    }
}

impl fmt::Display for PromotionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A promotion state together with the provenance that justifies it.
///
/// A state above [`PromotionState::Implemented`] without provenance is a claim with nothing behind
/// it, so [`Promotion::check`] rejects it — that is invariant I3 applied to the ladder itself.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Promotion {
    /// The rung this route currently stands on.
    pub state: PromotionState,
    /// The record behind the rung. `None` is only legal for `implemented`.
    pub provenance: Option<Provenance>,
}

impl Promotion {
    /// `implemented`: exists, loads, produces output. No measurement claimed.
    pub fn implemented() -> Self {
        Self {
            state: PromotionState::Implemented,
            provenance: None,
        }
    }

    /// Any rung above `implemented`, with the record that justifies it.
    pub fn with(state: PromotionState, provenance: Provenance) -> Self {
        Self {
            state,
            provenance: Some(provenance),
        }
    }

    /// `Err` with a human-readable reason when the state claims more than the provenance supports.
    pub fn check(&self) -> Result<(), String> {
        if self.state != PromotionState::Implemented && self.provenance.is_none() {
            return Err(format!(
                "promotion state `{}` carries no provenance record",
                self.state
            ));
        }
        Ok(())
    }

    /// The higher of two rungs, keeping the provenance of the winner.
    pub fn max(self, other: Promotion) -> Promotion {
        if other.state > self.state {
            other
        } else {
            self
        }
    }
}

impl fmt::Display for Promotion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.provenance {
            Some(p) => write!(f, "{} ({p})", self.state),
            None => write!(f, "{} (no provenance)", self.state),
        }
    }
}
