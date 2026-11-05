//! Errors. Every one of them is a refusal to start, never a fallback.
//!
//! Invariant I3 (*fail loud, never fall back silently*) is the reason this module exists in the
//! shape it does: each error carries the evidence a reader needs to act — the constraint that
//! refused a candidate, the arithmetic that overflowed the Infinity Cache, the exact key that was
//! not tuned.

use crate::autotune::AutotuneKey;
use crate::registry::CandidateMiss;
use crate::shape::ShapeKeys;
use std::fmt;
use std::path::PathBuf;

/// `Result` with this crate's error type.
pub type Result<T> = std::result::Result<T, PlanError>;

/// Nothing in the registry covers an op at a geometry.
///
/// Mirrors the reference's `select()` refusal, which records *which constraint* refused each candidate:
/// *"'no kernel' on its own is not an answer anyone can act on"*.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoImplementation {
    /// The op that was asked for.
    pub op: String,
    /// The geometry it was asked for at.
    pub query: ShapeKeys,
    /// Every candidate carrying that op, closest first, with the constraint that refused it.
    pub misses: Vec<CandidateMiss>,
}

impl fmt::Display for NoImplementation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "no implementation for op `{}` at [{}]",
            self.op, self.query
        )?;
        if self.misses.is_empty() {
            write!(
                f,
                "\n  no implementation in this registry has op `{}`",
                self.op
            )?;
        } else {
            for m in &self.misses {
                write!(
                    f,
                    "\n  {}: {} ({} constraint(s) satisfied)",
                    m.impl_name, m.reason, m.satisfied
                )?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for NoImplementation {}

/// A lookup missed the autotune cache.
///
/// This is the whole of `ARCHITECTURE.md` § *Cuts*, "the Planner's cost model": constraint match →
/// autotune lookup → **error if not tuned**. Loud, not clever.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotTuned {
    /// The key that missed. Boxed: it is the largest payload in the error enum.
    pub key: Box<AutotuneKey>,
    /// The candidates the constraint match returned — what a tuning run has to measure.
    pub candidates: Vec<String>,
}

impl fmt::Display for NotTuned {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "not tuned: op `{}` at [{}], power cap {} W\n  \
             plan input hash {}\n  plugins {}\n  \
             candidates needing a measurement: {}\n  \
             the planner has no cost model by design: constraint match, autotune lookup, error if \
             not tuned",
            self.key.op,
            self.key.shape,
            self.key.power_cap_w,
            self.key.plan_hash,
            self.key.plugin_build_hash,
            if self.candidates.is_empty() {
                "(none)".to_string()
            } else {
                self.candidates.join(", ")
            },
        )
    }
}

impl std::error::Error for NotTuned {}

/// A bucket's speculative GDN state does not fit the Infinity Cache (the 09-14 cliff).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IcOverflow {
    /// The bucket that overflowed.
    pub bucket: String,
    /// Concurrent (per-layer) speculative state bytes the bucket touches.
    pub working_set_bytes: u64,
    /// The Infinity Cache size it must fit in.
    pub ic_bytes: u64,
    /// The arithmetic, spelled out, so the refusal can be checked by hand.
    pub arithmetic: String,
}

impl fmt::Display for IcOverflow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "bucket {} overflows the Infinity Cache: {} > {} ({})\n  \
             GDN decode cost tracks total speculative state bytes with a hard cliff at the \
             Infinity Cache (09-14): 8:6 costs +2.2 ms/step and max_num_seqs=16 +5.5 ms/step\n  \
             {}",
            self.bucket,
            crate::explain::human_bytes(self.working_set_bytes),
            crate::explain::human_bytes(self.ic_bytes),
            self.ic_bytes,
            self.arithmetic,
        )
    }
}

impl std::error::Error for IcOverflow {}

/// A `!graph_safe` (or `may_sync`) implementation was bound into a captured bucket.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotGraphSafe {
    /// The bucket being captured.
    pub bucket: String,
    /// The model op that wanted it.
    pub op_name: String,
    /// The implementation that cannot be captured.
    pub impl_name: String,
    /// Which bit refused it.
    pub reason: String,
}

impl fmt::Display for NotGraphSafe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "bucket {} is captured, but `{}` binds `{}` which is {}\n  \
             mark the bucket eager, or bind a capturable implementation",
            self.bucket, self.op_name, self.impl_name, self.reason
        )
    }
}

impl std::error::Error for NotGraphSafe {}

/// Everything that stops a plan from resolving.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    /// The manifest did not parse or did not validate.
    #[error(transparent)]
    Manifest(#[from] ManifestError),

    /// No implementation covers an op at a geometry.
    #[error(transparent)]
    NoImplementation(#[from] NoImplementation),

    /// A lookup missed the autotune cache.
    #[error(transparent)]
    NotTuned(#[from] NotTuned),

    /// A bucket's speculative GDN state does not fit the Infinity Cache.
    #[error(transparent)]
    InfinityCache(#[from] Box<IcOverflow>),

    /// A `!graph_safe` implementation was bound into a captured bucket.
    #[error(transparent)]
    NotGraphSafe(#[from] Box<NotGraphSafe>),

    /// An autotune key was built without `m`.
    #[error("autotune key for op `{op}` has no `m` in its shape [{shape}]; the tile and split-K optimum flips per shape AND per M, so a record keyed without M is a record for the wrong call")]
    ShapeMissingM {
        /// The op whose key was malformed.
        op: String,
        /// The geometry that was offered.
        shape: ShapeKeys,
    },

    /// The manifest and the probed device disagree.
    #[error(
        "device mismatch: the manifest targets `{manifest}` but the device profile is `{profile}`"
    )]
    DeviceMismatch {
        /// `[device] target_gfx` from the manifest.
        manifest: String,
        /// `gfx_target` from the probe.
        profile: String,
    },

    /// An implementation declares something the device cannot do.
    #[error("implementation `{impl_name}` is not runnable on {gfx}: {reason}")]
    ImplUnsupported {
        /// The implementation.
        impl_name: String,
        /// The device target.
        gfx: String,
        /// What exactly it asked for.
        reason: String,
    },

    /// An implementation names a plugin the manifest does not declare.
    #[error("implementation `{impl_name}` comes from plugin `{plugin}`, which the manifest does not declare in [[plugins]]")]
    UnknownPlugin {
        /// The implementation.
        impl_name: String,
        /// The plugin it claims.
        plugin: String,
    },

    /// A `[plan.overrides]` entry could not be honoured.
    #[error("override for `{op_name}` rejected: {reason}")]
    OverrideRejected {
        /// The model op the override named.
        op_name: String,
        /// Why it could not be honoured.
        reason: String,
    },

    /// A tuned record names an implementation the constraint match did not return.
    #[error("tuned record for op `{op}` names implementation `{record_impl}`, which is not among the candidates that matched [{candidates}] — the record is stale, or the registry changed under it")]
    StaleRecord {
        /// The op.
        op: String,
        /// The implementation the record names.
        record_impl: String,
        /// The candidates the constraint match returned.
        candidates: String,
    },

    /// A promotion state claims more than its provenance supports.
    #[error("implementation `{impl_name}`: {reason}")]
    Promotion {
        /// The implementation.
        impl_name: String,
        /// What check failed.
        reason: String,
    },

    /// The model spec cannot be built for this configuration.
    #[error("model spec: {0}")]
    Model(String),

    /// The autotune protocol was used incorrectly (no samples left, empty sweep).
    #[error("autotune: {0}")]
    Autotune(String),

    /// The drafter's tier id does not match the target's.
    #[error("tier mismatch: target tier `{target}` vs drafter tier `{drafter}`; the same DFlash2 drafter accepts 3.67 tok/draft on MXFP4 and 2.72 on int4, so a mismatched pair is a ~35% throughput loss with entirely normal-looking output")]
    TierMismatch {
        /// `[model] tier_id`.
        target: String,
        /// `[model] drafter_tier_id`.
        drafter: String,
    },

    /// Reading or writing a file failed.
    #[error("{what} {}: {source}", path.display())]
    Io {
        /// What was being attempted.
        what: &'static str,
        /// The path involved.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },

    /// An autotune cache file did not parse.
    #[error("autotune cache {}: {source}", path.display())]
    CacheJson {
        /// The path involved.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: serde_json::Error,
    },
}

/// Manifest parse and validation failures.
///
/// The validator reports **every** problem it can see in one pass — an unknown key and three
/// missing required keys come back as four lines, not as four edit-compile cycles.
#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    /// The file could not be read.
    #[error("reading manifest {}: {source}", path.display())]
    Io {
        /// The path involved.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },

    /// The TOML did not parse at all.
    #[error("{origin}: TOML syntax error: {source}")]
    Syntax {
        /// Where the text came from.
        origin: String,
        /// The underlying error.
        #[source]
        source: Box<toml::de::Error>,
    },

    /// The TOML parsed but did not validate.
    #[error("{origin}: {} problem(s):\n{}", issues.len(), issues.iter().map(|i| format!("  - {i}")).collect::<Vec<_>>().join("\n"))]
    Invalid {
        /// Where the text came from.
        origin: String,
        /// Every problem found, in document order.
        issues: Vec<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::PlanError;

    #[test]
    fn plan_error_stays_small() {
        // `clippy::result_large_err` fires at 128 bytes; every large payload above is boxed so a
        // `Result<ExecutionPlan>` does not carry a fat error on the happy path.
        assert!(
            std::mem::size_of::<PlanError>() <= 128,
            "PlanError grew to {} bytes; box the new variant's payload",
            std::mem::size_of::<PlanError>()
        );
    }
}
