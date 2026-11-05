//! `saber-plan` — the control plane: manifest, device profile, constraint registry, autotune cache,
//! planner and `explain`.
//!
//! See `docs/ARCHITECTURE.md` § *Architecture — five planes* (1. Control plane, 2. Kernel plane) and
//! `docs/IMPLEMENTATION-PLAN.md` § P0. This crate is pure host Rust: it touches no HIP, no GPU and
//! no code object, and it is the only place behaviour is decided.
//!
//! ```text
//!   Manifest (typed TOML)  ─┐
//!   DeviceProfile          ─┤
//!   ModelSpec              ─┼─►  resolve()  ─►  constraint match per (op, M)  ─►  AutotuneKey
//!   Registry               ─┘                                                        │
//!                                                                                    ▼
//!   AutotuneCache ──────────────────────────────────────────────────►  bind()  ─►  ExecutionPlan
//!                                                                                    │
//!                                                                                    ▼
//!                                                                                 explain()
//! ```
//!
//! The rules this crate exists to enforce, each of which is a measurement on this box rather than a
//! preference:
//!
//! * **One manifest, not 137 environment variables.** Unknown key is an error; missing required key
//!   lists every one that is missing. [`Manifest`].
//! * **Constraints discriminate ops, never tiles.** A constraint is a *necessary* condition on the
//!   geometry; tile choice is an opaque `variant_id` plus a parameter blob from a measurement.
//!   [`Registry`], [`Constraint`].
//! * **No cost model.** Constraint match → autotune lookup → **error if not tuned**.
//!   [`AutotuneCache::require`], [`NotTuned`].
//! * **Discard the warm-up.** The first-measured variant per shape reads ~17% inflated, so a naive
//!   measure-in-order sweep rejects whatever it measured first, silently and forever.
//!   [`Autotuner::record_measurements`].
//! * **A `!graph_safe` op cannot enter a captured bucket** — not by convention, by type:
//!   [`CapturedBucketOp`] can only be built from a [`GraphSafe`] proof.
//! * **A bucket set is bounded by the Infinity Cache.** Speculative GDN state has a hard cliff at
//!   64 MiB (09-14). [`GdnStateBudget`].
//! * **Launch count is declared, not discovered.** [`PlannedBucket::n_launches`].
//!
//! # Example
//!
//! ```
//! use saber_plan::{reference, resolve, DeviceProfile, Manifest, ModelSpec};
//!
//! let manifest = Manifest::from_str(reference::EXAMPLE_MANIFEST_TOML, "example.toml").unwrap();
//! let device = DeviceProfile::gfx1201_reference();
//! let model = ModelSpec::qwen38_27b_dense(manifest.device.tp).unwrap();
//! let registry = reference::reference_registry();
//!
//! let resolved = resolve(&manifest, &device, &model, &registry).unwrap();
//! // Nothing is tuned yet, so every route is a work item rather than a binding.
//! assert_eq!(resolved.missing(&Default::default()).len(), resolved.requirements().len());
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod autotune;
pub mod device;
pub mod error;
pub mod explain;
pub mod from_manifest;
pub mod hash;
pub mod manifest;
pub mod model;
pub mod plan;
pub mod provenance;
pub mod reference;
pub mod registry;
pub mod shape;

pub use autotune::{
    medians_after_warmup, noise_floor_us, AutotuneCache, AutotuneKey, AutotuneRecord, Autotuner,
    CacheEntry, Measurement, PolicySplit, RunnerUp, Unstable, VariantParams,
};
pub use device::{
    Bandwidth, BandwidthSource, DeviceCaps, DeviceProfile, MeasuredBandwidths, WmmaShape,
};
pub use error::{
    IcOverflow, ManifestError, NoImplementation, NotGraphSafe, NotTuned, PlanError, Result,
};
pub use explain::{explain, explain_missing, human_bytes, NOISE_FLOOR_PCT, NOISE_FLOOR_US};
pub use from_manifest::{
    check_plugins_declared, implementations_for, load_artifacts, registry_from_artifacts,
    registry_from_prebuilt, PluginArtifact, RuleSets,
};
pub use manifest::{
    BucketSection, DeviceSection, Manifest, ModelSection, PlanOverride, PlanSection, PluginSection,
};
pub use model::{
    a_layout_key, a_tiled_min_m, ALayout, BucketKey, ModelOp, ModelSpec, Occurrences, Parallelism,
    Rows, AR_PAYLOAD_BYTES, AR_QUANT_MIN_BYTES, GDN_FUSED_MAX_ITEMS,
    GDN_STATE_BYTES_PER_LAYER_PER_SLOT,
};
pub use plan::{
    plan, resolve, BoundOp, Bucket, BucketOps, CapturedBucketOp, ExecutionPlan, GdnStateBudget,
    GraphSafe, Phase, PlanBody, PlannedBucket, PluginBinding, Requirement, Resolved, Stale,
    StaleWhy, WorkspaceAssignment, WorkspaceBudget, PLAN_SCHEMA_VERSION,
};
pub use provenance::{Promotion, PromotionState, Provenance, TieBreak};
pub use registry::{
    CandidateMiss, CandidateTrace, Constraint, ConstraintOp, ImplId, Implementation, LaunchModel,
    Registry, SelectionTrace, WorkspaceKind, WorkspaceReq, WorkspaceSizing,
};
pub use saber_kernels::AProducer;
pub use shape::{ShapeKeys, ShapeValue};
