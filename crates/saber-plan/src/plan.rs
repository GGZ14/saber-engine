//! The planner: `Manifest × DeviceProfile × ModelSpec × Registry × AutotuneCache → ExecutionPlan`.
//!
//! Resolution is two passes, and the split is not cosmetic:
//!
//! 1. [`resolve`] — constraint match per (op, bucket-M), which yields the **candidates** and the
//!    **autotune keys** the plan needs. This pass is what a tuning campaign enumerates
//!    ([`Resolved::missing`]), and it is what the plan *input* hash covers.
//! 2. [`Resolved::bind`] — one autotune lookup per key. A miss is [`crate::NotTuned`]; there is no
//!    cost model and no default variant to fall back to.
//!
//! Two refusals are structural rather than advisory:
//!
//! * A `!graph_safe` (or `may_sync`) implementation **cannot** be placed in a captured bucket: a
//!   [`CapturedBucketOp`] can only be built from a [`GraphSafe`] proof, and the only way to obtain
//!   one is [`GraphSafe::prove`] against an implementation that declares it.
//! * A bucket whose speculative GDN state overflows the Infinity Cache is rejected with the
//!   arithmetic printed (the 09-14 cliff).

use crate::autotune::{AutotuneCache, AutotuneKey, VariantParams};
use crate::device::DeviceProfile;
use crate::error::{IcOverflow, NotGraphSafe, PlanError, Result};
use crate::manifest::Manifest;
use crate::model::{ModelOp, ModelSpec};
use crate::provenance::{Promotion, PromotionState, TieBreak};
use crate::registry::{ImplId, Implementation, Registry, SelectionTrace, WorkspaceKind};
use crate::shape::ShapeKeys;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Which half of the step a bucket is captured for.
///
/// The two phases are not the same shape with a different `M`: they run **different ops**. A
/// prefill bucket takes the query-tiled attention kernel and the three-kernel GDN chunk path; a
/// decode bucket takes the split-KV attention pair and one of the two GDN decode routes. They
/// also budget GDN state differently ([`Bucket::gdn_state_slots`]), which is the thing that
/// makes a shared `M` actively misleading.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// One step per token per sequence: `M = batch × spec_width`, one GDN state slot per draft
    /// token. The default, because every bucket in the plan was a decode bucket until P2.
    #[default]
    Decode,
    /// One chunk of a prompt: `M = tokens`, batch 1, no speculation, **one** GDN state slot.
    Prefill,
}

impl Phase {
    /// The name `explain` and the manifest use.
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Decode => "decode",
            Phase::Prefill => "prefill",
        }
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One graph-capture bucket.
///
/// Decode: batch × speculative width × context band. Prefill: a chunk of `tokens` rows at batch
/// 1 with no speculative width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Bucket {
    /// Which half of the step.
    pub phase: Phase,
    /// Sequences in flight. Always 1 for a prefill bucket: chunked prefill runs one sequence's
    /// chunk per step, and a second sequence would be a second bucket.
    pub batch: u32,
    /// Draft tokens verified per step. One GDN state slot per draft token. Zero at prefill. A
    /// decode ROW BAND's is its MAX width (C2).
    pub spec_width: u32,
    /// Prompt tokens in a prefill chunk. Zero at a rectangular decode bucket, where `M` comes from
    /// batch × width; **a decode ROW BAND's row count** (C2, [`Bucket::band`]).
    pub tokens: u32,
    /// Context band.
    pub ctx_band: u32,
    /// Whether this bucket is graph-captured.
    pub captured: bool,
}

impl Bucket {
    /// A decode bucket. A constructor rather than a literal so that a later field is an additive
    /// change here instead of a compile error in every crate that builds one.
    pub fn decode(batch: u32, spec_width: u32, ctx_band: u32, captured: bool) -> Self {
        Self {
            phase: Phase::Decode,
            batch,
            spec_width,
            tokens: 0,
            ctx_band,
            captured,
        }
    }

    /// **A decode ROW BAND (workstream C2)**: `batch` slots at their OWN verify widths (each
    /// `1..=max_width`, 0 for padding), packed into `rows` rows — the smallest captured band at or
    /// above the step's `Σ w_i` runs it. `m()` is `rows`; the GDN state arena
    /// ([`Bucket::gdn_state_slots`]) is still `batch × max_width`, because a slot's snapshot run
    /// is reserved at the ceiling whatever it verifies this step. `rows` is clipped to
    /// `batch × max_width` by the caller that lists the bands.
    pub fn band(batch: u32, max_width: u32, rows: u32, ctx_band: u32, captured: bool) -> Self {
        Self {
            phase: Phase::Decode,
            batch,
            spec_width: max_width,
            tokens: rows,
            ctx_band,
            captured,
        }
    }

    /// Whether this is a decode row band (C2) rather than a `batch × width` rectangle.
    pub fn is_band(&self) -> bool {
        self.phase == Phase::Decode && self.tokens > 0
    }

    /// A prefill bucket: `tokens` rows at batch 1, no speculative width.
    pub fn prefill(tokens: u32, ctx_band: u32, captured: bool) -> Self {
        Self {
            phase: Phase::Prefill,
            batch: 1,
            spec_width: 0,
            tokens,
            ctx_band,
            captured,
        }
    }

    /// Rows the GEMMs run at: `batch * spec_width` at decode, `tokens` at prefill.
    pub fn m(&self) -> u32 {
        match self.phase {
            Phase::Decode if self.tokens > 0 => self.tokens,
            Phase::Decode => self.batch * self.spec_width,
            Phase::Prefill => self.tokens,
        }
    }

    /// GDN state slots one layer's kernel touches in this bucket.
    ///
    /// **Per sequence at prefill, per draft token at decode.** The 09-14 Infinity-Cache finding
    /// is about *speculative* state: a decode step holds one recurrent state per draft token,
    /// because every candidate continues from its own state and the snapshots are what the
    /// kernel spends 55-88% of its time writing. A prefill chunk has no candidates. It carries
    /// **one** state per sequence and scans the chunk through it, so a 8192-token chunk touches
    /// 1 slot, not 8192 — budgeting it by `M` would ask for 24 GB of state and refuse a bucket
    /// that in fact fits the cache with room to spare.
    pub fn gdn_state_slots(&self) -> u32 {
        match self.phase {
            Phase::Decode => self.batch * self.spec_width,
            Phase::Prefill => self.batch,
        }
    }

    /// Short stable label used in errors and in `explain`.
    pub fn label(&self) -> String {
        match self.phase {
            Phase::Decode if self.tokens > 0 => {
                format!("b{}r{}c{}", self.batch, self.tokens, self.ctx_band)
            }
            Phase::Decode => format!("b{}s{}c{}", self.batch, self.spec_width, self.ctx_band),
            Phase::Prefill => format!("p{}c{}", self.tokens, self.ctx_band),
        }
    }
}

impl fmt::Display for Bucket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let captured = if self.captured { "captured" } else { "eager" };
        match self.phase {
            Phase::Decode if self.tokens > 0 => write!(
                f,
                "decode band batch {} rows {} (max spec {}) ctx_band {} ({captured})",
                self.batch, self.tokens, self.spec_width, self.ctx_band
            ),
            Phase::Decode => write!(
                f,
                "decode batch {} spec {} ctx_band {} ({captured})",
                self.batch, self.spec_width, self.ctx_band
            ),
            Phase::Prefill => write!(
                f,
                "prefill tokens {} ctx_band {} ({captured})",
                self.tokens, self.ctx_band
            ),
        }
    }
}

/// A bucket's speculative GDN state, against the Infinity Cache and against VRAM.
///
/// The 09-14 finding is specific: **GDN decode cost tracks the speculative state bytes the kernel
/// touches, with a hard cliff at the 64 MB Infinity Cache.** One GDN kernel runs one layer at a
/// time over every `batch × spec_width` slot, so the bytes that meet the cache are
/// `batch × spec_width × 3.0 MiB / tp` — at batch 8 × width 5, TP=2, that is 60 MiB = 62.9 MB,
/// the "98% of the IC" in the ledger (98% against the 64 **MB** the ledger names, 93.8% against
/// the 64 **MiB** [`crate::device::GFX1201_INFINITY_CACHE_BYTES`] states; both are the same
/// edge). Batch 8 × width 6 is 72 MiB, over it, and cost prod +2.2 ms/step; `max_num_seqs=16`
/// (120 MiB) cost +5.5 ms/step. Batch 8 × width 8 is 96 MiB — half as much again as the cache.
///
/// The ledger writes those prod cells as `8:5` / `8:6`; they are `max_num_seqs × verify width`,
/// not `SABER_DRAFT_SCHEDULE` entries (ws-t1: that knob is inert under dflash).
///
/// [`GdnStateBudget::resident_bytes`] is the same quantity multiplied by the 48 GDN layers — what
/// the VRAM ledger in `saber-mem` has to hold — and is reported, not gated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GdnStateBudget {
    /// `batch * spec_width`.
    pub slots: u32,
    /// State bytes per slot per GDN layer, after the TP divide.
    pub bytes_per_slot_per_layer: u64,
    /// What one GDN layer's kernel touches: `slots * bytes_per_slot_per_layer`. Gated against the
    /// Infinity Cache.
    pub working_set_bytes: u64,
    /// All GDN layers resident: `working_set_bytes * gdn_layers`. Reported for the byte ledger.
    pub resident_bytes: u64,
    /// The Infinity Cache this was gated against.
    pub ic_bytes: u64,
}

impl GdnStateBudget {
    /// Fraction of the Infinity Cache the working set occupies.
    pub fn ic_fraction(&self) -> f64 {
        if self.ic_bytes == 0 {
            0.0
        } else {
            self.working_set_bytes as f64 / self.ic_bytes as f64
        }
    }
}

fn gdn_budget(
    model: &ModelSpec,
    bucket: &Bucket,
    device: &DeviceProfile,
) -> Result<GdnStateBudget> {
    let slots = bucket.gdn_state_slots();
    let per_slot = model
        .gdn_state_bytes_per_layer_per_slot
        .div_ceil(u64::from(model.tp));
    let working_set = u64::from(slots) * per_slot;
    let budget = GdnStateBudget {
        slots,
        bytes_per_slot_per_layer: per_slot,
        working_set_bytes: working_set,
        resident_bytes: working_set * u64::from(model.gdn_layers),
        ic_bytes: device.infinity_cache_bytes,
    };
    if working_set > device.infinity_cache_bytes {
        return Err(PlanError::InfinityCache(Box::new(IcOverflow {
            bucket: bucket.label(),
            working_set_bytes: working_set,
            ic_bytes: device.infinity_cache_bytes,
            arithmetic: format!(
                "{} x {} per GDN layer / tp {} = {} touched per layer; \
                 {} GDN layers resident = {}",
                match bucket.phase {
                    // One state per draft token: the 09-14 cliff is about speculative state.
                    Phase::Decode =>
                        format!("batch {} x spec_width {}", bucket.batch, bucket.spec_width),
                    // One state per sequence, scanned over the chunk.
                    Phase::Prefill => format!("{} sequence(s)", bucket.batch),
                },
                crate::explain::human_bytes(model.gdn_state_bytes_per_layer_per_slot),
                model.tp,
                crate::explain::human_bytes(working_set),
                model.gdn_layers,
                crate::explain::human_bytes(budget.resident_bytes),
            ),
        })));
    }
    Ok(budget)
}

/// A zero-sized proof that an implementation declares itself legal under graph capture.
///
/// It has no public constructor: the only way to obtain one is [`GraphSafe::prove`], and the only
/// thing that consumes one is [`CapturedBucketOp::new`]. That is `ARCHITECTURE.md` design rule 3,
/// *make illegal states unrepresentable*, applied to the one bug it exists for — an op that is
/// legal eager and illegal under capture.
///
/// A proof cannot be forged:
///
/// ```compile_fail
/// // error[E0603]: tuple struct constructor `GraphSafe` is private
/// let forged = saber_plan::GraphSafe(());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphSafe(());

impl GraphSafe {
    /// A proof, when the implementation declares `graph_safe` and does not declare `may_sync`.
    ///
    /// ```
    /// use saber_plan::{GraphSafe, Implementation};
    /// let eager_only = Implementation::new("ar_oneshot_2rank_exact", "saber", "all_reduce")
    ///     .capture(false, true);
    /// assert!(GraphSafe::prove(&eager_only).is_none());
    /// ```
    pub fn prove(imp: &Implementation) -> Option<GraphSafe> {
        if imp.capturable() {
            Some(GraphSafe(()))
        } else {
            None
        }
    }
}

/// A workspace an op needs, sized at this shape and variant.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct WorkspaceAssignment {
    /// What it is.
    pub name: String,
    /// Aliasable, persistent or peer-visible.
    pub kind: WorkspaceKind,
    /// Bytes at this shape and variant — what is **allocated**.
    pub bytes: u64,
    /// What this bucket actually **uses**, when the size law has a per-bucket form. Smaller than
    /// `bytes` for a buffer sized once for the worst shape the engine can schedule.
    pub used_bytes: Option<u64>,
}

/// A bucket-level workspace budget, aggregated by kind.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct WorkspaceBudget {
    /// The kind being budgeted.
    pub kind: WorkspaceKind,
    /// Bytes for the whole bucket under that kind's aggregation policy.
    pub bytes: u64,
    /// How the per-op requests were combined, printed by `explain`.
    pub policy: String,
    /// How many ops contributed.
    pub contributors: usize,
}

/// One resolved and bound op: everything the execution plane needs, and everything `explain` prints.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BoundOp {
    /// Model op name, e.g. `layer.mlp.down_proj`.
    pub op_name: String,
    /// Registry op selected on, e.g. `gemm_w4a8`.
    pub op: String,
    /// The geometry, including `m`.
    pub shape: ShapeKeys,
    /// The implementation bound.
    pub impl_name: String,
    /// The plugin that ships it.
    pub plugin: String,
    /// The tuned variant and its opaque parameters.
    pub variant: VariantParams,
    /// The measurement that chose it, microseconds.
    pub measured_us: f64,
    /// Margin over the runner-up in percent, when the sweep had one. Can be negative when the
    /// margin was inside the noise floor and [`BoundOp::tie_broken_by`] chose instead.
    pub margin_pct: Option<f64>,
    /// Why this route carries no measurement, when it carries none. `measured_us` is NaN then.
    pub not_measured: Option<String>,
    /// What resolved a below-floor tie, straight from the autotune record.
    ///
    /// Carried separately from [`BoundOp::promotion`] on purpose: a route whose implementation
    /// already ships at `default` or `certified` keeps the *implementation's* provenance, so
    /// reading the tie-break off the promotion would lose it precisely for the rows that are
    /// furthest up the ladder.
    pub tie_broken_by: Option<TieBreak>,
    /// Set when the two sweep passes named different winners: the binding is the rules' answer,
    /// not the measurement's, and `explain` says so where the number would otherwise be read as
    /// one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unstable: Option<crate::Unstable>,
    /// Where this route stands on the promotion ladder.
    pub promotion: Promotion,
    /// Declared legal under capture.
    pub graph_safe: bool,
    /// Declared to possibly synchronise.
    pub may_sync: bool,
    /// Wave size the code object was compiled with.
    pub wave_size: u32,
    /// How many times this op runs per step.
    pub occurrences: u32,
    /// Launches for one call.
    pub launches_each: u32,
    /// `launches_each * occurrences`.
    pub launches_total: u32,
    /// Workspaces, sized.
    pub workspaces: Vec<WorkspaceAssignment>,
    /// Why this route: every candidate and the constraint that kept or refused it.
    pub selection: SelectionTrace,
}

/// A [`BoundOp`] that has been proved legal for a captured bucket.
///
/// Only constructible from a [`GraphSafe`] proof:
///
/// ```compile_fail
/// # use saber_plan::{BoundOp, CapturedBucketOp};
/// fn capture(op: BoundOp) -> CapturedBucketOp {
///     // error[E0061]: this function takes 2 arguments but 1 argument was supplied
///     CapturedBucketOp::new(op)
/// }
/// ```
///
/// It is deliberately `Serialize` but not `Deserialize`: a plan is re-resolved, never loaded, so
/// there is no deserializer through which an unproved op could enter a captured bucket.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct CapturedBucketOp {
    op: BoundOp,
}

impl CapturedBucketOp {
    /// Places a bound op into a captured bucket, consuming the proof that it may be captured.
    pub fn new(op: BoundOp, _proof: GraphSafe) -> Self {
        Self { op }
    }

    /// The op.
    pub fn op(&self) -> &BoundOp {
        &self.op
    }

    /// Takes the op back out.
    pub fn into_inner(self) -> BoundOp {
        self.op
    }
}

/// A bucket's ops, in a form that records whether they were proved capturable.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BucketOps {
    /// Captured: every op carries a [`GraphSafe`] proof.
    Captured(Vec<CapturedBucketOp>),
    /// Eager: no proof required, and `explain` prints what forced it.
    Eager(Vec<BoundOp>),
}

impl BucketOps {
    /// The ops, however they are held.
    pub fn ops(&self) -> Vec<&BoundOp> {
        match self {
            BucketOps::Captured(ops) => ops.iter().map(CapturedBucketOp::op).collect(),
            BucketOps::Eager(ops) => ops.iter().collect(),
        }
    }

    /// How many op rows.
    pub fn len(&self) -> usize {
        match self {
            BucketOps::Captured(ops) => ops.len(),
            BucketOps::Eager(ops) => ops.len(),
        }
    }

    /// Whether the bucket has no ops.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One planned bucket: its ops, its static launch count, its state and workspace budgets.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlannedBucket {
    /// The bucket.
    pub bucket: Bucket,
    /// Rows the GEMMs run at.
    pub m: u32,
    /// The ops.
    pub ops: BucketOps,
    /// **Static launch count for the whole step**, asserted at capture (`CLAUDE.md` rule 7).
    pub n_launches: u32,
    /// Speculative GDN state against the Infinity Cache.
    pub gdn_state: GdnStateBudget,
    /// Workspace bytes by kind.
    pub workspaces: Vec<WorkspaceBudget>,
}

/// A plugin as bound by the plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PluginBinding {
    /// Plugin name.
    pub name: String,
    /// Artifact path.
    pub path: String,
    /// Build hash.
    pub build_hash: String,
}

/// The whole resolved plan: the bindings the [`ExecutionPlan::plan_hash`] covers, **and** the
/// measurements behind them, which it deliberately does not.
///
/// See [`ExecutionPlan::plan_hash`] for why the split exists.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlanBody {
    /// Bumped whenever the plan's shape changes.
    pub schema_version: u32,
    /// The tier being served: one content hash over checkpoint, drafter, config and template.
    pub tier_id: String,
    /// The drafter's tier id, when one is bound.
    pub drafter_tier_id: Option<String>,
    /// Model spec name.
    pub model_name: String,
    /// Tensor-parallel ranks.
    pub tp: u32,
    /// Device target.
    pub gfx_target: String,
    /// The enforced power cap every bound measurement is valid at.
    pub power_cap_w: u32,
    /// The hash of everything that went *into* resolution — the autotune key component.
    pub plan_input_hash: String,
    /// Canonical digest of the manifest.
    pub manifest_digest: String,
    /// Canonical digest of the device profile.
    pub device_digest: String,
    /// Canonical digest of the model spec.
    pub model_digest: String,
    /// Canonical digest of the registry.
    pub registry_fingerprint: String,
    /// Canonical digest of the launch-rule tables the registry was derived from, and the fifth
    /// component of `plan_input_hash` (see [`crate::Registry::policy_digest`]).
    #[serde(default)]
    pub policy_digest: String,
    /// The plugins bound.
    pub plugins: Vec<PluginBinding>,
    /// The buckets.
    pub buckets: Vec<PlannedBucket>,
}

/// The materialized plan: the single source of behaviour for the whole engine.
///
/// There is no `getenv` anywhere below this type. Behaviour comes from here or it does not exist.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExecutionPlan {
    /// Bindings and measurements.
    pub body: PlanBody,
    /// `sha256` over the **bindings** — see [`ExecutionPlan::plan_hash`].
    pub plan_hash: String,
    /// `sha256` over the **measurements** — see [`ExecutionPlan::measurement_hash`].
    pub measurement_hash: String,
}

impl ExecutionPlan {
    /// The plan hash: the cache key for everything compiled.
    ///
    /// # It covers the decisions, never the microseconds
    ///
    /// The hash is taken over [`bindings_view`]: the tier, the device, the four input digests, the
    /// plugin set, the bucket set, and per op its name, geometry, implementation, variant
    /// parameters, workspace assignments, capture bits and launch counts. It is **not** taken over
    /// `measured_us`, `margin_pct` or the provenance record.
    ///
    /// That is not a cosmetic choice. Autotune microseconds are measured beside a production
    /// server and move by tenths of a microsecond between runs; three identical sweeps on this box
    /// produced identical bindings and three different hashes when the measurement was inside the
    /// hash (`LEDGER.md`, *Autotune sweep*). Anything keyed on the plan hash — a compiled graph, a
    /// tier pack, `saber_mem::pack`'s `check_plan_hash`, which **refuses a pack outright** on a
    /// mismatch — would then be invalidated by re-running a benchmark that changed nothing.
    ///
    /// The numbers are still in the plan and still printed by `explain`; they are covered by
    /// [`ExecutionPlan::measurement_hash`], which is what to compare when the question is "was
    /// this re-measured?" rather than "is this the same plan?".
    pub fn plan_hash(&self) -> &str {
        &self.plan_hash
    }

    /// `sha256` over what was measured: per bound op the implementation, variant id, median
    /// microseconds, margin and promotion record.
    ///
    /// Two plans with the same [`ExecutionPlan::plan_hash`] and different measurement hashes are
    /// the same plan, re-measured. The reverse cannot happen: a different binding always changes
    /// both.
    pub fn measurement_hash(&self) -> &str {
        &self.measurement_hash
    }

    /// Recomputes both hashes and checks them. Cheap, and a good assertion at load.
    pub fn verify_hash(&self) -> bool {
        crate::hash::digest_of(&bindings_view(&self.body)) == self.plan_hash
            && crate::hash::digest_of(&measurement_view(&self.body)) == self.measurement_hash
    }

    /// The buckets.
    pub fn buckets(&self) -> &[PlannedBucket] {
        &self.body.buckets
    }

    /// Launches per step summed over every bucket — the CI budget metric.
    pub fn total_launches(&self) -> u32 {
        self.body.buckets.iter().map(|b| b.n_launches).sum()
    }
}

// ------------------------------------------------------------------ what each hash is taken over

/// One bound op, with everything a measurement can move removed.
#[derive(Debug, Serialize)]
struct BindingOp<'a> {
    op_name: &'a str,
    op: &'a str,
    shape: &'a ShapeKeys,
    impl_name: &'a str,
    plugin: &'a str,
    variant: &'a VariantParams,
    graph_safe: bool,
    may_sync: bool,
    wave_size: u32,
    occurrences: u32,
    launches_each: u32,
    launches_total: u32,
    workspaces: &'a [WorkspaceAssignment],
}

/// One bucket's bindings.
#[derive(Debug, Serialize)]
struct BindingBucket<'a> {
    bucket: &'a Bucket,
    m: u32,
    /// Whether the ops carry a `GraphSafe` proof. Part of the plan: a captured bucket and an eager
    /// one with the same ops are different plans.
    captured: bool,
    n_launches: u32,
    gdn_state: &'a GdnStateBudget,
    workspaces: &'a [WorkspaceBudget],
    ops: Vec<BindingOp<'a>>,
}

/// Everything [`ExecutionPlan::plan_hash`] is taken over.
#[derive(Debug, Serialize)]
struct BindingsView<'a> {
    schema_version: u32,
    tier_id: &'a str,
    drafter_tier_id: Option<&'a str>,
    model_name: &'a str,
    tp: u32,
    gfx_target: &'a str,
    power_cap_w: u32,
    plan_input_hash: &'a str,
    manifest_digest: &'a str,
    device_digest: &'a str,
    model_digest: &'a str,
    registry_fingerprint: &'a str,
    plugins: &'a [PluginBinding],
    buckets: Vec<BindingBucket<'a>>,
}

/// One bound op's measurement, and nothing else.
#[derive(Debug, Serialize)]
struct MeasurementOp<'a> {
    bucket: String,
    op_name: &'a str,
    impl_name: &'a str,
    variant_id: u32,
    measured_us: f64,
    margin_pct: Option<f64>,
    not_measured: Option<&'a str>,
    tie_broken_by: Option<TieBreak>,
    promotion: &'a Promotion,
}

/// The projection [`ExecutionPlan::plan_hash`] is taken over: the decisions.
///
/// The `SelectionTrace` is deliberately absent. It is a derivation, not a decision — prose
/// explaining which candidates the constraint match kept — and the thing it derives from
/// (`registry_fingerprint`) is hashed directly, so including it would only make the plan hash
/// depend on the wording of a message.
fn bindings_view(body: &PlanBody) -> BindingsView<'_> {
    BindingsView {
        schema_version: body.schema_version,
        tier_id: &body.tier_id,
        drafter_tier_id: body.drafter_tier_id.as_deref(),
        model_name: &body.model_name,
        tp: body.tp,
        gfx_target: &body.gfx_target,
        power_cap_w: body.power_cap_w,
        plan_input_hash: &body.plan_input_hash,
        manifest_digest: &body.manifest_digest,
        device_digest: &body.device_digest,
        model_digest: &body.model_digest,
        registry_fingerprint: &body.registry_fingerprint,
        plugins: &body.plugins,
        buckets: body
            .buckets
            .iter()
            .map(|b| BindingBucket {
                bucket: &b.bucket,
                m: b.m,
                captured: matches!(b.ops, BucketOps::Captured(_)),
                n_launches: b.n_launches,
                gdn_state: &b.gdn_state,
                workspaces: &b.workspaces,
                ops: b
                    .ops
                    .ops()
                    .into_iter()
                    .map(|o| BindingOp {
                        op_name: &o.op_name,
                        op: &o.op,
                        shape: &o.shape,
                        impl_name: &o.impl_name,
                        plugin: &o.plugin,
                        variant: &o.variant,
                        graph_safe: o.graph_safe,
                        may_sync: o.may_sync,
                        wave_size: o.wave_size,
                        occurrences: o.occurrences,
                        launches_each: o.launches_each,
                        launches_total: o.launches_total,
                        workspaces: &o.workspaces,
                    })
                    .collect(),
            })
            .collect(),
    }
}

/// The projection [`ExecutionPlan::measurement_hash`] is taken over: the numbers.
fn measurement_view(body: &PlanBody) -> Vec<MeasurementOp<'_>> {
    body.buckets
        .iter()
        .flat_map(|b| {
            let label = b.bucket.label();
            b.ops.ops().into_iter().map(move |o| MeasurementOp {
                bucket: label.clone(),
                op_name: &o.op_name,
                impl_name: &o.impl_name,
                variant_id: o.variant.variant_id,
                measured_us: o.measured_us,
                margin_pct: o.margin_pct,
                not_measured: o.not_measured.as_deref(),
                tie_broken_by: o.tie_broken_by,
                promotion: &o.promotion,
            })
        })
        .collect()
}

/// One (bucket, op) pair that needs a measurement, as enumerated by [`Resolved::requirements`].
#[derive(Clone, Debug, PartialEq)]
pub struct Requirement {
    /// The bucket it belongs to.
    pub bucket: Bucket,
    /// The model op name.
    pub op_name: String,
    /// The key a measurement must be recorded under.
    pub key: AutotuneKey,
    /// The candidates a sweep has to measure.
    pub candidates: Vec<String>,
}

/// A record an earlier sweep left behind that this resolution does not ask for.
///
/// Produced by [`Resolved::carry_forward`], printed by `saber tune`. It is data about the merge,
/// not an error: dropping it is what makes the incremental sweep land where a full one would.
#[derive(Clone, Debug, PartialEq)]
pub struct Stale {
    /// The key the dropped record was stored under.
    pub key: AutotuneKey,
    /// Why this resolution does not ask for it.
    pub why: StaleWhy,
}

/// Why a carried record is not carried.
#[derive(Clone, Debug, PartialEq)]
pub enum StaleWhy {
    /// The same op at the same shape IS required, under a different key — a policy edit, a
    /// registry move, a plugin rebuild or a cap change. The record measured a condition that is
    /// no longer the one the plan binds, and leaving it in the file is what gives an op two
    /// policy digests.
    Superseded(Box<AutotuneKey>),
    /// No route in this plan asks for this op at this shape at all: the cache was written for a
    /// different manifest, or the route has gone. A full sweep would not have written it.
    NotRequired,
}

impl fmt::Display for Stale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.why {
            StaleWhy::Superseded(now) => write!(
                f,
                "{} [{}] superseded: policy {} -> {}, plan {} -> {}",
                self.key.op,
                self.key.shape,
                crate::autotune::short_digest(&self.key.policy_digest),
                crate::autotune::short_digest(&now.policy_digest),
                crate::autotune::short_digest(&self.key.plan_hash),
                crate::autotune::short_digest(&now.plan_hash),
            ),
            StaleWhy::NotRequired => write!(
                f,
                "{} [{}] is not a route in this plan",
                self.key.op, self.key.shape
            ),
        }
    }
}

/// One matched op, before the autotune key exists. The key carries the input hash, and the input
/// hash covers the candidate sets, so the match has to come first.
#[derive(Clone, Debug)]
struct PreOp {
    model_op: usize,
    shape: ShapeKeys,
    candidates: Vec<ImplId>,
    trace: SelectionTrace,
}

#[derive(Clone, Debug)]
struct ResolvedOp {
    model_op: usize,
    shape: ShapeKeys,
    candidates: Vec<ImplId>,
    trace: SelectionTrace,
    key: AutotuneKey,
}

#[derive(Clone, Debug)]
struct ResolvedBucket {
    bucket: Bucket,
    gdn_state: GdnStateBudget,
    ops: Vec<ResolvedOp>,
}

/// The first pass: constraints matched, keys built, nothing measured yet.
#[derive(Debug)]
pub struct Resolved<'a> {
    manifest: &'a Manifest,
    device: &'a DeviceProfile,
    model: &'a ModelSpec,
    registry: &'a Registry,
    input_hash: String,
    buckets: Vec<ResolvedBucket>,
}

impl<'a> Resolved<'a> {
    /// The plan input hash: manifest × device × model × registry. Every autotune key carries it.
    pub fn input_hash(&self) -> &str {
        &self.input_hash
    }

    /// Every measurement this plan needs, in bucket then model order.
    pub fn requirements(&self) -> Vec<Requirement> {
        self.buckets
            .iter()
            .flat_map(|b| {
                b.ops.iter().map(|op| Requirement {
                    bucket: b.bucket,
                    op_name: self.model.ops[op.model_op].name.clone(),
                    key: op.key.clone(),
                    candidates: op.trace.chosen().iter().map(|s| (*s).to_string()).collect(),
                })
            })
            .collect()
    }

    /// The subset of [`Resolved::requirements`] a cache cannot answer — the tuning campaign's
    /// work list, and what `saber explain` prints instead of a plan when the cache is cold.
    pub fn missing(&self, autotune: &AutotuneCache) -> Vec<Requirement> {
        self.requirements()
            .into_iter()
            .filter(|r| autotune.lookup(&r.key).is_err())
            .collect()
    }

    /// An earlier sweep's cache, carried onto **this** resolution's keys: the incremental sweep's
    /// starting point, and the half of it that was missing.
    ///
    /// `saber tune --autotune-in` exists so that a rules edit costs only the routes that table can
    /// reach (I18). It loaded the old cache, measured [`Resolved::missing`] against it, and wrote
    /// the union — which is right about the routes whose key did not move and wrong about the ones
    /// whose key did: the old record is still in the file, under the OLD key, and the op now
    /// carries two policy digests. On 2026-09-21 that is exactly what the CI guard caught
    /// (`attn_gate_quant carries 2 policy digests`), and the workaround was a full sweep — which
    /// is the cost the incremental path exists to avoid.
    ///
    /// The rule is the one a full sweep obeys without trying: **a cache is the set of records this
    /// resolution asks for, and nothing else.** A full sweep starts from an empty cache, so it can
    /// only ever write required keys; an incremental one starts from a file and must therefore
    /// drop what the file holds and the plan does not. The two then produce the same key set, the
    /// same digest set and the same per-op counts, which is the property
    /// `an_incremental_merge_lands_where_a_full_sweep_would` asserts.
    ///
    /// Nothing is re-measured and nothing is rewritten: a carried record keeps its whole
    /// [`crate::AutotuneRecord`] — its median, its samples, its `unstable` pair and its provenance
    /// (including WHO measured it and when), because a record that survived the edit is the same
    /// measurement it always was.
    ///
    /// Returns the carried cache and, in key order, every record dropped and why — the sweep
    /// prints them, because a silent drop is how a cache loses a measurement nobody meant to lose.
    #[must_use]
    pub fn carry_forward(&self, prior: &AutotuneCache) -> (AutotuneCache, Vec<Stale>) {
        let required = self.requirements();
        let wanted: BTreeSet<String> = required.iter().map(|r| r.key.digest()).collect();
        // (op, shape) is what a route IS; everything else in the key is a condition the record was
        // measured under. A prior record that matches one and not the other has been superseded by
        // an edit, and naming the key that replaced it is what makes the drop readable.
        let mut by_route: BTreeMap<(&str, &ShapeKeys), &AutotuneKey> = BTreeMap::new();
        for r in &required {
            by_route
                .entry((r.key.op.as_str(), &r.key.shape))
                .or_insert(&r.key);
        }

        let mut carried = AutotuneCache::new();
        let mut stale = Vec::new();
        for entry in prior.entries() {
            if wanted.contains(&entry.key.digest()) {
                carried.insert(entry.key.clone(), entry.record.clone());
                continue;
            }
            let why = by_route
                .get(&(entry.key.op.as_str(), &entry.key.shape))
                .map_or(StaleWhy::NotRequired, |k| {
                    StaleWhy::Superseded(Box::new((*k).clone()))
                });
            stale.push(Stale {
                key: entry.key.clone(),
                why,
            });
        }
        (carried, stale)
    }

    /// The second pass: one autotune lookup per key, then the plan.
    pub fn bind(&self, autotune: &AutotuneCache) -> Result<ExecutionPlan> {
        let mut buckets = Vec::with_capacity(self.buckets.len());
        for rb in &self.buckets {
            let mut bound = Vec::with_capacity(rb.ops.len());
            for rop in &rb.ops {
                bound.push(self.bind_op(&rb.bucket, rop, autotune)?);
            }
            let n_launches = bound.iter().map(|o| o.launches_total).sum();
            let workspaces = aggregate_workspaces(&bound);
            let ops = if rb.bucket.captured {
                let mut captured = Vec::with_capacity(bound.len());
                for op in bound {
                    // The proof is re-derived from the registry, not carried from the check above:
                    // `CapturedBucketOp` cannot be built any other way.
                    let (_, imp) = self
                        .registry
                        .by_name(&op.impl_name)
                        .expect("bound implementation came from this registry");
                    let proof = GraphSafe::prove(imp).ok_or_else(|| {
                        PlanError::NotGraphSafe(Box::new(NotGraphSafe {
                            bucket: rb.bucket.label(),
                            op_name: op.op_name.clone(),
                            impl_name: op.impl_name.clone(),
                            reason: imp
                                .capture_refusal()
                                .unwrap_or_else(|| "not capturable".to_string()),
                        }))
                    })?;
                    captured.push(CapturedBucketOp::new(op, proof));
                }
                BucketOps::Captured(captured)
            } else {
                BucketOps::Eager(bound)
            };
            buckets.push(PlannedBucket {
                bucket: rb.bucket,
                m: rb.bucket.m(),
                ops,
                n_launches,
                gdn_state: rb.gdn_state,
                workspaces,
            });
        }

        let body = PlanBody {
            schema_version: PLAN_SCHEMA_VERSION,
            tier_id: self.manifest.model.tier_id.clone(),
            drafter_tier_id: self.manifest.model.drafter_tier_id.clone(),
            model_name: self.model.name.clone(),
            tp: self.model.tp,
            gfx_target: self.device.gfx_target.clone(),
            power_cap_w: self.manifest.device.power_cap_w,
            plan_input_hash: self.input_hash.clone(),
            manifest_digest: self.manifest.digest(),
            device_digest: self.device.digest(),
            model_digest: self.model.digest(),
            registry_fingerprint: self.registry.fingerprint(),
            policy_digest: self.registry.policy_digest().to_owned(),
            plugins: self
                .manifest
                .plugins
                .iter()
                .map(|p| PluginBinding {
                    name: p.name.clone(),
                    path: p.path.display().to_string(),
                    build_hash: p.build_hash.clone(),
                })
                .collect(),
            buckets,
        };
        let plan_hash = crate::hash::digest_of(&bindings_view(&body));
        let measurement_hash = crate::hash::digest_of(&measurement_view(&body));
        Ok(ExecutionPlan {
            body,
            plan_hash,
            measurement_hash,
        })
    }

    fn bind_op(
        &self,
        bucket: &Bucket,
        rop: &ResolvedOp,
        autotune: &AutotuneCache,
    ) -> Result<BoundOp> {
        let model_op: &ModelOp = &self.model.ops[rop.model_op];
        let candidate_names: Vec<&str> = rop
            .candidates
            .iter()
            .filter_map(|id| self.registry.get(*id))
            .map(|i| i.name.as_str())
            .collect();
        let over = self.manifest.plan.overrides.get(&model_op.name);

        if let Some(over) = over {
            if let Some(name) = &over.impl_name {
                if !candidate_names.contains(&name.as_str()) {
                    let trace = &rop.trace;
                    let why = trace
                        .candidates
                        .iter()
                        .find(|c| &c.impl_name == name)
                        .map(|c| c.reason.clone())
                        .unwrap_or_else(|| {
                            format!("no implementation named `{name}` has op `{}`", rop.trace.op)
                        });
                    return Err(PlanError::OverrideRejected {
                        op_name: model_op.name.clone(),
                        reason: format!("`{name}` does not cover this geometry: {why}"),
                    });
                }
            }
        }

        let record = autotune.require(&rop.key, &candidate_names)?;

        let (_, imp) =
            self.registry
                .by_name(&record.impl_name)
                .ok_or_else(|| PlanError::StaleRecord {
                    op: rop.key.op.clone(),
                    record_impl: record.impl_name.clone(),
                    candidates: candidate_names.join(", "),
                })?;
        if !candidate_names.contains(&imp.name.as_str()) {
            return Err(PlanError::StaleRecord {
                op: rop.key.op.clone(),
                record_impl: record.impl_name.clone(),
                candidates: candidate_names.join(", "),
            });
        }

        if let Some(over) = over {
            if let Some(name) = &over.impl_name {
                if name != &imp.name {
                    return Err(PlanError::OverrideRejected {
                        op_name: model_op.name.clone(),
                        reason: format!(
                            "the tuned record for this shape names `{}`; pinning `{name}` would \
                             serve a route no measurement covers — re-tune with `{name}` in the sweep",
                            imp.name
                        ),
                    });
                }
            }
            if let Some(variant_id) = over.variant_id {
                if variant_id != record.variant.variant_id {
                    return Err(PlanError::OverrideRejected {
                        op_name: model_op.name.clone(),
                        reason: format!(
                            "the tuned record for this shape is variant {}, not {variant_id}",
                            record.variant.variant_id
                        ),
                    });
                }
            }
        }

        self.check_implementation(imp)?;
        if bucket.captured && !imp.capturable() {
            return Err(PlanError::NotGraphSafe(Box::new(NotGraphSafe {
                bucket: bucket.label(),
                op_name: model_op.name.clone(),
                impl_name: imp.name.clone(),
                reason: imp
                    .capture_refusal()
                    .unwrap_or_else(|| "not capturable".to_string()),
            })));
        }

        let split_k = record.variant.split_k();
        let workspaces = imp
            .workspaces
            .iter()
            .map(|w| WorkspaceAssignment {
                name: w.name.clone(),
                kind: w.kind,
                bytes: w.sizing.bytes(&rop.shape, split_k),
                // The split count is the measurement's: the decode split law reads the context
                // band, and the record was taken at whatever it answered.
                used_bytes: w
                    .sizing
                    .used_bytes(&rop.shape, record.variant.get("splits")),
            })
            .collect();
        let launches_each = imp.launches.n_launches(split_k);

        // A measured record lifts `implemented` to `measured`; a route the plugin already declares
        // `default` or `certified` keeps its own, higher, provenance. A record that carries NO
        // measurement lifts nothing: `measured` means "has a provenance record: who, when, at what
        // power cap, against which harness", and "nobody timed this" is not that.
        let promotion = if record.is_measured() {
            imp.promotion.clone().max(Promotion::with(
                PromotionState::Measured,
                record.provenance.clone(),
            ))
        } else {
            imp.promotion.clone()
        };

        Ok(BoundOp {
            op_name: model_op.name.clone(),
            op: model_op.op.clone(),
            shape: rop.shape.clone(),
            impl_name: imp.name.clone(),
            plugin: imp.plugin.clone(),
            variant: record.variant.clone(),
            measured_us: record.measured_us,
            margin_pct: record.margin_pct(),
            not_measured: record.provenance.not_measured.clone(),
            tie_broken_by: record.provenance.tie_broken_by,
            unstable: record.unstable.clone(),
            promotion,
            graph_safe: imp.graph_safe,
            may_sync: imp.may_sync,
            wave_size: imp.wave_size,
            occurrences: model_op.calls_in(bucket.phase, bucket.batch, bucket.m()),
            launches_each,
            launches_total: launches_each
                * model_op.calls_in(bucket.phase, bucket.batch, bucket.m()),
            workspaces,
            selection: rop.trace.clone(),
        })
    }

    fn check_implementation(&self, imp: &Implementation) -> Result<()> {
        if self.manifest.plugin(&imp.plugin).is_none() {
            return Err(PlanError::UnknownPlugin {
                impl_name: imp.name.clone(),
                plugin: imp.plugin.clone(),
            });
        }
        if !self.device.supports_wave_size(imp.wave_size) {
            return Err(PlanError::ImplUnsupported {
                impl_name: imp.name.clone(),
                gfx: self.device.gfx_target.clone(),
                reason: format!(
                    "compiled for wave{}, which this target does not support (wave size is a \
                     translation-unit property, so a plugin mixing both must ship two code objects)",
                    imp.wave_size
                ),
            });
        }
        if imp.lds_bytes > self.device.lds_bytes {
            return Err(PlanError::ImplUnsupported {
                impl_name: imp.name.clone(),
                gfx: self.device.gfx_target.clone(),
                reason: format!(
                    "asks for {} B of LDS per workgroup, the target has {} B",
                    imp.lds_bytes, self.device.lds_bytes
                ),
            });
        }
        if let Err(reason) = imp.promotion.check() {
            return Err(PlanError::Promotion {
                impl_name: imp.name.clone(),
                reason,
            });
        }
        Ok(())
    }
}

/// Schema version of [`PlanBody`]. Bumped whenever the plan's shape changes.
pub const PLAN_SCHEMA_VERSION: u32 = 1;

fn aggregate_workspaces(ops: &[BoundOp]) -> Vec<WorkspaceBudget> {
    let kinds = [
        WorkspaceKind::Scratch,
        WorkspaceKind::Persistent,
        WorkspaceKind::IpcPeerVisible,
    ];
    kinds
        .iter()
        .filter_map(|kind| {
            let mine: Vec<&WorkspaceAssignment> = ops
                .iter()
                .flat_map(|o| o.workspaces.iter())
                .filter(|w| w.kind == *kind)
                .collect();
            if mine.is_empty() {
                return None;
            }
            // Scratch is aliasable between ops in a bucket, so the bucket needs the largest single
            // request. Persistent survives replay and peer-visible buffers are exported, so both
            // are summed.
            let (bytes, policy) = match kind {
                WorkspaceKind::Scratch => (
                    mine.iter().map(|w| w.bytes).max().unwrap_or(0),
                    "max over ops (aliasable within a bucket)",
                ),
                WorkspaceKind::Persistent => (
                    mine.iter().map(|w| w.bytes).sum(),
                    "sum over ops (survives replay, zeroed once)",
                ),
                WorkspaceKind::IpcPeerVisible => (
                    mine.iter().map(|w| w.bytes).sum(),
                    "sum over ops (exported to the peer rank)",
                ),
            };
            Some(WorkspaceBudget {
                kind: *kind,
                bytes,
                policy: policy.to_string(),
                contributors: mine.len(),
            })
        })
        .collect()
}

/// First pass: match constraints, build the autotune keys, check the bucket set.
pub fn resolve<'a>(
    manifest: &'a Manifest,
    device: &'a DeviceProfile,
    model: &'a ModelSpec,
    registry: &'a Registry,
) -> Result<Resolved<'a>> {
    if manifest.device.target_gfx != device.gfx_target {
        return Err(PlanError::DeviceMismatch {
            manifest: manifest.device.target_gfx.clone(),
            profile: device.gfx_target.clone(),
        });
    }
    if manifest.device.tp != model.tp {
        return Err(PlanError::Model(format!(
            "manifest asks for tp={} but the model spec is sharded for tp={}",
            manifest.device.tp, model.tp
        )));
    }
    if let Some(drafter) = &manifest.model.drafter_tier_id {
        if drafter != &manifest.model.tier_id {
            return Err(PlanError::TierMismatch {
                target: manifest.model.tier_id.clone(),
                drafter: drafter.clone(),
            });
        }
    }
    for name in manifest.plan.overrides.keys() {
        if model.op(name).is_none() {
            return Err(PlanError::OverrideRejected {
                op_name: name.clone(),
                reason: format!("`{}` has no op of that name", model.name),
            });
        }
    }

    let plugin_digest = manifest.plugin_build_digest();

    // PASS 1: match constraints. This is also what decides which registry rows the input hash
    // keys on, so it has to happen before the hash exists — a row no candidate set contains
    // cannot change what any record means, and hashing it anyway is what made eight Paro
    // rows invalidate an NVFP4 tier's whole cache.
    let mut selected: Vec<(Bucket, GdnStateBudget, Vec<PreOp>)> =
        Vec::with_capacity(manifest.buckets.len());
    let mut bindable: Vec<ImplId> = Vec::new();
    for section in &manifest.buckets {
        let bucket = manifest.bucket(section);
        let gdn_state = gdn_budget(model, &bucket, device)?;
        let mut ops = Vec::with_capacity(model.ops.len());
        for (index, model_op) in model.ops.iter().enumerate() {
            // A route that does not run in this bucket is not resolved for it. The GDN decode step
            // is why: the fused kernel and the conv+recurrent pair are different saber ops with
            // different launch counts, and exactly one of them runs at a given batch. Resolving
            // both would double the GDN entry in the launch budget; resolving neither would hide
            // it.
            if model_op.calls_in(bucket.phase, bucket.batch, bucket.m()) == 0 {
                continue;
            }
            let shape = model_op.shape_keys(
                bucket.phase,
                bucket.m(),
                bucket.batch,
                bucket.spec_width,
                bucket.ctx_band,
            );
            let candidates = registry.select(&model_op.op, &shape)?;
            let trace = registry.explain(&model_op.op, &shape);
            bindable.extend(candidates.iter().copied());
            ops.push(PreOp {
                model_op: index,
                shape,
                candidates,
                trace,
            });
        }
        selected.push((bucket, gdn_state, ops));
    }

    // The tier's launch POLICY is NOT here, and that is the I18 change. Without a policy
    // component anywhere, a rules edit leaves every record binding the variant the old policy
    // chose -- measured, provenanced, and answering a question nobody is asking any more. With it
    // HERE, in the hash every key shares, one line changed in any of the seven tables re-keys all
    // 85 records and costs a full sweep. It is per KEY instead (`AutotuneKey::policy_digest`),
    // cut from the tables owning the plugins that route's own candidates live in, so the guard
    // still fires on an edit and `--autotune-in` re-measures only what the edit could reach.
    let input_hash = crate::hash::digest_of(&(
        manifest.digest(),
        device.digest(),
        model.digest(),
        registry.bindable_fingerprint(&bindable),
    ));

    // PASS 2: the keys, now that the hash they carry exists.
    let mut buckets = Vec::with_capacity(selected.len());
    for (bucket, gdn_state, pre) in selected {
        let mut ops = Vec::with_capacity(pre.len());
        for op in pre {
            let key = AutotuneKey::new(
                &model.ops[op.model_op].op,
                op.shape.clone(),
                manifest.device.power_cap_w,
                &input_hash,
                &plugin_digest,
                &registry.policy_digest_for(registry.plugins_of(&op.candidates)),
            )?;
            ops.push(ResolvedOp {
                model_op: op.model_op,
                shape: op.shape,
                candidates: op.candidates,
                trace: op.trace,
                key,
            });
        }
        buckets.push(ResolvedBucket {
            bucket,
            gdn_state,
            ops,
        });
    }

    Ok(Resolved {
        manifest,
        device,
        model,
        registry,
        input_hash,
        buckets,
    })
}

/// Resolve, then bind: the whole planner.
///
/// Errors, all of them refusals to start:
/// [`PlanError::NoImplementation`] (nothing covers an op at a geometry),
/// [`PlanError::NotTuned`] (no measurement for a bound route),
/// [`PlanError::InfinityCache`] (a bucket's speculative GDN state overflows the IC),
/// [`PlanError::NotGraphSafe`] (a `!graph_safe` route in a captured bucket), and the configuration
/// mismatches.
pub fn plan(
    manifest: &Manifest,
    device: &DeviceProfile,
    model: &ModelSpec,
    registry: &Registry,
    autotune: &AutotuneCache,
) -> Result<ExecutionPlan> {
    resolve(manifest, device, model, registry)?.bind(autotune)
}
