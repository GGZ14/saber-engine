//! `ModelSpec` — the ops a plan has to resolve, with their shapes.
//!
//! For P0 this is a flat list of linear-layer shapes: the GEMM family is 44.5% of the decode wall
//! and the only prefill GEMM, so it is the port that validates whether the plan can express launch
//! logic as data. Attention, GDN and all-reduce arrive with the second plugin (saber) and are
//! added here as further [`ModelOp`] rows — nothing about the planner changes when they do.

use crate::error::{PlanError, Result};
use crate::plan::Phase;
use crate::shape::{ShapeKeys, ShapeValue};
use saber_kernels::AProducer;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// 3.0 MiB — one GDN layer's recurrent state for one sequence at fp32:
/// 48 value heads x 128 x 128 x 4 B. The number that sets the Infinity-Cache bucket bound.
pub const GDN_STATE_BYTES_PER_LAYER_PER_SLOT: u64 = 3 * 1024 * 1024;

/// How a weight is sharded across tensor-parallel ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Parallelism {
    /// Not sharded.
    Replicated,
    /// Output columns split across ranks: `N / tp`.
    Column,
    /// Input rows split across ranks: `K / tp`, followed by an all-reduce.
    Row,
}

impl Parallelism {
    /// The name `explain` prints.
    pub fn as_str(self) -> &'static str {
        match self {
            Parallelism::Replicated => "replicated",
            Parallelism::Column => "column",
            Parallelism::Row => "row",
        }
    }
}

impl fmt::Display for Parallelism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// `FUSED_MAX_ITEMS` from `rules/saber_gfx1201.toml [gdn]`, restated here because the *model* is
/// what knows how many value heads a rank owns and therefore which buckets clear the threshold.
///
/// It is a plugin constant living in two files, which is exactly the failure this project exists to
/// delete — so `gdn_fused_max_items_agrees_with_the_launch_rules` asserts the two are equal, and a
/// rules-file edit that moves it fails the build rather than silently re-routing the decode step.
pub const GDN_FUSED_MAX_ITEMS: u32 = 32;

/// `[ar] quant_min_bytes` from `rules/saber_gfx1201.toml`, restated here for the same reason as
/// [`GDN_FUSED_MAX_ITEMS`]: the *model* is what knows the message size, and the rules file is what
/// knows the threshold.
///
/// `_quant_ok` picks the rotated, `bits`-per-symbol kernel when `nbytes >= quant_min_bytes` and
/// takes the exact one below it — 128 KiB is where compression starts to pay, because under it the
/// transfer is not bandwidth-bound and the rotation, the amax reduction and the pack cost more than
/// they save. `ar_quant_min_bytes_agrees_with_the_launch_rules` asserts the copy against its
/// source, so a rules edit moves the band instead of silently disagreeing with it.
pub const AR_QUANT_MIN_BYTES: u64 = 131_072;

/// Bytes per element of an all-reduce payload. The hidden-state reduce is bf16 on this tier, which
/// is also what makes the quantised family eligible at all (`_quant_ok` requires bf16 or fp16).
pub const AR_PAYLOAD_BYTES: u64 = 2;

/// `a_tiled_min_m`, **read** from `rules/mxfp4_gfx1201.toml [knobs]` rather than restated.
///
/// It was a `const A_TILED_MIN_M = 513` here — one of four copies, each pinned to the rules file
/// by a host test — until P6 showed what a copy costs: L15 raised the MXFP4 copy to 99999 to ask
/// whether the A-tiled kernel was reached, the other three stayed at 513, and the run measured a
/// different layout mismatch. See `saber_kernels::rules::a_tiled_min_m`.
#[must_use]
pub fn a_tiled_min_m() -> u32 {
    saber_kernels::rules::a_tiled_min_m()
}

/// Where a GEMM's A-side activation layout comes from.
///
/// `IMPLEMENTATION-PLAN.md` P0 follow-up (a): *"Activation layout is a per-edge property … the
/// port collapsed it to one M threshold, which is right for the fused producers but cannot say
/// 'this edge is row-major because a different producer wrote it.'"*
///
/// `saber_model::graph` carries the full edge graph and finds exactly one edge in this model on
/// which the collapsed rule is wrong: **`gdn.out_proj`'s input**, written by `gdn_norm_quant`,
/// which has exactly one instantiation (store `q += row * N`) and therefore writes row-major
/// bytes at **every** M. At a prefill M the shape-only answer names the A-tiled kernel for it,
/// and reading row-major bytes with the A-tiled kernel is garbage rather than a regression — 48
/// layers of it, layer 0 first (P6).
///
/// **This type used to carry the opposite claim**, and `layer.attn.o_proj` was the row that
/// declared `RowMajor`: right about there being one exceptional edge, wrong about which. The
/// engine never read it (its kernels come from `saber_kernels::rules`), so no token depended on
/// it and nothing said so. It is derived now — [`ModelOp::a_written_by`] names the PRODUCER and
/// asks `saber_kernels::AProducer`, which is the same function `saber_exec::layer` asks — so the
/// plan plane cannot hold a different opinion from the exec plane about what a kernel writes.
///
/// [`ModelOp::shape_keys`] emits the resolved answer as the `a_layout` selection key the registry
/// rows constrain on.
pub type ALayout = saber_kernels::ALayout;

/// The `a_layout` selection key an [`ALayout`] gives at this `M`.
///
/// Free function rather than an inherent method, because [`ALayout`] is the kernel plane's type:
/// what a layout is called in a SHAPE KEY is this plane's business.
#[must_use]
pub fn a_layout_key(layout: ALayout, m: u32) -> &'static str {
    match layout {
        ALayout::RowMajor => "row_major",
        ALayout::FollowsM if m >= a_tiled_min_m() => "a_tiled",
        ALayout::FollowsM => "row_major",
    }
}

/// How often an op runs in a given bucket.
///
/// Not every route runs in every bucket, and pretending otherwise makes the launch budget wrong in
/// both directions. The GDN decode step is the case that forces this: `nseq * H <=
/// FUSED_MAX_ITEMS` picks the FUSED kernel (one launch) and anything above it takes the
/// conv_update + recurrent_update **pair** (two), so at TP=2 with H = 24 batch 1 is fused and
/// batch 2 and up is the pair. They are different saber ops with different launch counts, and a
/// flat per-step count cannot say that.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Occurrences {
    /// This many times in every bucket, whatever its phase.
    EveryStep(u32),
    /// This many times, but only in buckets of one phase.
    ///
    /// For the routes that are phase-exclusive outright rather than banded: the prefill
    /// attention kernel and the three-kernel GDN chunk path exist only in a prefill bucket, and
    /// the split-KV attention pair only in a decode one. A phase-blind row would resolve the
    /// prefill GDN trio into a decode bucket's launch budget, which is 144 launches of pure
    /// fiction.
    WhenPhase {
        /// Calls per step in a bucket of that phase.
        count: u32,
        /// The phase this route serves.
        phase: Phase,
    },
    /// This many times, but only in buckets whose batch is within `[min_batch, max_batch]`.
    /// A bucket outside the band does not resolve this op at all.
    ///
    /// For a condition that is genuinely about the number of SEQUENCES — the GDN decode step's
    /// `nseq * H <= FUSED_MAX_ITEMS`.
    WhenBatch {
        /// Calls per step when the bucket qualifies.
        count: u32,
        /// Smallest batch this route serves.
        min_batch: u32,
        /// Largest batch this route serves; `0` means the route never applies.
        max_batch: u32,
        /// Restrict to one phase as well. `None` serves both.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        phase: Option<Phase>,
    },
    /// This many times, but only in buckets whose row count `m = batch x spec_width` is within
    /// `[min_m, max_m]`.
    ///
    /// For a condition that is about the SIZE of a message rather than the number of sequences.
    /// The all-reduce is the case: `_quant_ok` compares `nbytes` against `quant_min_bytes`, and
    /// `nbytes = hidden x elem_bytes x m`, so a batch band would only be right at one spec width.
    /// The rules file says the same thing in tokens — "a batch-1 step at spec 8 is 8 x 5120 x
    /// bf16 = 80 KiB and takes the EXACT kernel, while a conc-8 step at 64 tokens is 640 KiB and
    /// takes wht6".
    WhenRows {
        /// Calls per step when the bucket qualifies.
        count: u32,
        /// Smallest row count this route serves.
        min_m: u32,
        /// Largest row count this route serves; `0` means the route never applies.
        max_m: u32,
        /// Restrict to one phase as well. `None` serves both — which is what the lossy
        /// all-reduce needs: a prefill chunk's 20 MiB reduce and a conc-8 decode step's 400 KiB
        /// reduce are the same route, and only the row band separates it from the exact one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        phase: Option<Phase>,
    },
}

impl Occurrences {
    fn default_every_step() -> Occurrences {
        Occurrences::EveryStep(1)
    }

    /// Calls in this bucket. Zero means the op does not run here.
    #[must_use]
    pub fn in_bucket(self, phase: Phase, batch: u32, m: u32) -> u32 {
        let band = |v: u32, lo: u32, hi: u32, count: u32| {
            if v >= lo && v <= hi {
                count
            } else {
                0
            }
        };
        let phase_ok = |want: Option<Phase>| want.is_none_or(|w| w == phase);
        match self {
            Occurrences::EveryStep(n) => n,
            Occurrences::WhenPhase { count, phase: want } => {
                if want == phase {
                    count
                } else {
                    0
                }
            }
            Occurrences::WhenBatch {
                count,
                min_batch,
                max_batch,
                phase: want,
            } => {
                if phase_ok(want) {
                    band(batch, min_batch, max_batch, count)
                } else {
                    0
                }
            }
            Occurrences::WhenRows {
                count,
                min_m,
                max_m,
                phase: want,
            } => {
                if phase_ok(want) {
                    band(m, min_m, max_m, count)
                } else {
                    0
                }
            }
        }
    }

    /// The largest number of calls this route can make in any bucket, for a static budget.
    #[must_use]
    pub fn peak(self) -> u32 {
        match self {
            Occurrences::EveryStep(n) | Occurrences::WhenPhase { count: n, .. } => n,
            Occurrences::WhenBatch {
                count, max_batch, ..
            } => {
                if max_batch == 0 {
                    0
                } else {
                    count
                }
            }
            Occurrences::WhenRows { count, max_m, .. } => {
                if max_m == 0 {
                    0
                } else {
                    count
                }
            }
        }
    }
}

/// A shape key the **bucket** supplies, because the op cannot know it.
///
/// Both of these were worst cases in the key until 2026-09-19, and both were wrong for the same
/// reason: an autotune record is a measurement of the geometry its key names, so a key that names
/// a geometry the engine never runs is a record of a kernel nobody launches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BucketKey {
    /// `q_len` = the bucket's speculative width. The registry's `q_len <= 10` stays a necessary
    /// condition on the *kernel*; this is the width actually verified.
    QLen,
    /// `max_ctx` = the bucket's context band. It sets the split count, and therefore the whole
    /// cost of a split-KV decode: two bands are two measurements, not one.
    MaxCtx,
}

/// How many rows an op runs at, given the bucket.
///
/// Almost everything runs at the bucket's `M`. The language-model head does not: logits are
/// computed for the positions that are **sampled**, and a prefill chunk samples one position
/// per sequence however many tokens it carries. A 8192-token chunk therefore runs `lm_head` at
/// M=1, not M=8192 -- which is the difference between a 40 MiB GEMM and a 2.3 GiB one.
///
/// This was not a modelling nicety: the autotune sweep refused the route outright
/// ("2302 MiB of inputs exceeds the 520 MiB budget"), which is how the error surfaced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rows {
    /// The bucket's `M`: `batch x spec_width` at decode, `tokens` at prefill.
    #[default]
    BucketM,
    /// One row per sampled position. At decode that is every verified draft token, so it is the
    /// bucket's `M`; at prefill it is one per sequence.
    Sampled,
}

impl Rows {
    /// Rows this op runs at in a bucket.
    #[must_use]
    pub fn at(self, phase: Phase, batch: u32, m: u32) -> u32 {
        match (self, phase) {
            (Rows::BucketM, _) | (Rows::Sampled, Phase::Decode) => m,
            (Rows::Sampled, Phase::Prefill) => batch,
        }
    }
}

/// One op the plan must resolve, with the shape it runs at and how often per step.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelOp {
    /// Unique name inside the model, e.g. `layer.gdn.in_proj_qkv`. `[plan.overrides]` keys on it.
    pub name: String,
    /// The op to select on, e.g. `gemm_w4a8`.
    pub op: String,
    /// Weight rows (output features) **after** sharding.
    pub n: u32,
    /// Weight columns (input features) **after** sharding.
    pub k: u32,
    /// How many times this op runs per step (one row covers all 48 GDN layers, for instance).
    ///
    /// Kept for the routes that run in every bucket; [`ModelOp::occurrences`] is what the planner
    /// reads, and it is what a bucket-dependent route sets instead.
    pub count: u32,
    /// How often this op runs in a given bucket. Defaults to `EveryStep(count)`.
    #[serde(default = "Occurrences::default_every_step")]
    pub occurrences: Occurrences,
    /// How the weight is sharded.
    pub parallelism: Parallelism,
    /// Which producer writes this op's A-side activation edge. `None` for an op that reads no
    /// fp8 activation — everything that is not a W4A8 GEMM, and a GEMM in a fixture model that
    /// does not model its producers.
    ///
    /// The **producer** is what is declared, never the layout: the layout is what that
    /// producer's kernel writes, and `saber_kernels::AProducer` is the one function that
    /// answers it — the same one `saber_exec::layer::Planner` asks at plan time. See
    /// [`ModelOp::a_written_by`] and [`ModelOp::a_layout`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub a_producer: Option<AProducer>,
    /// How many rows this op runs at in a bucket. Defaults to the bucket's `M`.
    #[serde(default)]
    pub rows: Rows,
    /// Extra selection keys: dtypes, flags.
    #[serde(default)]
    pub extra: BTreeMap<String, ShapeValue>,
    /// Keys whose value comes from the bucket rather than from the op.
    #[serde(default)]
    pub bucket_keys: Vec<BucketKey>,
}

impl ModelOp {
    /// A new op row.
    pub fn new(name: &str, op: &str, n: u32, k: u32, count: u32, parallelism: Parallelism) -> Self {
        Self {
            name: name.to_string(),
            op: op.to_string(),
            n,
            k,
            count,
            occurrences: Occurrences::EveryStep(count),
            parallelism,
            a_producer: None,
            rows: Rows::BucketM,
            extra: BTreeMap::new(),
            bucket_keys: Vec::new(),
        }
    }

    /// Declares which producer writes this op's A-side activation edge.
    ///
    /// The layout is **not** stated: it is derived from the producer by
    /// `saber_kernels::AProducer::a_layout`, which is the same question
    /// `saber_exec::layer::Planner::check_producer_layout` asks of the live rule tables. The
    /// previous spelling of this was `a_row_major()`, a bare assertion that one edge was
    /// row-major — and it was on the wrong row for two weeks (`layer.attn.o_proj` rather than
    /// `layer.gdn.out_proj`, P6) precisely because an assertion nothing derives cannot be wrong
    /// out loud.
    #[must_use]
    pub fn a_written_by(mut self, producer: AProducer) -> Self {
        self.a_producer = Some(producer);
        self
    }

    /// The A-side layout this op's edge carries, derived from its producer.
    ///
    /// `FollowsM` when no producer is declared, which is what [`ModelOp::new`] leaves and what a
    /// fixture model that does not model its producers gets. A GEMM in a REAL spec must name one:
    /// `every_w4a8_gemm_names_the_producer_that_writes_its_a_side` is the guard.
    #[must_use]
    pub fn a_layout(&self) -> ALayout {
        self.a_producer.map_or(ALayout::FollowsM, |p| {
            p.a_layout()
                .expect("the committed rule tables are compiled in and parse")
        })
    }

    /// Declares that this op runs on the SAMPLED positions rather than on every row of the
    /// bucket: one per sequence at prefill, every verified draft token at decode.
    #[must_use]
    pub fn sampled_rows(mut self) -> Self {
        self.rows = Rows::Sampled;
        self
    }

    /// Restricts this op to one phase.
    #[must_use]
    pub fn when_phase(mut self, phase: Phase) -> Self {
        self.occurrences = Occurrences::WhenPhase {
            count: self.count,
            phase,
        };
        self
    }

    /// Restricts an already-banded op to one phase as well.
    #[must_use]
    pub fn in_phase(mut self, want: Phase) -> Self {
        self.occurrences = match self.occurrences {
            Occurrences::EveryStep(count) => Occurrences::WhenPhase { count, phase: want },
            Occurrences::WhenPhase { count, .. } => Occurrences::WhenPhase { count, phase: want },
            Occurrences::WhenBatch {
                count,
                min_batch,
                max_batch,
                ..
            } => Occurrences::WhenBatch {
                count,
                min_batch,
                max_batch,
                phase: Some(want),
            },
            Occurrences::WhenRows {
                count,
                min_m,
                max_m,
                ..
            } => Occurrences::WhenRows {
                count,
                min_m,
                max_m,
                phase: Some(want),
            },
        };
        self
    }

    /// Restricts this op to a band of bucket batches.
    #[must_use]
    pub fn when_batch(mut self, min_batch: u32, max_batch: u32) -> Self {
        self.occurrences = Occurrences::WhenBatch {
            count: self.count,
            min_batch,
            max_batch,
            phase: None,
        };
        self
    }

    /// Restricts this op to a band of bucket row counts (`m = batch x spec_width`).
    #[must_use]
    pub fn when_rows(mut self, min_m: u32, max_m: u32) -> Self {
        self.occurrences = Occurrences::WhenRows {
            count: self.count,
            min_m,
            max_m,
            phase: None,
        };
        self
    }

    /// Calls in a bucket of this phase, batch and row count; zero means the op does not run
    /// there.
    #[must_use]
    pub fn calls_in(&self, phase: Phase, batch: u32, m: u32) -> u32 {
        self.occurrences.in_bucket(phase, batch, m)
    }

    /// Adds a selection key (dtype, flag).
    #[must_use]
    pub fn key(mut self, key: &str, value: impl Into<ShapeValue>) -> Self {
        self.extra.insert(key.to_string(), value.into());
        self
    }

    /// Adds a key whose value the BUCKET supplies, not the op.
    #[must_use]
    pub fn bucket_key(mut self, key: BucketKey) -> Self {
        if !self.bucket_keys.contains(&key) {
            self.bucket_keys.push(key);
        }
        self
    }

    /// The full geometry in a given bucket — what constraints are tested against and what the
    /// autotune key is built from.
    ///
    /// `batch` and `spec_width` are carried alongside `m` because `m = batch x spec_width` has
    /// already lost the distinction the kernel plane needs: attention's `q_len` is the spec width
    /// and the GDN decode step's route is decided by the batch, and neither is recoverable from
    /// the product.
    ///
    /// [`ModelOp::bucket_key`]s are written **after** the static ones, so a key an op declares as
    /// a worst case is replaced by the value this bucket actually serves.
    pub fn shape_keys(
        &self,
        phase: Phase,
        m: u32,
        batch: u32,
        spec_width: u32,
        ctx_band: u32,
    ) -> ShapeKeys {
        let m = self.rows.at(phase, batch, m);
        let mut keys = ShapeKeys::new()
            .with("m", m)
            .with("batch", batch)
            .with("spec_width", spec_width)
            .with("n", self.n)
            .with("k", self.k);
        if self.op.starts_with("gemm") {
            // The per-edge layout obligation, as a selection key. It is emitted only for a GEMM
            // because it is only a GEMM that reads an activation layout; a registry row for any
            // other op would have to carry a predicate on a key that means nothing to it.
            keys.insert(
                "a_layout",
                ShapeValue::from(a_layout_key(self.a_layout(), m)),
            );
        }
        for (k, v) in &self.extra {
            keys.insert(k, v.clone());
        }
        for bk in &self.bucket_keys {
            match bk {
                BucketKey::QLen => keys.insert("q_len", spec_width),
                BucketKey::MaxCtx => keys.insert("max_ctx", ctx_band),
            }
        }
        keys
    }
}

/// The model the plan is resolved for.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Model name, printed by `explain`.
    pub name: String,
    /// Tensor-parallel ranks the shapes below are already sharded for.
    pub tp: u32,
    /// Hidden size.
    pub hidden: u32,
    /// Total decoder layers.
    pub layers: u32,
    /// Linear-attention (gated delta net) layers. Sets the GDN state budget.
    pub gdn_layers: u32,
    /// Full-attention layers.
    pub attn_layers: u32,
    /// Bytes of recurrent state one GDN layer holds for one sequence slot, per rank *before* the
    /// TP divide. 3.0 MiB at fp32; fp16 state halves it (proven at TP=1, parked at TP=2).
    pub gdn_state_bytes_per_layer_per_slot: u64,
    /// The ops to resolve.
    pub ops: Vec<ModelOp>,
}

impl ModelSpec {
    /// Qwen3.8-27B dense, sharded for `tp` ranks.
    ///
    /// From `models/Qwen3.8-27B-PARO/config.json`: 64 layers, hidden 5120, 48 GDN + 16 attention
    /// (`full_attention_interval` 4), head_dim 256, GQA 6. Column-parallel weights divide `N` by
    /// `tp`; row-parallel weights divide `K`.
    ///
    /// Returns an error when a dimension does not divide by `tp` — TP=3 needs the padded head
    /// layout (36/6/18/54), which is a checkpoint-preparation decision and not something the
    /// planner may invent.
    ///
    /// The op list covers **both phases**. The linears are shared between them (same weights, a
    /// different row count, and the M band picks the family); attention and the GDN mixer are
    /// not, and their rows carry the phase they serve. The all-reduce is shared and banded on
    /// the message size alone, because a prefill chunk's 20 MiB reduce and a conc-8 decode
    /// step's 400 KiB reduce are the same route.
    pub fn qwen38_27b_dense(tp: u32) -> Result<Self> {
        if tp == 0 {
            return Err(PlanError::Model("tp must be >= 1".to_string()));
        }
        let hidden = 5120u32;
        let layers = 64u32;
        let gdn_layers = 48u32;
        let attn_layers = 16u32;

        let col = |name: &str, n: u32, k: u32, count: u32| -> Result<ModelOp> {
            divide(name, "N", n, tp).map(|n| {
                ModelOp::new(name, "gemm_w4a8", n, k, count, Parallelism::Column)
                    .key("w_dtype", "mxfp4")
                    .key("a_dtype", "int8")
            })
        };
        let row = |name: &str, n: u32, k: u32, count: u32| -> Result<ModelOp> {
            divide(name, "K", k, tp).map(|k| {
                ModelOp::new(name, "gemm_w4a8", n, k, count, Parallelism::Row)
                    .key("w_dtype", "mxfp4")
                    .key("a_dtype", "int8")
            })
        };

        // ---- the linears, MERGED as prod launches them ------------------------------------
        //
        // Not one row per weight tensor: one row per GEMM. Prod merges q/k/v into one
        // column-parallel linear, gate and up into another, and the GDN `in_proj` into a qkvz
        // and a ba half (`SABER_GDN_MERGE_INPROJ=1`) -- so the step launches 305 W4A8 GEMMs,
        // not 401, and a per-tensor list would inflate the launch budget by a quarter while
        // naming shapes no kernel is ever asked for.
        //
        // Three independent things say these are the right widths:
        //
        // * **Prod's own census.** The NVFP4 unit logs `linear layers: 304/304` when the
        //   load-time requant has covered every decoder linear -- 305 minus `lm_head`, which
        //   that patch handles separately. Unmerged, the count would be 369 or 353.
        // * **The tuned bands.** `rules/mxfp4_gfx1201.toml`'s measured decode ladder is written
        //   against `n8192` (the merged GDN qkvz at TP=2), `n7168` (the merged attention QKV),
        //   `gate48` (the merged `in_proj_ba`), `gate_up` at 17408, and `down` at "68 k-slabs"
        //   = K 8704. Those names only exist because the GEMMs are merged.
        // * **`saber_model::graph`**, which walks the layer graph edge by edge and arrives at
        //   the same 305. `saber-plan/tests/prefill.rs` asserts the two agree.
        //
        // The segment boundaries are all 128-aligned (`partition_align`), which is what lets a
        // merged linear be one launch rather than one per segment: qkvz is 5120 + 3072 at TP=2,
        // QKV is 6144 + 512 + 512, gate_up is 8704 + 8704. `in_proj_ba` is the exception at 48
        // rows and is merged anyway because it is two rows per value head, not a partitioned
        // read of a rotated A.
        // Every W4A8 GEMM names the PRODUCER that writes its A side. The layout follows from the
        // producer's kernel (`saber_kernels::AProducer`), which is the same function the exec
        // plane's `check_producer_layout` asks -- so the two planes cannot hold different
        // opinions about what `gdn_norm_quant` writes, which is the state this spec was in from
        // the day the `a_layout` key existed until P6 + I20.
        let mut ops = vec![
            // concat(q, k, v, z): (2*Hg + H)*128 + H*128 = 10240 + 6144.
            col("layer.gdn.in_proj_qkvz", 16384, hidden, gdn_layers)?
                .a_written_by(AProducer::AddRmsQuant),
            // concat(b, a): one row each per value head. 48 at TP=2 -- the `gate48` shape.
            col("layer.gdn.in_proj_ba", 96, hidden, gdn_layers)?
                .a_written_by(AProducer::AddRmsQuant),
            // **The one edge in the model the M threshold gets wrong**, and until I20 this spec
            // had the wrong row carrying it. Its A side is `gdn_norm_quant`'s output, and that
            // producer has exactly ONE instantiation (`gdn_norm_quant_t{block}`, store
            // `q += row * N`) -- as the reference does, and as prod says from the other end by
            // never calling `a_tiled_register` after it. So the bytes are row-major at EVERY M,
            // including a prefill M where every other GEMM in the same layer is A-tiled, and the
            // `a_layout` key is what makes the registry refuse the A-tiled family for it rather
            // than pick it and read WMMA fragments out of row-major bytes in all 48 GDN layers
            // (P6: layer 0 first, at p1024 and p2048).
            row("layer.gdn.out_proj", hidden, 6144, gdn_layers)?
                .a_written_by(AProducer::GdnNormQuant),
            // concat(q+gate, k, v). `attn_output_gate` doubles q_proj, so 12288 + 1024 + 1024.
            col("layer.attn.qkv", 14336, hidden, attn_layers)?.a_written_by(AProducer::AddRmsQuant),
            // The row that USED to carry `.a_row_major()`. `attn_gate_quant` is a fused producer
            // on this tier -- `attn_gate_quant_t256_tiled{0,1}` exist in the `attn_glue` plugin
            // and are keyed on the one band edge -- so `o_proj` follows M like every other
            // attention-layer GEMM. It was declared row-major on the grounds that this tier has
            // no fused gate producer, which stopped being true when the glue plugin grew the
            // tiled arm, and nothing read the declaration, so it was never wrong out loud.
            row("layer.attn.o_proj", hidden, 6144, attn_layers)?
                .a_written_by(AProducer::AttnGateQuant),
            // concat(gate, up).
            col("layer.mlp.gate_up", 34816, hidden, layers)?.a_written_by(AProducer::AddRmsQuant),
            row("layer.mlp.down_proj", hidden, 17408, layers)?
                .a_written_by(AProducer::SiluMulQuant),
            // Logits are computed for the SAMPLED positions. A prefill chunk samples one per
            // sequence, so this runs at M=1 there however long the chunk is -- not at M=8192,
            // which would be a 2.3 GiB GEMM the engine never launches.
            col("lm_head", 248320, hidden, 1)?
                .a_written_by(AProducer::AddRmsQuant)
                .sampled_rows(),
        ];

        // ---- the saber half of the step (the kernel inventory doc §2, §3) ---------------
        //
        // The linears above are 44.5% of the decode wall and were the port that proved the plan
        // could express launch logic as data. These three are the rest of what prod launches every
        // step, and adding them is what makes the per-bucket launch count a real budget rather
        // than a GEMM subtotal.
        //
        // Their selection keys are the reference's own (`head_dim`, `gqa`, `block_size`, `causal`,
        // `conv_width`, `head_k`, `head_v`, `world_size`, `exact`, `bits`, `numel`), because the
        // registry they are matched against is a transcription of the reference's constraint table and
        // the two must not drift apart.

        // Paged attention, DECODE. 16 full-attention layers (`full_attention_interval` 4); the
        // prefill kernel is the same op family at a different bucket class and is not a per-step
        // row here.
        //
        // `q_len` and `max_ctx` come from the BUCKET (2026-09-19). They were the worst case this
        // tier can schedule — a verify window of 8, and no context at all — which made the
        // registry match right and the autotune key wrong: the record would have been a
        // measurement of a 8-token window at an unstated context, while the bucket serves 5 at
        // 8192. The registry's `q_len <= 10` is untouched and stays a necessary condition on the
        // kernel; the per-call `q_len * gqa <= 64` check stays at the entry point, as the reference has
        // it.
        let q_heads = divide("layer.attn.paged_decode", "q_heads", 24, tp)?;
        ops.push(
            ModelOp::new(
                "layer.attn.paged_decode",
                "attn_decode_paged",
                q_heads,
                256,
                attn_layers,
                Parallelism::Column,
            )
            .key("head_dim", 256i64)
            .key("gqa", 6i64)
            .key("block_size", 16i64)
            .key("causal", 1i64)
            .key("q_dtype", "bf16")
            .key("kv_dtype", "fp8_e4m3")
            .bucket_key(BucketKey::QLen)
            .bucket_key(BucketKey::MaxCtx)
            .when_phase(Phase::Decode),
        );

        // ---- the attention layer's glue (workstream F, plugin 4) ----------------------
        //
        // Three ops per attention layer, in BOTH phases, one launch each. Until the `attn_glue`
        // plugin landed these were the last places in a forward pass where prod ran Triton or
        // ATen and Saber had nothing: `ModelSpec` named none of them, the graph counted two of
        // them with no launch count, and the paged K/V write was not modelled at all -- the
        // cache was state that every attention op read and no op wrote.
        //
        // `head_dim` and `rotary_dim` are COMPILED INTO the symbols (the rope pair count and
        // the LDS staging are sized by them), so the registry carries them as equalities and a
        // shape off them is a refusal rather than a slower path.
        //
        // `gqa` rides along on all three: the kernels take a KV-head count and the row's `n` is
        // the Q-head count, so without it the sweep would have to re-derive 6 from the config
        // — a second opinion about a number the registry already keys on elsewhere.
        let glue = |name: &str, op: &str| {
            ModelOp::new(name, op, q_heads, 256, attn_layers, Parallelism::Column)
                .key("head_dim", 256i64)
                .key("gqa", 6i64)
        };
        ops.push(
            glue("layer.attn.qk_norm_rope", "qk_norm_rope")
                // partial_rotary_factor 0.25 of head_dim 256. `get_rope` computes exactly this.
                .key("rotary_dim", 64i64)
                // The CONTEXT BAND, and it is in the key for the reason `attn_decode_paged`'s is
                // (2026-09-19): a record is a measurement of the geometry its key names, and this
                // op's `cos_sin` and `pos` operands are functions of the band and of nothing in
                // `(m, nq, nkv, head_dim)`. The step sizes the rope table by
                // `(max_blocks * KV_BLOCK).max(rows)` and reads it at `done_before + t`; a cell
                // that cannot say the band can only measure a 640-byte table where the step
                // binds a 1 MiB one, and can only fill `pos` with `0..rows` where the step gives a
                // decode sequence `band - q .. band - 1`. I19 found that and declared it, because
                // closing it re-keys the records; I20 closes it.
                .bucket_key(BucketKey::MaxCtx),
        );
        ops.push(glue("layer.attn.gate_quant", "attn_gate_quant"));
        ops.push(
            glue("layer.attn.kv_append", "kv_append")
                .key("block_size", 16i64)
                // `--kv-cache-dtype fp8`. There is no non-quantised instantiation, so a caller
                // asking for one is refused rather than served bf16.
                .key("kv_dtype", "fp8_e4m3"),
        );

        // The GDN decode step. TWO routes, and which one runs is decided by the BUCKET, not by a
        // measurement: `nseq * H <= FUSED_MAX_ITEMS` takes the fused kernel (conv update, grid
        // barrier, recurrent update, ONE launch) and anything above it takes the conv_update +
        // recurrent_update PAIR (TWO launches, because a recurrent head consumes q/k written by
        // different conv units, so the handoff crosses workgroups).
        //
        // At TP=2 that is H = 24 against a limit of 32: batch 1 is fused, batch 2 and up is the
        // pair. In prod this is a Python `if` over two environment variables; here it is the
        // bucket band on the two rows, which is the whole point of the exercise.
        let v_heads = divide("layer.gdn.decode_step", "value heads", 48, tp)?;
        let max_fused_batch = GDN_FUSED_MAX_ITEMS / v_heads.max(1);
        let gdn_route = |name: &str, op: &str| {
            ModelOp::new(name, op, v_heads, 128, gdn_layers, Parallelism::Column)
                .key("conv_width", 4i64)
                .key("head_k", 128i64)
                .key("head_v", 128i64)
        };
        ops.push(
            gdn_route("layer.gdn.decode_step.fused", "gdn_fused_update")
                .when_batch(1, max_fused_batch)
                .in_phase(Phase::Decode),
        );
        ops.push(
            gdn_route("layer.gdn.decode_step.pair", "gdn_conv_update")
                .when_batch(max_fused_batch + 1, u32::MAX)
                .in_phase(Phase::Decode),
        );

        // ---- the PREFILL class (P2) -----------------------------------------------------
        //
        // A prefill bucket is not a decode bucket with a bigger M: it runs DIFFERENT ops. The
        // linears above are shared -- the same weights at a different row count, with the M band
        // picking the family -- but the three routes below exist only here, and the two decode
        // routes above exist only there. That is what `WhenPhase` is for, and getting it wrong
        // in either direction puts launches in a budget that will never be submitted.

        // Paged attention, PREFILL: query-tiled, ONE launch per run of equal `q_len`. The
        // backend cuts the batch into maximal runs of equal query length and launches once per
        // run; a single-sequence chunk is one run, which is what the P2 bucket class is.
        ops.push(
            ModelOp::new(
                "layer.attn.paged_prefill",
                "attn_prefill_paged",
                q_heads,
                256,
                attn_layers,
                Parallelism::Column,
            )
            .key("head_dim", 256i64)
            .key("gqa", 6i64)
            .key("block_size", 16i64)
            .key("causal", 1i64)
            .key("q_dtype", "bf16")
            .key("kv_dtype", "fp8_e4m3")
            // No `q_len`: a prefill run's query length is the chunk, and the registry row
            // carries no predicate on it -- the decode row's `q_len <= 10` is what separates
            // the two families. The context band still keys the measurement.
            .bucket_key(BucketKey::MaxCtx)
            .when_phase(Phase::Prefill),
        );

        // The GDN prefill path is THREE kernels per layer, not one: the conv preamble (causal
        // conv + qkv split + qk l2 norm + gating + per-chunk cumsum), the per-chunk triangular
        // solve whose fp32 gram never reaches HBM, and the chunked scan that recomputes WY,
        // advances the state and writes the output. 48 layers x 3 = 144 launches, and none of
        // them appears in a decode bucket.
        //
        // `chunk` is a selection key because the three rows are templated on it and the
        // registry carries `chunk == 64` on each -- a query without it is refused ("chunk not
        // given") rather than defaulted, which is the reference's own behaviour.
        let gdn_prefill = |name: &str, op: &str| {
            ModelOp::new(name, op, v_heads, 128, gdn_layers, Parallelism::Column)
                .key("head_k", 128i64)
                .key("head_v", 128i64)
                .key("chunk", 64i64)
                .when_phase(Phase::Prefill)
        };
        ops.push(
            gdn_prefill("layer.gdn.prefill.conv_prep", "gdn_conv_prep").key("conv_width", 4i64),
        );
        ops.push(gdn_prefill("layer.gdn.prefill.kkt_solve", "gdn_kkt_solve"));
        ops.push(gdn_prefill(
            "layer.gdn.prefill.chunk_scan",
            "gdn_chunk_scan",
        ));

        // The gated RMS norm over the GDN output, 48 layers.
        //
        // A divergence worth naming: PROD binds `_GATED_RMSNORM` and never calls it
        // (the kernel inventory doc §3, "Bound but never called") -- its decode kernel
        // folds the same arithmetic into its epilogue, and on the MXFP4 tier the norm and the
        // quant are one fused producer (`gdn_norm_quant`, which is what `saber_model::graph`
        // names). Saber ported the standalone kernel as `saber_gdn_norm`, so the prefill path has
        // a route for it; the consistency test against the graph records the disagreement
        // rather than hiding it.
        ops.push(
            ModelOp::new(
                "layer.gdn.prefill.gated_rmsnorm",
                "gdn_gated_rmsnorm",
                v_heads,
                128,
                gdn_layers,
                Parallelism::Column,
            )
            .key("channels", 128i64)
            .when_phase(Phase::Prefill),
        );

        // The all-reduce: 2 per layer x 64 layers. It exists only when the model is sharded --
        // at TP=1 there is nothing to reduce, and a row asserting `world_size == 1` would be
        // asking for a kernel that has no reason to exist.
        //
        // TWO rows, because prod runs BOTH families every step and the SIZE of the message picks
        // between them: `_quant_ok` takes the rotated `bits`-per-symbol kernel at or above
        // `quant_min_bytes` and the exact one below it. The message is `hidden x elem_bytes x m`,
        // so the band is on rows, not on batch -- a batch band would only be right at one spec
        // width, and the rules file states the rule in tokens for exactly that reason.
        if tp > 1 {
            // The largest row count whose message still falls under the threshold. Derived, never
            // listed: `hidden` comes from the checkpoint and `AR_QUANT_MIN_BYTES` from the rules
            // file, so changing either moves the band.
            let bytes_per_row = u64::from(hidden) * AR_PAYLOAD_BYTES;
            let exact_max_m =
                u32::try_from((AR_QUANT_MIN_BYTES - 1) / bytes_per_row.max(1)).unwrap_or(u32::MAX);
            let ar = |name: &str, exact: i64| {
                ModelOp::new(
                    name,
                    "allreduce",
                    hidden,
                    hidden,
                    2 * layers,
                    Parallelism::Replicated,
                )
                .key("world_size", i64::from(tp))
                // `exact` is what separates the two families, and a request that does not
                // declare it gets no kernel rather than whichever one is listed first.
                .key("exact", exact)
                .key("dtype", "bf16")
                // `numel` is PER ROW: the message is `hidden x rows`, and the `% 64` condition
                // is row-count-independent because hidden itself is a multiple of 64. Stating
                // it per row keeps the necessary condition true at every bucket.
                .key("numel", i64::from(hidden))
            };
            ops.push(ar("layer.allreduce.exact", 1).when_rows(1, exact_max_m));
            ops.push(
                ar("layer.allreduce.wht", 0)
                    // The width prod runs (`SABER_AR_QBITS=6`). wht4 is a registry row and a
                    // measured option, not this tier's default.
                    .key("bits", 6i64)
                    .when_rows(exact_max_m + 1, u32::MAX),
            );
        }

        Ok(Self {
            name: format!("qwen3.8-27b-dense-tp{tp}"),
            tp,
            hidden,
            layers,
            gdn_layers,
            attn_layers,
            gdn_state_bytes_per_layer_per_slot: GDN_STATE_BYTES_PER_LAYER_PER_SLOT,
            ops,
        })
    }

    /// The op of that name.
    pub fn op(&self, name: &str) -> Option<&ModelOp> {
        self.ops.iter().find(|o| o.name == name)
    }

    /// Canonical digest, folded into the plan's input hash.
    pub fn digest(&self) -> String {
        crate::hash::digest_of(self)
    }
}

fn divide(name: &str, dim: &str, value: u32, tp: u32) -> Result<u32> {
    if value % tp != 0 {
        return Err(PlanError::Model(format!(
            "{name}: {dim}={value} does not divide by tp={tp}; TP={tp} needs a padded head layout \
             in the checkpoint, which the planner will not invent"
        )));
    }
    Ok(value / tp)
}
