//! saber-model — see docs/ARCHITECTURE.md and docs/IMPLEMENTATION-PLAN.md §1 for this crate's role.
//!
//! Workstream D covers the two P0 *accuracy* gates that must pass before any kernel is written
//! (`IMPLEMENTATION-PLAN.md` P0 item 8, P-1e):
//!
//! * [`template`] — `qwen-fixed-v22.3.jinja` rendered byte-identically to the reference server's sandboxed
//!   Jinja2, gated on the ~1000-conversation P-1e corpus.
//! * [`tokenizer`] — `tokenizer.json` through the `tokenizers` crate, gated on exact token-id
//!   *sequence* equality against HF, plus mrope positions and the rotary parameters for this
//!   checkpoint's `partial_rotary_factor 0.25` / `mrope_section [11, 11, 10]`.
//!
//! and two *structural* deliverables:
//!
//! * [`graph`] — the 64-layer forward pass as data: the op sequence, every weight bound by its
//!   exact tier-pack record name, a layout field on every producer→consumer edge, the shape
//!   keys the planner selects on, and a statically declared launch count per bucket.
//! * [`draft`] — DFlash2 block speculative decoding as pure host logic: the verify rule and
//!   its one extra token, the scheduler-side verify-width EMA, the draft-depth schedule, the
//!   longest-suffix n-gram matcher, and the `spec_decode_*` counter definitions. Gated case by
//!   case against prod's own kernels (`docs/abi/DFLASH2-SEMANTICS.md`).

pub mod checkpoint_head;
pub mod dflash_graph;
pub mod draft;
pub mod graph;
pub mod qwen4_exp;
pub mod template;
pub mod tokenizer;

pub use draft::{
    AcceptanceEma, DraftBlock, DraftSchedule, NgramMatcher, SpecDecodeCounters, TargetDecision,
    Verified, WidthPolicy,
};
pub use graph::{Bucket, Graph, KernelOp, Layout, ModelConfig, Phase};
pub use template::{ChatRequest, Renderer};
pub use tokenizer::{rope_params, MropePositions, RopeParams, TextConfig, Tokenizer};

/// Errors from the model plane.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// The chat template failed to compile or to render.
    #[error("chat template: {0}")]
    Template(String),
    /// The tokenizer failed to load or to encode.
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    /// `config.json` is missing a field the model plane needs, or it has the wrong type.
    #[error("model config: {0}")]
    Config(String),
    /// Filesystem error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// JSON parse error.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}
