//! Qwen3.8 Flash-Next (`model_type qwen4_exp`): the model facts and the pack plan.
//!
//! * [`config`] — `config.json` as typed data, refusing unknown and missing fields by name
//!   (`docs/FLASHNEXT-PLAN.md` §2, workstream FN2).
//! * [`pack_plan`] — the EP=2 tier pack's record set: which records exist per rank / shared, what
//!   source kind builds each one (the `quark_mxfp4` byte copy, bf16 passthrough or merge, f32
//!   casts), the expert tier's fixed-size records and the PLE table's NVMe index, and the byte
//!   ledger each record class contributes per rank. `tools/pack/build_flashnext_pack.py` does the
//!   bytes; `tools/pack/check_flashnext_pack.py` checks them.
//! * [`layer_graph`] — the decoder layer's op list per kind, its kernels and launch counts, asserted
//!   against the kernel plane's constants (FN9, `docs/EXEC.md` §15).
//! * [`ple_hash`] — the PLE n-gram hash on the host (FN6): token ids -> the 16 table rows per
//!   token, from the checkpoint's stored buffers, gated exactly against FN3's `ple_ids`.
//! * [`nvfp4_csf`] — FN144: the FP4-Local checkpoint's CSF scale codec (lossless decode to plain
//!   E4M3 g16 scales); [`st`] — a minimal safetensors shard-set reader.

pub mod config;
mod fields;
pub mod layer_graph;
pub mod nvfp4_csf;
pub mod pack_plan;
pub mod ple_hash;
pub mod st;

pub use config::{
    LayerKind, ModeloptAlgo, ModeloptLayer, QuantConfig, Qwen4ExpConfig, ScaleEncoding,
};
