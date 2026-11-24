//! Inkling 2 Mini (`model_type inkling2_loop`): the model facts and the pack plan (INK1,
//! `docs/INKLING2.md`).
//!
//! * [`config`] — `config.json` as typed data (unknown / missing fields refused by name), the
//!   48-visit loop schedule ([`config::Inkling2Config::visits`]) and the per-sequence state ledger.
//! * [`layer_graph`] — INK2: the op list of a visit, its kernel classes and launch law.
//! * [`engine_plan`] — INK2: the engine's per-rank arena / KV pools / conv states / runtime classes.
//! * [`pack_plan`] — the TP=2 / EP=2 record set and its per-rank bytes.

pub mod config;
pub mod engine_plan;
pub mod layer_graph;
pub mod pack_plan;

pub use config::{ExpertQuant, Inkling2Config, Visit};
pub use engine_plan::{EngineGeom, RankPlan};
pub use pack_plan::{mxfp8_layout, ExpertForm, PackPlan};

#[cfg(test)]
mod tests;
