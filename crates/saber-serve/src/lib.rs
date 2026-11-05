//! saber-serve — the OpenAI `/v1` serving plane.
//!
//! See `docs/ARCHITECTURE.md` § "Scope" and § "5. Serving plane", and
//! `docs/IMPLEMENTATION-PLAN.md` §1 / P2–P4. The parity requirements this crate
//! exists to meet, in the order they bind:
//!
//! 1. **One SSE update per engine step.** BetterBench classifies a stream as
//!    `per_token` / `batched` from `completion_tokens / n_chunks`; prod's headline
//!    `update p50` is ms/step *only* because prod emits one update per step. A
//!    per-token flush would make the number incomparable. [`server`] therefore emits
//!    exactly one chunk per [`engine::StepEvent`], carrying all of that step's text.
//! 2. **Exact `usage`**, `finish_reason` semantics, and the terminating `[DONE]`.
//! 3. **`reasoning`, not `reasoning_content`** (the reference 0.26+), with `content: null`
//!    while only reasoning has been produced.
//! 4. **`qwen3_coder` tool calls** parsed exactly as the reference server's parser does — gated by
//!    `tests/parser_corpus.rs` against a corpus recorded inside the prod image.
//! 5. **`/metrics` under the names `eval_drafter.py` and BetterBench read.**
//!
//! The engine itself is behind the [`engine::Engine`] trait; [`engine::MockEngine`]
//! replays a script so every test here runs without a GPU.

pub mod census;
pub mod conversation;
pub mod detok;
pub mod engine;
pub mod fatal;
pub mod ids;
pub mod images;
pub mod metrics;
pub mod parse;
pub mod prompt_logprobs;
pub mod protocol;
pub mod reqcheck;
pub mod sampler;
pub mod sampling;
pub mod server;
pub mod state;
pub mod warm;

pub use engine::{Engine, EngineError, FinishReason, MockEngine, StepEvent, Submission};
pub use server::{router, ServeError};
pub use state::{ServerState, ServerStateBuilder};
