//! `saber-shadow` — the shadow week's tools (`docs/SHADOW.md` §1 and §3), host-only.
//!
//! * [`mirror`] — the recording mirror: a transparent reverse proxy in front of an
//!   OpenAI-compatible endpoint that records every generation exchange to an append-only JSONL
//!   day file;
//! * [`replay`] — sends a day's recorded requests to a Saber endpoint, byte for byte, and stores
//!   the answers beside them;
//! * [`report`] — the daily reader: compares the two by the runbook's rules and prints the six
//!   stop conditions.
//!
//! **Why Rust, in the workspace, and not a Python tool.** The mirror's one hard requirement is
//! that it changes nothing a client receives, and the proof of that is a test against the
//! serving plane's own mock endpoint (`saber-serve`'s `MockEngine` behind the real router) — which
//! is in-process and runs under `cargo test --workspace` only from a Rust test. The same stack
//! (axum + hyper + reqwest) is already the serving plane's, so the proxy adds no dependency the
//! workspace does not already build, and a streamed body is forwarded by the same HTTP code that
//! produces it. The box has no Python HTTP stack worth standing on (`python3` only, no aiohttp).
//!
//! Nothing here reads the environment (structural rule 2): the log directory is an argument, and
//! `tools/shadow/shadow.sh` is what defaults it to `$SABER_SHARED_CACHE/shadow`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod mirror;
pub mod record;
pub mod replay;
pub mod report;
