//! saber-mem — see docs/ARCHITECTURE.md and docs/IMPLEMENTATION-PLAN.md §1 for this crate's role.
//!
//! Workstream E owns two pieces of it today:
//!
//! * [`pack`] — the tier-pack reader and the streaming `O_DIRECT` loader. The on-disk format is
//!   specified byte for byte in `docs/abi/PACK.md`; `include/saber_abi.h` mirrors the structs.
//! * [`ledger`] — the byte ledger. Over-commit is a startup error, not an OOM at token 4000.
//!
//! * [`park`] — workstream K4: session parking. An idle session's KV and GDN state leave VRAM as
//!   ONE image, into a fixed pool of pinned RAM slots first and an `O_DIRECT` NVMe file under
//!   pressure, hashed on the way out and verified on the way back.
//! * [`upload`] — workstream K: the cold-start path, pack stream to VRAM through a pinned staging
//!   ring, with per-record readiness and the ledger reservation in front of the arena.
//!
//! * [`xtier`], [`expert_tier`], [`expert_mover`] — workstream FN7: the Flash-Next expert tier
//!   (VRAM slots <- pinned RAM L2 <- the NVMe `experts.xtier` records), its placement contract
//!   with FN4's `xptr` table and its which-path counters (docs/EXEC.md §14).
//! * [`device_mover`] — workstream FN8: the tier's device mover (one ordered copy stream per rank,
//!   per-layer landed marks, the consumer fence) over HIP or a host model of an async stream.
//!
//! * [`kv_tier`] — workstream FN45a: the Flash-Next tiered KV cache (QSA pages: VRAM hot <- pinned
//!   RAM warm <- NVMe cold, write-through, gathered per layer from the step selection).
//!
//! The (separate) GDN-state allocator is not here yet.
//!
//! # `unsafe`
//!
//! All of it lives in [`dio`], is limited to the `preadv` call itself, and is documented there.
//! Every other module in this crate carries `#![forbid(unsafe_code)]`, which is what keeps the
//! "unsafe only in `saber-hip`/`saber-hsa`" rule meaningful for the one crate that has to talk to
//! the filesystem: a reviewer reads one function, not a crate.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod device_mover;
pub mod dio;
pub mod elastic;
pub mod expert_mover;
pub mod expert_tier;
pub mod flashnext_budget;
pub mod kv_ssd;
pub mod kv_tier;
pub mod ledger;
pub mod mem_valve;
pub mod p2p_fill;
pub mod pack;
pub mod park;
pub mod ple_tier;
pub mod prefix_tree;
pub mod route_trace;
pub mod sched_signal;
pub mod stage_ring;
pub mod step_ahead;
pub mod upload;
pub mod xtier;
