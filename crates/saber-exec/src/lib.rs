//! `saber-exec` — the execution plane: load a plugin, pack a kernarg image, launch, replay.
//!
//! See `docs/ARCHITECTURE.md` § *Kernel plane* and § *4. Execution plane*, and
//! `docs/IMPLEMENTATION-PLAN.md` P0 / P0.5. This crate is the seam between the two halves that
//! already exist:
//!
//! ```text
//!   prebuilt/gfx1201/*.manifest.toml + *.hsaco
//!            │
//!            ▼
//!   loader ──┴─► three validations ─► hipModuleLoadData ─► hipModuleGetFunction per symbol
//!            │        (b) kernarg rules · (c) wave size · (a) manifest vs code object
//!            ▼
//!   saber_kernels::resolve(op, shape, rules) ─► LaunchDescriptor
//!            │                                    (kernel, grid, block, kernarg slots, workspaces)
//!            ▼
//!   Launcher::launch ─► pack the image · assign workspaces by kind · dispatch · assert n_launches
//! ```
//!
//! What this crate must never do is decide anything. Tile choice, split-K, the A-tiled threshold
//! and the K-alignment override are `saber-kernels` rules data; graph safety, workspace kinds and
//! launch budgets are `saber-plan` data. Here there is only mechanism — which is why the bit-exact
//! gate can replay a descriptor and get the prod kernel's bytes back.
//!
//! Every fallible route carries a which-path counter ([`LoadCounters`], [`LaunchCounters`]),
//! declared at construction, because the failure being prevented is specific: diagnostics that arm
//! successfully and then never run.
//!
//! # Example
//!
//! ```no_run
//! use saber_exec::{ArgBindings, LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet};
//! use saber_hip::{HipThread, StreamFlags, DEFAULT_HIP_SONAME};
//! use saber_kernels::rules::{resolve, LaunchRules, Op, Shape};
//!
//! let hip = HipThread::spawn(DEFAULT_HIP_SONAME)?;      // NotAvailable on a host without ROCm
//! let load = LoadCounters::new();
//! let plugins = PluginSet::load_dir(&hip, std::path::Path::new("prebuilt/gfx1201"), &load)?;
//! let functions = plugins.functions(&hip)?;
//! let ctx = LaunchCtx::new(&hip, &functions);
//!
//! let rules = LaunchRules::prod_mxfp4()?;
//! let shape = Shape::new(8, 5120, 5120);
//! let desc = resolve(Op::gemm(), shape, &rules)?;
//!
//! let stream = hip.create_stream(StreamFlags::NonBlocking)?;
//! let counters = LaunchCounters::new();
//! let mut launcher = Launcher::new("step", &counters);
//! let bindings = ArgBindings::new(shape); // .with("a", …) for every buffer the descriptor names
//! launcher.launch(&ctx, &desc, &bindings, &stream)?;
//! # Ok::<(), saber_exec::Error>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod block_capture;
pub mod capture;
pub mod counters;
pub mod error;
pub mod forward;
pub mod launch;
pub mod layer;
pub mod loader;
pub mod mixed;
pub mod mrope;
pub mod overlap;
pub mod peer;
pub mod steploop;
pub mod tp2;
pub mod tune;
pub mod vision;

pub use block_capture::{
    bind_block, bind_steps, check_capturable, check_capturable_with, check_persistent_ready,
    BoundStep, CapturedBlock, PeerCapture,
};
pub use capture::{CapturableOp, CapturedBucket, GraphSafe};
pub use counters::{
    CaptureCounters, CaptureCounts, LaunchCounters, LaunchCounts, LoadCounters, LoadCounts,
};
pub use error::{Error, Result};
pub use forward::{
    check_token, i32_bytes, index_table, read_token, vocab_of, weight_window, Head, IndexError,
    Plans, Record, SeqFacts, StepIndex, Window, FIRST_STATE_SLOT,
};
pub use launch::{pack_kernarg, ArgBindings, DispatchRecord, LaunchCtx, LaunchReport, Launcher};
pub use layer::{
    check_weight_layout, state_address, BlockExecutor, BlockInputs, BlockPlan, BlockRun,
    BoundWeight, BufferKind, BufferPlan, Collectives, OpReport, Route, RowSpan, RuntimeParams,
    Source, StateAt, Step,
};
pub use loader::{
    check_kernarg_layout, check_wave_size, implied_struct_align, validate_plugin, FunctionTable,
    LoadedPlugin, PluginFacts, PluginSet,
};
pub use peer::{
    IpcExchange, ProtocolDigest, RankPeer, ResetEpochs, TcpExchange, ThreadPairExchange,
};
pub use steploop::{BucketShape, Stats, StepLoopConfig, StepLoopRow};
pub use tp2::{
    agree_on_collectives, collective_steps, wait_aimed, wait_bounded, wait_bounded_on, wait_polled,
    wait_polled_on, AimMemory, HostWait, RealClock, WaitAim, WaitClock, WaitSeen, WaitWindow,
};
pub use tune::{
    check_index_laws, enumerate_variants, interleaved_order, variant_id_of, IndexLawNote, QkCell,
    ShapeOutcome, SweepConfig, Sweeper, SymbolIndex, TuneOp, Variant, VariantSamples, VariantSet,
};
