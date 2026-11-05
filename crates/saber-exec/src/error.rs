//! Errors. Every refusal names the plugin, the kernel and the field it refused on.
//!
//! "A kernarg layout the host and the kernel disagree about is memory corruption, not an error"
//! (ARCHITECTURE.md, *Kernel plane*). The consequence for this crate is that a refusal has to be
//! actionable without a debugger: which plugin, which kernel symbol, which field, what the two
//! sides each said.

use std::path::PathBuf;

/// What went wrong.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The HIP runtime refused, or is not there at all.
    #[error("hip: {0}")]
    Hip(#[from] saber_hip::Error),

    /// The manifest / launch-rule layer refused.
    #[error("kernels: {0}")]
    Kernels(#[from] saber_kernels::Error),

    /// Reading a manifest or a code object off disk.
    #[error("{path}: {source}")]
    Io {
        /// The file that could not be read.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },

    /// A plugin failed one of the three load-time validations.
    #[error("{plugin}: kernel `{kernel}`: field `{field}`: {detail}")]
    Validate {
        /// Plugin (kernel-set stem) that was being loaded.
        plugin: String,
        /// The kernel symbol the disagreement is about, or `*` for a plugin-wide property.
        kernel: String,
        /// The field the two sides disagree about.
        field: String,
        /// What each side said.
        detail: String,
    },

    /// The launch could not be built from the descriptor.
    #[error("launch `{kernel}`: {detail}")]
    Launch {
        /// The kernel symbol the descriptor names.
        kernel: String,
        /// What was wrong.
        detail: String,
    },

    /// A workspace kind this build does not implement. Not a fallback: a refusal, with a counter.
    #[error("unsupported workspace kind `{kind}` for workspace `{name}` of `{kernel}`: {detail}")]
    UnsupportedWorkspaceKind {
        /// The kind, as the manifest spells it.
        kind: String,
        /// The workspace's name in the descriptor.
        name: String,
        /// The kernel that asked for it.
        kernel: String,
        /// Why it cannot be served yet.
        detail: String,
    },

    /// The plugin directory has no such plugin, or the plugin has no such kernel.
    #[error("{0}")]
    NotFound(String),

    /// The two ranks of a collective disagree about something the protocol depends on.
    ///
    /// Raised **before** either launches. A one-shot all-reduce whose ranks disagree does not
    /// produce a wrong answer, it produces a kernel that never retires, so this refusal is the
    /// whole point of exchanging a digest at all.
    #[error(
        "rank {rank} and rank {peer_rank} disagree about `{what}`: {mine:#018x} vs {theirs:#018x}. \
         Nothing was launched -- a one-shot all-reduce whose ranks disagree hangs rather than \
         answering wrongly."
    )]
    PeerDisagreement {
        /// What the two sides were comparing.
        what: String,
        /// This rank.
        rank: u32,
        /// This rank's digest.
        mine: u64,
        /// The other rank.
        peer_rank: u32,
        /// The other rank's digest.
        theirs: u64,
    },

    /// Device work did not complete inside the deadline the caller set.
    #[error("{what} did not complete within {seconds} s; giving up rather than blocking forever")]
    Timeout {
        /// What was being waited on.
        what: String,
        /// The deadline, in seconds.
        seconds: u64,
    },

    /// An op the planner says may not be recorded into a graph.
    ///
    /// Refused at the point an op enters a capture, not at replay: "a `!graph_safe` op cannot be
    /// bound into a captured bucket because the type will not permit it".
    #[error("op `{op}` (`{kernel}`) cannot be captured: {reason}")]
    NotCapturable {
        /// The op name.
        op: String,
        /// The kernel symbol behind it.
        kernel: String,
        /// Which of the planner's bits refused it.
        reason: String,
    },

    /// The capture produced a different number of nodes than the ops declared.
    ///
    /// Rule 7: the launch count per bucket is declared statically and **asserted at capture**. A
    /// plan that is wrong about what it costs is wrong now, not in a launch census later.
    #[error(
        "capture `{label}`: the ops declare {declared} launch(es) but the graph has {nodes} node(s) \
         and {issued} dispatch(es) were issued. Ops: [{ops}]"
    )]
    LaunchBudget {
        /// The bucket label.
        label: String,
        /// Sum of the ops' `n_launches`.
        declared: u32,
        /// Dispatches the launcher reported issuing.
        issued: u32,
        /// Nodes `hipGraphGetNodes` counted.
        nodes: usize,
        /// The sequence, for the message.
        ops: String,
    },

    /// The gate or the bench refused to run.
    #[error("{0}")]
    Refused(String),
}

/// This crate's result alias.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Error {
        Error::Io {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn validate(
        plugin: &str,
        kernel: &str,
        field: &str,
        detail: impl Into<String>,
    ) -> Error {
        Error::Validate {
            plugin: plugin.to_owned(),
            kernel: kernel.to_owned(),
            field: field.to_owned(),
            detail: detail.into(),
        }
    }

    pub(crate) fn launch(kernel: &str, detail: impl Into<String>) -> Error {
        Error::Launch {
            kernel: kernel.to_owned(),
            detail: detail.into(),
        }
    }

    /// The field name a [`Error::Validate`] refused on, for tests that assert *which* check fired.
    #[must_use]
    pub fn validate_field(&self) -> Option<&str> {
        match self {
            Error::Validate { field, .. } => Some(field),
            _ => None,
        }
    }
}
