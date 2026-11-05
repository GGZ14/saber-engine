//! Which-path counters.
//!
//! `CLAUDE.md` rule 5 and `ARCHITECTURE.md` § *Observability is a build-time contract*: a route
//! with a fallback branch declares a counter **at construction** or it does not merge. The failure
//! this prevents is specific and has happened twice on this box — diagnostics that arm
//! successfully and then never run — so every counter here is incremented on the path itself, not
//! from a wrapper that a later refactor can forget to call.
//!
//! The counters are plain atomics behind an `Arc`: the step loop reads them from the HIP thread,
//! `/metrics` will read them from a tokio worker, and neither may block the other.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Increment helper: relaxed is right here — these are diagnostics, never a synchronisation
/// mechanism, and an exact interleaving between two threads means nothing.
#[inline]
fn bump(c: &AtomicU64) {
    c.fetch_add(1, Ordering::Relaxed);
}

/// Plugin-load outcomes. Exactly the four the workstream declares, in the order a load tries them.
#[derive(Debug, Default)]
pub struct LoadCounters {
    loaded: AtomicU64,
    refused_kernarg: AtomicU64,
    refused_wave: AtomicU64,
    refused_metadata: AtomicU64,
}

/// A readable copy of [`LoadCounters`], for printing and for tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoadCounts {
    /// Plugins that passed all three validations and whose module and symbols are resolved.
    pub loaded: u64,
    /// Refused by the kernarg-layout rules (`saber_abi::check_kernarg` plus the structural rules).
    pub refused_kernarg: u64,
    /// Refused because a declared wave size disagreed with the code object, or the plugin mixed
    /// wave sizes across its kernels (a translation-unit property, not a per-kernel one).
    pub refused_wave: u64,
    /// Refused by the manifest-vs-code-object metadata check.
    pub refused_metadata: u64,
}

impl LoadCounters {
    /// A fresh counter set, shareable.
    #[must_use]
    pub fn new() -> Arc<LoadCounters> {
        Arc::new(LoadCounters::default())
    }

    pub(crate) fn hit_loaded(&self) {
        bump(&self.loaded);
    }
    pub(crate) fn hit_refused_kernarg(&self) {
        bump(&self.refused_kernarg);
    }
    pub(crate) fn hit_refused_wave(&self) {
        bump(&self.refused_wave);
    }
    pub(crate) fn hit_refused_metadata(&self) {
        bump(&self.refused_metadata);
    }

    /// Read every counter.
    #[must_use]
    pub fn snapshot(&self) -> LoadCounts {
        LoadCounts {
            loaded: self.loaded.load(Ordering::Relaxed),
            refused_kernarg: self.refused_kernarg.load(Ordering::Relaxed),
            refused_wave: self.refused_wave.load(Ordering::Relaxed),
            refused_metadata: self.refused_metadata.load(Ordering::Relaxed),
        }
    }
}

impl LoadCounts {
    /// `loaded=… refused_kernarg=… …`, one line, deterministic.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "loaded={} refused_kernarg={} refused_wave={} refused_metadata={}",
            self.loaded, self.refused_kernarg, self.refused_wave, self.refused_metadata
        )
    }
}

/// Launch-path outcomes, including every workspace decision.
///
/// `workspace_*_new` vs `workspace_*_reused` is the pair that makes the aliasing policy auditable:
/// a Scratch region that is never reused, or a Persistent region that is allocated twice for the
/// same `(plugin, op, shape)`, both show up here rather than as a slow step nobody can explain.
#[derive(Debug, Default)]
pub struct LaunchCounters {
    launched: AtomicU64,
    scratch_new: AtomicU64,
    scratch_grown: AtomicU64,
    scratch_reused: AtomicU64,
    persistent_new: AtomicU64,
    persistent_reused: AtomicU64,
    refused_workspace_kind: AtomicU64,
    refused_unbound_buffer: AtomicU64,
    refused_kernarg_drift: AtomicU64,
    refused_launch_count: AtomicU64,
    dispatched_unbudgeted: AtomicU64,
    ipc_exported: AtomicU64,
    ipc_opened: AtomicU64,
    ipc_local: AtomicU64,
    ipc_reused: AtomicU64,
    ipc_reset: AtomicU64,
    peer_access_enabled: AtomicU64,
    peer_access_already: AtomicU64,
    ar_fused_nq: AtomicU64,
    ar_plain: AtomicU64,
    ar_fused_gemm: AtomicU64,
}

/// A readable copy of [`LaunchCounters`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct LaunchCounts {
    pub launched: u64,
    pub scratch_new: u64,
    pub scratch_grown: u64,
    pub scratch_reused: u64,
    pub persistent_new: u64,
    pub persistent_reused: u64,
    pub refused_workspace_kind: u64,
    pub refused_unbound_buffer: u64,
    pub refused_kernarg_drift: u64,
    pub refused_launch_count: u64,
    pub dispatched_unbudgeted: u64,
    /// Peer-visible regions allocated here and exported to the other rank (fine-grained).
    pub ipc_exported: u64,
    /// Peer regions opened from the other rank's handle.
    pub ipc_opened: u64,
    /// Peer-visible regions allocated here and NOT exported -- protocol state the peer never
    /// reads, such as the per-block sequence counters.
    pub ipc_local: u64,
    /// Peer-visible regions served from the table rather than allocated again.
    pub ipc_reused: u64,
    /// Calls to `reset_ipc_state`; both ranks must make the same number.
    pub ipc_reset: u64,
    /// `hipDeviceEnablePeerAccess` returned success.
    pub peer_access_enabled: u64,
    /// ... or reported the link was already up, which is success too.
    pub peer_access_already: u64,
    /// All-reduces dispatched with the `_nq` epilogue: ONE kernel doing the reduce, the residual
    /// add, the RMS norm and the per-token quant. The fused branch of the planner's route.
    pub ar_fused_nq: u64,
    /// All-reduces dispatched as a bare reduce (exact or rotated), the norm following as its own
    /// dispatch. The unfused branch of the same route.
    pub ar_plain: u64,
    /// All-reduces dispatched INSIDE the split decode GEMM before them, with the `_nq` epilogue
    /// (fusion-map row 3, ws-h38): ONE kernel doing the GEMM, the reduce, the residual add, the
    /// RMS norm and the per-token quant. The third branch of the same route.
    pub ar_fused_gemm: u64,
}

impl LaunchCounters {
    /// A fresh counter set, shareable.
    #[must_use]
    pub fn new() -> Arc<LaunchCounters> {
        Arc::new(LaunchCounters::default())
    }

    pub(crate) fn hit_launched(&self) {
        bump(&self.launched);
    }
    pub(crate) fn hit_scratch_new(&self) {
        bump(&self.scratch_new);
    }
    pub(crate) fn hit_scratch_grown(&self) {
        bump(&self.scratch_grown);
    }
    pub(crate) fn hit_scratch_reused(&self) {
        bump(&self.scratch_reused);
    }
    pub(crate) fn hit_persistent_new(&self) {
        bump(&self.persistent_new);
    }
    pub(crate) fn hit_persistent_reused(&self) {
        bump(&self.persistent_reused);
    }
    pub(crate) fn hit_refused_workspace_kind(&self) {
        bump(&self.refused_workspace_kind);
    }
    pub(crate) fn hit_refused_unbound_buffer(&self) {
        bump(&self.refused_unbound_buffer);
    }
    pub(crate) fn hit_refused_kernarg_drift(&self) {
        bump(&self.refused_kernarg_drift);
    }
    pub(crate) fn hit_refused_launch_count(&self) {
        bump(&self.refused_launch_count);
    }
    pub(crate) fn hit_dispatched_unbudgeted(&self) {
        bump(&self.dispatched_unbudgeted);
    }
    pub(crate) fn hit_ipc_exported(&self) {
        bump(&self.ipc_exported);
    }
    pub(crate) fn hit_ipc_opened(&self) {
        bump(&self.ipc_opened);
    }
    pub(crate) fn hit_ipc_local(&self) {
        bump(&self.ipc_local);
    }
    pub(crate) fn hit_ipc_reused(&self) {
        bump(&self.ipc_reused);
    }
    pub(crate) fn hit_ipc_reset(&self) {
        bump(&self.ipc_reset);
    }
    pub(crate) fn hit_peer_access_enabled(&self) {
        bump(&self.peer_access_enabled);
    }
    pub(crate) fn hit_ar_fused_nq(&self) {
        bump(&self.ar_fused_nq);
    }
    pub(crate) fn hit_ar_plain(&self) {
        bump(&self.ar_plain);
    }
    pub(crate) fn hit_ar_fused_gemm(&self) {
        bump(&self.ar_fused_gemm);
    }
    pub(crate) fn hit_peer_access_already(&self) {
        bump(&self.peer_access_already);
    }

    /// Read every counter.
    #[must_use]
    pub fn snapshot(&self) -> LaunchCounts {
        LaunchCounts {
            launched: self.launched.load(Ordering::Relaxed),
            scratch_new: self.scratch_new.load(Ordering::Relaxed),
            scratch_grown: self.scratch_grown.load(Ordering::Relaxed),
            scratch_reused: self.scratch_reused.load(Ordering::Relaxed),
            persistent_new: self.persistent_new.load(Ordering::Relaxed),
            persistent_reused: self.persistent_reused.load(Ordering::Relaxed),
            refused_workspace_kind: self.refused_workspace_kind.load(Ordering::Relaxed),
            refused_unbound_buffer: self.refused_unbound_buffer.load(Ordering::Relaxed),
            refused_kernarg_drift: self.refused_kernarg_drift.load(Ordering::Relaxed),
            refused_launch_count: self.refused_launch_count.load(Ordering::Relaxed),
            dispatched_unbudgeted: self.dispatched_unbudgeted.load(Ordering::Relaxed),
            ipc_exported: self.ipc_exported.load(Ordering::Relaxed),
            ipc_opened: self.ipc_opened.load(Ordering::Relaxed),
            ipc_local: self.ipc_local.load(Ordering::Relaxed),
            ipc_reused: self.ipc_reused.load(Ordering::Relaxed),
            ipc_reset: self.ipc_reset.load(Ordering::Relaxed),
            peer_access_enabled: self.peer_access_enabled.load(Ordering::Relaxed),
            peer_access_already: self.peer_access_already.load(Ordering::Relaxed),
            ar_fused_nq: self.ar_fused_nq.load(Ordering::Relaxed),
            ar_plain: self.ar_plain.load(Ordering::Relaxed),
            ar_fused_gemm: self.ar_fused_gemm.load(Ordering::Relaxed),
        }
    }
}

impl LaunchCounts {
    /// One deterministic line, for a report or a log.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "launched={} scratch(new={} grown={} reused={}) persistent(new={} reused={}) \
             refused(ws_kind={} unbound_buffer={} kernarg_drift={} launch_count={}) \
             dispatched_unbudgeted={} \
             ipc(exported={} opened={} local={} reused={} reset={}) \
             peer_access(enabled={} already={}) ar(fused_nq={} plain={} fused_gemm={})",
            self.launched,
            self.scratch_new,
            self.scratch_grown,
            self.scratch_reused,
            self.persistent_new,
            self.persistent_reused,
            self.refused_workspace_kind,
            self.refused_unbound_buffer,
            self.refused_kernarg_drift,
            self.refused_launch_count,
            self.dispatched_unbudgeted,
            self.ipc_exported,
            self.ipc_opened,
            self.ipc_local,
            self.ipc_reused,
            self.ipc_reset,
            self.peer_access_enabled,
            self.peer_access_already,
            self.ar_fused_nq,
            self.ar_plain,
            self.ar_fused_gemm,
        )
    }
}

/// Capture-path outcomes.
///
/// `captured` and `replayed` are the two halves of the execution plane's own census: a bucket is
/// captured once and replayed per step, so a `replayed` that is not vastly larger than `captured`
/// means something is re-capturing and the whole point has been lost.
#[derive(Debug, Default)]
pub struct CaptureCounters {
    captured: AtomicU64,
    replayed: AtomicU64,
    refused_not_graph_safe: AtomicU64,
    refused_launch_count: AtomicU64,
}

/// A readable copy of [`CaptureCounters`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CaptureCounts {
    /// Buckets recorded and instantiated.
    pub captured: u64,
    /// Graph submissions.
    pub replayed: u64,
    /// Ops refused because the planner's bits say they may not be captured.
    pub refused_not_graph_safe: u64,
    /// Captures refused because the node count did not match the declared budget.
    pub refused_launch_count: u64,
}

impl CaptureCounters {
    /// A fresh counter set, shareable.
    #[must_use]
    pub fn new() -> Arc<CaptureCounters> {
        Arc::new(CaptureCounters::default())
    }

    pub(crate) fn hit_captured(&self) {
        bump(&self.captured);
    }
    pub(crate) fn hit_replayed(&self) {
        bump(&self.replayed);
    }
    pub(crate) fn hit_refused_not_graph_safe(&self) {
        bump(&self.refused_not_graph_safe);
    }
    pub(crate) fn hit_refused_launch_count(&self) {
        bump(&self.refused_launch_count);
    }

    /// Read every counter.
    #[must_use]
    pub fn snapshot(&self) -> CaptureCounts {
        CaptureCounts {
            captured: self.captured.load(Ordering::Relaxed),
            replayed: self.replayed.load(Ordering::Relaxed),
            refused_not_graph_safe: self.refused_not_graph_safe.load(Ordering::Relaxed),
            refused_launch_count: self.refused_launch_count.load(Ordering::Relaxed),
        }
    }
}

impl CaptureCounts {
    /// One deterministic line.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "captured={} replayed={} refused(not_graph_safe={} launch_count={})",
            self.captured, self.replayed, self.refused_not_graph_safe, self.refused_launch_count
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_start_at_zero_and_count_up() {
        let c = LoadCounters::new();
        assert_eq!(c.snapshot(), LoadCounts::default());
        c.hit_loaded();
        c.hit_refused_wave();
        let s = c.snapshot();
        assert_eq!(s.loaded, 1);
        assert_eq!(s.refused_wave, 1);
        assert_eq!(s.refused_kernarg, 0);
        assert_eq!(
            s.line(),
            "loaded=1 refused_kernarg=0 refused_wave=1 refused_metadata=0"
        );
    }

    #[test]
    fn launch_counters_line_is_deterministic() {
        let c = LaunchCounters::new();
        c.hit_launched();
        c.hit_scratch_new();
        assert_eq!(c.snapshot().launched, 1);
        assert_eq!(c.snapshot().line(), c.snapshot().line());
    }
}
