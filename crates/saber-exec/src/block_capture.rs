//! One decoder block, captured as ONE graph.
//!
//! [`crate::layer`] builds the block and runs it op by op: 49 dispatches at the decode bucket, 56
//! at prefill, each one a `launch_op` on the host. That is the eager path, and it is what the P1
//! gate proved bit-exact against prod's own kernels. This module records the same sequence into a
//! HIP graph and replays it, which is what a serving step will actually do:
//!
//! > Plan → captured HIP graphs per bucket (batch × spec width × ctx band).
//! >
//! > — ARCHITECTURE.md, *4. Execution plane*
//!
//! # What makes a block capturable at all
//!
//! Nothing here re-decides anything. The descriptors are the ones [`BlockPlan`] already resolved,
//! the addresses are the ones [`BlockExecutor::address`] already hands the eager path, and the
//! capture is [`CapturedBucket::capture`] with its three rules unchanged. What this module adds is
//! the *binding*: turning a block's [`Step`]s into [`CapturableOp`]s, with two facts made explicit
//! rather than assumed.
//!
//! * **An op is its whole dispatch sequence.** The paged attention decode is one op and two
//!   kernels; its head declares `n_launches = 2`. Binding the two descriptors separately would
//!   ask each to account for itself and the launcher would refuse both, so they are bound with
//!   [`CapturableOp::sequence`] and the node count still comes out at the op's own budget.
//! * **A block with a peer-visible workspace is not capturable here.** The all-reduce is the only
//!   op in the graph that synchronises with another rank, and at TP=2-rank-0-with-no-peer it is
//!   [`Route::Identity`] — it launches nothing. [`check_capturable`] asserts that: if any
//!   descriptor in the block declared an `IpcPeerVisible` workspace, the capture is refused before
//!   it opens rather than recorded and replayed into a protocol the graph cannot participate in.
//!
//! # What the caller still owns
//!
//! **State.** A replay re-runs the same kernels over the same addresses, and the GDN conv and
//! recurrent caches and the paged KV cache are rewritten *in place*. Replaying twice without
//! restoring them computes the second token of a different sequence, correctly. So the caller
//! re-uploads state between replays; this module does not hide that, because hiding it is exactly
//! how a "graph == eager" result gets claimed for a chain that quietly drifted.
//!
//! **Persistent workspaces.** They survive replay by design — that is what Persistent means — and
//! this block has six: five split-K block counters and the GDN fused update's cross-workgroup
//! barrier. Each is a *protocol* variable that its kernel leaves at zero for the next launch, so a
//! correct replay needs nothing done to them; what a capture must guarantee is that they were
//! allocated and zeroed **before** it opened, because a `hipMemset` recorded inside a capture
//! becomes a node that re-zeroes protocol state on every replay.
//!
//! [`check_persistent_ready`] is that guarantee, asserted by name before any node runs. It is
//! deliberately NOT tested by dirtying a counter and launching: the GDN barrier is waited on until
//! it arrives, so a barrier that starts non-zero never completes and the card is wedged until the
//! process is killed. That was tried on 2026-09-19 and cost fifty minutes of a held GPU lock. The
//! refusal is the property worth proving; the hang proves nothing except that a spin-wait spins.

use std::sync::Arc;

use saber_hip::Stream;
use saber_kernels::rules::WorkspaceKind;
use saber_kernels::LaunchDescriptor;

use crate::capture::{CapturableOp, CapturedBucket, GraphSafe};
use crate::counters::CaptureCounters;
use crate::error::{Error, Result};
use crate::launch::{ArgBindings, LaunchCtx, Launcher};
use crate::layer::{BlockExecutor, BlockInputs, BlockPlan, Route};

/// One launched op of a block, resolved to device addresses.
///
/// Borrows its descriptors from the [`BlockPlan`] and owns its bindings, because the bindings are
/// addresses that only exist once an executor and its inputs do.
pub struct BoundStep<'p> {
    /// The op's name in the graph, for messages and per-op reports.
    pub name: &'p str,
    /// The dispatch sequence, in order. `descs[0]` declares the op's launch count.
    pub descs: &'p [LaunchDescriptor],
    /// The kernarg bindings: every buffer the descriptors name, at its device address.
    pub args: ArgBindings,
}

impl BoundStep<'_> {
    /// Dispatches this op issues, as its head declares them.
    #[must_use]
    pub fn launches(&self) -> u32 {
        self.descs[0].n_launches
    }

    /// The kernel symbols, in order.
    #[must_use]
    pub fn symbols(&self) -> Vec<&str> {
        self.descs.iter().map(|d| d.kernel.as_str()).collect()
    }
}

/// Resolve every launched op of `plan` to device addresses.
///
/// The ops that launch nothing — the identity all-reduce, and any op with no code object whose
/// outputs the caller supplied — are not bound and not returned, exactly as
/// [`BlockExecutor::run`] skips them. The result is therefore the block's *launch* sequence, and
/// its length is the number of graph nodes to expect only after the per-op launch counts are
/// summed.
///
/// # Errors
/// Any [`Source`](crate::Source) the executor cannot resolve: a weight, state or index key the
/// caller did not bind, or an arena offset past the end of the arena.
pub fn bind_block<'p>(
    exec: &BlockExecutor,
    plan: &'p BlockPlan<'_>,
    inputs: &BlockInputs,
) -> Result<Vec<BoundStep<'p>>> {
    bind_steps(exec, &plan.steps, inputs)
}

/// [`bind_block`] over an arbitrary SLICE of a plan's steps.
///
/// One layer of a step, or a step with a family of ops left out: the addresses are the plan's, so a
/// slice replayed in the plan's own order computes what the whole plan computes up to that point.
/// That is what makes a per-layer sub-graph a measurement of the layer *in its real position* —
/// with the previous layer's weights already through the cache — rather than of a layer in
/// isolation.
///
/// # Errors
/// Any [`Source`](crate::Source) the executor cannot resolve, or an empty slice.
pub fn bind_steps<'p>(
    exec: &BlockExecutor,
    steps: &'p [crate::layer::Step<'p>],
    inputs: &BlockInputs,
) -> Result<Vec<BoundStep<'p>>> {
    let mut out = Vec::with_capacity(steps.len());
    for step in steps {
        let Route::Kernel { descs, shape } = &step.route else {
            continue;
        };
        let mut args = ArgBindings::new(*shape);
        for (name, src) in &step.bindings {
            args.bind(name, exec.address(src, inputs)?);
        }
        out.push(BoundStep {
            name: step.name(),
            descs,
            args,
        });
    }
    if out.is_empty() {
        return Err(Error::Refused(
            "this block launches nothing: there is no graph to capture".to_owned(),
        ));
    }
    Ok(out)
}

/// Refuse a block whose descriptors are not capture-legal, before the capture opens.
///
/// One check, and it is the one this module can make on its own: a peer-visible workspace means
/// the op talks to another rank inside the dispatch, which a replayed graph cannot do here. The
/// planner's `graph_safe` / `may_sync` bits are the general answer and they arrive as a
/// [`GraphSafe`] proof; this is the part the execution plane does not have to take on trust.
///
/// # Errors
/// The first op that declares an `IpcPeerVisible` workspace, by name.
pub fn check_capturable(steps: &[BoundStep<'_>]) -> Result<()> {
    check_capturable_with(steps, PeerCapture::Refuse)
}

/// May a peer-visible workspace be captured?
///
/// ws-h9 answered "no" for one rank with no peer, and that is still the default. The answer changes
/// when BOTH ranks capture and replay: the reference's one-shot reduce is an ordinary dispatch whose
/// protocol state is a Persistent per-block sequence counter (`seq_ctrs`) that each launch advances
/// and whose `slot = seq & 1` alternates the double-buffered scratch. A replay advances it exactly
/// as a fresh launch does, so a graph is legal **provided both ranks replay the same graph the same
/// number of times in the same order** — which a two-rank step does by construction, one replay per
/// rank per step.
///
/// What still has to be true, and what the caller is asserting by passing [`PeerCapture::Lockstep`]:
///
/// * every peer-visible region was allocated, exchanged and zeroed **before** the capture opened
///   (`Launcher::prepare_workspaces` + `reset_ipc_state`), because an `hipIpcOpenMemHandle` inside a
///   capture is not a node and an address that moved afterwards is the peer's corruption;
/// * the launch digests were compared on the host before the capture, because a graph cannot
///   rendezvous;
/// * the kernel's wait is bounded (`SABER_AR_SPIN_MAX`), so a rank that falls out of step produces a
///   wrong answer that a byte comparison catches, rather than a hang that holds the card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerCapture {
    /// Refuse any `IpcPeerVisible` workspace. One rank with no peer cannot replay a collective.
    Refuse,
    /// Allow it: both ranks capture and replay in lockstep, and the protocol state was prepared and
    /// agreed before either capture opened.
    Lockstep,
}

/// [`check_capturable`] with the peer-visible decision made explicitly.
///
/// # Errors
/// The first op that declares an `IpcPeerVisible` workspace, when [`PeerCapture::Refuse`].
pub fn check_capturable_with(steps: &[BoundStep<'_>], peer: PeerCapture) -> Result<()> {
    for step in steps {
        for desc in step.descs {
            if peer == PeerCapture::Lockstep {
                continue;
            }
            if let Some(w) = desc
                .workspaces
                .iter()
                .find(|w| w.kind == WorkspaceKind::IpcPeerVisible)
            {
                return Err(Error::NotCapturable {
                    op: step.name.to_owned(),
                    kernel: desc.kernel.clone(),
                    reason: format!(
                        "declares the peer-visible workspace `{}`: a collective cannot be \
                         replayed from a graph on one rank",
                        w.name
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Refuse a capture whose Persistent regions the launcher does not already own.
///
/// Rule 1 of [`crate::capture`] is that every workspace is resolved before the capture opens, and
/// [`CapturedBucket::capture`] upholds it by preparing them itself. This asserts the result rather
/// than trusting the sequence: after preparation, every Persistent workspace every descriptor
/// declares must be a region this launcher owns — allocated once and zeroed once, outside the
/// capture. A region missing here would be allocated and memset **inside** the capture, and the
/// memset would become a graph node that re-zeroes a protocol variable on every replay: the split-K
/// counters would still work (they self-clear), and the GDN fused update's barrier would be reset
/// under the workgroups waiting on it.
///
/// This is the negative control for Persistent state, and it is a *refusal*, not an experiment.
/// Dirtying a barrier and launching wedges the card; naming the unprepared region costs nothing and
/// proves the thing that matters.
///
/// # Errors
/// The first Persistent workspace the launcher does not own, named with its op and kernel.
pub fn check_persistent_ready(launcher: &Launcher, steps: &[BoundStep<'_>]) -> Result<()> {
    for step in steps {
        for desc in step.descs {
            for w in &desc.workspaces {
                if w.kind != WorkspaceKind::Persistent {
                    continue;
                }
                if !launcher.has_persistent(desc, &step.args, &w.name) {
                    return Err(Error::Refused(format!(
                        "op `{}` (`{}`) declares the Persistent workspace `{}`, which this \
                         launcher has not prepared: capturing now would allocate and zero it \
                         INSIDE the capture, making the zeroing a node that runs on every replay",
                        step.name, desc.kernel, w.name
                    )));
                }
            }
        }
    }
    Ok(())
}

/// A decoder block recorded as one graph.
pub struct CapturedBlock {
    bucket: CapturedBucket,
    ops: usize,
}

impl CapturedBlock {
    /// Record every launched op of `plan` into one graph.
    ///
    /// `proof` is the planner's claim that these implementations may be captured; this module
    /// cannot manufacture one (see [`GraphSafe`]). Workspaces are prepared for every descriptor
    /// before the capture opens, the node count is asserted against the sum of the ops' declared
    /// launches, and — one assertion beyond [`CapturedBucket::capture`] — that sum is asserted
    /// against [`BlockPlan::expected_launches`], so the graph is the block's launch budget and not
    /// merely self-consistent.
    ///
    /// # Errors
    /// A binding that cannot be resolved, an op that is not capture-legal, a capture that
    /// produced a node count other than the declared one, or a budget that is not the plan's.
    ///
    /// Nine parameters, and none of them is a knob: four are the block (executor, plan, inputs,
    /// stream), three are the engine's own (context, launcher, counters), and two are the claims
    /// the caller has to make out loud (the label, and the planner's proof). Bundling them into a
    /// struct would hide which of them is a *decision*.
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        exec: &BlockExecutor,
        plan: &BlockPlan<'_>,
        inputs: &BlockInputs,
        stream: &Stream,
        label: impl Into<String>,
        proof: GraphSafe,
        counters: &Arc<CaptureCounters>,
    ) -> Result<CapturedBlock> {
        let steps = bind_block(exec, plan, inputs)?;
        check_capturable(&steps)?;
        // Prepare here, explicitly, so that the Persistent check below runs between preparation
        // and the capture opening. `CapturedBucket::capture` prepares again and every region is
        // then a reuse, which is exactly what the check is asserting.
        for step in &steps {
            for desc in step.descs {
                launcher.prepare_workspaces(ctx, desc, &step.args, stream)?;
            }
        }
        check_persistent_ready(launcher, &steps)?;
        let ops = steps
            .iter()
            .map(|s| CapturableOp::sequence(s.descs, &s.args, proof))
            .collect::<Result<Vec<_>>>()?;
        let bucket = CapturedBucket::capture(ctx, launcher, &ops, stream, label, counters)?;
        let expected = plan.expected_launches();
        if bucket.declared_launches() != expected {
            return Err(Error::LaunchBudget {
                label: bucket.label().to_owned(),
                declared: expected,
                issued: bucket.declared_launches(),
                nodes: bucket.node_count(),
                ops: format!(
                    "{} op(s) bound out of {} planned steps",
                    steps.len(),
                    plan.steps.len()
                ),
            });
        }
        Ok(CapturedBlock {
            bucket,
            ops: steps.len(),
        })
    }

    /// [`Self::capture_steps`] over ops that are already BOUND.
    ///
    /// A two-rank step is not a list of `Step`s: its embedding is resolved from the rank's own
    /// vocabulary window, its head is two descriptors the planner refuses to plan above TP=1, and
    /// its 129 all-reduces replace ops the planner routed to identity. All of those are
    /// `(descriptor, bindings)` pairs rather than plan steps, so this is the entry that takes them —
    /// and it is the only one that can be told a peer-visible workspace is allowed.
    ///
    /// # Errors
    /// An op that is not capture-legal under `peer`, a Persistent region the launcher has not
    /// prepared, or a node count other than the ops' declared sum.
    #[allow(clippy::too_many_arguments)]
    pub fn capture_bound(
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        bound: &[BoundStep<'_>],
        stream: &Stream,
        label: impl Into<String>,
        proof: GraphSafe,
        peer: PeerCapture,
        counters: &Arc<CaptureCounters>,
    ) -> Result<CapturedBlock> {
        check_capturable_with(bound, peer)?;
        for step in bound {
            for desc in step.descs {
                launcher.prepare_workspaces(ctx, desc, &step.args, stream)?;
            }
        }
        check_persistent_ready(launcher, bound)?;
        let ops = bound
            .iter()
            .map(|s| CapturableOp::sequence(s.descs, &s.args, proof))
            .collect::<Result<Vec<_>>>()?;
        let bucket = CapturedBucket::capture(ctx, launcher, &ops, stream, label, counters)?;
        Ok(CapturedBlock {
            bucket,
            ops: bound.len(),
        })
    }

    /// [`Self::capture`] over a SLICE of a plan's steps, with no plan-level budget cross-check.
    ///
    /// The budget is the caller's to assert here, because a slice has no `expected_launches()` of
    /// its own: a per-layer sub-graph's node count is whatever that layer's ops declare, and what
    /// makes the set of them a step is that their counts SUM to the plan's. The node count is still
    /// asserted against the ops' own declarations inside [`CapturedBucket::capture`], and the
    /// Persistent regions are still checked, so the only thing not checked is the thing only the
    /// caller knows.
    ///
    /// # Errors
    /// A binding that cannot be resolved, an op that is not capture-legal, a Persistent region the
    /// launcher has not prepared, or a node count other than the ops' declared sum.
    #[allow(clippy::too_many_arguments)]
    pub fn capture_steps(
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        exec: &BlockExecutor,
        steps: &[crate::layer::Step<'_>],
        inputs: &BlockInputs,
        stream: &Stream,
        label: impl Into<String>,
        proof: GraphSafe,
        counters: &Arc<CaptureCounters>,
    ) -> Result<CapturedBlock> {
        let bound = bind_steps(exec, steps, inputs)?;
        check_capturable(&bound)?;
        for step in &bound {
            for desc in step.descs {
                launcher.prepare_workspaces(ctx, desc, &step.args, stream)?;
            }
        }
        check_persistent_ready(launcher, &bound)?;
        let ops = bound
            .iter()
            .map(|s| CapturableOp::sequence(s.descs, &s.args, proof))
            .collect::<Result<Vec<_>>>()?;
        let bucket = CapturedBucket::capture(ctx, launcher, &ops, stream, label, counters)?;
        Ok(CapturedBlock {
            bucket,
            ops: bound.len(),
        })
    }

    /// Submit the graph. Returns as soon as it is queued; the caller orders completion.
    ///
    /// # Errors
    /// The HIP graph launch failing.
    pub fn replay(&self, ctx: &LaunchCtx<'_>, stream: &Stream) -> Result<()> {
        self.bucket.replay(ctx, stream)
    }

    /// Nodes in the graph — the block's launch count, asserted twice.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.bucket.node_count()
    }

    /// The launch count the ops declared.
    #[must_use]
    pub fn declared_launches(&self) -> u32 {
        self.bucket.declared_launches()
    }

    /// Ops recorded — fewer than the nodes, because one op can be two dispatches.
    #[must_use]
    pub fn ops(&self) -> usize {
        self.ops
    }

    /// The captured bucket underneath.
    #[must_use]
    pub fn bucket(&self) -> &CapturedBucket {
        &self.bucket
    }
}

impl std::fmt::Debug for CapturedBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CapturedBlock({}, {} ops, {} nodes)",
            self.bucket.label(),
            self.ops,
            self.bucket.node_count()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use saber_kernels::rules::{Shape, WorkspaceKind as WsKind, WorkspaceReq};

    fn desc(kernel: &str, launches: u32, ws: Vec<WorkspaceReq>) -> LaunchDescriptor {
        LaunchDescriptor {
            op: "test".into(),
            plugin: "test".into(),
            kernel: kernel.into(),
            grid: [1, 1, 1],
            block: [1, 1, 1],
            shared_bytes: 0,
            kernarg_bytes: 0,
            args: vec![],
            workspaces: ws,
            n_launches: launches,
        }
    }

    /// A peer-visible workspace is refused BEFORE the capture opens, and the message names the op
    /// and the workspace. Without this the all-reduce would be recorded on a rank with no peer.
    #[test]
    fn an_op_with_a_peer_visible_workspace_is_not_capturable() {
        let descs = vec![desc(
            "saber_ar_oneshot_2rank_exact_bf16",
            1,
            vec![WorkspaceReq {
                name: "ar_scratch_peer_0".into(),
                kind: WsKind::IpcPeerVisible,
                bytes: 1024,
                zero_once: false,
            }],
        )];
        let args = ArgBindings::new(Shape::new(1, 1, 1));
        let steps = vec![BoundStep {
            name: "layer.0.allreduce",
            descs: &descs,
            args,
        }];
        match check_capturable(&steps) {
            Err(Error::NotCapturable { op, reason, .. }) => {
                assert_eq!(op, "layer.0.allreduce");
                assert!(reason.contains("ar_scratch_peer_0"), "{reason}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    /// The ordinary case: Scratch and Persistent workspaces are capturable, because the launcher
    /// resolves both before the capture opens.
    #[test]
    fn scratch_and_persistent_workspaces_are_capturable() {
        let descs = vec![desc(
            "mxfp4_gemm_decode_bk128_ks4_tm1_wperm1_nt1",
            1,
            vec![
                WorkspaceReq {
                    name: "decode_partial".into(),
                    kind: WsKind::Scratch,
                    bytes: 4096,
                    zero_once: false,
                },
                WorkspaceReq {
                    name: "decode_counter".into(),
                    kind: WsKind::Persistent,
                    bytes: 256,
                    zero_once: true,
                },
            ],
        )];
        let args = ArgBindings::new(Shape::new(5, 5120, 5120));
        let steps = vec![BoundStep {
            name: "layer.0.mlp.down_proj",
            descs: &descs,
            args,
        }];
        assert!(check_capturable(&steps).is_ok());
        assert_eq!(steps[0].launches(), 1);
        assert_eq!(steps[0].symbols().len(), 1);
    }

    /// The Persistent refusal: a launcher that has prepared nothing refuses by name, and says
    /// what would have happened. No device is needed, which is the point — the property is
    /// checkable without ever launching the kernel whose barrier would hang.
    #[test]
    fn a_persistent_workspace_the_launcher_has_not_prepared_is_refused_by_name() {
        let descs = vec![desc(
            "saber_gdn_fused_update_w4k128v128_ab1",
            1,
            vec![WorkspaceReq {
                name: "gdn_fused_barrier".into(),
                kind: WsKind::Persistent,
                bytes: 4,
                zero_once: true,
            }],
        )];
        let args = ArgBindings::new(Shape::new(0, 0, 0));
        let steps = vec![BoundStep {
            name: "layer.0.gdn.fused_update",
            descs: &descs,
            args,
        }];
        let counters = crate::counters::LaunchCounters::new();
        let launcher = Launcher::new("test", &counters);
        match check_persistent_ready(&launcher, &steps) {
            Err(Error::Refused(m)) => {
                assert!(m.contains("gdn_fused_barrier"), "{m}");
                assert!(m.contains("every replay"), "{m}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        // A descriptor with no Persistent workspace has nothing to prepare, so it passes.
        let plain = vec![desc("add_rms_quant_t256_tiled0", 1, vec![])];
        let ok = vec![BoundStep {
            name: "layer.0.input_layernorm",
            descs: &plain,
            args: ArgBindings::new(Shape::new(5, 0, 5120)),
        }];
        assert!(check_persistent_ready(&launcher, &ok).is_ok());
    }

    /// An op is its whole sequence: two descriptors, one budget, and the head is what declares it.
    #[test]
    fn a_two_dispatch_op_declares_its_count_on_the_head() {
        let descs = vec![
            desc("saber_attn_decode_h256_gqa6_fp8kv_qk8pv8", 2, vec![]),
            desc("saber_attn_splitkv_combine_h256", 1, vec![]),
        ];
        let args = ArgBindings::new(Shape::new(0, 0, 0));
        let step = BoundStep {
            name: "layer.3.attn.decode",
            descs: &descs,
            args,
        };
        assert_eq!(step.launches(), 2, "the head's n_launches is the op's");
        assert_eq!(step.symbols().len(), 2);
    }
}
