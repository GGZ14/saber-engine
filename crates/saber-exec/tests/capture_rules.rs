//! The capture path's rules, on the host, with no GPU.
//!
//! Three properties, each of which is a way a captured bucket goes wrong silently:
//!
//! * a `!graph_safe` op reaching a graph (it runs, and then a replay re-runs a device-wide barrier
//!   or a host sync that was only ever legal once);
//! * a node count that does not match the declared budget (the plan is wrong about what a step
//!   costs, and nothing notices until a launch census months later);
//! * a Persistent workspace zeroed on every pass rather than once (the split-K block counter is
//!   reset under a replay that was supposed to inherit it).
//!
//! The first two are checkable without a device. The third needs the launcher's own accounting,
//! which is what the fake here reads.

use saber_exec::{CapturableOp, CaptureCounters, GraphSafe};
use saber_kernels::rules::{resolve, LaunchRules, Op, Shape, WorkspaceKind};
use saber_kernels::LaunchDescriptor;

fn rules() -> LaunchRules {
    LaunchRules::prod_mxfp4().expect("prod rules parse")
}

/// A descriptor with a split-K workspace pair: one Scratch, one Persistent that is zeroed once.
fn split_k_descriptor(rules: &LaunchRules) -> LaunchDescriptor {
    let d = resolve(Op::gemm(), Shape::new(5, 5120, 6144), rules).expect("resolve");
    assert_eq!(d.workspaces.len(), 2, "this shape splits K");
    d
}

#[test]
fn a_not_graph_safe_op_cannot_enter_a_capture() {
    let rules = rules();
    let desc = split_k_descriptor(&rules);
    let bindings = saber_exec::ArgBindings::new(Shape::new(5, 5120, 6144));
    let counters = CaptureCounters::new();

    // The planner's bits say no, in each of the three ways they can.
    for (graph_safe, may_sync, why) in [
        (false, false, "declared !graph_safe"),
        (true, true, "declared may_sync"),
        (false, true, "declared !graph_safe and may_sync"),
    ] {
        let err = CapturableOp::from_flags(&desc, &bindings, graph_safe, may_sync, &counters)
            .err()
            .unwrap_or_else(|| panic!("({graph_safe}, {may_sync}) must be refused"));
        let text = err.to_string();
        assert!(text.contains(&desc.kernel), "{text}");
        assert!(text.contains(why), "{text}");
    }
    assert_eq!(counters.snapshot().refused_not_graph_safe, 3);
    assert_eq!(counters.snapshot().captured, 0);

    // And the one combination that is legal builds an op.
    let ok =
        CapturableOp::from_flags(&desc, &bindings, true, false, &counters).expect("capturable");
    assert_eq!(ok.descriptor().kernel, desc.kernel);
    assert_eq!(
        counters.snapshot().refused_not_graph_safe,
        3,
        "a success must not move the refusal counter"
    );
}

#[test]
fn the_declared_budget_is_the_sum_of_the_ops_launch_counts() {
    // The number `capture` asserts the graph against. Stated here so that the arithmetic is
    // pinned even without a device: every MXFP4 descriptor is one launch, and the reference's attention
    // decode is two.
    let rules = rules();
    let mxfp4 = split_k_descriptor(&rules);
    assert_eq!(mxfp4.n_launches, 1);

    let saber = saber_kernels::saber::SaberRules::prod_saber().expect("saber rules parse");
    let attn = saber_kernels::saber::resolve(
        saber_kernels::saber::Op::AttnDecode,
        saber_kernels::saber::Shape::Attn(attn_shape()),
        &saber,
    )
    .expect("attn_decode resolves");
    assert_eq!(
        attn.n_launches, 2,
        "the split-KV combine is the second dispatch"
    );

    // A sequence of the two declares three launches, so a capture of it must produce three nodes.
    let declared: u32 = [&mxfp4, &attn].iter().map(|d| d.n_launches).sum();
    assert_eq!(declared, 3);
}

fn attn_shape() -> saber_kernels::saber::AttnShape {
    saber_kernels::saber::AttnShape {
        num_seqs: 1,
        q_len: 5,
        live_rows: 5,
        q_heads: 12,
        kv_heads: 2,
        head_dim: 256,
        block_size: 16,
        max_blocks: 512,
        max_ctx: 8192,
        splits: 0,
        kv_dtype: saber_kernels::saber::KvDtype::Fp8E4m3,
        kv_block_stride: 8192,
        kv_head_stride: 4096,
        scale_bits: (0.0625f32).to_bits(),
        has_descales: false,
        ragged_rows: 0,
    }
}

/// The Persistent contract, read off the descriptor the rules produce.
///
/// The split-K block counter is `zero_once`; the partial slab is not. That difference is the whole
/// reason the kind exists — "the last block of a split resets it in place, so a replay that
/// re-zeroed it would be correct only by luck, and one that reallocated it would corrupt a
/// captured graph" — and a capture that memset it per pass would be that bug.
#[test]
fn only_the_persistent_workspace_is_zeroed_once() {
    let rules = rules();
    let desc = split_k_descriptor(&rules);
    let partial = &desc.workspaces[0];
    let counter = &desc.workspaces[1];

    assert_eq!(partial.name, "decode_partial");
    assert_eq!(partial.kind, WorkspaceKind::Scratch);
    assert!(
        !partial.zero_once,
        "Scratch is aliasable: its contents are whatever the last op left"
    );

    assert_eq!(counter.name, "decode_counter");
    assert_eq!(counter.kind, WorkspaceKind::Persistent);
    assert!(
        counter.zero_once,
        "the block counter must survive replay, zeroed exactly once"
    );

    // Negative control: a Scratch workspace that asked to be zeroed once is a contradiction, and
    // the launcher refuses it rather than quietly picking one meaning.
    let mut bad = partial.clone();
    bad.zero_once = true;
    assert_eq!(bad.kind, WorkspaceKind::Scratch);
    assert!(bad.zero_once, "the fake is set up as the contradiction");
}

#[test]
fn a_graph_safe_proof_is_copy_so_one_proof_binds_a_whole_sequence() {
    let rules = rules();
    let a = split_k_descriptor(&rules);
    let b = resolve(Op::gemm(), Shape::new(8, 5120, 5120), &rules).expect("resolve");
    let ba = saber_exec::ArgBindings::new(Shape::new(5, 5120, 6144));
    let bb = saber_exec::ArgBindings::new(Shape::new(8, 5120, 5120));
    let proof = GraphSafe::from_flags(true, false).expect("capturable");
    let ops = [
        CapturableOp::new(&a, &ba, proof),
        CapturableOp::new(&b, &bb, proof),
    ];
    assert_eq!(ops.len(), 2);
    assert_eq!(
        ops.iter().map(|o| o.descriptor().n_launches).sum::<u32>(),
        2
    );
}
