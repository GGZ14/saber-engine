//! Constraint semantics, and the three outcomes of a selection: one, several, none.

use saber_plan::{
    reference, Constraint, Implementation, PlanError, Registry, ShapeKeys, WorkspaceKind,
};

fn gemm_shape(m: i64) -> ShapeKeys {
    // `a_layout` is a required key on every GEMM row: a query that does not declare what its
    // A-side edge carries gets NO kernel rather than whichever row happens to be listed first.
    // That is the reference's own behaviour for `exact` on the all-reduce, applied to the per-edge
    // layout obligation. `follows M` is what a fused producer writes.
    gemm_shape_with(m, if m >= 513 { "a_tiled" } else { "row_major" })
}

fn gemm_shape_with(m: i64, a_layout: &str) -> ShapeKeys {
    ShapeKeys::new()
        .with("m", m)
        .with("n", 5120i64)
        .with("k", 5120i64)
        .with("w_dtype", "mxfp4")
        .with("a_dtype", "int8")
        .with("a_layout", a_layout)
}

#[test]
fn constraint_ops_have_saber_semantics() {
    let shape = ShapeKeys::new()
        .with("m", 40i64)
        .with("k", 5120i64)
        .with("dtype", "bf16");

    assert!(Constraint::eq("m", 40).check(&shape).is_ok());
    assert!(Constraint::le("m", 40).check(&shape).is_ok());
    assert!(Constraint::le("m", 39).check(&shape).is_err());
    assert!(Constraint::ge("m", 40).check(&shape).is_ok());
    assert!(Constraint::ge("m", 41).check(&shape).is_err());
    assert!(Constraint::div("k", 128).check(&shape).is_ok());
    assert!(Constraint::div("k", 3).check(&shape).is_err());
    assert!(Constraint::is_in("dtype", "bf16 fp16")
        .check(&shape)
        .is_ok());
    assert!(Constraint::is_in("dtype", "fp16 fp32")
        .check(&shape)
        .is_err());

    // A divisor of zero never holds, rather than dividing by zero.
    assert!(Constraint::div("m", 0).check(&shape).is_err());
}

#[test]
fn refusals_say_what_was_wanted() {
    let shape = ShapeKeys::new()
        .with("m", 200i64)
        .with("k", 100i64)
        .with("dtype", "int4");
    assert_eq!(
        Constraint::le("m", 128).check(&shape).unwrap_err(),
        "m=200, needs <= 128"
    );
    assert_eq!(
        Constraint::eq("m", 40).check(&shape).unwrap_err(),
        "m=200, needs == 40"
    );
    assert_eq!(
        Constraint::ge("m", 2048).check(&shape).unwrap_err(),
        "m=200, needs >= 2048"
    );
    assert_eq!(
        Constraint::div("k", 128).check(&shape).unwrap_err(),
        "k=100, needs divisible by 128"
    );
    assert_eq!(
        Constraint::is_in("dtype", "mxfp4 nvfp4")
            .check(&shape)
            .unwrap_err(),
        "dtype=int4, needs one of: mxfp4 nvfp4"
    );
    assert_eq!(
        Constraint::eq("head_dim", 256).check(&shape).unwrap_err(),
        "head_dim not given"
    );
    assert_eq!(
        Constraint::le("dtype", 8).check(&shape).unwrap_err(),
        "dtype=int4 is not a number"
    );
    assert_eq!(
        Constraint::is_in("m", "16 32").check(&shape).unwrap_err(),
        "m=200, needs one of: 16 32"
    );
}

#[test]
fn one_match_is_returned_alone() {
    let registry = reference::reference_registry();
    // M=2048 behind a FUSED producer is A-tiled, which leaves exactly the one row that reads
    // that layout. The row-major rows refuse it on `a_layout`, not on M.
    let names = |m: i64, layout: &str| -> Vec<String> {
        registry
            .select("gemm_w4a8", &gemm_shape_with(m, layout))
            .unwrap()
            .iter()
            .map(|id| registry.get(*id).unwrap().name.clone())
            .collect()
    };
    assert_eq!(names(2048, "a_tiled"), vec!["gemm_w4a8_mxfp4_atiled"]);
    assert_eq!(names(8192, "a_tiled"), vec!["gemm_w4a8_mxfp4_atiled"]);

    // The SAME M behind a non-fused producer takes the row-major family instead. This is the
    // per-edge obligation: the layout decides, not the row count.
    assert_eq!(
        names(2048, "row_major"),
        vec!["gemm_w4a8_mxfp4_wperm", "gemm_w4a8_mxfp4_folded"]
    );
    // …and above the WPERM band, exactly one row-major row is left.
    assert_eq!(names(8192, "row_major"), vec!["gemm_w4a8_mxfp4_folded"]);
}

#[test]
fn several_matches_are_all_returned_for_the_autotune_stage() {
    let registry = reference::reference_registry();
    let hits = registry.select("gemm_w4a8", &gemm_shape(40)).unwrap();
    let names: Vec<&str> = hits
        .iter()
        .map(|id| registry.get(*id).unwrap().name.as_str())
        .collect();
    assert_eq!(
        names,
        vec![
            "gemm_w4a8_mxfp4_decode",
            "gemm_w4a8_mxfp4_wperm",
            "gemm_w4a8_mxfp4_folded"
        ]
    );
}

#[test]
fn zero_matches_names_the_closest_failing_constraint_per_candidate() {
    let registry = reference::reference_registry();
    // K=100 divides by neither 128 nor 64, and M sits above every band but the A-tiled one.
    let shape = ShapeKeys::new()
        .with("m", 99_999i64)
        .with("n", 5120i64)
        .with("k", 100i64)
        .with("w_dtype", "mxfp4")
        .with("a_dtype", "int8")
        .with("a_layout", "a_tiled");
    let err = registry.select("gemm_w4a8", &shape).unwrap_err();
    let PlanError::NoImplementation(miss) = err else {
        panic!("expected NoImplementation")
    };
    assert_eq!(miss.op, "gemm_w4a8");
    assert_eq!(miss.misses.len(), 4);
    // Closest first: A-tiled cleared m and n before failing on k; the row-major rows failed on
    // the layout or on m, depending on which predicate they reach first.
    assert_eq!(miss.misses[0].impl_name, "gemm_w4a8_mxfp4_atiled");
    assert_eq!(miss.misses[0].reason, "k=100, needs divisible by 64");
    assert_eq!(miss.misses[0].satisfied, 2);
    assert!(
        miss.misses[1..]
            .iter()
            .all(|m| m.reason == "m=99999, needs <= 4096"
                || m.reason == "m=99999, needs <= 128"
                || m.reason == "a_layout=a_tiled, needs one of: row_major"
                || m.reason == "k=100, needs divisible by 64"),
        "{:?}",
        miss.misses
    );

    let rendered = miss.to_string();
    assert!(
        rendered.contains("no implementation for op `gemm_w4a8`"),
        "{rendered}"
    );
    assert!(
        rendered.contains("k=100, needs divisible by 64"),
        "{rendered}"
    );
}

#[test]
fn an_op_nothing_implements_says_so() {
    let registry = reference::reference_registry();
    // An op nothing in this build carries. `attn_decode_paged` used to serve here and no longer
    // can: saber is the second plugin now, so the fixture implements it.
    let err = registry
        .select("moe_expert_gemm", &gemm_shape(40))
        .unwrap_err();
    let PlanError::NoImplementation(miss) = err else {
        panic!("expected NoImplementation")
    };
    assert!(miss.misses.is_empty());
    assert!(
        miss.to_string()
            .contains("no implementation in this registry has op `moe_expert_gemm`"),
        "{miss}"
    );
}

#[test]
fn explain_reports_every_candidate_with_its_verdict() {
    let registry = reference::reference_registry();
    let trace = registry.explain("gemm_w4a8", &gemm_shape(40));
    assert_eq!(trace.op, "gemm_w4a8");
    assert_eq!(trace.candidates.len(), 4);
    assert_eq!(
        trace.chosen(),
        vec![
            "gemm_w4a8_mxfp4_decode",
            "gemm_w4a8_mxfp4_wperm",
            "gemm_w4a8_mxfp4_folded"
        ]
    );
    let atiled = trace
        .candidates
        .iter()
        .find(|c| c.impl_name.ends_with("atiled"))
        .unwrap();
    assert!(!atiled.ok);
    assert_eq!(atiled.reason, "m=40, needs >= 2048");
    // The folded row is the one that makes a row-major edge routable at any M.
    let folded = trace
        .candidates
        .iter()
        .find(|c| c.impl_name.ends_with("folded"))
        .unwrap();
    assert!(folded.ok);
    let decode = trace
        .candidates
        .iter()
        .find(|c| c.impl_name.ends_with("decode"))
        .unwrap();
    assert!(decode.ok);
    assert_eq!(decode.reason, "ok");
    assert_eq!(decode.satisfied, 6);
}

#[test]
fn constraints_never_mention_tiles() {
    // The split that prevents the macro explosion: constraints discriminate ops, tile choice is a
    // variant id from a measurement. If a tile parameter ever appears as a constraint key, the
    // registry has started doing the autotuner's job.
    let forbidden = [
        "split_k", "tm", "wperm", "nt", "bk", "tile", "block", "ks", "npw",
    ];
    for imp in reference::reference_registry().implementations() {
        for c in &imp.constraints {
            assert!(
                !forbidden.contains(&c.key.as_str()),
                "{} constrains `{}`, which is a tile parameter",
                imp.name,
                c.key
            );
        }
    }
}

#[test]
fn workspace_kinds_and_launch_counts_come_from_the_registry() {
    let registry = reference::reference_registry();
    let (_, decode) = registry.by_name("gemm_w4a8_mxfp4_decode").unwrap();
    let kinds: Vec<WorkspaceKind> = decode.workspaces.iter().map(|w| w.kind).collect();
    assert_eq!(
        kinds,
        vec![WorkspaceKind::Scratch, WorkspaceKind::Persistent]
    );
    // split-K partials are DKS * M * N * 4 B, which is why the rule flips per shape and per M.
    let shape = gemm_shape(40);
    assert_eq!(
        decode.workspaces[0].sizing.bytes(&shape, 4),
        4 * 40 * 5120 * 4
    );
    assert_eq!(decode.workspaces[1].sizing.bytes(&shape, 4), 4096);
    // One launch, plus the reduce when K is split.
    assert_eq!(decode.launches.n_launches(1), 1);
    assert_eq!(decode.launches.n_launches(4), 2);

    let (_, ar) = registry.by_name("ar_oneshot_2rank_exact").unwrap();
    assert_eq!(ar.workspaces[0].kind, WorkspaceKind::IpcPeerVisible);
}

#[test]
fn fingerprint_changes_with_any_registry_change() {
    let base = reference::reference_registry();
    let mut changed = base.clone();
    let extra = Implementation::new("gemm_w4a8_experimental", "mxfp4", "gemm_w4a8")
        .constrain(Constraint::le("m", 16));
    changed.insert(extra);
    assert_ne!(base.fingerprint(), changed.fingerprint());
    assert_eq!(
        base.fingerprint(),
        reference::reference_registry().fingerprint()
    );
}

#[test]
fn ops_lists_what_the_registry_covers() {
    assert_eq!(
        reference::reference_registry().ops(),
        vec![
            "allreduce",
            "attn_decode_paged",
            "attn_gate_quant",
            "attn_prefill_paged",
            "gdn_chunk_scan",
            "gdn_conv_prep",
            "gdn_conv_update",
            "gdn_fused_update",
            "gdn_gated_rmsnorm",
            "gdn_kkt_solve",
            "gemm_w4a8",
            "kv_append",
            "qk_norm_rope"
        ]
    );
}

#[test]
fn an_empty_registry_refuses_everything() {
    let registry = Registry::new();
    assert!(registry.select("gemm_w4a8", &gemm_shape(40)).is_err());
    assert!(registry.ops().is_empty());
}
