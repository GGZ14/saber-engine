//! The prefill bucket class: what P2 binds, and the two obligations it carries.
//!
//! A prefill bucket is not a decode bucket with a bigger `M`. It runs different ops (the
//! query-tiled attention kernel, the three-kernel GDN chunk path, the standalone gated norm),
//! it budgets GDN state per **sequence** rather than per draft token, and it is the only place
//! where the per-edge activation-layout obligation has teeth — because it is the only place
//! where the A-tiled family is reachable at all.

mod common;

use std::collections::BTreeMap;

use common::tuned_cache;
use saber_model::graph::{Graph, ModelConfig, Phase as GraphPhase};
use saber_plan::{
    explain, plan, reference, resolve, DeviceProfile, Manifest, ModelSpec, Phase, PlannedBucket,
    Registry,
};

fn qwen_plan() -> saber_plan::ExecutionPlan {
    let manifest =
        Manifest::from_str(reference::EXAMPLE_MANIFEST_TOML, "example.toml").expect("manifest");
    let device = DeviceProfile::gfx1201_reference();
    let model = ModelSpec::qwen38_27b_dense(manifest.device.tp).expect("model");
    let registry: Registry = reference::reference_registry();
    let resolved = resolve(&manifest, &device, &model, &registry).expect("resolve");
    let cache = tuned_cache(&resolved);
    plan(&manifest, &device, &model, &registry, &cache).expect("plan")
}

fn bucket_of(plan: &saber_plan::ExecutionPlan, label: &str) -> PlannedBucket {
    plan.buckets()
        .iter()
        .find(|b| b.bucket.label() == label)
        .unwrap_or_else(|| panic!("no bucket {label} in {:?}", labels(plan)))
        .clone()
}

fn labels(plan: &saber_plan::ExecutionPlan) -> Vec<String> {
    plan.buckets().iter().map(|b| b.bucket.label()).collect()
}

// =====================================================================================
// The bucket class itself
// =====================================================================================

#[test]
fn the_example_manifest_carries_both_phases() {
    let plan = qwen_plan();
    assert_eq!(
        labels(&plan),
        vec!["b1s5c8192", "b8s5c8192", "p2048c8192", "p8192c8192"]
    );
    for (label, phase, m) in [
        ("b1s5c8192", Phase::Decode, 5),
        ("b8s5c8192", Phase::Decode, 40),
        ("p2048c8192", Phase::Prefill, 2048),
        ("p8192c8192", Phase::Prefill, 8192),
    ] {
        let b = bucket_of(&plan, label);
        assert_eq!(b.bucket.phase, phase, "{label}");
        assert_eq!(b.bucket.m(), m, "{label}");
        assert_eq!(b.m, m, "{label}");
    }
    // A prefill bucket is batch 1 with no speculative width; the manifest parser refuses the
    // decode keys outright, so this is belt and braces on the resolved shape.
    let pre = bucket_of(&plan, "p8192c8192");
    assert_eq!((pre.bucket.batch, pre.bucket.spec_width), (1, 0));
    assert_eq!(pre.bucket.tokens, 8192);
}

/// Obligation (5): a prefill chunk carries **one** recurrent state per sequence.
///
/// The 09-14 Infinity-Cache finding is about *speculative* state — one state per draft token,
/// because every candidate continues from its own and the snapshots are 55–88% of the decode
/// kernel. A prefill chunk has no candidates: it scans the chunk through a single state. Sizing
/// it by `M` would ask for 8192 × 1.5 MiB = 12 GiB per layer and refuse a bucket that in fact
/// uses 1.5 MiB.
#[test]
fn prefill_gdn_state_is_per_sequence_not_per_row() {
    let plan = qwen_plan();
    let pre = bucket_of(&plan, "p8192c8192");
    assert_eq!(
        pre.bucket.gdn_state_slots(),
        1,
        "one sequence, one recurrent state -- NOT one per token"
    );
    assert_eq!(pre.gdn_state.slots, 1);
    // 3.0 MiB per layer per slot, halved by TP=2.
    assert_eq!(pre.gdn_state.working_set_bytes, 3 * 1024 * 1024 / 2);
    assert!(
        pre.gdn_state.ic_fraction() < 0.03,
        "a prefill chunk is nowhere near the cache: {:.3}",
        pre.gdn_state.ic_fraction()
    );

    // The comparison that makes the point: the widest decode bucket touches 40x as much.
    let dec = bucket_of(&plan, "b8s5c8192");
    assert_eq!(dec.gdn_state.slots, 40);
    assert_eq!(
        dec.gdn_state.working_set_bytes,
        40 * pre.gdn_state.working_set_bytes
    );

    // And the arithmetic is by slot count, so a prefill bucket at any chunk size is identical.
    let small = bucket_of(&plan, "p2048c8192");
    assert_eq!(small.gdn_state, pre.gdn_state);
}

// =====================================================================================
// The launch budgets
// =====================================================================================

fn launches_by_op(b: &PlannedBucket) -> BTreeMap<String, u32> {
    let mut m: BTreeMap<String, u32> = BTreeMap::new();
    for op in b.ops.ops() {
        *m.entry(op.op.clone()).or_default() += op.launches_total;
    }
    m
}

fn calls_by_op(b: &PlannedBucket) -> BTreeMap<String, u32> {
    let mut m: BTreeMap<String, u32> = BTreeMap::new();
    for op in b.ops.ops() {
        *m.entry(op.op.clone()).or_default() += op.occurrences;
    }
    m
}

#[test]
fn the_prefill_budget_is_the_prefill_ops_and_nothing_decode() {
    let plan = qwen_plan();
    for label in ["p2048c8192", "p8192c8192"] {
        let b = bucket_of(&plan, label);
        let calls = calls_by_op(&b);
        // What a prefill chunk runs.
        assert_eq!(calls.get("attn_prefill_paged"), Some(&16), "{label}");
        assert_eq!(calls.get("gdn_conv_prep"), Some(&48), "{label}");
        assert_eq!(calls.get("gdn_kkt_solve"), Some(&48), "{label}");
        assert_eq!(calls.get("gdn_chunk_scan"), Some(&48), "{label}");
        assert_eq!(calls.get("gdn_gated_rmsnorm"), Some(&48), "{label}");
        assert_eq!(calls.get("gemm_w4a8"), Some(&305), "{label}");
        assert_eq!(calls.get("allreduce"), Some(&128), "{label}");
        // …and what it does not. Every one of these would be a launch the step never submits.
        for absent in ["attn_decode_paged", "gdn_fused_update", "gdn_conv_update"] {
            assert_eq!(calls.get(absent), None, "{label} must not run {absent}");
        }

        let launches = launches_by_op(&b);
        assert_eq!(launches["attn_prefill_paged"], 16, "one launch per run");
        assert_eq!(
            launches["gdn_conv_prep"] + launches["gdn_kkt_solve"] + launches["gdn_chunk_scan"],
            144,
            "the GDN prefill path is THREE kernels per layer over 48 layers"
        );
        assert_eq!(launches["gdn_gated_rmsnorm"], 48);
        assert_eq!(launches["allreduce"], 128, "2 per layer, all of them lossy");
    }
}

#[test]
fn the_decode_budget_did_not_move() {
    // The prefill rows are additive: a decode bucket resolves exactly what it did before, and
    // the 1010 in tests/explain.rs is the same number.
    let plan = qwen_plan();
    let b = bucket_of(&plan, "b8s5c8192");
    let calls = calls_by_op(&b);
    assert_eq!(calls.get("attn_decode_paged"), Some(&16));
    assert_eq!(calls.get("gdn_conv_update"), Some(&48));
    assert_eq!(calls.get("gemm_w4a8"), Some(&305));
    for absent in [
        "attn_prefill_paged",
        "gdn_conv_prep",
        "gdn_kkt_solve",
        "gdn_chunk_scan",
        "gdn_gated_rmsnorm",
        "gdn_fused_update",
    ] {
        assert_eq!(calls.get(absent), None, "decode must not run {absent}");
    }
}

/// Obligation (3): `explain` prints both phases' budgets rather than one summed total.
#[test]
fn explain_prints_the_prefill_budgets_beside_the_decode_ones() {
    let plan = qwen_plan();
    let text = explain(&plan);
    assert!(
        text.contains("decode         2 bucket(s), launches/step b1s5c8192="),
        "{text}"
    );
    assert!(
        text.contains("prefill        2 bucket(s), launches/step p2048c8192="),
        "{text}"
    );
    assert!(
        text.contains("bucket p8192c8192  prefill  tokens 8192  ctx_band 8192  M=8192  CAPTURED"),
        "{text}"
    );
    // The per-bucket GDN line says "slot(s)", because a prefill bucket has exactly one.
    assert!(text.contains("(1 slot(s) x 1.50 MiB per layer)"), "{text}");
}

// =====================================================================================
// The per-edge layout obligation
// =====================================================================================

/// The two planes agree on how many rows the head runs at.
///
/// `saber_plan::Rows::Sampled` and `saber_model::graph::Bucket::sampled_rows` are the same law
/// written twice, in the two crates that each describe this forward pass. They disagreed until
/// 2026-09-20: the plan said one row per sequence and the graph's block plan gave the head the
/// chunk's M, which is how a p2048 step came to spend 27.6 ms computing logits it discarded. The
/// op COUNTS agreed the whole time, which is why the existing consistency test did not catch it —
/// `lm_head` is one op either way, and the divergence was in its geometry.
#[test]
fn the_two_planes_agree_on_the_heads_row_count() {
    use saber_model::graph::Bucket as GBucket;
    use saber_plan::{Phase, Rows};

    // Prefill: the plan's `batch` is the chunk's sequence count, the graph's is `seqs`.
    for tokens in [1u32, 64, 2048, 8192] {
        let g = GBucket::Prefill { tokens, seqs: 1 };
        assert_eq!(
            Rows::Sampled.at(Phase::Prefill, 1, tokens),
            g.sampled_rows(),
            "prefill {tokens} tokens"
        );
        // And the head's rows are NOT the body's, which is the whole finding.
        if tokens > 1 {
            assert_ne!(g.sampled_rows(), g.m(), "prefill {tokens}");
        }
    }

    // Decode: every verified row is sampled, so both planes return M and the head is planned in
    // the same plan as the body without loss.
    for (batch, width) in [(1u32, 1u32), (1, 5), (8, 5)] {
        let g = GBucket::Decode {
            batch,
            verify_width: width,
        };
        assert_eq!(
            Rows::Sampled.at(Phase::Decode, batch, g.m()),
            g.sampled_rows(),
            "decode b{batch}s{width}"
        );
        assert_eq!(g.sampled_rows(), g.m());
    }

    // The row that carries the law in the plan is `lm_head`, and it is the ONLY one: every other
    // GEMM runs on every row of the chunk. If a second op ever becomes sampled, this says so.
    let plan = qwen_plan();
    let b = bucket_of(&plan, "p8192c8192");
    let head: Vec<&str> = b
        .ops
        .ops()
        .iter()
        .filter(|o| o.shape.num("m") == Some(1))
        .map(|o| o.op_name.as_str())
        .collect();
    assert_eq!(
        head,
        vec!["lm_head"],
        "only the head runs on the sampled rows in a p8192 chunk"
    );
}

/// Two GEMMs read row-major bytes at a prefill M, and the reasons are different — which is the
/// point of having the layout on the edge rather than deriving it from one number:
///
/// * **`gdn.out_proj`**'s A side is written by `gdn_norm_quant`, which has exactly ONE
///   instantiation (store `q += row * N`), so its bytes are row-major at **every** M.
/// * `lm_head` runs on the SAMPLED positions — one per sequence — so its M is 1 in a prefill
///   bucket however long the chunk is, and at M=1 nothing is A-tiled.
///
/// Everything else in the chunk is A-tiled, `attn.o_proj` included: `attn_gate_quant` grew a
/// `tiled0`/`tiled1` pair in the `attn_glue` plugin, so that edge follows M like every other one
/// in an attention layer.
///
/// **This test named `attn.o_proj` until I20**, because the spec did. Both planes now derive the
/// layout from the producer (`saber_kernels::AProducer`), and the graph's own edges say the same
/// thing — see `the_graph_and_the_model_spec_agree_on_every_gemms_a_layout`.
///
/// **G2 (2026-09-23)**: the GDN norm writes the tiled layout (`gdn_norm_tiled`), so `gdn.out_proj`
/// takes the A-tiled family too and `lm_head` (M=1, the sampled position) is the one row-major
/// GEMM left in a prefill bucket.
#[test]
fn the_gdn_out_proj_takes_the_a_tiled_family_at_a_prefill_m() {
    let plan = qwen_plan();
    let b = bucket_of(&plan, "p8192c8192");
    let mut a_tiled = 0usize;
    let mut row_major = Vec::new();
    for op in b.ops.ops().iter().filter(|o| o.op == "gemm_w4a8") {
        let layout = op.shape.get("a_layout").map(ToString::to_string);
        if op.impl_name.contains("atiled") {
            assert_eq!(layout.as_deref(), Some("a_tiled"), "{}", op.op_name);
            a_tiled += 1;
        } else {
            assert_eq!(layout.as_deref(), Some("row_major"), "{}", op.op_name);
            row_major.push(op.op_name.clone());
        }
    }
    assert_eq!(
        row_major,
        vec!["lm_head"],
        "the one GEMM that reads row-major bytes at a prefill M"
    );
    assert!(a_tiled >= 7, "everything else is A-tiled: {a_tiled} rows");

    let out_proj = b
        .ops
        .ops()
        .iter()
        .find(|o| o.op_name == "layer.gdn.out_proj")
        .copied()
        .expect("gdn.out_proj");
    // It reads the A-tiled bytes its producer now writes; the P6 garbage was the A-tiled kernel
    // over ROW-MAJOR bytes, which the producer-derived layout makes unplannable.
    assert_eq!(
        out_proj
            .shape
            .get("a_layout")
            .map(ToString::to_string)
            .as_deref(),
        Some("a_tiled")
    );
    assert!(
        out_proj.impl_name.contains("atiled"),
        "gdn.out_proj bound {}",
        out_proj.impl_name
    );

    // And the row that USED to be declared row-major is A-tiled, which is the other half of the
    // swap: `attn_gate_quant` is a fused producer on this tier.
    let o_proj = b
        .ops
        .ops()
        .iter()
        .find(|o| o.op_name == "layer.attn.o_proj")
        .copied()
        .expect("o_proj");
    assert!(o_proj.impl_name.contains("atiled"), "{}", o_proj.impl_name);
}

/// The negative control: with the wrong producer declared, the planner picks the wrong family.
///
/// Structural rule 8 in spirit — a guard that cannot be observed to fire is not a guard. Give
/// `gdn.out_proj`'s edge a FUSED producer and the same bucket routes it to the A-tiled kernel,
/// which is exactly the bug P0 follow-up (a) describes and exactly what P6 found in the engine.
#[test]
fn negative_control_a_fused_producer_sends_the_gdn_out_proj_a_tiled() {
    let manifest =
        Manifest::from_str(reference::EXAMPLE_MANIFEST_TOML, "example.toml").expect("manifest");
    let device = DeviceProfile::gfx1201_reference();
    let mut model = ModelSpec::qwen38_27b_dense(manifest.device.tp).expect("model");
    let op = model
        .ops
        .iter_mut()
        .find(|o| o.name == "layer.gdn.out_proj")
        .expect("gdn.out_proj row");
    op.a_producer = Some(saber_plan::AProducer::AddRmsQuant);

    let registry = reference::reference_registry();
    let resolved = resolve(&manifest, &device, &model, &registry).expect("resolve");
    let cache = tuned_cache(&resolved);
    let plan = plan(&manifest, &device, &model, &registry, &cache).expect("plan");
    let b = bucket_of(&plan, "p8192c8192");
    let out_proj = b
        .ops
        .ops()
        .iter()
        .find(|o| o.op_name == "layer.gdn.out_proj")
        .copied()
        .expect("gdn.out_proj");
    assert_eq!(
        out_proj
            .shape
            .get("a_layout")
            .map(|v| v.to_string())
            .as_deref(),
        Some("a_tiled"),
        "the collapsed rule calls this edge A-tiled"
    );
    assert!(
        out_proj.impl_name.contains("atiled"),
        "and routes it to {} -- the wrong family, silently",
        out_proj.impl_name
    );
}

/// **Every W4A8 GEMM names the producer that writes its A side**, and the layout is derived from
/// it rather than declared beside it.
///
/// The row that used to declare `RowMajor` was the wrong row for two weeks. A declaration nothing
/// derives cannot be wrong out loud, so there is no declaration any more: the spec names a
/// PRODUCER and `saber_kernels::AProducer` answers what that producer's kernel writes — the same
/// function `saber_exec::layer::Planner::check_producer_layout` asks of the live rule tables.
#[test]
fn every_w4a8_gemm_names_the_producer_that_writes_its_a_side() {
    let model = ModelSpec::qwen38_27b_dense(2).expect("model");
    let mut missing: Vec<&str> = Vec::new();
    let mut named = 0usize;
    for op in &model.ops {
        if op.op != "gemm_w4a8" {
            assert!(
                op.a_producer.is_none(),
                "{}: only a GEMM reads an activation layout",
                op.name
            );
            continue;
        }
        match op.a_producer {
            Some(_) => named += 1,
            None => missing.push(op.name.as_str()),
        }
    }
    assert!(
        missing.is_empty(),
        "these GEMM rows would silently take `FollowsM`: {missing:?}"
    );
    assert_eq!(named, 8, "the eight merged linears plus the head");

    // Under the reference's law `gdn.out_proj` was the one row-major row; with G2's tiled GDN
    // norm (`gdn_norm_tiled`) no GEMM row is row-major at every M.
    let row_major: Vec<&str> = model
        .ops
        .iter()
        .filter(|o| o.op == "gemm_w4a8" && o.a_layout() == saber_plan::ALayout::RowMajor)
        .map(|o| o.name.as_str())
        .collect();
    assert_eq!(row_major, Vec::<&str>::new());
}

/// **Obligation (4), the LAYOUT half.** `ModelSpec`'s `a_layout` selection key and
/// `saber_model::graph`'s producing EDGE are two descriptions of one fact, and they disagreed
/// about `attn.o_proj` and `gdn.out_proj` from the day the key existed until I20 — in opposite
/// directions, on the same two rows, with nothing reading either one.
///
/// Checked at every bucket class the committed manifest plans, because the answer is a function
/// of M: below the band edge every edge is row-major and the two planes cannot disagree.
#[test]
fn the_graph_and_the_model_spec_agree_on_every_gemms_a_layout() {
    use saber_model::graph::{Bucket as GBucket, Layout, Site};

    let model = ModelSpec::qwen38_27b_dense(2).expect("model");
    let graph = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");

    // The HEAD is a second plan (N6): `Graph::ops_for` resolves every op at the bucket it is
    // handed, and `head_bucket` is the one that says `lm_head` runs on the SAMPLED rows. A caller
    // that reads the head's A edge out of the body's resolution reads the body's row count, which
    // is `Rows::Sampled`'s whole point — so the two sites are resolved at the two buckets here,
    // exactly as `saber_exec` plans them.
    for (bucket, phase, batch, m) in [
        (
            GBucket::Decode {
                batch: 1,
                verify_width: 5,
            },
            Phase::Decode,
            1u32,
            5u32,
        ),
        (
            GBucket::Decode {
                batch: 8,
                verify_width: 5,
            },
            Phase::Decode,
            8,
            40,
        ),
        (
            GBucket::Prefill {
                tokens: 2048,
                seqs: 1,
            },
            Phase::Prefill,
            1,
            2048,
        ),
        (
            GBucket::Prefill {
                tokens: 8192,
                seqs: 1,
            },
            Phase::Prefill,
            1,
            8192,
        ),
    ] {
        let mut checked = 0usize;
        let head_bucket = graph.head_bucket(bucket);
        let resolved: Vec<_> = graph
            .ops_for(bucket)
            .into_iter()
            .filter(|r| r.op.site != Site::Head)
            .chain(
                graph
                    .ops_for(head_bucket)
                    .into_iter()
                    .filter(|r| r.op.site == Site::Head),
            )
            .collect();
        for r in resolved {
            if r.op.kernel.id() != "gemm_w4a8" {
                continue;
            }
            let want = match r.input_layouts.first() {
                Some(Layout::ATiledFp8) => "a_tiled",
                Some(Layout::RowMajorFp8) => "row_major",
                other => panic!("{}: A edge carries {other:?}", r.name()),
            };
            // `layer.<n>.gdn.out_proj` in the graph is `layer.gdn.out_proj` in the spec: the spec
            // has one row per GEMM, the graph one op per layer.
            let name = strip_layer_index(r.name());
            let spec = model
                .op(&name)
                .unwrap_or_else(|| panic!("`{name}` has no ModelSpec row"));
            let spec_m = spec.rows.at(phase, batch, m);
            assert_eq!(
                saber_plan::a_layout_key(spec.a_layout(), spec_m),
                want,
                "{} at {bucket:?}: the graph's edge and the spec's key disagree",
                r.name()
            );
            checked += 1;
        }
        assert_eq!(checked, 305, "every W4A8 GEMM in the pass, at {bucket:?}");
    }
}

/// `layer.<n>.<rest>` -> `layer.<rest>`; anything else unchanged.
fn strip_layer_index(name: &str) -> String {
    let Some(rest) = name.strip_prefix("layer.") else {
        return name.to_string();
    };
    match rest.split_once('.') {
        Some((n, tail)) if n.chars().all(|c| c.is_ascii_digit()) => format!("layer.{tail}"),
        _ => name.to_string(),
    }
}

// =====================================================================================
// Two descriptions of one forward pass
// =====================================================================================

/// Obligation (4): `saber_model::graph` and `ModelSpec` must not drift.
///
/// They are two descriptions of the same prefill pass — the graph walks layers and edges, the
/// spec lists routes and counts — and the ops they name have to line up. Where they genuinely
/// differ, the difference is listed here with its reason, and the list is asserted to be
/// exactly consumed: a new disagreement fails, and a disagreement that gets fixed fails too.
#[test]
fn the_graph_and_the_model_spec_agree_on_the_prefill_pass() {
    let model = ModelSpec::qwen38_27b_dense(2).expect("model");
    let graph = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");

    let mut spec_counts: BTreeMap<&str, u32> = BTreeMap::new();
    for op in &model.ops {
        let n = op.calls_in(Phase::Prefill, 1, 8192);
        if n > 0 {
            *spec_counts.entry(op.op.as_str()).or_default() += n;
        }
    }

    let mut graph_counts: BTreeMap<&str, u32> = BTreeMap::new();
    for r in graph.ops_for_bucket(8192, GraphPhase::Prefill) {
        *graph_counts.entry(r.op.kernel.id()).or_default() += 1;
    }

    // The declared differences, each with the reason it exists.
    let expected: BTreeMap<&str, (Option<u32>, Option<u32>, &str)> = BTreeMap::from([
        (
            "allreduce",
            (
                Some(128),
                Some(129),
                "the graph counts the vocab-parallel embedding's reduce; \
             ModelSpec models only the two per decoder layer",
            ),
        ),
        (
            "gdn_gated_rmsnorm",
            (
                Some(48),
                None,
                "the reference's standalone kernel, which Saber ported and prod binds \
             without ever calling",
            ),
        ),
        (
            "gdn_norm_quant",
            (
                None,
                Some(48),
                "the MXFP4 fused norm+quant producer -- the same arithmetic as \
             gdn_gated_rmsnorm in one launch, which is what the tier actually runs",
            ),
        ),
        (
            "add_rms_quant",
            (
                None,
                Some(129),
                "ModelSpec has never modelled the fused quant producers",
            ),
        ),
        ("silu_mul_quant", (None, Some(64), "same")),
        ("embedding", (None, Some(1), "same")),
        ("logits_gather", (None, Some(1), "same")),
    ]);

    let mut ids: Vec<&str> = spec_counts.keys().copied().collect();
    ids.extend(graph_counts.keys().copied());
    ids.sort_unstable();
    ids.dedup();

    let mut agreed = Vec::new();
    let mut unexplained = Vec::new();
    let mut stale: Vec<&str> = expected.keys().copied().collect();
    for id in ids {
        let (s, g) = (spec_counts.get(id).copied(), graph_counts.get(id).copied());
        match expected.get(id) {
            Some((want_s, want_g, why)) => {
                assert_eq!(
                    (s, g),
                    (*want_s, *want_g),
                    "{id}: the declared difference moved ({why})"
                );
                stale.retain(|k| *k != id);
            }
            None => {
                if s == g && s.is_some() {
                    agreed.push(id);
                } else {
                    unexplained.push(format!("{id}: ModelSpec {s:?} vs graph {g:?}"));
                }
            }
        }
    }
    assert!(
        unexplained.is_empty(),
        "the two descriptions of the prefill pass disagree, and nothing says why:\n  {}",
        unexplained.join("\n  ")
    );
    assert!(
        stale.is_empty(),
        "these differences are declared but no longer exist -- delete them: {stale:?}"
    );
    assert_eq!(
        agreed,
        vec![
            "attn_gate_quant",
            "attn_prefill_paged",
            "gdn_chunk_scan",
            "gdn_conv_prep",
            "gdn_kkt_solve",
            // The one that used to be a declared exception at 401 vs 305. `ModelSpec` now
            // models prod's MERGED linears, so the two descriptions agree outright -- and the
            // number they agree on is the one prod's own `linear layers: 304/304` census
            // corroborates (that count plus lm_head, which its requant patch handles apart).
            "gemm_w4a8",
            // The two the `attn_glue` plugin added on both sides at once, alongside
            // `attn_gate_quant` above: `ModelSpec` names them, the graph counts them, and the
            // paged K/V write is now an op instead of a state nothing wrote.
            "kv_append",
            "qk_norm_rope",
        ],
        "the ops both descriptions name must agree on the count"
    );
}

/// The same check on the decode side, so that adding the prefill rows cannot have quietly
/// changed what a decode bucket resolves.
#[test]
fn the_graph_and_the_model_spec_agree_on_the_decode_attention_and_gdn_counts() {
    let model = ModelSpec::qwen38_27b_dense(2).expect("model");
    let graph = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");

    let spec_at = |op: &str, batch: u32, m: u32| -> u32 {
        model
            .ops
            .iter()
            .filter(|o| o.op == op)
            .map(|o| o.calls_in(Phase::Decode, batch, m))
            .sum()
    };
    let graph_at = |op: &str, bucket: saber_model::graph::Bucket| -> u32 {
        graph
            .ops_for(bucket)
            .iter()
            .filter(|r| r.op.kernel.id() == op)
            .count() as u32
    };
    let b1 = saber_model::graph::Bucket::Decode {
        batch: 1,
        verify_width: 5,
    };
    let b8 = saber_model::graph::Bucket::Decode {
        batch: 8,
        verify_width: 5,
    };

    assert_eq!(spec_at("attn_decode_paged", 8, 40), 16);
    assert_eq!(graph_at("attn_decode_paged", b8), 16);
    // Batch 1 fuses, batch 8 takes the pair -- both descriptions, from the same rule.
    assert_eq!(spec_at("gdn_fused_update", 1, 5), 48);
    assert_eq!(graph_at("gdn_fused_update", b1), 48);
    assert_eq!(spec_at("gdn_fused_update", 8, 40), 0);
    assert_eq!(graph_at("gdn_fused_update", b8), 0);
    assert_eq!(spec_at("gdn_conv_update", 8, 40), 48);
    assert_eq!(graph_at("gdn_conv_update", b8), 48);
}

/// The A-tiled threshold is restated in `ModelSpec`; this is the drift test the other two
/// restated constants already have.
#[test]
fn a_tiled_min_m_agrees_with_the_launch_rules() {
    let rules = saber_kernels::rules::LaunchRules::prod_mxfp4().expect("rules");
    assert_eq!(saber_plan::a_tiled_min_m(), rules.knobs.a_tiled_min_m);
    // And the model's own key derivation agrees with the kernel plane's predicate at the edges.
    for m in [1, 512, 513, 2048, 8192] {
        let want = if saber_kernels::rules::a_tiled(m, &rules) {
            "a_tiled"
        } else {
            "row_major"
        };
        assert_eq!(
            saber_plan::a_layout_key(saber_plan::ALayout::FollowsM, m),
            want,
            "M={m}"
        );
        assert_eq!(
            saber_plan::a_layout_key(saber_plan::ALayout::RowMajor, m),
            "row_major",
            "a non-fused producer is row-major at every M"
        );
    }
}

/// The numbers themselves, printed. `cargo test -p saber-plan --test prefill -- --nocapture
/// budget_table` is the human-readable form of what the assertions above pin.
#[test]
fn budget_table() {
    let plan = qwen_plan();
    for b in plan.buckets() {
        println!(
            "{:<12} {:<8} M={:<5} {:>5} launches/step   gdn {} slot(s)",
            b.bucket.label(),
            b.bucket.phase.as_str(),
            b.m,
            b.n_launches,
            b.gdn_state.slots
        );
        for (op, n) in launches_by_op(b) {
            println!("    {op:<22} {n:>5}");
        }
    }
}
