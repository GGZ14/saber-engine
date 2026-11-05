//! **WHAT A DECODE BATCH CHANGES ABOUT THE ARITHMETIC** (ws-p9), on the host and with no card.
//!
//! `saber engine`'s batched, concurrency and mixed-arrival arms compare one sequence between a
//! batch-1 run and a batch-N one, and every red since ws-s8 has been attributed to "the batch/width
//! numeric class the census arm attributes to a layer" — ws-l17's finding, which is about two WIDTH
//! buckets: `b1s1` runs `saber_attn_decode…nw1` and `b1s5` runs `nw2`, and the near-tie flips.
//!
//! That is not what a BATCH moves, and this file is the check. At a fixed verify width the decode
//! attention keeps its symbol and its split count — ws-l15 pinned the latter — and what moves is:
//!
//! * the row-parallel GEMMs' **split-K**, which is a different partition of K and therefore a
//!   different sum (`ks4` at M = 8 against `ks2` at M = 32);
//! * every GEMM's **M tile** (`tm1` against `tm2`);
//! * the GDN recurrent update's **value split** (`vs2` against `vs1`).
//!
//! The negative control is prod's own law with `batch_invariant_attn` off, where the attention's
//! split count moves too — which is the defect ws-l15 closed and which must still be visible here,
//! or this file cannot fail.

use saber_exec::layer::{BlockPlan, PlanOptions, Route, RuntimeParams};
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_model::graph::{Bucket, Graph, ModelConfig};

fn plan_at(g: &Graph, batch: u32, width: u32, opts: PlanOptions) -> BlockPlan<'_> {
    let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
    BlockPlan::with_options(
        g,
        Bucket::Decode {
            batch,
            verify_width: width,
        },
        &layers,
        RuntimeParams {
            max_ctx: 4096,
            gdn_slots: 41,
        },
        &LaunchRules::prod_mxfp4().expect("rules"),
        &SaberRules::prod_saber().expect("saber rules"),
        opts,
    )
    .expect("a decode plan")
}

/// Every descriptor of a plan as `(op, kernel, grid)`.
fn facts(p: &BlockPlan<'_>) -> Vec<(String, String, [u32; 3])> {
    let mut out = Vec::new();
    for step in &p.steps {
        if let Route::Kernel { descs, .. } = &step.route {
            for d in descs {
                out.push((step.op.name.clone(), d.kernel.clone(), d.grid));
            }
        }
    }
    out
}

/// The distinct kernels one op family resolves to in a plan.
fn kernels_of(p: &BlockPlan<'_>, suffix: &str) -> std::collections::BTreeSet<String> {
    facts(p)
        .into_iter()
        .filter(|(op, _, _)| op.ends_with(suffix))
        .map(|(_, k, _)| k)
        .collect()
}

fn graph() -> Graph {
    Graph::qwen38_27b(ModelConfig::qwen38_27b(), 1).expect("graph")
}

/// The engine's own options: ws-l15's pinned split count.
fn body() -> PlanOptions {
    PlanOptions::BODY.with_batch_invariant_attn()
}

#[test]
fn a_batch_does_not_move_the_decode_attention_and_does_move_the_gemms() {
    let g = graph();
    let (a, b) = (plan_at(&g, 1, 8, body()), plan_at(&g, 4, 8, body()));

    // THE ATTENTION IS THE SAME COMPUTATION per row: same instantiation (`nw` is `q_len * gqa`
    // over the rows a warp holds, which is the WIDTH's business and not the batch's) and the same
    // split count (`grid[0]`, pinned by `batch_invariant_attn`). Only the sequence axis moves.
    assert_eq!(
        kernels_of(&a, ".attn.decode"),
        kernels_of(&b, ".attn.decode"),
        "a batch must not change which decode attention kernel a row runs"
    );
    // The SPLIT count is the decode kernel's own `grid[0]`. The op's second descriptor is the
    // split-KV combine, whose `grid[0]` is `num_seqs * q_len * q_heads` — one job per row per
    // head, each folding its own row's `splits` partials in the same order — so it scales with
    // the batch without changing any row's arithmetic, and reading it here would be reading the
    // job count as if it were the partition.
    let splits = |p: &BlockPlan<'_>| -> std::collections::BTreeSet<u32> {
        facts(p)
            .into_iter()
            .filter(|(_, k, _)| k.starts_with("saber_attn_decode_h"))
            .map(|(_, _, g)| g[0])
            .collect()
    };
    assert_eq!(
        splits(&a),
        splits(&b),
        "and must not change how a sequence's keys are partitioned"
    );
    let seq_axis = |p: &BlockPlan<'_>| -> std::collections::BTreeSet<u32> {
        facts(p)
            .into_iter()
            .filter(|(_, k, _)| k.starts_with("saber_attn_decode_h"))
            .map(|(_, _, g)| g[2])
            .collect()
    };
    assert_ne!(
        seq_axis(&a),
        seq_axis(&b),
        "the grid's sequence axis is the one thing that does move, or this test is reading \
         nothing"
    );

    // THE GEMMS DO MOVE, and this is the mechanism behind every batched/concurrency red since
    // ws-s8. `mlp.down_proj` is row-parallel at every one of the 64 layers.
    let down_a = kernels_of(&a, ".mlp.down_proj");
    let down_b = kernels_of(&b, ".mlp.down_proj");
    assert_ne!(
        down_a, down_b,
        "M = 8 and M = 32 resolve the same down_proj: the split-K/M-tile ladder has changed and \
         the ledger's attribution with it"
    );
    let ks = |s: &std::collections::BTreeSet<String>| -> Vec<String> {
        s.iter()
            .filter_map(|k| {
                k.split('_')
                    .find(|p| p.starts_with("ks"))
                    .map(str::to_owned)
            })
            .collect()
    };
    assert_ne!(
        ks(&down_a),
        ks(&down_b),
        "and the difference is the SPLIT-K count, which is a different partition of K and \
         therefore a different sum: {down_a:?} against {down_b:?}"
    );

    // AND THE GDN RECURRENT UPDATE, which is a STATE op and resolves a different symbol too.
    assert_ne!(
        kernels_of(&a, ".gdn.recurrent_update"),
        kernels_of(&b, ".gdn.recurrent_update"),
        "the GDN recurrent update's value split is a batch fact as well"
    );
}

/// At a verify width of 5 the split-K ladder does NOT move between M = 5 and M = 20 — only the M
/// tile does. Recorded because it is what makes the class's reach a function of the WIDTH: the
/// width-8 buckets differ in one more way than the width-5 ones, and a gate that saw the two as
/// the same thing would be asserting more than it measured.
#[test]
fn the_split_k_ladder_moves_at_width_eight_and_not_at_width_five() {
    let g = graph();
    let five = (plan_at(&g, 1, 5, body()), plan_at(&g, 4, 5, body()));
    let ks_of = |p: &BlockPlan<'_>| -> std::collections::BTreeSet<String> {
        kernels_of(p, ".mlp.down_proj")
            .iter()
            .filter_map(|k| {
                k.split('_')
                    .find(|p| p.starts_with("ks"))
                    .map(str::to_owned)
            })
            .collect()
    };
    assert_eq!(
        ks_of(&five.0),
        ks_of(&five.1),
        "M = 5 and M = 20 are in the same split-K band"
    );
    assert_ne!(
        kernels_of(&five.0, ".mlp.down_proj"),
        kernels_of(&five.1, ".mlp.down_proj"),
        "the M tile still moves, which is why the two buckets are not the same kernel"
    );
}

/// THE NEGATIVE CONTROL: prod's own split law. With `batch_invariant_attn` off the decode
/// attention's split count is a function of `kv_heads * num_seqs`, so the same sequence is
/// partitioned differently in a batch — ws-l15's defect, and the reason the flag exists. If this
/// ever stops differing, the assertion above about the attention has stopped meaning anything.
#[test]
fn prods_own_split_law_still_moves_the_attention_with_the_batch() {
    let g = graph();
    let splits = |batch: u32| -> std::collections::BTreeSet<u32> {
        facts(&plan_at(&g, batch, 8, PlanOptions::BODY))
            .into_iter()
            .filter(|(_, k, _)| k.starts_with("saber_attn_decode_h"))
            .map(|(_, _, g)| g[0])
            .collect()
    };
    assert_ne!(splits(1), splits(4));
}

/// And a bucket against ITSELF resolves every op identically — the comparison's own zero.
#[test]
fn a_bucket_against_itself_differs_nowhere() {
    let g = graph();
    assert_eq!(
        facts(&plan_at(&g, 4, 8, body())),
        facts(&plan_at(&g, 4, 8, body()))
    );
}
