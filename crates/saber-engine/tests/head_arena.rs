//! **H46: the shared decode-head arena's host laws** — the reads-before-write audit over every
//! decode head the engine can plan, and what one arena at the widest head refunds.
//!
//! C7 laid every prefill rung over one arena after proving no rung takes a value from the arena's
//! previous run but its explicit zeros. The decode heads get the same question: a head plan reads
//! the two edges the row gather writes into its arena before every run (`plan.inputs`: the hidden
//! state and the residual), and `BlockPlan::stale_reads` — `reads_before_write` minus those — must be
//! EMPTY for every row count, or a head over a shared arena would read another row count's bytes.
//! The engine asserts the same thing at boot (`head_carries_nothing`) and refuses a head that fails.
//!
//! Negative control (structural rule 8): the BODY plan's layer 0 reads its `.zero_resid` (a
//! constant no op writes), so its `stale_reads` is NOT empty — the audit can see a stale read.

use saber_exec::layer::{BlockPlan, Collectives, PlanOptions, RuntimeParams};
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_model::graph::{Bucket, Graph, ModelConfig};
use saber_sched::decode_law::SharedHeadArena;

const LADDER: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
const POW2: [u32; 4] = [1, 2, 4, 8];
const WIDTHS: [u32; 3] = [1, 5, 8];

fn params() -> RuntimeParams {
    RuntimeParams {
        max_ctx: 4096,
        gdn_slots: 65,
    }
}

fn head<'g>(g: &'g Graph, r: &LaunchRules, rows: u32, tp: u32) -> BlockPlan<'g> {
    let opts = PlanOptions::HEAD.rank(0).with_collectives(if tp > 1 {
        Collectives::Real
    } else {
        Collectives::Identity
    });
    BlockPlan::with_options(
        g,
        Bucket::Decode {
            batch: rows,
            verify_width: 1,
        },
        &[],
        params(),
        r,
        &SaberRules::prod_saber().expect("saber"),
        opts,
    )
    .expect("a head plan")
}

/// Every decode head row count a boot can plan: batches x widths (deduped) at a batch set.
fn head_rows(batches: &[u32]) -> Vec<u32> {
    let mut v: Vec<u32> = batches
        .iter()
        .flat_map(|b| WIDTHS.iter().map(move |w| b * w))
        .collect();
    // C2's row bands (16..64), which the ragged schedule plans heads for.
    v.extend([16u32, 24, 32, 40, 48, 56, 64]);
    v.sort_unstable();
    v.dedup();
    v
}

/// THE AUDIT: no decode head at any row count, at TP=2 or TP=1, reads an arena buffer before
/// writing it other than its two gathered inputs.
#[test]
fn no_decode_head_reads_a_stale_arena_byte() {
    let r = LaunchRules::prod_mxfp4().expect("rules");
    for tp in [2u32, 1] {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), tp).expect("graph");
        for rows in head_rows(&LADDER) {
            let p = head(&g, &r, rows, tp);
            assert_eq!(p.inputs.len(), 2, "r{rows} TP={tp}: inputs {:?}", p.inputs);
            let rbw = p.reads_before_write();
            assert!(
                rbw.iter().all(|(b, _)| p.inputs.contains(b)),
                "r{rows} TP={tp}: reads before write {rbw:?} beyond the inputs {:?}",
                p.inputs
            );
            assert!(
                p.stale_reads().is_empty(),
                "r{rows} TP={tp}: {:?}",
                p.stale_reads()
            );
        }
    }
}

/// The control: the body's layer 0 reads the explicit zero residual no op writes.
#[test]
fn the_audit_sees_the_body_s_zero_residual() {
    let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
    let r = LaunchRules::prod_mxfp4().expect("rules");
    let p = BlockPlan::with_options(
        &g,
        Bucket::Decode {
            batch: 1,
            verify_width: 8,
        },
        &[0],
        params(),
        &r,
        &SaberRules::prod_saber().expect("saber"),
        PlanOptions::BODY
            .rank(0)
            .with_collectives(Collectives::Real)
            .with_batch_invariant_attn()
            .with_table_blocks(257),
    )
    .expect("a body plan");
    // T7: `stale_reads()` counts the executor's every-run re-seed, so the control is the plan
    // under the pre-T7 policy (an own arena, not re-seeded).
    let stale =
        p.stale_reads_under(saber_exec::layer::ConstantSeeds::SharedArenaOnly { shared: false });
    assert!(
        stale.iter().any(|(b, _)| b.ends_with(".zero_resid")),
        "the control must see layer 0's zero residual: {stale:?}"
    );
}

/// THE REFUND: the shared arena bills the widest head, the separate law the sum. Printed for the
/// ledger at both batch sets; the ladder's heads cost the shared arena NOTHING beyond `1,2,4,8`'s
/// (the widest head, r64, is in both).
#[test]
fn the_shared_head_arena_refunds_all_but_the_widest() {
    let r = LaunchRules::prod_mxfp4().expect("rules");
    let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
    let law = |batches: &[u32]| {
        SharedHeadArena::of(
            head_rows(batches)
                .into_iter()
                .map(|rows| (rows, head(&g, &r, rows, 2).arena_bytes))
                .collect(),
        )
    };
    let pow2 = law(&POW2);
    let ladder = law(&LADDER);
    let mib = |b: u64| b as f64 / 1_048_576.0;
    eprintln!(
        "head arenas TP=2 (host plan): 1,2,4,8 separate {:.1} MiB -> shared {:.1} (refund {:.1}); \
         1..8 separate {:.1} -> shared {:.1} (refund {:.1})",
        mib(pow2.separate_bytes()),
        mib(pow2.bytes()),
        mib(pow2.refund_bytes()),
        mib(ladder.separate_bytes()),
        mib(ladder.bytes()),
        mib(ladder.refund_bytes()),
    );
    assert_eq!(pow2.widest_rows(), 64);
    assert_eq!(
        ladder.bytes(),
        pow2.bytes(),
        "the widest head is in both sets"
    );
    assert!(ladder.separate_bytes() > pow2.separate_bytes());
    assert!(pow2.refund_bytes() > 0);
}
