//! **WHERE ROW `r` OF AN EDGE IS** (ws-l18), on the host and with no card.
//!
//! `bytes / rows` is the truth for the residual stream and an assumption everywhere else
//! (`docs/GATES.md` blind spot 17). This file states the exception and measures it: `l{n}.gdn.conv`
//! is ONE edge holding **q, k and v** as three row-major blocks laid end to end, because the conv
//! kernel writes three tensors — `q[(row · Hg + h) · D]`, `k` the same, `v[(row · H + h) · D]` —
//! and the planner gives them one buffer.
//!
//! At Qwen3.8-27B's GDN geometry (16 key heads, 48 value heads, 128 dims, bf16) that edge is
//! 20,480 bytes a row and q is 4,096 of them, so **`bytes / rows` of the conv edge is five rows of
//! q and none of k or v**. An op-deep probe slicing by it and reading "row 0" reads q rows 0..=4 —
//! and since every other decode edge it reads IS row-major, it compares one row there and five
//! here. That is how ws-p9's attribution came to name `gdn.conv_update` as the first op to write
//! differing bytes between two buckets: the conv is the first op in a layer whose "row 0" span
//! covers rows the rest of the probe never compared (and the first that mixes rows at all).
//!
//! The NEGATIVE CONTROL is the naive law itself, asserted to be wrong in exactly that way: if
//! `bytes / rows` ever became a row of the conv edge this file would stop having anything to say,
//! and the assertion that it spans five q rows is what makes that visible.

use saber_exec::layer::{BlockPlan, PlanOptions, RuntimeParams};
use saber_exec::RowSpan;
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_model::graph::{Bucket, Graph, ModelConfig};

/// Qwen3.8-27B's GDN geometry at TP=1, in bytes per row.
const Q_ROW: u64 = 16 * 128 * 2;
const K_ROW: u64 = 16 * 128 * 2;
const V_ROW: u64 = 48 * 128 * 2;
const CONV_ROW: u64 = Q_ROW + K_ROW + V_ROW;

fn plan_at(g: &Graph, batch: u32, width: u32) -> BlockPlan<'_> {
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
        PlanOptions::BODY.with_batch_invariant_attn(),
    )
    .expect("a decode plan")
}

fn graph() -> Graph {
    Graph::qwen38_27b(ModelConfig::qwen38_27b(), 1).expect("graph")
}

#[test]
fn the_conv_edge_is_three_blocks_and_the_naive_slice_is_five_rows_of_q() {
    let g = graph();
    for (batch, width) in [(1u32, 8u32), (4, 8)] {
        let p = plan_at(&g, batch, width);
        let rows = u64::from(batch * width);
        let b = p.locate("l30.gdn.conv").expect("the conv edge is planned");
        assert_eq!(b.bytes, rows * CONV_ROW, "b{batch}s{width}");

        let RowSpan::Blocks { blocks } = &b.rows else {
            panic!(
                "b{batch}s{width}: the conv edge is q|k|v, not one block: {:?}",
                b.rows
            );
        };
        assert_eq!(
            blocks,
            &vec![
                (0, Q_ROW),
                (rows * Q_ROW, K_ROW),
                (rows * (Q_ROW + K_ROW), V_ROW)
            ],
            "b{batch}s{width}"
        );
        assert_eq!(b.rows.bytes(), CONV_ROW);
        assert_eq!(b.rows.rows(b.bytes), rows);

        // THE NEGATIVE CONTROL. The law this replaces reads `[r · (bytes / rows), bytes / rows)`.
        // At r = 0 that is the right START and the wrong LENGTH: five rows of q.
        let naive = b.bytes / rows;
        assert_eq!(naive, CONV_ROW, "b{batch}s{width}");
        assert_eq!(
            naive / Q_ROW,
            5,
            "b{batch}s{width}: the naive slice spans five q rows, which is why it is not a row"
        );
    }
}

#[test]
fn the_same_logical_row_is_the_same_span_in_both_buckets() {
    let g = graph();
    let (a, b) = (plan_at(&g, 1, 8), plan_at(&g, 4, 8));
    let (ca, cb) = (
        a.locate("l30.gdn.conv").expect("edge"),
        b.locate("l30.gdn.conv").expect("edge"),
    );
    // Slot 0 of the batch holds rows 0..8 in both buckets, and under the plan's own law each of
    // those rows is the same q, k and v of the same token on both sides. Under the naive law only
    // the LENGTHS agree: the q offsets do (q starts the buffer), but rows 1..7 read into k and v
    // in one bucket and stay inside q in the other.
    for r in 0..8u64 {
        let (sa, sb) = (ca.rows.spans(r), cb.rows.spans(r));
        assert_eq!(sa.len(), 3);
        assert_eq!(sa[0], sb[0], "row {r}: q");
        assert_eq!(
            sa[0],
            (r * Q_ROW, Q_ROW),
            "row {r}: q is where the kernel wrote it"
        );
        // k and v sit at different BASES in the two buckets — which is the whole reason the span
        // has to come from the plan and not from the byte count.
        assert_eq!(sa[1], (8 * Q_ROW + r * K_ROW, K_ROW), "row {r}: k at b1s8");
        assert_eq!(sb[1], (32 * Q_ROW + r * K_ROW, K_ROW), "row {r}: k at b4s8");
        assert_eq!(
            sa[2],
            (8 * (Q_ROW + K_ROW) + r * V_ROW, V_ROW),
            "row {r}: v at b1s8"
        );
        assert_eq!(
            sb[2],
            (32 * (Q_ROW + K_ROW) + r * V_ROW, V_ROW),
            "row {r}: v at b4s8"
        );
    }
}

#[test]
fn the_residual_stream_is_one_block_and_a_row_of_it_is_a_row() {
    let g = graph();
    let p = plan_at(&g, 4, 8);
    let b = p.locate("l31.resid.a").expect("the residual stream");
    let hidden = u64::from(g.config.hidden_size) * 2;
    assert_eq!(b.rows, RowSpan::Contiguous { bytes: hidden });
    assert_eq!(b.rows.rows(b.bytes), 32);
    assert_eq!(b.rows.spans(24), vec![(24 * hidden, hidden)]);
}
