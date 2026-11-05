//! **H46: bucket independence as a law** — the b1s8 step does not know which other decode buckets
//! the boot planned.
//!
//! B1 read the engine gate's acceptance (b1s8 only) move 4.302 -> 4.306 when batch 7 was planned
//! beside `1,2,4,8`. Everything the b1s8 step binds on the host is asserted here to be a function of
//! its own shape and NOT of the bucket set: the body plan (every buffer's offset and size, every
//! step's kernel symbols), the head plan, the drafter's b1 shape (its index on the wire, its arena
//! layout) and the drafter's ONE shared arena (sized to the widest, batch 8, in every set that
//! holds 8). The card half of the law is `h46_hold.sh`'s attribution arms (LEDGER H46).
//!
//! Negative control (structural rule 8): the same comparison between b1s8 and b2s8 must see the
//! plans differ, or "identical" below would be vacuous.

use saber_engine::draft_plan::{planned_shapes, widest_arena, DraftPlan};
use saber_exec::layer::{BlockPlan, Collectives, PlanOptions, Route, RuntimeParams};
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_model::dflash_graph::DflashConfig;
use saber_model::graph::{Bucket, Graph, ModelConfig};

const SETS: [&[u32]; 4] = [
    &[1, 2, 4, 8],
    &[1, 2, 3, 4, 5, 6, 7, 8],
    &[1, 2, 3, 4, 5, 6, 8],
    &[1, 2, 4, 7, 8],
];
const WIDTHS: [u32; 3] = [1, 5, 8];

fn plan<'g>(
    g: &'g Graph,
    batch: u32,
    width: u32,
    opts: PlanOptions,
    layers: &[u32],
) -> BlockPlan<'g> {
    BlockPlan::with_options(
        g,
        Bucket::Decode {
            batch,
            verify_width: width,
        },
        layers,
        RuntimeParams {
            max_ctx: 4096,
            gdn_slots: 65,
        },
        &LaunchRules::prod_mxfp4().expect("rules"),
        &SaberRules::prod_saber().expect("saber"),
        opts,
    )
    .expect("a plan")
}

/// A plan's bytes as the executor sees them: arena size, every buffer's (name, offset, bytes), and
/// every step's (name, kernel symbols).
fn fingerprint(p: &BlockPlan<'_>) -> String {
    let mut s = format!("arena {}\n", p.arena_bytes);
    for b in &p.buffers {
        s.push_str(&format!("{} {} {}\n", b.name, b.offset, b.bytes));
    }
    for st in &p.steps {
        let k = match &st.route {
            Route::Kernel { descs, .. } => descs
                .iter()
                .map(|d| d.kernel.clone())
                .collect::<Vec<_>>()
                .join(","),
            other => format!("{other:?}").chars().take(40).collect(),
        };
        s.push_str(&format!("{} {k}\n", st.name()));
    }
    s
}

fn body() -> PlanOptions {
    PlanOptions::BODY
        .rank(0)
        .with_collectives(Collectives::Real)
        .with_batch_invariant_attn()
        .with_table_blocks(257)
}

/// The b1s8 body and head plans are functions of their own shape: planned alongside any bucket
/// set, they are the same bytes. The control: b2s8's are not b1s8's.
#[test]
fn the_b1s8_plans_do_not_know_the_bucket_set() {
    let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
    let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
    let head = PlanOptions::HEAD
        .rank(0)
        .with_collectives(Collectives::Real);
    let mut seen: Option<(String, String)> = None;
    for set in SETS {
        // Plan every bucket of the set first, the way the boot does, then b1s8.
        for b in set {
            for w in WIDTHS {
                let _ = plan(&g, *b, w, body(), &layers[..1]);
            }
        }
        let fb = fingerprint(&plan(&g, 1, 8, body(), &layers));
        let fh = fingerprint(&plan(&g, 8, 1, head, &[]));
        match &seen {
            None => seen = Some((fb, fh)),
            Some((b0, h0)) => {
                assert_eq!(&fb, b0, "the b1s8 body plan moved with the set {set:?}");
                assert_eq!(&fh, h0, "the r8 head plan moved with the set {set:?}");
            }
        }
    }
    let (b0, _) = seen.expect("planned");
    assert_ne!(
        fingerprint(&plan(&g, 2, 8, body(), &layers)),
        b0,
        "the control: b2s8 must differ from b1s8"
    );
}

/// The drafter: the b1 shape's wire index and arena layout, and the one shared arena's size, are
/// the same in every bucket set (the widest shape, batch 8, sizes the arena in each).
#[test]
fn the_drafter_s_b1_shape_does_not_know_the_bucket_set() {
    let cfg = DflashConfig::qwen38_dflash2(2, 7);
    let mut seen: Option<(usize, String, u64)> = None;
    for set in SETS {
        let shapes = planned_shapes(&cfg, 16, 4096, &WIDTHS, set, 8);
        let ix = shapes
            .iter()
            .position(|s| s.reqs == 1 && s.ctx_rows == 8)
            .expect("the b1 shape");
        let layout = format!("{:?}", DraftPlan::arena_for(&cfg, shapes[ix]));
        let widest = widest_arena(&cfg, &shapes);
        match &seen {
            None => seen = Some((ix, layout, widest)),
            Some((i0, l0, w0)) => {
                assert_eq!(ix, *i0, "the b1 shape's wire index moved with {set:?}");
                assert_eq!(&layout, l0, "the b1 arena layout moved with {set:?}");
                assert_eq!(widest, *w0, "the shared drafter arena moved with {set:?}");
            }
        }
    }
    // Control: the b2 shape's layout is not b1's.
    let shapes = planned_shapes(&cfg, 16, 4096, &WIDTHS, &[1, 2], 8);
    let b2 = shapes
        .iter()
        .find(|s| s.reqs == 2 && s.ctx_rows == 16)
        .expect("b2");
    assert_ne!(
        format!("{:?}", DraftPlan::arena_for(&cfg, *b2)),
        seen.expect("seen").1
    );
}
