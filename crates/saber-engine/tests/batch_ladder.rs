//! **B1: the batch ladder's class census**, on the host and with no card.
//!
//! The ladder (`--engine-decode-batches 1,2,3,4,5,6,7,8`) replays a step of `n` live sequences at
//! batch `n` where the power-of-two set (`1,2,4,8`) pads it into the next power of two. A row's
//! bytes can depend on the bucket only through what the step's M selects: the decode GEMM's split
//! band (`[[split_k_fill]]`: fill 110 at M <= 24, 78 above; `[[decode_bk]]`), the all-reduce's
//! exact / lossy size rule (`quant_min_bytes`, exact at <= 12 rows), and the fp8 head's row band
//! (16-row tile / 32-row band / H33's 48-row band — one k order, gated 0 differing at every band).
//! Attention's split count is the plan's and not the batch's
//! (`the_decode_attention_split_count_is_the_plan_s_not_the_batch_s`), and the tile height `tm`
//! is launch geometry (the fold-class law treats every one-split decode cell as one class).
//!
//! So this census plans every `(batch, width)` of the ladder with the engine's own rules and
//! compares each new bucket's row-local class vector against (a) the bucket the power-of-two set
//! pads its live count into — the numbers `main` serves that step with — and (b) the solo bucket
//! `b1` at the same width. The answer (asserted):
//!
//! * every new bucket at widths 1 and 5, and b5s8 / b6s8 / b7s8, is in the SAME class as the
//!   padded bucket it replaces: the ladder cannot move a row's bytes there;
//! * **b3s8 moves** `gdn.out_proj`, `attn.o_proj` and `mlp.down_proj` from split 2 (b4s8's) to
//!   split 4 (the fill-110 row at M = 24) — the class b1s8 / b2s8 already sum in. A 3-sequence
//!   width-8 step joins the solo side's GEMM class; its reduce stays lossy (24 rows > 12), as
//!   b4s8's is.
//!
//! The negative control (structural rule 8): the census must see b4s8 vs b2s8 as a move (the
//! same three GEMMs), or "no move" above would be vacuous.

use saber_engine::draft_plan::{pick_shape, planned_shapes, widest_arena};
use saber_exec::layer::{BlockPlan, Collectives, PlanOptions, Route, RuntimeParams};
use saber_kernels::dflash::{resolve_head_w8, DflashRules};
use saber_kernels::rules::{fold_class, LaunchRules};
use saber_kernels::saber::SaberRules;
use saber_model::dflash_graph::DflashConfig;
use saber_model::graph::{Bucket, Graph, ModelConfig};

const LADDER: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
const POW2: [u32; 4] = [1, 2, 4, 8];
const WIDTHS: [u32; 3] = [1, 5, 8];

fn graph() -> Graph {
    Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph")
}

/// The engine's decode rules at the defaults (the G5/G6 knobs are prefill-only; set as served).
fn rules() -> LaunchRules {
    let mut base = LaunchRules::prod_mxfp4().expect("rules");
    base.knobs.folded_mt = true;
    base.knobs.atiled_mid = true;
    base.knobs.skinny = true;
    base
}

fn plan<'g>(
    g: &'g Graph,
    r: &LaunchRules,
    batch: u32,
    width: u32,
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
        r,
        &SaberRules::prod_saber().expect("saber"),
        PlanOptions::BODY
            .rank(0)
            .with_collectives(Collectives::Real)
            .with_batch_invariant_attn()
            .with_table_blocks(257),
    )
    .expect("a decode plan")
}

/// The class a row's bytes are summed in, per row-local op: a GEMM's split (the one-split cells
/// are one class whatever the slab or tile, `rules::fold_class`), a reduce's family, and the
/// attention kernel as bound. `(op without the layer prefix, class)`, one layer of each kind.
fn classes(g: &Graph, r: &LaunchRules, batch: u32, width: u32) -> Vec<(String, String)> {
    // Layers 0..4: three GDN layers and the first full-attention layer (interval 4).
    let layers: Vec<u32> = (0..4).collect();
    let p = plan(g, r, batch, width, &layers);
    let mut out: Vec<(String, String)> = Vec::new();
    for s in &p.steps {
        if let Route::Kernel { descs, .. } = &s.route {
            for d in descs {
                let k = d.kernel.as_str();
                let class = if k.starts_with("mxfp4_gemm_decode_") {
                    if fold_class(k).is_some() {
                        "fold".to_string()
                    } else {
                        let ks = k
                            .split('_')
                            .find(|t| t.starts_with("ks"))
                            .unwrap_or("ks?")
                            .to_string();
                        format!("split-{ks}")
                    }
                } else if k.starts_with("saber_ar_") {
                    if k.contains("_exact") {
                        "ar-exact".into()
                    } else {
                        "ar-lossy".into()
                    }
                } else if s.name().contains("attn.decode") {
                    k.to_string()
                } else {
                    continue;
                };
                let op = s
                    .name()
                    .split_once("layer.")
                    .map_or(s.name().to_string(), |(_, rest)| {
                        rest.split_once('.').map_or(rest, |(_, op)| op).to_string()
                    });
                if !out.iter().any(|(o, _)| *o == op) {
                    out.push((op, class));
                }
            }
        }
    }
    out
}

fn moved(a: &[(String, String)], b: &[(String, String)]) -> Vec<String> {
    assert_eq!(
        a.iter().map(|(o, _)| o).collect::<Vec<_>>(),
        b.iter().map(|(o, _)| o).collect::<Vec<_>>(),
        "the two plans list the same row-local ops"
    );
    a.iter()
        .zip(b)
        .filter(|((_, ca), (_, cb))| ca != cb)
        .map(|((o, ca), (_, cb))| format!("{o} ({ca} -> {cb})"))
        .collect()
}

fn pad_into(live: u32) -> u32 {
    POW2.iter()
        .copied()
        .find(|b| *b >= live)
        .expect("a pow2 bucket")
}

/// **THE CLASS LAW (B1).** Every new bucket against the power-of-two bucket its live count was
/// padded into, and against solo; asserted as in the module header.
#[test]
fn the_ladder_moves_a_row_s_class_only_at_b3s8() {
    let g = graph();
    let r = rules();
    let mut table = Vec::new();
    for width in WIDTHS {
        let solo = classes(&g, &r, 1, width);
        for batch in LADDER.iter().copied().filter(|b| !POW2.contains(b)) {
            let new = classes(&g, &r, batch, width);
            let old = classes(&g, &r, pad_into(batch), width);
            let vs_main = moved(&old, &new);
            let vs_solo = moved(&solo, &new);
            table.push(format!(
                "b{batch}s{width} (M {:2}) vs b{}s{width}: {} | vs b1s{width}: {}",
                batch * width,
                pad_into(batch),
                if vs_main.is_empty() {
                    "same class".to_string()
                } else {
                    vs_main.join(", ")
                },
                if vs_solo.is_empty() {
                    "same class".to_string()
                } else {
                    vs_solo.join(", ")
                },
            ));
            if (batch, width) == (3, 8) {
                let mut ops: Vec<&str> = vs_main
                    .iter()
                    .map(|m| m.split(" (").next().unwrap_or(m))
                    .collect();
                ops.sort_unstable();
                assert_eq!(
                    ops,
                    vec!["attn.o_proj", "gdn.out_proj", "mlp.down_proj"],
                    "b3s8 vs b4s8: {vs_main:?}"
                );
                assert!(
                    vs_main
                        .iter()
                        .all(|m| m.ends_with("(split-ks2 -> split-ks4)")),
                    "b3s8 moves split 2 -> 4: {vs_main:?}"
                );
                // ... into b2s8's class, which is the solo class but for the reduce.
                assert!(moved(&classes(&g, &r, 2, 8), &new).is_empty());
            } else {
                assert!(vs_main.is_empty(), "b{batch}s{width}: {vs_main:?}");
            }
        }
    }
    for l in &table {
        eprintln!("{l}");
    }
}

/// The negative control: the census sees a split move where one exists (b2s8 -> b4s8).
#[test]
fn the_census_sees_the_b2s8_to_b4s8_split_move() {
    let g = graph();
    let r = rules();
    let mv = moved(&classes(&g, &r, 2, 8), &classes(&g, &r, 4, 8));
    assert_eq!(mv.len(), 3, "{mv:?}");
    assert!(moved(&classes(&g, &r, 4, 8), &classes(&g, &r, 4, 8)).is_empty());
}

/// **The fp8 head's band per decode row count** (F33/F34/H33 routing; every band gated 0
/// differing against the 16-row tile per slice, so the band never moves a row's bytes — this
/// states which band each new row count takes).
#[test]
fn every_ladder_row_count_takes_a_gated_head_band() {
    let dr = DflashRules::prod_dflash().expect("dflash rules");
    let mut lines = Vec::new();
    for width in WIDTHS {
        for batch in LADDER {
            let m = batch * width;
            let d = resolve_head_w8(m, 124_160, 5120, &dr).expect("the head resolves");
            let band = if m <= 16 {
                "tile16"
            } else if m <= 32 || m > 48 {
                "band32"
            } else {
                "band48"
            };
            assert!(
                d.kernel.contains(match band {
                    "tile16" => "_m16n16",
                    "band32" => "_m32n16",
                    _ => "_m48n16",
                }),
                "M={m}: {}",
                d.kernel
            );
            lines.push(format!("b{batch}s{width} M={m:2} {band} {}", d.kernel));
        }
    }
    for l in &lines {
        eprintln!("{l}");
    }
}

/// **The drafter's shapes cover the ladder** (`planned_shapes` = the boot check's list), the
/// billed drafter arena is unchanged (the widest, batch 8, is in both sets), and every live count
/// picks its own shape with no padding row-group.
#[test]
fn the_drafter_plans_every_ladder_shape_at_the_same_arena() {
    let cfg = DflashConfig::qwen38_dflash2(2, 7);
    let ladder = planned_shapes(&cfg, 16, 4096, &WIDTHS, &LADDER, 8);
    let pow2 = planned_shapes(&cfg, 16, 4096, &WIDTHS, &POW2, 8);
    // The store shape (1, block 8) is b1s8, so 1 + 24 - 1 and 1 + 12 - 1.
    assert_eq!(ladder.len(), 24);
    assert_eq!(pow2.len(), 12);
    assert_eq!(widest_arena(&cfg, &ladder), widest_arena(&cfg, &pow2));
    let pairs: Vec<(u32, u32)> = ladder
        .iter()
        .map(|s| (s.reqs, s.ctx_rows / s.reqs.max(1)))
        .collect();
    for w in WIDTHS {
        for live in 1..=8u32 {
            let i = pick_shape(&pairs, live, w).expect("a shape");
            assert_eq!(pairs[i], (live, w), "live {live} width {w}");
        }
    }
    // The control: the power-of-two list pads 3 into the 4-request shape.
    let ppairs: Vec<(u32, u32)> = pow2
        .iter()
        .map(|s| (s.reqs, s.ctx_rows / s.reqs.max(1)))
        .collect();
    assert_eq!(ppairs[pick_shape(&ppairs, 3, 8).expect("a shape")], (4, 8));
}

/// **What the new body arenas cost**, at every layer of the TP=2 model: the ledger bills one
/// `engine.arena.b{b}s{w}` per pair (the head's `engine.arena.head.r{rows}` is billed per new row
/// count on the card). Printed for the ledger; asserted linear in the rows (each arena is `M`
/// rows of the same regions), so the ladder's body bill is the power-of-two bill times
/// `sum(ladder rows) / sum(pow2 rows)`.
#[test]
fn the_new_body_arenas_are_linear_in_rows() {
    let g = graph();
    let r = rules();
    let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
    let mut per_row = None;
    let mut new_bytes = 0u64;
    let mut old_bytes = 0u64;
    for width in WIDTHS {
        for batch in LADDER {
            let bytes = plan(&g, &r, batch, width, &layers).arena_bytes;
            let rows = u64::from(batch * width);
            let pr = bytes / rows;
            let base = *per_row.get_or_insert(pr);
            assert!(
                pr.abs_diff(base) * 100 <= base,
                "b{batch}s{width}: {pr} B/row against {base}"
            );
            if POW2.contains(&batch) {
                old_bytes += bytes;
            } else {
                new_bytes += bytes;
            }
        }
    }
    eprintln!(
        "body arenas: power-of-two {:.1} MiB, the ladder adds {:.1} MiB ({} B per row)",
        old_bytes as f64 / 1048576.0,
        new_bytes as f64 / 1048576.0,
        per_row.unwrap_or(0)
    );
    assert!(new_bytes > 0);
}
