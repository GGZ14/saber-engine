//! **C9: the fused mixed forward's host laws**, over the engine's own plans and with no card.
//!
//! The fused step is assembled from the prefill plan at the mixed rung and the SERVED decode plan
//! at the bucket (`saber_exec::mixed`). These tests pin, for every decode bucket the engine serves
//! (batch 1..8 x widths 1 / 5 / 8, with and without the `all` decode fusions) and every rung a
//! fused step can take:
//!
//! * the launch order ([`fused_schedule`]): every prefill step once, in order; every decode-side
//!   op once, right after its own layer's chunk group; the shared rope / append taken once;
//! * the offset rebinding ([`rebase`]): every arena binding of a decode-side op lands in the
//!   prefill arena at the decode rows (`c..c + D`) of a buffer the SAME layer's chunk group also
//!   binds — the same edge — and nowhere else;
//! * the disjoint ranges: the chunk group writes rows `0..c` and the decode group rows `c..c + D`;
//! * negative controls: a decode range past the rung is refused, and fusion-map row 5 (`gdn.norm`
//!   fused into `fused_update`, h42) is refused rather than run twice.

use std::collections::{BTreeMap, BTreeSet};

use saber_engine::small_chunk::{rules_for_rows, small_chunk_rules, SmallChunkClass};
use saber_exec::layer::{BlockPlan, Collectives, PlanOptions, Route, RuntimeParams, Source};
use saber_exec::mixed::{
    fused_schedule, is_chunk_side, is_decode_side, rebase, rebind, FusedBinding, FusedOp, Rebound,
};
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_model::graph::{Bucket, Graph, ModelConfig};
use saber_sched::mixed_law::FusedRows;

const LADDER: [u32; 11] = [96, 128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048];

fn graph() -> Graph {
    Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph")
}

fn body() -> PlanOptions {
    PlanOptions::BODY
        .rank(0)
        .with_collectives(Collectives::Real)
        .with_batch_invariant_attn()
        .with_table_blocks(257)
}

struct Rules {
    base: LaunchRules,
    small: (u32, LaunchRules),
}

fn engine_rules() -> Rules {
    let mut base = LaunchRules::prod_mxfp4().expect("rules");
    base.knobs.folded_mt = true;
    base.knobs.atiled_mid = true;
    base.knobs.skinny = true;
    let small = small_chunk_rules(&base, 128, SmallChunkClass::Exact).expect("small");
    Rules {
        base,
        small: (128, small),
    }
}

fn plan<'g>(g: &'g Graph, b: Bucket, r: &LaunchRules, o: PlanOptions) -> BlockPlan<'g> {
    let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
    BlockPlan::with_options(
        g,
        b,
        &layers,
        RuntimeParams {
            max_ctx: 4096,
            gdn_slots: 65,
        },
        r,
        &SaberRules::prod_saber().expect("saber"),
        o,
    )
    .expect("a plan")
}

fn layer_of(op: &str) -> &str {
    let rest = op.strip_prefix("layer.").expect("a layer op");
    &op[..6 + rest.find('.').expect("dot") + 1]
}

/// The whole law for one (rung, decode plan, chunk rows) triple.
fn check(prefill: &BlockPlan<'_>, decode: &BlockPlan<'_>, c: u32) {
    let d_rows = decode.bucket.m();
    let sched = fused_schedule(prefill, decode).expect("a schedule");
    // Every prefill step once, in order.
    let chunk: Vec<usize> = sched
        .iter()
        .filter_map(|o| match o {
            FusedOp::Chunk(i) => Some(*i),
            FusedOp::Decode(_) => None,
        })
        .collect();
    assert_eq!(chunk, (0..prefill.steps.len()).collect::<Vec<_>>());
    // Every decode-side op once; nothing else from the decode plan.
    let dec: Vec<usize> = sched
        .iter()
        .filter_map(|o| match o {
            FusedOp::Decode(i) => Some(*i),
            FusedOp::Chunk(_) => None,
        })
        .collect();
    let want: Vec<usize> = decode
        .steps
        .iter()
        .enumerate()
        .filter(|(_, s)| is_decode_side(s.name()))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(dec, want);
    assert!(!dec.is_empty());
    // Each decode op right after its own layer's chunk group, and after the shared rope/append.
    for (k, o) in sched.iter().enumerate() {
        let FusedOp::Decode(i) = o else { continue };
        let l = layer_of(decode.steps[*i].name());
        let before: Vec<&str> = sched[..k]
            .iter()
            .filter_map(|o| match o {
                FusedOp::Chunk(j) => Some(prefill.steps[*j].name()),
                FusedOp::Decode(_) => None,
            })
            .filter(|n| n.starts_with(l))
            .collect();
        assert!(
            before.iter().any(|n| is_chunk_side(n)),
            "{} runs before its layer's chunk group",
            decode.steps[*i].name()
        );
    }
    // The chunk groups' buffers, per layer.
    let mut chunk_bufs: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for s in prefill.steps.iter().filter(|s| is_chunk_side(s.name())) {
        for (_, src) in &s.bindings {
            if let Source::Arena { buffer, .. } = src {
                chunk_bufs
                    .entry(layer_of(s.name()).to_owned())
                    .or_default()
                    .insert(buffer.clone());
            }
        }
    }
    // Every decode-side arena binding rebases onto the decode rows of an edge its layer's chunk
    // group binds too, or is scratch private to the decode-side ops.
    let mut private = 0;
    for i in &dec {
        let s = &decode.steps[*i];
        let l = layer_of(s.name());
        for (arg, src) in &s.bindings {
            let Source::Arena { buffer, offset, .. } = src else {
                continue;
            };
            // A decode-private scratch buffer stays on the decode arena.
            if let Rebound::Decode(o) =
                rebind(decode, prefill, buffer, *offset, u64::from(c)).expect("rebind")
            {
                assert_eq!(o, *offset);
                assert!(prefill.buffer(buffer).is_none(), "{buffer}");
                private += 1;
                continue;
            }
            assert!(
                chunk_bufs.get(l).is_some_and(|b| b.contains(buffer)),
                "{}.{arg} binds `{buffer}`, which {l}'s chunk group does not",
                s.name()
            );
            let at = rebase(decode, prefill, buffer, *offset, u64::from(c)).expect("rebase");
            let p = prefill.buffer(buffer).expect("buffer");
            let d = decode.buffer(buffer).expect("buffer");
            let per = p.rows.bytes();
            assert_eq!(per, d.rows.bytes(), "{buffer}: one row layout");
            // Inside the buffer, at or after row c of its block, and the decode rows fit.
            assert!(at >= p.offset && at < p.offset + p.bytes, "{buffer}");
            let first = p.rows.spans(u64::from(c));
            assert!(
                first.iter().any(|(o, _)| p.offset + o <= at),
                "{}.{arg}: `{buffer}` rebased before row {c}",
                s.name()
            );
            assert!(u64::from(c + d_rows) <= p.rows.rows(p.bytes));
        }
    }
    let _ = private;
    // The cached binding plan (what the engine binds with) agrees with the law at this row.
    let fb = FusedBinding::new(prefill, decode, u64::from(prefill.bucket.m() - d_rows))
        .expect("a binding plan");
    assert_eq!(fb.ops, sched);
    for (i, per) in &fb.rebound {
        for (j, r) in per.iter().enumerate() {
            let (_, src) = &decode.steps[*i].bindings[j];
            match (r, src) {
                (Some((true, at0, row)), Source::Arena { buffer, offset, .. }) => assert_eq!(
                    at0 + u64::from(c) * row,
                    rebase(decode, prefill, buffer, *offset, u64::from(c)).expect("rebase")
                ),
                (Some((false, at, _)), Source::Arena { offset, .. }) => assert_eq!(at, offset),
                (None, Source::Arena { .. }) | (Some(_), _) => panic!("binding {j} of step {i}"),
                (None, _) => {}
            }
        }
    }
}

#[test]
fn every_served_bucket_fuses_at_every_rung_that_holds_it() {
    let g = graph();
    let r = engine_rules();
    let mut checked = 0;
    for fusion in [false, true] {
        for batch in 1..=8u32 {
            for width in [1u32, 5, 8] {
                let o = if fusion {
                    body().with_decode_fusion()
                } else {
                    body()
                };
                let decode = plan(
                    &g,
                    Bucket::Decode {
                        batch,
                        verify_width: width,
                    },
                    &r.base,
                    o,
                );
                let d = batch * width;
                // The rungs a chunk beside this bucket can take, at their smallest and largest c.
                let mut rungs = BTreeSet::new();
                for c in [1u32, 96, 178, 500, 1400, 1600, 2048 - d] {
                    if let Some(f) = FusedRows::of_in_class(c, d, &LADDER, 2048) {
                        rungs.insert((f.rung, c));
                    }
                }
                let mut done_rungs = BTreeSet::new();
                for (rung, c) in rungs {
                    // One chunk size per rung is enough for the plan law (the plan is the rung's).
                    if !done_rungs.insert(rung) || (!fusion && batch % 3 != 1) {
                        continue;
                    }
                    let prefill = plan(
                        &g,
                        Bucket::Prefill {
                            tokens: rung,
                            seqs: 1,
                        },
                        rules_for_rows(&r.base, Some(&r.small), rung),
                        body(),
                    );
                    check(&prefill, &decode, c);
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 60, "{checked}");
}

#[test]
fn a_decode_range_past_the_rung_is_refused() {
    let g = graph();
    let r = engine_rules();
    let decode = plan(
        &g,
        Bucket::Decode {
            batch: 8,
            verify_width: 8,
        },
        &r.base,
        body(),
    );
    let prefill = plan(
        &g,
        Bucket::Prefill {
            tokens: 256,
            seqs: 1,
        },
        rules_for_rows(&r.base, Some(&r.small), 256),
        body(),
    );
    let s = decode
        .steps
        .iter()
        .find(|s| s.name().ends_with(".attn.decode"))
        .expect("attention");
    let (buffer, offset) = s
        .bindings
        .iter()
        .find_map(|(_, src)| match src {
            Source::Arena { buffer, offset, .. } => Some((buffer.clone(), *offset)),
            _ => None,
        })
        .expect("an arena binding");
    // 64 decode rows at row 192 fit a 256-row rung; at row 193 they do not.
    assert!(rebase(&decode, &prefill, &buffer, offset, 192).is_ok());
    let e = rebase(&decode, &prefill, &buffer, offset, 193).expect_err("past the rung");
    assert!(e.to_string().contains("run past"), "{e}");
}

#[test]
fn fusion_row_5_is_refused_not_run_twice() {
    // h42's row 5 folds `gdn.norm` (row-local) into `fused_update` (decode-side) at b1: the
    // fused step would run the norm's rows from the prefill plan AND inside the decode kernel.
    let g = graph();
    let r = engine_rules();
    let decode = plan(
        &g,
        Bucket::Decode {
            batch: 1,
            verify_width: 1,
        },
        &r.base,
        body().with_h42_fusion(),
    );
    let folded = decode
        .steps
        .iter()
        .any(|s| matches!(&s.route, Route::Fused { into, .. } if is_decode_side(into)));
    let prefill = plan(
        &g,
        Bucket::Prefill {
            tokens: 128,
            seqs: 1,
        },
        rules_for_rows(&r.base, Some(&r.small), 128),
        body(),
    );
    if folded {
        let e = fused_schedule(&prefill, &decode).expect_err("row 5");
        assert!(e.to_string().contains("fused into"), "{e}");
    } else {
        // If the planner does not fold at this shape the law is vacuous here; the served set
        // (`all`) never folds into a decode-side op (the test above schedules it).
        assert!(fused_schedule(&prefill, &decode).is_ok());
    }
}

#[test]
fn cmp27aa_a_decode_side_fold_into_its_own_layers_decode_op_is_scheduled_once() {
    // CMP27AH's `cuf` folds `gdn.recurrent_update` (decode-side) into `gdn.conv_update`
    // (decode-side): the fused step runs the conv's launch in the layer's decode group and skips
    // the folded step. The control: a ROW-LOCAL op folded into the same decode-side op stays
    // refused (its row-local half would run twice).
    let g = graph();
    let r = engine_rules();
    let mut decode = plan(
        &g,
        Bucket::Decode {
            batch: 4,
            verify_width: 8,
        },
        &r.base,
        body(),
    );
    let prefill = plan(
        &g,
        Bucket::Prefill {
            tokens: 256,
            seqs: 1,
        },
        rules_for_rows(&r.base, Some(&r.small), 256),
        body(),
    );
    let rec = decode
        .steps
        .iter()
        .position(|s| s.name().ends_with(".gdn.recurrent_update"))
        .expect("a b4 decode plan carries the GDN pair");
    let conv = decode.steps[rec]
        .name()
        .replace(".recurrent_update", ".conv_update");
    assert!(decode.steps.iter().any(|s| s.name() == conv), "{conv}");
    let before = fused_schedule(&prefill, &decode).expect("the unfused schedule");
    decode.steps[rec].route = Route::Fused {
        into: conv.clone(),
        reason: "test: the cuf pair",
    };
    let after = fused_schedule(&prefill, &decode).expect("the cuf fold schedules");
    let expect: Vec<FusedOp> = before
        .iter()
        .copied()
        .filter(|o| *o != FusedOp::Decode(rec))
        .collect();
    assert_eq!(after, expect, "the folded step is the only op dropped");
    // The control: a row-local step of the same layer folded into the conv update.
    let lay = layer_of(&conv).to_owned();
    let row_local = decode
        .steps
        .iter()
        .position(|s| {
            let n = s.name();
            n.starts_with(&lay)
                && !is_decode_side(n)
                && !is_chunk_side(n)
                && matches!(s.route, Route::Kernel { .. })
        })
        .expect("a row-local step in the layer");
    decode.steps[row_local].route = Route::Fused {
        into: conv,
        reason: "test: the control",
    };
    let e = fused_schedule(&prefill, &decode).expect_err("row-local into decode-side");
    assert!(e.to_string().contains("run twice"), "{e}");
}
