//! **T7: the stale-read law over EVERY plan the engine builds**, on the host and with no card.
//!
//! T7's bisecting boot poison named the one arena buffer a run read without anything having written
//! it in that run: layer 0's `.zero_resid` (the explicit zero residual `input_layernorm` adds) in
//! every DECODE bucket, an own arena that C7's re-seed never covered (it re-seeded the shared
//! prefill arena only). A `0x3c` fill of that buffer alone moved every request of a boot (solo ids
//! `0xf3abd92c1b8d7e93`, acceptance 10883/2513, identical to the whole-arena poison); a `0x00` fill
//! of the prefill-side candidates did nothing and neither did any other arena buffer (LEDGER T7).
//!
//! The fix is the executor's: every plan's constants are re-seeded before EVERY run
//! (`ConstantSeeds::EveryRun`, `run_step`), and the law is `BlockPlan::stale_reads()` EMPTY for
//! every prefill rung (the default ladder at TP=2 and TP=1, the small-chunk rules at <= 128 rows),
//! every decode bucket (the ladder `1..8` x widths `1,5,8`, fusions `all` and off, TP=2 and TP=1),
//! every row band (16..64 at `b8s8`) and every decode head -- asserted at boot too
//! (`plan_carries_nothing`, which refuses the boot by name).
//!
//! Negative control (structural rule 8): the SAME plans under the pre-T7 policy
//! (`ConstantSeeds::SharedArenaOnly { shared: false }`, a decode bucket's own arena) carry exactly
//! `layer.0.input_layernorm.zero_resid` -- the failing control the law must see.

use saber_engine::small_chunk::{rules_for_rows, small_chunk_rules, SmallChunkClass};
use saber_exec::layer::{BlockPlan, Collectives, ConstantSeeds, PlanOptions, RuntimeParams};
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_model::graph::{Bucket, Graph, ModelConfig};

const LADDER: [u32; 11] = [96, 128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048];
const BATCHES: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
const WIDTHS: [u32; 3] = [1, 5, 8];
const BANDS: [u32; 7] = [16, 24, 32, 40, 48, 56, 64];
const ZERO: &str = "layer.0.input_layernorm.zero_resid";

fn params() -> RuntimeParams {
    RuntimeParams {
        max_ctx: 4096,
        gdn_slots: 65,
    }
}

fn coll(tp: u32) -> Collectives {
    if tp > 1 {
        Collectives::Real
    } else {
        Collectives::Identity
    }
}

fn names(v: &[(String, String)]) -> Vec<&str> {
    v.iter().map(|(b, _)| b.as_str()).collect()
}

/// Every plan the boot builds, by label, at one TP.
fn every_plan<'g>(g: &'g Graph, tp: u32) -> Vec<(String, BlockPlan<'g>)> {
    let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
    let saber = SaberRules::prod_saber().expect("saber");
    let mut base = LaunchRules::prod_mxfp4().expect("rules");
    base.knobs.folded_mt = true;
    base.knobs.atiled_mid = true;
    base.knobs.skinny = true;
    let small = small_chunk_rules(&base, 128, SmallChunkClass::Exact).expect("small");
    let decode = LaunchRules::prod_mxfp4().expect("rules");
    let mut out = Vec::new();
    for rows in LADDER {
        let p = BlockPlan::with_options(
            g,
            Bucket::Prefill {
                tokens: rows,
                seqs: 1,
            },
            &layers,
            params(),
            rules_for_rows(&base, Some(&(128, small.clone())), rows),
            &saber,
            PlanOptions::BODY
                .rank(0)
                .with_collectives(coll(tp))
                .with_batch_invariant_attn(),
        )
        .expect("a prefill plan");
        out.push((format!("tp{tp} p{rows}"), p));
    }
    let body = PlanOptions::BODY
        .rank(0)
        .with_collectives(coll(tp))
        .with_batch_invariant_attn()
        .with_table_blocks(257);
    for (tag, opts) in [
        ("all", body.with_decode_fusion()),
        ("off", body.without_fusion()),
    ] {
        for batch in BATCHES {
            for width in WIDTHS {
                let p = BlockPlan::with_options(
                    g,
                    Bucket::Decode {
                        batch,
                        verify_width: width,
                    },
                    &layers,
                    params(),
                    &decode,
                    &saber,
                    opts,
                )
                .expect("a decode plan");
                out.push((format!("tp{tp} b{batch}s{width} fusion {tag}"), p));
            }
        }
    }
    for rows in BANDS {
        let p = BlockPlan::with_options(
            g,
            Bucket::DecodeBand {
                batch: 8,
                verify_width: 8,
                rows,
            },
            &layers,
            params(),
            &decode,
            &saber,
            body.with_decode_fusion(),
        )
        .expect("a band plan");
        out.push((format!("tp{tp} band r{rows}"), p));
    }
    let mut heads: Vec<u32> = BATCHES
        .iter()
        .flat_map(|b| WIDTHS.iter().map(move |w| b * w))
        .chain(BANDS)
        .collect();
    heads.sort_unstable();
    heads.dedup();
    for rows in heads {
        let p = BlockPlan::with_options(
            g,
            Bucket::Decode {
                batch: rows,
                verify_width: 1,
            },
            &[],
            params(),
            &decode,
            &saber,
            PlanOptions::HEAD.rank(0).with_collectives(coll(tp)),
        )
        .expect("a head plan");
        out.push((format!("tp{tp} head r{rows}"), p));
    }
    out
}

/// THE LAW: no plan the engine builds reads an arena byte that neither its own steps wrote first
/// nor the executor re-seeds before every run.
#[test]
fn no_plan_carries_a_stale_read() {
    let mut n = 0usize;
    for tp in [2u32, 1] {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), tp).expect("graph");
        for (label, p) in every_plan(&g, tp) {
            let stale = p.stale_reads();
            assert!(stale.is_empty(), "{label}: stale reads {stale:?}");
            n += 1;
        }
    }
    eprintln!("stale-read law: {n} plans, every one empty");
    assert!(n > 150, "the census must cover the plan set ({n})");
}

/// THE FAILING CONTROL: under the pre-T7 re-seed policy every BODY plan on an own arena carries
/// exactly the zero residual T7's poison named (and a head carries nothing either way).
#[test]
fn the_pre_fix_policy_carries_the_zero_residual() {
    let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
    let mut bodies = 0usize;
    for (label, p) in every_plan(&g, 2) {
        let own = p.stale_reads_under(ConstantSeeds::SharedArenaOnly { shared: false });
        if label.contains("head") {
            assert!(own.is_empty(), "{label}: {own:?}");
            continue;
        }
        assert_eq!(
            names(&own),
            vec![ZERO],
            "{label}: the control must see the zero residual"
        );
        // C7's shared prefill arena re-seeded it, so the prefill rungs were clean under C7.
        let shared = p.stale_reads_under(ConstantSeeds::SharedArenaOnly { shared: true });
        assert!(shared.is_empty(), "{label}: {shared:?}");
        bodies += 1;
    }
    assert!(bodies > 50);
}

/// What the walker reads before writing, now that an op's private intermediates count as the
/// owning op's writes: ONLY the constant (C7's list was the constant plus `conv_prep.gates` and
/// `qk_norm_rope.k`, "writes the walker cannot see"; T7's `buf=` poison of each moved nothing).
#[test]
fn reads_before_write_is_the_constant_alone() {
    let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
    for (label, p) in every_plan(&g, 2) {
        if label.contains("head") {
            continue;
        }
        assert_eq!(names(&p.reads_before_write()), vec![ZERO], "{label}");
    }
}
