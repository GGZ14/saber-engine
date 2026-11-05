//! **C7: the shared prefill arena and the ragged plan's class law**, on the host and with no card.
//!
//! * THE SHARED ARENA: every rung of the default ladder planned exactly as the engine plans it (the
//!   small-chunk rules at ≤ 128 rows, the mid-M folded tiles on), and the sizing law over their
//!   arenas: one allocation of the widest rung's bytes holds every rung's every buffer, and the
//!   refund is the sum minus the max.
//! * THE RAGGED CLASS LAW: a chunk re-planned at its live rows rounded to 16 (`ragged_rows`) is
//!   bit-exact with its rung only where every op of the ragged plan binds a kernel of the rung's
//!   summation class. This census names the rows where that holds and where it does not.

use saber_engine::small_chunk::{rules_for_rows, small_chunk_rules, SmallChunkClass};
use saber_exec::layer::{BlockPlan, PlanOptions, Route, RuntimeParams};
use saber_kernels::rules::{summation_class, LaunchRules};
use saber_kernels::saber::SaberRules;
use saber_model::graph::{Bucket, Graph, ModelConfig};
use saber_sched::prefill_law::{ragged_rows, SharedPrefillArena, RAGGED_ROW_GRANULE};

const LADDER: [u32; 11] = [96, 128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048];

struct Rules {
    base: LaunchRules,
    small: (u32, LaunchRules),
}

fn engine_rules() -> Rules {
    let mut base = LaunchRules::prod_mxfp4().expect("rules");
    base.knobs.folded_mt = true;
    // G6 (the default since 2026-09-24): the mid-band layout law and the skinny 48-wide gate.
    base.knobs.atiled_mid = true;
    base.knobs.skinny = true;
    let small = small_chunk_rules(&base, 128, SmallChunkClass::Exact).expect("small");
    Rules {
        base,
        small: (128, small),
    }
}

/// `(kernel symbols in order, arena bytes, declared launches)` of a prefill plan at `rows`.
fn plan(g: &Graph, rows: u32, r: &Rules) -> (Vec<String>, u64, u32) {
    let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
    let p = BlockPlan::with_options(
        g,
        Bucket::Prefill {
            tokens: rows,
            seqs: 1,
        },
        &layers,
        RuntimeParams {
            max_ctx: 4096,
            gdn_slots: 65,
        },
        rules_for_rows(&r.base, Some(&r.small), rows),
        &SaberRules::prod_saber().expect("saber rules"),
        PlanOptions::BODY.with_batch_invariant_attn(),
    )
    .expect("a prefill body plan");
    let mut out = Vec::new();
    for s in &p.steps {
        if let Route::Kernel { descs, .. } = &s.route {
            for d in descs {
                out.push(d.kernel.clone());
            }
        }
    }
    (out, p.arena_bytes, p.declared_launches)
}

#[test]
fn the_shared_arena_holds_every_rung_of_the_ladder() {
    let r = engine_rules();
    for tp in [1u32, 2] {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), tp).expect("graph");
        let rungs: Vec<(u32, u64)> = LADDER
            .iter()
            .map(|rows| (*rows, plan(&g, *rows, &r).1))
            .collect();
        let law = SharedPrefillArena::of(rungs.clone());
        eprintln!("tp{tp}: {}", law.line(0));
        for (rows, bytes) in &rungs {
            eprintln!("tp{tp}   p{rows}: {:.1} MiB", *bytes as f64 / 1048576.0);
            assert!(law.holds(*bytes), "tp{tp} p{rows}");
        }
        assert_eq!(law.widest_rows(), 2048, "tp{tp}: the widest rung sizes it");
        // The arena is monotone in rows, so the refund is every rung but the top.
        let below: u64 = rungs
            .iter()
            .filter(|(r, _)| *r < 2048)
            .map(|(_, b)| *b)
            .sum();
        assert_eq!(law.refund_bytes(), below, "tp{tp}");
    }
}

/// **The ragged class law, as a census over the default ladder at TP=2.** A chunk RE-PLANNED at
/// its live rows rounded to 16 keeps its rung's kernel class (every symbol equal, or equal under
/// `summation_class`, same launch count) on p192..p1536 — and MOVES it on the two ends:
///
/// * p96/p128: the decode GEMM's tile height (`tm`) follows M, and at M <= 64 the committed
///   decode band splits K (`ks2`/`ks4`): a different sum for the shortest prompts;
/// * p2048: every ragged M below 2048 leaves the 512-thread producers (`quant_large_min_m`) and
///   the `tn4_lbk64_wm8sp` tile for `tn2_lbk128` + the 256-thread ones (C5's class split).
///
/// So a ragged law that only RE-PLANS is bit-exact by class on the middle rungs alone; the ends
/// would need a plan resolved at the rung's M and run at the chunk's (not built, C7).
#[test]
fn ragged_class_census() {
    let r = engine_rules();
    let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
    let mut prev = 0u32;
    for rung in LADDER {
        let (kr, ar, lr) = plan(&g, rung, &r);
        let mut moved = 0usize;
        let mut total = 0usize;
        let mut m = ragged_rows(prev + 1, rung, RAGGED_ROW_GRANULE);
        while m < rung {
            let (km, am, lm) = plan(&g, m, &r);
            let same = km.len() == kr.len()
                && lm == lr
                && km
                    .iter()
                    .zip(&kr)
                    .all(|(a, b)| a == b || summation_class(a) == summation_class(b));
            assert!(am <= ar, "p{rung} M={m}: a ragged plan wider than its rung");
            total += 1;
            if !same {
                moved += 1;
            }
            m += RAGGED_ROW_GRANULE;
        }
        eprintln!("p{rung}: {moved} of {total} ragged M move class");
        match rung {
            96 | 128 | 2048 => assert_eq!(moved, total, "p{rung}: the control ends move"),
            _ => assert_eq!(moved, 0, "p{rung}: a middle rung moved class"),
        }
        prev = rung;
    }
}

/// C7's hold-1 failure, pinned: over the shared arena every served prompt's ids moved (0 / 60)
/// because layer 0 reads an explicit ZERO residual that nothing writes — zero only because each
/// rung's own arena was zeroed at boot.
#[test]
fn prefill_reads_before_write() {
    let r = engine_rules();
    let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
    for rows in [320u32, 1536] {
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let p = BlockPlan::with_options(
            &g,
            Bucket::Prefill {
                tokens: rows,
                seqs: 1,
            },
            &layers,
            RuntimeParams {
                max_ctx: 4096,
                gdn_slots: 65,
            },
            rules_for_rows(&r.base, Some(&r.small), rows),
            &SaberRules::prod_saber().expect("saber rules"),
            PlanOptions::BODY.with_batch_invariant_attn(),
        )
        .expect("plan");
        let rbw = p.reads_before_write();
        // The one buffer no op writes is layer 0's zero residual: a CONSTANT, re-seeded by
        // `zero_constants` before every run (C7 over the shared arena; T7 everywhere). C7 also
        // listed conv_prep's private `gates` and qk_norm_rope's `k` -- writes the walker could
        // not see; since T7 an op's private intermediate counts as its owner's write, and T7's
        // `buf=` poison of each (0x3c at boot) moved nothing on the card.
        let names: Vec<&str> = rbw.iter().map(|(b, _)| b.as_str()).collect();
        assert_eq!(
            names,
            vec!["layer.0.input_layernorm.zero_resid"],
            "p{rows}: a new buffer is read before any op of the plan writes it; a shared \
             arena would hand it another rung's bytes"
        );
        assert_eq!(
            p.buffers
                .iter()
                .filter(|b| b.name.ends_with(saber_exec::layer::ZERO_CONSTANT_SUFFIX))
                .count(),
            1,
            "p{rows}: the constants zero_constants re-seeds"
        );
    }
}
