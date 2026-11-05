//! **C8: the fused mixed step's class law**, on the host and with no card.
//!
//! A fused mixed step runs ONE forward over `M = chunk rows + decode rows`: every row-local op
//! (the GEMMs, their producers, the norms, the reduces) once over all rows at the mixed rung's
//! plan, and the per-sequence ops (attention, GDN, the head, the drafter) as two launches over
//! disjoint row ranges with each side's own kernels. So a row is bit-exact against its unmixed
//! step exactly where every row-local op it passes through binds a kernel of the SAME summation
//! class in the mixed plan as in its own plan. This census asks that question op by op with the
//! engine's own rules (`rules::fold_class`, the C6/G2/G5/G6 class machinery, and the all-reduce's
//! exact-vs-lossy size rule), for every decode bucket the engine serves and every rung the mixed
//! step could run at.
//!
//! **T4's correction (2026-09-24): this census counts KERNEL-class moves, not byte moves.** At
//! b1s8 the split-K GEMM moves it lists are byte-neutral on the card: C9's `mixfused` per-row arm
//! under `--ar-exact` reads the fused decode row 0 bytes from the serial one at p192/p256/p320
//! (B1, and T4 on main `f09e957`), so at <= 12 decode rows the decode row's BYTE move is the
//! reduce family alone -- what `predict_mixed_decode_row` computes
//! (`at_b1s8_the_decode_rows_byte_move_is_the_reduce_family_alone` pins it). The "never bit-exact"
//! below is a statement about kernels; at 15..56 decode rows it has not been read on the card.
//!
//! **The answer is a refutation.** The decode GEMM at ONE split is the folded order (C6), and the
//! folded / A-tiled families are one class (G2/G5/G6) — but the decode rows do not run at one
//! split: the committed decode band splits K on `attn.qkv` (ks2) and on the three GEMMs whose
//! output is reduced (`gdn.out_proj`, `attn.o_proj`, `mlp.down_proj`: ks2 / ks4) at EVERY served
//! decode M, and a split-K sum reassociates. A decode step of <= 12 rows also reduces over the
//! EXACT all-reduce (its message is under `quant_min_bytes`), and a mixed step's message never
//! is. So the fused step moves the decode rows' class on four GEMMs a layer (and the reduce at
//! <= 12 decode rows) for every (rung, batch x width) pair — a numerics candidate, gated by the
//! quality gates, never a bit-exact form. The chunk rows keep their class at every mixed rung up
//! to 1536 (every mixed GEMM is in the fold class, the producers stay 256-thread); a mixed rung of
//! 2048 over a chunk from a lower rung moves the chunk's producers (C5's split).

use saber_engine::mixed_class::{mixed_proxy_rules, mixed_proxy_saber};
use saber_engine::small_chunk::{rules_for_rows, small_chunk_rules, SmallChunkClass};
use saber_exec::layer::{BlockPlan, Collectives, PlanOptions, Route, RuntimeParams};
use saber_kernels::rules::{fold_class, LaunchRules};
use saber_kernels::saber::SaberRules;
use saber_model::graph::{Bucket, Graph, ModelConfig};

const LADDER: [u32; 11] = [96, 128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048];

/// The engine's rules at the defaults: the committed rules for decode, the small-chunk rules for
/// rungs <= 128, the G5 mid tiles and the G6 layout law + skinny gate.
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

/// One row-local op's kernel in a plan: `(op name with the layer prefix, kernel symbol)`.
type Ops = Vec<(String, String)>;

/// The row-local ops of a plan (every GEMM, producer and reduce; the per-sequence ops excluded),
/// in plan order.
fn row_local_ops(g: &Graph, b: Bucket, r: &LaunchRules, o: PlanOptions) -> Ops {
    row_local_ops_with(g, b, r, &SaberRules::prod_saber().expect("saber rules"), o)
}

fn row_local_ops_with(
    g: &Graph,
    b: Bucket,
    r: &LaunchRules,
    saber: &SaberRules,
    o: PlanOptions,
) -> Ops {
    let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
    let p = BlockPlan::with_options(
        g,
        b,
        &layers,
        RuntimeParams {
            max_ctx: 4096,
            gdn_slots: 65,
        },
        r,
        saber,
        o,
    )
    .expect("a plan");
    let mut out = Vec::new();
    for s in &p.steps {
        if let Route::Kernel { descs, .. } = &s.route {
            if is_per_sequence(s.name()) {
                continue;
            }
            for d in descs {
                out.push((s.name().to_string(), d.kernel.clone()));
            }
        }
    }
    out
}

/// The ops a fused mixed step runs as two launches over disjoint rows (each side keeps its own
/// kernel): attention, the rope + KV append, the GDN conv / scan / update and the embedding (a
/// gather, row-local but kernel-invariant in M).
fn is_per_sequence(op: &str) -> bool {
    [
        ".attn.decode",
        ".attn.prefill",
        ".attn.qk_norm_rope",
        ".attn.kv_append",
        ".gdn.conv_",
        ".gdn.kkt_solve",
        ".gdn.chunk_scan",
        ".gdn.recurrent_update",
        ".gdn.fused_update",
        "embed_tokens",
    ]
    .iter()
    .any(|t| {
        if *t == "embed_tokens" {
            op == *t
        } else {
            op.contains(t)
        }
    })
}

fn is_gemm(k: &str) -> bool {
    k.starts_with("mxfp4_gemm_")
}

fn is_ar(k: &str) -> bool {
    k.starts_with("saber_ar_")
}

/// The class a row-local kernel sums in, for this census: the fold class for a GEMM in it, the
/// all-reduce's family (exact / lossy, and its width), and the producer's thread count otherwise.
fn class_of(k: &str) -> String {
    if is_gemm(k) {
        return fold_class(k).map_or_else(|| format!("OFF-CLASS {k}"), str::to_string);
    }
    if is_ar(k) {
        return if k.contains("_exact") {
            "ar-exact".into()
        } else if let Some(i) = k.find("_wht") {
            format!("ar-{}", &k[i + 1..i + 5])
        } else {
            format!("ar-other {k}")
        };
    }
    // Producers: the arithmetic is the thread count's (t256 / t512); the tiled flag is layout.
    k.split("_tiled").next().unwrap_or(k).to_string()
}

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

fn decode_ops(g: &Graph, r: &Rules, batch: u32, width: u32) -> Ops {
    // The UNFUSED decode plan: every served fusion is gated bit-exact against the composition it
    // replaces (step-tp2's fusion gate), so the composition's kernels ARE the decode rows' class.
    row_local_ops(
        g,
        Bucket::Decode {
            batch,
            verify_width: width,
        },
        &r.base,
        body(),
    )
}

fn prefill_ops(g: &Graph, r: &Rules, rows: u32) -> Ops {
    row_local_ops(
        g,
        Bucket::Prefill {
            tokens: rows,
            seqs: 1,
        },
        rules_for_rows(&r.base, Some(&r.small), rows),
        body(),
    )
}

/// Ops (by name, layer prefix stripped, deduplicated) whose class differs between the two plans.
/// The two plans list the same row-local ops in the same order.
fn moved(a: &Ops, b: &Ops) -> Vec<String> {
    assert_eq!(
        a.iter().map(|(n, _)| n).collect::<Vec<_>>(),
        b.iter().map(|(n, _)| n).collect::<Vec<_>>(),
        "the two plans must list the same row-local ops"
    );
    let mut out: Vec<String> = Vec::new();
    for ((n, ka), (_, kb)) in a.iter().zip(b) {
        if class_of(ka) != class_of(kb) {
            let op = n
                .split_once('.')
                .and_then(|(_, rest)| rest.split_once('.'))
                .map_or(n.as_str(), |(_, op)| op)
                .to_string();
            let tag = format!("{op} ({} -> {})", short(ka), short(kb));
            if !out.contains(&tag) {
                out.push(tag);
            }
        }
    }
    out
}

fn short(k: &str) -> String {
    k.replace("mxfp4_gemm_", "")
        .replace("_wperm1", "")
        .replace("saber_ar_oneshot_2rank_", "ar_")
}

/// The smallest ladder rung holding `rows`.
fn rung(rows: u32) -> Option<u32> {
    LADDER.iter().copied().find(|r| *r >= rows)
}

/// **THE CLASS LAW (C8).** For every decode bucket the mixed step's decode half can be (batch
/// 1..7 — one sequence is the chunk's — at the served widths 1, 5, 8) and every rung a chunk of
/// at least one row puts the mixed step on, the decode rows' row-local ops in the mixed plan
/// against their own decode plan: which move class.
///
/// Asserted: (a) no (rung, batch x width) pair keeps the decode rows bit-exact — `attn.qkv` and
/// the three reduced GEMMs are split-K in every served decode bucket; (b) `mlp.gate_up` (one split
/// at every decode M) never moves — the GDN `in_proj` pair moves only where its band splits it; (c) the reduce moves exactly when the
/// decode half is <= 12 rows (exact under 128 KiB); (d) the producers never move below the 2048
/// rung.
#[test]
fn the_fused_mixed_step_moves_the_decode_rows_class() {
    let g = graph();
    let r = engine_rules();
    let mut table: Vec<String> = Vec::new();
    for batch in 1..=7u32 {
        for width in [1u32, 5, 8] {
            let d = batch * width;
            let dec = decode_ops(&g, &r, batch, width);
            // Every rung that can carry a chunk of >= 1 row with these decode rows.
            let mut rungs: Vec<u32> = LADDER.iter().copied().filter(|x| *x > d).collect();
            rungs.dedup();
            for mixed_rung in rungs {
                let mix = prefill_ops(&g, &r, mixed_rung);
                let mv = moved(&dec, &mix);
                // (a) never bit-exact
                assert!(
                    !mv.is_empty(),
                    "b{batch}s{width} at p{mixed_rung}: the census found the decode rows bit-exact"
                );
                // The split GEMMs are what moves.
                for must in ["attn.qkv", "attn.o_proj", "mlp.down_proj", "gdn.out_proj"] {
                    assert!(
                        mv.iter().any(|m| m.starts_with(must)),
                        "b{batch}s{width} at p{mixed_rung}: {must} did not move: {mv:?}"
                    );
                }
                // (b) the one-split GEMMs stay.
                let stays = "mlp.gate_up";
                assert!(
                    !mv.iter().any(|m| m.starts_with(stays)),
                    "b{batch}s{width} at p{mixed_rung}: {stays} moved: {mv:?}"
                );
                // (c) the reduce: exact at <= 12 decode rows, lossy in every mixed plan.
                let ar_moves = mv.iter().any(|m| m.contains("all_reduce"));
                assert_eq!(
                    ar_moves,
                    d * 5120 * 2 < 131_072,
                    "b{batch}s{width} at p{mixed_rung}: the reduce's family law"
                );
                // (d) producers: the 256-thread class below the 2048 rung.
                let prod_moves = mv.iter().any(|m| {
                    let op = m.split(" (").next().unwrap_or(m);
                    op.ends_with("layernorm")
                        || op == "mlp.act"
                        || op == "gdn.norm"
                        || op == "attn.gate_quant"
                });
                assert_eq!(
                    prod_moves,
                    mixed_rung == 2048,
                    "b{batch}s{width} at p{mixed_rung}: producers {mv:?}"
                );
                if mixed_rung == rung(d + 1).expect("a rung") {
                    table.push(format!(
                        "b{batch}s{width} ({d:2} rows) at p{mixed_rung:4}: {} op(s) move: {}",
                        mv.len(),
                        mv.join(", ")
                    ));
                }
            }
        }
    }
    for line in &table {
        eprintln!("{line}");
    }
}

/// **The chunk rows' side of the law.** A chunk from rung `c` riding a mixed step at rung
/// `R >= c` keeps every row-local op's class up to `R = 1536`; at `R = 2048` its producers leave
/// the 256-thread class unless the chunk was a p2048 chunk already.
#[test]
fn the_chunk_rows_keep_their_class_below_the_2048_rung() {
    let g = graph();
    let r = engine_rules();
    let plans: Vec<(u32, Ops)> = LADDER
        .iter()
        .map(|x| (*x, prefill_ops(&g, &r, *x)))
        .collect();
    for (c, pc) in &plans {
        for (big, pr) in plans.iter().filter(|(x, _)| x >= c) {
            let mv = moved(pc, pr);
            if *big < 2048 || *c == 2048 {
                assert!(mv.is_empty(), "p{c} riding p{big}: {mv:?}");
            } else {
                assert!(
                    !mv.is_empty() && mv.iter().all(|m| !m.contains("OFF-CLASS")),
                    "p{c} riding p2048: only the producers move, the GEMMs stay: {mv:?}"
                );
            }
        }
    }
}

/// The negative control (structural rule 8): the census's own comparison must see a split. The
/// decode GEMM at ONE split (C6's small chunk) is in the fold class, and the same cell at the
/// committed decode band's split is NOT — if this failed, (a) above would be vacuous.
#[test]
fn the_census_sees_a_split_decode_cell_as_a_class_move() {
    let g = graph();
    let r = engine_rules();
    // p96 under the small-chunk rules: the decode kernel at ks1 on every GEMM.
    let small = prefill_ops(&g, &r, 96);
    assert!(small
        .iter()
        .filter(|(_, k)| is_gemm(k))
        .all(|(_, k)| fold_class(k).is_some()));
    // The same 96 rows under the COMMITTED rules would not exist (decode_max_m 64), so compare a
    // 64-row decode bucket, b8s8, whose band splits: some GEMM is off the class.
    let dec = decode_ops(&g, &r, 8, 8);
    assert!(dec
        .iter()
        .any(|(_, k)| is_gemm(k) && fold_class(k).is_none()));
    // And the census's `moved` over a plan against itself is empty.
    assert!(moved(&small, &small).is_empty());
}

/// **The proxy reproduces the fused class (C8).** A decode bucket planned under
/// `mixed_class::mixed_proxy_rules` + `mixed_proxy_saber` with the decode fusions off binds, op by
/// op, the SAME class the mixed plan binds its decode rows to at every rung below 2048 — so a
/// quality gate served on the proxy measures the fused step's decode-row numerics (over-exposed:
/// every decode step, not only the ones carrying a chunk). The committed decode plan is the
/// control that moves (the first test).
#[test]
fn the_proxy_decode_plan_is_the_fused_class() {
    let g = graph();
    let r = engine_rules();
    let pr = mixed_proxy_rules(&r.base);
    let pr4 = mixed_proxy_saber(&SaberRules::prod_saber().expect("saber"));
    for batch in 1..=8u32 {
        for width in [1u32, 5, 8] {
            let d = batch * width;
            let proxy = row_local_ops_with(
                &g,
                Bucket::Decode {
                    batch,
                    verify_width: width,
                },
                &pr,
                &pr4,
                body(),
            );
            for mixed_rung in LADDER.iter().copied().filter(|x| *x > d && *x < 2048) {
                let mv = moved(&proxy, &prefill_ops(&g, &r, mixed_rung));
                assert!(
                    mv.is_empty(),
                    "b{batch}s{width} proxy vs p{mixed_rung}: {mv:?}"
                );
            }
        }
    }
}

/// **T4: the census's split-K claim at b1s8, against the card.** The class census above counts
/// KERNEL-class moves, and at <= 12 decode rows it finds the split-K decode GEMMs (`attn.qkv` ks2,
/// the GDN `in_proj` pair, and the three reduced GEMMs ks4) moving to the rung's fold class
/// beside the reduce (exact -> wht6). Those GEMM moves are NOT byte moves on the card: C9's
/// `mixfused` per-row arm under `--ar-exact` (the reduce exact on BOTH sides, the GEMMs still
/// split-K in the serial form and fold-class in the fused one) reads A's decode row 0 bytes, the
/// chunk rows 0 bytes and the four steps after identical at p192 / p256 (K2) / p320 (K3) -- B1's
/// hold 1 (`ws-b1/hold1-f9b7974-0924-133219/engine-bat-X3.log`) and T4's hold 2 on main `f09e957` plus
/// T4 (`ws-t4/stress-92c414d-0924-182322/engine-arx8.log`). So the decode row's BYTE move at
/// b1s8 is the reduce family alone, which is exactly what `predict_mixed_decode_row`
/// (saber-cli) computes from the plans. This test pins that correspondence: at the card's rungs,
/// every non-reduce op the census moves at b1s8 is one of the card-verified byte-neutral split-K
/// GEMMs, the reduce is what moves under the default band, and under `--ar-exact` nothing but
/// those GEMMs moves -- so the prediction ("0 bytes under one reduce class") is the census's own
/// answer once the verified GEMM moves are discounted. A new kind of move at b1s8 (a producer, a
/// different GEMM) fails here, because the card never said anything about it.
#[test]
fn at_b1s8_the_decode_rows_byte_move_is_the_reduce_family_alone() {
    // (op, what the card read). Kernel-class moves the per-row arm reads byte-neutral at b1s8.
    const CARD_BYTE_NEUTRAL: &[&str] = &[
        "attn.qkv",
        "attn.o_proj",
        "mlp.down_proj",
        "gdn.out_proj",
        "gdn.in_proj_qkvz",
        "gdn.in_proj_ba",
    ];
    let g = graph();
    let r = engine_rules();
    let served = SaberRules::prod_saber().expect("saber");
    // `--ar-exact`: the band above every message this run sends (the engine raises
    // `quant_min_bytes` the same way), so both plans reduce exact.
    let mut exact = served.clone();
    exact.ar.quant_min_bytes = u64::MAX / 4;
    let dec = |saber: &SaberRules| {
        row_local_ops_with(
            &g,
            Bucket::Decode {
                batch: 1,
                verify_width: 8,
            },
            &r.base,
            saber,
            body(),
        )
    };
    let pre = |rows: u32, saber: &SaberRules| {
        row_local_ops_with(
            &g,
            Bucket::Prefill {
                tokens: rows,
                seqs: 1,
            },
            rules_for_rows(&r.base, Some(&r.small), rows),
            saber,
            body(),
        )
    };
    let op_of = |m: &str| m.split(" (").next().unwrap_or(m).to_string();
    for rung in [192u32, 256, 320] {
        // The default band: the reduce moves (exact -> wht6), and every other move is verified.
        let mv = moved(&dec(&served), &pre(rung, &served));
        assert!(
            mv.iter().any(|m| m.contains("all_reduce")),
            "p{rung}: the default band must move the reduce at b1s8 (the control): {mv:?}"
        );
        for m in mv.iter().filter(|m| !m.contains("all_reduce")) {
            assert!(
                CARD_BYTE_NEUTRAL.contains(&op_of(m).as_str()),
                "p{rung}: `{m}` moves at b1s8 and the card has not read it byte-neutral"
            );
        }
        // `--ar-exact`: no reduce moves; what is left is only the verified GEMM moves, so the
        // byte prediction is 0 bytes -- the card's reading.
        let mx = moved(&dec(&exact), &pre(rung, &exact));
        assert!(
            !mx.iter().any(|m| m.contains("all_reduce")),
            "p{rung}: under --ar-exact the reduce keeps its family: {mx:?}"
        );
        assert!(
            !mx.is_empty(),
            "p{rung}: the split-K GEMMs still change KERNEL class under --ar-exact (the census's \
             half of the claim, which stands): {mx:?}"
        );
        for m in &mx {
            assert!(
                CARD_BYTE_NEUTRAL.contains(&op_of(m).as_str()),
                "p{rung}: `{m}` moves under --ar-exact and the card has not read it byte-neutral"
            );
        }
    }
}
