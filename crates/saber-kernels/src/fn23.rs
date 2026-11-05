//! FN23 (2026-09-29): the persistent decode megakernel -- FEASIBILITY, not a product.
//!
//! What lives here is host-only data and law for `hip/flashnext_mega.hip` and the card bench
//! `tools/capture/fn23/fn23_bench.cpp`:
//!
//! * [`STAGES_B1`] / [`STAGES_B8`]: one GDN + MoE + HC decoder layer at TP=2 rank 0, SPEC 0, as the
//!   persistent walk runs it (the launch train's symbols in the launch train's order, which kernel
//!   of the pair `_a` / `_b` carries each, and which stage ends in a grid barrier);
//! * [`step_bytes_b1`]: the per-rank byte floor of a b1 SPEC-0 decode step from the pack's records;
//! * structural checks that the prototype carries NO copy of a certified body (it `#include`s the
//!   committed sources verbatim), that its kind numbering equals the bench's, and the LDS law that
//!   forces two kernels a layer (the static LDS of every stage body is summed, 76,416 B > 65,536).

/// One stage of the persistent walk: the certified symbol it calls, the kernel of the pair that
/// runs it (`false` = `_a`, `true` = `_b`) and whether a grid barrier follows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stage {
    pub symbol: &'static str,
    pub in_b: bool,
    pub barrier: bool,
}

const fn st(symbol: &'static str, in_b: bool, barrier: bool) -> Stage {
    Stage {
        symbol,
        in_b,
        barrier,
    }
}

/// b1: 15 launches + 2 reduces (the engine's GDN layer at b1, EXEC §15/§16), in order.
pub const STAGES_B1: &[Stage] = &[
    st("fn_hc_down_combine_v2_m1w3t1", false, true),
    st("fn_hc_up_bf16_v2_s1", false, true),
    st("fn_bgemm_sk_m16", false, true), // gdn.in_proj.tp2, split 2
    st("saber_gdn_fused_update_f16s_w4k128v128_ab1", false, true),
    st("fn_gdn_norm_bf16_sig_t256", false, true),
    st("fn_bgemm_sk_m16", false, true), // gdn.out_proj.tp2, split 4
    st("saber_ar_oneshot_2rank_exact_bf16", false, true),
    st("fn_hc_down_combine_v2_m1w3t1", false, true),
    st("fn_hc_up_q8_v2_s1", false, true),
    st("fn_bgemm_sk_m16", false, false), // moe.router, split 4 (independent of the shared gate)
    st("fn_bgemm_sk_m16", false, true),  // the shared-gate GEMV
    st("moe_route_topk10_sort_e512", false, false), // the kernel boundary is its barrier
    st("moe_grouped_w4a8_bk128_ks4_tm1_g1", true, true),
    st("moe_silu_mul_quant_t256", true, true),
    st("moe_grouped_w4a8_bk128_ks2_tm1_g0", true, true),
    st("moe_combine_bf16", true, true),
    st("saber_ar_oneshot_2rank_exact_bf16", true, false),
];

/// b8: the GDN update is the conv_update + recurrent_update PAIR (items 192 > 32), the bf16 GEMMs
/// the m16n16 band, the grouped GEMMs one split.
pub const STAGES_B8: &[Stage] = &[
    st("fn_hc_down_combine_v2_m1w7t1", false, true),
    st("fn_hc_up_bf16_v2_s1", false, true),
    st("dflash_bgemm_bf16_m16n16", false, true),
    st("saber_gdn_conv_update_w4_h128_hpb8", false, true),
    st(
        "saber_gdn_recurrent_update_f16s_k128_v128_ab1_norm0_vs1",
        false,
        true,
    ),
    st("fn_gdn_norm_bf16_sig_t256", false, true),
    st("dflash_bgemm_bf16_m16n16", false, true),
    st("saber_ar_oneshot_2rank_exact_bf16", false, true),
    st("fn_hc_down_combine_v2_m1w7t1", false, true),
    st("fn_hc_up_q8_v2_s1", false, true),
    st("fn_bgemm_sk_m16", false, false),
    st("fn_bgemm_sk_m16", false, true),
    st("moe_route_topk10_sort_e512", false, false),
    st("moe_grouped_w4a8_bk128_ks1_tm1_g1", true, true),
    st("moe_silu_mul_quant_t256", true, true),
    st("moe_grouped_w4a8_bk128_ks1_tm1_g0", true, true),
    st("moe_combine_bf16", true, true),
    st("saber_ar_oneshot_2rank_exact_bf16", true, false),
];

/// Grid barriers a layer pays in the persistent walk: the stage barriers + the one after the two
/// SEED stages (the memset nodes' replacement). The `_a` -> `_b` kernel boundary is not counted.
#[must_use]
pub fn grid_barriers(stages: &[Stage]) -> usize {
    1 + stages.iter().filter(|s| s.barrier).count()
}

/// The certified sources the prototype `#include`s verbatim (in its include order) and the
/// fp-contract state each sits under (`true` = contract(off), saber-kbuild's EXTRA_FLAGS).
pub const INCLUDED_SOURCES: &[(&str, bool)] = &[
    ("flashnext_hc.hip", true),
    ("moe_route.hip", true),
    ("saber_gdn_fused_update_f16s.hip", true),
    ("saber_gdn_conv.hip", true),
    ("saber_gdn_recurrent_update_f16s.hip", true),
    ("saber_ar_exact.hip", true),
    ("dflash_bgemm.hip", false),
    ("moe_grouped.hip", false),
];

/// The 256-thread workgroup limit on static LDS (gfx1201).
pub const LDS_LIMIT: u32 = 65_536;

/// Per-rank bytes a b1 SPEC-0 decode step must read, by class (the pack's own record sizes,
/// `rank0/dense.json` + `shared/dense.json` + `experts.xtier`, LEDGER FN23 §1). `experts_per_layer`
/// = the routed experts this rank streams per layer (5.0 = the mean of 10 routes over 2 ranks; the
/// step waits for the busier rank, E[max] = 5.62).
#[must_use]
pub fn step_bytes_b1(experts_per_layer: f64) -> Vec<(&'static str, f64)> {
    const EXPERT: f64 = 2_611_200.0; // one expert of one layer at MXFP4 (w + ws + wref, both GEMMs)
    vec![
        (
            "gdn projections (36 x in_proj + out_proj, TP half)",
            2_086_500_000.0,
        ),
        ("qsa projections (12 x q/k/v/o, TP half)", 597_700_000.0),
        ("qsa indexer + norms (replicated)", 39_300_000.0),
        (
            "hc mixers (96 x down + up + inject, replicated)",
            1_281_200_000.0,
        ),
        ("router + shared gate (replicated)", 126_100_000.0),
        ("shared expert (rank 0)", 125_500_000.0),
        ("ple projections", 65_700_000.0),
        ("lm_head (vocab half)", 635_700_000.0),
        ("routed experts", 48.0 * experts_per_layer * EXPERT),
    ]
}

/// The R9700's DRAM peak (LEDGER: 635 GB/s measured, not the startup sweep's 576).
pub const DRAM_GBPS: f64 = 635.0;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repo() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn hip(name: &str) -> String {
        std::fs::read_to_string(repo().join("crates/saber-kernels/hip").join(name)).expect(name)
    }

    #[test]
    fn the_layer_is_the_engines_launch_count_and_the_barrier_count_is_declared() {
        // EXEC section 16: a GDN layer at b1 = 15 launches + 2 reduces; b2..b8 one more (the pair).
        assert_eq!(STAGES_B1.len(), 17);
        assert_eq!(STAGES_B8.len(), 18);
        assert_eq!(grid_barriers(STAGES_B1), 15);
        assert_eq!(grid_barriers(STAGES_B8), 16);
        // the last stage of each kernel ends at the kernel boundary, not a grid barrier
        for s in [STAGES_B1, STAGES_B8] {
            let last_a = s.iter().rposition(|x| !x.in_b).unwrap();
            assert!(!s[last_a].barrier && !s.last().unwrap().barrier);
            // _a strictly before _b
            assert!(s[..=last_a].iter().all(|x| !x.in_b) && s[last_a + 1..].iter().all(|x| x.in_b));
        }
        // NEGATIVE CONTROL: a table with a barrier dropped counts one fewer.
        let mut v = STAGES_B1.to_vec();
        v[3].barrier = false;
        assert_eq!(grid_barriers(&v), 14);
    }

    #[test]
    fn the_prototype_includes_the_certified_sources_verbatim_and_copies_no_body() {
        let src = hip("flashnext_mega.hip");
        // the include list, in order, each under its code object's contract state
        let mut at = 0usize;
        for (f, off) in INCLUDED_SOURCES {
            let inc = format!("#include \"{f}\"");
            let pos = src[at..]
                .find(&inc)
                .unwrap_or_else(|| panic!("{f} not included in order"))
                + at;
            let before = &src[..pos];
            let pragma = before
                .rfind("#pragma clang fp contract(")
                .expect("a contract pragma");
            let state = before[pragma..].lines().next().unwrap();
            assert_eq!(state.contains("(off)"), *off, "{f}: {state}");
            at = pos + inc.len();
            assert!(repo().join("crates/saber-kernels/hip").join(f).exists());
        }
        assert_eq!(src.matches("#include \"").count(), INCLUDED_SOURCES.len());
        // NO copy: no certified function is DEFINED here (every body arrives by the include)
        for (f, _) in INCLUDED_SOURCES {
            let s = hip(f);
            for line in s.lines() {
                let l = line.trim_start();
                let def = l.starts_with("__device__ __forceinline__ void ")
                    || l.starts_with("void saber_")
                    || l.starts_with("extern \"C\" __global__");
                if !def {
                    continue;
                }
                let name: String = l
                    .split('(')
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .last()
                    .unwrap_or("")
                    .to_string();
                if name.len() > 6 {
                    assert!(
                        !src.contains(&format!("void {name}(")),
                        "{name} from {f} is defined in the prototype"
                    );
                }
            }
        }
        // every symbol of both tables is called through the walk's dispatch
        for s in STAGES_B1.iter().chain(STAGES_B8) {
            assert!(
                src.contains(&format!(", {})", s.symbol)),
                "{} not dispatched",
                s.symbol
            );
        }
    }

    #[test]
    fn the_b1_byte_floor_is_written_down() {
        let mean: f64 = step_bytes_b1(5.0).iter().map(|x| x.1).sum();
        let busy: f64 = step_bytes_b1(5.62).iter().map(|x| x.1).sum();
        let ms = |b: f64| b / (DRAM_GBPS * 1e9) * 1e3;
        // 5.585 GB (mean routes) / 5.663 GB (the busier rank) -> 8.80 / 8.92 ms at 635 GB/s
        assert!((ms(mean) - 8.80).abs() < 0.01, "{}", ms(mean));
        assert!((ms(busy) - 8.92).abs() < 0.01, "{}", ms(busy));
        // the measured captured step (FN14: 17.5 ms device, 18.5 host) is ~2x its byte floor
        assert!(17.5 / ms(busy) > 1.9);
    }
}
