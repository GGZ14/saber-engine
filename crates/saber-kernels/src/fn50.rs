//! FN50: Qwen's FIRST-PARTY fp8 MTP experts as a gated option (`--fn-mtp-experts fp8`).
//!
//! Qwen's fp8 file carries the MTP layer's 512 ROUTED experts as e4m3 block-128 weights with bf16
//! `weight_scale_inv`; the MTP shared expert is bf16 in every checkpoint. The drafter's xptr table
//! is therefore MIXED -- a routed entry is `[preshuffled e4m3 weight, fp32 [N/128, K/128] scales]`,
//! the shared entry `[bf16 weight, 0]` -- and each grouped GEMM is two launches over the SAME route
//! tiles (`flashnext_mtp8.hip`, derived from FN6's bodies by `tools/gen_fn50_mtp8.py`, one inserted
//! tile-skip line each): the W8A8 body exits on a bf16 tile, the bf16 split-K body on an fp8 tile.
//! The A sides are FN6's: `dflash_act_quant_g128` over the token rows before gate_up and over the
//! compact silu rows before down (Qwen's own inference form: dynamic per-128 activations).
//!
//! The MoE block of the fp8 arm: router GEMM, shared-gate GEMV, route (3, unchanged) + quant,
//! gate_up fp8, gate_up bf16 (shared), silu, quant, down fp8, down bf16 (shared), combine = 11.

/// The code object.
pub const PLUGIN: &str = "flashnext_mtp8";
/// The activation quant's code object (FN6's / the drafter's).
pub const QUANT_PLUGIN: &str = "dflash_mlp";
/// The per-128 activation quant (`absmax / 448` as a division, FN6's spec).
pub const ACT_QUANT_SYMBOL: &str = "dflash_act_quant_g128";
/// Threads per workgroup of the activation quant (4 waves; one wave = one 128-group).
pub const ACT_QUANT_THREADS: u64 = 128;

/// The W8A8 mixed-table body (`gather` = gate_up, A rows by token).
#[must_use]
pub const fn fp8_symbol(gather: bool) -> &'static str {
    if gather {
        "fn_moe_a8w8_bs128_mx_g1"
    } else {
        "fn_moe_a8w8_bs128_mx_g0"
    }
}

/// The bf16 mixed-table body (the shared expert's tiles only).
#[must_use]
pub const fn bf16_symbol(gather: bool) -> &'static str {
    if gather {
        "fn_moe_bgemm_sk_m16_mx_g1"
    } else {
        "fn_moe_bgemm_sk_m16_mx_g0"
    }
}

/// Launches of the fp8 arm's MoE block (the bf16 arm's 7 + 2 quants + 2 bf16 shared launches).
pub const MOE_BLOCK_LAUNCHES_FP8: usize = crate::moe::LAUNCHES_PER_MOE_BLOCK + 4;

/// Workgroups of the activation quant over `rows x n` (one wave per 128-group, 4 waves a group).
#[must_use]
pub const fn act_quant_blocks(rows: u64, n: u64) -> u64 {
    (rows * (n / 128)).div_ceil(4)
}

/// Bytes of one routed expert in the fp8 pack file: gate_up e4m3 `[1280, 2560]` + its fp32
/// scales `[10, 20]` + down e4m3 `[2560, 640]` + its fp32 scales `[20, 5]`.
pub const EXPERT_BYTES_FP8: u64 = crate::fn6::MTP_EXPERT_BYTES_FP8;
/// The bf16 arm's bytes per routed expert.
pub const EXPERT_BYTES_BF16: u64 = crate::fn6::MTP_EXPERT_BYTES_BF16;

/// Every symbol of the code object.
#[must_use]
pub fn symbols() -> Vec<&'static str> {
    vec![
        fp8_symbol(true),
        fp8_symbol(false),
        bf16_symbol(true),
        bf16_symbol(false),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mixed_table_object_declares_its_four_symbols_and_the_quant_is_fn6s() {
        let m = crate::manifest::Manifest::load(
            crate::prebuilt::dir().join(format!("{PLUGIN}.manifest.toml")),
        )
        .unwrap();
        let mut have: Vec<&str> = m.kernel.iter().map(|k| k.name.as_str()).collect();
        have.sort_unstable();
        let mut want = symbols();
        want.sort_unstable();
        assert_eq!(have, want);
        let q = crate::manifest::Manifest::load(
            crate::prebuilt::dir().join(format!("{QUANT_PLUGIN}.manifest.toml")),
        )
        .unwrap();
        assert!(q.kernel.iter().any(|k| k.name == ACT_QUANT_SYMBOL));
        // half the bf16 bytes plus the scales
        assert_eq!(EXPERT_BYTES_FP8, 4_916_400);
        assert_eq!(EXPERT_BYTES_BF16, 9_830_400);
        assert_eq!(MOE_BLOCK_LAUNCHES_FP8, 11);
        assert_eq!(act_quant_blocks(1, 2560), 5);
        assert_eq!(act_quant_blocks(176, 640), 220);
    }
}
