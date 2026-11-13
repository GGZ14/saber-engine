//! MACaQ (2026-11-11): the MACaQ-W4A8 activation transform's producer twins (`--macaq-had on`).
//!
//! The full MACaQ recipe (`w4src`: a block-32 Hadamard + a per-channel scale `t` before every W4A8 GEMM)
//! puts the transform on the ACTIVATION side: every producer of a W4A8 GEMM's e4m3 input computes its
//! shipped bf16-rounded value `v`, then `x = H32 (v * t) / sqrt(32)` per 32-block along K (the fixed order
//! and arithmetic of `hip/macaq_had.h`), then its own per-token e4m3 law on `x`. Each served producer
//! symbol has ONE twin (`*_had32_*`) in a separate code object, its parent's kernarg list plus one trailing
//! pointer `had_t` (f32 [K] of this rank's columns, a pack record). With the arm off no twin is named.
//!
//! [`had_twin`] is the one map: a parent descriptor in, the twin's out (renamed, `had_t` appended at the
//! next 8-byte offset). A producer without a twin is REFUSED by name -- the arm never runs a parent
//! producer against Hadamard-domain weights.

use crate::rules::{ArgValue, LaunchDescriptor, Slot};
use crate::{Error, Result};

/// The trailing buffer every twin takes.
pub const HAD_T: &str = "had_t";

/// The pack plan's transform tag for a MACaQ-Hadamard pack (`manifest.plan.descriptor.transforms`).
pub const TRANSFORM_TAG: &str = "macaq_had32";

/// `(parent plugin, parent kernel prefix, twin plugin, twin kernel prefix)`: a parent kernel
/// `<prefix><rest>` becomes `<twin prefix><rest>`.
pub const TWINS: &[(&str, &str, &str, &str)] = &[
    (
        "mxfp4_quant",
        "add_rms_quant_t",
        "mxfp4_had",
        "add_rms_quant_had32_t",
    ),
    (
        "mxfp4_quant",
        "silu_mul_quant_t",
        "mxfp4_had",
        "silu_mul_quant_had32_t",
    ),
    (
        "mxfp4_quant",
        "gdn_norm_quant_t",
        "mxfp4_had",
        "gdn_norm_quant_had32_t",
    ),
    (
        "attn_glue",
        "attn_gate_quant_t",
        "attn_glue_had",
        "attn_gate_quant_had32_t",
    ),
    (
        "saber_ar_exact",
        "saber_ar_oneshot_2rank_exact_nq_ssqfma_bf16",
        "saber_ar_had",
        "saber_ar_oneshot_2rank_exact_nq_ssqfma_had32_bf16",
    ),
    (
        "mxfp4_gemm_decode_arnq",
        "mxfp4_gemm_decode_arnq_bk",
        "mxfp4_gemm_decode_arnq_had",
        "mxfp4_gemm_decode_arnq_had32_bk",
    ),
];

/// Every twin symbol the code objects carry (the prebuilt smoke list and the oracle read this).
pub const SYMBOLS: &[(&str, &str)] = &[
    ("mxfp4_had", "add_rms_quant_had32_t256_tiled0"),
    ("mxfp4_had", "add_rms_quant_had32_t256_tiled1"),
    ("mxfp4_had", "add_rms_quant_had32_t512_tiled0"),
    ("mxfp4_had", "add_rms_quant_had32_t512_tiled1"),
    ("mxfp4_had", "silu_mul_quant_had32_t256_tiled0"),
    ("mxfp4_had", "silu_mul_quant_had32_t256_tiled1"),
    ("mxfp4_had", "silu_mul_quant_had32_t512_tiled0"),
    ("mxfp4_had", "silu_mul_quant_had32_t512_tiled1"),
    ("mxfp4_had", "gdn_norm_quant_had32_t256"),
    ("mxfp4_had", "gdn_norm_quant_had32_t256_tiled1"),
    ("attn_glue_had", "attn_gate_quant_had32_t256_tiled0"),
    ("attn_glue_had", "attn_gate_quant_had32_t256_tiled1"),
    (
        "saber_ar_had",
        "saber_ar_oneshot_2rank_exact_nq_ssqfma_had32_bf16",
    ),
    (
        "mxfp4_gemm_decode_arnq_had",
        "mxfp4_gemm_decode_arnq_had32_bk64_ks4_tm1_wperm1_nt1",
    ),
    (
        "mxfp4_gemm_decode_arnq_had",
        "mxfp4_gemm_decode_arnq_had32_bk128_ks4_tm1_wperm1_nt1",
    ),
    (
        "mxfp4_gemm_decode_arnq_had",
        "mxfp4_gemm_decode_arnq_had32_bk128_ks2_tm1_wperm1_nt1",
    ),
];

/// The twin of a parent producer descriptor (see the module doc). Refuses, by name, a descriptor that
/// has no twin or already names `had_t`.
pub fn had_twin(d: &LaunchDescriptor) -> Result<LaunchDescriptor> {
    if d.arg(HAD_T).is_some() {
        return Err(Error::Rule(format!(
            "macaq had32: `{}` already takes `{HAD_T}`",
            d.kernel
        )));
    }
    let (plugin, kernel) = TWINS
        .iter()
        .find(|(p, k, _, _)| d.plugin == *p && d.kernel.starts_with(k))
        .map(|(_, k, tp, tk)| ((*tp).to_owned(), format!("{tk}{}", &d.kernel[k.len()..])))
        .ok_or_else(|| {
            Error::Rule(format!(
                "macaq had32: `{}` (plugin `{}`) has no Hadamard twin; `--macaq-had on` cannot run \
                 a parent producer against Hadamard-domain weights",
                d.kernel, d.plugin
            ))
        })?;
    if !SYMBOLS.iter().any(|(p, k)| *p == plugin && *k == kernel) {
        return Err(Error::Rule(format!(
            "macaq had32: `{}` maps to `{kernel}`, which `{plugin}` does not carry",
            d.kernel
        )));
    }
    let mut t = d.clone();
    let offset = d.kernarg_bytes.next_multiple_of(8);
    t.args.push(Slot {
        name: HAD_T.to_owned(),
        offset,
        size: 8,
        value: ArgValue::Buffer {
            buffer: HAD_T.to_owned(),
        },
    });
    t.kernarg_bytes = offset + 8;
    t.plugin = plugin;
    t.kernel = kernel;
    Ok(t)
}

/// Is this descriptor a twin?
#[must_use]
pub fn is_twin(d: &LaunchDescriptor) -> bool {
    SYMBOLS
        .iter()
        .any(|(p, k)| d.plugin == *p && d.kernel == *k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::{resolve, LaunchRules, Op, Shape};

    fn rules() -> LaunchRules {
        LaunchRules::prod_mxfp4().expect("prod rules")
    }

    #[test]
    fn every_served_producer_shape_has_a_twin_with_had_t_last() {
        let r = rules();
        for (op, shape) in [
            (Op::AddRmsQuant { eps: 1e-6 }, Shape::new(1, 0, 5120)),
            (Op::AddRmsQuant { eps: 1e-6 }, Shape::new(2048, 0, 5120)),
            (Op::SiluMulQuant, Shape::new(8, 8704, 0)),
            (Op::SiluMulQuant, Shape::new(2048, 8704, 0)),
            (Op::SiluMulQuant, Shape::new(8, 17408, 0)),
            (
                Op::GdnNormQuant {
                    zs: 8192,
                    eps: 1e-6,
                },
                Shape::new(8, 3072, 0),
            ),
            (
                Op::GdnNormQuant {
                    zs: 8192,
                    eps: 1e-6,
                },
                Shape::new(2048, 3072, 0),
            ),
        ] {
            let d = resolve(op, shape, &r).expect("parent");
            let t = had_twin(&d).expect("twin");
            assert!(is_twin(&t), "{}", t.kernel);
            let last = t.args.last().unwrap();
            assert_eq!(last.name, HAD_T);
            assert_eq!(last.offset % 8, 0);
            assert!(last.offset >= d.kernarg_bytes);
            assert_eq!(t.kernarg_bytes, last.offset + 8);
            assert_eq!(
                &t.args[..d.args.len()],
                &d.args[..],
                "parent slots unchanged"
            );
            assert_eq!((t.grid, t.block), (d.grid, d.block));
            assert!(had_twin(&t).is_err(), "a twin is not twinned again");
        }
    }

    #[test]
    fn a_producer_without_a_twin_is_refused_by_name() {
        let r = rules();
        let mut d = resolve(Op::AddRmsQuant { eps: 1e-6 }, Shape::new(1, 0, 5120), &r).unwrap();
        d.kernel = "saber_ar_oneshot_2rank_exact_nq_bf16".into();
        d.plugin = "saber_ar_exact".into();
        let e = had_twin(&d).unwrap_err().to_string();
        assert!(e.contains("no Hadamard twin"), "{e}");
    }
}
