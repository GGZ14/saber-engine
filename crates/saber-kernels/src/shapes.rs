//! The production GEMM shapes, and the M grid the launch rules are gated on.
//!
//! The shape list is read off a running server (`SABER_MXFP4_MHIST`), not derived from the
//! model config: the config-derived guess had the fused qkv at N=4096 and missed 7168 and 8192
//! completely, which is exactly the band where DKS=2 lives. Deriving this list by hand got it
//! wrong once; the kernel source says not to do it again.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProdShape {
    pub id: &'static str,
    pub n: u32,
    pub k: u32,
}

/// The shapes P0 gates on (TP=1 widths unless the id says otherwise).
pub const PROD_SHAPES: &[ProdShape] = &[
    ProdShape {
        id: "n10240_k5120",
        n: 10240,
        k: 5120,
    },
    ProdShape {
        id: "n6144_k5120",
        n: 6144,
        k: 5120,
    },
    ProdShape {
        id: "n5120_k6144",
        n: 5120,
        k: 6144,
    },
    ProdShape {
        id: "n12288_k5120",
        n: 12288,
        k: 5120,
    },
    ProdShape {
        id: "n1024_k5120",
        n: 1024,
        k: 5120,
    },
    ProdShape {
        id: "gate_up_n17408_k5120",
        n: 17408,
        k: 5120,
    },
    ProdShape {
        id: "down_n5120_k17408",
        n: 5120,
        k: 17408,
    },
    ProdShape {
        id: "lm_head_n248320_k5120",
        n: 248320,
        k: 5120,
    },
];

/// M values the decode/prefill bands are gated on: the split-K band edges (1, 8, 24/25, 64/65,
/// 112/113, 128) and one prefill-class M.
pub const M_GRID: &[u32] = &[1, 8, 24, 25, 64, 65, 112, 113, 128, 2048];

/// Every shape with its TP=2 shards. A column-parallel linear shards N, a row-parallel one shards
/// K; both appear in the model, so both halves are gated.
pub fn all_with_tp2_shards() -> Vec<ProdShape> {
    let mut out: Vec<ProdShape> = PROD_SHAPES.to_vec();
    for s in PROD_SHAPES {
        if s.n % 32 == 0 {
            out.push(ProdShape {
                id: leak(format!("{}_tp2n", s.id)),
                n: s.n / 2,
                k: s.k,
            });
        }
        if (s.k / 2) % 64 == 0 {
            out.push(ProdShape {
                id: leak(format!("{}_tp2k", s.id)),
                n: s.n,
                k: s.k / 2,
            });
        }
    }
    out
}

/// Shape ids are `&'static str` because they are constants everywhere else; the two derived
/// shards per shape are built once per process and never freed.
fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shards_keep_the_kernels_alignment_rules() {
        for s in all_with_tp2_shards() {
            assert_eq!(
                s.k % 64,
                0,
                "{}: K must be a multiple of the 64-deep slab",
                s.id
            );
            assert!(s.n > 0);
        }
    }
}
