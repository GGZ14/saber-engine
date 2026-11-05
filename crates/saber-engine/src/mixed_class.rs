//! **The fused mixed step's decode-row class, served as a proxy** (workstream C8).
//!
//! A fused mixed step (LEDGER C8, `docs/EXEC.md` § the mixed step) runs ONE forward over the
//! chunk's rows and the decode rows: every row-local op once over `M = chunk + decode` rows at the
//! mixed rung's plan, attention / GDN / head / drafter as two launches over disjoint rows with each
//! side's own kernels. The class census (`tests/mixed_c8.rs`) proves what that does to a decode
//! row: every GEMM it passes through sums in the fold class (`rules::fold_class`: the folded / A-tiled
//! families at the mixed M, identical to the decode kernel at ONE split), where its own decode step
//! splits K on six of the seven layer GEMMs (`attn.qkv`, `gdn.in_proj_qkvz` / `_ba`, and the three
//! reduced ones), and every reduce it passes through is the lossy `wht` family, where its own decode
//! step reduces EXACTLY at <= 12 rows (under `quant_min_bytes`). Attention, GDN, the head and the
//! drafter keep their own kernels; the chunk rows keep their class below the 2048 rung.
//!
//! So the decode rows' numbers in a fused mixed step are the numbers of a DECODE step planned under
//! these two transforms, and that decode step can be served today, on every decode step, without
//! the fused forward: [`mixed_proxy_rules`] (every decode-band cell at one split) and
//! [`mixed_proxy_saber`] (the reduce's lossy family at every size), with the decode launch-train
//! fusions off (they are exact compositions whose `_nq` / `arnq` halves exist only at the split
//! cells and the exact reduce). Served on EVERY step it over-exposes: a fused step touches only
//! the decode rows of the steps that carry a chunk. A quality gate that passes on the proxy passes
//! on the fused form a fortiori; one that fails bounds it. Nothing here is a serving candidate.

use saber_kernels::rules::{BandRow, LaunchRules};
use saber_kernels::saber::SaberRules;

/// The committed rules with every decode-band cell at or below `decode_max_m` at ONE split (`bk`
/// from the tile height, as the committed `m_max = 64` row derives it). The prefill rules, the
/// rules file and every row above `decode_max_m` are untouched.
#[must_use]
pub fn mixed_proxy_rules(base: &LaunchRules) -> LaunchRules {
    let mut r = base.clone();
    r.name = format!("{}+c8-mixed-proxy-ks1", base.name);
    // First match wins: this row shadows every committed row at M <= decode_max_m.
    r.decode_band.insert(
        0,
        BandRow {
            m_max: Some(base.knobs.decode_max_m),
            ks: Some(1),
            bk_from_tm: true,
            ..BandRow::default()
        },
    );
    r
}

/// The saber rules with the reduce's lossy (`wht`) family at every message size: the family a
/// mixed step's reduce takes, since its message is `M >= 96` rows (>= 983,040 B) and always above
/// `quant_min_bytes`. The lossy family's other conditions (16-bit payload, whole groups) are kept.
#[must_use]
pub fn mixed_proxy_saber(base: &SaberRules) -> SaberRules {
    let mut r = base.clone();
    r.name = format!("{}+c8-mixed-proxy-lossy-ar", base.name);
    r.ar.quant_min_bytes = 0;
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use saber_kernels::rules::{fold_class, resolve, Op, Shape};

    fn kernel(r: &LaunchRules, m: u32, n: u32, k: u32) -> String {
        resolve(Op::gemm(), Shape { m, n, k }, r)
            .expect("resolves")
            .kernel
    }

    /// Every TP=2 layer linear at every decode M: the committed band splits some of them (the
    /// census's move), the proxy puts every one in the fold class — the class the fused step's
    /// mixed plan binds them to.
    #[test]
    fn the_proxy_puts_every_decode_gemm_in_the_fold_class() {
        let base = LaunchRules::prod_mxfp4().expect("rules");
        let proxy = mixed_proxy_rules(&base);
        // (N, K) of the TP=2 layer linears: qkv, o, qkvz, ba, out, gate_up, down.
        let shapes = [
            (7168u32, 5120u32),
            (5120, 3072),
            (8192, 5120),
            (48, 5120),
            (5120, 3072),
            (17408, 5120),
            (5120, 8704),
        ];
        let mut split = 0;
        for m in 1..=64u32 {
            for (n, k) in shapes {
                let committed = kernel(&base, m, n, k);
                if fold_class(&committed).is_none() {
                    split += 1;
                }
                let p = kernel(&proxy, m, n, k);
                assert!(
                    fold_class(&p).is_some(),
                    "M={m} N={n} K={k}: the proxy bound {p}, outside the fold class"
                );
            }
        }
        // The negative control: the committed band is NOT in the class at these cells, or the
        // proxy would be the default and prove nothing.
        assert!(split > 64, "the committed band splits only {split} cells");
        // Above decode_max_m nothing moves.
        for (n, k) in shapes {
            assert_eq!(kernel(&base, 256, n, k), kernel(&proxy, 256, n, k));
        }
    }

    /// The reduce: a b1s8 message (8 x 5120 bf16 = 80 KiB) is exact under the committed rules and
    /// lossy under the proxy; a 96-row one is lossy under both.
    #[test]
    fn the_proxy_reduces_lossy_at_every_size() {
        use saber_kernels::saber::{ArDtype, ArShape};
        let base = SaberRules::prod_saber().expect("saber");
        let proxy = mixed_proxy_saber(&base);
        let at = |rows: u64| ArShape {
            n_elem: rows * 5120,
            elem_bytes: 2,
            world_size: 2,
        };
        assert!(!base.ar_wants_lossy(&at(8), ArDtype::Bf16));
        assert!(proxy.ar_wants_lossy(&at(8), ArDtype::Bf16));
        assert!(proxy.ar_wants_lossy(&at(1), ArDtype::Bf16));
        assert!(base.ar_wants_lossy(&at(96), ArDtype::Bf16));
        assert!(proxy.ar_wants_lossy(&at(96), ArDtype::Bf16));
    }
}
