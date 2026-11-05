//! **The small prefill rungs on the decode GEMM family** (workstream C6).
//!
//! A prefill chunk runs every layer linear at its BUCKET's M (C5's row law), and below
//! `a_tiled_min_m` that is the FOLDED kernel, whose M tile is `folded_bmf` = 256 rows: a 72-row
//! chunk planned at p96 streams every weight once and computes 256 rows of WMMA per column block,
//! exactly what p256 computes. The 64-row decode step instead runs the split-K decode kernel, one
//! workgroup per 128 columns holding ALL of its M rows (`TM = ceil(M / 16)` tiles, to
//! `dec_mtile * dec_max_tm` = 128 rows), so its cost is the weight stream and not the tile.
//!
//! The rules already describe that kernel above 64 rows: `[[decode_band]]`'s `m_min = 65` rows are
//! the reference launcher's own `M > 64` ladder (`launch_impl`, measured cell by cell to M = 128),
//! reachable in prod by raising `SABER_MXFP4_DECODE_MAX_M`. What this module does is build that
//! variant FOR THE PREFILL PLANS AT OR BELOW `max_m` ROWS ONLY — the decode plans, the rungs above,
//! the committed rules file and every other op keep their own rules (G2 owns the prefill GEMM
//! rules; nothing here edits `rules/mxfp4_gfx1201.toml`).
//!
//! **Two classes, and which one is exact.** The decode kernel at ONE split accumulates K in
//! ascending order in fp32, the folded kernel's order (the `lm_head` cell in
//! `saber_kernels::rules`: bit-identical at every M measured), so [`SmallChunkClass::Exact`] forces
//! `ks = 1` at every shape (the band's `bk` kept: 64 at `nblk >= 110`, 128 elsewhere). The band as
//! the reference ships it ([`SmallChunkClass::Band`]) splits `down` (2 or 4) and the 48-wide GDN
//! gate (4): a reassociated sum, i.e. a NUMERICS change, gated on quality like C5's ladder.

use saber_kernels::rules::{ATiledLaw, BandRow, LaunchRules};

/// Which decode-band law the small rungs take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmallChunkClass {
    /// Every shape at ONE split: the folded kernel's summation order, so the chunk's arithmetic
    /// is unchanged (gated bit-exact, `step-tp2 --small-chunk-gemm`).
    Exact,
    /// The reference's `M > 64` band as it stands (splits on `down` and the GDN gate): the
    /// numerics class moves for those shapes.
    Band,
}

/// The rules a prefill rung of at most `max_m` rows is planned under: the committed rules with
/// `decode_max_m` raised to `max_m` and, for [`SmallChunkClass::Exact`], every band row above the
/// committed `decode_max_m` at one split.
///
/// # Errors
/// A `max_m` that the decode kernel cannot hold (above `dec_mtile * dec_max_tm`), that is not above
/// the committed `decode_max_m` (nothing would move), or that reaches `a_tiled_min_m` (the rung
/// would then be an A-tiled one, whose producers write another layout).
pub fn small_chunk_rules(
    base: &LaunchRules,
    max_m: u32,
    class: SmallChunkClass,
) -> Result<LaunchRules, String> {
    let k = &base.knobs;
    let widest = k.dec_mtile * k.dec_max_tm;
    if max_m <= k.decode_max_m || max_m > widest {
        return Err(format!(
            "--small-chunk-gemm {max_m}: the decode kernel holds {} < M <= {widest} rows above \
             the committed decode band",
            k.decode_max_m
        ));
    }
    // The layout law's lowest tiled M: `a_tiled_min_m`, or G6's mid band edge when it is on.
    let edge = ATiledLaw::of(base).edge();
    if edge != 0 && max_m >= edge {
        return Err(format!(
            "--small-chunk-gemm {max_m} reaches the layout law's edge {edge}: that rung's producers \
             write the fragment-tiled layout the decode kernel does not read"
        ));
    }
    let mut r = base.clone();
    r.name = format!("{}+c6-small-chunk-{max_m}-{class:?}", base.name);
    let first = k.decode_max_m + 1;
    r.knobs.decode_max_m = max_m;
    if class == SmallChunkClass::Exact {
        // First match wins, so these two rows shadow every committed row above the old edge: the
        // gate_up cell keeps the band's BK = 64 (prod's measurement at M = 128), the rest BK = 128.
        let pre = [
            BandRow {
                m_min: Some(first),
                nblk_min: Some(110),
                ks: Some(1),
                bk: Some(64),
                ..BandRow::default()
            },
            BandRow {
                m_min: Some(first),
                ks: Some(1),
                bk: Some(128),
                ..BandRow::default()
            },
        ];
        r.decode_band.splice(0..0, pre);
    }
    Ok(r)
}

/// The rules a prefill plan of `rows` rows takes: the small-chunk variant at or below its edge,
/// the committed rules otherwise. `small` is `None` when the option is off.
#[must_use]
pub fn rules_for_rows<'r>(
    base: &'r LaunchRules,
    small: Option<&'r (u32, LaunchRules)>,
    rows: u32,
) -> &'r LaunchRules {
    match small {
        Some((max_m, r)) if rows <= *max_m => r,
        _ => base,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use saber_kernels::rules::{resolve, Op, Shape};

    fn kernel(r: &LaunchRules, m: u32, n: u32, k: u32) -> String {
        resolve(Op::gemm(), Shape { m, n, k }, r)
            .expect("resolves")
            .kernel
    }

    /// Every TP=2 layer linear at the small rungs: the committed rules bind the folded tile, the
    /// exact variant the decode kernel at ONE split, the band variant the reference's splits.
    /// The NEGATIVE CONTROL is the band: it must split `down` and the 48-wide gate, or the two
    /// classes would be the same plan and the exact gate would prove nothing about a split.
    #[test]
    fn the_small_rungs_bind_the_decode_kernel_at_one_split() {
        let base = LaunchRules::prod_mxfp4().expect("rules");
        let exact = small_chunk_rules(&base, 128, SmallChunkClass::Exact).expect("exact");
        let band = small_chunk_rules(&base, 128, SmallChunkClass::Band).expect("band");
        let shapes = [
            (17408, 5120),
            (5120, 8704),
            (8192, 5120),
            (7168, 5120),
            (5120, 3072),
            (48, 5120),
        ];
        for m in [96u32, 128] {
            let mut split_seen = 0;
            for (n, k) in shapes {
                assert!(
                    kernel(&base, m, n, k).starts_with("mxfp4_gemm_folded"),
                    "committed rules at M={m} n={n}"
                );
                let e = kernel(&exact, m, n, k);
                assert!(
                    e.starts_with("mxfp4_gemm_decode_bk") && e.contains("_ks1_"),
                    "exact at M={m} n={n}: {e}"
                );
                assert!(e.contains(&format!("_tm{}_", m.div_ceil(16))), "{e}");
                let b = kernel(&band, m, n, k);
                assert!(b.starts_with("mxfp4_gemm_decode_bk"), "{b}");
                if !b.contains("_ks1_") {
                    split_seen += 1;
                }
            }
            assert_eq!(split_seen, 2, "the band splits down and the gate at M={m}");
        }
        // Above the edge nothing moves, and the decode band below 65 rows is the committed one.
        for (n, k) in shapes {
            assert_eq!(kernel(&exact, 192, n, k), kernel(&base, 192, n, k));
            for m in [8u32, 40, 64] {
                assert_eq!(
                    kernel(&exact, m, n, k),
                    kernel(&base, m, n, k),
                    "M={m} n={n}"
                );
            }
        }
        assert_eq!(
            exact.knobs.decode_max_m, 128,
            "the variant's edge is the option's"
        );
        assert_eq!(
            base.knobs.decode_max_m, 64,
            "the committed rules are untouched"
        );
    }

    #[test]
    fn an_edge_the_kernel_cannot_hold_is_refused() {
        let base = LaunchRules::prod_mxfp4().expect("rules");
        for bad in [0u32, 64, 129, 256] {
            assert!(
                small_chunk_rules(&base, bad, SmallChunkClass::Exact).is_err(),
                "{bad}"
            );
        }
        let s = (96u32, base.clone());
        assert!(std::ptr::eq(rules_for_rows(&base, Some(&s), 96), &s.1));
        assert!(std::ptr::eq(rules_for_rows(&base, Some(&s), 128), &base));
        assert!(std::ptr::eq(rules_for_rows(&base, None, 96), &base));
    }
}
