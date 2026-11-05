//! Who writes a W4A8 GEMM's A side, and which layout that producer's kernel actually emits.
//!
//! **This is the one function both planes ask.** P6's prefill garbage above 512 rows was a
//! consumer re-deriving the A layout from `M` for an edge a non-fused producer had written, and
//! the fix in the exec plane was to stop guessing: `saber_exec::layer::Planner` resolves a GEMM's
//! family from its A EDGE, and refuses a producer whose kernel writes a layout its edge does not
//! declare. The plan plane (`saber_plan::ModelSpec`) had the same statement written by hand — and
//! had it backwards, `layer.attn.o_proj` carrying `.a_row_major()` where `layer.gdn.out_proj`
//! should have. Nothing read it, so nothing said so.
//!
//! A declaration that is written twice is a declaration that will disagree with itself. So both
//! planes name the PRODUCER now and ask [`AProducer::writes_tiled`] what it writes:
//!
//! * `saber_exec::layer::Planner::check_producer_layout` asks it per op at plan time, against the
//!   live rule tables, and refuses a disagreement with the graph's declared edge;
//! * `saber_plan::ModelOp::a_written_by` asks it once per row, against the committed tables, and
//!   collapses the answer into the `a_layout` selection key the registry constrains on.
//!
//! Prod answers the same question and does not use a threshold for it either: `ref_arnq.py`
//! calls `ref_mxfp4.a_tiled_register(q, M, K)` after `ar_add_rms_quant` and after
//! `silu_mul_quant` and **not** after `gdn_norm_quant`, and the GEMM's `a_tiled_take(q, M, K)`
//! *pops* that registration — a buffer nobody registered takes the row-major family however large
//! M is. The registry is a property of the producer; so is this.

use serde::{Deserialize, Serialize};

use crate::glue::GlueRules;
use crate::rules::{self, ALayout, LaunchRules, Op};
use crate::Result;

/// The producers that write a W4A8 GEMM's A-side activation edge on this tier.
///
/// Four, across two plugins: three fused quant producers in `mxfp4_quant` and the attention
/// output gate in `attn_glue`. Every W4A8 GEMM in Qwen3.8-27B reads an edge one of these wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AProducer {
    /// The fused residual-add + RMS norm + per-token quant. Feeds every QKV/in_proj/gate_up
    /// linear, and the head.
    AddRmsQuant,
    /// The fused SwiGLU + per-token quant. Feeds `mlp.down_proj`.
    SiluMulQuant,
    /// The GDN gated RMS norm + per-token quant. Feeds `gdn.out_proj` — and it is the one
    /// producer here with a single instantiation (`gdn_norm_quant_t{block}`, whose store is
    /// `q += row * N`), so it writes ROW-MAJOR AT EVERY M. The reference has one too:
    /// `saber_gdn_norm_quant` is templated on the thread count alone, unlike
    /// `saber_add_rms_quant` and `saber_silu_mul_quant` beside it in the same file.
    GdnNormQuant,
    /// `core * sigmoid(gate)` with the per-token quant `attn.o_proj` reads. A FUSED producer:
    /// `attn_gate_quant_t256_tiled{0,1}` exist and `rules/attn_glue_gfx1201.toml` keys them on
    /// the same band edge, so this edge follows M.
    AttnGateQuant,
}

impl AProducer {
    /// Does this producer's kernel write the fragment-tiled layout at `m`, under these tables?
    ///
    /// The mxfp4 producers go through [`rules::producer_writes_tiled`], which is the same call
    /// `saber_exec::layer` makes; the gate goes through [`GlueRules::tiled`], which is the glue
    /// plugin's own answer. Nothing here re-derives a threshold.
    #[must_use]
    pub fn writes_tiled(self, m: u32, mxfp4: &LaunchRules, glue: &GlueRules) -> bool {
        // `eps` and `zs` are kernarg values, not selection keys: they reach the descriptor and
        // never the layout decision, so any legal value asks the same question here. The match
        // is over the VARIANT, which is what `producer_writes_tiled` reads.
        let op = match self {
            AProducer::AddRmsQuant => Some(Op::AddRmsQuant { eps: 0.0 }),
            AProducer::SiluMulQuant => Some(Op::SiluMulQuant),
            AProducer::GdnNormQuant => Some(Op::GdnNormQuant { zs: 0, eps: 0.0 }),
            AProducer::AttnGateQuant => None,
        };
        match op {
            Some(op) => rules::producer_writes_tiled(op, m, mxfp4).unwrap_or(false),
            None => glue.tiled_under(m, rules::ATiledLaw::of(mxfp4)),
        }
    }

    /// The `a_layout` selection key this producer's edge carries, over the **committed** tables.
    ///
    /// Two-valued, because that is all the key can say: `FollowsM` for a producer that switches
    /// at [`rules::a_tiled_min_m`], `RowMajor` for one that does not switch at all. A producer
    /// that wrote the tiled layout at EVERY M would be neither, and there is none — see
    /// `a_producer_layouts_are_derived_from_the_band` for the assertion, which is where a fifth
    /// producer with a different band would fail rather than be silently collapsed.
    ///
    /// # Errors
    /// The committed tables failing to parse, which no build reaches.
    pub fn a_layout(self) -> Result<ALayout> {
        let mxfp4 = LaunchRules::prod_mxfp4()?;
        let glue = GlueRules::prod()?;
        Ok(
            if self.writes_tiled(rules::a_tiled_min_m(), &mxfp4, &glue) {
                ALayout::FollowsM
            } else {
                ALayout::RowMajor
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [AProducer; 4] = [
        AProducer::AddRmsQuant,
        AProducer::SiluMulQuant,
        AProducer::GdnNormQuant,
        AProducer::AttnGateQuant,
    ];

    /// The derivation, and its control: a producer's layout is read off the band, and the band is
    /// the one number. A producer whose kernel switched somewhere else would show up here as a
    /// `writes_tiled` that is true below the edge or false above it, and the collapsed
    /// `a_layout` would be a lie rather than a summary.
    #[test]
    fn a_producer_layouts_are_derived_from_the_band() {
        let mxfp4 = LaunchRules::prod_mxfp4().expect("mxfp4 rules");
        let glue = GlueRules::prod().expect("glue rules");
        let edge = rules::a_tiled_min_m();
        assert!(edge > 1, "the band edge is a real M");
        for p in ALL {
            let layout = p.a_layout().expect("committed tables");
            assert!(
                !p.writes_tiled(edge - 1, &mxfp4, &glue),
                "{p:?} writes the tiled layout BELOW the band edge {edge}"
            );
            for m in [edge, edge + 1, 8192] {
                assert_eq!(
                    p.writes_tiled(m, &mxfp4, &glue),
                    layout == ALayout::FollowsM,
                    "{p:?} at M={m} disagrees with its own collapsed a_layout"
                );
            }
        }
    }

    /// `gdn_norm_quant` has a single instantiation in the reference, so under the reference's
    /// law it was the one producer row-major at every M. Saber's tiled store (G2,
    /// `gdn_norm_tiled`) makes it switch at the band edge like the other three; the control is
    /// the knob off, where it is row-major at every M again, named rather than counted.
    #[test]
    fn every_producer_follows_the_band_and_the_gdn_norm_only_under_its_knob() {
        let row_major: Vec<AProducer> = ALL
            .into_iter()
            .filter(|p| p.a_layout().expect("committed tables") == ALayout::RowMajor)
            .collect();
        assert_eq!(row_major, Vec::<AProducer>::new());
        let mut mxfp4 = LaunchRules::prod_mxfp4().expect("mxfp4 rules");
        mxfp4.knobs.gdn_norm_tiled = false;
        let glue = GlueRules::prod().expect("glue rules");
        let row_major_off: Vec<AProducer> = ALL
            .into_iter()
            .filter(|p| {
                [rules::a_tiled_min_m(), 2048, 8192]
                    .iter()
                    .all(|&m| !p.writes_tiled(m, &mxfp4, &glue))
            })
            .collect();
        assert_eq!(row_major_off, vec![AProducer::GdnNormQuant]);
    }
}
