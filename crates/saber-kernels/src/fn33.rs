//! FN33: the Flash-Next MTP drafter's head at DRAFT precision -- the 27B's int2 draft head
//! (`dflash_head.hip`'s `_lx` scorer, `dflash_topk.hip`'s exact top-k) + an EXACT bf16 rerank of
//! the top-K candidates by the target head's own GEMM symbol, then FN22's device chain unchanged.
//!
//! The draft token of a pass is the argmax, over the int2 head's top-K candidates of the rank's
//! vocabulary half, of the candidates' EXACT scores -- the bf16 head's GEMM (the route of the pass's
//! row count, `FlashNextRules::bgemm_route("head.tp2", ..)`) on the candidates' weight rows gathered
//! into a `[rows x K, H]` block: one wave walks each 16-column group's K in ascending 16-k steps
//! into one fp32 accumulator whatever N or the column's position is, so each candidate's score is
//! the full head's logit bit for bit (gated on the card: `fnchaincheck` score diffs 0). The
//! drafter's numerics change only WHICH tokens are proposed; the target's verify step decides every
//! committed token (F40), so this is a speed lever measured by acceptance.
//!
//! The int2 pack (quantised ONCE at boot on the card from the rank's bf16 head half, prod's
//! `_quantize_head_now` law: asymmetric min/max at 2 bits per (row, group of 128), quarter-split
//! packing, premultiplied bf16 scales `S = 4 sc`, `ZS = 4 sc + zp sc`) is the ledger class
//! `fn.draft.int2` ([`pack_bytes`], 85.3 MiB a rank at TP=2; the exact ledger takes it out of the
//! expert slots' remainder). [`quant_row`] is the host law the boot check compares sample rows with.

/// Hidden size (the head's K).
pub const K: usize = 2560;
/// The quantisation group along K.
pub const G: usize = 128;
/// Groups per row.
pub const NG: usize = K / G;
/// Packed bytes per row (2 bits a weight, quarter-split: byte j carries k = j, Q + j, 2Q + j, 3Q + j).
pub const ROW_BYTES: usize = K / 4;
/// Candidates the int2 scorer emits per 64-column block (the 27B's KCAND).
pub const KC: u32 = 8;
/// The scorer's column block.
pub const BLOCK_N: u32 = 64;
/// The widest top-K the exact top-k kernel serves (`dflash_topk_f32_gather_k80`).
pub const KMAX: u32 = 80;
/// The default K: the 27B's law -- a top-k caller needs 4x its selector's top-k (RERANK 20 -> 80);
/// the drafter's caller is an argmax, and K is the rerank ceiling's knob (LEDGER FN33).
pub const DEFAULT_K: u32 = 80;
/// Rows of the scorer's padded input (one WMMA M tile; the drafter's selected rows are <= 8).
pub const XROWS: u32 = 16;
/// Rows of the padded input buffer (the rerank GEMM's widest row tile reads up to 32).
pub const XBUF_ROWS: u32 = 64;
/// Launches the int2 tail adds before FN22's pack: x gather, group sums, the int2 scorer, the
/// top-k threshold, the top-k gather, the candidate-column gather, the weight-row gather, the
/// rerank GEMM, the candidates' argmax.
pub const PRE_LAUNCHES: u32 = 9;
/// FN22's tail after its argmax part: pack, the exact reduce, select.
pub const POST_LAUNCHES: u32 = 3;

/// `--fn-draft-head`: the drafter's head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DraftHead {
    /// The shared bf16 head over the whole vocabulary half (FN22's pass: the control).
    #[default]
    Bf16,
    /// The int2 scorer + the exact bf16 rerank of its top-K.
    Int2,
}

impl std::str::FromStr for DraftHead {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim() {
            "bf16" => Ok(DraftHead::Bf16),
            "int2" => Ok(DraftHead::Int2),
            t => Err(format!("--fn-draft-head bf16 | int2, not {t}")),
        }
    }
}

impl std::fmt::Display for DraftHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DraftHead::Bf16 => "bf16",
            DraftHead::Int2 => "int2",
        })
    }
}

/// The int2 pack's bytes for `n` head rows: codes + the two bf16 scale planes.
#[must_use]
pub fn pack_bytes(n: u64) -> u64 {
    n * (ROW_BYTES as u64 + 2 * NG as u64 * 2)
}

/// Column blocks of an `n`-row head half.
#[must_use]
pub fn nblk(n: u32) -> u32 {
    n.div_ceil(BLOCK_N)
}

/// The rerank GEMM's column count for `rows` selected rows at `k` candidates each (the GEMM's
/// 64-column tile).
#[must_use]
pub fn rerank_cols(rows: u32, k: u32) -> u32 {
    (rows * k).div_ceil(64) * 64
}

fn bf16_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// Round-to-nearest-even f32 -> bf16 (finite inputs; torch's `.to(bfloat16)`).
#[must_use]
pub fn f32_bf16(v: f32) -> u16 {
    let a = v.to_bits();
    let a = a.wrapping_add(0x7fff + ((a >> 16) & 1));
    (a >> 16) as u16
}

/// The host law of one row's pack (the kernel `fn_draft_head_int2_quant`, element for element):
/// `(codes [ROW_BYTES], S [NG], ZS [NG])`.
#[must_use]
pub fn quant_row(w: &[u16]) -> (Vec<u8>, Vec<u16>, Vec<u16>) {
    assert_eq!(w.len(), K);
    let q4 = K / 4;
    let mut codes = vec![0u8; ROW_BYTES];
    let mut s = vec![0u16; NG];
    let mut zs = vec![0u16; NG];
    for g in 0..NG {
        let x = &w[g * G..(g + 1) * G];
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &b in x {
            let v = bf16_f32(b);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let sc = ((hi - lo) / 3.0).max(1e-8);
        let zp = (-lo / sc).round_ties_even().clamp(0.0, 3.0);
        for (i, &b) in x.iter().enumerate() {
            let q = (bf16_f32(b) / sc + zp).round_ties_even().clamp(0.0, 3.0) as u8;
            let k = g * G + i;
            let (quarter, j) = (k / q4, k % q4);
            codes[j] |= q << (2 * quarter);
        }
        s[g] = f32_bf16(4.0 * sc);
        zs[g] = f32_bf16(4.0 * sc + zp * sc);
    }
    (codes, s, zs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pack_is_85_mib_a_rank_and_the_default_k_is_the_27bs() {
        assert_eq!(pack_bytes(124_160), 89_395_200);
        assert_eq!(nblk(124_160), 1940);
        assert_eq!(DEFAULT_K, 4 * 20);
        const { assert!(DEFAULT_K <= KMAX) };
        assert_eq!(rerank_cols(1, 80), 128);
        assert_eq!(rerank_cols(8, 80), 640);
        assert_eq!("int2".parse::<DraftHead>(), Ok(DraftHead::Int2));
        assert_eq!(DraftHead::default().to_string(), "bf16");
    }

    #[test]
    fn a_row_dequantises_within_half_a_step() {
        let mut st = 11u64;
        let w: Vec<u16> = (0..K)
            .map(|_| {
                st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                f32_bf16(((st >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1)
            })
            .collect();
        let (c, s, zs) = quant_row(&w);
        let q4 = K / 4;
        for k in 0..K {
            let g = k / G;
            let q = (c[k % q4] >> (2 * (k / q4))) & 3;
            let sc = bf16_f32(s[g]) / 4.0;
            let zp = (bf16_f32(zs[g]) - bf16_f32(s[g])) / sc;
            let deq = (f32::from(q) - zp) * sc;
            assert!((deq - bf16_f32(w[k])).abs() <= 0.52 * sc + 1e-3, "k {k}");
        }
    }
}
