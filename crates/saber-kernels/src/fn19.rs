//! Flash-Next QSA sparse attention (FN19): the selection law, the step law and the geometry of the
//! `flashnext_qsa` / `flashnext_qsa_attn` code objects (docs/EXEC.md §18, LEDGER FN19).
//!
//! **The model's law** (upstream `upstream/models/qwen4_exp/amd/{indexer_qsa,qsa}.py`,
//! `amd/ops/qsa.py`): per query row at position `p`, the indexer scores every VISIBLE compressed
//! group `g < (p + 1) / 4` as `sum_h relu(q_h . k_g) / sqrt(128)` over its 4 query heads; a row with
//! at most [`TOPK`] visible groups takes all of them (QSA == dense attention there), above it the
//! [`TOPK`] best (exact ties at the 512th value: the lower group -- Saber's refinement of upstream's
//! arrival-order radix select); the row attends the selected groups' 4 tokens each plus its open
//! group's causal tail `[4 * ((p + 1) / 4), p]` ([`WIDTH`] = 2,051 tokens at most, no sink).
//!
//! **The step law** ([`sparse_step`]): a prefill chunk whose LAST row sits at position >= 2,051
//! (some row sees more than 512 groups) runs the sparse pipeline; every other chunk runs FN6's dense
//! prefill body (== the model: every row's selection is the identity), byte-identical to FN12-FN17.
//! A decode / verify step runs the pipeline whenever the pools' context exceeds 2,051 (the graphs
//! are captured once per bucket): a sequence whose rows all sit at positions <= 2,050 selects the
//! identity, and the gather body over the identity list at the dense route's split count is the
//! dense body's bytes (the byte oracle, GATES FN19), so short sequences keep FN13's bytes.
//! `--fn-qsa dense` is the control arm (today's dense attention over every prior token).

use crate::{Error, Result};

/// Compressed groups selected per query row (`indexer_budget / indexer_compress_ratio`).
pub const TOPK: u32 = 512;
/// Tokens per compressed group (`indexer_compress_ratio`).
pub const RATIO: u32 = 4;
/// The expanded index row: 512 x 4 + the open group's tail (<= 3).
pub const WIDTH: u32 = TOPK * RATIO + RATIO - 1;
/// The last position whose row is dense by the model's law: `(p + 1) / 4 <= 512` <=> `p <= 2,050`.
pub const DENSE_MAX_POS: u32 = TOPK * RATIO + RATIO - 2;
/// Rows per score / top-k slice of a prefill chunk (the score buffer's height: `SEL_ROWS x groups`
/// f32 is reused slice after slice).
pub const SEL_ROWS: u32 = 256;
/// Query positions per prefill attention CTA (FN6's prefill body: 24 warps x 16 rows / GQA 12).
pub const PREFILL_CTA_ROWS: u32 = 32;
/// Rows per score workgroup tile (the kernel's `sq` LDS tile).
pub const SCORE_TILE_ROWS: u32 = 32;
/// Groups per score workgroup.
pub const SCORE_TILE_GROUPS: u32 = 64;
/// A union key list's pad: a multiple of the prefill tile (48 keys) and of the mask byte (8).
pub const KEY_PAD: u32 = 48;
/// `sizeof(QsaKeys)` (flashnext_qsa_attn.hip).
pub const QSA_KEYS_BYTES: usize = 32;

/// Symbols.
pub const SCORE_SYMBOL: &str = "fn_qsa_score_r32";
pub const TOPK_SYMBOL: &str = "fn_qsa_topk512";
pub const UNION_SYMBOL: &str = "fn_qsa_union";
pub const PREFILL_SYMBOL: &str = "fn_qsa_attn_prefill_h256_gqa12_bf16kv";
pub const DECODE_SYMBOL: &str = "fn_qsa_attn_decode_h256_gqa12_bf16kv_nw4";

/// `--fn-qsa`: the model's sparse law above 2,048 (the default) or today's dense attention (the
/// control arm).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QsaMode {
    /// The model's law: the indexer's selection above 2,048 (the default since FN19).
    #[default]
    Sparse,
    /// Every prior token at every context (FN12-FN17's attention; the control arm).
    Dense,
}

impl std::str::FromStr for QsaMode {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.trim() {
            "sparse" => Ok(QsaMode::Sparse),
            "dense" => Ok(QsaMode::Dense),
            t => Err(format!("--fn-qsa sparse | dense, not {t}")),
        }
    }
}

impl std::fmt::Display for QsaMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            QsaMode::Sparse => "sparse",
            QsaMode::Dense => "dense",
        })
    }
}

/// The groups row `pos` may select: `(pos + 1) / 4`.
#[must_use]
pub fn visible(pos: u32) -> u32 {
    (pos + 1) / RATIO
}

/// THE STEP LAW: does a step run the sparse pipeline? `decode`: the pools' context (`pool_ctx`)
/// decides; a prefill chunk: its last row's position `p0 + m - 1`.
#[must_use]
pub fn sparse_step(mode: QsaMode, decode: bool, p0: u32, m: u32, pool_ctx: u32) -> bool {
    if mode == QsaMode::Dense {
        return false;
    }
    if decode {
        pool_ctx > DENSE_MAX_POS + 1
    } else {
        p0 + m > DENSE_MAX_POS + 1
    }
}

/// A decode step whose rows ALL sit at positions `<= 2,050` (sequence `s` feeds rows
/// `pos[s] .. pos[s] + w`): every selection is the identity, so the step runs FN6's dense decode --
/// byte-identical to the gather over the identity lists (the byte oracle) and without the
/// selection launches (the step's captured graph is keyed on it).
#[must_use]
pub fn short_rows(pos: &[u32], w: u32) -> bool {
    pos.iter().all(|&p| p + w.max(1) - 1 <= DENSE_MAX_POS)
}

/// Score / top-k slices of a step of `rows` rows.
#[must_use]
pub fn sel_slices(rows: u32) -> u32 {
    rows.div_ceil(SEL_ROWS).max(1)
}

/// Launches the sparse pipeline adds in place of FN6's attention: `(score + top-k) x slices + union`
/// (the attention itself -- and the combine at the pair form -- replaces the dense one 1:1).
#[must_use]
pub fn selection_launches(rows: u32) -> u32 {
    2 * sel_slices(rows) + 1
}

/// The score buffer's row stride (groups) for rows up to `max_pos`: a multiple of the score tile.
#[must_use]
pub fn score_stride(max_pos: u32) -> u32 {
    visible(max_pos).max(1).div_ceil(SCORE_TILE_GROUPS) * SCORE_TILE_GROUPS
}

/// The `who` scratch row stride (groups): the union scans groups `0..=p_last / 4`.
#[must_use]
pub fn who_stride(max_pos: u32) -> u32 {
    score_stride(max_pos) + SCORE_TILE_GROUPS
}

/// A union list's key capacity: `ctx_bound` tokens (every key a CTA can list), capped by what
/// `rows` rows can select (+ the CTA's own span), rounded to [`KEY_PAD`].
#[must_use]
pub fn key_stride(ctx_bound: u32, rows: u32) -> u32 {
    let cap = (ctx_bound + 64).min(rows * WIDTH + 64);
    cap.div_ceil(KEY_PAD) * KEY_PAD
}

/// The context band a sparse DECODE step's split law is read at: a sequence's list holds at most
/// `w x 2,051` keys (the selections of its `w` rows, overlapping; + its span), never more than
/// the pools' context.
#[must_use]
pub fn decode_band(w: u32, pool_ctx: u32) -> u32 {
    pool_ctx.min(w * WIDTH + 64)
}

/// Scratch bytes of a sparse step (the arena's selection buffers): scores (one slice), sel, who,
/// kslot + kmask + kbits, nkeys.
pub fn scratch_bytes(rows: u32, ctas: u32, max_pos: u32, kstride: u32) -> Result<u64> {
    if rows == 0 || ctas == 0 {
        return Err(Error::Rule("an empty sparse step".into()));
    }
    let slice = u64::from(rows.min(SEL_ROWS));
    let ss = u64::from(score_stride(max_pos));
    Ok(slice * ss * 4
        + u64::from(rows) * u64::from(TOPK) * 4
        + u64::from(ctas) * u64::from(who_stride(max_pos)) * 4
        + u64::from(ctas) * u64::from(kstride) * (4 + 4 + 4)
        + u64::from(ctas) * 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dense_range_is_the_models_identity_selection() {
        assert_eq!(WIDTH, 2051);
        assert_eq!(DENSE_MAX_POS, 2050);
        assert_eq!(visible(DENSE_MAX_POS), TOPK);
        assert_eq!(visible(DENSE_MAX_POS + 1), TOPK + 1);
        // a prompt of 2,051 tokens is dense (its last row at 2,050); 2,052 is the first sparse
        assert!(!sparse_step(QsaMode::Sparse, false, 0, 2051, 0));
        assert!(sparse_step(QsaMode::Sparse, false, 0, 2052, 0));
        assert!(sparse_step(QsaMode::Sparse, false, 2048, 4, 0));
        assert!(!sparse_step(QsaMode::Dense, false, 2048, 2048, 0));
        // decode: the pools' context decides (the graphs are captured per bucket)
        assert!(!sparse_step(QsaMode::Sparse, true, 0, 1, 2048));
        assert!(sparse_step(QsaMode::Sparse, true, 0, 1, 8192));
        assert!(!sparse_step(QsaMode::Dense, true, 0, 1, 8192));
        assert!(short_rows(&[2050, 0], 1) && !short_rows(&[2051], 1) && !short_rows(&[2047], 5));
    }

    #[test]
    fn the_selection_geometry() {
        assert_eq!(sel_slices(2048), 8);
        assert_eq!(sel_slices(40), 1);
        assert_eq!(selection_launches(2048), 17);
        assert_eq!(selection_launches(1), 3);
        assert_eq!(score_stride(16_383), 4096);
        assert_eq!(score_stride(0), 64);
        assert!(who_stride(16_383) > visible(16_383));
        assert_eq!(key_stride(2048, 32) % KEY_PAD, 0);
        assert!(key_stride(131_072, 32) >= 32 * WIDTH);
        assert!(key_stride(4096, 1) >= WIDTH + 3);
        assert_eq!(decode_band(1, 32_768), 2115);
        assert_eq!(decode_band(5, 8192), 8192);
        assert!("sparse".parse::<QsaMode>() == Ok(QsaMode::Sparse));
        assert!("dense".parse::<QsaMode>() == Ok(QsaMode::Dense));
        assert!("x".parse::<QsaMode>().is_err());
        assert!(scratch_bytes(2048, 64, 16_383, key_stride(16_384, 32)).unwrap() < 64 << 20);
    }
}
