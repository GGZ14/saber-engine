//! Flash-Next's attention, QSA-indexer store, PLE and MTP laws (workstream FN6).
//!
//! Everything here is DECLARED geometry: the numbers a Flash-Next layer planner (F2) binds and the
//! host tests hold. The kernels are three code objects:
//!
//! | plugin | symbols | what |
//! |---|---|---|
//! | `flashnext_attn` | `fn_attn_{prefill,decode,declc}_h256_gqa12_bf16kv*`, `fn_attn_splitkv_combine_h256_pf1` | the reference's certified attention bodies (sliced verbatim by the retired source generator) at GQA 12 over bf16 K/V pages |
//! | `flashnext_glue` | `fn_kv_append_ring_h256_bf16`, `fn_idx_q_compress_h128`, `fn_ple_gate_{x,combine}`, `fn_ple_conv`, `fn_ple_rows_fp8`, `fn_mtp_norms`, `fn_mtp_add`, `fn_moe_silu_mul_bf16_t256` | Saber's own arithmetic, `-ffp-contract=off`, gated bit-exact vs `tools/capture/fn6/fn6_ref.py` |
//! | `flashnext_moe` | `fn_moe_bgemm_sk_m16_g{0,1}`, `fn_moe_a8w8_bs128_g{0,1}` | the grouped MTP-expert GEMMs, derived by `tools/gen_fn6_grouped.py` from FN5's split-K band (bf16, the default) and the drafter's W8A8 body (fp8, GATED) |
//!
//! The laws (docs/EXEC.md section 13): the verify-width law at GQA 12, the indexer-cache
//! contract (what FN9's scorer reads), the PLE row-staging contract (what FN7's tier writes), the
//! PLE conv-state geometry, and the launch counts per QSA layer / PLE layer / MTP pass.

use serde::Deserialize;

use crate::{Error, Result};

/// Code objects.
pub const PLUGIN_ATTN: &str = "flashnext_attn";
pub const PLUGIN_GLUE: &str = "flashnext_glue";

/// FN21: Flash-Next's own `qk_norm_rope` instantiation (head 256, rotary 64, interleaved mrope),
/// byte-identical to attn_glue's `qk_norm_rope_wg8_h256_rot64_mi1_red5` (the 27B symbol, untouched)
/// with every load issued first and the eight waves' fp64 rsqrts as one sequence. Same kernargs and
/// the same launch: grid `ceil(m (nq + nkv) / 8)`, 256 threads (LEDGER FN21).
pub const QK_NORM_ROPE_V2: &str = "fn_qk_norm_rope_h256_rot64_mi1_v2";
/// FN21: `fn_idx_q_compress_h128` restructured, byte-identical, same kernargs; FOUR rows a
/// workgroup: grid `ceil(m / 4)`, 256 threads (LEDGER FN21).
pub const IDX_Q_COMPRESS_V2: &str = "fn_idx_q_compress_h128_v2";
pub const IDX_Q_COMPRESS_V2_ROWS: u32 = 4;
pub const PLUGIN_MOE: &str = "flashnext_moe";

// ------------------------------------------------------------------------------ attention

/// QSA head width (`head_dim`).
pub const HEAD_DIM: u32 = 256;
/// Query heads per KV head: 24 q over 2 KV heads (at TP=2: 12 q over ONE kv head per rank).
pub const GQA: u32 = 12;
/// Query heads and KV heads per rank at TP=2 (upstream shards the 2 KV heads over 2 ranks).
pub const Q_HEADS_TP2: u32 = 12;
pub const KV_HEADS_TP2: u32 = 1;
/// The paged cache's block (tokens per page), the reference's and upstream's `MultipleOf(16)`.
pub const KV_BLOCK: u32 = 16;
/// The decode kernel: 16 query rows per warp, at most 4 warps (instantiated NW 1..4) -> 64 rows.
pub const DECODE_ROWS_PER_WARP: u32 = 16;
pub const DECODE_MAX_NW: u32 = 4;
pub const DECODE_ROW_CAP: u32 = DECODE_ROWS_PER_WARP * DECODE_MAX_NW;
/// **The verify-width law**: a decode step's query length per sequence is at most
/// `DECODE_ROW_CAP / GQA` = 5 at GQA 12 (60 of the 64 rows). MTP-4 verifies 5 rows (the committed
/// token + 4 drafts) and fits; a width of 6 (MTP-5) has no instantiation (72 rows = 5 warps).
pub const MAX_VERIFY_ROWS: u32 = DECODE_ROW_CAP / GQA;
/// The decode kernel's KV tile (keys per m-tile loop), the reference's D_TILE.
pub const DECODE_TILE: u32 = 16;
/// The prefill block: 24 warps x 16 rows = 384 rows = 32 query positions x 12 heads.
pub const PREFILL_NWARPS: u32 = 24;
pub const PREFILL_TILE: u32 = 48;
pub const PREFILL_BLOCK_Q: u32 = PREFILL_NWARPS * 16 / GQA;
/// The split-KV merge's weight row (the reference's DECLC_MAX_SPLITS).
pub const DECLC_MAX_SPLITS: u32 = 128;
/// QSA layers of the target (every 4th layer, 3..47) and of the MTP drafter.
pub const TARGET_QSA_LAYERS: u32 = 12;
pub const MTP_QSA_LAYERS: u32 = 1;

/// Warps the decode kernel needs for `q_len` query rows per sequence: `ceil(q_len * 12 / 16)`.
/// Refuses a width past the verify-width law by name.
pub fn decode_nw(q_len: u32) -> Result<u32> {
    verify_width_law(q_len)?;
    Ok((q_len * GQA).div_ceil(DECODE_ROWS_PER_WARP))
}

/// The verify-width law as a check: `1 <= q_len <= MAX_VERIFY_ROWS`.
pub fn verify_width_law(q_len: u32) -> Result<()> {
    if q_len == 0 || q_len > MAX_VERIFY_ROWS {
        return Err(Error::Rule(format!(
            "flash-next decode attention: {q_len} query rows x GQA {GQA} = {} rows; the decode \
             kernel holds {DECODE_ROW_CAP} ({DECODE_MAX_NW} warps x {DECODE_ROWS_PER_WARP}), so \
             the verify width is 1..={MAX_VERIFY_ROWS} (MTP-4)",
            q_len * GQA
        )));
    }
    Ok(())
}

/// The committed attention / MTP-expert launch table (workstream FN10).
pub const PROD_ATTN_RULES_TEXT: &str = include_str!("../rules/flashnext_attn_gfx1201.toml");

/// The split law's constants at one kv head (`[decode_splits]`): the reference's, row for row.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct SplitLawConsts {
    pub kv_heads: u32,
    pub ceiling: u32,
    pub floor: u32,
    pub min_tiles_per_segment: u32,
    pub tile: u32,
}

/// The decode form a route row names.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RowForm {
    /// Decode + the last-arriving merge, one launch.
    Declc,
    /// Decode + the split-KV combine, two launches.
    Pair,
}

/// One `[[decode_route]]` row.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct DecodeRouteRow {
    /// Query rows per sequence (the verify width, 1..=5).
    pub rows: u32,
    /// The captured decode batch (1, 2, 4, 8).
    pub batch: u32,
    /// The context band's upper edge (tokens).
    pub max_ctx: u32,
    pub fill_target: u32,
    pub form: RowForm,
}

/// The MTP-expert arm a split row belongs to.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SplitArm {
    Bf16,
    Fp8,
}

/// One `[[mtp_expert_split]]` row.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct MtpSplitRow {
    pub arm: SplitArm,
    pub max_tokens: u32,
    pub gate_up: u32,
    pub down: u32,
}

/// `rules/flashnext_attn_gfx1201.toml`, parsed.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct AttnRules {
    pub schema_version: u32,
    pub name: String,
    pub decode_splits: SplitLawConsts,
    pub decode_route: Vec<DecodeRouteRow>,
    pub mtp_expert_split: Vec<MtpSplitRow>,
}

/// A decode step's attention launch: the split count and the form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeRoute {
    pub splits: u32,
    pub form: AttnForm,
    /// The decode symbol (the routed NW; `declc` or plain decode by form).
    pub symbol: String,
}

/// The captured decode batches the route table is keyed by (the B1 law's default bucket set).
pub const DECODE_BATCHES: [u32; 4] = [1, 2, 4, 8];

impl AttnRules {
    /// The committed table.
    pub fn prod() -> Result<Self> {
        Self::from_toml(PROD_ATTN_RULES_TEXT)
    }

    /// Parse and validate a table.
    pub fn from_toml(text: &str) -> Result<Self> {
        let r: AttnRules = toml::from_str(text).map_err(|e| Error::Parse(e.to_string()))?;
        r.validate()?;
        Ok(r)
    }

    fn validate(&self) -> Result<()> {
        let law = &self.decode_splits;
        if law.kv_heads != KV_HEADS_TP2 || law.tile != DECODE_TILE {
            return Err(Error::Rule(format!(
                "flashnext attention rules: kv_heads {} / tile {} are compiled as {KV_HEADS_TP2} / \
                 {DECODE_TILE}",
                law.kv_heads, law.tile
            )));
        }
        if law.ceiling > DECLC_MAX_SPLITS || !law.ceiling.is_power_of_two() || law.floor == 0 {
            return Err(Error::Rule(format!(
                "flashnext attention rules: ceiling {} must be a power of two <= {DECLC_MAX_SPLITS}",
                law.ceiling
            )));
        }
        // Every (width, batch) has a row whose band reaches the top: no context falls off the
        // table.
        for w in 1..=MAX_VERIFY_ROWS {
            for b in DECODE_BATCHES {
                if !self
                    .decode_route
                    .iter()
                    .any(|r| r.rows == w && r.batch == b && r.max_ctx == u32::MAX)
                {
                    return Err(Error::Rule(format!(
                        "flashnext attention rules: no open-ended row for width {w} batch {b}"
                    )));
                }
            }
        }
        for r in &self.decode_route {
            verify_width_law(r.rows)?;
            if r.fill_target == 0 || !DECODE_BATCHES.contains(&r.batch) {
                return Err(Error::Rule(format!(
                    "flashnext attention rules: bad row {r:?}"
                )));
            }
        }
        for arm in [SplitArm::Bf16, SplitArm::Fp8] {
            if !self
                .mtp_expert_split
                .iter()
                .any(|r| r.arm == arm && r.max_tokens == u32::MAX)
            {
                return Err(Error::Rule(format!(
                    "flashnext attention rules: no open-ended MTP split row for {arm:?}"
                )));
            }
        }
        for r in &self.mtp_expert_split {
            let ok = match r.arm {
                SplitArm::Bf16 => [r.gate_up, r.down].iter().all(|s| matches!(s, 1 | 2 | 4)),
                SplitArm::Fp8 => r.gate_up == 1 && r.down == 1,
            };
            if !ok {
                return Err(Error::Rule(format!(
                    "flashnext attention rules: MTP split row {r:?} (bf16 takes 1/2/4, the fp8 \
                     body has no split)"
                )));
            }
        }
        Ok(())
    }

    /// The route row for `q_len` rows per sequence, `num_seqs` sequences, a plan context
    /// `max_ctx`: the smallest captured batch holding `num_seqs`, the first band holding `max_ctx`.
    pub fn row(&self, q_len: u32, max_ctx: u32, num_seqs: u32) -> Result<&DecodeRouteRow> {
        verify_width_law(q_len)?;
        let widest = DECODE_BATCHES[DECODE_BATCHES.len() - 1];
        let batch = DECODE_BATCHES
            .iter()
            .copied()
            .find(|&b| b >= num_seqs.max(1))
            .ok_or_else(|| {
                Error::Rule(format!(
                    "flashnext decode attention: {num_seqs} sequences exceed the widest captured \
                     batch {widest}"
                ))
            })?;
        self.decode_route
            .iter()
            .filter(|r| r.rows == q_len && r.batch == batch && r.max_ctx >= max_ctx)
            .min_by_key(|r| r.max_ctx)
            .ok_or_else(|| {
                Error::Rule(format!(
                    "flashnext decode attention: no route row for width {q_len} batch {batch} \
                     context {max_ctx}"
                ))
            })
    }

    /// The split law at one kv head with a given fill target (the reference's arithmetic verbatim).
    #[must_use]
    pub fn splits_with_fill(&self, fill: u32, max_ctx: u32, num_seqs: u32) -> u32 {
        let law = &self.decode_splits;
        let tiles = max_ctx.div_ceil(law.tile);
        let units = (law.kv_heads * num_seqs).max(1);
        let raw = fill / units;
        let mut s = law.ceiling;
        while s > law.floor && s > raw {
            s >>= 1;
        }
        let cap = tiles / law.min_tiles_per_segment;
        while s > law.floor && s > cap {
            s >>= 1;
        }
        if s > tiles {
            s = tiles;
        }
        s.max(1)
    }

    /// The split count a decode step takes (its row's fill target through the law).
    pub fn decode_splits(&self, q_len: u32, max_ctx: u32, num_seqs: u32) -> Result<u32> {
        let row = self.row(q_len, max_ctx, num_seqs)?;
        Ok(self.splits_with_fill(row.fill_target, max_ctx, num_seqs))
    }

    /// The decode launch: splits, form, symbol. One segment is always the pair (declc has nothing
    /// to merge).
    pub fn decode_route(&self, q_len: u32, max_ctx: u32, num_seqs: u32) -> Result<DecodeRoute> {
        let row = self.row(q_len, max_ctx, num_seqs)?;
        let splits = self.splits_with_fill(row.fill_target, max_ctx, num_seqs);
        let form = if splits < 2 || row.form == RowForm::Pair {
            AttnForm::DecodePair
        } else {
            AttnForm::DecodeDeclc
        };
        let symbol = match form {
            AttnForm::DecodeDeclc => routed_declc_symbol(q_len)?,
            _ => format!("fn_attn_decode_h256_gqa12_bf16kv_nw{ROUTED_DECODE_NW}"),
        };
        Ok(DecodeRoute {
            splits,
            form,
            symbol,
        })
    }

    /// The grouped MTP-expert GEMM's K splits `(gate_up, down)` at `tokens` rows.
    pub fn mtp_expert_splits(&self, experts: MtpExperts, tokens: u32) -> Result<(u32, u32)> {
        let arm = match experts {
            MtpExperts::Bf16 => SplitArm::Bf16,
            MtpExperts::Fp8Gated => SplitArm::Fp8,
        };
        self.mtp_expert_split
            .iter()
            .filter(|r| r.arm == arm && r.max_tokens >= tokens)
            .min_by_key(|r| r.max_tokens)
            .map(|r| (r.gate_up, r.down))
            .ok_or_else(|| Error::Rule(format!("no MTP split row for {arm:?} at T {tokens}")))
    }
}

/// **The shipped split law** (FN6's; the reference's committed constants: fill 192 over kv_heads x
/// sequences, powers of two in 16..=128, >= 2 tiles a segment) at ONE kv head per rank, kept as
/// the pre-table CONTROL: `AttnRules::decode_splits` over the committed table must equal it at
/// every cell (a fill target that moves a split count moves the segmentation and so the bytes --
/// FN10's byte gate, LEDGER FN10).
#[must_use]
pub fn decode_splits_fn6_law(max_ctx: u32, num_seqs: u32) -> u32 {
    const FILL: u32 = 192;
    const CEIL: u32 = 128;
    const FLOOR: u32 = 16;
    const MIN_TILES: u32 = 2;
    let tiles = max_ctx.div_ceil(DECODE_TILE);
    let units = (KV_HEADS_TP2 * num_seqs).max(1);
    let raw = FILL / units;
    let mut s = CEIL;
    while s > FLOOR && s > raw {
        s >>= 1;
    }
    let cap = tiles / MIN_TILES;
    while s > FLOOR && s > cap {
        s >>= 1;
    }
    if s > tiles {
        s = tiles;
    }
    s.max(1)
}

/// The split count for a decode step, read from the committed table (`decode_route` rows).
pub fn decode_splits(q_len: u32, max_ctx: u32, num_seqs: u32) -> Result<u32> {
    AttnRules::prod()?.decode_splits(q_len, max_ctx, num_seqs)
}

/// The decode launch for a step (splits, form, symbol), read from the committed table.
pub fn decode_route(q_len: u32, max_ctx: u32, num_seqs: u32) -> Result<DecodeRoute> {
    AttnRules::prod()?.decode_route(q_len, max_ctx, num_seqs)
}

/// The grouped MTP-expert GEMM's `(gate_up, down)` K splits at `tokens`, from the committed table.
pub fn mtp_expert_splits(experts: MtpExperts, tokens: u32) -> Result<(u32, u32)> {
    AttnRules::prod()?.mtp_expert_splits(experts, tokens)
}

/// The segmentation the reference's kernels actually run (`saber_attn_tps` / `saber_attn_used`): the BYTE
/// class of a split count is `(tiles per segment, segments used)`, not the number -- two split
/// counts with the same tiles-per-segment are byte-identical; any other pair re-cuts the online
/// softmax and the f16 partials, a numerics change.
#[must_use]
pub fn segmentation(max_ctx: u32, splits: u32) -> (u32, u32) {
    let tiles = max_ctx.div_ceil(DECODE_TILE);
    let tps = tiles.div_ceil(splits.max(1));
    (tps, tiles.div_ceil(tps.max(1)))
}

/// The prefill symbol.
pub const PREFILL_SYMBOL: &str = "fn_attn_prefill_h256_gqa12_bf16kv";
/// The split-KV combine (the reference's `PF16 = 1` merge; GQA-independent, a copy in this object).
pub const COMBINE_SYMBOL: &str = "fn_attn_splitkv_combine_h256_pf1";

/// The decode symbol for `q_len` rows per sequence.
pub fn decode_symbol(q_len: u32) -> Result<String> {
    Ok(format!(
        "fn_attn_decode_h256_gqa12_bf16kv_nw{}",
        decode_nw(q_len)?
    ))
}

/// The decode + last-arriving merge symbol (one launch; `drop` = the gate's negative control).
pub fn declc_symbol(q_len: u32, drop: bool) -> Result<String> {
    Ok(format!(
        "fn_attn_declc_h256_gqa12_bf16kv_nw{}{}",
        decode_nw(q_len)?,
        if drop { "_drop1" } else { "" }
    ))
}

/// **The routed decode NW** (FN6 hold 3, card 1, 210 W, IC-cold): every width 1..=5 runs the
/// `nw4` instantiation. Its live rows are byte-identical to the minimal NW's (gated: `attn.nw4==nwX`,
/// 0 of every byte differ -- the extra warps hold only dead rows and a wave's rescale is wave-local),
/// it is the ONE spill-free decode body (nw1/nw3 spill 38/37 VGPRs, the reference's own), and it is
/// faster in 47 of 48 measured cells (the 48th within 1.3 %): b1 w1 at 2k 168.8 -> 72.2 us, at 32k
/// 319.0 -> 177.5 us.
pub const ROUTED_DECODE_NW: u32 = DECODE_MAX_NW;

/// The routed decode + last-arriving merge symbol for a width inside the verify-width law.
pub fn routed_declc_symbol(q_len: u32) -> Result<String> {
    verify_width_law(q_len)?;
    Ok(format!(
        "fn_attn_declc_h256_gqa12_bf16kv_nw{ROUTED_DECODE_NW}"
    ))
}

/// Bytes of bf16 K|V per token per QSA layer per rank: 1 kv head x 256 x (K, V) x 2 B.
#[must_use]
pub const fn kv_bytes_per_token_per_layer() -> u64 {
    (KV_HEADS_TP2 * HEAD_DIM * 2 * 2) as u64
}

// ------------------------------------------------------------------------------ the indexer

/// `indexer_n_heads` / `indexer_kv_heads` / `indexer_head_dim` / `indexer_compress_ratio` /
/// `indexer_budget` (tokens) of the checkpoint.
pub const IDX_Q_HEADS: u32 = 4;
pub const IDX_K_HEADS: u32 = 1;
pub const IDX_HEAD: u32 = 128;
pub const IDX_RATIO: u32 = 4;
pub const IDX_TOKEN_BUDGET: u32 = 2_048;
/// `index_qk_proj`'s N: 4 q heads + 1 k head x 128.
pub const IDX_QK_N: u32 = (IDX_Q_HEADS + IDX_K_HEADS) * IDX_HEAD;
/// Compressed rows per KV block (the compressed cache shares the main KV block table).
pub const IDX_GROUPS_PER_BLOCK: u32 = KV_BLOCK / IDX_RATIO;
/// One compressed row: 128 bf16.
pub const IDX_ROW_BYTES: u32 = IDX_HEAD * 2;
/// The rope positions a ring row carries (3 i64, the mrope axes).
pub const RING_POS_BYTES: u32 = 3 * 8;

/// **The indexer-cache contract** (docs/EXEC.md section 13), what FN9's scorer reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexerCache {
    /// QSA layers holding one (target 12 + MTP 1 = 13).
    pub layers: u32,
    /// Verify rows beyond the committed token a step may store (MTP-k's k).
    pub spec: u32,
}

impl IndexerCache {
    /// Compressed bytes per token per layer: one 256 B row per 4 tokens.
    #[must_use]
    pub const fn bytes_per_token_per_layer(&self) -> u32 {
        IDX_ROW_BYTES / IDX_RATIO
    }
    /// Compressed bytes per token over every QSA layer (13 layers -> 832 B, the plan's number).
    #[must_use]
    pub const fn bytes_per_token(&self) -> u32 {
        self.layers * self.bytes_per_token_per_layer()
    }
    /// Bytes of one layer's compressed page (one per 16-token KV block).
    #[must_use]
    pub const fn page_bytes(&self) -> u32 {
        IDX_GROUPS_PER_BLOCK * IDX_ROW_BYTES
    }
    /// Raw-key ring rows per sequence per layer: the open group's committed keys plus every row a
    /// speculative step stores before acceptance is known, rounded up to whole groups
    /// (upstream's `QSAKeyStateCache`: ratio x ceil((ratio + spec) / ratio)).
    #[must_use]
    pub const fn ring_rows(&self) -> u32 {
        IDX_RATIO * (IDX_RATIO + self.spec).div_ceil(IDX_RATIO)
    }
    /// Ring bytes per sequence per layer (key rows + their three i64 positions).
    #[must_use]
    pub const fn ring_bytes_per_seq_per_layer(&self) -> u32 {
        self.ring_rows() * (IDX_ROW_BYTES + RING_POS_BYTES)
    }
}

/// The compressed row a token at logical position `p` completes, if any: group `p / 4` when
/// `(p + 1) % 4 == 0`, stored at page `block_table[g / 4]`, row `g % 4`.
#[must_use]
pub const fn completes_group(p: u64) -> Option<u64> {
    if p >= 3 && (p + 1) % IDX_RATIO as u64 == 0 {
        Some(p / IDX_RATIO as u64)
    } else {
        None
    }
}

// ------------------------------------------------------------------------------ PLE

/// Rows looked up per token (16 heads: 8 two-gram, 8 three-gram) and their width.
pub const PLE_ROWS_PER_TOKEN: u32 = 16;
pub const PLE_ROW_DIM: u32 = 160;
/// The gathered embedding: 16 x 160 = 2,560 (`ple_embed_dim`).
pub const PLE_EMB: u32 = PLE_ROWS_PER_TOKEN * PLE_ROW_DIM;
/// The HC stream the PLE gates against and adds into.
pub const PLE_STREAM: u32 = 10_240;
/// `key_proj` [10240, 2560] / `value_proj` [2560, 2560], both bf16 (FN5's `ple.key_proj` /
/// `mtp.fc`-shaped rows of `rules/flashnext_gfx1201.toml`).
pub const PLE_KEY_N: u32 = PLE_STREAM;
pub const PLE_VALUE_N: u32 = 2_560;

/// The PLE table source the staging buffer holds (docs/EXEC.md section 10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PleSource {
    /// The checkpoint's bf16 rows in place (the DEFAULT): 320 B rows.
    CkptBf16,
    /// Qwen's first-party fp8 table (GATED, the user's word + F6's gates): 160 B rows + one bf16
    /// per-TABLE scale.
    QwenFp8,
}

/// **The PLE row-staging contract** (docs/EXEC.md section 13), what FN7's tier writes: per token
/// `t` of the step, its 16 rows in the hash's HEAD order (column h of the host hash's `[T, 16]`
/// ids) at `staging + (t * 16 + h) * row_bytes`, dense (no padding between rows or tokens), so
/// the bf16 staging buffer IS the `[T, 2560]` embedding the key / value GEMMs read as A (the
/// gather is the layout, zero launches). The fp8 source adds one conversion launch
/// (`fn_ple_rows_fp8`) and the scale.
#[must_use]
pub const fn ple_row_bytes(src: PleSource) -> u32 {
    match src {
        PleSource::CkptBf16 => PLE_ROW_DIM * 2,
        PleSource::QwenFp8 => PLE_ROW_DIM,
    }
}

/// Staged bytes per token (5,120 bf16 / 2,560 fp8).
#[must_use]
pub const fn ple_staging_bytes_per_token(src: PleSource) -> u32 {
    PLE_ROWS_PER_TOKEN * ple_row_bytes(src)
}

/// The dilated short conv: kernel 4, dilation `ngram_size` = 3 -> 9 rows of history.
pub const PLE_CONV_KERNEL: u32 = 4;
pub const PLE_CONV_DILATION: u32 = 3;
pub const PLE_CONV_STATE_ROWS: u32 = (PLE_CONV_KERNEL - 1) * PLE_CONV_DILATION;

/// **The conv-state geometry** (docs/EXEC.md section 13): per sequence a HISTORY of
/// `9 + spec` rows of the conv's (normed) input, 10,240 bf16 each, in TWO slots (the step reads
/// one and writes the other: the kernel's reads of the old history and its writes of the new one
/// never alias), plus a per-sequence `drop` = the previous step's rejected verify rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PleConvState {
    pub spec: u32,
}

impl PleConvState {
    /// History rows per slot: 9 + spec (13 at MTP-4).
    #[must_use]
    pub const fn hist_rows(&self) -> u32 {
        PLE_CONV_STATE_ROWS + self.spec
    }
    /// Bytes of one slot.
    #[must_use]
    pub const fn slot_bytes(&self) -> u64 {
        self.hist_rows() as u64 * PLE_STREAM as u64 * 2
    }
    /// Bytes per sequence (two slots).
    #[must_use]
    pub const fn bytes_per_seq(&self) -> u64 {
        2 * self.slot_bytes()
    }
    /// The largest `drop` the history can absorb (the rows past the 9 the conv needs).
    #[must_use]
    pub const fn max_drop(&self) -> u32 {
        self.spec
    }
}

// ------------------------------------------------------------------------------ MTP

/// An MTP routed expert: gate_up [1280, 2560] (merged, gate rows first) + down [2560, 640].
pub const MTP_INTER: u32 = 640;
pub const MTP_HIDDEN: u32 = 2_560;
/// Bytes one expert reads per pass, bf16 (the default).
pub const MTP_EXPERT_BYTES_BF16: u64 =
    (2 * MTP_INTER as u64 * MTP_HIDDEN as u64 + MTP_HIDDEN as u64 * MTP_INTER as u64) * 2;
/// Bytes one expert reads per pass, fp8 block-128 (the gated option): the e4m3 weights plus the
/// fp32 block scales ([10, 20] gate_up, [20, 5] down).
pub const MTP_EXPERT_BYTES_FP8: u64 = 2 * MTP_INTER as u64 * MTP_HIDDEN as u64
    + MTP_HIDDEN as u64 * MTP_INTER as u64
    + (200 + 100) * 4;

/// The MTP expert GEMM source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpExperts {
    /// Quark's bf16 experts on the grouped bf16 band (the DEFAULT).
    Bf16,
    /// Qwen's first-party fp8 block-128 experts on the grouped W8A8 body (GATED: never a default
    /// without its quality gate and the user's word).
    Fp8Gated,
}

impl MtpExperts {
    /// Expert-weight bytes one draft pass reads per rank for `experts` distinct experts touched.
    #[must_use]
    pub const fn pass_bytes(self, experts: u64) -> u64 {
        experts
            * match self {
                Self::Bf16 => MTP_EXPERT_BYTES_BF16,
                Self::Fp8Gated => MTP_EXPERT_BYTES_FP8,
            }
    }
    /// The grouped GEMM symbol (`gather` = gate_up, A rows by token).
    #[must_use]
    pub const fn symbol(self, gather: bool) -> &'static str {
        match (self, gather) {
            (Self::Bf16, true) => "fn_moe_bgemm_sk_m16_g1",
            (Self::Bf16, false) => "fn_moe_bgemm_sk_m16_g0",
            (Self::Fp8Gated, true) => "fn_moe_a8w8_bs128_g1",
            (Self::Fp8Gated, false) => "fn_moe_a8w8_bs128_g0",
        }
    }
    /// Launches of the MoE block: FN4's train, SEVEN (router GEMM, the shared-gate GEMV, route +
    /// sort, gate_up, silu, down, combine) -- the fp8 option adds the two per-128 activation quants (`dflash_act_quant_g128`
    /// before gate_up over the token rows and before down over the expert-major rows).
    #[must_use]
    pub const fn moe_block_launches(self) -> usize {
        match self {
            Self::Bf16 => crate::moe::LAUNCHES_PER_MOE_BLOCK,
            Self::Fp8Gated => crate::moe::LAUNCHES_PER_MOE_BLOCK + 2,
        }
    }
}

// ------------------------------------------------------------------------------ launch laws

/// The attention form of a QSA layer's step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnForm {
    /// A prefill chunk: one attention launch.
    Prefill,
    /// A decode step whose split law answers >= 2 segments: the decode + last-arriver merge, one
    /// launch.
    DecodeDeclc,
    /// A decode step at ONE segment: decode + combine (declc has nothing to merge), two.
    DecodePair,
}

/// Launches of a QSA layer's ATTENTION side (the HC mixes and the MoE block are FN5's / FN4's):
///   1 the merged projection GEMM (q|gate, k, v, index_qk: N 7,296 per rank at TP=2 -- a pack
///     merge rule like the GDN in_proj's; the four checkpoint records share one A input)
///   2 `qk_norm_rope_*` (attn_glue, 12 q / 1 kv heads per rank -- the 27B's symbols at nq 12,
///     nkv 1)
///   3 `fn_idx_q_compress_h128` (indexer q norm/rope + the compressed-key store)
///   4 `fn_kv_append_ring_h256_bf16` (bf16 K|V pages + the indexer's raw-key ring; after 3,
///     which reads the ring)
///   5 the attention (prefill / declc; 6 with the decode pair)
///   6 `fn_attn_gate_bf16_t256` (FN5; sigmoid(gate) x attn -> bf16, o_proj's A)
///   7 o_proj (bf16 GEMM, K 3,072 at TP=2)
#[must_use]
pub const fn qsa_attn_launches(form: AttnForm) -> usize {
    match form {
        AttnForm::Prefill | AttnForm::DecodeDeclc => 7,
        AttnForm::DecodePair => 8,
    }
}

/// The merged projection's N at TP=2: 12 heads x (q 256 + gate 256) + k 256 + v 256 + the
/// replicated indexer 640.
pub const QSA_PROJ_N_TP2: u32 = Q_HEADS_TP2 * 2 * HEAD_DIM + 2 * KV_HEADS_TP2 * HEAD_DIM + IDX_QK_N;

/// Launches of the PLE at layer 1 (on top of FN5's mix): key GEMM, value GEMM,
/// `fn_ple_gate_combine` (FN5's standalone combine folded in), `fn_ple_conv` (the add into the
/// stream folded in). The fp8 table adds `fn_ple_rows_fp8`.
#[must_use]
pub const fn ple_launches(src: PleSource) -> usize {
    match src {
        PleSource::CkptBf16 => 4,
        PleSource::QwenFp8 => 5,
    }
}
/// What the same work costs unfused: FN5's `fn_hc_combine` + the four above.
pub const PLE_LAUNCHES_UNFUSED: usize = 5;

/// Launches of one MTP draft pass BEFORE its decoder layer: `fn_mtp_norms` (both pre-fc norms),
/// `fc_embedding`, `fc_hidden` (M = 4 x rows: the four streams share the weight), `fn_mtp_add`.
/// At TP=2 the fc records are column halves ([1280, 2560] per rank); these counts take both
/// halves on every rank (replicated, 2 x 13.1 MB read) -- the alternative is a gather per GEMM.
pub const MTP_PRE_LAYER_LAUNCHES: usize = 4;

/// Launches of one MTP draft pass up to (not including) the head: the pre-layer 4, the layer's
/// attention-side mix (FN5, 2), the QSA attention side, the MoE-side mix (2), the MoE block, the
/// final mixer (FN5, 2). MTP's own QSA layer and HC are the target's kernels -- no new code.
#[must_use]
pub const fn mtp_pass_launches(form: AttnForm, experts: MtpExperts) -> usize {
    MTP_PRE_LAYER_LAUNCHES + 2 + qsa_attn_launches(form) + 2 + experts.moe_block_launches() + 2
}

/// Every symbol of the three code objects, by plugin (the manifest test holds them).
#[must_use]
pub fn symbols() -> Vec<(&'static str, String)> {
    let mut v = vec![
        (PLUGIN_ATTN, PREFILL_SYMBOL.to_owned()),
        (PLUGIN_ATTN, COMBINE_SYMBOL.to_owned()),
    ];
    for nw in 1..=DECODE_MAX_NW {
        v.push((
            PLUGIN_ATTN,
            format!("fn_attn_decode_h256_gqa12_bf16kv_nw{nw}"),
        ));
        v.push((
            PLUGIN_ATTN,
            format!("fn_attn_declc_h256_gqa12_bf16kv_nw{nw}"),
        ));
        v.push((
            PLUGIN_ATTN,
            format!("fn_attn_declc_h256_gqa12_bf16kv_nw{nw}_drop1"),
        ));
    }
    for s in [
        "fn_kv_append_ring_h256_bf16",
        "fn_idx_q_compress_h128",
        IDX_Q_COMPRESS_V2,
        QK_NORM_ROPE_V2,
        "fn_ple_gate_x",
        "fn_ple_gate_combine",
        "fn_ple_conv",
        "fn_ple_rows_fp8",
        "fn_mtp_norms",
        "fn_mtp_add",
        "fn_moe_silu_mul_bf16_t256",
    ] {
        v.push((PLUGIN_GLUE, s.to_owned()));
    }
    for e in [MtpExperts::Bf16, MtpExperts::Fp8Gated] {
        for g in [true, false] {
            v.push((PLUGIN_MOE, e.symbol(g).to_owned()));
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The verify-width law, asserted: widths 1..=5 fit the 64-row decode kernel at GQA 12 and map
    /// to NW 1, 2, 3, 3, 4; width 6 (MTP-5) and 0 are refused by name.
    #[test]
    fn the_verify_width_law_at_gqa_12() {
        assert_eq!(DECODE_ROW_CAP, 64);
        assert_eq!(MAX_VERIFY_ROWS, 5);
        let nws: Vec<u32> = (1..=5).map(|w| decode_nw(w).expect("fits")).collect();
        assert_eq!(nws, vec![1, 2, 3, 3, 4]);
        for w in 1..=5 {
            assert!(w * GQA <= DECODE_ROW_CAP);
        }
        // NEGATIVE CONTROL: the next width needs a fifth warp the object does not have.
        let e = decode_nw(6).expect_err("MTP-5 must be refused");
        assert!(e.to_string().contains("1..=5"), "{e}");
        assert!(decode_nw(0).is_err());
        assert_eq!(6 * GQA, 72);
        const { assert!(6 * GQA > DECODE_ROW_CAP) };
        for w in 1..=5 {
            assert_eq!(
                routed_declc_symbol(w).unwrap(),
                "fn_attn_declc_h256_gqa12_bf16kv_nw4"
            );
        }
        assert!(routed_declc_symbol(6).is_err());
        // The 27B's law at GQA 6 would have allowed 10 -- the constant is GQA's, not the kernel's.
        assert_eq!(DECODE_ROW_CAP / 6, 10);
        assert_eq!(
            declc_symbol(5, false).unwrap(),
            "fn_attn_declc_h256_gqa12_bf16kv_nw4"
        );
        assert_eq!(
            decode_symbol(1).unwrap(),
            "fn_attn_decode_h256_gqa12_bf16kv_nw1"
        );
    }

    #[test]
    fn prefill_block_and_split_law() {
        assert_eq!(PREFILL_BLOCK_Q, 32);
        assert_eq!(PREFILL_BLOCK_Q * GQA, PREFILL_NWARPS * 16);
        // one sequence at 2k: 128 tiles, fill 192 -> 128, cap 64 -> 64
        assert_eq!(decode_splits(1, 2048, 1).unwrap(), 64);
        assert_eq!(decode_splits(5, 32768, 1).unwrap(), 128);
        assert_eq!(decode_splits(3, 2048, 8).unwrap(), 16);
        assert_eq!(decode_splits(1, 16, 1).unwrap(), 1);
        for ctx in [1u32, 17, 100, 2048, 8192, 32768] {
            for b in 1..=8 {
                let s = decode_splits(1, ctx, b).unwrap();
                assert!((1..=DECLC_MAX_SPLITS).contains(&s));
            }
        }
    }

    /// FN10's byte law as a host test: the committed table's split count equals FN6's shipped law
    /// at EVERY width, batch and context, so every row's bytes are the shipped bytes and the form
    /// is the only thing a row changes. NEGATIVE CONTROL: the same table with the fill target
    /// doubled (the "one kv head = half the work units" derivation) moves the split count -- and
    /// so the segmentation, and so the bytes -- at batch 2..8 (b1 is held by the ceiling / the
    /// two-tile floor, not the fill).
    #[test]
    fn the_table_reproduces_the_shipped_split_law_everywhere() {
        let r = AttnRules::prod().expect("committed table");
        assert_eq!(r.decode_route.len(), 60);
        let ctxs: Vec<u32> = (0..=17)
            .map(|p| 1u32 << p)
            .chain([
                33, 100, 1000, 2047, 2049, 3000, 8191, 8193, 20000, 32768, 65536, 131_072,
            ])
            .collect();
        for w in 1..=MAX_VERIFY_ROWS {
            for b in 1..=8 {
                for &c in &ctxs {
                    assert_eq!(
                        r.decode_splits(w, c, b).unwrap(),
                        decode_splits_fn6_law(c, b),
                        "w{w} b{b} ctx{c}"
                    );
                }
            }
        }
        let doubled = PROD_ATTN_RULES_TEXT.replace("fill_target = 192", "fill_target = 384");
        let d = AttnRules::from_toml(&doubled).expect("control table parses");
        let mut moved = Vec::new();
        for b in [1u32, 2, 4, 8] {
            for c in [2048u32, 8192, 32768] {
                let (a, x) = (
                    r.decode_splits(5, c, b).unwrap(),
                    d.decode_splits(5, c, b).unwrap(),
                );
                if a != x {
                    assert_ne!(segmentation(c, a), segmentation(c, x));
                    moved.push((b, c, a, x));
                }
            }
        }
        assert!(moved.iter().all(|m| m.0 > 1), "{moved:?}");
        assert_eq!(moved.len(), 8, "{moved:?}");
        // the split NUMBER is not the byte class; the segmentation is
        assert_eq!(segmentation(2048, 64), segmentation(2048, 100));
        assert_ne!(segmentation(2048, 64), segmentation(2048, 128));
    }

    #[test]
    fn the_route_table_refuses_what_it_does_not_hold() {
        let r = AttnRules::prod().unwrap();
        assert!(
            r.row(6, 2048, 1).is_err(),
            "width 6 is past the verify-width law"
        );
        assert!(r.row(0, 2048, 1).is_err());
        assert!(
            r.row(1, 2048, 9).is_err(),
            "9 sequences exceed the widest captured batch"
        );
        // batch 3 replays batch 4's row; a context of 2,049 is the 8k band
        assert_eq!(r.row(2, 2049, 3).unwrap().batch, 4);
        assert_eq!(r.row(2, 2049, 3).unwrap().max_ctx, 8192);
        assert_eq!(r.row(2, 2048, 1).unwrap().max_ctx, 2048);
        assert_eq!(r.row(2, 40_000, 1).unwrap().max_ctx, u32::MAX);
        // a table missing an open-ended row, or a split the body has no cell for, is refused
        let cut = PROD_ATTN_RULES_TEXT.replacen("max_ctx = 4294967295", "max_ctx = 32768", 1);
        assert!(AttnRules::from_toml(&cut).is_err());
        let bad = PROD_ATTN_RULES_TEXT.replacen("gate_up = 2", "gate_up = 3", 1);
        assert!(AttnRules::from_toml(&bad).is_err());
    }

    /// Every route answers a symbol the object holds and a form consistent with its split count.
    #[test]
    fn every_route_is_a_committed_symbol_and_a_declared_launch_count() {
        let r = AttnRules::prod().unwrap();
        let syms: Vec<String> = symbols().into_iter().map(|(_, s)| s).collect();
        for w in 1..=MAX_VERIFY_ROWS {
            for b in [1u32, 2, 3, 4, 5, 8] {
                for c in [16u32, 700, 2048, 8192, 32768] {
                    let d = r.decode_route(w, c, b).unwrap();
                    assert!(syms.contains(&d.symbol), "{d:?}");
                    if d.splits < 2 {
                        assert_eq!(d.form, AttnForm::DecodePair);
                    }
                    match d.form {
                        AttnForm::DecodeDeclc => assert!(d.symbol.contains("declc")),
                        AttnForm::DecodePair => assert!(d.symbol.contains("_decode_")),
                        AttnForm::Prefill => unreachable!(),
                    }
                    assert!(qsa_attn_launches(d.form) >= 7);
                }
            }
        }
    }

    /// FN10's routing, pinned: the pair form (decode + combine, 8 launches a QSA layer) in the 57
    /// of 60 measured cells where it won beyond the noise in both holds, the declc (7) kept where
    /// the pair's gain sat inside a cell's floor once (2,048 / b8 / w2; 8,192 and above / b8 /
    /// w1); every row at the shipped fill target.
    #[test]
    fn fn10_routes_the_pair_where_it_won() {
        let r = AttnRules::prod().unwrap();
        let pair = r
            .decode_route
            .iter()
            .filter(|x| x.form == RowForm::Pair)
            .count();
        assert_eq!(pair, 57);
        for (w, c, b) in [(2, 2048, 8), (1, 8192, 8), (1, 32768, 8)] {
            assert_eq!(r.row(w, c, b).unwrap().form, RowForm::Declc);
        }
        assert!(r.decode_route.iter().all(|x| x.fill_target == 192));
        let d = r.decode_route(5, 32768, 1).unwrap();
        assert_eq!(
            (d.splits, d.form, d.symbol.as_str()),
            (
                128,
                AttnForm::DecodePair,
                "fn_attn_decode_h256_gqa12_bf16kv_nw4"
            )
        );
        assert_eq!(qsa_attn_launches(d.form), 8);
        let k = r.decode_route(2, 2048, 8).unwrap();
        assert_eq!((k.splits, k.form), (16, AttnForm::DecodeDeclc));
        assert_eq!(k.symbol, "fn_attn_declc_h256_gqa12_bf16kv_nw4");
        assert_eq!(mtp_pass_launches(d.form, MtpExperts::Bf16), 25);
    }

    /// The grouped MTP-expert GEMM's split rows pin the measured cells: gate_up ks 2 at T = 1 (won
    /// in FN6 hold 4, 105.0 vs 119.5, AND in FN10 hold 1, 78.3 vs 81.7), 1 above; down ks 1 at
    /// every T (FN6's ks 4 gain of 2-5 % at T 1..5 reversed in FN10 hold 1: inside the noise, so
    /// one class); splits LOSE from T 8 in both; the fp8 body takes no split.
    #[test]
    fn the_mtp_expert_split_rows_pin_the_measured_cells() {
        let r = AttnRules::prod().unwrap();
        let bf = |t| r.mtp_expert_splits(MtpExperts::Bf16, t).unwrap();
        assert_eq!(bf(1), (2, 1));
        for t in [2, 5, 6, 8, 16, 64, 2048, 100_000] {
            assert_eq!(bf(t), (1, 1), "T {t}");
        }
        for t in [1, 5, 8, 2048] {
            assert_eq!(
                r.mtp_expert_splits(MtpExperts::Fp8Gated, t).unwrap(),
                (1, 1)
            );
        }
        assert_eq!(mtp_expert_splits(MtpExperts::Bf16, 1).unwrap(), (2, 1));
    }

    /// The indexer-cache contract's numbers: 64 B/token/layer, 832 B/token over 13 layers (the
    /// plan's), 4 groups per 16-token page, an 8-row ring at MTP-4.
    #[test]
    fn the_indexer_cache_contract() {
        let c = IndexerCache {
            layers: TARGET_QSA_LAYERS + MTP_QSA_LAYERS,
            spec: 4,
        };
        assert_eq!(IDX_QK_N, 640);
        assert_eq!(c.bytes_per_token_per_layer(), 64);
        assert_eq!(c.bytes_per_token(), 832);
        assert_eq!(c.page_bytes(), 1024);
        assert_eq!(c.ring_rows(), 8);
        assert_eq!(
            IndexerCache {
                layers: 13,
                spec: 0
            }
            .ring_rows(),
            4
        );
        assert_eq!(
            IndexerCache {
                layers: 13,
                spec: 5
            }
            .ring_rows(),
            12
        );
        assert_eq!(c.ring_bytes_per_seq_per_layer(), 8 * 280);
        assert_eq!(completes_group(3), Some(0));
        assert_eq!(completes_group(7), Some(1));
        assert_eq!(completes_group(4), None);
        assert_eq!(completes_group(2), None);
        // every position of a 2,048-token context lands in exactly one group, 512 groups
        let groups: Vec<u64> = (0..2048).filter_map(completes_group).collect();
        assert_eq!(groups.len(), 512);
        assert_eq!(groups, (0..512).collect::<Vec<_>>());
        assert_eq!(IDX_TOKEN_BUDGET / IDX_RATIO, 512);
    }

    #[test]
    fn the_ple_staging_and_conv_state_geometry() {
        assert_eq!(PLE_EMB, 2560);
        assert_eq!(ple_staging_bytes_per_token(PleSource::CkptBf16), 5120);
        assert_eq!(ple_staging_bytes_per_token(PleSource::QwenFp8), 2560);
        assert_eq!(PLE_CONV_STATE_ROWS, 9);
        let st = PleConvState { spec: 4 };
        assert_eq!(st.hist_rows(), 13);
        assert_eq!(st.slot_bytes(), 13 * 10_240 * 2);
        assert_eq!(st.bytes_per_seq(), 532_480);
        // the plan's 180 KiB is the 9-row, spec-0, single-slot minimum
        assert_eq!(PleConvState { spec: 0 }.slot_bytes(), 180 * 1024);
        assert_eq!(st.max_drop(), 4);
    }

    #[test]
    fn launch_counts_are_declared() {
        assert_eq!(qsa_attn_launches(AttnForm::Prefill), 7);
        assert_eq!(qsa_attn_launches(AttnForm::DecodeDeclc), 7);
        assert_eq!(qsa_attn_launches(AttnForm::DecodePair), 8);
        assert_eq!(QSA_PROJ_N_TP2, 7296);
        assert_eq!(ple_launches(PleSource::CkptBf16), 4);
        assert!(ple_launches(PleSource::CkptBf16) < PLE_LAUNCHES_UNFUSED);
        assert_eq!(ple_launches(PleSource::QwenFp8), 5);
        assert_eq!(MtpExperts::Bf16.moe_block_launches(), 7);
        assert_eq!(MtpExperts::Fp8Gated.moe_block_launches(), 9);
        assert_eq!(
            mtp_pass_launches(AttnForm::DecodeDeclc, MtpExperts::Bf16),
            4 + 2 + 7 + 2 + 7 + 2
        );
        assert_eq!(
            mtp_pass_launches(AttnForm::DecodeDeclc, MtpExperts::Bf16),
            24
        );
    }

    #[test]
    fn mtp_expert_bytes() {
        assert_eq!(MTP_EXPERT_BYTES_BF16, 9_830_400);
        assert_eq!(MTP_EXPERT_BYTES_FP8, 4_916_400);
        // b1: 10 routed experts touched (all local at EP=1; ~5 per rank at EP=2)
        assert_eq!(MtpExperts::Bf16.pass_bytes(10), 98_304_000);
        assert_eq!(MtpExperts::Fp8Gated.pass_bytes(10), 49_164_000);
        assert_eq!(MtpExperts::Bf16.symbol(true), "fn_moe_bgemm_sk_m16_g1");
    }

    #[test]
    fn every_declared_symbol_is_in_its_committed_object() {
        let dir = crate::prebuilt::dir();
        for (plugin, sym) in symbols() {
            let m = crate::manifest::Manifest::load(dir.join(format!("{plugin}.manifest.toml")))
                .unwrap_or_else(|e| panic!("{plugin}: {e}"));
            assert!(m.kernel(&sym).is_some(), "{plugin} lacks {sym}");
        }
        // and every symbol of the three objects is declared here (no stray instantiation)
        let all: Vec<String> = symbols().into_iter().map(|(_, s)| s).collect();
        for plugin in [PLUGIN_ATTN, PLUGIN_GLUE, PLUGIN_MOE] {
            let m = crate::manifest::Manifest::load(dir.join(format!("{plugin}.manifest.toml")))
                .expect("manifest");
            for k in &m.kernel {
                assert!(
                    all.contains(&k.name),
                    "{plugin}: {} is not declared",
                    k.name
                );
            }
        }
    }
}
