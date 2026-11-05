//! Rotary parameters and mrope position ids for Qwen3.8-27B.
//!
//! Two things about this checkpoint make positions model-specific (ARCHITECTURE.md,
//! "Rust-specific hazards"):
//!
//! * **`partial_rotary_factor 0.25`** — only the first `head_dim * 0.25 = 64` of the 256
//!   dimensions of each head are rotated; the remaining 192 pass through unrotated. So the
//!   inverse-frequency table has `64 / 2 = 32` entries, not 128.
//! * **`mrope_interleaved` with `mrope_section [11, 11, 10]`** — the 32 frequency pairs are
//!   split between the temporal / height / width position streams. `11 + 11 + 10 == 32`, i.e.
//!   the sections tile exactly the rotary half-dimension; that identity is asserted at load.
//!
//! For **text-only** input all three streams carry the same value (the token's text position),
//! so the interleave is the identity and a `[3, seq_len]` position tensor has three equal rows.
//! See [`MropePositions::text_only`] for what changes when DeepStack vision lands.

use serde::Deserialize;

use crate::ModelError;

/// The text half of `config.json` (`config["text_config"]`), as far as positions care.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    /// Per-head dimension (256 here; note `num_attention_heads * head_dim != hidden_size`).
    pub head_dim: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (GQA).
    pub num_key_value_heads: usize,
    /// Model hidden size.
    pub hidden_size: usize,
    /// Decoder layer count.
    pub num_hidden_layers: usize,
    /// Every `full_attention_interval`-th layer is full attention; the rest are GDN.
    pub full_attention_interval: usize,
    /// Vocabulary size (248320).
    pub vocab_size: usize,
    /// Maximum position the rope table is defined for.
    pub max_position_embeddings: usize,
    /// Fraction of `head_dim` that is rotated.
    pub partial_rotary_factor: f64,
    /// Rope configuration block.
    pub rope_parameters: RopeParameters,
}

/// `config["text_config"]["rope_parameters"]`.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    /// Rope base (10 000 000 here).
    pub rope_theta: f64,
    /// `"default"` — no NTK/yarn/longrope scaling on this checkpoint.
    pub rope_type: String,
    /// Frequency-pair split between the t/h/w position streams.
    pub mrope_section: Vec<usize>,
    /// Whether the t/h/w streams are interleaved along the frequency axis rather than
    /// laid out as three contiguous blocks.
    pub mrope_interleaved: bool,
    /// Repeated here by the checkpoint; must agree with [`TextConfig::partial_rotary_factor`].
    pub partial_rotary_factor: Option<f64>,
}

/// Everything the rotary kernels need, resolved from `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeParams {
    /// `head_dim` (256).
    pub head_dim: usize,
    /// `int(head_dim * partial_rotary_factor)` = 64. Dimensions `[0, rotary_dim)` rotate.
    pub rotary_dim: usize,
    /// Rope base.
    pub theta: f64,
    /// `rotary_dim / 2` inverse frequencies, `theta^(-2i/rotary_dim)`.
    pub inv_freq: Vec<f64>,
    /// `[11, 11, 10]` — t/h/w frequency-pair counts, summing to `rotary_dim / 2`.
    pub mrope_section: [usize; 3],
    /// `true` on this checkpoint.
    pub mrope_interleaved: bool,
    /// Rope table extent.
    pub max_position_embeddings: usize,
}

impl RopeParams {
    /// Number of `(cos, sin)` frequency pairs, i.e. `rotary_dim / 2`.
    #[must_use]
    pub fn n_freqs(&self) -> usize {
        self.inv_freq.len()
    }

    /// Which mrope stream (0 = t, 1 = h, 2 = w) frequency pair `i` reads from.
    ///
    /// * **Interleaved** (this checkpoint): pair `i` reads stream `i % 3`, clamped to the
    ///   section lengths — the reference server's `apply_interleaved_rope` overwrites `x[0]`'s slots
    ///   `1 : section[1] * 3 : 3` from `x[1]` and `2 : section[2] * 3 : 3` from `x[2]`,
    ///   leaving every other slot on the temporal stream.
    /// * **Chunked** (the classic Qwen2-VL layout): the first `section[0]` pairs are t, the
    ///   next `section[1]` are h, the rest are w.
    ///
    /// For text-only input all three streams are equal, so this only matters once vision lands.
    #[must_use]
    pub fn mrope_stream_of(&self, i: usize) -> usize {
        if self.mrope_interleaved {
            match i % 3 {
                1 if i < self.mrope_section[1] * 3 => 1,
                2 if i < self.mrope_section[2] * 3 => 2,
                _ => 0,
            }
        } else if i < self.mrope_section[0] {
            0
        } else if i < self.mrope_section[0] + self.mrope_section[1] {
            1
        } else {
            2
        }
    }
}

/// Read the rotary parameters out of a parsed `config.json`.
///
/// Accepts either the whole multimodal config (with a `text_config` object) or the text config
/// on its own. Fails loud: a `rope_type` other than `"default"`, an mrope section that does not
/// tile `rotary_dim / 2`, or a missing field is an error, never a silent default.
pub fn rope_params(config: &serde_json::Value) -> Result<RopeParams, ModelError> {
    let text = config.get("text_config").unwrap_or(config);
    let cfg: TextConfig = serde_json::from_value(text.clone())
        .map_err(|e| ModelError::Config(format!("text_config: {e}")))?;
    rope_params_from_text_config(&cfg)
}

/// [`rope_params`] for an already-deserialised [`TextConfig`].
pub fn rope_params_from_text_config(cfg: &TextConfig) -> Result<RopeParams, ModelError> {
    let rp = &cfg.rope_parameters;
    if rp.rope_type != "default" {
        return Err(ModelError::Config(format!(
            "rope_type {:?} is not implemented (only \"default\")",
            rp.rope_type
        )));
    }
    if let Some(prf) = rp.partial_rotary_factor {
        if (prf - cfg.partial_rotary_factor).abs() > f64::EPSILON {
            return Err(ModelError::Config(format!(
                "partial_rotary_factor disagrees: text_config {} vs rope_parameters {prf}",
                cfg.partial_rotary_factor
            )));
        }
    }
    // reference/transformers: rotary_dim = int(head_dim * partial_rotary_factor)
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let rotary_dim = (cfg.head_dim as f64 * cfg.partial_rotary_factor) as usize;
    if rotary_dim == 0 || rotary_dim % 2 != 0 || rotary_dim > cfg.head_dim {
        return Err(ModelError::Config(format!(
            "rotary_dim {rotary_dim} from head_dim {} * partial_rotary_factor {} is not a usable \
             even dimension",
            cfg.head_dim, cfg.partial_rotary_factor
        )));
    }
    let section: [usize; 3] = rp.mrope_section.clone().try_into().map_err(|_| {
        ModelError::Config(format!(
            "mrope_section must have 3 entries, got {:?}",
            rp.mrope_section
        ))
    })?;
    let n_freqs = rotary_dim / 2;
    if section.iter().sum::<usize>() != n_freqs {
        return Err(ModelError::Config(format!(
            "mrope_section {section:?} sums to {} but rotary_dim / 2 is {n_freqs}",
            section.iter().sum::<usize>()
        )));
    }
    #[allow(clippy::cast_precision_loss)]
    let inv_freq = (0..rotary_dim)
        .step_by(2)
        .map(|i| 1.0 / rp.rope_theta.powf(i as f64 / rotary_dim as f64))
        .collect();

    Ok(RopeParams {
        head_dim: cfg.head_dim,
        rotary_dim,
        theta: rp.rope_theta,
        inv_freq,
        mrope_section: section,
        mrope_interleaved: rp.mrope_interleaved,
        max_position_embeddings: cfg.max_position_embeddings,
    })
}

/// A `[3, seq_len]` mrope position tensor: row 0 temporal, row 1 height, row 2 width.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MropePositions {
    /// `[t, h, w]`, each `seq_len` long.
    pub rows: [Vec<u32>; 3],
    /// The value the *next* token after this block would take on all three rows.
    ///
    /// For text-only this is `start + seq_len`. With vision it is `max(rows) + 1`, which is
    /// what makes a text run after an image resume past the image's grid extent instead of
    /// past its token count.
    pub next_position: u32,
}

impl MropePositions {
    /// Text-only positions: all three streams carry the plain text position.
    ///
    /// **What changes when DeepStack vision arrives.** A vision block (between
    /// `<|vision_start|>` and `<|vision_end|>`) contributes `t_frames * h_patches * w_patches`
    /// tokens whose three rows are *not* equal: row 0 is the temporal index (constant within a
    /// frame, scaled by the video's second-per-grid), row 1 the patch row and row 2 the patch
    /// column, each offset by the running position. `next_position` then becomes
    /// `max over the block + 1` rather than `start + n_tokens`, and the interleave in
    /// [`RopeParams::mrope_stream_of`] stops being the identity. Everything below this line
    /// stays valid for the text runs between vision blocks.
    #[must_use]
    pub fn text_only(seq_len: usize, start: u32) -> Self {
        let row: Vec<u32> = (0..seq_len)
            .map(|i| start + u32::try_from(i).expect("sequence length fits in u32"))
            .collect();
        let next_position = start + u32::try_from(seq_len).expect("sequence length fits in u32");
        Self {
            rows: [row.clone(), row.clone(), row],
            next_position,
        }
    }

    /// Sequence length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows[0].len()
    }

    /// Whether the block is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows[0].is_empty()
    }

    /// True when every stream carries the same position — the text-only case, in which the
    /// mrope interleave is a no-op and a plain 1-D rope kernel is bit-exact.
    #[must_use]
    pub fn is_uniform(&self) -> bool {
        self.rows[0] == self.rows[1] && self.rows[1] == self.rows[2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> serde_json::Value {
        serde_json::json!({
            "text_config": {
                "head_dim": 256,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "full_attention_interval": 4,
                "vocab_size": 248_320,
                "max_position_embeddings": 262_144,
                "partial_rotary_factor": 0.25,
                "rope_parameters": {
                    "rope_theta": 10_000_000.0,
                    "rope_type": "default",
                    "mrope_section": [11, 11, 10],
                    "mrope_interleaved": true,
                    "partial_rotary_factor": 0.25
                }
            }
        })
    }

    #[test]
    fn rope_params_match_hand_computation() {
        let p = rope_params(&cfg()).unwrap();
        // head_dim 256 * 0.25 -> 64 rotated dims -> 32 frequency pairs.
        assert_eq!(p.head_dim, 256);
        assert_eq!(p.rotary_dim, 64);
        assert_eq!(p.n_freqs(), 32);
        assert_eq!(p.mrope_section, [11, 11, 10]);
        assert_eq!(p.mrope_section.iter().sum::<usize>(), p.n_freqs());
        assert!(p.mrope_interleaved);

        // inv_freq[i] = 10_000_000 ** (-2i / 64); values from CPython.
        assert_eq!(p.inv_freq[0], 1.0);
        for (i, want) in [
            (1_usize, 0.604_296_390_238_132_8_f64),
            (2, 0.365_174_127_254_837_67),
            (15, 0.000_523_299_114_681_494_7),
            (16, 0.000_316_227_766_016_837_94),
            (30, 2.738_419_634_264_361_4e-7),
            (31, 1.654_817_099_943_181_6e-7),
        ] {
            let got = p.inv_freq[i];
            assert!(
                (got - want).abs() <= want.abs() * 1e-15,
                "inv_freq[{i}] = {got:e}, want {want:e}"
            );
        }
        // The last pair is theta^-((rotary_dim-2)/rotary_dim) = theta^(-62/64).
        assert!((p.inv_freq[31] - 10_000_000f64.powf(-62.0 / 64.0)).abs() < 1e-22);
    }

    #[test]
    fn rope_params_reject_inconsistent_config() {
        let mut bad = cfg();
        bad["text_config"]["rope_parameters"]["mrope_section"] = serde_json::json!([11, 11, 11]);
        assert!(rope_params(&bad).is_err());

        let mut bad = cfg();
        bad["text_config"]["rope_parameters"]["rope_type"] = serde_json::json!("yarn");
        assert!(rope_params(&bad).is_err());

        let mut bad = cfg();
        bad["text_config"]["partial_rotary_factor"] = serde_json::json!(0.5);
        assert!(rope_params(&bad).is_err());
    }

    #[test]
    fn interleaved_stream_assignment() {
        let p = rope_params(&cfg()).unwrap();
        // section[1] * 3 == 33 > 32 and section[2] * 3 == 30, so slots 0..30 cycle t,h,w and
        // the tail (30, 31) is t, h.
        let streams: Vec<usize> = (0..p.n_freqs()).map(|i| p.mrope_stream_of(i)).collect();
        assert_eq!(&streams[..6], &[0, 1, 2, 0, 1, 2]);
        assert_eq!(&streams[29..], &[2, 0, 1]);
        assert_eq!(streams.iter().filter(|s| **s == 0).count(), 11);
        assert_eq!(streams.iter().filter(|s| **s == 1).count(), 11);
        assert_eq!(streams.iter().filter(|s| **s == 2).count(), 10);
    }

    #[test]
    fn text_only_positions_are_uniform() {
        let p = MropePositions::text_only(5, 0);
        assert_eq!(p.rows[0], vec![0, 1, 2, 3, 4]);
        assert!(p.is_uniform());
        assert_eq!(p.next_position, 5);
        assert_eq!(p.len(), 5);

        let cont = MropePositions::text_only(3, p.next_position);
        assert_eq!(cont.rows[2], vec![5, 6, 7]);
        assert_eq!(cont.next_position, 8);
        assert!(!MropePositions::text_only(0, 0).next_position.eq(&1));
        assert!(MropePositions::text_only(0, 0).is_empty());
    }
}
