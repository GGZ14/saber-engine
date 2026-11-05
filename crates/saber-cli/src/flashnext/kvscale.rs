//! FN68b: the fp8 K/V pages' SCALE LAW (`--fn-kv-dtype fp8 --fn-kv-scales pack|unit|FILE`).
//!
//! FN68 stored e4m3 at one per-tensor scale 1.0 (the checkpoint carries no KV scales). FN68b reads
//! one scale per (QSA layer, KV head) for K and for V from a `fn68b-kv-scales/1` JSON (calibrated
//! once on the host from the FN3 prompts OUTSIDE the 25-prompt gate, `tools/capture/fn68b/
//! report.py calib`, stored in the pack as `kv_scales.json`): the append divides by the scale
//! (`fn_kv_append_ring_h256_fp8`'s `k_scale` / `v_scale` arguments, unchanged code object), the
//! attention multiplies back through the saber descale pointers (`k_descale` folds into the
//! logit multiplier, `v_descale` into the output's 1/l -- prefill in-kernel, decode in the
//! combine). At TP=2 a rank owns ONE KV head, so a (layer, head) scale is a per-rank scalar; the
//! descale tables are `[layer][k|v][DESC_N]` f32 (one entry a sequence: the kernels index
//! `seq * kv_heads + kvh` with `kv_heads` 1).
//!
//! `unit` = FN68's law (every scale 1.0, the descale pointers null: FN68's bytes, the control). A
//! layer the file does not name (the MTP layer 48 unless calibrated) runs at 1.0.

use std::path::Path;

use anyhow::{ensure, Context};

/// Layers a table covers: the 48 target layers + the MTP layer (index 48).
pub const TABLE_LAYERS: usize = 49;
/// Descale entries a (layer, K|V): the widest `num_seqs * kv_heads` any attention launch indexes
/// (a width-1-class verify step's per-row pseudo-sequences: <= 8 x 5 = 40; asserted per launch).
pub const DESC_N: usize = 128;
/// The pack member holding the calibrated law.
pub const PACK_FILE: &str = "kv_scales.json";
/// The file format tag.
pub const FORMAT: &str = "fn68b-kv-scales/1";

/// One boot's K/V scales, per layer and KV head (`[layer][head]`, 1.0 where unnamed).
#[derive(Clone, Debug, PartialEq)]
pub struct KvScales {
    pub k: Vec<[f32; 2]>,
    pub v: Vec<[f32; 2]>,
    /// Where the law came from (the boot line) and its law string.
    pub src: String,
    pub law: String,
}

impl KvScales {
    /// FN68's law: every scale 1.0.
    #[must_use]
    pub fn unit() -> KvScales {
        KvScales {
            k: vec![[1.0; 2]; TABLE_LAYERS],
            v: vec![[1.0; 2]; TABLE_LAYERS],
            src: "unit".into(),
            law: "1.0 (FN68)".into(),
        }
    }

    /// Whether every scale is 1.0 (the descale pointers then stay null: FN68's kernargs).
    #[cfg(test)]
    #[must_use]
    pub fn is_unit(&self) -> bool {
        self.k
            .iter()
            .chain(&self.v)
            .all(|h| h.iter().all(|&s| s == 1.0))
    }

    /// Parse a `fn68b-kv-scales/1` document: `{"format", "law", "layers": {"<L>": {"k": [h0, h1],
    /// "v": [h0, h1]}}}` -- every scale finite and > 0, layers < 49.
    pub fn parse(text: &str, src: &str) -> anyhow::Result<KvScales> {
        let v: serde_json::Value = serde_json::from_str(text).context("kv scales: not JSON")?;
        ensure!(
            v.get("format").and_then(serde_json::Value::as_str) == Some(FORMAT),
            "kv scales {src}: format is not {FORMAT}"
        );
        let mut s = KvScales::unit();
        s.src = src.into();
        s.law = v
            .get("law")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .into();
        let layers = v
            .get("layers")
            .and_then(serde_json::Value::as_object)
            .context("kv scales: no `layers` object")?;
        for (l, e) in layers {
            let li: usize = l
                .parse()
                .with_context(|| format!("kv scales: layer `{l}`"))?;
            ensure!(
                li < TABLE_LAYERS,
                "kv scales: layer {li} past {TABLE_LAYERS}"
            );
            for (w, dst) in [("k", &mut s.k[li]), ("v", &mut s.v[li])] {
                let a = e
                    .get(w)
                    .and_then(serde_json::Value::as_array)
                    .with_context(|| format!("kv scales: layer {li} has no `{w}` pair"))?;
                ensure!(a.len() == 2, "kv scales: layer {li} `{w}`: two heads");
                for (h, x) in a.iter().enumerate() {
                    let f = x.as_f64().context("kv scales: a number")? as f32;
                    ensure!(
                        f.is_finite() && f > 0.0,
                        "kv scales: layer {li} `{w}` head {h}: {f} is not a positive scale"
                    );
                    dst[h] = f;
                }
            }
        }
        Ok(s)
    }

    /// Load from a file.
    pub fn load(path: &Path) -> anyhow::Result<KvScales> {
        let t = std::fs::read_to_string(path)
            .with_context(|| format!("--fn-kv-scales: {}", path.display()))?;
        KvScales::parse(&t, &path.display().to_string())
    }

    /// Rank `rank`'s (K, V) scale at `layer` (TP=2: the rank's KV head is its rank).
    #[must_use]
    pub fn at(&self, layer: usize, rank: usize) -> (f32, f32) {
        (self.k[layer][rank], self.v[layer][rank])
    }

    /// Rank `rank`'s descale table bytes: `[layer][k|v][DESC_N]` f32 little-endian.
    #[must_use]
    pub fn table(&self, rank: usize) -> Vec<u8> {
        let mut b = Vec::with_capacity(TABLE_LAYERS * 2 * DESC_N * 4);
        for l in 0..TABLE_LAYERS {
            let (k, v) = self.at(l, rank);
            for s in [k, v] {
                for _ in 0..DESC_N {
                    b.extend_from_slice(&s.to_le_bytes());
                }
            }
        }
        b
    }

    /// Byte offset of (layer, K = 0 | V = 1) in [`Self::table`].
    #[must_use]
    pub fn table_off(layer: usize, which: usize) -> u64 {
        ((layer * 2 + which) * DESC_N * 4) as u64
    }

    /// A 64-bit identity of every scale (FNV-1a over the f32 bits): the prompt cache's key chain
    /// carries it, so a snapshot written under one law never reloads under another.
    #[must_use]
    pub fn ident(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for s in self.k.iter().chain(&self.v).flatten() {
            for byte in s.to_le_bytes() {
                h ^= u64::from(byte);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    /// The boot line.
    #[must_use]
    pub fn line(&self, rank: usize) -> String {
        let named: Vec<String> = (0..TABLE_LAYERS)
            .filter(|&l| self.k[l] != [1.0; 2] || self.v[l] != [1.0; 2])
            .map(|l| {
                let (k, v) = self.at(l, rank);
                format!("L{l} k {k:.4e} v {v:.4e}")
            })
            .collect();
        format!(
            "fnkvscale rank{rank}: {} ({}; ident {:016x}){}{}",
            self.src,
            self.law,
            self.ident(),
            if named.is_empty() { "" } else { ": " },
            named.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = r#"{"format": "fn68b-kv-scales/1", "law": "amax x 2 / 448",
        "layers": {"3": {"k": [0.05, 0.0625], "v": [0.004, 0.008]}, "47": {"k": [1, 1], "v": [0.5, 2]}}}"#;

    #[test]
    fn a_scales_file_parses_per_layer_and_head_and_unnamed_layers_stay_unit() {
        let s = KvScales::parse(DOC, "t").unwrap();
        assert_eq!(s.at(3, 0), (0.05, 0.004));
        assert_eq!(s.at(3, 1), (0.0625, 0.008));
        assert_eq!(s.at(47, 1), (1.0, 2.0));
        assert_eq!(s.at(48, 0), (1.0, 1.0));
        assert_eq!(s.at(7, 1), (1.0, 1.0));
        assert!(!s.is_unit());
        assert!(KvScales::unit().is_unit());
        assert_ne!(s.ident(), KvScales::unit().ident());
        assert!(
            s.line(0).contains("L3 k 5.0000e-2 v 4.0000e-3"),
            "{}",
            s.line(0)
        );
    }

    #[test]
    fn a_bad_scales_file_is_refused_by_name() {
        let bad = [
            r#"{"format": "x", "layers": {}}"#,
            r#"{"format": "fn68b-kv-scales/1", "layers": {"49": {"k": [1, 1], "v": [1, 1]}}}"#,
            r#"{"format": "fn68b-kv-scales/1", "layers": {"3": {"k": [1], "v": [1, 1]}}}"#,
            r#"{"format": "fn68b-kv-scales/1", "layers": {"3": {"k": [0, 1], "v": [1, 1]}}}"#,
            r#"{"format": "fn68b-kv-scales/1", "layers": {"3": {"k": [1, 1]}}}"#,
        ];
        for b in bad {
            assert!(KvScales::parse(b, "t").is_err(), "{b}");
        }
    }

    #[test]
    fn the_descale_table_holds_each_layers_scale_in_every_sequence_entry() {
        let s = KvScales::parse(DOC, "t").unwrap();
        let t = s.table(1);
        assert_eq!(t.len(), TABLE_LAYERS * 2 * DESC_N * 4);
        let f = |off: u64, i: usize| {
            let o = off as usize + 4 * i;
            f32::from_le_bytes(t[o..o + 4].try_into().unwrap())
        };
        for i in [0, 39, DESC_N - 1] {
            assert_eq!(f(KvScales::table_off(3, 0), i), 0.0625);
            assert_eq!(f(KvScales::table_off(3, 1), i), 0.008);
            assert_eq!(f(KvScales::table_off(47, 1), i), 2.0);
            assert_eq!(f(KvScales::table_off(5, 0), i), 1.0);
        }
    }
}
