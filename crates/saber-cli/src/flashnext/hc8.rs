//! FN108: the HC MIXING MATRICES at fp8 -- the reference's `hc8m` law (`qwen4exp-w4nl64-i8-hc8m.recipe`,
//! `ref/kernels/ref_hc_fp8.h`): e4m3 codes with one f32 scale per group of a row along K, 128 for the
//! downs `[320, 10240]`, K / 4 = 80 for the ups `[10240, 320]`, weight-only (the stream stays bf16).
//!
//! `saber flashnext-pack --pack <the -trunk8 variant> --hc8` ADDS `shared/hc8.{pack,json}` (refuses
//! an existing file; nothing else is touched): per TARGET mixing matrix `<name>` (the 96 trunk
//! connections + the final mixer = 194 records; the MTP layer's three stay bf16) one record
//! `<name>.hc8` = the codes `[N, K]` row-major followed by the scales `[N, K / G]` f32 -- the one
//! pointer `flashnext_hc8`'s `_h8` symbols take (`tools/gen_fn108_hc8.py`).
//!
//! The quantizer is the host reference's twin (`trunk_quant.hc8m_codes`): per (row, group)
//! `s = amax / 448` in fp32 (all-zero group: s = 0, codes 0), each element `bf16 / s` (fp32
//! division) to e4m3 round-to-nearest-even (`saber_cpux::e4m3_rne`). The value the kernel multiplies
//! is `bf16(f32(code) x s)` ([`dequant_bf16`]): an `_h8` symbol on the record == its parent symbol on
//! that bf16 matrix, byte for byte (the card gate `tools/capture/fn108/hc8_gate.py`).

use std::io::Write;
use std::path::Path;

use anyhow::{ensure, Context};
use saber_kernels::manifest::sha256_hex;
use serde_json::json;

use super::store::PackRecs;
use super::trunk4::{ckpt_dir, source_records};

/// The additional pack file's stem (in `shared/`).
pub const STEM: &str = "hc8";
/// The ledger class of the records (`fn.w.shared.hc8`).
pub const CLASS: &str = "hc8";
/// The plan class of the bf16 records they replace (`fn.w.shared.hc`).
pub const BF16_CLASS: &str = "hc";
/// Target mixing matrices: 48 layers x 2 connections x (down, up) + the final mixer's two.
pub const RECORDS: usize = 194;

/// The reference's group: 128 along K for a down (K = 10,240), K / 4 for an up (K = 320 -> 80).
#[must_use]
pub fn group(k: usize) -> usize {
    if k >= 1024 {
        128
    } else {
        k / 4
    }
}

/// The record name of a source mixing matrix.
#[must_use]
pub fn rec_name(src: &str) -> String {
    format!("{src}.hc8")
}

/// Whether `name` is a TARGET mixing matrix (a decoder layer's connection or the final mixer; the
/// MTP layer's `mtp.` mixers are not).
#[must_use]
pub fn target_mixer(name: &str) -> bool {
    name.starts_with(super::model::P)
        && (name.ends_with("input_mix_weight_down.weight")
            || name.ends_with("input_mix_weight_up.weight"))
}

fn bf16(b: &[u8], i: usize) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])) << 16)
}

/// Quantize a bf16 `[n, k]` mixing matrix: the record bytes (codes `[n, k]`, then f32 scales
/// `[n, k / G]`).
pub fn quantize(w: &[u8], n: usize, k: usize) -> anyhow::Result<Vec<u8>> {
    ensure!(w.len() == 2 * n * k, "bf16 [{n}, {k}]: {} B", w.len());
    let g = group(k);
    ensure!(
        g % 16 == 0 && k % g == 0,
        "hc8m needs K % G, G % 16 (K {k}, G {g})"
    );
    let ng = k / g;
    let mut codes = vec![0u8; n * k];
    let mut sc = vec![0f32; n * ng];
    for r in 0..n {
        for q in 0..ng {
            let mut amax = 0f32;
            for c in q * g..(q + 1) * g {
                let v = bf16(w, r * k + c);
                ensure!(v.is_finite(), "row {r}: non-finite weight");
                amax = amax.max(v.abs());
            }
            let s = amax / 448.0;
            sc[r * ng + q] = s;
            if s == 0.0 {
                continue;
            }
            for c in q * g..(q + 1) * g {
                let x = bf16(w, r * k + c) / s;
                ensure!(x.abs() < 464.0, "row {r} col {c}: {x} out of e4m3 range");
                codes[r * k + c] = saber_cpux::e4m3_rne(x);
            }
        }
    }
    codes.extend(sc.iter().flat_map(|x| x.to_le_bytes()));
    Ok(codes)
}

#[allow(dead_code)] // used by dequant_bf16
/// fp32 -> bf16 bits, round to nearest even (finite inputs: the card's `(__bf16)` conversion).
#[must_use]
pub fn f32_bf16(v: f32) -> u16 {
    let b = v.to_bits();
    let r = b + 0x7fff + ((b >> 16) & 1);
    (r >> 16) as u16
}

#[allow(dead_code)] // used by dequant_bf16
fn e4m3_f32(b: u8) -> f32 {
    let (e, m) = (i32::from((b >> 3) & 15), f32::from(b & 7));
    let v = if e == 0 {
        m / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    if b & 0x80 != 0 {
        -v
    } else {
        v
    }
}

#[allow(dead_code)] // the host law (the tests; the card gate is its torch twin)
/// The bf16 matrix an `_h8` symbol multiplies: `bf16(f32(code) x scale)` (fp32 product, one RNE
/// rounding to bf16), little-endian `[n, k]`.
pub fn dequant_bf16(rec: &[u8], n: usize, k: usize) -> anyhow::Result<Vec<u8>> {
    let g = group(k);
    ensure!(
        rec.len() == n * k + 4 * n * (k / g),
        "hc8 [{n}, {k}]: {} B",
        rec.len()
    );
    let sc = &rec[n * k..];
    let mut out = Vec::with_capacity(2 * n * k);
    for r in 0..n {
        for c in 0..k {
            let i = r * (k / g) + c / g;
            let s = f32::from_le_bytes([sc[4 * i], sc[4 * i + 1], sc[4 * i + 2], sc[4 * i + 3]]);
            let v = e4m3_f32(rec[r * k + c]) * s;
            out.extend_from_slice(&f32_bf16(v).to_le_bytes());
        }
    }
    Ok(out)
}

/// The bytes of `shared/hc8.pack` (the ledger's added class under `--fn-trunk w8-hc8[-head]`).
pub fn shared_bytes(pack: &Path) -> anyhow::Result<u64> {
    let p = pack.join(format!("shared/{STEM}.json"));
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&p).with_context(|| {
            format!(
                "{}: --fn-trunk w8-hc8 needs `saber flashnext-pack --pack <the -trunk8 variant> --hc8`",
                p.display()
            )
        })?)?;
    let recs = v["records"].as_array().context("records")?;
    ensure!(
        recs.len() == RECORDS,
        "{}: {} records (want {RECORDS})",
        p.display(),
        recs.len()
    );
    v["bytes"].as_u64().context("bytes")
}

/// The bf16 target mixing matrices' bytes in `shared/dense.json` (class [`BF16_CLASS`]) -- NOT
/// uploaded under `--fn-trunk w8-hc8[-head]`.
pub fn dropped_bf16(pack: &Path) -> anyhow::Result<u64> {
    let mut tot = 0u64;
    let mut n = 0usize;
    for r in source_records(pack, "shared")? {
        let name = r["name"].as_str().context("name")?;
        if target_mixer(name) {
            ensure!(
                r["class"] == BF16_CLASS,
                "{name}: class {} (want {BF16_CLASS})",
                r["class"]
            );
            tot += r["bytes"].as_u64().context("bytes")?;
            n += 1;
        }
    }
    ensure!(n == RECORDS, "{n} target mixing matrices (want {RECORDS})");
    Ok(tot)
}

/// Add `shared/hc8.{pack,json}` to `pack` (refuses an existing member). Returns (records, bytes).
pub fn build(pack: &Path) -> anyhow::Result<(usize, u64)> {
    let pack = pack.canonicalize()?;
    let (pf, jf) = (
        pack.join("shared").join(format!("{STEM}.pack")),
        pack.join("shared").join(format!("{STEM}.json")),
    );
    ensure!(
        !pf.exists() && !jf.exists(),
        "{}: the hc8 member exists (never rewritten in place)",
        pf.display()
    );
    let recs = PackRecs::open(&pack, &ckpt_dir(&pack)?)?;
    let tmp = pack.join("shared").join(format!("{STEM}.pack.tmp"));
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    let (mut outr, mut off) = (Vec::new(), 0u64);
    for r in source_records(&pack, "shared")? {
        let name = r["name"].as_str().context("name")?;
        if !target_mixer(name) {
            continue;
        }
        ensure!(r["dtype"] == "BF16", "{name}: not a bf16 record");
        let shape: Vec<usize> = r["shape"]
            .as_array()
            .context("shape")?
            .iter()
            .map(|x| x.as_u64().unwrap_or(0) as usize)
            .collect();
        ensure!(shape.len() == 2, "{name}: shape {shape:?}");
        let (n, k) = (shape[0], shape[1]);
        let w = recs.raw(0, name)?;
        let b = quantize(&w, n, k).with_context(|| name.to_string())?;
        outr.push(
            json!({"name": rec_name(name), "class": CLASS, "source": name,
            "shape": [n, k], "group": group(k), "codes_bytes": n * k,
            "storage": {"pack": format!("{STEM}.pack"), "data_off": off, "length": b.len(),
            "sha256": sha256_hex(&b)}}),
        );
        f.write_all(&b)?;
        let pad = (256 - b.len() as u64 % 256) % 256;
        f.write_all(&vec![0u8; pad as usize])?;
        off += b.len() as u64 + pad;
    }
    ensure!(
        outr.len() == RECORDS,
        "{} target mixing matrices (want {RECORDS})",
        outr.len()
    );
    f.into_inner().context("flush")?.sync_all()?;
    std::fs::rename(&tmp, &pf)?;
    let doc = json!({"records": outr, "bytes": off, "law": "the reference hc8m (FN108): e4m3 codes [N, K] \
        row-major then f32 scales [N, K/G], s = amax/448 per (row, G along K), G = 128 (K 10240) / \
        80 (K 320), RNE; the kernel multiplies bf16(code x s)"});
    std::fs::write(&jf, serde_json::to_vec_pretty(&doc)?)?;
    println!(
        "flashnext-pack shared: {} hc8 records, {:.3} GiB -> {}",
        outr.len(),
        off as f64 / (1u64 << 30) as f64,
        pf.display()
    );
    Ok((outr.len(), off))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_bf16(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| f32_bf16(*x).to_le_bytes()).collect()
    }

    /// FN108: the groups are the reference's (128 down, 80 up), the record is codes then scales, the
    /// value law is bf16(code x scale) and an outlier only coarsens ITS group; NEGATIVE CONTROL:
    /// the 128 x 128 block scale (FN61b's bs128) loses the other groups of that row.
    #[test]
    fn fn108_the_hc8m_record_is_ref_row_group_law() {
        assert_eq!(group(10240), 128);
        assert_eq!(group(320), 80);
        let (n, k) = (16usize, 320usize);
        let mut v: Vec<f32> = (0..n * k)
            .map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 2e-6)
            .collect();
        v[3] = 1.0; // row 0, group 0: an outlier
        let w = to_bf16(&v);
        let rec = quantize(&w, n, k).unwrap();
        assert_eq!(rec.len(), n * k + 4 * n * 4);
        let s0 = f32::from_le_bytes(rec[n * k..n * k + 4].try_into().unwrap());
        assert_eq!(s0, 1.0 / 448.0);
        assert_eq!(rec[3], saber_cpux::e4m3_rne(448.0));
        let deq = dequant_bf16(&rec, n, k).unwrap();
        let err = |a: &[u8]| -> f32 {
            (0..n * k)
                .filter(|i| *i >= 80 && *i < k) // row 0, groups 1..3
                .map(|i| (bf16(a, i) - bf16(&w, i)).abs())
                .fold(0.0, f32::max)
        };
        // row 0's other groups (values <= 1e-6) keep their own scale: e4m3's 2^-4 relative step
        assert!(err(&deq) < 1e-6 * 0.07, "{}", err(&deq));
        // CONTROL: one scale for the row (as a 128-block would give) crushes them
        let mut ctl = rec.clone();
        for q in 1..4 {
            ctl[n * k + 4 * q..n * k + 4 * q + 4].copy_from_slice(&s0.to_le_bytes());
        }
        assert!(err(&dequant_bf16(&ctl, n, k).unwrap()) > err(&deq) * 4.0);
        assert!(target_mixer(
            "model.language_model.layers.3.attn_hyper_connection.input_mix_weight_down.weight"
        ));
        assert!(target_mixer(
            "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight"
        ));
        assert!(!target_mixer(
            "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight"
        ));
        assert!(!target_mixer(
            "model.language_model.layers.3.attn_hyper_connection.block_inject_weight.weight"
        ));
    }
}
