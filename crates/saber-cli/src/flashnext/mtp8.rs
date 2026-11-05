//! FN50: Qwen's FIRST-PARTY fp8 MTP experts as an ADDITIONAL pack file (the pack is never
//! rewritten in place). `saber flashnext-pack --mtp-experts fp8` reads the 512 routed MTP experts
//! of `model-fp8-mtp-ple.safetensors` (e4m3 block 128 x 128, bf16 `weight_scale_inv`) and writes,
//! per EP rank, `rank<r>/mtp_fp8.pack` + `rank<r>/mtp_fp8.json` (records in FN2's form: pack file,
//! data offset, length, sha256). Per expert: `mtp.layer.0.expert.<e>.gate_up.w8` (gate rows then up
//! rows, `[1280, 2560]` e4m3 in the reference's preshuffled order -- the `w` of `fn_moe_a8w8_bs128_*`),
//! `.gate_up.s8` (`[10, 20]` fp32 = the bf16 scales widened, exact), `.down.w8` (`[2560, 640]`),
//! `.down.s8` (`[20, 5]`). The engine loads them only under `--fn-mtp-experts fp8`.

use std::io::Write;
use std::path::Path;

use anyhow::{ensure, Context};
use saber_kernels::manifest::sha256_hex;
use serde_json::json;

use super::store::Ckpt;

/// The additional pack file's stem (per rank directory).
pub const STEM: &str = "mtp_fp8";
/// Routed MTP experts per EP rank.
pub const ELOC: u32 = 256;
const H: usize = 2560;
const I: usize = 640;

/// The record names of expert `e` (gate_up weight, gate_up scales, down weight, down scales).
#[must_use]
pub fn names(e: u32) -> [String; 4] {
    let p = format!("mtp.layer.0.expert.{e}");
    [
        format!("{p}.gate_up.w8"),
        format!("{p}.gate_up.s8"),
        format!("{p}.down.w8"),
        format!("{p}.down.s8"),
    ]
}

/// the reference's preshuffle of an e4m3 `[n, k]` matrix: `w.view(n/16, 16, k/32, 2, 16).permute(0, 2, 3,
/// 1, 4)` -- n-group `g`, k-run `r = k / 16`: byte `g * k * 16 + r * 256 + (n % 16) * 16 + k % 16`.
#[must_use]
pub fn preshuffle(w: &[u8], n: usize, k: usize) -> Vec<u8> {
    assert_eq!(w.len(), n * k);
    assert!(n % 16 == 0 && k % 32 == 0);
    let mut o = vec![0u8; n * k];
    for row in 0..n {
        for col in 0..k {
            o[(row / 16) * k * 16 + (col / 16) * 256 + (row % 16) * 16 + col % 16] =
                w[row * k + col];
        }
    }
    o
}

/// bf16 scales widened to fp32 (exact).
#[must_use]
pub fn scales_f32(b: &[u8]) -> Vec<u8> {
    b.chunks_exact(2)
        .flat_map(|c| (u32::from(u16::from_le_bytes([c[0], c[1]])) << 16).to_le_bytes())
        .collect()
}

/// The four record payloads of global expert `e` from the fp8 checkpoint.
pub fn expert_bytes(ck: &Ckpt, e: u32) -> anyhow::Result<[Vec<u8>; 4]> {
    let x = format!("mtp.layers.0.mlp.experts.{e}");
    let w = |n: &str, rows: usize, cols: usize| -> anyhow::Result<Vec<u8>> {
        let b = ck.raw(&format!("{x}.{n}.weight"))?;
        ensure!(b.len() == rows * cols, "{x}.{n}.weight: {} B", b.len());
        Ok(b)
    };
    let s = |n: &str, cnt: usize| -> anyhow::Result<Vec<u8>> {
        let b = ck.raw(&format!("{x}.{n}.weight_scale_inv"))?;
        ensure!(
            b.len() == cnt * 2,
            "{x}.{n}.weight_scale_inv: {} B",
            b.len()
        );
        Ok(scales_f32(&b))
    };
    let mut gu = w("gate_proj", I, H)?;
    gu.extend(w("up_proj", I, H)?);
    let mut gs = s("gate_proj", 5 * 20)?;
    gs.extend(s("up_proj", 5 * 20)?);
    Ok([
        preshuffle(&gu, 2 * I, H),
        gs,
        preshuffle(&w("down_proj", H, I)?, H, I),
        s("down_proj", 20 * 5)?,
    ])
}

/// Write `rank<r>/mtp_fp8.{pack,json}` for both ranks (refuses if either exists). Returns the bytes
/// per rank.
pub fn build(pack: &Path, fp8_ckpt: &Path) -> anyhow::Result<[u64; 2]> {
    let ck = Ckpt::open(fp8_ckpt)?;
    let mut tot = [0u64; 2];
    for r in 0..2u32 {
        let dir = pack.join(format!("rank{r}"));
        let (pf, jf) = (
            dir.join(format!("{STEM}.pack")),
            dir.join(format!("{STEM}.json")),
        );
        ensure!(
            !pf.exists() && !jf.exists(),
            "{}: the fp8 MTP file exists (never rewritten in place)",
            pf.display()
        );
        let tmp = dir.join(format!("{STEM}.pack.tmp"));
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        let mut recs = Vec::new();
        let mut off = 0u64;
        for i in 0..ELOC {
            let e = r * ELOC + i;
            for (name, b) in names(e).iter().zip(expert_bytes(&ck, e)?) {
                recs.push(json!({"name": name, "class": "mtp.experts", "storage": {
                    "pack": format!("{STEM}.pack"), "data_off": off, "length": b.len(),
                    "sha256": sha256_hex(&b)}}));
                f.write_all(&b)?;
                // 256-byte aligned records (the a8w8 body reads 16-byte vectors)
                let pad = (256 - b.len() as u64 % 256) % 256;
                f.write_all(&vec![0u8; pad as usize])?;
                off += b.len() as u64 + pad;
            }
        }
        f.into_inner().context("flush")?.sync_all()?;
        std::fs::rename(&tmp, &pf)?;
        let doc = json!({"records": recs, "source": fp8_ckpt.join("model-fp8-mtp-ple.safetensors"),
            "class": "mtp.experts", "bytes": off, "layout": "fp8_preshuffle + fp8_block_scale (FN50)"});
        std::fs::write(&jf, serde_json::to_vec_pretty(&doc)?)?;
        tot[r as usize] = off;
        println!(
            "flashnext-pack rank{r}: {} records, {:.3} GiB -> {}",
            recs.len(),
            off as f64 / (1u64 << 30) as f64,
            pf.display()
        );
    }
    Ok(tot)
}

/// The fp8 file's bytes on `rank` (the ledger's `mtp.experts` class under the fp8 arm).
pub fn rank_bytes(pack: &Path, rank: u32) -> anyhow::Result<u64> {
    let p = pack.join(format!("rank{rank}/{STEM}.json"));
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&p).with_context(|| {
            format!(
                "{}: --fn-mtp-experts fp8 needs `saber flashnext-pack --mtp-experts fp8`",
                p.display()
            )
        })?)?;
    v["bytes"].as_u64().context("bytes")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_preshuffle_is_the_reference_fragment_order_and_scales_widen_exactly() {
        let (n, k) = (32usize, 64usize);
        let w: Vec<u8> = (0..n * k).map(|i| (i * 7 % 251) as u8).collect();
        let s = preshuffle(&w, n, k);
        // the reference: view(n/16, 16, k/32, 2, 16).permute(0, 2, 3, 1, 4)
        let mut want = Vec::new();
        for g in 0..n / 16 {
            for kc in 0..k / 32 {
                for h in 0..2 {
                    for r in 0..16 {
                        for c in 0..16 {
                            want.push(w[(g * 16 + r) * k + kc * 32 + h * 16 + c]);
                        }
                    }
                }
            }
        }
        assert_eq!(s, want);
        // control: the identity order differs
        assert_ne!(s, w);
        let b = [0x80u8, 0x3f, 0x00, 0xc0]; // bf16 1.0, -2.0
        let f = scales_f32(&b);
        assert_eq!(f32::from_le_bytes(f[0..4].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_le_bytes(f[4..8].try_into().unwrap()), -2.0);
        assert_eq!(names(3)[1], "mtp.layer.0.expert.3.gate_up.s8");
    }
}
