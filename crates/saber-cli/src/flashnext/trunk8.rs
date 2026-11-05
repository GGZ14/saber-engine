//! FN61b: the Flash-Next DENSE TRUNK at fp8 W8 (e4m3, one fp32 scale per 128 x 128 block -- the
//! format of the drafter's W8A8 GEMM `dflash_gemm_a8w8_bs128_*` and of Qwen's own fp8 MTP experts)
//! as a pack VARIANT beside the served pack (`saber flashnext-pack --trunk w8 --out-pack DIR`):
//! the variant links every file of the source pack (never rewritten) and ADDS `trunk8.{pack,json}`
//! to `rank0/`, `rank1/` (no shared member: see below). Per bf16 source record `<name>` two
//! records: `<name>.w8.w` (e4m3 `[N, K]` in the reference's preshuffled order, `mtp8::preshuffle`: the
//! kernel's `w`) and `<name>.w8.ws` (fp32 `[N/128, K/128]`, the kernel's `ws`).
//!
//! The quantizer (`trunk_quant.fp8_bs128`, the host reference's twin): per 128 x 128 block,
//! `s = amax / 448` in fp32 (no scale for an all-zero block: s = 0, every byte 0), each element
//! `x / s` (fp32 division) to e4m3 round-to-nearest-even (`saber_cpux::e4m3_rne`).
//!
//! Families (FN61's `trunk4::trunk_family`) whose shapes the 128 x 128 tile takes: GDN
//! `in_proj_qkvz` `[8192, 2560]` + `out_proj` `[2560, 3072]`, QSA q `[6144, 2560]` / k / v
//! `[256, 2560]` / o `[2560, 3072]`, the lm_head `[124160, 2560]` (rank halves). The HC mixers stay
//! bf16: down `[320, 10240]` (N = 320 is not a multiple of the 128-wide n tile) and up
//! `[10240, 320]` (K = 320 is not a multiple of the 128-deep k tile), and they run inside FN11's
//! fused mix kernels (norm prologue, amax/quant tails), not on a GEMM route. GDN `in_proj_ba`
//! (N = 48), the indexer, router, shared gate, norms, PLE and every MTP record stay as they are.

use std::io::Write;
use std::path::Path;

use anyhow::{ensure, Context};
use saber_kernels::manifest::sha256_hex;
use serde_json::json;

use super::store::PackRecs;
use super::trunk4::{ckpt_dir, link_variant, source_records, trunk_family};

/// The additional pack file's stem (per rank directory).
pub const STEM: &str = "trunk8";
/// The weight block (both axes).
pub const BLK: usize = 128;

/// FN61c: the bytes of `rank<r>/trunk8.pack` (the ledger's added weight class under `--fn-trunk w8`).
pub fn rank_bytes(pack: &Path, rank: u32, head: bool) -> anyhow::Result<u64> {
    let p = pack.join(format!("rank{rank}/{STEM}.json"));
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&p).with_context(|| {
            format!(
                "{}: --fn-trunk w8 needs a `saber flashnext-pack --trunk w8` variant",
                p.display()
            )
        })?)?;
    json_bytes(&v, head)
}

/// FN61e: whether a dense record is one of the bf16 trunk records the W8 records replace (dropped
/// from VRAM under `--fn-trunk w8` / `w8-head`): a target layer's GDN `in_proj_qkvz` / `out_proj`
/// or QSA `q` / `k` / `v` / `o` projection. `in_proj_ba`, the indexer, the MTP layer and the head
/// stay.
#[must_use]
pub fn dropped(name: &str) -> bool {
    if let Some(r) = name.strip_prefix("layer.") {
        return r.ends_with(".gdn.in_proj_qkvz.w");
    }
    let Some(r) = name.strip_prefix("model.language_model.layers.") else {
        return false;
    };
    [
        ".linear_attn.out_proj.weight",
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
    ]
    .iter()
    .any(|s| r.ends_with(s))
}

/// FN61e: the dropped bf16 records' bytes of `rank` by plan class (the pack's `rank<r>/dense.json`
/// records: 36 GDN layers x 2 + 12 QSA layers x 4 = 120 records).
pub fn dropped_bf16(pack: &Path, rank: u32) -> anyhow::Result<Vec<(String, u64)>> {
    let p = pack.join(format!("rank{rank}/dense.json"));
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p)?)?;
    dropped_by_class(&v).with_context(|| p.display().to_string())
}

fn dropped_by_class(v: &serde_json::Value) -> anyhow::Result<Vec<(String, u64)>> {
    let mut by: std::collections::BTreeMap<String, u64> = Default::default();
    let mut n = 0;
    for r in v["records"].as_array().context("records")? {
        if dropped(r["name"].as_str().context("name")?) {
            *by.entry(r["class"].as_str().context("class")?.to_string())
                .or_default() += r["bytes"].as_u64().context("bytes")?;
            n += 1;
        }
    }
    ensure!(
        n == 120,
        "{n} dropped bf16 trunk records (want 36 x 2 + 12 x 4)"
    );
    Ok(by.into_iter().collect())
}

/// The record class of the W8 head (`lm_head.weight.w8.{w,ws}`), uploaded only by `w8-head`.
pub const HEAD_CLASS: &str = "trunk8.head";

/// FN61g: the bf16 head record's (plan class, bytes) of `rank` -- NOT uploaded under
/// `--fn-trunk w8-head` (every post reads the W8 head; the drafter's rerank too).
pub fn head_bf16(pack: &Path, rank: u32) -> anyhow::Result<(String, u64)> {
    let p = pack.join(format!("rank{rank}/dense.json"));
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p)?)?;
    head_bf16_of(&v).with_context(|| p.display().to_string())
}

fn head_bf16_of(v: &serde_json::Value) -> anyhow::Result<(String, u64)> {
    let mut hit = Vec::new();
    for r in v["records"].as_array().context("records")? {
        if r["name"].as_str() == Some("lm_head.weight") {
            hit.push((
                r["class"].as_str().context("class")?.to_string(),
                r["bytes"].as_u64().context("bytes")?,
            ));
        }
    }
    ensure!(
        hit.len() == 1,
        "{} bf16 lm_head.weight records (want 1)",
        hit.len()
    );
    Ok(hit.remove(0))
}

/// FN61g: FN33's int2 draft-head pack of a bf16 head half `w` (`[n, 2560]` little-endian bf16)
/// built on the HOST -- `saber_kernels::fn33::quant_row` a row (the host law the card's
/// `fn_draft_head_int2_quant` was gated against) -- for `w8-head`, where the bf16 head is never
/// resident on the card. Returns (codes `[n, ROW_BYTES]`, S `[n, NG]` bf16, ZS `[n, NG]` bf16) in
/// the card pack's layout.
pub fn int2_pack_host(w: &[u8]) -> anyhow::Result<[Vec<u8>; 3]> {
    use saber_kernels::fn33::{quant_row, K, NG, ROW_BYTES};
    let rb = 2 * K;
    ensure!(
        !w.is_empty() && w.len() % rb == 0,
        "head half: {} B is not [n, {K}] bf16",
        w.len()
    );
    let n = w.len() / rb;
    let mut codes = vec![0u8; n * ROW_BYTES];
    let mut s = vec![0u8; n * NG * 2];
    let mut zs = vec![0u8; n * NG * 2];
    let th = std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get);
    let per = n.div_ceil(th.clamp(1, 16));
    std::thread::scope(|scope| {
        for (((wc, cc), sc), zc) in w
            .chunks(per * rb)
            .zip(codes.chunks_mut(per * ROW_BYTES))
            .zip(s.chunks_mut(per * NG * 2))
            .zip(zs.chunks_mut(per * NG * 2))
        {
            scope.spawn(move || {
                for (i, row) in wc.chunks_exact(rb).enumerate() {
                    let r: Vec<u16> = row
                        .chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    let (c, sv, zv) = quant_row(&r);
                    cc[i * ROW_BYTES..(i + 1) * ROW_BYTES].copy_from_slice(&c);
                    for g in 0..NG {
                        let o = (i * NG + g) * 2;
                        sc[o..o + 2].copy_from_slice(&sv[g].to_le_bytes());
                        zc[o..o + 2].copy_from_slice(&zv[g].to_le_bytes());
                    }
                }
            });
        }
    });
    Ok([codes, s, zs])
}

/// FN61d: a `trunk8.json`'s uploaded bytes -- every record, or (`head` false, `--fn-trunk w8`)
/// every record but the head's.
fn json_bytes(v: &serde_json::Value, head: bool) -> anyhow::Result<u64> {
    let all = v["bytes"].as_u64().context("bytes")?;
    if head {
        return Ok(all);
    }
    let mut head_bytes = 0u64;
    for r in v["records"].as_array().context("records")? {
        if r["class"].as_str() == Some(HEAD_CLASS) {
            head_bytes += r["storage"]["length"].as_u64().context("length")?;
        }
    }
    all.checked_sub(head_bytes)
        .context("head bytes past the total")
}

/// Whether a trunk record of `[n, k]` takes the W8 tile.
#[must_use]
pub fn fits(n: usize, k: usize) -> bool {
    n % BLK == 0 && k % BLK == 0
}

/// The record names of a source record (weight, block scales).
#[must_use]
pub fn names(src: &str) -> [String; 2] {
    [format!("{src}.w8.w"), format!("{src}.w8.ws")]
}

fn bf16(b: &[u8], i: usize) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])) << 16)
}

/// Quantize a bf16 `[n, k]` matrix: (row-major e4m3 `[n, k]`, fp32 scales `[n/128, k/128]`).
pub fn quantize_rows(w: &[u8], n: usize, k: usize) -> anyhow::Result<(Vec<u8>, Vec<f32>)> {
    ensure!(w.len() == 2 * n * k, "bf16 [{n}, {k}]: {} B", w.len());
    ensure!(fits(n, k), "W8 needs N % 128, K % 128 (got {n} x {k})");
    let (nb, kb) = (n / BLK, k / BLK);
    let mut q = vec![0u8; n * k];
    let mut sc = vec![0f32; nb * kb];
    for bn in 0..nb {
        for bk in 0..kb {
            let mut amax = 0f32;
            for r in bn * BLK..(bn + 1) * BLK {
                for c in bk * BLK..(bk + 1) * BLK {
                    let v = bf16(w, r * k + c);
                    ensure!(v.is_finite(), "row {r}: non-finite weight");
                    amax = amax.max(v.abs());
                }
            }
            let s = amax / 448.0;
            sc[bn * kb + bk] = s;
            if s == 0.0 {
                continue;
            }
            for r in bn * BLK..(bn + 1) * BLK {
                for c in bk * BLK..(bk + 1) * BLK {
                    let x = bf16(w, r * k + c) / s;
                    // x / s may land a hair above 448 (fp32 division): RNE takes it to 448
                    ensure!(x.abs() < 464.0, "row {r} col {c}: {x} out of e4m3 range");
                    q[r * k + c] = saber_cpux::e4m3_rne(x);
                }
            }
        }
    }
    Ok((q, sc))
}

/// The pack's two members of a bf16 `[n, k]` record: (preshuffled e4m3, fp32 scale bytes).
pub fn quantize(w: &[u8], n: usize, k: usize) -> anyhow::Result<[Vec<u8>; 2]> {
    let (q, sc) = quantize_rows(w, n, k)?;
    Ok([
        super::mtp8::preshuffle(&q, n, k),
        sc.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ])
}

/// Build the variant `out` from the pack `src`: link every source file, then write the W8 members
/// per rank. Refuses an existing `out/rank*/trunk8.*`. Returns (records, bytes) per rank.
pub fn build(src: &Path, out: &Path) -> anyhow::Result<[(usize, u64); 2]> {
    let (src, out) = link_variant(src, out)?;
    let recs = PackRecs::open(&src, &ckpt_dir(&src)?)?;
    let mut tot = [(0usize, 0u64); 2];
    for (rank, dir) in ["rank0", "rank1"].iter().enumerate() {
        let (pf, jf) = (
            out.join(dir).join(format!("{STEM}.pack")),
            out.join(dir).join(format!("{STEM}.json")),
        );
        ensure!(
            !pf.exists() && !jf.exists(),
            "{}: the W8 trunk file exists (never rewritten in place)",
            pf.display()
        );
        let tmp = out.join(dir).join(format!("{STEM}.pack.tmp"));
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        let mut outr = Vec::new();
        let mut kept = Vec::new();
        let mut off = 0u64;
        for r in source_records(&src, dir)? {
            let name = r["name"].as_str().context("name")?;
            let Some(fam) = trunk_family(name) else {
                continue;
            };
            ensure!(r["dtype"] == "BF16", "{name}: not a bf16 record");
            let shape: Vec<usize> = r["shape"]
                .as_array()
                .context("shape")?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            ensure!(shape.len() == 2, "{name}: shape {shape:?}");
            let (n, k) = (shape[0], shape[1]);
            if !fits(n, k) {
                kept.push(name.to_string());
                continue;
            }
            let w = recs.raw(rank as u32, name)?;
            let parts = quantize(&w, n, k).with_context(|| name.to_string())?;
            let shapes = [json!([n, k]), json!([n / BLK, k / BLK])];
            for ((rn, b), sh) in names(name).iter().zip(parts.iter()).zip(shapes) {
                outr.push(
                    json!({"name": rn, "class": format!("trunk8.{fam}"), "source": name,
                    "shape": sh, "storage": {"pack": format!("{STEM}.pack"), "data_off": off,
                    "length": b.len(), "sha256": sha256_hex(b)}}),
                );
                f.write_all(b)?;
                let pad = (256 - b.len() as u64 % 256) % 256;
                f.write_all(&vec![0u8; pad as usize])?;
                off += b.len() as u64 + pad;
            }
        }
        f.into_inner().context("flush")?.sync_all()?;
        std::fs::rename(&tmp, &pf)?;
        let doc = json!({"records": outr, "source_pack": src, "bytes": off, "kept_bf16": kept,
            "layout": "fp8 e4m3 [N, K] preshuffled (mtp8::preshuffle) + fp32 scales [N/128, K/128], \
                       s = amax/448 per 128x128 block, RNE (FN61b)"});
        std::fs::write(&jf, serde_json::to_vec_pretty(&doc)?)?;
        tot[rank] = (outr.len(), off);
        println!(
            "flashnext-pack {dir}: {} W8 records, {:.3} GiB -> {} ({} trunk records kept bf16)",
            outr.len(),
            off as f64 / (1u64 << 30) as f64,
            pf.display(),
            kept.len()
        );
    }
    Ok(tot)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FN61d: `--fn-trunk w8` budgets every record but the head's; `w8-head` all of them; the
    /// NEGATIVE CONTROL (a head record under another class) is not subtracted.
    #[test]
    fn fn61d_w8_budgets_the_trunk_without_the_head() {
        let rec = |class: &str, len: u64| json!({"class": class, "storage": {"length": len}});
        let v = json!({"bytes": 1000u64, "records": [rec("trunk8.gdn", 600), rec("trunk8.qsa", 300),
            rec(HEAD_CLASS, 90), rec(HEAD_CLASS, 10)]});
        assert_eq!(json_bytes(&v, true).unwrap(), 1000);
        assert_eq!(json_bytes(&v, false).unwrap(), 900);
        let ctl = json!({"bytes": 1000u64, "records": [rec("trunk8.gdn", 600), rec("trunk8.qsa", 300),
            rec("trunk8.lm_head", 100)]});
        assert_ne!(
            json_bytes(&ctl, false).unwrap(),
            900,
            "control: the class is the law"
        );
        let bad = json!({"bytes": 50u64, "records": [rec(HEAD_CLASS, 90)]});
        assert!(json_bytes(&bad, false).is_err());
    }

    /// FN61e: the dropped set is the W8-replaced records only, by name; in_proj_ba, the indexer,
    /// the MTP layer, conv and the head stay (NEGATIVE CONTROL: a set missing one layer refuses).
    #[test]
    fn fn61e_the_dropped_bf16_records_are_the_w8_replaced_ones() {
        let p = "model.language_model.layers.";
        for (n, d) in [
            ("layer.0.gdn.in_proj_qkvz.w", true),
            ("layer.0.gdn.in_proj_ba.w", false),
            (
                "model.language_model.layers.0.linear_attn.out_proj.weight",
                true,
            ),
            (
                "model.language_model.layers.0.linear_attn.conv1d.weight",
                false,
            ),
            (
                "model.language_model.layers.3.self_attn.q_proj.weight",
                true,
            ),
            (
                "model.language_model.layers.3.self_attn.k_proj.weight",
                true,
            ),
            (
                "model.language_model.layers.3.self_attn.v_proj.weight",
                true,
            ),
            (
                "model.language_model.layers.3.self_attn.o_proj.weight",
                true,
            ),
            (
                "model.language_model.layers.3.self_attn.qkv_idx.weight",
                false,
            ),
            ("mtp.layers.0.self_attn.o_proj.weight", false),
            ("lm_head.weight", false),
        ] {
            assert_eq!(dropped(n), d, "{n}");
        }
        let mut recs = Vec::new();
        for l in 0..48 {
            if l % 4 == 3 {
                for (s, c) in [
                    ("q", "attn.qo"),
                    ("k", "attn.kv"),
                    ("v", "attn.kv"),
                    ("o", "attn.qo"),
                ] {
                    recs.push(json!({"name": format!("{p}{l}.self_attn.{s}_proj.weight"),
                        "class": c, "bytes": 10}));
                }
            } else {
                recs.push(
                    json!({"name": format!("layer.{l}.gdn.in_proj_qkvz.w"), "class": "gdn",
                    "bytes": 7}),
                );
                recs.push(
                    json!({"name": format!("layer.{l}.gdn.in_proj_ba.w"), "class": "gdn",
                    "bytes": 1000}),
                );
                recs.push(
                    json!({"name": format!("{p}{l}.linear_attn.out_proj.weight"),
                    "class": "gdn", "bytes": 3}),
                );
            }
        }
        let by = dropped_by_class(&json!({ "records": recs.clone() })).unwrap();
        assert_eq!(
            by,
            vec![
                ("attn.kv".to_string(), 240),
                ("attn.qo".to_string(), 240),
                ("gdn".to_string(), 360)
            ]
        );
        recs.pop();
        assert!(dropped_by_class(&json!({ "records": recs })).is_err());
    }

    /// FN61g: the host int2 pack is `quant_row` row for row in the card layout, across the thread
    /// chunk boundaries (n not a multiple of the chunk); NEGATIVE CONTROL: one moved weight moves
    /// its own row's pack and no other row's. The head's dense record is found by name.
    #[test]
    fn fn61g_the_host_int2_pack_is_the_row_law_and_the_bf16_head_is_named() {
        use saber_kernels::fn33::{quant_row, K, NG, ROW_BYTES};
        let n = 37usize;
        let mut st = 0x61f0_0d5eu64;
        let mut w: Vec<u8> = (0..n * K)
            .flat_map(|_| {
                st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let v = ((st >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2;
                ((v.to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect();
        let [c, s, z] = int2_pack_host(&w).unwrap();
        assert_eq!(
            (c.len(), s.len(), z.len()),
            (n * ROW_BYTES, n * NG * 2, n * NG * 2)
        );
        let row = |w: &[u8], i: usize| -> Vec<u16> {
            w[i * 2 * K..(i + 1) * 2 * K]
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect()
        };
        let le = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        for i in 0..n {
            let (cr, sr, zr) = quant_row(&row(&w, i));
            assert_eq!(&c[i * ROW_BYTES..(i + 1) * ROW_BYTES], &cr[..], "row {i}");
            assert_eq!(&s[i * NG * 2..(i + 1) * NG * 2], &le(&sr)[..], "row {i}");
            assert_eq!(&z[i * NG * 2..(i + 1) * NG * 2], &le(&zr)[..], "row {i}");
        }
        // control: row 20's largest weight doubled moves row 20 only
        let o = 20 * 2 * K;
        let v = f32::from_bits(u32::from(u16::from_le_bytes([w[o], w[o + 1]])) << 16);
        let b = (((v.abs() * 8.0 + 1.0).to_bits() >> 16) as u16).to_le_bytes();
        w[o..o + 2].copy_from_slice(&b);
        let [c2, s2, _] = int2_pack_host(&w).unwrap();
        assert_ne!(
            (&c2[20 * ROW_BYTES..21 * ROW_BYTES], &s2[40 * NG..42 * NG]),
            (&c[20 * ROW_BYTES..21 * ROW_BYTES], &s[40 * NG..42 * NG])
        );
        assert_eq!(&c2[..20 * ROW_BYTES], &c[..20 * ROW_BYTES]);
        assert_eq!(&c2[21 * ROW_BYTES..], &c[21 * ROW_BYTES..]);
        assert!(int2_pack_host(&w[..2 * K - 2]).is_err());
        // the bf16 head record by name (one), its class and bytes
        let d = json!({"records": [{"name": "lm_head.weight", "class": "lm_head", "bytes": 9},
            {"name": "mtp.lm_head.weight", "class": "x", "bytes": 1}]});
        assert_eq!(head_bf16_of(&d).unwrap(), ("lm_head".to_string(), 9));
        assert!(head_bf16_of(&json!({"records": []})).is_err());
    }

    fn to_bf16(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    #[test]
    fn the_block_scale_is_amax_over_448_and_grid_values_are_exact() {
        // 256 x 256 = 2 x 2 blocks; block (bn, bk) holds e4m3 grid values times 2^(bn - bk), with
        // one element = 448 so amax / 448 = 2^(bn - bk) exactly and every value is on the grid
        let (n, k) = (256usize, 256usize);
        let grid = [0.5f32, -1.0, 1.5, -3.0, 6.0, 0.0, 13.0, -416.0];
        let v: Vec<f32> = (0..n * k)
            .map(|i| {
                let (r, c) = (i / k, i % k);
                let s = 2f32.powi((r / BLK) as i32 - (c / BLK) as i32);
                let base = if r % BLK == 0 && c % BLK == 0 {
                    448.0
                } else {
                    grid[(r + 3 * c) % 8]
                };
                base * s
            })
            .collect();
        let (q, sc) = quantize_rows(&to_bf16(&v), n, k).unwrap();
        assert_eq!(sc, vec![1.0, 0.5, 2.0, 1.0]);
        for (i, &x) in v.iter().enumerate() {
            let (r, c) = (i / k, i % k);
            let s = sc[(r / BLK) * 2 + c / BLK];
            assert_eq!(saber_cpux::e4m3_value(q[i]) as f32 * s, x, "r {r} c {c}");
        }
        // negative control: a per-row scale is NOT this layout's (row 1 of block 0 has no 448)
        let row_amax = v[k..k + BLK].iter().fold(0f32, |a, x| a.max(x.abs()));
        assert_ne!(row_amax / 448.0, sc[0]);
    }

    #[test]
    fn rounding_is_nearest_even_and_the_layout_is_the_preshuffle() {
        // one block; amax 448 (scale 1); 1.0625 sits between 1.0 and 1.125 -> even (1.0)
        let mut v = vec![0f32; BLK * BLK];
        v[0] = 448.0;
        v[1] = 1.0625;
        v[2] = 1.1875; // between 1.125 (odd m=1) and 1.25 (even m=2) -> 1.25
        let [w, s] = quantize(&to_bf16(&v), BLK, BLK).unwrap();
        assert_eq!(s, 1f32.to_le_bytes());
        let (q, _) = quantize_rows(&to_bf16(&v), BLK, BLK).unwrap();
        assert_eq!(saber_cpux::e4m3_value(q[1]), 1.0);
        assert_eq!(saber_cpux::e4m3_value(q[2]), 1.25);
        assert_eq!(w, crate::flashnext::mtp8::preshuffle(&q, BLK, BLK));
        // negative control: row 1's bytes are not at row-major offset 128 in the preshuffle
        let mut v2 = v.clone();
        v2[BLK] = 2.0;
        let (q2, _) = quantize_rows(&to_bf16(&v2), BLK, BLK).unwrap();
        let w2 = crate::flashnext::mtp8::preshuffle(&q2, BLK, BLK);
        assert_eq!(w2[BLK], w[BLK]);
        assert_eq!(w2[16], saber_cpux::e4m3_rne(2.0));
    }

    #[test]
    fn hc_shapes_stay_bf16_and_the_rest_fit() {
        for (n, k, f) in [
            (8192, 2560, true),
            (2560, 3072, true),
            (6144, 2560, true),
            (256, 2560, true),
            (124_160, 2560, true),
            (320, 10240, false),
            (10240, 320, false),
            (48, 2560, false),
        ] {
            assert_eq!(fits(n, k), f, "{n} x {k}");
        }
        assert!(quantize_rows(&to_bf16(&vec![1.0; 320 * 128]), 320, 128).is_err());
    }
}
