//! FN61: the Flash-Next DENSE TRUNK at MXFP4 W4 (the 27B's / the expert tier's format) as a pack
//! VARIANT beside the served pack (`saber flashnext-pack --trunk w4 --out-pack DIR`): the variant
//! directory links every file of the source pack (never rewritten) and ADDS `trunk4.{pack,json}` to
//! `rank0/`, `rank1/` and `shared/`. Per bf16 source record `<name>` of a trunk family three records
//! in the shared expert's trio form: `<name>.w4.w` (e2m1 nibbles `[N, K/2]`, low nibble = even k,
//! WMMA fragment order = `flashnext_mx.permute_w`), `<name>.w4.ws` (e8m0 `[K/32, N]`) and
//! `<name>.w4.wref` (e8m0 `[N]` = the row max of ws; the card folds `d = wref - ws` into e4m3).
//!
//! The quantizer (the OCP MX rule, Quark's): per 32 along K, `e = floor(log2(max(amax, 2^-126))) - 2`
//! clamped to [-127, 127], each element `x / 2^e` rounded to the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4,
//! 6} half-to-even by code, saturating at 6. `Rule::Mse` takes `e + 1` for a block whose squared
//! error is strictly smaller (a builder choice; the card's law is the same). Families: GDN
//! in_proj_qkvz + out_proj, QSA q/k/v/o, the lm_head (rank records); the HC mixers' down / up
//! (shared). Router, shared gate, norms, GDN ba / conv / state, indexer, PLE, HC inject (N = 4) and
//! every MTP record stay bf16. The host twin of the values is `tools/capture/fn61/trunk_quant.py`
//! (`check_trunk4.py` compares the pack's bytes with it).
//!
//! FN67: the CALIBRATED variant (`--trunk w4 --w4-rule gptq --calib DIR` -> `-trunk4c`): the codes
//! and block exponents are NOT computed here but read from the calibration directory written by
//! `the internal reference harness calib` (GPTQ on this same fold grid; `tools/capture/fn67/gptq.py`):
//! one u16 per element (low byte = the signed code, high byte = the element's block e8m0) at the
//! tensor's OWN checkpoint byte offset in a file named like its shard, so each pack record's own
//! checkpoint extents (TP shards, the merged in_proj_qkvz) read it exactly. Only the calibrated
//! families get W4 records (HC is never calibrated: it stays as the source pack has it); build it
//! from the `-trunk8` variant to keep W8 records for a family left at W8 (the MIXED form).

use std::collections::HashSet;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context};
use saber_kernels::manifest::sha256_hex;
use serde_json::{json, Value};

use super::store::PackRecs;

/// The additional pack file's stem (per rank directory and `shared/`).
pub const STEM: &str = "trunk4";

/// The block-exponent rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// OCP / Quark: floor(log2 amax) - 2.
    Ocp,
    /// OCP's exponent or one above, whichever has the smaller squared error.
    Mse,
    /// FN67: the calibrated codes + exponents of a calibration directory (`--calib`).
    Gptq,
}

impl std::str::FromStr for Rule {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<Rule> {
        match s {
            "ocp" => Ok(Rule::Ocp),
            "mse" => Ok(Rule::Mse),
            "gptq" => Ok(Rule::Gptq),
            _ => anyhow::bail!("--w4-rule ocp|mse|gptq, got {s}"),
        }
    }
}

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const MIDS: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];

/// |x| (already divided by the block scale) -> the e2m1 magnitude code, half-to-even, saturating.
#[must_use]
pub fn e2m1_code(a: f32) -> u8 {
    let mut i = 0usize;
    while i < 7 && a > MIDS[i] {
        i += 1;
    }
    // a == MIDS[i]: between codes i and i + 1 -> the even one
    if i < 7 && a == MIDS[i] && i % 2 == 1 {
        i += 1;
    }
    i as u8
}

fn bf16(b: &[u8], i: usize) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])) << 16)
}

/// floor(log2(max(x, 2^-126))) for a finite x >= 0.
fn flog2(x: f32) -> i32 {
    let x = x.max(f32::MIN_POSITIVE);
    ((x.to_bits() >> 23) & 0xFF) as i32 - 127
}

/// One block's codes (sign in bit 3) at exponent `e` and its squared error.
fn block_at(x: &[f32], e: i32) -> ([u8; 32], f64) {
    let s = 2f32.powi(e);
    let mut c = [0u8; 32];
    let mut err = 0f64;
    for (j, &v) in x.iter().enumerate() {
        let code = e2m1_code(v.abs() / s);
        let q = E2M1[code as usize] * s;
        let qv = if v < 0.0 { -q } else { q };
        let dv = f64::from(qv) - f64::from(v);
        err += dv * dv;
        c[j] = code | if v < 0.0 && code != 0 { 8 } else { 0 };
    }
    (c, err)
}

/// Quantize a bf16 `[n, k]` matrix: (WPERM nibbles `[n, k/2]`, e8m0 `[k/32, n]`, wref `[n]`).
pub fn quantize(w: &[u8], n: usize, k: usize, rule: Rule) -> anyhow::Result<[Vec<u8>; 3]> {
    ensure!(
        rule != Rule::Gptq,
        "the gptq rule reads its codes (from_calib), it does not quantize"
    );
    ensure!(w.len() == 2 * n * k, "bf16 [{n}, {k}]: {} B", w.len());
    ensure!(
        n % 16 == 0 && k % 32 == 0,
        "MXFP4 W4 needs N % 16, K % 32 (got {n} x {k})"
    );
    let kb = k / 32;
    let mut packed = vec![0u8; n * k / 2];
    let mut ws = vec![0u8; kb * n];
    let mut wref = vec![0u8; n];
    let mut x = [0f32; 32];
    for r in 0..n {
        let mut mx = 0u8;
        for b in 0..kb {
            let mut amax = 0f32;
            for (j, v) in x.iter_mut().enumerate() {
                *v = bf16(w, r * k + b * 32 + j);
                ensure!(v.is_finite(), "row {r}: non-finite weight");
                amax = amax.max(v.abs());
            }
            let mut e = if amax == 0.0 {
                -127
            } else {
                (flog2(amax) - 2).clamp(-127, 127)
            };
            let (mut c, err) = block_at(&x, e);
            if rule == Rule::Mse && e < 127 {
                let (c1, err1) = block_at(&x, e + 1);
                if err1 < err {
                    e += 1;
                    c = c1;
                }
            }
            let byte = (e + 127) as u8;
            ws[b * n + r] = byte;
            mx = mx.max(byte);
            for j in 0..16 {
                packed[r * k / 2 + b * 16 + j] = c[2 * j] | (c[2 * j + 1] << 4);
            }
        }
        wref[r] = mx;
    }
    Ok([permute_w(&packed, n, k), ws, wref])
}

/// `[N, K/2]` -> WMMA fragment order (`flashnext_mx.permute_w`: reshape(N/16, 16, K/16, 2, 4)
/// .transpose(0, 2, 3, 1, 4)).
#[must_use]
pub fn permute_w(p: &[u8], n: usize, k: usize) -> Vec<u8> {
    let ks = k / 16;
    let mut o = vec![0u8; p.len()];
    let mut i = 0;
    for nt in 0..n / 16 {
        for s in 0..ks {
            for h in 0..2 {
                for r in 0..16 {
                    for j in 0..4 {
                        o[i] = p[(nt * 16 + r) * (k / 2) + s * 8 + h * 4 + j];
                        i += 1;
                    }
                }
            }
        }
    }
    o
}

/// The trunk family of a pack record (`gdn` / `qsa` / `hc` / `head`), or `None` = stays bf16.
#[must_use]
pub fn trunk_family(name: &str) -> Option<&'static str> {
    if name.starts_with("mtp") {
        return None;
    }
    if name.ends_with(".gdn.in_proj_qkvz.w") || name.ends_with("linear_attn.out_proj.weight") {
        return Some("gdn");
    }
    if name.contains(".self_attn.")
        && ["q_proj", "k_proj", "v_proj", "o_proj"]
            .iter()
            .any(|p| name.ends_with(&format!(".{p}.weight")))
    {
        return Some("qsa");
    }
    if name.ends_with("input_mix_weight_down.weight")
        || name.ends_with("input_mix_weight_up.weight")
    {
        return Some("hc");
    }
    if name == "lm_head.weight" {
        return Some("head");
    }
    None
}

/// The record names of a source record.
#[must_use]
pub fn names(src: &str) -> [String; 3] {
    [
        format!("{src}.w4.w"),
        format!("{src}.w4.ws"),
        format!("{src}.w4.wref"),
    ]
}

pub(super) fn source_records(pack: &Path, dir: &str) -> anyhow::Result<Vec<Value>> {
    let p = pack.join(dir).join("dense.json");
    let v: Value =
        serde_json::from_str(&std::fs::read_to_string(&p).context(p.display().to_string())?)?;
    Ok(v["records"].as_array().context("records")?.clone())
}

/// Link every file of the pack `src` into the variant directory `out` (created; `rank0/`, `rank1/`
/// and `shared/` file by file, everything else as one link). Returns both canonical paths. FN61b's
/// `-trunk8` variant shares it.
pub fn link_variant(
    src: &Path,
    out: &Path,
) -> anyhow::Result<(std::path::PathBuf, std::path::PathBuf)> {
    let src = src.canonicalize()?;
    std::fs::create_dir_all(out)?;
    let out = out.canonicalize()?;
    ensure!(
        src != out,
        "the variant is a new directory (the served pack is never rewritten)"
    );
    for e in std::fs::read_dir(&src)? {
        let e = e?;
        let name = e.file_name();
        let (s, d) = (e.path(), out.join(&name));
        let sub = matches!(name.to_str(), Some("rank0" | "rank1" | "shared"));
        if sub {
            std::fs::create_dir_all(&d)?;
            for f in std::fs::read_dir(&s)? {
                let f = f?;
                let t = d.join(f.file_name());
                if !t.exists() {
                    std::os::unix::fs::symlink(f.path(), &t)?;
                }
            }
        } else if !d.exists() {
            std::os::unix::fs::symlink(&s, &d)?;
        }
    }
    Ok((src, out))
}

/// Build the variant `out` from the pack `src`: link every source file, then write the W4 members.
/// Refuses an existing `out/*/trunk4.*`. Returns (records, bytes) per [rank0, rank1, shared].
pub fn build(src: &Path, out: &Path, rule: Rule) -> anyhow::Result<[(usize, u64); 3]> {
    build_with(
        src,
        out,
        rule,
        json!(null),
        |recs, rank, name, _fam, _rec, n, k| {
            let w = recs.raw(rank, name)?;
            quantize(&w, n, k, rule).map(Some)
        },
    )
}

/// The trio of a record from its calibrated u16 elements (`[n, k]` row-major, low byte = signed
/// code, high byte = the block's e8m0): the same layout `quantize` writes. Every 32-block of a row
/// must carry one exponent (a record whose K cut is not 32-aligned is refused).
pub fn from_calib(u: &[u8], n: usize, k: usize) -> anyhow::Result<[Vec<u8>; 3]> {
    ensure!(u.len() == 2 * n * k, "u16 [{n}, {k}]: {} B", u.len());
    ensure!(
        n % 16 == 0 && k % 32 == 0,
        "MXFP4 W4 needs N % 16, K % 32 (got {n} x {k})"
    );
    let kb = k / 32;
    let mut packed = vec![0u8; n * k / 2];
    let mut ws = vec![0u8; kb * n];
    let mut wref = vec![0u8; n];
    for r in 0..n {
        let mut mx = 0u8;
        for b in 0..kb {
            let at = |j: usize| 2 * (r * k + b * 32 + j);
            let e = u[at(0) + 1];
            ensure!(e != 255, "row {r} block {b}: e8m0 255 (NaN)");
            for j in 0..32 {
                ensure!(
                    u[at(j) + 1] == e,
                    "row {r} block {b}: the exponent changes inside a 32-block"
                );
                ensure!(
                    u[at(j)] < 16,
                    "row {r} col {}: code {} > 15",
                    b * 32 + j,
                    u[at(j)]
                );
            }
            ws[b * n + r] = e;
            mx = mx.max(e);
            for j in 0..16 {
                packed[r * k / 2 + b * 16 + j] = u[at(2 * j)] | (u[at(2 * j + 1)] << 4);
            }
        }
        wref[r] = mx;
    }
    Ok([permute_w(&packed, n, k), ws, wref])
}

/// FN67: a calibration directory (`calib.json` + the sparse u16 shard files).
pub struct Calib {
    dir: PathBuf,
    /// (file, offset, length, sha256, name)
    tensors: Vec<(String, u64, u64, String, String)>,
    verified: std::cell::RefCell<HashSet<usize>>,
    /// sha256 of calib.json (the provenance the variant records)
    pub doc_sha256: String,
}

impl Calib {
    pub fn open(dir: &Path) -> anyhow::Result<Calib> {
        let p = dir.join("calib.json");
        let raw = std::fs::read(&p).with_context(|| p.display().to_string())?;
        let v: Value = serde_json::from_slice(&raw)?;
        ensure!(
            v["finished"].is_string(),
            "{}: the calibration did not finish",
            p.display()
        );
        let mut tensors = Vec::new();
        for (name, t) in v["tensors"].as_object().context("tensors")? {
            tensors.push((
                t["file"].as_str().context("file")?.to_string(),
                t["offset"].as_u64().context("offset")?,
                t["length"].as_u64().context("length")?,
                t["sha256"].as_str().context("sha256")?.to_string(),
                name.clone(),
            ));
        }
        Ok(Calib {
            dir: dir.to_path_buf(),
            tensors,
            verified: std::cell::RefCell::new(HashSet::new()),
            doc_sha256: sha256_hex(&raw),
        })
    }

    /// The calibrated tensor holding `[off, off + len)` of `file`, its bytes checked once.
    fn covering(&self, file: &str, off: u64, len: u64) -> anyhow::Result<Option<usize>> {
        let Some(i) = self
            .tensors
            .iter()
            .position(|t| t.0 == file && off >= t.1 && off + len <= t.1 + t.2)
        else {
            return Ok(None);
        };
        if !self.verified.borrow().contains(&i) {
            let t = &self.tensors[i];
            let f = std::fs::File::open(self.dir.join(&t.0))?;
            let mut b = vec![0u8; t.2 as usize];
            f.read_exact_at(&mut b, t.1)?;
            ensure!(
                sha256_hex(&b) == t.3,
                "{}: calibrated bytes != calib.json's sha256",
                t.4
            );
            self.verified.borrow_mut().insert(i);
        }
        Ok(Some(i))
    }

    /// A pack record's calibrated u16 elements through its own checkpoint extents; `None` when
    /// no extent is calibrated, an error when only some are.
    pub fn record_u16(&self, rec: &Value) -> anyhow::Result<Option<Vec<u8>>> {
        let ext = rec["storage"]["checkpoint"]
            .as_array()
            .context("not a checkpoint-identity record")?;
        let mut out = Vec::new();
        let mut hit = 0usize;
        let mut total = 0usize;
        for e in ext {
            let file = e["file"].as_str().context("file")?;
            let (off, len, stride, count) = (
                e["offset"].as_u64().context("offset")?,
                e["length"].as_u64().context("length")?,
                e["stride"].as_u64().unwrap_or(0),
                e["count"].as_u64().unwrap_or(1),
            );
            total += 1;
            let span = if count > 1 {
                stride * (count - 1) + len
            } else {
                len
            };
            if self.covering(file, off, span)?.is_none() {
                continue;
            }
            hit += 1;
            let f = std::fs::File::open(self.dir.join(file))?;
            for i in 0..count {
                let mut b = vec![0u8; len as usize];
                f.read_exact_at(&mut b, off + i * stride)?;
                out.extend_from_slice(&b);
            }
        }
        ensure!(
            hit == 0 || hit == total,
            "{}: {hit} of {total} checkpoint extents calibrated",
            rec["name"]
        );
        Ok((hit > 0).then_some(out))
    }
}

/// FN67: the calibrated variant (`-trunk4c`): W4 records for every record of a calibrated tensor,
/// read from `calib` (HC is never calibrated; a family not in the calibration is left as `src` has
/// it -- build from the `-trunk8` variant for the MIXED W4/W8 form).
pub fn build_calibrated(src: &Path, out: &Path, calib: &Path) -> anyhow::Result<[(usize, u64); 3]> {
    let cal = Calib::open(calib)?;
    let extra = json!({"calib_dir": calib, "calib_json_sha256": cal.doc_sha256});
    build_with(
        src,
        out,
        Rule::Gptq,
        extra,
        |_recs, _rank, name, fam, rec, n, k| {
            if fam == "hc" {
                return Ok(None);
            }
            match cal.record_u16(rec).with_context(|| name.to_string())? {
                Some(u) => from_calib(&u, n, k)
                    .map(Some)
                    .with_context(|| name.to_string()),
                None => Ok(None),
            }
        },
    )
}

fn build_with(
    src: &Path,
    out: &Path,
    rule: Rule,
    extra: Value,
    mut make: impl FnMut(
        &PackRecs,
        u32,
        &str,
        &str,
        &Value,
        usize,
        usize,
    ) -> anyhow::Result<Option<[Vec<u8>; 3]>>,
) -> anyhow::Result<[(usize, u64); 3]> {
    let (src, out) = link_variant(src, out)?;
    let recs = PackRecs::open(&src, &ckpt_dir(&src)?)?;
    let mut tot = [(0usize, 0u64); 3];
    for (slot, (dir, rank)) in [("rank0", 0u32), ("rank1", 1), ("shared", 0)]
        .iter()
        .enumerate()
    {
        let (pf, jf) = (
            out.join(dir).join(format!("{STEM}.pack")),
            out.join(dir).join(format!("{STEM}.json")),
        );
        ensure!(
            !pf.exists() && !jf.exists(),
            "{}: the W4 trunk file exists (never rewritten in place)",
            pf.display()
        );
        let tmp = out.join(dir).join(format!("{STEM}.pack.tmp"));
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        let mut outr = Vec::new();
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
            let Some(parts) = make(&recs, *rank, name, fam, &r, n, k)? else {
                continue;
            };
            for (rn, b) in names(name).iter().zip(parts.iter()) {
                outr.push(
                    json!({"name": rn, "class": format!("trunk4.{fam}"), "source": name,
                    "shape": [n, k], "storage": {"pack": format!("{STEM}.pack"), "data_off": off,
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
        let doc = json!({"records": outr, "source_pack": src, "rule": format!("{rule:?}"),
            "bytes": off, "layout": "mxfp4 trio: wperm nibbles + e8m0 [K/32, N] + wref [N] (FN61)",
            "calibration": extra});
        std::fs::write(&jf, serde_json::to_vec_pretty(&doc)?)?;
        tot[slot] = (outr.len(), off);
        println!(
            "flashnext-pack {dir}: {} W4 records, {:.3} GiB -> {}",
            outr.len(),
            off as f64 / (1u64 << 30) as f64,
            pf.display()
        );
    }
    Ok(tot)
}

pub(super) fn ckpt_dir(pack: &Path) -> anyhow::Result<std::path::PathBuf> {
    let m: Value = serde_json::from_str(&std::fs::read_to_string(pack.join("manifest.json"))?)?;
    Ok(m["checkpoint"]["dir"]
        .as_str()
        .context("checkpoint.dir")?
        .into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_bf16(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    #[test]
    fn e2m1_rounds_half_to_even_and_saturates() {
        let a = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, 0.0, 6.0, 0.26];
        let c: Vec<u8> = a.iter().map(|&x| e2m1_code(x)).collect();
        assert_eq!(c, [0, 2, 2, 4, 4, 6, 6, 7, 0, 7, 1]);
    }

    #[test]
    fn the_trio_dequantizes_to_the_ocp_values_and_the_layout_is_permute_w() {
        // a 16 x 64 matrix: row r, block b scaled 2^(r - b) so wref and ws differ
        let (n, k) = (16usize, 64usize);
        let v: Vec<f32> = (0..n * k)
            .map(|i| {
                let (r, c) = (i / k, i % k);
                let base = [0.5f32, -1.0, 1.5, 3.0, -6.0, 2.0, 0.0, 4.0][(c + r) % 8];
                base * 2f32.powi(r as i32 - (c / 32) as i32)
            })
            .collect();
        let [wp, ws, wref] = quantize(&to_bf16(&v), n, k, Rule::Ocp).unwrap();
        // amax 6 * 2^(r-b) -> e = r - b + 2 - 2 = r - b; every element exact on the grid
        for r in 0..n {
            for b in 0..2 {
                assert_eq!(i32::from(ws[b * n + r]) - 127, r as i32 - b as i32);
            }
            assert_eq!(wref[r], ws[r].max(ws[n + r]));
        }
        // undo the permutation (python's unpermute_w) and decode
        let ks = k / 16;
        let mut p = vec![0u8; n * k / 2];
        let mut i = 0;
        for nt in 0..n / 16 {
            for s in 0..ks {
                for h in 0..2 {
                    for r in 0..16 {
                        for j in 0..4 {
                            p[(nt * 16 + r) * (k / 2) + s * 8 + h * 4 + j] = wp[i];
                            i += 1;
                        }
                    }
                }
            }
        }
        for r in 0..n {
            for c in 0..k {
                let byte = p[r * k / 2 + c / 2];
                let code = if c % 2 == 0 { byte & 15 } else { byte >> 4 };
                let mag = E2M1[usize::from(code & 7)];
                let e = i32::from(ws[(c / 32) * n + r]) - 127;
                let val = if code & 8 != 0 { -mag } else { mag } * 2f32.powi(e);
                assert_eq!(val, v[r * k + c], "r {r} c {c}");
            }
        }
        // negative control: the unpermuted bytes are not the pack's layout
        assert_ne!(p, wp);
    }

    #[test]
    fn fn67_calibrated_u16_packs_like_the_quantizer_and_refuses_a_split_block() {
        // the 16 x 64 grid matrix of the trio test: row r, block b at 2^(r - b), every value exact
        let (n, k) = (16usize, 64usize);
        let v: Vec<f32> = (0..n * k)
            .map(|i| {
                let (r, c) = (i / k, i % k);
                let base = [0.5f32, -1.0, 1.5, 3.0, -6.0, 2.0, 0.0, 4.0][(c + r) % 8];
                base * 2f32.powi(r as i32 - (c / 32) as i32)
            })
            .collect();
        let want = quantize(&to_bf16(&v), n, k, Rule::Ocp).unwrap();
        // the calibration store's form: low byte = signed code, high byte = the block's e8m0
        let mut u = Vec::with_capacity(2 * n * k);
        for r in 0..n {
            for c in 0..k {
                let e = r as i32 - (c / 32) as i32;
                let x = v[r * k + c];
                let code = e2m1_code(x.abs() / 2f32.powi(e));
                u.push(code | if x < 0.0 && code != 0 { 8 } else { 0 });
                u.push((e + 127) as u8);
            }
        }
        assert_eq!(from_calib(&u, n, k).unwrap(), want);
        // negative control: one element's exponent differs from its block's
        let mut bad = u.clone();
        bad[2 * 5 + 1] ^= 1;
        assert!(from_calib(&bad, n, k).is_err());
        // and the gptq rule never quantizes
        assert!(quantize(&to_bf16(&v), n, k, Rule::Gptq).is_err());
        assert_eq!("gptq".parse::<Rule>().unwrap(), Rule::Gptq);
    }

    #[test]
    fn mse_takes_the_higher_exponent_only_when_it_is_strictly_better() {
        // 16 rows x one block; row 0's first element x, the other rows 1.0 (any value)
        let mat = |x: f32| {
            let mut m = vec![0f32; 16 * 32];
            m[0] = x;
            for r in 1..16 {
                m[r * 32] = 1.0;
            }
            to_bf16(&m)
        };
        let ws0 = |x: f32, rule| quantize(&mat(x), 16, 32, rule).unwrap()[1][0];
        // 7: OCP e 0 saturates to 6 (err 1); e 1: 3.5 -> the even code 4 -> 8 (err 1): a tie stays
        assert_eq!((ws0(7.0, Rule::Ocp), ws0(7.0, Rule::Mse)), (127, 127));
        // 7.5: e 0 -> 6 (err 1.5); e 1: 3.75 -> 4 -> 8 (err .5): MSE moves up, OCP does not
        assert_eq!((ws0(7.5, Rule::Ocp), ws0(7.5, Rule::Mse)), (127, 128));
    }

    #[test]
    fn the_families_are_the_trunk_and_nothing_else() {
        for (n, f) in [
            ("layer.4.gdn.in_proj_qkvz.w", Some("gdn")),
            ("layer.4.gdn.in_proj_ba.w", None),
            (
                "model.language_model.layers.0.linear_attn.out_proj.weight",
                Some("gdn"),
            ),
            (
                "model.language_model.layers.3.self_attn.q_proj.weight",
                Some("qsa"),
            ),
            (
                "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight",
                None,
            ),
            ("mtp.layers.0.self_attn.q_proj.weight", None),
            (
                "model.language_model.layers.7.attn_hyper_connection.input_mix_weight_up.weight",
                Some("hc"),
            ),
            (
                "model.language_model.layers.7.mlp_hyper_connection.block_inject_weight.weight",
                None,
            ),
            (
                "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
                Some("hc"),
            ),
            (
                "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
                None,
            ),
            ("model.language_model.layers.7.mlp.gate.weight", None),
            ("lm_head.weight", Some("head")),
        ] {
            assert_eq!(trunk_family(n), f, "{n}");
        }
    }
}
