//! FN144: the FP4-Local LOSSLESS pack, loader side (`saber flashnext-pack --source fp4-local`).
//!
//! * `--dry-run`: the plan ([`PackPlan::fp4local`]) checked against the checkpoint's safetensors
//!   headers (every source tensor present with the dtype / shape its record kind reads), the
//!   MXFP8 -> bf16 records proven exact element by element, and the byte accounting (per class,
//!   per rank, against the Quark pack's plan; the expert tier's stride and VRAM slots; the CSF
//!   bytes as stored; the PLE table; the vision tower). Writes `plan.json` + `accounting.json`
//!   into `--report`; writes no pack byte.
//! * `--layers A:B --out-pack DIR`: a SMALL real pack for the P3 kernel fixture -- per rank an
//!   xtier **v2** `NVFP4_E4M3G16` file of layers `[A, B)` (header `layers = B - A`, record `i` =
//!   (layer `A + i / 256`, expert `256 r + i % 256`)), each record built from the checkpoint with
//!   no rounding: `w` = the e2m1 bytes in `permute_w` order, `ws` = the CSF-decoded E4M3 scales
//!   transposed `[K/16, N]`, `gs` = the f32 globals; + `manifest.json`. Re-checked by
//!   `tools/pack/check_flashnext_pack.py DIR` (v2: an independent numpy re-derivation).

use std::collections::BTreeMap;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{bail, ensure, Context};
use saber_mem::xtier::{
    header_block, record_digest, record_layout_nvfp4, table_digest, XtierFile, XtierFormat,
    XtierHeader, XTIER_HEADER,
};
use saber_model::qwen4_exp::nvfp4_csf;
use saber_model::qwen4_exp::pack_plan::{Fp4Local, PackPlan, Placement, SourceKind};
use saber_model::qwen4_exp::st::ShardSet;
use saber_model::qwen4_exp::{ModeloptAlgo, QuantConfig, Qwen4ExpConfig};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::trunk4::permute_w;

/// The Quark checkpoint whose plan the accounting compares against.
pub const QUARK_CKPT: &str = "/mnt/nvme/models/Qwen3.8-Flash-Next-Quark-MXFP4";

fn config(dir: &Path) -> anyhow::Result<Qwen4ExpConfig> {
    let p = dir.join("config.json");
    let t = std::fs::read_to_string(&p).with_context(|| p.display().to_string())?;
    Qwen4ExpConfig::from_config_json(&t).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))
}

/// The dtype a record kind READS from its source tensor (and, for a `ws` / scale, its own).
fn source_dtype(kind: SourceKind, part: Option<&str>) -> &'static str {
    match kind {
        SourceKind::Bf16 | SourceKind::Bf16Merged | SourceKind::F32FromBf16 => "BF16",
        SourceKind::Raw => "I64",
        SourceKind::QuarkMxfp4 => "U8",
        SourceKind::ModeloptFp4Local(Fp4Local::Mxfp8 | Fp4Local::Mxfp8ToBf16) => {
            if part == Some("ws") {
                "U8"
            } else {
                "F8_E4M3"
            }
        }
        SourceKind::ModeloptFp4Local(Fp4Local::F32Copy) => "F32",
        SourceKind::ModeloptFp4Local(Fp4Local::Nvfp4Trio) => match part {
            Some("w") => "U8",
            Some("ws") => "F8_E4M3",
            _ => "F32",
        },
    }
}

fn bf16_exact(x: f32) -> bool {
    let b = x.to_bits();
    b & 0xFFFF == 0
}

fn e8m0(e: u8) -> f32 {
    // UE8M0: 2^(e - 127); 255 = NaN.
    match e {
        255 => f32::NAN,
        0 => 2f32.powi(-127),
        _ => f32::from_bits(u32::from(e) << 23),
    }
}

fn gib(b: u64) -> f64 {
    b as f64 / f64::from(1u32 << 30)
}

/// `--dry-run`: the plan, the header check, the exactness proof and the byte accounting.
pub fn dry_run(ckpt: &Path, report: &Path) -> anyhow::Result<Value> {
    let c = config(ckpt)?;
    let plan = PackPlan::fp4local(&c).map_err(|e| anyhow::anyhow!(e))?;
    let mut st = ShardSet::open(ckpt).map_err(|e| anyhow::anyhow!(e))?;
    // ---- 1. every source tensor: in the index, dtype + shape as the record reads it
    let (mut checked, mut absent, mut bad) = (0u64, 0u64, Vec::<String>::new());
    let mut seen = std::collections::HashSet::new();
    for r in &plan.dense {
        for s in &r.sources {
            let want = source_dtype(r.kind, r.part);
            if !seen.insert((s.tensor.clone(), want)) {
                continue;
            }
            if st.shard_of(&s.tensor).is_none() {
                bad.push(format!("{}: not in the index ({})", s.tensor, r.name));
                continue;
            }
            let Some(info) = st.info(&s.tensor) else {
                absent += 1;
                continue;
            };
            let shape: Vec<u64> = if info.shape.is_empty() {
                vec![1]
            } else {
                info.shape.clone()
            };
            if info.dtype != want || shape != s.shape {
                bad.push(format!(
                    "{}: {} {:?}, the plan reads {want} {:?} ({})",
                    s.tensor, info.dtype, shape, s.shape, r.name
                ));
            }
            checked += 1;
        }
    }
    // The expert tier's sources (CSF planes + e2m1 + globals), and the CSF bytes as stored.
    let (h, inter) = (u64::from(c.hidden_size), u64::from(c.moe.intermediate));
    let mut csf_bytes = 0u64;
    let mut csf_planes = 0u64;
    for l in 0..c.num_layers() {
        for e in 0..c.moe.num_experts {
            for (p, n, k) in [
                ("gate_proj", inter, h),
                ("up_proj", inter, h),
                ("down_proj", h, inter),
            ] {
                let x = format!("model.language_model.layers.{l}.mlp.experts.{e}.{p}");
                for (sfx, dt, shape) in [
                    ("weight", "U8", Some(vec![n, k / 2])),
                    (
                        "weight_scale.nvfp4_csf_fixed",
                        "U8",
                        Some(vec![n / 16, 16 * (1 + k / 32)]),
                    ),
                    ("weight_scale.nvfp4_csf_exceptions", "U32", None),
                    ("weight_scale_2", "F32", Some(vec![])),
                ] {
                    let t = format!("{x}.{sfx}");
                    match st.info(&t) {
                        None if st.shard_of(&t).is_none() => {
                            bad.push(format!("{t}: not in the index"))
                        }
                        None => absent += 1,
                        Some(i) => {
                            if i.dtype != dt || shape.as_ref().is_some_and(|s| *s != i.shape) {
                                bad.push(format!("{t}: {} {:?}", i.dtype, i.shape));
                            }
                            if sfx.starts_with("weight_scale.") {
                                csf_bytes += i.range.1 - i.range.0;
                            }
                            checked += 1;
                        }
                    }
                }
                csf_planes += 1;
            }
        }
    }
    ensure!(
        bad.is_empty(),
        "fp4local dry run: {} source mismatches, first: {}",
        bad.len(),
        bad.iter().take(5).cloned().collect::<Vec<_>>().join("; ")
    );
    // ---- 2. the MXFP8 -> bf16 records are exact (every element of every present source)
    let (mut exact_elems, mut inexact) = (0u64, 0u64);
    let mut exact_seen = std::collections::HashSet::new();
    for r in &plan.dense {
        if r.kind != SourceKind::ModeloptFp4Local(Fp4Local::Mxfp8ToBf16) {
            continue;
        }
        for s in &r.sources {
            if !st.present(&s.tensor) || !exact_seen.insert(s.tensor.clone()) {
                continue;
            }
            let (_, w) = st.read(&s.tensor).map_err(|e| anyhow::anyhow!(e))?;
            let (_, sc) = st
                .read(&format!("{}_scale", s.tensor))
                .map_err(|e| anyhow::anyhow!(e))?;
            let k = s.shape[1] as usize;
            for (j, &b) in w.iter().enumerate() {
                let v = nvfp4_csf::e4m3_to_f32(b) * e8m0(sc[(j / k) * (k / 32) + (j % k) / 32]);
                if bf16_exact(v) && v.is_finite() {
                    exact_elems += 1;
                } else {
                    inexact += 1;
                }
            }
        }
    }
    ensure!(
        inexact == 0,
        "fp4local: {inexact} MXFP8 elements are not exact in bf16"
    );
    // ---- 3. the byte accounting against the Quark plan
    let quark = config(Path::new(QUARK_CKPT))
        .ok()
        .map(|q| PackPlan::ep2(&q));
    let ledger = |p: &PackPlan| -> BTreeMap<(String, String), u64> {
        p.ledger()
            .into_iter()
            .map(|(pl, cl, b)| ((pl, cl.key().to_string()), b))
            .collect()
    };
    let lf = ledger(&plan);
    let lq = quark.as_ref().map(ledger).unwrap_or_default();
    let mut rows = Vec::new();
    let mut keys: Vec<&(String, String)> = lf.keys().chain(lq.keys()).collect();
    keys.sort();
    keys.dedup();
    for k in keys {
        rows.push(json!({"placement": k.0, "class": k.1,
            "fp4_local": lf.get(k).copied().unwrap_or(0), "quark": lq.get(k).copied().unwrap_or(0)}));
    }
    let per_rank = |l: &BTreeMap<(String, String), u64>, r: u32| -> u64 {
        l.iter()
            .filter(|((p, _), _)| *p == format!("rank{r}") || p == "shared")
            .map(|(_, b)| *b)
            .sum()
    };
    // Dense resident bytes of rank `r` (its records + shared), the expert tier and PLE excluded.
    let dense_rank = |p: &PackPlan, r: u32| -> u64 {
        p.dense
            .iter()
            .filter(|d| d.placement == Placement::Rank(r) || d.placement == Placement::Shared)
            .map(|d| d.bytes)
            .sum()
    };
    let written: u64 = plan
        .dense
        .iter()
        .filter(|r| !r.kind.identity())
        .map(|r| r.bytes)
        .sum();
    let t = &plan.experts[0];
    let q_stride = quark.as_ref().map_or(0, |q| q.experts[0].record.stride);
    // FN7's slot law: the VRAM expert budget is fixed bytes; slots scale by the stride.
    let slots_quark = 7_301u64;
    let slots_fp4 = slots_quark * q_stride / t.record.stride.max(1);
    // Vision (not in the plan; FN133 P6 dequantizes it to bf16).
    let QuantConfig::ModelOpt { .. } = &c.quant else {
        bail!("not ModelOpt")
    };
    let mut vision = BTreeMap::<&str, (u64, u64)>::new();
    let names: Vec<String> = st
        .names()
        .filter(|n| n.starts_with("model.visual."))
        .cloned()
        .collect();
    for n in &names {
        let Some(i) = st.info(n) else { continue };
        let elems: u64 = i.shape.iter().product::<u64>().max(1);
        let module = n.rsplit_once('.').map_or("", |x| x.0);
        let algo = c.quant.modelopt_layer(module).map(|l| l.algo);
        let (key, bf16) = match (algo, n.rsplit('.').next()) {
            (Some(ModeloptAlgo::Mxfp8), Some("weight")) => ("mxfp8->bf16 exact", elems * 2),
            (Some(ModeloptAlgo::W4a16Nvfp4), Some("weight")) => {
                ("nvfp4->bf16 one rounding", elems * 4)
            }
            (Some(_), _) => ("scales", 0),
            _ => (
                "bf16/f32 as stored",
                elems * if i.dtype == "F32" { 4 } else { 2 },
            ),
        };
        let e = vision.entry(key).or_default();
        e.0 += i.range.1 - i.range.0;
        e.1 += bf16;
    }
    let ple_rows = plan.ple.rows;
    let acct = json!({
        "source": "fp4_local",
        "checkpoint": ckpt.display().to_string(),
        "headers": {"checked": checked, "absent_shards": absent, "mismatches": 0},
        "mxfp8_bf16_exact": {"elements": exact_elems, "inexact": inexact},
        "ledger": rows,
        "per_rank_resident": {
            "fp4_local": [per_rank(&lf, 0), per_rank(&lf, 1)],
            "quark": [per_rank(&lq, 0), per_rank(&lq, 1)],
        },
        "per_rank_dense": {
            "fp4_local": [dense_rank(&plan, 0), dense_rank(&plan, 1)],
            "quark": quark.as_ref().map(|q| vec![dense_rank(q, 0), dense_rank(q, 1)]),
        },
        "dense_written_bytes": written,
        "expert_tier": {
            "format": t.record.format, "stride": t.record.stride, "pages": t.record.stride / 4096,
            "quark_stride": q_stride, "records_per_rank": t.records(),
            "file_bytes_per_rank": t.records() * t.record.stride,
            "quark_file_bytes_per_rank": t.records() * q_stride,
            "vram_slots_at_quark_budget": {"quark": slots_quark, "fp4_local": slots_fp4},
            "csf_bytes_as_stored": csf_bytes, "csf_planes": csf_planes,
            "plain_e4m3_bytes": t.records() * 2 * (t.record.gate_up_ws.1 + t.record.down_ws.1),
        },
        "ple": {"rows": ple_rows, "nvfp4_row_bytes": plan.ple.row_bytes_nvfp4(),
                "nvfp4_bytes": ple_rows * plan.ple.row_bytes_nvfp4(),
                "bf16_bytes": ple_rows * plan.ple.row_bytes_bf16()},
        "vision": vision.iter().map(|(k, (s, b))| json!({"form": k, "stored": s, "bf16": b}))
            .collect::<Vec<_>>(),
    });
    std::fs::create_dir_all(report)?;
    let pj = serde_json::to_string_pretty(&plan.to_json(&c))?;
    std::fs::write(report.join("plan.json"), &pj)?;
    std::fs::write(
        report.join("accounting.json"),
        serde_json::to_string_pretty(&acct)?,
    )?;
    println!(
        "fn144 dry-run: {} dense records ({} identity, {:.3} GiB written), {checked} source tensors checked, {absent} absent, 0 mismatches",
        plan.dense.len(),
        plan.dense.iter().filter(|r| r.kind.identity()).count(),
        gib(written)
    );
    println!(
        "fn144 dry-run: mxfp8->bf16 {exact_elems} elements, all exact; expert tier {} stride {} B ({} pages; quark {} B), {:.2} GiB a rank (quark {:.2}); VRAM slots at the quark budget {slots_fp4} (quark {slots_quark})",
        t.record.format,
        t.record.stride,
        t.record.stride / 4096,
        q_stride,
        gib(t.records() * t.record.stride),
        gib(t.records() * q_stride)
    );
    let qd = |r: u32| quark.as_ref().map_or(0, |q| dense_rank(q, r));
    println!(
        "fn144 dry-run: dense resident a rank (rank + shared) {:.3} / {:.3} GiB (quark {:.3} / {:.3}); CSF as stored {:.2} GiB over {csf_planes} planes; plan sha256 {}",
        gib(dense_rank(&plan, 0)),
        gib(dense_rank(&plan, 1)),
        gib(qd(0)),
        gib(qd(1)),
        gib(csf_bytes),
        hex::encode(Sha256::digest(pj.as_bytes()))
    );
    Ok(acct)
}

/// One expert's v2 record (`stride` bytes, zero-padded) from the checkpoint.
pub fn expert_record(
    st: &mut ShardSet,
    layer: u32,
    expert: u32,
    hidden: usize,
    inter: usize,
) -> anyhow::Result<Vec<u8>> {
    let (sub, _, stride) = record_layout_nvfp4(hidden as u64, inter as u64);
    let mut rec = vec![0u8; stride as usize];
    let p = format!("model.language_model.layers.{layer}.mlp.experts.{expert}");
    let mut proj =
        |name: &str, n: usize, k: usize| -> anyhow::Result<(Vec<u8>, Vec<u8>, [u8; 4])> {
            let x = format!("{p}.{name}");
            let er = |e: String| anyhow::anyhow!(e);
            let (wi, w) = st.read(&format!("{x}.weight")).map_err(er)?;
            ensure!(
                wi.shape == [n as u64, k as u64 / 2],
                "{x}.weight {:?}",
                wi.shape
            );
            let (_, fixed) = st
                .read(&format!("{x}.weight_scale.nvfp4_csf_fixed"))
                .map_err(er)?;
            let (_, exc) = st
                .read_u32(&format!("{x}.weight_scale.nvfp4_csf_exceptions"))
                .map_err(er)?;
            let d = nvfp4_csf::decode(&fixed, &exc, n, k / 16)
                .map_err(|e| anyhow::anyhow!("{x}: {e}"))?;
            let (gi, g) = st.read(&format!("{x}.weight_scale_2")).map_err(er)?;
            ensure!(gi.dtype == "F32" && g.len() == 4, "{x}.weight_scale_2");
            Ok((w, d.scales, [g[0], g[1], g[2], g[3]]))
        };
    let (gw, gsc, gg) = proj("gate_proj", inter, hidden)?;
    let (uw, usc, ug) = proj("up_proj", inter, hidden)?;
    let (dw, dsc, dg) = proj("down_proj", hidden, inter)?;
    let put = |rec: &mut [u8], at: (u64, u64), b: &[u8]| {
        assert_eq!(b.len() as u64, at.1);
        rec[at.0 as usize..(at.0 + at.1) as usize].copy_from_slice(b);
    };
    // gate_up: rows gate then up, `[2I, H/2]` -> fragment order; scales `[2I, H/16]` -> `[H/16, 2I]`.
    let mut gu = gw;
    gu.extend_from_slice(&uw);
    put(&mut rec, sub[0], &permute_w(&gu, 2 * inter, hidden));
    let transpose = |a: &[u8], n: usize, c: usize| {
        let mut t = vec![0u8; a.len()];
        for r in 0..n {
            for j in 0..c {
                t[j * n + r] = a[r * c + j];
            }
        }
        t
    };
    let mut gus = gsc;
    gus.extend_from_slice(&usc);
    put(&mut rec, sub[1], &transpose(&gus, 2 * inter, hidden / 16));
    put(&mut rec, sub[2], &[gg, ug].concat());
    put(&mut rec, sub[3], &permute_w(&dw, hidden, inter));
    put(&mut rec, sub[4], &transpose(&dsc, hidden, inter / 16));
    put(&mut rec, sub[5], &dg);
    Ok(rec)
}

/// `--layers A:B --out-pack DIR`: the small v2 pack (module docs).
pub fn build_smoke(ckpt: &Path, out: &Path, layers: (u32, u32)) -> anyhow::Result<()> {
    let c = config(ckpt)?;
    let plan = PackPlan::fp4local(&c).map_err(|e| anyhow::anyhow!(e))?;
    let (lo, hi) = layers;
    ensure!(
        lo < hi && hi <= c.num_layers(),
        "--layers {lo}:{hi} outside 0:{}",
        c.num_layers()
    );
    ensure!(
        !out.starts_with("/mnt/nvme/saber-packs/")
            || out.starts_with("/mnt/nvme/saber-packs-smoke"),
        "the smoke pack never lands in the served pack root"
    );
    let plan_json = serde_json::to_string_pretty(&plan.to_json(&c))?;
    let plan_sha: [u8; 32] = Sha256::digest(plan_json.as_bytes()).into();
    std::fs::create_dir_all(out)?;
    std::fs::write(out.join("plan.json"), &plan_json)?;
    let (hidden, inter) = (c.hidden_size as usize, c.moe.intermediate as usize);
    let (sub, payload, stride) = record_layout_nvfp4(hidden as u64, inter as u64);
    let mut st = ShardSet::open(ckpt).map_err(|e| anyhow::anyhow!(e))?;
    let mut tiers = Vec::new();
    for t in &plan.experts {
        let r = t.rank;
        let dir = out.join(format!("rank{r}"));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("experts.xtier");
        let records = u64::from(hi - lo) * u64::from(t.experts);
        let table_off = XTIER_HEADER;
        let table_bytes = records * 32;
        let data_off = (table_off + table_bytes).div_ceil(4096) * 4096;
        let mut f = std::fs::File::create(&path)?;
        f.seek(SeekFrom::Start(data_off))?;
        let mut table = Vec::with_capacity(table_bytes as usize);
        for l in lo..hi {
            for e in t.first_expert..t.first_expert + t.experts {
                let rec = expert_record(&mut st, l, e, hidden, inter)?;
                f.write_all(&rec)?;
                table.extend_from_slice(&record_digest(&rec));
            }
        }
        let h = XtierHeader {
            rank: r,
            layers: hi - lo,
            first_expert: t.first_expert,
            experts: t.experts,
            stride,
            payload,
            sub,
            hidden: c.hidden_size,
            inter: c.moe.intermediate,
            table_off,
            table_bytes,
            data_off,
            file_bytes: data_off + records * stride,
            table_sha256: table_digest(&table),
            plan_sha256: plan_sha,
            split_parts: 0,
            split_index: 0,
            format: XtierFormat::Nvfp4E4m3G16,
        };
        f.seek(SeekFrom::Start(0))?;
        f.write_all(&header_block(&h))?;
        f.seek(SeekFrom::Start(table_off))?;
        f.write_all(&table)?;
        f.set_len(h.file_bytes)?;
        f.sync_all()?;
        drop(f);
        // Read back through the v2 reader (header law + table digest + length).
        let x = XtierFile::open(&path, saber_mem::dio::OpenMode::Buffered)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        ensure!(
            x.header() == &h,
            "{}: header read back differs",
            path.display()
        );
        ensure!(
            x.header().require_mxfp4().is_err(),
            "the v2 tier must refuse an MXFP4 engine"
        );
        println!(
            "fn144 smoke rank{r}: {records} records x {stride} B ({}) -> {} (table {})",
            h.format.key(),
            path.display(),
            hex::encode(h.table_sha256)
        );
        let lay = |x: (u64, u64)| json!([x.0, x.1]);
        tiers.push(json!({
            "file": format!("rank{r}/experts.xtier"), "rank": r, "first_expert": t.first_expert,
            "experts": t.experts, "layers": hi - lo, "layer_offset": lo, "format": h.format.key(),
            "record": {"gate_up_w": lay(sub[0]), "gate_up_ws": lay(sub[1]), "gate_up_gs": lay(sub[2]),
                       "down_w": lay(sub[3]), "down_ws": lay(sub[4]), "down_gs": lay(sub[5]),
                       "payload": payload, "stride": stride},
            "table_sha256": hex::encode(h.table_sha256),
        }));
    }
    let man = json!({
        "model": "qwen4_exp", "source": "fp4_local", "xtier_version": 2,
        "checkpoint": {"dir": ckpt.display().to_string()},
        "plan_sha256": hex::encode(plan_sha), "layers": [lo, hi],
        "expert_tiers": tiers,
        "dense": "not written (FN144 smoke: the expert tier only; dense MXFP8 records are identity references)",
    });
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(&man)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CKPT: &str = "/mnt/nvme/models/Qwen3.8-Flash-Next-FP4-Local";

    /// One v2 record vs the checkpoint, element by element (skips when the checkpoint is absent):
    /// the e2m1 bytes in fragment order, the decoded scales transposed, the globals in order.
    #[test]
    fn fn144_a_v2_record_is_the_checkpoints_bytes_relaid() {
        let x = "model.language_model.layers.0.mlp.experts.7";
        let mut st = match ShardSet::open(Path::new(CKPT)) {
            Ok(s) => s,
            Err(_) => {
                eprintln!("SKIP: {CKPT} absent");
                return;
            }
        };
        if !st.present(&format!("{x}.up_proj.weight")) {
            eprintln!("SKIP: shard not complete");
            return;
        }
        let (h, i) = (2560usize, 640usize);
        let rec = expert_record(&mut st, 0, 7, h, i).unwrap();
        let (sub, _, stride) = record_layout_nvfp4(h as u64, i as u64);
        assert_eq!(rec.len() as u64, stride);
        let (_, upw) = st.read(&format!("{x}.up_proj.weight")).unwrap();
        let (_, fixed) = st
            .read(&format!("{x}.up_proj.weight_scale.nvfp4_csf_fixed"))
            .unwrap();
        let (_, exc) = st
            .read_u32(&format!("{x}.up_proj.weight_scale.nvfp4_csf_exceptions"))
            .unwrap();
        let sc = nvfp4_csf::decode(&fixed, &exc, i, h / 16).unwrap().scales;
        let (_, g) = st.read(&format!("{x}.up_proj.weight_scale_2")).unwrap();
        // gs: [gate, up]; up is the second f32.
        let gs = sub[2].0 as usize;
        assert_eq!(&rec[gs + 4..gs + 8], &g[..]);
        // ws `[H/16, 2I]`: up row r, group j at j * 2I + I + r.
        for (r, j) in [(0usize, 0usize), (5, 17), (639, 159)] {
            assert_eq!(
                rec[sub[1].0 as usize + j * 2 * i + i + r],
                sc[r * (h / 16) + j]
            );
        }
        // w: permute_w of [gate; up] -- up row 7's first 4 bytes sit in n-tile (I + 7) / 16,
        // k-slab 0, half 0, row (I + 7) % 16.
        let at = sub[0].0 as usize + (i + 7) / 16 * 16 * (h / 2) + ((i + 7) % 16) * 4;
        assert_eq!(&rec[at..at + 4], &upw[7 * (h / 2)..7 * (h / 2) + 4]);
        assert!(
            rec[sub[5].0 as usize + 4..].iter().all(|&b| b == 0),
            "zero pad"
        );
    }
}
