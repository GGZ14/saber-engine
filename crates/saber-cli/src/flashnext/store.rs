//! Weight bytes for the Flash-Next executor: whole checkpoint tensors (TP=1, gate A: the harness
//! reads the checkpoint the same way) and FN2's pack records (TP=2: each rank's `dense.json` +
//! `aliases.json`, then `shared/`), every record checked against its sha256 as FN9b did.

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{bail, ensure, Context};
use saber_kernels::manifest::sha256_hex;
use serde_json::Value;

/// One safetensors tensor: file, absolute byte range, dtype, shape.
#[derive(Debug, Clone)]
pub struct TensorLoc {
    pub file: String,
    pub begin: u64,
    pub end: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
}

/// The checkpoint by tensor name (index + lazily parsed shard headers).
#[derive(Debug)]
pub struct Ckpt {
    pub dir: PathBuf,
    map: HashMap<String, String>,
    shards: Mutex<HashMap<String, HashMap<String, TensorLoc>>>,
}

impl Ckpt {
    pub fn open(dir: &Path) -> anyhow::Result<Ckpt> {
        let idx: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("model.safetensors.index.json"))
                .with_context(|| format!("{}/model.safetensors.index.json", dir.display()))?,
        )?;
        let map = idx["weight_map"]
            .as_object()
            .context("weight_map")?
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
            .collect();
        Ok(Ckpt {
            dir: dir.to_path_buf(),
            map,
            shards: Mutex::new(HashMap::new()),
        })
    }

    fn shard(&self, file: &str) -> anyhow::Result<HashMap<String, TensorLoc>> {
        let mut g = self.shards.lock().expect("shard map");
        if let Some(s) = g.get(file) {
            return Ok(s.clone());
        }
        let f = File::open(self.dir.join(file)).with_context(|| file.to_string())?;
        let mut n = [0u8; 8];
        f.read_exact_at(&mut n, 0)?;
        let hl = u64::from_le_bytes(n);
        let mut h = vec![0u8; hl as usize];
        f.read_exact_at(&mut h, 8)?;
        let v: Value = serde_json::from_slice(&h)?;
        let base = 8 + hl;
        let mut out = HashMap::new();
        for (k, t) in v.as_object().context("safetensors header")? {
            if k == "__metadata__" {
                continue;
            }
            let off = t["data_offsets"].as_array().context("data_offsets")?;
            out.insert(
                k.clone(),
                TensorLoc {
                    file: file.to_string(),
                    begin: base + off[0].as_u64().context("off")?,
                    end: base + off[1].as_u64().context("off")?,
                    dtype: t["dtype"].as_str().unwrap_or_default().to_string(),
                    shape: t["shape"]
                        .as_array()
                        .context("shape")?
                        .iter()
                        .filter_map(Value::as_u64)
                        .collect(),
                },
            );
        }
        g.insert(file.to_string(), out.clone());
        Ok(out)
    }

    pub fn loc(&self, name: &str) -> anyhow::Result<TensorLoc> {
        let file = self
            .map
            .get(name)
            .with_context(|| format!("{name}: not in the checkpoint index"))?;
        self.shard(file)?
            .get(name)
            .cloned()
            .with_context(|| format!("{name}: not in {file}"))
    }

    /// The raw bytes of a tensor.
    pub fn raw(&self, name: &str) -> anyhow::Result<Vec<u8>> {
        let t = self.loc(name)?;
        let f = File::open(self.dir.join(&t.file))?;
        let mut b = vec![0u8; (t.end - t.begin) as usize];
        f.read_exact_at(&mut b, t.begin)?;
        Ok(b)
    }

    /// A bf16 tensor's raw bytes (refuses another dtype).
    pub fn bf16(&self, name: &str) -> anyhow::Result<Vec<u8>> {
        let t = self.loc(name)?;
        ensure!(t.dtype == "BF16", "{name}: {} (want BF16)", t.dtype);
        ensure!(
            t.end - t.begin == 2 * t.shape.iter().product::<u64>(),
            "{name}: extent != shape"
        );
        self.raw(name)
    }

    /// A bf16 tensor widened to f32 bytes (exact), as the harness's `ck.f32` does.
    pub fn f32_from_bf16(&self, name: &str) -> anyhow::Result<Vec<u8>> {
        let b = self.bf16(name)?;
        Ok(b.chunks_exact(2)
            .flat_map(|c| (u32::from(u16::from_le_bytes([c[0], c[1]])) << 16).to_le_bytes())
            .collect())
    }
}

/// FN2's pack records: rank r's `dense.json` + `aliases.json`, and `shared/dense.json`.
#[derive(Debug)]
pub struct PackRecs {
    pub dir: PathBuf,
    pub ckpt_dir: PathBuf,
    ranks: [HashMap<String, Value>; 2],
    shared: HashMap<String, Value>,
    /// FN121 (`--fn-ple-prefetch embed`): the whole token-embedding table in host RAM (the two
    /// ranks' identity records' bytes, rank 0's then rank 1's = row `t` at `t x 5,120`), read once
    /// at boot; [`embed_rows`] copies from it instead of a page-cache read of the checkpoint file.
    emb_ram: EmbRam,
}

/// FN121: the host-RAM embedding table (its `Debug` prints the size, never the bytes).
#[derive(Default)]
pub struct EmbRam(std::sync::OnceLock<Vec<u8>>);

impl std::fmt::Debug for EmbRam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EmbRam({:?} B)", self.0.get().map(Vec::len))
    }
}

/// The embedding table's rows a rank's identity record holds (the vocab half).
pub const EMBED_HALF: u32 = 124_160;
/// One embedding row's bytes (2,560 bf16).
pub const EMBED_ROW_BYTES: u64 = 5120;

impl PackRecs {
    pub fn open(dir: &Path, ckpt_dir: &Path) -> anyhow::Result<PackRecs> {
        let load = |p: PathBuf| -> anyhow::Result<HashMap<String, Value>> {
            let v: Value = serde_json::from_str(
                &std::fs::read_to_string(&p).with_context(|| p.display().to_string())?,
            )?;
            let mut m = HashMap::new();
            for key in ["records", "aliases"] {
                if let Some(a) = v[key].as_array() {
                    for r in a {
                        m.insert(
                            r["name"].as_str().unwrap_or_default().to_string(),
                            r.clone(),
                        );
                    }
                }
            }
            Ok(m)
        };
        let mut ranks = [HashMap::new(), HashMap::new()];
        for (r, slot) in ranks.iter_mut().enumerate() {
            *slot = load(dir.join(format!("rank{r}/dense.json")))?;
            slot.extend(load(dir.join(format!("rank{r}/aliases.json")))?);
            // FN50: the ADDITIONAL fp8 MTP expert file, when built (`saber flashnext-pack`)
            let f8 = dir.join(format!("rank{r}/{}.json", super::mtp8::STEM));
            if f8.exists() {
                slot.extend(load(f8)?);
            }
            // FN110: the ADDITIONAL MXFP4 MTP expert file, when built (`build_mtp_x4.py`)
            let x4 = dir.join(format!("rank{r}/{}.json", super::mtpx4::STEM));
            if x4.exists() {
                slot.extend(load(x4)?);
            }
            // FN61: the W4 trunk members of a `-trunk4` variant (`saber flashnext-pack --trunk w4`)
            let t4 = dir.join(format!("rank{r}/{}.json", super::trunk4::STEM));
            if t4.exists() {
                slot.extend(load(t4)?);
            }
            // FN61b: the W8 trunk members of a `-trunk8` variant (`saber flashnext-pack --trunk w8`)
            let t8 = dir.join(format!("rank{r}/{}.json", super::trunk8::STEM));
            if t8.exists() {
                slot.extend(load(t8)?);
            }
        }
        let mut shared = load(dir.join("shared/dense.json"))?;
        let t4 = dir.join(format!("shared/{}.json", super::trunk4::STEM));
        if t4.exists() {
            shared.extend(load(t4)?);
        }
        // FN108: the fp8 HC mixing matrices (`saber flashnext-pack --hc8`)
        let h8 = dir.join(format!("shared/{}.json", super::hc8::STEM));
        if h8.exists() {
            shared.extend(load(h8)?);
        }
        Ok(PackRecs {
            dir: dir.to_path_buf(),
            ckpt_dir: ckpt_dir.to_path_buf(),
            ranks,
            shared,
            emb_ram: EmbRam::default(),
        })
    }

    /// FN121 (`--fn-ple-prefetch embed`): read the whole embedding table into host RAM (once;
    /// a second call is a no-op) and check `probes` rows against the file. Returns the bytes held.
    pub fn embed_to_ram(&self, probes: usize) -> anyhow::Result<u64> {
        if let Some(t) = self.emb_ram.0.get() {
            return Ok(t.len() as u64);
        }
        let half = u64::from(EMBED_HALF) * EMBED_ROW_BYTES;
        let mut t = vec![0u8; (2 * half) as usize];
        for r in 0..2u32 {
            let (f, off, len) = self.extent(r, "model.language_model.embed_tokens.weight")?;
            ensure!(len == half, "embed record rank{r}: {len} B");
            let file = File::open(&f).with_context(|| f.display().to_string())?;
            let dst = &mut t[(u64::from(r) * half) as usize..(u64::from(r + 1) * half) as usize];
            for (k, c) in dst.chunks_mut(64 << 20).enumerate() {
                file.read_exact_at(c, off + (k as u64) * (64 << 20))?;
            }
        }
        // the copy law, checked: rows across both halves == the file path's rows
        let vocab = 2 * EMBED_HALF;
        let ids: Vec<u32> = (0..probes as u64)
            .map(|i| ((i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 17) % u64::from(vocab)) as u32)
            .chain([0, EMBED_HALF - 1, EMBED_HALF, vocab - 1])
            .collect();
        let want = embed_rows(self, &ids)?;
        let rb = EMBED_ROW_BYTES as usize;
        for (i, &id) in ids.iter().enumerate() {
            ensure!(
                t[id as usize * rb..(id as usize + 1) * rb] == want[i * rb..(i + 1) * rb],
                "FN121: the RAM embedding row {id} differs from the file's"
            );
        }
        let n = t.len() as u64;
        self.emb_ram
            .0
            .set(t)
            .map_err(|_| anyhow::anyhow!("FN121: the RAM embedding table was set twice"))?;
        Ok(n)
    }

    /// The record `name` for `rank` (the rank's own first, then `shared/`), and where it lives.
    pub fn record(&self, rank: u32, name: &str) -> anyhow::Result<(&Value, PathBuf)> {
        if let Some(r) = self.ranks[rank as usize].get(name) {
            return Ok((r, self.dir.join(format!("rank{rank}"))));
        }
        if let Some(r) = self.shared.get(name) {
            return Ok((r, self.dir.join("shared")));
        }
        bail!("{name}: no record for rank {rank} or shared")
    }

    /// A record's bytes, checked against its sha256.
    pub fn raw(&self, rank: u32, name: &str) -> anyhow::Result<Vec<u8>> {
        let (rec, home) = self.record(rank, name)?;
        let st = &rec["storage"];
        let mut out = Vec::new();
        if let Some(ext) = st["checkpoint"].as_array() {
            for e in ext {
                let f = File::open(self.ckpt_dir.join(e["file"].as_str().context("file")?))?;
                let (off, len, stride, count) = (
                    e["offset"].as_u64().context("offset")?,
                    e["length"].as_u64().context("length")?,
                    e["stride"].as_u64().unwrap_or(0),
                    e["count"].as_u64().unwrap_or(1),
                );
                for i in 0..count {
                    let mut b = vec![0u8; len as usize];
                    f.read_exact_at(&mut b, off + i * stride)?;
                    out.extend_from_slice(&b);
                }
            }
        } else {
            let f = File::open(home.join(st["pack"].as_str().context("pack")?))?;
            let mut b = vec![0u8; st["length"].as_u64().context("length")? as usize];
            f.read_exact_at(&mut b, st["data_off"].as_u64().context("data_off")?)?;
            out = b;
        }
        let want = st["sha256"]
            .as_str()
            .or_else(|| rec["sha256"].as_str())
            .context("sha256")?;
        ensure!(
            sha256_hex(&out) == want,
            "rank{rank} {name}: bytes != the record's sha256"
        );
        Ok(out)
    }

    /// A record's storage extent into the checkpoint (for row gathers of an identity record).
    pub fn extent(&self, rank: u32, name: &str) -> anyhow::Result<(PathBuf, u64, u64)> {
        let (rec, _) = self.record(rank, name)?;
        let e = rec["storage"]["checkpoint"]
            .as_array()
            .and_then(|a| a.first())
            .with_context(|| format!("{name}: not a checkpoint-identity record"))?;
        Ok((
            self.ckpt_dir.join(e["file"].as_str().context("file")?),
            e["offset"].as_u64().context("offset")?,
            e["length"].as_u64().context("length")?,
        ))
    }
}

/// The embedding rows of `ids` from the two ranks' vocab-parallel identity records
/// (`rank0` rows 0..124,160, `rank1` the rest), bf16 `[T, 2560]`. The embedding is a HOST gather
/// at prefill (a copy, not a launch).
pub fn embed_rows(p: &PackRecs, ids: &[u32]) -> anyhow::Result<Vec<u8>> {
    const HALF: u32 = EMBED_HALF;
    const RB: u64 = EMBED_ROW_BYTES;
    // FN121 (`--fn-ple-prefetch embed`): the same bytes from the host-RAM table
    if let Some(t) = p.emb_ram.0.get() {
        let rb = RB as usize;
        let mut out = vec![0u8; ids.len() * rb];
        for (i, &id) in ids.iter().enumerate() {
            ensure!(id < 2 * HALF, "token {id} outside the vocab");
            out[i * rb..(i + 1) * rb].copy_from_slice(&t[id as usize * rb..(id as usize + 1) * rb]);
        }
        return Ok(out);
    }
    let mut ext = Vec::new();
    for r in 0..2 {
        let (f, off, len) = p.extent(r, "model.language_model.embed_tokens.weight")?;
        ensure!(len == u64::from(HALF) * RB, "embed record rank{r}: {len} B");
        ext.push((File::open(&f)?, off));
    }
    let mut out = vec![0u8; ids.len() * RB as usize];
    for (i, &t) in ids.iter().enumerate() {
        let (r, row) = if t < HALF { (0, t) } else { (1, t - HALF) };
        ensure!(row < HALF, "token {t} outside the vocab");
        ext[r].0.read_exact_at(
            &mut out[i * RB as usize..(i + 1) * RB as usize],
            ext[r].1 + u64::from(row) * RB,
        )?;
    }
    Ok(out)
}
