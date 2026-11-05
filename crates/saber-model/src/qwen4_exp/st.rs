//! A minimal safetensors shard-set reader (FN144): the checkpoint's `model.safetensors.index.json`
//! names the shard of every tensor; a shard's header is the 8-byte little-endian length + the JSON
//! table; a tensor's bytes are read with one positioned read. Shards not on disk yet (a running
//! download) are reported as absent, never guessed.

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// One tensor's header entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    /// Shard file name (relative to the checkpoint dir).
    pub file: String,
    /// safetensors dtype spelling.
    pub dtype: String,
    /// Shape.
    pub shape: Vec<u64>,
    /// Absolute byte range in the shard file.
    pub range: (u64, u64),
}

/// The checkpoint's shards.
#[derive(Debug)]
pub struct ShardSet {
    dir: PathBuf,
    map: HashMap<String, String>,
    headers: HashMap<String, Option<HashMap<String, TensorInfo>>>,
    files: HashMap<String, File>,
}

fn complete(path: &Path) -> Option<(File, u64)> {
    let f = File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    Some((f, len))
}

impl ShardSet {
    /// Open the index of `dir`.
    pub fn open(dir: &Path) -> Result<Self, String> {
        let p = dir.join("model.safetensors.index.json");
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))?;
        let map = v
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("{}: no weight_map", p.display()))?
            .iter()
            .map(|(k, f)| (k.clone(), f.as_str().unwrap_or_default().to_string()))
            .collect();
        Ok(Self {
            dir: dir.to_path_buf(),
            map,
            headers: HashMap::new(),
            files: HashMap::new(),
        })
    }

    /// Every tensor name in the index.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.map.keys()
    }

    /// The shard file the index names for `name`.
    #[must_use]
    pub fn shard_of(&self, name: &str) -> Option<&str> {
        self.map.get(name).map(String::as_str)
    }

    fn load(&mut self, file: &str) -> Option<&HashMap<String, TensorInfo>> {
        if !self.headers.contains_key(file) {
            let parsed = complete(&self.dir.join(file)).and_then(|(f, len)| {
                let mut lb = [0u8; 8];
                f.read_exact_at(&mut lb, 0).ok()?;
                let n = u64::from_le_bytes(lb);
                if n > 100 << 20 || 8 + n > len {
                    return None;
                }
                let mut hb = vec![0u8; n as usize];
                f.read_exact_at(&mut hb, 8).ok()?;
                let v: Value = serde_json::from_slice(&hb).ok()?;
                let mut out = HashMap::new();
                let mut end = 0u64;
                for (k, e) in v.as_object()? {
                    if k == "__metadata__" {
                        continue;
                    }
                    let off = e.get("data_offsets")?.as_array()?;
                    let (a, b) = (off.first()?.as_u64()?, off.get(1)?.as_u64()?);
                    end = end.max(b);
                    out.insert(
                        k.clone(),
                        TensorInfo {
                            file: file.to_string(),
                            dtype: e.get("dtype")?.as_str()?.to_string(),
                            shape: e
                                .get("shape")?
                                .as_array()?
                                .iter()
                                .filter_map(Value::as_u64)
                                .collect(),
                            range: (8 + n + a, 8 + n + b),
                        },
                    );
                }
                // A shard still downloading is shorter than its header's data: absent.
                if 8 + n + end > len {
                    return None;
                }
                self.files.insert(file.to_string(), f);
                Some(out)
            });
            self.headers.insert(file.to_string(), parsed);
        }
        self.headers.get(file).and_then(Option::as_ref)
    }

    /// The header entry of `name`, `None` when the index lacks it or its shard is not complete.
    pub fn info(&mut self, name: &str) -> Option<TensorInfo> {
        let file = self.map.get(name)?.clone();
        self.load(&file)?.get(name).cloned()
    }

    /// Whether the shard holding `name` is complete on disk.
    pub fn present(&mut self, name: &str) -> bool {
        self.info(name).is_some()
    }

    /// The bytes of `name`.
    pub fn read(&mut self, name: &str) -> Result<(TensorInfo, Vec<u8>), String> {
        let info = self
            .info(name)
            .ok_or_else(|| format!("{name}: not in a complete shard"))?;
        let f = self
            .files
            .get(&info.file)
            .ok_or_else(|| format!("{name}: shard not open"))?;
        let mut b = vec![0u8; (info.range.1 - info.range.0) as usize];
        f.read_exact_at(&mut b, info.range.0)
            .map_err(|e| format!("{}: {e}", info.file))?;
        Ok((info, b))
    }

    /// The bytes of `name` as little-endian u32 words.
    pub fn read_u32(&mut self, name: &str) -> Result<(TensorInfo, Vec<u32>), String> {
        let (i, b) = self.read(name)?;
        if i.dtype != "U32" && i.dtype != "I32" {
            return Err(format!("{name}: dtype {} is not U32", i.dtype));
        }
        let w = b
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok((i, w))
    }
}
