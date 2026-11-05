//! The head a pack carries is the CHECKPOINT's head (workstream N6).
//!
//! **The law** (the user's decision of 2026-09-22, superseding the bf16 default of the same day):
//! the `lm_head` is served at the checkpoint's own precision and is never requantized. The pack
//! builder does not take a per-model choice; it reads the checkpoint's head tensor and picks the
//! record kind that carries it unchanged:
//!
//! | `lm_head.weight` dtype | `lm_head.weight_scale` | head | records |
//! |---|---|---|---|
//! | `F8_E4M3` | present (per-channel) | [`CheckpointHead::Fp8Channel`] | `lm_head.w8` (layout 15, raw) + `lm_head.ws` (layout 16, f32) |
//! | `BF16` | absent | [`CheckpointHead::Bf16`] | `lm_head.w` (bf16) |
//!
//! Anything else — an fp8 head with no scale, an f16/f32 head, a checkpoint with no
//! `lm_head.weight` at all (tied embeddings) — is a refusal that names the tensor and its dtype,
//! because there is no record kind for it yet and silently picking one would be a requantization.
//!
//! What it reads: the safetensors HEADER only (the 8-byte little-endian length and the JSON table
//! after it) of the file `model.safetensors.index.json` maps `lm_head.weight` to, or of the lone
//! `model.safetensors` when there is no index. No tensor byte is read.

use std::io::Read;
use std::path::{Path, PathBuf};

/// The head kind a checkpoint's `lm_head` is stored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointHead {
    /// e4m3 weight + a per-output-channel scale (this repo's Qwen3.8-27B-NVFP4).
    Fp8Channel,
    /// A plain bf16 weight.
    Bf16,
}

impl CheckpointHead {
    /// The `saber-packplan --head` / `build_devfinal.sh --head` value that builds it unchanged.
    #[must_use]
    pub fn pack_head(self) -> &'static str {
        match self {
            CheckpointHead::Fp8Channel => "fp8",
            CheckpointHead::Bf16 => "bf16",
        }
    }
}

/// What the resolution read, for the line the builder prints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadEvidence {
    /// The resolved kind.
    pub head: CheckpointHead,
    /// The safetensors file the header was read from.
    pub file: PathBuf,
    /// `lm_head.weight`'s dtype and shape, as the header states them.
    pub weight: (String, Vec<u64>),
    /// `lm_head.weight_scale`'s, when the checkpoint has one.
    pub scale: Option<(String, Vec<u64>)>,
}

impl HeadEvidence {
    /// One line: what was read and what it resolved to.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "head auto: {} has lm_head.weight {} {:?}{} -> `--head {}`",
            self.file.display(),
            self.weight.0,
            self.weight.1,
            match &self.scale {
                Some((d, s)) => format!(" + lm_head.weight_scale {d} {s:?}"),
                None => " and no lm_head.weight_scale".to_owned(),
            },
            self.head.pack_head()
        )
    }
}

fn entry(header: &serde_json::Value, name: &str) -> Option<(String, Vec<u64>)> {
    let t = header.get(name)?;
    let dtype = t.get("dtype")?.as_str()?.to_owned();
    let shape = t
        .get("shape")?
        .as_array()?
        .iter()
        .filter_map(serde_json::Value::as_u64)
        .collect();
    Some((dtype, shape))
}

/// The law over one safetensors header table (the JSON object after the length prefix).
///
/// # Errors
/// No `lm_head.weight`; an fp8 head without its per-channel scale; any other dtype — each by name.
pub fn head_from_header(header: &serde_json::Value, file: &Path) -> Result<HeadEvidence, String> {
    let weight = entry(header, "lm_head.weight").ok_or_else(|| {
        format!(
            "{}: no `lm_head.weight` (a tied-embedding checkpoint?): there is no head tensor to \
             serve at its own precision; pass `--head bf16|fp8|mxfp4` explicitly",
            file.display()
        )
    })?;
    let scale = entry(header, "lm_head.weight_scale");
    let head = match (weight.0.as_str(), &scale) {
        ("F8_E4M3", Some((sd, ss))) if ss.first() == weight.1.first() => {
            if !matches!(sd.as_str(), "BF16" | "F32" | "F16") {
                return Err(format!(
                    "{}: `lm_head.weight_scale` is {sd}: the fp8 head's scale record is f32 widened \
                     from a float scale",
                    file.display()
                ));
            }
            CheckpointHead::Fp8Channel
        }
        ("F8_E4M3", Some((_, ss))) => {
            return Err(format!(
                "{}: `lm_head.weight` is F8_E4M3 {:?} and its `weight_scale` is {ss:?}: not one \
                 scale per output channel, so it is not the per-channel fp8 head the fp8 record \
                 carries",
                file.display(),
                weight.1
            ))
        }
        ("F8_E4M3", None) => {
            return Err(format!(
                "{}: `lm_head.weight` is F8_E4M3 with no `lm_head.weight_scale`: an unscaled fp8 \
                 head has no record kind",
                file.display()
            ))
        }
        ("BF16", _) => CheckpointHead::Bf16,
        (other, _) => {
            return Err(format!(
                "{}: `lm_head.weight` is {other}: the head is served at the checkpoint's own \
                 precision and there is no {other} head record; pass `--head` explicitly",
                file.display()
            ))
        }
    };
    Ok(HeadEvidence {
        head,
        file: file.to_path_buf(),
        weight,
        scale,
    })
}

/// Read one safetensors file's header table.
///
/// # Errors
/// I/O, a length prefix past 100 MB, or JSON that does not parse.
pub fn read_header(file: &Path) -> Result<serde_json::Value, String> {
    let mut f = std::fs::File::open(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)
        .map_err(|e| format!("{}: {e}", file.display()))?;
    let n = u64::from_le_bytes(len);
    if n > 100 << 20 {
        return Err(format!(
            "{}: a {n}-byte header is not a safetensors file",
            file.display()
        ));
    }
    let mut buf = vec![0u8; usize::try_from(n).map_err(|e| e.to_string())?];
    f.read_exact(&mut buf)
        .map_err(|e| format!("{}: {e}", file.display()))?;
    serde_json::from_slice(&buf).map_err(|e| format!("{}: header JSON: {e}", file.display()))
}

/// The checkpoint directory's head: `model.safetensors.index.json`'s `weight_map` names the file
/// that holds `lm_head.weight` (else the lone `model.safetensors`), and that file's header decides.
///
/// # Errors
/// The index or the file cannot be read, or [`head_from_header`] refuses.
pub fn read_checkpoint_head(dir: &Path) -> Result<HeadEvidence, String> {
    let index = dir.join("model.safetensors.index.json");
    let file = if index.is_file() {
        let t = std::fs::read_to_string(&index).map_err(|e| format!("{}: {e}", index.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&t).map_err(|e| format!("{}: {e}", index.display()))?;
        match v
            .get("weight_map")
            .and_then(|m| m.get("lm_head.weight"))
            .and_then(serde_json::Value::as_str)
        {
            Some(f) => dir.join(f),
            None => {
                return Err(format!(
                    "{}: the weight map has no `lm_head.weight`",
                    index.display()
                ))
            }
        }
    } else {
        dir.join("model.safetensors")
    };
    head_from_header(&read_header(&file)?, &file)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hdr(v: serde_json::Value) -> Result<CheckpointHead, String> {
        head_from_header(&v, Path::new("synthetic.safetensors")).map(|e| e.head)
    }

    /// The law over synthetic headers: fp8 + per-channel scale -> fp8, bf16 -> bf16; the
    /// CONTROLS are every shape the law must refuse rather than guess (an unscaled fp8 head, a
    /// per-tensor scale, an f16 head, no head), each by name.
    #[test]
    fn the_head_kind_is_the_checkpoints_dtype() {
        let fp8 = serde_json::json!({
            "lm_head.weight": {"dtype": "F8_E4M3", "shape": [248_320, 5120], "data_offsets": [0, 1]},
            "lm_head.weight_scale": {"dtype": "BF16", "shape": [248_320, 1], "data_offsets": [1, 2]},
        });
        assert_eq!(hdr(fp8), Ok(CheckpointHead::Fp8Channel));
        let bf = serde_json::json!({
            "lm_head.weight": {"dtype": "BF16", "shape": [248_320, 5120], "data_offsets": [0, 1]},
        });
        assert_eq!(hdr(bf), Ok(CheckpointHead::Bf16));
        for (v, needle) in [
            (
                serde_json::json!({"lm_head.weight": {"dtype": "F8_E4M3", "shape": [8, 16]}}),
                "no `lm_head.weight_scale`",
            ),
            (
                serde_json::json!({
                    "lm_head.weight": {"dtype": "F8_E4M3", "shape": [8, 16]},
                    "lm_head.weight_scale": {"dtype": "F32", "shape": [1]},
                }),
                "not one scale per output channel",
            ),
            (
                serde_json::json!({"lm_head.weight": {"dtype": "F16", "shape": [8, 16]}}),
                "is F16",
            ),
            (
                serde_json::json!({"model.embed_tokens.weight": {"dtype": "BF16", "shape": [8]}}),
                "no `lm_head.weight`",
            ),
        ] {
            let e = hdr(v).expect_err(needle);
            assert!(e.contains(needle), "{needle}: {e}");
        }
    }

    /// A synthetic bf16-head CHECKPOINT on disk (index + one file), read the way the builder reads
    /// the real one: the resolution follows the index to the file and reads only its header.
    #[test]
    fn a_synthetic_bf16_checkpoint_resolves_to_bf16() {
        // Under `target/`, removed on drop -- never `/tmp` (FN51h).
        let dir = saber_test_scratch::ScratchDir::new("saber-n6-ckpt");
        let header = serde_json::json!({
            "__metadata__": {"format": "pt"},
            "lm_head.weight": {"dtype": "BF16", "shape": [64, 32], "data_offsets": [0, 4096]},
        });
        let h = serde_json::to_vec(&header).expect("json");
        let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&h);
        bytes.resize(bytes.len() + 4096, 0);
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), &bytes).expect("write");
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map": {"lm_head.weight": "model-00002-of-00002.safetensors"}}"#,
        )
        .expect("write index");
        let ev = read_checkpoint_head(&dir).expect("resolves");
        assert_eq!(ev.head, CheckpointHead::Bf16);
        assert!(ev.line().contains("--head bf16"), "{}", ev.line());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// This box's checkpoint, when it is here: Qwen3.8-27B-NVFP4's head is per-channel fp8.
    #[test]
    fn this_checkpoint_is_fp8() {
        let dir = Path::new("/mnt/nvme/models/Qwen3.8-27B-NVFP4");
        if !dir.is_dir() {
            eprintln!("skipped: the checkpoint is not on this box");
            return;
        }
        let ev = read_checkpoint_head(dir).expect("resolves");
        assert_eq!(ev.head, CheckpointHead::Fp8Channel, "{}", ev.line());
        assert_eq!(ev.weight, ("F8_E4M3".to_owned(), vec![248_320, 5120]));
        assert_eq!(ev.scale, Some(("BF16".to_owned(), vec![248_320, 1])));
    }
}
