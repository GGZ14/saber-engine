//! HF2 (2026-11-23): where a Flash-Next pack's checkpoint and model files are on THIS machine.
//!
//! A pack's `manifest.json` records the checkpoint it was built from as an absolute directory
//! (`checkpoint.dir`, e.g. `/mnt/nvme/models/Qwen3.8-Flash-Next-Quark-MXFP4`), and its
//! `ple/ple_index.json` names the PLE shards under the same directory. A downloaded pack sits
//! wherever the user put it, so the boot resolves the directory in this order:
//!
//! 1. `--fn-ckpt-dir DIR` when given (it must be a checkpoint directory);
//! 2. the recorded directory when it exists -- **the control: a pack whose recorded path is
//!    valid boots exactly as before** (no other path is looked at);
//! 3. the recorded path REBASED under a storage root: `$SABER_NVME`, then the directory above the
//!    pack's parent (a pack at `<root>/saber-packs/<pack>` gives `<root>`), joined with the
//!    recorded path's trailing components, longest first (`<root>/models/<name>`, then
//!    `<root>/<name>`). The first that holds `model.safetensors.index.json` wins.
//!
//! The small model files (`config.json`, `generation_config.json`, `chat_template.jinja`,
//! `tokenizer.json`) are read from the pack's own `base/` directory when the pack ships them,
//! else from the checkpoint directory.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use serde_json::Value;

/// The pack directory that carries the base checkpoint's small files (HF2).
pub const PACK_BASE_DIR: &str = "base";

/// The file that marks a checkpoint directory.
pub const CKPT_MARKER: &str = "model.safetensors.index.json";

/// The model files a Flash-Next boot reads by name (from [`PACK_BASE_DIR`] first).
pub const MODEL_FILES: [&str; 4] = [
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
];

/// How the checkpoint directory was found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CkptFrom {
    /// `--fn-ckpt-dir`.
    Flag,
    /// The manifest's own absolute path (the control).
    Recorded,
    /// The recorded path rebased under a storage root.
    Rebased,
}

/// The recorded `checkpoint.dir` of a pack's manifest.
pub fn recorded_ckpt(pack: &Path) -> anyhow::Result<PathBuf> {
    let p = pack.join("manifest.json");
    let v: Value = serde_json::from_str(
        &std::fs::read_to_string(&p).with_context(|| p.display().to_string())?,
    )?;
    Ok(PathBuf::from(
        v.pointer("/checkpoint/dir")
            .and_then(Value::as_str)
            .context("manifest checkpoint.dir")?,
    ))
}

fn is_ckpt(d: &Path) -> bool {
    d.join(CKPT_MARKER).is_file()
}

/// The storage roots a recorded path may be rebased under, in order: `env_root`
/// (`$SABER_NVME`), then the directory above the pack's parent.
#[must_use]
pub fn storage_roots(pack: &Path, env_root: Option<&Path>) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Some(r) = env_root.filter(|r| !r.as_os_str().is_empty()) {
        out.push(r.to_path_buf());
    }
    let abs = std::fs::canonicalize(pack).unwrap_or_else(|_| pack.to_path_buf());
    if let Some(r) = abs.parent().and_then(Path::parent) {
        if !out.iter().any(|o| o == r) {
            out.push(r.to_path_buf());
        }
    }
    out
}

/// The candidates for `recorded` under `roots`: every root joined with the recorded path's
/// trailing components, longest suffix first (the whole path itself is not a candidate).
#[must_use]
pub fn rebase_candidates(recorded: &Path, roots: &[PathBuf]) -> Vec<PathBuf> {
    let comps: Vec<_> = recorded
        .components()
        .filter(|c| matches!(c, std::path::Component::Normal(_)))
        .collect();
    let mut out = Vec::new();
    for r in roots {
        for k in 1..comps.len() {
            let mut p = r.clone();
            for c in &comps[k..] {
                p.push(c);
            }
            if !out.contains(&p) {
                out.push(p);
            }
        }
    }
    out
}

/// Resolve the checkpoint directory (the law in the module doc). Pure in its inputs except for
/// the filesystem probes.
pub fn resolve_ckpt(
    pack: &Path,
    recorded: &Path,
    flag: Option<&Path>,
    env_root: Option<&Path>,
) -> anyhow::Result<(PathBuf, CkptFrom)> {
    if let Some(d) = flag {
        if !is_ckpt(d) {
            bail!(
                "--fn-ckpt-dir {}: no {CKPT_MARKER} there (the directory of the checkpoint the pack \
                 was built from, {})",
                d.display(),
                recorded.display()
            );
        }
        return Ok((d.to_path_buf(), CkptFrom::Flag));
    }
    if recorded.is_dir() {
        return Ok((recorded.to_path_buf(), CkptFrom::Recorded));
    }
    let cands = rebase_candidates(recorded, &storage_roots(pack, env_root));
    if let Some(c) = cands.iter().find(|c| is_ckpt(c)) {
        return Ok((c.clone(), CkptFrom::Rebased));
    }
    bail!(
        "the pack's checkpoint {} is not on this machine, and none of {} holds it ({CKPT_MARKER}); \
         download it (tools/fetch_packs.sh flashnext-macaq), set SABER_NVME to the directory that \
         holds models/ and saber-packs/, or pass --fn-ckpt-dir DIR",
        recorded.display(),
        cands
            .iter()
            .map(|c| c.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// The checkpoint directory of `pack` for this boot (`--fn-ckpt-dir`, `$SABER_NVME`).
pub fn ckpt_of(pack: &Path, flag: Option<&Path>) -> anyhow::Result<(PathBuf, CkptFrom)> {
    let env = std::env::var_os("SABER_NVME").map(PathBuf::from);
    resolve_ckpt(pack, &recorded_ckpt(pack)?, flag, env.as_deref())
}

/// The PLE index's checkpoint root: `None` (the index's own absolute paths) when the checkpoint
/// is where the manifest recorded it, else the resolved directory.
#[must_use]
pub fn ple_root(pack: &Path, ckpt: &Path) -> Option<PathBuf> {
    match recorded_ckpt(pack) {
        Ok(r) if r == ckpt => None,
        _ => Some(ckpt.to_path_buf()),
    }
}

/// Where the boot reads the model file `name`: the pack's `base/` copy when the pack ships one,
/// else the checkpoint's.
#[must_use]
pub fn model_file(pack: &Path, ckpt: &Path, name: &str) -> PathBuf {
    let b = pack.join(PACK_BASE_DIR).join(name);
    if b.is_file() {
        b
    } else {
        ckpt.join(name)
    }
}

/// The directory the serve plane reads its tokenizer / template / generation config from: the
/// pack's `base/` when it ships all of [`MODEL_FILES`], else the checkpoint.
#[must_use]
pub fn model_files_dir(pack: &Path, ckpt: &Path) -> PathBuf {
    let b = pack.join(PACK_BASE_DIR);
    if MODEL_FILES.iter().all(|n| b.join(n).is_file()) {
        b
    } else {
        ckpt.to_path_buf()
    }
}

/// The boot line naming what was resolved.
#[must_use]
pub fn boot_line(pack: &Path, ckpt: &Path, from: CkptFrom) -> String {
    let files = model_files_dir(pack, ckpt);
    format!(
        "fnpaths: checkpoint {} ({}), model files {}",
        ckpt.display(),
        match from {
            CkptFrom::Flag => "--fn-ckpt-dir",
            CkptFrom::Recorded => "the manifest's path",
            CkptFrom::Rebased => "rebased under the storage root",
        },
        if files == ckpt {
            "from the checkpoint".to_string()
        } else {
            format!("from the pack's {}", files.display())
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(p: &Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"{}").unwrap();
    }

    /// root/saber-packs/pk (manifest recording `recorded`) + optionally root/models/ck.
    fn tree(root: &Path, recorded: &Path, with_ckpt: bool) -> PathBuf {
        let pack = root.join("saber-packs/pk");
        std::fs::create_dir_all(&pack).unwrap();
        std::fs::write(
            pack.join("manifest.json"),
            serde_json::json!({"checkpoint": {"dir": recorded}}).to_string(),
        )
        .unwrap();
        if with_ckpt {
            touch(&root.join("models/ck").join(CKPT_MARKER));
        }
        pack
    }

    #[test]
    fn a_valid_recorded_path_is_used_as_is() {
        let s = saber_test_scratch::ScratchDir::new("hf2-recorded");
        let real = s.path().join("orig/models/ck");
        touch(&real.join(CKPT_MARKER));
        // a second copy under the pack's root must NOT be preferred
        let pack = tree(&s.path().join("other"), &real, true);
        let (d, from) = resolve_ckpt(&pack, &real, None, Some(&s.path().join("other"))).unwrap();
        assert_eq!((d.as_path(), from), (real.as_path(), CkptFrom::Recorded));
        assert_eq!(ple_root(&pack, &d), None);
    }

    #[test]
    fn a_missing_recorded_path_is_rebased_under_the_packs_root() {
        let s = saber_test_scratch::ScratchDir::new("hf2-rebase");
        let rec = Path::new("/no/such/hf2/models/ck");
        let root = s.path().join("fresh");
        let pack = tree(&root, rec, true);
        let (d, from) = resolve_ckpt(&pack, rec, None, None).unwrap();
        assert_eq!(from, CkptFrom::Rebased);
        assert_eq!(d, std::fs::canonicalize(&root).unwrap().join("models/ck"));
        assert_eq!(ple_root(&pack, &d), Some(d.clone()));
    }

    #[test]
    fn the_env_root_comes_before_the_packs_root() {
        let s = saber_test_scratch::ScratchDir::new("hf2-env");
        let rec = Path::new("/no/such/hf2/models/ck");
        let pack = tree(&s.path().join("a"), rec, true);
        let env = s.path().join("b");
        touch(&env.join("models/ck").join(CKPT_MARKER));
        let (d, _) = resolve_ckpt(&pack, rec, None, Some(&env)).unwrap();
        assert_eq!(d, env.join("models/ck"));
        // a bare <root>/<name> also resolves
        let env2 = s.path().join("c");
        touch(&env2.join("ck").join(CKPT_MARKER));
        let pack2 = tree(&s.path().join("d"), rec, false);
        let (d2, _) = resolve_ckpt(&pack2, rec, None, Some(&env2)).unwrap();
        assert_eq!(d2, env2.join("ck"));
    }

    #[test]
    fn the_flag_wins_and_must_be_a_checkpoint() {
        let s = saber_test_scratch::ScratchDir::new("hf2-flag");
        let real = s.path().join("orig/models/ck");
        touch(&real.join(CKPT_MARKER));
        let pack = tree(&s.path().join("r"), &real, true);
        let mine = s.path().join("mine");
        assert!(resolve_ckpt(&pack, &real, Some(&mine), None).is_err());
        touch(&mine.join(CKPT_MARKER));
        let (d, from) = resolve_ckpt(&pack, &real, Some(&mine), None).unwrap();
        assert_eq!((d, from), (mine, CkptFrom::Flag));
    }

    #[test]
    fn nothing_found_is_refused_by_name() {
        let s = saber_test_scratch::ScratchDir::new("hf2-none");
        let rec = Path::new("/no/such/hf2/models/ck");
        let pack = tree(&s.path().join("r"), rec, false);
        let e = resolve_ckpt(&pack, rec, None, None)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("--fn-ckpt-dir") && e.contains("/no/such/hf2/models/ck"),
            "{e}"
        );
    }

    #[test]
    fn model_files_come_from_the_pack_base_when_shipped() {
        let s = saber_test_scratch::ScratchDir::new("hf2-files");
        let pack = s.path().join("pk");
        let ck = s.path().join("ck");
        std::fs::create_dir_all(&pack).unwrap();
        assert_eq!(
            model_file(&pack, &ck, "config.json"),
            ck.join("config.json")
        );
        assert_eq!(model_files_dir(&pack, &ck), ck);
        touch(&pack.join("base/config.json"));
        assert_eq!(
            model_file(&pack, &ck, "config.json"),
            pack.join("base/config.json")
        );
        // the serve plane switches only when every file is shipped
        assert_eq!(model_files_dir(&pack, &ck), ck);
        for n in MODEL_FILES {
            touch(&pack.join("base").join(n));
        }
        assert_eq!(model_files_dir(&pack, &ck), pack.join("base"));
    }

    #[test]
    fn candidates_are_longest_suffix_first() {
        let c = rebase_candidates(Path::new("/mnt/nvme/models/ck"), &[PathBuf::from("/r")]);
        assert_eq!(
            c,
            vec![
                PathBuf::from("/r/nvme/models/ck"),
                PathBuf::from("/r/models/ck"),
                PathBuf::from("/r/ck")
            ]
        );
    }
}
