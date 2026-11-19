//! The local pack cache: where a model's packs live on disk, whether they are there, and whether
//! they are what the registry says.
//!
//! Layout (a partial mirror of the Hugging Face repo, so the cache itself can serve as a mirror):
//!
//! ```text
//! <root>/<owner>/<repo>/<tp path>/manifest.json, rank0.pack[, rank1.pack], .saber-pull.json
//! <root>/<owner>/<repo>/tokenizer.json, <chat template>      (the entry's `files`)
//! ```
//!
//! `<root>` is `--packs-dir`, else `$SABER_PACKS`, else `$XDG_CACHE_HOME/saber/packs`, else
//! `~/.cache/saber/packs`. A pack an older tool put directly under `<root>` by one of the entry's
//! `legacy_dirs` names is found and served in place (and never deleted by `saber models rm`).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::registry::Entry;

/// The marker a verified pull leaves in a TP folder.
pub const MARKER: &str = ".saber-pull.json";

/// The process environment the front door reads (and nothing else): paths and the HF token.
#[derive(Clone, Debug, Default)]
pub struct Env {
    pub packs_root: PathBuf,
    /// Where the packs root came from, for messages (`--packs-dir`, `SABER_PACKS`, `default`).
    pub packs_root_from: &'static str,
    /// The user registry file (it may not exist).
    pub user_registry: PathBuf,
    /// The Hugging Face token file (`$HF_HOME/token`, else `~/.cache/huggingface/token`).
    pub hf_token_file: PathBuf,
    /// `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`, if set.
    pub hf_token_env: Option<String>,
    /// `HF_ENDPOINT`, if set (replaces the registry's default Hugging Face endpoint).
    pub hf_endpoint: Option<String>,
}

fn var(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.trim().is_empty())
}

impl Env {
    /// The real environment; `packs_dir` = the `--packs-dir` flag.
    #[must_use]
    pub fn from_process(packs_dir: Option<PathBuf>) -> Env {
        let home = var("HOME").map_or_else(|| PathBuf::from("/"), PathBuf::from);
        let cache = var("XDG_CACHE_HOME").map_or_else(|| home.join(".cache"), PathBuf::from);
        let config = var("XDG_CONFIG_HOME").map_or_else(|| home.join(".config"), PathBuf::from);
        let (packs_root, from) = match (packs_dir, var("SABER_PACKS")) {
            (Some(p), _) => (p, "--packs-dir"),
            (None, Some(p)) => (PathBuf::from(p), "SABER_PACKS"),
            (None, None) => (cache.join("saber/packs"), "default"),
        };
        let hf_home = var("HF_HOME").map_or_else(|| cache.join("huggingface"), PathBuf::from);
        Env {
            packs_root,
            packs_root_from: from,
            user_registry: config.join("saber/models.toml"),
            hf_token_file: hf_home.join("token"),
            hf_token_env: var("HF_TOKEN").or_else(|| var("HUGGING_FACE_HUB_TOKEN")),
            hf_endpoint: var("HF_ENDPOINT"),
        }
    }

    /// The token to send, if the user has one. Never printed.
    #[must_use]
    pub fn hf_token(&self) -> Option<String> {
        self.hf_token_env.clone().or_else(|| {
            std::fs::read_to_string(&self.hf_token_file)
                .ok()
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
        })
    }
}

/// `<root>/<owner>/<repo>`.
#[must_use]
pub fn repo_dir(root: &Path, e: &Entry) -> PathBuf {
    let (o, n) = e.repo_parts();
    root.join(o).join(n)
}

/// `<root>/<owner>/<repo>/<tp path>`.
#[must_use]
pub fn tp_dir(root: &Path, e: &Entry, tp: u32) -> Option<PathBuf> {
    e.at(tp).map(|t| repo_dir(root, e).join(&t.path))
}

/// What is on disk for one entry at one TP.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    /// Complete. `verified` = a pull checked every file's hash; `legacy` = an older tool's
    /// folder under the root (served in place, not managed).
    Ready {
        dir: PathBuf,
        verified: bool,
        legacy: bool,
    },
    /// Started but not complete (a pull resumes it).
    Partial {
        dir: PathBuf,
        have: u64,
    },
    Missing,
}

impl Status {
    #[must_use]
    pub fn ready(&self) -> bool {
        matches!(self, Status::Ready { .. })
    }
    #[must_use]
    pub fn dir(&self) -> Option<&Path> {
        match self {
            Status::Ready { dir, .. } | Status::Partial { dir, .. } => Some(dir),
            Status::Missing => None,
        }
    }
}

/// Where `e`'s TP=`tp` pack is, if anywhere.
#[must_use]
pub fn locate(root: &Path, e: &Entry, tp: u32) -> Status {
    let Some(spec) = e.at(tp) else {
        return Status::Missing;
    };
    let dir = repo_dir(root, e).join(&spec.path);
    if dir.join(MARKER).is_file() && check_pack(&dir, tp, None).is_ok() {
        return Status::Ready {
            dir,
            verified: true,
            legacy: false,
        };
    }
    for l in &spec.legacy_dirs {
        let d = root.join(l);
        if check_pack(&d, tp, e.requires_tag.as_deref()).is_ok() {
            return Status::Ready {
                dir: d,
                verified: false,
                legacy: true,
            };
        }
    }
    if dir.is_dir() {
        let partial = std::fs::read_dir(&dir)
            .map(|rd| {
                rd.flatten()
                    .any(|f| f.file_name().to_string_lossy().ends_with(".part"))
            })
            .unwrap_or(false);
        if !partial && check_pack(&dir, tp, e.requires_tag.as_deref()).is_ok() {
            // complete, put there by hand (e.g. `hf download --local-dir`): served, not verified
            return Status::Ready {
                dir,
                verified: false,
                legacy: false,
            };
        }
        let have = dir_bytes(&dir);
        if have > 0 {
            return Status::Partial { dir, have };
        }
    }
    Status::Missing
}

/// The shared files (`files`) of `e` that are not on disk yet.
#[must_use]
pub fn missing_files(root: &Path, e: &Entry) -> Vec<String> {
    let rd = repo_dir(root, e);
    e.files
        .iter()
        .filter(|f| {
            !std::fs::metadata(rd.join(f))
                .map(|m| m.is_file() && m.len() > 0)
                .unwrap_or(false)
        })
        .cloned()
        .collect()
}

/// Total bytes of the regular files directly in `dir` (and `.part` files).
#[must_use]
pub fn dir_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter_map(|f| f.metadata().ok())
                .filter(std::fs::Metadata::is_file)
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// What a pack's manifest says about it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackInfo {
    pub tp_size: Option<u32>,
    /// The versioned format tags among its transforms (`macaq_iu8/1`).
    pub tags: Vec<String>,
}

/// Why a folder is not a usable pack.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackProblem {
    NoManifest,
    BadManifest(String),
    WrongTp { want: u32, have: u32 },
    MissingRank(String),
    BadMagic(String),
    WrongTag { want: String, have: Vec<String> },
}

impl std::fmt::Display for PackProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PackProblem::NoManifest => write!(f, "it has no manifest.json"),
            PackProblem::BadManifest(e) => write!(f, "its manifest.json does not parse ({e})"),
            PackProblem::WrongTp { want, have } => write!(
                f,
                "it is a TP={have} pack ({have} rank file{}), not TP={want}",
                if *have == 1 { "" } else { "s" }
            ),
            PackProblem::MissingRank(r) => write!(f, "{r} is missing"),
            PackProblem::BadMagic(r) => {
                write!(
                    f,
                    "{r} is not a Saber pack file (no SBRPACK1 header: truncated or corrupt)"
                )
            }
            PackProblem::WrongTag { want, have } => write!(
                f,
                "its manifest has no `{want}` format tag (it has: {})",
                if have.is_empty() {
                    "none".to_string()
                } else {
                    have.join(", ")
                }
            ),
        }
    }
}

fn collect_transforms(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::Object(m) => {
            for (k, x) in m {
                if k == "transforms" {
                    if let Some(a) = x.as_array() {
                        out.extend(a.iter().filter_map(|s| s.as_str().map(str::to_string)));
                    }
                } else {
                    collect_transforms(x, out);
                }
            }
        }
        serde_json::Value::Array(a) => a.iter().for_each(|x| collect_transforms(x, out)),
        _ => {}
    }
}

/// Is `dir` a TP=`tp` Saber pack (with `tag` in its manifest, when given)? Reads the manifest and
/// eight bytes of each rank file; never the weights.
pub fn check_pack(dir: &Path, tp: u32, tag: Option<&str>) -> Result<PackInfo, PackProblem> {
    let m = dir.join("manifest.json");
    let text = std::fs::read_to_string(&m).map_err(|_| PackProblem::NoManifest)?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| PackProblem::BadManifest(e.to_string()))?;
    let tp_size = v["tp_size"].as_u64().map(|x| x as u32);
    if let Some(have) = tp_size {
        if have != tp {
            return Err(PackProblem::WrongTp { want: tp, have });
        }
    }
    for r in 0..tp {
        let name = format!("rank{r}.pack");
        let p = dir.join(&name);
        let mut head = [0u8; 8];
        let ok = std::fs::File::open(&p)
            .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut head))
            .is_ok();
        if !p.is_file() {
            return Err(PackProblem::MissingRank(name));
        }
        if !ok || &head != b"SBRPACK1" {
            return Err(PackProblem::BadMagic(name));
        }
    }
    if tp == 1 && dir.join("rank1.pack").exists() {
        return Err(PackProblem::WrongTp { want: 1, have: 2 });
    }
    let mut tr = Vec::new();
    collect_transforms(&v, &mut tr);
    let tags: Vec<String> = tr
        .iter()
        .filter_map(|s| s.split_whitespace().next())
        .filter(|w| {
            w.split_once('/')
                .is_some_and(|(n, ver)| !n.is_empty() && ver.parse::<u32>().is_ok())
        })
        .map(str::to_string)
        .collect();
    if let Some(t) = tag {
        if !tags.iter().any(|x| x == t) {
            return Err(PackProblem::WrongTag {
                want: t.to_string(),
                have: tags,
            });
        }
    }
    Ok(PackInfo { tp_size, tags })
}

/// One file a pull verified.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarkedFile {
    pub path: String,
    pub bytes: u64,
    #[serde(default)]
    pub sha256: Option<String>,
}

/// The `.saber-pull.json` a verified pull writes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Marker {
    pub repo: String,
    pub revision: String,
    pub path: String,
    pub tp: u32,
    pub files: Vec<MarkedFile>,
    pub verified_unix: u64,
}

pub fn write_marker(dir: &Path, m: &Marker) -> std::io::Result<()> {
    let tmp = dir.join(format!("{MARKER}.tmp"));
    std::fs::write(&tmp, serde_json::to_string_pretty(m).unwrap_or_default())?;
    std::fs::rename(tmp, dir.join(MARKER))
}

#[cfg(test)]
pub mod testkit {
    //! A fake pack on disk.
    use std::path::Path;

    pub fn fake_pack(dir: &Path, tp: u32, tags: &[&str]) {
        std::fs::create_dir_all(dir).unwrap();
        let tr: Vec<String> = tags
            .iter()
            .map(|t| format!("\"{t} extra words\""))
            .collect();
        std::fs::write(
            dir.join("manifest.json"),
            format!(
                r#"{{"magic":"SBRPACK1","tp_size":{tp},"plan":{{"descriptor":{{"transforms":["merge_dim0",{}]}}}}}}"#,
                tr.join(",")
            ),
        )
        .unwrap();
        for r in 0..tp {
            std::fs::write(dir.join(format!("rank{r}.pack")), b"SBRPACK1rest").unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testkit::fake_pack;
    use super::*;
    use crate::models::registry::Registry;
    use saber_test_scratch::ScratchDir;

    #[test]
    fn layout_mirrors_the_repo() {
        let r = Registry::load(None).unwrap();
        let e = r.resolve("w4a8").unwrap();
        let root = Path::new("/c");
        assert_eq!(
            repo_dir(root, e),
            Path::new("/c/Launch80/Qwen3.8-27B-MACaQ-W4A8-SABER")
        );
        assert_eq!(
            tp_dir(root, e, 1).unwrap(),
            Path::new("/c/Launch80/Qwen3.8-27B-MACaQ-W4A8-SABER/tp1")
        );
        let m = r.resolve("mxfp4").unwrap();
        assert_eq!(
            tp_dir(root, m, 2).unwrap(),
            Path::new("/c/Launch80/Qwen3.8-27B-MACaQ-MXFP4-SABER/qwen38-27b-macaq7-bc3")
        );
    }

    #[test]
    fn locate_managed_legacy_partial_missing() {
        let s = ScratchDir::new("models-locate");
        let r = Registry::load(None).unwrap();
        let e = r.resolve("w4a8").unwrap();
        assert_eq!(locate(&s, e, 2), Status::Missing);
        // a legacy flat folder is found in place
        fake_pack(&s.join("qwen38-27b-macaq-w4a8"), 2, &["macaq_iu8/1"]);
        assert_eq!(
            locate(&s, e, 2),
            Status::Ready {
                dir: s.join("qwen38-27b-macaq-w4a8"),
                verified: false,
                legacy: true
            }
        );
        // a legacy folder of another format is not this model's pack
        fake_pack(&s.join("qwen38-27b-macaq-w4a8-tp1"), 1, &["macaq_iu7/1"]);
        assert_eq!(locate(&s, e, 1), Status::Missing);
        std::fs::remove_dir_all(s.join("qwen38-27b-macaq-w4a8-tp1")).unwrap();
        // a managed folder with its marker wins over it
        let d = tp_dir(&s, e, 2).unwrap();
        fake_pack(&d, 2, &["macaq_iu8/1"]);
        write_marker(
            &d,
            &Marker {
                repo: e.repo.clone(),
                revision: "main".into(),
                path: "tp2".into(),
                tp: 2,
                files: vec![],
                verified_unix: 0,
            },
        )
        .unwrap();
        assert!(matches!(
            locate(&s, e, 2),
            Status::Ready {
                verified: true,
                legacy: false,
                ..
            }
        ));
        // TP=1: a half-downloaded folder is partial
        let d1 = tp_dir(&s, e, 1).unwrap();
        std::fs::create_dir_all(&d1).unwrap();
        std::fs::write(d1.join("rank0.pack.part"), b"SBRP").unwrap();
        assert!(matches!(locate(&s, e, 1), Status::Partial { have: 4, .. }));
    }

    #[test]
    fn pack_checks_name_the_problem() {
        let s = ScratchDir::new("models-check");
        let d = s.join("p");
        assert_eq!(check_pack(&d, 2, None), Err(PackProblem::NoManifest));
        fake_pack(&d, 2, &["macaq_iu8/1"]);
        let info = check_pack(&d, 2, Some("macaq_iu8/1")).unwrap();
        assert_eq!(info.tags, ["macaq_iu8/1"]);
        assert_eq!(
            check_pack(&d, 1, None),
            Err(PackProblem::WrongTp { want: 1, have: 2 })
        );
        assert!(matches!(
            check_pack(&d, 2, Some("macaq_iu8/2")),
            Err(PackProblem::WrongTag { .. })
        ));
        std::fs::write(d.join("rank1.pack"), b"garbage!!").unwrap();
        assert_eq!(
            check_pack(&d, 2, None),
            Err(PackProblem::BadMagic("rank1.pack".into()))
        );
        std::fs::remove_file(d.join("rank1.pack")).unwrap();
        assert_eq!(
            check_pack(&d, 2, None),
            Err(PackProblem::MissingRank("rank1.pack".into()))
        );
    }

    #[test]
    fn env_precedence_for_the_packs_root() {
        // only the explicit flag is testable without touching the process environment
        let e = Env::from_process(Some(PathBuf::from("/flag")));
        assert_eq!(e.packs_root, Path::new("/flag"));
        assert_eq!(e.packs_root_from, "--packs-dir");
    }
}
