//! The model catalog (`tools/models.json`, embedded at build time; `[downloads] catalog` names
//! another file): what the setup wizard offers, where each component lands, and the hardware
//! fit rules. `tools/saber-setup` (the terminal wizard) reads the same file.

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::config::{Config, FlagValue, Preset, Runtime};
use crate::system::SystemReport;

/// The catalog this binary was built with.
pub const EMBEDDED: &str = include_str!("../../../tools/models.json");

/// The catalog versions this build reads.
pub const CATALOG_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Catalog {
    pub version: u32,
    #[serde(default)]
    pub about: String,
    /// The Hugging Face endpoint `hf_repo` sources resolve against.
    #[serde(default = "default_hf")]
    pub hf_endpoint: String,
    pub models: Vec<Model>,
}

fn default_hf() -> String {
    "https://huggingface.co".into()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub title: String,
    /// `27B`, `Flash-Next` or `mock`.
    pub kind: String,
    pub preset: Preset,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub download_gb: f64,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub fit: FitRules,
    #[serde(default)]
    pub components: Vec<Component>,
    // the terminal wizard's fields (kept so both tools read one file)
    #[serde(default)]
    pub launcher: String,
    #[serde(default)]
    pub container: String,
    #[serde(default)]
    pub packs: Vec<String>,
    #[serde(default)]
    pub extra_files: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FitRules {
    #[serde(default)]
    pub min_gpus: u32,
    /// Accepted `gfx` name prefixes (`gfx12`). Empty = any.
    #[serde(default)]
    pub arch_prefixes: Vec<String>,
    #[serde(default)]
    pub min_vram_gib: f64,
    /// Tensor-parallel widths the hosted packs cover.
    #[serde(default)]
    pub tp: Vec<u32>,
    #[serde(default)]
    pub min_host_ram_gib: f64,
    #[serde(default)]
    pub min_disk_gib: f64,
}

/// Where a component lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dest {
    /// `<nvme>/saber-packs`
    Packs,
    /// `<nvme>/models`
    Models,
    /// `<repo>/.cache`
    RepoCache,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    /// One directory named `name` (renamed into place whole).
    #[default]
    Dir,
    /// Loose files into the destination root (each renamed into place).
    Files,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Source {
    /// `owner/repo` on the Hub.
    #[serde(default)]
    pub hf_repo: String,
    /// The repo path the component's files live under (`""` = the repo root).
    #[serde(default)]
    pub path: String,
    #[serde(default = "default_rev")]
    pub revision: String,
}

fn default_rev() -> String {
    "main".into()
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FileSpec {
    /// Relative to the component (a `files` layout: the file's name in the destination root).
    pub path: String,
    /// The file's path in the source repo when it differs from `path` (relative to the
    /// source's `path`; empty = `path`): a download saved under another name.
    #[serde(default)]
    pub from: String,
    /// An explicit URL (else the source's repo path + `from`, or + `path`).
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub sha256: Option<String>,
    /// The config path this file feeds (`tokenizer` / `chat_template`).
    #[serde(default)]
    pub config_path: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Component {
    pub id: String,
    #[serde(default)]
    pub label: String,
    pub dest: Dest,
    #[serde(default)]
    pub layout: Layout,
    /// The directory name (a `dir` layout).
    #[serde(default)]
    pub name: String,
    /// Local names that count as this component installed (the maintainers' own names).
    #[serde(default)]
    pub aliases: Vec<String>,
    /// The engine flag that names this pack (`--pack`, `--engine-draft-pack`).
    #[serde(default)]
    pub flag: String,
    #[serde(default)]
    pub approx_bytes: u64,
    #[serde(default)]
    pub source: Source,
    /// An explicit file list (else the Hub listing of `source.path`).
    #[serde(default)]
    pub files: Vec<FileSpec>,
    /// Files that must start with the `SBRPACK1` magic (docs/abi/PACK.md).
    #[serde(default)]
    pub pack_magic: Vec<String>,
    /// Files that must exist and be non-empty.
    #[serde(default)]
    pub required: Vec<String>,
}

/// The magic every pack file starts with.
pub const PACK_MAGIC: &[u8; 8] = b"SBRPACK1";

impl Catalog {
    pub fn parse(text: &str) -> anyhow::Result<Catalog> {
        let c: Catalog = serde_json::from_str(text).context("parsing the model catalog")?;
        if c.version > CATALOG_VERSION {
            anyhow::bail!(
                "the model catalog is version {}; this build reads up to {CATALOG_VERSION}",
                c.version
            );
        }
        let mut ids = std::collections::BTreeSet::new();
        for m in &c.models {
            if m.id.is_empty() || !ids.insert(m.id.as_str()) {
                anyhow::bail!("the model catalog has an empty or duplicate id `{}`", m.id);
            }
            let mut cids = std::collections::BTreeSet::new();
            for comp in &m.components {
                if !cids.insert(comp.id.as_str()) {
                    anyhow::bail!("model `{}`: duplicate component `{}`", m.id, comp.id);
                }
                if comp.layout == Layout::Dir && !safe_name(&comp.name) {
                    anyhow::bail!(
                        "model `{}` component `{}`: `{}` is not a plain directory name",
                        m.id,
                        comp.id,
                        comp.name
                    );
                }
                if comp.layout == Layout::Files && comp.files.is_empty() {
                    anyhow::bail!(
                        "model `{}` component `{}`: a files layout lists its files",
                        m.id,
                        comp.id
                    );
                }
                for f in &comp.files {
                    if !safe_rel(&f.path) || (!f.from.is_empty() && !safe_rel(&f.from)) {
                        anyhow::bail!("model `{}`: unsafe file path `{}`", m.id, f.path);
                    }
                    if f.url.is_empty() && comp.source.hf_repo.is_empty() {
                        anyhow::bail!(
                            "model `{}` component `{}`: `{}` has no url and no hf_repo",
                            m.id,
                            comp.id,
                            f.path
                        );
                    }
                }
                if comp.files.is_empty() && comp.source.hf_repo.is_empty() {
                    anyhow::bail!(
                        "model `{}` component `{}`: no source (hf_repo or files)",
                        m.id,
                        comp.id
                    );
                }
            }
        }
        Ok(c)
    }

    #[must_use]
    pub fn embedded() -> Catalog {
        Catalog::parse(EMBEDDED).expect("tools/models.json parses (a unit test checks it)")
    }

    /// `path` (when non-empty and readable), else the embedded catalog.
    pub fn load(path: &Path) -> anyhow::Result<Catalog> {
        if path.as_os_str().is_empty() {
            return Ok(Catalog::embedded());
        }
        let t = std::fs::read_to_string(path)
            .with_context(|| format!("reading the model catalog {}", path.display()))?;
        Catalog::parse(&t).with_context(|| format!("in {}", path.display()))
    }

    #[must_use]
    pub fn model(&self, id: &str) -> Option<&Model> {
        self.models.iter().find(|m| m.id == id)
    }
}

/// One path segment: no separators, not `.`/`..`, not hidden.
#[must_use]
pub fn safe_name(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('.')
        && !s.contains('/')
        && !s.contains('\\')
        && !s.contains('\0')
}

/// A relative path of safe segments.
#[must_use]
pub fn safe_rel(s: &str) -> bool {
    !s.is_empty() && !s.starts_with('/') && s.split('/').all(safe_name)
}

/// The destination roots for a config.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Roots {
    pub packs: PathBuf,
    pub models: PathBuf,
    pub repo_cache: PathBuf,
}

impl Roots {
    #[must_use]
    pub fn for_config(cfg: &Config, home: &Path) -> Roots {
        let p = cfg.engine.paths.resolved(home);
        Roots {
            packs: p.nvme.join("saber-packs"),
            models: p.nvme.join("models"),
            repo_cache: p.repo.join(".cache"),
        }
    }

    #[must_use]
    pub fn root(&self, d: Dest) -> &Path {
        match d {
            Dest::Packs => &self.packs,
            Dest::Models => &self.models,
            Dest::RepoCache => &self.repo_cache,
        }
    }
}

impl Component {
    /// The installed location: the first of `name` and its aliases that exists (a `dir`
    /// layout), else `None`. A `files` layout is installed when every file exists.
    #[must_use]
    pub fn installed_at(&self, roots: &Roots) -> Option<PathBuf> {
        let root = roots.root(self.dest);
        match self.layout {
            Layout::Dir => std::iter::once(&self.name)
                .chain(&self.aliases)
                .map(|n| root.join(n))
                .find(|p| p.is_dir()),
            Layout::Files => self
                .files
                .iter()
                .all(|f| root.join(&f.path).is_file())
                .then(|| root.to_path_buf()),
        }
    }

    /// Where a fresh download lands (`dir`: `<root>/<name>`; `files`: the root).
    #[must_use]
    pub fn target(&self, roots: &Roots) -> PathBuf {
        let root = roots.root(self.dest);
        match self.layout {
            Layout::Dir => root.join(&self.name),
            Layout::Files => root.to_path_buf(),
        }
    }

    /// The staging directory a download fills (same file system as the target, hidden from
    /// the packs listing by its dot).
    #[must_use]
    pub fn staging(&self, roots: &Roots, model: &str) -> PathBuf {
        let root = roots.root(self.dest);
        let tag = match self.layout {
            Layout::Dir => self.name.clone(),
            Layout::Files => format!("{model}-{}", self.id),
        };
        root.join(format!(".saber-download-{tag}"))
    }

    /// The post-download checks (`pack_magic`, `required`) on `dir`: the problems, if any.
    #[must_use]
    pub fn verify_tree(&self, dir: &Path) -> Vec<String> {
        let mut bad = Vec::new();
        for f in &self.pack_magic {
            let p = dir.join(f);
            let ok = std::fs::File::open(&p)
                .and_then(|mut h| {
                    use std::io::Read;
                    let mut b = [0u8; 8];
                    h.read_exact(&mut b).map(|()| b)
                })
                .is_ok_and(|b| &b == PACK_MAGIC);
            if !ok {
                bad.push(format!("{f} is missing or is not a Saber pack file"));
            }
        }
        for f in &self.required {
            if !std::fs::metadata(dir.join(f)).is_ok_and(|m| m.len() > 0) {
                bad.push(format!("{f} is missing or empty"));
            }
        }
        bad
    }
}

/// How well a model fits the probed machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Verdict {
    Fits,
    /// Runs, with a caveat.
    Tight,
    NoFit,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Fit {
    pub verdict: Verdict,
    /// Plain-English reasons (the failing or tight rules; empty = fits cleanly).
    pub reasons: Vec<String>,
}

/// Check `rules` against the probe. Disk is checked against the free space where the model
/// would land (`free_disk_bytes`), less what is already installed (`installed_bytes`).
#[must_use]
pub fn fit(
    rules: &FitRules,
    sys: &SystemReport,
    free_disk_bytes: Option<u64>,
    need_bytes: u64,
) -> Fit {
    const GIB: f64 = (1u64 << 30) as f64;
    let mut no = Vec::new();
    let mut tight = Vec::new();
    if rules.min_gpus > 0 {
        let ok: Vec<_> = sys
            .gpus
            .iter()
            .filter(|g| !g.integrated)
            .filter(|g| {
                rules.arch_prefixes.is_empty()
                    || rules
                        .arch_prefixes
                        .iter()
                        .any(|p| g.gfx.starts_with(p.as_str()))
            })
            .collect();
        if (ok.len() as u32) < rules.min_gpus {
            no.push(format!(
                "needs {} supported GPUs ({}), found {}",
                rules.min_gpus,
                if rules.arch_prefixes.is_empty() {
                    "any".to_string()
                } else {
                    rules
                        .arch_prefixes
                        .iter()
                        .map(|p| format!("{p}xx"))
                        .collect::<Vec<_>>()
                        .join(" / ")
                },
                ok.len()
            ));
        } else if rules.min_vram_gib > 0.0 {
            let small: Vec<_> = ok
                .iter()
                .filter(|g| {
                    g.vram_bytes
                        .is_some_and(|v| (v as f64) / GIB < rules.min_vram_gib)
                })
                .collect();
            if ok.len() - small.len() < rules.min_gpus as usize {
                no.push(format!(
                    "needs {:.0} GiB of VRAM on each of {} cards",
                    rules.min_vram_gib, rules.min_gpus
                ));
            }
        }
        if !rules.tp.is_empty()
            && !rules
                .tp
                .contains(&(ok.len().min(rules.min_gpus as usize) as u32))
        {
            tight.push(format!(
                "the hosted packs are built for TP={}",
                rules
                    .tp
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join("/")
            ));
        }
    }
    if rules.min_host_ram_gib > 0.0 {
        match sys.host_ram_bytes {
            Some(r) if (r as f64) / GIB + 0.5 < rules.min_host_ram_gib => no.push(format!(
                "needs {:.0} GiB of host RAM, this machine has {:.0} GiB",
                rules.min_host_ram_gib,
                r as f64 / GIB
            )),
            None => tight.push("host RAM could not be read".into()),
            _ => {}
        }
    }
    if need_bytes > 0 {
        match free_disk_bytes {
            Some(f) if f < need_bytes => no.push(format!(
                "needs {} of free disk for the download, {} free",
                crate::downloads::human(need_bytes),
                crate::downloads::human(f)
            )),
            Some(f) if (f as f64) < need_bytes as f64 * 1.1 => tight.push(format!(
                "the download leaves under 10 % of the disk free ({} free)",
                crate::downloads::human(f)
            )),
            _ => {}
        }
    }
    let verdict = if !no.is_empty() {
        Verdict::NoFit
    } else if !tight.is_empty() {
        Verdict::Tight
    } else {
        Verdict::Fits
    };
    no.extend(tight);
    Fit {
        verdict,
        reasons: no,
    }
}

/// The config that serves `model` from its installed components: the model's preset and
/// runtime, the pack flags pointing at what is installed (only when they differ from the
/// preset's own), the tokenizer / chat-template paths its files feed. Everything else in `cfg`
/// (paths, proxy, usage, overrides that are not pack flags) is kept.
pub fn config_for(cfg: &Config, model: &Model, roots: &Roots) -> anyhow::Result<Config> {
    let mut c = cfg.clone();
    if c.engine.preset != model.preset {
        // a different subcommand: the old overrides belong to the other flag set
        c.engine.flags.clear();
        c.engine.extra_args.clear();
    }
    c.engine.preset = model.preset;
    c.engine.runtime = if model.preset == Preset::Mock {
        Runtime::Direct
    } else {
        Runtime::Container
    };
    if model.preset == Preset::Mock {
        let m = Config::mock(c.engine.port);
        if c.engine.paths.tokenizer == crate::config::Paths::default().tokenizer {
            c.engine.paths.tokenizer = m.engine.paths.tokenizer;
        }
    }
    for comp in &model.components {
        let Some(at) = comp.installed_at(roots) else {
            anyhow::bail!(
                "{} ({}) is not installed yet: download it first",
                if comp.label.is_empty() {
                    &comp.id
                } else {
                    &comp.label
                },
                model.name
            );
        };
        if !comp.flag.is_empty() {
            let arg = crate::apply::pack_arg(model.preset, &at);
            c.engine
                .flags
                .insert(comp.flag.clone(), FlagValue::Str(arg));
        }
        for f in &comp.files {
            let p = at.join(&f.path);
            match f.config_path.as_str() {
                "tokenizer" => c.engine.paths.tokenizer = p,
                "chat_template" => c.engine.paths.chat_template = p,
                _ => {}
            }
        }
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::{GpuInfo, SystemReport};

    fn gpu(gfx: &str, vram_gib: u64) -> GpuInfo {
        GpuInfo {
            gfx: gfx.into(),
            supported: gfx.starts_with("gfx12"),
            vram_bytes: Some(vram_gib << 30),
            ..GpuInfo::default()
        }
    }

    fn sys(gpus: Vec<GpuInfo>, ram_gib: u64) -> SystemReport {
        SystemReport {
            gpus,
            host_ram_bytes: Some(ram_gib << 30),
            ..SystemReport::default()
        }
    }

    #[test]
    fn the_embedded_catalog_parses_and_names_both_models() {
        let c = Catalog::embedded();
        assert_eq!(c.version, 1);
        let ids: Vec<_> = c.models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["27b", "flashnext", "mock"]);
        let fnx = c.model("flashnext").unwrap();
        assert_eq!(fnx.preset, Preset::Flashnext);
        assert_eq!(
            fnx.components[0].aliases,
            ["qwen38-flashnext-quark-ep2-trunk8"]
        );
        // the terminal wizard's fields are there too
        assert_eq!(fnx.launcher, "tools/serve_flashnext.sh");
        assert_eq!(c.model("27b").unwrap().packs.len(), 2);
        assert!(c.model("mock").unwrap().components.is_empty());
    }

    #[test]
    fn bad_catalogs_are_refused_by_name() {
        let base = |comp: &str| {
            format!(
                r#"{{"version":1,"models":[{{"id":"a","name":"A","kind":"27B","preset":"qwen27b","components":[{comp}]}}]}}"#
            )
        };
        assert!(Catalog::parse(&base("")).is_ok());
        let e = Catalog::parse(&base(
            r#"{"id":"t","dest":"packs","name":"../x","source":{"hf_repo":"o/r"}}"#,
        ))
        .unwrap_err();
        assert!(format!("{e:#}").contains("plain directory name"), "{e:#}");
        let e = Catalog::parse(&base(r#"{"id":"t","dest":"packs","name":"x"}"#)).unwrap_err();
        assert!(format!("{e:#}").contains("no source"), "{e:#}");
        let e = Catalog::parse(&base(
            r#"{"id":"t","dest":"packs","name":"x","files":[{"path":"/etc/passwd","url":"http://h/x"}]}"#,
        ))
        .unwrap_err();
        assert!(format!("{e:#}").contains("unsafe"), "{e:#}");
        assert!(Catalog::parse(r#"{"version":2,"models":[]}"#).is_err());
        assert!(Catalog::parse(
            r#"{"version":1,"models":[{"id":"a","name":"A","kind":"x","preset":"mock"},{"id":"a","name":"B","kind":"x","preset":"mock"}]}"#
        )
        .is_err());
    }

    #[test]
    fn fit_rules() {
        let c = Catalog::embedded();
        let r27 = &c.model("27b").unwrap().fit;
        let rfn = &c.model("flashnext").unwrap().fit;
        let box_ = sys(
            vec![gpu("gfx1201", 32), gpu("gfx1201", 32), gpu("gfx1036", 2)],
            60,
        );
        // a two-card gfx12 host: both fit (the iGPU is ignored only when marked integrated: it is not gfx12)
        assert_eq!(
            fit(r27, &box_, Some(500_000_000_000), 26_000_000_000).verdict,
            Verdict::Fits
        );
        assert_eq!(
            fit(rfn, &box_, Some(500_000_000_000), 240_000_000_000).verdict,
            Verdict::Fits
        );
        // one card
        let one = sys(vec![gpu("gfx1201", 32)], 64);
        let f = fit(r27, &one, None, 0);
        assert_eq!(f.verdict, Verdict::NoFit);
        assert!(f.reasons[0].contains("needs 2 supported GPUs"), "{f:?}");
        // the wrong architecture
        let old = sys(vec![gpu("gfx1100", 24), gpu("gfx1100", 24)], 64);
        assert_eq!(fit(r27, &old, None, 0).verdict, Verdict::NoFit);
        // small cards
        let small = sys(vec![gpu("gfx1200", 16), gpu("gfx1200", 16)], 64);
        let f = fit(r27, &small, None, 0);
        assert!(f.reasons.iter().any(|r| r.contains("GiB of VRAM")), "{f:?}");
        // not enough RAM for Flash-Next, enough for the 27B
        let lowram = sys(vec![gpu("gfx1201", 32), gpu("gfx1201", 32)], 32);
        assert_eq!(fit(r27, &lowram, None, 0).verdict, Verdict::Fits);
        let f = fit(rfn, &lowram, None, 0);
        assert_eq!(f.verdict, Verdict::NoFit);
        assert!(f.reasons[0].contains("host RAM"), "{f:?}");
        // disk: short, tight
        let f = fit(r27, &box_, Some(10_000_000_000), 26_000_000_000);
        assert_eq!(f.verdict, Verdict::NoFit);
        assert!(f.reasons[0].contains("free disk"), "{f:?}");
        assert_eq!(
            fit(r27, &box_, Some(27_000_000_000), 26_000_000_000).verdict,
            Verdict::Tight
        );
        // the mock fits anything
        let none = sys(vec![], 8);
        assert_eq!(
            fit(&c.model("mock").unwrap().fit, &none, None, 0).verdict,
            Verdict::Fits
        );
    }

    #[test]
    fn install_detection_aliases_and_config_for() {
        let dir = saber_test_scratch::ScratchDir::new("console-catalog");
        let mut cfg = Config::default();
        cfg.engine.paths.nvme = dir.path().join("nvme");
        cfg.engine.paths.repo = dir.path().join("repo");
        let roots = Roots::for_config(&cfg, Path::new("/h"));
        let c = Catalog::embedded();
        let fnx = c.model("flashnext").unwrap();
        assert!(fnx.components[0].installed_at(&roots).is_none());
        assert!(config_for(&cfg, fnx, &roots).is_err());
        // the maintainers' local name counts
        let alias = roots.packs.join("qwen38-flashnext-quark-ep2-trunk8");
        std::fs::create_dir_all(&alias).unwrap();
        std::fs::create_dir_all(roots.models.join("Qwen3.8-Flash-Next-Quark-MXFP4")).unwrap();
        assert_eq!(
            fnx.components[0].installed_at(&roots).as_deref(),
            Some(alias.as_path())
        );
        let out = config_for(&cfg, fnx, &roots).unwrap();
        assert_eq!(
            out.engine.flags.get("--pack"),
            Some(&FlagValue::Str(alias.display().to_string()))
        );
        // the 27B: container paths for the packs, the tokenizer files into the paths
        let m27 = c.model("27b").unwrap();
        for n in ["qwen38-27b-macaq7-bc3", "qwen38-dflash2-drafter-quarkhead"] {
            std::fs::create_dir_all(roots.packs.join(n)).unwrap();
        }
        std::fs::create_dir_all(&roots.repo_cache).unwrap();
        assert!(config_for(&cfg, m27, &roots).is_err());
        for f in ["quark27b.tokenizer.json", "template_corpus.jinja"] {
            std::fs::write(roots.repo_cache.join(f), b"x").unwrap();
        }
        let mut with_old = cfg.clone();
        with_old
            .engine
            .flags
            .insert("--fn-max-ctx".into(), FlagValue::Int(4096));
        let out = config_for(&with_old, m27, &roots).unwrap();
        assert_eq!(out.engine.preset, Preset::Qwen27b);
        assert!(!out.engine.flags.contains_key("--fn-max-ctx"));
        assert_eq!(
            out.engine.flags.get("--engine-draft-pack"),
            Some(&FlagValue::Str(
                "/packs/qwen38-dflash2-drafter-quarkhead".into()
            ))
        );
        assert_eq!(
            out.engine.paths.tokenizer,
            roots.repo_cache.join("quark27b.tokenizer.json")
        );
        out.validate().unwrap();
        // the mock: a direct child, no download
        let out = config_for(&cfg, c.model("mock").unwrap(), &roots).unwrap();
        assert_eq!(out.engine.runtime, Runtime::Direct);
        out.validate().unwrap();
    }

    #[test]
    fn verify_tree_checks_magic_and_required() {
        let dir = saber_test_scratch::ScratchDir::new("console-catalog-verify");
        let comp = Component {
            id: "t".into(),
            label: String::new(),
            dest: Dest::Packs,
            layout: Layout::Dir,
            name: "p".into(),
            aliases: vec![],
            flag: String::new(),
            approx_bytes: 0,
            source: Source::default(),
            files: vec![],
            pack_magic: vec!["rank0.pack".into()],
            required: vec!["manifest.json".into()],
        };
        let d = dir.path();
        assert_eq!(comp.verify_tree(d).len(), 2);
        std::fs::write(d.join("rank0.pack"), b"NOTAPACKxx").unwrap();
        std::fs::write(d.join("manifest.json"), b"{}").unwrap();
        assert_eq!(
            comp.verify_tree(d),
            ["rank0.pack is missing or is not a Saber pack file"]
        );
        std::fs::write(d.join("rank0.pack"), b"SBRPACK1rest").unwrap();
        assert!(comp.verify_tree(d).is_empty());
    }
}
