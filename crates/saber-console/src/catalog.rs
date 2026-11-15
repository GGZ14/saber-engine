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
    /// `tools/fetch_packs.sh`'s target for this model (the terminal wizard's install step).
    #[serde(default)]
    pub fetch: String,
    /// The single-card (TP=1) form: what differs from the TP=2 entry above (the `-tp1` packs,
    /// their sizes and fit). A model without it is TP=2 only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tp1: Option<Tp1Form>,
}

/// A model's TP=1 form: the fields that differ from its TP=2 entry (absent = the same).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Tp1Form {
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub container: String,
    #[serde(default)]
    pub packs: Vec<String>,
    #[serde(default)]
    pub download_gb: Option<f64>,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub fetch: String,
    #[serde(default)]
    pub fit: Option<FitRules>,
    /// The TP=1 components (the single-card packs; a file shared with TP=2, such as the
    /// tokenizer, is listed again).
    #[serde(default)]
    pub components: Vec<Component>,
}

impl Model {
    /// The TP widths this model's packs are hosted for (`[2]`, or `[1, 2]` with a TP=1 form; the
    /// mock opens no GPU, so it offers no choice).
    #[must_use]
    pub fn tp_choices(&self) -> Vec<u32> {
        if self.tp1.is_some() && self.preset.tp_choices().contains(&1) {
            vec![1, 2]
        } else {
            vec![2]
        }
    }

    /// The model as served at `tp`: the TP=1 form folded over the entry at `tp == 1` (the
    /// components, packs, sizes and fit of the single-card packs), the entry itself otherwise.
    #[must_use]
    pub fn at_tp(&self, tp: u32) -> Model {
        let mut m = self.clone();
        if tp != 1 || self.preset == Preset::Mock {
            return m;
        }
        let Some(t) = &self.tp1 else { return m };
        if !t.description.is_empty() {
            m.description.clone_from(&t.description);
        }
        if !t.container.is_empty() {
            m.container.clone_from(&t.container);
        }
        if !t.packs.is_empty() {
            m.packs.clone_from(&t.packs);
        }
        if let Some(g) = t.download_gb {
            m.download_gb = g;
        }
        if !t.note.is_empty() {
            m.note.clone_from(&t.note);
        }
        if !t.fetch.is_empty() {
            m.fetch.clone_from(&t.fetch);
        }
        if let Some(f) = &t.fit {
            m.fit = f.clone();
        }
        if !t.components.is_empty() {
            m.components.clone_from(&t.components);
        }
        m
    }

    /// The TP this machine should run the model at: TP=2 with two supported cards (or a model
    /// without a TP=1 form), TP=1 with one.
    #[must_use]
    pub fn recommended_tp(&self, supported_gpus: u32) -> u32 {
        if supported_gpus < 2 && self.tp_choices().contains(&1) {
            1
        } else {
            2
        }
    }
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
            if m.tp1.is_some() && !m.preset.tp_choices().contains(&1) {
                anyhow::bail!(
                    "model `{}`: a TP=1 form for the {:?} preset, which serves TP=2 only",
                    m.id,
                    m.preset
                );
            }
            let tp1 = m
                .tp1
                .as_ref()
                .map(|t| t.components.as_slice())
                .unwrap_or(&[]);
            for (form, comps) in [("", m.components.as_slice()), (" (TP=1)", tp1)] {
                let mut cids = std::collections::BTreeSet::new();
                for comp in comps {
                    if !cids.insert(comp.id.as_str()) {
                        anyhow::bail!("model `{}`{form}: duplicate component `{}`", m.id, comp.id);
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
    config_for_tp(cfg, model, cfg.engine.tp, roots)
}

/// [`config_for`] at `tp` (`engine.tp` is set to it): the model's TP=1 form's components at
/// `tp == 1`. A TP the model is not hosted for is refused by name.
pub fn config_for_tp(
    cfg: &Config,
    model: &Model,
    tp: u32,
    roots: &Roots,
) -> anyhow::Result<Config> {
    let tp = if model.preset == Preset::Flashnext {
        2
    } else {
        tp
    };
    if !model.tp_choices().contains(&tp) {
        anyhow::bail!(
            "{} is hosted for TP={} only (no single-card packs)",
            model.name,
            model
                .tp_choices()
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join("/")
        );
    }
    let model = &model.at_tp(tp);
    let mut c = cfg.clone();
    if c.engine.preset != model.preset {
        // a different subcommand: the old overrides belong to the other flag set
        c.engine.flags.clear();
        c.engine.extra_args.clear();
    }
    c.engine.preset = model.preset;
    c.engine.tp = tp;
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

/// Switch `cfg` to `tp` (the Settings page's TP choice): `engine.tp` (and, at TP=1, the card)
/// is set, and a pack flag that names one of `catalog`'s packs at the old TP is moved to the
/// same component at the new TP (the TP=1 form's component of the same id); a pack flag naming
/// the preset's own default of the old TP is dropped (the new TP's preset names its own). Any
/// other pack override is kept (the validator then says if its tp_size does not match).
#[must_use]
pub fn with_tp(cfg: &Config, tp: u32, device: Option<u32>, catalog: &Catalog) -> Config {
    let mut c = cfg.clone();
    let from = c.engine.tp;
    c.engine.tp = tp;
    if let Some(d) = device {
        c.engine.device = d;
    }
    if from == tp || c.engine.preset != Preset::Qwen27b {
        return c;
    }
    let name_of = |v: &FlagValue| -> Option<String> {
        let a = v.arg()?;
        let n = a.rsplit('/').next()?.to_string();
        (!n.is_empty()).then_some(n)
    };
    let stock = |t: u32, flag: &str| -> &'static str {
        match (t, flag) {
            (1, "--pack") => crate::compose::tp1::PACK,
            (1, _) => crate::compose::tp1::DRAFT_PACK,
            (_, "--pack") => "qwen38-27b-macaq7-bc3",
            _ => "qwen38-dflash2-drafter-quarkhead",
        }
    };
    for flag in ["--pack", "--engine-draft-pack"] {
        let Some(name) = c.engine.flags.get(flag).and_then(name_of) else {
            continue;
        };
        if name == stock(from, flag) {
            c.engine.flags.remove(flag);
            continue;
        }
        // the catalog's same component at the other TP
        let twin = catalog
            .models
            .iter()
            .filter(|m| m.preset == Preset::Qwen27b && m.tp1.is_some())
            .find_map(|m| {
                let (a, b) = (m.at_tp(from), m.at_tp(tp));
                let comp = a
                    .components
                    .iter()
                    .find(|x| x.flag == flag && (x.name == name || x.aliases.contains(&name)))?;
                b.components
                    .iter()
                    .find(|y| y.id == comp.id && y.flag == flag)
                    .map(|y| y.name.clone())
            });
        if let Some(t) = twin {
            if t == stock(tp, flag) {
                c.engine.flags.remove(flag);
            } else {
                c.engine
                    .flags
                    .insert(flag.to_string(), FlagValue::Str(format!("/packs/{t}")));
            }
        }
    }
    c
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
    fn the_27b_has_a_single_card_form_and_flashnext_does_not() {
        let c = Catalog::embedded();
        let m27 = c.model("27b").unwrap();
        let fnx = c.model("flashnext").unwrap();
        assert_eq!(m27.tp_choices(), [1, 2]);
        assert_eq!(fnx.tp_choices(), [2]);
        let one = m27.at_tp(1);
        assert_eq!(one.container, "saber-27b-tp1-serve");
        assert_eq!(one.fit.min_gpus, 1);
        assert_eq!(one.fit.tp, [1]);
        let ids: Vec<&str> = one.components.iter().map(|x| x.id.as_str()).collect();
        assert_eq!(ids, ["target", "drafter", "tokenizer"]);
        assert!(one.components[0].name.contains("tp1"));
        assert_eq!(one.components[0].pack_magic, ["rank0.pack"]);
        // the TP=2 entry is unchanged by the fold, and TP=2 of a TP=2-only model is itself
        assert_eq!(m27.at_tp(2), *m27);
        assert_eq!(fnx.at_tp(1), *fnx);
        // the machine's recommendation
        assert_eq!(m27.recommended_tp(1), 1);
        assert_eq!(m27.recommended_tp(2), 2);
        assert_eq!(fnx.recommended_tp(1), 2);
        // one card: the TP=1 form fits, the TP=2 entry does not
        let one_card = sys(vec![gpu("gfx1201", 32)], 64);
        assert_eq!(fit(&one.fit, &one_card, None, 0).verdict, Verdict::Fits);
        assert_eq!(fit(&m27.fit, &one_card, None, 0).verdict, Verdict::NoFit);
        // a TP=1 form on a TP=2-only preset is refused by name
        let bad = r#"{"version":1,"models":[{"id":"f","name":"F","kind":"Flash-Next","preset":"flashnext","tp1":{}}]}"#;
        let e = Catalog::parse(bad).unwrap_err();
        assert!(format!("{e:#}").contains("TP=2 only"), "{e:#}");
    }

    #[test]
    fn config_for_a_tp_and_the_settings_switch() {
        let dir = saber_test_scratch::ScratchDir::new("console-catalog-tp");
        let mut cfg = Config::default();
        cfg.engine.paths.nvme = dir.path().join("nvme");
        cfg.engine.paths.repo = dir.path().join("repo");
        let roots = Roots::for_config(&cfg, Path::new("/h"));
        let c = Catalog::embedded();
        let m27 = c.model("27b").unwrap();
        std::fs::create_dir_all(&roots.repo_cache).unwrap();
        for f in ["quark27b.tokenizer.json", "template_corpus.jinja"] {
            std::fs::write(roots.repo_cache.join(f), b"x").unwrap();
        }
        // the TP=1 packs not installed: refused by name
        let e = config_for_tp(&cfg, m27, 1, &roots).unwrap_err();
        assert!(format!("{e:#}").contains("one card"), "{e:#}");
        let one = m27.at_tp(1);
        for x in &one.components[..2] {
            std::fs::create_dir_all(roots.packs.join(&x.name)).unwrap();
        }
        let out = config_for_tp(&cfg, m27, 1, &roots).unwrap();
        assert_eq!((out.engine.preset, out.engine.tp), (Preset::Qwen27b, 1));
        let packs = |c: &Config| {
            (
                c.engine.flags.get("--pack").and_then(FlagValue::arg),
                c.engine
                    .flags
                    .get("--engine-draft-pack")
                    .and_then(FlagValue::arg),
            )
        };
        assert_eq!(
            packs(&out),
            (
                Some("/packs/qwen38-27b-macaq7-bc3-tp1".into()),
                Some("/packs/qwen38-dflash2-drafter-quarkhead-tp1".into())
            )
        );
        out.validate().unwrap();
        // the Settings switch: the catalog's packs are the presets' own at each TP, so the
        // overrides go and each TP's preset names its packs
        let back = with_tp(&out, 2, None, &c);
        assert_eq!(back.engine.tp, 2);
        assert_eq!(packs(&back), (None, None));
        let again = with_tp(&back, 1, Some(1), &c);
        assert_eq!((again.engine.tp, again.engine.device), (1, 1));
        assert_eq!(packs(&again), (None, None));
        // a pack the catalog does not list stays an override (the validator checks its tp_size)
        let mut q = out.clone();
        q.engine.flags.insert(
            "--pack".into(),
            FlagValue::Str("/packs/qwen38-27b-quark-awq-mxfp4-tp1".into()),
        );
        let q2 = with_tp(&q, 2, None, &c);
        assert_eq!(
            q2.engine.flags.get("--pack"),
            Some(&FlagValue::Str(
                "/packs/qwen38-27b-quark-awq-mxfp4-tp1".into()
            ))
        );
        // the preset's own packs (no override) stay with the preset; an unknown pack stays
        let mut stock = Config::default();
        stock.engine.preset = Preset::Qwen27b;
        stock.engine.flags.insert(
            "--pack".into(),
            FlagValue::Str("/packs/qwen38-27b-macaq7-bc3".into()),
        );
        stock.engine.flags.insert(
            "--engine-draft-pack".into(),
            FlagValue::Str("/packs/my-own-drafter".into()),
        );
        let s1 = with_tp(&stock, 1, None, &c);
        assert_eq!(s1.engine.flags.get("--pack"), None);
        assert_eq!(
            s1.engine.flags.get("--engine-draft-pack"),
            Some(&FlagValue::Str("/packs/my-own-drafter".into()))
        );
        // Flash-Next: only the field moves (its validate refuses 1)
        let f = with_tp(&Config::default(), 1, None, &c);
        assert_eq!(f.engine.flags, Config::default().engine.flags);
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
