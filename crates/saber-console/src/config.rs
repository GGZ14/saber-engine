//! The console's config store: `~/.config/saber/config.toml`, typed and versioned.
//!
//! The file holds what the console needs to compose the engine's command line: a stock preset
//! (the launchers' defaults, see [`crate::compose`]), the engine's port, how it runs (inside the
//! dev image like `tools/serve_*.sh`, or as a direct child process) and per-flag overrides. The
//! path is resolved by the caller (`saber ui --config`); this crate reads no environment variable.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

/// The schema version this build writes. A file with a newer version is refused by name, an
/// older one is migrated in [`Config::from_toml`].
pub const CONFIG_VERSION: u32 = 5;

/// The stock engine presets. Each one reproduces a launcher's default boot exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Preset {
    /// `tools/serve_flashnext.sh` (Qwen3.8 Flash-Next, `saber engine --fn-serve`, TP=2).
    Flashnext,
    /// `tools/serve_27b.sh` (Qwen3.8-27B, `saber serve --engine real`): TP=2 on two cards, or
    /// `tools/serve_27b.sh --tp 1` on one ([`EngineConfig::tp`]).
    Qwen27b,
    /// `saber serve --engine mock`: the real serving plane over a scripted executor, no GPU.
    Mock,
}

impl Preset {
    /// True when the preset opens the GPUs (and so must pass the card guard).
    #[must_use]
    pub fn uses_gpus(self) -> bool {
        !matches!(self, Preset::Mock)
    }

    /// The tensor-parallel widths the preset serves: the 27B on one card or two, Flash-Next on
    /// two only (no single-card Flash-Next pack exists), the mock at either (it opens no GPU).
    #[must_use]
    pub fn tp_choices(self) -> &'static [u32] {
        match self {
            Preset::Flashnext => &[2],
            Preset::Qwen27b | Preset::Mock => &[1, 2],
        }
    }

    /// The `saber` subcommand the preset boots (the flag schema's section).
    #[must_use]
    pub fn subcommand(self) -> &'static str {
        match self {
            Preset::Flashnext => "engine",
            Preset::Qwen27b | Preset::Mock => "serve",
        }
    }
}

/// How the engine process is started.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Runtime {
    /// Inside the dev image, the way the launchers run it (`podman run ... saber engine ...`).
    Container,
    /// A direct child process of the console (the mock, or a host build).
    Direct,
}

/// One flag override. `true` adds a bare switch, `false` removes the flag (and its value) from
/// the preset's command line, a list repeats the flag once a value (`--api-key a --api-key b`),
/// anything else is the flag's value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FlagValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<String>),
}

impl FlagValue {
    /// The value as it goes on the command line (`None` for a switch).
    #[must_use]
    pub fn arg(&self) -> Option<String> {
        match self {
            FlagValue::Bool(_) => None,
            FlagValue::Int(v) => Some(v.to_string()),
            FlagValue::Float(v) => Some(v.to_string()),
            FlagValue::Str(v) => Some(v.clone()),
            FlagValue::List(v) => Some(v.join(",")),
        }
    }

    /// The value as the UI and the validator read it (`None` for a switch).
    #[must_use]
    pub fn display(&self) -> String {
        match self {
            FlagValue::Bool(b) => b.to_string(),
            FlagValue::List(v) => v.join(" "),
            other => other.arg().unwrap_or_default(),
        }
    }
}

/// Host paths the presets mount or read. The defaults are the launchers' defaults.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Paths {
    /// The repo checkout the engine runs from (mounted at /work in the container).
    pub repo: PathBuf,
    /// `SABER_SHARED`: captures, the optional prompt set, the launcher's output dir.
    pub shared: PathBuf,
    /// `SABER_NVME`: models/, saber-packs/ and saber-kv/.
    pub nvme: PathBuf,
    /// The 27B preset's tokenizer.json (host path, mounted read-only).
    pub tokenizer: PathBuf,
    /// The 27B preset's chat template (host path, mounted read-only).
    pub chat_template: PathBuf,
    /// The direct runtime's binary. Empty = the running `saber` binary itself.
    pub binary: PathBuf,
}

impl Paths {
    /// Every path with a leading `~/` expanded against `home` (the caller resolves `home`; this
    /// crate reads no environment variable).
    #[must_use]
    pub fn resolved(&self, home: &Path) -> Paths {
        let x = |p: &Path| match p.strip_prefix("~") {
            Ok(rest) => home.join(rest),
            Err(_) => p.to_path_buf(),
        };
        Paths {
            repo: x(&self.repo),
            shared: x(&self.shared),
            nvme: x(&self.nvme),
            tokenizer: x(&self.tokenizer),
            chat_template: x(&self.chat_template),
            binary: x(&self.binary),
        }
    }
}

impl Paths {
    /// These paths with the default checkout (`~/saber`) replaced by `repo`: the repo itself and
    /// any default file under it (the tokenizer and chat template the fetch script puts in its
    /// `.cache`). A path that is not a default is kept.
    #[must_use]
    pub fn rooted_at(mut self, repo: &Path) -> Paths {
        let d = Paths::default();
        if self.repo == d.repo {
            self.repo = repo.to_path_buf();
        }
        for p in [&mut self.tokenizer, &mut self.chat_template] {
            if let Ok(rest) = p.strip_prefix(&d.repo) {
                *p = repo.join(rest);
            }
        }
        self
    }
}

impl Default for Paths {
    fn default() -> Self {
        Paths {
            repo: PathBuf::from("~/saber"),
            shared: PathBuf::from("~/saber-cache"),
            nvme: PathBuf::from("/mnt/nvme"),
            tokenizer: PathBuf::from("~/saber/.cache/quark27b.tokenizer.json"),
            chat_template: PathBuf::from("~/saber/.cache/template_corpus.jinja"),
            binary: PathBuf::new(),
        }
    }
}

/// The container the GPU presets run in (the launchers' `$CTR run` arguments).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ContainerConfig {
    /// `podman` or `docker`.
    pub runtime: String,
    pub image: String,
    /// The container name. Empty = the launcher's name for the preset.
    pub name: String,
    /// `GPU_MAX_HW_QUEUES` for the HIP runtime inside the container (the launchers' default: 3).
    pub gpu_max_hw_queues: u32,
}

impl Default for ContainerConfig {
    fn default() -> Self {
        ContainerConfig {
            runtime: "podman".into(),
            image: "localhost/saber-dev:latest".into(),
            name: String::new(),
            gpu_max_hw_queues: 3,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineConfig {
    pub preset: Preset,
    /// Tensor parallelism: 2 = both cards (every preset's default), 1 = ONE card (the 27B only:
    /// `tools/serve_27b.sh --tp 1`'s boot: the `-tp1` packs, its context / sequence / batch
    /// defaults, the container `saber-27b-tp1-serve`). Flash-Next is TP=2 only.
    pub tp: u32,
    /// TP=1: the HIP device (card index) the engine runs on (the container sees only it).
    /// Ignored at TP=2 (both cards, 0 and 1).
    pub device: u32,
    /// The engine's HTTP port (`--fn-serve` / `--port`).
    pub port: u16,
    pub runtime: Runtime,
    /// Start the engine when the console starts.
    pub autostart: bool,
    /// Flash-Next: the launcher's agent-load defaults (`FN_AGENT_DEFAULTS=1`).
    pub agent_defaults: bool,
    /// Seconds a drain waits for running + waiting requests to reach 0 before the stop.
    pub drain_timeout_s: u64,
    /// Seconds a stop waits after SIGTERM before it kills.
    pub stop_timeout_s: u64,
    pub paths: Paths,
    pub container: ContainerConfig,
    /// Per-flag overrides, keyed by the long flag (`"--fn-max-ctx" = 131072`).
    pub flags: BTreeMap<String, FlagValue>,
    /// Extra arguments appended verbatim (the launchers' "any extra arguments").
    pub extra_args: Vec<String>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            preset: Preset::Flashnext,
            tp: 2,
            device: 0,
            port: 8080,
            runtime: Runtime::Container,
            autostart: false,
            agent_defaults: true,
            drain_timeout_s: 120,
            stop_timeout_s: 30,
            paths: Paths::default(),
            container: ContainerConfig::default(),
            flags: BTreeMap::new(),
            extra_args: Vec::new(),
        }
    }
}

impl EngineConfig {
    /// The TP=1 card the engine runs on, or `None` at TP=2 (both cards).
    #[must_use]
    pub fn single_card(&self) -> Option<u32> {
        (self.tp == 1 && self.preset.uses_gpus()).then_some(self.device)
    }

    /// The cards (HIP devices) this config's engine opens.
    #[must_use]
    pub fn devices(&self) -> Vec<u32> {
        match self.single_card() {
            Some(d) => vec![d],
            None => vec![0, 1],
        }
    }

    /// Migrate a version-4 engine section to version 5 (called by [`Config::from_toml`]):
    ///
    /// - TP=1 used to be written as flag overrides of the TP=2 27B preset (`tools/saber-setup
    ///   --tp 1`: `--engine-tp 1`, `--engine-device N`, `--engine-peer-device false`, the `-tp1`
    ///   packs and the single-card context / sequence / batch values). It is now `tp = 1` +
    ///   `device = N`; an override that only restates the TP=1 preset's own value is dropped, one
    ///   that differs (another pack, another context) stays an override.
    /// - An on/off flag set to a bare `false` (a toggle) is written as its `off` value
    ///   ([`crate::compose::ON_OFF_FLAGS`]): `false` used to drop the flag, which left the
    ///   engine's default (on) in place.
    pub fn migrate_v4(&mut self) {
        for k in crate::compose::ON_OFF_FLAGS {
            if let Some(FlagValue::Bool(b)) = self.flags.get(*k) {
                let v = if *b { "on" } else { "off" };
                self.flags
                    .insert((*k).to_string(), FlagValue::Str(v.into()));
            }
        }
        if self.preset != Preset::Qwen27b {
            return;
        }
        let int = |v: Option<&FlagValue>| -> Option<i64> {
            match v? {
                FlagValue::Int(n) => Some(*n),
                FlagValue::Str(s) => s.trim().parse().ok(),
                _ => None,
            }
        };
        if int(self.flags.get("--engine-tp")) != Some(1) {
            return;
        }
        self.tp = 1;
        self.flags.remove("--engine-tp");
        if let Some(d) = int(self.flags.get("--engine-device")) {
            if let Ok(d) = u32::try_from(d) {
                self.device = d;
                self.flags.remove("--engine-device");
            }
        }
        if self.flags.get("--engine-peer-device") == Some(&FlagValue::Bool(false)) {
            self.flags.remove("--engine-peer-device");
        }
        use crate::compose::tp1;
        let same =
            |v: Option<&FlagValue>, want: &str| v.and_then(FlagValue::arg).as_deref() == Some(want);
        // the old single-card context / sequences stay overrides (the file keeps the boot it had;
        // the TP=1 default is 1 sequence at 131,072 since 2026-11-14); the batches go: the
        // console derives them from the sequences (the old 1,2,4 = 4 sequences' ladder)
        for (k, want) in [
            ("--engine-decode-batches", tp1::V4_BATCHES.to_string()),
            ("--pack", format!("/packs/{}", tp1::PACK)),
            ("--engine-draft-pack", format!("/packs/{}", tp1::DRAFT_PACK)),
        ] {
            if same(self.flags.get(k), &want) {
                self.flags.remove(k);
            }
        }
    }
}

/// The metrics scraper's settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MetricsConfig {
    /// Seconds between scrapes of the engine's `/metrics`.
    pub interval_s: u64,
    /// Seconds of history the in-memory ring keeps.
    pub retention_s: u64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        MetricsConfig {
            interval_s: 5,
            retention_s: 24 * 3600,
        }
    }
}

impl MetricsConfig {
    /// Ring capacity in points (24 h at 5 s = 17,280).
    #[must_use]
    pub fn capacity(&self) -> usize {
        (self.retention_s / self.interval_s.max(1)).max(1) as usize
    }
}

/// A key the console's `/v1` proxy accepts, with the label its usage rows carry.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProxyKey {
    pub label: String,
    /// The key as text: only on its way in (a legacy file, the wizard). [`Config::save`] stores
    /// [`ProxyKey::hash`] instead and writes this empty.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub key: String,
    /// The salted hash ([`crate::auth`]) the proxy checks.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub hash: String,
    /// When the key was made (Unix ms; 0 = unknown).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub created_ms: u64,
    /// A revoked key is refused (it stays listed until it is deleted).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub revoked: bool,
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde's skip_serializing_if signature
fn is_zero(v: &u64) -> bool {
    *v == 0
}

impl ProxyKey {
    /// A key from its text (hashed when the config is saved).
    #[must_use]
    pub fn new(label: &str, key: &str) -> ProxyKey {
        ProxyKey {
            label: label.into(),
            key: key.into(),
            hash: String::new(),
            created_ms: 0,
            revoked: false,
        }
    }

    /// `presented` is this key (a revoked key never matches). Constant time.
    #[must_use]
    pub fn matches(&self, presented: &str) -> bool {
        if self.revoked || presented.is_empty() {
            return false;
        }
        if !self.hash.is_empty() {
            return crate::auth::verify(&self.hash, presented);
        }
        crate::auth::eq_ct(self.key.as_bytes(), presented.as_bytes())
    }
}

impl std::fmt::Debug for ProxyKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // never print a key
        f.debug_struct("ProxyKey")
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

/// The console's `/v1` reverse proxy and how the console authenticates to the engine.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProxyConfig {
    /// The Bearer key the console sends to the engine (its `/metrics`, `/v1/models`,
    /// `/v1/requests` polls and every proxied request). Empty = the first key of the engine's own
    /// `--api-key` flag override, else of its `--api-key-file`, else none (engine auth off).
    pub engine_key: String,
    /// A file holding the engine key (`LABEL=KEY` / `LABEL KEY` / a bare key; the first line
    /// that is not a comment). Used when `engine_key` is empty.
    pub engine_key_file: PathBuf,
    /// Keys clients present to the console's `/v1` proxy. Empty = the admin token when one is
    /// set, else the proxy is open (loopback only).
    pub keys: Vec<ProxyKey>,
}

/// The persistent usage history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UsageConfig {
    pub enabled: bool,
    /// The SQLite database (WAL). `~/` expands against the console's home.
    pub db_path: PathBuf,
    /// Days per-request rows are kept; the minute / hour / day rollups are kept forever.
    pub raw_retention_days: u32,
    /// Keep request and response bodies for the request inspector (OFF by default: privacy).
    pub store_content: bool,
    /// The most bytes of each body kept when `store_content` is on.
    pub content_max_bytes: usize,
    /// Seconds between polls of the engine's `/v1/requests` (direct-to-engine traffic).
    pub poll_interval_s: u64,
    /// The fixed UTC offset day and month rollups are cut at (minutes east of UTC). Applies to
    /// rows written after a change; earlier days keep their cut.
    pub utc_offset_minutes: i32,
}

impl Default for UsageConfig {
    fn default() -> Self {
        UsageConfig {
            enabled: true,
            db_path: PathBuf::from("~/.local/share/saber/usage.db"),
            raw_retention_days: 90,
            store_content: false,
            content_max_bytes: 256 * 1024,
            poll_interval_s: 5,
            utc_offset_minutes: 0,
        }
    }
}

/// The console's own settings: where it keeps its history and profiles, and the apply
/// flow's readiness law.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConsoleSettings {
    /// The console data dir (config history, user profiles). `~/` expands against the home.
    pub data_dir: PathBuf,
    /// Seconds an applied config has to reach `ready` before the apply counts as failed.
    pub ready_timeout_s: u64,
    /// Roll back to the last working config when an applied one fails to reach ready (the
    /// timeout above) or the engine exits during load.
    pub auto_rollback: bool,
    /// Applied configs kept in the history (oldest pruned first).
    pub history_keep: usize,
    /// The address `saber ui` binds when `--host` is not given (empty = 127.0.0.1).
    /// Takes effect the next time the console starts.
    pub host: String,
    /// The console's port when `--port` is not given (0 = 8090).
    pub port: u16,
    /// The admin token as text: only on its way in (a legacy file, the wizard).
    /// [`Config::save`] stores [`ConsoleSettings::admin_token_hash`] instead and writes this
    /// empty. `--admin-token` overrides both.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub admin_token: String,
    /// The admin token's salted hash ([`crate::auth`]); empty = no token (required off
    /// loopback). Set by the Security page's rotate, the wizard or `saber ui --gen-admin-token`.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub admin_token_hash: String,
}

impl Default for ConsoleSettings {
    fn default() -> Self {
        ConsoleSettings {
            data_dir: PathBuf::from("~/.local/share/saber/console"),
            ready_timeout_s: 900,
            auto_rollback: true,
            history_keep: 50,
            host: String::new(),
            port: 0,
            admin_token: String::new(),
            admin_token_hash: String::new(),
        }
    }
}

/// The setup wizard's place. The wizard shows while `completed` is false; every step
/// saves here, so a reload resumes where it was.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SetupState {
    pub completed: bool,
    /// "I'll configure manually": the wizard was skipped.
    pub skipped: bool,
    /// `system` | `model` | `profile` | `network` | `review` | `done`.
    pub step: String,
    /// The catalog id picked in the Model step.
    pub model: String,
    /// The built-in profile picked in the Profile step.
    pub profile: String,
    /// The TP picked in the Model step (0 = none yet: the machine's recommendation, TP=2 with
    /// two supported cards, TP=1 with one, for a model hosted at both).
    pub tp: u32,
}

impl Default for SetupState {
    fn default() -> Self {
        SetupState {
            completed: false,
            skipped: false,
            step: "system".into(),
            model: String::new(),
            profile: String::new(),
            tp: 0,
        }
    }
}

/// The download manager's settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DownloadsConfig {
    /// A model catalog file (`tools/models.json`'s format). Empty = the one built into this
    /// binary.
    pub catalog: PathBuf,
    /// The Hugging Face endpoint (a mirror). Empty = the catalog's.
    pub hf_endpoint: String,
    /// A file holding a Hugging Face token (only needed while a repo is private).
    pub hf_token_file: PathBuf,
    /// The downloader (`curl`; resumable, follows redirects). Empty = `curl` on the PATH.
    pub curl: PathBuf,
    /// Bytes kept free on the destination after a download (the disk-space precheck).
    pub reserve_bytes: u64,
}

impl Default for DownloadsConfig {
    fn default() -> Self {
        DownloadsConfig {
            catalog: PathBuf::new(),
            hf_endpoint: String::new(),
            hf_token_file: PathBuf::new(),
            curl: PathBuf::new(),
            reserve_bytes: 1 << 30,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub version: u32,
    #[serde(default)]
    pub engine: EngineConfig,
    #[serde(default)]
    pub metrics: MetricsConfig,
    #[serde(default)]
    pub proxy: ProxyConfig,
    #[serde(default)]
    pub usage: UsageConfig,
    #[serde(default)]
    pub console: ConsoleSettings,
    #[serde(default)]
    pub setup: SetupState,
    #[serde(default)]
    pub downloads: DownloadsConfig,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            version: CONFIG_VERSION,
            engine: EngineConfig::default(),
            metrics: MetricsConfig::default(),
            proxy: ProxyConfig::default(),
            usage: UsageConfig::default(),
            console: ConsoleSettings::default(),
            setup: SetupState::default(),
            downloads: DownloadsConfig::default(),
        }
    }
}

/// The engine's reserved key label for the console (`saber_serve::auth::CONSOLE_KEY_LABEL`):
/// with engine auth on, the engine keeps the console's `x-saber-console-ref` only from a request
/// that presented the key with this label.
pub const CONSOLE_KEY_LABEL: &str = "console";

/// The console's key in a key file's text (`LABEL=KEY`, `LABEL KEY` or a bare key per line, `#`
/// comments: the engine's own `--api-key-file` format): the line labelled
/// [`CONSOLE_KEY_LABEL`], else the first key.
#[must_use]
pub fn first_key(text: &str) -> Option<String> {
    let entries: Vec<(Option<&str>, &str)> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            l.split_once('=')
                .or_else(|| l.split_once(char::is_whitespace))
                .map_or((None, l), |(a, k)| (Some(a.trim()), k.trim()))
        })
        .filter(|(_, k)| !k.is_empty())
        .collect();
    entries
        .iter()
        .find(|(l, _)| *l == Some(CONSOLE_KEY_LABEL))
        .or_else(|| entries.first())
        .map(|(_, k)| (*k).to_string())
}

impl Config {
    /// The mock preset as a direct child: what `saber ui --engine mock` starts from.
    #[must_use]
    pub fn mock(port: u16) -> Config {
        let mut c = Config::default();
        c.engine.preset = Preset::Mock;
        c.engine.runtime = Runtime::Direct;
        c.engine.port = port;
        c.engine.autostart = true;
        c.engine.paths.tokenizer = PathBuf::from("~/saber/.cache/template_corpus.tokenizer.json");
        c
    }

    pub fn from_toml(text: &str) -> anyhow::Result<Config> {
        #[derive(Deserialize)]
        struct Version {
            version: Option<u32>,
        }
        let v: Version = toml::from_str(text).context("parsing the console config")?;
        match v.version {
            None => anyhow::bail!("the console config has no `version` key"),
            Some(n) if n > CONFIG_VERSION => anyhow::bail!(
                "the console config is version {n}; this build reads up to {CONFIG_VERSION} \
                 (a newer saber wrote it)"
            ),
            Some(_) => {}
        }
        let mut c: Config = toml::from_str(text).context("parsing the console config")?;
        // A file written before the wizard existed (version < 4) was configured by hand or by
        // the Settings page: the wizard does not take it over.
        if v.version.is_some_and(|n| n < 4) {
            c.setup.completed = true;
            c.setup.step = "done".into();
        }
        if v.version.is_some_and(|n| n < 5) {
            c.engine.migrate_v4();
        }
        // migrated: the next save writes this build's version
        c.version = CONFIG_VERSION;
        c.validate()?;
        Ok(c)
    }

    pub fn to_toml(&self) -> anyhow::Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }

    /// Cross-field checks that the types cannot carry.
    pub fn validate(&self) -> anyhow::Result<()> {
        let e = &self.engine;
        if e.port == 0 {
            anyhow::bail!("engine.port must be a real port, not 0");
        }
        if e.preset == Preset::Mock && e.runtime == Runtime::Container {
            anyhow::bail!("the mock preset runs as a direct child (engine.runtime = \"direct\")");
        }
        if e.preset.uses_gpus() && e.runtime == Runtime::Direct {
            anyhow::bail!(
                "the {:?} preset runs in the dev image (engine.runtime = \"container\"): the host \
                 has no ROCm",
                e.preset
            );
        }
        for k in e.flags.keys() {
            if !k.starts_with("--") {
                anyhow::bail!("engine.flags key `{k}` must be a long flag (`--name`)");
            }
        }
        if !e.preset.tp_choices().contains(&e.tp) {
            anyhow::bail!(
                "engine.tp = {} is not served by the {:?} preset (it serves TP={}){}",
                e.tp,
                e.preset,
                e.preset
                    .tp_choices()
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(" or TP="),
                if e.preset == Preset::Flashnext {
                    ": Flash-Next runs on two cards only"
                } else {
                    ""
                }
            );
        }
        if e.device > 15 {
            anyhow::bail!("engine.device = {} is not a card index (0..15)", e.device);
        }
        if self.metrics.interval_s == 0 {
            anyhow::bail!("metrics.interval_s must be at least 1");
        }
        if self.usage.raw_retention_days == 0 {
            anyhow::bail!("usage.raw_retention_days must be at least 1");
        }
        if self.usage.poll_interval_s == 0 {
            anyhow::bail!("usage.poll_interval_s must be at least 1");
        }
        if self.console.ready_timeout_s < 5 {
            anyhow::bail!("console.ready_timeout_s must be at least 5");
        }
        if !(1..=1000).contains(&self.console.history_keep) {
            anyhow::bail!("console.history_keep must be within 1..=1000");
        }
        if !self.console.host.is_empty() {
            if self.console.host != "localhost"
                && self.console.host.parse::<std::net::IpAddr>().is_err()
            {
                anyhow::bail!(
                    "console.host `{}` is not an IP address (use 127.0.0.1, 0.0.0.0 or one of \
                     this machine's addresses)",
                    self.console.host
                );
            }
            if !crate::is_loopback(&self.console.host)
                && self.console.admin_token.len() < 16
                && self.console.admin_token_hash.is_empty()
            {
                anyhow::bail!(
                    "console.host {} is reachable from other machines: set an admin token of at \
                     least 16 characters (console.admin_token)",
                    self.console.host
                );
            }
        }
        if !self.console.admin_token.is_empty() && self.console.admin_token.len() < 16 {
            anyhow::bail!("console.admin_token must be at least 16 characters");
        }
        if !self.console.admin_token_hash.is_empty()
            && !crate::auth::is_hash(&self.console.admin_token_hash)
        {
            anyhow::bail!(
                "console.admin_token_hash is not a console hash (sha256$<salt>$<digest>): set the \
                 token again with `saber ui --gen-admin-token` or the Security page"
            );
        }
        if self.console.port != 0 && self.console.port == self.engine.port {
            anyhow::bail!(
                "console.port and engine.port are both {}: pick two different ports",
                self.engine.port
            );
        }
        if !matches!(self.setup.tp, 0..=2) {
            anyhow::bail!("setup.tp = {} is not 0 (not picked), 1 or 2", self.setup.tp);
        }
        if !matches!(
            self.setup.step.as_str(),
            "system" | "model" | "profile" | "network" | "review" | "done"
        ) {
            anyhow::bail!("setup.step `{}` is not a wizard step", self.setup.step);
        }
        if self.usage.utc_offset_minutes.abs() > 14 * 60 {
            anyhow::bail!("usage.utc_offset_minutes must be within +-840");
        }
        let mut keys = std::collections::BTreeSet::new();
        let mut labels = std::collections::BTreeSet::new();
        for k in &self.proxy.keys {
            if k.label.trim().is_empty() {
                anyhow::bail!("proxy.keys: a key has an empty label");
            }
            if !labels.insert(k.label.as_str()) {
                anyhow::bail!("proxy.keys: the label `{}` is used twice", k.label);
            }
            if k.key.is_empty() && k.hash.is_empty() {
                anyhow::bail!("proxy.keys: `{}` has neither a key nor a hash", k.label);
            }
            if !k.key.is_empty() {
                if k.key.len() < 8 {
                    anyhow::bail!(
                        "proxy.keys: the key for `{}` is shorter than 8 bytes",
                        k.label
                    );
                }
                if !keys.insert(k.key.as_str()) {
                    anyhow::bail!("proxy.keys: the key for `{}` is a duplicate", k.label);
                }
            }
            if !k.hash.is_empty() && !crate::auth::is_hash(&k.hash) {
                anyhow::bail!("proxy.keys: the hash for `{}` is malformed", k.label);
            }
        }
        Ok(())
    }

    /// The Bearer key the console sends to the engine ([`ProxyConfig::engine_key`]'s order);
    /// `home` expands `~/` in file paths. `None` = engine auth is off as far as the config says.
    #[must_use]
    pub fn engine_bearer(&self, home: &Path) -> Option<String> {
        let x = |p: &Path| match p.strip_prefix("~") {
            Ok(rest) => home.join(rest),
            Err(_) => p.to_path_buf(),
        };
        let read = |p: &Path| {
            std::fs::read_to_string(x(p))
                .ok()
                .and_then(|t| first_key(&t))
        };
        let p = &self.proxy;
        if !p.engine_key.is_empty() {
            return Some(p.engine_key.clone());
        }
        if !p.engine_key_file.as_os_str().is_empty() {
            if let Some(k) = read(&p.engine_key_file) {
                return Some(k);
            }
        }
        let e = &self.engine;
        // the flag overrides, then the verbatim extra arguments
        let flag = |name: &str| -> Option<String> {
            match e.flags.get(name) {
                Some(FlagValue::Str(v)) => return Some(v.clone()),
                // the repeated `--api-key` specs as the lines of one key file
                Some(FlagValue::List(v)) if !v.is_empty() => return Some(v.join("\n")),
                _ => {}
            }
            let mut it = e.extra_args.iter();
            while let Some(a) = it.next() {
                if a == name {
                    return it.next().cloned();
                }
                if let Some(v) = a.strip_prefix(&format!("{name}=")) {
                    return Some(v.to_string());
                }
            }
            None
        };
        if let Some(spec) = flag("--api-key") {
            if let Some(k) = first_key(&spec) {
                return Some(k);
            }
        }
        flag("--api-key-file").and_then(|f| read(Path::new(&f)))
    }

    /// Replace every secret held as text (the admin token, the proxy keys) with its salted hash
    ///. Idempotent; the engine key stays (the console must send it).
    pub fn seal(&mut self) -> anyhow::Result<()> {
        if !self.console.admin_token.is_empty() {
            self.console.admin_token_hash = crate::auth::hash_secret(&self.console.admin_token)?;
            self.console.admin_token.clear();
        }
        for k in &mut self.proxy.keys {
            if !k.key.is_empty() {
                k.hash = crate::auth::hash_secret(&k.key)?;
                k.key.clear();
            }
            if k.created_ms == 0 {
                k.created_ms = crate::now_ms();
            }
        }
        Ok(())
    }

    /// True when a secret is still held as text (a legacy file: the console migrates it).
    #[must_use]
    pub fn has_plain_secrets(&self) -> bool {
        !self.console.admin_token.is_empty() || self.proxy.keys.iter().any(|k| !k.key.is_empty())
    }

    /// The config as the API shows it: no secret, and no hash either ([`crate::auth::REDACTED`]
    /// marks one that is set).
    #[must_use]
    pub fn redacted(&self) -> Config {
        let mut c = self.clone();
        let mark = |set: bool| {
            if set {
                crate::auth::REDACTED.to_string()
            } else {
                String::new()
            }
        };
        c.console.admin_token_hash =
            mark(!c.console.admin_token.is_empty() || !c.console.admin_token_hash.is_empty());
        c.console.admin_token.clear();
        c.proxy.engine_key = mark(!c.proxy.engine_key.is_empty());
        for k in &mut c.proxy.keys {
            k.hash = mark(!k.key.is_empty() || !k.hash.is_empty());
            k.key.clear();
        }
        c
    }

    /// A config from a client (the Settings page, an apply, a rollback to a history entry) with
    /// the console's CURRENT credentials: the admin token and the proxy keys are managed on the
    /// Security page (`/api/auth/*`), never by a config save or a rollback (a rollback must not
    /// bring a revoked key back). A redacted engine key keeps the stored one.
    #[must_use]
    pub fn with_credentials_of(mut self, current: &Config) -> Config {
        self.console.admin_token = current.console.admin_token.clone();
        self.console.admin_token_hash = current.console.admin_token_hash.clone();
        self.proxy.keys = current.proxy.keys.clone();
        if self.proxy.engine_key == crate::auth::REDACTED {
            self.proxy.engine_key = current.proxy.engine_key.clone();
        }
        self
    }

    /// The admin credential this config holds.
    #[must_use]
    pub fn admin(&self) -> crate::auth::Admin {
        if !self.console.admin_token_hash.is_empty() {
            crate::auth::Admin::Hash(self.console.admin_token_hash.clone())
        } else if !self.console.admin_token.is_empty() {
            // a legacy file before its migration: hash it in memory
            crate::auth::hash_secret(&self.console.admin_token).map_or(
                crate::auth::Admin::Flag(self.console.admin_token.clone()),
                crate::auth::Admin::Hash,
            )
        } else {
            crate::auth::Admin::None
        }
    }

    /// Load `path`; a missing file is the default config (not written until saved).
    pub fn load(path: &Path) -> anyhow::Result<Option<Config>> {
        match std::fs::read_to_string(path) {
            Ok(t) => Ok(Some(
                Config::from_toml(&t).with_context(|| format!("in {}", path.display()))?,
            )),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }

    /// Write atomically (temp file + rename), keeping the previous file as `config.toml.prev`.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        self.validate()?;
        // never a secret as text on disk
        let mut sealed = self.clone();
        sealed.seal()?;
        let text = sealed.to_toml()?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let tmp = path.with_extension("toml.tmp");
        {
            // owner-only: the file can hold the admin token and API keys
            use std::io::Write;
            use std::os::unix::fs::OpenOptionsExt;
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(&tmp)
                .with_context(|| format!("writing {}", tmp.display()))?;
            f.write_all(text.as_bytes())
                .with_context(|| format!("writing {}", tmp.display()))?;
        }
        if path.exists() {
            let _ = std::fs::copy(path, path.with_extension("toml.prev"));
        }
        std::fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_default_and_overrides() {
        let mut c = Config::default();
        c.engine
            .flags
            .insert("--fn-max-ctx".into(), FlagValue::Int(131072));
        c.engine
            .flags
            .insert("--fn-thinking".into(), FlagValue::Str("off".into()));
        c.engine
            .flags
            .insert("--fn-ring-idle".into(), FlagValue::Bool(false));
        c.engine
            .flags
            .insert("--fn-vram-elastic-kv-max-gib".into(), FlagValue::Float(1.5));
        c.engine.flags.insert(
            "--api-key".into(),
            FlagValue::List(vec!["a=k1-123456".into(), "b=k2-123456".into()]),
        );
        c.engine.extra_args = vec!["--fn-route-trace".into(), "/x".into()];
        let t = c.to_toml().unwrap();
        let back = Config::from_toml(&t).unwrap();
        assert_eq!(back, c);
        // and once more through the file store
        let dir = saber_test_scratch::ScratchDir::new("console-config");
        let p = dir.path().join("saber/config.toml");
        c.save(&p).unwrap();
        assert_eq!(Config::load(&p).unwrap().unwrap(), c);
        c.engine.port = 8100;
        c.save(&p).unwrap();
        assert!(p.with_extension("toml.prev").exists());
        assert_eq!(Config::load(&p).unwrap().unwrap().engine.port, 8100);
        assert!(Config::load(&dir.path().join("absent.toml"))
            .unwrap()
            .is_none());
    }

    #[test]
    fn mock_round_trips() {
        let c = Config::mock(8099);
        assert_eq!(Config::from_toml(&c.to_toml().unwrap()).unwrap(), c);
    }

    #[test]
    fn refuses_newer_version_unknown_keys_and_bad_combos() {
        assert!(Config::from_toml("version = 6").is_err());
        // a version-2 file (before the settings plane) reads with the console section at its defaults
        assert_eq!(
            Config::from_toml("version = 2").unwrap().console,
            ConsoleSettings::default()
        );
        // a version-1 file (before the usage store) reads with the new sections at their defaults
        assert_eq!(
            Config::from_toml("version = 1").unwrap().usage,
            UsageConfig::default()
        );
        assert!(Config::from_toml("[engine]\nport = 1").is_err());
        assert!(Config::from_toml("version = 1\n[engine]\nbogus = 1").is_err());
        assert!(Config::from_toml("version = 1\n[engine]\npreset = \"mock\"").is_err());
        assert!(Config::from_toml("version = 1\n[engine]\nruntime = \"direct\"").is_err());
        assert!(Config::from_toml("version = 1\n[engine.flags]\nfn-max-ctx = 1").is_err());
        // a minimal file is the defaults
        assert_eq!(Config::from_toml("version = 5").unwrap(), Config::default());
        assert_eq!(Config::from_toml("version = 4").unwrap(), Config::default());
        // the TP: the 27B at 1 or 2, Flash-Next at 2 only
        assert!(Config::from_toml("version = 5\n[engine]\npreset = \"qwen27b\"\ntp = 1").is_ok());
        assert!(Config::from_toml("version = 5\n[engine]\npreset = \"qwen27b\"\ntp = 3").is_err());
        let e = Config::from_toml("version = 5\n[engine]\ntp = 1").unwrap_err();
        assert!(format!("{e:#}").contains("two cards only"), "{e:#}");
        // a version-3 file (before the wizard) counts as set up
        let v3 = Config::from_toml("version = 3").unwrap();
        assert!(v3.setup.completed);
        assert_eq!(v3.setup.step, "done");
        assert!(!Config::from_toml("version = 4").unwrap().setup.completed);
        assert!(Config::from_toml("version = 4\n[setup]\nstep = \"bogus\"").is_err());
        // a non-loopback console bind needs a real admin token
        assert!(Config::from_toml("version = 4\n[console]\nhost = \"0.0.0.0\"").is_err());
        assert!(Config::from_toml(
            "version = 4\n[console]\nhost = \"0.0.0.0\"\nadmin_token = \"0123456789abcdef\""
        )
        .is_ok());
        assert!(Config::from_toml("version = 4\n[console]\nhost = \"example.com\"").is_err());
        assert!(Config::from_toml("version = 4\n[console]\nport = 8080").is_err());
        assert!(
            Config::from_toml("version = 2\n[[proxy.keys]]\nlabel = \"a\"\nkey = \"short\"")
                .is_err()
        );
        assert!(Config::from_toml("version = 2\n[usage]\nraw_retention_days = 0").is_err());
    }

    /// A version-4 file (`tools/saber-setup --tp 1` wrote TP=1 as flag overrides of the TP=2
    /// preset) reads as `tp = 1` on its card, the overrides that restate the TP=1 preset dropped.
    #[test]
    fn a_version_4_tp1_override_set_migrates_to_the_tp_field() {
        let v4 = r#"version = 4
[engine]
preset = "qwen27b"
port = 8081

[engine.flags]
"--engine-tp" = 1
"--engine-device" = 1
"--engine-peer-device" = false
"--engine-decode-batches" = "1,2,4"
"--engine-max-ctx" = 65536
"--engine-max-seqs" = 2
"--pack" = "/packs/qwen38-27b-macaq7-bc3-tp1"
"--engine-draft-pack" = "/packs/qwen38-dflash2-drafter-quarkhead-tp1"
"--enable-prefix-caching" = false
"--vision" = "lazy"

[setup]
completed = true
step = "done"
model = "27b"
"#;
        let c = Config::from_toml(v4).unwrap();
        assert_eq!((c.engine.tp, c.engine.device), (1, 1));
        let keys: Vec<&str> = c.engine.flags.keys().map(String::as_str).collect();
        // the old context / sequences (the boot the file had) and the unrelated flags stay
        // overrides; the batches go (derived from the sequences now)
        assert_eq!(
            keys,
            [
                "--enable-prefix-caching",
                "--engine-max-ctx",
                "--engine-max-seqs",
                "--vision"
            ]
        );
        assert_eq!(
            c.engine.flags["--enable-prefix-caching"],
            FlagValue::Str("off".into())
        );
        assert_eq!(c.engine.flags["--engine-max-seqs"], FlagValue::Int(2));
        // it saves as version 5 and reads back the same
        let t = c.to_toml().unwrap();
        assert!(t.contains("version = 5") && t.contains("tp = 1"), "{t}");
        assert_eq!(Config::from_toml(&t).unwrap(), c);
        // another pack than the preset's stays an override (the validator checks its tp_size)
        let other = v4.replace(
            "/packs/qwen38-27b-macaq7-bc3-tp1",
            "/packs/qwen38-27b-nvfp4-devfinal-tp1-fp8head",
        );
        let c = Config::from_toml(&other).unwrap();
        assert_eq!(c.engine.tp, 1);
        assert_eq!(
            c.engine.flags["--pack"],
            FlagValue::Str("/packs/qwen38-27b-nvfp4-devfinal-tp1-fp8head".into())
        );
        // a TP=2 file is untouched (but its prefix toggle is spelled out)
        let c = Config::from_toml(
            "version = 4\n[engine]\npreset = \"qwen27b\"\n[engine.flags]\n\"--engine-max-ctx\" = 65536\n\"--enable-prefix-caching\" = true",
        )
        .unwrap();
        assert_eq!(c.engine.tp, 2);
        assert_eq!(c.engine.flags["--engine-max-ctx"], FlagValue::Int(65536));
        assert_eq!(
            c.engine.flags["--enable-prefix-caching"],
            FlagValue::Str("on".into())
        );
        // a version-5 file is not migrated: an explicit --engine-tp override stays (the
        // validator refuses one that disagrees with engine.tp)
        let c = Config::from_toml(
            "version = 5\n[engine]\npreset = \"qwen27b\"\n[engine.flags]\n\"--engine-tp\" = 1",
        )
        .unwrap();
        assert_eq!(c.engine.tp, 2);
        assert!(c.engine.flags.contains_key("--engine-tp"));
        assert_eq!(c.engine.devices(), [0, 1]);
    }

    #[test]
    fn engine_bearer_order() {
        let home = Path::new("/nonexistent-home");
        let mut c = Config::default();
        assert_eq!(c.engine_bearer(home), None);
        c.engine.extra_args = vec!["--api-key".into(), "ops=extra-key-123".into()];
        assert_eq!(c.engine_bearer(home).as_deref(), Some("extra-key-123"));
        c.engine
            .flags
            .insert("--api-key".into(), FlagValue::Str("ci=flag-key-123".into()));
        assert_eq!(c.engine_bearer(home).as_deref(), Some("flag-key-123"));
        let dir = saber_test_scratch::ScratchDir::new("console-bearer");
        let f = dir.path().join("k.txt");
        std::fs::write(&f, "# keys\n\nconsole file-key-123\n").unwrap();
        c.proxy.engine_key_file = f;
        assert_eq!(c.engine_bearer(home).as_deref(), Some("file-key-123"));
        c.proxy.engine_key = "explicit-key-1".into();
        assert_eq!(c.engine_bearer(home).as_deref(), Some("explicit-key-1"));
        assert_eq!(first_key("bare-key-1234").as_deref(), Some("bare-key-1234"));
        // the key labelled `console` wins over the first (the engine keeps its refs)
        assert_eq!(
            first_key("ci=ci-key-12345\nconsole=con-key-12345\n").as_deref(),
            Some("con-key-12345")
        );
        c.proxy.engine_key.clear();
        c.proxy.engine_key_file = PathBuf::new();
        c.engine.flags.insert(
            "--api-key".into(),
            FlagValue::List(vec![
                "ci=ci-key-12345".into(),
                "console=con-key-12345".into(),
            ]),
        );
        assert_eq!(c.engine_bearer(home).as_deref(), Some("con-key-12345"));
    }

    #[test]
    fn secrets_are_sealed_on_save_redacted_on_read_and_kept_by_saves() {
        let dir = saber_test_scratch::ScratchDir::new("console-seal");
        let p = dir.path().join("config.toml");
        let mut c = Config::default();
        c.console.admin_token = "sa-admin-plaintext-0123".into();
        c.proxy.keys = vec![ProxyKey::new("laptop", "sk-saber-plaintext-1")];
        c.proxy.engine_key = "engine-key-plain".into();
        assert!(c.has_plain_secrets());
        c.save(&p).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(
            !text.contains("plaintext"),
            "no secret as text on disk:\n{text}"
        );
        assert!(text.contains("admin_token_hash = \"sha256$"), "{text}");
        let back = Config::load(&p).unwrap().unwrap();
        assert!(!back.has_plain_secrets());
        assert!(back.admin().check("sa-admin-plaintext-0123"));
        assert!(!back.admin().check("sa-admin-plaintext-0124"));
        assert!(back.proxy.keys[0].matches("sk-saber-plaintext-1"));
        assert!(back.proxy.keys[0].created_ms > 0);
        let mut revoked = back.proxy.keys[0].clone();
        revoked.revoked = true;
        assert!(
            !revoked.matches("sk-saber-plaintext-1"),
            "revoked = refused"
        );
        // the API's view holds neither secrets nor hashes
        let r = back.redacted();
        let js = serde_json::to_string(&r).unwrap();
        assert!(
            !js.contains("sha256$") && !js.contains("engine-key-plain"),
            "{js}"
        );
        assert_eq!(r.console.admin_token_hash, crate::auth::REDACTED);
        // a client's config (redacted, or with keys of its own) keeps the current credentials
        let mut incoming = r.clone();
        incoming.proxy.keys.clear();
        incoming.console.admin_token_hash.clear();
        let merged = incoming.with_credentials_of(&back);
        assert_eq!(merged.proxy.keys, back.proxy.keys);
        assert_eq!(
            merged.console.admin_token_hash,
            back.console.admin_token_hash
        );
        assert_eq!(merged.proxy.engine_key, "engine-key-plain");
        merged.validate().unwrap();
        // a malformed hash is refused by name
        let mut bad = back.clone();
        bad.console.admin_token_hash = "sha256$nothex$00".into();
        assert!(bad.validate().is_err());
    }
}
