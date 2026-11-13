//! The settings plane: check a draft config (validation + the memory estimate + the
//! diff and command line), apply it with a drain-and-restart and a progress record, roll back
//! by hand or automatically, and the read-only model manager.
//!
//! The apply law:
//! 1. **validate** -- errors block (nothing is saved);
//! 2. **save** -- the config file and a history entry;
//! 3. a restart happens only when the engine runs and the composed command line changed (a field
//!    that does not reach the command line -- metrics, usage, proxy keys, timeouts -- never
//!    restarts it); a stopped engine is started;
//! 4. **drain** (running + waiting to 0, bounded), **stop**, **start**, **loading**, **ready**
//!    (bounded by `console.ready_timeout_s`);
//! 5. a config that exits during load or does not reach ready in time is marked failed and, with
//!    `console.auto_rollback`, the last config that reached ready is restored and started.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
use serde_json::{json, Value};

use crate::compose::{effective_flags, EffectiveFlag, EngineCommand};
use crate::config::{Config, Preset};
use crate::estimate::{self, Estimate, Inputs, Law, PackInfo};
use crate::history::{History, Origin, Outcome};
use crate::profiles::ProfileStore;
use crate::scrape::MetricsStore;
use crate::supervisor::{LoadOutcome, Supervisor};
use crate::validate::{self, Facts, PackFact, Report, SchemaIndex};

/// What a check of a draft config returns (the `/api/config/validate` body).
#[derive(Clone, Debug, Serialize)]
pub struct Check {
    pub report: Report,
    pub estimate: Estimate,
    pub command: Vec<String>,
    pub command_line: String,
    pub effective: Vec<EffectiveFlag>,
    /// Config fields and command-line flags that differ from the running (else the saved) config.
    pub diff: Vec<DiffRow>,
    /// The diff's base: `running` or `saved`.
    pub diff_base: &'static str,
    /// Applying it restarts the running engine.
    pub restart_required: bool,
    /// Applying it changes nothing.
    pub unchanged: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DiffRow {
    /// `flag` (a command-line flag) or `config` (a config field).
    pub kind: &'static str,
    pub field: String,
    pub old: Option<String>,
    pub new: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum StepState {
    Pending,
    Running,
    Done,
    Failed,
    Skipped,
}

#[derive(Clone, Debug, Serialize)]
pub struct Step {
    pub id: &'static str,
    pub label: &'static str,
    pub state: StepState,
    pub message: Option<String>,
    pub t_ms: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum JobState {
    Running,
    Done,
    Failed,
    RolledBack,
}

#[derive(Clone, Debug, Serialize)]
pub struct ApplyJob {
    pub id: u64,
    pub origin: Origin,
    pub state: JobState,
    pub steps: Vec<Step>,
    pub restart: bool,
    pub history_id: Option<u64>,
    /// The history entry an auto-rollback restored, and its outcome.
    pub rolled_back_to: Option<u64>,
    pub message: Option<String>,
    pub started_ms: u64,
    pub finished_ms: Option<u64>,
}

const STEPS: &[(&str, &str)] = &[
    ("validate", "Validate"),
    ("save", "Save"),
    ("drain", "Drain"),
    ("stop", "Stop"),
    ("start", "Start"),
    ("loading", "Loading"),
    ("ready", "Ready"),
];

/// The settings plane's state shared by the settings routes.
pub struct Console {
    pub sup: Arc<Supervisor>,
    pub metrics: Arc<MetricsStore>,
    pub schema: SchemaIndex,
    pub history: History,
    pub profiles: ProfileStore,
    pub console_port: u16,
    pub data_dir: PathBuf,
    job: Mutex<Option<ApplyJob>>,
    seq: AtomicU64,
    /// The download manager (its worker is spawned by [`crate::run`], or a test).
    pub downloads: Arc<crate::downloads::Downloads>,
    /// The root the system probe reads sysfs / dev / proc under (`/`; a fake tree in tests).
    pub sys_root: PathBuf,
    /// Replaces the probe's commands (container runtime, card guard) in tests.
    pub host_facts: Option<crate::system::HostFacts>,
    /// The audit log (`<data dir>/audit.jsonl`).
    pub audit: crate::audit::Audit,
}

/// Flatten a JSON value into `path -> leaf` pairs.
fn leaves(prefix: &str, v: &Value, out: &mut Vec<(String, String)>) {
    match v {
        Value::Object(m) => {
            for (k, x) in m {
                let p = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                leaves(&p, x, out);
            }
        }
        Value::Null => {}
        other => out.push((prefix.to_string(), other.to_string())),
    }
}

/// The diff between two configs: config fields (the engine's flag overrides are compared on the
/// command line instead) and the composed command lines' flags.
#[must_use]
pub fn diff(old: &Config, old_cmd: &[String], new: &Config, new_cmd: &[String]) -> Vec<DiffRow> {
    let mut rows = Vec::new();
    let strip = |c: &Config| {
        let mut v = serde_json::to_value(c).unwrap_or(Value::Null);
        if let Some(e) = v.get_mut("engine").and_then(Value::as_object_mut) {
            e.remove("flags");
            e.remove("extra_args");
        }
        // never show a key in a diff
        if let Some(p) = v.get_mut("proxy").and_then(Value::as_object_mut) {
            if let Some(k) = p.get_mut("engine_key") {
                if k.as_str().is_some_and(|s| !s.is_empty()) {
                    *k = json!("(set)");
                }
            }
            if let Some(Value::Array(keys)) = p.get_mut("keys") {
                let n = keys.len();
                p.insert("keys".into(), json!(format!("{n} keys")));
            }
        }
        let mut out = Vec::new();
        leaves("", &v, &mut out);
        out
    };
    let (a, b) = (strip(old), strip(new));
    for (k, v) in &b {
        let o = a.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone());
        if o.as_deref() != Some(v.as_str()) {
            rows.push(DiffRow {
                kind: "config",
                field: k.clone(),
                old: o,
                new: Some(v.clone()),
            });
        }
    }
    for (k, v) in &a {
        if !b.iter().any(|(x, _)| x == k) {
            rows.push(DiffRow {
                kind: "config",
                field: k.clone(),
                old: Some(v.clone()),
                new: None,
            });
        }
    }
    let flags = |argv: &[String]| -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        for (k, v) in crate::compose::parse_argv(argv) {
            let v = v.unwrap_or_else(|| "(on)".into());
            match out.iter_mut().find(|(x, _)| *x == k) {
                Some(e) => {
                    e.1.push(' ');
                    e.1.push_str(&v);
                }
                None => out.push((k, v)),
            }
        }
        out
    };
    let (fa, fb) = (flags(old_cmd), flags(new_cmd));
    let mask = |k: &str, v: String| {
        if k == "--api-key" {
            "(set)".to_string()
        } else {
            v
        }
    };
    for (k, v) in &fb {
        let o = fa.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone());
        if o.as_deref() != Some(v.as_str()) {
            rows.push(DiffRow {
                kind: "flag",
                field: k.clone(),
                old: o.map(|o| mask(k, o)),
                new: Some(mask(k, v.clone())),
            });
        }
    }
    for (k, v) in &fa {
        if !fb.iter().any(|(x, _)| x == k) {
            rows.push(DiffRow {
                kind: "flag",
                field: k.clone(),
                old: Some(mask(k, v.clone())),
                new: None,
            });
        }
    }
    if old_cmd.first() != new_cmd.first() || old_cmd.get(1) != new_cmd.get(1) {
        rows.push(DiffRow {
            kind: "flag",
            field: "(binary)".into(),
            old: Some(
                old_cmd
                    .iter()
                    .take(2)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" "),
            ),
            new: Some(
                new_cmd
                    .iter()
                    .take(2)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" "),
            ),
        });
    }
    rows
}

/// A pack flag's value -> its host path for `preset` (the 27B runs in a container with the packs
/// dir mounted at `/packs`).
#[must_use]
pub fn pack_host_path(preset: Preset, packs_dir: &Path, arg: &str) -> PathBuf {
    match (preset, arg.strip_prefix("/packs/")) {
        (Preset::Qwen27b | Preset::Mock, Some(rest)) => packs_dir.join(rest),
        _ => PathBuf::from(arg),
    }
}

/// The flag value that names `pack` for `preset`.
#[must_use]
pub fn pack_arg(preset: Preset, pack: &Path) -> String {
    match preset {
        Preset::Flashnext => pack.display().to_string(),
        Preset::Qwen27b | Preset::Mock => format!(
            "/packs/{}",
            pack.file_name()
                .map(|n| n.to_string_lossy())
                .unwrap_or_default()
        ),
    }
}

/// The installed packs listing (`/api/packs`).
#[derive(Clone, Debug, Serialize)]
pub struct PackRow {
    #[serde(flatten)]
    pub info: PackInfo,
    /// The value `--pack` / `--engine-draft-pack` takes for this preset.
    pub arg: String,
    /// `target` / `drafter` when the effective config uses it.
    pub active_as: Option<&'static str>,
}

impl Console {
    #[must_use]
    pub fn new(
        sup: Arc<Supervisor>,
        metrics: Arc<MetricsStore>,
        schema: &Value,
        console_port: u16,
    ) -> Console {
        let cfg = sup.config_now();
        let data_dir = crate::expand_home(&cfg.console.data_dir, sup.home());
        let downloads = crate::downloads::Downloads::persistent(None, &data_dir);
        let audit = crate::audit::Audit::new(&data_dir);
        Console {
            history: History::new(&data_dir, cfg.console.history_keep),
            profiles: ProfileStore::new(&data_dir),
            schema: SchemaIndex::from_value(schema),
            sup,
            metrics,
            console_port,
            data_dir,
            job: Mutex::new(None),
            seq: AtomicU64::new(0),
            downloads,
            sys_root: PathBuf::from("/"),
            host_facts: None,
            audit,
        }
    }

    pub fn packs_dir(&self, cfg: &Config) -> PathBuf {
        cfg.engine
            .paths
            .resolved(self.sup.home())
            .nvme
            .join("saber-packs")
    }

    fn models_dir(&self, cfg: &Config) -> PathBuf {
        cfg.engine
            .paths
            .resolved(self.sup.home())
            .nvme
            .join("models")
    }

    fn pack_fact(&self, cfg: &Config, arg: Option<&str>) -> Option<PackFact> {
        let arg = arg?;
        let host = pack_host_path(cfg.engine.preset, &self.packs_dir(cfg), arg);
        let info = host.is_dir().then(|| estimate::read_pack(&host));
        Some(PackFact {
            arg: arg.to_string(),
            host_path: host.display().to_string(),
            info,
        })
    }

    /// Validate + estimate + diff `cfg` (reads the packs, the model config and the hw sample).
    pub fn check(&self, cfg: &Config) -> Check {
        let cmd = self.sup.compose_for(cfg);
        let m = validate::effective_map(&cmd);
        let preset = cfg.engine.preset;
        let (pack_flag, draft_flag) = match preset {
            Preset::Flashnext => ("--pack", "--draft-pack"),
            Preset::Qwen27b | Preset::Mock => ("--pack", "--engine-draft-pack"),
        };
        let v = |k: &str| m.get(k).and_then(|v| v.as_deref());
        let (pack, draft) = if preset.uses_gpus() {
            (
                self.pack_fact(cfg, v(pack_flag)),
                self.pack_fact(cfg, v(draft_flag)),
            )
        } else {
            (None, None)
        };
        let model = if preset.uses_gpus() {
            let mut fb: Vec<PathBuf> = Vec::new();
            if preset == Preset::Qwen27b {
                if let Ok(rd) = std::fs::read_dir(self.models_dir(cfg)) {
                    let mut ds: Vec<PathBuf> = rd
                        .flatten()
                        .map(|e| e.path())
                        .filter(|p| {
                            // the target's checkpoints, not a drafter's / a head's
                            let n = p
                                .file_name()
                                .map(|n| n.to_string_lossy().to_lowercase())
                                .unwrap_or_default();
                            n.contains("27b")
                                && !["dflash", "dspine", "embed", "drafter", "head"]
                                    .iter()
                                    .any(|x| n.contains(x))
                        })
                        .collect();
                    ds.sort();
                    fb.extend(ds.into_iter().map(|d| d.join("config.json")));
                }
            }
            estimate::read_model(pack.as_ref().and_then(|p| p.info.as_ref()), &fb)
        } else {
            None
        };
        let est = self.estimate(cfg, &cmd, pack.as_ref(), draft.as_ref(), model.clone());
        let facts = Facts {
            console_port: self.console_port,
            pack,
            draft_pack: draft,
            model,
            estimate: Some(est.clone()),
        };
        let report = validate::validate(cfg, &self.schema, &cmd, &facts);
        let st = self.sup.status();
        let saved = self.sup.config_now();
        let (base_cfg, base_cmd, base) = if st.state.alive() {
            (saved.clone(), st.command.clone(), "running")
        } else {
            (saved.clone(), self.sup.compose_for(&saved).engine, "saved")
        };
        let d = diff(&base_cfg, &base_cmd, cfg, &cmd.engine);
        let restart_required = st.state.alive() && st.command != cmd.engine;
        Check {
            report,
            estimate: est,
            command_line: cmd.engine_line(),
            effective: effective_flags(&cfg.engine, &cmd.engine),
            command: cmd.engine,
            unchanged: d.is_empty(),
            diff: d,
            diff_base: base,
            restart_required,
        }
    }

    fn estimate(
        &self,
        cfg: &Config,
        cmd: &EngineCommand,
        pack: Option<&PackFact>,
        draft: Option<&PackFact>,
        model: Option<estimate::ModelInfo>,
    ) -> Estimate {
        let m = validate::effective_map(cmd);
        let preset = cfg.engine.preset;
        let schema_default = |k: &str| -> Option<f64> {
            self.schema
                .get(preset, k)
                .and_then(|f| f.default.as_deref())
                .and_then(|d| d.parse().ok())
        };
        let num = |k: &str| -> Option<f64> {
            m.get(k)
                .and_then(|v| v.as_deref())
                .and_then(|v| v.parse().ok())
                .or_else(|| schema_default(k))
        };
        let s = |k: &str| m.get(k).and_then(|v| v.as_deref());
        let hw = self.metrics.hw_latest().map(|(_, h)| h);
        let vram = hw
            .as_ref()
            // the largest card (an APU's iGPU reports a small carve-out beside the dGPUs)
            .and_then(|h| h.gpus.iter().filter_map(|g| g.vram_total).reduce(f64::max))
            .map(|v| v as u64);
        let host_ram = hw.as_ref().and_then(|h| h.host.mem_total).map(|v| v as u64);
        let tp = validate::effective_tp(preset, &m);
        let info = |p: Option<&PackFact>| p.and_then(|p| p.info.clone());
        let (law, max_ctx, max_seqs, kv_b, st_b, budget) = match preset {
            Preset::Flashnext => (
                Law::Flashnext,
                num("--fn-max-ctx").unwrap_or(if s("--fn-kv-tier") == Some("off") {
                    8_192.0
                } else {
                    10_240.0
                }) as u64,
                estimate::FN_SEQ_SLOTS,
                if s("--fn-kv-dtype") == Some("bf16") {
                    2
                } else {
                    1
                },
                if m.contains_key("--fn-gdn-state-f32") {
                    4
                } else {
                    2
                },
                num("--fn-budget-mib").or(Some(30_500.0)),
            ),
            Preset::Qwen27b => (
                Law::Dense27b,
                num("--engine-max-ctx").unwrap_or(4096.0) as u64,
                num("--engine-max-seqs").unwrap_or(2.0) as u64,
                1,
                if s("--engine-gdn-state-dtype") == Some("f32") {
                    4
                } else {
                    2
                },
                num("--engine-budget-mib"),
            ),
            Preset::Mock => (
                Law::Mock,
                num("--engine-max-ctx").unwrap_or(4096.0) as u64,
                num("--engine-max-seqs").unwrap_or(2.0) as u64,
                1,
                2,
                None,
            ),
        };
        let inputs = Inputs {
            law,
            tp,
            vram_per_card: vram,
            budget_mib: budget.map(|b| b as u64),
            max_ctx,
            max_seqs,
            kv_dtype_bytes: kv_b,
            state_dtype_bytes: st_b,
            target: info(pack),
            drafter: info(draft),
            model,
            kv_headroom_mib: num("--engine-kv-headroom-mib").unwrap_or(1024.0) as u64,
            fn_kv_hot_gib: num("--fn-kv-hot-gib").unwrap_or(0.916),
            fn_admit_slab_tokens: num("--fn-kv-admit-slab-tokens").map(|v| v as u64),
            fn_kv_tier_on: s("--fn-kv-tier") != Some("off"),
            host_ram,
            mock_kv_gib: num("--kv-gib").unwrap_or(8.0) as u64,
        };
        estimate::estimate(&inputs)
    }

    /// The installed packs, with which the config uses.
    pub fn packs(&self, cfg: &Config) -> (PathBuf, Vec<PackRow>) {
        let dir = self.packs_dir(cfg);
        let cmd = self.sup.compose_for(cfg);
        let m = validate::effective_map(&cmd);
        let preset = cfg.engine.preset;
        let draft_flag = if preset == Preset::Flashnext {
            "--draft-pack"
        } else {
            "--engine-draft-pack"
        };
        let host = |k: &str| {
            m.get(k)
                .and_then(|v| v.as_deref())
                .map(|a| pack_host_path(preset, &dir, a))
        };
        let (target, drafter) = (host("--pack"), host(draft_flag));
        let mut rows: Vec<PackRow> = std::fs::read_dir(&dir)
            .map(|rd| {
                rd.flatten()
                    // a pack is a named directory (not `.cache` and the like)
                    .filter(|e| {
                        e.path().is_dir() && !e.file_name().to_string_lossy().starts_with('.')
                    })
                    .map(|e| {
                        let p = e.path();
                        let active_as = if target.as_deref() == Some(p.as_path()) {
                            Some("target")
                        } else if drafter.as_deref() == Some(p.as_path()) {
                            Some("drafter")
                        } else {
                            None
                        };
                        PackRow {
                            arg: pack_arg(preset, &p),
                            info: estimate::read_pack(&p),
                            active_as,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        rows.sort_by(|a, b| a.info.name.cmp(&b.info.name));
        (dir, rows)
    }

    #[must_use]
    pub fn job(&self) -> Option<ApplyJob> {
        self.job.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn update(&self, f: impl FnOnce(&mut ApplyJob)) {
        if let Some(j) = self.job.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            f(j);
        }
    }

    fn step(&self, id: &str, state: StepState, message: Option<String>) {
        self.update(|j| {
            if let Some(s) = j.steps.iter_mut().find(|s| s.id == id) {
                s.state = state;
                s.t_ms = Some(crate::now_ms());
                if message.is_some() {
                    s.message = message;
                }
            }
        });
    }

    fn finish(&self, state: JobState, message: Option<String>) {
        self.update(|j| {
            j.state = state;
            j.finished_ms = Some(crate::now_ms());
            j.message = message;
            for s in &mut j.steps {
                if s.state == StepState::Pending {
                    s.state = StepState::Skipped;
                }
            }
        });
    }

    /// Start an apply of `cfg` (validation first: errors refuse it with the report). Returns
    /// the job; its progress is [`Console::job`].
    pub fn apply(
        self: &Arc<Self>,
        cfg: Config,
        origin: Origin,
        note: String,
    ) -> Result<ApplyJob, Box<Check>> {
        // The credentials are the Security page's, never an apply's or a rollback's
        let cfg = cfg.with_credentials_of(&self.sup.config_now());
        let check = self.check(&cfg);
        if !check.report.ok {
            return Err(Box::new(check));
        }
        let id = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        let job = ApplyJob {
            id,
            origin,
            state: JobState::Running,
            steps: STEPS
                .iter()
                .map(|(id, label)| Step {
                    id,
                    label,
                    state: StepState::Pending,
                    message: None,
                    t_ms: None,
                })
                .collect(),
            restart: check.restart_required,
            history_id: None,
            rolled_back_to: None,
            message: None,
            started_ms: crate::now_ms(),
            finished_ms: None,
        };
        {
            let mut g = self.job.lock().unwrap_or_else(|e| e.into_inner());
            if g.as_ref().is_some_and(|j| j.state == JobState::Running) {
                let mut c = check;
                c.report.ok = false;
                c.report.errors.push(validate::Issue {
                    field: "apply".into(),
                    level: validate::Level::Error,
                    message: "another apply is in progress".into(),
                });
                return Err(Box::new(c));
            }
            *g = Some(job.clone());
        }
        self.step(
            "validate",
            StepState::Done,
            Some(format!("{} warnings", check.report.warnings.len())),
        );
        let me = self.clone();
        let line = check.command_line.clone();
        tokio::spawn(async move { me.run(cfg, origin, note, line).await });
        Ok(job)
    }

    async fn run(self: Arc<Self>, cfg: Config, origin: Origin, note: String, line: String) {
        let sup = self.sup.clone();
        let _op = sup.lock_op().await;
        let st = sup.status();
        let old = sup.config_now();
        let alive = st.state.alive();
        let new_cmd = sup.compose_for(&cfg);
        let restart = alive && st.command != new_cmd.engine;
        let start = !alive;
        self.history.set_keep(cfg.console.history_keep);

        // a first apply over a config that is running and ready: keep it as the baseline, so an
        // auto-rollback has somewhere to go
        if alive
            && st.state == crate::supervisor::EngineState::Ready
            && self.history.last_working(None).is_none()
        {
            let _ = self.history.record(
                &old,
                Origin::Baseline,
                Outcome::Ready,
                "the config running before the first apply",
                &st.command.join(" "),
            );
        }

        // save
        self.step("save", StepState::Running, None);
        let outcome = if restart || start {
            Outcome::Pending
        } else {
            Outcome::Saved
        };
        if let Err(e) = cfg.save(&sup.config_path) {
            self.step("save", StepState::Failed, Some(format!("{e:#}")));
            self.finish(JobState::Failed, Some(format!("saving the config: {e:#}")));
            return;
        }
        sup.set_config(cfg.clone());
        let hid = match self.history.record(&cfg, origin, outcome, &note, &line) {
            Ok(id) => Some(id),
            Err(e) => {
                sup.logs
                    .push("console", format!("saber ui: history: {e:#}"));
                None
            }
        };
        self.update(|j| j.history_id = hid);
        self.step(
            "save",
            StepState::Done,
            Some(match hid {
                Some(id) => format!("saved; history #{id}"),
                None => "saved (history not written)".into(),
            }),
        );
        if !restart && !start {
            self.finish(
                JobState::Done,
                Some(if alive {
                    "saved; the engine's command line is unchanged, so it keeps running (no restart)".into()
                } else {
                    "saved".into()
                }),
            );
            return;
        }

        match self.boot(alive, st.state).await {
            Ok(()) => {
                if let Some(id) = hid {
                    let _ = self.history.set_outcome(id, Outcome::Ready, None);
                }
                self.finish(
                    JobState::Done,
                    Some("the engine is ready on the new config".into()),
                );
            }
            Err(why) => {
                if let Some(id) = hid {
                    let _ = self
                        .history
                        .set_outcome(id, Outcome::Failed, Some(why.clone()));
                }
                sup.logs
                    .push("console", format!("saber ui: apply failed: {why}"));
                if !cfg.console.auto_rollback || origin == Origin::AutoRollback {
                    self.finish(JobState::Failed, Some(why));
                    return;
                }
                self.auto_rollback(hid, &why).await;
            }
        }
    }

    /// drain -> stop -> start -> loading -> ready; `Err` = why it did not reach ready.
    async fn boot(&self, alive: bool, state: crate::supervisor::EngineState) -> Result<(), String> {
        let sup = &self.sup;
        if alive {
            if state == crate::supervisor::EngineState::Ready {
                self.step("drain", StepState::Running, None);
                sup.drain().await;
                self.step("drain", StepState::Done, None);
            } else {
                self.step(
                    "drain",
                    StepState::Skipped,
                    Some("not ready: nothing to drain".into()),
                );
            }
            self.step("stop", StepState::Running, None);
            if let Err(e) = sup.stop_inner().await {
                self.step("stop", StepState::Failed, Some(format!("{e:#}")));
                return Err(format!("stopping the engine: {e:#}"));
            }
            self.step("stop", StepState::Done, None);
        } else {
            self.step(
                "drain",
                StepState::Skipped,
                Some("the engine was stopped".into()),
            );
            self.step("stop", StepState::Skipped, None);
        }
        self.step("start", StepState::Running, None);
        let t0 = crate::now_ms();
        if let Err(e) = sup.start_inner().await {
            self.step("start", StepState::Failed, Some(format!("{e:#}")));
            return Err(format!("start refused: {e:#}"));
        }
        let generation = sup.status().generation;
        self.step("start", StepState::Done, None);
        self.step("loading", StepState::Running, None);
        let timeout = sup.config_now().console.ready_timeout_s;
        match sup
            .wait_loaded(generation, Duration::from_secs(timeout))
            .await
        {
            LoadOutcome::Ready => {
                self.step("loading", StepState::Done, None);
                self.step("ready", StepState::Done, None);
                Ok(())
            }
            LoadOutcome::Exited { code, message } => {
                // the engine's own last word (clap's refusal, a FATAL line) says why
                let errs: Vec<String> = sup
                    .logs
                    .since(0, usize::MAX)
                    .into_iter()
                    .filter(|l| l.stream == "err" && l.t_ms >= t0 && !l.text.trim().is_empty())
                    .map(|l| l.text)
                    .collect();
                let telling = |t: &str| {
                    let t = t.trim_start().to_ascii_lowercase();
                    t.starts_with("error") || t.contains("fatal") || t.contains("panicked")
                };
                let last_err = errs
                    .iter()
                    .rev()
                    .find(|t| telling(t))
                    .or(errs.last())
                    .cloned();
                let why = format!(
                    "the engine exited during load ({}){}{}",
                    code.map_or("a signal".to_string(), |c| format!("code {c}")),
                    message
                        .filter(|m| !m.starts_with("the engine exited with code"))
                        .map(|m| format!(": {m}"))
                        .unwrap_or_default(),
                    last_err
                        .map(|l| format!("; last error line: {l}"))
                        .unwrap_or_default()
                );
                self.step("loading", StepState::Failed, Some(why.clone()));
                Err(why)
            }
            LoadOutcome::Timeout => {
                let why = format!("the engine did not reach ready within {timeout} s");
                self.step("loading", StepState::Failed, Some(why.clone()));
                // stop the stuck load before anything else boots
                let _ = sup.stop_inner().await;
                Err(why)
            }
            LoadOutcome::Superseded => {
                let why = "another start replaced this one".to_string();
                self.step("loading", StepState::Failed, Some(why.clone()));
                Err(why)
            }
        }
    }

    async fn auto_rollback(&self, failed: Option<u64>, why: &str) {
        let sup = &self.sup;
        let Some(target) = self.history.last_working(failed) else {
            self.finish(
                JobState::Failed,
                Some(format!(
                    "{why}; no earlier config reached ready, so there is nothing to roll back to"
                )),
            );
            return;
        };
        let tid = target.meta.id;
        self.update(|j| {
            j.rolled_back_to = Some(tid);
            j.steps.push(Step {
                id: "rollback",
                label: "Roll back",
                state: StepState::Running,
                message: Some(format!("restoring history #{tid}")),
                t_ms: Some(crate::now_ms()),
            });
        });
        sup.logs.push(
            "console",
            format!("saber ui: auto-rollback to history #{tid} ({why})"),
        );
        let cfg = target.config.with_credentials_of(&sup.config_now());
        self.audit.record(&crate::audit::AuditEntry {
            ts_ms: crate::now_ms(),
            who: "console".into(),
            ip: None,
            action: "config.auto-rollback".into(),
            route: String::new(),
            status: 0,
            detail: format!("to history #{tid}: {why}"),
        });
        let _ = cfg.save(&sup.config_path);
        sup.set_config(cfg.clone());
        let line = sup.compose_for(&cfg).engine_line();
        let rid = self
            .history
            .record(
                &cfg,
                Origin::AutoRollback,
                Outcome::Pending,
                &format!("auto-rollback to #{tid}"),
                &line,
            )
            .ok();
        // the failed boot is already stopped (exit / timeout's stop); start the restored one
        let res = async {
            if sup.status().state.alive() {
                sup.stop_inner().await.map_err(|e| format!("{e:#}"))?;
            }
            sup.start_inner().await.map_err(|e| format!("{e:#}"))?;
            let generation = sup.status().generation;
            let timeout = cfg.console.ready_timeout_s;
            match sup
                .wait_loaded(generation, Duration::from_secs(timeout))
                .await
            {
                LoadOutcome::Ready => Ok(()),
                other => Err(format!("{other:?}")),
            }
        }
        .await;
        let set_rb = |state: StepState, msg: String| {
            self.update(|j| {
                if let Some(s) = j.steps.iter_mut().find(|s| s.id == "rollback") {
                    s.state = state;
                    s.message = Some(msg);
                    s.t_ms = Some(crate::now_ms());
                }
            });
        };
        match res {
            Ok(()) => {
                if let Some(r) = rid {
                    let _ = self.history.set_outcome(r, Outcome::Ready, None);
                }
                set_rb(StepState::Done, format!("history #{tid} is running again"));
                self.finish(
                    JobState::RolledBack,
                    Some(format!(
                        "the new config failed ({why}); rolled back to the last working config (history #{tid}), which is ready"
                    )),
                );
            }
            Err(e) => {
                if let Some(r) = rid {
                    let _ = self
                        .history
                        .set_outcome(r, Outcome::Failed, Some(e.clone()));
                }
                set_rb(StepState::Failed, e.clone());
                self.finish(
                    JobState::Failed,
                    Some(format!(
                        "the new config failed ({why}); the rollback to history #{tid} failed too: {e}"
                    )),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_rows_and_masking() {
        let mut a = Config::mock(8099);
        a.proxy.keys = vec![crate::config::ProxyKey::new("x", "secret-1234")];
        let mut b = a.clone();
        b.engine.port = 8100;
        b.metrics.interval_s = 10;
        b.proxy
            .keys
            .push(crate::config::ProxyKey::new("y", "secret-5678"));
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let oc = s(&["saber", "serve", "--port", "8099", "--api-key", "k1", "--x"]);
        let nc = s(&[
            "saber",
            "serve",
            "--port",
            "8100",
            "--api-key",
            "k2",
            "--y",
            "3",
        ]);
        let d = diff(&a, &oc, &b, &nc);
        let f: Vec<(&str, &str)> = d.iter().map(|r| (r.kind, r.field.as_str())).collect();
        assert!(f.contains(&("config", "engine.port")));
        assert!(f.contains(&("config", "metrics.interval_s")));
        assert!(f.contains(&("config", "proxy.keys")));
        assert!(f.contains(&("flag", "--port")));
        assert!(f.contains(&("flag", "--y")));
        assert!(f.contains(&("flag", "--x")));
        let text = serde_json::to_string(&d).unwrap();
        assert!(
            !text.contains("secret") && !text.contains("k1") && !text.contains("k2"),
            "{text}"
        );
        assert!(diff(&a, &oc, &a, &oc).is_empty());
    }

    #[test]
    fn pack_paths_per_preset() {
        let d = Path::new("/n/saber-packs");
        assert_eq!(
            pack_host_path(Preset::Qwen27b, d, "/packs/qwen38-27b-x"),
            Path::new("/n/saber-packs/qwen38-27b-x")
        );
        assert_eq!(
            pack_host_path(Preset::Flashnext, d, "/n/saber-packs/fn"),
            Path::new("/n/saber-packs/fn")
        );
        assert_eq!(
            pack_arg(Preset::Qwen27b, Path::new("/n/saber-packs/q")),
            "/packs/q"
        );
        assert_eq!(
            pack_arg(Preset::Flashnext, Path::new("/n/saber-packs/q")),
            "/n/saber-packs/q"
        );
    }
}
