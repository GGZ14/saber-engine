//! The manifest — typed TOML, one parser, one validator, one owner.
//!
//! This is the replacement for the 137 flat `SABER_*` environment variables. The failure those
//! caused was never the count: it was *two independent readers deriving behaviour from the same
//! name*, in two languages, with no one owning the agreement. So there is exactly one parser here,
//! it is typed, and it refuses anything it does not understand.
//!
//! Validation reports **every** problem in one pass, with the full key path:
//!
//! ```text
//! saber.toml: 3 problem(s):
//!   - missing required key `device.power_cap_w`
//!   - unknown key `device.powercap` (expected one of: power_cap_w, target_gfx, tp)
//!   - `buckets[1].batch` must be >= 1, found 0
//! ```
//!
//! An unknown key is an error, never a warning: a manifest that silently ignores a key is the
//! env-var bug class wearing a different hat.

use crate::error::ManifestError;
use crate::plan::{Bucket, Phase};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use toml::{Table, Value};

/// `[model]` — which tier is being served.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSection {
    /// Path to the tier pack or checkpoint directory.
    pub path: PathBuf,
    /// The tier id: one content hash over checkpoint, drafter, adapters, serving config, chat
    /// template and autotune fingerprint. The release unit, not the checkpoint.
    pub tier_id: String,
    /// The drafter's tier id, when a drafter is bound. The planner refuses a mismatch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drafter_tier_id: Option<String>,
}

/// `[device]` — the target, and the power cap every measurement is valid at.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceSection {
    /// LLVM target the plan is resolved for, e.g. `gfx1201`.
    pub target_gfx: String,
    /// **Required.** The enforced power cap in watts. A record measured uncapped is invalid at the
    /// enforced 210 W, so the cap is part of every autotune key — there is no default for it.
    pub power_cap_w: u32,
    /// Tensor-parallel ranks. Defaults to 1.
    #[serde(default = "default_tp")]
    pub tp: u32,
}

fn default_tp() -> u32 {
    1
}

/// One `[[plugins]]` entry: a `.hsaco` code object plus its manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginSection {
    /// Plugin name, referenced by every implementation it ships.
    pub name: String,
    /// Path to the code object / plugin manifest.
    pub path: PathBuf,
    /// Build hash of the artifact. Part of every autotune key: a rebuilt plugin invalidates
    /// measurements taken against the old one.
    pub build_hash: String,
}

impl Serialize for BucketSection {
    /// Emits only the keys the parser accepts for this bucket's phase.
    ///
    /// A derived impl would write `batch` and `spec_width` into a prefill bucket, which the
    /// parser then refuses as decode keys -- so the manifest would not survive its own round
    /// trip. The phase-specific key set is the point of the check, not an accident of it.
    fn serialize<S: serde::Serializer>(&self, ser: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let extra = usize::from(self.capture.is_some());
        let mut st = ser.serialize_struct("BucketSection", 3 + extra)?;
        match self.phase {
            Phase::Decode => {
                st.serialize_field("batch", &self.batch)?;
                st.serialize_field("spec_width", &self.spec_width)?;
            }
            Phase::Prefill => {
                st.serialize_field("phase", "prefill")?;
                st.serialize_field("tokens", &self.tokens)?;
            }
        }
        st.serialize_field("ctx_band", &self.ctx_band)?;
        if let Some(c) = self.capture {
            st.serialize_field("capture", &c)?;
        }
        st.end()
    }
}

/// One `[[buckets]]` entry: the graph-capture bucket set is load-bearing (dropping capture sizes
/// 1/2/4 cost 8% of single-stream decode) and bounded from above by the Infinity Cache.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct BucketSection {
    /// Which half of the step. `phase = "prefill"` takes `tokens` instead of
    /// `batch`/`spec_width`; omitting it means decode, which is what every bucket was before P2.
    #[serde(default)]
    pub phase: Phase,
    /// Sequences in flight. Always 1 in a prefill bucket.
    pub batch: u32,
    /// Speculative width: draft tokens verified per step. One GDN state slot per draft token.
    /// Zero in a prefill bucket.
    pub spec_width: u32,
    /// Prompt tokens in a prefill chunk. Zero in a decode bucket.
    #[serde(default)]
    pub tokens: u32,
    /// Context band this bucket is captured for.
    pub ctx_band: u32,
    /// Overrides `[plan] capture` for this bucket. `None` inherits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture: Option<bool>,
}

/// One `[plan.overrides]` entry, keyed by model-op name.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanOverride {
    /// Pin the route to this implementation. It must still satisfy the constraints.
    #[serde(rename = "impl", default, skip_serializing_if = "Option::is_none")]
    pub impl_name: Option<String>,
    /// Assert the tuned record's variant id. A mismatch is an error, not a silent re-pick.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant_id: Option<u32>,
}

/// `[plan]` — planner-level switches and per-op overrides.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanSection {
    /// Whether buckets are graph-captured unless a bucket says otherwise. Defaults to true.
    #[serde(default = "default_capture")]
    pub capture: bool,
    /// Per-model-op overrides.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub overrides: BTreeMap<String, PlanOverride>,
}

fn default_capture() -> bool {
    true
}

impl Default for PlanSection {
    fn default() -> Self {
        Self {
            capture: true,
            overrides: BTreeMap::new(),
        }
    }
}

/// The whole manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// `[model]`
    pub model: ModelSection,
    /// `[device]`
    pub device: DeviceSection,
    /// `[[plugins]]`
    pub plugins: Vec<PluginSection>,
    /// `[[buckets]]`
    pub buckets: Vec<BucketSection>,
    /// `[plan]`
    #[serde(default)]
    pub plan: PlanSection,
}

impl Manifest {
    /// Parses and validates a manifest from text. `origin` names the source in error messages.
    pub fn from_str(text: &str, origin: &str) -> std::result::Result<Self, ManifestError> {
        let table: Table = toml::from_str(text).map_err(|e| ManifestError::Syntax {
            origin: origin.to_string(),
            source: Box::new(e),
        })?;
        Validator::new(origin).run(&table)
    }

    /// Reads, parses and validates a manifest file.
    pub fn from_path(path: &Path) -> std::result::Result<Self, ManifestError> {
        let text = std::fs::read_to_string(path).map_err(|source| ManifestError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_str(&text, &path.display().to_string())
    }

    /// Serializes back to TOML. Round-trips: parsing the output yields an equal manifest.
    ///
    /// # Panics
    ///
    /// Never for values this crate can construct — every field is a string, integer, boolean or a
    /// string-keyed table.
    pub fn to_toml_string(&self) -> String {
        toml::to_string_pretty(self).expect("manifest serializes to TOML infallibly")
    }

    /// Whether a bucket is captured, resolving `[[buckets]] capture` against `[plan] capture`.
    pub fn bucket_captured(&self, bucket: &BucketSection) -> bool {
        bucket.capture.unwrap_or(self.plan.capture)
    }

    /// The planner's bucket for a manifest section, with `capture` resolved.
    ///
    /// One place, because the field set is phase-dependent: a decode section carries
    /// `batch`/`spec_width` and a prefill one carries `tokens`, and a literal spelled out at
    /// each call site is a compile error in another crate every time a field is added.
    pub fn bucket(&self, section: &BucketSection) -> Bucket {
        Bucket {
            phase: section.phase,
            batch: section.batch,
            spec_width: section.spec_width,
            tokens: section.tokens,
            ctx_band: section.ctx_band,
            captured: self.bucket_captured(section),
        }
    }

    /// The plugin entry of that name.
    pub fn plugin(&self, name: &str) -> Option<&PluginSection> {
        self.plugins.iter().find(|p| p.name == name)
    }

    /// A digest over every declared plugin, in name order.
    ///
    /// This is the `plugin_build_hash` component of every autotune key. It covers *all* plugins
    /// rather than the one an implementation happens to come from, because the plugin is only known
    /// after selection while the key is needed before it — and because over-invalidating on a
    /// rebuild is the safe direction.
    pub fn plugin_build_digest(&self) -> String {
        let mut rows: Vec<(String, String)> = self
            .plugins
            .iter()
            .map(|p| (p.name.clone(), p.build_hash.clone()))
            .collect();
        rows.sort();
        crate::hash::digest_of(&rows)
    }

    /// Canonical digest of the manifest, folded into the plan's input hash.
    pub fn digest(&self) -> String {
        crate::hash::digest_of(self)
    }
}

impl fmt::Display for Manifest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_toml_string())
    }
}

/// Collects every problem in one pass, then builds the typed manifest.
struct Validator {
    origin: String,
    issues: Vec<String>,
}

const TOP_LEVEL_KEYS: &[&str] = &["buckets", "device", "model", "plan", "plugins"];
const MODEL_KEYS: &[&str] = &["drafter_tier_id", "path", "tier_id"];
const DEVICE_KEYS: &[&str] = &["power_cap_w", "target_gfx", "tp"];
const PLUGIN_KEYS: &[&str] = &["build_hash", "name", "path"];
const BUCKET_KEYS: &[&str] = &[
    "batch",
    "capture",
    "ctx_band",
    "phase",
    "spec_width",
    "tokens",
];
const PLAN_KEYS: &[&str] = &["capture", "overrides"];
const OVERRIDE_KEYS: &[&str] = &["impl", "variant_id"];

impl Validator {
    fn new(origin: &str) -> Self {
        Self {
            origin: origin.to_string(),
            issues: Vec::new(),
        }
    }

    fn issue(&mut self, msg: String) {
        self.issues.push(msg);
    }

    fn unknown_keys(&mut self, table: &Table, path: &str, allowed: &[&str]) {
        for key in table.keys() {
            if !allowed.contains(&key.as_str()) {
                let at = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                self.issue(format!(
                    "unknown key `{at}` (expected one of: {})",
                    allowed.join(", ")
                ));
            }
        }
    }

    fn section<'a>(&mut self, root: &'a Table, name: &str, required: bool) -> Option<&'a Table> {
        match root.get(name) {
            None => {
                if required {
                    self.issue(format!("missing required section `[{name}]`"));
                }
                None
            }
            Some(Value::Table(t)) => Some(t),
            Some(other) => {
                self.issue(format!(
                    "`{name}` must be a table, found {}",
                    type_of(other)
                ));
                None
            }
        }
    }

    fn array<'a>(&mut self, root: &'a Table, name: &str) -> Vec<&'a Table> {
        match root.get(name) {
            None => {
                self.issue(format!(
                    "missing required section `[[{name}]]`: a plan needs at least one entry"
                ));
                Vec::new()
            }
            Some(Value::Array(items)) => {
                if items.is_empty() {
                    self.issue(format!(
                        "`[[{name}]]` is empty: a plan needs at least one entry"
                    ));
                }
                items
                    .iter()
                    .enumerate()
                    .filter_map(|(i, v)| match v {
                        Value::Table(t) => Some(t),
                        other => {
                            self.issue(format!(
                                "`{name}[{i}]` must be a table, found {}",
                                type_of(other)
                            ));
                            None
                        }
                    })
                    .collect()
            }
            Some(other) => {
                self.issue(format!(
                    "`{name}` must be an array of tables, found {}",
                    type_of(other)
                ));
                Vec::new()
            }
        }
    }

    fn get_str(&mut self, t: &Table, path: &str, key: &str, required: bool) -> Option<String> {
        match t.get(key) {
            None => {
                if required {
                    self.issue(format!("missing required key `{path}.{key}`"));
                }
                None
            }
            Some(Value::String(s)) if s.is_empty() => {
                self.issue(format!("`{path}.{key}` must not be empty"));
                None
            }
            Some(Value::String(s)) => Some(s.clone()),
            Some(other) => {
                self.issue(format!(
                    "`{path}.{key}` must be a string, found {}",
                    type_of(other)
                ));
                None
            }
        }
    }

    fn get_int(
        &mut self,
        t: &Table,
        path: &str,
        key: &str,
        required: bool,
        min: i64,
        max: i64,
    ) -> Option<i64> {
        match t.get(key) {
            None => {
                if required {
                    self.issue(format!("missing required key `{path}.{key}`"));
                }
                None
            }
            Some(Value::Integer(v)) => {
                if *v < min || *v > max {
                    self.issue(format!(
                        "`{path}.{key}` must be in {min}..={max}, found {v}"
                    ));
                    None
                } else {
                    Some(*v)
                }
            }
            Some(other) => {
                self.issue(format!(
                    "`{path}.{key}` must be an integer, found {}",
                    type_of(other)
                ));
                None
            }
        }
    }

    fn get_bool(&mut self, t: &Table, path: &str, key: &str) -> Option<bool> {
        match t.get(key) {
            None => None,
            Some(Value::Boolean(b)) => Some(*b),
            Some(other) => {
                self.issue(format!(
                    "`{path}.{key}` must be a boolean, found {}",
                    type_of(other)
                ));
                None
            }
        }
    }

    fn run(mut self, root: &Table) -> std::result::Result<Manifest, ManifestError> {
        self.unknown_keys(root, "", TOP_LEVEL_KEYS);

        let model = self.section(root, "model", true).map(|t| {
            self.unknown_keys(t, "model", MODEL_KEYS);
            let path = self.get_str(t, "model", "path", true);
            let tier_id = self.get_str(t, "model", "tier_id", true);
            let drafter_tier_id = self.get_str(t, "model", "drafter_tier_id", false);
            (path, tier_id, drafter_tier_id)
        });

        let device = self.section(root, "device", true).map(|t| {
            self.unknown_keys(t, "device", DEVICE_KEYS);
            let target_gfx = self.get_str(t, "device", "target_gfx", true);
            let power_cap_w = self.get_int(t, "device", "power_cap_w", true, 1, 2000);
            let tp = self.get_int(t, "device", "tp", false, 1, 8).unwrap_or(1);
            (target_gfx, power_cap_w, tp)
        });

        let plugin_tables: Vec<Table> = self.array(root, "plugins").into_iter().cloned().collect();
        let mut plugins = Vec::new();
        for (i, t) in plugin_tables.iter().enumerate() {
            let path = format!("plugins[{i}]");
            self.unknown_keys(t, &path, PLUGIN_KEYS);
            let name = self.get_str(t, &path, "name", true);
            let file = self.get_str(t, &path, "path", true);
            let build_hash = self.get_str(t, &path, "build_hash", true);
            if let (Some(name), Some(file), Some(build_hash)) = (name, file, build_hash) {
                if plugins.iter().any(|p: &PluginSection| p.name == name) {
                    self.issue(format!("duplicate plugin name `{name}` at `{path}`"));
                }
                plugins.push(PluginSection {
                    name,
                    path: PathBuf::from(file),
                    build_hash,
                });
            }
        }

        let bucket_tables: Vec<Table> = self.array(root, "buckets").into_iter().cloned().collect();
        let mut buckets: Vec<BucketSection> = Vec::new();
        for (i, t) in bucket_tables.iter().enumerate() {
            let path = format!("buckets[{i}]");
            self.unknown_keys(t, &path, BUCKET_KEYS);
            let phase = match t.get("phase") {
                None => Some(Phase::Decode),
                Some(Value::String(v)) if v == "decode" => Some(Phase::Decode),
                Some(Value::String(v)) if v == "prefill" => Some(Phase::Prefill),
                Some(other) => {
                    self.issue(format!(
                        "`{path}.phase` must be \"decode\" or \"prefill\", found {}",
                        type_of(other)
                    ));
                    None
                }
            };
            let capture = self.get_bool(t, &path, "capture");
            let ctx_band = self.get_int(t, &path, "ctx_band", true, 1, 1 << 24);
            // A prefill bucket is M = tokens at batch 1 with no speculation, so it takes a
            // different key set. Asking for `spec_width` there -- or for `tokens` in a decode
            // bucket -- is a manifest error rather than a field that quietly reads zero: the two
            // shapes are captured separately and a bucket that names the wrong one would be
            // planned for a step that never runs.
            let (batch, spec_width, tokens) = match phase {
                Some(Phase::Prefill) => {
                    for k in ["batch", "spec_width"] {
                        if t.contains_key(k) {
                            self.issue(format!(
                                "`{path}.{k}` is a decode key; a prefill bucket is `tokens` rows \
                                 at batch 1 with no speculative width"
                            ));
                        }
                    }
                    let tokens = self.get_int(t, &path, "tokens", true, 1, 1 << 24);
                    (Some(1), Some(0), tokens)
                }
                _ => {
                    if t.contains_key("tokens") {
                        self.issue(format!(
                            "`{path}.tokens` is a prefill key; a decode bucket's rows are \
                             batch x spec_width"
                        ));
                    }
                    (
                        self.get_int(t, &path, "batch", true, 1, 4096),
                        self.get_int(t, &path, "spec_width", true, 1, 64),
                        Some(0),
                    )
                }
            };
            if let (Some(phase), Some(batch), Some(spec_width), Some(tokens), Some(ctx_band)) =
                (phase, batch, spec_width, tokens, ctx_band)
            {
                let (batch, spec_width, tokens, ctx_band) = (
                    batch as u32,
                    spec_width as u32,
                    tokens as u32,
                    ctx_band as u32,
                );
                if buckets.iter().any(|b| {
                    b.phase == phase
                        && b.batch == batch
                        && b.spec_width == spec_width
                        && b.tokens == tokens
                        && b.ctx_band == ctx_band
                }) {
                    let what = match phase {
                        Phase::Decode => {
                            format!("batch={batch} spec_width={spec_width} ctx_band={ctx_band}")
                        }
                        Phase::Prefill => format!("prefill tokens={tokens} ctx_band={ctx_band}"),
                    };
                    self.issue(format!("duplicate bucket {what} at `{path}`"));
                }
                buckets.push(BucketSection {
                    phase,
                    batch,
                    spec_width,
                    tokens,
                    ctx_band,
                    capture,
                });
            }
        }

        let mut plan = PlanSection::default();
        if let Some(t) = self.section(root, "plan", false).cloned() {
            self.unknown_keys(&t, "plan", PLAN_KEYS);
            if let Some(capture) = self.get_bool(&t, "plan", "capture") {
                plan.capture = capture;
            }
            match t.get("overrides") {
                None => {}
                Some(Value::Table(overrides)) => {
                    for (name, value) in overrides {
                        let path = format!("plan.overrides.{name}");
                        let Value::Table(entry) = value else {
                            self.issue(format!(
                                "`{path}` must be a table, found {}",
                                type_of(value)
                            ));
                            continue;
                        };
                        self.unknown_keys(entry, &path, OVERRIDE_KEYS);
                        let impl_name = self.get_str(entry, &path, "impl", false);
                        let variant_id = self
                            .get_int(entry, &path, "variant_id", false, 0, i64::from(u32::MAX))
                            .map(|v| v as u32);
                        if impl_name.is_none() && variant_id.is_none() {
                            self.issue(format!(
                                "`{path}` overrides nothing: set `impl`, `variant_id`, or both"
                            ));
                        }
                        plan.overrides.insert(
                            name.clone(),
                            PlanOverride {
                                impl_name,
                                variant_id,
                            },
                        );
                    }
                }
                Some(other) => self.issue(format!(
                    "`plan.overrides` must be a table, found {}",
                    type_of(other)
                )),
            }
        }

        if !self.issues.is_empty() {
            return Err(ManifestError::Invalid {
                origin: self.origin,
                issues: self.issues,
            });
        }

        // Every `Some` below is guaranteed by the absence of issues above.
        let (path, tier_id, drafter_tier_id) = model.expect("validated");
        let (target_gfx, power_cap_w, tp) = device.expect("validated");
        Ok(Manifest {
            model: ModelSection {
                path: PathBuf::from(path.expect("validated")),
                tier_id: tier_id.expect("validated"),
                drafter_tier_id,
            },
            device: DeviceSection {
                target_gfx: target_gfx.expect("validated"),
                power_cap_w: power_cap_w.expect("validated") as u32,
                tp: tp as u32,
            },
            plugins,
            buckets,
            plan,
        })
    }
}

fn type_of(v: &Value) -> &'static str {
    match v {
        Value::String(_) => "a string",
        Value::Integer(_) => "an integer",
        Value::Float(_) => "a float",
        Value::Boolean(_) => "a boolean",
        Value::Datetime(_) => "a datetime",
        Value::Array(_) => "an array",
        Value::Table(_) => "a table",
    }
}
