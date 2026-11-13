//! Server-side validation of a config before it is saved or applied.
//!
//! Pure functions over the config, the flag schema, the composed command line and the facts the
//! caller read from disk ([`Facts`]: the packs, the model shape, the estimate). Every issue
//! names its field (a long flag, or a config path such as `engine.port`) so the UI shows it at
//! the control. Errors block an apply; warnings do not.
//!
//! Two layers:
//! - per override: the flag exists for the preset's subcommand, the value has the flag's type,
//!   lies in its range ([`crate::schema::RANGES`]) and is one of its choices;
//! - cross-flag, on the EFFECTIVE command line (preset + agent defaults + overrides + extras):
//!   TP vs the pack's layout and the expert split, the KV drive tier needs a KV dir inside the
//!   container's writable mount, a drafter needs its pack (and a power-of-two block), port
//!   conflicts, the context against the model and the memory estimate, the vision flags.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;

use crate::compose::EngineCommand;
use crate::config::{Config, FlagValue, Preset};
use crate::estimate::{Estimate, ModelInfo, PackInfo};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Error,
    Warning,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Issue {
    /// `--fn-max-ctx`, or a config path (`engine.port`, `config`).
    pub field: String,
    pub level: Level,
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Report {
    pub ok: bool,
    pub errors: Vec<Issue>,
    pub warnings: Vec<Issue>,
}

impl Report {
    fn push(&mut self, level: Level, field: &str, message: impl Into<String>) {
        let i = Issue {
            field: field.to_string(),
            level,
            message: message.into(),
        };
        match level {
            Level::Error => self.errors.push(i),
            Level::Warning => self.warnings.push(i),
        }
    }
    fn err(&mut self, field: &str, m: impl Into<String>) {
        self.push(Level::Error, field, m);
    }
    fn warn(&mut self, field: &str, m: impl Into<String>) {
        self.push(Level::Warning, field, m);
    }
}

/// One flag as the validator reads it from the schema JSON.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub struct FlagInfo {
    pub name: String,
    pub section: String,
    pub kind: String,
    #[serde(default)]
    pub choices: Vec<String>,
    #[serde(default)]
    pub multiple: bool,
    #[serde(default)]
    pub delimiter: Option<String>,
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
    #[serde(default)]
    pub default: Option<String>,
}

/// The schema's flags by (section, name).
#[derive(Clone, Debug, Default)]
pub struct SchemaIndex {
    flags: BTreeMap<(String, String), FlagInfo>,
}

impl SchemaIndex {
    #[must_use]
    pub fn from_value(v: &serde_json::Value) -> SchemaIndex {
        let mut flags = BTreeMap::new();
        if let Some(a) = v["flags"].as_array() {
            for f in a {
                if let Ok(fi) = serde_json::from_value::<FlagInfo>(f.clone()) {
                    flags.insert((fi.section.clone(), fi.name.clone()), fi);
                }
            }
        }
        SchemaIndex { flags }
    }

    /// The flag for the preset's subcommand (or a global one).
    #[must_use]
    pub fn get(&self, preset: Preset, name: &str) -> Option<&FlagInfo> {
        self.flags
            .get(&(preset.subcommand().to_string(), name.to_string()))
            .or_else(|| self.flags.get(&("global".to_string(), name.to_string())))
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.flags.is_empty()
    }
}

/// What the caller read from disk for this config.
#[derive(Clone, Debug, Default)]
pub struct Facts {
    /// The console's own port (the engine may not take it).
    pub console_port: u16,
    /// The target pack (host path resolved), when the effective `--pack` names one.
    pub pack: Option<PackFact>,
    pub draft_pack: Option<PackFact>,
    pub model: Option<ModelInfo>,
    pub estimate: Option<Estimate>,
}

#[derive(Clone, Debug, Default)]
pub struct PackFact {
    /// The flag's value (container path for the 27B: `/packs/NAME`).
    pub arg: String,
    /// The host path it maps to.
    pub host_path: String,
    /// `None` = the directory does not exist.
    pub info: Option<PackInfo>,
}

/// The effective command line as `flag -> last value` (a bare switch = `Some(None)`).
#[must_use]
pub fn effective_map(cmd: &EngineCommand) -> BTreeMap<String, Option<String>> {
    crate::compose::parse_argv(&cmd.engine)
        .into_iter()
        .collect()
}

fn parse_num(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Type / range / choice check of one value word against its flag.
fn check_word(f: &FlagInfo, w: &str) -> Result<(), String> {
    let range = |v: f64| -> Result<(), String> {
        if let Some(lo) = f.min {
            if v < lo {
                return Err(format!("{v} is below the minimum {lo}"));
            }
        }
        if let Some(hi) = f.max {
            if v > hi {
                return Err(format!("{v} is above the maximum {hi}"));
            }
        }
        Ok(())
    };
    match f.kind.as_str() {
        "uint" => {
            let v: u64 = w
                .trim()
                .parse()
                .map_err(|_| format!("`{w}` is not a whole number >= 0"))?;
            range(v as f64)
        }
        "int" => {
            let v: i64 = w
                .trim()
                .parse()
                .map_err(|_| format!("`{w}` is not a whole number"))?;
            range(v as f64)
        }
        "float" => {
            let v = parse_num(w).ok_or_else(|| format!("`{w}` is not a number"))?;
            range(v)
        }
        "enum" => {
            if f.choices.iter().any(|c| c == w) {
                Ok(())
            } else {
                Err(format!("`{w}` is not one of {}", f.choices.join(", ")))
            }
        }
        "path" | "string" => {
            if w.trim().is_empty() {
                Err("the value is empty".into())
            } else {
                Ok(())
            }
        }
        _ => Ok(()),
    }
}

/// The per-override checks.
fn check_override(r: &mut Report, f: &FlagInfo, v: &FlagValue) {
    let name = f.name.as_str();
    match v {
        // `false` removes the flag from the command line: valid for any flag
        FlagValue::Bool(false) => {}
        FlagValue::Bool(true) => {
            // a bare switch: a bool, or an enum / optional flag with a bare form (`--vision`)
            if f.kind != "bool" && f.kind != "enum" {
                r.err(
                    name,
                    format!("{name} takes a {} value, not a bare switch", f.kind),
                );
            }
        }
        _ if f.kind == "bool" => r.err(
            name,
            format!("{name} is a switch: on (true) or off (false)"),
        ),
        FlagValue::List(vs) => {
            if !f.multiple {
                r.err(name, format!("{name} takes one value, not a list"));
                return;
            }
            if vs.is_empty() {
                r.err(name, "the list is empty");
            }
            for w in vs {
                if let Err(e) = check_word(f, w) {
                    r.err(name, e);
                }
            }
        }
        FlagValue::Float(x) if f.kind == "uint" || f.kind == "int" => {
            r.err(name, format!("{x} is not a whole number"));
        }
        other => {
            let s = other.arg().unwrap_or_default();
            let words: Vec<&str> = match (&f.delimiter, f.multiple) {
                (Some(d), true) => s.split(d.as_str()).collect(),
                _ => vec![s.as_str()],
            };
            for w in words {
                if let Err(e) = check_word(f, w) {
                    r.err(name, e);
                }
            }
        }
    }
}

fn val<'a>(m: &'a BTreeMap<String, Option<String>>, k: &str) -> Option<&'a str> {
    m.get(k).and_then(|v| v.as_deref())
}
fn num(m: &BTreeMap<String, Option<String>>, k: &str) -> Option<f64> {
    val(m, k).and_then(parse_num)
}
fn has(m: &BTreeMap<String, Option<String>>, k: &str) -> bool {
    m.contains_key(k)
}

/// The tp the effective command runs.
#[must_use]
pub fn effective_tp(preset: Preset, m: &BTreeMap<String, Option<String>>) -> u32 {
    let k = match preset {
        Preset::Flashnext => "--tp",
        Preset::Qwen27b | Preset::Mock => "--engine-tp",
    };
    num(m, k).map_or(1, |v| v as u32)
}

/// Validate `cfg` (pure). `cmd` is `cfg.engine` composed; `schema` is the flag schema.
#[must_use]
pub fn validate(cfg: &Config, schema: &SchemaIndex, cmd: &EngineCommand, facts: &Facts) -> Report {
    let mut r = Report::default();
    if let Err(e) = cfg.validate() {
        let msg = e.to_string();
        let field = if msg.starts_with("engine.port") {
            "engine.port"
        } else {
            "config"
        };
        r.err(field, msg);
    }
    let e = &cfg.engine;
    let preset = e.preset;

    // ---- per override ----
    for (k, v) in &e.flags {
        if !k.starts_with("--") {
            continue; // Config::validate said it
        }
        if schema.is_empty() {
            break;
        }
        match schema.get(preset, k) {
            None => r.err(
                k,
                format!(
                    "{k} is not a flag of `saber {}` (the {:?} preset)",
                    preset.subcommand(),
                    preset
                ),
            ),
            Some(f) => check_override(&mut r, f, v),
        }
    }

    // ---- ports ----
    let port_flag = match preset {
        Preset::Flashnext => "--fn-serve",
        Preset::Qwen27b | Preset::Mock => "--port",
    };
    if let Some(v) = e.flags.get(port_flag) {
        if v.arg().as_deref() != Some(e.port.to_string().as_str()) {
            r.err(
                port_flag,
                format!(
                    "the engine's port is engine.port ({}); {port_flag} may not differ (the console probes engine.port)",
                    e.port
                ),
            );
        }
    }
    if facts.console_port != 0 && e.port == facts.console_port {
        r.err(
            "engine.port",
            format!("port {} is the console's own port", e.port),
        );
    }
    if e.port != 0 && e.port < 1024 {
        r.warn("engine.port", "a port below 1024 needs root to bind");
    }

    let m = effective_map(cmd);
    let tp = effective_tp(preset, &m);

    // ---- model / pack / parallelism ----
    if preset.uses_gpus() {
        match &facts.pack {
            Some(PackFact {
                info: None,
                arg,
                host_path,
            }) => r.err(
                "--pack",
                format!("the pack {arg} does not exist (host path {host_path})"),
            ),
            Some(PackFact {
                info: Some(info), ..
            }) => {
                if let Some(err) = &info.error {
                    r.warn("--pack", format!("the pack's {err}"));
                }
                if info.role == "drafter" {
                    r.err(
                        "--pack",
                        format!("{} is a drafter pack, not a target", info.name),
                    );
                }
                if let Some(pt) = info.tp_size {
                    if preset == Preset::Qwen27b && pt != tp {
                        r.err(
                            "--engine-tp",
                            format!(
                                "--engine-tp {tp} does not match the pack's tp_size {pt} ({})",
                                info.name
                            ),
                        );
                    }
                    if preset == Preset::Flashnext && tp == 1 && pt == 2 {
                        r.warn(
                            "--tp",
                            format!(
                                "--tp 1 on a TP=2 / EP=2 pack ({}) is the gate harness's form, not the served boot",
                                info.name
                            ),
                        );
                    }
                }
            }
            None => {}
        }
    }
    if preset == Preset::Flashnext {
        if tp == 1 && val(&m, "--fn-experts") == Some("tp2") {
            r.err(
                "--fn-experts",
                "--fn-experts tp2 splits every expert across two ranks: it needs --tp 2",
            );
        }
        if tp == 1 && val(&m, "--fn-kv-tier") == Some("on") {
            r.err(
                "--fn-kv-tier",
                "the tiered KV runs in the worker form at TP=2 only (--tp 2)",
            );
        }
    }

    // ---- the KV tiers ----
    if preset == Preset::Flashnext {
        let drive = ["--fn-kv-ssd-gib", "--fn-kv-pcache-gib"]
            .into_iter()
            .filter(|k| num(&m, k).is_some_and(|v| v > 0.0))
            .collect::<Vec<_>>();
        let dir = val(&m, "--fn-kv-dir").map(str::to_string).or_else(|| {
            schema
                .get(preset, "--fn-kv-dir")
                .and_then(|f| f.default.clone())
        });
        for k in &drive {
            match &dir {
                None => r.err(k, format!("{k} puts KV on the drive: it needs --fn-kv-dir")),
                Some(d) if d.trim().is_empty() => {
                    r.err(
                        "--fn-kv-dir",
                        format!("{k} needs a KV dir; --fn-kv-dir is empty"),
                    );
                }
                Some(d) => {
                    let mount = e.paths.nvme.join("saber-kv");
                    if e.runtime == crate::config::Runtime::Container
                        && !Path::new(d).starts_with(&mount)
                    {
                        r.err(
                            "--fn-kv-dir",
                            format!(
                                "{d} is outside the container's writable KV mount {}: the engine could not write it",
                                mount.display()
                            ),
                        );
                    }
                }
            }
        }
        if has(&m, "--fn-kv-ram-gib") && has(&m, "--fn-kv-admit-slab-tokens") {
            r.err(
                "--fn-kv-admit-slab-tokens",
                "--fn-kv-admit-slab-tokens and --fn-kv-ram-gib are exclusive (the token form and the byte form of one slab)",
            );
        }
        if val(&m, "--fn-kv-tier") == Some("off")
            && num(&m, "--fn-max-ctx").is_some_and(|c| c > 16_384.0)
        {
            r.warn(
                "--fn-max-ctx",
                "with the KV tier off, a context past 16,384 is paid for in VRAM expert slots",
            );
        }
    }

    // ---- the drafter ----
    if matches!(preset, Preset::Qwen27b | Preset::Mock) {
        let block = num(&m, "--engine-draft-block").unwrap_or(0.0) as u64;
        let dpack = val(&m, "--engine-draft-pack");
        if block > 0 && dpack.is_none() && preset == Preset::Qwen27b {
            r.err(
                "--engine-draft-pack",
                format!("--engine-draft-block {block} plans a drafter tier: it needs --engine-draft-pack"),
            );
        }
        if block > 0 && !block.is_power_of_two() {
            r.err(
                "--engine-draft-block",
                format!("the draft block must be a power of two, not {block}"),
            );
        }
        if block == 0 && dpack.is_some() {
            r.warn(
                "--engine-draft-pack",
                "a drafter pack without --engine-draft-block plans no drafter tier",
            );
        }
        if preset == Preset::Qwen27b {
            match &facts.draft_pack {
                Some(PackFact {
                    info: None,
                    arg,
                    host_path,
                }) => r.err(
                    "--engine-draft-pack",
                    format!("the drafter pack {arg} does not exist (host path {host_path})"),
                ),
                Some(PackFact {
                    info: Some(info), ..
                }) => {
                    if info.role != "drafter" {
                        r.warn(
                            "--engine-draft-pack",
                            format!("{} does not look like a drafter pack", info.name),
                        );
                    }
                    if let Some(pt) = info.tp_size {
                        if pt != tp {
                            r.err(
                                "--engine-draft-pack",
                                format!("the drafter pack is tp_size {pt}; the target runs --engine-tp {tp}"),
                            );
                        }
                    }
                }
                None => {}
            }
        }
    }

    // ---- context and sequences ----
    let (ctx_flag, ctx) = match preset {
        Preset::Flashnext => ("--fn-max-ctx", num(&m, "--fn-max-ctx")),
        Preset::Qwen27b | Preset::Mock => ("--engine-max-ctx", num(&m, "--engine-max-ctx")),
    };
    let ctx = ctx.or_else(|| {
        schema
            .get(preset, ctx_flag)
            .and_then(|f| f.default.as_deref().and_then(parse_num))
    });
    if let (Some(c), Some(model)) = (ctx, &facts.model) {
        if let Some(mp) = model.max_position {
            if c as u64 > mp {
                r.err(
                    ctx_flag,
                    format!("{c} tokens is past the model's max_position_embeddings ({mp})"),
                );
            }
        }
    }
    if let Some(est) = &facts.estimate {
        if let Some(s) = est.seqs_at_ctx {
            let field = if preset == Preset::Flashnext {
                ctx_flag
            } else {
                "--engine-max-seqs"
            };
            if s == 0 {
                r.warn(
                    ctx_flag,
                    format!(
                        "estimate: not even one sequence fits at {} tokens (see the memory estimate)",
                        est.max_ctx
                    ),
                );
            } else if s < est.requested_seqs && preset != Preset::Flashnext {
                r.warn(
                    field,
                    format!(
                        "estimate: {s} of {} sequences fit at the full {}-token context",
                        est.requested_seqs, est.max_ctx
                    ),
                );
            }
        }
        if !est.fits && est.seqs_at_ctx != Some(0) {
            for w in &est.warnings {
                r.warn("estimate", format!("estimate: {w}"));
            }
        }
    }
    if preset != Preset::Flashnext {
        if let (Some(c), Some(s)) = (ctx, num(&m, "--engine-max-seqs")) {
            if c * s > 16_777_216.0 {
                r.warn(
                    "--engine-max-seqs",
                    format!(
                        "{s} sequences x {c} tokens = {} tokens of KV: check the estimate",
                        c * s
                    ),
                );
            }
        }
    }

    // ---- vision ----
    match preset {
        Preset::Flashnext => {
            let vis = val(&m, "--fn-vision").or(if has(&m, "--fn-vision") {
                Some("on")
            } else {
                None
            });
            let on = matches!(vis, Some("on" | "lazy"));
            if on {
                if let Some(PackFact {
                    info: Some(info), ..
                }) = &facts.pack
                {
                    if !info.has_vision {
                        r.err(
                            "--fn-vision",
                            format!(
                                "the pack {} has no vision/ member: images cannot be served",
                                info.name
                            ),
                        );
                    }
                }
            }
            if !on && has(&m, "--fn-vision-max-patches") {
                r.warn(
                    "--fn-vision-max-patches",
                    "inert without --fn-vision on|lazy",
                );
            }
            if !on
                && val(&m, "--fn-image-cache") == Some("on")
                && e.flags.contains_key("--fn-image-cache")
            {
                r.warn(
                    "--fn-image-cache",
                    "inert without a vision tower (--fn-vision)",
                );
            }
        }
        Preset::Qwen27b | Preset::Mock => {
            let vis = val(&m, "--vision").or(if has(&m, "--vision") {
                Some("lazy")
            } else {
                None
            });
            let on = matches!(vis, Some("on" | "lazy"));
            if preset == Preset::Mock && on {
                r.warn(
                    "--vision",
                    "the mock engine serves no images: --vision is inert",
                );
            }
            if !on && has(&m, "--vision-max-patches") {
                r.warn("--vision-max-patches", "inert without --vision");
            }
            if !on && e.flags.contains_key("--limit-mm-images") {
                r.warn("--limit-mm-images", "inert without --vision");
            }
        }
    }

    // ---- server exposure ----
    if matches!(preset, Preset::Qwen27b | Preset::Mock)
        && val(&m, "--host").is_some_and(|h| !crate::is_loopback(h))
        && !has(&m, "--api-key")
        && !has(&m, "--api-key-file")
    {
        r.warn(
            "--host",
            "the engine listens beyond loopback with no --api-key: anyone on the network can use it",
        );
    }

    r.ok = r.errors.is_empty();
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compose::{compose, ComposeCtx};
    use std::path::PathBuf;

    fn schema() -> SchemaIndex {
        let f = |section: &str, name: &str, kind: &str| {
            let r = crate::schema::range_of(name);
            serde_json::json!({"name": name, "section": section, "kind": kind,
                "min": r.map(|r| r.0), "max": r.map(|r| r.1)})
        };
        let mut flags = vec![
            f("engine", "--tp", "uint"),
            f("engine", "--fn-max-ctx", "uint"),
            f("engine", "--fn-serve", "uint"),
            f("engine", "--fn-kv-ssd-gib", "float"),
            f("engine", "--fn-kv-pcache-gib", "float"),
            f("engine", "--fn-kv-ram-gib", "float"),
            f("engine", "--fn-kv-admit-slab-tokens", "uint"),
            f("engine", "--fn-kv-tier", "string"),
            f("engine", "--fn-vision-max-patches", "uint"),
            f("engine", "--pack", "path"),
            f("serve", "--engine-tp", "uint"),
            f("serve", "--engine-max-ctx", "uint"),
            f("serve", "--engine-max-seqs", "uint"),
            f("serve", "--engine-draft-block", "uint"),
            f("serve", "--engine-draft-pack", "path"),
            f("serve", "--port", "uint"),
            f("serve", "--host", "string"),
            f("serve", "--engine-rectangular-widths", "bool"),
            f("serve", "--limit-mm-images", "uint"),
            f("global", "--verbose", "bool"),
        ];
        flags.push(
            serde_json::json!({"name": "--fn-kv-dir", "section": "engine", "kind": "path",
            "default": "/mnt/nvme/saber-kv"}),
        );
        flags.push(
            serde_json::json!({"name": "--fn-experts", "section": "engine", "kind": "enum",
            "choices": ["ep", "tp2"]}),
        );
        flags.push(
            serde_json::json!({"name": "--fn-vision", "section": "engine", "kind": "enum",
            "choices": ["on", "lazy", "off"]}),
        );
        flags.push(
            serde_json::json!({"name": "--vision", "section": "serve", "kind": "enum",
            "choices": ["lazy", "on", "off"]}),
        );
        flags.push(
            serde_json::json!({"name": "--api-key", "section": "serve", "kind": "string",
            "multiple": true}),
        );
        flags.push(
            serde_json::json!({"name": "--engine-decode-batches", "section": "serve",
            "kind": "uint", "multiple": true, "delimiter": ","}),
        );
        SchemaIndex::from_value(&serde_json::json!({ "flags": flags }))
    }

    fn ctx() -> ComposeCtx {
        ComposeCtx {
            home: PathBuf::from("/h"),
            exe: PathBuf::from("/bin/saber"),
            fn3_prompts_present: false,
            fn_legacy_pack_present: false,
            p2p_mounts: vec![],
        }
    }

    fn run(cfg: &Config, facts: &Facts) -> Report {
        validate(cfg, &schema(), &compose(&cfg.engine, &ctx()), facts)
    }

    fn fields(r: &[Issue]) -> Vec<&str> {
        r.iter().map(|i| i.field.as_str()).collect()
    }

    fn pack(tp: u32, role: &'static str, vision: bool) -> PackFact {
        PackFact {
            arg: "/p".into(),
            host_path: "/p".into(),
            info: Some(PackInfo {
                name: "p".into(),
                tp_size: Some(tp),
                role,
                has_vision: vision,
                ..PackInfo::default()
            }),
        }
    }

    #[test]
    fn the_stock_presets_validate() {
        for c in [Config::default(), Config::mock(8099)] {
            let r = run(&c, &Facts::default());
            assert!(r.ok, "{:?}", r.errors);
        }
        let mut c = Config::default();
        c.engine.preset = Preset::Qwen27b;
        let r = run(&c, &Facts::default());
        assert!(r.ok, "{:?}", r.errors);
    }

    #[test]
    fn types_ranges_choices_and_unknown_flags() {
        let mut c = Config::default();
        let fl = &mut c.engine.flags;
        fl.insert("--fn-max-ctx".into(), FlagValue::Str("lots".into()));
        fl.insert("--tp".into(), FlagValue::Int(3));
        fl.insert("--fn-experts".into(), FlagValue::Str("both".into()));
        fl.insert("--fn-kv-ssd-gib".into(), FlagValue::Str("x".into()));
        fl.insert("--engine-max-seqs".into(), FlagValue::Int(2)); // a serve flag
        fl.insert("--verbose".into(), FlagValue::Str("loud".into()));
        fl.insert("--pack".into(), FlagValue::Bool(true));
        let r = run(&c, &Facts::default());
        let f = fields(&r.errors);
        for want in [
            "--fn-max-ctx",
            "--tp",
            "--fn-experts",
            "--fn-kv-ssd-gib",
            "--engine-max-seqs",
            "--verbose",
            "--pack",
        ] {
            assert!(f.contains(&want), "{want} missing from {f:?}");
        }
        assert!(!r.ok);
        // removing a flag (false) is valid for any flag; a float for a uint is not
        let mut c = Config::default();
        c.engine
            .flags
            .insert("--fn-max-ctx".into(), FlagValue::Bool(false));
        assert!(run(&c, &Facts::default()).ok);
        c.engine
            .flags
            .insert("--fn-max-ctx".into(), FlagValue::Float(1.5));
        assert!(!run(&c, &Facts::default()).ok);
        c.engine
            .flags
            .insert("--fn-max-ctx".into(), FlagValue::Int(100));
        let r = run(&c, &Facts::default());
        assert!(r.errors[0].message.contains("below the minimum"), "{r:?}");
        // a delimited list checks each element; a repeated list needs a multiple flag
        let mut m = Config::mock(8099);
        m.engine.flags.insert(
            "--engine-decode-batches".into(),
            FlagValue::Str("1,2,x".into()),
        );
        m.engine
            .flags
            .insert("--engine-max-ctx".into(), FlagValue::List(vec!["1".into()]));
        let r = run(&m, &Facts::default());
        assert_eq!(
            fields(&r.errors),
            ["--engine-decode-batches", "--engine-max-ctx"]
        );
        m.engine.flags.clear();
        m.engine.flags.insert(
            "--api-key".into(),
            FlagValue::List(vec!["a=12345678".into(), "b=12345678".into()]),
        );
        assert!(run(&m, &Facts::default()).ok);
    }

    #[test]
    fn ports() {
        let mut c = Config::mock(8099);
        let f = Facts {
            console_port: 8099,
            ..Facts::default()
        };
        assert_eq!(fields(&run(&c, &f).errors), ["engine.port"]);
        c.engine.flags.insert("--port".into(), FlagValue::Int(9000));
        let r = run(&c, &Facts::default());
        assert_eq!(fields(&r.errors), ["--port"]);
        c.engine.flags.clear();
        c.engine.port = 80;
        assert_eq!(
            fields(&run(&c, &Facts::default()).warnings),
            ["engine.port"]
        );
        c.engine.port = 0;
        assert_eq!(fields(&run(&c, &Facts::default()).errors), ["engine.port"]);
    }

    #[test]
    fn tp_vs_pack_and_expert_split() {
        let mut c = Config::default();
        c.engine.preset = Preset::Qwen27b;
        let f = Facts {
            pack: Some(pack(1, "target", false)),
            ..Facts::default()
        };
        assert_eq!(fields(&run(&c, &f).errors), ["--engine-tp"]);
        let f = Facts {
            pack: Some(pack(2, "drafter", false)),
            ..Facts::default()
        };
        assert_eq!(fields(&run(&c, &f).errors), ["--pack"]);
        let f = Facts {
            pack: Some(PackFact {
                arg: "/packs/x".into(),
                host_path: "/n/x".into(),
                info: None,
            }),
            ..Facts::default()
        };
        assert!(run(&c, &f).errors[0].message.contains("does not exist"));
        // Flash-Next: tp 1 with the tp2 expert split, the KV tier at tp 1
        let mut c = Config::default();
        c.engine.flags.insert("--tp".into(), FlagValue::Int(1));
        let f = Facts {
            pack: Some(pack(2, "target", false)),
            ..Facts::default()
        };
        let r = run(&c, &f);
        assert_eq!(fields(&r.errors), ["--fn-experts"]);
        assert_eq!(fields(&r.warnings), ["--tp"]);
        c.engine
            .flags
            .insert("--fn-experts".into(), FlagValue::Str("ep".into()));
        c.engine
            .flags
            .insert("--fn-kv-tier".into(), FlagValue::Str("on".into()));
        assert_eq!(fields(&run(&c, &f).errors), ["--fn-kv-tier"]);
    }

    #[test]
    fn kv_drive_tier_needs_a_kv_dir_in_the_mount() {
        let mut c = Config::default();
        c.engine
            .flags
            .insert("--fn-kv-ssd-gib".into(), FlagValue::Float(64.0));
        // the schema default (/mnt/nvme/saber-kv) is in the mount
        assert!(run(&c, &Facts::default()).ok);
        c.engine
            .flags
            .insert("--fn-kv-dir".into(), FlagValue::Str("/home/me/kv".into()));
        let r = run(&c, &Facts::default());
        assert_eq!(fields(&r.errors), ["--fn-kv-dir"]);
        assert!(r.errors[0].message.contains("outside"));
        c.engine
            .flags
            .insert("--fn-kv-dir".into(), FlagValue::Bool(false));
        let mut s = schema();
        s.flags
            .get_mut(&("engine".into(), "--fn-kv-dir".into()))
            .unwrap()
            .default = None;
        let r = validate(&c, &s, &compose(&c.engine, &ctx()), &Facts::default());
        assert_eq!(fields(&r.errors), ["--fn-kv-ssd-gib"]);
        // the exclusive slab forms
        let mut c = Config::default();
        c.engine
            .flags
            .insert("--fn-kv-ram-gib".into(), FlagValue::Float(3.0));
        c.engine
            .flags
            .insert("--fn-kv-admit-slab-tokens".into(), FlagValue::Int(1000));
        assert_eq!(
            fields(&run(&c, &Facts::default()).errors),
            ["--fn-kv-admit-slab-tokens"]
        );
    }

    #[test]
    fn drafter_needs_its_pack() {
        let mut c = Config::default();
        c.engine.preset = Preset::Qwen27b;
        c.engine
            .flags
            .insert("--engine-draft-pack".into(), FlagValue::Bool(false));
        assert_eq!(
            fields(&run(&c, &Facts::default()).errors),
            ["--engine-draft-pack"]
        );
        c.engine.flags.insert(
            "--engine-draft-pack".into(),
            FlagValue::Str("/packs/d".into()),
        );
        c.engine
            .flags
            .insert("--engine-draft-block".into(), FlagValue::Int(6));
        assert_eq!(
            fields(&run(&c, &Facts::default()).errors),
            ["--engine-draft-block"]
        );
        c.engine
            .flags
            .insert("--engine-draft-block".into(), FlagValue::Int(8));
        let f = Facts {
            draft_pack: Some(pack(1, "drafter", false)),
            ..Facts::default()
        };
        assert_eq!(fields(&run(&c, &f).errors), ["--engine-draft-pack"]);
        let f = Facts {
            draft_pack: Some(pack(2, "target", false)),
            ..Facts::default()
        };
        let r = run(&c, &f);
        assert!(r.ok);
        // (the 27B launcher binds 0.0.0.0 with no key: that warning stands too)
        assert_eq!(fields(&r.warnings), ["--engine-draft-pack", "--host"]);
        c.engine
            .flags
            .insert("--engine-draft-block".into(), FlagValue::Int(0));
        assert!(fields(&run(&c, &Facts::default()).warnings).contains(&"--engine-draft-pack"));
    }

    #[test]
    fn ctx_against_the_model_and_the_estimate() {
        let mut c = Config::mock(8099);
        c.engine
            .flags
            .insert("--engine-max-ctx".into(), FlagValue::Int(500_000));
        let f = Facts {
            model: Some(ModelInfo {
                max_position: Some(262_144),
                ..ModelInfo::default()
            }),
            estimate: Some(Estimate {
                seqs_at_ctx: Some(1),
                requested_seqs: 8,
                max_ctx: 500_000,
                fits: true,
                ..Estimate::default()
            }),
            ..Facts::default()
        };
        let r = run(&c, &f);
        assert_eq!(fields(&r.errors), ["--engine-max-ctx"]);
        assert_eq!(fields(&r.warnings), ["--engine-max-seqs"]);
        let f = Facts {
            estimate: Some(Estimate {
                seqs_at_ctx: Some(0),
                ..Estimate::default()
            }),
            ..Facts::default()
        };
        assert_eq!(fields(&run(&c, &f).warnings), ["--engine-max-ctx"]);
        c.engine
            .flags
            .insert("--engine-max-seqs".into(), FlagValue::Int(64));
        assert!(fields(&run(&c, &Facts::default()).warnings).contains(&"--engine-max-seqs"));
    }

    #[test]
    fn vision_flags() {
        let mut c = Config::default();
        c.engine
            .flags
            .insert("--fn-vision".into(), FlagValue::Str("on".into()));
        let f = Facts {
            pack: Some(pack(2, "target", false)),
            ..Facts::default()
        };
        assert_eq!(fields(&run(&c, &f).errors), ["--fn-vision"]);
        let f = Facts {
            pack: Some(pack(2, "target", true)),
            ..Facts::default()
        };
        assert!(run(&c, &f).ok);
        c.engine
            .flags
            .insert("--fn-vision".into(), FlagValue::Str("off".into()));
        c.engine
            .flags
            .insert("--fn-vision-max-patches".into(), FlagValue::Int(4096));
        assert_eq!(fields(&run(&c, &f).warnings), ["--fn-vision-max-patches"]);
        let mut m = Config::mock(8099);
        m.engine
            .flags
            .insert("--vision".into(), FlagValue::Bool(true));
        assert_eq!(fields(&run(&m, &Facts::default()).warnings), ["--vision"]);
        m.engine.flags.clear();
        m.engine
            .flags
            .insert("--limit-mm-images".into(), FlagValue::Int(4));
        assert_eq!(
            fields(&run(&m, &Facts::default()).warnings),
            ["--limit-mm-images"]
        );
    }

    #[test]
    fn exposed_engine_without_a_key_warns() {
        let mut c = Config::mock(8099);
        c.engine
            .flags
            .insert("--host".into(), FlagValue::Str("0.0.0.0".into()));
        assert_eq!(fields(&run(&c, &Facts::default()).warnings), ["--host"]);
        c.engine
            .flags
            .insert("--api-key".into(), FlagValue::Str("a=12345678".into()));
        assert!(run(&c, &Facts::default()).warnings.is_empty());
    }
}
