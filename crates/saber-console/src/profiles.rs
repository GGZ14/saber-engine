//! Profiles: named sets of flag overrides over the stock presets.
//!
//! - **Built-in** profiles ([`builtin`], [`builtin_for`]): Balanced / Max throughput / Lowest
//!   latency / Long context / Low power, one set per serving subcommand (`engine` = Flash-Next,
//!   `serve` = the 27B and the mock) and, for `serve`, per TP (one card's KV pool is far smaller
//!   than two cards', so the TP=1 set asks for less). A profile never changes the TP: it is
//!   `engine.tp`, chosen beside the model. Applying one keeps the config's model and server flags ([`STICKY`]) and
//!   replaces every other override with the profile's. **No profile touches the GPU power cap**
//!   (the host's own policy; the console never writes it): Low power only picks power-saving
//!   arms such as the drive keep-awake OFF.
//! - **User** profiles: a snapshot of a config's engine overrides (flags, extra arguments, the
//!   agent defaults), saved / renamed / duplicated / deleted in `<data dir>/profiles.toml`, and
//!   imported / exported as TOML. Applying one replaces the overrides whole.
//!
//! [`apply_profile`] is the composition (pure); the store does the file I/O.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::config::{Config, FlagValue};

/// Flags a built-in profile never replaces: what model is served and how it is reached.
pub const STICKY: &[&str] = &[
    "--pack",
    "--engine-draft-pack",
    "--draft-pack",
    "--fn-served-name",
    "--served-model-name",
    "--host",
    "--api-key",
    "--api-key-file",
    "--fn-kv-dir",
    "--tokenizer",
    "--chat-template",
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// The subcommand its flags belong to (`engine` / `serve`).
    pub section: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub builtin: bool,
    #[serde(default)]
    pub flags: BTreeMap<String, FlagValue>,
    #[serde(default)]
    pub extra_args: Vec<String>,
    /// Flash-Next's agent-load defaults (`None` = leave the config's choice).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_defaults: Option<bool>,
}

fn p(name: &str, section: &str, description: &str, flags: &[(&str, FlagValue)]) -> Profile {
    Profile {
        name: name.to_string(),
        description: description.to_string(),
        section: section.to_string(),
        builtin: true,
        flags: flags
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect(),
        extra_args: Vec::new(),
        agent_defaults: None,
    }
}

/// The built-in profiles of `section` at TP=2.
#[must_use]
pub fn builtin(section: &str) -> Vec<Profile> {
    builtin_for(section, 2)
}

/// The built-in profiles of `section` for a config at `tp` (the same five names at either TP;
/// `serve` at TP=1 sizes them for one card).
#[must_use]
pub fn builtin_for(section: &str, tp: u32) -> Vec<Profile> {
    use FlagValue::{Int, Str};
    if section == "serve" && tp == 1 {
        return vec![
            p(
                "Balanced",
                "serve",
                "The single-card launcher's defaults, unchanged (tools/serve_27b.sh --tp 1): 4 \
                 sequences at a 65,536-token context, decode batches 1,2,4.",
                &[],
            ),
            p(
                "Max throughput",
                "serve",
                "4 sequences at a 16,384-token context, so all four fit at full context in the one \
                 card's KV pool.",
                &[("--engine-max-ctx", Int(16_384))],
            ),
            p(
                "Lowest latency",
                "serve",
                "2 sequences: each running request keeps a small decode batch, the rest queue.",
                &[("--engine-max-seqs", Int(2))],
            ),
            p(
                "Long context",
                "serve",
                "A 131,072-token context with 1 sequence: the longest conversation one card holds. \
                 Check the memory estimate: a larger single-card pack leaves a smaller pool.",
                &[
                    ("--engine-max-ctx", Int(131_072)),
                    ("--engine-max-seqs", Int(1)),
                ],
            ),
            p(
                "Low power",
                "serve",
                "2 sequences (less sustained load). The GPU power cap is never changed by the console.",
                &[("--engine-max-seqs", Int(2))],
            ),
        ];
    }
    match section {
        "engine" => vec![
            p(
                "Balanced",
                "engine",
                "The launcher's served defaults, unchanged: tiered fp8 KV, MTP auto, the prefix tree, \
                 the agent-load defaults. The reference every other profile is measured against.",
                &[],
            ),
            p(
                "Max throughput",
                "engine",
                "Caps the context at 32,768 tokens so all 8 sequence slots fit at full context in the \
                 admission slab (more concurrent requests, less pinned RAM). Long prompts past 32k are refused.",
                &[("--fn-max-ctx", Int(32_768))],
            ),
            p(
                "Lowest latency",
                "engine",
                "Turns ON the drive keep-awake (one 4 KiB read every 40 ms keeps the NVMe out of its deep \
                 idle state, whose 8.9 ms wake stalls the first step after a pause): steadier update p99 \
                 for a little drive power. The context stays at the default.",
                &[("--fn-ple-drive-keepalive-ms", Int(40))],
            ),
            p(
                "Long context",
                "engine",
                "A 262,144-token context on the tiered KV (the hot pool in VRAM, the pages in pinned host \
                 RAM). The admission slab holds one full-context request plus one 10,240-token request; \
                 check the memory estimate for the RAM it pins.",
                &[("--fn-max-ctx", Int(262_144)), ("--fn-kv-tier", Str("on".into()))],
            ),
            p(
                "Low power",
                "engine",
                "Power-saving arms only: the drive keep-awake OFF (the drive follows its own idle states), \
                 the vision tower released after 60 s idle. The GPU power cap is never changed by the \
                 console.",
                &[
                    ("--fn-ple-drive-keepalive-ms", Int(0)),
                    ("--fn-vision-idle-s", Int(60)),
                ],
            ),
        ],
        "serve" => vec![
            p(
                "Balanced",
                "serve",
                "The launcher's served defaults, unchanged (the 27B: DFlash2 drafter, widths 1,5,8, 8 \
                 sequences at 262,144 tokens).",
                &[],
            ),
            p(
                "Max throughput",
                "serve",
                "8 sequences at a 32,768-token context: the KV pool holds every sequence at full context.",
                &[("--engine-max-seqs", Int(8)), ("--engine-max-ctx", Int(32_768))],
            ),
            p(
                "Lowest latency",
                "serve",
                "2 sequences: each running request keeps a small decode batch (the b1/b2 step), the rest \
                 queue.",
                &[("--engine-max-seqs", Int(2))],
            ),
            p(
                "Long context",
                "serve",
                "A 262,144-token context with 2 sequences, so both fit at full context.",
                &[("--engine-max-ctx", Int(262_144)), ("--engine-max-seqs", Int(2))],
            ),
            p(
                "Low power",
                "serve",
                "2 sequences (less sustained load). The GPU power cap is never changed by the console.",
                &[("--engine-max-seqs", Int(2))],
            ),
        ],
        _ => Vec::new(),
    }
}

/// Compose `profile` over `cfg` (pure): a built-in keeps the [`STICKY`] overrides and replaces
/// the rest; a user profile replaces the overrides, extra arguments and agent defaults whole.
pub fn apply_profile(cfg: &Config, profile: &Profile) -> anyhow::Result<Config> {
    let sec = cfg.engine.preset.subcommand();
    if profile.section != sec {
        anyhow::bail!(
            "the profile `{}` is for `saber {}`; this config boots `saber {sec}`",
            profile.name,
            profile.section
        );
    }
    let mut out = cfg.clone();
    // a built-in is applied in its form for this config's TP (same name, the TP's flags)
    let tp_form = profile
        .builtin
        .then(|| {
            builtin_for(sec, cfg.engine.tp)
                .into_iter()
                .find(|b| b.name == profile.name)
        })
        .flatten();
    let profile = tp_form.as_ref().unwrap_or(profile);
    if profile.builtin {
        out.engine.flags.retain(|k, _| STICKY.contains(&k.as_str()));
        for (k, v) in &profile.flags {
            out.engine.flags.insert(k.clone(), v.clone());
        }
    } else {
        out.engine.flags = profile.flags.clone();
        out.engine.extra_args = profile.extra_args.clone();
    }
    if let Some(a) = profile.agent_defaults {
        out.engine.agent_defaults = a;
    }
    Ok(out)
}

/// A user profile snapshotting `cfg`'s overrides.
#[must_use]
pub fn snapshot(cfg: &Config, name: &str, description: &str) -> Profile {
    Profile {
        name: name.to_string(),
        description: description.to_string(),
        section: cfg.engine.preset.subcommand().to_string(),
        builtin: false,
        flags: cfg.engine.flags.clone(),
        extra_args: cfg.engine.extra_args.clone(),
        agent_defaults: Some(cfg.engine.agent_defaults),
    }
}

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    #[serde(default)]
    profile: Vec<Profile>,
}

/// One profile as TOML (the export form).
pub fn export_toml(p: &Profile) -> anyhow::Result<String> {
    let mut p = p.clone();
    p.builtin = false;
    Ok(toml::to_string_pretty(&p)?)
}

/// Parse an import: one profile, or a whole `profiles.toml` (`[[profile]]`).
pub fn import_toml(text: &str) -> anyhow::Result<Vec<Profile>> {
    let v: toml::Value = toml::from_str(text).context("parsing the profile TOML")?;
    let mut ps: Vec<Profile> = if v.get("profile").is_some() {
        let f: File = toml::from_str(text).context("parsing the profiles file")?;
        f.profile
    } else {
        vec![toml::from_str::<Profile>(text).context("parsing the profile")?]
    };
    for p in &mut ps {
        p.builtin = false;
        check_name(&p.name)?;
        if p.section != "engine" && p.section != "serve" {
            anyhow::bail!("profile `{}`: section must be `engine` or `serve`", p.name);
        }
        if let Some(k) = p.flags.keys().find(|k| !k.starts_with("--")) {
            anyhow::bail!("profile `{}`: `{k}` is not a long flag", p.name);
        }
    }
    Ok(ps)
}

fn check_name(name: &str) -> anyhow::Result<()> {
    let n = name.trim();
    if n.is_empty() {
        anyhow::bail!("a profile needs a name");
    }
    if n.len() > 64 {
        anyhow::bail!("a profile name is at most 64 characters");
    }
    if n != name {
        anyhow::bail!("a profile name may not start or end with spaces");
    }
    if ["engine", "serve"]
        .iter()
        .any(|s| builtin(s).iter().any(|b| b.name.eq_ignore_ascii_case(n)))
    {
        anyhow::bail!("`{n}` is a built-in profile's name");
    }
    Ok(())
}

/// The user profiles in `<data dir>/profiles.toml`.
pub struct ProfileStore {
    path: PathBuf,
    lock: std::sync::Mutex<()>,
}

impl ProfileStore {
    #[must_use]
    pub fn new(data_dir: &Path) -> ProfileStore {
        ProfileStore {
            path: data_dir.join("profiles.toml"),
            lock: std::sync::Mutex::new(()),
        }
    }

    fn read(&self) -> anyhow::Result<Vec<Profile>> {
        match std::fs::read_to_string(&self.path) {
            Ok(t) => Ok(toml::from_str::<File>(&t)
                .with_context(|| format!("in {}", self.path.display()))?
                .profile),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(e).with_context(|| format!("reading {}", self.path.display())),
        }
    }

    fn write(&self, ps: &[Profile]) -> anyhow::Result<()> {
        if let Some(d) = self.path.parent() {
            std::fs::create_dir_all(d).with_context(|| format!("creating {}", d.display()))?;
        }
        let f = File {
            profile: ps.to_vec(),
        };
        let tmp = self.path.with_extension("toml.tmp");
        std::fs::write(&tmp, toml::to_string_pretty(&f)?)?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }

    fn guard(&self) -> std::sync::MutexGuard<'_, ()> {
        self.lock.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn list(&self) -> anyhow::Result<Vec<Profile>> {
        let _g = self.guard();
        self.read()
    }

    pub fn get(&self, name: &str) -> anyhow::Result<Option<Profile>> {
        Ok(self.list()?.into_iter().find(|p| p.name == name))
    }

    /// Save (insert, or replace the profile of that name when `replace`).
    pub fn save(&self, p: Profile, replace: bool) -> anyhow::Result<()> {
        check_name(&p.name)?;
        let _g = self.guard();
        let mut ps = self.read()?;
        let mut p = p;
        p.builtin = false;
        match ps.iter().position(|x| x.name == p.name) {
            Some(i) if replace => ps[i] = p,
            Some(_) => anyhow::bail!("a profile named `{}` exists", p.name),
            None => ps.push(p),
        }
        self.write(&ps)
    }

    pub fn rename(&self, from: &str, to: &str) -> anyhow::Result<()> {
        check_name(to)?;
        let _g = self.guard();
        let mut ps = self.read()?;
        if from != to && ps.iter().any(|x| x.name == to) {
            anyhow::bail!("a profile named `{to}` exists");
        }
        let p = ps
            .iter_mut()
            .find(|x| x.name == from)
            .ok_or_else(|| anyhow::anyhow!("no profile named `{from}`"))?;
        p.name = to.to_string();
        self.write(&ps)
    }

    /// Copy `from` under a free name (`X copy`, `X copy 2`, ...); returns the new name.
    pub fn duplicate(&self, from: &str) -> anyhow::Result<String> {
        let _g = self.guard();
        let mut ps = self.read()?;
        let src = ps
            .iter()
            .find(|x| x.name == from)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no profile named `{from}`"))?;
        let mut name = format!("{from} copy");
        let mut n = 2;
        while ps.iter().any(|x| x.name == name) {
            name = format!("{from} copy {n}");
            n += 1;
        }
        check_name(&name)?;
        ps.push(Profile {
            name: name.clone(),
            ..src
        });
        self.write(&ps)?;
        Ok(name)
    }

    pub fn delete(&self, name: &str) -> anyhow::Result<()> {
        let _g = self.guard();
        let mut ps = self.read()?;
        let n = ps.len();
        ps.retain(|x| x.name != name);
        if ps.len() == n {
            anyhow::bail!("no profile named `{name}`");
        }
        self.write(&ps)
    }

    /// Import every profile of `text`; a clashing name gets ` (imported)` appended.
    pub fn import(&self, text: &str) -> anyhow::Result<Vec<String>> {
        let incoming = import_toml(text)?;
        let _g = self.guard();
        let mut ps = self.read()?;
        let mut names = Vec::new();
        for mut p in incoming {
            let base = p.name.clone();
            let mut n = 1;
            while ps.iter().any(|x| x.name == p.name) {
                p.name = if n == 1 {
                    format!("{base} (imported)")
                } else {
                    format!("{base} (imported {n})")
                };
                n += 1;
            }
            check_name(&p.name)?;
            names.push(p.name.clone());
            ps.push(p);
        }
        self.write(&ps)?;
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Preset;

    #[test]
    fn builtins_compose_over_the_stock_presets_and_keep_sticky_flags() {
        for sec in ["engine", "serve"] {
            let b = builtin(sec);
            let names: Vec<&str> = b.iter().map(|p| p.name.as_str()).collect();
            assert_eq!(
                names,
                [
                    "Balanced",
                    "Max throughput",
                    "Lowest latency",
                    "Long context",
                    "Low power"
                ]
            );
            for p in &b {
                assert!(!p.description.is_empty());
                // no profile names the power cap
                for k in p.flags.keys() {
                    assert!(!k.contains("power") && !k.contains("watt"), "{k}");
                }
            }
        }
        let mut c = Config::default();
        c.engine
            .flags
            .insert("--pack".into(), FlagValue::Str("/n/p".into()));
        c.engine
            .flags
            .insert("--fn-thinking".into(), FlagValue::Str("off".into()));
        let long = builtin("engine")
            .into_iter()
            .find(|p| p.name == "Long context")
            .unwrap();
        let out = apply_profile(&c, &long).unwrap();
        assert_eq!(
            out.engine.flags.get("--pack"),
            Some(&FlagValue::Str("/n/p".into()))
        );
        assert_eq!(out.engine.flags.get("--fn-thinking"), None);
        assert_eq!(
            out.engine.flags.get("--fn-max-ctx"),
            Some(&FlagValue::Int(262_144))
        );
        // Balanced = the stock preset (sticky kept)
        let bal = builtin("engine").remove(0);
        let out = apply_profile(&c, &bal).unwrap();
        assert_eq!(out.engine.flags.len(), 1);
        // the wrong section is refused
        let serve = builtin("serve").remove(1);
        assert!(apply_profile(&c, &serve).is_err());
        let mut m = Config::mock(8099);
        m.engine.preset = Preset::Mock;
        let out = apply_profile(&m, &serve).unwrap();
        assert_eq!(
            out.engine.flags.get("--engine-max-ctx"),
            Some(&FlagValue::Int(32_768))
        );
    }

    #[test]
    fn the_27b_profiles_follow_the_tp_and_never_change_it() {
        let one = builtin_for("serve", 1);
        let two = builtin_for("serve", 2);
        let names = |v: &[Profile]| v.iter().map(|p| p.name.clone()).collect::<Vec<_>>();
        assert_eq!(names(&one), names(&two));
        assert_eq!(builtin_for("engine", 1), builtin("engine"));
        let mut c = Config::default();
        c.engine.preset = Preset::Qwen27b;
        c.engine.tp = 1;
        c.engine.device = 1;
        // the TP=2 list's entry applies in its TP=1 form
        let long2 = two.iter().find(|p| p.name == "Long context").unwrap();
        let out = apply_profile(&c, long2).unwrap();
        assert_eq!(
            out.engine.flags.get("--engine-max-seqs"),
            Some(&FlagValue::Int(1))
        );
        assert_eq!(
            out.engine.flags.get("--engine-max-ctx"),
            Some(&FlagValue::Int(131_072))
        );
        assert_eq!((out.engine.tp, out.engine.device), (1, 1));
        for p in &one {
            assert!(!p.flags.contains_key("--engine-tp"), "{}", p.name);
            let o = apply_profile(&c, p).unwrap();
            assert_eq!(o.engine.tp, 1);
        }
    }

    #[test]
    fn user_profiles_snapshot_and_replace() {
        let mut c = Config::default();
        c.engine
            .flags
            .insert("--fn-max-ctx".into(), FlagValue::Int(65_536));
        c.engine.extra_args = vec!["--fn-route-trace".into(), "/x".into()];
        c.engine.agent_defaults = false;
        let snap = snapshot(&c, "mine", "d");
        let back = apply_profile(&Config::default(), &snap).unwrap();
        assert_eq!(back.engine.flags, c.engine.flags);
        assert_eq!(back.engine.extra_args, c.engine.extra_args);
        assert!(!back.engine.agent_defaults);
    }

    #[test]
    fn store_save_rename_duplicate_delete_import_export() {
        let dir = saber_test_scratch::ScratchDir::new("console-profiles");
        let s = ProfileStore::new(dir.path());
        assert!(s.list().unwrap().is_empty());
        let mut c = Config::default();
        c.engine
            .flags
            .insert("--fn-max-ctx".into(), FlagValue::Int(65_536));
        c.engine.flags.insert(
            "--api-key".into(),
            FlagValue::List(vec!["a=12345678".into()]),
        );
        s.save(snapshot(&c, "work", "my box"), false).unwrap();
        assert!(s.save(snapshot(&c, "work", ""), false).is_err());
        assert!(s.save(snapshot(&c, "Balanced", ""), false).is_err());
        assert!(s.save(snapshot(&c, "", ""), false).is_err());
        s.rename("work", "office").unwrap();
        assert!(s.get("work").unwrap().is_none());
        let d = s.duplicate("office").unwrap();
        assert_eq!(d, "office copy");
        assert_eq!(s.duplicate("office").unwrap(), "office copy 2");
        assert!(s.rename("office copy", "office").is_err());
        s.delete("office copy 2").unwrap();
        assert!(s.delete("nope").is_err());
        // export -> import round trip (a clash gets a suffix)
        let p = s.get("office").unwrap().unwrap();
        let t = export_toml(&p).unwrap();
        let parsed = import_toml(&t).unwrap();
        assert_eq!(parsed, vec![p.clone()]);
        let names = s.import(&t).unwrap();
        assert_eq!(names, ["office (imported)"]);
        let imported = s.get("office (imported)").unwrap().unwrap();
        assert_eq!(imported.flags, p.flags);
        assert_eq!(imported.agent_defaults, p.agent_defaults);
        // the whole file imports too
        let file = std::fs::read_to_string(dir.path().join("profiles.toml")).unwrap();
        let all = import_toml(&file).unwrap();
        assert_eq!(all.len(), 3);
        assert!(import_toml("name = \"x\"\nsection = \"gpu\"").is_err());
        assert!(import_toml("name = \"x\"\nsection = \"serve\"\n[flags]\nport = 1").is_err());
        assert!(import_toml("not toml [").is_err());
    }
}
