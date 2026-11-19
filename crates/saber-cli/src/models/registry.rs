//! The model registry: `models.toml` (compiled in) merged with the user's override file.
//!
//! The registry is data only. A model is one `[[model]]` block: its Hugging Face repo, the
//! folder each tensor-parallel degree's pack sits in, the served id, the drafter it pairs with
//! (by that drafter's own entry name), the pack format tag it needs and the serving shapes per
//! TP. docs/MODELS.md is the schema reference.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::ui::UserError;

/// The registry shipped with this binary.
pub const SHIPPED: &str = include_str!("../../../../models.toml");

/// The schema version this build reads.
pub const SCHEMA: u32 = 1;

/// The TP degrees the engine serves.
pub const TPS: [u32; 2] = [1, 2];

/// The pack format tags this build of the engine can load. A pack's tag is the first word of the
/// format string the engine checks in the pack's manifest (`macaq_iu8/1 i4row+i8row ...`).
#[must_use]
pub fn engine_tags() -> Vec<String> {
    [
        saber_kernels::macaq_iu8::FORMAT,
        saber_kernels::dflash_iu8::FORMAT,
    ]
    .iter()
    .filter_map(|f| f.split_whitespace().next())
    .map(str::to_string)
    .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// A model you serve.
    Target,
    /// A speculative-decoding drafter a target names.
    Drafter,
}

/// Where an entry came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Origin {
    Shipped,
    User,
}

/// One TP degree of an entry: where its pack is and (targets) how to serve it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TpSpec {
    /// The pack's folder inside the repo (`tp2`), relative, no `..`.
    pub path: String,
    /// The pack's size in bytes (the listing at pull time has the exact figure; this one is
    /// shown offline).
    pub bytes: u64,
    /// Older flat folder names under the packs root that hold the same pack (e.g. what
    /// `tools/fetch_packs.sh` made); found and served in place, never deleted.
    #[serde(default)]
    pub legacy_dirs: Vec<String>,
    #[serde(default)]
    pub max_ctx: Option<u32>,
    #[serde(default)]
    pub max_seqs: Option<u32>,
    #[serde(default)]
    pub prefill_rows: Option<String>,
    #[serde(default)]
    pub decode_batches: Option<String>,
    #[serde(default)]
    pub decode_widths: Option<String>,
    #[serde(default)]
    pub draft_block: Option<u32>,
    #[serde(default)]
    pub budget_mib: Option<u32>,
    /// Engine flags appended after `--port` at this TP.
    #[serde(default)]
    pub tail_args: Vec<String>,
}

/// One `[[model]]` block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub kind: Kind,
    #[serde(default)]
    pub description: String,
    /// `owner/name` on the endpoint.
    pub repo: String,
    #[serde(default = "main_rev")]
    pub revision: String,
    /// Overrides the registry's endpoint for this entry (`https://...` or `file:///mirror`).
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub served_id: Option<String>,
    /// The drafter entry's name (or alias).
    #[serde(default)]
    pub drafter: Option<String>,
    /// The pack format tag the engine must support (`macaq_iu8/1`).
    #[serde(default)]
    pub requires_tag: Option<String>,
    #[serde(default)]
    pub license: String,
    /// Files at the repo root every TP needs (tokenizer, chat template).
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default)]
    pub tokenizer: Option<String>,
    #[serde(default)]
    pub chat_template: Option<String>,
    /// Engine flags placed after the shapes, before `--served-model-name`.
    #[serde(default)]
    pub engine_args: Vec<String>,
    #[serde(default)]
    pub default_tp: Option<u32>,
    /// Keyed by the TP degree (`"1"`, `"2"`).
    #[serde(default)]
    pub tp: BTreeMap<String, TpSpec>,
    #[serde(skip_deserializing, default = "shipped")]
    pub origin: Origin,
}

fn main_rev() -> String {
    "main".into()
}
fn shipped() -> Origin {
    Origin::Shipped
}

impl Entry {
    /// The TP degrees this entry has packs for, ascending.
    #[must_use]
    pub fn tps(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.tp.keys().filter_map(|k| k.parse().ok()).collect();
        v.sort_unstable();
        v
    }

    #[must_use]
    pub fn at(&self, tp: u32) -> Option<&TpSpec> {
        self.tp.get(&tp.to_string())
    }

    /// The TP to use when nothing else decides: `default_tp`, else the largest.
    #[must_use]
    pub fn preferred_tp(&self) -> u32 {
        self.default_tp
            .or_else(|| self.tps().last().copied())
            .unwrap_or(2)
    }

    /// `owner`, `name`.
    #[must_use]
    pub fn repo_parts(&self) -> (&str, &str) {
        self.repo.split_once('/').unwrap_or(("", &self.repo))
    }

    /// Every name the entry answers to, lower-cased.
    fn keys(&self) -> impl Iterator<Item = String> + '_ {
        std::iter::once(self.name.to_ascii_lowercase())
            .chain(self.aliases.iter().map(|a| a.to_ascii_lowercase()))
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    #[serde(default)]
    schema: Option<u32>,
    #[serde(default)]
    default: Option<String>,
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default, rename = "model")]
    models: Vec<Entry>,
}

/// The merged registry.
#[derive(Clone, Debug)]
pub struct Registry {
    pub default: String,
    pub endpoint: String,
    pub entries: Vec<Entry>,
    /// The user file that was merged in, if any.
    pub user_file: Option<PathBuf>,
}

/// Is `s` a safe file or folder name (no separators, no dot-dot)?
fn safe_name(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_.+".contains(c))
}

/// Is `s` a safe relative path (segments of safe names)?
fn safe_rel(s: &str) -> bool {
    !s.is_empty() && !s.starts_with('/') && s.split('/').all(safe_name)
}

fn parse_csv(s: &str) -> Option<Vec<u32>> {
    s.split(',').map(|x| x.trim().parse().ok()).collect()
}

fn parse_file(text: &str, origin: &str) -> Result<File, UserError> {
    let f: File = toml::from_str(text).map_err(|e| {
        UserError::new(format!("{origin} is not a valid registry file:\n{e}")).fix(format!(
            "fix the line above, or move {origin} aside to use the shipped registry only"
        ))
    })?;
    if let Some(s) = f.schema {
        if s != SCHEMA {
            return Err(UserError::new(format!(
                "{origin} says `schema = {s}`; this saber reads schema {SCHEMA}"
            ))
            .fix("update saber (git pull, then rebuild), or set `schema = 1`".to_string()));
        }
    }
    Ok(f)
}

impl Registry {
    /// The shipped registry merged with `user` (its text and its path for messages).
    pub fn load(user: Option<(&str, &Path)>) -> Result<Registry, UserError> {
        Registry::from_texts(SHIPPED, user)
    }

    /// `shipped` merged with `user`: a user entry replaces the shipped entry of the same name
    /// (case-insensitive) and takes any alias it names from the shipped entries; the user's
    /// `default` / `endpoint` win.
    pub fn from_texts(shipped: &str, user: Option<(&str, &Path)>) -> Result<Registry, UserError> {
        let base = parse_file(shipped, "the shipped models.toml")?;
        let mut entries = base.models;
        let mut default = base.default.unwrap_or_default();
        let mut endpoint = base
            .endpoint
            .unwrap_or_else(|| "https://huggingface.co".into());
        let mut user_file = None;
        if let Some((text, path)) = user {
            let label = path.display().to_string();
            let u = parse_file(text, &label)?;
            check_unique(&u.models, &label)?;
            for mut e in u.models {
                e.origin = Origin::User;
                let claimed: Vec<String> = e.keys().collect();
                entries.retain(|s| !s.name.eq_ignore_ascii_case(&e.name));
                for s in &mut entries {
                    s.aliases
                        .retain(|a| !claimed.contains(&a.to_ascii_lowercase()));
                }
                entries.push(e);
            }
            if let Some(d) = u.default {
                default = d;
            }
            if let Some(ep) = u.endpoint {
                endpoint = ep;
            }
            user_file = Some(path.to_path_buf());
        }
        let r = Registry {
            default,
            endpoint,
            entries,
            user_file,
        };
        r.validate()?;
        Ok(r)
    }

    /// Every problem in the merged registry, reported together.
    pub fn validate(&self) -> Result<(), UserError> {
        let mut errs: Vec<String> = Vec::new();
        if let Err(e) = check_unique(&self.entries, "the registry") {
            errs.push(e.what);
        }
        for e in &self.entries {
            let at = |m: String| format!("model `{}`: {m}", e.name);
            if !safe_name(&e.name) {
                errs.push(at("`name` may use letters, digits and - _ . + only".into()));
            }
            for a in &e.aliases {
                if !safe_name(a) {
                    errs.push(at(format!(
                        "alias `{a}` may use letters, digits and - _ . + only"
                    )));
                }
            }
            match e.repo.split_once('/') {
                Some((o, n)) if safe_name(o) && safe_name(n) => {}
                _ => errs.push(at(format!("`repo = \"{}\"` is not `owner/name`", e.repo))),
            }
            if e.tp.is_empty() {
                errs.push(at(
                    "no `[model.tp.N]` table: it has no pack to download".into()
                ));
            }
            for (k, t) in &e.tp {
                let tp: Option<u32> = k.parse().ok().filter(|t| TPS.contains(t));
                if tp.is_none() {
                    errs.push(at(format!("`[model.tp.{k}]`: TP must be 1 or 2")));
                }
                if !safe_rel(&t.path) {
                    errs.push(at(format!(
                        "`[model.tp.{k}] path = \"{}\"` must be a relative folder (no `..`)",
                        t.path
                    )));
                }
                for l in &t.legacy_dirs {
                    if !safe_name(l) {
                        errs.push(at(format!("legacy dir `{l}` must be a plain folder name")));
                    }
                }
                if e.kind == Kind::Target {
                    errs.extend(check_shapes(k, t).into_iter().map(&at));
                }
            }
            if let Some(d) = e.default_tp {
                if e.at(d).is_none() {
                    errs.push(at(format!("`default_tp = {d}` has no `[model.tp.{d}]`")));
                }
            }
            if let Some(t) = &e.requires_tag {
                let ok = t
                    .split_once('/')
                    .is_some_and(|(n, v)| !n.is_empty() && v.parse::<u32>().is_ok());
                if !ok {
                    errs.push(at(format!(
                        "`requires_tag = \"{t}\"` must look like `name/1`"
                    )));
                }
            }
            for f in &e.files {
                if !safe_rel(f) {
                    errs.push(at(format!("file `{f}` must be a relative path (no `..`)")));
                }
            }
            match e.kind {
                Kind::Target => {
                    if e.served_id.as_deref().is_none_or(str::is_empty) {
                        errs.push(at("a target needs `served_id`".into()));
                    }
                    for (field, v) in [
                        ("tokenizer", &e.tokenizer),
                        ("chat_template", &e.chat_template),
                    ] {
                        match v {
                            None => errs.push(at(format!("a target needs `{field}`"))),
                            Some(f) if !e.files.contains(f) => errs.push(at(format!(
                                "`{field} = \"{f}\"` is not in `files` (so it would never be downloaded)"
                            ))),
                            _ => {}
                        }
                    }
                    if let Some(d) = &e.drafter {
                        match self.find(d) {
                            None => errs.push(at(format!(
                                "`drafter = \"{d}\"` names no entry in the registry"
                            ))),
                            Some(de) if de.kind != Kind::Drafter => errs.push(at(format!(
                                "`drafter = \"{d}\"` is a {:?} entry, not a drafter",
                                de.kind
                            )
                            .to_lowercase())),
                            Some(de) => {
                                for tp in e.tps() {
                                    if de.at(tp).is_none() {
                                        errs.push(at(format!(
                                            "its drafter `{}` has no TP={tp} pack",
                                            de.name
                                        )));
                                    }
                                }
                            }
                        }
                    }
                }
                Kind::Drafter => {
                    for (field, set) in [
                        ("served_id", e.served_id.is_some()),
                        ("drafter", e.drafter.is_some()),
                        ("tokenizer", e.tokenizer.is_some()),
                        ("chat_template", e.chat_template.is_some()),
                    ] {
                        if set {
                            errs.push(at(format!("a drafter has no `{field}`")));
                        }
                    }
                }
            }
        }
        match self.find(&self.default) {
            Some(d) if d.kind == Kind::Target => {}
            Some(_) => errs.push(format!(
                "`default = \"{}\"` is a drafter, not a target",
                self.default
            )),
            None if self.default.is_empty() => errs.push("no `default = \"...\"` model".into()),
            None => errs.push(format!("`default = \"{}\"` names no entry", self.default)),
        }
        if errs.is_empty() {
            return Ok(());
        }
        let file = self
            .user_file
            .as_ref()
            .map_or("models.toml".to_string(), |p| p.display().to_string());
        Err(UserError::new(format!(
            "the model registry has {} problem{}:\n  - {}",
            errs.len(),
            if errs.len() == 1 { "" } else { "s" },
            errs.join("\n  - ")
        ))
        .fix(format!("edit {file} (docs/MODELS.md has the schema)")))
    }

    /// The entry `name` names: its name or an alias (case-insensitive), or `default`.
    #[must_use]
    pub fn find(&self, name: &str) -> Option<&Entry> {
        let n = name.trim().to_ascii_lowercase();
        self.entries
            .iter()
            .find(|e| e.name.eq_ignore_ascii_case(&n))
            .or_else(|| self.entries.iter().find(|e| e.keys().any(|k| k == n)))
            .or_else(|| (n == "default").then(|| self.find(&self.default)).flatten())
    }

    /// The entry, or an error with the closest names.
    pub fn resolve(&self, name: &str) -> Result<&Entry, UserError> {
        if let Some(e) = self.find(name) {
            return Ok(e);
        }
        let s = self.suggest(name);
        let mut err = UserError::new(format!("there is no model called `{name}`"));
        if let Some(first) = s.first() {
            err = err
                .note(format!("did you mean: {}?", s.join(", ")))
                .fix(format!("saber models info {first}"));
        }
        Err(err.fix("saber models list   (every model and its aliases)".to_string()))
    }

    /// The names and aliases closest to `name` (edit distance, prefix and substring), best first.
    #[must_use]
    pub fn suggest(&self, name: &str) -> Vec<String> {
        let n = name.trim().to_ascii_lowercase();
        let mut scored: Vec<(usize, String)> = Vec::new();
        for e in &self.entries {
            let mut best: Option<usize> = None;
            for k in e.keys() {
                let d = edit_distance(&n, &k);
                let score = if !n.is_empty() && (k.starts_with(&n) || k.contains(&n)) {
                    0
                } else if d <= 2 + k.len().min(n.len()) / 6 {
                    d
                } else {
                    continue;
                };
                best = Some(best.map_or(score, |b: usize| b.min(score)));
            }
            if let Some(b) = best {
                // the shortest handle reads best in a suggestion
                let handle = std::iter::once(&e.name)
                    .chain(e.aliases.iter())
                    .filter(|k| {
                        let k = k.to_ascii_lowercase();
                        edit_distance(&n, &k) == b || k.contains(&n)
                    })
                    .min_by_key(|k| k.len())
                    .unwrap_or(&e.name)
                    .clone();
                scored.push((b, handle));
            }
        }
        scored.sort();
        scored.dedup_by(|a, b| a.1 == b.1);
        scored.into_iter().take(3).map(|(_, s)| s).collect()
    }

    #[must_use]
    pub fn default_entry(&self) -> &Entry {
        self.find(&self.default)
            .expect("validated: the default resolves")
    }

    /// The targets that name `drafter` as theirs.
    #[must_use]
    pub fn users_of<'a>(&'a self, drafter: &Entry) -> Vec<&'a Entry> {
        self.entries
            .iter()
            .filter(|e| {
                e.kind == Kind::Target
                    && e.drafter
                        .as_deref()
                        .and_then(|d| self.find(d))
                        .is_some_and(|d| d.name == drafter.name)
            })
            .collect()
    }

    /// The endpoint an entry downloads from.
    #[must_use]
    pub fn endpoint_of<'a>(&'a self, e: &'a Entry) -> &'a str {
        e.endpoint.as_deref().unwrap_or(&self.endpoint)
    }
}

fn check_unique(entries: &[Entry], label: &str) -> Result<(), UserError> {
    let mut seen: BTreeMap<String, &str> = BTreeMap::new();
    let mut errs = Vec::new();
    for e in entries {
        let mut own: Vec<String> = e.keys().collect();
        own.sort();
        own.dedup();
        for k in own {
            if let Some(prev) = seen.insert(k.clone(), &e.name) {
                errs.push(format!(
                    "`{k}` names both `{prev}` and `{}` in {label}",
                    e.name
                ));
            }
        }
    }
    if errs.is_empty() {
        Ok(())
    } else {
        Err(UserError::new(errs.join("; ")))
    }
}

fn check_shapes(k: &str, t: &TpSpec) -> Vec<String> {
    let mut errs = Vec::new();
    let w = |m: &str| format!("`[model.tp.{k}]` {m}");
    if t.max_ctx.is_none() {
        errs.push(w("needs `max_ctx` (a target is served with it)"));
    }
    let seqs = t.max_seqs;
    if seqs.is_none() {
        errs.push(w("needs `max_seqs`"));
    }
    for (f, v, need) in [
        ("prefill_rows", &t.prefill_rows, false),
        ("decode_batches", &t.decode_batches, true),
        ("decode_widths", &t.decode_widths, true),
    ] {
        match v {
            None if need => errs.push(w(&format!("needs `{f}`"))),
            None => {}
            Some(s) => match parse_csv(s) {
                None => errs.push(w(&format!(
                    "`{f} = \"{s}\"` must be comma-separated numbers"
                ))),
                Some(list) if f == "decode_batches" => {
                    if let (Some(m), Some(n)) = (list.iter().max(), seqs) {
                        if *m > n {
                            errs.push(w(&format!(
                                "`decode_batches` goes to {m} but `max_seqs = {n}` (the engine refuses a batch larger than its sequences)"
                            )));
                        }
                    }
                }
                Some(_) => {}
            },
        }
    }
    errs
}

/// Levenshtein distance (bytes; the names are ASCII).
#[must_use]
pub fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(text: &str) -> Result<Registry, UserError> {
        Registry::from_texts(SHIPPED, Some((text, Path::new("/u/models.toml"))))
    }

    #[test]
    fn the_shipped_registry_is_valid() {
        let r = Registry::load(None).unwrap();
        assert_eq!(r.default_entry().name, "Qwen3.8-27B-MACaQ-W4A8-SABER");
        assert_eq!(
            r.default_entry().served_id.as_deref(),
            Some("Qwen3.8-27B-MACaQ-W4A8-SABER")
        );
        let targets: Vec<_> = r
            .entries
            .iter()
            .filter(|e| e.kind == Kind::Target)
            .collect();
        let drafters: Vec<_> = r
            .entries
            .iter()
            .filter(|e| e.kind == Kind::Drafter)
            .collect();
        assert_eq!((targets.len(), drafters.len()), (2, 2));
        for e in &r.entries {
            assert_eq!(e.tps(), vec![1, 2], "{}", e.name);
            assert_eq!(e.origin, Origin::Shipped);
        }
    }

    #[test]
    fn aliases_resolve_case_insensitively() {
        let r = Registry::load(None).unwrap();
        for (k, want) in [
            ("w4a8", "Qwen3.8-27B-MACaQ-W4A8-SABER"),
            ("default", "Qwen3.8-27B-MACaQ-W4A8-SABER"),
            ("W4A8", "Qwen3.8-27B-MACaQ-W4A8-SABER"),
            ("mxfp4", "Qwen3.8-27B-MACaQ-MXFP4-SABER"),
            (
                "qwen3.8-27b-macaq-mxfp4-saber",
                "Qwen3.8-27B-MACaQ-MXFP4-SABER",
            ),
            ("drafter-fp8", "Qwen3.8-27B-DFlash2-Drafter-FP8-SABER"),
        ] {
            assert_eq!(r.resolve(k).unwrap().name, want, "{k}");
        }
    }

    #[test]
    fn unknown_names_get_did_you_mean() {
        let r = Registry::load(None).unwrap();
        let e = r.resolve("w4a9").unwrap_err();
        assert!(
            e.render(false).contains("did you mean: w4a8"),
            "{}",
            e.render(false)
        );
        let e = r.resolve("mxfp").unwrap_err();
        assert!(e.render(false).contains("mxfp4"), "{}", e.render(false));
        let e = r.resolve("zzzzzzzz").unwrap_err();
        let t = e.render(false);
        assert!(!t.contains("did you mean"), "{t}");
        assert!(t.contains("saber models list"), "{t}");
    }

    #[test]
    fn edit_distance_basics() {
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("w4a8", "w4a8"), 0);
    }

    const MINE: &str = r#"
default = "mine"
[[model]]
name = "mine"
aliases = ["w4a8"]
kind = "target"
repo = "me/mine"
served_id = "Mine"
drafter = "drafter-w4a8"
files = ["tokenizer.json", "t.jinja"]
tokenizer = "tokenizer.json"
chat_template = "t.jinja"
[model.tp.2]
path = "tp2"
bytes = 10
max_ctx = 4096
max_seqs = 2
decode_batches = "1,2"
decode_widths = "1"
"#;

    #[test]
    fn a_user_entry_is_added_and_steals_its_aliases() {
        let r = user(MINE).unwrap();
        assert_eq!(r.default_entry().name, "mine");
        assert_eq!(r.default_entry().origin, Origin::User);
        // the user's alias wins over the shipped entry's
        assert_eq!(r.resolve("w4a8").unwrap().name, "mine");
        // the shipped entry is still there under its other names
        assert_eq!(
            r.resolve("27b").unwrap().name,
            "Qwen3.8-27B-MACaQ-W4A8-SABER"
        );
        assert_eq!(r.entries.len(), 5);
    }

    #[test]
    fn a_user_entry_replaces_a_shipped_one_by_name() {
        let text = MINE
            .replace("name = \"mine\"", "name = \"qwen3.8-27b-macaq-w4a8-saber\"")
            .replace("default = \"mine\"", "");
        let r = user(&text).unwrap();
        assert_eq!(r.entries.len(), 4);
        let e = r.resolve("Qwen3.8-27B-MACaQ-W4A8-SABER").unwrap();
        assert_eq!(e.served_id.as_deref(), Some("Mine"));
        assert_eq!(e.origin, Origin::User);
        // the shipped default named it; it now resolves to the user's entry
        assert_eq!(r.default_entry().served_id.as_deref(), Some("Mine"));
    }

    #[test]
    fn bad_entries_are_named_with_the_fix() {
        let cases: &[(&str, &str)] = &[
            (r#"drafter = "drafter-w4a8""#, r#"drafter = "nope""#),
            (r#"drafter = "drafter-w4a8""#, r#"drafter = "mxfp4""#),
            ("path = \"tp2\"", "path = \"../escape\""),
            ("[model.tp.2]", "[model.tp.3]"),
            ("decode_batches = \"1,2\"", "decode_batches = \"1,2,4\""),
            ("served_id = \"Mine\"", ""),
            (
                "chat_template = \"t.jinja\"",
                "chat_template = \"other.jinja\"",
            ),
            ("repo = \"me/mine\"", "repo = \"mine\""),
            ("kind = \"target\"", "kind = \"target\"\nbogus = 1"),
            ("default = \"mine\"", "default = \"drafter-w4a8\""),
        ];
        for (from, to) in cases {
            let text = MINE.replace(from, to);
            assert_ne!(text, MINE, "{from}");
            let e = user(&text).unwrap_err();
            let t = e.render(false);
            assert!(t.contains("fix:"), "{to}: {t}");
            assert!(t.contains("/u/models.toml"), "{to}: {t}");
        }
    }

    #[test]
    fn a_drafter_without_the_targets_tp_is_refused() {
        let text = format!(
            "{MINE}\n[[model]]\nname = \"d1\"\nkind = \"drafter\"\nrepo = \"me/d1\"\n[model.tp.1]\npath = \"tp1\"\nbytes = 1\n"
        )
        .replace(r#"drafter = "drafter-w4a8""#, r#"drafter = "d1""#);
        let t = user(&text).unwrap_err().render(false);
        assert!(t.contains("has no TP=2 pack"), "{t}");
    }

    #[test]
    fn duplicate_names_inside_one_file_are_refused() {
        let text = format!("{MINE}\n{}", MINE.replace("default = \"mine\"", ""));
        let t = user(&text).unwrap_err().render(false);
        assert!(t.contains("names both"), "{t}");
    }

    #[test]
    fn a_wrong_schema_says_update() {
        let t = user("schema = 2\n").unwrap_err().render(false);
        assert!(t.contains("schema 1") && t.contains("update saber"), "{t}");
    }

    #[test]
    fn the_engine_knows_the_shipped_tags() {
        let tags = engine_tags();
        let r = Registry::load(None).unwrap();
        for e in &r.entries {
            if let Some(t) = &e.requires_tag {
                assert!(
                    tags.contains(t),
                    "{} needs {t}; engine has {tags:?}",
                    e.name
                );
            }
        }
    }

    #[test]
    fn users_of_a_drafter() {
        let r = Registry::load(None).unwrap();
        let d = r.resolve("drafter-fp8").unwrap();
        let u: Vec<_> = r.users_of(d).iter().map(|e| e.name.clone()).collect();
        assert_eq!(u, ["Qwen3.8-27B-MACaQ-MXFP4-SABER"]);
    }
}
