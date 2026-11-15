//! The engine's command line, composed from the config.
//!
//! Each [`Preset`] reproduces its launcher's default boot argument for argument
//! (`tools/serve_flashnext.sh`, `tools/serve_27b.sh` at TP=2 and `tools/serve_27b.sh --tp 1` at
//! TP=1; the test `launcher_parity` runs them with `SABER_PRINT_CMD=1` and compares). Config overrides then replace a preset flag in place, remove
//! it (`false`) or append it, exactly where the launcher puts "any extra arguments". Everything
//! the engine needs is an argument: the engine reads no environment variable. The container
//! wrapper only sets the HIP runtime's variables the launchers set (`HIP_VISIBLE_DEVICES`,
//! `GPU_MAX_HW_QUEUES`, `LD_LIBRARY_PATH`), and nothing here touches the GPU power cap.

use std::path::{Path, PathBuf};

use crate::config::{EngineConfig, FlagValue, Preset, Runtime};

/// The prefill row ladder both launchers pass (`DEFAULT_PREFILL_ROWS` in saber-cli).
pub const PREFILL_ROWS: &str = "96,128,192,256,320,384,512,768,1024,1536,2048";

/// Flash-Next's pack as `tools/fetch_packs.sh flashnext` installs it (under `<nvme>`).
pub const FN_PACK: &str = "saber-packs/qwen38-flashnext-quark-trunk8";
/// The same files under the name a pack built locally carries; the launcher prefers it.
pub const FN_LEGACY_PACK: &str = "saber-packs/qwen38-flashnext-quark-ep2-trunk8";

/// Flash-Next's agent-load defaults (`FN_AGENT_DEFAULTS=1`), in the launcher's order.
pub const FN_AGENT_DEFAULTS: &[(&str, Option<&str>)] = &[
    ("--fn-kv-lazy", Some("1024")),
    ("--fn-prefix-tree-ram-gib", Some("4.5")),
    ("--fn-prefill-yield", Some("1")),
    ("--fn-ring-idle", None),
    ("--fn-kv-admit-slab-tokens", Some("338592")),
    ("--spec-couple", Some("on")),
];

/// Flags whose values are `on` / `off` and whose engine default is `on`. A bare `false`
/// override (a toggle switched off) composes as `<flag> off`, `true` as `<flag> on`: removing the
/// flag would leave the engine's default, on, in place.
pub const ON_OFF_FLAGS: &[&str] = &["--enable-prefix-caching"];

/// The 27B on ONE card: `tools/serve_27b.sh --tp 1`'s defaults (the `-tp1` packs of the default
/// weights, the single-card context / sequence / batch ladder, its container name).
pub mod tp1 {
    /// The single-card target pack (under `<nvme>/saber-packs`, mounted at `/packs`).
    pub const PACK: &str = "qwen38-27b-macaq7-bc3-tp1";
    /// The single-card DFlash2 drafter pack.
    pub const DRAFT_PACK: &str = "qwen38-dflash2-drafter-quarkhead-tp1";
    /// Per-sequence context: one conversation of up to 128k tokens (the user's default of
    /// 2026-11-14; one card's ledger leaves ~62k tokens spare beside it for the prefix cache).
    pub const MAX_CTX: u64 = 131_072;
    /// One sequence: one card's KV goes to one conversation (other requests queue).
    pub const MAX_SEQS: u64 = 1;
    /// The decode batch ladder the sequences pick from (`BATCHES`: every rung <= the
    /// sequences; 1 sequence = `1`, 2 = `1,2`, 4 = `1,2,4`).
    pub const BATCH_LADDER: &str = "1,2,4,8";
    /// The batches of the launcher's TP=1 default boot.
    pub const BATCHES: &str = "1";
    /// The TP=1 defaults before 2026-11-14 (4 sequences at 65,536, batches 1,2,4): what a
    /// version-4 file's TP=1 overrides restate.
    pub const V4_MAX_CTX: u64 = 65_536;
    pub const V4_MAX_SEQS: u64 = 4;
    pub const V4_BATCHES: &str = "1,2,4";
    /// The container the launcher names a TP=1 server.
    pub const CONTAINER: &str = "saber-27b-tp1-serve";
}

/// What the composition may not compute by itself (it touches no file system).
#[derive(Clone, Debug)]
pub struct ComposeCtx {
    /// `~` in the config's paths.
    pub home: PathBuf,
    /// The running `saber` binary (the direct runtime's default).
    pub exe: PathBuf,
    /// Flash-Next: `$SHARED/ws-fn3/ref/run1` exists (the launcher passes `--prompt-ids` then).
    pub fn3_prompts_present: bool,
    /// Flash-Next: `<nvme>/saber-packs/qwen38-flashnext-quark-ep2-trunk8` exists (the
    /// launcher then serves that local name of the pack instead of `qwen38-flashnext-quark-trunk8`).
    pub fn_legacy_pack_present: bool,
    /// The container runtime's P2PDMA pool mounts (`tools/capture/p2p_mounts.sh`).
    pub p2p_mounts: Vec<String>,
}

/// The composed command.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineCommand {
    /// The engine's argv: binary, subcommand, flags (as the engine sees it).
    pub engine: Vec<String>,
    /// The full argv the console spawns (the container wrapper, or `engine` itself).
    pub spawn: Vec<String>,
    /// The container's name when the runtime is a container.
    pub container_name: Option<String>,
}

impl EngineCommand {
    /// The engine argv as one shell line (for display and the launcher comparison).
    #[must_use]
    pub fn engine_line(&self) -> String {
        self.engine
            .iter()
            .map(|a| shell_quote(a))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

type Flags = Vec<(String, Option<String>)>;

fn f(k: &str, v: impl Into<String>) -> (String, Option<String>) {
    (k.to_string(), Some(v.into()))
}
fn sw(k: &str) -> (String, Option<String>) {
    (k.to_string(), None)
}

fn p(path: &Path) -> String {
    path.display().to_string()
}

/// The container name a preset's launcher uses at TP=2 (so the launchers' guards and `stop` see
/// it).
#[must_use]
pub fn default_container_name(preset: Preset) -> &'static str {
    match preset {
        Preset::Flashnext => "saber-flashnext-serve",
        Preset::Qwen27b => "saber-27b-serve",
        Preset::Mock => "saber-mock-serve",
    }
}

/// The container name the launcher uses for this config (the 27B at TP=1 has its own).
#[must_use]
pub fn container_name_for(cfg: &EngineConfig) -> String {
    if !cfg.container.name.is_empty() {
        return cfg.container.name.clone();
    }
    if cfg.preset == Preset::Qwen27b && cfg.tp == 1 {
        return tp1::CONTAINER.to_string();
    }
    default_container_name(cfg.preset).to_string()
}

/// Compose the command for `cfg`.
#[must_use]
pub fn compose(cfg: &EngineConfig, ctx: &ComposeCtx) -> EngineCommand {
    let paths = cfg.paths.resolved(&ctx.home);
    let port = cfg.port.to_string();
    let (binary, sub, mut base): (String, &str, Flags) = match cfg.preset {
        Preset::Flashnext => {
            let pack = paths.nvme.join(if ctx.fn_legacy_pack_present {
                FN_LEGACY_PACK
            } else {
                FN_PACK
            });
            let mut b: Flags = vec![
                f("--pack", p(&pack)),
                f("--fn-trunk", "w8-hc8-head"),
                f("--fn-kv-dtype", "fp8"),
                f("--fn-vram-elastic-kv-max-gib", "0.5"),
                f("--fn-experts", "tp2"),
            ];
            if ctx.fn3_prompts_present {
                b.push(f("--prompt-ids", p(&paths.shared.join("ws-fn3/ref/run1"))));
            }
            b.extend([
                f("--tp", "2"),
                f("--hip-lib", "/opt/rocm/lib/libamdhip64.so.7"),
                f("--prebuilt", "/work/prebuilt/gfx1201"),
                f("--fn-out", p(&paths.shared.join("serve-flashnext/run"))),
                f("--fn-serve", &port),
                f("--fn-thinking", "on"),
            ]);
            (
                "/work/.cache/target-container/release/saber".into(),
                "engine",
                b,
            )
        }
        Preset::Qwen27b => {
            // TP=2: both cards; TP=1: one card (the container sees only `device`, so inside it
            // the engine's device is 0), the `-tp1` packs and the single-card defaults.
            let one = cfg.tp == 1;
            let mut b: Flags = vec![f("--engine", "real")];
            if one {
                b.extend([f("--engine-tp", "1"), f("--engine-device", "0")]);
            } else {
                b.extend([
                    f("--engine-tp", "2"),
                    f("--engine-device", "0"),
                    f("--engine-peer-device", "1"),
                ]);
            }
            let (pack, draft) = if one {
                (tp1::PACK, tp1::DRAFT_PACK)
            } else {
                ("qwen38-27b-macaq7-bc3", "qwen38-dflash2-drafter-quarkhead")
            };
            let max_ctx = if one { tp1::MAX_CTX } else { 262_144 }.to_string();
            let max_seqs = if one { tp1::MAX_SEQS } else { 8 }.to_string();
            b.extend([
                f("--hip-lib", "/opt/rocm/lib/libamdhip64.so.7"),
                f("--pack", format!("/packs/{pack}")),
                f("--prebuilt", "prebuilt/gfx1201"),
                f("--tokenizer", "/serve-tokenizer.json"),
                f("--chat-template", "/serve-template.jinja"),
                f("--engine-max-ctx", max_ctx),
                f("--engine-prefill-rows", PREFILL_ROWS),
                f(
                    "--engine-decode-batches",
                    if one { tp1::BATCH_LADDER } else { "1,2,3,4,8" },
                ),
                f("--engine-decode-widths", "1,5,8"),
                f("--engine-max-seqs", max_seqs),
                f("--engine-draft-block", "8"),
                f("--engine-draft-pack", format!("/packs/{draft}")),
                f("--engine-budget-mib", "30500"),
                sw("--engine-rectangular-widths"),
                f("--served-model-name", "Qwen3.8-27B-MACaQ-MXFP4"),
                f("--host", "0.0.0.0"),
                f("--port", &port),
            ]);
            (".cache/target-container/release/saber".into(), "serve", b)
        }
        Preset::Mock => {
            let r = &paths.repo;
            let b: Flags = vec![
                f("--engine", "mock"),
                f("--mock", p(&r.join("examples/mock-script.toml"))),
                f(
                    "--manifest",
                    p(&r.join("examples/qwen38-27b-nvfp4-tp2.toml")),
                ),
                f(
                    "--autotune",
                    p(&r.join("examples/autotune-gfx1201-210w.json")),
                ),
                f("--prebuilt", p(&r.join("prebuilt/gfx1201"))),
                f("--tokenizer", p(&paths.tokenizer)),
                f("--chat-template", p(&paths.chat_template)),
                f("--step-delay-ms", "20"),
                f("--host", "127.0.0.1"),
                f("--port", &port),
            ];
            let bin = if paths.binary.as_os_str().is_empty() {
                p(&ctx.exe)
            } else {
                p(&paths.binary)
            };
            (bin, "serve", b)
        }
    };

    // Overrides: a preset flag is replaced in place (or removed by `false`); any other flag goes
    // where the launcher puts the caller's extra arguments.
    let mut user: Flags = Vec::new();
    let mut removed: Vec<&str> = Vec::new();
    for (k, v) in &cfg.flags {
        let toggle;
        let v = match v {
            FlagValue::Bool(b) if ON_OFF_FLAGS.contains(&k.as_str()) => {
                toggle = FlagValue::Str(if *b { "on" } else { "off" }.into());
                &toggle
            }
            v => v,
        };
        let pos = base.iter().position(|(bk, _)| bk == k);
        match (pos, v) {
            (Some(i), FlagValue::Bool(false)) => {
                base.remove(i);
                removed.push(k);
            }
            (Some(i), FlagValue::List(vs)) => {
                base.remove(i);
                for (j, x) in vs.iter().enumerate() {
                    base.insert(i + j, (k.clone(), Some(x.clone())));
                }
            }
            (Some(i), v) => base[i].1 = v.arg(),
            (None, FlagValue::Bool(false)) => removed.push(k),
            (None, FlagValue::List(vs)) => {
                user.extend(vs.iter().map(|x| (k.clone(), Some(x.clone()))));
            }
            (None, v) => user.push((k.clone(), v.arg())),
        }
    }
    // The engine refuses a decode batch larger than its sequences ("the largest decode batch is
    // 4 and the engine has state for N sequence(s)"): with no batches of the user's own, the
    // preset's ladder (TP=1: 1,2,4,8; TP=2: 1,2,3,4,8) is trimmed to what the sequences can run
    // (TP=1's default 1 sequence -> 1; 2 -> 1,2; 4 -> 1,2,4). An explicit
    // `--engine-decode-batches` is the user's: the validator names a conflict instead.
    if cfg.preset == Preset::Qwen27b && !cfg.flags.contains_key("--engine-decode-batches") {
        let seqs = base
            .iter()
            .find(|(k, _)| k == "--engine-max-seqs")
            .and_then(|(_, v)| v.as_deref()?.trim().parse::<u64>().ok());
        if let (Some(seqs), Some(b)) = (
            seqs,
            base.iter_mut()
                .find(|(k, _)| k == "--engine-decode-batches"),
        ) {
            if let Some(list) = b.1.as_deref() {
                let all: Vec<u64> = list
                    .split(',')
                    .filter_map(|x| x.trim().parse().ok())
                    .collect();
                let kept = crate::estimate::trim_batches(&all, seqs);
                if kept != all {
                    b.1 = Some(
                        kept.iter()
                            .map(u64::to_string)
                            .collect::<Vec<_>>()
                            .join(","),
                    );
                }
            }
        }
    }
    let mut tail: Vec<String> = flatten(&user);
    tail.extend(cfg.extra_args.iter().cloned());

    if cfg.preset == Preset::Flashnext && cfg.agent_defaults {
        let has = |tail: &[String], k: &str| {
            tail.iter()
                .any(|a| a == k || a.starts_with(&format!("{k}=")))
        };
        for (k, v) in FN_AGENT_DEFAULTS {
            if removed.contains(k) {
                continue;
            }
            // The launcher: an explicit `--fn-kv-ram-gib` (the slab form) drops the lazy pair.
            if has(&tail, "--fn-kv-ram-gib")
                && matches!(*k, "--fn-kv-lazy" | "--fn-kv-admit-slab-tokens")
            {
                continue;
            }
            if has(&tail, k) {
                continue;
            }
            tail.push((*k).to_string());
            if let Some(v) = v {
                tail.push((*v).to_string());
            }
        }
    }

    let mut engine = vec![binary, sub.to_string()];
    engine.extend(flatten(&base));
    engine.extend(tail);

    match cfg.runtime {
        Runtime::Direct => EngineCommand {
            spawn: engine.clone(),
            engine,
            container_name: None,
        },
        Runtime::Container => {
            let name = container_name_for(cfg);
            let spawn = container_wrap(cfg, &paths, ctx, &name, &engine);
            EngineCommand {
                engine,
                spawn,
                container_name: Some(name),
            }
        }
    }
}

/// Where a flag's effective value comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Source {
    /// The stock preset's command line (the launcher's default boot).
    Preset,
    /// Flash-Next's agent-load defaults (`engine.agent_defaults`).
    AgentDefault,
    /// A config override (`engine.flags`).
    Override,
    /// `engine.extra_args`.
    Extra,
}

/// One flag of a composed command line.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct EffectiveFlag {
    pub name: String,
    /// Every value it carries, in order (`None` = a bare switch).
    pub values: Vec<Option<String>>,
    pub source: Source,
}

/// The flags of the composed engine command (`--name value` / `--name=value` / a bare
/// `--switch`), with where each came from; the last value of a flag wins for clap's `Set`, so
/// every value is kept in order. `cfg` must be the config `engine` was composed from.
#[must_use]
pub fn effective_flags(cfg: &EngineConfig, engine: &[String]) -> Vec<EffectiveFlag> {
    let mut stock = cfg.clone();
    stock.flags.clear();
    stock.extra_args.clear();
    stock.agent_defaults = false;
    let ctx_free = ComposeCtx {
        home: PathBuf::from("/"),
        exe: PathBuf::from("/"),
        fn3_prompts_present: false,
        fn_legacy_pack_present: false,
        p2p_mounts: vec![],
    };
    let preset_names: Vec<String> = parse_argv(&compose(&stock, &ctx_free).engine)
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    let extra: Vec<String> = parse_argv(&cfg.extra_args)
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    let mut out: Vec<EffectiveFlag> = Vec::new();
    for (k, v) in parse_argv(engine) {
        if let Some(e) = out.iter_mut().find(|e| e.name == k) {
            e.values.push(v);
            continue;
        }
        let source = if cfg.flags.contains_key(&k) {
            Source::Override
        } else if extra.contains(&k) {
            Source::Extra
        } else if preset_names.contains(&k) {
            Source::Preset
        } else {
            Source::AgentDefault
        };
        out.push(EffectiveFlag {
            name: k,
            values: vec![v],
            source,
        });
    }
    out
}

/// `[bin, sub, --a, 1, --b, --c=2]` -> `[(--a, 1), (--b, None), (--c, 2)]` (a word that does
/// not start with `--` after a flag is its value).
#[must_use]
pub fn parse_argv(argv: &[String]) -> Vec<(String, Option<String>)> {
    let mut out: Vec<(String, Option<String>)> = Vec::new();
    let mut it = argv.iter().skip_while(|a| !a.starts_with("--")).peekable();
    while let Some(a) = it.next() {
        if !a.starts_with("--") {
            // a second value of a multi-value flag (`--served-model-name a b`)
            if let Some(last) = out.last_mut() {
                let joined = match last.1.take() {
                    Some(v) => format!("{v} {a}"),
                    None => a.clone(),
                };
                last.1 = Some(joined);
            }
            continue;
        }
        if let Some((k, v)) = a.split_once('=') {
            out.push((k.to_string(), Some(v.to_string())));
            continue;
        }
        let v = match it.peek() {
            Some(n) if !n.starts_with("--") => it.next().cloned(),
            _ => None,
        };
        out.push((a.clone(), v));
    }
    out
}

fn flatten(flags: &Flags) -> Vec<String> {
    let mut out = Vec::new();
    for (k, v) in flags {
        out.push(k.clone());
        if let Some(v) = v {
            out.push(v.clone());
        }
    }
    out
}

fn canon(p: &Path) -> String {
    std::fs::canonicalize(p)
        .unwrap_or_else(|_| p.to_path_buf())
        .display()
        .to_string()
}

/// The launcher's `$CTR run` around the engine, in the FOREGROUND (`--rm`, no `-d`) so the
/// console reads the logs from the client's pipes and the client's exit code is the engine's.
fn container_wrap(
    cfg: &EngineConfig,
    paths: &crate::config::Paths,
    ctx: &ComposeCtx,
    name: &str,
    engine: &[String],
) -> Vec<String> {
    let c = &cfg.container;
    let mut a: Vec<String> = vec![c.runtime.clone(), "run".into(), "--rm".into()];
    let mut push = |xs: &[&str]| a.extend(xs.iter().map(|s| s.to_string()));
    push(&["--name", name]);
    push(&[
        "--security-opt",
        "label=disable",
        "--security-opt",
        "seccomp=unconfined",
    ]);
    push(&["--device", "/dev/kfd", "--device", "/dev/dri"]);
    if c.runtime == "podman" {
        push(&["--group-add", "keep-groups"]);
    }
    push(&["--ipc=host", "--network=host"]);
    let q = format!("GPU_MAX_HW_QUEUES={}", c.gpu_max_hw_queues);
    let repo = p(&paths.repo);
    let shared = p(&paths.shared);
    let mut v: Vec<String> = Vec::new();
    match cfg.preset {
        Preset::Flashnext => {
            let nv = p(&paths.nvme);
            for e in [
                "HIP_VISIBLE_DEVICES=0,1",
                &q,
                "LD_LIBRARY_PATH=/opt/rocm/lib",
                "PYTHONDONTWRITEBYTECODE=1",
            ] {
                v.extend(["-e".into(), e.to_string()]);
            }
            v.extend(ctx.p2p_mounts.iter().cloned());
            for m in [
                format!("{shared}:{shared}"),
                format!("{nv}/models:{nv}/models:ro"),
                format!("{nv}/saber-packs:{nv}/saber-packs:ro"),
                format!("{nv}/saber-kv:{nv}/saber-kv"),
                format!("{repo}:/work:ro"),
            ] {
                v.extend(["-v".into(), m]);
            }
        }
        Preset::Qwen27b | Preset::Mock => {
            let hip = match cfg.single_card() {
                Some(d) => format!("HIP_VISIBLE_DEVICES={d}"),
                None => "HIP_VISIBLE_DEVICES=0,1".to_string(),
            };
            let packs = p(&paths.nvme.join("saber-packs"));
            let cache = canon(&paths.repo.join(".cache"));
            for m in [
                format!("{repo}:/work"),
                format!("{packs}:/packs:ro"),
                format!("{cache}:{cache}"),
                format!("{shared}:{shared}"),
                format!("{}:/serve-tokenizer.json:ro", canon(&paths.tokenizer)),
                format!("{}:/serve-template.jinja:ro", canon(&paths.chat_template)),
                format!("{repo}/.cache/cargo-registry:/opt/rust/cargo/registry"),
            ] {
                v.extend(["-v".into(), m]);
            }
            for e in [
                hip.as_str(),
                &q,
                "CUDA_VISIBLE_DEVICES=",
                "LD_LIBRARY_PATH=/opt/rocm/lib",
                "CARGO_TARGET_DIR=/work/.cache/target-container",
            ] {
                v.extend(["-e".into(), e.to_string()]);
            }
        }
    }
    a.extend(v);
    let line = engine
        .iter()
        .map(|s| shell_quote(s))
        .collect::<Vec<_>>()
        .join(" ");
    a.extend([
        "-w".into(),
        "/work".into(),
        "--entrypoint".into(),
        "/bin/bash".into(),
        c.image.clone(),
        "-c".into(),
        format!("ulimit -c 0; exec {line}"),
    ]);
    a
}

/// POSIX single-quote a word unless it is plainly safe.
#[must_use]
pub fn shell_quote(s: &str) -> String {
    let safe = !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_./,:=+@%".contains(c));
    if safe {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn ctx(home: &Path) -> ComposeCtx {
        ComposeCtx {
            home: home.to_path_buf(),
            exe: PathBuf::from("/usr/bin/saber"),
            fn3_prompts_present: false,
            fn_legacy_pack_present: false,
            p2p_mounts: vec![],
        }
    }

    #[test]
    fn overrides_replace_remove_append() {
        let mut c = Config::default().engine;
        c.flags
            .insert("--fn-thinking".into(), FlagValue::Str("off".into()));
        c.flags
            .insert("--fn-max-ctx".into(), FlagValue::Int(131072));
        c.flags
            .insert("--fn-ring-idle".into(), FlagValue::Bool(false));
        c.flags
            .insert("--fn-kv-ram-gib".into(), FlagValue::Float(3.2));
        let e = compose(&c, &ctx(Path::new("/h"))).engine;
        let line = e.join(" ");
        assert!(line.contains("--fn-thinking off"));
        assert!(!line.contains("--fn-thinking on"));
        assert!(line.contains("--fn-max-ctx 131072"));
        assert!(!line.contains("--fn-ring-idle"));
        assert!(
            !line.contains("--fn-kv-lazy"),
            "the slab form drops the lazy pair"
        );
        assert!(line.contains("--fn-prefix-tree-ram-gib 4.5"));
    }

    #[test]
    fn lists_repeat_and_effective_sources() {
        let mut c = Config::default().engine;
        c.flags.insert(
            "--api-key".into(),
            FlagValue::List(vec!["a=k1".into(), "b=k2".into()]),
        );
        c.flags
            .insert("--fn-thinking".into(), FlagValue::Str("off".into()));
        c.extra_args = vec!["--fn-route-trace=/x".into()];
        let cmd = compose(&c, &ctx(Path::new("/h")));
        let line = cmd.engine.join(" ");
        assert!(line.contains("--api-key a=k1 --api-key b=k2"), "{line}");
        let eff = effective_flags(&c, &cmd.engine);
        let g = |n: &str| eff.iter().find(|e| e.name == n).unwrap();
        assert_eq!(g("--api-key").values.len(), 2);
        assert_eq!(g("--api-key").source, Source::Override);
        assert_eq!(g("--fn-thinking").source, Source::Override);
        assert_eq!(g("--fn-thinking").values, [Some("off".to_string())]);
        assert_eq!(g("--tp").source, Source::Preset);
        assert_eq!(g("--spec-couple").source, Source::AgentDefault);
        assert_eq!(g("--fn-route-trace").source, Source::Extra);
        assert_eq!(g("--fn-ring-idle").values, [None]);
        // a preset flag replaced by a list keeps its place
        let mut m = Config::mock(8099).engine;
        m.flags.insert(
            "--served-model-name".into(),
            FlagValue::List(vec!["a".into()]),
        );
        let l = compose(&m, &ctx(Path::new("/h"))).engine.join(" ");
        assert!(l.contains("--served-model-name a") || !l.contains("--served-model-name Qwen"));
    }

    #[test]
    fn container_wrapper_is_foreground_and_named() {
        let c = Config::default().engine;
        let cmd = compose(&c, &ctx(Path::new("/h")));
        assert_eq!(cmd.container_name.as_deref(), Some("saber-flashnext-serve"));
        assert_eq!(&cmd.spawn[..3], &["podman", "run", "--rm"]);
        assert!(!cmd.spawn.iter().any(|a| a == "-d"));
        assert!(cmd
            .spawn
            .last()
            .unwrap()
            .starts_with("ulimit -c 0; exec /work/.cache"));
        assert!(cmd.spawn.iter().any(|a| a == "/h/saber:/work:ro"));
    }

    #[test]
    fn mock_is_direct() {
        let c = Config::mock(8099).engine;
        let cmd = compose(&c, &ctx(Path::new("/h")));
        assert_eq!(cmd.spawn, cmd.engine);
        assert_eq!(cmd.engine[0], "/usr/bin/saber");
        assert!(cmd
            .engine_line()
            .contains("--engine mock --mock /h/saber/examples/mock-script.toml"));
    }

    fn on_path(bin: &str) -> bool {
        std::env::var_os("PATH")
            .is_some_and(|p| std::env::split_paths(&p).any(|d| d.join(bin).is_file()))
    }

    /// Run a launcher in its `SABER_PRINT_CMD=1` mode and return the engine argv it would boot.
    fn launcher_argv(
        script: &str,
        shared: &Path,
        env: &[(&str, &str)],
        args: &[&str],
    ) -> Vec<String> {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut cmd = std::process::Command::new("bash");
        cmd.arg(repo.join(script)).arg("--no-build").args(args);
        // the scratch NVMe root beside `shared` (the test's config points at the same one)
        let nvme = shared.parent().unwrap().join("nvme");
        cmd.env("SABER_PRINT_CMD", "1")
            .env("SABER_SHARED", shared)
            .env("SABER_NVME", &nvme)
            .env("SABER_PACKS", nvme.join("saber-packs"))
            .env_remove("PORT");
        if std::env::var_os("SABER_CONTAINER").is_none() && !on_path("podman") && !on_path("docker")
        {
            // the print mode runs no container, but the launcher's runtime check wants one named
            // (a build image without podman or docker)
            cmd.env("SABER_CONTAINER", "true");
        }
        if script.ends_with("serve_27b.sh") {
            // the console mounts the configured tokenizer / chat template at these two paths
            cmd.env("TOKENIZER", "/serve-tokenizer.json")
                .env("CHAT_TEMPLATE", "/serve-template.jinja");
        }
        for (k, v) in env {
            cmd.env(k, v);
        }
        let out = cmd.output().expect("running the launcher");
        assert!(
            out.status.success(),
            "{script}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect()
    }

    /// The stock presets compose exactly what the launchers boot (and the overrides land where
    /// the launchers put extra arguments).
    #[test]
    fn launcher_parity() {
        let dir = saber_test_scratch::ScratchDir::new("console-launcher-parity");
        let shared = dir.path().join("shared");
        std::fs::create_dir_all(&shared).unwrap();
        let mk = |preset: Preset| {
            let mut c = Config::default().engine;
            c.preset = preset;
            c.paths.shared = shared.clone();
            c.paths.nvme = dir.path().join("nvme");
            c
        };
        let ctx0 = ctx(Path::new("/h"));

        // Flash-Next: defaults, another port, extra flags, the slab form, no agent defaults.
        let c = mk(Preset::Flashnext);
        let ours = compose(&c, &ctx0).engine;
        assert_eq!(
            ours,
            launcher_argv("tools/serve_flashnext.sh", &shared, &[], &[])
        );

        let mut c = mk(Preset::Flashnext);
        c.port = 8100;
        c.flags
            .insert("--fn-max-ctx".into(), FlagValue::Int(131072));
        c.flags
            .insert("--fn-route-trace".into(), FlagValue::Str("/x/y".into()));
        let ours = compose(&c, &ctx0).engine;
        let theirs = launcher_argv(
            "tools/serve_flashnext.sh",
            &shared,
            &[("PORT", "8100")],
            &["--fn-max-ctx", "131072", "--fn-route-trace", "/x/y"],
        );
        assert_eq!(ours, theirs);

        let mut c = mk(Preset::Flashnext);
        c.flags
            .insert("--fn-kv-ram-gib".into(), FlagValue::Float(3.2));
        c.flags
            .insert("--fn-prefix-tree-ram-gib".into(), FlagValue::Float(1.5));
        let theirs = launcher_argv(
            "tools/serve_flashnext.sh",
            &shared,
            &[],
            &["--fn-kv-ram-gib", "3.2", "--fn-prefix-tree-ram-gib", "1.5"],
        );
        assert_eq!(compose(&c, &ctx0).engine, theirs);

        let mut c = mk(Preset::Flashnext);
        c.agent_defaults = false;
        let theirs = launcher_argv(
            "tools/serve_flashnext.sh",
            &shared,
            &[("FN_AGENT_DEFAULTS", "0")],
            &[],
        );
        assert_eq!(compose(&c, &ctx0).engine, theirs);

        // The optional prompt set present: the launcher adds --prompt-ids.
        std::fs::create_dir_all(shared.join("ws-fn3/ref/run1")).unwrap();
        let c = mk(Preset::Flashnext);
        let mut ctx1 = ctx0.clone();
        ctx1.fn3_prompts_present = true;
        assert_eq!(
            compose(&c, &ctx1).engine,
            launcher_argv("tools/serve_flashnext.sh", &shared, &[], &[])
        );

        // 27B: defaults, another port + an extra flag.
        let c = mk(Preset::Qwen27b);
        assert_eq!(
            compose(&c, &ctx0).engine,
            launcher_argv("tools/serve_27b.sh", &shared, &[], &[])
        );
        let mut c = mk(Preset::Qwen27b);
        c.port = 8100;
        c.flags
            .insert("--spec-lookup".into(), FlagValue::Str("on".into()));
        assert_eq!(
            compose(&c, &ctx0).engine,
            launcher_argv(
                "tools/serve_27b.sh",
                &shared,
                &[("PORT", "8100")],
                &["--spec-lookup", "on"]
            )
        );
        // The launchers' env knobs are in-place flag overrides here.
        let mut c = mk(Preset::Qwen27b);
        c.flags
            .insert("--engine-max-ctx".into(), FlagValue::Int(16384));
        c.flags.insert(
            "--engine-decode-widths".into(),
            FlagValue::Str("1,8".into()),
        );
        c.flags.insert(
            "--pack".into(),
            FlagValue::Str("/packs/qwen38-27b-quark-awq-mxfp4".into()),
        );
        let env = [
            ("MAX_CTX", "16384"),
            ("WIDTHS", "1,8"),
            ("TIER", "qwen38-27b-quark-awq-mxfp4"),
        ];
        // TIER must exist for the launcher only past the print, so any name works here.
        assert_eq!(
            compose(&c, &ctx0).engine,
            launcher_argv("tools/serve_27b.sh", &shared, &env, &[])
        );
        // 27B at TP=1: `tools/serve_27b.sh --tp 1` (defaults, another port + a flag, the env
        // knobs as overrides, the card)
        let mut c = mk(Preset::Qwen27b);
        c.tp = 1;
        assert_eq!(
            compose(&c, &ctx0).engine,
            launcher_argv("tools/serve_27b.sh", &shared, &[], &["--tp", "1"])
        );
        c.port = 8100;
        c.device = 1;
        c.flags
            .insert("--spec-lookup".into(), FlagValue::Str("on".into()));
        c.flags
            .insert("--engine-max-seqs".into(), FlagValue::Int(8));
        c.flags.insert(
            "--engine-decode-batches".into(),
            FlagValue::Str("1,2,4,8".into()),
        );
        assert_eq!(
            compose(&c, &ctx0).engine,
            launcher_argv(
                "tools/serve_27b.sh",
                &shared,
                &[
                    ("PORT", "8100"),
                    ("DEVICE", "1"),
                    ("MAX_SEQS", "8"),
                    ("BATCHES", "1,2,4,8")
                ],
                &["--tp", "1", "--spec-lookup", "on"]
            )
        );
        // TP in the environment, the launcher's other spelling
        let mut c = mk(Preset::Qwen27b);
        c.tp = 1;
        assert_eq!(
            compose(&c, &ctx0).engine,
            launcher_argv("tools/serve_27b.sh", &shared, &[("TP", "1")], &[])
        );
        // and `--tp 2` is the default boot
        let c = mk(Preset::Qwen27b);
        assert_eq!(
            compose(&c, &ctx0).engine,
            launcher_argv("tools/serve_27b.sh", &shared, &[], &["--tp", "2"])
        );

        let mut c = mk(Preset::Flashnext);
        c.flags
            .insert("--fn-kv-dtype".into(), FlagValue::Str("bf16".into()));
        c.flags
            .insert("--fn-experts".into(), FlagValue::Str("ep".into()));
        let env = [("FN_KV", "bf16"), ("FN_EXPERTS", "ep")];
        // (the prompt set exists in `shared` by now)
        assert_eq!(
            compose(&c, &ctx1).engine,
            launcher_argv("tools/serve_flashnext.sh", &shared, &env, &[])
        );

        // A locally built Flash-Next pack under its own name: the launcher prefers it.
        std::fs::create_dir_all(dir.path().join("nvme").join(FN_LEGACY_PACK)).unwrap();
        let c = mk(Preset::Flashnext);
        let mut ctx2 = ctx1.clone();
        ctx2.fn_legacy_pack_present = true;
        assert_eq!(
            compose(&c, &ctx2).engine,
            launcher_argv("tools/serve_flashnext.sh", &shared, &[], &[])
        );
    }

    #[test]
    fn tp1_wrapper_names_the_card_and_its_container() {
        let mut c = Config::default().engine;
        c.preset = Preset::Qwen27b;
        let two = compose(&c, &ctx(Path::new("/h")));
        assert_eq!(two.container_name.as_deref(), Some("saber-27b-serve"));
        assert!(two.spawn.iter().any(|a| a == "HIP_VISIBLE_DEVICES=0,1"));
        c.tp = 1;
        c.device = 1;
        let one = compose(&c, &ctx(Path::new("/h")));
        assert_eq!(one.container_name.as_deref(), Some(tp1::CONTAINER));
        assert!(one.spawn.iter().any(|a| a == "HIP_VISIBLE_DEVICES=1"));
        assert!(!one.spawn.iter().any(|a| a == "HIP_VISIBLE_DEVICES=0,1"));
        let line = one.engine_line();
        // inside the container the one card is device 0, and there is no peer
        assert!(
            line.contains("--engine-tp 1 --engine-device 0 --hip-lib"),
            "{line}"
        );
        assert!(!line.contains("--engine-peer-device"), "{line}");
        assert!(
            line.contains(&format!("--pack /packs/{}", tp1::PACK)),
            "{line}"
        );
        assert!(line.contains(&format!("--engine-draft-pack /packs/{}", tp1::DRAFT_PACK)));
        // the single-card default: one conversation of up to 131,072 tokens, batch 1
        assert!(line.contains("--engine-max-ctx 131072") && line.contains("--engine-max-seqs 1"));
        assert!(line.contains("--engine-decode-batches 1 "), "{line}");
        // a named container wins
        c.container.name = "mine".into();
        assert_eq!(
            compose(&c, &ctx(Path::new("/h"))).container_name.as_deref(),
            Some("mine")
        );
        // Flash-Next ignores the field (its validate refuses tp 1)
        let f = Config::default().engine;
        assert_eq!(container_name_for(&f), "saber-flashnext-serve");
    }

    #[test]
    fn an_on_off_toggle_switched_off_writes_off() {
        let mut c = Config::mock(8099).engine;
        // `false` used to drop the flag, which left the engine's default (on)
        c.flags
            .insert("--enable-prefix-caching".into(), FlagValue::Bool(false));
        let line = compose(&c, &ctx(Path::new("/h"))).engine.join(" ");
        assert!(line.ends_with("--enable-prefix-caching off"), "{line}");
        c.flags
            .insert("--enable-prefix-caching".into(), FlagValue::Bool(true));
        let line = compose(&c, &ctx(Path::new("/h"))).engine.join(" ");
        assert!(line.ends_with("--enable-prefix-caching on"), "{line}");
        c.flags.insert(
            "--enable-prefix-caching".into(),
            FlagValue::Str("off".into()),
        );
        let cmd = compose(&c, &ctx(Path::new("/h")));
        assert!(cmd
            .engine
            .join(" ")
            .ends_with("--enable-prefix-caching off"));
        let eff = effective_flags(&c, &cmd.engine);
        let pc = eff
            .iter()
            .find(|e| e.name == "--enable-prefix-caching")
            .unwrap();
        assert_eq!(pc.values, [Some("off".to_string())]);
        assert_eq!(pc.source, Source::Override);
        // another flag's `false` still removes it
        c.flags
            .insert("--step-delay-ms".into(), FlagValue::Bool(false));
        assert!(!compose(&c, &ctx(Path::new("/h")))
            .engine
            .iter()
            .any(|a| a == "--step-delay-ms"));
    }

    /// The batches follow the sequences: the TP's ladder (TP=1 1,2,4,8; TP=2 1,2,3,4,8) trimmed
    /// to the sequences (the engine refuses a batch larger than its sequences); explicit batches
    /// are the user's and are left to the validator.
    #[test]
    fn the_batches_follow_the_sequences() {
        let mut c = Config::default().engine;
        c.preset = Preset::Qwen27b;
        c.tp = 1;
        let line =
            |c: &crate::config::EngineConfig| compose(c, &ctx(Path::new("/h"))).engine_line();
        for (seqs, want) in [
            (None, "1"),
            (Some(2), "1,2"),
            (Some(4), "1,2,4"),
            (Some(8), "1,2,4,8"),
        ] {
            if let Some(s) = seqs {
                c.flags
                    .insert("--engine-max-seqs".into(), FlagValue::Int(s));
            }
            let l = line(&c);
            assert!(
                l.contains(&format!("--engine-decode-batches {want} ")),
                "{seqs:?}: {l}"
            );
        }
        // TP=2's ladder at 3 sequences, and its default 8
        let mut t = Config::default().engine;
        t.preset = Preset::Qwen27b;
        assert!(line(&t).contains("--engine-decode-batches 1,2,3,4,8 "));
        t.flags
            .insert("--engine-max-seqs".into(), FlagValue::Int(3));
        assert!(
            line(&t).contains("--engine-decode-batches 1,2,3 "),
            "{}",
            line(&t)
        );
        // the user's own batches stay as given
        c.flags.insert(
            "--engine-decode-batches".into(),
            FlagValue::Str("1,2".into()),
        );
        assert!(line(&c).contains("--engine-decode-batches 1,2 "));
    }

    #[test]
    fn quoting() {
        assert_eq!(shell_quote("a/b-c"), "a/b-c");
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
        assert_eq!(shell_quote(""), "''");
    }
}
