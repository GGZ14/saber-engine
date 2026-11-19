//! `saber serve --model <name>`: the front door. It picks the TP from the GPUs, finds (or pulls)
//! the packs, composes the engine's full command line from the registry entry (explicit flags
//! still override any of it), prints what it is about to do and runs it: in the dev container
//! from the host, or directly when already inside a container.
//!
//! The command it runs never carries `--model`, so it never comes back here.

use std::io::IsTerminal;
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use super::cache::{self, Status};
use super::plan::{Plan, Skip};
use super::registry::{Entry, Kind, TpSpec};
use super::ui::{human, tilde, Res, Style, UserError};
use super::{choose_tp, drafter_of, engine_supports, short, Ctx};

/// The HIP runtime inside the dev image.
pub const HIP_LIB: &str = "/opt/rocm/lib/libamdhip64.so.7";
/// The dev image the launchers run the engine in.
pub const IMAGE: &str = "localhost/saber-dev:latest";
/// The prebuilt kernels' arch when no GPU says otherwise.
pub const ARCH: &str = "gfx1201";

/// Flags of `saber serve` that turn the front door on (any of them) ...
const FRONT_FLAGS: &[&str] = &["--model", "--dry-run", "--print-argv"];
/// ... and flags that mean the caller is driving the engine by hand (any of them, without
/// `--model`, keeps the old behaviour).
const ENGINE_FLAGS: &[&str] = &["--engine", "--pack", "--mock", "--manifest", "--help", "-h"];

/// Does `saber serve <args>` go through the front door? With `--model` (or `--dry-run`), or with
/// none of the flags that drive the engine by hand: a bare `saber serve` serves the default model.
#[must_use]
pub fn wants_front_door(args: &[String]) -> bool {
    let has = |set: &[&str]| {
        args.iter().any(|a| {
            set.iter()
                .any(|f| a == f || a.starts_with(&format!("{f}=")))
        })
    };
    has(FRONT_FLAGS) || !has(ENGINE_FLAGS)
}

/// The front door's own options; everything else is an engine flag override.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Opts {
    pub model: Option<String>,
    pub tp: Option<u32>,
    pub drafter: Option<String>,
    pub dry_run: bool,
    pub print_argv: bool,
    pub yes: bool,
    pub device: Option<u32>,
    pub packs_dir: Option<PathBuf>,
    pub image: Option<String>,
    /// `container` / `direct`.
    pub runtime: Option<String>,
    /// `podman` / `docker`.
    pub container_runtime: Option<String>,
    pub name: Option<String>,
    pub help: bool,
    /// Engine flags in order: (flag, value).
    pub overrides: Vec<(String, Option<String>)>,
}

/// Parse `saber serve` arguments into the front door's options and the engine overrides.
pub fn parse(args: &[String]) -> Res<Opts> {
    let mut o = Opts::default();
    let mut i = 0;
    let usage = |m: String| {
        UserError::new(m)
            .usage()
            .fix("saber serve --model <name> --help".to_string())
    };
    while i < args.len() {
        let a = &args[i];
        let (flag, inline) = match a.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (a.clone(), None),
        };
        let takes = matches!(
            flag.as_str(),
            "--model"
                | "--tp"
                | "--drafter"
                | "--device"
                | "--packs-dir"
                | "--image"
                | "--runtime"
                | "--container-runtime"
                | "--name"
        );
        let mut value = || -> Res<String> {
            if let Some(v) = &inline {
                return Ok(v.clone());
            }
            i += 1;
            args.get(i)
                .filter(|v| !v.starts_with("--"))
                .cloned()
                .ok_or_else(|| usage(format!("{flag} needs a value")))
        };
        match flag.as_str() {
            "--model" | "-m" => o.model = Some(value()?),
            "--tp" => {
                let v = value()?;
                o.tp = Some(
                    v.parse()
                        .map_err(|_| usage(format!("--tp {v}: 1 (one GPU) or 2 (two GPUs)")))?,
                );
            }
            "--drafter" => o.drafter = Some(value()?),
            "--device" => {
                let v = value()?;
                o.device = Some(
                    v.parse()
                        .map_err(|_| usage(format!("--device {v}: a GPU index (0, 1, ...)")))?,
                );
            }
            "--packs-dir" => o.packs_dir = Some(PathBuf::from(value()?)),
            "--image" => o.image = Some(value()?),
            "--runtime" => {
                let v = value()?;
                if v != "container" && v != "direct" {
                    return Err(usage(format!("--runtime {v}: `container` (the dev image, the default on a host) or `direct` (this process, inside a container)")));
                }
                o.runtime = Some(v);
            }
            "--container-runtime" => {
                let v = value()?;
                if v != "podman" && v != "docker" {
                    return Err(usage(format!("--container-runtime {v}: podman or docker")));
                }
                o.container_runtime = Some(v);
            }
            "--name" => o.name = Some(value()?),
            "--dry-run" => o.dry_run = true,
            "--print-argv" => o.print_argv = true,
            "--yes" | "-y" => o.yes = true,
            "--help" | "-h" => o.help = true,
            f if f.starts_with('-') && !takes => {
                let v = match inline {
                    Some(v) => Some(v),
                    None => match args.get(i + 1) {
                        Some(n) if !n.starts_with("--") => {
                            i += 1;
                            Some(n.clone())
                        }
                        _ => None,
                    },
                };
                o.overrides.push((f.to_string(), v));
            }
            _ => {
                return Err(usage(format!(
                    "unexpected `{a}` (did you mean `--model {a}`?)"
                )));
            }
        }
        i += 1;
    }
    Ok(o)
}

pub const HELP: &str = "\
saber serve --model <name> [options] [engine flags...]

Serve a model from the registry (`saber models list`). Picks the TP from your GPUs, finds the
packs (or offers to download them), and composes the engine's whole command line from the
model's entry. `saber serve` with no --model serves the default model.

Options:
  --model, -m <name>     a model name or alias (default: the registry's default)
  --tp <1|2>             GPUs to use (default: 2 if you have two, else 1)
  --device <n>           TP=1: which GPU (default 0)
  --drafter <name|none>  another drafter entry, or none
  --dry-run              print what would happen (the exact commands) and stop
  --print-argv           print the engine's argv, one word a line, and stop
  --yes, -y              download missing packs without asking
  --packs-dir <dir>      where packs are cached (default $SABER_PACKS, else ~/.cache/saber/packs)
  --runtime <container|direct>   container = run in the dev image (the default on a host);
                         direct = run here (the default inside a container)
  --image <ref>          the dev image (default localhost/saber-dev:latest)
  --container-runtime <podman|docker>   (default: podman if installed, else docker)
  --name <container>     the container's name (default saber-serve, saber-serve-tp1 at TP=1)

Engine flags (any `saber serve --engine real` flag, e.g. --port 8081, --engine-max-ctx 65536,
--served-model-name X) replace the entry's value or are added after it.

Examples:
  saber serve                                   # the default model
  saber serve --model mxfp4 --tp 1 --dry-run    # show what would run
  saber serve --model w4a8 --port 8081 --engine-max-ctx 131072
";

/// The engine argv as (flag, value) pairs.
type Flags = Vec<(String, Option<String>)>;

fn f(k: &str, v: impl Into<String>) -> (String, Option<String>) {
    (k.to_string(), Some(v.into()))
}

/// Raw tokens (`["--vision", "lazy", "--x"]`) as pairs.
fn pairs(tokens: &[String]) -> Flags {
    let mut out = Vec::new();
    let mut it = tokens.iter().peekable();
    while let Some(t) = it.next() {
        let v = match it.peek() {
            Some(n) if !n.starts_with("--") => it.next().cloned(),
            _ => None,
        };
        out.push((t.clone(), v));
    }
    out
}

/// What the engine is given, before the caller's overrides.
pub struct Inputs<'a> {
    pub target: &'a Entry,
    pub spec: &'a TpSpec,
    pub tp: u32,
    pub drafter: Option<&'a Entry>,
    pub pack: PathBuf,
    pub draft_pack: Option<PathBuf>,
    pub tokenizer: PathBuf,
    pub chat_template: PathBuf,
    pub prebuilt: PathBuf,
    pub binary: String,
}

/// The engine's argv for `inp` with `overrides` applied: an override replaces the flag of the
/// same name in place (all of its occurrences), anything else is appended.
#[must_use]
pub fn compose(inp: &Inputs<'_>, overrides: &[(String, Option<String>)]) -> Vec<String> {
    let t = inp.spec;
    let mut b: Flags = vec![
        f("--engine", "real"),
        f("--engine-tp", inp.tp.to_string()),
        f("--engine-device", "0"),
    ];
    if inp.tp == 2 {
        b.push(f("--engine-peer-device", "1"));
    }
    b.extend([
        f("--hip-lib", HIP_LIB),
        f("--pack", inp.pack.display().to_string()),
        f("--prebuilt", inp.prebuilt.display().to_string()),
        f("--tokenizer", inp.tokenizer.display().to_string()),
        f("--chat-template", inp.chat_template.display().to_string()),
    ]);
    if let Some(c) = t.max_ctx {
        b.push(f("--engine-max-ctx", c.to_string()));
    }
    if let Some(r) = &t.prefill_rows {
        b.push(f("--engine-prefill-rows", r.clone()));
    }
    if let Some(x) = &t.decode_batches {
        b.push(f("--engine-decode-batches", x.clone()));
    }
    let widths = if inp.drafter.is_some() {
        t.decode_widths.clone().unwrap_or_else(|| "1".into())
    } else {
        "1".into()
    };
    b.push(f("--engine-decode-widths", widths));
    if let Some(s) = t.max_seqs {
        b.push(f("--engine-max-seqs", s.to_string()));
    }
    if let Some(dp) = &inp.draft_pack {
        b.push(f(
            "--engine-draft-block",
            t.draft_block.unwrap_or(8).to_string(),
        ));
        b.push(f("--engine-draft-pack", dp.display().to_string()));
    }
    if let Some(m) = t.budget_mib {
        b.push(f("--engine-budget-mib", m.to_string()));
    }
    b.extend(pairs(&inp.target.engine_args));
    b.push(f(
        "--served-model-name",
        inp.target.served_id.clone().unwrap_or_default(),
    ));
    b.push(f("--host", "0.0.0.0"));
    b.push(f("--port", "8080"));
    b.extend(pairs(&t.tail_args));

    // overrides: in place, else appended
    let mut extra: Flags = Vec::new();
    let mut seen: Vec<&str> = Vec::new();
    for (k, v) in overrides {
        if b.iter().any(|(bk, _)| bk == k) {
            if seen.contains(&k.as_str()) {
                // a repeated flag of the caller's: keep each value
                extra.push((k.clone(), v.clone()));
                continue;
            }
            let first = b.iter().position(|(bk, _)| bk == k).expect("present");
            b.retain(|(bk, _)| bk != k);
            b.insert(first, (k.clone(), v.clone()));
            seen.push(k);
        } else {
            extra.push((k.clone(), v.clone()));
        }
    }
    // the engine refuses a decode batch above its sequences: trim the entry's ladder to a
    // `--engine-max-seqs` override (an explicit `--engine-decode-batches` is the caller's)
    let ov = |k: &str| overrides.iter().any(|(x, _)| x == k);
    if ov("--engine-max-seqs") && !ov("--engine-decode-batches") {
        let seqs = b
            .iter()
            .find(|(k, _)| k == "--engine-max-seqs")
            .and_then(|(_, v)| v.as_deref()?.parse::<u32>().ok());
        if let (Some(n), Some(bat)) = (
            seqs,
            b.iter_mut().find(|(k, _)| k == "--engine-decode-batches"),
        ) {
            if let Some(list) = bat.1.clone() {
                let kept: Vec<String> = list
                    .split(',')
                    .filter(|x| x.trim().parse::<u32>().is_ok_and(|v| v <= n))
                    .map(str::to_string)
                    .collect();
                bat.1 = Some(if kept.is_empty() {
                    "1".into()
                } else {
                    kept.join(",")
                });
            }
        }
    }
    b.extend(extra);
    let mut argv = vec![inp.binary.clone(), "serve".into()];
    for (k, v) in b {
        argv.push(k);
        if let Some(v) = v {
            argv.push(v);
        }
    }
    argv
}

/// The value of `flag` in `argv`.
#[must_use]
pub fn arg_of<'a>(argv: &'a [String], flag: &str) -> Option<&'a str> {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .map(String::as_str)
}

/// The container around `engine`: the dev image, the GPUs, host networking, and every host path
/// the command names mounted read-only at the same path (so the engine argv reads the same in and
/// out of the container).
#[must_use]
pub fn container_argv(
    ctr: &str,
    image: &str,
    name: &str,
    hip_devices: &str,
    engine: &[String],
) -> Vec<String> {
    let mut a: Vec<String> = [
        ctr,
        "run",
        "--rm",
        "--name",
        name,
        "--security-opt",
        "label=disable",
        "--security-opt",
        "seccomp=unconfined",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    if ctr == "podman" {
        a.extend(["--group-add".into(), "keep-groups".into()]);
    }
    a.extend(["--ipc=host".into(), "--network=host".into()]);
    for e in [
        format!("HIP_VISIBLE_DEVICES={hip_devices}"),
        "GPU_MAX_HW_QUEUES=3".into(),
        "CUDA_VISIBLE_DEVICES=".into(),
        "LD_LIBRARY_PATH=/opt/rocm/lib".into(),
    ] {
        a.extend(["-e".into(), e]);
    }
    let mut mounts: Vec<PathBuf> = Vec::new();
    let hip = arg_of(engine, "--hip-lib");
    for (i, w) in engine.iter().enumerate() {
        if !w.starts_with('/') || Some(w.as_str()) == hip || (i > 0 && engine[i - 1] == "--hip-lib")
        {
            continue;
        }
        let p = PathBuf::from(w);
        // the binary and the inputs the front door resolved are mounted even before a pull has
        // made them; any other absolute path only when it exists
        let named = i == 0 || (i > 0 && MOUNT_FLAGS.contains(&engine[i - 1].as_str()));
        if !named && !p.exists() {
            continue;
        }
        if mounts.iter().any(|m| p.starts_with(m)) {
            continue;
        }
        mounts.retain(|m| !m.starts_with(&p));
        mounts.push(p);
    }
    for m in &mounts {
        let src = std::fs::canonicalize(m).unwrap_or_else(|_| m.clone());
        a.extend(["-v".into(), format!("{}:{}:ro", src.display(), m.display())]);
    }
    let line = engine
        .iter()
        .map(|s| saber_console::compose::shell_quote(s))
        .collect::<Vec<_>>()
        .join(" ");
    a.extend([
        "-w".into(),
        "/tmp".into(),
        "--entrypoint".into(),
        "/bin/bash".into(),
        image.into(),
        "-c".into(),
        format!("ulimit -c 0; exec {line}"),
    ]);
    a
}

/// Engine flags whose value is a host path the container must see.
const MOUNT_FLAGS: &[&str] = &[
    "--pack",
    "--engine-draft-pack",
    "--prebuilt",
    "--tokenizer",
    "--chat-template",
];

/// A command as a copy-pasteable shell block (` \` continuations before each flag).
#[must_use]
pub fn pretty(argv: &[String]) -> String {
    let mut out = String::from("  ");
    let mut col = 2;
    for (i, w) in argv.iter().enumerate() {
        let q = saber_console::compose::shell_quote(w);
        let flag = w.starts_with('-') && i > 0;
        if flag && col > 2 && col + q.len() > 70 {
            out.push_str(" \\\n    ");
            col = 4;
        } else if i > 0 {
            out.push(' ');
            col += 1;
        }
        out.push_str(&q);
        col += q.len();
    }
    out
}

/// Inside a container already?
fn in_container() -> bool {
    Path::new("/run/.containerenv").exists() || Path::new("/.dockerenv").exists()
}

fn on_path(bin: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|p| std::env::split_paths(&p).any(|d| d.join(bin).is_file()))
}

/// The prebuilt kernels for `arch`: next to the binary, in an ancestor of it (a checkout's
/// `.cache/target-container/release/saber`), or under the current directory.
fn find_prebuilt(exe: &Path, arch: &str) -> Option<PathBuf> {
    let mut cands: Vec<PathBuf> = exe
        .ancestors()
        .skip(1)
        .take(6)
        .map(|d| d.join("prebuilt").join(arch))
        .collect();
    if let Ok(cwd) = std::env::current_dir() {
        cands.push(cwd.join("prebuilt").join(arch));
    }
    cands.into_iter().find(|p| p.is_dir())
}

/// Running saber containers and the GPUs each holds (`None` = no HIP_VISIBLE_DEVICES: all).
fn holders(ctr: &str) -> Vec<(String, Option<Vec<u32>>)> {
    let Ok(o) = Command::new(ctr)
        .args(["ps", "--format", "{{.Names}}"])
        .output()
    else {
        return Vec::new();
    };
    let names = String::from_utf8_lossy(&o.stdout);
    let mut out = Vec::new();
    for n in names.lines().filter(|n| n.starts_with("saber")) {
        let env = Command::new(ctr)
            .args([
                "inspect",
                "-f",
                "{{range .Config.Env}}{{println .}}{{end}}",
                n,
            ])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            .unwrap_or_default();
        let held: Option<Vec<u32>> = env
            .lines()
            .find_map(|l| l.strip_prefix("HIP_VISIBLE_DEVICES="))
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
        out.push((n.to_string(), held));
    }
    out
}

/// Is `host:port` free to listen on?
fn port_free(port: u16) -> bool {
    TcpListener::bind(("0.0.0.0", port)).is_ok()
}

/// The front door. `validate` parses an engine argv with the CLI's own parser (typos in engine
/// flags are caught before anything starts). Returns the exit code.
pub fn run(args: &[String], validate: &dyn Fn(&[String]) -> Result<(), String>) -> i32 {
    super::quiet_pipe();
    super::report_code(serve(args, validate))
}

fn serve(args: &[String], validate: &dyn Fn(&[String]) -> Result<(), String>) -> Res<i32> {
    let o = parse(args)?;
    if o.help {
        print!("{HELP}");
        return Ok(0);
    }
    let ctx = Ctx::load(o.packs_dir.clone())?;
    let st = Style::stdout();
    let e = match &o.model {
        Some(m) => ctx.reg.resolve(m)?,
        None => ctx.reg.default_entry(),
    };
    if e.kind == Kind::Drafter {
        let users: Vec<&str> = ctx.reg.users_of(e).iter().map(|u| short(u)).collect();
        let mut err = UserError::new(format!(
            "{} is a drafter: it is served beside a model, not on its own",
            e.name
        ))
        .usage();
        if let Some(u) = users.first() {
            err = err.fix(format!("saber serve --model {u}"));
        }
        return Err(err);
    }
    let (tp, why) = choose_tp(e, o.tp, &ctx.gpus, "saber serve --model")?;
    let spec = e.at(tp).expect("chosen from the entry");
    let d = drafter_of(&ctx.reg, e, o.drafter.as_deref())?;
    let ov = |k: &str| {
        o.overrides
            .iter()
            .find(|(x, _)| x == k)
            .and_then(|(_, v)| v.clone())
    };
    // explicit paths win; only what is not given comes from the cache
    let skip = Skip {
        target_pack: ov("--pack").is_some(),
        drafter_pack: ov("--engine-draft-pack").is_some(),
        files: ov("--tokenizer").is_some() && ov("--chat-template").is_some(),
    };
    let mut problems: Vec<UserError> = Vec::new();
    for x in std::iter::once(e).chain(d) {
        if let Err(err) = engine_supports(x) {
            if !o.dry_run {
                return Err(err);
            }
            problems.push(err);
        }
    }
    let cmd_base = format!(
        "saber serve --model {}{}",
        short(e),
        if why.starts_with("you asked") {
            format!(" --tp {tp}")
        } else {
            String::new()
        }
    );
    let pull_cmd = format!(
        "saber models pull {}{}{}",
        short(e),
        if why.starts_with("you asked") {
            format!(" --tp {tp}")
        } else {
            String::new()
        },
        o.drafter
            .as_ref()
            .map_or(String::new(), |x| format!(" --drafter {x}"))
    );

    // packs: here, or pull them first
    let mut plan = Plan::build(&ctx, e, d, tp, false, false, &pull_cmd, &skip)?;
    if plan.has_work() {
        if o.dry_run || o.print_argv {
            if !o.print_argv {
                eprintln!(
                    "{} would download these first ({}):",
                    st.yellow("Not downloaded yet:"),
                    pull_cmd
                );
                plan.print(&ctx);
            }
        } else {
            eprintln!(
                "{} is not downloaded yet{}:",
                e.name,
                if tp == 2 { "" } else { " for one GPU" }
            );
            plan.print(&ctx);
            let go = o.yes || {
                if !std::io::stdin().is_terminal() {
                    return Err(UserError::new(format!(
                        "{} is not downloaded, and there is no terminal to ask",
                        e.name
                    ))
                    .fix(pull_cmd.clone())
                    .fix(format!("or: {cmd_base} --yes   (download, then serve)")));
                }
                super::ui::confirm(
                    &format!("Download {} now and then serve?", human(plan.need())),
                    true,
                    &cmd_base,
                )?
            };
            if !go {
                return Err(UserError::new("not downloaded; nothing started").fix(pull_cmd));
            }
            plan = Plan::build(&ctx, e, d, tp, true, true, &pull_cmd, &skip)?;
            plan.check_disk(&ctx)?;
            plan.execute(&ctx)?;
        }
    }
    // the packs it will serve, checked (manifest, TP, rank headers, format tag)
    let mut path_of = |x: &Entry, given: Option<String>| -> PathBuf {
        if let Some(g) = given {
            return PathBuf::from(g);
        }
        match cache::locate(ctx.root(), x, tp) {
            Status::Ready { dir, .. } => {
                if let Err(p) = cache::check_pack(&dir, tp, x.requires_tag.as_deref()) {
                    problems.push(
                        UserError::new(format!(
                            "the {} pack at {} is not usable: {p}",
                            x.name,
                            tilde(&dir)
                        ))
                        .fix(format!(
                            "saber models rm {} --tp {tp} --yes && {pull_cmd}",
                            short(x)
                        )),
                    );
                }
                dir
            }
            _ => cache::tp_dir(ctx.root(), x, tp).expect("has the TP"),
        }
    };
    let pack = path_of(e, ov("--pack"));
    let draft_pack = d.map(|x| path_of(x, ov("--engine-draft-pack")));
    let rd = cache::repo_dir(ctx.root(), e);
    let tokenizer = ov("--tokenizer").map_or_else(
        || rd.join(e.tokenizer.as_deref().unwrap_or("tokenizer.json")),
        PathBuf::from,
    );
    let chat_template = ov("--chat-template").map_or_else(
        || rd.join(e.chat_template.as_deref().unwrap_or("chat_template.jinja")),
        PathBuf::from,
    );

    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("saber"));
    let arch = ctx.gpus.gfx.clone().unwrap_or_else(|| ARCH.to_string());
    let prebuilt = match ov("--prebuilt") {
        Some(p) => PathBuf::from(p),
        None => find_prebuilt(&exe, &arch).unwrap_or_else(|| {
            problems.push(
                UserError::new(format!("cannot find the prebuilt kernels (prebuilt/{arch}) next to {}", exe.display()))
                    .fix(format!("run it from the saber checkout, or pass --prebuilt /path/to/saber/prebuilt/{arch}")),
            );
            PathBuf::from(format!("prebuilt/{arch}"))
        }),
    };
    let inp = Inputs {
        target: e,
        spec,
        tp,
        drafter: d,
        pack,
        draft_pack,
        tokenizer,
        chat_template,
        prebuilt,
        binary: exe.display().to_string(),
    };
    let engine = compose(&inp, &o.overrides);
    if let Err(msg) = validate(&engine) {
        // clap's message without its (very long) usage line
        let msg: Vec<&str> = msg
            .lines()
            .map(|l| l.strip_prefix("error: ").unwrap_or(l))
            .filter(|l| !l.starts_with("Usage:") && !l.starts_with("For more information"))
            .filter(|l| !l.trim().is_empty())
            .collect();
        return Err(UserError::new(format!(
            "the engine does not accept this command line: {}",
            msg.join("\n  ")
        ))
        .usage()
        .fix("check the flag's spelling: saber serve --help lists every engine flag".to_string()));
    }
    if o.print_argv {
        for w in &engine {
            println!("{w}");
        }
        return Ok(0);
    }

    // where it runs
    let runtime = o.runtime.clone().unwrap_or_else(|| {
        if in_container() {
            "direct".into()
        } else {
            "container".into()
        }
    });
    let ctr = o.container_runtime.clone().unwrap_or_else(|| {
        if on_path("podman") || !on_path("docker") {
            "podman".into()
        } else {
            "docker".into()
        }
    });
    let device = o.device.unwrap_or(0);
    if tp == 1 && ctx.gpus.count > 0 && device as usize >= ctx.gpus.count {
        return Err(UserError::new(format!(
            "--device {device}: there is no GPU {device} here (this machine has GPU 0..{}: {})",
            ctx.gpus.count - 1,
            ctx.gpus.describe()
        ))
        .usage()
        .fix(format!("{cmd_base} --device 0")));
    }
    let devices: Vec<u32> = if tp == 1 { vec![device] } else { vec![0, 1] };
    let hip_devices = devices
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let cname = o.name.clone().unwrap_or_else(|| {
        if tp == 1 {
            format!(
                "saber-serve-tp1{}",
                if device > 0 {
                    format!("-gpu{device}")
                } else {
                    String::new()
                }
            )
        } else {
            "saber-serve".into()
        }
    });
    let image = o.image.clone().unwrap_or_else(|| IMAGE.into());
    let spawn: Vec<String> = if runtime == "container" {
        container_argv(&ctr, &image, &cname, &hip_devices, &engine)
    } else {
        engine.clone()
    };

    // guards (local only: no network)
    let port: u16 = arg_of(&engine, "--port")
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let served = arg_of(&engine, "--served-model-name")
        .unwrap_or("?")
        .to_string();
    if ctx.gpus.count == 0 {
        problems.push(
            UserError::new("no supported GPU found (Saber's kernels are built for RDNA4, gfx12)")
                .note("checked /sys/class/kfd: is the amdgpu driver loaded and are you in the `render` and `video` groups?")
                .fix("ls /dev/kfd /dev/dri && groups"),
        );
    }
    if !port_free(port) {
        problems.push(
            UserError::new(format!(
                "port {port} is already in use (another server is listening there)"
            ))
            .fix(format!("{cmd_base} --port {}", port + 1)),
        );
    }
    if runtime == "container" && on_path(&ctr) {
        let all = holders(&ctr);
        let holds = |h: &Option<Vec<u32>>, d: u32| h.as_ref().is_none_or(|v| v.contains(&d));
        let clash: Vec<String> = all
            .iter()
            .filter(|(_, h)| devices.iter().any(|d| holds(h, *d)))
            .map(|(n, h)| {
                format!(
                    "{n} (GPU {})",
                    h.as_ref().map_or("all".to_string(), |v| v
                        .iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join(","))
                )
            })
            .collect();
        if !clash.is_empty() {
            let free: Vec<u32> = (0..ctx.gpus.count as u32)
                .filter(|d| !all.iter().any(|(_, h)| holds(h, *d)))
                .collect();
            let mut err = UserError::new(format!("the GPUs are in use by {}", clash.join(", ")))
                .fix(format!(
                    "stop it first: {ctr} stop {}",
                    clash[0].split(' ').next().unwrap_or("")
                ));
            if let Some(f) = free.first() {
                err = err.fix(format!(
                    "or serve on the free GPU: saber serve --model {} --tp 1 --device {f}",
                    short(e)
                ));
            }
            problems.push(err);
        }
        let img_ok = Command::new(&ctr)
            .args(["image", "inspect", &image])
            .output()
            .is_ok_and(|o| o.status.success());
        if !img_ok {
            problems.push(
                UserError::new(format!("the container image {image} is not built")).fix(
                    "tools/build-image.sh   (once; it needs only podman or docker and the network)",
                ),
            );
        }
    } else if runtime == "container" {
        problems.push(
            UserError::new(format!(
                "{ctr} is not installed (the engine runs in the dev container)"
            ))
            .fix("install podman (or docker), or run inside the dev image with --runtime direct"),
        );
    }

    // the summary
    let dname = d.map_or("none".to_string(), |x| x.name.clone());
    let head = if o.dry_run { "Would serve" } else { "Serving" };
    println!(
        "{} {} as {} on :{port}, drafter {}, TP {tp}",
        st.bold(head),
        st.bold(&e.name),
        st.cyan(&served),
        dname,
    );
    println!("  {}", st.dim(&format!("TP {tp}: {why}")));
    println!("  packs     {}", tilde(&inp.pack));
    if let Some(dp) = &inp.draft_pack {
        println!("  drafter   {}", tilde(dp));
    }
    println!(
        "  shapes    {} tokens a sequence, {} sequence{}, verify widths {}",
        arg_of(&engine, "--engine-max-ctx")
            .and_then(|v| v.parse().ok())
            .map_or("?".to_string(), super::group),
        arg_of(&engine, "--engine-max-seqs").unwrap_or("?"),
        if arg_of(&engine, "--engine-max-seqs") == Some("1") {
            ""
        } else {
            "s"
        },
        arg_of(&engine, "--engine-decode-widths").unwrap_or("1"),
    );
    if runtime == "container" {
        println!("  runs in   {ctr} container `{cname}` ({image}), GPUs {hip_devices}");
    } else {
        println!("  runs      directly in this process's environment (GPUs as visible here)");
    }
    let curl = format!(
        "curl -s http://localhost:{port}/v1/chat/completions -H 'Content-Type: application/json' -d '{{\"model\": \"{served}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'"
    );
    if o.dry_run {
        println!();
        println!("{}", st.bold("Engine command:"));
        println!("{}", pretty(&engine));
        if runtime == "container" {
            println!();
            println!("{}", st.bold("Container command:"));
            println!("{}", pretty(&spawn));
        }
        println!();
        println!("{}", st.bold("Then try it:"));
        println!("  {curl}");
        if !problems.is_empty() {
            println!();
            println!(
                "{}",
                st.yellow(&format!(
                    "A real run would stop here ({} problem{}):",
                    problems.len(),
                    if problems.len() == 1 { "" } else { "s" }
                ))
            );
            for p in &problems {
                print!("{}", p.render(st.0));
            }
        }
        println!();
        println!("{}", st.dim("Dry run: nothing was started or downloaded."));
        return Ok(0);
    }
    if let Some(first) = problems.into_iter().next() {
        return Err(first);
    }
    println!();
    println!("Booting (the packs load from disk; a 27B takes a minute or two). Ctrl-C stops it.");
    println!("Once it says ready, try:\n  {curl}");
    println!();
    let mut child = match Command::new(&spawn[0]).args(&spawn[1..]).spawn() {
        Ok(c) => c,
        Err(err) => {
            return Err(UserError::new(format!("cannot start {} ({err})", spawn[0]))
                .fix(format!("{cmd_base} --dry-run   (shows the exact command)")))
        }
    };
    // announce readiness once /v1/models answers
    let ready_msg = format!(
        "{} {served} is listening on http://localhost:{port}/v1",
        st.green("Ready:")
    );
    std::thread::spawn(move || {
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(1800) {
            std::thread::sleep(Duration::from_secs(2));
            if models_ok(port) {
                eprintln!(
                    "\n{ready_msg} (after {})\n",
                    super::ui::duration(t0.elapsed().as_secs_f64())
                );
                return;
            }
        }
    });
    let status = child
        .wait()
        .map_err(|err| UserError::new(format!("lost the server process ({err})")))?;
    let code = status.code().unwrap_or(1);
    if code != 0 && code != 130 {
        let mut err = UserError::new(format!(
            "the server exited with code {code}{}",
            if code == 86 {
                " (a GPU rank failed; the log's last line says why)"
            } else {
                ""
            }
        ));
        if runtime == "container" {
            err = err.fix(format!(
                "{cmd_base} --dry-run   (check the command), then look at the log above"
            ));
        }
        return Err(err);
    }
    Ok(code)
}

/// `GET /v1/models` answers 200?
fn models_ok(port: u16) -> bool {
    use std::io::{Read, Write};
    let Ok(mut s) =
        TcpStream::connect_timeout(&([127, 0, 0, 1], port).into(), Duration::from_secs(1))
    else {
        return false;
    };
    let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
    if s.write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut buf = [0u8; 16];
    let n = s.read(&mut buf).unwrap_or(0);
    buf[..n].starts_with(b"HTTP/1.1 200") || buf[..n].starts_with(b"HTTP/1.0 200")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::registry::Registry;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| (*x).to_string()).collect()
    }

    #[test]
    fn the_front_door_triggers() {
        assert!(wants_front_door(&s(&["--model", "w4a8"])));
        assert!(wants_front_door(&s(&[])));
        assert!(wants_front_door(&s(&["--tp", "1"])));
        assert!(wants_front_door(&s(&["--dry-run"])));
        assert!(wants_front_door(&s(&["--model=w4a8", "--pack", "/x"])));
        assert!(!wants_front_door(&s(&["--engine", "real", "--pack", "/x"])));
        assert!(!wants_front_door(&s(&["--engine", "mock"])));
        assert!(!wants_front_door(&s(&["--help"])));
    }

    #[test]
    fn options_and_overrides_parse() {
        let o = parse(&s(&[
            "--model",
            "w4a8",
            "--tp=1",
            "--dry-run",
            "--port",
            "8081",
            "--engine-rectangular-widths",
            "--vision",
            "on",
            "-y",
        ]))
        .unwrap();
        assert_eq!(o.model.as_deref(), Some("w4a8"));
        assert_eq!(o.tp, Some(1));
        assert!(o.dry_run && o.yes);
        assert_eq!(
            o.overrides,
            vec![
                ("--port".to_string(), Some("8081".to_string())),
                ("--engine-rectangular-widths".to_string(), None),
                ("--vision".to_string(), Some("on".to_string())),
            ]
        );
        let e = parse(&s(&["w4a8"])).unwrap_err().render(false);
        assert!(e.contains("did you mean `--model w4a8`"), "{e}");
        let e = parse(&s(&["--tp", "two"])).unwrap_err().render(false);
        assert!(e.contains("1 (one GPU) or 2"), "{e}");
        let e = parse(&s(&["--model"])).unwrap_err().render(false);
        assert!(e.contains("--model needs a value"), "{e}");
    }

    fn inputs<'a>(r: &'a Registry, name: &str, tp: u32) -> Inputs<'a> {
        let e = r.resolve(name).unwrap();
        let d = crate::models::drafter_of(r, e, None).unwrap();
        Inputs {
            target: e,
            spec: e.at(tp).unwrap(),
            tp,
            drafter: d,
            pack: PathBuf::from("/c/t"),
            draft_pack: d.map(|_| PathBuf::from("/c/d")),
            tokenizer: PathBuf::from("/c/tok.json"),
            chat_template: PathBuf::from("/c/t.jinja"),
            prebuilt: PathBuf::from("/r/prebuilt/gfx1201"),
            binary: "/r/saber".into(),
        }
    }

    /// The default entry composes the production 27B boot's argv, flag for flag (paths and the
    /// served id are the caller's).
    #[test]
    fn the_default_entry_composes_the_two_card_boot() {
        let r = Registry::load(None).unwrap();
        let argv = compose(&inputs(&r, "default", 2), &[]);
        let want =
            "/r/saber serve --engine real --engine-tp 2 --engine-device 0 --engine-peer-device 1 \
            --hip-lib /opt/rocm/lib/libamdhip64.so.7 --pack /c/t --prebuilt /r/prebuilt/gfx1201 \
            --tokenizer /c/tok.json --chat-template /c/t.jinja --engine-max-ctx 262144 \
            --engine-prefill-rows 96,128,192,256,320,384,512,768,1024,1536,2048 \
            --engine-decode-batches 1,2,3,4,8 --engine-decode-widths 1,5,8 --engine-max-seqs 8 \
            --engine-draft-block 8 --engine-draft-pack /c/d --engine-budget-mib 30500 \
            --engine-rectangular-widths --served-model-name Qwen3.8-27B-MACaQ-W4A8-SABER \
            --host 0.0.0.0 --port 8080 --vision lazy";
        assert_eq!(
            argv.join(" "),
            want.split_whitespace().collect::<Vec<_>>().join(" ")
        );
    }

    #[test]
    fn one_card_and_overrides() {
        let r = Registry::load(None).unwrap();
        let argv = compose(
            &inputs(&r, "w4a8", 1),
            &[
                ("--port".into(), Some("8081".into())),
                ("--engine-max-seqs".into(), Some("2".into())),
                ("--warm-prompts".into(), Some("/w.txt".into())),
            ],
        );
        let j = argv.join(" ");
        assert!(
            j.contains("--engine-tp 1 --engine-device 0 --hip-lib"),
            "{j}"
        );
        assert!(!j.contains("--engine-peer-device"), "{j}");
        assert!(j.contains("--engine-max-ctx 131072"), "{j}");
        assert!(j.contains("--port 8081 "), "{j}");
        assert!(!j.contains("--port 8080"), "{j}");
        assert!(j.contains("--engine-max-seqs 2"), "{j}");
        // the ladder `1` stays within 2 sequences
        assert!(j.contains("--engine-decode-batches 1 "), "{j}");
        assert!(j.ends_with("--warm-prompts /w.txt"), "{j}");
        assert!(!j.contains("--vision"), "{j}");
        // the TP=2 ladder is trimmed to a smaller sequence count
        let argv = compose(
            &inputs(&r, "w4a8", 2),
            &[("--engine-max-seqs".into(), Some("3".into()))],
        );
        assert_eq!(arg_of(&argv, "--engine-decode-batches"), Some("1,2,3"));
    }

    #[test]
    fn no_drafter_means_width_one() {
        let r = Registry::load(None).unwrap();
        let mut i = inputs(&r, "w4a8", 2);
        i.drafter = None;
        i.draft_pack = None;
        let argv = compose(&i, &[]);
        assert_eq!(arg_of(&argv, "--engine-decode-widths"), Some("1"));
        assert!(arg_of(&argv, "--engine-draft-pack").is_none());
        assert!(arg_of(&argv, "--engine-draft-block").is_none());
    }

    #[test]
    fn the_container_mounts_named_paths_read_only() {
        let sc = saber_test_scratch::ScratchDir::new("models-front-ctr");
        let pack = sc.join("packs/t");
        std::fs::create_dir_all(&pack).unwrap();
        let engine = vec![
            "/nonexistent/saber".to_string(),
            "serve".into(),
            "--hip-lib".into(),
            HIP_LIB.into(),
            "--pack".into(),
            pack.display().to_string(),
            "--tokenizer".into(),
            "/not/yet/pulled.json".into(),
            "--warm-prompts".into(),
            "/does/not/exist.txt".into(),
        ];
        let a = container_argv("podman", IMAGE, "saber-serve", "0,1", &engine);
        let j = a.join(" ");
        assert!(j.starts_with("podman run --rm --name saber-serve"), "{j}");
        assert!(j.contains("--group-add keep-groups"), "{j}");
        assert!(j.contains("-e HIP_VISIBLE_DEVICES=0,1"), "{j}");
        assert!(
            j.contains(&format!(
                "{}:{}:ro",
                std::fs::canonicalize(&pack).unwrap().display(),
                pack.display()
            )),
            "{j}"
        );
        assert!(
            j.contains("-v /nonexistent/saber:/nonexistent/saber:ro"),
            "{j}"
        );
        assert!(!j.contains("libamdhip64.so.7:"), "{j}");
        // a resolved input is mounted before it exists (a dry run before the pull) ...
        assert!(
            j.contains("-v /not/yet/pulled.json:/not/yet/pulled.json:ro"),
            "{j}"
        );
        // ... any other path only when it does
        assert!(!j.contains("exist.txt:"), "{j}");
        assert!(
            a.last()
                .unwrap()
                .starts_with("ulimit -c 0; exec /nonexistent/saber serve"),
            "{j}"
        );
        let d = container_argv("docker", IMAGE, "x", "1", &engine).join(" ");
        assert!(!d.contains("keep-groups"), "{d}");
    }

    #[test]
    fn pretty_wraps_at_flags() {
        let p = pretty(&s(&["saber", "serve", "--a", "1", "--b", "two words"]));
        assert_eq!(p, "  saber serve --a 1 --b 'two words'");
    }
}
