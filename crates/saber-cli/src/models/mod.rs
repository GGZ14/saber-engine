//! `saber models` (list / info / pull / rm) and the `saber serve --model <name>` front door.
//!
//! The registry (`models.toml`, docs/MODELS.md) names each model's Hugging Face repo, its pack
//! folder per TP degree, its served id, its drafter and its serving shapes; this module turns a
//! name into packs on disk and an engine command line. It is the only part of `saber` that reads
//! the environment (`HOME`, `XDG_*`, `SABER_PACKS`, `HF_*`): the engine command it composes is
//! fully explicit, as every engine command line is.

pub mod cache;
pub mod fetch;
pub mod front;
pub mod plan;
pub mod registry;
pub mod ui;

use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use clap::{Args, Subcommand};
use serde::Serialize;

use cache::{Env, Status};
use registry::{Entry, Kind, Registry};
use ui::{human, tilde, Res, Style, UserError};

#[derive(Args, Debug, Clone)]
pub struct ModelsArgs {
    #[command(subcommand)]
    pub cmd: Option<ModelsCmd>,

    /// Where packs are cached (default: $SABER_PACKS, else ~/.cache/saber/packs)
    #[arg(long, global = true, value_name = "DIR")]
    pub packs_dir: Option<PathBuf>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum ModelsCmd {
    /// List every model: size, whether it is downloaded, its drafter (the default with no subcommand)
    #[command(visible_alias = "ls")]
    List {
        /// Machine-readable output
        #[arg(long)]
        json: bool,
    },
    /// Everything about one model: repo, served id, drafter, sizes, shapes, cache paths
    #[command(visible_alias = "show")]
    Info {
        /// A model name or alias (`saber models list` shows them)
        name: String,
        #[arg(long)]
        json: bool,
    },
    /// Download a model (and its drafter, if not already there). Resumable; verifies every file
    #[command(visible_alias = "download")]
    Pull {
        /// A model name or alias; omitted = the default model
        name: Option<String>,
        /// Tensor-parallel degree (1 = one GPU, 2 = two). Default: from the GPUs this machine has
        #[arg(long)]
        tp: Option<u32>,
        /// Use another drafter entry than the model's own (a name or alias), or `none`
        #[arg(long)]
        drafter: Option<String>,
        /// Show what would be downloaded, and where, without downloading
        #[arg(long)]
        dry_run: bool,
        /// Do not ask before downloading
        #[arg(long, short)]
        yes: bool,
    },
    /// Delete a downloaded model (its drafter too, unless another downloaded model uses it)
    #[command(visible_alias = "remove")]
    Rm {
        /// A model name or alias
        name: String,
        /// Only this TP degree's pack (default: every TP downloaded)
        #[arg(long)]
        tp: Option<u32>,
        /// Show what would be deleted without deleting
        #[arg(long)]
        dry_run: bool,
        /// Do not ask before deleting
        #[arg(long, short)]
        yes: bool,
    },
}

/// Everything a command needs: the merged registry, the environment, the GPUs.
pub struct Ctx {
    pub reg: Registry,
    pub env: Env,
    pub gpus: Gpus,
}

/// The supported (gfx12) discrete GPUs the kernel's KFD topology lists. Read from sysfs: no GPU
/// is opened.
#[derive(Clone, Debug, Default)]
pub struct Gpus {
    pub count: usize,
    pub gfx: Option<String>,
    pub names: Vec<String>,
}

impl Gpus {
    #[must_use]
    pub fn probe() -> Gpus {
        let all = saber_console::system::probe_gpus(Path::new("/"));
        let ok: Vec<_> = all
            .iter()
            .filter(|g| g.supported && !g.integrated)
            .collect();
        Gpus {
            count: ok.len(),
            gfx: ok.first().map(|g| g.gfx.clone()),
            names: ok
                .iter()
                .map(|g| {
                    if g.name.is_empty() {
                        g.gfx.clone()
                    } else {
                        g.name.clone()
                    }
                })
                .collect(),
        }
    }

    #[must_use]
    pub fn describe(&self) -> String {
        let names = dedup_join(&self.names);
        match self.count {
            0 => "no supported GPU".into(),
            1 => format!("1 GPU: {names}"),
            n if names.contains(", ") => format!("{n} GPUs: {names}"),
            n => format!("{n} x {names}"),
        }
    }
}

fn dedup_join(v: &[String]) -> String {
    let mut out: Vec<&str> = Vec::new();
    for x in v {
        if !out.contains(&x.as_str()) {
            out.push(x);
        }
    }
    out.join(", ")
}

impl Ctx {
    /// The registry (shipped + the user's file), the environment and the GPUs.
    pub fn load(packs_dir: Option<PathBuf>) -> Res<Ctx> {
        let env = Env::from_process(packs_dir);
        let user_text = match std::fs::read_to_string(&env.user_registry) {
            Ok(t) => Some(t),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                return Err(UserError::new(format!(
                    "cannot read {} ({e})",
                    env.user_registry.display()
                ))
                .fix(format!(
                    "fix its permissions, or move it aside: mv {0} {0}.off",
                    env.user_registry.display()
                )))
            }
        };
        let mut reg = Registry::load(
            user_text
                .as_deref()
                .map(|t| (t, env.user_registry.as_path())),
        )?;
        if let Some(ep) = &env.hf_endpoint {
            if reg.endpoint == "https://huggingface.co" {
                reg.endpoint.clone_from(ep);
            }
        }
        Ok(Ctx {
            reg,
            env,
            gpus: Gpus::probe(),
        })
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.env.packs_root
    }
}

/// The TP to use for `e`: the flag, else what the GPUs allow, else the entry's preference.
/// Returns the TP and a phrase saying why.
pub fn choose_tp(e: &Entry, flag: Option<u32>, gpus: &Gpus, what: &str) -> Res<(u32, String)> {
    let have = e.tps();
    if let Some(t) = flag {
        if e.at(t).is_none() {
            let mut err = UserError::new(format!(
                "{} has no TP={t} pack (it has TP={})",
                e.name,
                have.iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
            .usage();
            for h in have.iter().rev() {
                err = err.fix(format!("{what} {} --tp {h}", short(e)));
            }
            return Err(err);
        }
        if gpus.count > 0 && t as usize > gpus.count {
            return Err(UserError::new(format!(
                "TP={t} needs {t} GPUs; this machine has {}",
                gpus.describe()
            ))
            .usage()
            .fix(format!("{what} {} --tp {}", short(e), gpus.count)));
        }
        return Ok((t, format!("you asked for --tp {t}")));
    }
    match gpus.count {
        0 => {
            let t = e.preferred_tp();
            Ok((
                t,
                format!("no supported GPU found here; TP={t} is the model's default"),
            ))
        }
        n => {
            let fit: Vec<u32> = have.iter().copied().filter(|t| *t as usize <= n).collect();
            match fit.last() {
                Some(t) => Ok((*t, format!("{} found", gpus.describe()))),
                None => Err(UserError::new(format!(
                    "{} needs {} GPUs; this machine has {}",
                    e.name,
                    have.first().copied().unwrap_or(2),
                    gpus.describe()
                ))
                .fix("pick a model with a TP=1 pack: saber models list".to_string())),
            }
        }
    }
}

/// The handle commands print for `e`: its first alias (other than `default`), else its name.
#[must_use]
pub fn short(e: &Entry) -> &str {
    e.aliases
        .iter()
        .find(|a| a.as_str() != "default")
        .map_or(e.name.as_str(), String::as_str)
}

/// The entry `e` uses as its drafter (`override_` = `--drafter`, `none` = no drafter).
pub fn drafter_of<'a>(
    reg: &'a Registry,
    e: &'a Entry,
    override_: Option<&str>,
) -> Res<Option<&'a Entry>> {
    let name = match override_ {
        Some(n) if n.eq_ignore_ascii_case("none") => return Ok(None),
        Some(n) => n,
        None => match &e.drafter {
            Some(d) => d.as_str(),
            None => return Ok(None),
        },
    };
    let d = reg.resolve(name)?;
    if d.kind != Kind::Drafter {
        return Err(
            UserError::new(format!("`{name}` is a model to serve, not a drafter"))
                .usage()
                .fix("pick a drafter: saber models list   (the Drafters section)".to_string()),
        );
    }
    Ok(Some(d))
}

/// The pack format tag `e` needs, when this build cannot load it.
pub fn engine_supports(e: &Entry) -> Res<()> {
    let Some(t) = &e.requires_tag else {
        return Ok(());
    };
    let tags = registry::engine_tags();
    if tags.contains(t) {
        return Ok(());
    }
    Err(UserError::new(format!(
        "{} is a `{t}` pack, and this saber build cannot load it (it loads: {})",
        e.name,
        if tags.is_empty() {
            "no tagged packs".into()
        } else {
            tags.join(", ")
        }
    ))
    .fix("update saber: git pull, then rebuild (tools/saber --build)")
    .fix("or pick another model: saber models list"))
}

// ------------------------------------------------------------------------------------------------
// list / info

#[derive(Serialize)]
struct TpJson {
    tp: u32,
    path: String,
    bytes: u64,
    status: &'static str,
    dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_seqs: Option<u32>,
}

#[derive(Serialize)]
struct EntryJson<'a> {
    name: &'a str,
    aliases: &'a [String],
    kind: Kind,
    default: bool,
    description: &'a str,
    repo: &'a str,
    revision: &'a str,
    endpoint: &'a str,
    served_id: Option<&'a str>,
    drafter: Option<String>,
    used_by: Vec<String>,
    requires_tag: Option<&'a str>,
    engine_supports_tag: bool,
    license: &'a str,
    origin: &'a registry::Origin,
    tps: Vec<TpJson>,
}

fn status_word(s: &Status) -> &'static str {
    match s {
        Status::Ready { legacy: true, .. } => "downloaded (older layout)",
        Status::Ready { verified: true, .. } => "downloaded",
        Status::Ready { .. } => "downloaded (not verified)",
        Status::Partial { .. } => "partial",
        Status::Missing => "not downloaded",
    }
}

fn entry_json<'a>(ctx: &'a Ctx, e: &'a Entry) -> EntryJson<'a> {
    let root = ctx.root();
    EntryJson {
        name: &e.name,
        aliases: &e.aliases,
        kind: e.kind,
        default: ctx.reg.default_entry().name == e.name,
        description: &e.description,
        repo: &e.repo,
        revision: &e.revision,
        endpoint: ctx.reg.endpoint_of(e),
        served_id: e.served_id.as_deref(),
        drafter: e
            .drafter
            .as_deref()
            .and_then(|d| ctx.reg.find(d))
            .map(|d| d.name.clone()),
        used_by: ctx.reg.users_of(e).iter().map(|u| u.name.clone()).collect(),
        requires_tag: e.requires_tag.as_deref(),
        engine_supports_tag: engine_supports(e).is_ok(),
        license: &e.license,
        origin: &e.origin,
        tps: e
            .tps()
            .into_iter()
            .map(|tp| {
                let t = e.at(tp).expect("listed");
                let s = cache::locate(root, e, tp);
                TpJson {
                    tp,
                    path: t.path.clone(),
                    bytes: t.bytes,
                    status: status_word(&s),
                    dir: s.dir().map(|d| d.display().to_string()),
                    max_ctx: t.max_ctx,
                    max_seqs: t.max_seqs,
                }
            })
            .collect(),
    }
}

/// `tp2` / `tp1, tp2` / `partial` / `no`.
fn downloaded_cell(ctx: &Ctx, e: &Entry) -> String {
    let mut have = Vec::new();
    let mut partial = false;
    for tp in e.tps() {
        match cache::locate(ctx.root(), e, tp) {
            Status::Ready { .. } => have.push(format!("tp{tp}")),
            Status::Partial { .. } => partial = true,
            Status::Missing => {}
        }
    }
    if !have.is_empty() {
        have.join(", ")
    } else if partial {
        "partial".into()
    } else {
        "no".into()
    }
}

pub fn list(ctx: &Ctx, json: bool) -> Res<()> {
    if json {
        let v: Vec<_> = ctx.reg.entries.iter().map(|e| entry_json(ctx, e)).collect();
        println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
        return Ok(());
    }
    let st = Style::stdout();
    let def = ctx.reg.default_entry().name.clone();
    let machine_tp =
        |e: &Entry| choose_tp(e, None, &ctx.gpus, "").map_or(e.preferred_tp(), |x| x.0);
    let targets: Vec<&Entry> = ctx
        .reg
        .entries
        .iter()
        .filter(|e| e.kind == Kind::Target)
        .collect();
    let drafters: Vec<&Entry> = ctx
        .reg
        .entries
        .iter()
        .filter(|e| e.kind == Kind::Drafter)
        .collect();

    let mut rows: Vec<[String; 5]> = Vec::new();
    for e in &targets {
        let tp = machine_tp(e);
        let d = drafter_of(&ctx.reg, e, None).ok().flatten();
        let bytes =
            e.at(tp).map_or(0, |t| t.bytes) + d.and_then(|d| d.at(tp)).map_or(0, |t| t.bytes);
        let mark = if e.name == def { "*" } else { " " };
        let mut aliases: Vec<&str> = e.aliases.iter().map(String::as_str).collect();
        aliases.truncate(3);
        rows.push([
            format!("{mark} {}", e.name),
            aliases.join(", "),
            format!("{} (TP={tp})", human(bytes)),
            downloaded_cell(ctx, e),
            d.map_or("none".to_string(), |d| short(d).to_string()),
        ]);
    }
    print_table(
        st,
        &["  MODEL", "ALIASES", "SIZE", "DOWNLOADED", "DRAFTER"],
        &rows,
    );
    if !drafters.is_empty() {
        println!();
        let mut rows: Vec<[String; 5]> = Vec::new();
        for d in &drafters {
            let tp = machine_tp(d);
            let users: Vec<&str> = ctx.reg.users_of(d).iter().map(|u| short(u)).collect();
            rows.push([
                format!("  {}", d.name),
                d.aliases.first().cloned().unwrap_or_default(),
                format!("{} (TP={tp})", human(d.at(tp).map_or(0, |t| t.bytes))),
                downloaded_cell(ctx, d),
                if users.is_empty() {
                    "-".into()
                } else {
                    users.join(", ")
                },
            ]);
        }
        print_table(
            st,
            &["  DRAFTER", "ALIAS", "SIZE", "DOWNLOADED", "USED BY"],
            &rows,
        );
    }
    println!();
    println!(
        "{}",
        st.dim(&format!(
            "* = the default (`saber serve` with no --model). Sizes include the drafter, at the TP this machine runs ({}).",
            ctx.gpus.describe()
        ))
    );
    println!(
        "{}",
        st.dim(&format!(
            "Packs live in {}{}.",
            tilde(ctx.root()),
            match ctx.env.packs_root_from {
                "default" => String::new(),
                f => format!(" (from {f})"),
            }
        ))
    );
    println!(
        "Next: {}  |  {}  |  {}",
        st.cyan("saber models info <name>"),
        st.cyan("saber models pull <name>"),
        st.cyan("saber serve --model <name>")
    );
    Ok(())
}

fn print_table(st: Style, head: &[&str; 5], rows: &[[String; 5]]) {
    let mut w = [0usize; 5];
    for (i, h) in head.iter().enumerate() {
        w[i] = h.len();
    }
    for r in rows {
        for (i, c) in r.iter().enumerate() {
            w[i] = w[i].max(c.len());
        }
    }
    let line = |cells: Vec<String>| {
        cells
            .iter()
            .enumerate()
            .map(|(i, c)| {
                if i + 1 == cells.len() {
                    c.clone()
                } else {
                    format!("{c:<width$}", width = w[i])
                }
            })
            .collect::<Vec<_>>()
            .join("  ")
    };
    println!(
        "{}",
        st.bold(&line(head.iter().map(|s| (*s).to_string()).collect()))
    );
    for r in rows {
        println!("{}", line(r.to_vec()));
    }
}

pub fn info(ctx: &Ctx, name: &str, json: bool) -> Res<()> {
    let e = ctx.reg.resolve(name)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&entry_json(ctx, e)).unwrap_or_default()
        );
        return Ok(());
    }
    let st = Style::stdout();
    let is_default = ctx.reg.default_entry().name == e.name;
    println!(
        "{}{}",
        st.bold(&e.name),
        if is_default {
            st.dim("  (the default)")
        } else {
            String::new()
        }
    );
    if !e.description.is_empty() {
        println!("{}", wrap(&e.description, 96, ""));
    }
    println!();
    let kv = |k: &str, v: String| println!("  {k:<13}{v}");
    kv("kind", format!("{:?}", e.kind).to_lowercase());
    if !e.aliases.is_empty() {
        kv("aliases", e.aliases.join(", "));
    }
    kv(
        "repo",
        format!(
            "{}/{}",
            ctx.reg.endpoint_of(e).trim_end_matches('/'),
            e.repo
        ),
    );
    if e.revision != "main" {
        kv("revision", e.revision.clone());
    }
    if let Some(s) = &e.served_id {
        kv("served id", s.clone());
    }
    if e.kind == Kind::Target {
        let d = drafter_of(&ctx.reg, e, None).ok().flatten();
        kv(
            "drafter",
            d.map_or("none".into(), |d| format!("{} ({})", d.name, short(d))),
        );
    } else {
        let u: Vec<String> = ctx.reg.users_of(e).iter().map(|u| u.name.clone()).collect();
        kv(
            "used by",
            if u.is_empty() {
                "-".into()
            } else {
                u.join(", ")
            },
        );
    }
    if let Some(t) = &e.requires_tag {
        let ok = engine_supports(e).is_ok();
        kv(
            "pack format",
            format!(
                "{t}  {}",
                if ok {
                    st.green("(this saber loads it)")
                } else {
                    st.red("(this saber CANNOT load it: update saber)")
                }
            ),
        );
    }
    if !e.license.is_empty() {
        kv("license", e.license.clone());
    }
    if e.origin == registry::Origin::User {
        kv(
            "defined in",
            tilde(ctx.reg.user_file.as_deref().unwrap_or(Path::new("?"))),
        );
    }
    println!();
    let here = choose_tp(e, None, &ctx.gpus, "").map(|x| x.0).ok();
    for tp in e.tps().into_iter().rev() {
        let t = e.at(tp).expect("listed");
        let s = cache::locate(ctx.root(), e, tp);
        let word = status_word(&s);
        let word = if s.ready() {
            st.green(word)
        } else {
            st.yellow(word)
        };
        println!(
            "  {} ({} GPU{})  {}  {}{}",
            st.bold(&format!("TP={tp}")),
            tp,
            if tp == 1 { "" } else { "s" },
            human(t.bytes),
            word,
            if here == Some(tp) && ctx.gpus.count > 0 {
                st.dim("  <- this machine")
            } else {
                String::new()
            }
        );
        let where_ = s.dir().map_or_else(
            || {
                cache::tp_dir(ctx.root(), e, tp)
                    .map(|d| tilde(&d))
                    .unwrap_or_default()
            },
            tilde,
        );
        println!(
            "      {:<11}{where_}",
            if s.ready() { "at" } else { "goes to" }
        );
        if e.kind == Kind::Target {
            println!(
                "      {:<11}{} tokens a sequence, {} sequence{}, verify widths {}, decode batches {}",
                "serves",
                t.max_ctx.map_or("?".into(), group),
                t.max_seqs.unwrap_or(0),
                if t.max_seqs == Some(1) { "" } else { "s" },
                t.decode_widths.as_deref().unwrap_or("1"),
                t.decode_batches.as_deref().unwrap_or("?")
            );
        }
    }
    println!();
    if e.kind == Kind::Target {
        println!(
            "  pull:  {}",
            st.cyan(&format!("saber models pull {}", short(e)))
        );
        println!(
            "  serve: {}",
            st.cyan(&format!("saber serve --model {}", short(e)))
        );
    } else {
        println!(
            "  {}",
            st.dim("Pulled with the models that use it; serve one of those.")
        );
    }
    Ok(())
}

/// `262,144`.
pub fn group(n: u32) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn wrap(text: &str, width: usize, indent: &str) -> String {
    let mut out = String::new();
    let mut line = String::from(indent);
    for w in text.split_whitespace() {
        if line.len() + w.len() + 1 > width && !line.trim().is_empty() {
            out.push_str(line.trim_end());
            out.push('\n');
            line = String::from(indent);
        }
        line.push_str(w);
        line.push(' ');
    }
    out.push_str(line.trim_end());
    out
}

// ------------------------------------------------------------------------------------------------
// pull / rm

pub fn pull(
    ctx: &Ctx,
    name: Option<&str>,
    tp: Option<u32>,
    drafter: Option<&str>,
    dry_run: bool,
    yes: bool,
) -> Res<()> {
    let e = match name {
        Some(n) => ctx.reg.resolve(n)?,
        None => ctx.reg.default_entry(),
    };
    let (tp, why) = choose_tp(e, tp, &ctx.gpus, "saber models pull")?;
    let d = if e.kind == Kind::Target {
        drafter_of(&ctx.reg, e, drafter)?
    } else {
        None
    };
    engine_supports(e)?;
    if let Some(d) = d {
        engine_supports(d)?;
    }
    let retry = format!(
        "saber models pull {}{}",
        short(e),
        if why.starts_with("you asked") {
            format!(" --tp {tp}")
        } else {
            String::new()
        }
    );
    let serve_hint = format!(
        "saber serve --model {}{}{}",
        short(e),
        if why.starts_with("you asked") {
            format!(" --tp {tp}")
        } else {
            String::new()
        },
        drafter.map_or(String::new(), |x| format!(" --drafter {x}"))
    );
    let p = plan::Plan::build(
        ctx,
        e,
        d,
        tp,
        true,
        !dry_run,
        &retry,
        &plan::Skip::default(),
    )?;
    let st = Style::stderr();
    eprintln!("{} {} at TP={tp} ({why})", st.bold("Pulling"), e.name);
    p.print(ctx);
    if dry_run {
        eprintln!("{}", st.dim("Dry run: nothing was downloaded."));
        return Ok(());
    }
    if p.need() == 0 && !p.has_work() {
        eprintln!("{} everything is already here.", st.green("Nothing to do:"));
        if e.kind == Kind::Target {
            eprintln!("Serve it: {}", st.cyan(&serve_hint));
        }
        return Ok(());
    }
    p.check_disk(ctx)?;
    if !yes
        && std::io::stdin().is_terminal()
        && p.need() > (1 << 30)
        && !ui::confirm(&format!("Download {}?", human(p.need())), true, &retry)?
    {
        return Err(UserError::new("download cancelled; nothing changed").fix(retry));
    }
    p.execute(ctx)?;
    if e.kind == Kind::Target {
        eprintln!("Ready. Serve it: {}", st.cyan(&serve_hint));
    }
    Ok(())
}

/// One thing `rm` deletes.
struct Doomed {
    label: String,
    path: PathBuf,
    bytes: u64,
}

pub fn rm(ctx: &Ctx, name: &str, tp: Option<u32>, dry_run: bool, yes: bool) -> Res<()> {
    let e = ctx.reg.resolve(name)?;
    let st = Style::stderr();
    let root = ctx.root();
    let tps: Vec<u32> = match tp {
        Some(t) if e.at(t).is_none() => {
            return Err(UserError::new(format!("{} has no TP={t} pack", e.name))
                .usage()
                .fix(format!("saber models rm {}", short(e))));
        }
        Some(t) => vec![t],
        None => e.tps(),
    };
    let mut doomed: Vec<Doomed> = Vec::new();
    let mut kept: Vec<String> = Vec::new();
    let mut warn: Vec<String> = Vec::new();
    let consider =
        |x: &Entry, t: u32, doomed: &mut Vec<Doomed>, kept: &mut Vec<String>| match cache::locate(
            root, x, t,
        ) {
            Status::Ready {
                legacy: true, dir, ..
            } => kept.push(format!(
            "{} TP={t}: an older download; saber models never deletes those (by hand: rm -r {})",
            x.name,
            dir.display()
        )),
            Status::Ready { dir, .. } | Status::Partial { dir, .. } => doomed.push(Doomed {
                label: format!("{} TP={t}", x.name),
                bytes: cache::dir_bytes(&dir),
                path: dir,
            }),
            Status::Missing => {}
        };
    for &t in &tps {
        consider(e, t, &mut doomed, &mut kept);
        match e.kind {
            Kind::Target => {
                if let Some(d) = drafter_of(&ctx.reg, e, None)? {
                    let others: Vec<&str> = ctx
                        .reg
                        .users_of(d)
                        .into_iter()
                        .filter(|u| u.name != e.name && cache::locate(root, u, t).ready())
                        .map(|u| u.name.as_str())
                        .collect();
                    if others.is_empty() {
                        consider(d, t, &mut doomed, &mut kept);
                    } else if cache::locate(root, d, t).ready() {
                        kept.push(format!(
                            "drafter {} TP={t}: still used by {}",
                            d.name,
                            others.join(", ")
                        ));
                    }
                }
            }
            Kind::Drafter => {
                let users: Vec<&str> = ctx
                    .reg
                    .users_of(e)
                    .into_iter()
                    .filter(|u| cache::locate(root, u, t).ready())
                    .map(|u| u.name.as_str())
                    .collect();
                if !users.is_empty() && cache::locate(root, e, t).ready() {
                    warn.push(format!(
                        "{} (downloaded) uses this drafter at TP={t}; serving it would download the drafter again",
                        users.join(", ")
                    ));
                }
            }
        }
    }
    // the shared files go when no TP of the entry is left
    let left: Vec<u32> = e
        .tps()
        .into_iter()
        .filter(|t| !tps.contains(t) && cache::locate(root, e, *t).ready())
        .collect();
    let rd = cache::repo_dir(root, e);
    if left.is_empty() {
        for f in &e.files {
            let p = rd.join(f);
            if let Ok(m) = std::fs::metadata(&p) {
                doomed.push(Doomed {
                    label: format!("{} {f}", e.name),
                    bytes: m.len(),
                    path: p,
                });
            }
        }
    }
    for k in &kept {
        eprintln!("  {} {k}", st.dim("keep"));
    }
    if doomed.is_empty() && !kept.is_empty() {
        eprintln!("Nothing removed.");
        return Ok(());
    }
    if doomed.is_empty() {
        eprintln!(
            "{} is not downloaded{}: nothing to remove.",
            e.name,
            tp.map_or(String::new(), |t| format!(" at TP={t}"))
        );
        return Ok(());
    }
    let total: u64 = doomed.iter().map(|d| d.bytes).sum();
    for d in &doomed {
        eprintln!(
            "  {} {:<48} {:>9}  {}",
            st.red("delete"),
            d.label,
            human(d.bytes),
            tilde(&d.path)
        );
    }
    for w in &warn {
        eprintln!("  {} {w}", st.yellow("note"));
    }
    eprintln!("  frees {}", human(total));
    if dry_run {
        eprintln!("{}", st.dim("Dry run: nothing was deleted."));
        return Ok(());
    }
    let cmd = format!(
        "saber models rm {}{}",
        short(e),
        tp.map_or(String::new(), |t| format!(" --tp {t}"))
    );
    if !yes && !ui::confirm("Delete these files?", warn.is_empty(), &cmd)? {
        return Err(UserError::new("nothing was deleted").fix(format!("{cmd} --yes")));
    }
    for d in &doomed {
        let r = if d.path.is_dir() {
            std::fs::remove_dir_all(&d.path)
        } else {
            std::fs::remove_file(&d.path)
        };
        r.map_err(|err| {
            UserError::new(format!("cannot delete {} ({err})", d.path.display()))
                .fix(format!("check its permissions, then run `{cmd}` again"))
        })?;
        // empty parents up to the root go too (`<owner>/<repo>/`)
        let mut p = d.path.parent().map(Path::to_path_buf);
        while let Some(dir) = p {
            if dir == root || !dir.starts_with(root) || std::fs::remove_dir(&dir).is_err() {
                break;
            }
            p = dir.parent().map(Path::to_path_buf);
        }
    }
    eprintln!("{} {} freed.", st.green("Removed."), human(total));
    Ok(())
}

/// Output piped into `head` ends the process quietly (the default SIGPIPE), as other command
/// line tools do, instead of a panic on the next print.
pub fn quiet_pipe() {
    // SAFETY: restoring the default disposition of one signal; no handler runs.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

/// `saber models ...`: prints errors itself and returns the exit code.
pub fn run(a: ModelsArgs) -> i32 {
    quiet_pipe();
    let r = (|| -> Res<()> {
        let ctx = Ctx::load(a.packs_dir.clone())?;
        match a.cmd.clone().unwrap_or(ModelsCmd::List { json: false }) {
            ModelsCmd::List { json } => list(&ctx, json),
            ModelsCmd::Info { name, json } => info(&ctx, &name, json),
            ModelsCmd::Pull {
                name,
                tp,
                drafter,
                dry_run,
                yes,
            } => pull(&ctx, name.as_deref(), tp, drafter.as_deref(), dry_run, yes),
            ModelsCmd::Rm {
                name,
                tp,
                dry_run,
                yes,
            } => rm(&ctx, &name, tp, dry_run, yes),
        }
    })();
    report(r)
}

/// [`report`] for a command that has its own exit code.
pub fn report_code(r: Res<i32>) -> i32 {
    match r {
        Ok(c) => c,
        Err(e) => {
            eprint!("{}", e.render(Style::stderr().0));
            e.code
        }
    }
}

/// Print an error (if any) and turn it into an exit code.
pub fn report(r: Res<()>) -> i32 {
    match r {
        Ok(()) => 0,
        Err(e) => {
            eprint!("{}", e.render(Style::stderr().0));
            e.code
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpus(n: usize) -> Gpus {
        Gpus {
            count: n,
            gfx: Some("gfx1201".into()),
            names: vec!["R9700".into(); n],
        }
    }

    #[test]
    fn tp_follows_the_gpus() {
        let r = Registry::load(None).unwrap();
        let e = r.resolve("w4a8").unwrap();
        assert_eq!(choose_tp(e, None, &gpus(2), "x").unwrap().0, 2);
        assert_eq!(choose_tp(e, None, &gpus(4), "x").unwrap().0, 2);
        assert_eq!(choose_tp(e, None, &gpus(1), "x").unwrap().0, 1);
        // no GPU visible (a download box): the entry's default
        assert_eq!(choose_tp(e, None, &gpus(0), "x").unwrap().0, 2);
        assert_eq!(choose_tp(e, Some(1), &gpus(2), "x").unwrap().0, 1);
        let err = choose_tp(e, Some(2), &gpus(1), "saber serve --model")
            .unwrap_err()
            .render(false);
        assert!(
            err.contains("TP=2 needs 2 GPUs") && err.contains("saber serve --model w4a8 --tp 1"),
            "{err}"
        );
        let err = choose_tp(e, Some(3), &gpus(4), "saber models pull")
            .unwrap_err()
            .render(false);
        assert!(
            err.contains("no TP=3 pack")
                && err.contains("fix: saber models pull w4a8 --tp 2")
                && err.contains("fix: saber models pull w4a8 --tp 1"),
            "{err}"
        );
    }

    #[test]
    fn a_two_card_only_model_on_one_card_says_so() {
        let text = r#"
[[model]]
name = "big"
kind = "target"
repo = "o/big"
served_id = "Big"
files = ["t.json", "c.jinja"]
tokenizer = "t.json"
chat_template = "c.jinja"
[model.tp.2]
path = "tp2"
bytes = 1
max_ctx = 1
max_seqs = 1
decode_batches = "1"
decode_widths = "1"
"#;
        let r = Registry::from_texts(registry::SHIPPED, Some((text, Path::new("/u")))).unwrap();
        let e = r.resolve("big").unwrap();
        let err = choose_tp(e, None, &gpus(1), "x").unwrap_err().render(false);
        assert!(err.contains("needs 2 GPUs"), "{err}");
    }

    #[test]
    fn drafter_override_and_none() {
        let r = Registry::load(None).unwrap();
        let e = r.resolve("w4a8").unwrap();
        assert_eq!(
            drafter_of(&r, e, None).unwrap().unwrap().name,
            "Qwen3.8-27B-DFlash2-Drafter-W4A8-SABER"
        );
        assert_eq!(
            drafter_of(&r, e, Some("drafter-fp8"))
                .unwrap()
                .unwrap()
                .name,
            "Qwen3.8-27B-DFlash2-Drafter-FP8-SABER"
        );
        assert!(drafter_of(&r, e, Some("none")).unwrap().is_none());
        let err = drafter_of(&r, e, Some("mxfp4")).unwrap_err().render(false);
        assert!(err.contains("not a drafter"), "{err}");
    }

    #[test]
    fn short_handles() {
        let r = Registry::load(None).unwrap();
        assert_eq!(short(r.resolve("default").unwrap()), "nari3");
        assert_eq!(short(r.resolve("macaq-w4a8").unwrap()), "w4a8");
        assert_eq!(short(r.resolve("macaq").unwrap()), "mxfp4");
    }

    #[test]
    fn numbers_group() {
        assert_eq!(group(262_144), "262,144");
        assert_eq!(group(512), "512");
    }
}
