//! `saber-shadow mirror | replay | report` — see `docs/SHADOW.md` and the crate docs.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use clap::{Parser, Subcommand};
use saber_shadow::mirror::{self, Mirror, MirrorConfig};
use saber_shadow::record::{self, Exchange, LogEntry, ReplayEntry};
use saber_shadow::replay::{self, Pace, ReplayConfig};
use saber_shadow::report::{self, ReaderOptions};

#[derive(Parser)]
#[command(
    name = "saber-shadow",
    about = "The shadow week's mirror, replay and daily reader"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// A transparent recording reverse proxy. NEVER put in front of prod without the user's
    /// go-ahead (docs/SHADOW.md §1 [user]).
    Mirror {
        /// Address to listen on.
        #[arg(long)]
        listen: String,
        /// The upstream's base URL.
        #[arg(long)]
        upstream: String,
        /// Where the `<UTC date>.jsonl` day files go.
        #[arg(long)]
        log_dir: PathBuf,
        /// Poll the upstream's /metrics this often, seconds (0: never).
        #[arg(long, default_value_t = 30)]
        metrics_every_s: u64,
    },
    /// Send a day's recorded requests to a Saber endpoint and store the answers.
    Replay {
        /// The mirror's day file.
        #[arg(long)]
        log: PathBuf,
        /// Saber's base URL (`http://127.0.0.1:8090`).
        #[arg(long)]
        target: String,
        /// Where the answers go (JSONL).
        #[arg(long)]
        out: PathBuf,
        /// `recorded` (arrival offsets / --speed, concurrency uncapped) or `sequential`.
        #[arg(long, default_value = "recorded")]
        pace: String,
        /// Compression factor for `--pace recorded`.
        #[arg(long, default_value_t = 1.0)]
        speed: f64,
        /// Replay only the first N exchanges.
        #[arg(long)]
        limit: Option<usize>,
        /// Where the cards' `power1_cap` is read (empty: skip).
        #[arg(long, default_value = "/sys/class/drm")]
        power_cap_root: String,
        /// Per-request timeout, seconds.
        #[arg(long, default_value_t = 900)]
        timeout_s: u64,
    },
    /// The daily reader: the comparison and the six stop conditions. Exits 1 when one fired.
    Report {
        /// The mirror's day file.
        #[arg(long)]
        log: PathBuf,
        /// The replay's answers.
        #[arg(long)]
        answers: PathBuf,
        /// Saber's stdout for the window.
        #[arg(long)]
        saber_log: Option<PathBuf>,
        /// The power cap every card must read, watts.
        #[arg(long, default_value_t = 210.0)]
        cap_w: f64,
        /// An exchange id whose refusal is already on record as prod's own (repeatable).
        #[arg(long)]
        allow_refusal: Vec<u64>,
        /// Print the report as JSON.
        #[arg(long)]
        json: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    match cli.cmd {
        Cmd::Mirror {
            listen,
            upstream,
            log_dir,
            metrics_every_s,
        } => rt.block_on(async move {
            let m = Mirror::new(MirrorConfig {
                upstream: upstream.trim_end_matches('/').to_owned(),
                log_dir,
                metrics_every: (metrics_every_s > 0).then(|| Duration::from_secs(metrics_every_s)),
                fault_drop_chunk: None,
            })?;
            let listener = tokio::net::TcpListener::bind(&listen)
                .await
                .with_context(|| format!("binding {listen}"))?;
            println!(
                "mirror         http://{} -> {} (health: /shadow/health)",
                listener.local_addr()?,
                upstream
            );
            mirror::serve(m, listener, async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await
        }),
        Cmd::Replay {
            log,
            target,
            out,
            pace,
            speed,
            limit,
            power_cap_root,
            timeout_s,
        } => rt.block_on(async move {
            let mut xs: Vec<Exchange> = record::read_log(&log)?
                .into_iter()
                .filter_map(|e| match e {
                    LogEntry::Exchange(x) => Some(*x),
                    _ => None,
                })
                .collect();
            if let Some(n) = limit {
                xs.truncate(n);
            }
            let cfg = ReplayConfig {
                target: target.trim_end_matches('/').to_owned(),
                pace: match pace.as_str() {
                    "recorded" => Pace::Recorded { speed },
                    "sequential" => Pace::Sequential,
                    other => anyhow::bail!("--pace {other}: recorded or sequential"),
                },
                power_cap_root: (!power_cap_root.is_empty()).then(|| power_cap_root.into()),
                timeout: Duration::from_secs(timeout_s),
            };
            let (meta, answers) = replay::replay(&xs, &cfg).await?;
            if out.exists() {
                anyhow::bail!("{} exists; a replay file is written once", out.display());
            }
            record::append_line(&out, &ReplayEntry::ReplayMeta(meta))?;
            for a in answers {
                record::append_line(&out, &ReplayEntry::Answer(Box::new(a)))?;
            }
            println!(
                "replayed       {} exchange(s) -> {}",
                xs.len(),
                out.display()
            );
            Ok(())
        }),
        Cmd::Report {
            log,
            answers,
            saber_log,
            cap_w,
            allow_refusal,
            json,
        } => {
            let entries = record::read_log(&log)?;
            let (meta, answers) = record::read_replay(&answers)?;
            let saber_log = match saber_log {
                Some(p) => Some(
                    std::fs::read_to_string(&p)
                        .with_context(|| format!("reading {}", p.display()))?,
                ),
                None => None,
            };
            let r = report::compare(
                &entries,
                &meta,
                &answers,
                &ReaderOptions {
                    saber_log,
                    cap_w,
                    allowed_refusals: allow_refusal,
                },
            );
            if json {
                println!("{}", serde_json::to_string_pretty(&r)?);
            } else {
                print!("{}", r.render());
            }
            if r.fired() {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}
