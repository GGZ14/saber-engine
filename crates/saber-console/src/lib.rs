//! `saber ui`: the browser console. An always-on process that serves the dashboard and owns the
//! engine as a child process (it survives engine restarts and works before a model is installed).
//!
//! - [`config`]: the typed, versioned store (`~/.config/saber/config.toml`)
//! - [`schema`]: the flag schema, read from the CLI's clap structs
//! - [`compose`]: the engine command line from the config (the launchers' presets)
//! - [`guard`]: one server on the cards (the launchers' refusals)
//! - [`supervisor`]: spawn / stop / drain-and-restart, the log ring, the engine state
//! - [`scrape`]: the engine's `/metrics` -> an in-memory ring (24 h at 5 s)
//! - [`http`]: the `/api/*` JSON + SSE plane and the embedded SPA (`ui/`)
//! - [`proxy`]: the `/v1/*` reverse proxy to the engine (playground, usage capture)
//! - [`usage`]: the persistent usage history (SQLite, rollups, the engine poller)
//! - [`hw`]: GPU / host sampling from sysfs and `/proc`
//! - the settings plane: [`validate`] (per-field + cross-flag rules), [`estimate`]
//!   (the memory budget), [`profiles`], [`history`] and [`apply`] (diff, drain-and-restart,
//!   rollback / auto-rollback, the model manager)
//! - first run: [`catalog`] (`tools/models.json`, fit rules), [`system`] (the hardware
//!   and host check), [`downloads`] (resumable, verified pack downloads), [`setup`] (the
//!   wizard's state machine) and [`wizard`] (their routes)
//! - hardening: [`auth`] (hashed credentials, constant-time checks, the failed-auth
//!   limiter) and [`audit`] (who applied / started / stopped what, when)
//!
//! No environment variable is read here: every path arrives as an argument.

pub mod apply;
pub mod audit;
pub mod auth;
pub mod catalog;
pub mod compose;
pub mod config;
pub mod downloads;
pub mod estimate;
pub mod guard;
pub mod history;
pub mod http;
pub mod httpc;
pub mod hw;
pub mod logring;
pub mod profiles;
pub mod proxy;
pub mod schema;
pub mod scrape;
pub mod setup;
pub mod supervisor;
pub mod system;
pub mod usage;
pub mod validate;
pub mod wizard;

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;

/// Unix milliseconds.
#[must_use]
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// Everything `saber ui` resolves before the console starts.
pub struct ConsoleArgs {
    /// Address to bind (`None` = the config's `console.host`, else `127.0.0.1`).
    pub host: Option<String>,
    /// `None` = the config's `console.port`, else 8090.
    pub port: Option<u16>,
    /// Required when `host` is not a loopback address (or the config's stored hash).
    pub admin_token: Option<String>,
    /// The resolved credential ([`run`] fills it from the token / the config).
    pub admin: Option<auth::Admin>,
    pub config_path: PathBuf,
    /// The config used when `config_path` does not exist yet (not written until saved).
    pub default_config: config::Config,
    /// Replace the stored engine preset for this run (`--engine mock`); not saved.
    pub override_config: Option<config::Config>,
    /// `~` in the config's paths.
    pub home: PathBuf,
    /// The running `saber` binary.
    pub exe: PathBuf,
    /// The flag schema (built by the CLI from its own clap structs).
    pub schema: serde_json::Value,
}

/// True when `host` binds the loopback only.
#[must_use]
pub fn is_loopback(host: &str) -> bool {
    host == "localhost" || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// Refuse a non-loopback bind without an admin token.
pub fn check_bind(host: &str, admin: &auth::Admin) -> anyhow::Result<()> {
    if !is_loopback(host) && !admin.is_set() {
        anyhow::bail!(
            "--host {host} is not a loopback address: the console controls the engine, so a \
             non-loopback bind needs an admin token (--admin-token <secret>, or \
             `saber ui --gen-admin-token` once) (refusing to start)"
        );
    }
    Ok(())
}

/// The console's bind: the command line, else the config, else the LAN (0.0.0.0:8090) when an
/// admin token guards it, else 127.0.0.1:8090 (the console controls the engine, so an open one
/// never leaves the loopback). The wizard's Network step writes the config's. The admin
/// credential: `--admin-token`, else the config's stored hash, else none.
#[must_use]
pub fn resolve_bind(
    args: &ConsoleArgs,
    stored: Option<&config::Config>,
) -> (String, u16, auth::Admin) {
    let c = stored.map(|c| &c.console);
    let port = args
        .port
        .or_else(|| c.map(|c| c.port).filter(|&p| p != 0))
        .unwrap_or(8090);
    let admin = match args.admin_token.clone().filter(|t| !t.is_empty()) {
        Some(t) => auth::Admin::Flag(t),
        None => stored.map_or(auth::Admin::None, config::Config::admin),
    };
    let host = args
        .host
        .clone()
        .or_else(|| c.map(|c| c.host.clone()).filter(|h| !h.is_empty()))
        .unwrap_or_else(|| if admin.is_set() { "0.0.0.0" } else { "127.0.0.1" }.into());
    (host, port, admin)
}

/// `saber ui --gen-admin-token`: a new token's hash into the config (created when absent);
/// returns the token (the caller prints it once).
pub fn gen_admin_token(config_path: &std::path::Path) -> anyhow::Result<String> {
    let mut cfg = config::Config::load(config_path)?.unwrap_or_default();
    let token = setup::random_key("sa-admin-")?;
    cfg.console.admin_token = token.clone();
    cfg.console.admin_token_hash.clear();
    cfg.save(config_path)?;
    Ok(token)
}

/// Run the console until Ctrl-C / SIGTERM; the engine it started is stopped on the way out.
pub fn run(mut args: ConsoleArgs) -> anyhow::Result<()> {
    let stored = config::Config::load(&args.config_path).ok().flatten();
    // A file that still holds a secret as text (written before the hashes) is
    // rewritten with the hashes before anything listens
    if let Some(s) = stored.as_ref().filter(|s| s.has_plain_secrets()) {
        s.save(&args.config_path)?;
        println!(
            "saber ui: {}: the admin token / proxy keys are now stored as salted hashes",
            args.config_path.display()
        );
    }
    let stored = config::Config::load(&args.config_path).ok().flatten();
    let (host, port, admin) = resolve_bind(&args, stored.as_ref());
    check_bind(&host, &admin)?;
    args.host = Some(host);
    args.port = Some(port);
    args.admin = Some(admin);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    rt.block_on(serve(args))
}

async fn serve(args: ConsoleArgs) -> anyhow::Result<()> {
    let host = args.host.clone().unwrap_or_else(|| "127.0.0.1".into());
    let port = args.port.unwrap_or(8090);
    let stored = config::Config::load(&args.config_path)?;
    let from_file = stored.is_some();
    // `--engine mock` replaces the stored ENGINE only: the stored proxy / usage / metrics
    // settings still apply (a mock run records into the configured usage database)
    let cfg = match (args.override_config.clone(), stored) {
        (Some(o), Some(s)) => config::Config {
            engine: o.engine,
            ..s
        },
        (Some(o), None) => o,
        (None, Some(s)) => s,
        (None, None) => args.default_config.clone(),
    };
    cfg.validate()?;

    let logs = Arc::new(logring::LogRing::new(10_000));
    let metrics = Arc::new(scrape::MetricsStore::new(
        cfg.metrics.capacity(),
        cfg.metrics.interval_s,
    ));
    let autostart = cfg.engine.autostart;
    let usage_cfg = cfg.usage.clone();
    let (sup, port_rx) = supervisor::Supervisor::new(
        cfg,
        args.config_path.clone(),
        args.home.clone(),
        args.exe.clone(),
        logs.clone(),
    );

    // The usage store (a failure to open it is loud, and the console runs without it)
    let usage = if usage_cfg.enabled {
        let path = expand_home(&usage_cfg.db_path, &args.home);
        match usage::UsageStore::open(&path, usage_cfg.utc_offset_minutes) {
            Ok(u) => {
                println!("saber ui: usage history in {}", path.display());
                Some(Arc::new(u))
            }
            Err(e) => {
                eprintln!("saber ui: usage history OFF: {e:#}");
                logs.push("console", format!("saber ui: usage history OFF: {e:#}"));
                None
            }
        }
    } else {
        None
    };
    if let Some(u) = &usage {
        tokio::spawn(usage::poll(u.clone(), sup.clone()));
        tokio::spawn(usage::retention(u.clone(), sup.clone()));
    }
    let bsup = sup.clone();
    tokio::spawn(scrape::run(
        metrics.clone(),
        scrape::ScrapeCtx {
            port: port_rx,
            bearer: Arc::new(move || bsup.engine_bearer()),
            hw_root: PathBuf::from("/"),
            usage: usage.clone(),
        },
    ));

    let loopback = is_loopback(&host);
    let console = Arc::new(apply::Console::new(
        sup.clone(),
        metrics.clone(),
        &args.schema,
        port,
    ));
    // The single-flight download worker
    tokio::spawn(console.downloads.clone().worker());
    println!(
        "saber ui: config history and profiles in {}",
        console.data_dir.display()
    );
    let state = http::AppState {
        sup: sup.clone(),
        metrics,
        console,
        schema: Arc::new(args.schema),
        auth: Arc::new(auth::Auth::new(args.admin.unwrap_or(auth::Admin::None))),
        loopback,
        proxy: Arc::new(proxy::Proxy::new(usage.clone())),
        usage: usage.clone(),
    };
    let app = http::router(state);
    let ip: IpAddr = if host == "localhost" {
        IpAddr::from([127, 0, 0, 1])
    } else {
        host.parse().with_context(|| format!("--host {host}"))?
    };
    let listener = tokio::net::TcpListener::bind(SocketAddr::new(ip, port))
        .await
        .with_context(|| format!("binding {host}:{port}"))?;
    let bound = listener.local_addr()?;
    println!(
        "saber ui: console on http://{bound}/ (config {}{})",
        args.config_path.display(),
        if from_file {
            ""
        } else {
            ", not saved yet: defaults"
        }
    );
    logs.push("console", format!("saber ui: listening on {bound}"));

    if autostart {
        let s = sup.clone();
        tokio::spawn(async move {
            if let Err(e) = s.start().await {
                eprintln!("saber ui: autostart: {e:#}");
            }
        });
    }

    let shutdown = async {
        let ctrl_c = tokio::signal::ctrl_c();
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("installing the SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => {}
            _ = term.recv() => {}
        }
    };
    // Not a graceful shutdown: open SSE streams never end, so the server is dropped instead.
    tokio::select! {
        r = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        ) => r?,
        () = shutdown => {}
    }
    println!("saber ui: shutting down; stopping the engine it started");
    sup.stop().await?;
    if let Some(u) = usage {
        // commit what the writer still holds
        let _ = tokio::task::spawn_blocking(move || u.flush()).await;
    }
    Ok(())
}

/// `~/x` against `home`.
#[must_use]
pub fn expand_home(p: &std::path::Path, home: &std::path::Path) -> PathBuf {
    match p.strip_prefix("~") {
        Ok(rest) => home.join(rest),
        Err(_) => p.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_loopback_needs_a_token() {
        let none = auth::Admin::None;
        let tok = auth::Admin::Flag("s3cret-s3cret-s3cret".into());
        assert!(check_bind("127.0.0.1", &none).is_ok());
        assert!(check_bind("::1", &none).is_ok());
        assert!(check_bind("localhost", &none).is_ok());
        assert!(check_bind("0.0.0.0", &none).is_err());
        assert!(check_bind("192.168.1.10", &tok).is_ok());
        let h = auth::Admin::Hash(auth::hash_secret("cfg-token-0123456").unwrap());
        assert!(
            check_bind("0.0.0.0", &h).is_ok(),
            "a stored hash is a token"
        );
    }

    #[test]
    fn bind_comes_from_the_flags_then_the_config() {
        let mut args = ConsoleArgs {
            host: None,
            port: None,
            admin_token: None,
            admin: None,
            config_path: PathBuf::new(),
            default_config: config::Config::default(),
            override_config: None,
            home: PathBuf::new(),
            exe: PathBuf::new(),
            schema: serde_json::Value::Null,
        };
        assert_eq!(
            resolve_bind(&args, None),
            ("127.0.0.1".into(), 8090, auth::Admin::None)
        );
        // no host anywhere: a token-guarded console listens on the LAN, an open one stays on the loopback
        let mut guarded = config::Config::default();
        guarded.console.admin_token = "0123456789abcdef".into();
        guarded.seal().unwrap();
        assert_eq!(resolve_bind(&args, Some(&guarded)).0, "0.0.0.0");
        args.admin_token = Some("from-the-flag".into());
        assert_eq!(resolve_bind(&args, None).0, "0.0.0.0");
        args.admin_token = None;
        let mut c = config::Config::default();
        c.console.host = "0.0.0.0".into();
        c.console.port = 9000;
        c.console.admin_token = "0123456789abcdef".into();
        c.seal().unwrap();
        let (h, p, a) = resolve_bind(&args, Some(&c));
        assert_eq!((h.as_str(), p), ("0.0.0.0", 9000));
        assert_eq!(a.source(), "config");
        assert!(a.check("0123456789abcdef"));
        args.host = Some("127.0.0.1".into());
        args.port = Some(8091);
        args.admin_token = Some("from-the-flag".into());
        assert_eq!(
            resolve_bind(&args, Some(&c)),
            (
                "127.0.0.1".into(),
                8091,
                auth::Admin::Flag("from-the-flag".into())
            )
        );
    }

    #[test]
    fn gen_admin_token_stores_only_the_hash() {
        let d = saber_test_scratch::ScratchDir::new("console-gentoken");
        let p = d.path().join("config.toml");
        let t = gen_admin_token(&p).unwrap();
        assert!(t.starts_with("sa-admin-"));
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(!text.contains(&t), "{text}");
        let c = config::Config::load(&p).unwrap().unwrap();
        assert!(c.admin().check(&t));
    }
}
