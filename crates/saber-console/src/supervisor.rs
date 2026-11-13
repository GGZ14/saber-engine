//! The engine process supervisor: spawn / stop / restart the engine from the config, capture its
//! output into the log ring, and track its state.
//!
//! States: `stopped` -> `loading` (process alive, `/v1/models` not answering yet, the readiness
//! probe the launchers use) -> `ready`; a stop goes through `stopping` (and a restart through
//! `draining`, which waits for running + waiting requests to reach 0 first). An exit the console
//! did not ask for is `crashed`, or `fatal-86` when the code is 86 (a rank failed; the
//! log's last `saber: FATAL rank R: <reason>` line is the message).

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::watch;

use crate::compose::{compose, ComposeCtx, EngineCommand};
use crate::config::{Config, Preset, Runtime};
use crate::guard;
use crate::logring::LogRing;

/// Exit code of a failed rank.
pub const FATAL_RANK_EXIT: i32 = 86;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EngineState {
    Stopped,
    Loading,
    Ready,
    Draining,
    Stopping,
    Crashed,
    #[serde(rename = "fatal-86")]
    Fatal86,
}

impl EngineState {
    #[must_use]
    pub fn alive(self) -> bool {
        matches!(
            self,
            EngineState::Loading
                | EngineState::Ready
                | EngineState::Draining
                | EngineState::Stopping
        )
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Status {
    pub state: EngineState,
    pub preset: Preset,
    pub runtime: Runtime,
    pub port: u16,
    pub pid: Option<u32>,
    pub container: Option<String>,
    /// Unix ms the process started / reached ready.
    pub started_ms: Option<u64>,
    pub ready_ms: Option<u64>,
    pub exit_code: Option<i32>,
    /// The last refusal, fatal line or crash note.
    pub message: Option<String>,
    /// The served model id (`/v1/models`).
    pub model: Option<String>,
    /// The engine's argv as the engine sees it.
    pub command: Vec<String>,
    /// Bumped on every spawn.
    pub generation: u64,
    /// The console's own clock, for the client's uptime display.
    pub now_ms: u64,
}

pub struct Supervisor {
    pub config: Mutex<Config>,
    pub config_path: PathBuf,
    home: PathBuf,
    exe: PathBuf,
    status: Mutex<Status>,
    stop_requested: Mutex<bool>,
    pub logs: Arc<LogRing>,
    port_tx: watch::Sender<Option<u16>>,
    exit_tx: watch::Sender<u64>,
    op: tokio::sync::Mutex<()>,
}

impl Supervisor {
    #[must_use]
    pub fn new(
        config: Config,
        config_path: PathBuf,
        home: PathBuf,
        exe: PathBuf,
        logs: Arc<LogRing>,
    ) -> (Arc<Supervisor>, watch::Receiver<Option<u16>>) {
        let (port_tx, port_rx) = watch::channel(None);
        let (exit_tx, _) = watch::channel(0);
        let status = Status {
            state: EngineState::Stopped,
            preset: config.engine.preset,
            runtime: config.engine.runtime,
            port: config.engine.port,
            pid: None,
            container: None,
            started_ms: None,
            ready_ms: None,
            exit_code: None,
            message: None,
            model: None,
            command: Vec::new(),
            generation: 0,
            now_ms: 0,
        };
        let s = Supervisor {
            config: Mutex::new(config),
            config_path,
            home,
            exe,
            status: Mutex::new(status),
            stop_requested: Mutex::new(false),
            logs,
            port_tx,
            exit_tx,
            op: tokio::sync::Mutex::new(()),
        };
        (Arc::new(s), port_rx)
    }

    fn st(&self) -> std::sync::MutexGuard<'_, Status> {
        self.status.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[must_use]
    pub fn status(&self) -> Status {
        let mut s = self.st().clone();
        s.now_ms = crate::now_ms();
        if !s.state.alive() {
            // show what the next start would run
            if let Ok(c) = self.preview() {
                s.command = c.engine;
            }
            let cfg = self.config.lock().unwrap_or_else(|e| e.into_inner());
            s.preset = cfg.engine.preset;
            s.runtime = cfg.engine.runtime;
            s.port = cfg.engine.port;
        }
        s
    }

    /// The Bearer key the console sends to the engine ([`Config::engine_bearer`]).
    #[must_use]
    pub fn engine_bearer(&self) -> Option<String> {
        self.config
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .engine_bearer(&self.home)
    }

    /// The engine's port while its process is alive (the proxy's upstream).
    #[must_use]
    pub fn live_port(&self) -> Option<u16> {
        let s = self.st();
        s.state.alive().then_some(s.port)
    }

    /// A copy of the current config.
    #[must_use]
    pub fn config_now(&self) -> Config {
        self.config
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Treat an engine the console did not start as running and ready on `port` (the proxy and
    /// poller tests run against an in-process mock upstream). Stop does not reach it.
    #[doc(hidden)]
    pub fn adopt_for_test(&self, port: u16) {
        {
            let mut s = self.st();
            s.state = EngineState::Ready;
            s.port = port;
            s.ready_ms = Some(crate::now_ms());
            s.generation += 1;
        }
        let _ = self.port_tx.send(Some(port));
    }

    /// `~` for the config's paths.
    #[must_use]
    pub fn home(&self) -> &std::path::Path {
        &self.home
    }

    fn ctx(&self, cfg: &Config) -> ComposeCtx {
        let paths = cfg.engine.paths.resolved(&self.home);
        let p2p =
            if cfg.engine.preset == Preset::Flashnext && cfg.engine.runtime == Runtime::Container {
                p2p_mounts(&paths.repo)
            } else {
                Vec::new()
            };
        ComposeCtx {
            home: self.home.clone(),
            exe: self.exe.clone(),
            fn3_prompts_present: paths.shared.join("ws-fn3/ref/run1").is_dir(),
            fn_legacy_pack_present: paths.nvme.join(crate::compose::FN_LEGACY_PACK).is_dir(),
            p2p_mounts: p2p,
        }
    }

    /// The command the next start would spawn.
    pub fn preview(&self) -> anyhow::Result<EngineCommand> {
        let cfg = self
            .config
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        Ok(compose(&cfg.engine, &self.ctx(&cfg)))
    }

    pub async fn start(self: &Arc<Self>) -> anyhow::Result<()> {
        let _g = self.op.lock().await;
        self.start_inner().await
    }

    pub async fn stop(self: &Arc<Self>) -> anyhow::Result<()> {
        let _g = self.op.lock().await;
        self.stop_inner().await
    }

    /// Drain (wait for 0 running + waiting, bounded), stop, start.
    pub async fn restart(self: &Arc<Self>) -> anyhow::Result<()> {
        let _g = self.op.lock().await;
        if self.st().state == EngineState::Ready {
            self.drain().await;
        }
        self.stop_inner().await?;
        self.start_inner().await
    }

    /// Hold the supervisor's operation lock (the apply flow runs drain / stop / start under one
    /// hold so no other start or stop interleaves).
    pub(crate) async fn lock_op(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.op.lock().await
    }

    /// Replace the in-memory config (the caller saves the file).
    pub fn set_config(&self, c: Config) {
        *self.config.lock().unwrap_or_else(|e| e.into_inner()) = c;
    }

    /// The command `cfg` would spawn (the same composition as [`Supervisor::preview`]).
    #[must_use]
    pub fn compose_for(&self, cfg: &Config) -> EngineCommand {
        compose(&cfg.engine, &self.ctx(cfg))
    }

    /// Wait for spawn `generation` to reach ready, exit, or time out.
    pub async fn wait_loaded(&self, generation: u64, timeout: Duration) -> LoadOutcome {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            {
                let s = self.st();
                if s.generation != generation {
                    return LoadOutcome::Superseded;
                }
                match s.state {
                    EngineState::Ready => return LoadOutcome::Ready,
                    EngineState::Loading => {}
                    st if !st.alive() => {
                        return LoadOutcome::Exited {
                            code: s.exit_code,
                            message: s.message.clone(),
                        }
                    }
                    _ => {}
                }
            }
            if tokio::time::Instant::now() >= deadline {
                return LoadOutcome::Timeout;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    pub(crate) async fn drain(self: &Arc<Self>) {
        let (port, timeout) = {
            let c = self.config.lock().unwrap_or_else(|e| e.into_inner());
            (self.st().port.max(c.engine.port), c.engine.drain_timeout_s)
        };
        self.st().state = EngineState::Draining;
        self.note("draining: waiting for running + waiting requests to reach 0");
        let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout);
        let bearer = self.engine_bearer();
        while tokio::time::Instant::now() < deadline {
            if let Ok((200, body)) =
                crate::httpc::get_auth(port, "/metrics", Duration::from_secs(2), bearer.as_deref())
                    .await
            {
                let s = crate::scrape::parse(&body);
                let busy = s.get("num_requests_running").unwrap_or(0.0)
                    + s.get("num_requests_waiting").unwrap_or(0.0);
                if busy == 0.0 {
                    self.note("drained");
                    return;
                }
            } else {
                return;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        self.note(&format!(
            "drain timed out after {timeout} s; stopping anyway"
        ));
    }

    fn note(&self, msg: &str) {
        self.logs.push("console", format!("saber ui: {msg}"));
    }

    fn refuse(&self, msg: String) -> anyhow::Error {
        self.note(&format!("refused to start: {msg}"));
        self.st().message = Some(msg.clone());
        anyhow::anyhow!(msg)
    }

    pub(crate) async fn start_inner(self: &Arc<Self>) -> anyhow::Result<()> {
        if self.st().state.alive() {
            anyhow::bail!("the engine is already running");
        }
        let cfg = self
            .config
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        cfg.validate()?;
        let e = &cfg.engine;
        // The guards are blocking probes (systemctl, podman ps, flock): off the async workers.
        let (ctr, lock_dir, preset, port) = (
            e.container.runtime.clone(),
            self.home.join("saber/.cache"),
            e.preset,
            e.port,
        );
        let name = if e.runtime == Runtime::Container {
            Some(if e.container.name.is_empty() {
                crate::compose::default_container_name(preset).to_string()
            } else {
                e.container.name.clone()
            })
        } else {
            None
        };
        let guard_res = tokio::task::spawn_blocking(move || -> Result<(), guard::Refusal> {
            if preset.uses_gpus() {
                guard::check_cards(&ctr, &lock_dir)?;
            }
            if let Some(n) = &name {
                guard::check_name_free(&ctr, n)?;
            }
            guard::check_port(port)
        })
        .await?;
        if let Err(r) = guard_res {
            return Err(self.refuse(r.0));
        }

        let ctx = self.ctx(&cfg);
        let cmd = compose(e, &ctx);
        let paths = e.paths.resolved(&self.home);
        self.note(&format!("starting: {}", cmd.engine_line()));
        let mut c = Command::new(&cmd.spawn[0]);
        c.args(&cmd.spawn[1..])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(false)
            // its own process group: a Ctrl-C at the console's terminal reaches the console
            // only, and the console stops the engine on its own terms
            .process_group(0);
        if e.runtime == Runtime::Direct && paths.repo.is_dir() {
            c.current_dir(&paths.repo);
        }
        let mut child = match c.spawn() {
            Ok(ch) => ch,
            Err(err) => return Err(self.refuse(format!("spawning {}: {err}", cmd.spawn[0]))),
        };
        *self
            .stop_requested
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = false;
        let generation = {
            let mut s = self.st();
            s.generation += 1;
            s.state = EngineState::Loading;
            s.preset = e.preset;
            s.runtime = e.runtime;
            s.port = e.port;
            s.pid = child.id();
            s.container = cmd.container_name.clone();
            s.started_ms = Some(crate::now_ms());
            s.ready_ms = None;
            s.exit_code = None;
            s.message = None;
            s.model = None;
            s.command = cmd.engine.clone();
            s.generation
        };
        let _ = self.port_tx.send(Some(e.port));

        for (stream, name) in [
            (child.stdout.take().map(Either::Out), "out"),
            (child.stderr.take().map(Either::Err), "err"),
        ] {
            let Some(stream) = stream else { continue };
            let logs = self.logs.clone();
            tokio::spawn(async move {
                match stream {
                    Either::Out(s) => pump(BufReader::new(s), &logs, name).await,
                    Either::Err(s) => pump(BufReader::new(s), &logs, name).await,
                }
            });
        }

        // the waiter
        let me = self.clone();
        tokio::spawn(async move {
            let res = child.wait().await;
            let code = res.as_ref().ok().and_then(std::process::ExitStatus::code);
            // let the pipes drain into the ring before reading the last line
            tokio::time::sleep(Duration::from_millis(200)).await;
            let requested = *me.stop_requested.lock().unwrap_or_else(|e| e.into_inner());
            let fatal_line = me.logs.last_matching(|t| t.contains("FATAL"));
            {
                let mut s = me.st();
                if s.generation != generation {
                    return;
                }
                s.pid = None;
                s.exit_code = code;
                s.state = match (requested, code) {
                    (_, Some(FATAL_RANK_EXIT)) => EngineState::Fatal86,
                    (true, _) | (false, Some(0)) => EngineState::Stopped,
                    _ => EngineState::Crashed,
                };
                s.message = match s.state {
                    EngineState::Fatal86 => {
                        Some(fatal_line.unwrap_or_else(|| "a rank failed (exit 86)".into()))
                    }
                    EngineState::Crashed => Some(match code {
                        Some(c) => format!("the engine exited with code {c}"),
                        None => "the engine was killed by a signal".into(),
                    }),
                    _ => None,
                };
            }
            me.note(&format!(
                "engine exited ({})",
                code.map_or("signal".to_string(), |c| format!("code {c}"))
            ));
            let _ = me.port_tx.send(None);
            me.exit_tx.send_modify(|g| *g = generation);
        });

        // readiness: the launchers' probe (`/v1/models` answers)
        let me = self.clone();
        let port = e.port;
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;
                {
                    let s = me.st();
                    if s.generation != generation || s.state != EngineState::Loading {
                        return;
                    }
                }
                let bearer = me.engine_bearer();
                if let Ok((200, body)) = crate::httpc::get_auth(
                    port,
                    "/v1/models",
                    Duration::from_secs(2),
                    bearer.as_deref(),
                )
                .await
                {
                    let model = serde_json::from_str::<serde_json::Value>(&body)
                        .ok()
                        .and_then(|v| v["data"][0]["id"].as_str().map(str::to_string));
                    let mut s = me.st();
                    if s.generation == generation && s.state == EngineState::Loading {
                        s.state = EngineState::Ready;
                        s.ready_ms = Some(crate::now_ms());
                        s.model = model;
                        drop(s);
                        me.note("engine is ready");
                    }
                    return;
                }
            }
        });
        Ok(())
    }

    pub(crate) async fn stop_inner(self: &Arc<Self>) -> anyhow::Result<()> {
        let (pid, container, generation) = {
            let s = self.st();
            if !s.state.alive() {
                return Ok(());
            }
            (s.pid, s.container.clone(), s.generation)
        };
        let (ctr, timeout) = {
            let c = self.config.lock().unwrap_or_else(|e| e.into_inner());
            (c.engine.container.runtime.clone(), c.engine.stop_timeout_s)
        };
        *self
            .stop_requested
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = true;
        self.st().state = EngineState::Stopping;
        self.note("stopping (SIGTERM)");
        let mut exit_rx = self.exit_tx.subscribe();
        // a container is stopped BY NAME (killing the client leaves the container on the cards)
        match (&container, pid) {
            (Some(name), _) => {
                let _ = Command::new(&ctr)
                    .args(["stop", "-t", &timeout.to_string(), name])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .await;
            }
            (None, Some(pid)) => signal(pid, "TERM").await,
            (None, None) => {}
        }
        let wait = |secs: u64, rx: &mut watch::Receiver<u64>| {
            let mut rx = rx.clone();
            async move {
                tokio::time::timeout(Duration::from_secs(secs), rx.wait_for(|g| *g == generation))
                    .await
                    .is_ok()
            }
        };
        if wait(timeout + 5, &mut exit_rx).await {
            return Ok(());
        }
        self.note("the engine did not exit after SIGTERM; killing it");
        match (&container, pid) {
            (Some(name), _) => {
                let _ = Command::new(&ctr).args(["kill", name]).status().await;
            }
            (None, Some(pid)) => signal(pid, "KILL").await,
            (None, None) => {}
        }
        if wait(15, &mut exit_rx).await {
            Ok(())
        } else {
            anyhow::bail!("the engine did not exit after SIGKILL")
        }
    }
}

/// How a spawn's load ended ([`Supervisor::wait_loaded`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadOutcome {
    Ready,
    /// The process exited before ready (`code` = `None`: a signal).
    Exited {
        code: Option<i32>,
        message: Option<String>,
    },
    Timeout,
    /// Another spawn replaced it.
    Superseded,
}

enum Either {
    Out(tokio::process::ChildStdout),
    Err(tokio::process::ChildStderr),
}

async fn pump<R: tokio::io::AsyncRead + Unpin>(
    r: BufReader<R>,
    logs: &LogRing,
    stream: &'static str,
) {
    let mut lines = r.lines();
    while let Ok(Some(l)) = lines.next_line().await {
        logs.push(stream, l);
    }
}

async fn signal(pid: u32, sig: &str) {
    let _ = Command::new("kill")
        .arg(format!("-{sig}"))
        .arg(pid.to_string())
        .status()
        .await;
}

/// The launcher's P2PDMA pool mounts, from the launcher's own helper (one source of truth).
fn p2p_mounts(repo: &std::path::Path) -> Vec<String> {
    let helper = repo.join("tools/capture/p2p_mounts.sh");
    if !helper.exists() {
        // a repo without the helper would silently lose the host's P2PDMA pool
        // (820+ expert slots a rank): say so, once
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let pools = std::fs::read_dir("/sys/bus/pci/devices")
            .map(|d| d.flatten().any(|e| e.path().join("p2pmem/size").exists()))
            .unwrap_or(false);
        if pools && !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "saber ui: WARNING: {} is missing, so the engine gets NO P2PDMA pool mounts \
                 (engine.paths.repo must be a checkout with tools/capture/)",
                helper.display()
            );
        }
        return Vec::new();
    }
    let out = std::process::Command::new("bash")
        .arg("-c")
        .arg(r#". "$1"; P=(); p2p_mounts P; [ ${#P[@]} -gt 0 ] && printf '%s\0' "${P[@]}"; true"#)
        .arg("p2p")
        .arg(&helper)
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .split('\0')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// A fake engine binary: the mock preset's argv, run by a script that exits `code`.
    async fn run_fake(body: &str, port: u16) -> (Arc<Supervisor>, Status) {
        let dir = saber_test_scratch::ScratchDir::new("console-supervisor");
        let bin = dir.path().join("fake-saber");
        std::fs::write(&bin, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut cfg = Config::mock(port);
        cfg.engine.paths.binary = bin.clone();
        cfg.engine.paths.repo = dir.path().to_path_buf();
        let logs = Arc::new(LogRing::new(100));
        let (sup, _rx) = Supervisor::new(
            cfg,
            dir.path().join("c.toml"),
            dir.path().to_path_buf(),
            bin,
            logs,
        );
        let mut exit = sup.exit_tx.subscribe();
        sup.start().await.unwrap();
        tokio::time::timeout(Duration::from_secs(10), exit.wait_for(|g| *g == 1))
            .await
            .unwrap()
            .unwrap();
        let st = sup.status();
        drop(dir);
        (sup, st)
    }

    #[tokio::test]
    async fn exit_86_is_fatal_with_the_rank_line() {
        let (_s, st) = run_fake(
            "echo 'saber: FATAL rank 1: injected after 3 steps' >&2; exit 86",
            18_731,
        )
        .await;
        assert_eq!(st.state, EngineState::Fatal86);
        assert_eq!(st.exit_code, Some(86));
        assert_eq!(
            st.message.as_deref(),
            Some("saber: FATAL rank 1: injected after 3 steps")
        );
    }

    #[tokio::test]
    async fn other_exits_are_crashes_and_zero_is_stopped() {
        let (_s, st) = run_fake("exit 3", 18_732).await;
        assert_eq!(st.state, EngineState::Crashed);
        assert_eq!(st.exit_code, Some(3));
        let (_s, st) = run_fake("echo bye", 18_733).await;
        assert_eq!(st.state, EngineState::Stopped);
    }

    #[tokio::test]
    async fn stop_sends_sigterm_and_a_busy_port_refuses() {
        let l = std::net::TcpListener::bind("127.0.0.1:18734").unwrap();
        let dir = saber_test_scratch::ScratchDir::new("console-supervisor-stop");
        let bin = dir.path().join("fake-saber");
        std::fs::write(
            &bin,
            "#!/bin/sh\ntrap 'echo got-term; exit 0' TERM\nwhile :; do sleep 0.1; done\n",
        )
        .unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut cfg = Config::mock(18_734);
        cfg.engine.paths.binary = bin.clone();
        let logs = Arc::new(LogRing::new(100));
        let (sup, _rx) = Supervisor::new(
            cfg.clone(),
            dir.path().join("c.toml"),
            dir.path().into(),
            bin.clone(),
            logs.clone(),
        );
        let err = sup.start().await.unwrap_err().to_string();
        assert!(err.contains("port 18734 is already in use"), "{err}");
        assert_eq!(sup.status().state, EngineState::Stopped);
        drop(l);
        cfg.engine.port = 18_735;
        *sup.config.lock().unwrap() = cfg;
        sup.start().await.unwrap();
        assert_eq!(sup.status().state, EngineState::Loading);
        tokio::time::sleep(Duration::from_millis(300)).await;
        sup.stop().await.unwrap();
        let st = sup.status();
        assert_eq!(st.state, EngineState::Stopped);
        assert_eq!(st.exit_code, Some(0));
        assert!(logs.last_matching(|t| t == "got-term").is_some());
    }
}
