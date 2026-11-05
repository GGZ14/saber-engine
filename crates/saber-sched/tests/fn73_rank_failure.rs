//! FN73: **a failed rank exits the process** -- the teardown law, end to end on the host.
//!
//! Incident 2026-10-07 (FN55c hold p5_Li): rank 1 of a Flash-Next pair failed; rank 0's serve loop
//! stopped and its main thread then joined the HTTP thread, whose server ended only on a signal --
//! the process held both cards for ~40 minutes with the listener open.
//!
//! Each test re-runs this binary as a child (a second libtest filter `fn73form:<form>`, which matches no test: argv, not the
//! environment -- structural check 2) that stands up the serving plane
//! over the real scheduler in one engine form's process shape, with a FAKE rank 1 (a child process
//! that prints `saber: FATAL rank 1: ...` on its stderr and exits 86 after 1 s) watched by
//! `saber_serve::fatal::RankWatch`, and a client with one stream and one plain request in flight.
//! The parent asserts: exit code 86 within the bound, the final line `saber: FATAL rank 1: <the
//! child rank's reason>`, the in-flight requests ended with a 503 carrying the reason, a new
//! request refused (503 or a closed listener), and the port closed after the exit.
//!
//! * `27b`: the step loop runs the executor in-thread; the HTTP plane runs on the main thread's
//!   runtime (`saber serve --engine real`'s shape).
//! * `fn`: the step thread hands every step to an ENGINE thread (the main thread, a job channel,
//!   `FnExec`'s shape) and waits for the reply; the HTTP plane runs on a thread of its own that the
//!   main thread joins at the end (`saber engine --fn-serve`'s shape -- the incident's wait).
//! * `watchdog`: the fatal is set and the main thread never returns: the watchdog exits 86.

mod common;

use std::io::{BufRead, Read};
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use saber_sched::{
    build_state, mock_model_info, spawn_step_loop, stop_with_deadline, ArenaBytes, KvGeometry,
    Scheduler, SchedulerConfig, ScriptExecutor, StepBatch, StepExecutor, StepResult,
};
use saber_serve::detok::{ByteCodec, TextCodec};
use saber_serve::fatal;
use saber_serve::state::MinimalChatRenderer;
use serde_json::json;

use common::{boot_ledger, decode_buckets, plan_with, GIB};

const BOUND: Duration = Duration::from_secs(3);
const REASON: &str = "injected failure (fn73 host test): row read short";
const EOS: u32 = 255;

// ── the child ───────────────────────────────────────────────────────────────────

/// FN's executor shape: every step is a job the engine thread must answer.
#[derive(Debug)]
struct ChannelExec {
    inner: ScriptExecutor,
    tx: mpsc::Sender<mpsc::Sender<()>>,
}

impl StepExecutor for ChannelExec {
    fn execute(&mut self, batch: &StepBatch<'_>) -> StepResult {
        let (rtx, rrx) = mpsc::channel();
        if self.tx.send(rtx).is_err() || rrx.recv().is_err() {
            return StepResult::default();
        }
        self.inner.execute(batch)
    }
    fn admit(&mut self, seq: &saber_sched::AdmittedSeq) {
        self.inner.admit(seq);
    }
    fn release(&mut self, seq: saber_sched::SeqId) {
        self.inner.release(seq);
    }
}

fn config() -> SchedulerConfig {
    SchedulerConfig::from_plan(
        &plan_with(&decode_buckets(), 2),
        KvGeometry::PROD_FP8,
        ArenaBytes {
            kv_per_rank: 8 * GIB,
            gdn_state_per_rank: 8 * GIB,
        },
        262_144,
    )
    .expect("config")
}

fn script() -> ScriptExecutor {
    // long enough to be in flight when rank 1 fails (5 tokens a step, 20 ms a step: ~8 s)
    let text = "abcdefghij".repeat(200);
    let mut ids = ByteCodec.encode(&text).expect("encode");
    ids.push(EOS);
    ScriptExecutor::new(vec![ids; 8], 5, EOS).with_step_delay(Duration::from_millis(20))
}

/// The fake rank 1: its `saber: FATAL` line on stderr, then exit 86.
fn fake_rank1() -> std::process::Child {
    Command::new("sh")
        .arg("-c")
        .arg(format!(
            "sleep 1; echo 'fnworker rank1: FAILED: x' ; echo 'saber: FATAL rank 1: {REASON}' >&2; exit 86"
        ))
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the fake rank 1")
}

/// The client: one stream and one plain request in flight, then a request after the fatal.
fn client(port: u16) -> std::thread::JoinHandle<Vec<(String, bool, String)>> {
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rt");
        rt.block_on(async move {
            let base = format!("http://127.0.0.1:{port}/v1/chat/completions");
            let c = reqwest::Client::new();
            let body = |stream: bool| {
                json!({"model": "Qwen3.8", "stream": stream, "max_tokens": 4000,
                       "messages": [{"role": "user", "content": "hi"}]})
            };
            let s = c.post(&base).json(&body(true)).send();
            let p = c.post(&base).json(&body(false)).send();
            let (s, p) = tokio::join!(s, p);
            let mut out = Vec::new();
            let text = match s {
                Ok(r) => r.text().await.unwrap_or_default(),
                Err(e) => format!("ERR {e}"),
            };
            out.push((
                "stream_ended_with_503_and_reason".to_owned(),
                text.contains("ServiceUnavailableError") && text.contains(REASON),
                text.chars()
                    .rev()
                    .take(300)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect(),
            ));
            let (code, text) = match p {
                Ok(r) => (r.status().as_u16(), r.text().await.unwrap_or_default()),
                Err(e) => (0, format!("ERR {e}")),
            };
            out.push((
                "plain_request_503_with_reason".to_owned(),
                code == 503 && text.contains(REASON),
                format!("{code} {text}"),
            ));
            let after = c.post(&base).json(&body(false)).send().await;
            let (ok, d) = match after {
                Ok(r) => (r.status().as_u16() == 503, format!("{}", r.status())),
                Err(e) => (true, format!("refused: {e}")),
            };
            out.push(("new_request_refused".to_owned(), ok, d));
            out
        })
    })
}

fn report(checks: &[(String, bool, String)]) {
    for (name, ok, detail) in checks {
        println!(
            "fn73check {name} {} | {}",
            if *ok { "ok" } else { "FAIL" },
            detail.replace('\n', " ")
        );
    }
}

fn child_27b() {
    let sched = Arc::new(
        Scheduler::new(
            config(),
            script(),
            mock_model_info("Qwen3.8", "/x", EOS, 262_144),
            boot_ledger(),
        )
        .expect("arenas"),
    );
    let steps = spawn_step_loop(Arc::clone(&sched));
    let state = build_state(
        Arc::clone(&sched),
        Arc::new(ByteCodec),
        Arc::new(MinimalChatRenderer),
    );
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("rt");
    let listener = rt
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    println!("fn73port {port}");
    let mut r1 = fake_rank1();
    let _watch = fatal::RankWatch::start(r1.id(), 1, r1.stderr.take());
    let cl = client(port);
    rt.block_on(saber_serve::server::serve_with_shutdown(
        state,
        listener,
        fatal::wait(),
    ))
    .expect("serve");
    println!("fn73 serve returned");
    stop_with_deadline(sched, steps, Duration::from_secs(5)).expect("teardown");
    report(&cl.join().expect("client"));
    let _ = r1.wait();
    fatal::exit_if_fatal();
    println!("fn73 no fatal");
}

fn child_fn() {
    let (tx, jobs) = mpsc::channel::<mpsc::Sender<()>>();
    let exec = ChannelExec {
        inner: script(),
        tx,
    };
    let sched = Arc::new(
        Scheduler::new(
            config(),
            exec,
            mock_model_info("Qwen3.8", "/x", EOS, 262_144),
            boot_ledger(),
        )
        .expect("arenas"),
    );
    let steps = spawn_step_loop(Arc::clone(&sched));
    let state = build_state(
        Arc::clone(&sched),
        Arc::new(ByteCodec),
        Arc::new(MinimalChatRenderer),
    );
    let (ptx, prx) = mpsc::channel();
    // the HTTP plane on a thread of its own, as `flashnext::serve::start` builds it
    let http = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("rt");
        rt.block_on(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind");
            ptx.send(listener.local_addr().expect("addr").port())
                .expect("port");
            saber_serve::server::serve_with_shutdown(state, listener, fatal::wait())
                .await
                .expect("serve");
        });
        let _ = stop_with_deadline(sched, steps, Duration::from_secs(5));
    });
    let port = prx.recv().expect("port");
    println!("fn73port {port}");
    let mut r1 = fake_rank1();
    let _watch = fatal::RankWatch::start(r1.id(), 1, r1.stderr.take());
    let cl = client(port);
    // the engine thread's serve loop: a bounded wait with the fatal check (FN73), as `run_rank`
    let r: Result<(), String> = loop {
        if let Some(f) = fatal::get() {
            break Err(format!("the pair is down: {}", f.line()));
        }
        match jobs.recv_timeout(Duration::from_secs(1)) {
            Ok(reply) => {
                let _ = reply.send(());
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break Ok(()),
        }
    };
    if let Err(e) = &r {
        fatal::set_after(0, e.clone(), Duration::from_secs(1));
    }
    // the job channel goes BEFORE the join: a step waiting on a queued job returns
    drop(jobs);
    http.join().expect("http thread");
    println!("fn73 http joined");
    report(&cl.join().expect("client"));
    let _ = r1.wait();
    fatal::exit_if_fatal();
    println!("fn73 no fatal");
}

// ── FN73b: a rank the engine stopped is not a failure ─────────────────────────────

/// A fake rank 1 that waits for its `Stop` (stdin closed), then exits `code` (printing a FATAL
/// line first when `code` != 0).
fn fake_rank1_until_stop(code: i32) -> std::process::Child {
    let say = if code == 0 {
        String::new()
    } else {
        format!("echo 'saber: FATAL rank 1: {REASON}' >&2; ")
    };
    Command::new("sh")
        .arg("-c")
        .arg(format!("read x; {say}exit {code}"))
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the fake rank 1")
}

/// The leader's teardown after a run that FINISHED, as `Worker::stop_checked` /
/// `run_rank`'s caller do it: send `Stop`, (disarm), reap at once, hand the status back.
fn stop_and_reap(r1: &mut std::process::Child, watch: &fatal::RankWatch, disarm: bool) {
    drop(r1.stdin.take()); // `Stop`
    if disarm {
        watch.disarm();
    }
    // a blocking reap: the zombie lives microseconds, so an ARMED watch sees the pid gone
    let st = r1.wait().expect("reap");
    println!("fn73b rank 1 exited {st}");
    if disarm {
        watch.reaped(1, st);
    }
    // several of the watch's 50 ms polls
    std::thread::sleep(Duration::from_millis(400));
}

/// The gate / FN harness shape: the run ends, the leader stops and reaps rank 1.
fn child_clean(code: i32, disarm: bool) {
    let mut r1 = fake_rank1_until_stop(code);
    let watch = fatal::RankWatch::start(r1.id(), 1, r1.stderr.take());
    std::thread::sleep(Duration::from_millis(300));
    stop_and_reap(&mut r1, &watch, disarm);
    fatal::exit_if_fatal();
    println!("fn73 no fatal");
}

/// The served shape stopped by SIGTERM (the FN form: HTTP thread + engine thread + job channel):
/// the HTTP plane's shutdown future = the fatal OR SIGTERM (which calls `fatal::stop_requested`,
/// as `engine::shutdown_or_fatal` does); then the leader stops and reaps rank 1.
fn child_sigterm() {
    let (tx, jobs) = mpsc::channel::<mpsc::Sender<()>>();
    let exec = ChannelExec {
        inner: script(),
        tx,
    };
    let sched = Arc::new(
        Scheduler::new(
            config(),
            exec,
            mock_model_info("Qwen3.8", "/x", EOS, 262_144),
            boot_ledger(),
        )
        .expect("arenas"),
    );
    let steps = spawn_step_loop(Arc::clone(&sched));
    let state = build_state(
        Arc::clone(&sched),
        Arc::new(ByteCodec),
        Arc::new(MinimalChatRenderer),
    );
    let (ptx, prx) = mpsc::channel();
    let http = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("rt");
        rt.block_on(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind");
            let mut term =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("SIGTERM");
            ptx.send(listener.local_addr().expect("addr").port())
                .expect("port");
            saber_serve::server::serve_with_shutdown(state, listener, async move {
                tokio::select! {
                    _ = term.recv() => {
                        println!("fn73b signal received");
                        fatal::stop_requested();
                    }
                    () = fatal::wait() => {}
                }
            })
            .await
            .expect("serve");
        });
        let _ = stop_with_deadline(sched, steps, Duration::from_secs(5));
    });
    let port = prx.recv().expect("port");
    let mut r1 = fake_rank1_until_stop(0);
    let watch = fatal::RankWatch::start(r1.id(), 1, r1.stderr.take());
    println!("fn73port {port}");
    let r: Result<(), String> = loop {
        if let Some(f) = fatal::get() {
            break Err(format!("the pair is down: {}", f.line()));
        }
        match jobs.recv_timeout(Duration::from_millis(200)) {
            Ok(reply) => {
                let _ = reply.send(());
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if http.is_finished() {
                    break Ok(());
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break Ok(()),
        }
    };
    drop(jobs);
    http.join().expect("http thread");
    println!("fn73 http joined ({r:?})");
    stop_and_reap(&mut r1, &watch, true);
    fatal::exit_if_fatal();
    println!("fn73 no fatal");
}

fn child_watchdog() {
    fatal::set(1, REASON);
    // a wait that never returns (a kernel spinning on a dead peer's flags)
    loop {
        std::thread::sleep(Duration::from_secs(60));
    }
}

#[test]
fn fn73_child_entry() {
    let Some(form) = std::env::args().find_map(|a| a.strip_prefix("fn73form:").map(str::to_owned))
    else {
        return;
    };
    fatal::install(BOUND, |c| std::process::exit(c));
    // libtest printed `test fn73_child_entry ... ` with no newline: every line below is whole
    println!();
    match form.as_str() {
        "27b" => child_27b(),
        "fn" => child_fn(),
        "watchdog" => child_watchdog(),
        "clean" => child_clean(0, true),
        "clean-armed" => child_clean(0, false),
        "stopfail" => child_clean(86, true),
        "sigterm" => child_sigterm(),
        other => panic!("fn73form:{other}"),
    }
}

// ── the parent ──────────────────────────────────────────────────────────────────

struct Run {
    code: Option<i32>,
    secs: f64,
    stdout: Vec<String>,
}

fn run_child(form: &str) -> Run {
    run_child_then(form, |_| {})
}

/// [`run_child`], calling `then(pid)` once the child printed its port line.
fn run_child_then(form: &str, then: impl FnOnce(u32) + Send + 'static) -> Run {
    let exe = std::env::current_exe().expect("exe");
    let t = Instant::now();
    let mut ch = Command::new(exe)
        .args([
            "--exact",
            "fn73_child_entry",
            &format!("fn73form:{form}"),
            "--nocapture",
            "--test-threads",
            "1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the child");
    let mut err = ch.stderr.take().expect("stderr");
    let errt = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = err.read_to_string(&mut s);
        s
    });
    let out = std::io::BufReader::new(ch.stdout.take().expect("stdout"));
    let pid = ch.id();
    let mut then = Some(then);
    let mut stdout: Vec<String> = Vec::new();
    for l in out.lines().map_while(Result::ok) {
        if l.starts_with("fn73port ") {
            if let Some(f) = then.take() {
                f(pid);
            }
        }
        stdout.push(l);
    }
    // bounded: a child that hangs is the defect under test
    let deadline = Instant::now() + Duration::from_secs(60);
    let code = loop {
        if let Some(s) = ch.try_wait().expect("wait") {
            break s.code();
        }
        if Instant::now() > deadline {
            let _ = ch.kill();
            let _ = ch.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let secs = t.elapsed().as_secs_f64();
    let stderr = errt.join().unwrap_or_default();
    eprintln!(
        "---- child {form} ({secs:.2} s, code {code:?}) stdout:\n{}\n---- stderr:\n{stderr}",
        stdout.join("\n")
    );
    Run { code, secs, stdout }
}

fn assert_fatal_exit(form: &str, run: &Run, max_secs: f64) {
    assert_eq!(
        run.code,
        Some(fatal::FATAL_EXIT_CODE),
        "{form}: the exit code"
    );
    assert!(
        run.secs < max_secs,
        "{form}: exited after {:.2} s (bound {max_secs})",
        run.secs
    );
    let last = run
        .stdout
        .iter()
        .rev()
        .find(|l| !l.trim().is_empty())
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        last,
        format!("saber: FATAL rank 1: {REASON}"),
        "{form}: the final line"
    );
}

fn assert_served_law(form: &str, run: &Run) {
    let checks: Vec<&String> = run
        .stdout
        .iter()
        .filter(|l| l.starts_with("fn73check "))
        .collect();
    assert_eq!(checks.len(), 3, "{form}: the client's three checks ran");
    for c in checks {
        assert!(c.contains(" ok |"), "{form}: {c}");
    }
    let port: u16 = run
        .stdout
        .iter()
        .find_map(|l| l.strip_prefix("fn73port "))
        .and_then(|p| p.trim().parse().ok())
        .expect("the port line");
    assert!(
        TcpStream::connect(("127.0.0.1", port)).is_err(),
        "{form}: the listener is closed"
    );
}

#[test]
fn the_27b_form_exits_86_with_the_rank_reason_and_503s() {
    let run = run_child("27b");
    // the fake rank fails at ~1 s; the teardown must not need the watchdog (3 s) to finish
    assert_fatal_exit("27b", &run, 1.0 + BOUND.as_secs_f64());
    assert!(
        run.stdout.iter().any(|l| l == "fn73 serve returned"),
        "27b: the orderly teardown ran (not the watchdog)"
    );
    assert_served_law("27b", &run);
}

#[test]
fn the_fn_form_exits_86_with_the_rank_reason_and_503s() {
    let run = run_child("fn");
    assert_fatal_exit("fn", &run, 1.0 + BOUND.as_secs_f64());
    assert!(
        run.stdout.iter().any(|l| l == "fn73 http joined"),
        "fn: the HTTP thread was joined (the incident's wait returned)"
    );
    assert_served_law("fn", &run);
}

#[test]
fn the_watchdog_exits_86_when_a_wait_never_returns() {
    let run = run_child("watchdog");
    assert_fatal_exit("watchdog", &run, BOUND.as_secs_f64() + 2.0);
    assert!(
        run.secs >= BOUND.as_secs_f64() - 0.1,
        "the watchdog waited its bound"
    );
}

/// The negative control: a rank that exits 0 (it obeyed `Stop`) is not a fatal.
#[test]
fn a_clean_rank_exit_is_not_a_fatal() {
    let mut ch = Command::new("sh")
        .args(["-c", "exit 0"])
        .spawn()
        .expect("spawn");
    let w = fatal::RankWatch::start(ch.id(), 1, None::<std::process::ChildStderr>);
    std::thread::sleep(Duration::from_millis(400));
    assert!(fatal::get().is_none(), "a clean exit set the fatal");
    w.disarm();
    let _ = ch.wait();
}

fn assert_clean_exit(form: &str, run: &Run) {
    assert_eq!(run.code, Some(0), "{form}: a clean stop exits 0");
    assert!(
        !run.stdout.iter().any(|l| l.contains("FATAL")),
        "{form}: no FATAL line"
    );
    assert!(
        run.stdout.iter().any(|l| l == "fn73 no fatal"),
        "{form}: the teardown ran to its end"
    );
}

/// FN73b: the leader stops rank 1 and reaps it at once (the 27B gate's `Worker::stop_checked`,
/// FN's `run_rank` caller): exit 0, no FATAL line.
#[test]
fn fn73b_a_rank_the_leader_stopped_and_reaped_is_not_a_fatal() {
    let run = run_child("clean");
    assert_clean_exit("clean", &run);
}

/// FN73b's negative control = the P27-3b race itself: the same reap with the watch ARMED sees the
/// pid gone (`ExitSeen::Gone`) and raises the fatal on a clean exit 0.
#[test]
fn fn73b_control_an_armed_watch_calls_the_leaders_reap_a_failure() {
    let run = run_child("clean-armed");
    assert_eq!(run.code, Some(fatal::FATAL_EXIT_CODE), "the control's race");
    assert!(
        run.stdout
            .iter()
            .any(|l| l.contains("exited (already reaped)")),
        "the control's reason is the reaped pid"
    );
}

/// FN73b: a rank that exits NON-ZERO after the leader's `Stop` is still a failed rank (exit 86,
/// its own reason), though the watch was disarmed for the reap.
#[test]
fn fn73b_a_rank_that_fails_on_its_stop_is_still_a_fatal() {
    let run = run_child("stopfail");
    assert_fatal_exit("stopfail", &run, 5.0);
}

/// FN73b: the served FN shape stopped by SIGTERM: exit 0, no FATAL line.
#[test]
fn fn73b_a_served_sigterm_stop_exits_0() {
    let run = run_child_then("sigterm", |pid| {
        std::thread::sleep(Duration::from_millis(500));
        let _ = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status();
    });
    assert_clean_exit("sigterm", &run);
    assert!(
        run.stdout.iter().any(|l| l == "fn73b signal received"),
        "sigterm: the signal stopped it"
    );
}
