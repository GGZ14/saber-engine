//! FN73: the teardown law -- **a failed rank exits the process**.
//!
//! Incident 2026-10-07 (FN55c hold p5_Li): rank 1 of a Flash-Next pair failed its worker; rank 0's
//! serve loop stopped, then its main thread joined the HTTP thread, whose server ended only on
//! SIGINT/SIGTERM. The process held both cards for ~40 minutes with the listener open.
//!
//! The law, process-wide:
//! * a rank failure calls [`set`] (first caller wins: `rank` + a one-line reason);
//! * every blocking wait of the serve plane observes it: the HTTP plane's shutdown future
//!   ([`wait`]), the scheduler's step loop (fails every in-flight and queued request with the
//!   reason; new requests get 503) and the engines' serve loops (a bounded wait with a fatal check);
//! * the process exits with [`FATAL_EXIT_CODE`] and a final line `saber: FATAL rank R: <reason>`
//!   ([`exit_if_fatal`]);
//! * a watchdog thread is the last resort: if the process is still alive [`watchdog_bound`]
//!   after [`set`], it prints the line and calls the installed hard exit (`_exit`: no atexit
//!   handler runs, so a live HIP thread cannot turn the code into a 139).
//!
//! [`RankWatch`] is the leader's view of its child rank: the child's exit (non-zero or a signal)
//! sets the fatal, with the child's own `saber: FATAL rank 1:` line (forwarded from its stderr)
//! as the reason.
//!
//! FN73b: **a rank the engine stopped is not a failure.** The leader disarms the watch the moment
//! it starts its own teardown ([`RankWatch::disarm`] at the head of the reap; a SIGINT/SIGTERM
//! stop disarms every watch via [`stop_requested`]), so an exit the leader asked for -- reaped by
//! the leader before the watch's `/proc` poll saw it ([`ExitSeen::Gone`]) -- never raises a fatal.
//! The leader that reaps the child hands the status back ([`RankWatch::reaped`]): a non-zero exit
//! it did not cause by its own kill is still the rank's failure.

use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The process exit code of a fatal teardown (distinct from 1 = an ordinary error, 101 = a panic,
/// 124 = `timeout`, 134 = abort, 137/143 = SIGKILL/SIGTERM, 139 = SIGSEGV).
pub const FATAL_EXIT_CODE: i32 = 86;

/// The watchdog's default bound.
pub const DEFAULT_WATCHDOG: Duration = Duration::from_secs(10);

/// The prefix of the final line (a leader parses its child's).
pub const LINE_PREFIX: &str = "saber: FATAL rank ";

/// One fatal: which rank failed and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fatal {
    pub rank: u32,
    pub reason: String,
    pub at: Instant,
}

impl Fatal {
    /// `saber: FATAL rank R: <reason>` (one line).
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "{LINE_PREFIX}{}: {}",
            self.rank,
            self.reason.replace('\n', " | ")
        )
    }
}

/// Parse a `saber: FATAL rank R: <reason>` line.
#[must_use]
pub fn parse_line(line: &str) -> Option<(u32, String)> {
    let rest = line.trim_end().strip_prefix(LINE_PREFIX)?;
    let (r, reason) = rest.split_once(": ")?;
    Some((r.parse().ok()?, reason.to_owned()))
}

type Hook = Box<dyn Fn(&Fatal) + Send + Sync>;

static CELL: OnceLock<Fatal> = OnceLock::new();
static NOTIFY: OnceLock<tokio::sync::Notify> = OnceLock::new();
static HOOKS: Mutex<Vec<Hook>> = Mutex::new(Vec::new());
static INSTALLED: OnceLock<(Duration, fn(i32) -> !)> = OnceLock::new();
/// FN73b: the process was asked to stop (SIGINT / SIGTERM): every rank exit from here on is the
/// stop's, not a rank failure.
static STOPPING: AtomicBool = AtomicBool::new(false);

/// FN73b: the process is stopping on a signal (the HTTP plane's shutdown future calls it): every
/// [`RankWatch`] is disarmed -- a child rank in the same process group gets the same SIGINT, and
/// its exit is the stop's.
pub fn stop_requested() {
    STOPPING.store(true, Ordering::Relaxed);
}

/// Has [`stop_requested`] run?
#[must_use]
pub fn stopping() -> bool {
    STOPPING.load(Ordering::Relaxed)
}

fn notify() -> &'static tokio::sync::Notify {
    NOTIFY.get_or_init(tokio::sync::Notify::new)
}

fn default_exit(code: i32) -> ! {
    std::process::exit(code)
}

/// The binary's choice of watchdog bound and hard exit (`saber_hip::hard_exit` in `saber`);
/// first call wins. Without it the bound is [`DEFAULT_WATCHDOG`] and the exit `process::exit`.
pub fn install(bound: Duration, hard_exit: fn(i32) -> !) {
    let _ = INSTALLED.set((bound, hard_exit));
}

/// The watchdog's bound.
#[must_use]
pub fn watchdog_bound() -> Duration {
    INSTALLED.get().map_or(DEFAULT_WATCHDOG, |x| x.0)
}

fn hard_exit_fn() -> fn(i32) -> ! {
    INSTALLED
        .get()
        .map_or(default_exit as fn(i32) -> !, |x| x.1)
}

/// The fatal, once set.
#[must_use]
pub fn get() -> Option<&'static Fatal> {
    CELL.get()
}

/// Set the fatal (first caller wins; returns whether this call set it). Prints
/// `fatal rank R: <reason> -- tearing down`, wakes every [`wait`], runs every [`on_fatal`] hook on
/// a thread of its own and arms the watchdog.
pub fn set(rank: u32, reason: impl Into<String>) -> bool {
    let f = Fatal {
        rank,
        reason: reason.into(),
        at: Instant::now(),
    };
    if CELL.set(f).is_err() {
        return false;
    }
    let f = CELL.get().expect("just set");
    let bound = watchdog_bound();
    eprintln!(
        "fatal rank {}: {} -- tearing down (503 to every request, listener closed, exit {} within \
         {} s)",
        f.rank,
        f.reason,
        FATAL_EXIT_CODE,
        bound.as_secs_f64()
    );
    notify().notify_waiters();
    let _ = std::thread::Builder::new()
        .name("saber-fatal-hooks".into())
        .spawn(|| {
            let f = CELL.get().expect("set");
            let hooks = HOOKS
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for h in hooks.iter() {
                h(f);
            }
        });
    let exit = hard_exit_fn();
    let _ = std::thread::Builder::new()
        .name("saber-fatal-watchdog".into())
        .spawn(move || {
            std::thread::sleep(bound);
            let f = CELL.get().expect("set");
            eprintln!(
                "fatal watchdog: the process is still alive {} s after the fatal; exiting {}",
                bound.as_secs_f64(),
                FATAL_EXIT_CODE
            );
            final_line(f);
            exit(FATAL_EXIT_CODE)
        });
    true
}

/// [`set`], after giving another reporter `grace` to get there first (a leader that read EOF from
/// a dead peer waits for the peer's own reason).
pub fn set_after(rank: u32, reason: impl Into<String>, grace: Duration) -> bool {
    let until = Instant::now() + grace;
    while get().is_none() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(10));
    }
    set(rank, reason)
}

/// Run `hook` when the fatal is set (on the hooks thread; at once on that thread if it already is).
pub fn on_fatal(hook: impl Fn(&Fatal) + Send + Sync + 'static) {
    let mut hooks = HOOKS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(f) = get() {
        hook(f);
    }
    hooks.push(Box::new(hook));
}

/// Resolves when the fatal is set (the HTTP plane's shutdown future selects on it).
pub async fn wait() {
    loop {
        let n = notify().notified();
        tokio::pin!(n);
        n.as_mut().enable();
        if get().is_some() {
            return;
        }
        n.await;
    }
}

fn final_line(f: &Fatal) {
    let line = f.line();
    let _ = std::io::stderr().flush();
    println!("{line}");
    let _ = std::io::stdout().flush();
    eprintln!("{line}");
    let _ = std::io::stderr().flush();
}

/// If the fatal is set: print the final line (stdout and stderr) and exit [`FATAL_EXIT_CODE`]
/// through the installed hard exit. The caller has released its devices.
pub fn exit_if_fatal() {
    if let Some(f) = get() {
        final_line(f);
        hard_exit_fn()(FATAL_EXIT_CODE)
    }
}

/// The leader's watch over a child rank: the exit of `pid` with a non-zero code or a signal sets
/// the fatal (reason: the child's own `saber: FATAL` line when it printed one on the forwarded
/// stderr, else its exit status). A clean exit (0) is the child obeying `Stop` and sets nothing.
#[derive(Debug, Clone)]
pub struct RankWatch {
    disarmed: Arc<AtomicBool>,
    reason: Arc<Mutex<Option<String>>>,
}

impl RankWatch {
    /// Start watching `pid` (rank `rank`); `stderr` = the child's piped stderr, forwarded line by
    /// line to this process's stderr.
    pub fn start(
        pid: u32,
        rank: u32,
        stderr: Option<impl std::io::Read + Send + 'static>,
    ) -> RankWatch {
        let w = RankWatch {
            disarmed: Arc::new(AtomicBool::new(false)),
            reason: Arc::new(Mutex::new(None)),
        };
        if let Some(e) = stderr {
            let reason = Arc::clone(&w.reason);
            let _ = std::thread::Builder::new()
                .name(format!("rank{rank}-stderr"))
                .spawn(move || {
                    let mut r = std::io::BufReader::new(e);
                    let mut line = String::new();
                    loop {
                        line.clear();
                        match r.read_line(&mut line) {
                            Ok(0) | Err(_) => break,
                            Ok(_) => {
                                if let Some((_, why)) = parse_line(&line) {
                                    *reason
                                        .lock()
                                        .unwrap_or_else(std::sync::PoisonError::into_inner) =
                                        Some(why);
                                }
                                let mut o = std::io::stderr().lock();
                                let _ = o.write_all(line.as_bytes());
                                let _ = o.flush();
                            }
                        }
                    }
                });
        }
        let me = w.clone();
        let _ = std::thread::Builder::new()
            .name(format!("rank{rank}-watch"))
            .spawn(move || loop {
                if me.is_disarmed() || get().is_some() {
                    return;
                }
                if let Some(status) = proc_exit(pid) {
                    // FN73b: the leader's own reap (after it disarmed) shows as `Gone` here
                    if me.is_disarmed() {
                        return;
                    }
                    if status == ExitSeen::Code(0) {
                        return;
                    }
                    // the stderr forwarder may still hold the child's last line
                    std::thread::sleep(Duration::from_millis(100));
                    let own = me
                        .reason
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone();
                    set(
                        rank,
                        own.unwrap_or_else(|| format!("rank {rank} (pid {pid}) {status}")),
                    );
                    return;
                }
                std::thread::sleep(Duration::from_millis(50));
            });
        w
    }

    /// The leader is about to stop or kill the child itself: its exit is no longer a fatal (FN73b:
    /// called at the head of every reap, before the first `try_wait` can take the zombie).
    pub fn disarm(&self) {
        self.disarmed.store(true, Ordering::Relaxed);
    }

    /// Disarmed by the leader or by a signal stop ([`stop_requested`]).
    #[must_use]
    pub fn is_disarmed(&self) -> bool {
        self.disarmed.load(Ordering::Relaxed) || stopping()
    }

    /// The child's own `saber: FATAL` reason, if it printed one on the forwarded stderr.
    #[must_use]
    pub fn child_reason(&self) -> Option<String> {
        self.reason
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// FN73b: the leader reaped the child after asking it to stop (NOT after killing it): a clean
    /// exit is nothing; a non-zero code or a signal is still rank `rank`'s failure -- the fatal,
    /// with the child's own reason when it printed one. A signal stop ([`stop_requested`]) makes
    /// every exit the stop's. Returns whether this call set the fatal.
    pub fn reaped(&self, rank: u32, status: std::process::ExitStatus) -> bool {
        self.disarm();
        if status.success() || stopping() {
            return false;
        }
        // the stderr forwarder may still hold the child's last line
        std::thread::sleep(Duration::from_millis(100));
        set(
            rank,
            self.child_reason()
                .unwrap_or_else(|| format!("rank {rank} exited {status} after the stop")),
        )
    }
}

/// How a watched child ended, as `/proc/<pid>/stat` tells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitSeen {
    Code(i32),
    Signal(i32),
    /// Reaped already (no `/proc` entry): the status is the reaper's.
    Gone,
}

impl std::fmt::Display for ExitSeen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExitSeen::Code(c) => write!(f, "exited with code {c}"),
            ExitSeen::Signal(s) => write!(f, "was killed by signal {s}"),
            ExitSeen::Gone => write!(f, "exited (already reaped)"),
        }
    }
}

/// `Some` once `pid` is a zombie (or gone): field 3 (state) `Z`/`X`, field 52 (the wait status).
#[must_use]
pub fn proc_exit(pid: u32) -> Option<ExitSeen> {
    let Ok(s) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
        return Some(ExitSeen::Gone);
    };
    let after = &s[s.rfind(')')? + 1..];
    let f: Vec<&str> = after.split_whitespace().collect();
    // f[0] = field 3 (state); field 52 = f[49]
    match f.first() {
        Some(&"Z") | Some(&"X") => {}
        _ => return None,
    }
    let ws: i32 = f.get(49).and_then(|x| x.parse().ok()).unwrap_or(0);
    Some(if ws & 0x7f == 0 {
        ExitSeen::Code((ws >> 8) & 0xff)
    } else {
        ExitSeen::Signal(ws & 0x7f)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_line_round_trips_and_is_one_line() {
        let f = Fatal {
            rank: 1,
            reason: "row read: short read\nat offset 7".into(),
            at: Instant::now(),
        };
        assert_eq!(
            f.line(),
            "saber: FATAL rank 1: row read: short read | at offset 7"
        );
        assert_eq!(
            parse_line(&format!("{}\n", f.line())),
            Some((1, "row read: short read | at offset 7".into()))
        );
        assert_eq!(parse_line("fnworker rank1: FAILED: x"), None);
    }

    #[test]
    fn proc_exit_reads_a_zombie_code_and_a_signal() {
        let mut a = std::process::Command::new("sh")
            .args(["-c", "exit 7"])
            .spawn()
            .unwrap();
        let mut b = std::process::Command::new("sh")
            .args(["-c", "kill -9 $$"])
            .spawn()
            .unwrap();
        let mut c = std::process::Command::new("sleep")
            .arg("5")
            .spawn()
            .unwrap();
        let t = Instant::now();
        let (mut sa, mut sb) = (None, None);
        while (sa.is_none() || sb.is_none()) && t.elapsed() < Duration::from_secs(5) {
            sa = sa.or(proc_exit(a.id()));
            sb = sb.or(proc_exit(b.id()));
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(sa, Some(ExitSeen::Code(7)));
        assert_eq!(sb, Some(ExitSeen::Signal(9)));
        // the control: a live child is not an exit
        assert_eq!(proc_exit(c.id()), None);
        let _ = c.kill();
        for ch in [&mut a, &mut b, &mut c] {
            let _ = ch.wait();
        }
        assert_eq!(proc_exit(a.id()), Some(ExitSeen::Gone));
    }
}
