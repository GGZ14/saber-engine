//! The two-rank forward: the collectives a TP=2 pass needs, spliced into a rank's plan.
//!
//! [`crate::layer::BlockPlan`] routes every `layer.allreduce` to [`Route::Identity`] — "rank 0 of
//! a TP=2 tier with no peer: the output edge ALIASES the input edge". That is the honest answer
//! for a single card, and it is what makes this module possible rather than a rewrite: **the
//! alias is exactly the buffer a real all-reduce should reduce in place.** The plan allocates no
//! buffer for the reduce's output edge and points every consumer at the producer's, so replacing
//! the identity with `saber`'s one-shot two-rank kernel over that same address gives every consumer
//! the summed value with no copy and no second buffer.
//!
//! In-place is safe by construction, not by hope: the kernel's reduce phase is
//! `out4[k] = in4[k] + my_scratch[k]` with one thread per index, and the push phase that reads all
//! of `in4` has completed and been flagged before it runs (`hip/saber_ar_exact.hip`). One thread
//! reads index `k` and writes index `k`; nothing else reads it afterwards.
//!
//! # What this module does, since ws-h16
//!
//! Almost nothing, and that is the point. The 129 reduces used to be spliced in here, because
//! `BlockPlan` routed every `layer.allreduce` to `Route::Identity` and something had to put a
//! kernel back. The planner now resolves the real route itself
//! ([`crate::layer::Collectives::Real`]), so a two-rank pass is an ordinary bound op sequence and
//! what is left here is the two things a collective needs that a plan cannot carry:
//!
//! * [`agree_on_collectives`] — the launch digests, compared on the host before anything launches,
//!   because a graph cannot rendezvous;
//! * [`wait_bounded`] — every wait in a two-rank pass, polled to a deadline, so a rank that falls
//!   out of step fails loud instead of holding a card.
//!
//! [`collective_steps`] names them for a report or a timing arm.

use std::sync::Arc;
use std::time::{Duration, Instant};

use saber_hip::{HipThread, Stream};
use saber_kernels::rules::WorkspaceKind;

use crate::block_capture::BoundStep;
use crate::error::{Error, Result};
use crate::launch::{LaunchCtx, Launcher};
use crate::peer::RankPeer;

/// How long any single wait in a two-rank pass may take before it is called a hang.
///
/// The all-reduce kernel spins on its peer's flag with a bounded spin (`SABER_AR_SPIN_MAX`), and
/// every host wait here polls `hipStreamQuery`. Twenty seconds is four orders of magnitude above a
/// 2048-token forward and still short enough that a wedged pair releases the GPU lock while
/// someone is still watching.
pub const WAIT_DEADLINE: Duration = Duration::from_secs(20);

/// How a host wait polls: how long it spins before it starts sleeping, and how long it sleeps.
///
/// There is no single right answer, and the wrong one costs real time in both directions — which is
/// why this is a parameter with two named presets rather than a constant:
///
/// * a **gate** waits once per collective and the wait is the thing being measured, so it spins:
///   a flat millisecond of sleep after each of 129 collectives measured its own sleep and reported
///   it as the all-reduce share (ws-h10, corrected);
/// * a **serving step** waits once per step on BOTH ranks at the same time, and a step is ~17 ms,
///   so ten milliseconds of `spin_loop` burns two cores per step to save nothing. Worse, measured
///   by ws-l13 on 2026-09-20: with the long spin the engine's per-step p50 was 17.17 ms and the
///   **tail reached 386 ms**, because a rank the OS descheduled left the other one's all-reduce
///   spinning on a flag, and a pair's step time is the slower rank's. A short spin then sleeps
///   costs at most one sleep of overshoot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaitWindow {
    /// Spin for this long before sleeping at all.
    pub spin: Duration,
    /// Then sleep this long between polls.
    pub sleep: Duration,
    /// Give up after this.
    pub deadline: Duration,
}

impl WaitWindow {
    /// A gate or a measurement: spin 10 ms, then sleep 1 ms.
    pub const GATE: WaitWindow = WaitWindow {
        spin: Duration::from_millis(10),
        sleep: Duration::from_millis(1),
        deadline: WAIT_DEADLINE,
    };
    /// A serving step: spin 200 µs, then sleep 250 µs (ws-l13's measurement).
    pub const STEP: WaitWindow = WaitWindow {
        spin: Duration::from_micros(200),
        sleep: Duration::from_micros(250),
        deadline: WAIT_DEADLINE,
    };
}

/// Poll `hipStreamQuery` on the [`WaitWindow::GATE`] schedule instead of blocking in
/// `hipStreamSynchronize`.
///
/// # Errors
/// The deadline passing, which is how a pair that is waiting for a push that is not coming stops
/// instead of holding a card.
pub fn wait_bounded(ctx: &LaunchCtx<'_>, stream: &Stream, what: &str) -> Result<()> {
    wait_bounded_on(ctx.hip, stream, what, WaitWindow::GATE)
}

/// [`wait_bounded`] on a chosen schedule, for a caller that holds a [`HipThread`] rather than a
/// [`LaunchCtx`] — the engine's step loop is one.
///
/// # Errors
/// The deadline passing, or the stream query failing.
pub fn wait_bounded_on(
    hip: &HipThread,
    stream: &Stream,
    what: &str,
    window: WaitWindow,
) -> Result<()> {
    wait_aimed(hip, stream, what, window, None).map(|_| ())
}

/// FN57: whether this handle's waits are ONE bounded runtime wait instead of a poll -- true once
/// the engine has set an explicit device schedule ([`HostWait::schedule`]): `Spin` (the runtime's
/// active wait) or `BlockingSync` (the interrupt wait).
#[must_use]
pub fn runtime_waits(hip: &HipThread) -> bool {
    matches!(
        hip.schedule_set(),
        Some(saber_hip::DeviceSchedule::Spin | saber_hip::DeviceSchedule::BlockingSync)
    )
}

/// FN57: the stream's work so far, waited by the RUNTIME (an event recorded now, then
/// `hipEventSynchronize` on the HIP thread) with the caller's deadline -- the `spin` / `block`
/// arms' wait. What it saw: one "poll", the wait's wall, no gap (the overshoot is the runtime's).
///
/// # Errors
/// The deadline passing, or a HIP call failing.
pub fn wait_runtime(
    hip: &HipThread,
    stream: &Stream,
    what: &str,
    deadline: Duration,
) -> Result<WaitSeen> {
    let start = Instant::now();
    let ev = hip.create_event(saber_hip::EventFlags::DisableTiming)?;
    hip.record_event(&ev, stream)?;
    hip.sync_event_within(&ev, what, deadline)
        .map_err(|e| match e {
            saber_hip::Error::Timeout { what, seconds } => Error::Timeout { what, seconds },
            e => Error::from(e),
        })?;
    Ok(WaitSeen {
        waited: start.elapsed(),
        last_gap: Duration::ZERO,
        polls: 1,
        slept_to_aim: false,
    })
}

/// What one wait saw — enough for the host to bound its own OVERSHOOT without a device clock.
///
/// The work completed at some instant after the last poll that said "not yet" and no later than the
/// poll that said "done", so `last_gap` (the time between the two polls' returns) is an upper bound
/// on how long the host sat idle after the card had finished. Under [`WaitWindow::STEP`] that gap
/// is one sleep quantum (250 µs asked, ~310 µs taken at Linux's default 50 µs timer slack) and the
/// expected overshoot half of it; under an [`WaitAim`] it is one poll (ws-h32).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WaitSeen {
    /// From the wait's start to the return of the poll that saw the work done.
    pub waited: Duration,
    /// Between the last "not yet" poll's return and the "done" poll's return: the overshoot bound.
    /// Zero when the first poll already saw the work done.
    pub last_gap: Duration,
    /// How many times the stream was queried.
    pub polls: u32,
    /// Whether the aim's one coarse sleep ran.
    pub slept_to_aim: bool,
}

/// **THE ADAPTIVE WAIT'S LAW** (ws-h32): where a wait expects the card to finish, and how close to
/// that instant it is willing to sleep.
///
/// [`WaitWindow::STEP`] spins for its first 200 µs — when the card is surely still busy — and then
/// sleeps in 250 µs quanta through the part where it finishes, so every wait overshoots the card's
/// completion by up to a whole quantum and by half of one on average. A decode step repeats the
/// same shape, so the host KNOWS roughly when the card will be done: the previous waits at this
/// site and shape. The aim inverts the window: **one coarse sleep to `expect - guard`, then spin**
/// (poll back to back) until the work is done, for at most `spin_cap` past `expect`, after which it
/// falls back to the window's sleeps so a stalled peer never burns a core for the whole deadline.
///
/// THE HOST LAW (held by `the_aimed_wait_never_sleeps_past_the_card` on a mock whose completion time
/// is known): when the card finishes anywhere in `[expect - guard + slack, expect + spin_cap]`, the
/// wait returns within ONE POLL of it. Outside that band it degrades to the window, never worse.
///
/// THE CPU IT COSTS: one core busy-polling for `guard` plus the sleep's own wake-up lateness per wait
/// (≈ 0.25–0.35 ms) — and the HIP thread answering those polls — where `STEP` spun 200 µs at the
/// START of every wait for nothing. Three waits a step at ~23 ms is ~1 ms of spin per step, ~4 % of
/// one core per rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaitAim {
    /// When, from the wait's start, the work is expected to be done.
    pub expect: Duration,
    /// How far ahead of `expect` the coarse sleep ends. It must cover the sleep's own wake-up
    /// lateness (Linux's default timer slack is 50 µs) plus how much EARLIER than predicted the card
    /// may finish; the predictor feeds the minimum of recent waits, so the second term is small.
    pub guard: Duration,
    /// How long past `expect` the wait keeps spinning before it falls back to the window's sleeps.
    pub spin_cap: Duration,
}

impl WaitAim {
    /// The serving preset's guard and cap (ws-h32). The guard is 250 µs: the step wait's own p90
    /// − p50 spread measured by ws-h31 is ~80 µs and the sleep's lateness ~60 µs, so a step that
    /// finishes a guard early is a step 3x outside what either ever showed.
    pub const GUARD: Duration = Duration::from_micros(250);
    /// Spin at most this far past the expectation before sleeping again.
    pub const SPIN_CAP: Duration = Duration::from_millis(1);

    /// An aim at `expect` with the serving preset's guard and cap.
    #[must_use]
    pub const fn at(expect: Duration) -> WaitAim {
        WaitAim {
            expect,
            guard: WaitAim::GUARD,
            spin_cap: WaitAim::SPIN_CAP,
        }
    }
}

/// The wait's notion of time (FN57): the real clock in the engine, a virtual one in the tests, so
/// the sleeper's laws are held exactly instead of against a loaded host's scheduler.
pub trait WaitClock {
    /// Time since the clock's origin.
    fn now(&self) -> Duration;
    /// Sleep (give the core back) for about `d`.
    fn sleep(&self, d: Duration);
    /// One busy-poll pause.
    fn spin(&self);
}

/// The wall clock: [`Instant`], `std::thread::sleep`, `std::hint::spin_loop`.
#[derive(Debug, Clone, Copy)]
pub struct RealClock {
    origin: Instant,
}

impl RealClock {
    /// A clock whose origin is now.
    #[must_use]
    pub fn start() -> RealClock {
        RealClock {
            origin: Instant::now(),
        }
    }
}

impl WaitClock for RealClock {
    fn now(&self) -> Duration {
        self.origin.elapsed()
    }
    fn sleep(&self, d: Duration) {
        std::thread::sleep(d);
    }
    fn spin(&self) {
        std::hint::spin_loop();
    }
}

/// The wait itself, over any readiness probe: [`wait_bounded_on`]'s loop, factored out so that the
/// law it follows can be held on the host against a mock with a known completion time.
///
/// Without an `aim` this is exactly the window's schedule (spin, then sleep). With one, see
/// [`WaitAim`]. The first poll always runs BEFORE any sleep, so work that is already done costs one
/// query and no sleep at all.
///
/// # Errors
/// The deadline passing, or the probe failing.
pub fn wait_polled<F>(
    ready: F,
    what: &str,
    window: WaitWindow,
    aim: Option<WaitAim>,
) -> Result<WaitSeen>
where
    F: FnMut() -> Result<bool>,
{
    wait_polled_on(&RealClock::start(), ready, what, window, aim)
}

/// [`wait_polled`] on any [`WaitClock`] (FN57: the tests run it on a virtual clock).
///
/// # Errors
/// The deadline passing, or the probe failing.
pub fn wait_polled_on<C, F>(
    clock: &C,
    mut ready: F,
    what: &str,
    window: WaitWindow,
    aim: Option<WaitAim>,
) -> Result<WaitSeen>
where
    C: WaitClock,
    F: FnMut() -> Result<bool>,
{
    let start = clock.now();
    let mut seen = WaitSeen::default();
    let mut last_not_ready: Option<Duration> = None;
    loop {
        let done = ready()?;
        let now = clock.now();
        seen.polls = seen.polls.saturating_add(1);
        if done {
            seen.waited = now.saturating_sub(start);
            seen.last_gap = last_not_ready.map_or(Duration::ZERO, |t| now.saturating_sub(t));
            return Ok(seen);
        }
        last_not_ready = Some(now);
        let waited = now.saturating_sub(start);
        if waited > window.deadline {
            return Err(Error::Timeout {
                what: what.to_owned(),
                seconds: window.deadline.as_secs(),
            });
        }
        match aim {
            Some(a) => {
                let wake = a.expect.saturating_sub(a.guard);
                if !seen.slept_to_aim && waited < wake {
                    // THE ONE COARSE SLEEP, and only once: a second would be the window again.
                    seen.slept_to_aim = true;
                    clock.sleep(wake - waited);
                } else if waited < a.expect + a.spin_cap {
                    clock.spin();
                } else {
                    clock.sleep(window.sleep);
                }
            }
            None => {
                if waited < window.spin {
                    clock.spin();
                } else {
                    clock.sleep(window.sleep);
                }
            }
        }
    }
}

/// **THE HOST-WAIT ARM** (FN57, both engines): how a step's host thread waits for the card.
///
/// * `Spin` -- the HIP runtime's active wait (`hipEventSynchronize` under the default device
///   schedule): the waiting HIP thread busy-polls the completion signal for the whole step. The
///   Flash-Next engine's sleeper before FN57.
/// * `Aimed` -- ws-h32's law ([`WaitAim`]): one coarse sleep to the site's expected finish minus a
///   guard, then a bounded poll, then the window's sleeps. The 27B's sleeper before FN57.
/// * `Block` -- the interrupt wait: `hipSetDeviceFlags(hipDeviceScheduleBlockingSync)` on the rank's
///   device at boot (every runtime wait -- event, stream, synchronous copy -- sleeps in the kernel on
///   the signal's interrupt after a short poll) and the step's wait is one `hipEventSynchronize`
///   bounded by the caller's deadline ([`saber_hip::HipThread::sync_event_within`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostWait {
    /// The runtime's active wait.
    Spin,
    /// ws-h32's aimed sleep-then-poll.
    Aimed,
    /// The interrupt wait.
    Block,
}

impl std::str::FromStr for HostWait {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.trim() {
            "spin" => Ok(HostWait::Spin),
            "aimed" => Ok(HostWait::Aimed),
            "block" => Ok(HostWait::Block),
            t => Err(format!("--host-wait spin | aimed | block, not {t}")),
        }
    }
}

impl std::fmt::Display for HostWait {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            HostWait::Spin => "spin",
            HostWait::Aimed => "aimed",
            HostWait::Block => "block",
        })
    }
}

impl HostWait {
    /// The device schedule this arm sets at boot (`None` = leave the runtime's default).
    #[must_use]
    pub fn schedule(self) -> Option<saber_hip::DeviceSchedule> {
        match self {
            HostWait::Block => Some(saber_hip::DeviceSchedule::BlockingSync),
            HostWait::Spin => Some(saber_hip::DeviceSchedule::Spin),
            HostWait::Aimed => None,
        }
    }
}

/// The aim's memory for a sleeper outside the 27B's `StepWaits` (FN57, the Flash-Next worker): per
/// key, the last [`AimMemory::HISTORY`] waits, and the aim = their MINIMUM (a minimum can only make
/// a wait wake early, never sleep past the card -- [`WaitAim`]'s law).
#[derive(Debug, Default, Clone)]
pub struct AimMemory {
    seen: Vec<(u64, std::collections::VecDeque<Duration>)>,
}

impl AimMemory {
    /// How many recent waits an aim is the minimum of.
    pub const HISTORY: usize = 8;

    /// The aim for the next wait at `key`, if one has been seen.
    #[must_use]
    pub fn aim(&self, key: u64) -> Option<WaitAim> {
        let (_, ring) = self.seen.iter().find(|(k, _)| *k == key)?;
        ring.iter().min().copied().map(WaitAim::at)
    }

    /// Remember a wait at `key` (a wait whose first poll saw the work done says nothing).
    pub fn observe(&mut self, key: u64, seen: &WaitSeen) {
        if seen.polls <= 1 {
            return;
        }
        let at = match self.seen.iter().position(|(k, _)| *k == key) {
            Some(i) => i,
            None => {
                self.seen.push((
                    key,
                    std::collections::VecDeque::with_capacity(Self::HISTORY),
                ));
                self.seen.len() - 1
            }
        };
        let ring = &mut self.seen[at].1;
        if ring.len() == Self::HISTORY {
            ring.pop_front();
        }
        ring.push_back(seen.waited);
    }
}

/// [`wait_bounded_on`] with an optional [`WaitAim`], returning what the wait saw.
///
/// # Errors
/// The deadline passing, or the stream query failing.
pub fn wait_aimed(
    hip: &HipThread,
    stream: &Stream,
    what: &str,
    window: WaitWindow,
    aim: Option<WaitAim>,
) -> Result<WaitSeen> {
    if runtime_waits(hip) {
        return wait_runtime(hip, stream, what, window.deadline);
    }
    wait_polled(|| Ok(hip.stream_ready(stream)?), what, window, aim)
}

/// Agree with the peer on every collective in a bound step, **before** anything launches.
///
/// A graph cannot rendezvous, and neither can a `run_steps` loop that does not know which of its
/// ops talk to another rank. So the agreement happens once, here, over the ops that declare a
/// peer-visible workspace: each rank computes the launch digest — the descriptor plus the workspace
/// layout it is about to use — and [`RankPeer::agree`] refuses if they differ. That is the only
/// moment at which a mismatch is still a message rather than a kernel spinning on a flag that will
/// never come.
///
/// Returns how many collectives were agreed.
///
/// # Errors
/// A digest the peer does not share, or the exchange failing.
pub fn agree_on_collectives(
    launcher: &Launcher,
    peer: &Arc<RankPeer>,
    bound: &[BoundStep<'_>],
) -> Result<usize> {
    let mut n = 0usize;
    for step in bound {
        for desc in step.descs {
            if !desc
                .workspaces
                .iter()
                .any(|w| w.kind == WorkspaceKind::IpcPeerVisible)
            {
                continue;
            }
            peer.agree(step.name, launcher.collective_digest(desc, &step.args))?;
            n += 1;
        }
    }
    peer.rendezvous("collectives-agreed")?;
    Ok(n)
}

/// Every collective in a bound step: the ops that declare a peer-visible workspace.
#[must_use]
pub fn collective_steps<'a, 'b>(bound: &'a [BoundStep<'b>]) -> Vec<&'a BoundStep<'b>> {
    bound
        .iter()
        .filter(|s| {
            s.descs.iter().any(|d| {
                d.workspaces
                    .iter()
                    .any(|w| w.kind == WorkspaceKind::IpcPeerVisible)
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// THE VIRTUAL CLOCK (FN57's hygiene item: the sleeper laws used to be held on the wall clock and
    /// flaked under host load). Time moves only when the sleeper acts: a poll costs `POLL`, a spin
    /// pause `SPIN`, a sleep exactly what was asked plus `SLACK` (Linux's default 50 µs timer
    /// slack, the lateness every real sleep has) -- so every law below is exact and repeatable.
    struct Virtual {
        t: Cell<Duration>,
    }

    const POLL: Duration = Duration::from_nanos(800);
    const SPIN: Duration = Duration::from_nanos(200);
    const SLACK: Duration = Duration::from_micros(50);

    impl Virtual {
        fn new() -> Virtual {
            Virtual {
                t: Cell::new(Duration::ZERO),
            }
        }
        fn advance(&self, d: Duration) {
            self.t.set(self.t.get() + d);
        }
    }

    impl WaitClock for Virtual {
        fn now(&self) -> Duration {
            self.t.get()
        }
        fn sleep(&self, d: Duration) {
            self.advance(d + SLACK);
        }
        fn spin(&self) {
            self.advance(SPIN);
        }
    }

    /// A card with a KNOWN completion instant on the virtual clock: the probe (one poll's cost)
    /// says "done" from `done_after` on. Returns the overshoot past the card and what the wait saw.
    fn overshoot(done_after: Duration, aim: Option<WaitAim>) -> (Duration, WaitSeen) {
        let clock = Virtual::new();
        let seen = wait_polled_on(
            &clock,
            || {
                clock.advance(POLL);
                Ok(clock.now() >= done_after)
            },
            "mock",
            WaitWindow::STEP,
            aim,
        )
        .expect("a mock that completes cannot time out");
        (clock.now().saturating_sub(done_after), seen)
    }

    fn median(mut xs: Vec<Duration>) -> Duration {
        xs.sort_unstable();
        xs[xs.len() / 2]
    }

    /// Sixteen completion instants from 150 µs before the aim to 300 µs after it, 30 µs apart —
    /// one STEP quantum and a half of spread, so neither arm can be lucky about its phase.
    fn completions(expect: Duration) -> impl Iterator<Item = Duration> {
        (0..16u64).map(move |k| expect - Duration::from_micros(150) + Duration::from_micros(30 * k))
    }

    fn median_overshoot(aim: bool) -> Duration {
        let expect = Duration::from_millis(4);
        let mut over = Vec::new();
        for done in completions(expect) {
            let (o, seen) = overshoot(done, aim.then(|| WaitAim::at(expect)));
            if aim {
                assert!(seen.slept_to_aim, "the aim's coarse sleep must run");
            }
            over.push(o);
        }
        median(over)
    }

    /// THE HOST LAW: a card finishing anywhere from a little before to a little after the aim is
    /// seen within one poll of it — on the virtual clock the median overshoot is under 2 µs (one
    /// poll + one spin pause), exactly, every run.
    #[test]
    fn the_aimed_wait_never_sleeps_past_the_card() {
        let m = median_overshoot(true);
        assert!(
            m < Duration::from_micros(2),
            "the aimed wait overshot the card by a median {m:?}"
        );
    }

    /// THE NEGATIVE CONTROL: the same sixteen completions under `WaitWindow::STEP` (no aim) sit a
    /// sleep quantum's worth past the card — the law above fails for it, which is the point.
    #[test]
    fn the_step_window_overshoots_by_a_quantum() {
        let m = median_overshoot(false);
        assert!(
            m > Duration::from_micros(50),
            "the control must overshoot (median {m:?}); if it does not, the law above proves nothing"
        );
    }

    /// A card that runs LONG past the aim (a stalled peer) falls back to the window's sleeps after
    /// the spin cap: it still completes, within one sleep (+ its slack) of the card. On the
    /// virtual clock this is exact (FN57: it asserted a wall-clock wait and flaked under load).
    #[test]
    fn a_late_card_falls_back_to_the_window() {
        let (o, seen) = overshoot(
            Duration::from_millis(5),
            Some(WaitAim::at(Duration::from_millis(1))),
        );
        assert!(
            o <= WaitWindow::STEP.sleep + SLACK + POLL,
            "overshoot {o:?} past one window sleep"
        );
        assert!(seen.waited >= Duration::from_millis(5));
        assert!(seen.last_gap > Duration::ZERO);
        assert!(seen.slept_to_aim);
    }

    /// Work that is already done costs one query and no sleep, aimed or not.
    #[test]
    fn done_work_is_one_poll() {
        for aim in [None, Some(WaitAim::at(Duration::from_millis(10)))] {
            let clock = Virtual::new();
            let seen = wait_polled_on(&clock, || Ok(true), "done", WaitWindow::STEP, aim).unwrap();
            assert_eq!(seen.polls, 1);
            assert!(!seen.slept_to_aim);
            assert_eq!(seen.last_gap, Duration::ZERO);
            assert_eq!(clock.now(), Duration::ZERO);
        }
    }

    /// The deadline still binds under an aim: a card that never finishes is a named timeout.
    #[test]
    fn the_deadline_binds_under_an_aim() {
        let w = WaitWindow {
            spin: Duration::from_micros(10),
            sleep: Duration::from_micros(100),
            deadline: Duration::from_millis(3),
        };
        let clock = Virtual::new();
        let r = wait_polled_on(
            &clock,
            || {
                clock.advance(POLL);
                Ok(false)
            },
            "never",
            w,
            Some(WaitAim::at(Duration::from_millis(1))),
        );
        assert!(matches!(r, Err(Error::Timeout { .. })), "{r:?}");
        assert!(clock.now() < Duration::from_millis(4));
    }

    /// The real clock still runs the same loop: done work is one poll on the wall clock too.
    #[test]
    fn the_real_clock_runs_the_same_loop() {
        let seen = wait_polled(|| Ok(true), "done", WaitWindow::STEP, None).unwrap();
        assert_eq!(seen.polls, 1);
    }

    /// FN57: the aim memory takes the MINIMUM of the recent waits at a key, forgets the oldest past
    /// its history, and ignores a wait whose first poll saw the work done.
    #[test]
    fn the_aim_memory_is_the_minimum_of_recent_waits() {
        let mut m = AimMemory::default();
        assert!(m.aim(7).is_none());
        let w = |ms: u64, polls: u32| WaitSeen {
            waited: Duration::from_millis(ms),
            last_gap: Duration::ZERO,
            polls,
            slept_to_aim: false,
        };
        m.observe(7, &w(1, 1));
        assert!(m.aim(7).is_none(), "a one-poll wait says nothing");
        m.observe(7, &w(20, 5));
        m.observe(7, &w(18, 5));
        m.observe(9, &w(3, 5));
        assert_eq!(m.aim(7).unwrap().expect, Duration::from_millis(18));
        assert_eq!(m.aim(9).unwrap().expect, Duration::from_millis(3));
        for _ in 0..AimMemory::HISTORY {
            m.observe(7, &w(25, 5));
        }
        assert_eq!(m.aim(7).unwrap().expect, Duration::from_millis(25));
    }

    /// FN57: the arm parses by name, prints its name, and only `block` moves the device schedule.
    #[test]
    fn the_host_wait_arm_parses_and_names_its_schedule() {
        for (t, a) in [
            ("spin", HostWait::Spin),
            ("aimed", HostWait::Aimed),
            ("block", HostWait::Block),
        ] {
            assert_eq!(t.parse::<HostWait>().unwrap(), a);
            assert_eq!(a.to_string(), t);
        }
        assert!("poll".parse::<HostWait>().is_err());
        assert_eq!(
            HostWait::Block.schedule(),
            Some(saber_hip::DeviceSchedule::BlockingSync)
        );
        assert_eq!(
            HostWait::Spin.schedule(),
            Some(saber_hip::DeviceSchedule::Spin)
        );
        assert_eq!(HostWait::Aimed.schedule(), None);
    }
}
