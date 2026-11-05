//! FN59: the HIP thread's BACKGROUND LANE -- a non-blocking batch submit.
//!
//! [`crate::HipThread::run_batch`] blocks the caller until every job of the batch has been ISSUED
//! (not completed) on the HIP thread. For the expert tier's end-of-step copy batch (~45 pinned
//! 2.6 MB copies a b1 verify step) that issue is host time on the rank thread's critical path
//! (FN56: `tier_end` 1.4-1.7 ms exposed). [`crate::HipThread::submit_batch`] posts the same lowered
//! batch to the SAME thread (the affinity law: every HIP call on the owning thread) and returns a
//! [`BatchTicket`] at once. The HIP thread issues the background jobs one at a time WHENEVER ITS
//! CHANNEL IS EMPTY, so an ordinary call posted later may run between two background copies --
//! that is the point: the rank thread's next calls are not queued behind ~1 ms of copy issue.
//!
//! **What may overtake, and what may not** (the law the worker enforces, by the call's [`Deps`]):
//! * `Deps::All` (every call that does not declare its names -- the default of `run`, every
//!   destructor, every host-blocking sync, every allocation): the lane is DRAINED first, so the
//!   call sees exactly the order it would have seen had the batch been issued by `run_batch`.
//! * `Deps::On(names)` (launches, records, stream waits, async copies, queries): the lane is
//!   drained THROUGH the last pending job that touches one of those streams or events, then the
//!   call runs. A call on a stream and events the batch never names overtakes it; anything that
//!   could observe the batch's order (a wait on its landed event, more work on its copy stream, a
//!   re-record of an event it waits on) does not.
//! * Background batches are issued FIFO among themselves, each in its own order.
//! * A capture pauses the lane (`BeginCapture` drains it first); a batch submitted while a
//!   capture is open is refused by name, never issued into a capture.
//! * Shutdown (and a disconnected channel) DRAINS the lane before the thread exits: no submitted
//!   copy is lost.
//!
//! Handle lifetime: a background job holds only `Send` integer tokens. Every handle's destructor
//! is a `Deps::All` call posted on this same channel, so it cannot run before the batch that names
//! the handle has been issued -- the same guarantee `run_batch` gives, by FIFO instead of by
//! blocking. (The caller's law: every handle in a submitted batch belongs to THIS `HipThread`.)

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError};

use crate::error::{Error, Result};
use crate::handles::{RawEvent, RawStream};
use crate::thread::{Ctx, Msg, RawJob};

/// A stream or event a job or call names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Name {
    Stream(RawStream),
    Event(RawEvent),
}

/// What a call posted to the HIP thread may observe of the background lane.
#[derive(Debug)]
pub(crate) enum Deps {
    /// Opaque or host-blocking: drain the lane first.
    All,
    /// Up to two names (no allocation on the hot path).
    On([Option<Name>; 2]),
    /// A batch's names.
    Many(Vec<Name>),
    /// `hipStreamBeginCapture`: drain, then pause the lane.
    BeginCapture,
    /// `hipStreamEndCapture`: resume the lane after the call.
    EndCapture,
}

impl Deps {
    pub(crate) fn on1(a: Name) -> Deps {
        Deps::On([Some(a), None])
    }
    pub(crate) fn on2(a: Name, b: Name) -> Deps {
        Deps::On([Some(a), Some(b)])
    }
    fn names(&self) -> &[Option<Name>] {
        match self {
            Deps::On(n) => n,
            _ => &[],
        }
    }
    fn touches(&self, n: Name) -> bool {
        match self {
            Deps::On(_) => self.names().contains(&Some(n)),
            Deps::Many(v) => v.contains(&n),
            _ => false,
        }
    }
}

/// A batch on its way to the lane.
pub(crate) struct Background {
    pub(crate) jobs: Vec<RawJob>,
    pub(crate) done: SyncSender<Result<usize>>,
}

struct Pending {
    jobs: Vec<RawJob>,
    next: usize,
    done: SyncSender<Result<usize>>,
}

/// The lane: FIFO of submitted batches, issued job by job.
#[derive(Default)]
pub(crate) struct Lane {
    q: VecDeque<Pending>,
    paused: bool,
}

impl Lane {
    fn runnable(&self) -> bool {
        !self.paused && !self.q.is_empty()
    }

    fn push(&mut self, b: Background) {
        if self.paused {
            let _ = b.done.send(Err(Error::Invalid(format!(
                "a background batch of {} jobs was submitted while a stream capture is open; \
                 refused (copies are never issued into a capture)",
                b.jobs.len()
            ))));
            return;
        }
        if b.jobs.is_empty() {
            let _ = b.done.send(Ok(0));
            return;
        }
        self.q.push_back(Pending {
            jobs: b.jobs,
            next: 0,
            done: b.done,
        });
    }

    /// Issue the front batch's next job; retire the batch when it is done or has failed.
    fn step(&mut self, issue: &mut dyn FnMut(RawJob) -> Result<()>) {
        let Some(p) = self.q.front_mut() else { return };
        let i = p.next;
        let n = p.jobs.len();
        match issue(p.jobs[i]) {
            Ok(()) => {
                p.next += 1;
                if p.next == n {
                    let p = self.q.pop_front().expect("front");
                    let _ = p.done.send(Ok(n));
                }
            }
            Err(e) => {
                tracing::error!(
                    job = i,
                    batch = n,
                    error = %e,
                    "HIP background batch failed; the jobs after this one were not issued"
                );
                let p = self.q.pop_front().expect("front");
                let _ = p.done.send(Err(e));
            }
        }
    }

    /// Number of pending jobs (across batches, in issue order) through the last one `deps` touches.
    fn through(&self, deps: &Deps) -> usize {
        let mut k = 0;
        let mut last = 0;
        for p in &self.q {
            for j in &p.jobs[p.next..] {
                k += 1;
                let (a, b) = j.names();
                if a.is_some_and(|n| deps.touches(n)) || b.is_some_and(|n| deps.touches(n)) {
                    last = k;
                }
            }
        }
        last
    }

    /// Issue the first `k` pending jobs (a batch that fails retires all its remaining jobs).
    fn drain_jobs(&mut self, mut k: usize, issue: &mut dyn FnMut(RawJob) -> Result<()>) {
        while k > 0 {
            let Some(p) = self.q.front() else { return };
            let left = p.jobs.len() - p.next;
            let before = self.q.len();
            self.step(issue);
            let failed = self.q.len() < before && left > 1;
            k = k.saturating_sub(if failed { left } else { 1 });
        }
    }

    fn drain_all(&mut self, issue: &mut dyn FnMut(RawJob) -> Result<()>) {
        while !self.q.is_empty() {
            self.step(issue);
        }
    }

    #[cfg(test)]
    pub(crate) fn pending_jobs(&self) -> usize {
        self.q.iter().map(|p| p.jobs.len() - p.next).sum()
    }
}

/// The HIP thread's loop, generic over the runtime so the lane's law is testable on a host with no
/// ROCm (the tests drive it with a recording fake).
pub(crate) fn serve<L, F>(rx: &Receiver<Msg<L>>, lib: &L, ctx: &Ctx, exec: F)
where
    F: Fn(RawJob, &L, &Ctx) -> Result<()>,
{
    let mut lane = Lane::default();
    let mut issue = |j: RawJob| exec(j, lib, ctx);
    loop {
        let msg = if lane.runnable() {
            match rx.try_recv() {
                Ok(m) => m,
                Err(TryRecvError::Empty) => {
                    lane.step(&mut issue);
                    continue;
                }
                Err(TryRecvError::Disconnected) => break,
            }
        } else {
            match rx.recv() {
                Ok(m) => m,
                Err(_) => break,
            }
        };
        match msg {
            Msg::Call(task, deps) => {
                match deps {
                    Deps::All | Deps::BeginCapture => lane.drain_all(&mut issue),
                    Deps::On(_) | Deps::Many(_) => {
                        let k = lane.through(&deps);
                        lane.drain_jobs(k, &mut issue);
                    }
                    Deps::EndCapture => {}
                }
                task(lib, ctx);
                match deps {
                    Deps::BeginCapture => lane.paused = true,
                    Deps::EndCapture => lane.paused = false,
                    _ => {}
                }
            }
            Msg::Background(b) => lane.push(b),
            Msg::Shutdown => break,
        }
    }
    // THE DRAIN LAW: nothing submitted is lost at exit.
    lane.paused = false;
    lane.drain_all(&mut issue);
}

/// The handle [`crate::HipThread::submit_batch`] returns.
///
/// It tells the HOST when the batch has been issued (or failed, by name). The DEVICE-side wait is
/// the batch's own `RecordEvent` job plus a `stream_wait_event` on it -- which names the event, so
/// the lane issues the record before the wait runs, whenever the wait is posted.
#[must_use = "a submitted batch's failure is reported only through its ticket (and the log)"]
#[derive(Debug)]
pub struct BatchTicket {
    rx: Receiver<Result<usize>>,
    state: Option<Result<usize>>,
    jobs: usize,
}

impl BatchTicket {
    pub(crate) fn new(rx: Receiver<Result<usize>>, jobs: usize) -> BatchTicket {
        BatchTicket {
            rx,
            state: None,
            jobs,
        }
    }

    /// Jobs in the batch.
    #[must_use]
    pub fn jobs(&self) -> usize {
        self.jobs
    }

    /// Non-blocking: has every job been issued (or has the batch failed)?
    pub fn is_issued(&mut self) -> bool {
        if self.state.is_none() {
            match self.rx.try_recv() {
                Ok(r) => self.state = Some(r),
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => self.state = Some(Err(Error::ThreadGone)),
            }
        }
        self.state.is_some()
    }

    /// Block until the batch is issued; the jobs issued, or the first failure (by name), or
    /// `ThreadGone` if the HIP thread died first -- never a hang.
    pub fn wait(mut self) -> Result<usize> {
        if let Some(r) = self.state.take() {
            return r;
        }
        self.rx.recv().unwrap_or(Err(Error::ThreadGone))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;
    use std::sync::mpsc::{self, Sender};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use crate::ffi::MemcpyKind;

    /// A recording runtime: every issued job appends a label; a copy of 13 bytes fails.
    #[derive(Default)]
    struct Fake {
        log: Arc<Mutex<Vec<String>>>,
        delay: Duration,
    }

    fn label(j: RawJob) -> String {
        match j {
            RawJob::Memcpy { bytes, .. } => format!("copy{bytes}"),
            RawJob::RecordEvent { event, .. } => format!("rec{}", event.as_ptr() as usize),
            RawJob::StreamWaitEvent { event, .. } => format!("wait{}", event.as_ptr() as usize),
            other => format!("{other:?}"),
        }
    }

    fn exec(j: RawJob, f: &Fake, _c: &Ctx) -> Result<()> {
        std::thread::sleep(f.delay);
        if let RawJob::Memcpy { bytes: 13, .. } = j {
            return Err(Error::Invalid(
                "hipMemcpyAsync failed: injected (13 bytes)".into(),
            ));
        }
        f.log.lock().unwrap().push(label(j));
        Ok(())
    }

    struct Rig {
        tx: Sender<Msg<Fake>>,
        log: Arc<Mutex<Vec<String>>>,
        join: Option<std::thread::JoinHandle<()>>,
    }

    impl Rig {
        fn new(delay_ms: u64) -> Rig {
            let (tx, rx) = mpsc::channel::<Msg<Fake>>();
            let log = Arc::new(Mutex::new(Vec::new()));
            let l2 = log.clone();
            let join = std::thread::spawn(move || {
                let f = Fake {
                    log: l2,
                    delay: Duration::from_millis(delay_ms),
                };
                serve(&rx, &f, &Ctx::new_for_tests(), exec);
            });
            Rig {
                tx,
                log,
                join: Some(join),
            }
        }
        fn submit(&self, jobs: Vec<RawJob>) -> BatchTicket {
            crate::thread::post_background(&self.tx, jobs).expect("posted")
        }
        /// Post a call; returns the log as the call saw it.
        fn call(&self, deps: Deps, tag: &str) -> Vec<String> {
            let (t, r) = mpsc::sync_channel(1);
            let tag = tag.to_string();
            let log = self.log.clone();
            self.tx
                .send(Msg::Call(
                    Box::new(move |_f: &Fake, _c: &Ctx| {
                        let mut g = log.lock().unwrap();
                        g.push(tag);
                        let _ = t.send(g.clone());
                    }),
                    deps,
                ))
                .unwrap();
            r.recv().unwrap()
        }
        fn shutdown(&mut self) {
            let _ = self.tx.send(Msg::Shutdown);
            self.join.take().unwrap().join().unwrap();
        }
    }

    fn st(i: usize) -> RawStream {
        RawStream::from_raw(i as *mut c_void)
    }
    fn ev(i: usize) -> RawEvent {
        RawEvent::from_raw(i as *mut c_void)
    }
    fn copy(bytes: u64, s: usize) -> RawJob {
        RawJob::Memcpy {
            dst: 0x1000,
            src: 0x2000,
            bytes,
            kind: MemcpyKind::HostToDevice,
            stream: st(s),
        }
    }
    fn rec(e: usize, s: usize) -> RawJob {
        RawJob::RecordEvent {
            event: ev(e),
            stream: st(s),
        }
    }

    /// Batches issue in submit order, each in its own order, and an opaque call (`Deps::All` --
    /// a destructor, a host sync, an allocation) sees them ALL issued before it runs.
    #[test]
    fn batches_keep_their_order_and_an_opaque_call_drains_the_lane() {
        let mut rig = Rig::new(1);
        let a = rig.submit(vec![copy(1, 7), copy(2, 7), rec(100, 7)]);
        let b = rig.submit(vec![copy(3, 7), rec(101, 7)]);
        let seen = rig.call(Deps::All, "free");
        assert_eq!(
            seen,
            ["copy1", "copy2", "rec100", "copy3", "rec101", "free"],
            "a destructor / sync never overtakes a submitted copy"
        );
        assert_eq!(a.wait().unwrap(), 3);
        assert_eq!(b.wait().unwrap(), 2);
        rig.shutdown();
    }

    /// The point of the lane: a call on a stream the batch does not name OVERTAKES the pending
    /// copies; a wait on the batch's landed event drains THROUGH that record (never a wait on a
    /// stale record), and later jobs may still be pending after it.
    #[test]
    fn an_unrelated_call_overtakes_and_a_gate_wait_drains_through_its_record() {
        let mut rig = Rig::new(2);
        let mut jobs: Vec<RawJob> = (0..20).map(|i| copy(100 + i, 7)).collect();
        jobs.insert(5, rec(500, 7));
        let mut t = rig.submit(jobs);
        let seen = rig.call(Deps::on1(Name::Stream(st(9))), "launch");
        let copies_before = seen.iter().filter(|l| l.starts_with("copy")).count();
        assert!(
            copies_before < 20,
            "a launch on another stream waited for the whole batch ({copies_before} copies)"
        );
        let seen = rig.call(Deps::on2(Name::Stream(st(9)), Name::Event(ev(500))), "gate");
        let at_rec = seen
            .iter()
            .position(|l| l == "rec500")
            .expect("record issued");
        let at_gate = seen.iter().position(|l| l == "gate").unwrap();
        assert!(
            at_rec < at_gate,
            "the gate waited on a record not yet issued"
        );
        // more work on the COPY stream is ordered behind every pending copy on it
        let seen = rig.call(Deps::on1(Name::Stream(st(7))), "work7");
        assert_eq!(
            seen.iter().filter(|l| l.starts_with("copy")).count(),
            20,
            "work on the batch's own stream must not overtake it"
        );
        assert!(t.is_issued());
        assert_eq!(t.wait().unwrap(), 21);
        rig.shutdown();
    }

    /// The ticket: not issued while copies are pending, `wait` returns the job count.
    #[test]
    fn the_ticket_reports_issue_without_blocking_and_wait_returns_the_count() {
        let mut rig = Rig::new(5);
        let mut t = rig.submit((0..8).map(|i| copy(i + 1, 7)).collect());
        assert_eq!(t.jobs(), 8);
        assert!(!t.is_issued(), "40 ms of issue cannot be done already");
        assert_eq!(t.wait().unwrap(), 8);
        assert_eq!(rig.log.lock().unwrap().len(), 8);
        rig.shutdown();
    }

    /// THE DRAIN LAW: shutting the thread down with copies still pending issues every one of them
    /// before the thread exits.
    #[test]
    fn shutdown_drains_every_submitted_copy() {
        let mut rig = Rig::new(2);
        let t1 = rig.submit((0..10).map(|i| copy(i + 1, 7)).collect());
        let t2 = rig.submit(vec![copy(50, 8), rec(9, 8)]);
        rig.shutdown();
        assert_eq!(rig.log.lock().unwrap().len(), 12, "a copy was lost at exit");
        assert_eq!(t1.wait().unwrap(), 10);
        assert_eq!(t2.wait().unwrap(), 2);
    }

    /// A failed copy fails ITS batch by name (jobs after it not issued), the next batch still
    /// issues, and a ticket whose thread died reports `ThreadGone` instead of hanging.
    #[test]
    fn a_failed_copy_is_reported_by_name_and_the_lane_goes_on() {
        let mut rig = Rig::new(0);
        let bad = rig.submit(vec![copy(1, 7), copy(13, 7), copy(2, 7)]);
        let good = rig.submit(vec![copy(3, 7)]);
        let err = bad.wait().expect_err("the injected failure");
        assert!(err.to_string().contains("hipMemcpyAsync"), "{err}");
        assert_eq!(good.wait().unwrap(), 1);
        let seen = rig.call(Deps::All, "end");
        assert_eq!(seen, ["copy1", "copy3", "end"]);
        rig.shutdown();

        let (tx, rx) = mpsc::sync_channel::<Result<usize>>(1);
        drop(tx);
        let mut t = BatchTicket::new(rx, 3);
        assert!(t.is_issued());
        assert!(matches!(t.wait(), Err(Error::ThreadGone)));
    }

    /// A capture pauses the lane: it is drained first, and a batch submitted during the capture
    /// is refused by name (never issued into it).
    #[test]
    fn a_capture_drains_then_pauses_the_lane() {
        let mut rig = Rig::new(1);
        let t = rig.submit(vec![copy(1, 7), copy(2, 7)]);
        let seen = rig.call(Deps::BeginCapture, "begin");
        assert_eq!(seen, ["copy1", "copy2", "begin"]);
        assert_eq!(t.wait().unwrap(), 2);
        let refused = rig.submit(vec![copy(3, 7)]);
        let err = refused.wait().expect_err("refused during capture");
        assert!(err.to_string().contains("capture"), "{err}");
        rig.call(Deps::EndCapture, "end");
        let ok = rig.submit(vec![copy(4, 7)]);
        assert_eq!(ok.wait().unwrap(), 1);
        rig.shutdown();
    }

    #[test]
    fn through_counts_to_the_last_touching_job() {
        let mut lane = Lane::default();
        let (t, _r) = mpsc::sync_channel(1);
        lane.push(Background {
            jobs: vec![copy(1, 7), rec(5, 7), copy(2, 7), copy(3, 7)],
            done: t,
        });
        assert_eq!(lane.pending_jobs(), 4);
        assert_eq!(lane.through(&Deps::on1(Name::Event(ev(5)))), 2);
        assert_eq!(lane.through(&Deps::on1(Name::Stream(st(7)))), 4);
        assert_eq!(lane.through(&Deps::on1(Name::Stream(st(8)))), 0);
        assert_eq!(lane.through(&Deps::Many(vec![Name::Event(ev(6))])), 0);
    }
}
