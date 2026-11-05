//! Stopping the engine, against a deadline.
//!
//! A serving process holds two things nobody else can use while it lives: the shared GPU lock and
//! the card's memory. So the interesting property of a shutdown is not that it is tidy — it is that
//! it is **bounded**. On 2026-09-19 a `saber serve --engine real` inside a gate script took a
//! signal, printed its result, and then sat in `wait` for **2 h 55 m** with 18.7 GiB of a card held
//! and the lock taken, because the drop of its engine never returned.
//!
//! [`stop_with_deadline`] is the answer, and these are its two halves: it returns promptly when the
//! teardown is healthy, and it **returns an error rather than hanging** when the teardown is not.
//! The second is the one worth having, so it is a real stuck drop and not a mock of one.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{request, scheduler, EOS};
use saber_sched::{spawn_step_loop, stop_with_deadline, MockExecutor, SeqScript};
use saber_serve::engine::Engine;

/// A payload whose drop takes `for_` — a stand-in for a HIP thread that will not join.
struct SlowDrop {
    for_: Duration,
    dropped: Arc<AtomicBool>,
}

impl Drop for SlowDrop {
    fn drop(&mut self) {
        std::thread::sleep(self.for_);
        self.dropped.store(true, Ordering::SeqCst);
    }
}

#[test]
fn a_healthy_teardown_returns_the_steps_it_ran() {
    let mut exec = MockExecutor::new();
    exec.push_script(SeqScript::per_token([10, 11, 12, EOS]));
    let sched = Arc::new(scheduler(exec));
    let sub = sched.submit(request(&[1, 2, 3], 64)).expect("submit");
    let steps = spawn_step_loop(Arc::clone(&sched));
    // Let the request finish so there is something to count.
    let deadline = Instant::now() + Duration::from_secs(5);
    while sched.counters().steps < 4 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(2));
    }
    drop(sub);

    let t0 = Instant::now();
    let ran = stop_with_deadline(sched, steps, Duration::from_secs(5)).expect("a healthy teardown");
    let took = t0.elapsed();
    assert!(ran >= 4, "the loop ran {ran} steps");
    assert!(
        took < Duration::from_secs(1),
        "a healthy teardown took {took:?}"
    );
}

/// NEGATIVE CONTROL, and the reason this function exists: a teardown that will not finish is
/// **reported**, not waited on. Without this the deadline is a parameter nobody has seen used.
#[test]
fn a_teardown_that_will_not_finish_is_reported_rather_than_waited_on() {
    let exec = MockExecutor::new();
    let sched = Arc::new(scheduler(exec));
    let steps = spawn_step_loop(Arc::clone(&sched));
    let dropped = Arc::new(AtomicBool::new(false));
    let stuck = SlowDrop {
        // Far longer than the deadline below, and longer than this test's own runtime: if
        // `stop_with_deadline` waited for it, the test would be the hang it is asserting about.
        for_: Duration::from_secs(30),
        dropped: Arc::clone(&dropped),
    };

    let t0 = Instant::now();
    let err = stop_with_deadline((sched, stuck), steps, Duration::from_millis(200))
        .expect_err("the drop cannot finish inside the deadline");
    let took = t0.elapsed();

    assert!(
        took < Duration::from_secs(2),
        "the call itself waited {took:?} on a drop that takes 30 s"
    );
    assert_eq!(err.deadline, Duration::from_millis(200));
    assert!(
        err.to_string().contains("did not shut down"),
        "the message must say what is stuck: {err}"
    );
    assert!(
        !dropped.load(Ordering::SeqCst),
        "the drop is still running, which is the point: the caller leaves without it"
    );
    // The teardown thread is detached and still sleeping; the test process may exit over it, which
    // is exactly what the binary does with `abort`.
}
