//! The engine's shutdown census, live on `/metrics` (ws-v8).
//!
//! Every which-path counter the engine, its drafter and the scheduler keep used to reach the
//! outside world ONCE, as the lines `saber serve --engine real` prints after the listener closes
//! (`docs/SHADOW.md` §2). A shadow window that crashed lost its census, and a mid-window fallback
//! could not be dated. This module is the transport that puts them on `/metrics` while the server
//! runs, under the `saber:` family, without the scrape ever waiting on the device:
//!
//! * [`CensusDef`] names one counter: its exposition name, its help text, whether it is a counter
//!   or a gauge, and **the census line and label it mirrors** (`"counters: eager {}"`), so the
//!   doc table and the parity test read the same statement of the mapping;
//! * [`CensusMirror`] is a fixed array of atomics behind a sequence lock. The engine thread
//!   PUBLISHES into it at the end of a step (it already holds its own state; the publish is a
//!   store per counter), and the scrape READS it without taking any lock the step path holds;
//! * the mirror is **disarmed** until the first scrape: [`CensusMirror::publish`] is one relaxed
//!   load and a return on a server nobody scrapes, which is the "zero cost when nobody scrapes"
//!   half of the contract. The first scrape arms it and waits (bounded) for the first publish.
//!
//! The engine-specific definitions live with the counters they read (`saber-engine`'s
//! `EngineCounters::CENSUS`, `saber-sched`'s `SchedCounters::CENSUS`); this crate only carries
//! them to the renderer, because it is the one crate both of those depend on.

use std::sync::atomic::{fence, AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

/// Prometheus type of one census entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CensusKind {
    /// Monotone over the process's life.
    Counter,
    /// A level (`saber:engine_clean`).
    Gauge,
}

impl CensusKind {
    /// The `# TYPE` word.
    pub fn word(self) -> &'static str {
        match self {
            Self::Counter => "counter",
            Self::Gauge => "gauge",
        }
    }
}

/// One census counter: what it is called on `/metrics`, and what it mirrors at shutdown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CensusDef {
    /// The exposition name, `saber:` prefix included (`saber:engine_eager_steps_total`).
    pub name: &'static str,
    /// The `# HELP` text.
    pub help: &'static str,
    /// Counter or gauge.
    pub kind: CensusKind,
    /// The shutdown census line this mirrors, as `"<line tag>: <pattern>"` where the pattern is
    /// the line's own text around the number, `{}` standing for it (`"counters: eager {}"`,
    /// `"parking: refused {},"`). The parity test finds the number in the printed line by this
    /// pattern, so a census line and its `/metrics` twin cannot drift apart silently.
    pub census: &'static str,
}

impl CensusDef {
    /// A counter.
    pub const fn counter(name: &'static str, help: &'static str, census: &'static str) -> Self {
        Self {
            name,
            help,
            kind: CensusKind::Counter,
            census,
        }
    }

    /// A gauge.
    pub const fn gauge(name: &'static str, help: &'static str, census: &'static str) -> Self {
        Self {
            name,
            help,
            kind: CensusKind::Gauge,
            census,
        }
    }

    /// The census line's tag (`"counters"`), the text before the first `": "`.
    pub fn line_tag(&self) -> &'static str {
        self.census.split_once(": ").map_or(self.census, |(t, _)| t)
    }

    /// The pattern inside the line (`"eager {}"`).
    pub fn pattern(&self) -> &'static str {
        self.census.split_once(": ").map_or("", |(_, p)| p)
    }
}

/// One value, with its definition, as the renderer consumes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CensusSample {
    /// What it is.
    pub def: CensusDef,
    /// Its value at the snapshot.
    pub value: u64,
}

/// Finds the number a [`CensusDef::census`] pattern names inside a printed census.
///
/// `census` is the whole shutdown output (or any text holding its lines); the line is the LAST
/// one whose first word is the def's tag — a server's log carries boot lines under the same tags
/// (`parking        OFF …`, `session   4 sequence slot(s) …`) and the census is printed last, and the number is the one that sits where the pattern's
/// `{}` does. `None` when the line or the pattern is missing — which is itself a parity failure.
pub fn find_in_census(census: &str, def: &CensusDef) -> Option<u64> {
    let tag = def.line_tag();
    let (before, after) = def.pattern().split_once("{}")?;
    let line = census
        .lines()
        .rev()
        .find(|l| l.split_whitespace().next() == Some(tag))?;
    let bytes = line.as_bytes();
    let mut from = 0;
    while let Some(off) = line[from..].find(before) {
        let at = from + off;
        let start = at + before.len();
        let digits = line[start..].bytes().take_while(u8::is_ascii_digit).count();
        // A pattern that is the tail of a longer word (`"ager {}"` inside `"eager 3"`) is not a
        // match: the character before it must not continue a word.
        let boundary = at == 0
            || before
                .as_bytes()
                .first()
                .is_none_or(|b| !b.is_ascii_alphanumeric())
            || !bytes[at - 1].is_ascii_alphanumeric();
        if digits > 0 && boundary && line[start + digits..].starts_with(after) {
            return line[start..start + digits].parse().ok();
        }
        from = at + before.len().max(1);
        if from >= line.len() {
            break;
        }
    }
    None
}

/// Pairs `defs` with the values `visit` writes, in order — a snapshot taken under a lock the caller
/// already holds (the scheduler's own counters, read at scrape time under its core lock).
///
/// # Panics
///
/// When `visit` writes a different number of values than there are defs.
pub fn collect(defs: &[CensusDef], visit: impl FnOnce(&mut dyn FnMut(u64))) -> Vec<CensusSample> {
    let mut values = Vec::with_capacity(defs.len());
    visit(&mut |v| values.push(v));
    assert_eq!(
        values.len(),
        defs.len(),
        "census: {} values for {} definitions",
        values.len(),
        defs.len()
    );
    defs.iter()
        .zip(values)
        .map(|(def, value)| CensusSample { def: *def, value })
        .collect()
}

/// A published snapshot of a fixed list of counters, readable without a lock.
///
/// One writer at a time (the engine thread, under the executor's own lock) and any number of
/// readers. A sequence lock: the writer makes the sequence odd, stores every value, and makes it
/// even again; a reader retries while it is odd or changed underneath, so it never returns a torn
/// snapshot — `saber:engine_clean` and the counters it is computed from are always one step's.
#[derive(Debug)]
pub struct CensusMirror {
    defs: Vec<CensusDef>,
    values: Box<[AtomicU64]>,
    seq: AtomicU64,
    armed: AtomicBool,
    /// Wakes a first scrape that is waiting for the first publish.
    first: (Mutex<bool>, Condvar),
}

impl CensusMirror {
    /// A disarmed mirror over `defs`, with nothing published.
    pub fn new(defs: Vec<CensusDef>) -> Self {
        let values = defs.iter().map(|_| AtomicU64::new(0)).collect();
        Self {
            defs,
            values,
            seq: AtomicU64::new(0),
            armed: AtomicBool::new(false),
            first: (Mutex::new(false), Condvar::new()),
        }
    }

    /// The definitions, in publish order.
    pub fn defs(&self) -> &[CensusDef] {
        &self.defs
    }

    /// Whether a scrape has asked for the census yet.
    pub fn armed(&self) -> bool {
        self.armed.load(Ordering::Relaxed)
    }

    /// Asks the engine to publish from now on.
    pub fn arm(&self) {
        self.armed.store(true, Ordering::Relaxed);
    }

    /// Whether anything has been published yet.
    pub fn published(&self) -> bool {
        self.seq.load(Ordering::Acquire) != 0
    }

    /// Publishes one snapshot: `fill` is handed a writer and must call it once per def, in order.
    /// A no-op (one relaxed load) while disarmed.
    ///
    /// # Panics
    ///
    /// When `fill` writes a different number of values than there are defs — a census whose
    /// definitions and values disagree is a programming error, and it is caught by the first test
    /// that publishes.
    pub fn publish(&self, fill: impl FnOnce(&mut dyn FnMut(u64))) {
        if !self.armed() {
            return;
        }
        let s = self.seq.load(Ordering::Relaxed);
        self.seq.store(s | 1, Ordering::Relaxed);
        fence(Ordering::Release);
        let mut i = 0usize;
        fill(&mut |v| {
            if let Some(slot) = self.values.get(i) {
                slot.store(v, Ordering::Relaxed);
            }
            i += 1;
        });
        // Even and strictly larger than any value a reader could have seen before.
        self.seq.store((s | 1).wrapping_add(1), Ordering::Release);
        assert_eq!(
            i,
            self.defs.len(),
            "census: {} values published for {} definitions",
            i,
            self.defs.len()
        );
        if s == 0 {
            let (lock, cv) = &self.first;
            *lock.lock().unwrap_or_else(|e| e.into_inner()) = true;
            cv.notify_all();
        }
    }

    /// The last published snapshot, or `None` when nothing has been published.
    pub fn read(&self) -> Option<Vec<CensusSample>> {
        loop {
            let s1 = self.seq.load(Ordering::Acquire);
            if s1 == 0 {
                return None;
            }
            if s1 & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let out: Vec<CensusSample> = self
                .defs
                .iter()
                .zip(self.values.iter())
                .map(|(def, v)| CensusSample {
                    def: *def,
                    value: v.load(Ordering::Relaxed),
                })
                .collect();
            fence(Ordering::Acquire);
            if self.seq.load(Ordering::Relaxed) == s1 {
                return Some(out);
            }
        }
    }

    /// Waits up to `deadline` for the first publish. `true` when there is one.
    pub fn wait_first(&self, deadline: Duration) -> bool {
        if self.published() {
            return true;
        }
        let (lock, cv) = &self.first;
        let guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        let (guard, _) = cv
            .wait_timeout_while(guard, deadline, |done| !*done)
            .unwrap_or_else(|e| e.into_inner());
        drop(guard);
        self.published()
    }
}

/// Declares a census over a struct of `u64` counters: a `const` list of [`CensusDef`]s and a
/// visitor that writes the fields' values in the same order, so the two cannot drift apart.
///
/// ```ignore
/// saber_serve::census_counters!(SchedCounters, CENSUS, census_values, [
///     (steps, "saber:sched_steps_total", "Steps that executed a bucket.", "sched: steps {},"),
/// ]);
/// ```
#[macro_export]
macro_rules! census_counters {
    ($ty:ty, $defs:ident, $visit:ident, [ $( ($field:ident, $name:literal, $help:literal, $census:literal) ),* $(,)? ]) => {
        impl $ty {
            /// The `/metrics` twin of every counter in this census, in publish order (ws-v8).
            pub const $defs: &'static [$crate::census::CensusDef] = &[
                $( $crate::census::CensusDef::counter($name, $help, $census) ),*
            ];

            /// Writes every counter [`Self::CENSUS`] names, in its order.
            pub fn $visit(&self, put: &mut dyn FnMut(u64)) {
                $( put(self.$field); )*
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEFS: [CensusDef; 3] = [
        CensusDef::counter("saber:t_eager_total", "e", "counters: eager {}"),
        CensusDef::counter("saber:t_refused_total", "r", "parking: refused {},"),
        CensusDef::gauge("saber:t_clean", "c", "clean: clean {}"),
    ];

    fn three(put: &mut dyn FnMut(u64)) {
        put(1);
        put(2);
        put(3);
    }

    #[test]
    fn a_disarmed_mirror_publishes_nothing() {
        let m = CensusMirror::new(DEFS.to_vec());
        m.publish(three);
        assert!(m.read().is_none(), "disarmed: the publish must be a no-op");
        assert!(!m.wait_first(Duration::ZERO));
        m.arm();
        m.publish(three);
        let got: Vec<u64> = m.read().unwrap().iter().map(|s| s.value).collect();
        assert_eq!(got, vec![1, 2, 3]);
        assert!(m.wait_first(Duration::ZERO));
    }

    #[test]
    #[should_panic(expected = "values published")]
    fn a_short_publish_is_a_programming_error() {
        let m = CensusMirror::new(DEFS.to_vec());
        m.arm();
        m.publish(|put| put(1));
    }

    #[test]
    fn the_pattern_finds_the_number_the_line_prints() {
        // A boot line under the same tag comes FIRST in a real log (ws-v8's card run matched it).
        let text = "counters       OFF at boot, eager 99\n\
                    counters       prefill 3 (0 silent), path graph 4 / eager 7, feed device 1\n\
                    parking        parks 2 (policy 1, refused 5), restores 0 (failed 0)\n\
                    other          x";
        assert_eq!(find_in_census(text, &DEFS[0]), Some(7));
        let refused = CensusDef::counter("x", "x", "parking: refused {})");
        assert_eq!(find_in_census(text, &refused), Some(5));
        // NEGATIVE CONTROLS: a label on another line, a missing line, the tail of a longer word.
        let wrong_line = CensusDef::counter("x", "x", "other: eager {}");
        assert_eq!(find_in_census(text, &wrong_line), None);
        let missing = CensusDef::counter("x", "x", "nowhere: eager {}");
        assert_eq!(find_in_census(text, &missing), None);
        let tail = CensusDef::counter("x", "x", "counters: ager {}");
        assert_eq!(find_in_census(text, &tail), None);
    }

    #[test]
    fn readers_never_see_a_torn_snapshot() {
        let m = std::sync::Arc::new(CensusMirror::new(DEFS.to_vec()));
        m.arm();
        let w = std::sync::Arc::clone(&m);
        let writer = std::thread::spawn(move || {
            for i in 0..20_000u64 {
                w.publish(|put| {
                    put(i);
                    put(i);
                    put(i);
                });
            }
        });
        for _ in 0..20_000 {
            if let Some(s) = m.read() {
                assert!(s[0].value == s[1].value && s[1].value == s[2].value);
            }
        }
        writer.join().unwrap();
    }
}
