//! Cold start, as a ledger of phases rather than one number.
//!
//! # Why this exists
//!
//! `docs/ARCHITECTURE.md` § 4 promises **~10 s cold** against prod's **210 s** to `/v1/models` on a
//! fresh cache dir (`docs/LEDGER.md`, 2026-09-19). A single boot time cannot be argued with: it
//! either is or is not ten seconds, and when it is not, nothing in it says which part to attack. So
//! the engine records what it spent on each phase between `spawn` and ready, and the serve path
//! prints the list with the first request's TTFT under it.
//!
//! This is deliberately **not** a profiler. It is a fixed, short list of phases that correspond to
//! real dependencies — you cannot upload weights before the pack is open, you cannot capture a graph
//! before its plan is built — because the point of measuring them separately is to know which ones
//! may overlap and which may not.
//!
//! # The rules it follows
//!
//! * Every phase is named at construction, so a phase that is never entered still appears, at zero.
//!   A missing line would read as "fast" when it means "did not run".
//! * Phases nest one level: a phase may have sub-phases whose times sum to less than its own (the
//!   remainder is its own overhead, and that is worth seeing).
//! * Nothing here allocates on a hot path or is read by anything but a log line. `CLAUDE.md` rule 6:
//!   a measurement is data, never control flow. No code branches on a phase time.

use std::fmt::Write as _;
use std::time::{Duration, Instant};

/// The phases of a cold start, in dependency order.
///
/// The order is the order they must happen in, not the order they are recorded; a phase that has
/// been overlapped with another still appears in its dependency position, with a note.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Phase {
    /// `dlopen` of the HIP runtime, the one HIP thread, device open.
    HipOpen,
    /// The pack's header, its record table and the table's digest.
    PackOpen,
    /// Building the execution plan for every bucket: shapes, routes, launch budgets.
    PlanBuild,
    /// Loading `.hsaco` code objects and binding kernels.
    CodeObjects,
    /// Streaming the weights into VRAM, verified. Workstream K7's path.
    WeightUpload,
    /// Allocating KV, GDN state, index tables and the scratch arenas.
    Arenas,
    /// The peer connection, the IPC handle exchange and the all-reduce workspace. TP > 1 only.
    PeerExchange,
    /// Capturing each bucket's graph.
    GraphCapture,
    /// The scheduler, its arenas and the byte ledger's reservations.
    Scheduler,
    /// Binding the HTTP socket.
    ServerBind,
    /// The host plane: the tokenizer and the chat template. **Nothing about it touches the device**,
    /// which is why it is the one phase that can be hidden behind the whole of the rest.
    HostPlane,
}

impl Phase {
    /// Every phase, in dependency order.
    #[must_use]
    pub const fn all() -> [Phase; 11] {
        [
            Phase::HipOpen,
            Phase::PackOpen,
            Phase::PlanBuild,
            Phase::CodeObjects,
            Phase::WeightUpload,
            Phase::Arenas,
            Phase::PeerExchange,
            Phase::GraphCapture,
            Phase::Scheduler,
            Phase::ServerBind,
            Phase::HostPlane,
        ]
    }

    /// The name printed in the ledger.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Phase::HipOpen => "hip open",
            Phase::PackOpen => "pack open + verify",
            Phase::PlanBuild => "plan build",
            Phase::CodeObjects => "code objects",
            Phase::WeightUpload => "weight upload",
            Phase::Arenas => "arenas",
            Phase::PeerExchange => "peer exchange",
            Phase::GraphCapture => "graph capture",
            Phase::Scheduler => "scheduler",
            Phase::ServerBind => "server bind",
            Phase::HostPlane => "host plane",
        }
    }
}

/// What one phase cost, and what it did.
#[derive(Clone, Debug, Default)]
pub struct PhaseRecord {
    /// Wall time inside the phase.
    pub elapsed: Duration,
    /// A short fact the phase wants in its line: record counts, GB/s, bucket counts.
    pub detail: String,
    /// Whether this phase ran **concurrently** with an earlier one. Its time is then wall time it
    /// did not add to the total, and the ledger says so rather than double-counting it.
    pub overlapped: bool,
}

/// A cold start's phases, recorded as they happen.
///
/// `Send`, because the engine thread fills it and the caller prints it.
#[derive(Clone, Debug)]
pub struct Phases {
    started: Instant,
    records: Vec<(Phase, PhaseRecord)>,
    /// Wall time from construction to `ready`.
    pub total: Duration,
}

impl Default for Phases {
    fn default() -> Self {
        Self::new()
    }
}

impl Phases {
    /// A ledger with every phase present at zero, so a phase that never ran still prints.
    #[must_use]
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            records: Phase::all()
                .into_iter()
                .map(|p| (p, PhaseRecord::default()))
                .collect(),
            total: Duration::ZERO,
        }
    }

    /// When this cold start began.
    #[must_use]
    pub fn started(&self) -> Instant {
        self.started
    }

    /// Time a phase and record what it did.
    pub fn record(&mut self, phase: Phase, elapsed: Duration, detail: impl Into<String>) {
        if let Some((_, r)) = self.records.iter_mut().find(|(p, _)| *p == phase) {
            r.elapsed += elapsed;
            let detail = detail.into();
            if !detail.is_empty() {
                r.detail = detail;
            }
        }
    }

    /// Mark a phase as having run concurrently with an earlier one.
    pub fn overlapped(&mut self, phase: Phase) {
        if let Some((_, r)) = self.records.iter_mut().find(|(p, _)| *p == phase) {
            r.overlapped = true;
        }
    }

    /// Run `f`, timing it as `phase`.
    pub fn time<T>(&mut self, phase: Phase, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let out = f();
        self.record(phase, t.elapsed(), "");
        out
    }

    /// Stop the clock. Called when the engine reports ready.
    pub fn finish(&mut self) {
        self.total = self.started.elapsed();
    }

    /// What one phase cost.
    #[must_use]
    pub fn get(&self, phase: Phase) -> Duration {
        self.records
            .iter()
            .find(|(p, _)| *p == phase)
            .map_or(Duration::ZERO, |(_, r)| r.elapsed)
    }

    /// The phases whose time is on the critical path — everything not marked overlapped.
    #[must_use]
    pub fn serial(&self) -> Duration {
        self.records
            .iter()
            .filter(|(_, r)| !r.overlapped)
            .map(|(_, r)| r.elapsed)
            .sum()
    }

    /// The ledger, as the boot log prints it.
    ///
    /// `unaccounted` is the difference between the total and the phases, and it is printed rather
    /// than hidden: a cold start with two seconds nobody claims is a cold start nobody understands.
    #[must_use]
    pub fn render(&self, title: &str) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "{title}");
        let _ = writeln!(out, "  {:<22} {:>9}  {:>7}  detail", "phase", "ms", "share");
        let total = self.total.as_secs_f64().max(f64::MIN_POSITIVE);
        let mut accounted = Duration::ZERO;
        for (phase, r) in &self.records {
            let ms = r.elapsed.as_secs_f64() * 1e3;
            if !r.overlapped {
                accounted += r.elapsed;
            }
            let share = if r.overlapped {
                "  (ovl)".to_owned()
            } else {
                format!("{:>6.1}%", 100.0 * r.elapsed.as_secs_f64() / total)
            };
            let _ = writeln!(
                out,
                "  {:<22} {ms:>9.1}  {share}  {}",
                phase.name(),
                r.detail
            );
        }
        let rest = self.total.saturating_sub(accounted);
        let _ = writeln!(
            out,
            "  {:<22} {:>9.1}  {:>6.1}%  everything the phases above do not claim",
            "unaccounted",
            rest.as_secs_f64() * 1e3,
            100.0 * rest.as_secs_f64() / total
        );
        let _ = writeln!(
            out,
            "  {:<22} {:>9.1}",
            "TOTAL to ready",
            self.total.as_secs_f64() * 1e3
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_phase_appears_even_when_it_did_not_run() {
        // A phase that is missing from the ledger reads as "fast" when it means "did not run" —
        // which at TP=1 is exactly what `peer exchange` is.
        let mut p = Phases::new();
        p.record(
            Phase::WeightUpload,
            Duration::from_millis(1100),
            "1237 records, 11.2 GB/s",
        );
        p.finish();
        let out = p.render("boot");
        for phase in Phase::all() {
            assert!(
                out.contains(phase.name()),
                "{} is missing from:\n{out}",
                phase.name()
            );
        }
        assert!(out.contains("1237 records"));
        assert!(
            out.contains("peer exchange"),
            "TP=1 still lists the peer phase"
        );
    }

    #[test]
    fn overlapped_phases_are_not_counted_against_the_critical_path() {
        let mut p = Phases::new();
        p.record(Phase::WeightUpload, Duration::from_millis(1000), "");
        p.record(Phase::PlanBuild, Duration::from_millis(400), "");
        p.overlapped(Phase::PlanBuild);
        p.finish();
        assert_eq!(p.serial(), Duration::from_millis(1000));
        assert_eq!(p.get(Phase::PlanBuild), Duration::from_millis(400));
        let out = p.render("boot");
        assert!(
            out.contains("(ovl)"),
            "an overlapped phase is marked:\n{out}"
        );

        // NEGATIVE CONTROL: without the mark the same 400 ms IS on the critical path, so the
        // distinction is doing work rather than decorating the output.
        let mut q = Phases::new();
        q.record(Phase::WeightUpload, Duration::from_millis(1000), "");
        q.record(Phase::PlanBuild, Duration::from_millis(400), "");
        q.finish();
        assert_eq!(q.serial(), Duration::from_millis(1400));
    }

    #[test]
    fn the_unaccounted_remainder_is_printed_rather_than_hidden() {
        let mut p = Phases::new();
        p.record(Phase::PackOpen, Duration::from_millis(10), "");
        p.total = Duration::from_millis(1000);
        let out = p.render("boot");
        assert!(out.contains("unaccounted"), "{out}");
        // 990 of the 1000 ms belong to nobody, and the line has to say so.
        let line = out
            .lines()
            .find(|l| l.contains("unaccounted"))
            .expect("the line");
        assert!(line.contains("990.0"), "{line}");
    }
}
