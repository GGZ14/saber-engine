//! What the HOST spends on a decode step, phase by phase.
//!
//! # Why this exists
//!
//! `ms/step` is a wall, and at TP = 2 the wall is not the card. ws-t1's per-op attribution reads
//! the `b1s8` TARGET step at **18.79 ms** of device time (GEMMs 13.3 / GDN 1.8 / attention 0.5 /
//! all-reduce 1.8 / producers 1.3) and ws-s9's proposal at **~5.4 ms**, against a served step of
//! **24.19 ms** — and ws-t3 then found a per-step HOST cost that a diagnostic arm upstream of the
//! measurement could inflate by **+2.7 ms**, with two named candidates and **no instrument that
//! could tell them apart** (`docs/LEDGER.md`, ws-t3 part 5: "separating them needs a per-step
//! host-time counter the engine does not have; that is the next instrument, not another hold").
//!
//! This is that counter. It is not a profiler: it is a fixed, short list of phases that a decode
//! step passes through **in order**, each one a thing the host does that the device is not doing —
//! a socket frame, a blocking read-back, a table build, a wait — so that "the step got slower" can
//! be answered with a name instead of a guess.
//!
//! # The rules it follows
//!
//! * **It is OFF unless a caller asks for it, and a number taken with it on is not a published
//!   number.** `docs/GATES.md` blind spot 19: a diagnostic must not sit upstream of a published
//!   figure. Off, a phase boundary is one relaxed atomic load; on, it is two `Instant::now()` calls
//!   and a mutex push, and that cost is real enough to be measured rather than assumed
//!   (`docs/EXEC.md` §7.7).
//! * **Every phase is named at construction**, so a phase that never ran prints at zero rather than
//!   vanishing — a missing line reads as "free" when it means "did not happen".
//! * **Nothing branches on a time** (`CLAUDE.md` rule 6). The samples are data: a table, a log line
//!   and nothing else.
//! * **The sum is checked against the step's own wall.** [`HostPhase::Step`] brackets the whole of
//!   `decode_step`, so `unaccounted = step − Σ(phases)` is printed and a phase list that does not
//!   add up says so instead of implying it is complete.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use saber_exec::tp2::{WaitAim, WaitSeen};

/// One phase of a step, on the HOST, in the order a step passes through them.
///
/// Every variant but [`HostPhase::Step`] is disjoint from every other: a nanosecond counted in one
/// is counted in no other, which is what lets the remainder be printed as "unaccounted" and mean
/// something.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum HostPhase {
    /// **Prefill only (C5).** Choosing the chunk's planned shape and pointing it at this chunk:
    /// the bucket pick, the session's facts, the image rows' scatter plan, the padded ids, and the
    /// plan's two retargets (the attention's live rows, the GDN state slot).
    ChunkPlan,
    /// Building this step's index tables into the pinned blob, and its ids into the staging: the
    /// `forward` index laws, on the host, with no device traffic (`docs/EXEC.md` §2).
    Index,
    /// The sampler seam's per-row temperatures and penalty ids, uploaded before anything launches.
    SeamIn,
    /// **Prefill only (C5).** The head's row selector for the chunk's last live row: one BLOCKING
    /// four-byte host-to-device copy, before the step is led.
    RowSel,
    /// **Rank swap.** The step command and the step-varying index region, to the peer
    /// (`payload_of` + `RankLink::lead`). 1,932 KiB per step at `max_seqs` 8, symmetric — the
    /// follower answers with a buffer of the same width (ws-h29's "what is still on this wire").
    Lead,
    /// **Rank swap.** The block table's own frame, on the steps a sequence crossed a page boundary.
    Paged,
    /// **Rank swap.** The seam's input frame, on a step that samples with a temperature.
    SeamLead,
    /// The index blob's and the ids' asynchronous upload: the enqueue, not the DMA.
    Blob,
    /// Binding the eager walk's op list (`step_ops`: the body's descriptors against this plan's
    /// arena, the aux taps, the head's ops) — host work before the first eager launch. Zero on a
    /// graph replay.
    Bind,
    /// Submitting the step's work: one `hipGraphLaunch` on the captured path, or the whole eager
    /// op walk when there is no graph for the shape.
    Submit,
    /// The step's one wait — `WaitWindow::STEP`, 200 µs of spin and then 250 µs sleeps
    /// (`saber_exec::tp2`), or since ws-h32 the aimed wait ([`StepWaits`]). At TP = 1 this is
    /// folded into [`HostPhase::Picks`]' read-back instead. Since ws-h32 it also carries the pick
    /// table's copy, which is enqueued behind the step rather than read after it.
    Wait,
    /// **THE LOGITS GATHER**, on the host: one D2H of the pick table and then one blocking two-byte
    /// read per ROW at the column it names. `host_gathers` counts the per-row half of it.
    Picks,
    /// **Rank swap.** The two ranks' picks, one frame for every row of the step.
    SwapPicks,
    /// The seam's outputs read back, and the peer's half of the same rows.
    SeamOut,
    /// The acceptance law and the row merge, on the host, per sequence over its own rows.
    Accept,
    /// **Rank swap.** The proposal's table frames and the candidate seam.
    DraftFrame,
    /// Seeding the proposal's index tables onto the card.
    DraftSeed,
    /// The proposal's EAGER phase: the context store, whose kernargs are a function of the
    /// acceptance this step's host just decided (ws-s3).
    DraftEager,
    /// The proposal's captured backbone and tail, submitted and waited for.
    DraftReplay,
    /// The walk's tokens (and the candidate shards) read back.
    DraftRead,
    /// **Prefill only (C5). Rank swap.** The drafter's context-store frames: one per stored tile
    /// and the end marker, each a round trip on the peer socket.
    StoreFrame,
    /// **Prefill only (C5).** The drafter's context store over the prompt's rows: every tile's
    /// table upload and eager launches (`store_ctx_tile`), and the schedule that tiles them.
    Store,
    /// **Prefill only (C6).** The context store's ONE wait: since C6 a chunk's tiles are enqueued
    /// back to back (each tile's tables in their own staging slice) and the host waits once, after
    /// the last tile, so [`HostPhase::Store`] is the enqueue and this is the tiles' device time
    /// the host still sees.
    StoreWait,
    /// **The instrument's own device clock** (ws-h32): creating, recording and reading the event
    /// pair each traced wait brackets the card with. Only ever non-zero with the trace on, and
    /// billed here so that it is neither hidden in another phase nor left as `unaccounted`.
    Instrument,
    /// **Prefill table only (C5), and OUTSIDE the step:** a sequence's admission on the engine
    /// thread (its GDN slot cleared, the follower told, the drafter's pages cleared). It runs
    /// between steps, so it is printed beside the chunk and never counted against its wall.
    Admit,
    /// **OUTSIDE the step (P27-13).** The leader's engine thread blocked between a decode step's
    /// reply and the request that collects its proposal (`ProposeAfterCommit`): the executor's
    /// host turn after the forward -- the host sampler deciding each slot's last token (F37) and
    /// the commit. Printed beside the step, never counted against its wall.
    Turn,
    /// **OUTSIDE the step (P27-13).** The leader's engine thread blocked between the previous
    /// decode cycle's last reply (the step's, or its proposal's) and the next decode step's
    /// request: the executor's after-work, the scheduler's turn, the stream writers and the next
    /// batch's build. A gap with a prefill chunk or any other request in it is not counted (it is
    /// that request's time, not idle).
    Idle,
    /// The whole of the step, host wall. The one phase that overlaps the others, on purpose.
    Step,
}

impl HostPhase {
    /// Every phase, in the order a step passes through them.
    #[must_use]
    pub const fn all() -> [HostPhase; 28] {
        [
            HostPhase::ChunkPlan,
            HostPhase::Index,
            HostPhase::SeamIn,
            HostPhase::RowSel,
            HostPhase::Lead,
            HostPhase::Paged,
            HostPhase::SeamLead,
            HostPhase::Blob,
            HostPhase::Bind,
            HostPhase::Submit,
            HostPhase::Wait,
            HostPhase::Picks,
            HostPhase::SwapPicks,
            HostPhase::SeamOut,
            HostPhase::Accept,
            HostPhase::DraftFrame,
            HostPhase::DraftSeed,
            HostPhase::DraftEager,
            HostPhase::DraftReplay,
            HostPhase::DraftRead,
            HostPhase::StoreFrame,
            HostPhase::Store,
            HostPhase::StoreWait,
            HostPhase::Instrument,
            HostPhase::Admit,
            HostPhase::Turn,
            HostPhase::Idle,
            HostPhase::Step,
        ]
    }

    /// The name this phase prints under.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            HostPhase::ChunkPlan => "chunk plan",
            HostPhase::Index => "index build",
            HostPhase::SeamIn => "seam upload",
            HostPhase::RowSel => "row sel h2d",
            HostPhase::Lead => "lead (blob+ids)",
            HostPhase::Paged => "paged frame",
            HostPhase::SeamLead => "seam frame",
            HostPhase::Blob => "blob h2d",
            HostPhase::Bind => "op bind",
            HostPhase::Submit => "submit/replay",
            HostPhase::Wait => "wait",
            HostPhase::Picks => "picks d2h",
            HostPhase::SwapPicks => "swap picks",
            HostPhase::SeamOut => "seam read",
            HostPhase::Accept => "acceptance",
            HostPhase::DraftFrame => "draft frames",
            HostPhase::DraftSeed => "draft seed",
            HostPhase::DraftEager => "draft eager",
            HostPhase::DraftReplay => "draft replay",
            HostPhase::DraftRead => "draft read",
            HostPhase::StoreFrame => "store frames",
            HostPhase::Store => "ctx store",
            HostPhase::StoreWait => "ctx store wait",
            HostPhase::Instrument => "trace clock",
            HostPhase::Admit => "admit (outside)",
            HostPhase::Turn => "turn (outside)",
            HostPhase::Idle => "idle (outside)",
            HostPhase::Step => "STEP (whole)",
        }
    }

    /// Whether this phase is a frame on the peer socket. Printed so that the wire's share of a
    /// two-rank step can be read off the table without adding the rows up by hand.
    #[must_use]
    pub const fn is_rank_swap(self) -> bool {
        matches!(
            self,
            HostPhase::Lead
                | HostPhase::Paged
                | HostPhase::SeamLead
                | HostPhase::SwapPicks
                | HostPhase::DraftFrame
                | HostPhase::StoreFrame
        )
    }

    /// Whether this phase runs OUTSIDE the step's own bracket (admission, and P27-13's turn and
    /// idle): printed beside the step and never summed into its `accounted`.
    #[must_use]
    pub const fn is_outside(self) -> bool {
        matches!(self, HostPhase::Admit | HostPhase::Turn | HostPhase::Idle)
    }

    /// Its slot in the sample table.
    const fn slot(self) -> usize {
        self as usize
    }
}

/// The per-step host-time counter the engine did not have (ws-t3's open item 1).
///
/// Shared between the engine thread, which fills it, and the caller, which reads it — so the
/// samples live behind a `Mutex` and the switch behind an `AtomicBool`. The switch is read once per
/// phase boundary and is the whole of the off-path cost.
#[derive(Debug)]
pub struct HostTrace {
    on: AtomicBool,
    /// Which kind of step the engine thread is in (C5): a [`StepKind`] as a byte. Every sample is
    /// filed under it, so a prefill chunk's phases never land in the decode table.
    kind: AtomicU8,
    /// One sample list per phase, in nanoseconds, per [`StepKind`]. Nanoseconds because a rank
    /// swap of a few hundred microseconds and a table build of a few hundred nanoseconds are both
    /// on this table and a microsecond grid would round the second one to "free".
    ns: Mutex<[Vec<Vec<u64>>; 2]>,
    /// Every traced wait's split into card and host (ws-h32), in arrival order, per [`StepKind`].
    waits: Mutex<[Vec<WaitSample>; 2]>,
    /// **The prefill chunks, one sample each, by planned bucket** (C5): the chunk's live rows, its
    /// whole wall on the engine thread, and the card's span of its step when the wait clocked it.
    chunks: Mutex<Vec<ChunkSample>>,
    /// The last traced wait's card span, for the chunk that is closing to claim.
    last_card: Mutex<Option<u64>>,
    /// The aimed wait's policy and memory. Not a diagnostic: this is the ONE piece of per-step host
    /// state every wait site already holds a reference to, which is why it lives here rather than
    /// being threaded through every step function beside the trace.
    policy: StepWaits,
    /// T4's race checks (`--engine-race-check`), set once at boot on a checked run. Here for the
    /// aimed wait's reason: every wait site, the pick swap and the proposal already hold the trace.
    race: std::sync::OnceLock<crate::race::RaceCheck>,
    /// T6's shadow digest (`--engine-race-check shadow`), set once at boot. Never beside `race`:
    /// the shadow exists to run with every T4/T5 check OFF.
    shadow: std::sync::OnceLock<crate::race::ShadowCheck>,
}

impl Default for HostTrace {
    fn default() -> Self {
        Self::new()
    }
}

impl HostTrace {
    /// A trace with every phase present and empty, switched OFF.
    #[must_use]
    pub fn new() -> HostTrace {
        HostTrace {
            on: AtomicBool::new(false),
            kind: AtomicU8::new(StepKind::Decode as u8),
            ns: Mutex::new(Self::empty()),
            waits: Mutex::new([Vec::new(), Vec::new()]),
            chunks: Mutex::new(Vec::new()),
            last_card: Mutex::new(None),
            policy: StepWaits::default(),
            race: std::sync::OnceLock::new(),
            shadow: std::sync::OnceLock::new(),
        }
    }

    /// Arm T6's shadow digest for the rest of the run (once; refused beside the full checks).
    ///
    /// # Errors
    /// Already armed, or the full checks are.
    pub fn set_shadow(&self, shadow: crate::race::ShadowCheck) -> Result<(), String> {
        if self.race.get().is_some() {
            return Err(
                "the shadow digest runs with the full race checks OFF; both were asked for"
                    .to_owned(),
            );
        }
        self.shadow
            .set(shadow)
            .map_err(|_| "the shadow digest is armed once per engine".to_owned())
    }

    /// The shadow digest, on a shadowed run.
    #[must_use]
    pub fn shadow(&self) -> Option<&crate::race::ShadowCheck> {
        self.shadow.get()
    }

    /// Arm T4's race checks for the rest of the run (once; a second call is refused by name).
    ///
    /// # Errors
    /// The checks were already armed.
    pub fn set_race(&self, check: crate::race::RaceCheck) -> Result<(), String> {
        self.race
            .set(check)
            .map_err(|_| "the race checks are armed once per engine".to_owned())
    }

    /// The race checks, on a checked run.
    #[must_use]
    pub fn race(&self) -> Option<&crate::race::RaceCheck> {
        self.race.get()
    }

    fn empty() -> [Vec<Vec<u64>>; 2] {
        [
            vec![Vec::new(); HostPhase::all().len()],
            vec![Vec::new(); HostPhase::all().len()],
        ]
    }

    /// File every sample until the guard drops under `kind` (C5). The engine thread opens a
    /// [`StepKind::Prefill`] scope around a prefill chunk; everything else is a decode step's.
    /// Cheap whether or not the trace records: one relaxed store each way.
    #[must_use]
    pub fn scope(&self, kind: StepKind) -> KindScope<'_> {
        let before = self.kind.swap(kind as u8, Ordering::Relaxed);
        KindScope {
            trace: self,
            before,
        }
    }

    fn slot_kind(&self) -> usize {
        usize::from(self.kind.load(Ordering::Relaxed) == StepKind::Prefill as u8)
    }

    /// Close one prefill chunk's sample (C5): its planned bucket, its live rows, its wall, and the
    /// card span the step's wait clocked (if the wait ran traced since the last chunk closed).
    pub fn chunk(&self, rows: u32, live: u32, wall: Duration) {
        if !self.on() {
            return;
        }
        let card_ns = self.last_card.lock().ok().and_then(|mut c| c.take());
        if let Ok(mut c) = self.chunks.lock() {
            c.push(ChunkSample {
                rows,
                live,
                wall_ns: wall.as_nanos().min(u128::from(u64::MAX)) as u64,
                card_ns,
            });
        }
    }

    /// The aimed wait's policy and memory (ws-h32). Live whether or not the trace records.
    #[must_use]
    pub fn waits(&self) -> &StepWaits {
        &self.policy
    }

    /// Record one wait's split, when recording.
    pub fn wait_split(&self, sample: WaitSample) {
        if !self.on() {
            return;
        }
        let k = self.slot_kind();
        if let Ok(mut c) = self.last_card.lock() {
            *c = sample.card_ns;
        }
        if let Ok(mut w) = self.waits.lock() {
            w[k].push(sample);
        }
    }

    /// Turn recording on or off. Off is the default and is what every published number runs under.
    pub fn set(&self, on: bool) {
        self.on.store(on, Ordering::Relaxed);
    }

    /// Whether this trace is recording.
    #[must_use]
    pub fn on(&self) -> bool {
        self.on.load(Ordering::Relaxed)
    }

    /// Open a phase: `Some(Instant)` when recording, `None` when not.
    ///
    /// Returning an `Option` rather than always reading the clock is the point: with the trace off
    /// a phase boundary is one relaxed load and no syscall, so the instrument cannot be a term in
    /// a measurement that did not ask for it.
    #[must_use]
    pub fn start(&self) -> Option<Instant> {
        self.on().then(Instant::now)
    }

    /// Close a phase opened by [`HostTrace::start`]. A `None` is a phase that was never opened and
    /// records nothing.
    pub fn stop(&self, phase: HostPhase, at: Option<Instant>) {
        if let Some(t) = at {
            self.add(phase, t.elapsed());
        }
    }

    /// Record a duration against a phase directly, for a caller that already has one.
    pub fn add(&self, phase: HostPhase, d: Duration) {
        if !self.on() {
            return;
        }
        let k = self.slot_kind();
        if let Ok(mut ns) = self.ns.lock() {
            if let Some(slot) = ns[k].get_mut(phase.slot()) {
                slot.push(d.as_nanos().min(u128::from(u64::MAX)) as u64);
            }
        }
    }

    /// The samples so far, left in place.
    #[must_use]
    pub fn snapshot(&self) -> HostTraceReport {
        let ns = self
            .ns
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| Self::empty());
        let waits = self
            .waits
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| [Vec::new(), Vec::new()]);
        let chunks = self.chunks.lock().map(|g| g.clone()).unwrap_or_default();
        HostTraceReport::both(ns, waits, chunks)
    }

    /// The samples so far, and reset — so that one arm's table is that arm's and not the boot's.
    #[must_use]
    pub fn take(&self) -> HostTraceReport {
        let taken = match self.ns.lock() {
            Ok(mut g) => std::mem::replace(&mut *g, Self::empty()),
            Err(_) => Self::empty(),
        };
        let waits = self
            .waits
            .lock()
            .map(|mut g| std::mem::take(&mut *g))
            .unwrap_or_else(|_| [Vec::new(), Vec::new()]);
        let chunks = self
            .chunks
            .lock()
            .map(|mut g| std::mem::take(&mut *g))
            .unwrap_or_default();
        HostTraceReport::both(taken, waits, chunks)
    }

    /// Drop every sample without reading them, for a caller that wants the next arm clean.
    pub fn reset(&self) {
        if let Ok(mut g) = self.ns.lock() {
            *g = Self::empty();
        }
        if let Ok(mut w) = self.waits.lock() {
            for k in w.iter_mut() {
                k.clear();
            }
        }
        if let Ok(mut c) = self.chunks.lock() {
            c.clear();
        }
        if let Ok(mut c) = self.last_card.lock() {
            *c = None;
        }
    }
}

/// Which kind of step a sample belongs to (C5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StepKind {
    /// A decode (or verify) step, and everything the engine thread does outside a prefill chunk.
    Decode = 0,
    /// A prefill chunk, from the leader's pick of its bucket to its commit.
    Prefill = 1,
}

/// Restores the kind that was current when [`HostTrace::scope`] opened.
#[derive(Debug)]
pub struct KindScope<'a> {
    trace: &'a HostTrace,
    before: u8,
}

impl Drop for KindScope<'_> {
    fn drop(&mut self) {
        self.trace.kind.store(self.before, Ordering::Relaxed);
    }
}

/// **One arm's step ledger** (P27-13), per decode step, ms: `wall = device + host + turn + idle`.
///
/// `device` is the card's own spans as the traced waits clocked them (the target step's, the
/// drafter's backbone + tail: each from an event recorded before its submit to one recorded before
/// its wait, so the copies enqueued in between and the graph's launch gaps are INSIDE it); `host`
/// = the phases inside the cycle minus `device` = everything the host does while no card span is
/// open, the waits' overshoot included; `turn` / `idle` are the engine thread's blocked time
/// outside the cycle ([`HostPhase::Turn`], [`HostPhase::Idle`]). `instrument` is the trace's own
/// clock, inside `host` (subtract it for the untraced figure).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepLedger {
    /// Steps the arm ran.
    pub steps: usize,
    /// Every phase inside the cycle, summed (the table's `accounted`).
    pub inside: f64,
    /// The target step's card span.
    pub target: f64,
    /// The drafter's card spans (backbone, its eager store included, + tail).
    pub drafter: f64,
    /// The executor's turn between the step and its proposal.
    pub turn: f64,
    /// The engine thread's idle between cycles.
    pub idle: f64,
    /// The trace's own clock (inside `host`).
    pub instrument: f64,
}

impl StepLedger {
    /// The card's share: target + drafter.
    #[must_use]
    pub fn device(&self) -> f64 {
        self.target + self.drafter
    }

    /// The host's exposed share inside the cycle (never below zero).
    #[must_use]
    pub fn host(&self) -> f64 {
        (self.inside - self.device()).max(0.0)
    }

    /// The whole cycle on the engine thread.
    #[must_use]
    pub fn wall(&self) -> f64 {
        self.inside + self.turn + self.idle
    }

    /// The printed line (`hledger`).
    #[must_use]
    pub fn line(&self, label: &str) -> String {
        format!(
            "hledger   {label}: {:.3} ms/step = device {:.3} (target {:.3} + drafter {:.3}) + host {:.3} \
             (trace clock {:.3} of it) + turn {:.3} + idle {:.3}; host+turn+idle {:.3} ms over {} step(s)",
            self.wall(),
            self.device(),
            self.target,
            self.drafter,
            self.host(),
            self.instrument,
            self.turn,
            self.idle,
            self.host() + self.turn + self.idle,
            self.steps
        )
    }
}

/// One prefill chunk (C5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkSample {
    /// The planned bucket it ran in.
    pub rows: u32,
    /// The prompt rows it carried.
    pub live: u32,
    /// Its whole wall on the engine thread.
    pub wall_ns: u64,
    /// The card's span of its step (submit to the wait's end), when the wait was clocked.
    pub card_ns: Option<u64>,
}

/// What one phase cost over an arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseStat {
    /// Which phase.
    pub phase: HostPhase,
    /// How many times it was entered over the arm. **Not** the step count: a phase entered twice a
    /// step (the proposal's frames) reports twice the steps.
    pub hits: usize,
    /// Everything it cost over the arm.
    pub total: Duration,
    /// The median occurrence.
    pub p50: Duration,
    /// The p90 occurrence, which is where a per-step stall shows up that a median hides (ws-t3's
    /// sweep spread).
    pub p90: Duration,
    /// The worst occurrence.
    pub max: Duration,
}

impl PhaseStat {
    /// What this phase costs a STEP on average: its whole cost divided by the arm's steps, which is
    /// the only normalisation that adds up to the step's own wall.
    #[must_use]
    pub fn per_step_ms(&self, steps: usize) -> f64 {
        if steps == 0 {
            return 0.0;
        }
        self.total.as_secs_f64() * 1e3 / steps as f64
    }
}

/// One arm's host-time table.
#[derive(Clone, Debug)]
pub struct HostTraceReport {
    /// Steps the arm ran, taken from [`HostPhase::Step`]'s own hit count.
    pub steps: usize,
    /// Every phase, in order, including the ones that never ran.
    pub rows: Vec<PhaseStat>,
    /// Every traced wait's card/host split (ws-h32).
    pub waits: Vec<WaitSample>,
    /// **The prefill chunks' own table** (C5), when any ran: the same phases, filed under
    /// [`StepKind::Prefill`], with [`HostPhase::Step`] bracketing the whole chunk. `None` inside a
    /// prefill report itself.
    pub prefill: Option<Box<HostTraceReport>>,
    /// Every prefill chunk, in order (C5). Empty on the decode-only report of an arm with none.
    pub chunks: Vec<ChunkSample>,
}

impl HostTraceReport {
    fn both(
        ns: [Vec<Vec<u64>>; 2],
        waits: [Vec<WaitSample>; 2],
        chunks: Vec<ChunkSample>,
    ) -> HostTraceReport {
        let [ns_d, ns_p] = ns;
        let [w_d, w_p] = waits;
        let mut decode = HostTraceReport::of(&ns_d, w_d);
        let prefill = HostTraceReport::of(&ns_p, w_p);
        if prefill.steps > 0 || !chunks.is_empty() {
            decode.prefill = Some(Box::new(prefill));
        }
        decode.chunks = chunks;
        decode
    }

    fn of(ns: &[Vec<u64>], waits: Vec<WaitSample>) -> HostTraceReport {
        let rows: Vec<PhaseStat> = HostPhase::all()
            .into_iter()
            .map(|phase| {
                let mut xs = ns.get(phase.slot()).cloned().unwrap_or_default();
                xs.sort_unstable();
                let at = |q: f64| -> Duration {
                    if xs.is_empty() {
                        return Duration::ZERO;
                    }
                    let i = (((xs.len() - 1) as f64) * q).round() as usize;
                    Duration::from_nanos(xs[i.min(xs.len() - 1)])
                };
                PhaseStat {
                    phase,
                    hits: xs.len(),
                    total: Duration::from_nanos(xs.iter().sum::<u64>()),
                    p50: at(0.5),
                    p90: at(0.9),
                    max: at(1.0),
                }
            })
            .collect();
        let steps = rows
            .iter()
            .find(|r| r.phase == HostPhase::Step)
            .map_or(0, |r| r.hits);
        HostTraceReport {
            steps,
            rows,
            waits,
            prefill: None,
            chunks: Vec::new(),
        }
    }

    /// One phase's row.
    #[must_use]
    pub fn get(&self, phase: HostPhase) -> PhaseStat {
        self.rows
            .iter()
            .find(|r| r.phase == phase)
            .copied()
            .unwrap_or(PhaseStat {
                phase,
                hits: 0,
                total: Duration::ZERO,
                p50: Duration::ZERO,
                p90: Duration::ZERO,
                max: Duration::ZERO,
            })
    }

    /// Every phase but [`HostPhase::Step`], summed: the host time a step's parts account for.
    #[must_use]
    pub fn accounted(&self) -> Duration {
        self.rows
            .iter()
            .filter(|r| r.phase != HostPhase::Step && !r.phase.is_outside())
            .map(|r| r.total)
            .sum()
    }

    /// The rank swaps' share, for a two-rank step.
    #[must_use]
    pub fn rank_swaps(&self) -> Duration {
        self.rows
            .iter()
            .filter(|r| r.phase.is_rank_swap())
            .map(|r| r.total)
            .sum()
    }

    /// The table, as the gate prints it. `label` names the arm it was taken over.
    ///
    /// Phases that never ran are printed at zero rather than dropped: a phase list with holes in it
    /// invites the reader to assume the hole is free.
    ///
    /// C5: an arm that ran prefill chunks prints their table after the decode one, under
    /// `htrace    <label> PREFILL`, then one line per planned bucket (`hchunk`).
    #[must_use]
    pub fn table(&self, label: &str) -> String {
        let mut out = self.table_one(label);
        if let Some(p) = &self.prefill {
            out.push_str(&p.table_one(&format!("{label} PREFILL")));
        }
        out.push_str(&self.chunk_table(label));
        out
    }

    /// **THE PREFILL CHUNKS BY BUCKET** (C5): per planned row count, the chunks it ran, their live
    /// rows, the padding, the wall and the card's span. `sum s` is that bucket's share of the arm.
    #[must_use]
    pub fn chunk_table(&self, label: &str) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        if self.chunks.is_empty() {
            return out;
        }
        let mut buckets: Vec<u32> = self.chunks.iter().map(|c| c.rows).collect();
        buckets.sort_unstable();
        buckets.dedup();
        let total_wall: u64 = self.chunks.iter().map(|c| c.wall_ns).sum();
        let _ = writeln!(
            out,
            "hchunk    {label}: {} prefill chunk(s), {:.3} s of wall, by planned bucket",
            self.chunks.len(),
            total_wall as f64 / 1e9
        );
        let _ = writeln!(
            out,
            "hchunk    {:>7} {:>7} {:>9} {:>9} {:>9} {:>10} {:>10} {:>10} {:>10} {:>9}",
            "bucket",
            "chunks",
            "live avg",
            "live min",
            "live max",
            "wall p50",
            "wall mean",
            "card p50",
            "card mean",
            "sum s"
        );
        for b in buckets {
            let xs: Vec<&ChunkSample> = self.chunks.iter().filter(|c| c.rows == b).collect();
            let live: Vec<u32> = xs.iter().map(|c| c.live).collect();
            let wall: Vec<u64> = xs.iter().map(|c| c.wall_ns).collect();
            let card: Vec<u64> = xs.iter().filter_map(|c| c.card_ns).collect();
            let mean_ms = |v: &[u64]| {
                if v.is_empty() {
                    f64::NAN
                } else {
                    v.iter().sum::<u64>() as f64 / v.len() as f64 / 1e6
                }
            };
            let _ = writeln!(
                out,
                "hchunk    {:>7} {:>7} {:>9.1} {:>9} {:>9} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>9.3}",
                format!("p{b}"),
                xs.len(),
                f64::from(live.iter().sum::<u32>()) / live.len().max(1) as f64,
                live.iter().min().copied().unwrap_or(0),
                live.iter().max().copied().unwrap_or(0),
                quantile_us(&wall, 0.5) / 1e3,
                mean_ms(&wall),
                quantile_us(&card, 0.5) / 1e3,
                mean_ms(&card),
                wall.iter().sum::<u64>() as f64 / 1e9,
            );
        }
        out
    }

    fn table_one(&self, label: &str) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let step = self.get(HostPhase::Step);
        let _ = writeln!(
            out,
            "htrace    {label}: {} step(s), {} p50 {:.3} ms / p90 {:.3} / max {:.3}",
            self.steps,
            HostPhase::Step.name(),
            step.p50.as_secs_f64() * 1e3,
            step.p90.as_secs_f64() * 1e3,
            step.max.as_secs_f64() * 1e3,
        );
        let _ = writeln!(
            out,
            "htrace    {:<17} {:>7} {:>11} {:>10} {:>10} {:>10}",
            "phase", "hits", "ms/step", "p50 µs", "p90 µs", "max µs"
        );
        for r in &self.rows {
            if r.phase == HostPhase::Step {
                continue;
            }
            let _ = writeln!(
                out,
                "htrace    {:<17} {:>7} {:>11.4} {:>10.1} {:>10.1} {:>10.1}{}",
                r.phase.name(),
                r.hits,
                r.per_step_ms(self.steps),
                r.p50.as_secs_f64() * 1e6,
                r.p90.as_secs_f64() * 1e6,
                r.max.as_secs_f64() * 1e6,
                if r.phase.is_rank_swap() {
                    "  <- wire"
                } else {
                    ""
                },
            );
        }
        let accounted = self.accounted();
        let unaccounted = step.total.saturating_sub(accounted);
        let per = |d: Duration| -> f64 {
            if self.steps == 0 {
                0.0
            } else {
                d.as_secs_f64() * 1e3 / self.steps as f64
            }
        };
        let _ = writeln!(
            out,
            "htrace    {:<17} {:>7} {:>11.4}   (of which the peer socket {:.4} ms/step)",
            "accounted",
            "",
            per(accounted),
            per(self.rank_swaps()),
        );
        let _ = writeln!(
            out,
            "htrace    {:<17} {:>7} {:>11.4}   = the step's own wall minus its parts; a phase list \
             that does not add up says so",
            "unaccounted",
            "",
            per(unaccounted),
        );
        if let Some(l) = self.ledger() {
            let _ = writeln!(out, "{}", l.line(label));
        }
        out.push_str(&self.wait_table(label));
        out
    }

    /// **THE STEP LEDGER** (P27-13): one decode cycle's wall split into the card's own spans, the
    /// host's exposed time inside the cycle, and the engine thread's time outside it. `None` when
    /// no step ran or no target wait was clocked (the device term would read zero and mean
    /// "unknown").
    #[must_use]
    pub fn ledger(&self) -> Option<StepLedger> {
        if self.steps == 0 {
            return None;
        }
        let steps = self.steps as f64;
        let card_ms = |site: WaitSite| -> Option<f64> {
            let xs: Vec<u64> = self
                .waits
                .iter()
                .filter(|w| w.site == site)
                .filter_map(|w| w.card_ns)
                .collect();
            (!xs.is_empty()).then(|| xs.iter().sum::<u64>() as f64 / 1e6 / steps)
        };
        let target = card_ms(WaitSite::Step)?;
        let drafter = card_ms(WaitSite::DraftBackbone).unwrap_or(0.0)
            + card_ms(WaitSite::DraftTail).unwrap_or(0.0);
        let ms = |p: HostPhase| self.get(p).total.as_secs_f64() * 1e3 / steps;
        let inside = self.accounted().as_secs_f64() * 1e3 / steps;
        Some(StepLedger {
            steps: self.steps,
            inside,
            target,
            drafter,
            turn: ms(HostPhase::Turn),
            idle: ms(HostPhase::Idle),
            instrument: ms(HostPhase::Instrument),
        })
    }

    /// **THE WAIT, SPLIT** (ws-h32): per wait site, the host's wall against the card's own span.
    ///
    /// `card` is the device's elapsed time between an event recorded just before the work was
    /// submitted and one recorded just before the wait began (so it covers every copy enqueued
    /// ahead of the wait too). `host` is the host's wall over the same interval, from the first
    /// event's record to the poll that saw the work done. `h-c` = `host - card` is therefore the
    /// launch latency plus the wait's OVERSHOOT; `gap` is the host-only bound on the overshoot (the
    /// interval between the last "not yet" poll and the "done" one, `saber_exec::tp2::WaitSeen`).
    #[must_use]
    pub fn wait_table(&self, label: &str) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        if self.waits.is_empty() {
            return out;
        }
        let _ = writeln!(
            out,
            "hwait     {label}: the wait split into CARD and HOST OVERSHOOT, per site (µs)"
        );
        let _ = writeln!(
            out,
            "hwait     {:<15} {:>6} {:>6} {:>9} {:>9} {:>9} {:>8} {:>8} {:>8} {:>8} {:>6}",
            "site",
            "hits",
            "aimed",
            "wait p50",
            "card p50",
            "card p90",
            "h-c p50",
            "h-c p90",
            "gap p50",
            "gap p90",
            "polls"
        );
        for site in WaitSite::all() {
            let xs: Vec<&WaitSample> = self.waits.iter().filter(|w| w.site == site).collect();
            if xs.is_empty() {
                continue;
            }
            let wait: Vec<u64> = xs.iter().map(|w| w.wait_ns).collect();
            let card: Vec<u64> = xs.iter().filter_map(|w| w.card_ns).collect();
            let hc: Vec<u64> = xs
                .iter()
                .filter_map(|w| Some(w.host_ns?.saturating_sub(w.card_ns?)))
                .collect();
            let gap: Vec<u64> = xs.iter().map(|w| w.gap_ns).collect();
            let polls: Vec<u64> = xs.iter().map(|w| u64::from(w.polls) * 1000).collect();
            let _ = writeln!(
                out,
                "hwait     {:<15} {:>6} {:>6} {:>9.1} {:>9.1} {:>9.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>6.0}",
                site.name(),
                xs.len(),
                xs.iter().filter(|w| w.aimed).count(),
                quantile_us(&wait, 0.5),
                quantile_us(&card, 0.5),
                quantile_us(&card, 0.9),
                quantile_us(&hc, 0.5),
                quantile_us(&hc, 0.9),
                quantile_us(&gap, 0.5),
                quantile_us(&gap, 0.9),
                quantile_us(&polls, 0.5),
            );
        }
        out
    }

    /// This table against an earlier one, phase by phase, per step — the shape ws-t3's control
    /// wanted: the same arm before and after a diagnostic, with the phase that moved named.
    ///
    /// `before` is the baseline. Rows are printed in the order a step passes through them and the
    /// biggest mover is named at the end, so a reader does not have to scan the column.
    #[must_use]
    pub fn delta(&self, before: &HostTraceReport, label: &str) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "htrace    {label}: {} step(s) before, {} after; STEP p50 {:.3} -> {:.3} ms",
            before.steps,
            self.steps,
            before.get(HostPhase::Step).p50.as_secs_f64() * 1e3,
            self.get(HostPhase::Step).p50.as_secs_f64() * 1e3,
        );
        let _ = writeln!(
            out,
            "htrace    {:<17} {:>11} {:>11} {:>11}",
            "phase", "before", "after", "delta"
        );
        let mut worst = (HostPhase::Step, 0.0f64);
        for r in &self.rows {
            let b = before.get(r.phase).per_step_ms(before.steps);
            let a = r.per_step_ms(self.steps);
            if r.phase != HostPhase::Step && (a - b).abs() > worst.1.abs() {
                worst = (r.phase, a - b);
            }
            let _ = writeln!(
                out,
                "htrace    {:<17} {:>11.4} {:>11.4} {:>+11.4}",
                r.phase.name(),
                b,
                a,
                a - b
            );
        }
        let _ = writeln!(
            out,
            "htrace    THE PHASE THAT MOVED: {} ({:+.4} ms/step)",
            worst.0.name(),
            worst.1
        );
        out
    }
}

/// A quantile of nanosecond samples, in microseconds; `NaN` for none (a column with no samples
/// prints as `NaN` rather than as a free zero).
fn quantile_us(xs: &[u64], q: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_unstable();
    let i = (((v.len() - 1) as f64) * q).round() as usize;
    v[i.min(v.len() - 1)] as f64 / 1e3
}

/// Which wait of a step (ws-h32). Three a decode step with a resident drafter at TP = 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WaitSite {
    /// The target step: the verify graph (and, since ws-h32, the pick table's copy behind it).
    Step,
    /// The drafter's captured backbone.
    DraftBackbone,
    /// The drafter's captured tail, after the candidate seam.
    DraftTail,
}

impl WaitSite {
    /// Every site, in the order a step passes through them.
    #[must_use]
    pub const fn all() -> [WaitSite; 3] {
        [WaitSite::Step, WaitSite::DraftBackbone, WaitSite::DraftTail]
    }

    /// The name this site prints under.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            WaitSite::Step => "step",
            WaitSite::DraftBackbone => "draft backbone",
            WaitSite::DraftTail => "draft tail",
        }
    }
}

/// One traced wait: what the host saw and, when the device clock ran, what the card took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaitSample {
    /// Which wait.
    pub site: WaitSite,
    /// Whether it ran aimed ([`StepWaits`]) or on the plain window.
    pub aimed: bool,
    /// The wait phase's own wall: from the wait's start to the poll that saw the work done.
    pub wait_ns: u64,
    /// The host-only overshoot bound ([`WaitSeen::last_gap`]).
    pub gap_ns: u64,
    /// Polls it took.
    pub polls: u32,
    /// The card's span between the submit's event and the wait's, when the clock ran.
    pub card_ns: Option<u64>,
    /// The host's wall over the same span: the submit's event record to the done poll.
    pub host_ns: Option<u64>,
}

impl WaitSample {
    /// A sample from what the wait saw, with the device clock's two numbers when there are any.
    #[must_use]
    pub fn of(
        site: WaitSite,
        aimed: bool,
        seen: &WaitSeen,
        card_host: Option<(Duration, Duration)>,
    ) -> WaitSample {
        let ns = |d: Duration| d.as_nanos().min(u128::from(u64::MAX)) as u64;
        WaitSample {
            site,
            aimed,
            wait_ns: ns(seen.waited),
            gap_ns: ns(seen.last_gap),
            polls: seen.polls,
            card_ns: card_host.map(|(c, _)| ns(c)),
            host_ns: card_host.map(|(_, h)| ns(h)),
        }
    }
}

/// **THE AIMED WAIT'S MEMORY** (ws-h32): per wait site and shape, the last few waits' lengths, and
/// the aim the next one takes from them.
///
/// The aim is the MINIMUM of the last [`StepWaits::HISTORY`] waits at the same site and key, because
/// the one mistake that costs time is sleeping PAST the card: a minimum can only make the wait wake
/// early (and spin longer), never late — see `saber_exec::tp2::WaitAim` for the law. A wait whose
/// FIRST poll saw the work already done carries no information about the card and is not recorded.
/// Off, every wait is `WaitWindow::STEP`'s, which is the control (`EngineConfig::fixed_step_wait`).
/// A fresh `StepWaits` is off; the engine turns it on at open unless the config asks for the
/// control, so a caller that never opened an engine never aims anything.
#[derive(Debug)]
pub struct StepWaits {
    aimed: AtomicBool,
    seen: Mutex<Vec<(WaitSite, u64, VecDeque<Duration>)>>,
}

impl Default for StepWaits {
    fn default() -> Self {
        StepWaits {
            aimed: AtomicBool::new(false),
            seen: Mutex::new(Vec::new()),
        }
    }
}

impl StepWaits {
    /// How many recent waits the aim is the minimum of.
    pub const HISTORY: usize = 8;

    /// Aim waits (the default) or run every one on the fixed window (the control).
    pub fn set_aimed(&self, on: bool) {
        self.aimed.store(on, Ordering::Relaxed);
    }

    /// Whether waits are aimed.
    #[must_use]
    pub fn aimed(&self) -> bool {
        self.aimed.load(Ordering::Relaxed)
    }

    /// The aim for the next wait at `site` and `key`, if waits are aimed and one has been seen.
    #[must_use]
    pub fn aim(&self, site: WaitSite, key: u64) -> Option<WaitAim> {
        if !self.aimed() {
            return None;
        }
        let seen = self.seen.lock().ok()?;
        let (_, _, ring) = seen.iter().find(|(s, k, _)| *s == site && *k == key)?;
        ring.iter().min().copied().map(WaitAim::at)
    }

    /// Remember what a wait at `site` and `key` saw.
    pub fn observe(&self, site: WaitSite, key: u64, seen: &WaitSeen) {
        if seen.polls <= 1 {
            // Done before the first poll: says nothing about how long the card takes.
            return;
        }
        let Ok(mut all) = self.seen.lock() else {
            return;
        };
        let at = match all.iter().position(|(s, k, _)| *s == site && *k == key) {
            Some(i) => i,
            None => {
                all.push((site, key, VecDeque::with_capacity(Self::HISTORY)));
                all.len() - 1
            }
        };
        let ring = &mut all[at].2;
        if ring.len() == Self::HISTORY {
            ring.pop_front();
        }
        ring.push_back(seen.waited);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Off is free and silent: nothing is recorded and `start` never reads the clock.
    #[test]
    fn a_trace_that_is_off_records_nothing() {
        let t = HostTrace::new();
        assert!(!t.on());
        assert!(t.start().is_none(), "an off trace must not read the clock");
        t.add(HostPhase::Wait, Duration::from_millis(5));
        let r = t.snapshot();
        assert_eq!(r.steps, 0);
        assert_eq!(r.get(HostPhase::Wait).hits, 0);
    }

    /// On, every phase is recorded, and the table still carries the phases that never ran.
    #[test]
    fn every_phase_is_present_even_when_it_never_ran() {
        let t = HostTrace::new();
        t.set(true);
        t.add(HostPhase::Step, Duration::from_micros(1000));
        t.add(HostPhase::Wait, Duration::from_micros(400));
        let r = t.snapshot();
        assert_eq!(r.rows.len(), HostPhase::all().len());
        assert_eq!(r.steps, 1);
        assert_eq!(r.get(HostPhase::Paged).hits, 0, "a phase that never ran");
        let table = r.table("unit");
        for p in HostPhase::all() {
            assert!(
                table.contains(p.name()),
                "{} is missing from the table",
                p.name()
            );
        }
    }

    /// The remainder is printed rather than hidden: the parts do not have to add up, and when they
    /// do not the table says by how much.
    #[test]
    fn the_unaccounted_remainder_is_the_step_minus_its_parts() {
        let t = HostTrace::new();
        t.set(true);
        t.add(HostPhase::Step, Duration::from_micros(1000));
        t.add(HostPhase::Wait, Duration::from_micros(400));
        t.add(HostPhase::Picks, Duration::from_micros(100));
        let r = t.snapshot();
        assert_eq!(r.accounted(), Duration::from_micros(500));
        assert!(r.table("unit").contains("unaccounted"));
    }

    /// `take` is per-arm: the second arm's table is the second arm's.
    #[test]
    fn take_resets_so_an_arm_is_its_own() {
        let t = HostTrace::new();
        t.set(true);
        t.add(HostPhase::Step, Duration::from_micros(1000));
        let first = t.take();
        assert_eq!(first.steps, 1);
        let second = t.take();
        assert_eq!(second.steps, 0, "the samples were drained with the first");
    }

    /// The percentiles are over OCCURRENCES and the `ms/step` column is over STEPS: a phase that
    /// runs twice a step costs twice its median.
    #[test]
    fn a_phase_entered_twice_a_step_costs_twice_its_median() {
        let t = HostTrace::new();
        t.set(true);
        for _ in 0..4 {
            t.add(HostPhase::Step, Duration::from_micros(1000));
            t.add(HostPhase::DraftFrame, Duration::from_micros(50));
            t.add(HostPhase::DraftFrame, Duration::from_micros(50));
        }
        let r = t.snapshot();
        let row = r.get(HostPhase::DraftFrame);
        assert_eq!(row.hits, 8);
        assert_eq!(row.p50, Duration::from_micros(50));
        assert!(
            (row.per_step_ms(r.steps) - 0.1).abs() < 1e-9,
            "0.05 ms twice a step"
        );
    }

    /// The aim is the minimum of the recent waits at the SAME site and key, and nothing else.
    #[test]
    fn the_aim_is_the_minimum_of_its_own_site_and_key() {
        let w = StepWaits::default();
        w.set_aimed(true);
        let seen = |us: u64| WaitSeen {
            waited: Duration::from_micros(us),
            last_gap: Duration::ZERO,
            polls: 5,
            slept_to_aim: false,
        };
        assert!(w.aim(WaitSite::Step, 1).is_none(), "nothing seen, no aim");
        for us in [900, 800, 1000] {
            w.observe(WaitSite::Step, 1, &seen(us));
        }
        w.observe(WaitSite::Step, 4, &seen(100));
        w.observe(WaitSite::DraftTail, 1, &seen(50));
        assert_eq!(
            w.aim(WaitSite::Step, 1).map(|a| a.expect),
            Some(Duration::from_micros(800))
        );
        // A wait that was done at its first poll teaches nothing.
        w.observe(
            WaitSite::Step,
            1,
            &WaitSeen {
                polls: 1,
                ..seen(10)
            },
        );
        assert_eq!(
            w.aim(WaitSite::Step, 1).map(|a| a.expect),
            Some(Duration::from_micros(800))
        );
        // The ring forgets: HISTORY later waits push the 800 out.
        for _ in 0..StepWaits::HISTORY {
            w.observe(WaitSite::Step, 1, &seen(950));
        }
        assert_eq!(
            w.aim(WaitSite::Step, 1).map(|a| a.expect),
            Some(Duration::from_micros(950))
        );
        // The control: fixed waits take no aim at all.
        w.set_aimed(false);
        assert!(w.aim(WaitSite::Step, 1).is_none());
    }

    /// The split prints one row per site that waited, with the host-minus-card column.
    #[test]
    fn the_wait_split_prints_card_and_overshoot() {
        let t = HostTrace::new();
        t.set(true);
        let seen = WaitSeen {
            waited: Duration::from_micros(1000),
            last_gap: Duration::from_micros(300),
            polls: 3,
            slept_to_aim: false,
        };
        t.add(HostPhase::Step, Duration::from_micros(1200));
        t.wait_split(WaitSample::of(
            WaitSite::Step,
            false,
            &seen,
            Some((Duration::from_micros(900), Duration::from_micros(1050))),
        ));
        let r = t.take();
        assert_eq!(r.waits.len(), 1);
        let text = r.table("unit");
        assert!(text.contains("hwait"), "{text}");
        assert!(text.contains("150.0"), "host - card = 150 µs:\n{text}");
        assert!(t.take().waits.is_empty(), "take drains the waits too");
        // Off records nothing.
        let off = HostTrace::new();
        off.wait_split(WaitSample::of(WaitSite::Step, false, &seen, None));
        assert!(off.snapshot().waits.is_empty());
    }

    /// P27-13: the step ledger splits a cycle into the card's spans, the host inside the cycle and
    /// the turn / idle outside it; turn and idle never enter `accounted` (the control: the same
    /// samples filed as an inside phase move `accounted` and the host term by exactly that much).
    #[test]
    fn the_step_ledger_splits_device_host_turn_and_idle() {
        let mk = |outside: HostPhase| {
            let t = HostTrace::new();
            t.set(true);
            let seen = WaitSeen {
                waited: Duration::from_micros(1000),
                last_gap: Duration::from_micros(5),
                polls: 3,
                slept_to_aim: false,
            };
            for _ in 0..2 {
                t.add(HostPhase::Step, Duration::from_micros(10_000));
                t.add(HostPhase::Wait, Duration::from_micros(8_100));
                t.add(HostPhase::DraftReplay, Duration::from_micros(1_550));
                t.add(HostPhase::Picks, Duration::from_micros(350));
                t.add(outside, Duration::from_micros(400));
                t.wait_split(WaitSample::of(
                    WaitSite::Step,
                    true,
                    &seen,
                    Some((Duration::from_micros(8_000), Duration::from_micros(8_050))),
                ));
                t.wait_split(WaitSample::of(
                    WaitSite::DraftBackbone,
                    true,
                    &seen,
                    Some((Duration::from_micros(1_500), Duration::from_micros(1_520))),
                ));
            }
            t.take()
        };
        let r = mk(HostPhase::Idle);
        let l = r.ledger().expect("a clocked step has a ledger");
        assert_eq!(l.steps, 2);
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        assert!(close(l.inside, 10.0), "{l:?}");
        assert!(close(l.device(), 9.5), "{l:?}");
        assert!(close(l.host(), 0.5), "{l:?}");
        assert!(close(l.idle, 0.4) && close(l.turn, 0.0), "{l:?}");
        assert!(close(l.wall(), 10.4), "{l:?}");
        assert!(r.table("unit").contains("hledger"));
        let t = mk(HostPhase::Turn).ledger().unwrap();
        assert!(close(t.turn, 0.4) && close(t.host(), 0.5), "{t:?}");
        // Control: the same 0.4 ms filed INSIDE the cycle is host time, not idle.
        let c = mk(HostPhase::Accept).ledger().unwrap();
        assert!(close(c.host(), 0.9) && close(c.idle, 0.0), "{c:?}");
        // No clocked target wait: no ledger rather than a zero device term.
        let none = HostTrace::new();
        none.set(true);
        none.add(HostPhase::Step, Duration::from_micros(1000));
        assert!(none.take().ledger().is_none());
    }

    /// C5: a prefill chunk's samples go to the PREFILL table and never to the decode one, the
    /// chunk table groups by bucket and claims the wait's card span, and the scope restores the
    /// kind it found. The control is the same samples with no scope: they land in the decode table.
    #[test]
    fn a_prefill_scope_files_its_samples_apart_and_restores_the_kind() {
        let t = HostTrace::new();
        t.set(true);
        t.add(HostPhase::Step, Duration::from_micros(1000));
        t.add(HostPhase::Wait, Duration::from_micros(400));
        {
            let _p = t.scope(StepKind::Prefill);
            t.add(HostPhase::ChunkPlan, Duration::from_micros(30));
            t.add(HostPhase::Submit, Duration::from_micros(6000));
            t.add(HostPhase::Wait, Duration::from_micros(60_000));
            t.wait_split(WaitSample::of(
                WaitSite::Step,
                false,
                &WaitSeen {
                    waited: Duration::from_micros(60_000),
                    last_gap: Duration::from_micros(200),
                    polls: 9,
                    slept_to_aim: false,
                },
                Some((Duration::from_micros(65_000), Duration::from_micros(66_000))),
            ));
            t.add(HostPhase::Step, Duration::from_micros(70_000));
            t.chunk(2048, 300, Duration::from_micros(70_000));
            t.add(HostPhase::Step, Duration::from_micros(50_000));
            t.chunk(256, 178, Duration::from_micros(50_000));
        }
        // Back in decode: this lands in the decode table.
        t.add(HostPhase::Wait, Duration::from_micros(100));
        let r = t.take();
        assert_eq!(r.steps, 1, "one decode step");
        assert_eq!(
            r.get(HostPhase::Wait).hits,
            2,
            "both decode waits, no prefill one"
        );
        assert_eq!(r.get(HostPhase::ChunkPlan).hits, 0);
        let p = r.prefill.as_ref().expect("a prefill table");
        assert_eq!(p.steps, 2);
        assert_eq!(p.get(HostPhase::Wait).hits, 1);
        assert_eq!(p.waits.len(), 1, "the prefill wait is filed under prefill");
        assert!(r.waits.is_empty());
        assert_eq!(r.chunks.len(), 2);
        assert_eq!(
            r.chunks[0].card_ns,
            Some(65_000_000),
            "the wait's card span"
        );
        assert_eq!(r.chunks[1].card_ns, None, "claimed once, not twice");
        let text = r.table("unit");
        assert!(text.contains("unit PREFILL"), "{text}");
        assert!(text.contains("p2048") && text.contains("p256"), "{text}");
        // The control: no scope, and the same phase is a decode sample.
        let c = HostTrace::new();
        c.set(true);
        c.add(HostPhase::ChunkPlan, Duration::from_micros(30));
        let rc = c.take();
        assert_eq!(rc.get(HostPhase::ChunkPlan).hits, 1);
        assert!(rc.prefill.is_none());
    }

    /// The delta names the phase that moved, and the NEGATIVE CONTROL is two identical arms, where
    /// nothing may be named as having moved by anything.
    #[test]
    fn the_delta_names_the_phase_that_moved() {
        let mk = |wait_us: u64| {
            let t = HostTrace::new();
            t.set(true);
            for _ in 0..8 {
                t.add(HostPhase::Step, Duration::from_micros(1000 + wait_us));
                t.add(HostPhase::Wait, Duration::from_micros(wait_us));
                t.add(HostPhase::Picks, Duration::from_micros(30));
            }
            t.snapshot()
        };
        let before = mk(100);
        let after = mk(2800);
        let text = after.delta(&before, "case");
        assert!(
            text.contains("THE PHASE THAT MOVED: wait"),
            "the mover must be named:\n{text}"
        );
        // The control: the same arm twice. Nothing moved, and the line says so with a zero rather
        // than by omitting itself.
        let flat = mk(100).delta(&mk(100), "control");
        assert!(
            flat.contains("+0.0000"),
            "the control must read zero:\n{flat}"
        );
    }
}
