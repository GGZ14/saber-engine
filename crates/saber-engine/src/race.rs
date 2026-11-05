//! **THE RACE CHECKS** (workstream T4, `--engine-race-check` / `saber engine --race-check`).
//!
//! Two gate boots in ~45 at TP=2 read a different per-step trace from every other boot at the same
//! settings (B1's `engine-L`, H46's `bat-HL5`), and neither reproduced. A served engine must be
//! deterministic run to run, so every candidate a two-rank step has for a one-off is made
//! OBSERVABLE here, at no cost when the flag is off (one `Option` test at each site):
//!
//! * **the all-reduce handshake** — every reduce resolves to its `_chk` twin
//!   (`saber_kernels::saber::Ar::race_check`; `hip/ar_race_check.h`), which checks each call's payload
//!   against a check word the sender wrote beside it, re-reads its flag after the spin, and checks
//!   row 3's split-K counter. The counts land in each `ar_seq` region's tail, and
//!   [`RaceCheck::after_picks`] reads every region after every step, logs each NEW event with the
//!   step and the region's call counter, and keeps a digest of every region's per-block sequence
//!   counters for the end-of-run line — the lockstep the protocol assumes, compared across the
//!   two ranks' lines;
//! * **the aimed wait** — [`RaceCheck::wait_done`]: an event recorded
//!   before the wait must be complete when the wait returns, and the stream must still be idle;
//! * **the drafter's read** — the proposal's `out.tokens` rows are filled with a sentinel before the
//!   proposal launches and read after its sync ([`RaceCheck::draft_sentinel`]): a sentinel left in
//!   a live row is a read that beat the walk's store.
//!
//! Every event is logged at ERROR with the rank and the step, and counted; nothing is retried and
//! nothing changes what the step computes, so a checked boot shows the event in its digests too.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use saber_hip::{DeviceAddr, HipThread, Stream};
use saber_kernels::saber::{AR_CHECK_DIAG_OFFSET, AR_CHECK_DIAG_WORDS};

use crate::forward::ForwardError;
use crate::tp2::RankLink;

/// The diagnostic words (`CHK_D_*` in `hip/ar_race_check.h`), by name.
pub mod diag {
    /// Blocks (row owners) checked.
    pub const BLOCKS: usize = 0;
    /// First-read hash != the sender's check word.
    pub const MISMATCH: usize = 1;
    /// ... and a system-scope re-read matched.
    pub const CONVERGED: usize = 2;
    /// ... and it never did.
    pub const NEVER: usize = 3;
    /// Flag == s + 1 after the spin (benign).
    pub const AHEAD: usize = 4;
    /// Flag neither s nor s + 1 after the spin.
    pub const DRIFT: usize = 5;
    /// The spin ran out.
    pub const TIMEOUT: usize = 6;
    /// Row 3: a split-K election counter found un-reset.
    pub const CNT: usize = 7;
    /// The first mismatch's sequence number.
    pub const FIRST_S: usize = 8;
    /// ... its block (row).
    pub const FIRST_B: usize = 9;
    /// ... its piece.
    pub const FIRST_P: usize = 10;
    /// The most re-read polls a converged mismatch took.
    pub const POLLS: usize = 11;
    /// Block 0's last sequence number: the region's call counter.
    pub const LAST_S: usize = 12;
}

/// Every count the checks keep, for the end-of-run line.
#[derive(Debug, Default)]
pub struct RaceCounters {
    /// Steps checked (pick swaps).
    pub steps: AtomicU64,
    /// All-reduce blocks (row owners) whose payload was checked.
    pub ar_blocks: AtomicU64,
    /// Payload mismatches.
    pub ar_mismatch: AtomicU64,
    /// ... that converged on a re-read (the data landed late).
    pub ar_converged: AtomicU64,
    /// ... that never did (another call's data).
    pub ar_never: AtomicU64,
    /// Flag one ahead after the spin (benign, counted for scale).
    pub ar_ahead: AtomicU64,
    /// Flag drift.
    pub ar_drift: AtomicU64,
    /// Spins that ran out.
    pub ar_timeout: AtomicU64,
    /// Row 3 split-K counters found un-reset.
    pub ar_cnt: AtomicU64,
    /// The digest of every region's per-block sequence counters at the last step's pick swap —
    /// equal on the two ranks exactly when every reduce of the run paired.
    pub last_seq: AtomicU64,
    /// The frame serial at that point (equal on the two ranks; the protocol checks it per frame).
    pub last_serial: AtomicU64,
    /// Waits that returned with the pre-wait event, or the stream, not complete.
    pub wait_early: AtomicU64,
    /// Waits checked.
    pub waits: AtomicU64,
    /// Proposal reads that found the sentinel in a live row.
    pub draft_sentinel: AtomicU64,
    /// Proposal reads checked.
    pub draft_reads: AtomicU64,
    /// T5: proposals digested in FULL (a first proposal after a prefill and the one after it).
    pub fp_full: AtomicU64,
    /// T5: every other proposal, digested LIGHT (the small head / selector regions).
    pub fp_light: AtomicU64,
    /// T5: a seeded table (or the seam's merged candidates) read back off the card after its
    /// sync and found different from the bytes the host staged: a copy that landed wrong.
    pub table_mismatch: AtomicU64,
    /// T5: a host write into the drafter's pinned seed staging over a slice whose asynchronous
    /// copy the stream had not been drained past, or a pending slice whose bytes changed before
    /// the drain: the staging-reuse hole, caught at the write.
    pub stage_clobbered: AtomicU64,
}

impl RaceCounters {
    fn get(c: &AtomicU64) -> u64 {
        c.load(Ordering::Relaxed)
    }

    /// Every check's events, zero when the run was clean.
    #[must_use]
    pub fn events(&self) -> u64 {
        Self::get(&self.ar_mismatch)
            + Self::get(&self.ar_drift)
            + Self::get(&self.ar_timeout)
            + Self::get(&self.ar_cnt)
            + Self::get(&self.wait_early)
            + Self::get(&self.draft_sentinel)
            + Self::get(&self.table_mismatch)
            + Self::get(&self.stage_clobbered)
    }

    /// The one line a gate greps: `racecheck rank R: events E | ...`.
    #[must_use]
    pub fn line(&self, rank: u32) -> String {
        let g = Self::get;
        format!(
            "racecheck rank {rank}: events {} | steps {} | ar blocks {} mismatch {} (converged {} \
             never {}) drift {} timeout {} splitk-unreset {} ahead {} | seq-final {:#018x} at \
             serial {} | waits {} early {} | draft reads {} sentinel {} | proposals digested \
             {} full {} light | tables-mismatch {} stage-clobbered {}",
            self.events(),
            g(&self.steps),
            g(&self.ar_blocks),
            g(&self.ar_mismatch),
            g(&self.ar_converged),
            g(&self.ar_never),
            g(&self.ar_drift),
            g(&self.ar_timeout),
            g(&self.ar_cnt),
            g(&self.ar_ahead),
            g(&self.last_seq),
            g(&self.last_serial),
            g(&self.waits),
            g(&self.wait_early),
            g(&self.draft_reads),
            g(&self.draft_sentinel),
            g(&self.fp_full),
            g(&self.fp_light),
            g(&self.table_mismatch),
            g(&self.stage_clobbered),
        )
    }
}

/// One `ar_seq` region's diagnostic words.
pub type Diag = [u32; AR_CHECK_DIAG_WORDS];

/// The events between two readings of one region, as the log line states them.
#[must_use]
pub fn diag_delta(prev: &Diag, now: &Diag) -> Option<String> {
    let d = |i: usize| now[i].wrapping_sub(prev[i]);
    let evs = [
        diag::MISMATCH,
        diag::DRIFT,
        diag::TIMEOUT,
        diag::CNT,
        diag::NEVER,
    ];
    if evs.iter().all(|&i| d(i) == 0) {
        return None;
    }
    Some(format!(
        "mismatch +{} (converged +{}, never +{}, first at seq {} block {} piece {}, max polls {}), \
         drift +{}, timeout +{}, splitk-unreset +{}; the region's call counter {} -> {}",
        d(diag::MISMATCH),
        d(diag::CONVERGED),
        d(diag::NEVER),
        now[diag::FIRST_S],
        now[diag::FIRST_B],
        now[diag::FIRST_P],
        now[diag::POLLS],
        d(diag::DRIFT),
        d(diag::TIMEOUT),
        d(diag::CNT),
        prev[diag::LAST_S],
        now[diag::LAST_S],
    ))
}

/// FNV-1a over the per-block sequence counters of every region, in key order, keys included.
#[must_use]
pub fn seq_digest(regions: &[(String, Vec<u32>)]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |b: &[u8]| {
        for &x in b {
            h ^= u64::from(x);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for (k, words) in regions {
        eat(k.as_bytes());
        for w in words {
            eat(&w.to_le_bytes());
        }
    }
    h
}

/// The checks' state on one rank.
#[derive(Debug)]
pub struct RaceCheck {
    rank: u32,
    /// `(key, address, bytes)` of every `ar_seq` region, taken from the launcher once every
    /// collective is prepared (`Launcher::ar_seq_regions`); the launcher keeps them alive.
    regions: Vec<(String, DeviceAddr, u64)>,
    /// The per-block counters the kernels index (`max_blocks`).
    seq_words: usize,
    prev: Mutex<BTreeMap<String, Diag>>,
    /// The counts, shared with whoever prints the end-of-run line.
    pub counters: std::sync::Arc<RaceCounters>,
    /// T5: which proposals are digested in full ([`ProposalMarks`]).
    marks: ProposalMarks,
}

impl RaceCheck {
    /// The checks over `regions`, on `rank`.
    #[must_use]
    pub fn new(rank: u32, regions: Vec<(String, DeviceAddr, u64)>, seq_words: usize) -> RaceCheck {
        RaceCheck {
            rank,
            regions,
            seq_words,
            prev: Mutex::new(BTreeMap::new()),
            counters: std::sync::Arc::new(RaceCounters::default()),
            marks: ProposalMarks::default(),
        }
    }

    /// How many regions are read after every step.
    #[must_use]
    pub fn regions(&self) -> usize {
        self.regions.len()
    }

    /// Read every region: its sequence counters and its diagnostic words.
    fn read(
        &self,
        hip: &HipThread,
        stream: &Stream,
    ) -> Result<Vec<(String, Vec<u32>, Diag)>, ForwardError> {
        let diag_at = AR_CHECK_DIAG_OFFSET as usize;
        let want = diag_at + AR_CHECK_DIAG_WORDS * 4;
        let mut out = Vec::with_capacity(self.regions.len());
        let mut buf = vec![0u8; want];
        for (key, addr, bytes) in &self.regions {
            if (*bytes as usize) < want {
                return Err(ForwardError::Device(format!(
                    "racecheck: `{key}` is {bytes} B and the checks read {want}; the `ar_seq` \
                     rule is older than the checks"
                )));
            }
            hip.memcpy_d2h_addr_blocking(&mut buf, *addr, stream)
                .map_err(|e| ForwardError::Device(format!("racecheck: reading `{key}`: {e}")))?;
            let word =
                |i: usize| u32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap_or_default());
            let seq: Vec<u32> = (0..self.seq_words.min(diag_at / 4)).map(word).collect();
            let mut d: Diag = [0; AR_CHECK_DIAG_WORDS];
            for (i, w) in d.iter_mut().enumerate() {
                *w = word(diag_at / 4 + i);
            }
            out.push((key.clone(), seq, d));
        }
        Ok(out)
    }

    /// **After every pick swap, on both ranks**: read every region, log each new all-reduce event
    /// with `step`, and keep the sequence counters' digest and the frame serial for the end-of-run
    /// line. Nothing crosses the wire: the check changes no exchange.
    ///
    /// # Errors
    /// A read failing. An EVENT is not an error: it is logged and counted, and the step goes on,
    /// so the run's digests say what it did to the ids.
    pub fn after_picks(
        &self,
        hip: &HipThread,
        stream: &Stream,
        link: &RankLink,
    ) -> Result<(), ForwardError> {
        let step = self.counters.steps.fetch_add(1, Ordering::Relaxed);
        let now = self.read(hip, stream)?;
        let mut prev = self
            .prev
            .lock()
            .map_err(|_| ForwardError::Device("racecheck: a poisoned lock".to_owned()))?;
        for (key, _, d) in &now {
            let zero: Diag = [0; AR_CHECK_DIAG_WORDS];
            let p = prev.get(key).unwrap_or(&zero);
            let bump = |c: &AtomicU64, i: usize| {
                c.fetch_add(u64::from(d[i].wrapping_sub(p[i])), Ordering::Relaxed);
            };
            bump(&self.counters.ar_blocks, diag::BLOCKS);
            bump(&self.counters.ar_mismatch, diag::MISMATCH);
            bump(&self.counters.ar_converged, diag::CONVERGED);
            bump(&self.counters.ar_never, diag::NEVER);
            bump(&self.counters.ar_ahead, diag::AHEAD);
            bump(&self.counters.ar_drift, diag::DRIFT);
            bump(&self.counters.ar_timeout, diag::TIMEOUT);
            bump(&self.counters.ar_cnt, diag::CNT);
            if let Some(what) = diag_delta(p, d) {
                tracing::error!(
                    rank = self.rank,
                    step,
                    region = %key,
                    "racecheck ALL-REDUCE EVENT at rank {} step {step} in `{key}`: {what}",
                    self.rank
                );
            }
            prev.insert(key.clone(), *d);
        }
        drop(prev);
        // THE LOCKSTEP, recorded rather than swapped: a swap here would be one more frame in an
        // exchange whose order differs by step kind (the seam, the fused step's two heads -- hold 1
        // found one), so each rank keeps its LAST reading and the end-of-run lines are compared
        // (`t4_digest.py`): the counters never re-synchronise once lost, so a final digest that
        // agrees says every reduce of the run paired.
        let seqs: Vec<(String, Vec<u32>)> = now.into_iter().map(|(k, s, _)| (k, s)).collect();
        self.counters
            .last_seq
            .store(seq_digest(&seqs), Ordering::Relaxed);
        self.counters
            .last_serial
            .store(link.serial(), Ordering::Relaxed);
        Ok(())
    }

    /// The wait's check, after it returned: the event recorded before it (`armed`) must be
    /// complete and the stream idle. `overshoot_us` is what the wait itself reported.
    pub fn wait_done(&self, what: &str, event_done: bool, stream_idle: bool, overshoot_us: u64) {
        self.counters.waits.fetch_add(1, Ordering::Relaxed);
        if !(event_done && stream_idle) {
            self.counters.wait_early.fetch_add(1, Ordering::Relaxed);
            tracing::error!(
                rank = self.rank,
                step = self.counters.steps.load(Ordering::Relaxed),
                "racecheck EARLY WAIT at rank {}: `{what}` returned with the pre-wait event \
                 {} and the stream {} (last poll gap {overshoot_us} us)",
                self.rank,
                if event_done { "done" } else { "NOT DONE" },
                if stream_idle { "idle" } else { "BUSY" },
            );
        }
    }

    /// The proposal read's check: how many of the live rows' ids are still the sentinel.
    pub fn draft_sentinel(&self, raw: &[u8], live_bytes: usize) {
        self.counters.draft_reads.fetch_add(1, Ordering::Relaxed);
        let left = raw[..live_bytes.min(raw.len())]
            .chunks_exact(8)
            .filter(|c| c.iter().all(|&b| b == DRAFT_SENTINEL))
            .count();
        if left > 0 {
            self.counters.draft_sentinel.fetch_add(1, Ordering::Relaxed);
            tracing::error!(
                rank = self.rank,
                step = self.counters.steps.load(Ordering::Relaxed),
                "racecheck DRAFTER READ at rank {}: {left} live id(s) still hold the sentinel \
                 after the proposal's sync -- the read beat the walk's store",
                self.rank
            );
        }
    }

    /// **T5: the next proposal is a FIRST one** (after a prompt's prefill, or a fork's). The
    /// store shape cannot say so -- it is also the one-request decode shape -- so the leader marks
    /// it in `propose_after_prefill` and tells the follower beside `Command::Propose`
    /// (`t5.first`, under the check only), or the follower's own step knows it was a prefill.
    pub fn mark_first(&self) {
        self.marks.mark_first();
    }

    /// **T5: a proposal starts.** Returns its ordinal on this rank and whether it is digested in
    /// FULL: a FIRST proposal (marked, [`RaceCheck::mark_first`], or `first`) and the one proposal
    /// that follows it; every other one is digested light. Both ranks run the same proposals in
    /// the same order, so the ordinal and the choice agree across the pair.
    pub fn proposal_begin(&self, first: bool) -> (u64, bool) {
        self.marks.begin(first)
    }

    /// **T5: the proposal's digest line**, one per proposal per rank:
    /// `racecheck fpd rank R n N full|light ix I: stage=hash ...`, the stages in pipeline order.
    /// `tools/gates/t5_digest.py` keys it by `(rank, n)` and names, on a boot that differs, the
    /// FIRST stage whose hash is not the modal boots'.
    pub fn proposal_end(&self, n: u64, full: bool, ix: usize, d: &ProposalDigest) {
        let c = if full {
            &self.counters.fp_full
        } else {
            &self.counters.fp_light
        };
        c.fetch_add(1, Ordering::Relaxed);
        tracing::info!(
            "racecheck fpd rank {} n {n} {} ix {ix}: {}",
            self.rank,
            if full { "full" } else { "light" },
            d.line()
        );
    }

    /// **T5: a read-back that must equal the bytes the host staged** (a seeded table after its
    /// sync, the seam's merged candidates after theirs). A difference is an EVENT: the copy
    /// landed wrong, or something wrote the buffer between the sync and the read.
    pub fn table_check(&self, what: &str, want: &[u8], got: &[u8]) {
        if want != got {
            self.counters.table_mismatch.fetch_add(1, Ordering::Relaxed);
            let at = want.iter().zip(got).position(|(a, b)| a != b);
            tracing::error!(
                rank = self.rank,
                step = self.counters.steps.load(Ordering::Relaxed),
                "racecheck TABLE LANDED WRONG at rank {}: `{what}` read back {} byte(s) against \
                 {} staged, first difference at {at:?}",
                self.rank,
                got.len(),
                want.len()
            );
        }
    }

    /// **T5: the pinned seed staging reused before its copy ran** (or changed under it).
    pub fn stage_clobbered(&self, what: &str) {
        self.counters
            .stage_clobbered
            .fetch_add(1, Ordering::Relaxed);
        tracing::error!(
            rank = self.rank,
            step = self.counters.steps.load(Ordering::Relaxed),
            "racecheck STAGE CLOBBERED at rank {}: {what}",
            self.rank
        );
    }

    /// The end-of-run line.
    #[must_use]
    pub fn line(&self) -> String {
        self.counters.line(self.rank)
    }
}

/// **Which proposals a digest takes in FULL** (T5, shared by T6's shadow): a FIRST proposal
/// (marked, or `first`) and the one proposal after it; every other one LIGHT. The ordinal counts
/// every proposal on this rank.
#[derive(Debug, Default)]
pub struct ProposalMarks {
    n: AtomicU64,
    next_full: std::sync::atomic::AtomicBool,
    mark: std::sync::atomic::AtomicBool,
}

impl ProposalMarks {
    /// The next proposal is a first one.
    pub fn mark_first(&self) {
        self.mark.store(true, Ordering::Relaxed);
    }

    /// A proposal starts: its ordinal and whether it is digested in full.
    pub fn begin(&self, first: bool) -> (u64, bool) {
        let n = self.n.fetch_add(1, Ordering::Relaxed);
        let first = self.mark.swap(false, Ordering::Relaxed) || first;
        let full = first || self.next_full.swap(false, Ordering::Relaxed);
        if first {
            self.next_full.store(true, Ordering::Relaxed);
        }
        (n, full)
    }
}

/// **T6: THE SHADOW DIGEST** (`--engine-race-check shadow` / `saber engine --race-check shadow`,
/// both ranks; OFF by default, a diagnostic).
///
/// T5's instrument perturbs what it watches: a blocking read-back sync before every proposal launch
/// and ~25 after, the `_chk` all-reduce twins, a sentinel fill. 168 checked boots read 0 events
/// against T4's 1 in 56 unchecked -- not significant (p = 0.22), but the shape of a timing race the
/// checks' syncs hide. The shadow keeps ONLY what costs the host no wait:
///
/// * **on the card**, after the proposal's last kernel on its own stream, two instrument dispatches
///   (`saber_kernels::race_digest`) digest the seeded tables, the aux window, the shape's arena
///   regions (every region on a FULL proposal; the head's selection, the selector's inputs and the
///   tail's outputs on a LIGHT one) and, FULL only, the drafter K/V pages the attention read, into
///   a small record that rides the pinned batch the proposal's tokens are already read with;
/// * **on the host**, from bytes it already holds: the frame, the pinned staging snapshot at the
///   launch, the candidate seam (this rank's shard, the peer's, the merge) and the tokens.
///
/// No `_chk` twin, no per-step `ar_seq` read, no wait re-query, no sentinel, no read-back, no frame
/// on the wire: the follower marks its own first proposals from its own steps. The line is T5's
/// (`racecheck fpd rank R n N full|light ix I: ...`), so `tools/gates/t5_digest.py` reads it; the
/// end-of-run line is `racecheck shadow rank R: ...`.
#[derive(Debug)]
pub struct ShadowCheck {
    rank: u32,
    marks: ProposalMarks,
    /// Proposals digested in full.
    pub full: AtomicU64,
    /// Proposals digested light.
    pub light: AtomicU64,
    /// Records whose length did not match the digest table (counted; that line's device stages
    /// are absent, so it differs from every clean boot's by construction).
    pub short: AtomicU64,
    /// T8: the GDN state slot the last prefill chunk on this rank read `h0` from and wrote `ht` to
    /// (`u32::MAX` before the first): a FULL proposal digests the TARGET's state there.
    prefill_slot: std::sync::atomic::AtomicU32,
    /// T10: the prefill chunks' layer record since the last FULL proposal ([`Self::fold_prefill_stages`]).
    prefill_stages: std::sync::Mutex<Vec<(String, u64)>>,
}

impl ShadowCheck {
    /// The shadow on `rank`.
    #[must_use]
    pub fn new(rank: u32) -> ShadowCheck {
        ShadowCheck {
            rank,
            marks: ProposalMarks::default(),
            full: AtomicU64::new(0),
            light: AtomicU64::new(0),
            short: AtomicU64::new(0),
            prefill_slot: std::sync::atomic::AtomicU32::new(u32::MAX),
            prefill_stages: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// The next proposal is a first one ([`ProposalMarks::mark_first`]).
    pub fn mark_first(&self) {
        self.marks.mark_first();
    }

    /// T8: a prefill chunk on this rank ran on GDN state slot `slot`.
    pub fn note_prefill_slot(&self, slot: u32) {
        self.prefill_slot.store(slot, Ordering::Relaxed);
    }

    /// **T10: fold one prefill chunk's layer record** (`p.*` stages, forward order) into the set
    /// the next FULL proposal line carries: a stage seen before in this window is combined with
    /// the new chunk's digest (order-keyed), a new one is appended -- so the window's first
    /// differing stage in forward order is the first differing stage of any of its chunks at or
    /// before that layer.
    pub fn fold_prefill_stages(&self, stages: &[(String, u64)]) {
        let Ok(mut acc) = self.prefill_stages.lock() else {
            return;
        };
        for (k, h) in stages {
            match acc.iter_mut().find(|(n, _)| n == k) {
                Some((_, a)) => *a = saber_kernels::race_digest::sm64(*a ^ h.rotate_left(17)),
                None => acc.push((k.clone(), *h)),
            }
        }
    }

    /// T10: the layer record since the last FULL proposal (and reset it).
    pub fn take_prefill_stages(&self) -> Vec<(String, u64)> {
        self.prefill_stages
            .lock()
            .map(|mut a| std::mem::take(&mut *a))
            .unwrap_or_default()
    }

    /// T8: the GDN slot of the last prefill chunk on this rank, if any ran.
    #[must_use]
    pub fn prefill_slot(&self) -> Option<u32> {
        let s = self.prefill_slot.load(Ordering::Relaxed);
        (s != u32::MAX).then_some(s)
    }

    /// A proposal starts ([`ProposalMarks::begin`]).
    pub fn proposal_begin(&self, first: bool) -> (u64, bool) {
        self.marks.begin(first)
    }

    /// The proposal's digest line (T5's format).
    pub fn proposal_end(&self, n: u64, full: bool, ix: usize, d: &ProposalDigest) {
        let c = if full { &self.full } else { &self.light };
        c.fetch_add(1, Ordering::Relaxed);
        tracing::info!(
            "racecheck fpd rank {} n {n} {} ix {ix}: {}",
            self.rank,
            if full { "full" } else { "light" },
            d.line()
        );
    }

    /// The end-of-run line: `racecheck shadow rank R: proposals digested F full L light | ...`.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "racecheck shadow rank {}: proposals digested {} full {} light | short records {}",
            self.rank,
            self.full.load(Ordering::Relaxed),
            self.light.load(Ordering::Relaxed),
            self.short.load(Ordering::Relaxed),
        )
    }
}

/// The byte the proposal's `out.tokens` rows are filled with before it launches (an id of −1).
pub const DRAFT_SENTINEL: u8 = 0xff;

/// FNV-1a 64 over `bytes` (the digest every T5 stage carries).
#[must_use]
pub fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// **T5: one proposal's stage digests**, in the order the proposal produced them: the frame, the
/// pinned staging snapshot at launch, the seeded tables read back, the aux window, the drafter's
/// K/V pages before the launch, the arena's regions after the backbone, the candidate seam (this
/// rank's shard, the peer's, the merge), the tail's regions, the K/V pages after, the tokens.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ProposalDigest {
    stages: Vec<(String, u64)>,
}

impl ProposalDigest {
    /// Append one stage: `name` and the digest of its bytes.
    pub fn push(&mut self, name: impl Into<String>, bytes: &[u8]) {
        self.stages.push((name.into(), fnv(bytes)));
    }

    /// Append one stage whose digest is already taken.
    pub fn push_hash(&mut self, name: impl Into<String>, h: u64) {
        self.stages.push((name.into(), h));
    }

    /// The stages, in order.
    #[must_use]
    pub fn stages(&self) -> &[(String, u64)] {
        &self.stages
    }

    /// `name=hash name=hash ...` (the hash as 16 hex digits).
    #[must_use]
    pub fn line(&self) -> String {
        let mut s = String::with_capacity(self.stages.len() * 28);
        for (i, (k, h)) in self.stages.iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            s.push_str(k);
            s.push('=');
            s.push_str(&format!("{h:016x}"));
        }
        s
    }
}

/// **T5: the pinned seed staging's pending slices** -- the spans whose asynchronous copy the
/// stream has not been drained past (C6's context store stages every tile's tables in its own
/// slice and waits once per chunk). The law it checks: no host write reaches a pending slice
/// before the drain. Kept by the drafter resident; consulted only under the race check.
#[derive(Debug, Default)]
pub struct StageSpans {
    pending: Vec<(usize, usize, u64)>,
}

impl StageSpans {
    /// A write of `len` bytes at `at` is about to land in the staging: the pending slices it
    /// overlaps (each one a copy that may not have run yet).
    #[must_use]
    pub fn overlaps(&self, at: usize, len: usize) -> usize {
        self.pending
            .iter()
            .filter(|(o, l, _)| at < o + l && *o < at + len)
            .count()
    }

    /// A slice whose copy is now in flight, with the digest of the bytes it must still hold.
    pub fn push(&mut self, at: usize, len: usize, h: u64) {
        self.pending.push((at, len, h));
    }

    /// The stream was drained: every pending copy has run. Returns the slices whose bytes no
    /// longer hold what was staged (`buf` is the staging), and forgets them all.
    pub fn drain(&mut self, buf: &[u8]) -> usize {
        let bad = self
            .pending
            .iter()
            .filter(|(o, l, h)| buf.get(*o..o + l).is_none_or(|b| fnv(b) != *h))
            .count();
        self.pending.clear();
        bad
    }

    /// Slices pending.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Nothing pending.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clean_reading_has_no_delta_and_an_event_names_its_first_call() {
        let a: Diag = [0; AR_CHECK_DIAG_WORDS];
        let mut b = a;
        b[diag::BLOCKS] = 40;
        b[diag::AHEAD] = 3;
        b[diag::LAST_S] = 129;
        // Blocks checked and the benign "ahead" are not events.
        assert_eq!(diag_delta(&a, &b), None);
        let mut c = b;
        c[diag::MISMATCH] = 1;
        c[diag::CONVERGED] = 1;
        c[diag::FIRST_S] = 77;
        c[diag::FIRST_B] = 5;
        c[diag::FIRST_P] = 12;
        let s = diag_delta(&b, &c).expect("a mismatch is an event");
        assert!(
            s.contains("mismatch +1") && s.contains("seq 77 block 5 piece 12"),
            "{s}"
        );
        // Control: drift alone is an event too.
        let mut d = b;
        d[diag::DRIFT] = 2;
        assert!(diag_delta(&b, &d).expect("drift").contains("drift +2"));
    }

    #[test]
    fn the_sequence_digest_sees_one_counter_and_one_key() {
        let base = vec![("saber/allreduce m=8".to_owned(), vec![5u32, 5, 5, 4])];
        let same = base.clone();
        assert_eq!(seq_digest(&base), seq_digest(&same));
        let mut off = base.clone();
        off[0].1[3] = 6;
        assert_ne!(seq_digest(&base), seq_digest(&off));
        let mut renamed = base.clone();
        renamed[0].0.push('x');
        assert_ne!(seq_digest(&base), seq_digest(&renamed));
    }

    #[test]
    fn the_first_proposal_and_the_next_are_full_and_the_rest_light() {
        let rc = RaceCheck::new(1, Vec::new(), 512);
        assert_eq!(rc.proposal_begin(false), (0, false));
        assert_eq!(rc.proposal_begin(true), (1, true));
        assert_eq!(rc.proposal_begin(false), (2, true));
        assert_eq!(rc.proposal_begin(false), (3, false));
        // A fork's store-shape proposal right after a first proposal keeps the next one full.
        assert_eq!(rc.proposal_begin(true), (4, true));
        assert_eq!(rc.proposal_begin(true), (5, true));
        assert_eq!(rc.proposal_begin(false), (6, true));
        assert_eq!(rc.proposal_begin(false), (7, false));
        // The mark is the same as the argument, and it is consumed by one proposal.
        rc.mark_first();
        assert_eq!(rc.proposal_begin(false), (8, true));
        assert_eq!(rc.proposal_begin(false), (9, true));
        assert_eq!(rc.proposal_begin(false), (10, false));
    }

    #[test]
    fn the_shadow_marks_its_proposals_like_the_full_check_and_counts_them() {
        let sh = ShadowCheck::new(1);
        assert_eq!(sh.proposal_begin(false), (0, false));
        sh.mark_first();
        assert_eq!(sh.proposal_begin(false), (1, true));
        assert_eq!(sh.proposal_begin(false), (2, true));
        assert_eq!(sh.proposal_begin(false), (3, false));
        let mut d = ProposalDigest::default();
        d.push("frame", b"x");
        sh.proposal_end(1, true, 0, &d);
        sh.proposal_end(3, false, 2, &d);
        assert_eq!(
            sh.line(),
            "racecheck shadow rank 1: proposals digested 1 full 1 light | short records 0"
        );
    }

    #[test]
    fn a_table_that_landed_wrong_is_an_event_and_one_that_landed_is_not() {
        let rc = RaceCheck::new(0, Vec::new(), 512);
        rc.table_check("q.input_ids", &[1, 2, 3], &[1, 2, 3]);
        assert_eq!(rc.counters.events(), 0);
        // Control: one byte off.
        rc.table_check("q.input_ids", &[1, 2, 3], &[1, 9, 3]);
        assert_eq!(rc.counters.table_mismatch.load(Ordering::Relaxed), 1);
        assert_eq!(rc.counters.events(), 1);
        assert!(rc.line().contains("tables-mismatch 1 stage-clobbered 0"));
    }

    #[test]
    fn the_stage_spans_see_a_write_over_a_pending_slice_and_a_changed_one() {
        let mut s = StageSpans::default();
        let buf = vec![7u8; 64];
        s.push(0, 16, fnv(&buf[0..16]));
        s.push(16, 16, fnv(&buf[16..32]));
        // Past the pending slices: clean. Over one: seen.
        assert_eq!(s.overlaps(32, 16), 0);
        assert_eq!(s.overlaps(8, 16), 2);
        assert_eq!(s.overlaps(0, 1), 1);
        // A drain over unchanged bytes finds nothing and forgets the slices.
        assert_eq!(s.drain(&buf), 0);
        assert!(s.is_empty());
        // Control: a slice whose bytes changed before the drain.
        s.push(0, 16, fnv(&buf[0..16]));
        let mut changed = buf.clone();
        changed[3] = 0;
        assert_eq!(s.drain(&changed), 1);
    }

    #[test]
    fn the_digest_line_keeps_the_stage_order() {
        let mut d = ProposalDigest::default();
        d.push("frame", b"abc");
        d.push_hash("tokens", 0x1234);
        assert_eq!(
            d.line(),
            format!("frame={:016x} tokens=0000000000001234", fnv(b"abc"))
        );
        assert_eq!(d.stages()[0].0, "frame");
    }

    #[test]
    fn the_sentinel_counts_only_whole_ids_in_live_rows() {
        let rc = RaceCheck::new(0, Vec::new(), 512);
        let mut raw = vec![0u8; 64];
        rc.draft_sentinel(&raw, 32);
        assert_eq!(rc.counters.draft_sentinel.load(Ordering::Relaxed), 0);
        // A sentinel PAST the live rows is a padding row's, not a race.
        raw[40..48].fill(DRAFT_SENTINEL);
        rc.draft_sentinel(&raw, 32);
        assert_eq!(rc.counters.draft_sentinel.load(Ordering::Relaxed), 0);
        // One in a live row is.
        raw[8..16].fill(DRAFT_SENTINEL);
        rc.draft_sentinel(&raw, 32);
        assert_eq!(rc.counters.draft_sentinel.load(Ordering::Relaxed), 1);
        assert_eq!(rc.counters.draft_reads.load(Ordering::Relaxed), 3);
        assert!(rc.line().contains("sentinel 1"));
    }
}
