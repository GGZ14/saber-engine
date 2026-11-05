//! FN112: THE OVERLAP SCHEDULER's arm (`--fn-overlap on`) -- a decode / verify step's
//! token-INDEPENDENT host work queued behind its graph before the host waits.
//!
//! The census (LEDGER FN112) classes every named host phase of a served step (FN75 / FN93) by what
//! it reads: a phase that reads nothing step N writes may run before step N ends; a phase that
//! reads step N's bytes but whose SHAPE is known before (which rows, which addresses, how many
//! bytes) can be QUEUED on the card behind step N, so the bytes cross the link the moment the
//! step ends instead of after the host wakes and asks; the rest (the seam's log-sum-exps, the
//! rank exchange, the next step's inputs from the committed tokens) stays where it is. This arm
//! takes the second class: the device seam pick (`fn_pick_top64_bf16`, the same launch), the
//! rows a request's sampling / logprobs / probes / row sha need (a function of the REQUESTS, not
//! of any token) and the tier end's route log -- three D2H copies into one pinned staging on the
//! step's stream, one event behind them, ONE wait. The kernels, their inputs and their stream
//! order are the control's (the seam kernel launches behind the same step; the reads are the same
//! ranges); the bytes the host reads are the same bytes (bit-exact by construction, gated).
//!
//! Byte law of the staging: the layout is fixed at boot from the pools' row cap
//! ([`Layout::of`]); a step's queued reads are consumed by the SAME step's tier end and pick, and
//! a pick whose (logits address, rows, read set) differs from the queued one falls back to the
//! control's reads (counted `miss`).

use saber_hip::{Event, PinnedBuf};

/// The staging's three regions (bytes, offsets) for `rows_max` pick rows of `half` bf16 logits,
/// `seam_bytes_row` bytes of device seam a row, and the route log of `layers` x `rows_max` rows x
/// `log_row` bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub rows_max: usize,
    pub seam_off: usize,
    pub seam_row: usize,
    pub log_off: usize,
    pub log_row: usize,
    pub layers: usize,
    pub raw_off: usize,
    pub raw_row: usize,
    pub total: usize,
}

impl Layout {
    /// The staging of a rank: seam, then the route log, then the raw rows (each region 4 KiB
    /// aligned so a copy's destination never shares a page with another's).
    #[must_use]
    pub fn of(
        rows_max: usize,
        seam_row: usize,
        layers: usize,
        log_row: usize,
        half: usize,
    ) -> Self {
        let al = |x: usize| x.div_ceil(4096) * 4096;
        let seam_off = 0;
        let log_off = al(seam_off + rows_max * seam_row);
        let raw_off = al(log_off + layers * rows_max * log_row);
        let raw_row = half * 2;
        let total = al(raw_off + rows_max * raw_row);
        Layout {
            rows_max,
            seam_off,
            seam_row,
            log_off,
            log_row,
            layers,
            raw_off,
            raw_row,
            total,
        }
    }

    /// The route log's bytes for a step of `m` rows (the tier end's read: `layers x m x log_row`).
    #[must_use]
    pub fn log_bytes(&self, m: usize) -> usize {
        self.layers * m * self.log_row
    }
}

/// The reads queued behind the step in flight.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pend {
    /// The step's logits (arena address) and its pick rows.
    pub lg: u64,
    pub rows: usize,
    /// The device seam was launched and read.
    pub seam: bool,
    /// The raw rows were read (`rows` x `raw_row` bytes).
    pub raw: bool,
    /// The route log was read for a step of `m` rows (`None`: not read).
    pub log_m: Option<u32>,
}

/// One rank's arm state: the staging, the event behind the queued reads, the counters.
pub struct Ov112 {
    pub lay: Layout,
    pub stg: PinnedBuf,
    pub ev: Event,
    pub pend: Option<Pend>,
    /// Steps queued; seam rows, raw rows and route-log reads queued; picks / logs served from the
    /// staging; picks that fell back (`miss`); the card's ms for the queued reads (the step's end
    /// event -> the reads' event: the dependent remainder ON THE DEVICE).
    pub steps: u64,
    pub seam_rows: u64,
    pub raw_rows: u64,
    pub logs: u64,
    pub used_pick: u64,
    pub used_log: u64,
    pub miss: u64,
    pub dev_ms: f64,
    /// The host's pick-read + route-log-read ms under the arm (what remains of the two phases).
    pub host_ms: f64,
}

impl Ov112 {
    /// A fresh arm over its staging and event.
    #[must_use]
    pub fn new(lay: Layout, stg: PinnedBuf, ev: Event) -> Self {
        Ov112 {
            lay,
            stg,
            ev,
            pend: None,
            steps: 0,
            seam_rows: 0,
            raw_rows: 0,
            logs: 0,
            used_pick: 0,
            used_log: 0,
            miss: 0,
            dev_ms: 0.0,
            host_ms: 0.0,
        }
    }

    /// The tier end's route log for a step of `m` rows, if this step queued it (consumed).
    pub fn take_log(&mut self, m: u32) -> Option<Vec<u8>> {
        let p = self.pend.as_mut()?;
        if p.log_m != Some(m) {
            return None;
        }
        p.log_m = None;
        self.used_log += 1;
        let n = self.lay.log_bytes(m as usize);
        Some(self.stg.as_slice()[self.lay.log_off..self.lay.log_off + n].to_vec())
    }

    /// Whether the pick of (`lg`, `rows`) with the read set (`seam`, `raw`) is the queued one
    /// (consumed either way: a mismatch is a `miss` and the control's reads run).
    pub fn take_pick(&mut self, lg: u64, rows: usize, seam: bool, raw: bool) -> bool {
        let Some(p) = self.pend.take() else {
            return false;
        };
        let hit = p.lg == lg && p.rows == rows && p.seam == seam && p.raw == raw;
        if hit {
            self.used_pick += 1;
        } else {
            self.miss += 1;
        }
        hit
    }

    /// The end-of-run line.
    #[must_use]
    pub fn line(&self, rank: u32) -> String {
        let n = self.steps.max(1) as f64;
        format!(
            "fnoverlap112 rank{rank}: arm on | {} steps queued (staging {:.1} MiB) | seam rows {} raw rows {} route logs {} | picks served {} logs served {} misses {} | reads on the card {:.3} ms a step (the device remainder) | host pick-read + log-read {:.3} ms a step",
            self.steps,
            self.lay.total as f64 / 1048576.0,
            self.seam_rows,
            self.raw_rows,
            self.logs,
            self.used_pick,
            self.used_log,
            self.miss,
            self.dev_ms / n,
            self.host_ms / n,
        )
    }
}

/// The census's class of a named host phase (LEDGER FN112): `I` reads nothing step N writes,
/// `Q` reads step N's bytes at a shape known before step N ends (queueable behind it), `D`
/// depends on step N's committed tokens / values.
#[cfg(test)]
#[must_use]
pub fn phase_class(name: &str) -> char {
    match name {
        "te_fence" | "pre_tier" | "pre_tables" => 'I',
        "te_log" | "pick_read" => 'Q',
        _ => 'D',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_staging_layout_holds_every_region_apart_and_aligned() {
        // the served shape: 8 x 5 rows, 64 x 8 B seam, 48 layers x 40 B log, 124,160 logits a rank
        let l = Layout::of(40, 512, 48, 40, 124_160);
        assert_eq!(l.seam_off, 0);
        assert!(l.log_off >= 40 * 512 && l.log_off % 4096 == 0);
        assert!(l.raw_off >= l.log_off + l.log_bytes(40) && l.raw_off % 4096 == 0);
        assert!(l.total >= l.raw_off + 40 * 124_160 * 2);
        assert_eq!(l.log_bytes(4), 48 * 4 * 40);
        // ~9.6 MiB a rank: inside the serve plane's 4 GiB pinned reserve
        assert!(l.total < 10 << 20);
    }

    #[test]
    fn the_census_classes_name_the_queueable_reads_and_the_rest() {
        assert_eq!(phase_class("pick_read"), 'Q');
        assert_eq!(phase_class("te_log"), 'Q');
        assert_eq!(phase_class("pick_seam"), 'D');
        assert_eq!(phase_class("pick_xchg"), 'D');
        assert_eq!(phase_class("pre_ple"), 'D');
        assert_eq!(phase_class("pre_tier"), 'I');
    }
}
