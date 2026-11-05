//! FN53b: THE MEMORY-PLANE PRESSURE VALVE's controller (`--fn-mem-valve on`; LEDGER FN53a §4
//! / FN53b). A pure state machine: signals in, at most one move out per tick, every move
//! ledgered. The engine samples the signals between steps and applies the move on both ranks.
//!
//! **The law.** RAM PRESSURE = MemAvailable below `floor + act_gib` held for `act_ms`; the
//! moves escalate in a fixed order -- (1) DEMOTE one valve band segment (its coldest records to
//! the drive, [`crate::expert_tier::VALVE_SEG`] a rank) until `max_segs` are closed, (2) PARK
//! idle sequences (when the engine offers it), (3) REFUSE admission by name (when offered) --
//! never swap. RELIEF = MemAvailable above `floor + undo_gib` held for `undo_ms`; the moves undo
//! in the reverse order, one at a time. Between the two thresholds nothing moves (the dead band:
//! no oscillation). At most one move every `min_gap_ms`; after a move its window restarts.
//! Placement only: no move changes a byte the engine computes.
//!
//! **FN53c: the trigger is the BOOT's free floor minus a margin, never a constant.** The served
//! 262k boot pins ~35 of 36.4 pinnable GiB and leaves MemAvailable ~17-19 GiB idle -- AT FN53b's
//! constant `16 + 2` trigger, so the valve would demote at boot and flap. [`MemValve::at_boot`]
//! takes the MemAvailable measured after the boot's pins settled and resolves ([`resolve`]):
//! TRIGGER = `max(floor, boot - margin)` (pressure = the EFFECTIVE free below it: only NEW
//! pressure -- a job eating RAM -- fires it, never the idle boot), REOPEN = `max(trigger +
//! reopen_gap, boot - reopen_below_boot)` (the hysteresis scaled the same way: above one
//! segment pair's bytes, so a reopen cannot re-trigger). The EFFECTIVE free is MemAvailable plus
//! the engine's own RSS growth since the boot sample ([`Signals::own_growth_gib`]): the engine's
//! planned growth (graphs, request buffers) is not pressure the valve answers. [`MemValve::new`]
//! keeps FN53b's constants (`floor + act`, `floor + undo`) as the control.

#![forbid(unsafe_code)]

/// The controller's parameters (FN53a §4's numbers are the defaults).
#[derive(Debug, Clone, PartialEq)]
pub struct ValveParams {
    /// The host floor (GiB of MemAvailable; FN51's 16).
    pub floor_gib: f64,
    /// Pressure below `floor + act_gib`.
    pub act_gib: f64,
    /// Relief above `floor + undo_gib`.
    pub undo_gib: f64,
    /// Pressure must hold this long before a move (ms).
    pub act_ms: u64,
    /// Relief must hold this long before an undo (ms).
    pub undo_ms: u64,
    /// At most one move per this many ms.
    pub min_gap_ms: u64,
    /// Band segments the valve may close (`--fn-mem-valve-max-tail` / 256).
    pub max_segs: u32,
    /// The engine can park idle sequences (stage 2).
    pub park: bool,
    /// The engine can refuse admission by name (stage 3).
    pub refuse: bool,
    /// FN53c: the boot law's margin below the boot's free floor (GiB).
    pub margin_gib: f64,
    /// FN53c: the reopen threshold sits at least this far above the trigger (GiB; more than one
    /// segment pair's 2 x 0.62 GiB, so a reopen never re-triggers).
    pub reopen_gap_gib: f64,
    /// FN53c: ... and this far below the boot's free floor at most (GiB), so the tier reopens
    /// whole once the pressure is gone.
    pub reopen_below_boot_gib: f64,
}

impl Default for ValveParams {
    fn default() -> Self {
        ValveParams {
            floor_gib: 16.0,
            act_gib: 2.0,
            undo_gib: 4.0,
            act_ms: 2_000,
            undo_ms: 30_000,
            min_gap_ms: 1_000,
            max_segs: 4,
            park: false,
            refuse: false,
            margin_gib: 2.0,
            reopen_gap_gib: 1.5,
            reopen_below_boot_gib: 0.5,
        }
    }
}

/// FN53c: the resolved thresholds (GiB of EFFECTIVE free memory).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    /// The boot's MemAvailable after its pins settled (`None` = FN53b's constant law).
    pub boot_gib: Option<f64>,
    /// Pressure below this.
    pub trigger_gib: f64,
    /// Relief above this.
    pub reopen_gib: f64,
}

impl std::fmt::Display for Thresholds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.boot_gib {
            Some(b) => write!(
                f,
                "law boot: boot free {b:.2} GiB -> trigger {:.2} GiB, reopen {:.2} GiB",
                self.trigger_gib, self.reopen_gib
            ),
            None => write!(
                f,
                "law fixed: trigger {:.2} GiB, reopen {:.2} GiB",
                self.trigger_gib, self.reopen_gib
            ),
        }
    }
}

/// FN53c: THE BOOT LAW -- trigger `max(floor, boot - margin)`, reopen `max(trigger + gap, boot -
/// below)`.
#[must_use]
pub fn resolve(p: &ValveParams, boot_gib: f64) -> Thresholds {
    let trigger_gib = p.floor_gib.max(boot_gib - p.margin_gib);
    let reopen_gib = (trigger_gib + p.reopen_gap_gib).max(boot_gib - p.reopen_below_boot_gib);
    Thresholds {
        boot_gib: Some(boot_gib),
        trigger_gib,
        reopen_gib,
    }
}

/// FN53b's constant law (the control): `floor + act`, `floor + undo`.
#[must_use]
pub fn fixed(p: &ValveParams) -> Thresholds {
    Thresholds {
        boot_gib: None,
        trigger_gib: p.floor_gib + p.act_gib,
        reopen_gib: p.floor_gib + p.undo_gib,
    }
}

/// One sample of the signals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Signals {
    /// Monotonic time (ms).
    pub t_ms: u64,
    /// `/proc/meminfo` MemAvailable (GiB).
    pub avail_gib: f64,
    /// Prefill rows over all rows of the recent steps (ledgered with each move).
    pub prefill_frac: f64,
    /// FN53c: the engine's own RSS growth since the boot sample (GiB, >= 0; both ranks): the
    /// effective free is `avail_gib + own_growth_gib`.
    pub own_growth_gib: f64,
}

impl Signals {
    /// FN53c: the free memory the thresholds compare against.
    #[must_use]
    pub fn effective_gib(&self) -> f64 {
        self.avail_gib + self.own_growth_gib.max(0.0)
    }
}

/// One move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    /// Close one more band segment (closed count after the move).
    Demote(u32),
    /// Reopen one band segment (closed count after the move).
    Promote(u32),
    /// Park idle sequences.
    Park,
    /// Unpark them.
    Unpark,
    /// Refuse new admissions by name.
    Refuse,
    /// Admit again.
    Admit,
}

/// A ledgered move.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ledgered {
    /// The move.
    pub mv: Move,
    /// The signals it was decided on.
    pub at: Signals,
}

impl std::fmt::Display for Ledgered {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "fnvalve t_ms {} move {:?} avail {:.2} GiB own_growth {:.2} GiB effective {:.2} GiB prefill_frac {:.2}",
            self.at.t_ms,
            self.mv,
            self.at.avail_gib,
            self.at.own_growth_gib,
            self.at.effective_gib(),
            self.at.prefill_frac
        )
    }
}

/// The controller.
#[derive(Debug, Clone)]
pub struct MemValve {
    p: ValveParams,
    th: Thresholds,
    closed: u32,
    parked: bool,
    refusing: bool,
    low_since: Option<u64>,
    high_since: Option<u64>,
    last_move: Option<u64>,
    ledger: Vec<Ledgered>,
}

impl MemValve {
    /// A closed valve (no segment demoted) on FN53b's CONSTANT thresholds (the control).
    #[must_use]
    pub fn new(p: ValveParams) -> MemValve {
        let th = fixed(&p);
        MemValve::with(p, th)
    }

    /// FN53c: a closed valve on the BOOT law ([`resolve`]): `boot_gib` = MemAvailable measured
    /// after the boot's pins settled.
    #[must_use]
    pub fn at_boot(p: ValveParams, boot_gib: f64) -> MemValve {
        let th = resolve(&p, boot_gib);
        MemValve::with(p, th)
    }

    fn with(p: ValveParams, th: Thresholds) -> MemValve {
        MemValve {
            p,
            th,
            closed: 0,
            parked: false,
            refusing: false,
            low_since: None,
            high_since: None,
            last_move: None,
            ledger: Vec::new(),
        }
    }

    /// Band segments closed now.
    #[must_use]
    pub fn closed(&self) -> u32 {
        self.closed
    }

    /// Whether admissions are refused now.
    #[must_use]
    pub fn refusing(&self) -> bool {
        self.refusing
    }

    /// Whether idle sequences are parked now.
    #[must_use]
    pub fn parked(&self) -> bool {
        self.parked
    }

    /// Every move so far, in order.
    #[must_use]
    pub fn ledger(&self) -> &[Ledgered] {
        &self.ledger
    }

    /// The parameters.
    #[must_use]
    pub fn params(&self) -> &ValveParams {
        &self.p
    }

    /// FN53c: the resolved thresholds.
    #[must_use]
    pub fn thresholds(&self) -> Thresholds {
        self.th
    }

    /// One tick: update the windows from `s`, return the move to apply now (if any). The caller
    /// applies it before the next step (the state already counts it).
    pub fn tick(&mut self, s: Signals) -> Option<Move> {
        let free = s.effective_gib();
        let pressure = free < self.th.trigger_gib;
        let relief = free > self.th.reopen_gib;
        self.low_since = if pressure {
            Some(self.low_since.unwrap_or(s.t_ms))
        } else {
            None
        };
        self.high_since = if relief {
            Some(self.high_since.unwrap_or(s.t_ms))
        } else {
            None
        };
        if self
            .last_move
            .is_some_and(|t| s.t_ms.saturating_sub(t) < self.p.min_gap_ms)
        {
            return None;
        }
        let held = |since: Option<u64>, ms: u64| since.is_some_and(|t| s.t_ms - t >= ms);
        let mv = if held(self.low_since, self.p.act_ms) {
            if self.closed < self.p.max_segs {
                self.closed += 1;
                Some(Move::Demote(self.closed))
            } else if self.p.park && !self.parked {
                self.parked = true;
                Some(Move::Park)
            } else if self.p.refuse && !self.refusing {
                self.refusing = true;
                Some(Move::Refuse)
            } else {
                None
            }
        } else if held(self.high_since, self.p.undo_ms) {
            if self.refusing {
                self.refusing = false;
                Some(Move::Admit)
            } else if self.parked {
                self.parked = false;
                Some(Move::Unpark)
            } else if self.closed > 0 {
                self.closed -= 1;
                Some(Move::Promote(self.closed))
            } else {
                None
            }
        } else {
            None
        };
        if let Some(m) = mv {
            self.last_move = Some(s.t_ms);
            // each further move waits a whole window again
            if self.low_since.is_some() {
                self.low_since = Some(s.t_ms);
            }
            if self.high_since.is_some() {
                self.high_since = Some(s.t_ms);
            }
            self.ledger.push(Ledgered { mv: m, at: s });
        }
        mv
    }
}

/// FN53c: `VmRSS` (GiB) from the text of a `/proc/<pid>/status`.
#[must_use]
pub fn vm_rss_gib(status: &str) -> Option<f64> {
    let line = status.lines().find(|l| l.starts_with("VmRSS:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1_048_576.0)
}

/// `MemAvailable` (GiB) from the text of `/proc/meminfo`.
#[must_use]
pub fn mem_available_gib(meminfo: &str) -> Option<f64> {
    let line = meminfo.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1_048_576.0)
}
