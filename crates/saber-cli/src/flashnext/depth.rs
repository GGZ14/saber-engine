//! FN92: the b1 ADAPTIVE MTP DEPTH (`--fn-mtp-depth adaptive`, an ARM; `table` = the default =
//! `verify::AUTO_TABLE`'s b1 cell).
//!
//! The law (the reference 1.3.0's adaptive draft window, re-derived for Saber; FN91 priced it): per
//! SEQUENCE a decayed conditional acceptance `a_j` per draft position `j` (each verify that
//! evaluates position `j` moves it with a half-life of [`HALF_LIFE`] evaluations from the prior
//! [`PRIOR_ACC`]); the expected tokens a step commits at depth `d` are
//! `E(d) = 1 + sum_{k <= d} prod_{j <= k} a_j`; per STEP the depth `D` maximises `E(D) / T(D)`
//! with `T` the MEASURED iteration cost at depth `D` (the verify step at width `D + 1` + the
//! drafter round of `D` passes + the host gap between iterations, each an EMA on the leader); `D`
//! moves only for a [`HYST`] gain and [`HOLD`] steps after its last move; the sequence's STOP is the
//! first draft whose emit chance `prod_{j <= d} a_j` is under [`MIN_CHANCE`] (`D <= stop`).
//!
//! BYTES: a b1 verify at any width 2..5 is SPEC 0's kernel class (FN24's width-1 class), so the
//! depth moves how many tokens a step commits, never which. The law is CAUSAL: a decision reads
//! only commits that already happened (the tests' negative control reads the step it decides --
//! the shape a bug would take -- and the causality check must catch it).
//!
//! b >= 2 is untouched: the table decides there (the reference's own cells lose at 2 / 8 sequences).

/// The deepest draft (the verify cap: FN6's width law 1..5 rows -> 4 drafts).
pub const KMAX: usize = 4;
/// Half-life of a position's estimate, in verifies that evaluate it (the reference's 16).
pub const HALF_LIFE: f64 = 16.0;
/// A position nothing has measured (optimistic: a deep start evaluates every position).
pub const PRIOR_ACC: f64 = 0.8;
/// A depth move needs this relative gain in E / T.
pub const HYST: f64 = 0.02;
/// ... and this many steps since the last move.
pub const HOLD: u32 = 8;
/// A sequence stops at the first draft whose emit chance is under this.
pub const MIN_CHANCE: f64 = 0.1;
/// The step-time EMA weight of a new observation.
pub const T_ALPHA: f64 = 0.125;
/// The longest host gap between two b1 iterations that is the iteration's own (FN83: 0.44 ms of
/// host a b1 iteration); a longer one is a request boundary or the scheduler idling.
pub const GAP_MAX_MS: f64 = 10.0;
/// The cost PRIOR of a depth not yet timed, ms at b1 (index = depth 1..4): the verify step at
/// width d + 1 + the drafter round of d passes. FN92 hold h1 (harness, 210 W, -trunk8, FN3's 8
/// prompts x 128 at b1, 2 boots a depth, 2026-10-08): iteration p50 18.86 / 22.03 / 24.38 / 26.82
/// ms = the round's device p50 below + the rest (the verify step, the walk, the host) here. The
/// slope (+2.4..3.2 ms a depth) is what the law trades against E(d): h1's first priors (+1.6 a
/// depth) never left depth 4 because only the CURRENT depth is timed -- only the RATIOS are used
/// once any depth is timed (an untimed depth = the timed depths' mean measured / prior ratio x
/// its prior).
pub const VERIFY_PRIOR_MS: [f64; KMAX] = [18.04, 20.39, 21.91, 23.54];
/// The drafter round prior per depth (passes 1..4), ms (h1 device p50).
pub const ROUND_PRIOR_MS: [f64; KMAX] = [0.82, 1.64, 2.47, 3.28];

/// One sequence's acceptance estimate per draft position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeqAcc {
    a: [f64; KMAX],
    n: [u32; KMAX],
}

impl Default for SeqAcc {
    fn default() -> Self {
        Self {
            a: [PRIOR_ACC; KMAX],
            n: [0; KMAX],
        }
    }
}

/// The per-evaluation decay of a position's estimate (half-life [`HALF_LIFE`]).
fn keep() -> f64 {
    (-std::f64::consts::LN_2 / HALF_LIFE).exp()
}

impl SeqAcc {
    /// One verify: `drafted` drafts proposed, the first `accepted` taken. Positions `1..=accepted`
    /// were evaluated and accepted, position `accepted + 1` (when drafted) evaluated and rejected,
    /// the rest not evaluated.
    pub fn observe(&mut self, drafted: u32, accepted: u32) {
        let k = keep();
        let ev = (accepted + u32::from(accepted < drafted)).min(drafted) as usize;
        for j in 0..ev.min(KMAX) {
            let x = if (j as u32) < accepted { 1.0 } else { 0.0 };
            self.a[j] = k * self.a[j] + (1.0 - k) * x;
            self.n[j] = self.n[j].saturating_add(1);
        }
    }

    /// The conditional acceptance at draft position `j` (1-based).
    #[must_use]
    #[cfg(test)]
    pub fn at(&self, j: usize) -> f64 {
        self.a[j - 1]
    }

    /// Expected tokens a verify at depth `d` commits: `1 + sum_{k <= d} prod_{j <= k} a_j`.
    #[must_use]
    pub fn expect(&self, d: usize) -> f64 {
        let mut e = 1.0;
        let mut p = 1.0;
        for j in 0..d.min(KMAX) {
            p *= self.a[j];
            e += p;
        }
        e
    }

    /// The sequence's stop: the deepest depth whose every draft's emit chance is >= MIN_CHANCE
    /// (at least 1: a sequence at 0 would never evaluate a position again).
    #[must_use]
    pub fn stop(&self) -> usize {
        let mut p = 1.0;
        for j in 0..KMAX {
            p *= self.a[j];
            if p < MIN_CHANCE {
                return j.max(1);
            }
        }
        KMAX
    }
}

/// Why a decision came out as it did (rule 5's `which_path`, one counter per path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Path {
    /// The first decision of the controller (the cold depth = the argmax on the priors).
    Cold,
    /// Inside the hold since the last move: kept.
    Hold,
    /// The best depth is the current one, or beats it by < HYST: kept.
    Keep,
    /// Moved deeper.
    Up,
    /// Moved shallower.
    Down,
    /// The sequence's stop is under the current depth: capped (also past the hold).
    Stop,
}

impl Path {
    pub const ALL: [Path; 6] = [
        Path::Cold,
        Path::Hold,
        Path::Keep,
        Path::Up,
        Path::Down,
        Path::Stop,
    ];
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Path::Cold => "cold",
            Path::Hold => "hold",
            Path::Keep => "keep",
            Path::Up => "up",
            Path::Down => "down",
            Path::Stop => "stop",
        }
    }
}

/// The step half of the law: the measured costs, the current depth, the hold.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthCtl {
    /// The ceiling (the verify cap or the pools' width - 1).
    pub ceil: usize,
    verify_ms: [Option<f64>; KMAX],
    round_ms: [Option<f64>; KMAX],
    gap_ms: Option<f64>,
    d: Option<usize>,
    held: u32,
    /// Decisions by path (rule 5).
    pub paths: [u64; 6],
    /// Steps decided at each depth 1..4.
    pub at: [u64; KMAX],
}

fn ema(slot: &mut Option<f64>, x: f64) {
    if !x.is_finite() || x < 0.0 {
        return;
    }
    *slot = Some(match *slot {
        None => x,
        Some(e) => (1.0 - T_ALPHA) * e + T_ALPHA * x,
    });
}

/// A table of costs with untimed entries priced as `prior x` the timed entries' mean ratio.
fn priced(meas: &[Option<f64>; KMAX], prior: &[f64; KMAX]) -> [f64; KMAX] {
    let r: Vec<f64> = meas
        .iter()
        .zip(prior)
        .filter_map(|(m, p)| m.map(|m| m / p))
        .collect();
    let s = if r.is_empty() {
        1.0
    } else {
        r.iter().sum::<f64>() / r.len() as f64
    };
    let mut o = [0.0; KMAX];
    for i in 0..KMAX {
        o[i] = meas[i].unwrap_or(prior[i] * s);
    }
    o
}

impl DepthCtl {
    #[must_use]
    pub fn new(ceil: usize) -> Self {
        Self {
            ceil: ceil.clamp(1, KMAX),
            verify_ms: [None; KMAX],
            round_ms: [None; KMAX],
            gap_ms: None,
            d: None,
            held: 0,
            paths: [0; 6],
            at: [0; KMAX],
        }
    }

    /// The current depth (the ceiling before the first decision).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.d.unwrap_or(self.ceil)
    }

    /// A b1 verify step at width `w` (depth `w - 1`) took `ms` (leader wall: the step + the walk).
    pub fn time_verify(&mut self, w: u32, ms: f64) {
        if (2..=KMAX as u32 + 1).contains(&w) {
            ema(&mut self.verify_ms[w as usize - 2], ms);
        }
    }

    /// A b1 drafter round of `k` passes took `ms` (leader wall).
    pub fn time_round(&mut self, k: u32, ms: f64) {
        if (1..=KMAX as u32).contains(&k) {
            ema(&mut self.round_ms[k as usize - 1], ms);
        }
    }

    /// The host gap of one iteration (its wall - the verify - the round): shared by every depth.
    /// A gap past [`GAP_MAX_MS`] is idle time between requests, not the iteration's: ignored.
    pub fn time_gap(&mut self, ms: f64) {
        if ms <= GAP_MAX_MS {
            ema(&mut self.gap_ms, ms);
        }
    }

    /// The cost of an iteration at depth `d` (1-based), ms.
    #[must_use]
    pub fn cost(&self, d: usize) -> f64 {
        let v = priced(&self.verify_ms, &VERIFY_PRIOR_MS);
        let r = priced(&self.round_ms, &ROUND_PRIOR_MS);
        v[d - 1] + r[d - 1] + self.gap_ms.unwrap_or(0.0)
    }

    /// Committed tokens per ms at depth `d` for the sequence `s`.
    #[must_use]
    pub fn yield_at(&self, s: &SeqAcc, d: usize) -> f64 {
        s.expect(d) / self.cost(d)
    }

    /// The depth of the NEXT step for the one sequence `s` (its commits so far observed; nothing
    /// of the step this depth will verify).
    pub fn decide(&mut self, s: &SeqAcc) -> (usize, Path) {
        let top = self.ceil.min(s.stop()).max(1);
        let best = (1..=top)
            .max_by(|&x, &y| {
                self.yield_at(s, x)
                    .partial_cmp(&self.yield_at(s, y))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    // a tie keeps the shallower depth
                    .then(y.cmp(&x))
            })
            .unwrap_or(1);
        let (d, p) = match self.d {
            None => (best, Path::Cold),
            Some(cur) if cur > top => (top.min(best.max(1)), Path::Stop),
            Some(cur) if self.held < HOLD => (cur, Path::Hold),
            Some(cur) => {
                if best != cur && self.yield_at(s, best) > (1.0 + HYST) * self.yield_at(s, cur) {
                    (best, if best > cur { Path::Up } else { Path::Down })
                } else {
                    (cur, Path::Keep)
                }
            }
        };
        if self.d != Some(d) {
            self.held = 0;
        } else {
            self.held = self.held.saturating_add(1);
        }
        self.d = Some(d);
        self.paths[Path::ALL.iter().position(|&x| x == p).unwrap_or(0)] += 1;
        self.at[d - 1] += 1;
        (d, p)
    }

    /// Set the depth a peer decided (the follower rank mirrors the leader's decision).
    pub fn follow(&mut self, d: usize) {
        self.d = Some(d.clamp(1, self.ceil));
    }

    /// `fnmtpdepth` boot / end-of-run summary.
    #[must_use]
    pub fn summary(&self) -> String {
        let v = priced(&self.verify_ms, &VERIFY_PRIOR_MS);
        let r = priced(&self.round_ms, &ROUND_PRIOR_MS);
        let paths: Vec<String> = Path::ALL
            .iter()
            .zip(self.paths)
            .map(|(p, n)| format!("{} {n}", p.name()))
            .collect();
        format!(
            "steps by depth 1..4 {:?} | paths {} | verify ms {:?} (timed {:?}) | round ms {:?} | gap {:.2} ms | depth now {}",
            self.at,
            paths.join(", "),
            v.map(|x| (x * 100.0).round() / 100.0),
            self.verify_ms.map(|x| x.is_some()),
            r.map(|x| (x * 100.0).round() / 100.0),
            self.gap_ms.unwrap_or(0.0),
            self.depth()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run a law over an acceptance script: per step, decide the depth from what is known, then
    /// the step at that depth accepts `min(script[i], d)`. `peek` = the negative control (the
    /// step's own outcome read before its depth is fixed).
    fn run(script: &[u32], peek: bool) -> Vec<usize> {
        let mut ctl = DepthCtl::new(KMAX);
        let mut s = SeqAcc::default();
        let mut ds = Vec::new();
        for &x in script {
            let (mut d, _) = ctl.decide(&s);
            if peek {
                // the negative control: the depth read off THIS step's outcome (the verify's
                // acceptance before the verify ran)
                s.observe(KMAX as u32, x);
                ctl.follow(x.max(1) as usize);
                d = ctl.depth();
            }
            ds.push(d);
            if !peek {
                s.observe(d as u32, x.min(d as u32));
            }
        }
        ds
    }

    /// CAUSALITY: changing step i's outcome may change decisions after i, never at or before it.
    fn causal(peek: bool) -> Result<(), String> {
        let base: Vec<u32> = (0..96u32)
            .map(|i| [4, 4, 3, 0, 4, 1, 4, 2][i as usize % 8])
            .collect();
        let d0 = run(&base, peek);
        for i in 0..base.len() {
            for alt in 0..=KMAX as u32 {
                let mut s = base.clone();
                s[i] = alt;
                let d1 = run(&s, peek);
                if d0[..=i] != d1[..=i] {
                    return Err(format!(
                        "step {i}: outcome {alt} moved a decision at or before it"
                    ));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn fn92_the_law_is_causal_and_the_peeking_control_is_caught() {
        assert_eq!(causal(false), Ok(()));
        // the negative control must FAIL the same check
        assert!(
            causal(true).is_err(),
            "the peeking law passed the causality check"
        );
    }

    #[test]
    fn fn92_estimates_follow_the_text() {
        let mut s = SeqAcc::default();
        assert!((s.expect(4) - (1.0 + 0.8 + 0.64 + 0.512 + 0.4096)).abs() < 1e-12);
        // code-like text: every draft accepted -> estimates climb toward 1
        for _ in 0..64 {
            s.observe(4, 4);
        }
        assert!(s.at(4) > 0.95 && s.stop() == KMAX);
        // prose-like: the first draft taken half the time, the second never
        let mut p = SeqAcc::default();
        for i in 0..160 {
            p.observe(4, u32::from(i % 2 == 0));
        }
        assert!(p.at(1) > 0.4 && p.at(1) < 0.6, "{}", p.at(1));
        assert!(p.at(2) < 0.1);
        assert_eq!(p.stop(), 1);
        // an unevaluated position does not move (it was never drafted)
        let mut q = SeqAcc::default();
        q.observe(1, 0);
        assert_eq!(q.at(2), PRIOR_ACC);
        assert_eq!(q.n, [1, 0, 0, 0]);
        // a rejection at position 2 evaluates positions 1 and 2 only
        q.observe(4, 1);
        assert_eq!(q.n, [2, 1, 0, 0]);
    }

    #[test]
    fn fn92_the_depth_follows_yield_with_hold_and_hysteresis() {
        let mut ctl = DepthCtl::new(KMAX);
        // flat costs (each depth costs the same): the deepest depth always yields most
        for w in 2..=5 {
            ctl.time_verify(w, 20.0);
        }
        for k in 1..=4 {
            ctl.time_round(k, 0.0);
        }
        let hi = SeqAcc::default();
        assert_eq!(ctl.decide(&hi), (4, Path::Cold));
        // steeply rising cost: depth 1 wins, but only after the hold
        let mut c2 = ctl.clone();
        for (i, w) in (2..=5).enumerate() {
            for _ in 0..64 {
                c2.time_verify(w, 10.0 + 20.0 * i as f64);
            }
        }
        for i in 0..HOLD {
            assert_eq!(c2.decide(&hi), (4, Path::Hold), "step {i}");
        }
        let (d, p) = c2.decide(&hi);
        assert_eq!(p, Path::Down);
        assert!(d < 4);
        // a sequence whose stop is 1 caps the depth at once (no hold)
        let mut lo = SeqAcc::default();
        for _ in 0..96 {
            lo.observe(4, 0);
        }
        let mut c3 = ctl.clone();
        assert_eq!(c3.decide(&lo), (1, Path::Stop));
        // a near-tie (< HYST) keeps the current depth
        let mut c4 = DepthCtl::new(KMAX);
        let mut s = SeqAcc::default();
        for _ in 0..96 {
            s.observe(4, 4);
        }
        for w in 2..=5 {
            c4.time_verify(
                w,
                10.0 * (s.expect(w as usize - 1)) * (1.0 + 0.001 * f64::from(w)),
            );
        }
        let (d0, _) = c4.decide(&s);
        for _ in 0..3 * HOLD {
            assert_eq!(c4.decide(&s).0, d0);
        }
        assert!(c4.paths[Path::ALL.iter().position(|&x| x == Path::Keep).unwrap()] > 0);
    }

    #[test]
    fn fn92_the_priors_pick_h1s_best_static_depth_per_text() {
        // FN92 h1: per-prompt conditional acceptance (k 4 runs) and the measured best static
        // depth of that text (ms a committed token, 2 boots a depth)
        for (a, best) in [
            ([0.68, 0.55, 0.38, 0.25], 2), // p05 (prose-like: k 2 best)
            ([0.96, 0.96, 0.88, 0.87], 4), // p08
            ([0.85, 0.93, 0.74, 0.80], 4), // p03
            ([1.00, 0.98, 0.96, 0.82], 4), // C05 (code edit)
        ] {
            let mut ctl = DepthCtl::new(KMAX);
            let s = SeqAcc { a, n: [100; KMAX] };
            assert_eq!(ctl.decide(&s).0, best, "{a:?}");
        }
    }

    #[test]
    fn fn92_untimed_depths_take_the_timed_ratio() {
        let mut ctl = DepthCtl::new(KMAX);
        ctl.time_verify(4, VERIFY_PRIOR_MS[2] * 1.5);
        ctl.time_round(3, ROUND_PRIOR_MS[2] * 1.5);
        for d in 1..=4 {
            let want = 1.5 * (VERIFY_PRIOR_MS[d - 1] + ROUND_PRIOR_MS[d - 1]);
            assert!((ctl.cost(d) - want).abs() < 1e-9, "depth {d}");
        }
        ctl.time_gap(2.0);
        ctl.time_gap(500.0); // a request boundary: ignored
        assert!(
            (ctl.cost(1) - (1.5 * (VERIFY_PRIOR_MS[0] + ROUND_PRIOR_MS[0]) + 2.0)).abs() < 1e-9
        );
        // a garbage time is ignored
        ctl.time_verify(4, f64::NAN);
        ctl.time_verify(9, 1.0);
        assert!(
            (ctl.cost(3) - (1.5 * (VERIFY_PRIOR_MS[2] + ROUND_PRIOR_MS[2]) + 2.0)).abs() < 1e-9
        );
    }

    #[test]
    fn fn92_the_width_never_moves_a_committed_token() {
        // a target whose argmax at every position is a fixed function of the prefix; a drafter
        // that is right with a text-dependent rate. Whatever depth the law picks, the greedy walk
        // commits exactly the target's stream (the host half of FN24's byte law).
        let target = |n: usize| -> u32 { ((n * 2_654_435_761) % 1_000) as u32 };
        let drafter_right = |n: usize| -> bool { (n % 7) != 3 && (n % 11) != 5 };
        let mut out: Vec<u32> = Vec::new();
        let mut ctl = DepthCtl::new(KMAX);
        let mut s = SeqAcc::default();
        let mut depths = std::collections::BTreeSet::new();
        while out.len() < 500 {
            let (d, _) = ctl.decide(&s);
            depths.insert(d);
            let n = out.len();
            let drafts: Vec<u32> = (0..d)
                .map(|j| {
                    if drafter_right(n + j) {
                        target(n + j)
                    } else {
                        target(n + j) + 1
                    }
                })
                .collect();
            let am: Vec<u32> = (0..=d).map(|j| target(n + j)).collect();
            let a = super::super::worker::verify::greedy_walk(&drafts, &am);
            out.extend_from_slice(&am[..=a as usize]);
            s.observe(d as u32, a);
            ctl.time_verify(d as u32 + 1, 18.0 + 0.8 * d as f64);
        }
        let want: Vec<u32> = (0..out.len()).map(target).collect();
        assert_eq!(out, want);
        assert!(!depths.is_empty());
    }
}
