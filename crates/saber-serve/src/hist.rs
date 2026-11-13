//! Latency histograms for `/metrics` and `/v1/stats`.
//!
//! **The law this module holds: nothing here runs on the step path.** Every sample is a span
//! between instants the serving plane or the scheduler already took (a request's arrival, its
//! first event, its finish; the step ledger's joined record), recorded with relaxed atomic adds
//! after the fact -- no new clock read inside a step, no lock.
//!
//! Buckets are fixed per family (`*_BOUNDS_US`, microseconds, upper bounds inclusive as
//! Prometheus' `le`), so a snapshot is plain integers (`EngineStats` stays `Eq`) and two
//! snapshots of one family can be compared bucket by bucket. Quantiles are estimated the way
//! PromQL's `histogram_quantile` does: linear interpolation inside the bucket that holds the
//! rank, the first bucket's lower edge 0, and a rank in the `+Inf` bucket reported as the last
//! finite bound.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Time to first token: 10 ms .. 80 s.
pub const TTFT_BOUNDS_US: &[u64] = &[
    10_000, 25_000, 50_000, 75_000, 100_000, 150_000, 200_000, 300_000, 500_000, 750_000,
    1_000_000, 1_500_000, 2_000_000, 3_000_000, 5_000_000, 7_500_000, 10_000_000, 20_000_000,
    40_000_000, 80_000_000,
];

/// Inter-token latency (one SSE update to the next) and the engine step: 2 ms .. 2.5 s, dense
/// around the 15-30 ms band a served step falls in, so a p99 has resolution where it matters.
pub const STEP_BOUNDS_US: &[u64] = &[
    2_000, 5_000, 7_500, 10_000, 12_500, 15_000, 17_500, 20_000, 22_500, 25_000, 30_000, 35_000,
    40_000, 50_000, 60_000, 75_000, 100_000, 150_000, 200_000, 300_000, 500_000, 1_000_000,
    2_500_000,
];

/// End-to-end request latency: 50 ms .. 8 min.
pub const E2E_BOUNDS_US: &[u64] = &[
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_000_000,
    3_000_000,
    5_000_000,
    7_500_000,
    10_000_000,
    15_000_000,
    20_000_000,
    30_000_000,
    45_000_000,
    60_000_000,
    90_000_000,
    120_000_000,
    240_000_000,
    480_000_000,
];

/// Queue wait (arrival -> admission): 0.1 ms .. 60 s.
pub const QUEUE_BOUNDS_US: &[u64] = &[
    100, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000,
    2_500_000, 5_000_000, 10_000_000, 30_000_000, 60_000_000,
];

/// The bucket a sample of `us` falls in: the first bound `>= us`, or `bounds.len()` (`+Inf`).
#[must_use]
pub fn bucket_of(bounds: &[u64], us: u64) -> usize {
    bounds.partition_point(|&b| b < us)
}

/// A shared, lock-free histogram.
#[derive(Debug)]
pub struct Histogram {
    bounds_us: &'static [u64],
    /// Per bucket (NOT cumulative), `bounds.len() + 1` entries, the last one `+Inf`.
    buckets: Box<[AtomicU64]>,
    sum_us: AtomicU64,
    count: AtomicU64,
}

impl Histogram {
    #[must_use]
    pub fn new(bounds_us: &'static [u64]) -> Self {
        debug_assert!(bounds_us.windows(2).all(|w| w[0] < w[1]), "bounds ascend");
        Self {
            bounds_us,
            buckets: (0..=bounds_us.len()).map(|_| AtomicU64::new(0)).collect(),
            sum_us: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    pub fn observe_us(&self, us: u64) {
        self.buckets[bucket_of(self.bounds_us, us)].fetch_add(1, Ordering::Relaxed);
        self.sum_us.fetch_add(us, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn observe(&self, d: Duration) {
        self.observe_us(u64::try_from(d.as_micros()).unwrap_or(u64::MAX));
    }

    /// A sample in milliseconds (the step ledger's unit); a negative or NaN one counts as 0.
    pub fn observe_ms(&self, ms: f64) {
        let us = if ms.is_finite() && ms > 0.0 {
            (ms * 1e3).round() as u64
        } else {
            0
        };
        self.observe_us(us);
    }

    /// Fold a request's local samples in (one add per touched bucket).
    pub fn merge(&self, local: &LocalHist) {
        debug_assert_eq!(local.counts.len(), self.buckets.len());
        for (b, &n) in self.buckets.iter().zip(&local.counts) {
            if n > 0 {
                b.fetch_add(n, Ordering::Relaxed);
            }
        }
        self.sum_us.fetch_add(local.sum_us, Ordering::Relaxed);
        self.count.fetch_add(local.count, Ordering::Relaxed);
    }

    #[must_use]
    pub fn local(&self) -> LocalHist {
        LocalHist {
            bounds_us: self.bounds_us,
            counts: vec![0; self.buckets.len()],
            sum_us: 0,
            count: 0,
        }
    }

    #[must_use]
    pub fn snapshot(&self) -> HistSnapshot {
        HistSnapshot {
            bounds_us: self.bounds_us,
            counts: self
                .buckets
                .iter()
                .map(|b| b.load(Ordering::Relaxed))
                .collect(),
            sum_us: self.sum_us.load(Ordering::Relaxed),
            count: self.count.load(Ordering::Relaxed),
        }
    }
}

/// One request's own samples (its inter-token gaps), folded into the shared histogram once, at
/// the request's end.
#[derive(Debug, Clone)]
pub struct LocalHist {
    bounds_us: &'static [u64],
    counts: Vec<u64>,
    sum_us: u64,
    count: u64,
}

impl LocalHist {
    pub fn observe(&mut self, d: Duration) {
        let us = u64::try_from(d.as_micros()).unwrap_or(u64::MAX);
        self.counts[bucket_of(self.bounds_us, us)] += 1;
        self.sum_us = self.sum_us.saturating_add(us);
        self.count += 1;
    }

    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }
}

/// A histogram's values at one instant: integers only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistSnapshot {
    pub bounds_us: &'static [u64],
    /// Per bucket (not cumulative), `bounds_us.len() + 1` entries.
    pub counts: Vec<u64>,
    pub sum_us: u64,
    pub count: u64,
}

impl HistSnapshot {
    /// An empty snapshot of a family (what a scrape shows before the first sample).
    #[must_use]
    pub fn empty(bounds_us: &'static [u64]) -> Self {
        Self {
            bounds_us,
            counts: vec![0; bounds_us.len() + 1],
            sum_us: 0,
            count: 0,
        }
    }

    /// The `q` quantile (0..=1) in milliseconds, `None` with no samples.
    #[must_use]
    pub fn quantile_ms(&self, q: f64) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let rank = q.clamp(0.0, 1.0) * self.count as f64;
        let mut below = 0u64;
        for (i, &n) in self.counts.iter().enumerate() {
            let upto = below + n;
            if n > 0 && (upto as f64) >= rank {
                let Some(&hi) = self.bounds_us.get(i) else {
                    // the +Inf bucket: the last finite bound, as `histogram_quantile`
                    return self.bounds_us.last().map(|&b| b as f64 / 1e3);
                };
                let lo = if i == 0 { 0 } else { self.bounds_us[i - 1] };
                let frac = ((rank - below as f64) / n as f64).clamp(0.0, 1.0);
                return Some((lo as f64 + (hi - lo) as f64 * frac) / 1e3);
            }
            below = upto;
        }
        self.bounds_us.last().map(|&b| b as f64 / 1e3)
    }

    #[must_use]
    pub fn mean_ms(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum_us as f64 / self.count as f64 / 1e3)
    }

    /// The family in Prometheus text format, in seconds: `name_bucket{..,le=".."}` cumulative,
    /// `name_sum`, `name_count`. `labels` is the inner label list without braces
    /// (`engine="0",model_name="X"`).
    pub fn render(&self, out: &mut String, name: &str, help: &str, labels: &str) {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} histogram");
        let mut cum = 0u64;
        for (i, &n) in self.counts.iter().enumerate() {
            cum += n;
            let le = match self.bounds_us.get(i) {
                Some(&b) => fmt_secs(b),
                None => "+Inf".to_owned(),
            };
            let _ = writeln!(out, "{name}_bucket{{{labels},le=\"{le}\"}} {cum}.0");
        }
        let _ = writeln!(
            out,
            "{name}_sum{{{labels}}} {}",
            fmt_secs_f(self.sum_us as f64 / 1e6)
        );
        let _ = writeln!(out, "{name}_count{{{labels}}} {}.0", self.count);
    }

    /// `{count, mean_ms, p50_ms, p95_ms, p99_ms}` for `/v1/stats`.
    #[must_use]
    pub fn summary_json(&self) -> serde_json::Value {
        let r = |v: Option<f64>| v.map(|x| (x * 1000.0).round() / 1000.0);
        serde_json::json!({
            "count": self.count,
            "mean_ms": r(self.mean_ms()),
            "p50_ms": r(self.quantile_ms(0.50)),
            "p95_ms": r(self.quantile_ms(0.95)),
            "p99_ms": r(self.quantile_ms(0.99)),
        })
    }
}

/// A bound in seconds, shortest exact decimal (`0.0125`, `1`, `80`).
fn fmt_secs(us: u64) -> String {
    let whole = us / 1_000_000;
    let frac = us % 1_000_000;
    if frac == 0 {
        return whole.to_string();
    }
    let f = format!("{frac:06}");
    format!("{whole}.{}", f.trim_end_matches('0'))
}

fn fmt_secs_f(s: f64) -> String {
    let t = format!("{s:.6}");
    let t = t.trim_end_matches('0');
    if let Some(stripped) = t.strip_suffix('.') {
        format!("{stripped}.0")
    } else {
        t.to_owned()
    }
}

/// The step-side pair (the engine step's wall and the queue wait), shared between whoever
/// records them (the Flash-Next step ledger, or the scheduler for an executor without one) and
/// the scrape.
#[derive(Debug)]
pub struct StepHistograms {
    pub step: Histogram,
    pub queue: Histogram,
}

impl Default for StepHistograms {
    fn default() -> Self {
        Self {
            step: Histogram::new(STEP_BOUNDS_US),
            queue: Histogram::new(QUEUE_BOUNDS_US),
        }
    }
}

impl StepHistograms {
    /// A step's wall (ms) -- decode and mixed steps only: a prefill step alone is a prompt's
    /// cost, which TTFT reports, and folding it in would make the step p99 a prompt length.
    pub fn record_step_ms(&self, wall_ms: f64, prefill_alone: bool) {
        if !prefill_alone {
            self.step.observe_ms(wall_ms);
        }
    }

    /// The admitted requests' arrival -> admission waits, microseconds.
    pub fn record_queue_us(&self, lags: &[u32]) {
        for &u in lags {
            self.queue.observe_us(u64::from(u));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const B: &[u64] = &[1_000, 2_000, 4_000];

    #[test]
    fn buckets_are_le_inclusive() {
        assert_eq!(bucket_of(B, 0), 0);
        assert_eq!(
            bucket_of(B, 1_000),
            0,
            "a sample at a bound is in that bucket (le)"
        );
        assert_eq!(bucket_of(B, 1_001), 1);
        assert_eq!(bucket_of(B, 4_000), 2);
        assert_eq!(bucket_of(B, 4_001), 3, "+Inf");
    }

    #[test]
    fn quantiles_interpolate_like_histogram_quantile() {
        let h = Histogram::new(B);
        assert_eq!(
            h.snapshot().quantile_ms(0.5),
            None,
            "no samples, no quantile"
        );
        // 4 samples in (1,2] ms, 4 in (2,4] ms
        for _ in 0..4 {
            h.observe_us(1_500);
            h.observe_us(3_000);
        }
        let s = h.snapshot();
        assert_eq!(s.count, 8);
        assert_eq!(s.counts, vec![0, 4, 4, 0]);
        // rank 4 = the top of bucket 1 -> its upper bound
        assert!((s.quantile_ms(0.5).unwrap() - 2.0).abs() < 1e-9);
        // rank 6 = halfway into bucket 2 (2..4 ms) -> 3 ms
        assert!((s.quantile_ms(0.75).unwrap() - 3.0).abs() < 1e-9);
        // rank 2 = halfway into bucket 1 (1..2) -> 1.5
        assert!((s.quantile_ms(0.25).unwrap() - 1.5).abs() < 1e-9);
        assert!((s.mean_ms().unwrap() - 2.25).abs() < 1e-9);
        // a sample past the last bound: the quantile reports the last finite bound
        let h = Histogram::new(B);
        h.observe_us(9_000);
        assert!((h.snapshot().quantile_ms(0.99).unwrap() - 4.0).abs() < 1e-9);
        // the first bucket's lower edge is 0
        let h = Histogram::new(B);
        h.observe_us(500);
        h.observe_us(500);
        assert!((h.snapshot().quantile_ms(0.5).unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn local_merge_equals_direct_observes() {
        let a = Histogram::new(STEP_BOUNDS_US);
        let b = Histogram::new(STEP_BOUNDS_US);
        let mut l = b.local();
        for ms in [3u64, 14, 22, 22, 41, 900, 4000] {
            a.observe(Duration::from_millis(ms));
            l.observe(Duration::from_millis(ms));
        }
        b.merge(&l);
        assert_eq!(a.snapshot(), b.snapshot());
        assert_eq!(l.count(), 7);
    }

    #[test]
    fn prometheus_rendering_is_cumulative_in_seconds() {
        let h = Histogram::new(&[12_500, 1_000_000]);
        h.observe_us(10_000);
        h.observe_us(500_000);
        h.observe_us(3_000_000);
        let mut out = String::new();
        h.snapshot()
            .render(&mut out, "saber:x_seconds", "X.", "engine=\"0\"");
        assert!(out.contains("# TYPE saber:x_seconds histogram"));
        assert!(out.contains("saber:x_seconds_bucket{engine=\"0\",le=\"0.0125\"} 1.0"));
        assert!(out.contains("saber:x_seconds_bucket{engine=\"0\",le=\"1\"} 2.0"));
        assert!(out.contains("saber:x_seconds_bucket{engine=\"0\",le=\"+Inf\"} 3.0"));
        assert!(
            out.contains("saber:x_seconds_sum{engine=\"0\"} 3.51\n"),
            "{out}"
        );
        assert!(out.contains("saber:x_seconds_count{engine=\"0\"} 3.0"));
    }

    #[test]
    fn step_pair_skips_prefill_alone() {
        let s = StepHistograms::default();
        s.record_step_ms(21.0, false);
        s.record_step_ms(900.0, true);
        s.record_step_ms(f64::NAN, false);
        s.record_queue_us(&[100, 2_000_000]);
        assert_eq!(s.step.snapshot().count, 2);
        assert_eq!(s.queue.snapshot().count, 2);
    }
}
