//! The metrics scraper: polls the engine's Prometheus `/metrics`, derives rates, and keeps an
//! in-memory ring (24 h at 5 s by default) for the dashboard's JSON and SSE endpoints.
//!
//! Names are read without the exporter's `saber:` prefix, so `saber:generation_tokens_total`
//! is the series `generation_tokens_total` here.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;
use std::time::Duration;

use serde::Serialize;
use tokio::sync::{broadcast, watch};

/// One parsed exposition: `name{labels}` -> value, plus per-name sums over label sets.
#[derive(Clone, Debug, Default)]
pub struct Scrape {
    pub series: BTreeMap<String, f64>,
    pub sums: BTreeMap<String, f64>,
}

impl Scrape {
    #[must_use]
    pub fn get(&self, name: &str) -> Option<f64> {
        self.sums.get(name).copied()
    }
}

/// Strip the exporter prefix (`saber:`).
fn canonical(name: &str) -> &str {
    name.strip_prefix("saber:").unwrap_or(name)
}

/// Parse the Prometheus text format (the subset the engine writes: no timestamps, no escapes
/// inside label values that contain `}`).
#[must_use]
pub fn parse(text: &str) -> Scrape {
    let mut s = Scrape::default();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, rest) = match line.find('}') {
            Some(i) if line[..i].contains('{') => (&line[..=i], line[i + 1..].trim()),
            _ => match line.split_once(char::is_whitespace) {
                Some((k, r)) => (k, r.trim()),
                None => continue,
            },
        };
        let Some(v) = rest
            .split_whitespace()
            .next()
            .and_then(|v| v.parse::<f64>().ok())
        else {
            continue;
        };
        let name = key.split('{').next().unwrap_or(key);
        let cname = canonical(name).to_string();
        let ckey = format!("{cname}{}", &key[name.len()..]);
        *s.sums.entry(cname).or_insert(0.0) += v;
        s.series.insert(ckey, v);
    }
    s
}

/// One point of the ring. `None` = no value (engine down, first scrape, counter reset).
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct Point {
    /// Unix seconds.
    pub t: f64,
    pub up: bool,
    pub gen_tps: Option<f64>,
    pub prompt_tps: Option<f64>,
    pub running: Option<f64>,
    pub waiting: Option<f64>,
    /// Accepted draft tokens / drafted tokens over the interval.
    pub accept_rate: Option<f64>,
    /// Accepted tokens per draft over the interval.
    pub accepted_per_draft: Option<f64>,
    /// Finished requests per second.
    pub req_rate: Option<f64>,
    /// TTFT p50 / p99 and the engine step's p99 over the interval, in ms, from
    /// the engine's histogram buckets (`saber:time_to_first_token_seconds`,
    /// `saber:step_time_seconds`). `None` with no sample in the interval (or an older engine).
    pub ttft_p50_ms: Option<f64>,
    pub ttft_p99_ms: Option<f64>,
    pub step_p99_ms: Option<f64>,
    /// More interval quantiles (ms) for the throughput / latency page.
    pub ttft_p95_ms: Option<f64>,
    pub step_p50_ms: Option<f64>,
    pub itl_p50_ms: Option<f64>,
    pub itl_p99_ms: Option<f64>,
    pub e2e_p50_ms: Option<f64>,
    pub e2e_p99_ms: Option<f64>,
    pub queue_p99_ms: Option<f64>,
    /// Total GPU power (W) sampled by the console (sysfs), with or without an engine.
    pub gpu_power_w: Option<f64>,
}

/// Columnar form of a window of points (what uPlot takes).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Columns {
    pub interval_s: u64,
    pub t: Vec<f64>,
    pub up: Vec<bool>,
    pub gen_tps: Vec<Option<f64>>,
    pub prompt_tps: Vec<Option<f64>>,
    pub running: Vec<Option<f64>>,
    pub waiting: Vec<Option<f64>>,
    pub accept_rate: Vec<Option<f64>>,
    pub accepted_per_draft: Vec<Option<f64>>,
    pub req_rate: Vec<Option<f64>>,
    pub ttft_p50_ms: Vec<Option<f64>>,
    pub ttft_p99_ms: Vec<Option<f64>>,
    pub step_p99_ms: Vec<Option<f64>>,
    pub ttft_p95_ms: Vec<Option<f64>>,
    pub step_p50_ms: Vec<Option<f64>>,
    pub itl_p50_ms: Vec<Option<f64>>,
    pub itl_p99_ms: Vec<Option<f64>>,
    pub e2e_p50_ms: Vec<Option<f64>>,
    pub e2e_p99_ms: Vec<Option<f64>>,
    pub queue_p99_ms: Vec<Option<f64>>,
    pub gpu_power_w: Vec<Option<f64>>,
}

/// A histogram family's cumulative buckets `(le, count)` in `le` order (`+Inf` last), summed over
/// label sets other than `le`. `name` is canonical (no exporter prefix).
#[must_use]
pub fn hist_buckets(s: &Scrape, name: &str) -> Vec<(f64, f64)> {
    let head = format!("{name}_bucket{{");
    let mut by: BTreeMap<u64, (f64, f64)> = BTreeMap::new();
    for (k, v) in s.series.range(head.clone()..) {
        if !k.starts_with(&head) {
            break;
        }
        let Some(le) = k
            .split("le=\"")
            .nth(1)
            .and_then(|r| r.split('"').next())
            .and_then(|x| {
                if x == "+Inf" {
                    Some(f64::INFINITY)
                } else {
                    x.parse::<f64>().ok()
                }
            })
        else {
            continue;
        };
        by.entry(le.to_bits()).or_insert((le, 0.0)).1 += v;
    }
    let mut out: Vec<(f64, f64)> = by.into_values().collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

/// PromQL's `histogram_quantile` over cumulative buckets (seconds in, ms out): linear inside the
/// bucket holding the rank, a rank in `+Inf` reported as the last finite bound.
#[must_use]
pub fn bucket_quantile_ms(q: f64, buckets: &[(f64, f64)]) -> Option<f64> {
    let total = buckets.last()?.1;
    if total <= 0.0 {
        return None;
    }
    let rank = q.clamp(0.0, 1.0) * total;
    let (mut lo, mut below) = (0.0, 0.0);
    for &(le, cum) in buckets {
        if cum >= rank && cum > below {
            if le.is_infinite() {
                return Some(lo * 1e3);
            }
            let frac = ((rank - below) / (cum - below)).clamp(0.0, 1.0);
            return Some((lo + (le - lo) * frac) * 1e3);
        }
        if le.is_finite() {
            lo = le;
        }
        below = cum;
    }
    Some(lo * 1e3)
}

/// The interval's quantile: the bucket deltas between two scrapes (a reset or an absent family
/// gives `None`).
fn interval_quantile_ms(q: f64, cur: &Scrape, prev: &Scrape, name: &str) -> Option<f64> {
    let c = hist_buckets(cur, name);
    let p = hist_buckets(prev, name);
    if c.is_empty() || c.len() != p.len() {
        return None;
    }
    let mut d = Vec::with_capacity(c.len());
    for (a, b) in c.iter().zip(&p) {
        if a.0.to_bits() != b.0.to_bits() || a.1 < b.1 {
            return None;
        }
        d.push((a.0, a.1 - b.1));
    }
    bucket_quantile_ms(q, &d)
}

fn rate(cur: Option<f64>, prev: Option<f64>, dt: f64) -> Option<f64> {
    match (cur, prev) {
        (Some(c), Some(p)) if c >= p && dt > 0.0 => Some((c - p) / dt),
        _ => None,
    }
}

fn ratio(
    num: Option<f64>,
    nprev: Option<f64>,
    den: Option<f64>,
    dprev: Option<f64>,
) -> Option<f64> {
    let dn = num? - nprev?;
    let dd = den? - dprev?;
    (dn >= 0.0 && dd > 0.0).then(|| dn / dd)
}

/// Derive a point from this scrape and the previous one (`dt` seconds apart).
#[must_use]
pub fn derive(t: f64, cur: &Scrape, prev: Option<(&Scrape, f64)>) -> Point {
    let mut p = Point {
        t,
        up: true,
        running: cur.get("num_requests_running"),
        waiting: cur.get("num_requests_waiting"),
        ..Point::default()
    };
    if let Some((pr, pt)) = prev {
        let dt = t - pt;
        let g = |n: &str| (cur.get(n), pr.get(n));
        let (a, b) = g("generation_tokens_total");
        p.gen_tps = rate(a, b, dt);
        let (a, b) = g("prompt_tokens_total");
        p.prompt_tps = rate(a, b, dt);
        let (a, b) = g("request_success_total");
        p.req_rate = rate(a, b, dt);
        let (acc, accp) = g("spec_decode_num_accepted_tokens_total");
        let (dtk, dtkp) = g("spec_decode_num_draft_tokens_total");
        let (dr, drp) = g("spec_decode_num_drafts_total");
        p.accept_rate = ratio(acc, accp, dtk, dtkp);
        p.accepted_per_draft = ratio(acc, accp, dr, drp);
        let ttft = "time_to_first_token_seconds";
        p.ttft_p50_ms = interval_quantile_ms(0.50, cur, pr, ttft);
        p.ttft_p99_ms = interval_quantile_ms(0.99, cur, pr, ttft);
        p.step_p99_ms = interval_quantile_ms(0.99, cur, pr, "step_time_seconds");
        p.ttft_p95_ms = interval_quantile_ms(0.95, cur, pr, ttft);
        p.step_p50_ms = interval_quantile_ms(0.50, cur, pr, "step_time_seconds");
        let itl = "inter_token_latency_seconds";
        p.itl_p50_ms = interval_quantile_ms(0.50, cur, pr, itl);
        p.itl_p99_ms = interval_quantile_ms(0.99, cur, pr, itl);
        let e2e = "e2e_request_latency_seconds";
        p.e2e_p50_ms = interval_quantile_ms(0.50, cur, pr, e2e);
        p.e2e_p99_ms = interval_quantile_ms(0.99, cur, pr, e2e);
        p.queue_p99_ms = interval_quantile_ms(0.99, cur, pr, "request_queue_time_seconds");
    }
    p
}

// ---------------------------------------------------------------------------------------------
// The per-family series ring (memory plane, spec decode, histogram buckets, hw)

/// The families (canonical names) the series ring keeps, by prefix. Per-layer families are
/// left out (hundreds of series).
pub const WATCH: &[&str] = &[
    "kv_blocks_total",
    "kv_blocks_free",
    "kv_reserved_tokens",
    "kv_max_reserved_tokens",
    "kv_length_tokens",
    "kv_lazy_",
    "gpu_cache_usage_perc",
    "admission_refused_kv",
    "saber_fn_kv_",
    "saber_xtier_routed_total",
    "saber_xtier_moves_total",
    "saber_xtier_bytes_total",
    "saber_xtier_records",
    "saber_xtier_elastic_",
    "saber_ple_rows_total",
    "saber_ple_bytes_read_total",
    "saber_ledger_",
    "saber_fn_ptree_hits_total",
    "saber_fn_ptree_reloaded_tokens_total",
    "saber_fn_ptree_prompt_tokens_total",
    "spec_decode_num_accepted_tokens_per_pos_total",
    "spec_decode_num_drafts_total",
    "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total",
    "saber_fn_mtp_accepted_per_pos_total",
    "saber_fn_mtp_evaluated_per_pos_total",
    "request_success_total",
    "time_to_first_token_seconds_bucket",
    "step_time_seconds_bucket",
    "inter_token_latency_seconds_bucket",
    "e2e_request_latency_seconds_bucket",
    "request_queue_time_seconds_bucket",
    "hw_",
];

/// The most series the ring holds (new ones past it are dropped).
pub const MAX_SERIES: usize = 512;

fn watched(name: &str) -> bool {
    !name.contains("_layer")
        && !name.contains("_plane_")
        && WATCH.iter().any(|w| name.starts_with(w))
}

/// A counter family (its series are stored as per-second rates).
#[must_use]
pub fn is_counter(name: &str) -> bool {
    name.ends_with("_total")
        || name.ends_with("_bucket")
        || name.ends_with("_count")
        || name.ends_with("_sum")
}

/// `name{labels}` with the `engine` and `model_name` labels dropped (one engine, one model).
#[must_use]
pub fn short_key(key: &str) -> String {
    let Some(open) = key.find('{') else {
        return key.to_string();
    };
    let name = &key[..open];
    let body = key[open + 1..].trim_end_matches('}');
    let mut labels = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    let mut esc = false;
    for ch in body.chars() {
        if esc {
            cur.push(ch);
            esc = false;
            continue;
        }
        match ch {
            '\\' => {
                cur.push(ch);
                esc = true;
            }
            '"' => {
                cur.push(ch);
                quoted = !quoted;
            }
            ',' if !quoted => labels.push(std::mem::take(&mut cur)),
            _ => cur.push(ch),
        }
    }
    if !cur.is_empty() {
        labels.push(cur);
    }
    let kept: Vec<String> = labels
        .into_iter()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.starts_with("engine=") && !l.starts_with("model_name=") && !l.is_empty())
        .collect();
    if kept.is_empty() {
        name.to_string()
    } else {
        format!("{name}{{{}}}", kept.join(","))
    }
}

/// The watched series of a scrape, keyed short, summed over the dropped labels.
#[must_use]
pub fn watched_values(s: &Scrape) -> BTreeMap<String, f64> {
    let mut o = BTreeMap::new();
    for (k, v) in &s.series {
        let name = k.split('{').next().unwrap_or(k);
        if watched(name) {
            *o.entry(short_key(k)).or_insert(0.0) += v;
        }
    }
    o
}

/// Counters as rates (per second over `dt`), gauges as values. A counter without a previous
/// value, or one that went down (a restart), has no value this interval.
#[must_use]
pub fn derive_series(
    cur: &BTreeMap<String, f64>,
    prev: Option<(&BTreeMap<String, f64>, f64)>,
) -> BTreeMap<String, f64> {
    let mut o = BTreeMap::new();
    for (k, v) in cur {
        let name = k.split('{').next().unwrap_or(k);
        if is_counter(name) {
            if let Some((p, dt)) = prev {
                if let Some(pv) = p.get(k) {
                    if *v >= *pv && dt > 0.0 {
                        o.insert(k.clone(), (v - pv) / dt);
                    }
                }
            }
        } else {
            o.insert(k.clone(), *v);
        }
    }
    o
}

/// Columnar ring of named series aligned on one time axis (f32 values, NaN = no value).
pub struct SeriesRing {
    t: VecDeque<f64>,
    cols: BTreeMap<String, VecDeque<f32>>,
    cap: usize,
}

/// `/api/series`.
#[derive(Clone, Debug, Default, Serialize)]
pub struct SeriesWindow {
    pub interval_s: u64,
    pub t: Vec<f64>,
    pub series: BTreeMap<String, Vec<Option<f64>>>,
}

impl SeriesRing {
    #[must_use]
    pub fn new(cap: usize) -> SeriesRing {
        SeriesRing {
            t: VecDeque::new(),
            cols: BTreeMap::new(),
            cap: cap.max(1),
        }
    }

    pub fn push(&mut self, t: f64, vals: &BTreeMap<String, f64>) {
        let n = self.t.len();
        for k in vals.keys() {
            if !self.cols.contains_key(k) && self.cols.len() < MAX_SERIES {
                self.cols
                    .insert(k.clone(), std::iter::repeat_n(f32::NAN, n).collect());
            }
        }
        self.t.push_back(t);
        for (k, col) in &mut self.cols {
            col.push_back(vals.get(k).map_or(f32::NAN, |v| *v as f32));
        }
        if self.t.len() > self.cap {
            self.t.pop_front();
            for col in self.cols.values_mut() {
                col.pop_front();
            }
        }
        // a series with no value left in the ring is forgotten
        self.cols.retain(|_, c| c.iter().any(|v| !v.is_nan()));
    }

    /// The last `seconds`, thinned from the end to `max_points`, for series whose family name
    /// starts with one of `prefixes` (all when empty).
    #[must_use]
    pub fn window(
        &self,
        seconds: f64,
        max_points: usize,
        prefixes: &[&str],
        interval_s: u64,
    ) -> SeriesWindow {
        let end = self.t.back().copied().unwrap_or(0.0);
        let start = self
            .t
            .iter()
            .position(|t| *t >= end - seconds)
            .unwrap_or(self.t.len());
        let len = self.t.len() - start;
        let step = len.div_ceil(max_points.max(1)).max(1);
        let mut idx: Vec<usize> = (start..self.t.len()).rev().step_by(step).collect();
        idx.reverse();
        let mut w = SeriesWindow {
            interval_s: interval_s * step as u64,
            t: idx.iter().map(|&i| self.t[i]).collect(),
            series: BTreeMap::new(),
        };
        for (k, col) in &self.cols {
            let name = k.split('{').next().unwrap_or(k);
            if !prefixes.is_empty() && !prefixes.iter().any(|p| name.starts_with(p)) {
                continue;
            }
            let v: Vec<Option<f64>> = idx
                .iter()
                .map(|&i| {
                    let x = col[i];
                    (!x.is_nan()).then_some(f64::from(x))
                })
                .collect();
            if v.iter().any(Option::is_some) {
                w.series.insert(k.clone(), v);
            }
        }
        w
    }
}

pub struct MetricsStore {
    ring: Mutex<VecDeque<Point>>,
    latest: Mutex<Option<(f64, Scrape)>>,
    series: Mutex<SeriesRing>,
    hw: Mutex<Option<(f64, crate::hw::HwSample)>>,
    cap: usize,
    pub interval_s: u64,
    tx: broadcast::Sender<Point>,
}

impl MetricsStore {
    #[must_use]
    pub fn new(cap: usize, interval_s: u64) -> MetricsStore {
        let (tx, _) = broadcast::channel(256);
        MetricsStore {
            ring: Mutex::new(VecDeque::with_capacity(cap.min(20_000))),
            latest: Mutex::new(None),
            series: Mutex::new(SeriesRing::new(cap)),
            hw: Mutex::new(None),
            cap: cap.max(1),
            interval_s,
            tx,
        }
    }

    pub fn push(&self, p: Point) {
        {
            let mut r = self.ring.lock().unwrap_or_else(|e| e.into_inner());
            if r.len() == self.cap {
                r.pop_front();
            }
            r.push_back(p.clone());
        }
        let _ = self.tx.send(p);
    }

    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<Point> {
        self.tx.subscribe()
    }

    /// The raw series of the last scrape (unix seconds, `name{labels}` -> value).
    #[must_use]
    pub fn latest(&self) -> Option<(f64, BTreeMap<String, f64>)> {
        let g = self.latest.lock().unwrap_or_else(|e| e.into_inner());
        g.as_ref().map(|(t, s)| (*t, s.series.clone()))
    }

    /// The per-family series (see [`WATCH`]).
    #[must_use]
    pub fn series_window(
        &self,
        seconds: f64,
        max_points: usize,
        prefixes: &[&str],
    ) -> SeriesWindow {
        self.series
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .window(seconds, max_points, prefixes, self.interval_s)
    }

    pub fn push_series(&self, t: f64, vals: &BTreeMap<String, f64>) {
        self.series
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(t, vals);
    }

    /// The last hardware sample (unix seconds).
    #[must_use]
    pub fn hw_latest(&self) -> Option<(f64, crate::hw::HwSample)> {
        self.hw.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// The last `seconds` of the ring, thinned to at most `max_points`.
    #[must_use]
    pub fn window(&self, seconds: f64, max_points: usize) -> Columns {
        let r = self.ring.lock().unwrap_or_else(|e| e.into_inner());
        let end = r.back().map_or(0.0, |p| p.t);
        let pts: Vec<&Point> = r.iter().filter(|p| p.t >= end - seconds).collect();
        let step = pts.len().div_ceil(max_points.max(1)).max(1);
        let mut c = Columns {
            interval_s: self.interval_s * step as u64,
            ..Columns::default()
        };
        // Keep the newest point exactly: thin from the end.
        let mut sel: Vec<&Point> = pts.iter().rev().step_by(step).copied().collect();
        sel.reverse();
        for p in sel {
            c.t.push(p.t);
            c.up.push(p.up);
            c.gen_tps.push(p.gen_tps);
            c.prompt_tps.push(p.prompt_tps);
            c.running.push(p.running);
            c.waiting.push(p.waiting);
            c.accept_rate.push(p.accept_rate);
            c.accepted_per_draft.push(p.accepted_per_draft);
            c.req_rate.push(p.req_rate);
            c.ttft_p50_ms.push(p.ttft_p50_ms);
            c.ttft_p99_ms.push(p.ttft_p99_ms);
            c.step_p99_ms.push(p.step_p99_ms);
            c.ttft_p95_ms.push(p.ttft_p95_ms);
            c.step_p50_ms.push(p.step_p50_ms);
            c.itl_p50_ms.push(p.itl_p50_ms);
            c.itl_p99_ms.push(p.itl_p99_ms);
            c.e2e_p50_ms.push(p.e2e_p50_ms);
            c.e2e_p99_ms.push(p.e2e_p99_ms);
            c.queue_p99_ms.push(p.queue_p99_ms);
            c.gpu_power_w.push(p.gpu_power_w);
        }
        c
    }
}

/// What the scrape loop needs besides the store.
pub struct ScrapeCtx {
    /// The engine's port while its process is alive.
    pub port: watch::Receiver<Option<u16>>,
    /// The Bearer key for the engine's `/metrics` (engine auth with `--api-auth-all`).
    pub bearer: std::sync::Arc<dyn Fn() -> Option<String> + Send + Sync>,
    /// The hardware sampler's filesystem root (`/`).
    pub hw_root: std::path::PathBuf,
    /// The usage store, for the sampled GPU energy.
    pub usage: Option<std::sync::Arc<crate::usage::UsageStore>>,
}

/// The scrape loop. `port` carries the engine's port while its process is alive (`None` when
/// stopped: the loop then records "down" points so the charts show the gap). The hardware is
/// sampled every tick whether or not an engine runs.
pub async fn run(store: std::sync::Arc<MetricsStore>, mut ctx: ScrapeCtx) {
    let mut tick = tokio::time::interval(Duration::from_secs(store.interval_s.max(1)));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut prev: Option<(Scrape, f64)> = None;
    let mut prev_watch: Option<(BTreeMap<String, f64>, f64)> = None;
    let sampler = std::sync::Arc::new(Mutex::new(crate::hw::HwSampler::new(ctx.hw_root.clone())));
    let mut prev_power: Option<(f64, f64)> = None;
    loop {
        tick.tick().await;
        let t = crate::now_ms() as f64 / 1000.0;
        let cur_port = *ctx.port.borrow_and_update();
        let bearer = (ctx.bearer)();
        let got = match cur_port {
            Some(pt) => {
                crate::httpc::get_auth(pt, "/metrics", Duration::from_secs(2), bearer.as_deref())
                    .await
                    .ok()
                    .filter(|(st, _)| *st == 200)
                    .map(|(_, body)| parse(&body))
            }
            None => None,
        };
        let smp = sampler.clone();
        let hw = tokio::task::spawn_blocking(move || {
            smp.lock().unwrap_or_else(|e| e.into_inner()).sample()
        })
        .await
        .ok();
        let power = hw.as_ref().and_then(crate::hw::HwSample::total_power_w);
        // energy: trapezoid between samples, an interval of at most 3 ticks
        if let (Some(u), Some(p)) = (&ctx.usage, power) {
            if let Some((pt, pp)) = prev_power {
                let dt = t - pt;
                if dt > 0.0 && dt <= 3.0 * store.interval_s.max(1) as f64 {
                    u.send(crate::usage::Op::Energy {
                        ts_ms: (t * 1e3) as i64,
                        joules: (p + pp) / 2.0 * dt,
                    });
                }
            }
        }
        prev_power = power.map(|p| (t, p));
        let mut watch_now = got.as_ref().map(watched_values).unwrap_or_default();
        let mut series = derive_series(&watch_now, prev_watch.as_ref().map(|(m, pt)| (m, t - pt)));
        if let Some(h) = &hw {
            for (k, v) in h.series() {
                series.insert(k, v);
            }
            *store.hw.lock().unwrap_or_else(|e| e.into_inner()) = Some((t, h.clone()));
        }
        store.push_series(t, &series);
        match got {
            Some(s) => {
                let mut p = derive(t, &s, prev.as_ref().map(|(s, t)| (s, *t)));
                p.gpu_power_w = power;
                store.push(p);
                *store.latest.lock().unwrap_or_else(|e| e.into_inner()) = Some((t, s.clone()));
                prev = Some((s, t));
                prev_watch = Some((std::mem::take(&mut watch_now), t));
            }
            None => {
                store.push(Point {
                    t,
                    gpu_power_w: power,
                    ..Point::default()
                });
                prev = None;
                prev_watch = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "# HELP saber:num_requests_running x\n# TYPE saber:num_requests_running gauge\n\
saber:num_requests_running{model_name=\"Q\"} 2.0\nsaber:num_requests_waiting{model_name=\"Q\"} 1.0\n\
saber:generation_tokens_total{model_name=\"Q\"} 1000.0\nsaber:prompt_tokens_total{model_name=\"Q\"} 500.0\n\
saber:spec_decode_num_drafts_total{model_name=\"Q\"} 100.0\n\
saber:spec_decode_num_draft_tokens_total{model_name=\"Q\"} 400.0\n\
saber:spec_decode_num_accepted_tokens_total{model_name=\"Q\"} 200.0\n\
saber:request_success_total{finished_reason=\"stop\"} 3\nsaber:request_success_total{finished_reason=\"length\"} 1\n";

    #[test]
    fn parses_prefixes_labels_and_sums() {
        let s = parse(A);
        assert_eq!(s.get("num_requests_running"), Some(2.0));
        assert_eq!(s.get("request_success_total"), Some(4.0));
        assert_eq!(
            s.series
                .get("request_success_total{finished_reason=\"stop\"}"),
            Some(&3.0)
        );
    }

    #[test]
    fn derives_rates_and_survives_resets() {
        let a = parse(A);
        let b = parse(
            &A.replace("} 1000.0", "} 1500.0")
                .replace("400.0", "600.0")
                .replace("} 200.0", "} 350.0")
                .replace(
                    "drafts_total{model_name=\"Q\"} 100.0",
                    "drafts_total{model_name=\"Q\"} 150.0",
                ),
        );
        let p = derive(105.0, &b, Some((&a, 100.0)));
        assert_eq!(p.gen_tps, Some(100.0));
        assert_eq!(p.prompt_tps, Some(0.0));
        assert_eq!(p.accept_rate, Some(150.0 / 200.0));
        assert_eq!(p.accepted_per_draft, Some(3.0));
        // a restart resets the counters: no negative rate
        let p = derive(110.0, &a, Some((&b, 105.0)));
        assert_eq!(p.gen_tps, None);
        assert_eq!(p.running, Some(2.0));
    }

    /// The engine's histograms become interval quantiles (bucket deltas
    /// between two scrapes), summed over label sets, with the reset and absent-family cases.
    #[test]
    fn interval_quantiles_from_histogram_buckets() {
        let h = |c1: u64, c2: u64, inf: u64| {
            format!(
                "saber:time_to_first_token_seconds_bucket{{engine=\"0\",model_name=\"Q\",le=\"0.1\"}} {c1}.0\n\
                 saber:time_to_first_token_seconds_bucket{{engine=\"0\",model_name=\"Q\",le=\"0.2\"}} {c2}.0\n\
                 saber:time_to_first_token_seconds_bucket{{engine=\"0\",model_name=\"Q\",le=\"+Inf\"}} {inf}.0\n\
                 saber:step_time_seconds_bucket{{engine=\"0\",model_name=\"Q\",le=\"0.02\"}} {c1}.0\n\
                 saber:step_time_seconds_bucket{{engine=\"0\",model_name=\"Q\",le=\"+Inf\"}} {c1}.0\n"
            )
        };
        let a = parse(&h(10, 10, 10));
        let b = parse(&h(10, 14, 14));
        let bk = hist_buckets(&b, "time_to_first_token_seconds");
        assert_eq!(bk, vec![(0.1, 10.0), (0.2, 14.0), (f64::INFINITY, 14.0)]);
        // 4 new samples, all in (0.1, 0.2]: p50 = rank 2 of 4 -> 150 ms
        let p = derive(105.0, &b, Some((&a, 100.0)));
        assert!((p.ttft_p50_ms.unwrap() - 150.0).abs() < 1e-9, "{p:?}");
        assert!((p.ttft_p99_ms.unwrap() - 199.0).abs() < 1e-6, "{p:?}");
        assert_eq!(p.step_p99_ms, None, "no step sample in the interval");
        // a reset (counts went down) is no value, not a negative bucket
        assert_eq!(derive(110.0, &a, Some((&b, 105.0))).ttft_p50_ms, None);
        // an engine without the family
        let old = parse(A);
        assert_eq!(derive(110.0, &old, Some((&old, 105.0))).ttft_p99_ms, None);
        // past the last bound: the last finite bound
        assert_eq!(
            bucket_quantile_ms(0.99, &[(0.1, 0.0), (f64::INFINITY, 3.0)]),
            Some(100.0)
        );
    }

    #[test]
    fn series_keys_rates_and_ring() {
        assert_eq!(
            short_key("saber_xtier_routed_total{rank=\"0\",path=\"ram\"}"),
            "saber_xtier_routed_total{rank=\"0\",path=\"ram\"}"
        );
        assert_eq!(
            short_key("spec_decode_num_accepted_tokens_per_pos_total{engine=\"0\",model_name=\"a,b\",position=\"2\"}"),
            "spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"}"
        );
        assert_eq!(short_key("kv_blocks_free{engine=\"0\"}"), "kv_blocks_free");
        let text = "saber:kv_blocks_free{engine=\"0\"} 10\n\
            saber_xtier_routed_total{rank=\"0\",path=\"ram\"} 100\n\
            saber_xtier_layer_routed_total{rank=\"0\",layer=\"1\",path=\"ram\"} 9\n\
            saber:num_requests_running 1\n";
        let a = watched_values(&parse(text));
        assert_eq!(a.len(), 2, "{a:?}");
        let b = watched_values(&parse(&text.replace("} 100", "} 150")));
        let d = derive_series(&b, Some((&a, 5.0)));
        assert_eq!(d["saber_xtier_routed_total{rank=\"0\",path=\"ram\"}"], 10.0);
        assert_eq!(d["kv_blocks_free"], 10.0);
        assert!(!derive_series(&b, None)
            .contains_key("saber_xtier_routed_total{rank=\"0\",path=\"ram\"}"));
        let mut r = SeriesRing::new(3);
        for i in 0..5 {
            let mut m = BTreeMap::new();
            m.insert("g".to_string(), f64::from(i));
            if i == 4 {
                m.insert("late".to_string(), 1.0);
            }
            r.push(f64::from(i), &m);
        }
        let w = r.window(1e9, 100, &[], 5);
        assert_eq!(w.t, vec![2.0, 3.0, 4.0]);
        assert_eq!(w.series["g"], vec![Some(2.0), Some(3.0), Some(4.0)]);
        assert_eq!(w.series["late"], vec![None, None, Some(1.0)]);
        assert!(!r.window(1e9, 100, &["late"], 5).series.contains_key("g"));
    }

    #[test]
    fn ring_is_bounded_and_thins_from_the_end() {
        let m = MetricsStore::new(10, 5);
        for i in 0..25 {
            m.push(Point {
                t: f64::from(i) * 5.0,
                up: true,
                ..Point::default()
            });
        }
        let w = m.window(1e9, 100);
        assert_eq!(w.t.len(), 10);
        assert_eq!(*w.t.last().unwrap(), 120.0);
        let w = m.window(1e9, 4);
        assert!(w.t.len() <= 4);
        assert_eq!(*w.t.last().unwrap(), 120.0);
        assert_eq!(w.interval_s, 15);
        assert_eq!(m.window(10.0, 100).t, vec![110.0, 115.0, 120.0]);
    }
}
