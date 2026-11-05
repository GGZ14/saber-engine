//! FN47a: the STEP-AHEAD expert prefetch study over FN18's served route traces (host only; the
//! tier's insert path untouched).
//!
//! A decode step reads every routed expert that is not VRAM-resident (a MISS) over UVA inside the
//! grouped GEMM, at link rate, serialized with that layer. A step-ahead prefetch copies a
//! PREDICTED set into a staging pool on the copy stream while earlier layers compute; the kernel
//! takes a staged expert through `xptr` and falls back to UVA for the rest -- bit-exact by
//! construction (the prediction chooses what is copied, never what is computed). This module
//! replays the real [`ExpertTier`] (the same warm start, slots and law as `replay`) over a trace,
//! and at every decode step asks each predictor for its set per layer, BEFORE that step's routes:
//!
//! * [`Pred::Batch`]`(n)`: the union of the whole batch's routes (live and pad rows) over the last
//!   `n` decode steps.
//! * [`Pred::Profile`]`(k)`: the warm-start profile's top `k` NON-resident experts per layer.
//! * [`Pred::BatchProfile`]`(n, k)`: the union of the two.
//! * [`Pred::Oracle`]: the step's own misses (the bound).
//!
//! The copy set is the prediction minus the VRAM-resident records (the tier's placement is fixed
//! through a step: promotions land at `end_step`). Per predictor it counts the misses predicted
//! (recall), the copies (precision), and runs the TIMING LAW: one rank's step as `layers` equal
//! compute slices, a layer's experts needed at its middle, one shared link serving the copy stream
//! FIFO (at the copy-engine rate) and the demand UVA reads (at the link rate; they stall compute
//! and push the copy queue back), the issue either at step start or `lead` layers ahead; the step
//! ends when both compute and the copy queue are done. The EP step is the max of the two ranks.
//! [`SeqDiag`] adds the per-SEQUENCE recall (row group = sequence, positional identity across two
//! decode steps of the same shape) beside the batch's on the same steps.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::expert_tier::{ExpertTier, Placement, RouteProfile, TierParams};
use crate::route_trace::{replay_header, CountMover, InsertLaw, Window, TRACE_MAGIC};

/// A predictor of a decode step's routed experts per layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pred {
    /// Union of the batch's routes over the last `n` decode steps.
    Batch(usize),
    /// The profile's top `k` non-resident experts per layer (static ranking).
    Profile(usize),
    /// `Batch(n)` united with `Profile(k)`.
    BatchProfile(usize, usize),
    /// The step's own misses.
    Oracle,
    /// CROSS-LAYER: per row, the top `k` experts at layer L by the co-routing table P(e at L |
    /// the row's experts at L - 1) (trained on another trace, [`XTable`]); known only after layer
    /// L - 1's router, so it is issued one layer ahead (both timing columns use lead 1).
    XLayer(usize),
}

impl Pred {
    /// A short name.
    #[must_use]
    pub fn name(self) -> String {
        match self {
            Pred::Batch(n) => format!("last-batch n={n}"),
            Pred::Profile(k) => format!("profile top{k}"),
            Pred::BatchProfile(n, k) => format!("batch n={n} + top{k}"),
            Pred::Oracle => "oracle".to_string(),
            Pred::XLayer(k) => format!("x-layer top{k}/row"),
        }
    }
    /// Parse `batch1`, `batch4`, `top8`, `batch1+top8`, `oracle`.
    pub fn parse(s: &str) -> Result<Pred, String> {
        let num = |x: &str, pre: &str| -> Result<usize, String> {
            x.strip_prefix(pre)
                .ok_or_else(|| format!("predictor '{s}'"))?
                .parse::<usize>()
                .map_err(|_| format!("predictor '{s}'"))
        };
        if s == "oracle" {
            return Ok(Pred::Oracle);
        }
        if let Some((a, b)) = s.split_once('+') {
            return Ok(Pred::BatchProfile(num(a, "batch")?, num(b, "top")?));
        }
        if s.starts_with("xlayer") {
            return Ok(Pred::XLayer(num(s, "xlayer")?));
        }
        if s.starts_with("batch") {
            return Ok(Pred::Batch(num(s, "batch")?.max(1)));
        }
        Ok(Pred::Profile(num(s, "top")?))
    }
    fn hist(self) -> usize {
        match self {
            Pred::Batch(n) | Pred::BatchProfile(n, _) => n,
            _ => 0,
        }
    }
}

/// The link / compute model of the timing law.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkModel {
    /// One expert record's bytes (the pack's stride).
    pub stride: u64,
    /// Demand (in-kernel UVA) read rate, GB/s.
    pub link_gbps: f64,
    /// Copy-engine rate, GB/s.
    pub copy_gbps: f64,
    /// Compute ms of a step with no link time, by decode bucket.
    pub compute_ms: Vec<(u32, f64)>,
    /// The issue lead in layers for the one-layer-ahead form (layer L issued when L - lead starts).
    pub lead: usize,
}

impl LinkModel {
    /// Compute ms for a bucket (the nearest listed bucket at or above, else the largest).
    #[must_use]
    pub fn compute(&self, bucket: u32) -> f64 {
        self.compute_ms
            .iter()
            .find(|(b, _)| *b >= bucket)
            .or(self.compute_ms.last())
            .map_or(0.0, |x| x.1)
    }
    /// Parse `"1:28.4,8:55"`.
    pub fn parse_compute(s: &str) -> Result<Vec<(u32, f64)>, String> {
        let mut v = Vec::new();
        for p in s.split(',').filter(|p| !p.trim().is_empty()) {
            let (b, ms) = p
                .split_once(':')
                .ok_or_else(|| format!("'{p}': want b:ms"))?;
            v.push((
                b.trim().parse().map_err(|_| format!("bucket '{b}'"))?,
                ms.trim().parse().map_err(|_| format!("ms '{ms}'"))?,
            ));
        }
        v.sort_by_key(|x| x.0);
        Ok(v)
    }
    fn read_ms(&self) -> f64 {
        self.stride as f64 / (self.link_gbps * 1e9) * 1e3
    }
    fn copy_ms(&self) -> f64 {
        self.stride as f64 / (self.copy_gbps * 1e9) * 1e3
    }
}

/// One rank's step under the timing law. `copies[l]` = the copy items of layer `l` (true = a miss
/// of the step, i.e. useful), `misses[l]` = the layer's misses. `lead` = `None` issues every copy at
/// step start, `Some(d)` issues layer `l`'s when layer `l - d` starts. Returns (step ms, peak
/// staging slots held, copy items still queued or late at their layer).
#[must_use]
pub fn simulate(
    copies: &[Vec<bool>],
    misses: &[u32],
    compute_ms: f64,
    m: &LinkModel,
    lead: Option<usize>,
) -> (f64, u32, u32) {
    let layers = misses.len();
    let c = compute_ms / layers.max(1) as f64;
    let (tr, tc) = (m.read_ms(), m.copy_ms());
    // the copy queue: (layer, useful, issue time), FIFO
    let mut q: Vec<(usize, bool, f64)> = Vec::new();
    let mut head = 0usize;
    let mut link_free = 0.0f64;
    let mut arrived = vec![0u32; layers];
    let mut clock = 0.0f64;
    let mut issued = 0usize; // layers issued so far
    let mut peak = 0u32;
    let mut late = 0u32;
    let issue_upto = |upto: usize, issued: &mut usize, q: &mut Vec<(usize, bool, f64)>, t: f64| {
        while *issued < upto.min(layers) {
            for &u in &copies[*issued] {
                q.push((*issued, u, t));
            }
            *issued += 1;
        }
    };
    if lead.is_none() {
        issue_upto(layers, &mut issued, &mut q, 0.0);
        peak = q.len() as u32;
    }
    for l in 0..layers {
        if let Some(d) = lead {
            issue_upto(l + d + 1, &mut issued, &mut q, clock);
            let held: usize = (l..issued).map(|x| copies[x].len()).sum();
            peak = peak.max(held as u32);
        }
        let need = clock + 0.5 * c;
        while head < q.len() {
            let (ql, u, t) = q[head];
            let done = link_free.max(t) + tc;
            if done > need {
                break;
            }
            link_free = done;
            if ql >= l && u {
                arrived[ql] += 1;
            } else if ql < l {
                late += 1;
            }
            head += 1;
        }
        let demand = misses[l].saturating_sub(arrived[l]);
        let stall = f64::from(demand) * tr;
        if demand > 0 {
            link_free = link_free.max(need) + stall;
        }
        clock = need + stall + 0.5 * c;
    }
    // drain: the queue's tail still holds the link
    for &(ql, _, t) in &q[head..] {
        link_free = link_free.max(t) + tc;
        if ql < layers {
            late += 1;
        }
    }
    (clock.max(link_free), peak, late)
}

/// Accumulated results of one (predictor, bucket) cell.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Cell {
    /// Decode steps.
    pub steps: u64,
    /// Rank-steps (steps x ranks).
    pub rank_steps: u64,
    /// Misses (routed, not VRAM-resident), summed over ranks.
    pub misses: u64,
    /// Copies issued (predicted, not resident).
    pub copies: u64,
    /// Copies that were misses.
    pub useful: u64,
    /// Sum of the EP step ms (max over ranks): today (no prefetch).
    pub ms_today: f64,
    /// ... with every copy issued at step start.
    pub ms_start: f64,
    /// ... with layer L issued when L - lead starts.
    pub ms_lead: f64,
    /// Compute ms summed (the no-link floor).
    pub ms_compute: f64,
    /// Peak staging slots a rank-step, step-start form (for percentiles).
    pub pool_start: Vec<u32>,
    /// ... lead form.
    pub pool_lead: Vec<u32>,
    /// Copies late or still queued at the step's end.
    pub late: u64,
}

impl Cell {
    /// The `p`-quantile of a pool list.
    #[must_use]
    pub fn pct(v: &[u32], p: f64) -> u32 {
        if v.is_empty() {
            return 0;
        }
        let mut s = v.to_vec();
        s.sort_unstable();
        s[((s.len() - 1) as f64 * p).round() as usize]
    }
}

/// The per-sequence diagnostic (steps whose previous decode step had the same rows / live / width).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SeqDiag {
    /// Steps counted.
    pub steps: u64,
    /// Sequence-level misses (a sequence's routed, non-resident experts; summed over seqs/ranks).
    pub misses: u64,
    /// ... predicted by the SAME sequence's routes at t-1.
    pub by_seq: u64,
    /// ... predicted by the whole batch's routes at t-1.
    pub by_batch: u64,
}

/// The study's output.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Study {
    /// `(predictor index, bucket)` -> cell; bucket 9 = b8 steps with every row live (full batches).
    pub cells: BTreeMap<(usize, u32), Cell>,
    /// Per bucket.
    pub seq: BTreeMap<u32, SeqDiag>,
    /// Decode steps seen.
    pub decode_steps: u64,
}

/// The cross-layer co-routing table: `succ[L][f]` = the experts at layer L most often routed by a
/// row that routed `f` at layer L - 1, with P(e | f) (decode rows, live only), the top `keep`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct XTable {
    /// `succ[L][f]` = `(e, p)` descending; `succ[0]` is empty.
    pub succ: Vec<Vec<Vec<(u16, f32)>>>,
}

/// Train an [`XTable`] on a trace's decode steps (optionally inside `window`).
pub fn train_xlayer(
    trace: &Path,
    window: Option<&[Window]>,
    keep: usize,
) -> Result<XTable, String> {
    let mut rd = RawReader::open(trace).map_err(|e| e.to_string())?;
    let (layers, topk) = (rd.layers as usize, rd.topk as usize);
    let ne = 512usize;
    let mut c = vec![vec![0u32; ne * ne]; layers];
    let mut n = vec![vec![0u32; ne]; layers];
    while let Some(s) = rd.next().map_err(|e| e.to_string())? {
        if s.prefill {
            continue;
        }
        if let Some(ws) = window {
            if !ws.iter().any(|w| w.holds(s.t)) {
                continue;
            }
        }
        for l in 1..layers {
            for row in 0..s.live as usize {
                let a = &s.ids[s.off[l - 1] + row * topk..s.off[l - 1] + (row + 1) * topk];
                let b = &s.ids[s.off[l] + row * topk..s.off[l] + (row + 1) * topk];
                for &f in a {
                    let f = usize::from(f) % ne;
                    n[l][f] += 1;
                    for &e in b {
                        c[l][f * ne + usize::from(e) % ne] += 1;
                    }
                }
            }
        }
    }
    let mut succ = vec![Vec::new(); layers];
    for l in 1..layers {
        succ[l] = (0..ne)
            .map(|f| {
                let row = &c[l][f * ne..(f + 1) * ne];
                let mut v: Vec<(u16, f32)> = row
                    .iter()
                    .enumerate()
                    .filter(|(_, &x)| x > 0)
                    .map(|(e, &x)| (e as u16, x as f32 / n[l][f].max(1) as f32))
                    .collect();
                v.sort_by(|a, b| b.1.total_cmp(&a.1));
                v.truncate(keep);
                v
            })
            .collect();
    }
    Ok(XTable { succ })
}

/// The cross-layer prediction of one step: per layer the union over rows of each row's top `k`.
fn xlayer_predict(s: &Raw, topk: usize, xt: &XTable, k: usize, layers: usize) -> Vec<Vec<u16>> {
    let mut out = vec![Vec::new(); layers];
    let mut score = vec![0f32; 512];
    let mut touched: Vec<u16> = Vec::new();
    for (l, o) in out.iter_mut().enumerate().skip(1) {
        for row in 0..s.rows as usize {
            let a = &s.ids[s.off[l - 1] + row * topk..s.off[l - 1] + (row + 1) * topk];
            for &f in a {
                for &(e, p) in &xt.succ[l][usize::from(f) % 512] {
                    if score[usize::from(e)] == 0.0 {
                        touched.push(e);
                    }
                    score[usize::from(e)] += p;
                }
            }
            touched.sort_by(|a, b| score[usize::from(*b)].total_cmp(&score[usize::from(*a)]));
            o.extend(touched.iter().take(k));
            for &e in &touched {
                score[usize::from(e)] = 0.0;
            }
            touched.clear();
        }
        o.sort_unstable();
        o.dedup();
    }
    out
}

type Bits = [u64; 8];
/// Per rank, per sequence, per layer.
type SeqSets = Vec<Vec<Vec<Bits>>>;

fn set(b: &mut Bits, i: usize) {
    b[i >> 6] |= 1 << (i & 63);
}
fn get(b: &Bits, i: usize) -> bool {
    b[i >> 6] >> (i & 63) & 1 == 1
}
fn and(a: &Bits, b: &Bits) -> Bits {
    std::array::from_fn(|i| a[i] & b[i])
}
fn or(a: &Bits, b: &Bits) -> Bits {
    std::array::from_fn(|i| a[i] | b[i])
}
fn pop(a: &Bits) -> u32 {
    a.iter().map(|x| x.count_ones()).sum()
}

/// One raw target record: per layer the routed global ids (decode: `rows x topk`, raw).
struct Raw {
    prefill: bool,
    bucket: u32,
    width: u32,
    rows: u32,
    live: u32,
    t: f64,
    /// Decode: `layers x rows x topk`; prefill: the distinct ids, `off` per layer.
    ids: Vec<u16>,
    off: Vec<usize>,
}

struct RawReader {
    r: BufReader<File>,
    layers: u32,
    topk: u32,
}

impl RawReader {
    fn open(p: &Path) -> std::io::Result<RawReader> {
        let mut r = BufReader::with_capacity(8 << 20, File::open(p)?);
        let mut m = [0u8; 8];
        r.read_exact(&mut m)?;
        if &m != TRACE_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not an fn18-route-trace/1 file",
            ));
        }
        let mut h = [0u8; 16];
        r.read_exact(&mut h)?;
        let u = |o: usize| u32::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3]]);
        Ok(RawReader {
            r,
            layers: u(0),
            topk: u(4),
        })
    }
    /// The next TARGET record (MTP / PLE records skipped).
    fn next(&mut self) -> std::io::Result<Option<Raw>> {
        loop {
            let mut h = [0u8; 28];
            match self.r.read_exact(&mut h) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
                Err(e) => return Err(e),
            }
            let u16a = |o: usize| u32::from(u16::from_le_bytes([h[o], h[o + 1]]));
            let u32a = |o: usize| u32::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3]]);
            let mut p = vec![0u8; u32a(24) as usize];
            self.r.read_exact(&mut p)?;
            if h[0] > 1 {
                continue;
            }
            let mut tb = [0u8; 8];
            tb.copy_from_slice(&h[16..24]);
            let u16s: Vec<u16> = p
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let mut raw = Raw {
                prefill: h[0] == 0,
                bucket: u16a(2),
                width: u16a(4),
                rows: u32a(8),
                live: u32a(12),
                t: f64::from_le_bytes(tb),
                ids: Vec::new(),
                off: vec![0],
            };
            if raw.prefill {
                let mut at = 0;
                for _ in 0..self.layers {
                    let n = usize::from(u16s[at]);
                    at += 1;
                    for _ in 0..n {
                        raw.ids.push(u16s[at]);
                        at += 2;
                    }
                    raw.off.push(raw.ids.len());
                }
            } else {
                let per = (raw.rows * self.topk) as usize;
                for l in 0..=self.layers as usize {
                    raw.off.push(l * per);
                }
                raw.off.remove(0);
                raw.ids = u16s;
            }
            return Ok(Some(raw));
        }
    }
}

/// Run the study over one trace for `ranks` (each its own tier, the same `params` / warm start /
/// law as [`crate::route_trace::replay`]), every predictor in `preds`.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn study(
    trace: &Path,
    ranks: &[u32],
    per_rank: u32,
    params: &TierParams,
    profile: Option<&RouteProfile>,
    law: &InsertLaw,
    window: Option<&[Window]>,
    preds: &[Pred],
    m: &LinkModel,
    xt: Option<&XTable>,
) -> Result<Study, String> {
    if preds.iter().any(|p| matches!(p, Pred::XLayer(_))) && xt.is_none() {
        return Err("an x-layer predictor needs a trained XTable".to_string());
    }
    let mut rd = RawReader::open(trace).map_err(|e| e.to_string())?;
    let (layers, topk) = (rd.layers as usize, rd.topk as usize);
    let pr = per_rank as usize;
    let mut tiers = Vec::new();
    for &r in ranks {
        let h = replay_header(r, rd.layers, per_rank, m.stride);
        tiers.push(
            ExpertTier::new(&h, params.clone(), profile, 0, 1 << 40).map_err(|e| e.to_string())?,
        );
    }
    let mut mv = CountMover::default();
    // profile ranking per rank per layer: local ids by descending count
    let rank_order: Vec<Vec<Vec<usize>>> = ranks
        .iter()
        .map(|&r| {
            (0..layers)
                .map(|l| {
                    let mut v: Vec<usize> = (0..pr).collect();
                    if let Some(p) = profile {
                        let row = &p.counts[l];
                        let base = (r * per_rank) as usize;
                        v.sort_by_key(|&e| {
                            std::cmp::Reverse(row.get(base + e).copied().unwrap_or(0))
                        });
                    }
                    v
                })
                .collect()
        })
        .collect();
    let max_hist = preds.iter().map(|p| p.hist()).max().unwrap_or(1).max(1);
    // history[rank] = the last decode steps' routed bitsets per layer (newest last)
    let mut hist: Vec<std::collections::VecDeque<Vec<Bits>>> =
        vec![std::collections::VecDeque::new(); ranks.len()];
    // previous decode step's per-sequence sets: (rows, live, width, [rank][seq][layer])
    let mut prev_seq: Option<(u32, u32, u32, SeqSets)> = None;
    let mut out = Study::default();
    while let Some(s) = rd.next().map_err(|e| e.to_string())? {
        if let Some(ws) = window {
            if !ws.iter().any(|w| w.holds(s.t)) {
                continue;
            }
        }
        let bucket = if s.prefill { 0 } else { s.bucket };
        // per rank: routed / non-resident bitsets per layer at step start
        let mut routed = vec![vec![[0u64; 8]; layers]; ranks.len()];
        let mut nonres = vec![vec![[0u64; 8]; layers]; ranks.len()];
        let mut seqs: Vec<Vec<Vec<Bits>>> = Vec::new();
        let width = s.width.max(1);
        let nseq = if s.prefill {
            0
        } else {
            (s.live / width) as usize
        };
        for (ri, &r) in ranks.iter().enumerate() {
            let first = (r * per_rank) as usize;
            let t = &mut tiers[ri];
            t.set_max_inserts(law.cap(bucket));
            t.set_promote_budget(law.budget(bucket));
            t.set_second_chance(law.second_chance(bucket));
            t.begin_step();
            for l in 0..layers {
                for e in 0..pr {
                    if !matches!(
                        t.placement(l as u32, (first + e) as u32),
                        Some(Placement::Vram(_))
                    ) {
                        set(&mut nonres[ri][l], e);
                    }
                }
                let ids = &s.ids[s.off[l]..s.off[l + 1]];
                for &e in ids {
                    let e = usize::from(e);
                    if e >= first && e < first + pr {
                        set(&mut routed[ri][l], e - first);
                    }
                }
                let g: Vec<u32> = ids.iter().map(|&x| u32::from(x)).collect();
                t.resolve(l as u32, &g, &mut mv)
                    .map_err(|e| e.to_string())?;
            }
            if !s.prefill {
                let mut sv = vec![vec![[0u64; 8]; layers]; nseq];
                for (q, seq) in sv.iter_mut().enumerate() {
                    for (l, b) in seq.iter_mut().enumerate() {
                        for row in q * width as usize..(q + 1) * width as usize {
                            let at = s.off[l] + row * topk;
                            for &e in &s.ids[at..at + topk] {
                                let e = usize::from(e);
                                if e >= first && e < first + pr {
                                    set(b, e - first);
                                }
                            }
                        }
                    }
                }
                seqs.push(sv);
            }
            t.refill(&mut mv).map_err(|e| e.to_string())?;
            t.end_step(&mut mv).map_err(|e| e.to_string())?;
        }
        if s.prefill {
            continue;
        }
        out.decode_steps += 1;
        let buckets: Vec<u32> = if s.bucket == 8 && s.live >= s.rows {
            vec![s.bucket, 9]
        } else {
            vec![s.bucket]
        };
        let compute = m.compute(s.bucket);
        let misses: Vec<Vec<Bits>> = (0..ranks.len())
            .map(|ri| {
                (0..layers)
                    .map(|l| and(&routed[ri][l], &nonres[ri][l]))
                    .collect()
            })
            .collect();
        // the per-sequence diagnostic
        if let Some((pr_rows, pr_live, pr_w, ps)) = &prev_seq {
            if *pr_rows == s.rows && *pr_live == s.live && *pr_w == s.width && nseq > 0 {
                let mut d = SeqDiag {
                    steps: 1,
                    ..SeqDiag::default()
                };
                for ri in 0..ranks.len() {
                    let pb = hist[ri].back();
                    for q in 0..nseq {
                        for l in 0..layers {
                            let mq = and(&seqs[ri][q][l], &nonres[ri][l]);
                            d.misses += u64::from(pop(&mq));
                            d.by_seq += u64::from(pop(&and(&mq, &ps[ri][q][l])));
                            if let Some(pb) = pb {
                                d.by_batch += u64::from(pop(&and(&mq, &pb[l])));
                            }
                        }
                    }
                }
                for &b in &buckets {
                    let e = out.seq.entry(b).or_default();
                    e.steps += d.steps;
                    e.misses += d.misses;
                    e.by_seq += d.by_seq;
                    e.by_batch += d.by_batch;
                }
            }
        }
        let mut xpred: BTreeMap<usize, Vec<Vec<u16>>> = BTreeMap::new();
        for &p in preds {
            if let (Pred::XLayer(k), Some(xt)) = (p, xt) {
                xpred.insert(k, xlayer_predict(&s, topk, xt, k, layers));
            }
        }
        for (pi, &p) in preds.iter().enumerate() {
            let (mut today, mut start, mut ahead) = (0f64, 0f64, 0f64);
            let mut cell_add = Cell::default();
            for ri in 0..ranks.len() {
                let mut copies = vec![Vec::new(); layers];
                let mut miss_n = vec![0u32; layers];
                for l in 0..layers {
                    let mut pred: Bits = [0; 8];
                    match p {
                        Pred::Oracle => pred = misses[ri][l],
                        Pred::Batch(n) | Pred::BatchProfile(n, _) => {
                            for h in hist[ri].iter().rev().take(n) {
                                pred = or(&pred, &h[l]);
                            }
                        }
                        Pred::Profile(_) => {}
                        Pred::XLayer(k) => {
                            let first = (ranks[ri] * per_rank) as usize;
                            for &e in &xpred[&k][l] {
                                let e = usize::from(e);
                                if e >= first && e < first + pr {
                                    set(&mut pred, e - first);
                                }
                            }
                        }
                    }
                    if let Pred::Profile(k) | Pred::BatchProfile(_, k) = p {
                        let mut got = 0;
                        for &e in &rank_order[ri][l] {
                            if got >= k {
                                break;
                            }
                            if get(&nonres[ri][l], e) {
                                set(&mut pred, e);
                                got += 1;
                            }
                        }
                    }
                    let c = and(&pred, &nonres[ri][l]);
                    miss_n[l] = pop(&misses[ri][l]);
                    for e in 0..pr {
                        if get(&c, e) {
                            copies[l].push(get(&misses[ri][l], e));
                        }
                    }
                    cell_add.copies += copies[l].len() as u64;
                    cell_add.useful += u64::from(pop(&and(&c, &misses[ri][l])));
                    cell_add.misses += u64::from(miss_n[l]);
                }
                let none = vec![Vec::new(); layers];
                let (t0, _, _) = simulate(&none, &miss_n, compute, m, None);
                let (first_lead, lead) = match p {
                    Pred::XLayer(_) => (Some(1), Some(1)),
                    _ => (None, Some(m.lead)),
                };
                let (t1, p1, l1) = simulate(&copies, &miss_n, compute, m, first_lead);
                let (t2, p2, l2) = simulate(&copies, &miss_n, compute, m, lead);
                today = today.max(t0);
                start = start.max(t1);
                ahead = ahead.max(t2);
                cell_add.pool_start.push(p1);
                cell_add.pool_lead.push(p2);
                cell_add.late += u64::from(l1.max(l2));
            }
            for &b in &buckets {
                let c = out.cells.entry((pi, b)).or_default();
                c.steps += 1;
                c.rank_steps += ranks.len() as u64;
                c.misses += cell_add.misses;
                c.copies += cell_add.copies;
                c.useful += cell_add.useful;
                c.ms_today += today;
                c.ms_start += start;
                c.ms_lead += ahead;
                c.ms_compute += compute;
                c.pool_start.extend_from_slice(&cell_add.pool_start);
                c.pool_lead.extend_from_slice(&cell_add.pool_lead);
                c.late += cell_add.late;
            }
        }
        for ri in 0..ranks.len() {
            hist[ri].push_back(routed[ri].clone());
            while hist[ri].len() > max_hist {
                hist[ri].pop_front();
            }
        }
        prev_seq = Some((s.rows, s.live, s.width, seqs));
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------------
// FN47b: the PRE-GATE's recall, scored on the card's own logits (the capture arm
// `--fn-pregate-capture`): a layer's router applied to an EARLIER stream predicts its routes.

/// The candidate list sizes the capture scores (top-K a row).
pub const PREGATE_KS: [usize; 3] = [10, 16, 24];
/// The per-layer copy budgets (records a layer's lead can carry: ~0.37 ms at b1, ~1.1 ms at b8,
/// 2x and 4x that, at 122 us a record).
pub const PREGATE_BUDGETS: [usize; 4] = [3, 9, 18, 36];
/// The router's width (Flash-Next: 512 experts, top-10).
pub const PREGATE_EXPERTS: usize = 512;

/// One prediction method's tallies at one step shape (rows).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PregateTally {
    /// Layer-steps scored.
    pub layers: u64,
    /// This rank's routed records that were RAM-resident (the misses: what a prefetch must copy).
    pub misses: u64,
    /// This rank's distinct routed records (resident or not).
    pub routed: u64,
    /// Per K: misses the union of the rows' top-K named.
    pub hits: [u64; 3],
    /// Per K: routed records (any placement) the union named.
    pub routed_hits: [u64; 3],
    /// Per K: the union's RAM-resident records (the copies a prefetch would issue).
    pub copies: [u64; 3],
    /// Per budget (of the largest K's union, issued best-first): misses among the first B copies.
    pub budget_hits: [u64; 4],
    /// Per budget: copies actually issued (min(B, copies)).
    pub budget_copies: [u64; 4],
    /// The largest per-layer copy count seen at the largest K.
    pub max_copies: u64,
}

impl PregateTally {
    /// Recall of the misses at K index `k` (0 when there were none).
    #[must_use]
    pub fn recall(&self, k: usize) -> f64 {
        if self.misses == 0 {
            0.0
        } else {
            self.hits[k] as f64 / self.misses as f64
        }
    }
}

fn bf16_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// The top `k` expert ids of one row of bf16 router logits, best first (ties: lower id first).
#[must_use]
pub fn row_topk(row: &[u16], k: usize) -> Vec<u32> {
    let mut ix: Vec<u32> = (0..row.len() as u32).collect();
    ix.sort_by(|&a, &b| {
        bf16_f32(row[b as usize])
            .partial_cmp(&bf16_f32(row[a as usize]))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    ix.truncate(k);
    ix
}

/// Score one layer-step: `logits` = the predictor's `[m, 512]` bf16 rows, `routed` = the step's
/// TRUE route ids `[m * 10]` (global), `place(e)` = this rank's placement of global expert `e`
/// (`None`: another rank's; `Some(true)`: RAM-resident, `Some(false)`: VRAM).
pub fn pregate_score(
    t: &mut PregateTally,
    logits: &[u16],
    m: usize,
    routed: &[u32],
    place: impl Fn(u32) -> Option<bool>,
) {
    let kmax = PREGATE_KS[PREGATE_KS.len() - 1];
    // best (lowest) rank position a row gave each expert, and how many rows named it
    let mut best = vec![usize::MAX; PREGATE_EXPERTS];
    let mut rows_named = vec![0u32; PREGATE_EXPERTS];
    for r in 0..m {
        let row = &logits[r * PREGATE_EXPERTS..(r + 1) * PREGATE_EXPERTS];
        for (pos, e) in row_topk(row, kmax).into_iter().enumerate() {
            let e = e as usize;
            best[e] = best[e].min(pos);
            rows_named[e] += 1;
        }
    }
    let mut truth: Vec<u32> = routed
        .iter()
        .copied()
        .filter(|&e| (e as usize) < PREGATE_EXPERTS && place(e).is_some())
        .collect();
    truth.sort_unstable();
    truth.dedup();
    let mut miss = vec![false; PREGATE_EXPERTS];
    let mut is_routed = vec![false; PREGATE_EXPERTS];
    for &e in &truth {
        is_routed[e as usize] = true;
        miss[e as usize] = place(e) == Some(true);
    }
    t.layers += 1;
    t.routed += truth.len() as u64;
    t.misses += miss.iter().filter(|&&b| b).count() as u64;
    for (ki, &k) in PREGATE_KS.iter().enumerate() {
        for e in 0..PREGATE_EXPERTS {
            if best[e] < k && place(e as u32).is_some() {
                if is_routed[e] {
                    t.routed_hits[ki] += 1;
                }
                if place(e as u32) == Some(true) {
                    t.copies[ki] += 1;
                    if miss[e] {
                        t.hits[ki] += 1;
                    }
                }
            }
        }
    }
    // the largest K's copies, best-first: lowest rank position, then most rows, then id
    let mut cand: Vec<usize> = (0..PREGATE_EXPERTS)
        .filter(|&e| best[e] < kmax && place(e as u32) == Some(true))
        .collect();
    t.max_copies = t.max_copies.max(cand.len() as u64);
    cand.sort_by_key(|&e| (best[e], std::cmp::Reverse(rows_named[e]), e));
    for (bi, &b) in PREGATE_BUDGETS.iter().enumerate() {
        let take = cand.len().min(b);
        t.budget_copies[bi] += take as u64;
        t.budget_hits[bi] += cand[..take].iter().filter(|&&e| miss[e]).count() as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lm() -> LinkModel {
        LinkModel {
            stride: 2_617_344,
            link_gbps: 27.7,
            copy_gbps: 21.4,
            compute_ms: vec![(1, 28.0), (8, 55.0)],
            lead: 2,
        }
    }

    #[test]
    fn the_timing_law_hides_a_fitting_prefetch_and_charges_a_miss() {
        let m = lm();
        let layers = 48;
        let misses = vec![2u32; layers];
        let none = vec![Vec::new(); layers];
        // today: every miss is a demand read
        let (t0, p0, _) = simulate(&none, &misses, 55.0, &m, None);
        assert!((t0 - (55.0 + 96.0 * m.read_ms())).abs() < 1e-9);
        assert_eq!(p0, 0);
        // the oracle at step start: 96 copies = 11.7 ms of copy stream, inside 55 ms of compute
        // except the first layers' (needed before their copies land)
        let all = vec![vec![true, true]; layers];
        let (t1, p1, _) = simulate(&all, &misses, 55.0, &m, None);
        assert_eq!(p1, 96);
        assert!(t1 < t0 && t1 >= 55.0 - 1e-9, "{t1} vs {t0}");
        // one layer ahead holds at most (lead + 1) layers' copies
        let (t2, p2, _) = simulate(&all, &misses, 55.0, &m, Some(2));
        assert_eq!(p2, 6);
        assert!(t2 < t0 && t2 >= 55.0 - 1e-9);
        // useless copies only add link time: never faster than today
        let junk = vec![vec![false; 8]; layers];
        let (t3, _, late) = simulate(&junk, &misses, 55.0, &m, None);
        assert!(t3 >= t0 - 1e-9);
        assert!(late > 0, "a 47 ms junk stream outlives its first layers");
        // a copy stream longer than the compute extends the step
        let lots = vec![vec![false; 20]; layers];
        let (t4, _, _) = simulate(&lots, &vec![0; layers], 55.0, &m, None);
        assert!((t4 - 960.0 * m.copy_ms()).abs() < 1e-6);
    }

    /// The study's misses are the replay's RAM + NVMe routes (the same tier, the same law), the
    /// oracle copies exactly them, and no predictor recalls more misses than exist.
    #[test]
    fn the_study_counts_the_replays_misses_and_the_oracle_bounds_it() {
        use crate::route_trace::{replay, StepKind, TraceWriter};
        // Under `target/`, removed when `_scratch` drops -- never `/tmp` (FN51h).
        let _scratch = saber_test_scratch::ScratchDir::new("fn47a");
        let path = _scratch.join("fn47a.bin");
        let (layers, topk) = (4u32, 10u32);
        {
            let mut w = TraceWriter::create(&path, layers, topk, 512).unwrap();
            let mut x: u64 = 0x2545_f491_4f6c_dd1d;
            let mut next = || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let u = (x % 10_000) as f64 / 10_000.0;
                (u * u * 512.0) as u32
            };
            for s in 0..60u32 {
                let (kind, rows, b, wd) = if s % 15 == 0 {
                    (StepKind::Prefill, 32, 0, 1)
                } else if s % 2 == 0 {
                    (StepKind::Decode, 8, 8, 1)
                } else {
                    (StepKind::Decode, 4, 1, 4)
                };
                let ids: Vec<u32> = (0..layers * rows * topk).map(|_| next()).collect();
                w.target(kind, rows, rows, b, wd, &ids).unwrap();
            }
            w.flush().unwrap();
        }
        let params = TierParams {
            vram_slots: 200,
            ram_slots: 4 * 256 - 200,
            max_inserts: 2,
            ..TierParams::default()
        };
        let law = InsertLaw::parse(2, "").unwrap();
        let preds = [
            Pred::Batch(1),
            Pred::Batch(4),
            Pred::Profile(4),
            Pred::Oracle,
        ];
        let st = study(
            &path,
            &[0, 1],
            256,
            &params,
            None,
            &law,
            None,
            &preds,
            &lm(),
            None,
        )
        .unwrap();
        let mut ram = 0;
        for r in [0, 1] {
            let o = replay(&path, r, 256, 2_617_344, &params, None, &law, None, None).unwrap();
            ram += o
                .buckets
                .iter()
                .filter(|(b, _)| **b != 0)
                .map(|(_, t)| t.paths.ram + t.paths.nvme)
                .sum::<u64>();
        }
        let _ = std::fs::remove_file(&path);
        for (pi, _) in preds.iter().enumerate() {
            let m: u64 = [1, 8].iter().map(|b| st.cells[&(pi, *b)].misses).sum();
            assert_eq!(m, ram, "predictor {pi}: the study's misses == the replay's");
            for b in [1, 8] {
                let c = &st.cells[&(pi, b)];
                assert!(c.useful <= c.misses && c.useful <= c.copies);
                assert!(c.ms_start >= c.ms_compute - 1e-9 && c.ms_today >= c.ms_compute - 1e-9);
            }
        }
        for b in [1, 8] {
            let o = &st.cells[&(3, b)];
            assert_eq!(
                (o.useful, o.copies),
                (o.misses, o.misses),
                "the oracle copies its misses"
            );
            assert!(o.ms_start <= o.ms_today + 1e-9);
        }
        assert!(
            st.cells.contains_key(&(0, 9)),
            "full b8 steps are tallied apart"
        );
        assert!(study(
            &path,
            &[0],
            256,
            &params,
            None,
            &law,
            None,
            &[Pred::XLayer(2)],
            &lm(),
            None
        )
        .is_err());
    }

    #[test]
    fn predictors_parse() {
        assert_eq!(Pred::parse("batch1"), Ok(Pred::Batch(1)));
        assert_eq!(Pred::parse("top8"), Ok(Pred::Profile(8)));
        assert_eq!(Pred::parse("batch2+top4"), Ok(Pred::BatchProfile(2, 4)));
        assert_eq!(Pred::parse("oracle"), Ok(Pred::Oracle));
        assert_eq!(Pred::parse("xlayer4"), Ok(Pred::XLayer(4)));
        assert!(Pred::parse("x").is_err());
        assert_eq!(
            LinkModel::parse_compute("8:55,1:28").unwrap(),
            vec![(1, 28.0), (8, 55.0)]
        );
        assert!((lm().compute(2) - 55.0).abs() < 1e-12);
    }

    #[test]
    fn the_pregate_scorer_counts_misses_copies_and_budgets() {
        // two rows; row 0 ranks experts 5 > 7 > 9 > the rest by id, row 1 ranks 7 > 11
        let mut lg = vec![0u16; 2 * PREGATE_EXPERTS];
        let set = |lg: &mut Vec<u16>, r: usize, e: usize, v: f32| {
            lg[r * PREGATE_EXPERTS + e] = (v.to_bits() >> 16) as u16;
        };
        for e in 0..PREGATE_EXPERTS {
            set(&mut lg, 0, e, -1.0 - e as f32 / 64.0);
            set(&mut lg, 1, e, -1.0 - e as f32 / 64.0);
        }
        set(&mut lg, 0, 5, 3.0);
        set(&mut lg, 0, 7, 2.0);
        set(&mut lg, 0, 9, 1.0);
        set(&mut lg, 1, 7, 3.0);
        set(&mut lg, 1, 11, 2.0);
        assert_eq!(row_topk(&lg[..PREGATE_EXPERTS], 4), vec![5, 7, 9, 0]);
        // this rank owns 0..256; RAM-resident: 5, 9, 200; VRAM the rest
        let place = |e: u32| match e {
            256.. => None,
            5 | 9 | 200 => Some(true),
            _ => Some(false),
        };
        // true routes: 5 (miss), 7 (VRAM), 200 (miss, never predicted), 300 (other rank)
        let routed = [5u32, 7, 200, 300, 5, 7, 200, 300, 0, 0];
        let mut t = PregateTally::default();
        pregate_score(&mut t, &lg, 1, &routed[..5], place);
        assert_eq!((t.layers, t.misses, t.routed), (1, 2, 3));
        // K 10: row 0 names 5, 7, 9, 0..6 -> copies 5, 9; hits 5
        assert_eq!(t.copies[0], 2);
        assert_eq!(t.hits[0], 1);
        assert_eq!(t.routed_hits[0], 2);
        assert!((t.recall(0) - 0.5).abs() < 1e-12);
        // budget 3: 5 then 9 (best-first) -> 1 hit of 2 copies
        assert_eq!((t.budget_copies[0], t.budget_hits[0]), (2, 1));
        // two rows: the union adds 11 (VRAM) only
        let mut t2 = PregateTally::default();
        pregate_score(&mut t2, &lg, 2, &routed, place);
        assert_eq!(t2.copies[0], 2);
        assert_eq!(t2.max_copies, 2);
    }
}
