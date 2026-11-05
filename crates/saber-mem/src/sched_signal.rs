//! FN104: THE SCHEDULER'S KNOWLEDGE AS A SIGNAL FOR THE EXPERT TIER (LEDGER FN104; the seed of
//! FLASHNEXT-PLAN §15's E7 arbiter).
//!
//! FN48 closed the history-only victim policies: every ranking built from a record's own
//! reference history ties LRFU 0.98, and MIN's remaining gap is information OUTSIDE a record's
//! history -- which conversation ends, which starts. The scheduler holds exactly that: which
//! sequences are live, which rows of a step are whose, when one is admitted and when it finishes.
//! This module is that signal and its replay:
//!
//! * [`SchedEvent`] -- the event stream an arbiter consumes: `Admitted` (a request entered prefill,
//!   with its prompt length / max tokens / prefix hit), `Joined` (it entered the decode batch at a
//!   position), `Finishing` (steps left), `Finished`; the per-step `routed(seq, layer, experts)`
//!   is [`StepAnnot::seqs`] (the sequence at each decode position) beside the step's route log
//!   (row `j * width + t` belongs to position `j`).
//! * [`infer`] -- the FN18 trace carries NO sequence id and no request boundary, so the stream is
//!   DERIVED from its step structure under the serve plane's own slot law (a new prefill = a new
//!   request, a chunk of the boot's maximum rows continues into the next; the decode batch is pool
//!   slots `0..n` in order, a `Ready` appends at the end, a `Leave` swap-removes -- worker.rs
//!   `Cmd::Ready` / `Cmd::Leave`): when the live count changes, the departed positions (and leave
//!   order) are the candidate that best continues every survivor's routing (the last
//!   [`InferOpts::hist`] steps' (layer, expert) sets). Steady steps check the signal (a position's
//!   routing is closer to its own history than to any other live sequence's).
//! * [`sched_replay`] -- FN18's exact tier replay with the scheduler's law applied ([`SchedLaw`]:
//!   finish-decay, working-set protection, steps-left, the admission prior and the pre-warm
//!   copies) + the census (per-sequence working sets, concurrent overlap, miss / victim
//!   attribution, the first-200-step misses resident at the previous finish).
//!
//! Every law is PLACEMENT ONLY (which record leaves, which is copied): the bytes the kernels read
//! never change (I2). The neutral [`SchedLaw`] reproduces [`crate::route_trace::replay`] exactly.

#![forbid(unsafe_code)]

use std::collections::VecDeque;
use std::path::Path;

use crate::expert_tier::{ExpertMover, ExpertTier, Placement, RouteProfile, TierError, TierParams};
use crate::route_trace::{replay_header, InsertLaw, ReplayOut, StepKind, TraceReader};

/// A sequence (request) id of the derived stream (0.. in admission order).
pub type SeqId = u32;

/// FN104: the scheduler -> tier event stream (the E7 arbiter's input).
#[derive(Debug, Clone, PartialEq)]
pub enum SchedEvent {
    /// A request entered prefill.
    Admitted {
        /// The sequence.
        seq: SeqId,
        /// Its prompt's rows (all chunks; known at admission in the serve plane).
        prompt_len: u32,
        /// Its `max_tokens` (`None` = not carried: the FN18 trace has no request metadata).
        max_tokens: Option<u32>,
        /// Whether a stored prefix (the prompt cache) served part of it.
        prefix_hit: bool,
    },
    /// It entered the decode batch at `position` (after its last prefill chunk).
    Joined {
        /// The sequence.
        seq: SeqId,
        /// Its decode position (slot).
        position: u32,
    },
    /// It has `steps_left` decode steps left (from `max_tokens`; the replay's oracle form knows
    /// the true count).
    Finishing {
        /// The sequence.
        seq: SeqId,
        /// Steps left.
        steps_left: u32,
    },
    /// It left (released its slot).
    Finished {
        /// The sequence.
        seq: SeqId,
    },
}

/// The derived stream's options.
#[derive(Debug, Clone, PartialEq)]
pub struct InferOpts {
    /// The boot's widest prefill chunk: a chunk of at least this many rows continues into the
    /// next prefill record (FN18's boots: 2,048; FN44's wide default: 8,192).
    pub max_chunk: u32,
    /// Steps of a sequence's routing history its continuation is scored against.
    pub hist: usize,
    /// The score a candidate pays per departure beyond the fewest the live count allows (a
    /// simultaneous leave + join must beat the plain continuation by this).
    pub depart_penalty: f64,
}

impl Default for InferOpts {
    fn default() -> Self {
        InferOpts {
            max_chunk: 2048,
            hist: 4,
            depart_penalty: 0.15,
        }
    }
}

/// One target step's annotation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StepAnnot {
    /// Events before this step (in order).
    pub events: Vec<SchedEvent>,
    /// Decode: the sequence at each live position; prefill: the chunk's sequence.
    pub seqs: Vec<SeqId>,
}

/// One derived sequence.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SeqInfo {
    /// Target step index of its first prefill chunk.
    pub admit: u32,
    /// Target step index of its first decode step (`None` = never decoded: a prefill-only
    /// request, or the trace ended).
    pub join: Option<u32>,
    /// Target step index before which it was seen gone (`None` = live at the trace's end).
    pub finish: Option<u32>,
    /// Prompt rows (all chunks).
    pub prompt_rows: u32,
    /// Decode steps it was live in.
    pub decode_steps: u32,
}

/// The inference's own checks.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InferStats {
    /// Steady (event-free) decode positions checked (live >= 2).
    pub steady: u64,
    /// Of them: the position's routing scored highest against its own sequence's history.
    pub steady_self_best: u64,
    /// Mean self / best-other similarity on the steady checks.
    pub self_sim: f64,
    /// See [`Self::self_sim`].
    pub other_sim: f64,
    /// Decode steps whose live set changed (decided by the candidate search).
    pub decided: u64,
    /// Of them: best minus second-best candidate score below 0.05 (weakly decided).
    pub weak: u64,
    /// Departures / joins / prefill-only requests (dropped from the pending queue).
    pub departures: u64,
    /// See [`Self::departures`].
    pub joins: u64,
    /// See [`Self::departures`].
    pub prefill_only: u64,
    /// Decode steps with more new positions than pending requests (anonymous sequences made).
    pub anomalies: u64,
}

/// The derived stream of a trace.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Annot {
    /// Per target step (prefill / decode records, in order).
    pub steps: Vec<StepAnnot>,
    /// Per sequence.
    pub seqs: Vec<SeqInfo>,
    /// The checks.
    pub stats: InferStats,
}

/// The (layer, expert) sets of a decode record's live positions (global ids, `l * experts + e`).
#[must_use]
pub fn position_sets(
    raw: &[u16],
    layers: u32,
    topk: u32,
    experts: u32,
    rows: u32,
    n: u32,
    w: u32,
) -> Vec<Vec<u32>> {
    let (k, rows) = (topk as usize, rows as usize);
    let mut out = vec![Vec::new(); n as usize];
    for l in 0..layers as usize {
        let base = l * rows * k;
        for (j, set) in out.iter_mut().enumerate() {
            for t in 0..w as usize {
                let r = j * w as usize + t;
                for &e in &raw[base + r * k..base + (r + 1) * k] {
                    set.push(l as u32 * experts + u32::from(e));
                }
            }
        }
    }
    for s in &mut out {
        s.sort_unstable();
        s.dedup();
    }
    out
}

fn count_in(sorted: &[u32], x: u32) -> bool {
    sorted.binary_search(&x).is_ok()
}

/// Fraction of `set` found in the history (each history step weighs 1 / its count).
fn sim_hist(h: &VecDeque<Vec<u32>>, set: &[u32]) -> f64 {
    if h.is_empty() || set.is_empty() {
        return 0.0;
    }
    let mut c = 0.0;
    for x in set {
        for s in h {
            if count_in(s, *x) {
                c += 1.0;
            }
        }
    }
    c / (h.len() as f64 * set.len() as f64)
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur = Vec::with_capacity(k);
    fn go(i: usize, n: usize, k: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        if n - i < k - cur.len() {
            return;
        }
        for x in i..n {
            cur.push(x);
            go(x + 1, n, k, cur, out);
            cur.pop();
        }
    }
    go(0, n, k, &mut cur, &mut out);
    out
}

fn permutations(v: &[usize]) -> Vec<Vec<usize>> {
    if v.len() <= 1 {
        return vec![v.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..v.len() {
        let mut rest = v.to_vec();
        let x = rest.remove(i);
        for mut p in permutations(&rest) {
            p.insert(0, x);
            out.push(p);
        }
    }
    out
}

/// A candidate continuation: (score, the new batch, the departed, the joining pending indices).
type Cand = (f64, Vec<SeqId>, Vec<SeqId>, Vec<usize>);

/// FN104: derive the scheduler's event stream from an FN18 trace (module docs).
pub fn infer(trace: &Path, o: &InferOpts) -> Result<Annot, String> {
    let mut r = TraceReader::open(trace).map_err(|e| e.to_string())?;
    r.keep_raw = true;
    let (layers, topk, experts) = (r.layers, r.topk, r.experts);
    let mut a = Annot::default();
    let mut ready: Vec<SeqId> = Vec::new();
    let mut hist: Vec<VecDeque<Vec<u32>>> = Vec::new();
    let mut prompt: Vec<Vec<u32>> = Vec::new();
    let mut pending: VecDeque<SeqId> = VecDeque::new();
    let mut open: Option<SeqId> = None;
    let mut events: Vec<SchedEvent> = Vec::new();
    let (mut ss, mut os) = (0.0f64, 0.0f64);
    let new_seq = |a: &mut Annot,
                   hist: &mut Vec<VecDeque<Vec<u32>>>,
                   prompt: &mut Vec<Vec<u32>>,
                   idx: u32|
     -> SeqId {
        a.seqs.push(SeqInfo {
            admit: idx,
            ..SeqInfo::default()
        });
        hist.push(VecDeque::new());
        prompt.push(Vec::new());
        (a.seqs.len() - 1) as SeqId
    };
    while let Some(s) = r.next_step().map_err(|e| e.to_string())? {
        if !s.kind.target() {
            continue;
        }
        let idx = a.steps.len() as u32;
        if s.kind == StepKind::Prefill {
            let mut set: Vec<u32> = Vec::new();
            for l in 0..s.layers() {
                set.extend(
                    s.layer(l)
                        .iter()
                        .map(|x| l as u32 * experts + u32::from(x.e)),
                );
            }
            let q = match open.take() {
                Some(q) => q,
                None => {
                    let q = new_seq(&mut a, &mut hist, &mut prompt, idx);
                    events.push(SchedEvent::Admitted {
                        seq: q,
                        prompt_len: 0,
                        max_tokens: None,
                        prefix_hit: false,
                    });
                    q
                }
            };
            a.seqs[q as usize].prompt_rows += s.rows;
            let p = &mut prompt[q as usize];
            p.extend(set);
            p.sort_unstable();
            p.dedup();
            if s.rows >= o.max_chunk {
                open = Some(q);
            } else {
                pending.push_back(q);
            }
            // the admission's prompt length is the request's (known up front in the serve plane)
            for e in &mut events {
                if let SchedEvent::Admitted {
                    seq, prompt_len, ..
                } = e
                {
                    if *seq == q {
                        *prompt_len = a.seqs[q as usize].prompt_rows;
                    }
                }
            }
            a.steps.push(StepAnnot {
                events: std::mem::take(&mut events),
                seqs: vec![q],
            });
            continue;
        }
        let w = s.width.max(1);
        let n = s.live / w;
        let sets = position_sets(&r.raw, layers, topk, experts, s.rows, n, w);
        let (rn, n_us) = (ready.len(), n as usize);
        if n_us == rn && pending.is_empty() {
            if rn >= 2 {
                for (j, set) in sets.iter().enumerate() {
                    let me = sim_hist(&hist[ready[j] as usize], set);
                    let other = (0..rn)
                        .filter(|&i| i != j)
                        .map(|i| sim_hist(&hist[ready[i] as usize], set))
                        .fold(0.0, f64::max);
                    a.stats.steady += 1;
                    a.stats.steady_self_best += u64::from(me > other);
                    ss += me;
                    os += other;
                }
            }
        } else {
            a.stats.decided += 1;
            let p = pending.len();
            let d_lo = rn.saturating_sub(n_us);
            let d_hi = rn.min((rn + p).saturating_sub(n_us));
            let mut best: Option<Cand> = None;
            let mut second = f64::NEG_INFINITY;
            if rn + p < n_us {
                a.stats.anomalies += 1;
            }
            for d in d_lo..=d_hi.max(d_lo) {
                if d > rn {
                    break;
                }
                let arr = n_us + d - rn;
                let arr = arr.min(p);
                for dset in combinations(rn, d) {
                    let orders = if d <= 3 {
                        permutations(&dset)
                    } else {
                        vec![dset.clone()]
                    };
                    for ord in orders {
                        let mut v = ready.clone();
                        for &k in &ord {
                            let at = v.iter().position(|&x| x == ready[k]).expect("live");
                            v.swap_remove(at);
                        }
                        for aset in combinations(p, arr) {
                            let mut cand = v.clone();
                            cand.extend(aset.iter().map(|&i| pending[i]));
                            let mut sc = -o.depart_penalty * (d - d_lo) as f64;
                            for (j, set) in sets.iter().enumerate() {
                                let Some(&q) = cand.get(j) else { break };
                                let h = &hist[q as usize];
                                sc += if h.is_empty() {
                                    // a joining sequence: its prompt's coverage of the row
                                    let pr = &prompt[q as usize];
                                    set.iter().filter(|x| count_in(pr, **x)).count() as f64
                                        / set.len().max(1) as f64
                                        * 0.5
                                } else {
                                    sim_hist(h, set)
                                };
                            }
                            let departed: Vec<SeqId> = ord.iter().map(|&k| ready[k]).collect();
                            match &best {
                                Some((b, ..)) if sc <= *b => second = second.max(sc),
                                _ => {
                                    if let Some((b, ..)) = &best {
                                        second = second.max(*b);
                                    }
                                    best = Some((sc, cand, departed, aset));
                                }
                            }
                        }
                    }
                }
            }
            let (bs, mut cand, departed, aset) =
                best.unwrap_or((0.0, ready.clone(), Vec::new(), Vec::new()));
            if bs - second < 0.05 {
                a.stats.weak += 1;
            }
            for q in &departed {
                events.push(SchedEvent::Finished { seq: *q });
                a.seqs[*q as usize].finish = Some(idx);
                a.stats.departures += 1;
            }
            // joined: the chosen pending ones; every pending request queued BEFORE the last
            // joined one that did not join finished at its prefill
            let last_joined = aset.last().copied();
            let mut keep = VecDeque::new();
            for (i, &q) in pending.iter().enumerate() {
                if aset.contains(&i) {
                    continue;
                }
                if last_joined.is_some_and(|lj| i < lj) {
                    events.push(SchedEvent::Finished { seq: q });
                    a.seqs[q as usize].finish = Some(idx);
                    a.stats.prefill_only += 1;
                } else {
                    keep.push_back(q);
                }
            }
            pending = keep;
            // anonymous sequences for positions nobody accounts for
            while cand.len() < n_us {
                let q = new_seq(&mut a, &mut hist, &mut prompt, idx);
                cand.push(q);
            }
            for (j, &q) in cand.iter().enumerate() {
                if !ready.contains(&q) {
                    events.push(SchedEvent::Joined {
                        seq: q,
                        position: j as u32,
                    });
                    a.seqs[q as usize].join = Some(idx);
                    a.stats.joins += 1;
                }
            }
            ready = cand;
        }
        for (j, set) in sets.into_iter().enumerate() {
            let q = ready[j] as usize;
            a.seqs[q].decode_steps += 1;
            let h = &mut hist[q];
            h.push_back(set);
            if h.len() > o.hist {
                h.pop_front();
            }
        }
        a.steps.push(StepAnnot {
            events: std::mem::take(&mut events),
            seqs: ready.clone(),
        });
    }
    if a.stats.steady > 0 {
        a.stats.self_sim = ss / a.stats.steady as f64;
        a.stats.other_sim = os / a.stats.steady as f64;
    }
    Ok(a)
}

/// FN104: the scheduler's law over the tier (every field neutral by [`Default`] = FN18's LRFU).
#[derive(Debug, Clone, PartialEq)]
pub struct SchedLaw {
    /// FINISH-DECAY: on `Finished(seq)` each record the sequence touched that no LIVE sequence
    /// touched within [`Self::alive_window`] steps has its LRFU score multiplied by this (1 = off;
    /// 0 = victim first).
    pub finish_decay: f64,
    /// See [`Self::finish_decay`] (steps; `u64::MAX` = any time in the live sequence's life).
    pub alive_window: u64,
    /// WORKING-SET PROTECT: a record a live sequence touched within this many steps is protected
    /// (leaves only after every unprotected record; 0 = off).
    pub protect: u64,
    /// STEPS-LEFT: a sequence with fewer steps left than this protects nothing (0 = off). The
    /// replay's steps left are the derived sequence's TRUE remaining steps (the oracle form of
    /// `max_tokens`).
    pub left_min: u32,
    /// The oracle ANTICIPATION: a sequence counts as finished this many steps before it leaves
    /// (its finish-decay fires then; 0 = at the event).
    pub anticipate: u32,
    /// ADMISSION PRIOR: on `Joined(seq)` each record its prompt routed gets
    /// `+ beta * ln(1 + rows routing it) / ln(1 + prompt rows)` on its LRFU score (0 = off).
    pub admit_beta: f64,
    /// PRE-WARM: on `Joined(seq)` up to this many of each layer's most-routed prompt records that
    /// sit in RAM are copied into slots whose victim value is below [`Self::prewarm_dead`]
    /// (0 = off). The copies are counted apart (`prewarm`), off the step.
    pub prewarm: u32,
    /// See [`Self::prewarm`].
    pub prewarm_dead: f64,
    /// Census: print the working-set / overlap / attribution lines.
    pub census: bool,
}

impl Default for SchedLaw {
    fn default() -> Self {
        SchedLaw {
            finish_decay: 1.0,
            alive_window: u64::MAX,
            protect: 0,
            left_min: 0,
            anticipate: 0,
            admit_beta: 0.0,
            prewarm: 0,
            prewarm_dead: 0.5,
            census: false,
        }
    }
}

impl SchedLaw {
    /// Parse `fd=0.25,alive=64,prot=32,left=16,ant=8,beta=0.5,prewarm=4,dead=0.5,census`
    /// (empty = neutral).
    pub fn parse(s: &str) -> Result<SchedLaw, String> {
        let mut l = SchedLaw::default();
        for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (k, v) = part.split_once('=').unwrap_or((part, ""));
            let f = || v.parse::<f64>().map_err(|_| format!("--sched {k}={v}"));
            match k {
                "fd" => l.finish_decay = f()?,
                "alive" => l.alive_window = f()? as u64,
                "prot" => l.protect = f()? as u64,
                "left" => l.left_min = f()? as u32,
                "ant" => l.anticipate = f()? as u32,
                "beta" => l.admit_beta = f()?,
                "prewarm" => l.prewarm = f()? as u32,
                "dead" => l.prewarm_dead = f()?,
                "census" => l.census = true,
                _ => {
                    return Err(format!(
                        "--sched key {k} (fd alive prot left ant beta prewarm dead census)"
                    ))
                }
            }
        }
        Ok(l)
    }
    /// Whether the law changes nothing (FN18's replay exactly).
    #[must_use]
    pub fn neutral(&self) -> bool {
        self.finish_decay == 1.0 && self.protect == 0 && self.admit_beta == 0.0 && self.prewarm == 0
    }
}

/// A counting mover that remembers the VRAM slots demoted (the victims' slots) and the RAM slots
/// they went to.
#[derive(Debug, Default)]
struct LogMover {
    h2d: u64,
    d2h: Vec<(u32, u32)>,
}

impl ExpertMover for LogMover {
    fn nvme_to_ram(&mut self, _reads: &[(u64, u32)]) -> Result<(), TierError> {
        Ok(())
    }
    fn ram_to_vram(&mut self, _ram: u32, _vram: u32) -> Result<(), TierError> {
        self.h2d += 1;
        Ok(())
    }
    fn vram_to_ram(&mut self, vram: u32, ram: u32) -> Result<(), TierError> {
        self.d2h.push((vram, ram));
        Ok(())
    }
    fn flush(&mut self) -> Result<(), TierError> {
        Ok(())
    }
}

/// The census and the law's counters of one replay.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SchedCensus {
    /// Decode misses (RAM reads) by the record's earlier users: the requesting sequence itself,
    /// another LIVE sequence, only FINISHED sequences (dead heat), nobody (warm start only).
    pub miss_self: u64,
    /// The requesting sequence used it only in its PROMPT (prefill), not in decode.
    pub miss_self_prompt: u64,
    /// See [`Self::miss_self`].
    pub miss_live: u64,
    /// See [`Self::miss_self`].
    pub miss_dead: u64,
    /// See [`Self::miss_self`].
    pub miss_none: u64,
    /// Victims (records demoted) whose users include a live sequence vs only finished / none.
    pub victim_live: u64,
    /// See [`Self::victim_live`].
    pub victim_dead: u64,
    /// A sequence's misses in its first 200 decode steps, and of them those RESIDENT when the
    /// previous sequence finished (evicted after the finish = too early).
    pub first200_miss: u64,
    /// See [`Self::first200_miss`].
    pub first200_was_resident: u64,
    /// Misses in later steps (after its first 200).
    pub later_miss: u64,
    /// Decode steps a sequence lived in its first 200 / later (the per-step denominators).
    pub first200_steps: u64,
    /// See [`Self::first200_steps`].
    pub later_steps: u64,
    /// Finish-decays applied (records), protected records summed over steps, pre-warm copies.
    pub decays: u64,
    /// See [`Self::decays`].
    pub protected: u64,
    /// See [`Self::decays`].
    pub prewarm: u64,
    /// Steps the pre-warm copies ran (joins).
    pub prewarm_events: u64,
}

const NOSEQ: SeqId = SeqId::MAX;
/// Users kept per record (most recent distinct sequences).
const USERS: usize = 8;

/// FN104: FN18's exact replay ([`crate::route_trace::replay`]) with the scheduler's law and the
/// census (module docs). Rank `rank`, its records only.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn sched_replay(
    trace: &Path,
    annot: &Annot,
    rank: u32,
    per_rank: u32,
    stride: u64,
    params: &TierParams,
    profile: Option<&RouteProfile>,
    law_ins: &InsertLaw,
    law: &SchedLaw,
) -> Result<(ReplayOut, SchedCensus, Vec<String>), String> {
    let mut r = TraceReader::open(trace).map_err(|e| e.to_string())?;
    r.keep_raw = true;
    let (layers, topk, experts) = (r.layers, r.topk, r.experts);
    let h = replay_header(rank, layers, per_rank, stride);
    let mut tier =
        ExpertTier::new(&h, params.clone(), profile, 0, 1 << 40).map_err(|e| e.to_string())?;
    let records = (layers * per_rank) as usize;
    let first = rank * per_rank;
    let local = |gid: u32| -> Option<u32> {
        let (l, e) = (gid / experts, gid % experts);
        (e >= first && e < first + per_rank).then(|| l * per_rank + (e - first))
    };
    let mut mv = LogMover::default();
    let mut out = ReplayOut {
        per_layer: vec![Default::default(); layers as usize],
        buckets: std::collections::BTreeMap::new(),
        steps: 0,
    };
    let mut c = SchedCensus::default();
    let nseq = annot.seqs.len();
    let mut live = vec![false; nseq];
    // per record: its most recent users (seq, last step)
    let mut users: Vec<Vec<(SeqId, u64, bool)>> = vec![Vec::new(); records];
    let mut dws_bits: Vec<Vec<u64>> = vec![Vec::new(); nseq];
    let mut dws_n: Vec<u32> = vec![0; nseq];
    // per sequence: its working set (records), the first-200 bookkeeping, steps decoded
    let mut ws: Vec<Vec<u32>> = vec![Vec::new(); nseq];
    let mut ws_mark: Vec<u32> = vec![NOSEQ; records];
    let mut ws_bits: Vec<Vec<u64>> = vec![Vec::new(); nseq];
    let mut ws_at: Vec<[u32; 4]> = vec![[0; 4]; nseq];
    let mut dsteps: Vec<u32> = vec![0; nseq];
    let mut prompt_hist: Vec<Vec<(u32, u32)>> = vec![Vec::new(); nseq];
    // the resident set at the last finish (first-200 census) and each sequence's snapshot
    let mut res_at_finish: Vec<bool> = vec![false; records];
    let mut snap_of: Vec<Option<std::rc::Rc<Vec<bool>>>> = vec![None; nseq];
    let mut last_snap: Option<std::rc::Rc<Vec<bool>>> = None;
    // the protection window: per step the (record) list touched, and per record the touches in it
    let mut win: VecDeque<Vec<u32>> = VecDeque::new();
    let mut win_n: Vec<u32> = vec![0; records];
    // the census of concurrent overlap
    let (mut ov_pairs, mut ov_jacc, mut ov_steps, mut ov_shared) = (0u64, 0.0f64, 0u64, 0.0f64);
    let mut ant_done = vec![false; nseq];
    let mut step_idx = 0usize;
    let mut now: u64 = 0;
    let finish_of = |q: SeqId| annot.seqs[q as usize].finish;
    let alive_other = |users: &Vec<(SeqId, u64, bool)>,
                       me: SeqId,
                       live: &Vec<bool>,
                       now: u64,
                       win: u64|
     -> bool {
        users
            .iter()
            .any(|&(q, t, _)| q != me && live[q as usize] && now.saturating_sub(t) <= win)
    };
    let decay_seq = |q: SeqId,
                     tier: &mut ExpertTier,
                     users: &Vec<Vec<(SeqId, u64, bool)>>,
                     live: &Vec<bool>,
                     ws: &Vec<Vec<u32>>,
                     now: u64,
                     c: &mut SchedCensus| {
        if law.finish_decay == 1.0 {
            return;
        }
        let mut all = ws[q as usize].clone();
        all.sort_unstable();
        all.dedup();
        let dead: Vec<u32> = all
            .into_iter()
            .filter(|&rec| !alive_other(&users[rec as usize], q, live, now, law.alive_window))
            .collect();
        c.decays += dead.len() as u64;
        tier.sched_scale(&dead, law.finish_decay);
    };
    while let Some(s) = r.next_step().map_err(|e| e.to_string())? {
        if !s.kind.target() {
            continue;
        }
        let an = annot
            .steps
            .get(step_idx)
            .ok_or("the annotation is shorter than the trace")?;
        step_idx += 1;
        // the events before the step
        for ev in &an.events {
            match *ev {
                SchedEvent::Admitted { seq, .. } => {
                    live[seq as usize] = true;
                    snap_of[seq as usize] = last_snap.clone();
                }
                SchedEvent::Joined { seq, .. } => {
                    live[seq as usize] = true;
                    let ph = std::mem::take(&mut prompt_hist[seq as usize]);
                    if law.admit_beta > 0.0 {
                        let norm = (1.0 + f64::from(annot.seqs[seq as usize].prompt_rows))
                            .ln()
                            .max(1.0);
                        for &(rec, cnt) in &ph {
                            tier.sched_bump(
                                rec,
                                law.admit_beta * (1.0 + f64::from(cnt)).ln() / norm,
                            );
                        }
                    }
                    if law.prewarm > 0 {
                        c.prewarm_events += 1;
                        let mut by_layer: Vec<Vec<(u32, u32)>> = vec![Vec::new(); layers as usize];
                        for &(rec, cnt) in &ph {
                            by_layer[(rec / per_rank) as usize].push((cnt, rec));
                        }
                        for (l, v) in by_layer.iter_mut().enumerate() {
                            v.sort_unstable_by(|a, b| b.cmp(a));
                            let cands: Vec<u32> = v.iter().map(|x| x.1).collect();
                            let n = tier
                                .sched_prewarm(
                                    l as u32,
                                    &cands,
                                    law.prewarm_dead,
                                    law.prewarm as usize,
                                    &mut mv,
                                )
                                .map_err(|e| e.to_string())?;
                            c.prewarm += n as u64;
                        }
                        mv.d2h.clear();
                    }
                }
                SchedEvent::Finished { seq } => {
                    if live[seq as usize] {
                        live[seq as usize] = false;
                        if !ant_done[seq as usize] {
                            decay_seq(seq, &mut tier, &users, &live, &ws, now, &mut c);
                        }
                    }
                    if law.census {
                        for (rec, x) in res_at_finish.iter_mut().enumerate() {
                            *x = matches!(
                                tier.placement(
                                    rec as u32 / per_rank,
                                    first + rec as u32 % per_rank
                                ),
                                Some(Placement::Vram(_))
                            );
                        }
                        last_snap = Some(std::rc::Rc::new(res_at_finish.clone()));
                    }
                }
                SchedEvent::Finishing { .. } => {}
            }
        }
        // the oracle anticipation: a sequence within `anticipate` steps of its end counts as gone
        if law.anticipate > 0 && law.finish_decay != 1.0 {
            for &q in &an.seqs {
                if let Some(f) = finish_of(q) {
                    if !ant_done[q as usize]
                        && (f as usize) <= step_idx - 1 + law.anticipate as usize
                    {
                        ant_done[q as usize] = true;
                        live[q as usize] = false;
                        decay_seq(q, &mut tier, &users, &live, &ws, now, &mut c);
                        live[q as usize] = true;
                    }
                }
            }
        }
        let bucket = if s.kind == StepKind::Prefill {
            0
        } else {
            s.bucket
        };
        tier.set_max_inserts(law_ins.cap(bucket));
        tier.set_promote_budget(law_ins.budget(bucket));
        tier.set_second_chance(law_ins.second_chance(bucket));
        // the step's per-position records (rank-local)
        let pos_recs: Vec<Vec<u32>> = if s.kind == StepKind::Prefill {
            let mut v = Vec::new();
            let mut ph = Vec::new();
            for l in 0..s.layers() {
                for x in s.layer(l) {
                    if let Some(rec) = local(l as u32 * experts + u32::from(x.e)) {
                        v.push(rec);
                        ph.push((rec, u32::from(x.live)));
                    }
                }
            }
            if let Some(&q) = an.seqs.first() {
                let p = &mut prompt_hist[q as usize];
                for (rec, cnt) in ph {
                    match p.iter_mut().find(|x| x.0 == rec) {
                        Some(x) => x.1 += cnt,
                        None => p.push((rec, cnt)),
                    }
                }
            }
            vec![v]
        } else {
            let w = s.width.max(1);
            let n = (s.live / w).min(an.seqs.len() as u32);
            position_sets(&r.raw, layers, topk, experts, s.rows, n, w)
                .into_iter()
                .map(|set| set.into_iter().filter_map(local).collect())
                .collect()
        };
        // the census: misses by earlier users, before the routes touch anything
        if law.census && s.kind == StepKind::Decode {
            let mut seen: Vec<u32> = Vec::new();
            for (j, recs) in pos_recs.iter().enumerate() {
                let q = an.seqs[j];
                let d = dsteps[q as usize];
                let early = d < 200;
                for &rec in recs {
                    let ram = !matches!(
                        tier.placement(rec / per_rank, first + rec % per_rank),
                        Some(Placement::Vram(_))
                    );
                    if !ram {
                        continue;
                    }
                    if early {
                        c.first200_miss += 1;
                        if snap_of[q as usize]
                            .as_ref()
                            .is_some_and(|sn| sn[rec as usize])
                        {
                            c.first200_was_resident += 1;
                        }
                    } else {
                        c.later_miss += 1;
                    }
                    if seen.contains(&rec) {
                        continue;
                    }
                    seen.push(rec);
                    let u = &users[rec as usize];
                    let reqs: Vec<SeqId> = pos_recs
                        .iter()
                        .enumerate()
                        .filter(|(_, rs)| rs.binary_search(&rec).is_ok())
                        .map(|(i, _)| an.seqs[i])
                        .collect();
                    if u.is_empty() {
                        c.miss_none += 1;
                    } else if u.iter().any(|(x, _, d)| *d && reqs.contains(x)) {
                        c.miss_self += 1;
                    } else if u.iter().any(|(x, _, _)| reqs.contains(x)) {
                        c.miss_self_prompt += 1;
                    } else if u.iter().any(|(x, _, _)| live[*x as usize]) {
                        c.miss_live += 1;
                    } else {
                        c.miss_dead += 1;
                    }
                }
            }
            for &q in &an.seqs {
                if dsteps[q as usize] < 200 {
                    c.first200_steps += 1;
                } else {
                    c.later_steps += 1;
                }
            }
            // concurrent overlap: every 64 steps, the live sequences' last-64-step sets
            if an.seqs.len() >= 2 && now % 64 == 0 {
                let mut sets: Vec<Vec<u32>> = vec![Vec::new(); an.seqs.len()];
                for (rec, u) in users.iter().enumerate() {
                    for &(q, t, _) in u {
                        if now.saturating_sub(t) < 64 {
                            if let Some(j) = an.seqs.iter().position(|&x| x == q) {
                                sets[j].push(rec as u32);
                            }
                        }
                    }
                }
                for i in 0..sets.len() {
                    for j in i + 1..sets.len() {
                        let inter = sets[i]
                            .iter()
                            .filter(|x| sets[j].binary_search(x).is_ok())
                            .count();
                        let uni = sets[i].len() + sets[j].len() - inter;
                        if uni > 0 {
                            ov_jacc += inter as f64 / uni as f64;
                            ov_pairs += 1;
                        }
                    }
                }
                // the share of this step's routed records another live sequence used in 64 steps
                let mut sh = 0usize;
                let mut tot = 0usize;
                for (j, recs) in pos_recs.iter().enumerate() {
                    for rec in recs {
                        tot += 1;
                        if sets
                            .iter()
                            .enumerate()
                            .any(|(i, st)| i != j && st.binary_search(rec).is_ok())
                        {
                            sh += 1;
                        }
                    }
                }
                ov_shared += sh as f64 / tot.max(1) as f64;
                ov_steps += 1;
            }
        }
        // protection for this step's victim ranking (live sequences' last `protect` steps)
        if law.protect > 0 {
            let mut prot = tier.take_sched_protect();
            if prot.len() != records {
                prot = vec![false; records];
            } else {
                prot.iter_mut().for_each(|x| *x = false);
            }
            let mut n = 0u64;
            for recs in &win {
                for &rec in recs {
                    if prot[rec as usize] {
                        continue;
                    }
                    let ok = users[rec as usize].iter().any(|&(q, t, _)| {
                        live[q as usize]
                            && !ant_done[q as usize]
                            && now.saturating_sub(t) < law.protect
                            && (law.left_min == 0
                                || finish_of(q).is_none_or(|f| {
                                    (f as usize).saturating_sub(step_idx - 1)
                                        >= law.left_min as usize
                                }))
                    });
                    if ok {
                        prot[rec as usize] = true;
                        n += 1;
                    }
                }
            }
            c.protected += n;
            tier.set_sched_protect(prot);
        }
        // the step (FN18's replay, line for line)
        tier.begin_step();
        now += 1;
        let mut paths = crate::expert_tier::PathCounts::default();
        for l in 0..s.layers() {
            let reps = params.policy.row_gamma != 0.0;
            let routed: Vec<u32> = s
                .layer(l)
                .iter()
                .flat_map(|x| {
                    std::iter::repeat_n(
                        u32::from(x.e),
                        if reps { usize::from(x.live.max(1)) } else { 1 },
                    )
                })
                .collect();
            let pc = tier
                .resolve(l as u32, &routed, &mut mv)
                .map_err(|e| e.to_string())?;
            out.per_layer[l].resident += pc.resident;
            out.per_layer[l].ram += pc.ram;
            out.per_layer[l].nvme += pc.nvme;
            paths.resident += pc.resident;
            paths.ram += pc.ram;
            paths.nvme += pc.nvme;
        }
        tier.refill(&mut mv).map_err(|e| e.to_string())?;
        mv.d2h.clear();
        let rep = tier.end_step(&mut mv).map_err(|e| e.to_string())?;
        if law.census {
            for &(_, ram) in &mv.d2h {
                if let Some(rec) = tier.ram_record(ram) {
                    if users[rec as usize]
                        .iter()
                        .any(|(q, _, _)| live[*q as usize])
                    {
                        c.victim_live += 1;
                    } else {
                        c.victim_dead += 1;
                    }
                }
            }
        }
        mv.d2h.clear();
        let b = out.buckets.entry(bucket).or_default();
        b.steps += 1;
        b.paths.resident += paths.resident;
        b.paths.ram += paths.ram;
        b.paths.nvme += paths.nvme;
        b.inserts += rep.moves.inserts;
        b.demotions += rep.moves.demotions;
        b.refills += rep.moves.refills;
        out.steps += 1;
        // the touches: users, working sets, the protection window
        let mut touched: Vec<u32> = Vec::new();
        for (j, recs) in pos_recs.iter().enumerate() {
            let Some(&q) = an.seqs.get(j) else { continue };
            for &rec in recs {
                let u = &mut users[rec as usize];
                let dec = s.kind == StepKind::Decode;
                match u.iter_mut().find(|x| x.0 == q) {
                    Some(x) => {
                        x.1 = now;
                        x.2 |= dec;
                    }
                    None => {
                        if u.len() >= USERS {
                            let k = u
                                .iter()
                                .enumerate()
                                .min_by_key(|(_, x)| (live[x.0 as usize], x.1))
                                .map(|(k, _)| k)
                                .unwrap_or(0);
                            u.swap_remove(k);
                        }
                        u.push((q, now, dec));
                    }
                }
                if ws_mark[rec as usize] != q {
                    ws_mark[rec as usize] = q;
                    let b = &mut ws_bits[q as usize];
                    if b.is_empty() {
                        b.resize(records.div_ceil(64), 0);
                    }
                    let (wd, bit) = (rec as usize / 64, 1u64 << (rec % 64));
                    if b[wd] & bit == 0 {
                        b[wd] |= bit;
                        ws[q as usize].push(rec);
                    }
                }
                if dec {
                    let b = &mut dws_bits[q as usize];
                    if b.is_empty() {
                        b.resize(records.div_ceil(64), 0);
                    }
                    let (wd, bit) = (rec as usize / 64, 1u64 << (rec % 64));
                    if b[wd] & bit == 0 {
                        b[wd] |= bit;
                        dws_n[q as usize] += 1;
                    }
                }
                touched.push(rec);
            }
            if s.kind == StepKind::Decode {
                let d = &mut dsteps[q as usize];
                *d += 1;
                let k = match *d {
                    16 => Some(0),
                    64 => Some(1),
                    256 => Some(2),
                    _ => None,
                };
                if let Some(k) = k {
                    ws_at[q as usize][k] = dws_n[q as usize];
                }
                ws_at[q as usize][3] = dws_n[q as usize];
            }
        }
        if law.protect > 0 {
            touched.sort_unstable();
            touched.dedup();
            for &rec in &touched {
                win_n[rec as usize] += 1;
            }
            win.push_back(touched);
            while win.len() as u64 > law.protect {
                if let Some(old) = win.pop_front() {
                    for rec in old {
                        win_n[rec as usize] -= 1;
                    }
                }
            }
        }
    }
    let mut lines = Vec::new();
    if law.census {
        let decoded: Vec<usize> = (0..nseq).filter(|&q| dsteps[q] > 0).collect();
        let pct = |v: &mut Vec<f64>, p: f64| -> f64 {
            if v.is_empty() {
                return f64::NAN;
            }
            v.sort_by(f64::total_cmp);
            v[((v.len() - 1) as f64 * p).round() as usize]
        };
        let mut life: Vec<f64> = decoded.iter().map(|&q| f64::from(dsteps[q])).collect();
        let mut wsz: Vec<f64> = decoded.iter().map(|&q| f64::from(ws_at[q][3])).collect();
        let sat = |k: usize, need: u32| -> (f64, usize) {
            let v: Vec<f64> = decoded
                .iter()
                .filter(|&&q| dsteps[q] >= need)
                .map(|&q| f64::from(ws_at[q][k]))
                .collect();
            (v.iter().sum::<f64>() / v.len().max(1) as f64, v.len())
        };
        let (s16, n16) = sat(0, 16);
        let (s64, n64) = sat(1, 64);
        let (s256, n256) = sat(2, 256);
        lines.push(format!(
            "fnsched census seqs {} decoded {} | life p50 {:.0} p90 {:.0} steps | decode working set (rank-local records of {records}) p50 {:.0} p90 {:.0} | saturation mean @16 {s16:.0} (n {n16}) @64 {s64:.0} (n {n64}) @256 {s256:.0} (n {n256})",
            nseq,
            decoded.len(),
            pct(&mut life, 0.5),
            pct(&mut life, 0.9),
            pct(&mut wsz, 0.5),
            pct(&mut wsz, 0.9)
        ));
        lines.push(format!(
            "fnsched census overlap: live pairs {ov_pairs} | Jaccard of last-64-step sets mean {:.3} | a step's routed records another live sequence used in 64 steps {:.1} %",
            ov_jacc / ov_pairs.max(1) as f64,
            100.0 * ov_shared / ov_steps.max(1) as f64
        ));
        let m = (c.miss_self + c.miss_self_prompt + c.miss_live + c.miss_dead + c.miss_none).max(1)
            as f64;
        lines.push(format!(
            "fnsched census decode misses (distinct a step) {:.0}: own sequence's earlier DECODE use {:.1} % | its prompt only {:.1} % | another LIVE sequence's {:.1} % | only FINISHED sequences' (dead heat) {:.1} % | nobody's {:.1} %",
            m,
            100.0 * c.miss_self as f64 / m,
            100.0 * c.miss_self_prompt as f64 / m,
            100.0 * c.miss_live as f64 / m,
            100.0 * c.miss_dead as f64 / m,
            100.0 * c.miss_none as f64 / m
        ));
        let v = (c.victim_live + c.victim_dead).max(1) as f64;
        lines.push(format!(
            "fnsched census victims {:.0}: a live sequence among the users {:.1} % | finished / none only {:.1} %",
            v,
            100.0 * c.victim_live as f64 / v,
            100.0 * c.victim_dead as f64 / v
        ));
        lines.push(format!(
            "fnsched census first 200 steps: misses {} = {:.2} a sequence-step ({} steps), RESIDENT at the previous finish {:.1} % | later: {:.2} a sequence-step ({} steps)",
            c.first200_miss,
            c.first200_miss as f64 / c.first200_steps.max(1) as f64,
            c.first200_steps,
            100.0 * c.first200_was_resident as f64 / c.first200_miss.max(1) as f64,
            c.later_miss as f64 / c.later_steps.max(1) as f64,
            c.later_steps
        ));
    }
    lines.push(format!(
        "fnsched law: finish-decays {} records | protected {:.1} records a step | pre-warm copies {} over {} joins ({:.1} a join)",
        c.decays,
        c.protected as f64 / out.steps.max(1) as f64,
        c.prewarm,
        c.prewarm_events,
        c.prewarm as f64 / c.prewarm_events.max(1) as f64
    ));
    Ok((out, c, lines))
}

/// The inference's report line.
#[must_use]
pub fn infer_line(a: &Annot) -> String {
    let s = &a.stats;
    let decoded = a.seqs.iter().filter(|q| q.decode_steps > 0).count();
    format!(
        "fnsched infer: sequences {} (decoded {decoded}, prefill-only {}) | joins {} departures {} | steady checks {} self-best {:.2} % (self sim {:.3} vs best other {:.3}) | changed steps {} weak {} | anomalies {}",
        a.seqs.len(),
        s.prefill_only,
        s.joins,
        s.departures,
        s.steady,
        100.0 * s.steady_self_best as f64 / s.steady.max(1) as f64,
        s.self_sim,
        s.other_sim,
        s.decided,
        s.weak,
        s.anomalies
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fn104_law_parse_and_neutral() {
        assert!(SchedLaw::default().neutral());
        let l = SchedLaw::parse(
            "fd=0.25,alive=64,prot=32,left=16,ant=8,beta=0.5,prewarm=4,dead=0.5,census",
        )
        .unwrap();
        assert!(!l.neutral());
        assert_eq!(
            (l.protect, l.left_min, l.anticipate, l.prewarm),
            (32, 16, 8, 4)
        );
        assert!(SchedLaw::parse("bogus=1").is_err());
    }

    #[test]
    fn fn104_combinations_and_permutations() {
        assert_eq!(combinations(4, 2).len(), 6);
        assert_eq!(combinations(3, 0), vec![Vec::<usize>::new()]);
        assert_eq!(permutations(&[1, 2, 3]).len(), 6);
    }

    /// A synthetic served run under the serve plane's slot law: `seqs` requests, at most `conc`
    /// live, each with a TOPIC (its own preferred experts a layer; `topic` = the share of its
    /// routes drawn from it, 0 = uniform routes), departures by swap-remove in random order.
    /// Returns the true (sequence, target step index) before which each departure is seen.
    fn synth_seqs(path: &Path, seqs: u32, conc: usize, topic: f64) -> Vec<(SeqId, u32)> {
        use crate::route_trace::TraceWriter;
        let layers = 4u32;
        let mut w = TraceWriter::create(path, layers, 10, 512).unwrap();
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut topics: Vec<Vec<u32>> = Vec::new();
        let mut left: Vec<u32> = Vec::new();
        let mut ready: Vec<SeqId> = Vec::new();
        let mut next_seq = 0u32;
        let mut idx = 0u32;
        let mut gone = Vec::new();
        let row = |rnd: &mut dyn FnMut() -> u64, t: &[u32]| -> Vec<u32> {
            let mut r: Vec<u32> = Vec::new();
            while r.len() < 10 {
                let u = (rnd() % 10_000) as f64 / 10_000.0;
                let e = if u < topic {
                    t[(rnd() % t.len() as u64) as usize]
                } else {
                    (rnd() % 512) as u32
                };
                if !r.contains(&e) {
                    r.push(e);
                }
            }
            r
        };
        while next_seq < seqs || !ready.is_empty() {
            // admissions: a prefill chunk each, then the decode batch grows at its end
            while ready.len() < conc && next_seq < seqs {
                let q = next_seq;
                next_seq += 1;
                topics.push((0..40).map(|_| (rnd() % 512) as u32).collect());
                left.push(4 + (rnd() % 60) as u32);
                let rows = 32 + (rnd() % 64) as u32;
                let mut ids = Vec::new();
                for _ in 0..layers {
                    for _ in 0..rows {
                        ids.extend(row(&mut rnd, &topics[q as usize]));
                    }
                }
                w.target(StepKind::Prefill, rows, rows, 0, 1, &ids).unwrap();
                idx += 1;
                ready.push(q);
            }
            let n = ready.len() as u32;
            let b = [1u32, 2, 4, 8].into_iter().find(|&b| b >= n).unwrap();
            let mut ids = Vec::new();
            for _ in 0..layers {
                for j in 0..b as usize {
                    match ready.get(j) {
                        Some(&q) => ids.extend(row(&mut rnd, &topics[q as usize])),
                        None => ids.extend(0..10u32),
                    }
                }
            }
            w.target(StepKind::Decode, b, n, b, 1, &ids).unwrap();
            idx += 1;
            // departures (random order, swap-remove): seen gone before the next target step
            let mut done: Vec<SeqId> = ready
                .iter()
                .copied()
                .filter(|&q| {
                    left[q as usize] -= 1;
                    left[q as usize] == 0
                })
                .collect();
            if done.len() > 1 {
                let k = (rnd() % done.len() as u64) as usize;
                done.rotate_left(k);
            }
            for q in done {
                let k = ready.iter().position(|&x| x == q).unwrap();
                ready.swap_remove(k);
                gone.push((q, idx));
            }
        }
        w.flush().unwrap();
        gone
    }

    #[test]
    fn fn104_the_derived_stream_recovers_the_slot_law_and_neutral_replay_is_fn18s() {
        use crate::route_trace::replay;
        let d = saber_test_scratch::ScratchDir::new("fn104-sched");
        let p = d.join("trace.bin");
        let truth = synth_seqs(&p, 60, 8, 0.6);
        let a = infer(&p, &InferOpts::default()).unwrap();
        assert_eq!(a.seqs.len(), 60, "{}", infer_line(&a));
        let mut right = 0;
        for &(q, at) in &truth {
            // the next requests' prefills may sit between the departure and the next decode
            if a.seqs[q as usize]
                .finish
                .is_some_and(|f| f >= at && f <= at + 9)
            {
                right += 1;
            }
        }
        // the batch's last departures are seen only at the trace's end
        assert!(
            right * 100 >= (truth.len() - 8) * 95,
            "{right} of {} departures: {}",
            truth.len(),
            infer_line(&a)
        );
        assert!(
            a.stats.steady_self_best * 10 >= a.stats.steady * 9,
            "{}",
            infer_line(&a)
        );
        // the negative control: no topic, no signal -- the steady check falls to chance
        let p2 = d.join("trace_uniform.bin");
        synth_seqs(&p2, 60, 8, 0.0);
        let u = infer(&p2, &InferOpts::default()).unwrap();
        assert!(
            u.stats.steady_self_best * 2 < u.stats.steady,
            "control: {}",
            infer_line(&u)
        );
        // the neutral law is FN18's replay exactly; the arms serve every route on one path
        let params = TierParams {
            vram_slots: 4 * 40,
            ram_slots: 4 * 216,
            max_inserts: 2,
            read_through: 0.5,
            lrfu_lambda: 0.98,
            ..TierParams::default()
        };
        let ins = InsertLaw::parse(2, "").unwrap();
        let today = replay(&p, 0, 256, 4096, &params, None, &ins, None, None).unwrap();
        let (n, _, _) = sched_replay(
            &p,
            &a,
            0,
            256,
            4096,
            &params,
            None,
            &ins,
            &SchedLaw::default(),
        )
        .unwrap();
        assert_eq!(n, today, "the neutral law == FN18's replay");
        let routed = |o: &ReplayOut| {
            o.per_layer
                .iter()
                .map(|c| c.resident + c.ram + c.nvme)
                .sum::<u64>()
        };
        for cell in [
            "fd=0",
            "fd=0,ant=8",
            "prot=16,left=4",
            "beta=1",
            "prewarm=4,dead=1",
            "census",
        ] {
            let law = SchedLaw::parse(cell).unwrap();
            let (o, c, _) = sched_replay(&p, &a, 0, 256, 4096, &params, None, &ins, &law).unwrap();
            assert_eq!(
                routed(&o),
                routed(&today),
                "{cell}: every route on one path"
            );
            if cell == "census" {
                assert_eq!(o, today, "the census observes only");
                assert!(
                    c.miss_self + c.miss_self_prompt + c.miss_live + c.miss_dead + c.miss_none > 0
                );
            }
            if cell == "fd=0" {
                assert!(c.decays > 0);
            }
            if cell.starts_with("prot") {
                assert!(c.protected > 0);
            }
        }
    }

    #[test]
    fn fn104_position_sets_split_rows_by_position() {
        // 1 layer, topk 2, 2 positions x width 2 = 4 rows
        let raw: Vec<u16> = vec![1, 2, 3, 2, 7, 8, 7, 9];
        let s = position_sets(&raw, 1, 2, 512, 4, 2, 2);
        assert_eq!(s, vec![vec![1, 2, 3], vec![7, 8, 9]]);
    }
}
