//! LR-F: a discrete-event replay of the REAL scheduler core (`saber_sched::Scheduler`) against a
//! modelled Flash-Next step time, in virtual time.
//!
//! The scheduler runs unmodified on a virtual clock ([`Scheduler::with_clock`]); the executor
//! advances that clock by the modelled duration of every step it is handed, emits ~2.6 tokens a
//! row a decode step (speculative acceptance) and EOS at each request's drawn output length. The
//! driver submits closed-loop sessions' requests at their virtual arrival times and honours the
//! admit-slack hold the scheduler asks for. Everything the scheduler measures (stalled decoder
//! time, prefill-alone time) is therefore in model time.
//!
//! Step model (`StepModel::flashnext`): decode `T(b) = 12 + 15.5 (b - 1) / 7` ms (FN165's b1 ~12 /
//! b8 27.5, LINEAR INTERPOLATION in between); prefill pass `760 + 0.075 rows` ms above 2,048 rows
//! (LR-B's fit, the fixed ring staging) and `60 + 0.075 rows` at or below (FN157's 82-85 ms short
//! passes). Shared with `tests/lrf_admit_slack.rs` through `#[path]`.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use saber_mem::ledger::{Ledger, Pool};
use saber_plan::{Bucket, Phase};
use saber_sched::{
    AdmitPolicy, AdmittedSeq, ArenaBytes, KvGeometry, Scheduler, SchedulerConfig, SeqId, SeqStep,
    StepBatch, StepExecutor, StepOutcome, StepResult,
};
use saber_serve::engine::{Engine, EngineRequest, Submission, TokenId};
use saber_serve::sampling::SamplingParams;

pub const EOS: TokenId = 7;
const GIB: u64 = 1 << 30;

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(x: &mut u64) -> f64 {
    (splitmix(x) >> 11) as f64 / (1u64 << 53) as f64
}

fn exp_draw(x: &mut u64, mean: f64) -> f64 {
    -mean * (1.0 - unit(x)).max(1e-12).ln()
}

/// The modelled step walls, ms.
#[derive(Clone, Copy, Debug)]
pub struct StepModel {
    pub dec_b1_ms: f64,
    pub dec_b8_ms: f64,
    pub pass_fixed_ms: f64,
    pub pass_small_fixed_ms: f64,
    pub per_row_ms: f64,
    /// P(a draft position is accepted), a geometric prefix capped at the slot's verify width.
    pub accept_p: f64,
    /// The fraction of a chunk's rows the prefix tree does NOT reload (LR-D: 0.84 in the live
    /// thrash window, 0.54 cumulative); the pass's fixed cost is paid whole.
    pub computed: f64,
}

impl StepModel {
    pub fn flashnext() -> Self {
        Self {
            dec_b1_ms: 12.0,
            dec_b8_ms: 27.5,
            pass_fixed_ms: 760.0,
            pass_small_fixed_ms: 60.0,
            per_row_ms: 0.075,
            accept_p: 0.66,
            computed: 0.84,
        }
    }
    pub fn decode_ms(&self, rows: usize) -> f64 {
        let b = rows.clamp(1, 8) as f64;
        self.dec_b1_ms + (self.dec_b8_ms - self.dec_b1_ms) * (b - 1.0) / 7.0
    }
    pub fn pass_ms(&self, rows: u32) -> f64 {
        let fixed = if rows <= 2048 {
            self.pass_small_fixed_ms
        } else {
            self.pass_fixed_ms
        };
        fixed + self.per_row_ms * f64::from(rows) * self.computed
    }
}

/// The served Flash-Next scheduler geometry (`flashnext::serve::scheduler_config(262144, 5,
/// false, 8192, 0, Some(kv_tokens))`): decode 1/2/4/8 x width 5, prefill rungs 2,048 / 8,192,
/// admission by reservation over `kv_tokens` (prod: 272,528 a rank), FN51's implicit clamp.
pub fn flashnext_config(kv_tokens: u64, admit: AdmitPolicy, yield_steps: u32) -> SchedulerConfig {
    let max_ctx = 262_144u32;
    let width = 5;
    let kv = KvGeometry {
        block_tokens: 16,
        bytes_per_token_per_rank: 13_312,
    };
    let mut buckets: Vec<Bucket> = [1u32, 2, 4, 8]
        .iter()
        .map(|&b| Bucket::decode(b, width, max_ctx, true))
        .collect();
    buckets.push(Bucket::prefill(2048, max_ctx, true));
    buckets.push(Bucket::prefill(8192, max_ctx, true));
    let gdn_slot = 36 * 24 * 128 * 128 * 2;
    let arenas = |gdn| ArenaBytes {
        kv_per_rank: kv_tokens * kv.bytes_per_token_per_rank,
        gdn_state_per_rank: gdn,
    };
    let probe = SchedulerConfig::for_mock(
        buckets.clone(),
        2,
        gdn_slot,
        kv,
        arenas(1 << 40),
        max_ctx as usize,
        "lrf:replay".into(),
    )
    .expect("probe");
    let per_seq = probe.gdn_bytes_per_seq();
    let mut c = SchedulerConfig::for_mock(
        buckets,
        2,
        gdn_slot,
        kv,
        arenas(8 * per_seq),
        max_ctx as usize,
        "lrf:replay".into(),
    )
    .expect("config");
    c.implicit_context = Some(10_240);
    c.implicit_headroom = 16_384;
    c.prefill = c.prefill.clone().with_yield_steps(yield_steps);
    c.admit = admit;
    c
}

pub fn ledger() -> Ledger {
    let mut l = Ledger::new();
    for r in 0..2 {
        l.declare(Pool::Vram(r), 32 * GIB).expect("vram");
    }
    l.declare(Pool::HostPinned, 4 * GIB).expect("pinned");
    l.declare(Pool::HostPageable, 48 * GIB).expect("pageable");
    l.declare(Pool::Nvme, 64 * GIB).expect("nvme");
    l
}

/// One request of the trace (the executor reads `session / turn / out_len` from its first three
/// prompt ids).
#[derive(Clone, Copy, Debug)]
pub struct Req {
    pub session: u32,
    pub turn: u32,
    pub prompt: usize,
    pub out_len: usize,
    pub max_tokens: usize,
}

/// A closed-loop trace: each session sends its next request `think` after its previous one
/// finished.
#[derive(Clone, Debug)]
pub struct Trace {
    pub name: String,
    pub sessions: u32,
    pub kind: TraceKind,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug)]
pub enum TraceKind {
    /// The live agent shape (LR-A/LR-D, 10-23): a 7,801-token system prompt + a 0-8k task, +~2.2k
    /// a turn (the previous answer + a 0.5-1.5k tool result), ~1.2k generated a turn (exp), an
    /// explicit `max_tokens`, a tool/think gap (exp) between turns, a fresh task after 4-12 turns
    /// (mean prompt ~17-20k = LR-A's 17.5k).
    Agent {
        think_ms: f64,
        max_tokens: usize,
        out_mean: f64,
    },
    /// BetterBench `--quick` conc-8 shape: short prompts, ~400 generated, the client turns
    /// around in ~1.7 ms (FN157).
    Closed {
        prompt: usize,
        out_len: usize,
        max_tokens: usize,
        think_ms: f64,
    },
}

struct Session {
    rng: u64,
    turn: u32,
    turns_left: u32,
    context: usize,
}

impl Trace {
    fn session(&self, s: u32) -> Session {
        let mut rng = self.seed ^ (u64::from(s) + 1).wrapping_mul(0xA24B_AED4_963E_E407);
        splitmix(&mut rng);
        let mut x = Session {
            rng,
            turn: 0,
            turns_left: 0,
            context: 0,
        };
        self.new_task(&mut x);
        x
    }
    fn new_task(&self, x: &mut Session) {
        x.turns_left = 4 + (splitmix(&mut x.rng) % 9) as u32;
        x.context = 7_801 + (unit(&mut x.rng) * 8_000.0) as usize;
    }
    /// The session's next request and the think gap before it (ms).
    fn next(&self, s: u32, x: &mut Session) -> (Req, f64) {
        match self.kind {
            TraceKind::Agent {
                think_ms,
                max_tokens,
                out_mean,
            } => {
                if x.turns_left == 0 {
                    self.new_task(x);
                }
                x.turns_left -= 1;
                let out = (exp_draw(&mut x.rng, out_mean) as usize).clamp(16, max_tokens);
                let prompt = x.context.min(200_000);
                let req = Req {
                    session: s,
                    turn: x.turn,
                    prompt,
                    out_len: out,
                    max_tokens,
                };
                x.turn += 1;
                // the next turn carries this answer + a tool result
                x.context += out + 500 + (unit(&mut x.rng) * 1_000.0) as usize;
                (req, exp_draw(&mut x.rng, think_ms))
            }
            TraceKind::Closed {
                prompt,
                out_len,
                max_tokens,
                think_ms,
            } => {
                let jitter = (unit(&mut x.rng) * 0.2 - 0.1) * out_len as f64;
                let req = Req {
                    session: s,
                    turn: x.turn,
                    prompt,
                    out_len: ((out_len as f64 + jitter) as usize).clamp(1, max_tokens),
                    max_tokens,
                };
                x.turn += 1;
                (req, think_ms)
            }
        }
    }
}

#[derive(Debug, Default)]
struct SeqLive {
    session: u32,
    turn: u32,
    out_len: usize,
    generated: usize,
    rng: u64,
    last_commit_us: Option<u64>,
}

/// The modelled executor: advances the virtual clock by each step's modelled wall.
#[derive(Debug)]
pub struct ReplayExec {
    clock_us: Arc<AtomicU64>,
    model: StepModel,
    seqs: HashMap<SeqId, SeqLive>,
    /// (session, turn) -> first-token time, us.
    pub first_token_us: BTreeMap<(u32, u32), u64>,
    /// (session, turn) finished in the last step.
    pub finished: Vec<(u32, u32)>,
    /// Inter-commit gaps, us, with their end time (for windowing).
    pub gaps: Vec<(u64, u64)>,
    /// (end us, generated tokens) per step.
    pub tokens: Vec<(u64, u32)>,
    pub decode_steps: u64,
    pub decode_rows: u64,
    /// LR-G: (session, turn) -> tokens generated so far (a preempted sequence's recompute
    /// continues its count, its EOS stays where it was drawn).
    progress: HashMap<(u32, u32), usize>,
    /// LR-G: slots handed a step that writes past their KV blocks (the mid-step failure a real
    /// engine would meet; must stay 0).
    pub outran: u64,
}

impl ReplayExec {
    pub fn new(clock_us: Arc<AtomicU64>, model: StepModel) -> Self {
        Self {
            clock_us,
            model,
            seqs: HashMap::new(),
            first_token_us: BTreeMap::new(),
            finished: Vec::new(),
            gaps: Vec::new(),
            tokens: Vec::new(),
            decode_steps: 0,
            decode_rows: 0,
            progress: HashMap::new(),
            outran: 0,
        }
    }
}

impl StepExecutor for ReplayExec {
    fn execute(&mut self, batch: &StepBatch<'_>) -> StepResult {
        let ms = match batch.phase {
            Phase::Prefill => self.model.pass_ms(
                batch
                    .slots
                    .iter()
                    .filter_map(|s| s.chunk.map(|c| c.prompt_tokens))
                    .sum(),
            ),
            Phase::Decode => {
                self.decode_steps += 1;
                self.decode_rows += batch.slots.len() as u64;
                self.model.decode_ms(batch.slots.len())
            }
        };
        let end = self.clock_us.load(Ordering::Relaxed) + (ms * 1e3) as u64;
        self.clock_us.store(end, Ordering::Relaxed);
        let mut out = Vec::with_capacity(batch.slots.len());
        let mut committed = 0u32;
        for slot in batch.slots {
            // the rows this step writes: a chunk's, or the verify's (committed + drafts)
            let rows_end = match slot.chunk {
                Some(c) => c.tokens_done_before + c.prompt_tokens,
                None => slot.context_len + 1 + slot.verify_width,
            };
            if u64::from(rows_end) > slot.kv_blocks.len() as u64 * 16 {
                self.outran += 1;
            }
            let s = self.seqs.get_mut(&slot.seq_id).expect("admitted");
            if slot.chunk.is_some_and(|c| !c.last) {
                out.push(SeqStep::speculative(slot.seq_id, Vec::new(), 0, 0));
                continue;
            }
            let mut accepted = 0;
            if slot.chunk.is_none() {
                while accepted < slot.verify_width && unit(&mut s.rng) < self.model.accept_p {
                    accepted += 1;
                }
            }
            let mut toks = Vec::with_capacity(accepted as usize + 1);
            let mut live = 0;
            for _ in 0..=accepted {
                if s.generated >= s.out_len {
                    toks.push(EOS + 1);
                    continue;
                }
                s.generated += 1;
                live += 1;
                toks.push(if s.generated >= s.out_len { EOS } else { 1000 });
            }
            committed += live;
            self.progress.insert((s.session, s.turn), s.generated);
            if let Some(prev) = s.last_commit_us {
                self.gaps.push((end, end - prev));
            } else {
                self.first_token_us
                    .entry((s.session, s.turn))
                    .or_insert(end);
            }
            s.last_commit_us = Some(end);
            if s.generated >= s.out_len {
                self.finished.push((s.session, s.turn));
            }
            let drafted = if slot.chunk.is_none() {
                slot.verify_width
            } else {
                0
            };
            out.push(SeqStep::speculative(slot.seq_id, toks, drafted, accepted));
        }
        self.tokens.push((end, committed));
        StepResult::new(out)
    }

    fn admit(&mut self, seq: &AdmittedSeq) {
        let p = &seq.prompt_tokens;
        let (session, turn, out_len) = (p[0] - 100, p[1] - 100, (p[2] - 100) as usize);
        let rng = 0xD1B5_4A32_D192_ED03 ^ (u64::from(session) << 32 | u64::from(turn));
        let generated = self.progress.get(&(session, turn)).copied().unwrap_or(0);
        self.seqs.insert(
            seq.seq_id,
            SeqLive {
                session,
                turn,
                out_len,
                generated,
                rng: rng ^ generated as u64,
                ..SeqLive::default()
            },
        );
    }

    fn release(&mut self, seq: SeqId) {
        self.seqs.remove(&seq);
    }
}

/// What one replay measured over its window.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub label: String,
    pub window_s: f64,
    pub tok_s: f64,
    /// Time-averaged sequences running (admitted, prefilling or decoding).
    pub mean_running: f64,
    /// Mean rows of a decode step (per step).
    pub mean_decode_rows: f64,
    /// Decoder-ms frozen behind prefill-alone steps, per second of window.
    pub stall_ms_per_s: f64,
    pub gap_p50_ms: f64,
    pub gap_p99_ms: f64,
    pub gap_p999_ms: f64,
    pub gap_max_ms: f64,
    pub ttft_p50_s: f64,
    pub ttft_p99_s: f64,
    pub requests: usize,
    /// Mean prompt of the requests that arrived in the window.
    pub prompt_mean: f64,
    pub waiting_mean: f64,
    pub holds: u64,
    pub caught: u64,
    pub held_ms: f64,
    pub declined: u64,
    pub hol_skips: u64,
    pub refused_kv: u64,
    /// LR-G: lazy growths, preemptions, recompute rows, margin refusals.
    pub lazy_grows: u64,
    pub preemptions: u64,
    pub recompute_tokens: u64,
    pub margin_refusals: u64,
    /// Time-averaged reserved / written KV tokens over the window (the slab's use).
    pub reserved_mean: f64,
    pub written_mean: f64,
    /// Slots stepped past their blocks (must be 0).
    pub outran: u64,
    /// The schedule's fingerprint: (phase, rows) of every step, hashed.
    pub schedule_hash: u64,
    pub steps: u64,
}

/// A replay variant.
#[derive(Clone, Debug)]
pub struct Variant {
    pub label: String,
    pub admit: AdmitPolicy,
    pub yield_steps: u32,
    /// LR-F's "smaller reserve" what-if: reserve `max(X, out_len)` (growth that always succeeds;
    /// an upper bound on a grow-on-demand reservation), 0 = the request's `max_tokens`.
    pub reserve_cap: usize,
}

impl Variant {
    pub fn base() -> Self {
        Self {
            label: "base".into(),
            admit: AdmitPolicy::default(),
            yield_steps: 0,
            reserve_cap: 0,
        }
    }
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let i = ((v.len() - 1) as f64 * q).round() as usize;
    v[i]
}

/// Run `trace` under `variant` for `horizon_s` of virtual time; measure over `[warm_s, horizon_s]`.
pub fn run(
    trace: &Trace,
    variant: &Variant,
    model: StepModel,
    kv_tokens: u64,
    warm_s: f64,
    horizon_s: f64,
) -> Report {
    let clock_us = Arc::new(AtomicU64::new(0));
    let base = Instant::now();
    let c2 = Arc::clone(&clock_us);
    let config = flashnext_config(kv_tokens, variant.admit, variant.yield_steps);
    let model_info = saber_sched::mock_model_info("lrf", "/replay", EOS, 262_144);
    let sched = Scheduler::new(
        config,
        ReplayExec::new(Arc::clone(&clock_us), model),
        model_info,
        ledger(),
    )
    .expect("arenas")
    .with_clock(Arc::new(move || {
        base + Duration::from_micros(c2.load(Ordering::Relaxed))
    }));

    let horizon = (horizon_s * 1e6) as u64;
    let warm = (warm_s * 1e6) as u64;
    let mut sessions: Vec<Session> = (0..trace.sessions).map(|s| trace.session(s)).collect();
    // pending arrivals: (time us, session, request)
    let mut pending: Vec<(u64, Req)> = Vec::new();
    for (s, x) in sessions.iter_mut().enumerate() {
        let (req, _) = trace.next(s as u32, x);
        // stagger the first wave over 2 s
        let mut r = trace.seed ^ s as u64;
        pending.push(((unit(&mut r) * 2e6) as u64, req));
    }
    let mut live: HashMap<(u32, u32), (u64, Submission)> = HashMap::new();
    let mut arrivals: BTreeMap<(u32, u32), u64> = BTreeMap::new();
    let mut ttft: Vec<f64> = Vec::new();
    let (mut run_area, mut wait_area, mut area_t) = (0f64, 0f64, 0f64);
    let (mut res_area, mut len_area) = (0f64, 0f64);
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut steps = 0u64;
    let (mut prompt_sum, mut prompt_n) = (0f64, 0f64);
    let mut warm_counters = None;
    let mut warm_exec = (0u64, 0u64);

    let submit = |req: &Req, at: u64, live: &mut HashMap<(u32, u32), (u64, Submission)>| {
        let mut ids: Vec<TokenId> = vec![100; req.prompt.max(3)];
        ids[0] = 100 + req.session;
        ids[1] = 100 + req.turn;
        ids[2] = 100 + req.out_len as u32;
        let max_tokens = if variant.reserve_cap > 0 {
            variant.reserve_cap.max(req.out_len).min(req.max_tokens)
        } else {
            req.max_tokens
        };
        let sub = sched
            .submit(EngineRequest {
                prompt_token_ids: ids,
                params: SamplingParams {
                    max_tokens: Some(max_tokens),
                    ..SamplingParams::default()
                },
            })
            .expect("submit");
        live.insert((req.session, req.turn), (at, sub));
    };

    loop {
        let now = clock_us.load(Ordering::Relaxed);
        if now >= horizon {
            break;
        }
        if warm_counters.is_none() && now >= warm {
            warm_counters = Some(sched.counters());
            warm_exec = sched.with_executor(|e| (e.decode_steps, e.decode_rows));
        }
        pending.sort_by_key(|p| std::cmp::Reverse(p.0));
        while pending.last().is_some_and(|p| p.0 <= now) {
            let (at, req) = pending.pop().expect("non-empty");
            arrivals.insert((req.session, req.turn), at);
            if at >= warm {
                prompt_sum += req.prompt as f64;
                prompt_n += 1.0;
            }
            // the request reaches the queue at its own instant (mid-step), not at the step's end
            clock_us.store(at, Ordering::Relaxed);
            submit(&req, at, &mut live);
            clock_us.store(now, Ordering::Relaxed);
        }
        let stats = sched.stats();
        let (kvs, _) = sched.kv_admission();
        let t0 = now;
        match sched.step() {
            StepOutcome::Idle => {
                let next = pending.last().map_or(horizon, |p| p.0);
                clock_us.store(next.max(now + 1), Ordering::Relaxed);
            }
            StepOutcome::NoBucket(e) => panic!("no bucket: {e:?}"),
            StepOutcome::Ran(sum) => {
                steps += 1;
                hash = (hash ^ (u64::from(sum.bucket.m()) << 8 | u64::from(sum.sequences)))
                    .wrapping_mul(0x100_0000_01B3);
            }
        }
        let t_step = clock_us.load(Ordering::Relaxed);
        // finishes -> the session's next request after its think gap
        let done = sched.with_executor(|e| std::mem::take(&mut e.finished));
        for key in done {
            let first = sched.with_executor(|e| e.first_token_us.get(&key).copied());
            if let (Some(a), Some(f)) = (arrivals.get(&key), first) {
                if *a >= warm {
                    ttft.push((f - a) as f64 / 1e6);
                }
            }
            live.remove(&key);
            let x = &mut sessions[key.0 as usize];
            let (req, think) = trace.next(key.0, x);
            pending.push((t_step + (think * 1e3) as u64, req));
        }
        // the admit-slack hold the step earned (None with the arm off)
        if let Some(cap) = sched.admit_slack_hold() {
            let now = clock_us.load(Ordering::Relaxed);
            let until = now + cap.as_micros() as u64;
            let next = pending.iter().map(|p| p.0).min().unwrap_or(u64::MAX);
            let (end, caught) = if next <= until {
                (next, true)
            } else {
                (until, false)
            };
            clock_us.store(end, Ordering::Relaxed);
            sched.hold_ended(Duration::from_micros(end - now), caught);
        }
        let t1 = clock_us.load(Ordering::Relaxed);
        if t0 >= warm {
            let dt = (t1 - t0) as f64;
            run_area += stats.num_requests_running as f64 * dt;
            wait_area += stats.num_requests_waiting as f64 * dt;
            res_area += kvs.reserved_tokens as f64 * dt;
            len_area += kvs.length_tokens as f64 * dt;
            area_t += dt;
        }
    }

    let end_counters = sched.counters();
    let w = warm_counters.unwrap_or_default();
    let window = (horizon - warm) as f64 / 1e6;
    let (toks, mut gaps, dsteps, drows) = sched.with_executor(|e| {
        let toks: u64 = e
            .tokens
            .iter()
            .filter(|(t, _)| *t >= warm && *t <= horizon)
            .map(|(_, n)| u64::from(*n))
            .sum();
        let gaps: Vec<f64> = e
            .gaps
            .iter()
            .filter(|(t, _)| *t >= warm)
            .map(|(_, g)| *g as f64 / 1e3)
            .collect();
        (
            toks,
            gaps,
            e.decode_steps - warm_exec.0,
            e.decode_rows - warm_exec.1,
        )
    });
    let arm = sched.admit_arm_counters();
    let adm = sched.admission_counters();
    let gap_max = gaps.iter().copied().fold(0.0, f64::max);
    Report {
        label: variant.label.clone(),
        window_s: window,
        tok_s: toks as f64 / window,
        mean_running: run_area / area_t.max(1.0),
        waiting_mean: wait_area / area_t.max(1.0),
        mean_decode_rows: drows as f64 / dsteps.max(1) as f64,
        stall_ms_per_s: (end_counters.stalled_decoder_us - w.stalled_decoder_us) as f64
            / 1e3
            / window,
        gap_p50_ms: pct(&mut gaps, 0.50),
        gap_p99_ms: pct(&mut gaps, 0.99),
        gap_p999_ms: pct(&mut gaps, 0.999),
        gap_max_ms: gap_max,
        ttft_p50_s: pct(&mut ttft, 0.50),
        ttft_p99_s: pct(&mut ttft, 0.99),
        requests: ttft.len(),
        prompt_mean: prompt_sum / prompt_n.max(1.0),
        holds: arm.slack_holds,
        caught: arm.slack_caught,
        held_ms: arm.slack_held_us as f64 / 1e3,
        declined: arm.slack_declined,
        hol_skips: arm.hol_skips,
        refused_kv: adm.refused_kv,
        lazy_grows: arm.lazy_grows,
        preemptions: arm.lazy_preemptions,
        recompute_tokens: arm.lazy_recompute_tokens,
        margin_refusals: arm.lazy_margin_refusals,
        reserved_mean: res_area / area_t.max(1.0),
        outran: sched.with_executor(|e| e.outran),
        written_mean: len_area / area_t.max(1.0),
        schedule_hash: hash,
        steps,
    }
}
