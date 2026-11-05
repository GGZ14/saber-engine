//! **A RAGGED STEP IS THE SAME STREAM** (workstream C2, host tests a and b).
//!
//! A decode ROW BAND runs every slot at its OWN verify width, the rows packed by the step's `cu`
//! prefix and the band's tail padded. This file drives the real executor over [`NullForward`] —
//! whose rows are a fixed function of the ids each slot's rows are READ from, by `cu`, exactly as
//! the device reads them — and asserts:
//!
//!   (a) THE ENGINE-IDENTITY LAW: requests decoded together in bands, each at its own and
//!       changing width (`[5, 3, 8, 2, 5, 1, 4, 5]`-style multisets, a first step at one row),
//!       commit each request's SOLO width-1 ids, token for token — greedy and seeded;
//!   (b) THE PADDING IS NEVER READ: the band's tail rows poisoned with an id the model would
//!       answer differently change no live row, and the mock's own check (every row past `cu`
//!       at `slot` −1 / position 0, every padding slot inert, `naccept >= 1`) holds on every step.
//!
//! NEGATIVE CONTROL: the same run with each slot read at the RECTANGLE's stride (`i * max_width`)
//! instead of by `cu` reads its neighbours' rows and the padding, and diverges from the solo ids.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use saber_engine::{Drafter, NullForward, RealExecutor, RecordedDrafter};
use saber_plan::{Bucket, Phase};
use saber_sched::{
    band_rows_for, AdmittedSeq, BatchSlot, PrefillChunk, SeqId, SlotSampling, StepBatch,
    StepExecutor, DEFAULT_ROW_BANDS,
};
use saber_serve::engine::{RequestId, TokenId};
use saber_serve::sampling::SamplingParams;

const PREFILL: u32 = 2048;
const BATCHES: &[u32] = &[1, 2, 4, 8];
const WIDTHS: &[u32] = &[1, 5, 8];
const MAX_W: u32 = 8;
const PROMPT: usize = 12;
const TOKENS: usize = 40;

/// A drafter proposing each sequence's REFERENCE continuation, wrong at every `p % 5 == 3`, so
/// steps accept anywhere from 0 to all of their guesses.
#[derive(Clone, Debug, Default)]
struct Oracle(Arc<Mutex<Scripts>>);

/// Per sequence: its reference stream and how much of it it has committed.
type Scripts = BTreeMap<SeqId, (Vec<TokenId>, usize)>;

impl Oracle {
    fn script(&self, seq: SeqId, reference: &[TokenId], at: usize) {
        self.0
            .lock()
            .expect("oracle")
            .insert(seq, (reference.to_vec(), at));
    }
}

impl Drafter for Oracle {
    fn name(&self) -> &'static str {
        "oracle"
    }
    fn max_width(&self) -> u32 {
        MAX_W - 1
    }
    fn propose(&mut self, seq: SeqId, _last: TokenId, width: u32) -> Vec<TokenId> {
        let st = self.0.lock().expect("oracle");
        let Some((reference, at)) = st.get(&seq) else {
            return Vec::new();
        };
        (0..width as usize)
            .map(|i| {
                let p = at + i;
                if p % 5 == 3 {
                    RecordedDrafter::WRONG
                } else {
                    reference.get(p).copied().unwrap_or(RecordedDrafter::WRONG)
                }
            })
            .collect()
    }
    fn commit(&mut self, seq: SeqId, committed: &[TokenId]) {
        if let Some((_, at)) = self.0.lock().expect("oracle").get_mut(&seq) {
            *at += committed.len();
        }
    }
    fn release(&mut self, seq: SeqId) {
        self.0.lock().expect("oracle").remove(&seq);
    }
}

#[derive(Clone, Debug)]
struct Req {
    seq: u64,
    params: SamplingParams,
    ids: Vec<TokenId>,
    last_commit: Vec<TokenId>,
}

fn prompt_of(seq: u64) -> Vec<TokenId> {
    (0..PROMPT as TokenId)
        .map(|i| 1000 + i + 97 * seq as TokenId)
        .collect()
}

fn blocks_of(seq: u64) -> Vec<u32> {
    (0..16).map(|b| b + 16 * seq as u32).collect()
}

fn sampling(r: &Req) -> SlotSampling {
    SlotSampling {
        params: r.params.clone(),
        choice_index: 0,
        step_index: 0,
        position: r.ids.len() as u64,
        penalties: None,
    }
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    }
}

fn seeded(seed: u64) -> SamplingParams {
    SamplingParams {
        temperature: 0.7,
        top_p: 0.95,
        top_k: 20,
        seed: Some(seed),
        ..SamplingParams::default()
    }
}

fn prefill(exec: &mut RealExecutor<NullForward>, r: &mut Req) {
    exec.admit(&AdmittedSeq {
        request_id: RequestId(r.seq),
        choice_index: 0,
        seq_id: SeqId(r.seq),
        prompt_tokens: prompt_of(r.seq),
        kv_blocks: blocks_of(r.seq),
        shared_prefix_blocks: 0,
        cow_source: None,
        gdn_set: r.seq as u32,
        gdn_slots: MAX_W,
    });
    let slots = vec![BatchSlot {
        request_id: RequestId(r.seq),
        seq_id: SeqId(r.seq),
        slot: 0,
        kv_blocks: blocks_of(r.seq),
        gdn_set: r.seq as u32,
        verify_width: 0,
        input_tokens: prompt_of(r.seq),
        context_len: 0,
        first_step: true,
        chunk: Some(PrefillChunk {
            prompt_tokens: PROMPT as u32,
            tokens_done_before: 0,
            last: true,
            next_prompt_token: None,
        }),
        sampling: sampling(r),
    }];
    let out = exec.execute(&StepBatch {
        phase: Phase::Prefill,
        bucket: Bucket::prefill(PREFILL, 8192, true),
        step_index: 0,
        slots: &slots,
        padded_slots: 0,
    });
    let t = out.seqs[0].tokens.clone();
    assert_eq!(t.len(), 1);
    r.ids.extend_from_slice(&t);
    r.last_commit = t;
}

/// One step: `widths[i]` rows for request `i` (a band), or every request at `rect` (a rectangle).
fn step(
    exec: &mut RealExecutor<NullForward>,
    reqs: &mut [&mut Req],
    widths: &[u32],
    rect: Option<u32>,
    index: u64,
) {
    let slots: Vec<BatchSlot> = reqs
        .iter()
        .zip(widths)
        .map(|(r, w)| BatchSlot {
            request_id: RequestId(r.seq),
            seq_id: SeqId(r.seq),
            slot: 0,
            kv_blocks: blocks_of(r.seq),
            gdn_set: r.seq as u32,
            verify_width: rect.unwrap_or(*w),
            input_tokens: r.last_commit.clone(),
            context_len: (PROMPT + r.ids.len() - 1) as u32,
            first_step: false,
            chunk: None,
            sampling: sampling(r),
        })
        .collect();
    let bucket = match rect {
        Some(w) => {
            let batch = *BATCHES
                .iter()
                .find(|b| **b as usize >= reqs.len())
                .expect("a batch");
            Bucket::decode(batch, w, 8192, true)
        }
        None => {
            let fed: u32 = widths.iter().sum();
            let rows = band_rows_for(DEFAULT_ROW_BANDS, 8, MAX_W)
                .into_iter()
                .find(|r| *r >= fed)
                .expect("a band holds it");
            Bucket::band(8, MAX_W, rows, 8192, true)
        }
    };
    let out = exec.execute(&StepBatch {
        phase: Phase::Decode,
        bucket,
        step_index: index,
        slots: &slots,
        padded_slots: bucket.batch - slots.len() as u32,
    });
    assert_eq!(out.seqs.len(), reqs.len(), "{}", exec.counters().line());
    for (r, s) in reqs.iter_mut().zip(&out.seqs) {
        assert!(!s.tokens.is_empty(), "seq {} committed nothing", r.seq);
        r.ids.extend_from_slice(&s.tokens);
        r.last_commit = s.tokens.clone();
    }
}

fn executor(oracle: &Oracle, forward: NullForward) -> RealExecutor<NullForward> {
    RealExecutor::with_drafter(forward, Box::new(oracle.clone()))
}

fn stub() -> NullForward {
    NullForward::with_widths(&[PREFILL], BATCHES, WIDTHS, 8).with_bands(DEFAULT_ROW_BANDS)
}

/// THE REFERENCE: one request alone at width 1.
fn solo(seq: u64, params: &SamplingParams) -> Vec<TokenId> {
    let oracle = Oracle::default();
    let mut exec = executor(&oracle, stub());
    let mut r = Req {
        seq,
        params: params.clone(),
        ids: Vec::new(),
        last_commit: Vec::new(),
    };
    prefill(&mut exec, &mut r);
    let mut i = 1;
    while r.ids.len() < TOKENS {
        step(&mut exec, &mut [&mut r], &[1], Some(1), i);
        i += 1;
    }
    assert!(exec.counters().clean(), "{}", exec.counters().line());
    r.ids.truncate(TOKENS);
    r.ids
}

/// The per-step width multisets the ragged run cycles through (a first step is one row).
const PATTERNS: &[[u32; 5]] = &[
    [1, 1, 1, 1, 1],
    [5, 3, 8, 2, 5],
    [8, 1, 4, 5, 3],
    [2, 8, 8, 3, 1],
    [4, 5, 1, 8, 6],
];

/// Five requests decoded together in bands at the patterns' widths; returns each one's ids and
/// the executor's counters line.
fn together(
    params: &[SamplingParams],
    forward: NullForward,
) -> (Vec<Vec<TokenId>>, saber_engine::EngineCounters) {
    let oracle = Oracle::default();
    let mut exec = executor(&oracle, forward);
    let mut reqs: Vec<Req> = params
        .iter()
        .enumerate()
        .map(|(i, p)| Req {
            seq: 10 + i as u64,
            params: p.clone(),
            ids: Vec::new(),
            last_commit: Vec::new(),
        })
        .collect();
    for r in &mut reqs {
        prefill(&mut exec, r);
        let reference = solo(r.seq, &r.params);
        oracle.script(SeqId(r.seq), &reference, r.ids.len());
    }
    let mut i = 1u64;
    while reqs.iter().any(|r| r.ids.len() < TOKENS) {
        let pattern = PATTERNS[(i as usize - 1) % PATTERNS.len()];
        let mut live: Vec<&mut Req> = reqs.iter_mut().filter(|r| r.ids.len() < TOKENS).collect();
        let widths: Vec<u32> = pattern[..live.len()].to_vec();
        step(&mut exec, &mut live, &widths, None, i);
        i += 1;
        assert!(i < 400, "not advancing");
    }
    let counters = exec.counters();
    (
        reqs.into_iter()
            .map(|mut r| {
                r.ids.truncate(TOKENS);
                r.ids
            })
            .collect(),
        counters,
    )
}

fn mixed() -> Vec<SamplingParams> {
    vec![greedy(), seeded(7), greedy(), seeded(11), greedy()]
}

/// (a) The engine-identity law in bands: every request's committed ids are its solo width-1 ids,
/// greedy and seeded, whatever width it and its neighbours verified at each step.
#[test]
fn a_ragged_batch_commits_every_requests_solo_ids() {
    let params = mixed();
    let (ids, counters) = together(&params, stub());
    for (i, (got, p)) in ids.iter().zip(&params).enumerate() {
        let want = solo(10 + i as u64, p);
        assert_eq!(got, &want, "request {i} ({p:?})");
    }
    assert!(counters.band_steps > 0, "{}", counters.line());
    assert!(counters.band_live_rows < counters.band_rows);
    assert!(counters.clean(), "{}", counters.line());
    assert_eq!(counters.budget_mismatches + counters.undeclared_shapes, 0);
}

/// (b) The padding is never read: the band's tail poisoned with an id the stub answers
/// differently changes no committed token (and the stub's per-step padding check held throughout,
/// or the step would have been refused).
#[test]
fn a_poisoned_band_padding_row_changes_no_live_row() {
    let params = mixed();
    let (clean, _) = together(&params, stub());
    let (poisoned, counters) = together(&params, stub().with_band_pad_id(123_457));
    assert_eq!(clean, poisoned);
    assert!(counters.clean(), "{}", counters.line());
}

/// NEGATIVE CONTROL: each slot read at the rectangle's stride instead of by `cu` reads its
/// neighbours' rows and the padding — the stream moves off its solo ids.
#[test]
fn control_reading_a_band_at_the_rectangles_stride_moves_the_stream() {
    let params = mixed();
    let (ids, _) = together(&params, stub().with_strided_band_reads());
    let moved = ids
        .iter()
        .zip(&params)
        .enumerate()
        .filter(|(i, (got, p))| **got != solo(10 + *i as u64, p))
        .count();
    assert!(
        moved > 0,
        "the strided control must diverge from the solo ids"
    );
}
