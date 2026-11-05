//! **PROMPT-LOOKUP SPECULATION IS OUTPUT-NEUTRAL** (workstream FN69).
//!
//! `saber_engine::LookupDrafter` puts a per-sequence n-gram index (prompt + committed output) in
//! front of the engine's drafter; a confident match REPLACES the drafter's block. The verify step
//! is unchanged, so the committed stream must be the drafter-only stream token for token -- greedy
//! (the argmax theorem) and seeded (F40's position-keyed draws: a draft is accepted iff it IS the
//! position's draw, whatever proposed it). This drives the real executor over [`NullForward`] with
//! a small vocabulary, so the stub's streams repeat and the lookup actually hits, against a
//! scripted drafter that knows the reference stream (wrong at every fifth position):
//!
//!   (a) greedy and seeded, width 8: lookup ON == lookup OFF == the width-1 reference;
//!   (b) the lookup was asked at every drafting point, hit, was USED, and its tokens were verified
//!       (the per-source tally), so the test is not the drafter-only run twice;
//!   (c) the executor's census carries the tally (`lookup` line, `/metrics` twins).

use std::sync::{Arc, Mutex};

use saber_engine::{Drafter, LookupConfig, LookupDrafter, NullForward, RealExecutor};
use saber_plan::{Bucket, Phase};
use saber_sched::{
    AdmittedSeq, BatchSlot, PrefillChunk, SeqId, SlotSampling, StepBatch, StepExecutor,
};
use saber_serve::engine::{RequestId, TokenId};
use saber_serve::sampling::SamplingParams;

const PREFILL: u32 = 2048;
const PROMPT: usize = 64;
const TOKENS: usize = 160;
const VOCAB: u32 = 64;

/// Proposes the reference stream's next tokens, wrong at every position `p % 5 == 3`.
#[derive(Clone, Debug, Default)]
struct Script(Arc<Mutex<(Vec<TokenId>, usize)>>);

impl Drafter for Script {
    fn name(&self) -> &'static str {
        "script"
    }
    fn max_width(&self) -> u32 {
        7
    }
    fn propose(&mut self, _seq: SeqId, _last: TokenId, width: u32) -> Vec<TokenId> {
        let g = self.0.lock().expect("script");
        let (want, at) = (&g.0, g.1);
        (at..at + width as usize)
            .map(|p| {
                if p % 5 == 3 {
                    VOCAB - 1
                } else {
                    want.get(p).copied().unwrap_or(VOCAB - 1)
                }
            })
            .collect()
    }
    fn commit(&mut self, _seq: SeqId, committed: &[TokenId]) {
        self.0.lock().expect("script").1 += committed.len();
    }
}

fn prompt() -> Vec<TokenId> {
    (0..PROMPT as TokenId).map(|i| 1 + (i * 7) % 61).collect()
}

fn sampling(params: &SamplingParams, position: usize, steps: u64) -> SlotSampling {
    SlotSampling {
        params: params.clone(),
        choice_index: 0,
        step_index: steps,
        position: position as u64,
        penalties: None,
    }
}

fn blocks() -> Vec<u32> {
    (0..16).collect()
}

/// One request to `TOKENS` at `width`; `lookup` wraps the scripted drafter.
fn run(
    params: &SamplingParams,
    width: u32,
    reference: &[TokenId],
    lookup: Option<LookupConfig>,
) -> (Vec<TokenId>, saber_engine::EngineCounters) {
    let script = Script(Arc::new(Mutex::new((reference.to_vec(), 1))));
    let inner: Box<dyn Drafter> = Box::new(script);
    let drafter: Box<dyn Drafter> = match lookup {
        Some(cfg) => Box::new(LookupDrafter::new(inner, cfg)),
        None => inner,
    };
    run_with(drafter, params, width, false)
}

/// The same with a given drafter; `keyed` = the pre-F40 step-keyed draw law (the control).
fn run_with(
    drafter: Box<dyn Drafter>,
    params: &SamplingParams,
    width: u32,
    keyed: bool,
) -> (Vec<TokenId>, saber_engine::EngineCounters) {
    let mut exec = RealExecutor::with_drafter(
        NullForward::with_widths(&[PREFILL], &[1], &[1, 8], 2).with_vocab(VOCAB),
        drafter,
    );
    exec.step_keyed_draws(keyed);
    let seq = SeqId(1);
    exec.admit(&AdmittedSeq {
        request_id: RequestId(1),
        choice_index: 0,
        seq_id: seq,
        prompt_tokens: prompt(),
        kv_blocks: blocks(),
        shared_prefix_blocks: 0,
        cow_source: None,
        gdn_set: 1,
        gdn_slots: 2,
    });
    let slots = vec![BatchSlot {
        request_id: RequestId(1),
        seq_id: seq,
        slot: 0,
        kv_blocks: blocks(),
        gdn_set: 1,
        verify_width: 0,
        input_tokens: prompt(),
        context_len: 0,
        first_step: true,
        chunk: Some(PrefillChunk {
            prompt_tokens: PROMPT as u32,
            tokens_done_before: 0,
            last: true,
            next_prompt_token: None,
        }),
        sampling: sampling(params, 0, 0),
    }];
    let out = exec.execute(&StepBatch {
        phase: Phase::Prefill,
        bucket: Bucket::prefill(PREFILL, 8192, true),
        step_index: 0,
        slots: &slots,
        padded_slots: 0,
    });
    let mut ids = out.seqs[0].tokens.clone();
    let mut last = ids.clone();
    let mut steps = 1u64;
    while ids.len() < TOKENS {
        let slots = vec![BatchSlot {
            request_id: RequestId(1),
            seq_id: seq,
            slot: 0,
            kv_blocks: blocks(),
            gdn_set: 1,
            verify_width: width - 1,
            input_tokens: last.clone(),
            context_len: (PROMPT + ids.len() - 1) as u32,
            first_step: false,
            chunk: None,
            sampling: sampling(params, ids.len(), steps),
        }];
        let out = exec.execute(&StepBatch {
            phase: Phase::Decode,
            bucket: Bucket::decode(1, width, 8192, true),
            step_index: steps,
            slots: &slots,
            padded_slots: 0,
        });
        let t = out.seqs[0].tokens.clone();
        assert!(!t.is_empty(), "committed nothing");
        ids.extend_from_slice(&t);
        last = t;
        steps += 1;
        assert!(steps < 1000, "not advancing");
    }
    ids.truncate(TOKENS);
    let c = exec.counters();
    assert!(c.clean(), "{}", c.line());
    (ids, c)
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    }
}

fn seeded() -> SamplingParams {
    SamplingParams {
        temperature: 0.7,
        top_p: 0.95,
        top_k: 20,
        seed: Some(4242),
        ..SamplingParams::default()
    }
}

fn law(params: &SamplingParams, cfg: LookupConfig) {
    let reference = run(params, 1, &[], None).0;
    assert_eq!(reference.len(), TOKENS);
    let (off, c_off) = run(params, 8, &reference, None);
    let (on, c_on) = run(params, 8, &reference, Some(cfg));
    assert_eq!(off, reference, "drafter-only left the width-1 stream");
    assert_eq!(on, reference, "lookup ON left the width-1 stream");
    // (b): the lookup was asked, hit, used, and its tokens were verified
    assert_eq!(c_off.lookup_lookups, 0, "{}", c_off.lookup_line());
    assert!(c_on.lookup_lookups > 0, "{}", c_on.lookup_line());
    assert!(c_on.lookup_hits > 0, "{}", c_on.lookup_line());
    assert!(c_on.lookup_used > 0, "{}", c_on.lookup_line());
    assert!(c_on.lookup_drafted > 0, "{}", c_on.lookup_line());
    assert_eq!(
        c_on.lookup_drafted + c_on.lookup_drafter_drafted,
        c_on.draft_tokens,
        "every verified draft row is tallied to one source: {}",
        c_on.lookup_line()
    );
    assert_eq!(
        c_on.lookup_accepted + c_on.lookup_drafter_accepted,
        c_on.accepted_tokens,
        "{}",
        c_on.lookup_line()
    );
}

/// (a) + (b), greedy, at a bar every repeated pair clears.
#[test]
fn greedy_lookup_on_is_the_drafter_only_stream() {
    law(&greedy(), LookupConfig::parse("1..2", 1).expect("cfg"));
}

/// (a) + (b), seeded: F40's draws make the committed stream a function of the seed alone.
#[test]
fn seeded_lookup_on_is_the_drafter_only_stream() {
    law(&seeded(), LookupConfig::parse("1..2", 1).expect("cfg"));
}

/// (c): the census line and its `/metrics` twins carry the tally.
#[test]
fn the_census_carries_the_lookup_line() {
    let reference = run(&greedy(), 1, &[], None).0;
    let (_, c) = run(
        &greedy(),
        8,
        &reference,
        Some(LookupConfig::parse("1..2", 1).expect("cfg")),
    );
    let text = saber_engine::shutdown_census(
        &saber_sched::SchedCounters::default(),
        &saber_sched::alloc::AdmissionCounters::default(),
        &c,
        None,
    );
    let line = text
        .lines()
        .find(|l| l.starts_with("lookup "))
        .expect("a lookup line");
    assert!(
        line.contains(&format!("lookups {},", c.lookup_lookups)),
        "{line}"
    );
    for def in saber_engine::EngineCounters::CENSUS
        .iter()
        .filter(|d| d.name.contains("lookup"))
    {
        assert!(
            saber_serve::census::find_in_census(&text, def).is_some(),
            "{} not found in the census",
            def.name
        );
    }
}

/// CONTROL for (a) seeded: under the pre-F40 step-keyed law (`step_keyed_draws`) the draw follows
/// the step, so the lookup -- which moves step boundaries -- moves the seeded stream. The seeded
/// claim above rests on F40's law, and this shows the test can see a moved stream.
#[test]
fn control_the_step_keyed_law_moves_with_the_lookup() {
    let params = seeded();
    let cfg = LookupConfig::parse("1..2", 1).expect("cfg");
    let reference = run(&params, 1, &[], None).0;
    let keyed = |lookup| {
        let script = Script(Arc::new(Mutex::new((reference.clone(), 1))));
        let inner: Box<dyn Drafter> = Box::new(script);
        let drafter: Box<dyn Drafter> = match lookup {
            Some(c) => Box::new(LookupDrafter::new(inner, c)),
            None => inner,
        };
        run_with(drafter, &params, 8, true)
    };
    let (off, on) = (keyed(None), keyed(Some(cfg)));
    assert_ne!(
        off, on,
        "the step-keyed control did not move with the lookup: the test cannot see a moved stream"
    );
}
