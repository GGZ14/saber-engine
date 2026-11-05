//! C4: the MIXED STEP — a decode step that also carries the queue head's next prefill chunk.
//!
//! Unmixed, a prompt chunk takes a step of its own and every decoding sequence waits it out (the
//! stall, counted here as `stalled_decoder_steps`). Mixed (`MixedPrefill`), the decode half runs
//! and the chunk rides in the same step. The laws the host can prove:
//!
//! * **the engine-identity law**: every request's tokens are the same mixed and unmixed, over a
//!   run with staggered arrivals (the MIXED-ARRIVAL arm's law), under an executor whose tokens
//!   are a function of each sequence's own history only — and the comparison FAILS under the
//!   control executor that lets the schedule leak into a token (`CLAUDE.md` rule 8);
//! * **the chunk law**: each sequence is fed the same chunks — the same rows, the same
//!   `tokens_done_before`, the same `last` — and ends its prompt in the same state, mixed or not;
//! * **the order law**: in a mixed step the chunk runs FIRST and the decode half never holds a
//!   prefilling sequence;
//! * **the budget law**: `chunk <= max_batched_tokens - decode rows`, and a spent budget makes the
//!   chunk wait (counted) rather than overrun;
//! * **one step**: a mixed step is one scheduler step, and the stall is zero whenever something
//!   was decoding.

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use common::{boot_ledger, decode_buckets, plan_with_prefill, GIB};
use saber_plan::Phase;
use saber_sched::{
    AdmittedSeq, ArenaBytes, KvGeometry, MixedPrefill, Scheduler, SchedulerConfig, SeqId, SeqStep,
    StepBatch, StepExecutor, StepResult,
};
use saber_serve::engine::{Engine, EngineRequest, StepEvent, TokenId};
use saber_serve::sampling::SamplingParams;

const EOS: TokenId = 255;

fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One executor call, as recorded.
#[derive(Clone, Debug)]
struct Call {
    step_index: u64,
    phase: Phase,
    seqs: Vec<SeqId>,
    /// Rows the step planned: the bucket's M.
    m: u32,
    /// Chunk rows fed (prefill only).
    chunk_rows: u32,
}

/// Tokens are a hash chain over each sequence's OWN history: the chunks it was fed (rows and
/// offsets) and what it generated. So a schedule can move a sequence between steps and batches
/// and its tokens cannot change — unless `leak_schedule`, the control, mixes the step index in.
#[derive(Debug, Default)]
struct ChainExec {
    leak_schedule: bool,
    state: BTreeMap<SeqId, u64>,
    /// Per sequence: `(tokens_done_before, rows, last, state after the chunk)`.
    chunks: BTreeMap<SeqId, Vec<(u32, u32, bool, u64)>>,
    calls: Vec<Call>,
}

impl StepExecutor for ChainExec {
    fn execute(&mut self, batch: &StepBatch<'_>) -> StepResult {
        self.calls.push(Call {
            step_index: batch.step_index,
            phase: batch.phase,
            seqs: batch.slots.iter().map(|s| s.seq_id).collect(),
            m: batch.m(),
            chunk_rows: batch
                .slots
                .iter()
                .filter_map(|s| s.chunk)
                .map(|c| c.prompt_tokens)
                .sum(),
        });
        let mut seqs = Vec::new();
        for slot in batch.slots {
            let st = self.state.get_mut(&slot.seq_id).expect("admitted");
            if let Some(c) = slot.chunk {
                for t in &slot.input_tokens {
                    *st = mix(*st ^ u64::from(*t));
                }
                *st = mix(*st ^ u64::from(c.tokens_done_before));
                self.chunks.entry(slot.seq_id).or_default().push((
                    c.tokens_done_before,
                    c.prompt_tokens,
                    c.last,
                    *st,
                ));
                if !c.last {
                    seqs.push(SeqStep {
                        seq_id: slot.seq_id,
                        tokens: Vec::new(),
                        draft_tokens: 0,
                        accepted_tokens: 0,
                        logprobs: None,
                        prompt_logprobs: None,
                    });
                    continue;
                }
            }
            // Two accepted drafts when the slot verifies at least two, always the same number for
            // the same width, so acceptance is a function of the width alone.
            let accepted = slot.verify_width.min(2);
            let mut tokens = Vec::new();
            for _ in 0..=accepted {
                *st = mix(*st);
                if self.leak_schedule {
                    *st ^= batch.step_index;
                }
                tokens.push(1000 + (*st % 4096) as TokenId);
            }
            seqs.push(SeqStep {
                seq_id: slot.seq_id,
                tokens,
                draft_tokens: slot.verify_width,
                accepted_tokens: accepted,
                logprobs: None,
                prompt_logprobs: None,
            });
        }
        StepResult { seqs }
    }

    fn admit(&mut self, seq: &AdmittedSeq) {
        self.state.insert(seq.seq_id, 0x5EED);
    }
}

fn sched(mixed: MixedPrefill, leak_schedule: bool) -> Arc<Scheduler<ChainExec>> {
    let decode: Vec<(u32, u32, u32)> = decode_buckets()
        .into_iter()
        .filter(|(_, _, band)| *band == 8192)
        .collect();
    let plan = plan_with_prefill(&decode, &[(256, 8192), (2048, 8192)], 2);
    let config = SchedulerConfig::from_plan(
        &plan,
        KvGeometry::PROD_FP8,
        ArenaBytes {
            kv_per_rank: 8 * GIB,
            gdn_state_per_rank: 8 * GIB,
        },
        8192,
    )
    .expect("config");
    let config = SchedulerConfig { mixed, ..config };
    let model =
        saber_sched::mock_model_info("Qwen3.8", "/models/mock", EOS, config.max_context_tokens);
    Arc::new(
        Scheduler::new(
            config,
            ChainExec {
                leak_schedule,
                ..ChainExec::default()
            },
            model,
            boot_ledger(),
        )
        .expect("arenas fit"),
    )
}

fn request(prompt_tokens: usize, max_tokens: usize, salt: u32) -> EngineRequest {
    EngineRequest {
        prompt_token_ids: (0..prompt_tokens)
            .map(|i| ((i as u32 * 7 + salt) % 200) + 1)
            .collect(),
        params: SamplingParams {
            max_tokens: Some(max_tokens),
            ..SamplingParams::default()
        },
    }
}

async fn tokens(submission: saber_serve::engine::Submission) -> Vec<TokenId> {
    use futures_util::StreamExt;
    let events: Vec<StepEvent> = submission.events.collect().await;
    events.iter().flat_map(|e| e.new_tokens.clone()).collect()
}

/// The workload: three requests at once, then arrivals while they decode — one of them a prompt
/// longer than the largest chunk, so it is fed in two chunks with decoding going on around it.
fn workload() -> Vec<(usize, usize, usize, u32)> {
    // (steps to run before submitting, prompt tokens, max tokens, salt)
    vec![
        (0, 300, 40, 1),
        (0, 120, 25, 2),
        (0, 700, 30, 3),
        (4, 2500, 20, 4),
        (3, 90, 35, 5),
        (2, 400, 15, 6),
        (0, 1500, 20, 7),
    ]
}

/// Runs the workload; returns each request's tokens, in submission order, and the scheduler.
async fn run(mixed: MixedPrefill, leak: bool) -> (Vec<Vec<TokenId>>, Arc<Scheduler<ChainExec>>) {
    let s = sched(mixed, leak);
    let mut subs = Vec::new();
    for (before, prompt, max, salt) in workload() {
        s.run_steps(before);
        subs.push(s.submit(request(prompt, max, salt)).expect("submit"));
    }
    s.run_steps(2000);
    assert!(!s.has_work(), "the run finished");
    let mut out = Vec::new();
    for sub in subs {
        out.push(tokens(sub).await);
    }
    (out, s)
}

#[tokio::test]
async fn a_mixed_run_commits_the_ids_of_the_unmixed_run_and_the_schedule_leak_control_fails() {
    let (unmixed, u) = run(MixedPrefill::OFF, false).await;
    let (mixed, m) = run(MixedPrefill::on(8192), false).await;
    for (i, (a, b)) in unmixed.iter().zip(&mixed).enumerate() {
        assert!(!a.is_empty(), "request {i} produced tokens");
        assert_eq!(a, b, "request {i}: mixed must commit the unmixed ids");
    }
    // The schedules really differ: the mixed run carried chunks in decode steps, and took fewer.
    let (uc, mc) = (u.counters(), m.counters());
    assert_eq!(uc.mixed_steps, 0);
    assert!(mc.mixed_steps >= 4, "mixed steps {}", mc.mixed_steps);
    assert!(mc.steps < uc.steps, "{} vs {}", mc.steps, uc.steps);

    // THE CONTROL: the same comparison with the schedule leaking into a token must FAIL — a
    // comparator that cannot tell two schedules apart would pass anything.
    let (unmixed_leak, _) = run(MixedPrefill::OFF, true).await;
    let (mixed_leak, _) = run(MixedPrefill::on(8192), true).await;
    assert_ne!(
        unmixed_leak, mixed_leak,
        "the control must see the schedule change"
    );
}

#[tokio::test]
async fn every_sequence_is_fed_the_same_chunks_and_ends_its_prompt_in_the_same_state() {
    let (_, u) = run(MixedPrefill::OFF, false).await;
    let (_, m) = run(MixedPrefill::on(8192), false).await;
    let a = u.with_executor(|e| e.chunks.clone());
    let b = m.with_executor(|e| e.chunks.clone());
    assert_eq!(a.len(), workload().len());
    assert_eq!(a, b, "chunk rows, offsets, last flags and post-chunk state");
    // The 2500-token prompt is two chunks (2048 + 452) either way.
    let long = a.values().find(|c| c.len() == 2).expect("the long prompt");
    assert_eq!((long[0].0, long[0].1, long[0].2), (0, 2048, false));
    assert_eq!((long[1].0, long[1].1, long[1].2), (2048, 452, true));
}

#[tokio::test]
async fn a_mixed_step_runs_its_chunk_first_and_its_decode_half_holds_no_prefilling_sequence() {
    let (_, m) = run(MixedPrefill::on(8192), false).await;
    let calls = m.with_executor(|e| e.calls.clone());
    let mut by_step: BTreeMap<u64, Vec<Call>> = BTreeMap::new();
    for c in calls {
        by_step.entry(c.step_index).or_default().push(c);
    }
    let mut mixed = 0;
    for (step, calls) in &by_step {
        match calls.as_slice() {
            [one] => assert!(!one.seqs.is_empty(), "step {step}"),
            [chunk, decode] => {
                mixed += 1;
                assert_eq!(chunk.phase, Phase::Prefill, "step {step}: the chunk first");
                assert_eq!(
                    decode.phase,
                    Phase::Decode,
                    "step {step}: then the decode half"
                );
                assert_eq!(chunk.seqs.len(), 1, "one chunk");
                assert!(
                    !decode.seqs.contains(&chunk.seqs[0]),
                    "step {step}: a prefilling sequence is not a decode row"
                );
            }
            other => panic!("step {step}: {} calls", other.len()),
        }
    }
    let c = m.counters();
    assert_eq!(mixed, c.mixed_steps);
    // A mixed step is ONE step, and the stall is gone: every chunk that ran alone ran with
    // nothing decoding.
    assert_eq!(c.steps as usize, by_step.len());
    assert_eq!(c.stalled_decoder_steps, 0);
    assert_eq!(c.stalled_decoder_us, 0);
    assert_eq!(c.prefill_alone_steps + c.mixed_steps, c.prefill_steps);
}

#[tokio::test]
async fn the_unmixed_schedule_counts_its_stall() {
    let (_, u) = run(MixedPrefill::OFF, false).await;
    let c = u.counters();
    assert_eq!(c.prefill_alone_steps, c.prefill_steps);
    // Arrivals while three sequences decode: each of those chunks left decoders waiting.
    assert!(
        c.stalled_decoder_steps >= 4,
        "stalled {}",
        c.stalled_decoder_steps
    );
    // And it is TIME as well as steps: every stalled slot-step was billed its step's wall.
    assert!(c.stalled_decoder_us >= c.stalled_decoder_steps.min(1));
}

#[test]
fn the_budget_law() {
    let off = MixedPrefill::OFF;
    assert_eq!(off.room(10), 0, "off: never room");
    let on = MixedPrefill::on(8192);
    assert_eq!(on.room(64), 8128);
    assert_eq!(on.room(8192), 0, "a spent budget");
    assert_eq!(on.room(9000), 0, "never negative");
    assert_eq!(MixedPrefill::PROD_MAX_BATCHED_TOKENS, 8192);
    assert_eq!(MixedPrefill::default(), MixedPrefill::OFF);
}

#[tokio::test]
async fn a_tight_budget_cuts_the_chunk_to_fit_and_a_spent_one_makes_it_wait() {
    // 200 rows of budget: a decode half of `m` rows leaves `200 - m` for the chunk.
    //
    // A budget that CUTS a chunk moves the prompt's chunk boundaries, which is a different
    // arithmetic on a device (H36's chunk-identity class) and a different hash chain here, so its
    // ids are not this law's; its law is the per-step bound below.
    let (ids_unmixed, _) = run(MixedPrefill::OFF, false).await;
    let (_, t) = run(MixedPrefill::on(200), false).await;
    let calls = t.with_executor(|e| e.calls.clone());
    let mut by_step: BTreeMap<u64, Vec<Call>> = BTreeMap::new();
    for c in calls {
        by_step.entry(c.step_index).or_default().push(c);
    }
    let mut cut = 0;
    for calls in by_step.values() {
        if let [chunk, decode] = calls.as_slice() {
            assert!(
                chunk.chunk_rows + decode.m <= 200,
                "{} chunk rows beside a {}-row decode half",
                chunk.chunk_rows,
                decode.m
            );
            cut += 1;
        }
    }
    assert!(cut > 0);
    assert!(t.counters().mixed_steps > 0);

    // A budget no decode half fits under: every owed chunk waits for the decoders to finish, is
    // counted, and still goes in alone — the run finishes with the same ids.
    let (ids_spent, s) = run(MixedPrefill::on(1), false).await;
    assert_eq!(ids_unmixed, ids_spent);
    let c = s.counters();
    assert_eq!(c.mixed_steps, 0);
    assert!(c.mixed_budget_spent > 0);
}

#[test]
fn the_fused_room_is_prods_law_and_the_rung_cut_is_the_control() {
    // C11: every fused form takes prod's room (the chunk is never cut to fit the rung; a chunk
    // with no in-class rung beside the decode rows runs the serial form). The PRE-C11 control is
    // C9's room, capped by `largest_rung − decode rows`.
    let serial = MixedPrefill::on(8192);
    let fused = MixedPrefill::fused(8192);
    let control = MixedPrefill::fused_rung_cut_control(8192);
    assert!(fused.enabled && fused.fused && !serial.fused);
    assert!(fused.uses_pack_law() && MixedPrefill::packed(8192).uses_pack_law());
    assert!(!serial.uses_pack_law() && !control.uses_pack_law());
    assert!(!MixedPrefill::OFF.uses_pack_law());
    assert_eq!(serial.room_for(40, 2048), 8152, "serial: uncapped");
    assert_eq!(
        fused.room_for(40, 2048),
        8152,
        "fused: prod's law, never the rung"
    );
    assert_eq!(
        control.room_for(40, 2048),
        2008,
        "the control: the rung binds"
    );
    assert_eq!(
        MixedPrefill::fused_rung_cut_control(300).room_for(40, 2048),
        260,
        "the budget binds"
    );
    assert_eq!(control.room_for(2048, 2048), 0, "spent");
    assert_eq!(MixedPrefill::OFF.room_for(40, 2048), 0);
}

fn steps_of(calls: Vec<Call>) -> BTreeMap<u64, Vec<Call>> {
    let mut by_step: BTreeMap<u64, Vec<Call>> = BTreeMap::new();
    for c in calls {
        by_step.entry(c.step_index).or_default().push(c);
    }
    by_step
}

#[tokio::test]
async fn a_fused_run_fits_the_widest_rung_or_carries_the_whole_unpacked_chunk() {
    // The mock executor runs the serial form either way (the trait's default), so the fused
    // flag moves ONLY the scheduler's chunk. C11: every mixed step's chunk + decode rows fit the
    // largest rung (2048 here) — the fused form — OR the chunk is the prompt's whole unpacked
    // 2048-row chunk, which the engine runs in the serial form. Nothing in between: a chunk is
    // never cut to fit.
    let (_, f) = run(MixedPrefill::fused(8192), false).await;
    let by_step = steps_of(f.with_executor(|e| e.calls.clone()));
    let (mut mixed, mut whole) = (0, 0);
    for calls in by_step.values() {
        if let [chunk, decode] = calls.as_slice() {
            if chunk.chunk_rows + decode.m > 2048 {
                assert_eq!(
                    chunk.chunk_rows, 2048,
                    "a chunk that does not fit the rung beside {} decode rows must be the whole \
                     unpacked 2048-row chunk",
                    decode.m
                );
                whole += 1;
            }
            mixed += 1;
        }
    }
    assert!(mixed > 0);
    assert!(
        whole > 0,
        "the 2500-token prompt's 2048-row chunk rode beside decoders whole"
    );
    assert_eq!(f.counters().mixed_steps, mixed);
    // The control: C9's rung cut fits every step in the rung by cutting the chunk short.
    let (_, c) = run(MixedPrefill::fused_rung_cut_control(8192), false).await;
    let by_step = steps_of(c.with_executor(|e| e.calls.clone()));
    let mut cut = 0;
    for calls in by_step.values() {
        if let [chunk, decode] = calls.as_slice() {
            assert!(chunk.chunk_rows + decode.m <= 2048);
            if chunk.chunk_rows > 1536 && chunk.chunk_rows < 2048 {
                cut += 1;
            }
        }
    }
    assert!(
        cut > 0,
        "the control must cut a chunk short, or this test proves nothing"
    );
}

// ---- C10: the packing law ------------------------------------------------------------------

/// Runs `wl` (as [`workload`]'s tuples) under `mixed`.
async fn run_wl(
    mixed: MixedPrefill,
    wl: &[(usize, usize, usize, u32)],
) -> (Vec<Vec<TokenId>>, Arc<Scheduler<ChainExec>>) {
    let s = sched(mixed, false);
    let mut subs = Vec::new();
    for (before, prompt, max, salt) in wl.iter().copied() {
        s.run_steps(before);
        subs.push(s.submit(request(prompt, max, salt)).expect("submit"));
    }
    s.run_steps(4000);
    assert!(!s.has_work(), "the run finished");
    let mut out = Vec::new();
    for sub in subs {
        out.push(tokens(sub).await);
    }
    (out, s)
}

type Workload = Vec<(usize, usize, usize, u32)>;

/// The three arrival patterns: eight at once (BetterBench's conc-8 burst), staggered, and one
/// long prompt with seven short ones queued behind it — each behind decoders already running,
/// so the chunks ride in mixed steps.
fn patterns() -> Vec<(&'static str, Workload)> {
    let lead = [(0, 30, 60, 90), (0, 40, 60, 91), (6, 20, 1, 92)];
    let burst: Workload = lead
        .iter()
        .copied()
        .chain((0..8).map(|i| (0, 20 + 9 * i, 12, 100 + i as u32)))
        .collect();
    let stagger: Workload = lead
        .iter()
        .copied()
        .chain((0..8).map(|i| (1 + i % 3, 25 + 11 * i, 10, 200 + i as u32)))
        .collect();
    let long: Workload = lead
        .iter()
        .copied()
        .chain(std::iter::once((0, 5000, 8, 300)))
        .chain((0..7).map(|i| (0, 18 + 5 * i, 10, 301 + i as u32)))
        .collect();
    vec![
        ("eight at once", burst),
        ("staggered", stagger),
        ("one long + seven short", long),
    ]
}

#[tokio::test]
async fn every_fused_form_commits_the_unmixed_ids_and_every_prompts_chunk_sequence() {
    // THE CHUNK-SEQUENCE LAW FOR EVERY MIXED FORM (C11): `fused` (one chunk) and `packed` both
    // feed every prompt exactly the unpacked cut, over every arrival pattern.
    for (name, wl) in patterns() {
        let (unmixed, u) = run_wl(MixedPrefill::OFF, &wl).await;
        let a = u.with_executor(|e| e.chunks.clone());
        for (form, mixed) in [
            ("fused", MixedPrefill::fused(8192)),
            ("packed", MixedPrefill::packed(8192)),
        ] {
            let (ids, p) = run_wl(mixed, &wl).await;
            assert_eq!(unmixed, ids, "{name} / {form}: must commit the unmixed ids");
            let b = p.with_executor(|e| e.chunks.clone());
            for (seq, fed) in &b {
                let fed: Vec<(u32, u32)> = fed.iter().map(|c| (c.0, c.1)).collect();
                let cut: Vec<u32> = a[seq].iter().map(|c| c.1).collect();
                saber_sched::mixed_law::chunk_sequence_holds(&cut, &fed)
                    .unwrap_or_else(|e| panic!("{name} / {form}: {seq}: {e}"));
            }
            assert_eq!(
                a, b,
                "{name} / {form}: chunk rows, offsets, last flags and post-chunk state"
            );
        }
    }
}

#[tokio::test]
async fn packed_steps_carry_several_chunks_in_class_and_the_one_chunk_law_is_the_control() {
    // Rungs here are 256 and 2048: two chunks share a step only when they and the decode rows fit
    // 256 rows (the 2048 rung is the other producer class).
    for (name, wl) in patterns() {
        let (_, p) = run_wl(MixedPrefill::packed(8192), &wl).await;
        let by_step = steps_of(p.with_executor(|e| e.calls.clone()));
        let mut packed = 0;
        for calls in by_step.values() {
            let (decode, chunks) = match calls.split_last() {
                Some((d, c)) if d.phase == Phase::Decode && !c.is_empty() => (d, c),
                _ => continue,
            };
            assert!(
                chunks.iter().all(|c| c.phase == Phase::Prefill),
                "{name}: chunks first"
            );
            if chunks.len() > 1 {
                packed += 1;
                let rows: u32 = chunks.iter().map(|c| c.chunk_rows).sum();
                assert!(
                    rows + decode.m <= 256,
                    "{name}: {rows} chunk rows beside {} decode rows leave the class",
                    decode.m
                );
                // Distinct prompts, in admission order (sequence ids ascend).
                let ids: Vec<SeqId> = chunks.iter().map(|c| c.seqs[0]).collect();
                assert!(ids.windows(2).all(|w| w[0] < w[1]), "{name}: arrival order");
            }
        }
        if name != "staggered" {
            assert!(packed > 0, "{name}: some step packed");
        }
        // THE CONTROL: C9's one-chunk law never carries two.
        let (_, f) = run_wl(MixedPrefill::fused(8192), &wl).await;
        let by_step = steps_of(f.with_executor(|e| e.calls.clone()));
        assert!(
            by_step.values().all(|c| c.len() <= 2),
            "{name}: the one-chunk law carried two chunks"
        );
    }
}

#[tokio::test]
async fn no_fused_form_moves_a_long_prompts_boundary_and_the_pre_c11_cut_does() {
    // C9's fused room (2048 − decode rows) cut the 5000-token prompt's first chunk short: a G4
    // boundary move (the C9 defect C10 found). C11: `fused` takes its chunk from the packing law
    // too, so both fused forms keep 2048 (the chunk rides whole, in the serial form); the
    // pre-fix room is the control.
    let wl = patterns().remove(2).1;
    let (_, u) = run_wl(MixedPrefill::OFF, &wl).await;
    let (_, f) = run_wl(MixedPrefill::fused(8192), &wl).await;
    let (_, p) = run_wl(MixedPrefill::packed(8192), &wl).await;
    let (_, c) = run_wl(MixedPrefill::fused_rung_cut_control(8192), &wl).await;
    let find = |s: &Arc<Scheduler<ChainExec>>| {
        s.with_executor(|e| {
            e.chunks
                .values()
                .find(|c| c.iter().map(|x| x.1).sum::<u32>() == 5000)
                .map(|c| c.iter().map(|x| (x.0, x.1)).collect::<Vec<_>>())
                .expect("the long prompt")
        })
    };
    let cut: Vec<u32> = find(&u).iter().map(|c| c.1).collect();
    assert_eq!(cut, vec![2048, 2048, 904]);
    assert!(saber_sched::mixed_law::chunk_sequence_holds(&cut, &find(&f)).is_ok());
    assert!(saber_sched::mixed_law::chunk_sequence_holds(&cut, &find(&p)).is_ok());
    let e = saber_sched::mixed_law::chunk_sequence_holds(&cut, &find(&c))
        .expect_err("the control: C9's rung cut moves the boundary");
    assert!(e.starts_with("chunk 0: fed "), "{e}");
}
