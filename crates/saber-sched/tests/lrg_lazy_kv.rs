//! LR-G: lazy KV growth (`AdmitPolicy::lazy_headroom`, `--fn-kv-lazy`) on the discrete-event
//! replay of the real scheduler (`examples/sched_replay`): a load that never meets a KV refusal
//! is today's schedule step for step; the agent shape gains what a bigger slab gains with zero
//! preemptions; under a stress that forces preemption every stream still arrives whole -- the
//! event index contiguous, exactly its drawn length, one finish -- and no step is ever built past
//! a sequence's blocks.

#[path = "../examples/sched_replay/replay.rs"]
mod replay;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use replay::{flashnext_config, ledger, run, ReplayExec, StepModel, Trace, TraceKind, Variant};
use saber_sched::{AdmitPolicy, Scheduler, StepOutcome};
use saber_serve::engine::{Engine, EngineRequest, FinishReason, StepEvent, TokenId};
use saber_serve::sampling::SamplingParams;

fn agent(seed: u64, out_mean: f64) -> Trace {
    Trace {
        name: "agent".into(),
        sessions: 16,
        kind: TraceKind::Agent {
            think_ms: 10_000.0,
            max_tokens: 32_768,
            out_mean,
        },
        seed,
    }
}

fn conc8(seed: u64) -> Trace {
    Trace {
        name: "conc8".into(),
        sessions: 8,
        kind: TraceKind::Closed {
            prompt: 512,
            out_len: 400,
            max_tokens: 1_024,
            think_ms: 1.7,
        },
        seed,
    }
}

fn lazy(h: u32, margin: u32) -> Variant {
    Variant {
        admit: AdmitPolicy {
            lazy_headroom: h,
            lazy_margin: margin,
            ..AdmitPolicy::default()
        },
        ..Variant::base()
    }
}

const KV: u64 = 272_528;

#[test]
fn lrg_a_load_without_kv_refusals_is_todays_schedule() {
    let m = StepModel::flashnext();
    for seed in 0..2 {
        let t = conc8(seed);
        let base = run(&t, &Variant::base(), m, KV, 10.0, 120.0);
        assert_eq!(base.refused_kv, 0);
        for v in [lazy(4096, 4096), lazy(256, 0)] {
            let r = run(&t, &v, m, KV, 10.0, 120.0);
            assert_eq!((r.schedule_hash, r.steps), (base.schedule_hash, base.steps));
            assert_eq!(r.preemptions, 0);
            assert_eq!(r.outran, 0);
        }
    }
}

#[test]
fn lrg_the_agent_shape_fills_the_batch_without_preemption() {
    let m = StepModel::flashnext();
    for seed in 0..2 {
        let t = agent(seed, 1_200.0);
        let base = run(&t, &Variant::base(), m, KV, 120.0, 900.0);
        let r = run(&t, &lazy(4096, 4096), m, KV, 120.0, 900.0);
        assert!(
            r.tok_s >= base.tok_s * 1.05,
            "{} vs {}",
            r.tok_s,
            base.tok_s
        );
        assert!(r.ttft_p50_s <= base.ttft_p50_s * 0.75, "{r:?}");
        assert!(r.mean_running > base.mean_running + 2.0, "{r:?}");
        assert_eq!(r.preemptions, 0, "{r:?}");
        assert!(r.lazy_grows > 0);
        assert_eq!((r.outran, base.outran), (0, 0));
        // a stress that preempts: still no step past a sequence's blocks
        let s = run(&agent(seed, 6_000.0), &lazy(1024, 0), m, KV, 120.0, 900.0);
        assert!(s.preemptions > 0, "{s:?}");
        assert_eq!(s.outran, 0, "{s:?}");
    }
}

/// A small arena and long outputs: growth fails, the youngest is preempted and recomputed, and
/// every request's stream still arrives whole.
#[test]
fn lrg_preempted_streams_arrive_whole_and_no_step_outruns_its_blocks() {
    let clock = Arc::new(AtomicU64::new(0));
    let base = Instant::now();
    let c2 = Arc::clone(&clock);
    let mut config = flashnext_config(
        24_000,
        AdmitPolicy {
            lazy_headroom: 256,
            lazy_margin: 0,
            ..AdmitPolicy::default()
        },
        0,
    );
    config.implicit_context = None;
    let model_info = saber_sched::mock_model_info("lrg", "/replay", replay::EOS, 262_144);
    let sched = Scheduler::new(
        config,
        ReplayExec::new(Arc::clone(&clock), StepModel::flashnext()),
        model_info,
        ledger(),
    )
    .expect("arenas")
    .with_clock(Arc::new(move || {
        base + Duration::from_micros(c2.load(Ordering::Relaxed))
    }));
    // 8 requests: 1-3k prompts, 1.5-3k outputs, max_tokens 8k: their written KV outgrows 24k
    let mut subs = Vec::new();
    for i in 0..8u32 {
        let prompt = 1_000 + 250 * i as usize;
        let out = 1_500 + 200 * i as usize;
        let mut ids: Vec<TokenId> = vec![100; prompt];
        ids[0] = 100 + i;
        ids[1] = 100;
        ids[2] = 100 + out as u32;
        let sub = sched
            .submit(EngineRequest {
                prompt_token_ids: ids,
                params: SamplingParams {
                    max_tokens: Some(8_192),
                    ..SamplingParams::default()
                },
            })
            .expect("submit");
        subs.push((out, sub));
    }
    let mut steps = 0;
    while sched.has_work() && steps < 200_000 {
        match sched.step() {
            StepOutcome::NoBucket(e) => panic!("{e:?}"),
            StepOutcome::Idle => clock.fetch_add(1_000, Ordering::Relaxed),
            StepOutcome::Ran(_) => 0,
        };
        steps += 1;
    }
    assert!(!sched.has_work(), "drained");
    let arm = sched.admit_arm_counters();
    assert!(arm.lazy_preemptions > 0, "the stress preempts: {arm:?}");
    assert!(arm.lazy_recompute_tokens > 0);
    let counters = sched.counters();
    assert_eq!(counters.executor_protocol_errors, 0);
    assert_eq!(counters.executor_token_count_errors, 0);
    assert_eq!(
        sched.with_executor(|e| e.outran),
        0,
        "no step past its blocks"
    );
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("rt");
    for (out, sub) in subs {
        let events: Vec<StepEvent> = rt.block_on(sub.events.collect());
        for (k, e) in events.iter().enumerate() {
            assert_eq!(e.step_index as usize, k, "the event index is contiguous");
            assert!(e.error.is_none(), "{e:?}");
        }
        let toks: usize = events.iter().map(|e| e.new_tokens.len()).sum();
        assert_eq!(toks, out, "the stream is whole");
        let fin: Vec<_> = events.iter().filter_map(|e| e.finished).collect();
        assert_eq!(fin, vec![FinishReason::Stop]);
    }
    let stats = sched.stats();
    let get = |name: &str| {
        stats
            .census
            .iter()
            .find(|s| s.def.name == name)
            .map(|s| s.value)
    };
    assert_eq!(
        get("saber:kv_lazy_preemptions_total"),
        Some(arm.lazy_preemptions)
    );
    assert_eq!(
        get("saber:kv_lazy_recompute_tokens_total"),
        Some(arm.lazy_recompute_tokens)
    );
    assert_eq!(get("saber:kv_lazy_grows_total"), Some(arm.lazy_grows));
    let (free, _) = sched.arena_free();
    assert_eq!(
        free,
        sched.kv_admission().0.total_blocks,
        "every block back"
    );
}
