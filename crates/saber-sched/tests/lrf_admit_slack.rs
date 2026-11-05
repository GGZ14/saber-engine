//! LR-F: the admission arms (`AdmitPolicy`) on the discrete-event replay of the real scheduler
//! (`examples/sched_replay`): off == today's schedule step for step; the arms fire only where
//! their preconditions hold; a KV-blocked head is passed at most `SKIP_LIMIT` times.

#[path = "../examples/sched_replay/replay.rs"]
mod replay;

use replay::{run, StepModel, Trace, TraceKind, Variant};
use saber_sched::AdmitPolicy;

fn agent(seed: u64) -> Trace {
    Trace {
        name: "agent".into(),
        sessions: 16,
        kind: TraceKind::Agent {
            think_ms: 10_000.0,
            max_tokens: 32_768,
            out_mean: 1_200.0,
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

fn with(admit: AdmitPolicy) -> Variant {
    Variant {
        admit,
        ..Variant::base()
    }
}

const KV: u64 = 272_528;

#[test]
fn lrf_the_arms_off_and_unfired_are_todays_schedule() {
    let m = StepModel::flashnext();
    for seed in 0..2 {
        // agent: the queue is never empty at a finish, so the slack hold can never fire
        let t = agent(seed);
        let base = run(&t, &Variant::base(), m, KV, 60.0, 400.0);
        assert!(base.refused_kv > 0, "the agent shape is KV-bound");
        for admit in [
            AdmitPolicy {
                slack_ms: 40,
                ..AdmitPolicy::default()
            },
            AdmitPolicy {
                slack_ms: 40,
                slack_force: true,
                ..AdmitPolicy::default()
            },
        ] {
            let r = run(&t, &with(admit), m, KV, 60.0, 400.0);
            assert_eq!(r.holds, 0);
            assert_eq!((r.schedule_hash, r.steps), (base.schedule_hash, base.steps));
        }
        // conc-8: nothing is ever KV-refused, so the skip arm and a smaller reserve cannot move it
        let t = conc8(seed);
        let base = run(&t, &Variant::base(), m, KV, 10.0, 120.0);
        assert_eq!(base.refused_kv, 0);
        let skip = run(
            &t,
            &with(AdmitPolicy {
                skip_kv_blocked: 8,
                ..AdmitPolicy::default()
            }),
            m,
            KV,
            10.0,
            120.0,
        );
        assert_eq!(
            (skip.schedule_hash, skip.hol_skips),
            (base.schedule_hash, 0)
        );
    }
}

#[test]
fn lrf_the_slack_hold_fires_on_a_closed_loop_and_is_capped() {
    let m = StepModel::flashnext();
    let t = conc8(1);
    let base = run(&t, &Variant::base(), m, KV, 10.0, 120.0);
    let r = run(
        &t,
        &with(AdmitPolicy {
            slack_ms: 40,
            ..AdmitPolicy::default()
        }),
        m,
        KV,
        10.0,
        120.0,
    );
    // the 1.7 ms client turn-around is learned and caught; every hold is under its cap
    assert!(r.holds > 100 && r.caught * 10 >= r.holds * 9, "{r:?}");
    assert!(r.held_ms <= r.holds as f64 * 40.0);
    assert_ne!(r.schedule_hash, base.schedule_hash);
    // FN157's ceiling: no throughput to win
    assert!(
        (r.tok_s / base.tok_s - 1.0).abs() < 0.005,
        "{} vs {}",
        r.tok_s,
        base.tok_s
    );
}

#[test]
fn lrf_a_blocked_head_is_passed_at_most_skip_limit_times_and_then_runs() {
    let m = StepModel::flashnext();
    let t = agent(3);
    let r = run(
        &t,
        &with(AdmitPolicy {
            skip_kv_blocked: 8,
            ..AdmitPolicy::default()
        }),
        m,
        KV,
        60.0,
        600.0,
    );
    assert!(r.hol_skips > 0, "{r:?}");
    // every request that arrived early enough got its first token (no starvation)
    assert!(r.requests > 50, "{r:?}");
    assert!(r.ttft_p99_s < 300.0, "{r:?}");
}
