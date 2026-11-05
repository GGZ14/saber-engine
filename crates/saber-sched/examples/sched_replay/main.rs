//! LR-F: the scheduler replay (see `replay.rs`).
//!
//! `cargo run --release -p saber-sched --example sched_replay -- [--trace agent|conc8]
//!  [--seeds N] [--horizon-s S] [--warm-s S] [--sessions N] [--think-ms MS] [--max-tokens N]
//!  [--out-mean N] [--kv-tokens N] [--variants base,yield1,slack40,slack40f,skip8,res4096,...]`
//!
//! Variant names: `base`; `yieldN` (`--fn-prefill-yield N`); `slackN` (admit-slack hold cap N ms,
//! gain-tested); `slackNf` (the same, forced = the unconditional form); `skipN` (pass a KV-blocked head,
//! scanning N deep); `resX` (reserve `max(X, output)`, the grow-on-demand upper bound); `+` joins
//! several (`skip8+res4096`). LR-G: `lazyH` (`--fn-kv-lazy H`: reserve prompt + min(max_tokens,
//! H), grow block by block, preempt the youngest for recompute), `marginM` (a lazy admission
//! leaves M tokens free). Prints one markdown row a variant, means over the seeds, deltas vs
//! the first variant.

mod replay;

use replay::{run, Report, StepModel, Trace, TraceKind, Variant};
use saber_sched::AdmitPolicy;

fn parse_variant(name: &str) -> Variant {
    let mut v = Variant {
        label: name.to_owned(),
        admit: AdmitPolicy::default(),
        yield_steps: 0,
        reserve_cap: 0,
    };
    for part in name.split('+') {
        let num = |p: &str| -> u32 {
            p.trim_end_matches('f')
                .parse()
                .unwrap_or_else(|_| panic!("variant {name}"))
        };
        if part == "base" {
        } else if let Some(n) = part.strip_prefix("yield") {
            v.yield_steps = num(n);
        } else if let Some(n) = part.strip_prefix("slack") {
            v.admit.slack_ms = num(n);
            v.admit.slack_force = n.ends_with('f');
        } else if let Some(n) = part.strip_prefix("skip") {
            v.admit.skip_kv_blocked = num(n);
        } else if let Some(n) = part.strip_prefix("lazy") {
            v.admit.lazy_headroom = num(n);
        } else if let Some(n) = part.strip_prefix("margin") {
            v.admit.lazy_margin = num(n);
        } else if let Some(n) = part.strip_prefix("res") {
            v.reserve_cap = num(n) as usize;
        } else {
            panic!("unknown variant {part}");
        }
    }
    v
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str, d: &str| -> String {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1).cloned())
            .unwrap_or_else(|| d.to_owned())
    };
    let trace_kind = get("--trace", "agent");
    let seeds: u64 = get("--seeds", "3").parse().expect("seeds");
    let horizon: f64 = get("--horizon-s", "1800").parse().expect("horizon");
    let warm: f64 = get("--warm-s", "120").parse().expect("warm");
    let kv_tokens: u64 = get("--kv-tokens", "272528").parse().expect("kv");
    let variants: Vec<Variant> = get("--variants", "base,yield1,slack40,slack40f")
        .split(',')
        .map(parse_variant)
        .collect();
    let (sessions, kind) = match trace_kind.as_str() {
        "agent" => (
            get("--sessions", "16").parse().expect("sessions"),
            TraceKind::Agent {
                think_ms: get("--think-ms", "2000").parse().expect("think"),
                max_tokens: get("--max-tokens", "32768").parse().expect("max"),
                out_mean: get("--out-mean", "1200").parse().expect("out"),
            },
        ),
        "conc8" => (
            get("--sessions", "8").parse().expect("sessions"),
            TraceKind::Closed {
                prompt: get("--prompt", "512").parse().expect("prompt"),
                out_len: get("--out-mean", "400").parse().expect("out"),
                max_tokens: get("--max-tokens", "1024").parse().expect("max"),
                think_ms: get("--think-ms", "1.7").parse().expect("think"),
            },
        ),
        t => panic!("--trace agent|conc8, not {t}"),
    };
    let mut model = StepModel::flashnext();
    model.computed = get("--computed", "0.84").parse().expect("computed");
    println!(
        "trace {trace_kind} sessions {sessions} {kind:?} seeds {seeds} window {warm}..{horizon} s kv_tokens {kv_tokens} computed {}", model.computed
    );
    println!(
        "| variant | tok/s | d % | mean running | mean decode rows | waiting | stall ms/s | gap p50 / p99 / p99.9 / max ms | d p99 % | TTFT p50 / p99 s | reqs (prompt) | holds (caught) held ms / declined | hol skips | refused_kv steps | grows | preempt | recompute tok | margin ref | reserved / written k |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    let mut base: Option<(f64, f64)> = None;
    for v in &variants {
        let rs: Vec<Report> = (0..seeds)
            .map(|s| {
                let trace = Trace {
                    name: trace_kind.clone(),
                    sessions,
                    kind,
                    seed: 0x5EED_0000 + s,
                };
                run(&trace, v, model, kv_tokens, warm, horizon)
            })
            .collect();
        let m = |f: &dyn Fn(&Report) -> f64| mean(&rs.iter().map(f).collect::<Vec<_>>());
        let tok = m(&|r| r.tok_s);
        let p99 = m(&|r| r.gap_p99_ms);
        let (bt, bp) = *base.get_or_insert((tok, p99));
        println!(
            "| {} | {:.1} | {:+.2} | {:.2} | {:.2} | {:.2} | {:.0} | {:.1} / {:.1} / {:.0} / {:.0} | {:+.1} | {:.2} / {:.2} | {:.0} ({:.0}k) | {:.0} ({:.0}) {:.0} / {:.0} | {:.0} | {:.0} | {:.0} | {:.1} | {:.0} | {:.0} | {:.0} / {:.0} |",
            v.label,
            tok,
            100.0 * (tok / bt - 1.0),
            m(&|r| r.mean_running),
            m(&|r| r.mean_decode_rows),
            m(&|r| r.waiting_mean),
            m(&|r| r.stall_ms_per_s),
            m(&|r| r.gap_p50_ms),
            p99,
            m(&|r| r.gap_p999_ms),
            m(&|r| r.gap_max_ms),
            100.0 * (p99 / bp - 1.0),
            m(&|r| r.ttft_p50_s),
            m(&|r| r.ttft_p99_s),
            m(&|r| r.requests as f64),
            m(&|r| r.prompt_mean / 1e3),
            m(&|r| r.holds as f64),
            m(&|r| r.caught as f64),
            m(&|r| r.held_ms),
            m(&|r| r.declined as f64),
            m(&|r| r.hol_skips as f64),
            m(&|r| r.refused_kv as f64),
            m(&|r| r.lazy_grows as f64),
            m(&|r| r.preemptions as f64),
            m(&|r| r.recompute_tokens as f64),
            m(&|r| r.margin_refusals as f64),
            m(&|r| r.reserved_mean / 1e3),
            m(&|r| r.written_mean / 1e3),
        );
        for r in &rs {
            eprintln!(
                "  {} seed: tok/s {:.1} steps {} hash {:016x}",
                r.label, r.tok_s, r.steps, r.schedule_hash
            );
        }
    }
}
