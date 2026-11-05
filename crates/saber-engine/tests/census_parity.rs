//! `/metrics` equals the shutdown census, counter for counter (ws-v8).
//!
//! The real scheduler over a [`RealExecutor`] over the host stub, served by the real `saber-serve`
//! router on a real socket, with FAULTS INJECTED so the counters that must read 0 on a serving
//! boot do not:
//!
//! * one dispatch too many on one step (`budget mismatches`, and `saber:engine_clean` goes to 0);
//! * one decode the forward refuses (`forward errors`);
//! * every fork refused (`forks … refused`, the scheduler's `fork failures`);
//! * a drafter that reports an uncollected proposal and a refused one (`drafted … UNCOLLECTED`);
//! * a sampled row (T = 0.8) beside the greedy ones, and an abort mid-stream.
//!
//! Then the step loop is stopped, `/metrics` is scraped over HTTP, the census block
//! `saber serve --engine real` prints at shutdown is rendered from the same scheduler
//! (`saber_engine::shutdown_census`, the CLI's own call), and every `saber:` value on `/metrics`
//! is found in that text by its def's pattern. The negative control renders the census with one
//! counter moved by one and must fail the same comparison.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use saber_engine::{
    engine_defs, shutdown_census, DecodeSlot, DraftCounts, ForwardError, ForwardFacts,
    ForwardRunner, NullForward, RealExecutor, RowRequest, StepOutcome, StepRows,
};
use saber_mem::ledger::{Ledger, Pool};
use saber_plan::Bucket;
use saber_sched::{
    build_state_with, spawn_step_loop, AdmissionCounters, ArenaBytes, KvGeometry, SchedCounters,
    Scheduler, SchedulerConfig, SeqId,
};
use saber_serve::census::{find_in_census, CensusDef};
use saber_serve::detok::ByteCodec;
use saber_serve::engine::TokenId;
use saber_serve::state::MinimalChatRenderer;
use serde_json::json;

const GIB: u64 = 1 << 30;
const PREFILL: u32 = 2048;
const BATCHES: &[u32] = &[1, 2, 4];

/// The host stub with three faults bolted on. Everything not overridden is the stub's.
#[derive(Debug)]
struct Faulty {
    inner: NullForward,
    /// The decode call (1-based) the forward refuses.
    fail_decode: u64,
    decodes: u64,
    /// What the "drafter" reports: an uncollected stash and a refused proposal.
    draft: DraftCounts,
}

impl ForwardRunner for Faulty {
    fn facts(&self) -> &ForwardFacts {
        self.inner.facts()
    }
    fn declared_launches(&self, rows: StepRows) -> Option<u32> {
        self.inner.declared_launches(rows)
    }
    fn admit(&mut self, seq: SeqId, prompt_len: u32, kv: &[u32]) -> Result<(), ForwardError> {
        self.inner.admit(seq, prompt_len, kv)
    }
    fn release(&mut self, seq: SeqId) {
        self.inner.release(seq);
    }
    fn prefill(
        &mut self,
        seq: SeqId,
        chunk: &[TokenId],
        done: u32,
        last: bool,
        sampling: &RowRequest,
    ) -> Result<StepOutcome, ForwardError> {
        self.inner.prefill(seq, chunk, done, last, sampling)
    }
    fn decode(
        &mut self,
        slots: &[DecodeSlot],
        batch: u32,
        width: u32,
    ) -> Result<StepOutcome, ForwardError> {
        self.decodes += 1;
        if self.decodes == self.fail_decode {
            return Err(ForwardError::Device("injected by census_parity".to_owned()));
        }
        self.inner.decode(slots, batch, width)
    }
    /// A group can be asked for, and every branch is refused: the fork-refusal path.
    fn can_fork(&self) -> bool {
        true
    }
    fn draft_counts(&self) -> Option<DraftCounts> {
        Some(self.draft)
    }
}

fn scheduler(forward: Faulty) -> Scheduler<RealExecutor<Faulty>> {
    let f = forward.facts();
    let kv = KvGeometry {
        block_tokens: 16,
        bytes_per_token_per_rank: f.kv_block_bytes / 16,
    };
    let mut buckets: Vec<Bucket> = BATCHES
        .iter()
        .map(|b| Bucket::decode(*b, 1, 8192, true))
        .collect();
    buckets.push(Bucket::prefill(PREFILL, 8192, true));
    let config = SchedulerConfig::for_mock(
        buckets,
        1,
        f.gdn_bytes_per_slot,
        kv,
        ArenaBytes {
            kv_per_rank: u64::from(f.kv_pool_blocks) * kv.block_bytes(),
            gdn_state_per_rank: 64 * f.gdn_bytes_per_slot,
        },
        65_536,
        "test:census-parity".to_owned(),
    )
    .expect("config");
    let mut ledger = Ledger::new();
    ledger.declare(Pool::Vram(0), 32 * GIB).expect("rank 0");
    ledger.declare(Pool::HostPinned, 4 * GIB).expect("pinned");
    ledger
        .declare(Pool::HostPageable, 48 * GIB)
        .expect("pageable");
    ledger.declare(Pool::Nvme, 16 * GIB).expect("nvme");
    Scheduler::new(
        config,
        RealExecutor::new(forward),
        saber_serve::engine::ModelInfo {
            served_model_name: "Qwen3.8".to_owned(),
            aliases: Vec::new(),
            max_model_len: 65_536,
            root: "/packs/census-parity".to_owned(),
            eos_token_ids: vec![151_645],
        },
        ledger,
    )
    .expect("arenas fit")
}

/// `saber:` name -> value, from a scrape.
fn saber_values(text: &str) -> BTreeMap<String, u64> {
    text.lines()
        .filter(|l| l.starts_with("saber:"))
        .filter_map(|l| {
            let (name_labels, value) = l.rsplit_once(' ')?;
            let name = name_labels.split('{').next()?;
            let value: f64 = value.parse().ok()?;
            Some((name.to_owned(), value as u64))
        })
        .collect()
}

/// Every census def the server exports.
fn all_defs() -> Vec<CensusDef> {
    let mut d = engine_defs();
    d.extend_from_slice(SchedCounters::CENSUS);
    d.extend_from_slice(AdmissionCounters::CENSUS);
    d
}

/// Compares a scrape against a printed census; returns the mismatches.
fn mismatches(scrape: &BTreeMap<String, u64>, census: &str) -> Vec<String> {
    let mut bad = Vec::new();
    for def in all_defs() {
        let Some(live) = scrape.get(def.name) else {
            bad.push(format!("{} missing from /metrics", def.name));
            continue;
        };
        match find_in_census(census, &def) {
            Some(printed) if printed == *live => {}
            other => bad.push(format!(
                "{}: /metrics {live}, census {other:?} ({})",
                def.name, def.census
            )),
        }
    }
    bad
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn metrics_equal_the_shutdown_census_counter_for_counter() {
    let forward = Faulty {
        inner: NullForward::new(&[PREFILL], BATCHES, 4).with_extra_launch_on(3),
        fail_decode: 6,
        decodes: 0,
        draft: DraftCounts {
            proposals: 11,
            refused: 1,
            tokens: 70,
            uncollected: 2,
            ..DraftCounts::default()
        },
    };
    let sched = Arc::new(scheduler(forward));
    let steps = spawn_step_loop(Arc::clone(&sched));
    let state = build_state_with(
        Arc::clone(&sched),
        Arc::new(ByteCodec),
        Arc::new(MinimalChatRenderer),
        20,
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, saber_serve::router(state))
            .await
            .unwrap();
    });
    let http = reqwest::Client::new();
    let scrape = || async {
        http.get(format!("{base}/metrics"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
    };

    // The FIRST scrape arms the mirror and must already carry the census (published by the idle
    // loop or by the scrape itself): every family present, the clean gauge at 1.
    let first = saber_values(&scrape().await);
    for def in all_defs() {
        assert!(
            first.contains_key(def.name),
            "{} not on the first scrape",
            def.name
        );
    }
    assert_eq!(first["saber:engine_clean"], 1, "clean before any step");

    // The traffic: greedy, sampled, a group of two (refused fork), one aborted mid-stream.
    let mut jobs = Vec::new();
    for (i, body) in [
        json!({"model": "Qwen3.8", "prompt": "alpha beta", "max_tokens": 12, "temperature": 0}),
        json!({"model": "Qwen3.8", "prompt": "gamma", "max_tokens": 12, "temperature": 0.8, "seed": 7}),
        json!({"model": "Qwen3.8", "prompt": "delta epsilon", "max_tokens": 12, "temperature": 0}),
        json!({"model": "Qwen3.8", "prompt": "zeta", "max_tokens": 6, "temperature": 0.7, "seed": 3, "n": 2}),
        json!({"model": "Qwen3.8", "prompt": "eta theta", "max_tokens": 12, "temperature": 0}),
    ]
    .into_iter()
    .enumerate()
    {
        let http = http.clone();
        let url = format!("{base}/v1/completions");
        jobs.push(tokio::spawn(async move {
            let _ = i;
            let r = http.post(url).json(&body).send().await.unwrap();
            let t = r.text().await.unwrap_or_default();
            println!("request {i}: {}", &t[..t.len().min(300)]);
        }));
    }
    for j in jobs {
        j.await.unwrap();
    }
    // A second wave after the refused decode: these run to `max_tokens`, and one stream is
    // dropped after its first chunk (the abort path).
    for body in [
        json!({"model": "Qwen3.8", "prompt": "iota", "max_tokens": 5, "temperature": 0}),
        json!({"model": "Qwen3.8", "prompt": "kappa lambda", "max_tokens": 7, "temperature": 0}),
    ] {
        let r = http
            .post(format!("{base}/v1/completions"))
            .json(&body)
            .send()
            .await
            .unwrap();
        let t = r.text().await.unwrap();
        assert!(t.contains("\"finish_reason\":\"length\""), "{t}");
    }
    {
        let mut r = http
            .post(format!("{base}/v1/chat/completions"))
            .json(
                &json!({"model": "Qwen3.8", "messages": [{"role": "user", "content": "mu"}],
                          "max_tokens": 8000, "temperature": 0, "stream": true}),
            )
            .send()
            .await
            .unwrap();
        let first = r.chunk().await.unwrap();
        assert!(first.is_some());
        drop(r);
    }
    // Idle: nothing running, nothing waiting.
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let text = scrape().await;
        if text.contains("saber:num_requests_running{engine=\"0\",model_name=\"Qwen3.8\"} 0.0")
            && text.contains("saber:num_requests_waiting{engine=\"0\",model_name=\"Qwen3.8\"} 0.0")
        {
            break;
        }
        assert!(Instant::now() < deadline, "the engine never went idle");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Stop stepping (the idle poll would move `idle` between the two readings), then read both.
    let ran = steps.stop();
    assert!(ran > 0);
    let live = saber_values(&scrape().await);
    let census = {
        let sc = sched.counters();
        let ac = sched.admission_counters();
        let (ec, dc) = sched.with_executor(|e| (e.counters(), e.forward_mut().draft_counts()));
        shutdown_census(&sc, &ac, &ec, dc.as_ref())
    };
    println!("{census}");

    // the KV pool's gauges and lazy-growth counters are serving-plane families too (LR-F/G)
    let serving_kv =
        |k: &str| k.starts_with("saber:kv_") || k == "saber:admission_refused_kv_requests_total";
    let bad = mismatches(&live, &census);
    assert!(
        bad.is_empty(),
        "parity:\n{}\n\ncensus:\n{census}",
        bad.join("\n")
    );
    assert_eq!(
        live.keys()
            .filter(
                // the serving plane's own families (not engine census lines): the refusal
                // families, the stream-abort routes and D3's image counters
                |k| !k.starts_with("saber:request_refused")
                    && !k.starts_with("saber:stream_abort")
                    && !k.starts_with("saber:image_")
                    && !serving_kv(k)
            )
            .count(),
        all_defs().len(),
        "every saber: family on /metrics is a census def (and none is missing)"
    );

    // The faults are visible, not only equal.
    assert!(
        live["saber:engine_budget_mismatches_total"] >= 1,
        "{census}"
    );
    assert!(live["saber:engine_forward_errors_total"] >= 1, "{census}");
    assert!(live["saber:engine_fork_refusals_total"] >= 1, "{census}");
    assert!(live["saber:sched_fork_failures_total"] >= 1, "{census}");
    assert!(live["saber:engine_sampled_rows_total"] >= 1, "{census}");
    assert!(live["saber:engine_greedy_rows_total"] >= 1, "{census}");
    assert_eq!(live["saber:draft_uncollected_total"], 2);
    assert_eq!(live["saber:draft_refused_total"], 1);
    assert_eq!(
        live["saber:engine_clean"], 0,
        "an injected fault makes the run unclean"
    );
    assert!(live["saber:sched_steps_total"] > 0);
    assert!(live["saber:sched_finished_length_total"] >= 2, "{census}");
    assert!(live["saber:sched_aborts_running_total"] >= 1, "{census}");

    // NEGATIVE CONTROL: the same census with one counter moved by one is caught, by name.
    let moved = {
        let sc = sched.counters();
        let ac = sched.admission_counters();
        let (mut ec, dc) = sched.with_executor(|e| (e.counters(), e.forward_mut().draft_counts()));
        ec.eager_steps += 1;
        shutdown_census(&sc, &ac, &ec, dc.as_ref())
    };
    let caught = mismatches(&live, &moved);
    assert_eq!(caught.len(), 1, "{caught:?}");
    assert!(
        caught[0].starts_with("saber:engine_eager_steps_total"),
        "{caught:?}"
    );

    server.abort();
}
