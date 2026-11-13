//! The usage store's laws -- rollups equal the raw rows (every period, every
//! merge), history survives a restart, a proxied request seen by both capture paths and a
//! re-polled engine page are recorded exactly once, retention prunes rows but never rollups,
//! and the monthly / lifetime / Prometheus readings come from the rollups.

use std::collections::BTreeMap;

use saber_console::usage::{
    self, ingest_page, Op, Period, RowFilter, UsageRow, UsageStore, DAY_MS, HOUR_MS, SRC_ENGINE,
    SRC_PROXY,
};
use serde_json::json;

const T0: i64 = 1_793_000_000_000; // 2026-10-27

fn row(uid: &str, ts: i64, model: &str, key: Option<&str>, i: i64) -> UsageRow {
    UsageRow {
        uid: uid.into(),
        ts_ms: ts,
        source: SRC_PROXY,
        endpoint: "/v1/chat/completions".into(),
        model: Some(model.into()),
        key: key.map(str::to_string),
        stream: Some(i % 2 == 0),
        status: Some(match i % 11 {
            0 => 500,
            5 => 499,
            7 => 400,
            _ => 200,
        }),
        prompt_tokens: Some(100 + i * 7),
        completion_tokens: Some(20 + i * 3),
        cached_tokens: Some(i * 2),
        ttft_ms: (i % 5 != 0).then_some(30.0 + i as f64 * 1.5),
        latency_ms: Some(200.0 + i as f64),
        decode_tps: (i % 3 != 0).then_some(40.0 + i as f64),
        finish_reason: Some("stop".into()),
        ..UsageRow::default()
    }
}

/// The scripted mix: three models, two keys + none, spread over 3 days, with merges.
fn script(store: &UsageStore) -> Vec<UsageRow> {
    let models = ["qwen-a", "qwen-b", "flash"];
    let keys = [Some("laptop"), None, Some("ci")];
    let mut rows = Vec::new();
    for i in 0..240_i64 {
        let ts = T0 + i * 17 * 60_000 + (i % 7) * 1_000;
        let r = row(
            &format!("p-{i}"),
            ts,
            models[(i % 3) as usize],
            keys[(i % 3) as usize],
            i,
        );
        store.record(r.clone());
        rows.push(r);
    }
    store.flush();
    rows
}

fn raw_sums(rows: &[UsageRow]) -> (i64, i64, i64, i64, i64) {
    let mut s = (0, 0, 0, 0, 0);
    for r in rows {
        s.0 += 1;
        s.1 += r.prompt_tokens.unwrap_or(0);
        s.2 += r.completion_tokens.unwrap_or(0);
        s.3 += r.cached_tokens.unwrap_or(0);
        s.4 += i64::from(r.is_error());
    }
    s
}

fn cell_sums(store: &UsageStore, p: Period) -> (i64, i64, i64, i64, i64) {
    let mut s = (0, 0, 0, 0, 0);
    for c in store.cells(p, i64::MIN, i64::MAX).unwrap() {
        s.0 += c.agg.requests;
        s.1 += c.agg.prompt;
        s.2 += c.agg.completion;
        s.3 += c.agg.cached;
        s.4 += c.agg.errors;
    }
    s
}

fn all_rows(store: &UsageStore) -> Vec<UsageRow> {
    store
        .rows(&RowFilter {
            limit: 100_000,
            ..RowFilter::default()
        })
        .unwrap()
        .1
}

#[test]
fn rollups_equal_the_raw_rows_through_merges() {
    let dir = saber_test_scratch::ScratchDir::new("console-usage-sums");
    let store = UsageStore::open(&dir.path().join("u.db"), -360).unwrap();
    let rows = script(&store);
    let raw = raw_sums(&rows);
    assert_eq!(raw_sums(&all_rows(&store)), raw);
    for p in [Period::Minute, Period::Hour, Period::Day] {
        assert_eq!(cell_sums(&store, p), raw, "{p:?}");
    }
    // the engine's view arrives for 40 of them (more tokens, a different ts): merged in place
    for i in 0..40_i64 {
        let mut e = row(
            &format!("p-{i}"),
            T0 + i * 17 * 60_000 - 3_000_000,
            "qwen-a",
            Some("console"),
            i,
        );
        e.source = SRC_ENGINE;
        e.completion_tokens = Some(1_000 + i);
        e.engine_boot = Some(9);
        e.engine_id = Some(i + 1);
        store.record(e);
    }
    store.flush();
    let rows = all_rows(&store);
    assert_eq!(rows.len(), 240, "merged, not added");
    let raw = raw_sums(&rows);
    assert_eq!(raw.0, 240);
    for p in [Period::Minute, Period::Hour, Period::Day] {
        assert_eq!(cell_sums(&store, p), raw, "{p:?} after merges");
    }
    let merged = store.row("p-3").unwrap().unwrap();
    assert_eq!(merged.source, SRC_PROXY | SRC_ENGINE);
    assert_eq!(
        merged.completion_tokens,
        Some(1_003),
        "the engine's count wins"
    );
    let m1 = store.row("p-1").unwrap().unwrap();
    assert_eq!(
        m1.key.as_deref(),
        Some("console"),
        "row 1 had no proxy key: the engine's label then"
    );
    let m0 = store.row("p-0").unwrap().unwrap();
    assert_eq!(m0.key.as_deref(), Some("laptop"), "the proxy's label wins");
    assert_eq!(m0.ts_ms, T0, "the proxy's arrival wins");

    // the summary's totals (rollups) equal the raw rows over the same span
    let s = usage::summary(&store, T0 - DAY_MS, T0 + 5 * DAY_MS, None).unwrap();
    assert_eq!(s["totals"]["requests"], 240);
    assert_eq!(s["totals"]["prompt_tokens"], raw.1);
    assert_eq!(s["totals"]["completion_tokens"], raw.2);
    assert_eq!(s["period"], "h");
    let by_model: i64 = s["by_model"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["requests"].as_i64().unwrap())
        .sum();
    assert_eq!(by_model, 240);
    let series: i64 = s["series"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["completion_tokens"].as_i64().unwrap())
        .sum();
    assert_eq!(series, raw.2);
    // the quantiles come from the merged histograms, within a bucket of the exact value
    let mut ttft: Vec<f64> = rows.iter().filter_map(|r| r.ttft_ms).collect();
    ttft.sort_by(f64::total_cmp);
    let exact = ttft[(ttft.len() as f64 * 0.5).ceil() as usize - 1];
    let p50 = s["totals"]["ttft_p50_ms"].as_f64().unwrap();
    assert!(
        (p50 / exact - 1.0).abs() < 0.2,
        "p50 {p50} vs exact {exact}"
    );
    let mean: f64 = rows.iter().filter_map(|r| r.decode_tps).sum::<f64>()
        / rows.iter().filter(|r| r.decode_tps.is_some()).count() as f64;
    assert!((s["totals"]["decode_tps_mean"].as_f64().unwrap() - mean).abs() < 1e-6);
    // a short span reads minute cells
    let s = usage::summary(&store, T0, T0 + HOUR_MS, None).unwrap();
    assert_eq!(s["period"], "m");
    assert_eq!(
        s["totals"]["requests"].as_i64().unwrap(),
        rows.iter().filter(|r| r.ts_ms < T0 + HOUR_MS).count() as i64
    );
}

#[test]
fn history_survives_a_restart() {
    let dir = saber_test_scratch::ScratchDir::new("console-usage-restart");
    let path = dir.path().join("nested/dir/u.db");
    let raw = {
        let store = UsageStore::open(&path, 0).unwrap();
        let rows = script(&store);
        store.send(Op::Cursor {
            boot: 42,
            after: 17,
        });
        store.send(Op::Energy {
            ts_ms: T0,
            joules: 3.6e6,
        });
        store.flush();
        raw_sums(&rows)
    };
    let store = UsageStore::open(&path, 0).unwrap();
    assert_eq!(raw_sums(&all_rows(&store)), raw);
    assert_eq!(cell_sums(&store, Period::Day), raw);
    assert_eq!(store.cursor().unwrap(), Some((42, 17)));
    let m = usage::monthly(&store, T0 + 3 * DAY_MS, 3, 7).unwrap();
    let kwh: f64 = m["months"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|r| r["energy_kwh"].as_f64())
        .sum();
    assert!((kwh - 1.0).abs() < 1e-9, "{m}");
}

fn page(boot: u64, recs: &[serde_json::Value]) -> serde_json::Value {
    let newest = recs
        .iter()
        .filter_map(|r| r["id"].as_u64())
        .max()
        .unwrap_or(0);
    json!({"object": "list", "data": recs, "boot_id": boot, "newest_id": newest, "dropped": 0})
}

fn eng(id: u64, client_ref: Option<&str>) -> serde_json::Value {
    json!({"id": id, "ts_ms": T0 + id as i64 * 1000, "endpoint": "/v1/chat/completions",
        "model": "qwen", "key": "console", "stream": true, "prompt_tokens": 50,
        "completion_tokens": 10, "cached_tokens": 5, "ttft_ms": 40.0, "latency_ms": 300.0,
        "decode_tps": 33.0, "finish_reason": "stop", "status": 200, "images": 0,
        "thinking": true, "client_ref": client_ref})
}

#[test]
fn proxy_and_poller_record_each_request_once() {
    let dir = saber_test_scratch::ScratchDir::new("console-usage-dedupe");
    let path = dir.path().join("u.db");
    let store = UsageStore::open(&path, 0).unwrap();
    // the proxy's row for c1 (no usage block: a stream without include_usage)
    let mut p = row("c1", T0 + 900, "qwen", Some("laptop"), 1);
    p.prompt_tokens = None;
    p.completion_tokens = None;
    p.cached_tokens = None;
    store.record(p);
    store.flush();
    // the engine's page: c1 (proxied), 2 (direct), 3 (proxied, the proxy row comes later)
    let pg = page(7, &[eng(1, Some("c1")), eng(2, None), eng(3, Some("c3"))]);
    let (n, cur, more) = ingest_page(&store, &pg, None);
    assert_eq!((n, cur, more), (3, (7, 3), false));
    store.flush();
    let mut p3 = row("c3", T0 + 2_900, "qwen", None, 3);
    p3.completion_tokens = None;
    store.record(p3);
    store.flush();
    let rows = all_rows(&store);
    assert_eq!(rows.len(), 3, "{rows:#?}");
    let c1 = store.row("c1").unwrap().unwrap();
    assert_eq!(c1.source, SRC_PROXY | SRC_ENGINE);
    assert_eq!(
        (c1.prompt_tokens, c1.completion_tokens),
        (Some(50), Some(10))
    );
    assert_eq!(c1.key.as_deref(), Some("laptop"));
    assert_eq!(
        c1.ttft_ms,
        row("x", 0, "", None, 1).ttft_ms,
        "the proxy's TTFT"
    );
    assert_eq!(c1.engine_id, Some(1));
    let direct = store.row("e:7:2").unwrap().unwrap();
    assert_eq!(direct.source, SRC_ENGINE);
    assert_eq!(direct.key.as_deref(), Some("console"));
    // a console restart re-polls the whole ring: nothing is added
    drop(store);
    let store = UsageStore::open(&path, 0).unwrap();
    assert_eq!(store.cursor().unwrap(), Some((7, 3)));
    ingest_page(&store, &pg, None);
    store.flush();
    assert_eq!(all_rows(&store).len(), 3);
    assert_eq!(cell_sums(&store, Period::Day).0, 3);
    assert_eq!(
        cell_sums(&store, Period::Day).2,
        all_rows(&store)
            .iter()
            .map(|r| r.completion_tokens.unwrap_or(0))
            .sum::<i64>()
    );
    // a new engine boot restarts its ids: id 1 of boot 8 is another request
    let (_, cur, _) = ingest_page(&store, &page(8, &[eng(1, None)]), Some((7, 3)));
    assert_eq!(cur, (8, 1));
    store.flush();
    assert_eq!(all_rows(&store).len(), 4);
    // the source filter tells direct traffic apart
    let (n, _) = store
        .rows(&RowFilter {
            source: Some("engine".into()),
            limit: 10,
            ..RowFilter::default()
        })
        .unwrap();
    assert_eq!(n, 2);
}

#[test]
fn retention_prunes_rows_and_keeps_rollups() {
    let dir = saber_test_scratch::ScratchDir::new("console-usage-prune");
    let store = UsageStore::open(&dir.path().join("u.db"), 0).unwrap();
    let rows = script(&store);
    let raw = raw_sums(&rows);
    let cut = T0 + DAY_MS;
    store.send(Op::Prune { before_ms: cut });
    store.flush();
    let left = all_rows(&store);
    assert_eq!(
        left.len(),
        rows.iter().filter(|r| r.ts_ms >= cut).count(),
        "rows before the cut are gone"
    );
    assert!(left.iter().all(|r| r.ts_ms >= cut));
    assert_eq!(store.writer_stats().rows_pruned as usize, 240 - left.len());
    // rollups are untouched: the long-range readings still see every request
    for p in [Period::Minute, Period::Hour, Period::Day] {
        assert_eq!(cell_sums(&store, p), raw, "{p:?}");
    }
    let m = usage::monthly(&store, T0 + 3 * DAY_MS, 2, 10).unwrap();
    let total: i64 = m["months"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["requests"].as_i64().unwrap())
        .sum();
    assert_eq!(total, 240);
    let days: i64 = m["days"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["completion_tokens"].as_i64().unwrap())
        .sum();
    assert_eq!(days, raw.2);
    let prom = usage::prometheus(&store).unwrap();
    let mut by_kind: BTreeMap<&str, i64> = BTreeMap::new();
    for l in prom
        .lines()
        .filter(|l| l.starts_with("saber_console_tokens_total{"))
    {
        let kind = l
            .split("kind=\"")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap();
        let v: i64 = l.rsplit(' ').next().unwrap().parse().unwrap();
        *by_kind.entry(kind).or_insert(0) += v;
    }
    let kinds: Vec<(&str, i64)> = by_kind.into_iter().collect();
    assert_eq!(
        kinds,
        vec![("cached", raw.3), ("completion", raw.2), ("prompt", raw.1)]
    );
    // CSV export: one header + one line a row
    let csv = usage::csv(&left);
    assert_eq!(csv.lines().count(), left.len() + 1);
    assert!(csv.starts_with("uid,ts_ms,time_utc,"));
}

#[test]
fn filters_sort_and_page() {
    let dir = saber_test_scratch::ScratchDir::new("console-usage-filter");
    let store = UsageStore::open(&dir.path().join("u.db"), 0).unwrap();
    let rows = script(&store);
    let f = |f: RowFilter| store.rows(&RowFilter { limit: 1000, ..f }).unwrap();
    let (n, r) = f(RowFilter {
        model: Some("flash".into()),
        ..RowFilter::default()
    });
    assert_eq!(n, 80);
    assert!(r.iter().all(|x| x.model.as_deref() == Some("flash")));
    let (n, _) = f(RowFilter {
        status: Some("error".into()),
        ..RowFilter::default()
    });
    assert_eq!(n, rows.iter().filter(|r| r.is_error()).count() as i64);
    let (n, _) = f(RowFilter {
        status: Some("aborted".into()),
        ..RowFilter::default()
    });
    assert_eq!(
        n,
        rows.iter().filter(|r| r.status == Some(499)).count() as i64
    );
    let (_, r) = f(RowFilter {
        sort: Some("latency_ms".into()),
        asc: false,
        limit: 3,
        ..RowFilter::default()
    });
    assert_eq!(r[0].uid, "p-239");
    let (n, r) = store
        .rows(&RowFilter {
            limit: 10,
            offset: 235,
            sort: Some("ts_ms".into()),
            asc: true,
            ..RowFilter::default()
        })
        .unwrap();
    assert_eq!((n, r.len()), (240, 5));
    let (n, _) = f(RowFilter {
        min_latency_ms: Some(400.0),
        q: Some("qwen".into()),
        ..RowFilter::default()
    });
    assert_eq!(
        n,
        rows.iter()
            .filter(
                |r| r.latency_ms.unwrap() >= 400.0 && r.model.as_deref().unwrap().contains("qwen")
            )
            .count() as i64
    );
    // an unknown sort column falls back to the timestamp (never interpolated)
    let (_, r) = f(RowFilter {
        sort: Some("ts_ms; DROP TABLE requests".into()),
        limit: 1,
        ..RowFilter::default()
    });
    assert_eq!(r[0].uid, "p-239");
    assert_eq!(store.row_count().unwrap(), 240);
}
