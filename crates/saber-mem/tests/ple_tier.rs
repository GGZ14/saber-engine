//! FN7: the PLE row tier over synthetic shards laid out like the checkpoint's (two shards per
//! file at the real shards' odd offsets, so rows straddle 512 B blocks exactly as they do on the
//! drive): every staged byte checked, the cache's hits counted, the forced-miss control, the
//! gated fp8 source refused without its gate.

use std::path::Path;

use saber_mem::dio::OpenMode;
use saber_mem::ple_tier::{PleError, PleGate, PleIndex, PleSourceKind, PleTier};
use saber_test_scratch::ScratchDir;

const RPS: u64 = 1000; // rows per shard
const SHARDS: u64 = 4;
const DIM: u64 = 160;

fn row_byte(kind: u64, row: u64, k: u64) -> u8 {
    let x = (row.wrapping_mul(0x9E37_79B9) ^ k.wrapping_mul(0x85EB_CA6B) ^ kind << 40) >> 3;
    (x ^ (x >> 11)) as u8
}

/// Write the fake checkpoint (2 files, 2 shards each, bf16 rows at offsets 360 / 104 like the
/// real shards 0 / 1) and the fp8 file (one file, offset 416,370 like the real one), and the index.
fn fixture(tag: &str) -> (ScratchDir, String) {
    let d = ScratchDir::new(&format!("saber-fn7-ple-{tag}"));
    let mut shards = Vec::new();
    let offs = [360u64, 104];
    for f in 0..2u64 {
        let mut bytes = vec![0u8; offs[0] as usize];
        let mut cur = offs[0];
        let mut first_off = [0u64; 2];
        for s in 0..2u64 {
            let k = 2 * f + s;
            if s == 1 {
                // pad so the second shard starts at 104 mod 512
                while cur % 512 != offs[1] {
                    bytes.push(0);
                    cur += 1;
                }
            }
            first_off[s as usize] = cur;
            for r in 0..RPS {
                let row = k * RPS + r;
                for e in 0..DIM * 2 {
                    bytes.push(row_byte(0, row, e));
                }
            }
            cur += RPS * DIM * 2;
            shards.push(
                serde_json::json!({"shard": k, "file": format!("model-{f}.safetensors"),
                "offset": first_off[s as usize], "rows": RPS}),
            );
        }
        bytes.resize(bytes.len().div_ceil(4096) * 4096 + 4096, 0);
        std::fs::write(d.join(format!("model-{f}.safetensors")), &bytes).unwrap();
    }
    let fp8_off = 416_370u64;
    let mut fp8 = vec![0u8; fp8_off as usize];
    let mut fshards = Vec::new();
    for k in 0..SHARDS {
        fshards.push(serde_json::json!({"shard": k, "offset": fp8.len()}));
        for r in 0..RPS {
            for e in 0..DIM {
                fp8.push(row_byte(1, k * RPS + r, e));
            }
        }
    }
    fp8.resize(fp8.len().div_ceil(4096) * 4096 + 4096, 0);
    std::fs::write(d.join("model-fp8.safetensors"), &fp8).unwrap();
    let idx = serde_json::json!({
        "table": {"rows": RPS * SHARDS, "rows_per_shard": RPS, "shards": SHARDS, "row_dim": DIM, "rows_per_token": 16},
        "sources": {
            "ckpt_bf16": {"default": true, "checkpoint": "/nonexistent/ckpt", "row_bytes": 320, "shards": shards},
            "qwen_fp8": {"default": false, "file": "/nonexistent/model-fp8.safetensors", "row_bytes": 160,
                         "table_scale": {"value": 0.00019931793212890625}, "shards": fshards}
        }
    });
    (d, idx.to_string())
}

fn ids(n: usize, seed: u64) -> Vec<u64> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            // Zipf-ish: a hot band and a long tail.
            let r = (x >> 33) % (RPS * SHARDS);
            if (x >> 20) % 3 == 0 {
                r % 40
            } else {
                r
            }
        })
        .collect()
}

fn check(staged: &[u8], ids: &[u64], kind: u64, rb: u64) {
    for (p, &r) in ids.iter().enumerate() {
        for e in 0..rb {
            assert_eq!(
                staged[p * rb as usize + e as usize],
                row_byte(kind, r, e),
                "position {p} row {r} byte {e}"
            );
        }
    }
}

fn tier(dir: &Path, text: &str, cache: u32, qd: usize) -> PleTier {
    let idx =
        PleIndex::from_json(text, PleSourceKind::CkptBf16, PleGate::Default, Some(dir)).unwrap();
    PleTier::new(idx, OpenMode::Buffered, qd, cache).unwrap()
}

#[test]
fn the_span_law_matches_the_real_shards() {
    let (d, text) = fixture("span");
    let idx = PleIndex::from_json(
        &text,
        PleSourceKind::CkptBf16,
        PleGate::Default,
        Some(d.path()),
    )
    .unwrap();
    assert_eq!(idx.bytes_per_token(), 5120);
    // Shard 0 at 360 mod 512: row 0 spans [0, 1024) (straddles), row 1 at 680 -> [512, 1024).
    assert_eq!(idx.span(0).unwrap(), (0, 0, 1024, 360));
    assert_eq!(idx.span(1).unwrap(), (0, 512, 512, 168));
    let straddles = (0..RPS).filter(|&r| idx.span(r).unwrap().2 == 1024).count();
    assert_eq!(
        straddles as f64 / RPS as f64,
        0.625,
        "the index's straddle_512 for offset 360"
    );
    assert!(matches!(idx.span(RPS * SHARDS), Err(PleError::Row { .. })));
}

#[test]
fn every_staged_byte_the_cache_and_the_forced_miss_control() {
    let (d, text) = fixture("gather");
    for qd in [1, 16] {
        let mut t = tier(&d, &text, 512, qd);
        let a = ids(16 * 64, 5);
        let mut staged = vec![0u8; a.len() * 320];
        let c1 = t.gather(&a, &mut staged).unwrap();
        check(&staged, &a, 0, 320);
        assert_eq!(c1.requested, 1024);
        assert_eq!(c1.cache_hits, 0, "a cold cache");
        assert_eq!(c1.nvme_reads, c1.distinct);
        assert!(c1.distinct < c1.requested, "the hot band dedupes");
        assert_eq!(c1.bytes_read, 512 * c1.nvme_reads + 512 * c1.straddles);
        // The same chunk again: the cache (512 rows) holds the most recent 512 distinct rows.
        let c2 = t.gather(&a, &mut staged).unwrap();
        check(&staged, &a, 0, 320);
        assert_eq!(c2.cache_hits + c2.nvme_reads, c2.distinct);
        assert_eq!(c2.cache_hits, c2.distinct.min(512));
        // THE CONTROL: evict, gather again -> every distinct row read from the drive.
        t.evict_all();
        let c3 = t.gather(&a, &mut staged).unwrap();
        check(&staged, &a, 0, 320);
        assert_eq!((c3.cache_hits, c3.nvme_reads), (0, c3.distinct));
        assert!(t
            .render_prometheus()
            .contains("saber_ple_rows_total{path=\"nvme\"}"));
    }
}

#[test]
fn no_cache_reads_every_distinct_row_every_time() {
    let (d, text) = fixture("nocache");
    let mut t = tier(&d, &text, 0, 4);
    let a = ids(160, 9);
    let mut staged = vec![0u8; a.len() * 320];
    for _ in 0..2 {
        let c = t.gather(&a, &mut staged).unwrap();
        assert_eq!(c.nvme_reads, c.distinct);
        check(&staged, &a, 0, 320);
    }
    let mut wrong = vec![0u8; 10];
    assert!(matches!(
        t.gather(&a, &mut wrong),
        Err(PleError::Staging(_))
    ));
}

#[test]
fn the_fp8_source_is_gated_and_plumbed() {
    let (d, text) = fixture("fp8");
    let refused = PleIndex::from_json(
        &text,
        PleSourceKind::QwenFp8,
        PleGate::Default,
        Some(d.path()),
    );
    assert!(matches!(refused, Err(PleError::Gated)), "no gate, no fp8");
    let idx = PleIndex::from_json(
        &text,
        PleSourceKind::QwenFp8,
        PleGate::ApprovedByUser,
        Some(d.path()),
    )
    .unwrap();
    assert_eq!((idx.row_bytes, idx.bytes_per_token()), (160, 2560));
    assert!((idx.table_scale - 0.000_199_317_93).abs() < 1e-9);
    let mut t = PleTier::new(idx, OpenMode::Buffered, 8, 64).unwrap();
    assert!(t.boot_line().contains("GATED"));
    let a = ids(16 * 8, 3);
    let mut staged = vec![0u8; a.len() * 160];
    t.gather(&a, &mut staged).unwrap();
    check(&staged, &a, 1, 160);
}

/// FN8: the io_uring reader returns the pool's bytes for the same jobs (the reader byte law), in
/// one wave and across several waves of its ring; a job naming a missing file is refused.
#[test]
fn the_io_uring_reader_returns_the_pools_bytes() {
    use saber_mem::ple_tier::{RowReaderPool, UringRowReader};
    let (dir, text) = fixture("uring");
    let idx = PleIndex::from_json(
        &text,
        PleSourceKind::CkptBf16,
        PleGate::Default,
        Some(dir.path()),
    )
    .unwrap();
    let rb = idx.row_bytes as usize;
    let pool = RowReaderPool::new(&idx.files, OpenMode::Buffered, 4, rb).unwrap();
    for (entries, n) in [(16u32, 16usize), (16, 200), (64, 1000)] {
        let mut ur = match UringRowReader::new(&idx.files, OpenMode::Buffered, entries, rb) {
            Ok(u) => u,
            Err(e) => {
                // A kernel with io_uring disabled refuses at setup, by name; nothing else may fail.
                assert!(e.to_string().contains("io_uring"), "{e}");
                return;
            }
        };
        let rows = ids(n, 0x0e1 + n as u64);
        let jobs: Vec<_> = rows.iter().map(|&r| idx.span(r).unwrap()).collect();
        let a = pool.read(&jobs).unwrap();
        let b = ur.read(&jobs).unwrap();
        assert_eq!(a, b, "{n} rows through a {entries}-entry ring");
        check(&b, &rows, 0, rb as u64);
        // THE CONTROL: a job naming a file the reader does not have is refused, not zero-filled.
        let mut bad = jobs.clone();
        bad[0].0 = 99;
        assert!(ur.read(&bad).is_err());
    }
}

/// FN121 (the PLE prefetch law): rows prefetched asynchronously land in the RAM cache at the next
/// settle / gather and the staged bytes are EXACTLY a plain gather's; a prefetched row is a hit at
/// the gate, a row never prefetched is a counted drive read (the stall), a second prefetch of rows
/// in flight submits nothing new, and the read log records every batch with its timings.
#[test]
fn fn121_prefetched_rows_are_the_gathers_bytes_and_unprefetched_rows_are_counted_reads() {
    let (dir, text) = fixture("fn121");
    let rows = ids(64, 0x121);
    // the reference: a plain gather on a cold tier
    let mut a = tier(dir.path(), &text, 4096, 8);
    let mut want = vec![0u8; rows.len() * 320];
    a.gather(&rows, &mut want).unwrap();
    check(&want, &rows, 0, 320);
    // the arm: prefetch the first 48 ids, then gather all 64
    let mut b = tier(dir.path(), &text, 4096, 8);
    b.set_read_log(true);
    let n = b.prefetch(&rows[..48]).unwrap();
    let distinct48: std::collections::HashSet<u64> = rows[..48].iter().copied().collect();
    assert_eq!(n, distinct48.len() as u64);
    // in flight: a repeat submits nothing, and nothing is resident yet
    assert_eq!(b.prefetch(&rows[..48]).unwrap(), 0);
    assert_eq!(b.prefetch_counts().pending, distinct48.len() as u64);
    b.settle().unwrap();
    assert_eq!(b.resident(&rows[..48]), 48);
    let mut got = vec![0u8; rows.len() * 320];
    let c = b.gather(&rows, &mut got).unwrap();
    assert_eq!(got, want, "a prefetched row is the gather's bytes");
    let distinct: std::collections::HashSet<u64> = rows.iter().copied().collect();
    let late = distinct.difference(&distinct48).count() as u64;
    assert_eq!(
        c.nvme_reads, late,
        "only the rows never prefetched read at the gate"
    );
    let log = b.take_read_log();
    assert_eq!(log.len(), if late > 0 { 2 } else { 1 });
    assert_eq!(log[0].kind, "prefetch");
    assert_eq!(log[0].reads.len(), distinct48.len());
    assert!(log[0].reads.iter().all(|r| r.dur_us > 0.0));
    assert!(b.take_read_log().is_empty());
    // THE CONTROL: without the settle a gather still settles first (never a wrong byte)
    let mut d = tier(dir.path(), &text, 4096, 8);
    d.prefetch(&rows).unwrap();
    let mut got2 = vec![0u8; rows.len() * 320];
    let c2 = d.gather(&rows, &mut got2).unwrap();
    assert_eq!(got2, want);
    assert_eq!(c2.nvme_reads, 0);
}
