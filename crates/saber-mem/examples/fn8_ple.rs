//! FN8 host trial (NO GPU): the PLE row tier's per-token read through FN7's reader pool (worker
//! threads, two hand-offs) vs the io_uring reader (submitted and reaped on the calling thread), on
//! the real drive (O_DIRECT), with the reader byte law (every token's rows identical).
//!
//!   fn8_ple <ple_index.json> <ids.bin (u64 [N][16])> <tokens> <qd_pool>
//!
//! Prints per-token p50 / p90 / mean µs for both readers (alternate tokens, disjoint reads), then a 2,048-token chunk (distinct rows) through the pool at qd 128 and the
//! ring at 128 / 256 entries.

use std::time::Instant;

use saber_mem::dio::OpenMode;
use saber_mem::ple_tier::{PleGate, PleIndex, PleSourceKind, RowReaderPool, UringRowReader};

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p) as usize]
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let text = std::fs::read_to_string(&a[0]).unwrap();
    let idx = PleIndex::from_json(&text, PleSourceKind::CkptBf16, PleGate::Default, None).unwrap();
    let raw = std::fs::read(&a[1]).unwrap();
    let ids: Vec<u64> = raw
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let n: usize = a[2].parse().unwrap();
    let qd: usize = a[3].parse().unwrap();
    let rb = idx.row_bytes as usize;
    let pool = RowReaderPool::new(&idx.files, OpenMode::Direct, qd, rb).unwrap();
    let mut ur = UringRowReader::new(&idx.files, OpenMode::Direct, 16, rb).unwrap();
    let toks = (ids.len() / 16).min(n);
    let (mut tp, mut tu) = (Vec::new(), Vec::new());
    let mut same = 0;
    for t in 0..toks {
        let jobs: Vec<_> = ids[16 * t..16 * t + 16]
            .iter()
            .map(|&r| idx.span(r).unwrap())
            .collect();
        // Timed: each token through ONE reader (even tokens the pool, odd the ring), so neither
        // meets rows the other just pulled into the drive's cache; the byte law is checked after.
        if t % 2 == 0 {
            let t0 = Instant::now();
            pool.read(&jobs).unwrap();
            tp.push(t0.elapsed().as_secs_f64() * 1e6);
        } else {
            let t1 = Instant::now();
            ur.read(&jobs).unwrap();
            tu.push(t1.elapsed().as_secs_f64() * 1e6);
        }
        let (x, y) = (pool.read(&jobs).unwrap(), ur.read(&jobs).unwrap());
        same += usize::from(x == y);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (mp, mu) = (mean(&tp), mean(&tu));
    println!(
        "ple token (16 rows) x {toks}: pool qd {qd} p50 {:.0} p90 {:.0} mean {mp:.0} us | io_uring (16 entries, \
         calling thread) p50 {:.0} p90 {:.0} mean {mu:.0} us | READER BYTE GATE {same}/{toks} tokens identical",
        pct(&mut tp.clone(), 0.5),
        pct(&mut tp.clone(), 0.9),
        pct(&mut tu.clone(), 0.5),
        pct(&mut tu.clone(), 0.9),
    );
    // A chunk: 2,048 tokens' distinct rows.
    let mut rows: Vec<u64> = ids[..(2048 * 16).min(ids.len())].to_vec();
    rows.sort_unstable();
    rows.dedup();
    let jobs: Vec<_> = rows.iter().map(|&r| idx.span(r).unwrap()).collect();
    let pool128 = RowReaderPool::new(&idx.files, OpenMode::Direct, 128, rb).unwrap();
    let t0 = Instant::now();
    let x = pool128.read(&jobs).unwrap();
    let p_ms = t0.elapsed().as_secs_f64() * 1e3;
    for e in [128u32, 256] {
        let mut u = UringRowReader::new(&idx.files, OpenMode::Direct, e, rb).unwrap();
        let t1 = Instant::now();
        let y = u.read(&jobs).unwrap();
        println!(
            "ple chunk: {} distinct rows: pool qd 128 {p_ms:.1} ms | io_uring {e} entries {:.1} ms | bytes {}",
            jobs.len(),
            t1.elapsed().as_secs_f64() * 1e3,
            if x == y { "IDENTICAL" } else { "DIFFER" }
        );
    }
}
