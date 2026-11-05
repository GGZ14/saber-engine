//! FN7 host bench (no GPU): the drive under the expert tier and the PLE tier, and the tier's
//! demand path replayed over FN3's routes. Every read is `O_DIRECT` on the real pack / checkpoint.
//!
//!   fn7_bench xtier-qd <experts.xtier> <n> <qd,..>          random whole-record reads
//!   fn7_bench ple-qd <ple_index.json> <rows> <qd,..>        random row reads (no cache)
//!   fn7_bench ple-replay <ple_index.json> <ids.bin> <tokens/chunk> <qd> <cache_rows,..>
//!   fn7_bench replay <experts.xtier> <routes.bin> <profile.json|-> <vram> <ram> <prefetch> <mode>
//!            mode = prefill (one step per prompt) | decode (one token a step) | conc8 (8 prompts
//!            interleaved, one token each a step) | forced (decode after evict_all)

use std::path::Path;
use std::time::Instant;

use saber_mem::dio::{AlignedBuf, OpenMode};
use saber_mem::expert_mover::DrainMover;
use saber_mem::expert_tier::{ExpertTier, PathCounts, RouteProfile, TierParams};
use saber_mem::ple_tier::{PleGate, PleIndex, PleSourceKind, PleTier, RowReaderPool};
use saber_mem::xtier::XtierFile;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn qds(s: &str) -> Vec<usize> {
    s.split(',').map(|x| x.parse().unwrap()).collect()
}

fn xtier_qd(a: &[String]) {
    let xt = XtierFile::open(Path::new(&a[0]), OpenMode::Direct).unwrap();
    let n: usize = a[1].parse().unwrap();
    let stride = xt.header().stride as usize;
    let recs = xt.header().records();
    for qd in qds(&a[2]) {
        let mut buf = AlignedBuf::zeroed(stride * qd);
        let mut rng = Rng(0x9e37 + qd as u64);
        let t = Instant::now();
        let (mut done, mut verified) = (0, 0);
        while done < n {
            let k = qd.min(n - done);
            let recs_now: Vec<u64> = (0..k).map(|_| rng.next() % recs).collect();
            std::thread::scope(|sc| {
                for (r, dst) in recs_now.iter().zip(buf.chunks_mut(stride)) {
                    let xt = &xt;
                    sc.spawn(move || xt.read_record(*r, dst).unwrap());
                }
            });
            if done == 0 {
                for (j, &r) in recs_now.iter().enumerate() {
                    verified += u32::from(xt.verify(r, &buf[j * stride..(j + 1) * stride]));
                }
            }
            done += k;
        }
        let s = t.elapsed().as_secs_f64();
        println!(
            "xtier-qd qd {qd:>3}: {n} records x {stride} B in {s:.3} s = {:.2} GB/s, {:.1} us/record \
             (first batch sha256 {verified}/{} verified) | drive roof 14.8 GB/s seq",
            (n * stride) as f64 / s / 1e9,
            s * 1e6 / n as f64,
            qd.min(n)
        );
    }
}

fn index(p: &str) -> PleIndex {
    let text = std::fs::read_to_string(p).unwrap();
    PleIndex::from_json(&text, PleSourceKind::CkptBf16, PleGate::Default, None).unwrap()
}

fn ple_qd(a: &[String]) {
    let idx = index(&a[0]);
    let n: usize = a[1].parse().unwrap();
    for qd in qds(&a[2]) {
        let rb = idx.row_bytes as usize;
        let pool = RowReaderPool::new(&idx.files, OpenMode::Direct, qd, rb).unwrap();
        let mut rng = Rng(0x51 + qd as u64);
        let jobs: Vec<_> = (0..n)
            .map(|_| idx.span(rng.next() % idx.rows).unwrap())
            .collect();
        let bytes: u64 = jobs.iter().map(|j| u64::from(j.2)).sum();
        let t = Instant::now();
        pool.read(&jobs).unwrap();
        let s = t.elapsed().as_secs_f64();
        let mut lat = Vec::new();
        for _ in 0..64 {
            let tok: Vec<_> = (0..16)
                .map(|_| idx.span(rng.next() % idx.rows).unwrap())
                .collect();
            let t1 = Instant::now();
            pool.read(&tok).unwrap();
            lat.push(t1.elapsed().as_secs_f64() * 1e6);
        }
        lat.sort_by(f64::total_cmp);
        println!(
            "ple-qd qd {qd:>4}: {n} rows in {s:.3} s = {:.0} k rows/s, {:.2} us/row, {:.2} GB/s of spans \
             ({:.1}% straddled) | one token (16 rows) p50 {:.0} us p90 {:.0} us | drive 4K QD1 45 us",
            n as f64 / s / 1e3,
            s * 1e6 / n as f64,
            bytes as f64 / s / 1e9,
            100.0 * jobs.iter().filter(|j| j.2 > 512).count() as f64 / n as f64,
            lat[32],
            lat[57]
        );
    }
}

fn ple_replay(a: &[String]) {
    let raw = std::fs::read(&a[1]).unwrap();
    let ids: Vec<u64> = raw
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let chunk: usize = a[2].parse::<usize>().unwrap() * 16;
    let qd: usize = a[3].parse().unwrap();
    for cache in qds(&a[4]) {
        let mut t = PleTier::new(index(&a[0]), OpenMode::Direct, qd, cache as u32).unwrap();
        let mut staged = vec![0u8; chunk * 320];
        let mut worst = 0f64;
        for c in ids.chunks(chunk) {
            let k = t.gather(c, &mut staged[..c.len() * 320]).unwrap();
            worst = worst.max(k.total_s);
        }
        let tot = t.totals();
        let chunks = ids.len().div_ceil(chunk);
        t.evict_all();
        let first = &ids[..chunk.min(ids.len())];
        let k = t.gather(first, &mut staged[..first.len() * 320]).unwrap();
        println!(
            "ple-replay {} tok, chunk {} tok, qd {qd}, cache {cache} rows ({:.0} MiB): requested {} \
             distinct {} cache hits {} ({:.1}% of distinct; {:.1}% of lookups served without a read) \
             nvme {} ({:.1}% straddled) | {:.3} ms/chunk mean, worst {:.2} ms, reads {:.2} GB/s | \
             forced-miss control: first chunk after evict_all nvme {} == distinct {}, hits {}",
            ids.len() / 16,
            chunk / 16,
            cache as f64 * 320.0 / 1_048_576.0,
            tot.requested,
            tot.distinct,
            tot.cache_hits,
            100.0 * tot.cache_hits as f64 / tot.distinct as f64,
            100.0 * (tot.requested - tot.nvme_reads) as f64 / tot.requested as f64,
            tot.nvme_reads,
            100.0 * tot.straddles as f64 / tot.nvme_reads.max(1) as f64,
            tot.total_s * 1e3 / chunks as f64,
            worst * 1e3,
            tot.bytes_read as f64 / tot.read_s.max(1e-9) / 1e9,
            k.nvme_reads,
            k.distinct,
            k.cache_hits
        );
    }
}

fn load_routes(p: &str) -> Vec<(usize, Vec<i32>)> {
    let raw = std::fs::read(p).unwrap();
    let u = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize;
    let mut o = 4;
    let mut v = Vec::new();
    for _ in 0..u(0) {
        let t = u(o);
        o += 4;
        let r: Vec<i32> = raw[o..o + 48 * t * 40]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        o += 48 * t * 40;
        v.push((t, r));
    }
    v
}

fn replay(a: &[String]) {
    let xt = XtierFile::open(Path::new(&a[0]), OpenMode::Direct).unwrap();
    let routes = load_routes(&a[1]);
    let prof = (a[2] != "-")
        .then(|| RouteProfile::from_json(&std::fs::read_to_string(&a[2]).unwrap()).unwrap());
    let p = TierParams {
        vram_slots: a[3].parse().unwrap(),
        ram_slots: a[4].parse().unwrap(),
        prefetch_per_step: a[5].parse().unwrap(),
        ..TierParams::default()
    };
    let mode = a[6].as_str();
    let mut tier = ExpertTier::new(xt.header(), p, prof.as_ref(), 1 << 46, 1 << 45).unwrap();
    println!("{}", tier.boot_line());
    let mut mv = DrainMover::new(&xt, 32);
    // Steps: (prompt, token lo, token hi) per participating prompt.
    let mut steps: Vec<Vec<(usize, usize, usize)>> = Vec::new();
    match mode {
        "prefill" => {
            for (i, (t, _)) in routes.iter().enumerate() {
                steps.push(vec![(i, 0, *t)]);
            }
        }
        "decode" | "forced" => {
            for (i, (t, _)) in routes.iter().enumerate() {
                for k in 0..*t {
                    steps.push(vec![(i, k, k + 1)]);
                }
            }
        }
        "conc8" => {
            for (g, grp) in routes.chunks(8).enumerate() {
                let maxt = grp.iter().map(|x| x.0).max().unwrap();
                for k in 0..maxt {
                    steps.push(
                        grp.iter()
                            .enumerate()
                            .filter(|(_, x)| k < x.0)
                            .map(|(j, _)| (g * 8 + j, k, k + 1))
                            .collect(),
                    );
                }
            }
        }
        _ => panic!("mode"),
    }
    if mode == "forced" {
        tier.evict_all();
    }
    let t0 = Instant::now();
    for s in &steps {
        tier.begin_step();
        for l in 0..48 {
            let mut rs = Vec::new();
            for &(pi, lo, hi) in s {
                let (t, r) = &routes[pi];
                for k in lo..hi {
                    let o = (l * t + k) * 10;
                    rs.extend(r[o..o + 10].iter().map(|&e| e as u32));
                }
            }
            tier.resolve(l as u32, &rs, &mut mv).unwrap();
            tier.check_dispatch(l as u32, &rs).unwrap();
        }
        tier.end_step(&mut mv).unwrap();
    }
    let wall = t0.elapsed().as_secs_f64();
    let (paths, moves): (PathCounts, _) = tier.totals();
    let n = steps.len() as f64;
    let nv: f64 = mv.batches.iter().map(|b| b.1).sum();
    let nvn: usize = mv.batches.iter().map(|b| b.0).sum();
    let worst = mv
        .batches
        .iter()
        .fold((0, 0f64), |w, b| if b.1 > w.1 { *b } else { w });
    let pc = |x: u64| 100.0 * x as f64 / paths.routed().max(1) as f64;
    println!(
        "replay {mode}: {} steps | routed {} = resident {} ({:.1}%) ram {} ({:.1}%) nvme {} ({:.2}%) \
         prefetched {} | per step: resident {:.1} ram {:.1} nvme {:.2} | inserts {} ({:.2}/step) \
         demotions {} ram drops {} prefetch reads {} read-through layers {} | link/step h2d {:.1} MB \
         d2h {:.1} MB | NVMe {} records in {:.1} ms ({:.2} GB/s; worst batch {} rec {:.2} ms) | wall {:.2} s",
        steps.len(),
        paths.routed(),
        paths.resident,
        pc(paths.resident),
        paths.ram,
        pc(paths.ram),
        paths.nvme,
        pc(paths.nvme),
        paths.prefetched,
        paths.resident as f64 / n,
        paths.ram as f64 / n,
        paths.nvme as f64 / n,
        moves.inserts,
        moves.inserts as f64 / n,
        moves.demotions,
        moves.ram_drops,
        moves.prefetch_reads,
        moves.read_through_layers,
        moves.bytes_h2d as f64 / n / 1e6,
        moves.bytes_d2h as f64 / n / 1e6,
        nvn,
        nv * 1e3,
        (nvn as f64 * 2_617_344.0) / nv.max(1e-9) / 1e9,
        worst.0,
        worst.1 * 1e3,
        wall
    );
    let pl = tier.per_layer();
    let show: Vec<String> = [0usize, 1, 12, 24, 36, 47]
        .iter()
        .map(|&l| format!("l{l} {}/{}/{}", pl[l].resident, pl[l].ram, pl[l].nvme))
        .collect();
    println!(
        "replay {mode}: per layer resident/ram/nvme: {}",
        show.join("  ")
    );
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.first().map(String::as_str) {
        Some("xtier-qd") => xtier_qd(&a[1..]),
        Some("ple-qd") => ple_qd(&a[1..]),
        Some("ple-replay") => ple_replay(&a[1..]),
        Some("replay") => replay(&a[1..]),
        _ => eprintln!("usage: see the module docs"),
    }
}
