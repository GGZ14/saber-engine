//! FN45c host bench (NO GPU): the drive under the KV tier's own I/O paths.
//!
//!   fn45c_ssd_bench <dir on the drive> [slab GiB = 4]
//!
//! 1. `KvSsd::write` of the whole slab in 4,096-block runs (park / write-through rate);
//! 2. `KvSsd::read` of q random 16 KiB blocks per batch, q = 1..512 (an unpark's scattered reads;
//!    and the price of SSD reads ON a decode step: 13 layers serial, q reads a layer);
//! 3. `PromptCache::store` + `load` of 8k / 32k / 100k-token entries (13,312 B a token a rank),
//!    the two ranks' loads concurrent from the one drive (the reload vs prefill cell).
//!
//! Lines `bench ...`; every file is deleted at the end.

use std::path::PathBuf;
use std::time::Instant;

use saber_mem::dio::{AlignedBuf, OpenMode};
use saber_mem::kv_ssd::{KvSsd, PromptCache};

const BLOCK: u64 = 16 * 1024;

fn main() {
    let mut a = std::env::args().skip(1);
    let dir = PathBuf::from(a.next().expect("dir"));
    let gib: u64 = a.next().map_or(4, |x| x.parse().expect("GiB"));
    std::fs::create_dir_all(&dir).expect("dir");
    let slots = (gib << 30) / BLOCK;
    let ssd =
        KvSsd::create(&dir.join("kv.ssd"), slots as u32, BLOCK, OpenMode::Direct).expect("slab");
    // 1. sequential write
    let run = 4096u32;
    let mut slab = AlignedBuf::zeroed((u64::from(run) * BLOCK) as usize);
    for (i, b) in slab.iter_mut().enumerate() {
        *b = (i * 31 % 251) as u8;
    }
    let t = Instant::now();
    let mut s = 0u32;
    while u64::from(s) < slots {
        let n = run.min((slots - u64::from(s)) as u32);
        let pairs: Vec<(u32, u32)> = (0..n).map(|i| (i, s + i)).collect();
        ssd.write(&slab, 0, &pairs).expect("write");
        s += n;
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "bench write: {:.2} GiB in {:.3} s = {:.2} GB/s (runs of {run} blocks, 8 threads)",
        gib as f64,
        dt,
        (slots * BLOCK) as f64 / dt / 1e9
    );
    // 2. random reads by batch size
    let mut rng = 0x1234_5678_9abc_def1u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut rbuf = AlignedBuf::zeroed((512 * BLOCK) as usize);
    for q in [1u32, 4, 7, 16, 32, 64, 128, 256, 512] {
        let reps = (4096 / q).clamp(16, 400);
        let mut lat = Vec::with_capacity(reps as usize);
        for _ in 0..reps {
            let mut ds: Vec<u32> = (0..q).map(|_| (next() % slots) as u32).collect();
            ds.sort_unstable();
            ds.dedup();
            // scattered drive slots, never adjacent to each other in RAM order -> q runs
            let pairs: Vec<(u32, u32)> =
                ds.iter().enumerate().map(|(i, &d)| (d, i as u32)).collect();
            let t = Instant::now();
            ssd.read(&mut rbuf, 0, &pairs).expect("read");
            lat.push(t.elapsed().as_secs_f64() * 1e6);
        }
        lat.sort_by(f64::total_cmp);
        let p50 = lat[lat.len() / 2];
        let p99 = lat[(lat.len() * 99 / 100).min(lat.len() - 1)];
        println!(
            "bench read q {q:4}: p50 {p50:8.1} us p99 {p99:8.1} us = {:.2} GB/s ({:.1} us a block) x{reps}",
            f64::from(q) * BLOCK as f64 / p50 / 1e3,
            p50 / f64::from(q)
        );
    }
    drop(ssd);
    let _ = std::fs::remove_file(dir.join("kv.ssd"));
    // 3. the prompt cache: store + concurrent two-rank load
    for tokens in [8_192u64, 32_768, 100_000] {
        let bytes = (tokens * 13 * 1024) as usize;
        let data: Vec<u8> = (0..bytes).map(|i| (i % 253) as u8).collect();
        let mut pcs: Vec<PromptCache> = (0..2)
            .map(|r| PromptCache::open(&dir.join(format!("pc{r}")), 8 << 30).expect("pc"))
            .collect();
        let key = [tokens as u8; 32];
        let t = Instant::now();
        for pc in &mut pcs {
            pc.store(key, tokens as u32, &[&data]).expect("store");
        }
        let st = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let ok = std::thread::scope(|sc| {
            let hs: Vec<_> = pcs
                .iter_mut()
                .map(|pc| sc.spawn(move || pc.load(&key).expect("load").len()))
                .collect();
            hs.into_iter()
                .map(|h| h.join().expect("join"))
                .all(|n| n == bytes)
        });
        let lt = t.elapsed().as_secs_f64();
        println!(
            "bench pcache {tokens:6} tokens: {:.2} GB a rank; store both ranks {:.0} ms; load both \
             ranks concurrent {:.0} ms = {:.2} GB/s; bytes ok {ok}",
            bytes as f64 / 1e9,
            st * 1e3,
            lt * 1e3,
            2.0 * bytes as f64 / lt / 1e9
        );
        drop(pcs);
        for r in 0..2 {
            let _ = std::fs::remove_dir_all(dir.join(format!("pc{r}")));
        }
    }
}
