//! FN60 phase 1: the host twin's throughput on THIS box, from real expert records in RAM.
//!
//!   cpux_bench tput XTIER --rows 1,8,24 --threads 1,6,12,20 --pool 384|4 --secs 3 [--first-cpu 0]
//!       each thread runs whole expert chains (gate_up -> silu-quant -> down) on experts drawn in
//!       turn from a pool of real records (384 = 1 GB = DRAM-cold; 4 = L3-warm); experts/ms,
//!       us/expert/thread, weight GB/s.
//!   cpux_bench lat XTIER --rows 1,2,4 --threads 1,2,4,8 --reps 200
//!       ONE expert's chain split across P threads by 16-column tiles (the b1 shape): latency.
//!   cpux_bench check XTIER DIR
//!       the twin vs a hold-1 chain dump (`chain_l*_T*_law*`): every H / q / sc / Y row byte for byte.
use saber_cpux::{
    expert_chain, gemm_tiles, silu_quant_row, ExpertScratch, MxGemm, SiluTable, HIDDEN, INTER,
};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

struct Xt {
    f: File,
    experts: u64,
    stride: u64,
    off: [u64; 6],
    data_off: u64,
}

fn open_xt(p: &str) -> Xt {
    let f = File::open(p).expect("xtier");
    let mut h = vec![0u8; 4096];
    f.read_exact_at(&mut h, 0).unwrap();
    assert_eq!(&h[0..8], b"SBRXPRT1");
    let u32a = |o: usize| u64::from(u32::from_le_bytes(h[o..o + 4].try_into().unwrap()));
    let u64a = |o: usize| u64::from_le_bytes(h[o..o + 8].try_into().unwrap());
    let mut off = [0u64; 6];
    for (i, o) in off.iter_mut().enumerate() {
        *o = u64a(48 + 16 * i);
    }
    Xt {
        experts: u32a(28),
        stride: u64a(32),
        off,
        data_off: u64a(168),
        f,
    }
}

impl Xt {
    fn record(&self, layer: u64, e: u64, dst: &mut [u8]) {
        self.f
            .read_exact_at(
                dst,
                self.data_off + (layer * self.experts + e) * self.stride,
            )
            .unwrap();
    }
    fn gemms<'a>(&self, r: &'a [u8]) -> (MxGemm<'a>, MxGemm<'a>) {
        let o = |i: usize| self.off[i] as usize;
        let gu = MxGemm {
            w: &r[o(0)..],
            ws: &r[o(1)..],
            wref: &r[o(2)..],
            n: 2 * INTER,
            k: HIDDEN,
        };
        let dn = MxGemm {
            w: &r[o(3)..],
            ws: &r[o(4)..],
            wref: &r[o(5)..],
            n: HIDDEN,
            k: INTER,
        };
        (gu, dn)
    }
}

fn arg_list(args: &[String], k: &str, d: &str) -> Vec<usize> {
    let v = args
        .iter()
        .position(|a| a == k)
        .map_or(d.to_string(), |i| args[i + 1].clone());
    v.split(',').map(|s| s.parse().unwrap()).collect()
}

fn pin(cpu: usize) {
    // SAFETY: a zeroed cpu_set_t is valid; CPU_SET on an index < CPU_SETSIZE.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

fn acts(rows: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let mut s = seed | 1;
    let a = (0..rows * HIDDEN)
        .map(|_| loop {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // realistic-ish: small exponents mostly
            let b = (s as u8) & 0xBF;
            if b & 0x7F != 0x7F {
                break b;
            }
        })
        .collect();
    (a, vec![0.01; rows])
}

fn load_pool(xt: &Xt, n: usize) -> Vec<u8> {
    let st = xt.stride as usize;
    let mut pool = vec![0u8; n * st];
    for i in 0..n {
        xt.record(
            10 + (i as u64 / xt.experts) % 30,
            i as u64 % xt.experts,
            &mut pool[i * st..(i + 1) * st],
        );
    }
    // SAFETY: mlock on our own allocation (failure is reported, not fatal: it mimics pinned RAM).
    let rc = unsafe { libc::mlock(pool.as_ptr().cast(), pool.len()) };
    eprintln!(
        "# pool {n} records ({:.2} GB), mlock rc {rc}",
        pool.len() as f64 / 1e9
    );
    pool
}

fn tput(args: &[String]) {
    let xt = Arc::new(open_xt(&args[2]));
    let rows_l = arg_list(args, "--rows", "1,8,24");
    let thr_l = arg_list(args, "--threads", "1,6,12,20");
    let pool_n = arg_list(args, "--pool", "384")[0];
    let secs = arg_list(args, "--secs", "3")[0];
    let first = arg_list(args, "--first-cpu", "0")[0];
    let pool = Arc::new(load_pool(&xt, pool_n));
    let silu = Arc::new(SiluTable::host());
    let st = xt.stride as usize;
    for &rows in &rows_l {
        for &thr in &thr_l {
            let next = Arc::new(AtomicUsize::new(0));
            let stop = Arc::new(AtomicBool::new(false));
            let go = Arc::new(Barrier::new(thr + 1));
            let mut hs = Vec::new();
            for t in 0..thr {
                let (xt, pool, silu, next, stop, go) = (
                    xt.clone(),
                    pool.clone(),
                    silu.clone(),
                    next.clone(),
                    stop.clone(),
                    go.clone(),
                );
                hs.push(std::thread::spawn(move || {
                    pin(first + t);
                    let (a, s) = acts(rows, 7 + t as u64);
                    let mut sc = ExpertScratch::new(rows);
                    let mut y = vec![0u16; rows * HIDDEN];
                    go.wait();
                    let mut done = 0usize;
                    while !stop.load(Ordering::Relaxed) {
                        let i = next.fetch_add(1, Ordering::Relaxed) % pool_n;
                        let r = &pool[i * st..(i + 1) * st];
                        let (gu, dn) = xt.gemms(r);
                        let (ksg, ksd) = if rows == 1 { (4, 2) } else { (1, 1) };
                        expert_chain(&gu, &dn, &a, &s, rows, ksg, ksd, &silu, &mut sc, &mut y);
                        done += 1;
                    }
                    std::hint::black_box(&y);
                    done
                }));
            }
            go.wait();
            let t0 = Instant::now();
            std::thread::sleep(Duration::from_secs(secs as u64));
            stop.store(true, Ordering::Relaxed);
            let n: usize = hs.into_iter().map(|h| h.join().unwrap()).sum();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let wall = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64()
                * 1e3;
            let epm = n as f64 / ms;
            println!(
                "TPUT pool {pool_n} rows {rows:2} threads {thr:2}: {epm:7.2} experts/ms  {:8.1} us/expert/thread  weights {:6.1} GB/s  ({n} chains in {ms:.0} ms) wall_ms {:.0}..{wall:.0}",
                thr as f64 * 1e3 / epm,
                epm * 1e3 * 2.615e6 / 1e9,
                wall - ms,
            );
        }
    }
}

fn lat(args: &[String]) {
    let xt = open_xt(&args[2]);
    let rows_l = arg_list(args, "--rows", "1,2,4");
    let thr_l = arg_list(args, "--threads", "1,2,4,8");
    let reps = arg_list(args, "--reps", "200")[0];
    let pool_n = 64usize;
    let pool = Arc::new(load_pool(&xt, pool_n));
    let st = xt.stride as usize;
    let silu = Arc::new(SiluTable::host());
    let xt = Arc::new(xt);
    for &rows in &rows_l {
        for &p in &thr_l {
            // P workers; each rep: tiles of gate_up split, barrier, silu (worker 0), barrier, down split.
            let bar = Arc::new(Barrier::new(p));
            let hbuf = Arc::new(SyncBuf(std::cell::UnsafeCell::new(vec![
                0u16;
                rows * 2 * INTER
            ])));
            let qbuf = Arc::new(SyncBuf(std::cell::UnsafeCell::new(vec![0u8; rows * INTER])));
            let sbuf = Arc::new(SyncBuf(std::cell::UnsafeCell::new(vec![0f32; rows])));
            let ybuf = Arc::new(SyncBuf(std::cell::UnsafeCell::new(vec![
                0u16;
                rows * HIDDEN
            ])));
            let mut hs = Vec::new();
            for w in 0..p {
                let (xt, pool, silu, bar) = (xt.clone(), pool.clone(), silu.clone(), bar.clone());
                let (hb, qb, sb, yb) = (hbuf.clone(), qbuf.clone(), sbuf.clone(), ybuf.clone());
                hs.push(std::thread::spawn(move || {
                    pin(w);
                    let (a, s) = acts(rows, 3);
                    let mut lats = Vec::new();
                    for rep in 0..reps {
                        let i = (rep * 7) % pool_n;
                        let r = &pool[i * st..(i + 1) * st];
                        let (gu, dn) = xt.gemms(r);
                        bar.wait();
                        let t0 = Instant::now();
                        let (ksg, ksd) = if rows == 1 { (4, 2) } else { (1, 1) };
                        let ntg = gu.n / 16;
                        // SAFETY: workers write disjoint 16-column tiles; barriers order the phases.
                        unsafe {
                            let h = &mut *hb.0.get();
                            gemm_tiles(&gu, &a, &s, rows, ksg, h, ntg * w / p, ntg * (w + 1) / p);
                        }
                        bar.wait();
                        if w == 0 {
                            // SAFETY: only worker 0 touches q/s between the barriers.
                            unsafe {
                                let h = &*hb.0.get();
                                let q = &mut *qb.0.get();
                                let sc = &mut *sb.0.get();
                                for rr in 0..rows {
                                    sc[rr] = silu_quant_row(
                                        &h[rr * 2 * INTER..(rr + 1) * 2 * INTER],
                                        &silu,
                                        &mut q[rr * INTER..(rr + 1) * INTER],
                                    );
                                }
                            }
                        }
                        bar.wait();
                        let ntd = dn.n / 16;
                        // SAFETY: disjoint tiles, as above.
                        unsafe {
                            let y = &mut *yb.0.get();
                            gemm_tiles(
                                &dn,
                                &*qb.0.get(),
                                &*sb.0.get(),
                                rows,
                                ksd,
                                y,
                                ntd * w / p,
                                ntd * (w + 1) / p,
                            );
                        }
                        bar.wait();
                        lats.push(t0.elapsed().as_secs_f64() * 1e6);
                    }
                    lats
                }));
            }
            let mut l = hs.into_iter().next().map(|h| h.join().unwrap()).unwrap();
            l.sort_by(f64::total_cmp);
            println!(
                "LAT rows {rows} threads {p}: one expert p50 {:7.1} us  p90 {:7.1} us",
                l[l.len() / 2],
                l[l.len() * 9 / 10]
            );
        }
    }
}

struct SyncBuf<T>(std::cell::UnsafeCell<T>);
// SAFETY: access is partitioned by tile and ordered by barriers (lat mode only).
unsafe impl<T> Sync for SyncBuf<T> {}

fn rd<T: Copy>(p: &str) -> Vec<T> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    let n = b.len() / std::mem::size_of::<T>();
    let mut v: Vec<T> = Vec::with_capacity(n);
    // SAFETY: plain-old-data reinterpretation of a byte dump of n T's.
    unsafe {
        std::ptr::copy_nonoverlapping(
            b.as_ptr(),
            v.as_mut_ptr().cast::<u8>(),
            n * std::mem::size_of::<T>(),
        );
        v.set_len(n);
    }
    v
}

fn check(args: &[String]) {
    let xt = open_xt(&args[2]);
    let dir = &args[3];
    let silu_card =
        SiluTable::from_bytes(&std::fs::read(format!("{dir}/silu_table.bin")).unwrap()).unwrap();
    let silu_host = SiluTable::host();
    let sd = silu_card
        .0
        .iter()
        .zip(&silu_host.0)
        .filter(|(a, b)| a != b)
        .count();
    println!("SILU host formula vs card table: {sd} of 65536 entries differ");
    let mut cases: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| {
            let n = e.unwrap().file_name().into_string().unwrap();
            n.starts_with("chain_").then_some(n)
        })
        .collect();
    cases.sort();
    let st = xt.stride as usize;
    let mut fails = 0;
    for c in &cases {
        let p = format!("{dir}/{c}");
        let layer: u64 = c.split('_').nth(1).unwrap()[1..].parse().unwrap();
        let ex: Vec<i32> = rd(&format!("{p}/experts.bin"));
        let a: Vec<u8> = rd(&format!("{p}/A.bin"));
        let asc: Vec<f32> = rd(&format!("{p}/As.bin"));
        let row_tok: Vec<i32> = rd(&format!("{p}/row_tok.bin"));
        let te: Vec<i32> = rd(&format!("{p}/te.bin"));
        let t0: Vec<i32> = rd(&format!("{p}/t0.bin"));
        let tr: Vec<i32> = rd(&format!("{p}/tr.bin"));
        let mut recs = vec![vec![0u8; st]; ex.len()];
        for (i, e) in ex.iter().enumerate() {
            xt.record(layer, *e as u64, &mut recs[i]);
        }
        for ks in [1usize, 2, 4] {
            let hc: Vec<u16> = rd(&format!("{p}/H_ks{ks}.bin"));
            let qc: Vec<u8> = rd(&format!("{p}/q_ks{ks}.bin"));
            let scc: Vec<f32> = rd(&format!("{p}/sc_ks{ks}.bin"));
            let yc: Vec<u16> = rd(&format!("{p}/Y_ks{ks}.bin"));
            let (mut bh, mut bq, mut bs, mut by, mut by2, mut rows_all) = (0, 0, 0, 0, 0, 0);
            let (mut el_y, mut el_all, mut el_h, mut el_hall, mut max_ulp, mut max_rel) =
                (0usize, 0usize, 0usize, 0usize, 0u32, 0f32);
            for t in 0..te.len() {
                let (e, r0, m) = (te[t] as usize, t0[t] as usize, tr[t] as usize);
                let (gu, dn) = xt.gemms(&recs[e]);
                let mut ar = vec![0u8; m * HIDDEN];
                let mut ars = vec![0f32; m];
                for i in 0..m {
                    let tok = row_tok[r0 + i] as usize;
                    ar[i * HIDDEN..(i + 1) * HIDDEN]
                        .copy_from_slice(&a[tok * HIDDEN..(tok + 1) * HIDDEN]);
                    ars[i] = asc[tok];
                }
                // stage by stage on the card's own inputs
                let mut h = vec![0u16; m * 2 * INTER];
                saber_cpux::gemm(&gu, &ar, &ars, m, ks, &mut h);
                for i in 0..m {
                    let hr = &hc[(r0 + i) * 2 * INTER..(r0 + i + 1) * 2 * INTER];
                    bh += usize::from(h[i * 2 * INTER..(i + 1) * 2 * INTER] != *hr);
                    let mut q = vec![0u8; INTER];
                    let s = silu_quant_row(hr, &silu_card, &mut q);
                    bq += usize::from(q[..] != qc[(r0 + i) * INTER..(r0 + i + 1) * INTER]);
                    bs += usize::from(s.to_bits() != scc[r0 + i].to_bits());
                }
                let mut y = vec![0u16; m * HIDDEN];
                saber_cpux::gemm(
                    &dn,
                    &qc[r0 * INTER..(r0 + m) * INTER],
                    &scc[r0..r0 + m],
                    m,
                    ks,
                    &mut y,
                );
                for i in 0..m {
                    by += usize::from(
                        y[i * HIDDEN..(i + 1) * HIDDEN]
                            != yc[(r0 + i) * HIDDEN..(r0 + i + 1) * HIDDEN],
                    );
                }
                for (j, (&u, &v)) in y
                    .iter()
                    .zip(&yc[r0 * HIDDEN..(r0 + m) * HIDDEN])
                    .enumerate()
                {
                    let _ = j;
                    if u != v {
                        el_y += 1;
                        let (fu, fv) = (saber_cpux::bf16_f32(u), saber_cpux::bf16_f32(v));
                        let d = (fu - fv).abs() / fv.abs().max(1e-30);
                        if d > max_rel {
                            max_rel = d;
                        }
                        let ulp = (i32::from(u as i16) - i32::from(v as i16)).unsigned_abs();
                        if ulp > max_ulp {
                            max_ulp = ulp;
                        }
                    }
                    el_all += 1;
                }
                for (&u, &v) in h.iter().zip(&hc[r0 * 2 * INTER..(r0 + m) * 2 * INTER]) {
                    el_h += usize::from(u != v);
                    el_hall += 1;
                }
                // end to end: the whole host chain from the card's A rows
                let mut sc = ExpertScratch::new(m);
                let mut y2 = vec![0u16; m * HIDDEN];
                expert_chain(&gu, &dn, &ar, &ars, m, ks, ks, &silu_card, &mut sc, &mut y2);
                for i in 0..m {
                    by2 += usize::from(
                        y2[i * HIDDEN..(i + 1) * HIDDEN]
                            != yc[(r0 + i) * HIDDEN..(r0 + i + 1) * HIDDEN],
                    );
                }
                rows_all += m;
            }
            let ok = bh + bq + bs + by + by2 == 0;
            fails += usize::from(!ok);
            println!(
                "CHECK {c} ks{ks}: rows {rows_all} differing H {bh} q {bq} sc {bs} Y(card q) {by} Y(end-to-end) {by2} {}",
                if ok { "BIT-EXACT" } else { "DIFFERS" }
            );
            println!(
                "ELEM {c} ks{ks}: H elements differing {el_h} of {el_hall}; Y(card q) elements differing {el_y} of {el_all}, max {max_ulp} bf16 ulp, max rel {max_rel:.2e}"
            );
        }
    }
    println!("CHECK cases differing: {fails} of {}", cases.len() * 3);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("tput") => tput(&args),
        Some("lat") => lat(&args),
        Some("check") => check(&args),
        _ => eprintln!("usage: cpux_bench tput|lat|check XTIER ..."),
    }
}
