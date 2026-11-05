//! FN60: the AVX-512 GEMM == the scalar law, byte for byte, on synthetic records spanning every
//! fold shift (d 0..15, the kMag subnormal rows included), every split count the card runs (1, 2, 4)
//! and both expert shapes; plus a negative control (a split law that differs in the way a wrong
//! split would) that must FAIL.

#![allow(clippy::needless_range_loop)]

use saber_cpux::{
    bf16_rne, e4m3_rne, e4m3_value, gemm, gemm_f64, gemm_ref, gemm_tiles, split_steps, MxGemm,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

struct Rec {
    w: Vec<u8>,
    ws: Vec<u8>,
    wref: Vec<u8>,
    n: usize,
    k: usize,
}

fn rec(n: usize, k: usize, seed: u64, dmax: u64) -> Rec {
    let mut r = Rng(seed);
    let w = (0..n * k / 2).map(|_| r.next() as u8).collect();
    let wref: Vec<u8> = (0..n).map(|_| 110 + (r.next() % 20) as u8).collect();
    let mut ws = vec![0u8; k / 32 * n];
    for b in 0..k / 32 {
        for i in 0..n {
            let d = if r.next().is_multiple_of(8) {
                r.next() % (dmax + 1)
            } else {
                r.next() % 3
            };
            ws[b * n + i] = wref[i].saturating_sub(d as u8);
        }
    }
    Rec { w, ws, wref, n, k }
}

fn acts(rows: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let mut r = Rng(seed);
    let a = (0..rows * k)
        .map(|_| loop {
            let b = r.next() as u8;
            if b & 0x7F != 0x7F {
                break b;
            }
        })
        .collect();
    let s = (0..rows).map(|i| 0.001 + i as f32 * 0.37e-3).collect();
    (a, s)
}

fn g(r: &Rec) -> MxGemm<'_> {
    MxGemm {
        w: &r.w,
        ws: &r.ws,
        wref: &r.wref,
        n: r.n,
        k: r.k,
    }
}

#[test]
fn avx512_gemm_equals_the_scalar_law() {
    if !saber_cpux::has_avx512() {
        eprintln!("no AVX-512 on this host: the dispatch IS the scalar law");
    }
    for (n, k, rows, seed) in [
        (64, 512, 3, 1u64),
        (32, 640, 5, 2),
        (16, 1024, 1, 3),
        (48, 256, 9, 4),
    ] {
        for dmax in [2u64, 8, 15] {
            let rc = rec(n, k, seed * 31 + dmax, dmax);
            let (a, s) = acts(rows, k, seed + 100);
            for ks in [1, 2, 4] {
                let mut want = vec![0u16; rows * n];
                let mut got = vec![0u16; rows * n];
                gemm_ref(&g(&rc), &a, &s, rows, ks, &mut want);
                gemm(&g(&rc), &a, &s, rows, ks, &mut got);
                assert_eq!(want, got, "n {n} k {k} rows {rows} dmax {dmax} ks {ks}");
                let mut f64b = vec![0u16; rows * n];
                gemm_f64(&g(&rc), &a, &s, rows, ks, &mut f64b);
                assert_eq!(
                    want, f64b,
                    "fp64 body, n {n} k {k} rows {rows} dmax {dmax} ks {ks}"
                );
                // the tile entry point over two halves == the whole
                let mut halves = vec![0u16; rows * n];
                let nt = n / 16;
                gemm_tiles(&g(&rc), &a, &s, rows, ks, &mut halves, 0, nt / 2);
                gemm_tiles(&g(&rc), &a, &s, rows, ks, &mut halves, nt / 2, nt);
                assert_eq!(want, halves);
            }
        }
    }
}

/// The negative control: a law with one fp32 rounding per 32 k (two WMMA steps) instead of per 16
/// must differ -- otherwise the gate above could not see a step-law change. Random rows almost
/// never show it at bf16 (0 of 2,048 outputs measured), so the control is a CRAFTED row: +X in step
/// 0, a 0.75-half-ulp term in each middle step (lost at one rounding a step, kept at one per two
/// steps), -X in the last step.
#[test]
fn a_wrong_step_law_is_caught() {
    const N: usize = 16;
    const K: usize = 2560;
    let w = vec![0x77u8; N * K / 2]; // every nibble 7 = +6.0
    let wref = vec![127u8; N]; // rf = 1
    let mut ws = vec![127u8; K / 32 * N];
    for b in 1..K / 32 - 1 {
        for i in 0..N {
            ws[b * N + i] = 127 - 8; // d = 8: w = 6 * 2^-8
        }
    }
    let rc = Rec {
        w,
        ws,
        wref,
        n: N,
        k: K,
    };
    let mut a = vec![0u8; K];
    for k in 0..16 {
        a[k] = 0x7E; // +448
        a[K - 16 + k] = 0xFE; // -448
    }
    for st in 2..K / 16 - 2 {
        a[st * 16] = 0x18; // 2^-4: one product 3 * 2^-11 = 0.75 half-ulp of 43,008
    }
    let s = [1.0f32];
    let mut want = vec![0u16; N];
    gemm(&g(&rc), &a, &s, 1, 1, &mut want);
    assert!(
        want.iter().all(|&v| v == 0),
        "the step law loses every middle term: {want:?}"
    );
    let gg = g(&rc);
    let mut acc = 0f32;
    for st in (0..K / 16).step_by(2) {
        let mut sum = f64::from(acc);
        for k in st * 16..st * 16 + 32 {
            let wb = saber_cpux::folded_byte(gg.nibble(0, k), gg.shift(0, k / 32));
            sum += e4m3_value(a[k]) * e4m3_value(wb);
        }
        acc = sum as f32;
    }
    let v = bf16_rne(acc * gg.rf(0) * s[0]);
    assert_ne!(
        v, want[0],
        "the control matched: the gate cannot see a step-law change"
    );
}

#[test]
fn split_steps_cover_k_once() {
    for k in [640usize, 2560] {
        for ks in [1, 2, 4] {
            let mut next = 0;
            for s in 0..ks {
                let (lo, hi) = split_steps(k, ks, s);
                assert_eq!(lo, next.min(lo).max(lo));
                assert!(lo <= hi);
                if hi > lo {
                    assert_eq!(lo, next);
                    next = hi;
                }
            }
            assert_eq!(next, k / 16);
        }
    }
}

#[test]
fn e4m3_rne_matches_the_nearest_code() {
    // every code round-trips; midpoints go to the even code
    for b in 0u8..=0x7E {
        let v = e4m3_value(b) as f32;
        assert_eq!(e4m3_rne(v), b);
        assert_eq!(e4m3_rne(-v), b | 0x80);
        if b < 0x7E {
            let hi = e4m3_value(b + 1) as f32;
            let mid = ((f64::from(v) + f64::from(hi)) / 2.0) as f32;
            let want = if b % 2 == 0 { b } else { b + 1 };
            assert_eq!(e4m3_rne(mid), want, "mid of {b:#x}");
        }
    }
}
