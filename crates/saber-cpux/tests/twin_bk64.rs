//! FN80b-1: the twin's law at slabs of 64 (the TP=2 down half, K 320 = 5 slabs of 64; 2.5 slabs
//! of 128 would read past the row): the split partition is the card's at DBK 64, the AVX-512 body
//! == the scalar law at every split the card runs, at ks 1 the slab size is not in the answer (the
//! same 16-k steps in the same order), and the negative control: a split-class change moves bytes.

use saber_cpux::{gemm_bk, gemm_ref_bk, split_steps, split_steps_bk, MxGemm, SLAB, SLAB_TP2};

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
fn the_bk64_partition_is_the_cards() {
    // K 320 at DBK 64: 5 slabs; ks 2 -> spb 3: slabs [0, 3) and [3, 5) = steps [0, 12), [12, 20).
    assert_eq!(split_steps_bk(320, 1, 0, SLAB_TP2), (0, 20));
    assert_eq!(split_steps_bk(320, 2, 0, SLAB_TP2), (0, 12));
    assert_eq!(split_steps_bk(320, 2, 1, SLAB_TP2), (12, 20));
    for s in 0..4 {
        assert_eq!(split_steps_bk(640, 4, s, SLAB), split_steps(640, 4, s));
    }
    for k in [320usize, 640, 2560] {
        for ks in [1, 2, 4] {
            let mut next = 0;
            for s in 0..ks {
                let (lo, hi) = split_steps_bk(k, ks, s, SLAB_TP2);
                if hi > lo {
                    assert_eq!(lo, next);
                    next = hi;
                }
            }
            assert_eq!(next, k / 16, "k {k} ks {ks}");
        }
    }
}

#[test]
fn the_bk64_body_equals_the_scalar_law() {
    for (n, k, rows, seed) in [
        (2560usize, 320usize, 4usize, 7u64),
        (64, 320, 9, 8),
        (32, 640, 3, 9),
    ] {
        for dmax in [2u64, 15] {
            let rc = rec(n, k, seed * 17 + dmax, dmax);
            let (a, s) = acts(rows, k, seed + 300);
            for ks in [1, 2] {
                let mut want = vec![0u16; rows * n];
                let mut got = vec![0u16; rows * n];
                gemm_ref_bk(&g(&rc), &a, &s, rows, ks, SLAB_TP2, &mut want);
                gemm_bk(&g(&rc), &a, &s, rows, ks, SLAB_TP2, &mut got);
                assert_eq!(want, got, "n {n} k {k} rows {rows} dmax {dmax} ks {ks}");
                if k % SLAB == 0 && ks == 1 {
                    let mut b128 = vec![0u16; rows * n];
                    gemm_bk(&g(&rc), &a, &s, rows, ks, SLAB, &mut b128);
                    assert_eq!(want, b128, "ks 1: the slab size is not in the answer");
                }
            }
        }
    }
}

#[test]
fn a_split_class_change_moves_bytes() {
    // The control. Random rows almost never move (every product sits on the 2^-18 grid, so an
    // fp32 sum is exact until it is large -- FN26: a split change moved no byte on real routes),
    // so the row is CRAFTED: +X in step 0 (split 0), -X in step 12 (split 1's first step at
    // DBK 64), then small terms (below half an ulp of X) in steps 14..19. ks 1 cancels X before
    // the small terms arrive and keeps them; ks 2's second partial is -X + small (lost) = -X, and
    // P0 + P1 = 0. One accumulator vs two partials: the bytes must differ.
    const N: usize = 16;
    const K: usize = 320;
    let w = vec![0x77u8; N * K / 2]; // every nibble 7 = +6.0
    let wref = vec![127u8; N]; // rf = 1
    let mut ws = vec![127u8; K / 32 * N];
    for b in 7..10 {
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
    a[..16].fill(0x7E); // +448
    a[192..208].fill(0xFE); // -448
    a[224..].fill(0x01); // 2^-9: a step's term 16 * 2^-9 * 6 * 2^-8 < half an ulp of 43,008
    let s = [1.0f32];
    let (mut one, mut two) = (vec![0u16; N], vec![0u16; N]);
    gemm_bk(&g(&rc), &a, &s, 1, 1, SLAB_TP2, &mut one);
    gemm_bk(&g(&rc), &a, &s, 1, 2, SLAB_TP2, &mut two);
    assert_ne!(one[0], 0, "ks 1 keeps the small terms");
    assert_eq!(two[0], 0, "ks 2 loses them against -X");
    assert_ne!(one, two, "the split-class control moved nothing");
}
