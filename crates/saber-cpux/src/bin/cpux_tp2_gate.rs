//! FN80b-1: the HOST half of the TP=2 half-width cells' gates, over `fn80b1_bench --mode gate`'s
//! dumps and the REAL half records of the -tp2 pack.
//!
//!   cpux_tp2_gate DUMP_DIR PACK_DIR
//!
//! (a) THE TWIN: for every dumped (T, rank half, expert) -- 64 experts x T 1 / 4 / 8 / 64 x both
//!     halves -- the card's gate_up half rows (the committed bk128 cell at N 640) == the twin's law
//!     at slabs of 128, and each down cell's rows (`moe_grouped_w4a8_bk64_ks{1,2}_tm1_g0`, K 320)
//!     == the twin's law at slabs of 64 (`saber_cpux::gemm_bk(.., SLAB_TP2, ..)`), every bf16 byte.
//!     Control: the twin of the NEIGHBOUR expert's half record must differ.
//! (b) THE TP ORACLE: the whole EP expert's silu rows split in K through each half's down cell
//!     (Y0, Y1) summed in fp32 vs the whole expert's down output Yw (the committed bk128 cell),
//!     per element within the stated bound
//!       B = 1/2 ulp_bf16(Y0) + 1/2 ulp_bf16(Y1) + 1/2 ulp_bf16(Yw) + 2^-24 |Y0 + Y1|
//!           + (steps_w + steps_0 + steps_1 + 4) * 2^-24 * rf * As * S,   S = sum_k |a_k w_k|
//!     (three bf16 roundings, the fp32 add, one fp32 rounding a 16-k step on a running sum bounded
//!     by S, the two epilogue products) -- the summation-order bound; the control (Y1 taken from
//!     the neighbour expert's half) must exceed it.
use saber_cpux::{e4m3_value, folded_byte, gemm, gemm_bk, split_steps_bk, MxGemm, SLAB, SLAB_TP2};
use std::fs::File;
use std::os::unix::fs::FileExt;

const H: usize = 2560;
const IH: usize = 320;
const IW: usize = 640;

struct Xt {
    f: File,
    stride: u64,
    sub: [(u64, u64); 6],
    data_off: u64,
    experts: u64,
    first: u64,
}

fn open_xt(p: &str) -> Xt {
    let f = File::open(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    let mut h = vec![0u8; 4096];
    f.read_exact_at(&mut h, 0).unwrap();
    assert_eq!(&h[0..8], b"SBRXPRT1", "{p}");
    let u32a = |o: usize| u64::from(u32::from_le_bytes(h[o..o + 4].try_into().unwrap()));
    let u64a = |o: usize| u64::from_le_bytes(h[o..o + 8].try_into().unwrap());
    let mut sub = [(0u64, 0u64); 6];
    for (i, s) in sub.iter_mut().enumerate() {
        *s = (u64a(48 + 16 * i), u64a(56 + 16 * i));
    }
    Xt {
        stride: u64a(32),
        sub,
        data_off: u64a(168),
        experts: u32a(28),
        first: u32a(24),
        f,
    }
}

/// The six sub-buffers of (layer, GLOBAL expert e).
fn record(x: &Xt, layer: u64, e: u64) -> Vec<Vec<u8>> {
    assert!(e >= x.first && e < x.first + x.experts);
    let base = x.data_off + (layer * x.experts + (e - x.first)) * x.stride;
    x.sub
        .iter()
        .map(|&(o, l)| {
            let mut v = vec![0u8; l as usize];
            x.f.read_exact_at(&mut v, base + o).unwrap();
            v
        })
        .collect()
}

fn mx(r: &[Vec<u8>], down: bool, n: usize, k: usize) -> MxGemm<'_> {
    let b = if down { 3 } else { 0 };
    MxGemm {
        w: &r[b],
        ws: &r[b + 1],
        wref: &r[b + 2],
        n,
        k,
    }
}

fn rd(p: &str) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"))
}
fn u16s(b: &[u8]) -> Vec<u16> {
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect()
}
fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}
fn i32s(b: &[u8]) -> Vec<i32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect()
}
fn bf(b: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(b) << 16))
}
fn ulp_bf16(b: u16) -> f64 {
    let e = i32::from((b >> 7) & 0xFF);
    2f64.powi(e.max(1) - 127 - 7)
}

struct Oracle {
    elems: u64,
    over: u64,
    exact: u64,
    max_ratio: f64,
}

/// The TP oracle over one list of experts: whole (q, qs, y), halves' split outputs y0 / y1.
#[allow(clippy::too_many_arguments)]
fn tp_oracle(
    whole: &[Xt; 2],
    layer: u64,
    ex: &[i32],
    t: usize,
    rp: usize,
    q: &[u8],
    qs: &[f32],
    yw: &[u16],
    y0: &[u16],
    y1: &[u16],
    ks_w: usize,
    ks_h: usize,
    y1_shift: usize,
) -> Oracle {
    let mut o = Oracle {
        elems: 0,
        over: 0,
        exact: 0,
        max_ratio: 0.0,
    };
    let steps = |k: usize, ks: usize, bk: usize| -> usize {
        (0..ks)
            .map(|s| {
                let (lo, hi) = split_steps_bk(k, ks, s, bk);
                hi - lo
            })
            .max()
            .unwrap()
    };
    let nst = (steps(IW, ks_w, SLAB) + 2 * steps(IH, ks_h, SLAB_TP2) + 4) as f64;
    let e24 = 2f64.powi(-24);
    let ne = ex.len();
    // per expert: |w| table [n][k] at the whole's K (the halves' terms are subsets)
    let results: Vec<Oracle> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..ne)
            .map(|i| {
                sc.spawn(move || {
                    let e = ex[i] as u64;
                    let rec = record(&whole[(e / 256) as usize], layer, e);
                    let g = mx(&rec, true, H, IW);
                    let mut aw = vec![0f32; H * IW];
                    for n in 0..H {
                        for k in 0..IW {
                            let w = folded_byte(g.nibble(n, k), g.shift(n, k / 32));
                            aw[n * IW + k] = e4m3_value(w).abs() as f32;
                        }
                    }
                    let mut r = Oracle {
                        elems: 0,
                        over: 0,
                        exact: 0,
                        max_ratio: 0.0,
                    };
                    let j = (i + y1_shift) % ne; // the control reads the neighbour's half
                    for row in 0..t {
                        let ri = i * rp + row;
                        let rj = j * rp + row;
                        let a: Vec<f64> =
                            (0..IW).map(|k| e4m3_value(q[ri * IW + k]).abs()).collect();
                        for n in 0..H {
                            let s: f64 = (0..IW).map(|k| a[k] * f64::from(aw[n * IW + k])).sum();
                            let (b0, b1, bw) = (y0[ri * H + n], y1[rj * H + n], yw[ri * H + n]);
                            let sum = bf(b0) + bf(b1);
                            let d = (sum - bf(bw)).abs();
                            let scale = f64::from(g.rf(n)) * f64::from(qs[ri]);
                            let bound = 0.5 * (ulp_bf16(b0) + ulp_bf16(b1) + ulp_bf16(bw))
                                + e24 * sum.abs()
                                + nst * e24 * scale * s;
                            r.elems += 1;
                            if d > bound {
                                r.over += 1;
                            }
                            if d == 0.0 {
                                r.exact += 1;
                            }
                            r.max_ratio = r.max_ratio.max(d / bound);
                        }
                    }
                    r
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for r in results {
        o.elems += r.elems;
        o.over += r.over;
        o.exact += r.exact;
        o.max_ratio = o.max_ratio.max(r.max_ratio);
    }
    o
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: cpux_tp2_gate DUMP_DIR PACK_DIR");
        std::process::exit(2);
    }
    let (dir, pack) = (&args[1], &args[2]);
    let half = [
        open_xt(&format!("{pack}/rank0/experts.tp2.xtier")),
        open_xt(&format!("{pack}/rank1/experts.tp2.xtier")),
    ];
    let whole = [
        open_xt(&format!("{pack}/rank0/experts.xtier")),
        open_xt(&format!("{pack}/rank1/experts.xtier")),
    ];
    println!(
        "# avx512 {} | twin law: one fp32 rounding a 16-k step, gate_up slabs of {SLAB}, down half slabs of {SLAB_TP2}",
        saber_cpux::has_avx512()
    );
    let mut fails = 0usize;
    let (mut rows_gu, mut rows_dn, mut ctl_caught, mut ctl_rows) = (0usize, 0usize, 0usize, 0usize);
    for t in [1usize, 4, 8, 64] {
        let mut lists: Vec<Vec<i32>> = Vec::new();
        for (r, half_r) in half.iter().enumerate() {
            let base = format!("{dir}/gate_t{t}_r{r}");
            let meta = i32s(&rd(&format!("{base}.meta")));
            let (layer, nexp, ks_gu, rp) = (
                meta[2] as u64,
                meta[3] as usize,
                meta[4] as usize,
                meta[5] as usize,
            );
            assert_eq!((meta[0] as usize, meta[1] as usize), (t, r));
            let ex = meta[8..8 + nexp].to_vec();
            lists.push(ex.clone());
            let a0 = rd(&format!("{base}.a0"));
            let as0 = f32s(&rd(&format!("{base}.as0")));
            let gu = u16s(&rd(&format!("{base}.gu")));
            let q = rd(&format!("{base}.q"));
            let qs = f32s(&rd(&format!("{base}.qs")));
            let ys = [
                u16s(&rd(&format!("{base}.y1"))),
                u16s(&rd(&format!("{base}.y2"))),
            ];
            let recs: Vec<Vec<Vec<u8>>> = ex
                .iter()
                .map(|&e| record(half_r, layer, e as u64))
                .collect();
            let (mut dgu, mut ddn) = (0usize, [0usize; 2]);
            let mut ctl = 0usize;
            for (i, rec) in recs.iter().enumerate() {
                // gate_up half: rows = the T tokens (gathered: row r reads token r)
                let mut want = vec![0u16; t * 2 * IH];
                gemm(&mx(rec, false, 2 * IH, H), &a0, &as0, t, ks_gu, &mut want);
                for row in 0..t {
                    let ri = i * rp + row;
                    dgu += usize::from(
                        want[row * 2 * IH..(row + 1) * 2 * IH]
                            != gu[ri * 2 * IH..(ri + 1) * 2 * IH],
                    );
                }
                let qa = &q[i * rp * IH..(i * rp + t) * IH];
                let qsa = &qs[i * rp..i * rp + t];
                for (ki, ks) in [1usize, 2].into_iter().enumerate() {
                    let mut y = vec![0u16; t * H];
                    gemm_bk(&mx(rec, true, H, IH), qa, qsa, t, ks, SLAB_TP2, &mut y);
                    for row in 0..t {
                        let ri = i * rp + row;
                        ddn[ki] +=
                            usize::from(y[row * H..(row + 1) * H] != ys[ki][ri * H..(ri + 1) * H]);
                    }
                }
                // the control: the neighbour expert's down half on these rows must differ
                let nb = &recs[(i + 1) % recs.len()];
                let mut y = vec![0u16; t * H];
                gemm_bk(&mx(nb, true, H, IH), qa, qsa, t, 1, SLAB_TP2, &mut y);
                for row in 0..t {
                    let ri = i * rp + row;
                    ctl += usize::from(y[row * H..(row + 1) * H] != ys[0][ri * H..(ri + 1) * H]);
                }
            }
            let n = nexp * t;
            println!(
                "  TWIN T={t} rank {r} ({nexp} experts): gate_up half ks{ks_gu} {dgu}/{n} rows differ | down bk64 ks1 {}/{n} | ks2 {}/{n} -> {}",
                ddn[0],
                ddn[1],
                if dgu + ddn[0] + ddn[1] == 0 { "BIT-EXACT" } else { "FAIL" }
            );
            println!(
                "  TWIN CONTROL T={t} rank {r}: the neighbour's half record moves {ctl}/{n} rows -> {}",
                if ctl > 0 { "caught" } else { "CONTROL FAILED" }
            );
            fails += usize::from(dgu + ddn[0] + ddn[1] > 0) + usize::from(ctl == 0);
            rows_gu += n;
            rows_dn += 2 * n;
            ctl_caught += ctl;
            ctl_rows += n;
        }
        // (b) the TP oracle, both lists, at the whole's cell split (and the other)
        let ks_w = if t == 1 { 2 } else { 1 };
        for (li, ex) in lists.iter().enumerate() {
            let (wname, h0, h1) = if li == 0 {
                (
                    format!("{dir}/gate_t{t}_whole"),
                    format!("{dir}/gate_t{t}_r0"),
                    format!("{dir}/gate_t{t}_l0_h1"),
                )
            } else {
                (
                    format!("{dir}/gate_t{t}_l1_whole"),
                    format!("{dir}/gate_t{t}_l1_h0"),
                    format!("{dir}/gate_t{t}_l1_h1"),
                )
            };
            let q = rd(&format!("{wname}.q"));
            let qs = f32s(&rd(&format!("{wname}.qs")));
            let yw = u16s(&rd(&format!("{wname}.y")));
            let rp = t.div_ceil(16) * 16;
            for ks_h in [1usize, 2] {
                let y0 = u16s(&rd(&format!("{h0}.ysplit{ks_h}")));
                let y1 = u16s(&rd(&format!("{h1}.ysplit{ks_h}")));
                let o = tp_oracle(&whole, 0, ex, t, rp, &q, &qs, &yw, &y0, &y1, ks_w, ks_h, 0);
                println!(
                    "  TP ORACLE T={t} list {li} halves ks{ks_h} vs whole ks{ks_w}: {} elements, {} over the bound, max |d|/B {:.3}, {:.2} % exactly equal -> {}",
                    o.elems,
                    o.over,
                    o.max_ratio,
                    100.0 * o.exact as f64 / o.elems as f64,
                    if o.over == 0 { "IN BOUND" } else { "FAIL" }
                );
                fails += usize::from(o.over > 0);
                if ks_h == 1 {
                    let c = tp_oracle(&whole, 0, ex, t, rp, &q, &qs, &yw, &y0, &y1, ks_w, ks_h, 1);
                    println!(
                        "  TP CONTROL T={t} list {li}: Y1 from the neighbour expert's half: {}/{} over the bound, max |d|/B {:.1} -> {}",
                        c.over,
                        c.elems,
                        c.max_ratio,
                        if c.over > 0 { "caught" } else { "CONTROL FAILED" }
                    );
                    fails += usize::from(c.over == 0);
                }
            }
        }
    }
    println!(
        "TWIN GATE {} ({fails} failure(s); gate_up half rows {rows_gu}, down half rows {rows_dn}, control moved {ctl_caught}/{ctl_rows})",
        if fails == 0 { "PASSED" } else { "FAILED" }
    );
    std::process::exit(i32::from(fails > 0));
}
