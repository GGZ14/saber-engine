//! FN60: the HOST twin of Flash-Next's routed-expert W4A8 chain, for experts whose bytes already
//! live in pinned RAM (the expert tier's L2): gate_up (`moe_grouped_w4a8_*_g1`) -> silu-quant
//! (`moe_silu_mul_quant_t256`) -> down (`moe_grouped_w4a8_*_g0`), byte for byte the card's rows.
//!
//! THE ARITHMETIC (what makes a twin possible). The card folds each e2m1 nibble and its block's
//! e8m0 exponent into an e4m3 byte (`kMag[d][i]`, d = clamp(wref - ws, 0, 15), subnormals down to
//! 2^-9) and runs `v_wmma_f32_16x16x16_fp8_fp8` over 16-k steps into an fp32 accumulator. Every
//! e4m3 operand is a multiple of 2^-9, so every product is a multiple of 2^-18 with <= 7
//! significant bits, and an fp32 accumulator built from such sums stays on the 2^-18 grid. The
//! HOST law here is `D = RNE_f32(C + sum_k a_k b_k)` computed EXACTLY (in fp64 every partial sum
//! is exact: grid 2^-18, magnitude < 2^35, < 53 bits), so any summation order inside a step gives
//! the same value. THE CARD IS NOT PROVEN TO DO THIS: FN60's probe (`tools/capture/fn60/twin.py
//! wmma`, LEDGER FN60) finds no host model of `v_wmma_f32_16x16x16_fp8_fp8` (best of ~70
//! candidates misses 1.1-20 % of words); on the real chain the law is byte-identical for every
//! realistic (amax-quantized) activation row and differs on ~1e-5 of gate_up bf16 elements
//! only for random full-range e4m3 rows. With that caveat, this law, one fp32 rounding a 16-k
//! step, reproduced the card. What must be mirrored is only: the step boundaries (16 k), the
//! split-K slabs (128 k, ceil(slabs / ks) a split, each split from 0), the split sum `0 + P0 + P1
//! ...` in fp32 (ks > 1; at ks 1 the accumulator directly), the epilogue `(acc * rf) * As` in
//! fp32 and the bf16 RNE store; then silu-quant's own fp32/bf16 rounding points.
//!
//! The weights are read in the xtier's WPERM fragment order, the layout the card reads: u32 word
//! `((n >> 4) * (K / 16) + kstep) * 32 + ((khalf << 4) | (n & 15))` holds k = kstep*16 + khalf*8
//! + j at nibble j (low nibble first).

#![allow(clippy::too_many_arguments)]

/// Flash-Next's hidden size (gate_up K, down N).
pub const HIDDEN: usize = 2560;
/// The routed expert's intermediate size (down K).
pub const INTER: usize = 640;

/// `mxfp4_common.h`'s `kMag`: the 8 e2m1 magnitudes shifted down by d binades as e4m3 bytes.
pub const KMAG: [[u32; 2]; 16] = [
    [0x3c38_3000, 0x4c48_4440],
    [0x3430_2800, 0x4440_3c38],
    [0x2c28_2000, 0x3c38_3430],
    [0x2420_1800, 0x3430_2c28],
    [0x1c18_1000, 0x2c28_2420],
    [0x1410_0800, 0x2420_1c18],
    [0x0c08_0400, 0x1c18_1410],
    [0x0604_0200, 0x1410_0c08],
    [0x0302_0100, 0x0c08_0604],
    [0x0101_0000, 0x0604_0302],
    [0x0100_0000, 0x0302_0101],
    [0x0000_0000, 0x0101_0100],
    [0x0000_0000, 0x0100_0000],
    [0, 0],
    [0, 0],
    [0, 0],
];

/// The e4m3 (OCP, finite) value of a byte; NaN codes 0x7F / 0xFF decode to NaN.
#[must_use]
pub fn e4m3_value(b: u8) -> f64 {
    let s = b >> 7;
    let e = i32::from((b >> 3) & 15);
    let m = f64::from(b & 7);
    if b & 0x7F == 0x7F {
        return f64::NAN;
    }
    let v = if e == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    if s == 1 {
        -v
    } else {
        v
    }
}

/// The folded weight byte the card stages for nibble `nib` of a block at shift `d` (0..=15).
#[must_use]
pub fn folded_byte(nib: u8, d: usize) -> u8 {
    let i = usize::from(nib & 7);
    let mag = ((KMAG[d][i >> 2] >> (8 * (i & 3))) & 0xFF) as u8;
    mag | ((nib & 8) << 4)
}

/// fp32 -> bf16 bits, round to nearest even (finite inputs; a NaN keeps a quiet NaN).
#[must_use]
pub fn bf16_rne(x: f32) -> u16 {
    let u = x.to_bits();
    if x.is_nan() {
        return ((u >> 16) | 0x40) as u16;
    }
    ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

/// bf16 bits -> fp32.
#[must_use]
pub fn bf16_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// fp32 (finite, |x| <= 448) -> e4m3 byte, round to nearest even, subnormals kept, the sign of a
/// zero kept (the raw `v_cvt_pk_fp8_f32` inside the producer's clamp).
#[must_use]
pub fn e4m3_rne(x: f32) -> u8 {
    let sign = ((x.to_bits() >> 31) as u8) << 7;
    let a = x.abs();
    // e4m3: subnormal step 2^-9 below 2^-6; normal binade e (unbiased -6..8) has step 2^(e-3).
    let q = if a < 2f32.powi(-6) {
        // multiples of 2^-9: m = round_even(a * 512)
        let v = a * 512.0; // exact (power of two scaling)
        let m = round_half_even(v);
        m as u8 // 0..8; 8 = 2^-6 = code 0x08, which is exactly m (subnormal m=8 == e=1,m=0)
    } else {
        let e = a.to_bits() >> 23; // biased fp32 exponent
        let ue = e as i32 - 127; // -6..=8
        let step = 2f32.powi(ue - 3);
        let m = round_half_even(a / step); // 8..=16 (exact division by a power of two)
        let (ue, m) = if m >= 16.0 { (ue + 1, 8.0) } else { (ue, m) };
        (((ue + 7) as u8) << 3) | ((m as u8) - 8)
    };
    q | sign
}

fn round_half_even(v: f32) -> f32 {
    let f = v.floor();
    let r = v - f;
    if r > 0.5 || (r == 0.5 && (f as i64) % 2 == 1) {
        f + 1.0
    } else {
        f
    }
}

/// One MXFP4 GEMM of an expert record: the WPERM nibbles, the e8m0 block scales `[K/32][N]` and the
/// per-row reference exponents `[N]`.
#[derive(Clone, Copy)]
pub struct MxGemm<'a> {
    pub w: &'a [u8],
    pub ws: &'a [u8],
    pub wref: &'a [u8],
    pub n: usize,
    pub k: usize,
}

impl MxGemm<'_> {
    /// The nibble of (n, k).
    #[must_use]
    pub fn nibble(&self, n: usize, k: usize) -> u8 {
        let kstep = k / 16;
        let lane = (((k % 16) / 8) << 4) | (n & 15);
        let widx = ((n >> 4) * (self.k / 16) + kstep) * 32 + lane;
        let word = u32::from_le_bytes(self.w[4 * widx..4 * widx + 4].try_into().unwrap());
        ((word >> (4 * (k % 8))) & 0xF) as u8
    }
    /// The block shift d for (n, block of 32 k).
    #[must_use]
    pub fn shift(&self, n: usize, blk: usize) -> usize {
        (i32::from(self.wref[n]) - i32::from(self.ws[blk * self.n + n])).clamp(0, 15) as usize
    }
    /// The epilogue's per-row factor rf = 2^(wref - 127) as the card builds it.
    #[must_use]
    pub fn rf(&self, n: usize) -> f32 {
        f32::from_bits(u32::from(self.wref[n]) << 23)
    }
    fn check(&self) {
        self.check_bk(SLAB);
    }
    fn check_bk(&self, bk: usize) {
        assert!(
            (bk == SLAB || bk == SLAB_TP2)
                && self.n.is_multiple_of(16)
                && self.k.is_multiple_of(bk),
            "N % 16, slabs of 128 (or 64) and K % slab"
        );
        assert!(self.w.len() >= self.n * self.k / 2);
        assert!(self.ws.len() >= self.n * self.k / 32);
        assert!(self.wref.len() >= self.n);
    }
}

/// The decode body's K slab (`moe_grouped`, DBK 128).
pub const SLAB: usize = 128;
/// FN80b-1: the TP=2 down half's slab (`moe_grouped_tp2`'s `bk64` cells and `moe_back_h2_*`'s
/// down: K 320 = 5 slabs of 64; 2.5 slabs of 128 would read past the row).
pub const SLAB_TP2: usize = 64;

/// The k range [lo, hi) (in 16-k steps) of split `s` of `ks` at K: slabs of 128 k, ceil(slabs/ks)
/// slabs a split, the last split short (or empty).
#[must_use]
pub fn split_steps(k: usize, ks: usize, s: usize) -> (usize, usize) {
    split_steps_bk(k, ks, s, SLAB)
}

/// [`split_steps`] at slabs of `bk` k (FN80b-1: the card's `slabs = K / DBK`, `spb = ceil(slabs /
/// ks)`, split s = slabs `[s * spb, min(slabs, s * spb + spb))` -- the decode body's partition at
/// any DBK).
#[must_use]
pub fn split_steps_bk(k: usize, ks: usize, s: usize, bk: usize) -> (usize, usize) {
    let slabs = k / bk;
    let spb = slabs.div_ceil(ks);
    let lo = (s * spb).min(slabs);
    let hi = (lo + spb).min(slabs);
    (lo * (bk / 16), hi * (bk / 16))
}

/// SCALAR reference of the grouped W4A8 GEMM rows (the law, written plainly): `a` = `rows` e4m3 rows
/// of K bytes, `a_scale` their per-row scales; `out` = rows x N bf16 bits.
pub fn gemm_ref(g: &MxGemm, a: &[u8], a_scale: &[f32], rows: usize, ks: usize, out: &mut [u16]) {
    gemm_ref_bk(g, a, a_scale, rows, ks, SLAB, out);
}

/// [`gemm_ref`] with the split-K slabs of `bk` k (FN80b-1's law at slabs of 64).
pub fn gemm_ref_bk(
    g: &MxGemm,
    a: &[u8],
    a_scale: &[f32],
    rows: usize,
    ks: usize,
    bk: usize,
    out: &mut [u16],
) {
    g.check_bk(bk);
    let (n_all, k_all) = (g.n, g.k);
    for m in 0..rows {
        let ar = &a[m * k_all..(m + 1) * k_all];
        for n in 0..n_all {
            let mut total = 0f32;
            let mut single = 0f32;
            for s in 0..ks {
                let (lo, hi) = split_steps_bk(k_all, ks, s, bk);
                let mut acc = 0f32;
                for st in lo..hi {
                    let mut sum = f64::from(acc);
                    for kk in 0..16 {
                        let k = st * 16 + kk;
                        let w = folded_byte(g.nibble(n, k), g.shift(n, k / 32));
                        sum += e4m3_value(ar[k]) * e4m3_value(w);
                    }
                    acc = sum as f32;
                }
                if ks == 1 {
                    single = acc;
                } else {
                    total += acc;
                }
            }
            let v = if ks == 1 { single } else { total };
            out[m * n_all + n] = bf16_rne(v * g.rf(n) * a_scale[m]);
        }
    }
}

/// The GEMM rows, AVX-512 when the host has it (bit-identical to [`gemm_ref`] by the argument in
/// the module doc; `tests/twin.rs` gates it), else the scalar law.
pub fn gemm(g: &MxGemm, a: &[u8], a_scale: &[f32], rows: usize, ks: usize, out: &mut [u16]) {
    gemm_bk(g, a, a_scale, rows, ks, SLAB, out);
}

/// [`gemm`] with the split-K slabs of `bk` k (FN80b-1: the TP=2 down half's twin at `bk` 64).
pub fn gemm_bk(
    g: &MxGemm,
    a: &[u8],
    a_scale: &[f32],
    rows: usize,
    ks: usize,
    bk: usize,
    out: &mut [u16],
) {
    g.check_bk(bk);
    assert!(a.len() >= rows * g.k && a_scale.len() >= rows && out.len() >= rows * g.n);
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            // SAFETY: the features were checked at run time; the slices were bounds-checked above.
            unsafe { avx512::gemm_i16(g, a, a_scale, rows, ks, out, 0, g.n / 16, bk) };
            return;
        }
    }
    gemm_ref_bk(g, a, a_scale, rows, ks, bk, out);
}

/// The GEMM rows for n-tiles [t0, t1) only (16 columns a tile): the worker pool's unit of work.
pub fn gemm_tiles(
    g: &MxGemm,
    a: &[u8],
    a_scale: &[f32],
    rows: usize,
    ks: usize,
    out: &mut [u16],
    t0: usize,
    t1: usize,
) {
    g.check();
    assert!(t1 <= g.n / 16 && t0 <= t1);
    assert!(a.len() >= rows * g.k && a_scale.len() >= rows && out.len() >= rows * g.n);
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            // SAFETY: as in `gemm`.
            unsafe { avx512::gemm_i16(g, a, a_scale, rows, ks, out, t0, t1, SLAB) };
            return;
        }
    }
    let _ = (t0, t1);
    gemm_ref(g, a, a_scale, rows, ks, out);
}

/// The fp64-FMA body of the same law (the integer body's control; bit-identical by construction).
pub fn gemm_f64(g: &MxGemm, a: &[u8], a_scale: &[f32], rows: usize, ks: usize, out: &mut [u16]) {
    g.check();
    assert!(a.len() >= rows * g.k && a_scale.len() >= rows && out.len() >= rows * g.n);
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            // SAFETY: as in `gemm`.
            unsafe { avx512::gemm(g, a, a_scale, rows, ks, out, 0, g.n / 16, SLAB) };
            return;
        }
    }
    gemm_ref(g, a, a_scale, rows, ks, out);
}

/// AVX-512 F/BW/DQ/VL + BF16 + VNNI present.
#[must_use]
pub fn has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512dq")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx512bf16")
            && std::arch::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The silu table: `moe_silu_mul_quant_impl`'s `(float)(__bf16)(x / (1.f + expf(-x)))` for every
/// bf16 input bit pattern, as bf16 bits (captured on the card, FN60 hold 1; `twin.py` proves the
/// host formula equal or names where it is not).
pub struct SiluTable(pub Vec<u16>);

impl SiluTable {
    /// From the 131,072-byte little-endian dump.
    #[must_use]
    pub fn from_bytes(b: &[u8]) -> Option<SiluTable> {
        (b.len() == 131_072).then(|| {
            SiluTable(
                b.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect(),
            )
        })
    }
    /// The host formula (fp32 expf from libm, fp32 divide, bf16 RNE) -- equal to the card's table
    /// only where `twin.py` says so.
    #[must_use]
    pub fn host() -> SiluTable {
        SiluTable(
            (0..=u16::MAX)
                .map(|u| {
                    let x = bf16_f32(u);
                    bf16_rne(x / (1.0 + (-x).exp()))
                })
                .collect(),
        )
    }
}

/// `moe_silu_mul_quant_impl` on one row: `gu` = 2 * inter bf16 (gate then up) -> `q` (inter e4m3
/// bytes) and the row scale.
pub fn silu_quant_row(gu: &[u16], silu: &SiluTable, q: &mut [u8]) -> f32 {
    let n = q.len();
    let mut p = [0f32; 1024];
    assert!(n <= 1024 && gu.len() >= 2 * n);
    let mut amax = 0f32;
    for j in 0..n {
        let t = bf16_f32(silu.0[usize::from(gu[j])]);
        let v = bf16_f32(bf16_rne(t * bf16_f32(gu[n + j])));
        p[j] = v;
        amax = amax.max(v.abs());
    }
    let sc = (amax * (1.0f32 / 448.0)).max(1.0f32 / (448.0 * 512.0));
    let rs = 1.0f32 / sc;
    for j in 0..n {
        q[j] = e4m3_rne((p[j] * rs).clamp(-448.0, 448.0));
    }
    sc
}

/// One routed expert's chain for `rows` token rows: gate_up -> silu-quant -> down. `a`/`a_scale` =
/// the block input's e4m3 rows (the card's gate_up A) for the expert's tokens; `y` = rows x HIDDEN
/// bf16 bits, the rows the card's combine reads. `scratch` holds the intermediate rows.
pub struct ExpertScratch {
    pub h: Vec<u16>,
    pub q: Vec<u8>,
    pub sc: Vec<f32>,
}

impl ExpertScratch {
    #[must_use]
    pub fn new(max_rows: usize) -> ExpertScratch {
        ExpertScratch {
            h: vec![0; max_rows * 2 * INTER],
            q: vec![0; max_rows * INTER],
            sc: vec![0.0; max_rows],
        }
    }
}

/// The chain, single-threaded.
pub fn expert_chain(
    gu: &MxGemm,
    dn: &MxGemm,
    a: &[u8],
    a_scale: &[f32],
    rows: usize,
    ks_gu: usize,
    ks_dn: usize,
    silu: &SiluTable,
    s: &mut ExpertScratch,
    y: &mut [u16],
) {
    gemm(gu, a, a_scale, rows, ks_gu, &mut s.h);
    for r in 0..rows {
        s.sc[r] = silu_quant_row(
            &s.h[r * 2 * INTER..(r + 1) * 2 * INTER],
            silu,
            &mut s.q[r * INTER..(r + 1) * INTER],
        );
    }
    gemm(dn, &s.q, &s.sc, rows, ks_dn, y);
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::MxGemm;
    use std::arch::x86_64::*;

    /// fp32 values of the 16 nibble codes at d = 0 (sign in bit 3).
    const NIB_F32: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    /// e4m3 byte -> f64 value, the 256-entry table.
    fn e4m3_table() -> [f64; 256] {
        let mut t = [0f64; 256];
        for (b, v) in t.iter_mut().enumerate() {
            let x = super::e4m3_value(b as u8);
            *v = if x.is_nan() { 0.0 } else { x };
        }
        t
    }

    const MAXR: usize = 64;

    /// n-tiles [t0, t1). Per tile and 16-k step: decode the 16 x 16 weights to fp64 once (the
    /// exponent of the d=0 value lowered by d binades -- exact for d <= 8, the kMag table for d >= 9),
    /// then per row an exact fp64 FMA sum seeded with the running accumulator, rounded once to fp32.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512bf16")]
    pub unsafe fn gemm(
        g: &MxGemm,
        a: &[u8],
        a_scale: &[f32],
        rows: usize,
        ks: usize,
        out: &mut [u16],
        t0: usize,
        t1: usize,
        bk: usize,
    ) {
        let tab = e4m3_table();
        let (n_all, k_all) = (g.n, g.k);
        let ksteps = k_all / 16;
        // The A rows as fp64 (exact), chunked by MAXR rows.
        let mut row0 = 0;
        while row0 < rows {
            let rr = (rows - row0).min(MAXR);
            let mut af = vec![0f64; rr * k_all];
            for r in 0..rr {
                for k in 0..k_all {
                    af[r * k_all + k] = tab[usize::from(a[(row0 + r) * k_all + k])];
                }
            }
            let nibtab = _mm512_loadu_ps(NIB_F32.as_ptr());
            let mut wd = [_mm512_setzero_pd(); 32]; // [k][half]
            let mut accs = vec![_mm512_setzero_pd(); rr * 2];
            let mut total = vec![_mm512_setzero_ps(); rr];
            let wp = g.w.as_ptr().cast::<u32>();
            for tile in t0..t1 {
                let n0 = tile * 16;
                for t in total.iter_mut() {
                    *t = _mm512_setzero_ps();
                }
                let wref = _mm512_cvtepu8_epi32(_mm_loadu_si128(g.wref.as_ptr().add(n0).cast()));
                for s in 0..ks {
                    let (lo, hi) = super::split_steps_bk(k_all, ks, s, bk);
                    for acc in accs.iter_mut() {
                        *acc = _mm512_setzero_pd();
                    }
                    let mut dsh = _mm512_setzero_si512();
                    let mut big = false;
                    let mut dv = _mm512_setzero_si512();
                    for st in lo..hi {
                        if st % 2 == 0 || st == lo {
                            let blk = st / 2;
                            let ws = _mm512_cvtepu8_epi32(_mm_loadu_si128(
                                g.ws.as_ptr().add(blk * n_all + n0).cast(),
                            ));
                            let d = _mm512_min_epi32(
                                _mm512_max_epi32(
                                    _mm512_sub_epi32(wref, ws),
                                    _mm512_setzero_si512(),
                                ),
                                _mm512_set1_epi32(15),
                            );
                            dv = d;
                            dsh = _mm512_slli_epi32::<23>(d);
                            big = _mm512_cmpgt_epi32_mask(d, _mm512_set1_epi32(8)) != 0;
                        }
                        let base = (tile * ksteps + st) * 32;
                        let wlo = _mm512_loadu_si512(wp.add(base).cast());
                        let whi = _mm512_loadu_si512(wp.add(base + 16).cast());
                        if !big {
                            for j in 0..8 {
                                for (h, wv) in [(0usize, wlo), (1usize, whi)] {
                                    let nib = _mm512_and_si512(
                                        _mm512_srlv_epi32(wv, _mm512_set1_epi32(4 * j as i32)),
                                        _mm512_set1_epi32(15),
                                    );
                                    let v = _mm512_permutexvar_ps(nib, nibtab);
                                    let nz = _mm512_test_epi32_mask(nib, _mm512_set1_epi32(7));
                                    let vi = _mm512_mask_sub_epi32(
                                        _mm512_castps_si512(v),
                                        nz,
                                        _mm512_castps_si512(v),
                                        dsh,
                                    );
                                    let vf = _mm512_castsi512_ps(vi);
                                    let k = h * 8 + j;
                                    wd[2 * k] = _mm512_cvtps_pd(_mm512_castps512_ps256(vf));
                                    wd[2 * k + 1] = _mm512_cvtps_pd(_mm256_castpd_ps(
                                        _mm512_extractf64x4_pd::<1>(_mm512_castps_pd(vf)),
                                    ));
                                }
                            }
                        } else {
                            // d >= 9 somewhere in this block: the kMag table, scalar.
                            let mut dl = [0i32; 16];
                            _mm512_storeu_si512(dl.as_mut_ptr().cast(), dv);
                            let mut wl = [0u32; 16];
                            let mut wh = [0u32; 16];
                            _mm512_storeu_si512(wl.as_mut_ptr().cast(), wlo);
                            _mm512_storeu_si512(wh.as_mut_ptr().cast(), whi);
                            for k in 0..16 {
                                let mut vals = [0f64; 16];
                                for (ln, v) in vals.iter_mut().enumerate() {
                                    let word = if k < 8 { wl[ln] } else { wh[ln] };
                                    let nib = ((word >> (4 * (k % 8))) & 15) as u8;
                                    *v = tab[usize::from(super::folded_byte(nib, dl[ln] as usize))];
                                }
                                wd[2 * k] = _mm512_loadu_pd(vals.as_ptr());
                                wd[2 * k + 1] = _mm512_loadu_pd(vals.as_ptr().add(8));
                            }
                        }
                        let kb = st * 16;
                        for r in 0..rr {
                            let ar = af.as_ptr().add(r * k_all + kb);
                            let mut e0 = accs[2 * r];
                            let mut e1 = accs[2 * r + 1];
                            let mut o0 = _mm512_setzero_pd();
                            let mut o1 = _mm512_setzero_pd();
                            let mut k = 0;
                            while k < 16 {
                                let av = _mm512_set1_pd(*ar.add(k));
                                e0 = _mm512_fmadd_pd(av, wd[2 * k], e0);
                                e1 = _mm512_fmadd_pd(av, wd[2 * k + 1], e1);
                                let bv = _mm512_set1_pd(*ar.add(k + 1));
                                o0 = _mm512_fmadd_pd(bv, wd[2 * k + 2], o0);
                                o1 = _mm512_fmadd_pd(bv, wd[2 * k + 3], o1);
                                k += 2;
                            }
                            // exact sums, one fp32 rounding per step (the WMMA law)
                            accs[2 * r] = _mm512_cvtps_pd(_mm512_cvtpd_ps(_mm512_add_pd(e0, o0)));
                            accs[2 * r + 1] =
                                _mm512_cvtps_pd(_mm512_cvtpd_ps(_mm512_add_pd(e1, o1)));
                        }
                    }
                    for r in 0..rr {
                        let lo8 = _mm512_cvtpd_ps(accs[2 * r]);
                        let hi8 = _mm512_cvtpd_ps(accs[2 * r + 1]);
                        let p = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(lo8), hi8);
                        total[r] = if ks == 1 {
                            p
                        } else {
                            _mm512_add_ps(total[r], p)
                        };
                    }
                }
                // epilogue: (acc * rf) * As -> bf16 RNE
                let rf = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(wref));
                for r in 0..rr {
                    let y = _mm512_mul_ps(
                        _mm512_mul_ps(total[r], rf),
                        _mm512_set1_ps(a_scale[row0 + r]),
                    );
                    // vcvtneps2bf16 treats a denormal input as zero; the card's cast keeps it.
                    let tiny = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(
                        _mm512_abs_ps(y),
                        _mm512_set1_ps(f32::MIN_POSITIVE),
                    ) & _mm512_cmp_ps_mask::<_CMP_NEQ_OQ>(y, _mm512_setzero_ps());
                    if tiny != 0 {
                        let mut yl = [0f32; 16];
                        _mm512_storeu_ps(yl.as_mut_ptr(), y);
                        for (i, v) in yl.iter().enumerate() {
                            out[(row0 + r) * n_all + n0 + i] = super::bf16_rne(*v);
                        }
                        continue;
                    }
                    let b = _mm512_cvtneps_pbh(y);
                    _mm256_storeu_si256(
                        out.as_mut_ptr().add((row0 + r) * n_all + n0).cast(),
                        std::mem::transmute::<__m256bh, __m256i>(b),
                    );
                }
            }
            row0 += rr;
        }
    }

    /// The same law in integer form (the DEFAULT body): every e4m3 activation is a_int = a * 2^9
    /// (|a_int| <= 229,376) split as ah * 512 + al (ah = floor(a_int / 512) in [-448, 448], al in
    /// [0, 512)); every folded weight is w_int = w * 2^9 = (2 * e2m1) * 2^(8 - d) (an int16 <= 3,072,
    /// exact for d <= 8; the kMag table for d >= 9). `vpdpwssd` sums 16 k of ah * w and al * w in
    /// int32 lanes (|sum| < 2^25: no wrap), then sum = acc + hi * 2^-9 + lo * 2^-18 in fp64 -- every
    /// term and partial sum exact -- and one fp32 rounding a step, as the fp64 body.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512bf16,avx512vnni")]
    pub unsafe fn gemm_i16(
        g: &MxGemm,
        a: &[u8],
        a_scale: &[f32],
        rows: usize,
        ks: usize,
        out: &mut [u16],
        t0: usize,
        t1: usize,
        bk: usize,
    ) {
        let tab = e4m3_table();
        let (n_all, k_all) = (g.n, g.k);
        let ksteps = k_all / 16;
        let kp = k_all / 2;
        // nibble -> w * 2^9 at d = 0, as int16 (vpermw takes the low 5 bits of each lane)
        let mut ntab = [0i16; 32];
        for (i, v) in ntab.iter_mut().enumerate().take(16) {
            let m = [0i16, 256, 512, 768, 1024, 1536, 2048, 3072][i & 7];
            *v = if i & 8 != 0 { -m } else { m };
        }
        let ntabv = _mm512_loadu_si512(ntab.as_ptr().cast());
        let mut row0 = 0;
        while row0 < rows {
            let rr = (rows - row0).min(MAXR);
            // the activation pairs (ah_k, ah_k+1) / (al_k, al_k+1) as int32 words
            let mut hp = vec![0i32; rr * kp];
            let mut lp = vec![0i32; rr * kp];
            for r in 0..rr {
                for k2 in 0..kp {
                    let mut h = [0i32; 2];
                    let mut l = [0i32; 2];
                    for j in 0..2 {
                        let ai =
                            (tab[usize::from(a[(row0 + r) * k_all + 2 * k2 + j])] * 512.0) as i32;
                        h[j] = ai >> 9;
                        l[j] = ai - (h[j] << 9);
                    }
                    hp[r * kp + k2] = (h[0] & 0xFFFF) | (h[1] << 16);
                    lp[r * kp + k2] = (l[0] & 0xFFFF) | (l[1] << 16);
                }
            }
            let mut accs = vec![_mm512_setzero_pd(); rr * 2];
            let mut total = vec![_mm512_setzero_ps(); rr];
            let mut wv = [_mm512_setzero_si512(); 8]; // the step's 8 k-pairs, int16 pairs per n lane
            let wp = g.w.as_ptr().cast::<u32>();
            let lo4 = _mm512_set1_epi32(0xF);
            let hi4 = _mm512_set1_epi32(0xF0);
            let c9 = _mm512_set1_pd(1.0 / 512.0);
            let c18 = _mm512_set1_pd(1.0 / 262_144.0);
            for tile in t0..t1 {
                let n0 = tile * 16;
                for t in total.iter_mut() {
                    *t = _mm512_setzero_ps();
                }
                let wref = _mm512_cvtepu8_epi32(_mm_loadu_si128(g.wref.as_ptr().add(n0).cast()));
                for s in 0..ks {
                    let (lo, hi) = super::split_steps_bk(k_all, ks, s, bk);
                    for acc in accs.iter_mut() {
                        *acc = _mm512_setzero_pd();
                    }
                    let mut d16 = _mm512_setzero_si512();
                    let mut dv = _mm512_setzero_si512();
                    let mut big = false;
                    for st in lo..hi {
                        if st % 2 == 0 || st == lo {
                            let blk = st / 2;
                            let ws = _mm512_cvtepu8_epi32(_mm_loadu_si128(
                                g.ws.as_ptr().add(blk * n_all + n0).cast(),
                            ));
                            let d = _mm512_min_epi32(
                                _mm512_max_epi32(
                                    _mm512_sub_epi32(wref, ws),
                                    _mm512_setzero_si512(),
                                ),
                                _mm512_set1_epi32(15),
                            );
                            dv = d;
                            d16 = _mm512_or_si512(d, _mm512_slli_epi32::<16>(d));
                            big = _mm512_cmpgt_epi32_mask(d, _mm512_set1_epi32(8)) != 0;
                        }
                        let base = (tile * ksteps + st) * 32;
                        let wlo = _mm512_loadu_si512(wp.add(base).cast());
                        let whi = _mm512_loadu_si512(wp.add(base + 16).cast());
                        if !big {
                            for p in 0..4 {
                                for (h, w) in [(0usize, wlo), (1usize, whi)] {
                                    let x = _mm512_srlv_epi32(w, _mm512_set1_epi32(8 * p as i32));
                                    let t = _mm512_or_si512(
                                        _mm512_and_si512(x, lo4),
                                        _mm512_slli_epi32::<12>(_mm512_and_si512(x, hi4)),
                                    );
                                    let v = _mm512_permutexvar_epi16(t, ntabv);
                                    wv[h * 4 + p] = _mm512_srav_epi16(v, d16);
                                }
                            }
                        } else {
                            let mut dl = [0i32; 16];
                            _mm512_storeu_si512(dl.as_mut_ptr().cast(), dv);
                            let mut wl = [0u32; 16];
                            let mut wh = [0u32; 16];
                            _mm512_storeu_si512(wl.as_mut_ptr().cast(), wlo);
                            _mm512_storeu_si512(wh.as_mut_ptr().cast(), whi);
                            for (q, slot) in wv.iter_mut().enumerate() {
                                let mut vals = [0i32; 16];
                                for (ln, v) in vals.iter_mut().enumerate() {
                                    let word = if q < 4 { wl[ln] } else { wh[ln] };
                                    let mut pair = [0i32; 2];
                                    for (j, pv) in pair.iter_mut().enumerate() {
                                        let kk = 2 * (q % 4) + j;
                                        let nib = ((word >> (4 * kk)) & 15) as u8;
                                        let fb = super::folded_byte(nib, dl[ln] as usize);
                                        *pv = (tab[usize::from(fb)] * 512.0) as i32;
                                    }
                                    *v = (pair[0] & 0xFFFF) | (pair[1] << 16);
                                }
                                *slot = _mm512_loadu_si512(vals.as_ptr().cast());
                            }
                        }
                        let kb2 = st * 8;
                        for r in 0..rr {
                            let hpr = hp.as_ptr().add(r * kp + kb2);
                            let lpr = lp.as_ptr().add(r * kp + kb2);
                            let mut sh0 = _mm512_setzero_si512();
                            let mut sl0 = _mm512_setzero_si512();
                            let mut sh1 = _mm512_setzero_si512();
                            let mut sl1 = _mm512_setzero_si512();
                            // k-pairs 0..3 are khalf 0 (k 0..7), 4..7 khalf 1 (k 8..15)
                            for q in 0..4 {
                                sh0 =
                                    _mm512_dpwssd_epi32(sh0, wv[q], _mm512_set1_epi32(*hpr.add(q)));
                                sl0 =
                                    _mm512_dpwssd_epi32(sl0, wv[q], _mm512_set1_epi32(*lpr.add(q)));
                                sh1 = _mm512_dpwssd_epi32(
                                    sh1,
                                    wv[4 + q],
                                    _mm512_set1_epi32(*hpr.add(4 + q)),
                                );
                                sl1 = _mm512_dpwssd_epi32(
                                    sl1,
                                    wv[4 + q],
                                    _mm512_set1_epi32(*lpr.add(4 + q)),
                                );
                            }
                            let sh = _mm512_add_epi32(sh0, sh1);
                            let sl = _mm512_add_epi32(sl0, sl1);
                            for half in 0..2 {
                                let (h8, l8) = if half == 0 {
                                    (_mm512_castsi512_si256(sh), _mm512_castsi512_si256(sl))
                                } else {
                                    (
                                        _mm512_extracti64x4_epi64::<1>(sh),
                                        _mm512_extracti64x4_epi64::<1>(sl),
                                    )
                                };
                                let mut t =
                                    _mm512_fmadd_pd(_mm512_cvtepi32_pd(h8), c9, accs[2 * r + half]);
                                t = _mm512_fmadd_pd(_mm512_cvtepi32_pd(l8), c18, t);
                                accs[2 * r + half] = _mm512_cvtps_pd(_mm512_cvtpd_ps(t));
                            }
                        }
                    }
                    for r in 0..rr {
                        let lo8 = _mm512_cvtpd_ps(accs[2 * r]);
                        let hi8 = _mm512_cvtpd_ps(accs[2 * r + 1]);
                        let p = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(lo8), hi8);
                        total[r] = if ks == 1 {
                            p
                        } else {
                            _mm512_add_ps(total[r], p)
                        };
                    }
                }
                let rf = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(wref));
                for r in 0..rr {
                    let y = _mm512_mul_ps(
                        _mm512_mul_ps(total[r], rf),
                        _mm512_set1_ps(a_scale[row0 + r]),
                    );
                    let tiny = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(
                        _mm512_abs_ps(y),
                        _mm512_set1_ps(f32::MIN_POSITIVE),
                    ) & _mm512_cmp_ps_mask::<_CMP_NEQ_OQ>(y, _mm512_setzero_ps());
                    if tiny != 0 {
                        let mut yl = [0f32; 16];
                        _mm512_storeu_ps(yl.as_mut_ptr(), y);
                        for (i, v) in yl.iter().enumerate() {
                            out[(row0 + r) * n_all + n0 + i] = super::bf16_rne(*v);
                        }
                        continue;
                    }
                    let b = _mm512_cvtneps_pbh(y);
                    _mm256_storeu_si256(
                        out.as_mut_ptr().add((row0 + r) * n_all + n0).cast(),
                        std::mem::transmute::<__m256bh, __m256i>(b),
                    );
                }
            }
            row0 += rr;
        }
    }
}
