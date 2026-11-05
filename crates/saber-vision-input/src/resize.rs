//! The resample prod runs: torchvision's `resize(uint8 tensor, BICUBIC, antialias=True)` on the
//! CPU, which (torchvision 0.24.1, `_do_native_uint8_resize_on_cpu(BICUBIC) == True`) hands the
//! **uint8** tensor straight to ATen's `_upsample_bicubic2d_aa` -- the PIL-compatible
//! fixed-point kernel (`aten/src/ATen/native/cpu/UpSampleKernel.cpp`, torch 2.11.0):
//!
//! * separable, **horizontal pass first** (into a uint8 intermediate), then vertical;
//!   a pass whose size is unchanged is skipped;
//! * per output index `i`: `center = scale * (i + 0.5)`, `support = 2 * scale` when downscaling
//!   (`scale >= 1`) else 2, taps `xmin = max(trunc(center - support + 0.5), 0)` ..
//!   `min(trunc(center + support + 0.5), in)`, weights = Keys cubic `a = -0.5` at
//!   `(j + xmin - center + 0.5) * invscale`, normalised by their f64 sum;
//! * weights quantised ONCE per pass to int16 at a precision chosen from the largest weight
//!   (`(int)(0.5 + wt_max * 2^(p+1)) < 2^15`, `p < 22`), rounded half away from zero;
//! * each output `clamp((2^(p-1) + sum(px * w)) >> p, 0, 255)` in int32 -- integer arithmetic, so
//!   the AVX kernel and the generic loop give the same bytes and so does this.
//!
//! The f64 weight arithmetic is the only place a compiler could contract into FMA; the
//! polynomial is evaluated here exactly as written (no `mul_add`), which the gate proves
//! byte-identical over the corpus (and an FMA arm is kept as a test-only control).

/// One pass's quantised taps: per output index, the first input index, the tap count and the
/// int16 weights (`taps` of them), plus the shared precision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Taps {
    pub xmin: Vec<usize>,
    pub xsize: Vec<usize>,
    /// `out_size * max_taps`, row-major.
    pub weights: Vec<i16>,
    pub max_taps: usize,
    pub precision: u32,
}

#[inline]
fn cubic_aa(x: f64, fma: bool) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if fma {
        if x < 1.0 {
            return ((A + 2.0).mul_add(x, -(A + 3.0)) * x).mul_add(x, 1.0);
        }
        if x < 2.0 {
            return (A.mul_add(x, -5.0 * A).mul_add(x, 8.0 * A)).mul_add(x, -4.0 * A);
        }
        return 0.0;
    }
    if x < 1.0 {
        return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    }
    if x < 2.0 {
        return ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    }
    0.0
}

/// ATen's `_compute_index_ranges_int16_weights` for bicubic, antialias = true,
/// align_corners = false, no explicit scale.
pub fn taps(input: usize, output: usize, fma: bool) -> Taps {
    let scale = input as f64 / output as f64;
    let interp_size = 4.0_f64;
    let support = if scale >= 1.0 {
        (interp_size * 0.5) * scale
    } else {
        interp_size * 0.5
    };
    let max_taps = (support.ceil() as usize) * 2 + 1;
    let mut wf = vec![0.0f64; output * max_taps];
    let mut xmin_v = Vec::with_capacity(output);
    let mut xsize_v = Vec::with_capacity(output);
    let mut wt_max = 0.0f64;
    for i in 0..output {
        let center = scale * (i as f64 + 0.5);
        let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
        // static_cast<int64_t> truncates toward zero.
        let xmin = ((center - support + 0.5) as i64).max(0);
        let xsize = ((center + support + 0.5) as i64).min(input as i64) - xmin;
        let xsize = xsize.clamp(0, max_taps as i64) as usize;
        let row = &mut wf[i * max_taps..(i + 1) * max_taps];
        let mut total = 0.0f64;
        for (j, w) in row.iter_mut().enumerate().take(xsize) {
            let v = cubic_aa(((j as i64 + xmin) as f64 - center + 0.5) * invscale, fma);
            *w = v;
            total += v;
        }
        let mut row_max = 0.0f64;
        if total != 0.0 {
            for w in row.iter_mut().take(xsize) {
                *w /= total;
                row_max = row_max.max(*w);
            }
        }
        wt_max = wt_max.max(row_max);
        xmin_v.push(xmin as usize);
        xsize_v.push(xsize);
    }
    let mut precision = 0u32;
    while precision < 22 {
        let next = (0.5 + wt_max * f64::from(1u32 << (precision + 1))) as i32;
        if next >= (1 << 15) {
            break;
        }
        precision += 1;
    }
    let scale_q = f64::from(1u32 << precision);
    let weights = wf
        .iter()
        .map(|&w| {
            let v = w * scale_q;
            (if v < 0.0 {
                (-0.5 + v) as i32
            } else {
                (0.5 + v) as i32
            }) as i16
        })
        .collect();
    Taps {
        xmin: xmin_v,
        xsize: xsize_v,
        weights,
        max_taps,
        precision,
    }
}

#[inline]
fn apply(taps: &Taps, o: usize, get: impl Fn(usize) -> u8) -> u8 {
    let w = &taps.weights[o * taps.max_taps..];
    let mut acc: i32 = 1 << (taps.precision - 1);
    for (j, &wj) in w.iter().enumerate().take(taps.xsize[o]) {
        acc += i32::from(get(taps.xmin[o] + j)) * i32::from(wj);
    }
    (acc >> taps.precision).clamp(0, 255) as u8
}

/// Resize one planar uint8 image `[c][h][w]` to `[c][oh][ow]`.
pub fn resize_planar(src: &[u8], c: usize, h: usize, w: usize, oh: usize, ow: usize) -> Vec<u8> {
    resize_planar_with(src, c, h, w, oh, ow, false)
}

/// [`resize_planar`] with the weight polynomial contracted to FMA (`fma = true`): the control.
pub fn resize_planar_with(
    src: &[u8],
    c: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    fma: bool,
) -> Vec<u8> {
    assert_eq!(src.len(), c * h * w);
    if oh == h && ow == w {
        return src.to_vec();
    }
    let horiz: Vec<u8> = if ow != w {
        let t = taps(w, ow, fma);
        let mut out = vec![0u8; c * h * ow];
        for plane in 0..c {
            for y in 0..h {
                let row = &src[(plane * h + y) * w..(plane * h + y + 1) * w];
                let dst = &mut out[(plane * h + y) * ow..(plane * h + y + 1) * ow];
                for (x, d) in dst.iter_mut().enumerate() {
                    *d = apply(&t, x, |i| row[i]);
                }
            }
        }
        out
    } else {
        src.to_vec()
    };
    if oh == h {
        return horiz;
    }
    let t = taps(h, oh, fma);
    let mut out = vec![0u8; c * oh * ow];
    for plane in 0..c {
        let p = &horiz[plane * h * ow..(plane + 1) * h * ow];
        for y in 0..oh {
            for x in 0..ow {
                out[(plane * oh + y) * ow + x] = apply(&t, y, |i| p[i * ow + x]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_one_in_fixed_point() {
        for (i, o) in [(512, 384), (230, 288), (4500, 4320), (997, 992), (7, 64)] {
            let t = taps(i, o, false);
            let one = 1i64 << t.precision;
            for k in 0..o {
                let s: i64 = t.weights[k * t.max_taps..k * t.max_taps + t.xsize[k]]
                    .iter()
                    .map(|&v| i64::from(v))
                    .sum();
                assert!(
                    (s - one).abs() <= t.xsize[k] as i64,
                    "{i}->{o} row {k}: {s} vs {one}"
                );
            }
        }
    }

    #[test]
    fn identity_size_is_a_copy_and_flat_stays_flat() {
        let src: Vec<u8> = (0..3 * 5 * 7).map(|v| v as u8).collect();
        assert_eq!(resize_planar(&src, 3, 5, 7, 5, 7), src);
        let flat = vec![77u8; 3 * 40 * 30];
        assert!(resize_planar(&flat, 3, 40, 30, 64, 32)
            .iter()
            .all(|&v| v == 77));
    }
}
