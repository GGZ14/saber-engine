//! The NVFP4 CSF scale codec of `nvidia/Qwen3.8-Flash-Next-FP4-Local` (FN144, the lossless path).
//!
//! The checkpoint stores each routed expert's E4M3 group-16 block scales (`[N, K/16]`, one byte per
//! 16 weights along K) COMPRESSED as `weight_scale.nvfp4_csf_fixed` (U8 `[N/16, 16 * (1 + C/2)]`,
//! `C = K/16`) + `weight_scale.nvfp4_csf_exceptions` (U32 `[E]`). The codec (reference:
//! the checkpoint publisher's reference reader (`b12x/_lib/quant/nvfp4_csf.py`), the checkpoint form read by
//! `scripts/convert_csf_blocks.py::_logical_source`; codec 0, layout 0):
//!
//! * the plane is cut into slabs of 16 rows; a slab is 16 per-row BASE bytes (each <= 240, so
//!   `base + 15` never carries) then the 16 rows' 4-bit codes, `C/2` bytes a row, LOW nibble first;
//! * a scale byte is `base[row] + code`;
//! * then every exception word overwrites one scale: `pos = w & 0xFF_FFFF` (`row * C + col`),
//!   `byte = w >> 24`.
//!
//! The codec is lossless: [`decode`] returns the plain E4M3 plane the CSF was cut from, byte for
//! byte. Its twin is `tools/pack/fp4local_csf.py` (numpy, a vectorized second implementation);
//! the byte law (LEDGER FN144) decodes every routed-expert plane both ways and compares sha256s.
//! [`encode`] (row base = the row minimum clamped to 240, every out-of-window scale an exception)
//! exists for the round-trip law over plain-scale NVFP4 planes; it is not NVIDIA's encoder (theirs
//! chooses bases to minimize exceptions), so only `decode(encode(x)) == x` is claimed.

use std::fmt;

/// Rows per slab.
pub const SLAB_ROWS: usize = 16;
/// The largest legal row base (`base + 15 <= 255`).
pub const MAX_BASE: u8 = 240;
/// Position bits of an exception word.
pub const POS_BITS: u32 = 24;

/// A decode variant. `Exact` is the codec; the others are the byte law's NEGATIVE CONTROLS and
/// must produce a different plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// The codec.
    Exact,
    /// Codes read HIGH nibble first (the wrong packing order).
    NibbleSwap,
    /// Exceptions ignored (the fixed stream alone).
    NoExceptions,
}

/// Why a CSF plane was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CsfError {
    /// The geometry does not fit the codec (rows % 16, odd columns, too many positions).
    Geometry(String),
    /// The fixed stream's length is not `rows/16 * 16 * (1 + cols/2)`.
    FixedLen {
        /// Expected bytes.
        want: usize,
        /// Bytes given.
        got: usize,
    },
    /// A row base above 240.
    Base {
        /// Row.
        row: usize,
        /// Its base.
        base: u8,
    },
    /// An exception outside the plane.
    Position {
        /// Exception ordinal.
        index: usize,
        /// Its position.
        pos: u32,
    },
}

impl fmt::Display for CsfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CsfError::Geometry(s) => write!(f, "csf geometry: {s}"),
            CsfError::FixedLen { want, got } => {
                write!(
                    f,
                    "csf fixed stream is {got} bytes, the geometry needs {want}"
                )
            }
            CsfError::Base { row, base } => write!(f, "csf row {row}: base {base} > 240"),
            CsfError::Position { index, pos } => {
                write!(f, "csf exception {index}: position {pos} outside the plane")
            }
        }
    }
}

impl std::error::Error for CsfError {}

/// A decoded plane and its census.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decoded {
    /// The plain E4M3 scales, row-major `[rows, cols]`.
    pub scales: Vec<u8>,
    /// Exception words applied.
    pub exceptions: usize,
    /// Exceptions that wrote the byte the fixed stream already gave (redundant; 0 expected).
    pub redundant: usize,
    /// Exceptions at a position an earlier exception already wrote (0 expected).
    pub repeated: usize,
    /// The largest row base.
    pub base_max: u8,
}

/// Bytes of the fixed stream of a `rows x cols` plane.
#[must_use]
pub fn fixed_len(rows: usize, cols: usize) -> usize {
    rows / SLAB_ROWS * SLAB_ROWS * (1 + cols / 2)
}

fn geometry(rows: usize, cols: usize) -> Result<(), CsfError> {
    if rows == 0 || rows % SLAB_ROWS != 0 || cols == 0 || cols % 2 != 0 {
        return Err(CsfError::Geometry(format!(
            "{rows} x {cols} (rows % 16 == 0 and even columns)"
        )));
    }
    if rows * cols > 1 << POS_BITS {
        return Err(CsfError::Geometry(format!(
            "{rows} x {cols} needs more than {POS_BITS} position bits"
        )));
    }
    Ok(())
}

/// Decode a CSF plane (`fixed`, `exceptions`) of `rows x cols` E4M3 scales.
pub fn decode(
    fixed: &[u8],
    exceptions: &[u32],
    rows: usize,
    cols: usize,
) -> Result<Decoded, CsfError> {
    decode_variant(fixed, exceptions, rows, cols, Variant::Exact)
}

/// [`decode`] under a [`Variant`] (the controls).
pub fn decode_variant(
    fixed: &[u8],
    exceptions: &[u32],
    rows: usize,
    cols: usize,
    variant: Variant,
) -> Result<Decoded, CsfError> {
    geometry(rows, cols)?;
    let want = fixed_len(rows, cols);
    if fixed.len() != want {
        return Err(CsfError::FixedLen {
            want,
            got: fixed.len(),
        });
    }
    let row_bytes = cols / 2;
    let slab_bytes = SLAB_ROWS * (1 + row_bytes);
    let mut scales = vec![0u8; rows * cols];
    let mut base_max = 0u8;
    for (s, slab) in fixed.chunks_exact(slab_bytes).enumerate() {
        let (bases, codes) = slab.split_at(SLAB_ROWS);
        for (rr, &base) in bases.iter().enumerate() {
            let row = s * SLAB_ROWS + rr;
            if base > MAX_BASE {
                return Err(CsfError::Base { row, base });
            }
            base_max = base_max.max(base);
            let src = &codes[rr * row_bytes..(rr + 1) * row_bytes];
            let dst = &mut scales[row * cols..(row + 1) * cols];
            for (j, &b) in src.iter().enumerate() {
                let (lo, hi) = match variant {
                    Variant::NibbleSwap => (b >> 4, b & 0xF),
                    _ => (b & 0xF, b >> 4),
                };
                dst[2 * j] = base + lo;
                dst[2 * j + 1] = base + hi;
            }
        }
    }
    let mut redundant = 0;
    let mut repeated = 0;
    let mut seen = vec![
        false;
        if variant == Variant::NoExceptions {
            0
        } else {
            rows * cols
        }
    ];
    let n = rows * cols;
    if variant != Variant::NoExceptions {
        for (i, &w) in exceptions.iter().enumerate() {
            let pos = w & ((1 << POS_BITS) - 1);
            if pos as usize >= n {
                return Err(CsfError::Position { index: i, pos });
            }
            let p = pos as usize;
            let v = (w >> POS_BITS) as u8;
            if seen[p] {
                repeated += 1;
            } else if scales[p] == v {
                redundant += 1;
            }
            seen[p] = true;
            scales[p] = v;
        }
    }
    Ok(Decoded {
        scales,
        exceptions: exceptions.len(),
        redundant,
        repeated,
        base_max,
    })
}

/// Encode a plain `[rows, cols]` E4M3 plane as CSF (row base = min clamped to 240; every scale
/// outside `[base, base + 15]` an exception, in position order). For the round-trip law only.
pub fn encode(scales: &[u8], rows: usize, cols: usize) -> Result<(Vec<u8>, Vec<u32>), CsfError> {
    geometry(rows, cols)?;
    if scales.len() != rows * cols {
        return Err(CsfError::Geometry(format!(
            "{} scales for {rows} x {cols}",
            scales.len()
        )));
    }
    let row_bytes = cols / 2;
    let mut fixed = vec![0u8; fixed_len(rows, cols)];
    let mut exc = Vec::new();
    let slab_bytes = SLAB_ROWS * (1 + row_bytes);
    for row in 0..rows {
        let r = &scales[row * cols..(row + 1) * cols];
        let base = (*r.iter().min().expect("cols > 0")).min(MAX_BASE);
        let (s, rr) = (row / SLAB_ROWS, row % SLAB_ROWS);
        let slab = &mut fixed[s * slab_bytes..(s + 1) * slab_bytes];
        slab[rr] = base;
        for (c, &v) in r.iter().enumerate() {
            let code = if v >= base && v - base <= 15 {
                v - base
            } else {
                exc.push(((v as u32) << POS_BITS) | (row * cols + c) as u32);
                0
            };
            let at = SLAB_ROWS + rr * row_bytes + c / 2;
            slab[at] |= if c % 2 == 0 { code } else { code << 4 };
        }
    }
    Ok((fixed, exc))
}

/// The value of an OCP E4M3 (fn) byte: bias 7, no infinities, `0x7F` / `0xFF` = NaN.
#[must_use]
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = i32::from((b >> 3) & 0xF);
    let m = f32::from(b & 7);
    if e == 15 && b & 7 == 7 {
        return f32::NAN;
    }
    if e == 0 {
        sign * m / 8.0 * 2f32.powi(-6)
    } else {
        sign * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        // A scale-like plane: a per-row level in [40, 120] plus small offsets, ~1 % outliers.
        let mut x = seed | 1;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut v = vec![0u8; rows * cols];
        for r in 0..rows {
            let level = 40 + (next() % 80) as u8;
            for c in 0..cols {
                let n = next();
                v[r * cols + c] = if n % 97 == 0 {
                    (n >> 8) as u8 & 0x7E
                } else {
                    level + (n >> 16) as u8 % 16
                };
            }
        }
        v
    }

    #[test]
    fn decode_inverts_encode_and_the_controls_differ() {
        for (rows, cols, seed) in [(640, 160, 1), (2560, 40, 2), (16, 2, 3)] {
            let p = plane(rows, cols, seed);
            let (fixed, exc) = encode(&p, rows, cols).unwrap();
            assert_eq!(fixed.len(), fixed_len(rows, cols));
            let d = decode(&fixed, &exc, rows, cols).unwrap();
            assert_eq!(d.scales, p);
            assert_eq!((d.redundant, d.repeated), (0, 0));
            assert!(d.base_max <= MAX_BASE);
            if rows > 16 {
                assert!(!exc.is_empty());
                for v in [Variant::NibbleSwap, Variant::NoExceptions] {
                    let c = decode_variant(&fixed, &exc, rows, cols, v).unwrap();
                    assert_ne!(c.scales, p, "{v:?} must differ");
                }
            }
        }
    }

    #[test]
    fn the_layout_is_the_reference_form() {
        // Two slabs of 16 rows x 4 columns, hand-built: row 17 base 7, codes 1,2,3,4 low first.
        let (rows, cols) = (32, 4);
        let mut fixed = vec![0u8; fixed_len(rows, cols)];
        let slab = 16 * 3;
        fixed[slab + 1] = 7;
        fixed[slab + 16 + 2] = 0x21;
        fixed[slab + 16 + 3] = 0x43;
        let exc = [(200u32 << 24) | (17 * 4 + 3)];
        let d = decode(&fixed, &exc, rows, cols).unwrap();
        assert_eq!(&d.scales[17 * 4..18 * 4], &[8, 9, 10, 200]);
        assert_eq!(d.exceptions, 1);
    }

    #[test]
    fn refusals_name_the_fault() {
        let (rows, cols) = (16, 4);
        let mut fixed = vec![0u8; fixed_len(rows, cols)];
        assert!(matches!(
            decode(&fixed[1..], &[], rows, cols),
            Err(CsfError::FixedLen { .. })
        ));
        assert!(matches!(
            decode(&fixed, &[64], rows, cols),
            Err(CsfError::Position { pos: 64, .. })
        ));
        fixed[3] = 241;
        assert!(matches!(
            decode(&fixed, &[], rows, cols),
            Err(CsfError::Base { row: 3, base: 241 })
        ));
        assert!(matches!(
            decode(&fixed, &[], 15, cols),
            Err(CsfError::Geometry(_))
        ));
    }

    #[test]
    fn e4m3_values() {
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9));
        assert_eq!(e4m3_to_f32(0xB8), -1.0);
        assert!(e4m3_to_f32(0x7F).is_nan());
    }
}
