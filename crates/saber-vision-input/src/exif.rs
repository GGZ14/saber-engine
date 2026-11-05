//! EXIF orientation, as the reference server applies it: `normalize_image` = `ImageOps.exif_transpose(image)`
//! (errors suppressed) before `image.load()` and the mode conversion. Only tag 0x0112 of IFD0 is
//! read; PIL's XMP `tiff:Orientation` fallback (used only when the EXIF block has no orientation)
//! is NOT reproduced -- stated in VISION.md §8.

/// Orientation 1..=8 from a TIFF blob (with or without the `Exif\0\0` APP1 prefix); `None` for
/// anything unreadable, which is what PIL's suppressed error amounts to.
pub fn orientation(blob: &[u8]) -> Option<u16> {
    let tiff = blob.strip_prefix(b"Exif\0\0").unwrap_or(blob);
    if tiff.len() < 8 {
        return None;
    }
    let le = match &tiff[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |o: usize| -> Option<u16> {
        let b = tiff.get(o..o + 2)?;
        Some(if le {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        })
    };
    let u32_at = |o: usize| -> Option<u32> {
        let b = tiff.get(o..o + 4)?;
        let a = [b[0], b[1], b[2], b[3]];
        Some(if le {
            u32::from_le_bytes(a)
        } else {
            u32::from_be_bytes(a)
        })
    };
    if u16_at(2)? != 42 {
        return None;
    }
    let ifd = u32_at(4)? as usize;
    let n = u16_at(ifd)? as usize;
    for k in 0..n {
        let e = ifd + 2 + 12 * k;
        if u16_at(e)? == 0x0112 {
            let ty = u16_at(e + 2)?;
            let v = match ty {
                3 => u16_at(e + 8)?,
                4 => u32_at(e + 8)? as u16,
                _ => return None,
            };
            return (1..=8).contains(&v).then_some(v);
        }
    }
    None
}

/// Apply PIL's `exif_transpose` method for `orientation` to an interleaved image of `ch`
/// channels. Returns the new `(pixels, width, height)`.
pub fn transpose(
    px: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    orientation: u16,
) -> (Vec<u8>, usize, usize) {
    // (out_w, out_h, source coordinates of out (r, c))
    let swap = matches!(orientation, 5..=8);
    let (ow, oh) = if swap { (h, w) } else { (w, h) };
    if orientation <= 1 {
        return (px.to_vec(), w, h);
    }
    let mut out = vec![0u8; px.len()];
    for r in 0..oh {
        for c in 0..ow {
            let (sr, sc) = match orientation {
                2 => (r, w - 1 - c),         // FLIP_LEFT_RIGHT
                3 => (h - 1 - r, w - 1 - c), // ROTATE_180
                4 => (h - 1 - r, c),         // FLIP_TOP_BOTTOM
                5 => (c, r),                 // TRANSPOSE
                6 => (h - 1 - c, r),         // ROTATE_270 (90 clockwise)
                7 => (h - 1 - c, w - 1 - r), // TRANSVERSE
                8 => (c, w - 1 - r),         // ROTATE_90 (90 counter-clockwise)
                _ => (r, c),
            };
            let d = (r * ow + c) * ch;
            let s = (sr * w + sc) * ch;
            out[d..d + ch].copy_from_slice(&px[s..s + ch]);
        }
    }
    (out, ow, oh)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(le: bool, v: u16) -> Vec<u8> {
        let mut b = b"Exif\0\0".to_vec();
        if le {
            b.extend_from_slice(b"II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0");
            b.extend_from_slice(&u32::from(v).to_le_bytes());
        } else {
            b.extend_from_slice(b"MM\0*\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01");
            b.extend_from_slice(&v.to_be_bytes());
            b.extend_from_slice(&[0, 0]);
        }
        b.extend_from_slice(&[0, 0, 0, 0]);
        b
    }

    #[test]
    fn reads_both_byte_orders() {
        assert_eq!(orientation(&blob(true, 6)), Some(6));
        assert_eq!(orientation(&blob(false, 8)), Some(8));
        assert_eq!(orientation(&blob(true, 9)), None);
        assert_eq!(orientation(b"junk"), None);
    }

    #[test]
    fn eight_orientations_are_the_dihedral_group() {
        // 3 x 2 image, one channel: values 0..6 row-major
        let px: Vec<u8> = (0..6).collect();
        let t = |o| transpose(&px, 3, 2, 1, o);
        assert_eq!(t(2), (vec![2, 1, 0, 5, 4, 3], 3, 2));
        assert_eq!(t(3), (vec![5, 4, 3, 2, 1, 0], 3, 2));
        assert_eq!(t(4), (vec![3, 4, 5, 0, 1, 2], 3, 2));
        assert_eq!(t(5), (vec![0, 3, 1, 4, 2, 5], 2, 3));
        assert_eq!(t(6), (vec![3, 0, 4, 1, 5, 2], 2, 3));
        assert_eq!(t(7), (vec![5, 2, 4, 1, 3, 0], 2, 3));
        assert_eq!(t(8), (vec![2, 5, 1, 4, 0, 3], 2, 3));
    }
}
