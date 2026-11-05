//! Bytes -> RGB, as the reference server's `ImageMediaIO(image_mode="RGB").load_bytes` does it with PIL 12.3.0:
//!
//! 1. `Image.open` -- the format is sniffed from the magic bytes (PNG / JPEG / WebP here), and
//!    an image over `MAX_IMAGE_PIXELS` (178,956,970, the reference server's default) is refused from the
//!    header with the reference server's text;
//! 2. `ImageOps.exif_transpose` (EXIF tag 0x0112 of the JPEG APP1 / PNG eXIf / WebP EXIF block);
//! 3. the mode conversion: RGB stays; L -> R=G=B; anything carrying transparency (RGBA, LA,
//!    P or L or RGB with a tRNS chunk) -> RGBA -> pasted over WHITE with the alpha as mask
//!    (`rgba_to_rgb`, PIL's `paste_mask_L` blend: `DIV255(bg*(255-a) + px*a)`).
//!
//! What is refused BY NAME rather than guessed (prod accepts these; the gap is stated):
//! 16-bit grayscale PNG (PIL's `I;16` -> RGB clip), CMYK / YCCK JPEG, and every other container
//! PIL can open (GIF, BMP, TIFF, ...).

use std::io::Cursor;

use crate::{exif, InputError, RgbImage};

/// The reference server's `MAX_IMAGE_PIXELS` default (`envs.py`, read in prod's image).
pub const MAX_IMAGE_PIXELS: u64 = 178_956_970;

/// The container, from the magic bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Png,
    Jpeg,
    WebP,
}

impl Format {
    pub fn sniff(b: &[u8]) -> Option<Self> {
        if b.starts_with(b"\x89PNG\r\n\x1a\n") {
            Some(Self::Png)
        } else if b.starts_with(&[0xff, 0xd8, 0xff]) {
            Some(Self::Jpeg)
        } else if b.len() >= 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WEBP" {
            Some(Self::WebP)
        } else {
            None
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::WebP => "WEBP",
        }
    }
}

/// PIL's "cannot identify" text after the reference server's wrapper and `sanitize_message`.
pub const UNIDENTIFIED: &str =
    "Failed to load image: cannot identify image file <_io.BytesIO object>";

fn failed(e: impl std::fmt::Display) -> InputError {
    InputError::Decode(format!("Failed to load image: {e}"))
}

fn check_pixels(w: usize, h: usize) -> Result<(), InputError> {
    let n = w as u64 * h as u64;
    if n > MAX_IMAGE_PIXELS {
        return Err(InputError::TooManyPixels(format!(
            "Image dimensions {w}x{h} ({n} pixels) exceed the maximum of {MAX_IMAGE_PIXELS} \
             pixels."
        )));
    }
    Ok(())
}

/// Interleaved 8-bit pixels of 1..=4 channels (L, LA, RGB, RGBA), before orientation.
struct Raw {
    px: Vec<u8>,
    w: usize,
    h: usize,
    ch: usize,
    exif: Option<Vec<u8>>,
    format: Format,
}

/// `rgba_to_rgb`'s blend: PIL `paste_mask_L` with `BLEND(mask, in1=bg, in2=px)` =
/// `DIV255(bg*(255-a) + px*a)`, `DIV255(v) = ((v+128) >> 8) + (v+128)) >> 8`.
#[inline]
fn over_white(px: u8, a: u8) -> u8 {
    let v = 255u32 * (255 - u32::from(a)) + u32::from(px) * u32::from(a) + 128;
    (((v >> 8) + v) >> 8) as u8
}

fn to_rgb(raw: Raw) -> RgbImage {
    let n = raw.w * raw.h;
    let mut rgb = Vec::with_capacity(n * 3);
    match raw.ch {
        1 => {
            for &l in &raw.px {
                rgb.extend_from_slice(&[l, l, l]);
            }
        }
        2 => {
            for p in raw.px.chunks_exact(2) {
                let v = over_white(p[0], p[1]);
                rgb.extend_from_slice(&[v, v, v]);
            }
        }
        3 => rgb = raw.px,
        _ => {
            for p in raw.px.chunks_exact(4) {
                rgb.extend_from_slice(&[
                    over_white(p[0], p[3]),
                    over_white(p[1], p[3]),
                    over_white(p[2], p[3]),
                ]);
            }
        }
    }
    RgbImage {
        rgb,
        width: raw.w,
        height: raw.h,
    }
}

fn png(bytes: &[u8]) -> Result<Raw, InputError> {
    let mut d = png::Decoder::new(Cursor::new(bytes));
    // EXPAND: palette -> RGB(A), tRNS -> alpha, gray < 8 bits scaled by bit replication -- the
    // same values PIL's unpackers give (L;1 -> 0/255, L;2 -> x85, L;4 -> x17).
    d.set_transformations(png::Transformations::EXPAND);
    let mut r = d.read_info().map_err(failed)?;
    let (w, h) = {
        let i = r.info();
        (i.width as usize, i.height as usize)
    };
    check_pixels(w, h)?;
    let mut buf = vec![
        0u8;
        r.output_buffer_size()
            .ok_or_else(|| failed("image too large"))?
    ];
    let frame = r.next_frame(&mut buf).map_err(failed)?;
    buf.truncate(frame.buffer_size());
    let ch = match frame.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return Err(failed("indexed PNG was not expanded")),
    };
    let px = match frame.bit_depth {
        png::BitDepth::Sixteen => {
            if ch <= 2 {
                return Err(InputError::Unsupported(
                    "16-bit grayscale PNG is not supported by Saber's image path (prod's PIL \
                     opens it as mode I;16 and clips to 8 bits); convert it to 8-bit"
                        .to_owned(),
                ));
            }
            // PIL opens 16-bit RGB(A) as rawmode RGB;16B / RGBA;16B: the high byte of each sample.
            buf.chunks_exact(2).map(|s| s[0]).collect()
        }
        _ => buf,
    };
    // `r.info()` after the frame: an eXIf chunk after IDAT is seen too.
    let exif = r.info().exif_metadata.as_ref().map(|c| c.to_vec());
    Ok(Raw {
        px,
        w,
        h,
        ch,
        exif,
        format: Format::Png,
    })
}

fn jpeg(bytes: &[u8]) -> Result<Raw, InputError> {
    use zune_core::bytestream::ZCursor;
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::DecoderOptions;
    let mut d = zune_jpeg::JpegDecoder::new(ZCursor::new(bytes));
    d.decode_headers().map_err(|e| failed(format!("{e:?}")))?;
    let (w, h) = d
        .dimensions()
        .ok_or_else(|| failed("JPEG without dimensions"))?;
    check_pixels(w, h)?;
    let (out, ch) = match d.input_colorspace() {
        Some(ColorSpace::Luma) => (ColorSpace::Luma, 1),
        Some(ColorSpace::YCbCr) | Some(ColorSpace::RGB) => (ColorSpace::RGB, 3),
        Some(cs) => {
            return Err(InputError::Unsupported(format!(
                "a {cs:?} JPEG is not supported by Saber's image path (prod's PIL opens CMYK/YCCK \
                 as mode CMYK); convert it to RGB"
            )))
        }
        None => return Err(failed("JPEG without a colorspace")),
    };
    let opts = DecoderOptions::default()
        .jpeg_set_out_colorspace(out)
        .set_max_width(1 << 20)
        .set_max_height(1 << 20);
    let mut d = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(bytes), opts);
    let px = d.decode().map_err(|e| failed(format!("{e:?}")))?;
    let exif = d.exif().cloned();
    Ok(Raw {
        px,
        w,
        h,
        ch,
        exif,
        format: Format::Jpeg,
    })
}

fn webp(bytes: &[u8]) -> Result<Raw, InputError> {
    let mut d = image_webp::WebPDecoder::new(Cursor::new(bytes)).map_err(failed)?;
    let (w, h) = d.dimensions();
    let (w, h) = (w as usize, h as usize);
    check_pixels(w, h)?;
    let ch = if d.has_alpha() { 4 } else { 3 };
    let mut px = vec![
        0u8;
        d.output_buffer_size()
            .ok_or_else(|| failed("image too large"))?
    ];
    d.read_image(&mut px).map_err(failed)?;
    let exif = d.exif_metadata().map_err(failed)?;
    Ok(Raw {
        px,
        w,
        h,
        ch,
        exif,
        format: Format::WebP,
    })
}

/// The whole of `load_bytes`: sniff, decode, orient, convert to RGB.
///
/// # Errors
/// The reference server's texts for an unidentifiable file and an oversized one; a decoder's own text (after
/// The reference server's "Failed to load image: " prefix) for a recognised but corrupt file; a named refusal for
/// the formats listed in the module docs.
pub fn decode(bytes: &[u8]) -> Result<(RgbImage, Format), InputError> {
    let format = Format::sniff(bytes).ok_or_else(|| {
        if bytes.starts_with(b"GIF8")
            || bytes.starts_with(b"BM")
            || bytes.starts_with(b"II*\0")
            || bytes.starts_with(b"MM\0*")
        {
            InputError::Unsupported(
                "this image container (GIF/BMP/TIFF) is not supported by Saber's image path: \
                 send PNG, JPEG or WebP"
                    .to_owned(),
            )
        } else {
            InputError::Decode(UNIDENTIFIED.to_owned())
        }
    })?;
    let raw = match format {
        Format::Png => png(bytes)?,
        Format::Jpeg => jpeg(bytes)?,
        Format::WebP => webp(bytes)?,
    };
    let fmt = raw.format;
    let orient = raw.exif.as_deref().and_then(exif::orientation).unwrap_or(1);
    let raw = if orient > 1 {
        let (px, w, h) = exif::transpose(&raw.px, raw.w, raw.h, raw.ch, orient);
        Raw { px, w, h, ..raw }
    } else {
        raw
    };
    Ok((to_rgb(raw), fmt))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_blend_endpoints() {
        assert_eq!(over_white(0, 0), 255);
        assert_eq!(over_white(0, 255), 0);
        assert_eq!(over_white(200, 255), 200);
        assert_eq!(over_white(0, 128), 127);
    }

    #[test]
    fn junk_is_prods_unidentified_text() {
        let e = decode(b"not an image at all").unwrap_err();
        assert_eq!(e.to_string(), UNIDENTIFIED);
        assert!(matches!(
            decode(b"GIF89a....").unwrap_err(),
            InputError::Unsupported(_)
        ));
    }
}
