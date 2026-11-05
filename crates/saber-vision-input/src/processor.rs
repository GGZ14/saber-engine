//! `Qwen2VLImageProcessor` (transformers 5.14.1, torchvision backend -- the one the reference server builds with
//! `use_fast=True`) on one RGB image: `smart_resize` -> bicubic antialias resample (uint8) ->
//! fused rescale + normalise in f32 -> the temporal duplicate -> patches flattened in prod's
//! order -> `pixel_values [L, 1536]` f32 and `image_grid_thw = [1, H/16, W/16]`.

use crate::resize::resize_planar;
use crate::{pyfmt, InputError, RgbImage};

/// The checkpoint's `preprocessor_config.json` (read in prod's image, 2026-09-23).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProcessorConfig {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    /// `size.shortest_edge` -- the pixel-count floor.
    pub min_pixels: u64,
    /// `size.longest_edge` -- the pixel-count ceiling.
    pub max_pixels: u64,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    /// `rescale_factor` = 1/255 as Python's double.
    pub rescale_factor: f64,
}

impl ProcessorConfig {
    /// Qwen3.8-27B's processor: 16-px patches, T = 2, 2x2 merge, 65,536..16,777,216 pixels,
    /// mean = std = 0.5.
    pub const QWEN38: Self = Self {
        patch_size: 16,
        temporal_patch_size: 2,
        merge_size: 2,
        min_pixels: 65_536,
        max_pixels: 16_777_216,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        rescale_factor: 1.0 / 255.0,
    };

    /// `patch_size * merge_size` = 32.
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// Row width of `pixel_values`: `3 * T * p * p` = 1536.
    pub fn row_len(&self) -> usize {
        3 * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// FN126: the pixel ceiling a tower arena of `arena_patches` patches holds: one patch is
    /// `patch_size^2` pixels of the resized image (T = 2 duplicates a frame, it adds no patches).
    pub fn arena_max_pixels(&self, arena_patches: u32) -> u64 {
        u64::from(arena_patches) * (self.patch_size * self.patch_size) as u64
    }

    /// FN126: this processor with its pixel ceiling (`max_pixels`) clamped to a tower arena of
    /// `arena_patches` patches -- `min(max_pixels, arena_patches x patch^2)`; every other field is
    /// prod's. [`smart_resize`] at the smaller ceiling is prod's own downscale (the same f64
    /// formula, the same floors), so an image whose prod resize already fits the arena resizes to
    /// the SAME target (and the same pixels), and a larger one is downscaled to at most
    /// `arena_patches` patches instead of being refused. `0` (a text-only plane) clamps nothing.
    #[must_use]
    pub fn clamped_to_arena(&self, arena_patches: u32) -> Self {
        if arena_patches == 0 {
            return *self;
        }
        Self {
            max_pixels: self.max_pixels.min(self.arena_max_pixels(arena_patches)),
            ..*self
        }
    }
}

/// transformers' `smart_resize`, in f64 with Python's semantics (`round` is round-half-even).
///
/// # Errors
/// An aspect ratio above 200, with transformers' own text.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: u64,
    max_pixels: u64,
) -> Result<(usize, usize), InputError> {
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    let ratio = h.max(w) / h.min(w);
    if ratio > 200.0 {
        return Err(InputError::AspectRatio(format!(
            "absolute aspect ratio must be smaller than 200, got {}",
            pyfmt::float_repr(ratio)
        )));
    }
    let mut h_bar = (h / f).round_ties_even() * f;
    let mut w_bar = (w / f).round_ties_even() * f;
    if h_bar * w_bar > max_pixels as f64 {
        let beta = ((h * w) / max_pixels as f64).sqrt();
        h_bar = f.max((h / beta / f).floor() * f);
        w_bar = f.max((w / beta / f).floor() * f);
    } else if h_bar * w_bar < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as usize, w_bar as usize))
}

/// One processed image.
#[derive(Debug, Clone, PartialEq)]
pub struct Processed {
    /// `[patches, row_len]` f32, row-major, prod's patch order.
    pub pixel_values: Vec<f32>,
    /// `image_grid_thw`.
    pub grid_thw: [u32; 3],
    /// The resize target `(H', W')`.
    pub resized_hw: (usize, usize),
    /// The resampled image, planar `[3][H'][W']` (kept for the gate's per-stage diagnosis).
    pub resized: Vec<u8>,
}

impl Processed {
    /// `L` = `(H'/p) * (W'/p)`.
    pub fn patches(&self) -> usize {
        (self.grid_thw[1] * self.grid_thw[2]) as usize
    }

    /// Image tokens = patches / merge^2.
    pub fn tokens(&self, cfg: &ProcessorConfig) -> usize {
        self.patches() / (cfg.merge_size * cfg.merge_size)
    }
}

/// The resize target and the patch count, without touching pixels (the reference server's
/// `get_number_of_image_patches`).
///
/// # Errors
/// As [`smart_resize`].
pub fn plan(
    height: usize,
    width: usize,
    cfg: &ProcessorConfig,
) -> Result<(usize, usize, usize), InputError> {
    let (rh, rw) = smart_resize(height, width, cfg.factor(), cfg.min_pixels, cfg.max_pixels)?;
    Ok((rh, rw, (rh / cfg.patch_size) * (rw / cfg.patch_size)))
}

/// The whole processor on one image.
///
/// # Errors
/// As [`smart_resize`].
pub fn process(img: &RgbImage, cfg: &ProcessorConfig) -> Result<Processed, InputError> {
    let (h, w) = (img.height, img.width);
    let (rh, rw) = smart_resize(h, w, cfg.factor(), cfg.min_pixels, cfg.max_pixels)?;
    // `pil_to_tensor`: HWC -> CHW.
    let mut planar = vec![0u8; 3 * h * w];
    for (i, px) in img.rgb.chunks_exact(3).enumerate() {
        for c in 0..3 {
            planar[c * h * w + i] = px[c];
        }
    }
    let resized = resize_planar(&planar, 3, h, w, rh, rw);
    // `_fuse_mean_std_and_rescale_factor`: torch.tensor(mean) [f32] * (1.0 / rescale) [python
    // float] -> f32; then `normalize(x.to(f32), mean, std)` = (x - mean) / std in f32.
    let inv = 1.0 / cfg.rescale_factor;
    let mean: [f32; 3] = std::array::from_fn(|c| (f64::from(cfg.image_mean[c]) * inv) as f32);
    let std: [f32; 3] = std::array::from_fn(|c| (f64::from(cfg.image_std[c]) * inv) as f32);
    let (p, m, t) = (cfg.patch_size, cfg.merge_size, cfg.temporal_patch_size);
    let (gh, gw) = (rh / p, rw / p);
    let row_len = cfg.row_len();
    let mut pv = vec![0f32; gh * gw * row_len];
    // reshape [C, gh/m, m, p, gw/m, m, p] -> permute to [gh/m, gw/m, m, m, C, p, p], then the
    // frame repeated T times inside each channel: row = C x T x p x p.
    let mut row = 0usize;
    for bh in 0..gh / m {
        for bw in 0..gw / m {
            for mh in 0..m {
                for mw in 0..m {
                    let base = row * row_len;
                    for c in 0..3 {
                        for tt in 0..t {
                            for py in 0..p {
                                let y = (bh * m + mh) * p + py;
                                for px in 0..p {
                                    let x = (bw * m + mw) * p + px;
                                    let v = f32::from(resized[(c * rh + y) * rw + x]);
                                    pv[base + ((c * t + tt) * p + py) * p + px] =
                                        (v - mean[c]) / std[c];
                                }
                            }
                        }
                    }
                    row += 1;
                }
            }
        }
    }
    Ok(Processed {
        pixel_values: pv,
        grid_thw: [1, gh as u32, gw as u32],
        resized_hw: (rh, rw),
        resized,
    })
}

/// torch's f32 -> bf16 (round to nearest, ties to even; NaN kept quiet) -- what the reference server hands the
/// tower, and what `vis_cast_f32_bf16` does on the card.
pub fn f32_to_bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    if v.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    let lsb = (b >> 16) & 1;
    ((b + 0x7fff + lsb) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_prods_integers() {
        let c = ProcessorConfig::QWEN38;
        let sr = |h, w| smart_resize(h, w, 32, c.min_pixels, c.max_pixels).unwrap();
        // the corpus' own (h, w) -> prod's resize_hw (the internal reference harness log)
        assert_eq!(sr(384, 512), (384, 512));
        assert_eq!(sr(1344, 1536), (1344, 1536));
        assert_eq!(sr(1000, 600), (992, 608));
        assert_eq!(sr(270, 230), (288, 256));
        assert_eq!(sr(4000, 4500), (3840, 4320));
        assert_eq!(sr(613, 997), (608, 992));
        assert_eq!(sr(427, 641), (416, 640));
        // Python's round is half-to-even: 48/32 = 1.5 -> 2, 80/32 = 2.5 -> 2
        assert_eq!(sr(48 * 10, 80 * 10).0, 480);
        assert_eq!(sr(80 * 20, 80 * 20).0, 1600);
    }

    /// FN126: the arena clamp is prod's smart-resize at a smaller ceiling. The integers are
    /// transformers' formula (Python `round` / `math.floor`, the same function the prod-pinned
    /// cases above check) evaluated at `max_pixels = arena x 256`; an 8 MP photo (3264 x 2448:
    /// 31,008 patches at prod's ceiling) lands at <= the arena's patches at every boot's arena,
    /// and an image prod already sizes inside the arena keeps prod's target.
    #[test]
    fn fn126_the_arena_clamp_is_prods_resize_at_a_smaller_ceiling() {
        let c = ProcessorConfig::QWEN38;
        assert_eq!(
            c.clamped_to_arena(65_536),
            c,
            "prod's own arena clamps nothing"
        );
        assert_eq!(c.clamped_to_arena(0), c, "a text-only plane clamps nothing");
        assert_eq!(c.clamped_to_arena(16_384).max_pixels, 4_194_304);
        let patches = |hw: (usize, usize)| (hw.0 / 16) * (hw.1 / 16);
        let at = |h, w, a: u32| {
            let k = c.clamped_to_arena(a);
            smart_resize(h, w, 32, k.min_pixels, k.max_pixels).unwrap()
        };
        // (h, w, arena) -> transformers' integers
        for (h, w, a, want) in [
            (2448, 3264, 65_536, (2432, 3264)),
            (2448, 3264, 16_384, (1760, 2336)),
            (2448, 3264, 12_288, (1536, 2048)),
            (2448, 3264, 8_192, (1248, 1664)),
            (2448, 3264, 4_096, (864, 1152)),
            (3456, 2304, 16_384, (2496, 1664)),
            (2160, 3840, 16_384, (1536, 2720)),
            (4000, 4500, 65_536, (3840, 4320)),
            (4000, 4500, 16_384, (1920, 2144)),
            (1920, 1080, 16_384, (1920, 1088)),
            (1920, 1080, 4_096, (1344, 768)),
            (704, 1280, 16_384, (704, 1280)),
            (768, 1024, 4_096, (768, 1024)),
        ] {
            let got = at(h, w, a);
            assert_eq!(got, want, "{h}x{w} at a {a}-patch arena");
            assert!(
                patches(got) <= a as usize,
                "{h}x{w}: {} > {a}",
                patches(got)
            );
        }
        assert_eq!(patches(at(2448, 3264, 65_536)), 31_008);
        assert_eq!(patches(at(2448, 3264, 16_384)), 16_060);
        // every aspect ratio prod admits (<= 200) fits every boot's arena after the clamp, and an
        // image whose prod target fits keeps that target exactly
        for a in [16_384u32, 12_288, 8_192, 4_096] {
            for h in (16..9000).step_by(97) {
                for w in [h / 3 + 1, h / 2 + 1, h, 2 * h, 7 * h, 150 * h] {
                    if w == 0 || (h.max(w) as f64 / h.min(w) as f64) > 200.0 {
                        continue;
                    }
                    let p = smart_resize(h, w, 32, c.min_pixels, c.max_pixels).unwrap();
                    let k = at(h, w, a);
                    assert!(patches(k) <= a as usize, "{h}x{w} at {a}: {k:?}");
                    if patches(p) <= a as usize {
                        assert_eq!(k, p, "{h}x{w} at {a}: a fitting image moved");
                    }
                }
            }
        }
    }

    /// FN126: an image that fits the arena is processed BYTE FOR BYTE as at prod's ceiling (the
    /// whole `Processed`: pixel_values, grid, target, the resampled image); one past it is
    /// downscaled to prod's integers and fits.
    #[test]
    fn fn126_a_fitting_image_is_byte_identical_under_the_clamp() {
        let c = ProcessorConfig::QWEN38;
        let k = c.clamped_to_arena(16_384);
        let img = |h: usize, w: usize| RgbImage {
            rgb: (0..h * w * 3)
                .map(|i| ((i * 2_654_435_761usize) >> 7) as u8)
                .collect(),
            width: w,
            height: h,
        };
        for (h, w) in [
            (704, 1280),
            (768, 1024),
            (613, 997),
            (1000, 600),
            (270, 230),
        ] {
            let im = img(h, w);
            assert_eq!(
                process(&im, &k).unwrap(),
                process(&im, &c).unwrap(),
                "{h}x{w}"
            );
        }
        let big = img(2448, 3264);
        let p = process(&big, &k).unwrap();
        assert_eq!(p.resized_hw, (1760, 2336));
        assert_eq!(p.grid_thw, [1, 110, 146]);
        assert_eq!(p.patches(), 16_060);
        assert_eq!(p.pixel_values.len(), 16_060 * 1536);
    }

    #[test]
    fn aspect_ratio_refusal_is_transformers_text() {
        let e = smart_resize(2, 402, 32, 65_536, 16_777_216).unwrap_err();
        assert_eq!(
            e.to_string(),
            "absolute aspect ratio must be smaller than 200, got 201.0"
        );
        assert!(smart_resize(2, 400, 32, 65_536, 16_777_216).is_ok());
    }

    #[test]
    fn bf16_rounding_is_rne() {
        assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f80_8000)), 0x3f80); // tie -> even
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f81_8000)), 0x3f82); // tie -> even
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f80_8001)), 0x3f81);
    }
}
