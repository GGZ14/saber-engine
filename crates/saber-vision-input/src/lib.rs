//! `saber-vision-input` — the IMAGE INPUT PATH (round D3, `docs/VISION.md` §8).
//!
//! What prod does between an OpenAI `image_url` part and the tower's `pixel_values`, reproduced
//! from prod's own code (the reference 0.27.1 `multimodal/media`, transformers 5.14.1
//! `Qwen2VLImageProcessor` on the torchvision backend, torchvision 0.24.1, torch 2.11.0 ATen, PIL
//! 12.3.0) and gated on prod's bytes (`tests/processor_gate.rs` over the corpus
//! `the internal reference harness` records):
//!
//! * [`url`] — the `image_url.url` forms and the reference server's refusal texts (`data:` base64 is served;
//!   `http(s)` is REFUSED BY NAME: prod fetches it by default, Saber does not fetch);
//! * [`decode`] — PNG / JPEG / WebP bytes -> RGB (EXIF orientation, alpha over white);
//! * [`processor`] — `smart_resize`, the uint8 bicubic-antialias resample ([`resize`]),
//!   normalise, the T = 2 duplicate, the patch flatten -> `pixel_values [L, 1536]` f32 +
//!   `image_grid_thw`;
//! * [`expand_image_pads`] — the rendered prompt's `<|image_pad|>` ids expanded to each image's
//!   token count, as the reference server's prompt update does.

#![forbid(unsafe_code)]

pub mod decode;
pub mod exif;
pub mod processor;
pub mod pyfmt;
pub mod resize;
pub mod url;

pub use decode::{decode, Format};
pub use processor::{process, smart_resize, Processed, ProcessorConfig};

/// `<|image_pad|>`, `<|vision_start|>`, `<|vision_end|>` in the checkpoint's tokenizer.
pub const IMAGE_PAD_ID: u32 = 248_056;
pub const VISION_START_ID: u32 = 248_053;
pub const VISION_END_ID: u32 = 248_054;

/// The reference server's `--limit-mm-per-prompt` default for a modality (`MultiModalConfig.get_limit_per_prompt`).
pub const DEFAULT_IMAGE_LIMIT: u32 = 999;

/// An 8-bit RGB image, interleaved, row-major (`np.asarray(pil_image)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbImage {
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

/// Every refusal on the image path. Each variant's text is prod's where prod has one (read off
/// prod's code and recorded by `vision_input_ref.py` through the reference server's own `sanitize_message`), and
/// a sentence naming Saber's limit where the refusal is Saber's own.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InputError {
    /// `pybase64.b64decode(data, validate=True)` -> `binascii.Error`.
    #[error("Non-base64 digit found")]
    BadBase64,
    /// A `data:` URL whose encoding is not `base64` (`NotImplementedError`, HTTP 501).
    #[error("Only base64 data URLs are supported for now.")]
    DataUrlNotBase64,
    /// `data:` with no `,` (text carries the URL's `repr(url[:32])`).
    #[error("{0}")]
    DataUrlNoComma(String),
    /// `file://` without `--allowed-local-media-path` (RuntimeError, HTTP 500 in prod).
    #[error("Cannot load local files without `--allowed-local-media-path`.")]
    FileUrl,
    /// Neither http, data nor file.
    #[error("The URL must be either a HTTP, data or file URL.")]
    UnknownScheme,
    /// `http(s)://` — Saber's refusal: prod FETCHES these by default (no domain allowlist).
    #[error("Saber does not fetch remote media: send the image inline as a base64 `data:` URL")]
    RemoteFetch,
    /// PIL could not open / decode the bytes (prod's `Failed to load image: ...`).
    #[error("{0}")]
    Decode(String),
    /// `MAX_IMAGE_PIXELS`.
    #[error("{0}")]
    TooManyPixels(String),
    /// A format prod decodes that Saber refuses by name.
    #[error("{0}")]
    Unsupported(String),
    /// transformers' `smart_resize` aspect-ratio refusal.
    #[error("{0}")]
    AspectRatio(String),
    /// More images than `--limit-mm-per-prompt` (the reference server's text; `param = "image"`).
    #[error("At most {limit} image(s) may be provided in one prompt. (parameter=image)")]
    TooManyImages { limit: u32 },
    /// Saber's arena: an image past `--vision-max-patches`.
    #[error(
        "the image is {patches} patches ({h}x{w} after smart_resize) and this server's vision \
         arena holds {max} (--vision-max-patches) (parameter=image)"
    )]
    TooManyPatches {
        patches: usize,
        max: usize,
        h: usize,
        w: usize,
    },
    /// The prompt's `<|image_pad|>` count does not match the image count.
    #[error("{0}")]
    Placeholder(String),
}

/// How an [`InputError`] reaches the wire: HTTP status, error `type`, `param` — what the reference server's
/// `create_error_response` gives each exception class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wire {
    pub status: u16,
    pub kind: &'static str,
    pub param: Option<&'static str>,
}

impl InputError {
    pub fn wire(&self) -> Wire {
        let bad = |param| Wire {
            status: 400,
            kind: "BadRequestError",
            param,
        };
        match self {
            Self::DataUrlNotBase64 => Wire {
                status: 501,
                kind: "NotImplementedError",
                param: None,
            },
            Self::FileUrl => Wire {
                status: 500,
                kind: "InternalServerError",
                param: None,
            },
            Self::TooManyImages { .. } | Self::TooManyPatches { .. } => bad(Some("image")),
            _ => bad(None),
        }
    }

    /// The census label of this refusal.
    pub fn label(&self) -> &'static str {
        match self {
            Self::BadBase64 | Self::DataUrlNotBase64 | Self::DataUrlNoComma(_) => "data_url",
            Self::FileUrl | Self::UnknownScheme | Self::RemoteFetch => "url",
            Self::Decode(_) | Self::Unsupported(_) => "decode",
            Self::TooManyPixels(_) | Self::AspectRatio(_) | Self::TooManyPatches { .. } => "size",
            Self::TooManyImages { .. } => "count",
            Self::Placeholder(_) => "placeholder",
        }
    }
}

/// The reference server's prompt update for images: the `k`-th `<|image_pad|>` id of the rendered, tokenized
/// prompt becomes `tokens[k]` copies of itself; everything else is untouched.
///
/// # Errors
/// A prompt whose `<|image_pad|>` count differs from the number of images.
pub fn expand_image_pads(ids: &[u32], tokens: &[usize]) -> Result<Vec<u32>, InputError> {
    let found = ids.iter().filter(|&&t| t == IMAGE_PAD_ID).count();
    if found != tokens.len() {
        return Err(InputError::Placeholder(format!(
            "the rendered prompt has {found} <|image_pad|> placeholder(s) for {} image(s)",
            tokens.len()
        )));
    }
    let mut out = Vec::with_capacity(ids.len() + tokens.iter().sum::<usize>());
    let mut k = 0;
    for &t in ids {
        if t == IMAGE_PAD_ID {
            out.extend(std::iter::repeat_n(IMAGE_PAD_ID, tokens[k]));
            k += 1;
        } else {
            out.push(t);
        }
    }
    Ok(out)
}

/// One image through the whole input path: URL -> bytes -> RGB -> processor.
///
/// # Errors
/// Any [`InputError`] of the stages.
pub fn load_and_process(
    image_url: &str,
    cfg: &ProcessorConfig,
) -> Result<(Processed, Format), InputError> {
    let bytes = url::load(image_url)?;
    let (img, fmt) = decode::decode(&bytes)?;
    Ok((processor::process(&img, cfg)?, fmt))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expansion_in_order_and_count_checked() {
        let ids = [1, IMAGE_PAD_ID, 2, IMAGE_PAD_ID, 3];
        let out = expand_image_pads(&ids, &[2, 3]).unwrap();
        assert_eq!(
            out,
            vec![
                1,
                IMAGE_PAD_ID,
                IMAGE_PAD_ID,
                2,
                IMAGE_PAD_ID,
                IMAGE_PAD_ID,
                IMAGE_PAD_ID,
                3
            ]
        );
        assert!(expand_image_pads(&ids, &[2]).is_err());
        assert_eq!(expand_image_pads(&[5, 6], &[]).unwrap(), vec![5, 6]);
    }

    #[test]
    fn wire_classes_are_prods() {
        assert_eq!(InputError::DataUrlNotBase64.wire().status, 501);
        assert_eq!(InputError::FileUrl.wire().status, 500);
        assert_eq!(
            InputError::TooManyImages { limit: 1 }.wire().param,
            Some("image")
        );
        assert_eq!(
            InputError::TooManyImages { limit: 1 }.to_string(),
            "At most 1 image(s) may be provided in one prompt. (parameter=image)"
        );
        assert_eq!(InputError::BadBase64.wire().status, 400);
    }
}
