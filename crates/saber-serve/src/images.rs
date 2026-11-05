//! Image parts on the OpenAI plane (round D3, `docs/VISION.md` §8).
//!
//! `messages[].content` may be a list of parts. The reference server 0.27.1 (`chat_utils.py`,
//! `_parse_chat_message_content_part`) serves, for an image:
//!
//! * `{"type": "image_url", "image_url": {"url": URL}}` — the OpenAI shape;
//! * `{"type": "image_url", "image_url": URL}` — pydantic lets the string through (measured in
//!   prod's image);
//! * `{"type": "input_image", "image_url": URL}` — the Responses-API shape.
//!
//! Each becomes `{"type": "image"}` plus the part's fields that are not content-part fields
//! (`_collect_extra_fields`) in the conversation the template renders, and the template emits
//! `<|vision_start|><|image_pad|><|vision_end|>` for it. Text parts and every other field are
//! passed to the template unchanged (Saber's pre-D3 behaviour, gated by the template corpus).
//!
//! Refusals, in prod's order (the count is checked as each part is ADDED, before any image is
//! loaded; the loads come after the whole conversation is parsed):
//!
//! 1. a part whose `type` is not a content-part type — prod's pydantic `ValidatorIterator` error
//!    (400 `BadRequestError`, `param: null`; its per-union-branch listing is not reproduced);
//! 2. an image on a TEXT-ONLY server — Saber's, by name (400, `param: image`);
//! 3. more images than `--limit-mm-per-prompt` (the reference server's text; prod's default 999);
//! 4. the URL / decode / size refusals of `saber_vision_input` (prod's texts where prod has one);
//! 5. an image past `--vision-max-patches` — Saber's arena, by name. Since FN126 the processor's
//!    pixel ceiling FOLLOWS the arena ([`VisionPolicy::arena_processor`]): an image past it is
//!    downscaled by prod's smart-resize, so only an arena below prod's 65,536-pixel floor (256
//!    patches) can still refuse here.

use serde_json::{Map, Value};

use saber_vision_input::{InputError, ProcessorConfig};

use crate::engine::ImageBatch;
use crate::protocol::ApiError;

/// The reference server's `_KNOWN_CONTENT_PART_FIELDS` (read in prod's image): what `_collect_extra_fields` drops.
pub(crate) const KNOWN_PART_FIELDS: &[&str] = &[
    "audio_embeds",
    "audio_url",
    "closed",
    "data",
    "file",
    "image_embeds",
    "image_pil",
    "image_url",
    "input_audio",
    "name",
    "prompt_cache_breakpoint",
    "refusal",
    "text",
    "thinking",
    "type",
    "uuid",
    "video_url",
];

/// Content-part types the reference server parses (`MM_PARSER_MAP` + `text`/`refusal`), read in prod's image.
const PART_TYPES: &[&str] = &[
    "audio_embeds",
    "audio_url",
    "image_embeds",
    "image_pil",
    "image_url",
    "input_audio",
    "input_image",
    "input_text",
    "output_text",
    "prompt_embeds",
    "refusal",
    "text",
    "thinking",
    "tool_reference",
    "video_url",
];

/// Media types Saber does not serve (prod's model has no audio/video path through this plane
/// either, and the embeds/PIL forms are offline-API shapes).
const REFUSED_MEDIA: &[&str] = &[
    "audio_embeds",
    "audio_url",
    "image_embeds",
    "image_pil",
    "input_audio",
    "prompt_embeds",
    "video_url",
];

/// The server's image policy (`--vision-max-patches`, `--limit-mm-per-prompt`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisionPolicy {
    /// `None` = text-only (the default): any image part is refused by name.
    pub max_patches: Option<u32>,
    /// Images per prompt (the reference server's `--limit-mm-per-prompt '{"image": N}'`, default 999).
    pub limit_per_prompt: u32,
    pub processor: ProcessorConfig,
}

impl VisionPolicy {
    /// FN126: the processor this plane RUNS -- prod's (`processor`), its pixel ceiling clamped to
    /// the booted tower arena (`max_patches x 16 x 16`; prod's 16,777,216 = 65,536 patches). An
    /// image past the arena is downscaled by prod's own smart-resize at that ceiling instead of
    /// being refused; one that already fits is processed byte for byte as prod would.
    #[must_use]
    pub fn arena_processor(&self) -> ProcessorConfig {
        self.processor
            .clamped_to_arena(self.max_patches.unwrap_or(0))
    }
}

/// FN126: the text every engine-side arena refusal carries (`saber_exec::vision`'s run plan,
/// Flash-Next's tower precheck / grouping) -- the HTTP plane counts an engine error carrying it
/// under `saber:image_refused_total{reason="arena"}`.
pub const ARENA_REFUSAL_MARK: &str = "this engine's tower arena";

impl Default for VisionPolicy {
    fn default() -> Self {
        Self {
            max_patches: None,
            limit_per_prompt: saber_vision_input::DEFAULT_IMAGE_LIMIT,
            processor: ProcessorConfig::QWEN38,
        }
    }
}

/// An image refusal: the wire error and its census label.
#[derive(Debug, Clone)]
pub struct Refusal {
    pub error: ApiError,
    pub label: &'static str,
}

impl From<InputError> for Refusal {
    fn from(e: InputError) -> Self {
        let w = e.wire();
        let mut error = ApiError::bad_request(e.to_string(), w.param);
        error.body.kind = w.kind.to_owned();
        error.body.code = w.status;
        Self {
            error,
            label: e.label(),
        }
    }
}

fn refusal(message: String, param: Option<&str>, label: &'static str) -> Refusal {
    Refusal {
        error: ApiError::bad_request(message, param),
        label,
    }
}

/// The conversation with its image parts normalised, and the image URLs in order.
#[derive(Debug, Clone, PartialEq)]
pub struct Extracted {
    pub messages: Vec<Value>,
    pub urls: Vec<String>,
}

fn image_url_of(part: &Map<String, Value>) -> Option<Option<String>> {
    let ty = part.get("type").and_then(Value::as_str);
    let is_image = matches!(ty, Some("image_url") | Some("input_image"));
    if !is_image {
        return None;
    }
    let url = match part.get("image_url") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(o)) => o.get("url").and_then(Value::as_str).map(str::to_owned),
        _ => None,
    };
    Some(url)
}

/// Walk the conversation. `Ok(None)`: no image part anywhere — the messages are used as sent.
///
/// # Errors
/// Refusals 1–3 of the module docs.
pub fn extract(messages: &[Value], policy: &VisionPolicy) -> Result<Option<Extracted>, Refusal> {
    let mut urls = Vec::new();
    let mut out = messages.to_vec();
    for (mi, msg) in out.iter_mut().enumerate() {
        let Some(Value::Array(parts)) = msg.get_mut("content") else {
            continue;
        };
        for (pi, part) in parts.iter_mut().enumerate() {
            let Some(obj) = part.as_object() else {
                continue; // reqcheck::content_parts owns non-object parts
            };
            let ty = obj.get("type");
            let known = ty
                .and_then(Value::as_str)
                .is_some_and(|t| PART_TYPES.contains(&t));
            if !known {
                let shown = match ty {
                    None => "no `type`".to_owned(),
                    Some(Value::String(s)) => {
                        format!("type {}", saber_vision_input::pyfmt::str_repr(s))
                    }
                    Some(other) => format!("type {other}"),
                };
                return Err(refusal(
                    format!(
                        "validation errors for ValidatorIterator: part {pi} of messages[{mi}] \
                         ({shown}) matches no content-part type; Saber serves `text` and \
                         `image_url` parts"
                    ),
                    None,
                    "part_type",
                ));
            }
            let t = ty.and_then(Value::as_str).unwrap_or_default();
            if REFUSED_MEDIA.contains(&t) {
                return Err(refusal(
                    format!(
                        "`{t}` content parts are not supported by this server: Saber's vision \
                         path serves images as `image_url` parts (parameter=type, value={t})"
                    ),
                    Some("type"),
                    "part_type",
                ));
            }
            let Some(url) = image_url_of(obj) else {
                continue;
            };
            if policy.max_patches.is_none() {
                return Err(refusal(
                    crate::engine::TEXT_ONLY_REFUSAL.to_owned(),
                    Some("image"),
                    "disabled",
                ));
            }
            // The reference server validates the count as the item is ADDED (`_validate_add`).
            if urls.len() as u32 + 1 > policy.limit_per_prompt {
                return Err(InputError::TooManyImages {
                    limit: policy.limit_per_prompt,
                }
                .into());
            }
            let Some(url) = url else {
                return Err(refusal(
                    format!("image part {pi} of messages[{mi}] carries no `image_url.url`"),
                    Some("image_url"),
                    "url",
                ));
            };
            urls.push(url);
            let mut norm = Map::new();
            norm.insert("type".to_owned(), Value::String("image".to_owned()));
            for (k, v) in obj {
                if !KNOWN_PART_FIELDS.contains(&k.as_str()) {
                    norm.insert(k.clone(), v.clone());
                }
            }
            *part = Value::Object(norm);
        }
    }
    if urls.is_empty() {
        return Ok(None);
    }
    Ok(Some(Extracted {
        messages: out,
        urls,
    }))
}

/// Load, decode and process every image (CPU; call it off the async runtime).
///
/// # Errors
/// Refusals 4–5 of the module docs.
pub fn load(urls: &[String], policy: &VisionPolicy) -> Result<(ImageBatch, Vec<usize>), Refusal> {
    load_counted(urls, policy).map(|l| (l.batch, l.tokens))
}

/// FN126: one request's images as [`load`] hands them on, + how many the arena clamp DOWNSCALED
/// (their resize target differs from the one at prod's own ceiling).
#[derive(Debug, Clone, PartialEq)]
pub struct Loaded {
    pub batch: ImageBatch,
    pub tokens: Vec<usize>,
    pub downscaled: u32,
}

/// [`load`] with the clamp's count ([`Loaded::downscaled`]). The processor is
/// [`VisionPolicy::arena_processor`]: prod's, at the arena's pixel ceiling.
///
/// # Errors
/// Refusals 4–5 of the module docs.
pub fn load_counted(urls: &[String], policy: &VisionPolicy) -> Result<Loaded, Refusal> {
    let max = policy.max_patches.unwrap_or(0) as usize;
    let cfg = policy.arena_processor();
    let prod = policy.processor;
    let mut grids = Vec::with_capacity(urls.len());
    let mut tokens = Vec::with_capacity(urls.len());
    let mut pixel_values = Vec::new();
    let mut downscaled = 0u32;
    for url in urls {
        // `saber_vision_input::load_and_process`, the source's size kept for the count
        let bytes = saber_vision_input::url::load(url)?;
        let (img, _fmt) = saber_vision_input::decode(&bytes)?;
        let p = saber_vision_input::process(&img, &cfg)?;
        if cfg.max_pixels < prod.max_pixels {
            let at_prod = saber_vision_input::smart_resize(
                img.height,
                img.width,
                prod.factor(),
                prod.min_pixels,
                prod.max_pixels,
            )?;
            if at_prod != p.resized_hw {
                downscaled += 1;
            }
        }
        if p.patches() > max {
            return Err(InputError::TooManyPatches {
                patches: p.patches(),
                max,
                h: p.resized_hw.0,
                w: p.resized_hw.1,
            }
            .into());
        }
        tokens.push(p.tokens(&cfg));
        grids.push(p.grid_thw);
        pixel_values.extend_from_slice(&p.pixel_values);
    }
    Ok(Loaded {
        batch: ImageBatch {
            grids,
            pixel_values,
        },
        tokens,
        downscaled,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn on() -> VisionPolicy {
        VisionPolicy {
            max_patches: Some(65_536),
            ..VisionPolicy::default()
        }
    }

    #[test]
    fn text_only_requests_are_untouched() {
        let m = vec![json!({"role": "user", "content": [{"type": "text", "text": "hi", "x": 1}]})];
        assert_eq!(extract(&m, &VisionPolicy::default()).unwrap(), None);
        assert_eq!(extract(&m, &on()).unwrap(), None);
    }

    #[test]
    fn image_parts_normalise_to_the_reference_shape() {
        let m = vec![json!({"role": "user", "content": [
            {"type": "text", "text": "a"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA==", "detail": "low"}, "uuid": "u", "extra": 7},
            {"type": "input_image", "image_url": "data:image/png;base64,AQ=="}]})];
        let e = extract(&m, &on()).unwrap().unwrap();
        assert_eq!(
            e.urls,
            vec!["data:image/png;base64,AA==", "data:image/png;base64,AQ=="]
        );
        assert_eq!(
            e.messages[0]["content"],
            json!([{"type": "text", "text": "a"}, {"type": "image", "extra": 7}, {"type": "image"}])
        );
    }

    #[test]
    fn refusals_in_prods_order() {
        let img = json!({"type": "image_url", "image_url": {"url": "data:,"}});
        let m = vec![json!({"role": "user", "content": [img.clone(), img.clone()]})];
        let off = extract(&m, &VisionPolicy::default()).unwrap_err();
        assert_eq!(
            (off.label, off.error.body.param.as_deref()),
            ("disabled", Some("image"))
        );
        let one = VisionPolicy {
            limit_per_prompt: 1,
            ..on()
        };
        let lim = extract(&m, &one).unwrap_err();
        assert_eq!(
            lim.error.body.message,
            "At most 1 image(s) may be provided in one prompt. (parameter=image)"
        );
        let bad = vec![json!({"role": "user", "content": [{"type": "image_file"}]})];
        assert_eq!(extract(&bad, &on()).unwrap_err().label, "part_type");
        let audio = vec![
            json!({"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "x"}}]}),
        ];
        assert_eq!(
            extract(&audio, &on())
                .unwrap_err()
                .error
                .body
                .param
                .as_deref(),
            Some("type")
        );
    }

    #[test]
    fn load_refuses_with_prods_texts_and_the_arena_by_name() {
        let p = on();
        let e = load(&["data:image/png,abc".to_owned()], &p).unwrap_err();
        assert_eq!(
            (e.error.body.code, e.error.body.kind.as_str()),
            (501, "NotImplementedError")
        );
        let e = load(&["https://example.com/x.png".to_owned()], &p).unwrap_err();
        assert_eq!(e.label, "url");
        // a 1x1 PNG is upscaled to 256x256 = 256 patches: past a 100-patch arena
        let png1 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGMQUDAAAACkAGE0Zn1yAAAAAElFTkSuQmCC";
        let small = VisionPolicy {
            max_patches: Some(100),
            ..on()
        };
        let e = load(&[png1.to_owned()], &small).unwrap_err();
        assert_eq!(
            (e.label, e.error.body.param.as_deref()),
            ("size", Some("image"))
        );
        let (batch, tokens) = load(&[png1.to_owned()], &p).unwrap();
        assert_eq!((batch.grids, tokens), (vec![[1, 16, 16]], vec![64]));
    }

    fn png_url(h: u32, w: u32) -> String {
        use base64::Engine as _;
        let mut out = Vec::new();
        {
            let mut e = png::Encoder::new(&mut out, w, h);
            e.set_color(png::ColorType::Rgb);
            e.set_depth(png::BitDepth::Eight);
            let mut wr = e.write_header().unwrap();
            let px: Vec<u8> = (0..(h * w * 3) as usize)
                .map(|i| ((i * 2_654_435_761usize) >> 9) as u8)
                .collect();
            wr.write_image_data(&px).unwrap();
        }
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(out)
        )
    }

    /// FN126: the plane's processor follows the booted arena. An 8 MP photo (3264 x 2448: 31,008
    /// patches at prod's ceiling, the size refusal on a 16,384-patch boot before FN126) is
    /// ANSWERED at a 16,384 boot -- downscaled to prod's integers at 4,194,304 px (1760 x 2336 =
    /// 16,060 patches), counted -- while prod's arena (65,536) keeps prod's target and counts
    /// nothing; an image that fits is byte for byte the same batch at both arenas.
    #[test]
    fn fn126_the_pixel_budget_follows_the_arena() {
        let at = |a: u32| VisionPolicy {
            max_patches: Some(a),
            ..VisionPolicy::default()
        };
        assert_eq!(at(16_384).arena_processor().max_pixels, 4_194_304);
        assert_eq!(at(65_536).arena_processor(), ProcessorConfig::QWEN38);
        assert_eq!(
            VisionPolicy::default().arena_processor(),
            ProcessorConfig::QWEN38
        );
        let big = png_url(2448, 3264);
        let l = load_counted(std::slice::from_ref(&big), &at(16_384)).unwrap();
        assert_eq!(l.batch.grids, vec![[1, 110, 146]]);
        assert_eq!(
            (l.batch.patches(), l.tokens.clone(), l.downscaled),
            (16_060, vec![4_015], 1)
        );
        let p = load_counted(std::slice::from_ref(&big), &at(65_536)).unwrap();
        assert_eq!((p.batch.patches(), p.downscaled), (31_008, 0));
        // a 1280 x 704 screenshot (FN125's agent shape, 3,520 patches) fits: the same bytes
        let fit = png_url(704, 1280);
        let a = load_counted(std::slice::from_ref(&fit), &at(16_384)).unwrap();
        let b = load_counted(std::slice::from_ref(&fit), &at(65_536)).unwrap();
        assert_eq!(a.downscaled, 0);
        assert_eq!(a, b);
        assert_eq!(a.batch.patches(), 3_520);
        // and the 5-image agent request is five fitting images (17,600 patches in all: the sum is
        // the engine's per-image tower runs' business, not a refusal here)
        let five: Vec<String> = (0..5).map(|_| fit.clone()).collect();
        let f = load_counted(&five, &at(16_384)).unwrap();
        assert_eq!((f.batch.patches(), f.downscaled), (17_600, 0));
    }
}
