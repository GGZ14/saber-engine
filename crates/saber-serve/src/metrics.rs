//! `/metrics` in Prometheus text format, under the exact names and labels the tools
//! on this box read.
//!
//! `eval_drafter.py` reads the three `saber:spec_decode_*` counters, BetterBench and
//! the GPU-test discipline in CLAUDE.md read `saber:num_requests_running`. The label
//! set is prod's: `{engine="0",model_name="Qwen3.8"}`. Names and labels were taken
//! from the live server, not from the reference server's source.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::engine::{EngineStats, FinishReason};

/// Counters the serving plane owns. Queue depth and the spec-decode counters come
/// from the engine at render time.
#[derive(Debug)]
pub struct Metrics {
    engine_label: String,
    model_name: String,
    prompt_tokens: AtomicU64,
    generation_tokens: AtomicU64,
    success_stop: AtomicU64,
    success_length: AtomicU64,
    success_tool_calls: AtomicU64,
    success_abort: AtomicU64,
    /// A streaming request aborted because the **client went away**: the response body was
    /// dropped, which is the connection closing.
    abort_on_disconnect: AtomicU64,
    /// A streaming request aborted because a **chunk could not be sent**. The fallback path: it
    /// only fires when the engine produced something to push and the push failed, so it costs a
    /// step of the sequence's life. `CLAUDE.md` rule 5 -- the two routes are counted separately
    /// because a shift from the first to the second is a regression in exactly this behaviour.
    abort_on_write_failure: AtomicU64,
    /// A request refused by [`crate::reqcheck`] before it reached the engine, by family: the
    /// pydantic-shaped 400 (`shape`) and the handler-shaped one (`handler`). Two counters, not
    /// one, because the two families carry different `type` strings on the wire and a shift from
    /// one to the other is exactly the regression this plane exists to avoid (CLAUDE.md rule 5).
    refused_shape: AtomicU64,
    refused_handler: AtomicU64,
    /// D3: images admitted on the OpenAI plane (decoded, processed, handed to the engine), the
    /// patches they carried, and the image refusals by stage (`saber_vision_input::InputError::
    /// label` order below). Prod exports no image counters; these are Saber's.
    images: AtomicU64,
    image_patches: AtomicU64,
    image_refused: [AtomicU64; IMAGE_REFUSAL_LABELS.len()],
    /// FN126: images the arena clamp downscaled (their resize target at the booted arena's pixel
    /// ceiling differs from prod's own) -- answered, not refused.
    image_downscaled: AtomicU64,
}

/// The image-refusal labels, in exposition order (`InputError::label` plus `disabled`, a vision
/// part sent to a text-only server, and FN126's `arena`: an ENGINE-side tower-arena refusal that
/// reached the client as an engine error -- 0 after FN126's clamp + per-image tower runs).
pub const IMAGE_REFUSAL_LABELS: [&str; 9] = [
    "disabled",
    "part_type",
    "data_url",
    "url",
    "decode",
    "size",
    "count",
    "placeholder",
    "arena",
];

impl Metrics {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            engine_label: "0".to_owned(),
            model_name: model_name.into(),
            prompt_tokens: AtomicU64::new(0),
            generation_tokens: AtomicU64::new(0),
            success_stop: AtomicU64::new(0),
            success_length: AtomicU64::new(0),
            success_tool_calls: AtomicU64::new(0),
            success_abort: AtomicU64::new(0),
            abort_on_disconnect: AtomicU64::new(0),
            abort_on_write_failure: AtomicU64::new(0),
            refused_shape: AtomicU64::new(0),
            refused_handler: AtomicU64::new(0),
            images: AtomicU64::new(0),
            image_patches: AtomicU64::new(0),
            image_refused: std::array::from_fn(|_| AtomicU64::new(0)),
            image_downscaled: AtomicU64::new(0),
        }
    }

    /// One request's images were admitted: `count` images, `patches` patches in all.
    pub fn record_images(&self, count: u64, patches: u64) {
        self.images.fetch_add(count, Ordering::Relaxed);
        self.image_patches.fetch_add(patches, Ordering::Relaxed);
    }

    /// FN126: `n` of a request's images were downscaled by the arena clamp.
    pub fn record_downscaled(&self, n: u64) {
        self.image_downscaled.fetch_add(n, Ordering::Relaxed);
    }

    /// FN126: images downscaled by the arena clamp so far.
    pub fn downscaled(&self) -> u64 {
        self.image_downscaled.load(Ordering::Relaxed)
    }

    /// FN126: an engine error that ended a request -- counted as an `arena` image refusal when it
    /// is the tower arena's ([`crate::images::ARENA_REFUSAL_MARK`]); any other is not an image's.
    pub fn record_engine_error(&self, error: &str) {
        if error.contains(crate::images::ARENA_REFUSAL_MARK) {
            self.record_image_refusal("arena");
        }
    }

    /// An image refusal, by [`IMAGE_REFUSAL_LABELS`] label (an unknown label is counted under
    /// `decode`, never dropped).
    pub fn record_image_refusal(&self, label: &str) {
        let i = IMAGE_REFUSAL_LABELS
            .iter()
            .position(|l| *l == label)
            .unwrap_or(4);
        self.image_refused[i].fetch_add(1, Ordering::Relaxed);
    }

    /// `(images, patches, refusals by label)`, for a test or the census line.
    pub fn image_counts(&self) -> (u64, u64, Vec<(&'static str, u64)>) {
        (
            self.images.load(Ordering::Relaxed),
            self.image_patches.load(Ordering::Relaxed),
            IMAGE_REFUSAL_LABELS
                .iter()
                .zip(&self.image_refused)
                .map(|(l, c)| (*l, c.load(Ordering::Relaxed)))
                .collect(),
        )
    }

    /// A request refused before the engine saw it. `kind` is the error body's own `type`, so the
    /// counter and the wire cannot drift apart.
    pub fn record_request_refusal(&self, kind: &str) {
        if kind == "Bad Request" {
            &self.refused_shape
        } else {
            &self.refused_handler
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    /// The two refusal families, for a test or a dashboard.
    pub fn refusal_paths(&self) -> (u64, u64) {
        (
            self.refused_shape.load(Ordering::Relaxed),
            self.refused_handler.load(Ordering::Relaxed),
        )
    }

    pub fn record_prompt_tokens(&self, n: u64) {
        self.prompt_tokens.fetch_add(n, Ordering::Relaxed);
    }

    pub fn record_generation_tokens(&self, n: u64) {
        self.generation_tokens.fetch_add(n, Ordering::Relaxed);
    }

    /// A client hung up and the engine was told at once.
    pub fn record_abort_on_disconnect(&self) {
        self.abort_on_disconnect.fetch_add(1, Ordering::Relaxed);
    }

    /// A client hung up and it took a failed chunk write to find out.
    pub fn record_abort_on_write_failure(&self) {
        self.abort_on_write_failure.fetch_add(1, Ordering::Relaxed);
    }

    /// The two disconnect routes, for a test or a dashboard.
    pub fn abort_paths(&self) -> (u64, u64) {
        (
            self.abort_on_disconnect.load(Ordering::Relaxed),
            self.abort_on_write_failure.load(Ordering::Relaxed),
        )
    }

    pub fn record_finish(&self, reason: FinishReason) {
        match reason {
            FinishReason::Stop => &self.success_stop,
            FinishReason::Length => &self.success_length,
            FinishReason::ToolCalls => &self.success_tool_calls,
            FinishReason::Abort => &self.success_abort,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    fn labels(&self) -> String {
        format!(
            "{{engine=\"{}\",model_name=\"{}\"}}",
            self.engine_label, self.model_name
        )
    }

    /// The full exposition. `stats` is read from the engine at scrape time.
    pub fn render(&self, stats: EngineStats) -> String {
        let l = self.labels();
        let mut out = String::with_capacity(2048);
        let mut metric = |name: &str, help: &str, kind: &str, labels: &str, value: u64| {
            let _ = writeln!(out, "# HELP {name} {help}");
            let _ = writeln!(out, "# TYPE {name} {kind}");
            let _ = writeln!(out, "{name}{labels} {value}.0");
        };

        metric(
            "saber:num_requests_running",
            "Number of requests in model execution batches.",
            "gauge",
            &l,
            stats.num_requests_running,
        );
        metric(
            "saber:num_requests_waiting",
            "Number of requests waiting to be processed.",
            "gauge",
            &l,
            stats.num_requests_waiting,
        );
        metric(
            "saber:prompt_tokens_total",
            "Number of prefill tokens processed.",
            "counter",
            &l,
            self.prompt_tokens.load(Ordering::Relaxed),
        );
        metric(
            "saber:generation_tokens_total",
            "Number of generation tokens processed.",
            "counter",
            &l,
            self.generation_tokens.load(Ordering::Relaxed),
        );
        metric(
            "saber:spec_decode_num_drafts_total",
            "Number of spec decoding drafts.",
            "counter",
            &l,
            stats.spec_decode_num_drafts,
        );
        metric(
            "saber:spec_decode_num_draft_tokens_total",
            "Number of draft tokens.",
            "counter",
            &l,
            stats.spec_decode_num_draft_tokens,
        );
        metric(
            "saber:spec_decode_num_accepted_tokens_total",
            "Number of accepted tokens.",
            "counter",
            &l,
            stats.spec_decode_num_accepted_tokens,
        );
        // The per-position curve, under prod's own name and label. The three totals above say
        // acceptance IS 0.9 per draft; this is the only thing in the exposition that says WHERE
        // it dies, and a harness outside the process (`eval_drafter.py`'s sibling reading, a
        // dashboard) has no other view of it. The reference server declares it as
        // `pos_labelnames = labelnames + ["position"]` over the same counter family, so the
        // label order here is prod's: engine, model_name, position.
        //
        // Emitted with the HELP/TYPE pair even when the vector is empty — a scrape of a server
        // that has not drafted yet must show the family exists rather than look like a build
        // without speculation.
        let _ = writeln!(
            out,
            "# HELP saber:spec_decode_num_accepted_tokens_per_pos_total Accepted tokens per draft \
             position."
        );
        let _ = writeln!(
            out,
            "# TYPE saber:spec_decode_num_accepted_tokens_per_pos_total counter"
        );
        for (pos, count) in stats.spec_decode_accepted_per_pos.iter().enumerate() {
            let _ = writeln!(
                out,
                "saber:spec_decode_num_accepted_tokens_per_pos_total{{engine=\"{}\",model_name=\"{}\",position=\"{}\"}} {}.0",
                self.engine_label, self.model_name, pos, count
            );
        }

        // request_success_total carries a third label, as prod does.
        let _ = writeln!(
            out,
            "# HELP saber:request_success_total Count of successfully processed requests."
        );
        let _ = writeln!(out, "# TYPE saber:request_success_total counter");
        for (reason, counter) in [
            ("stop", &self.success_stop),
            ("length", &self.success_length),
            ("tool_calls", &self.success_tool_calls),
            ("abort", &self.success_abort),
        ] {
            let _ = writeln!(
                out,
                "saber:request_success_total{{engine=\"{}\",finished_reason=\"{}\",model_name=\"{}\"}} {}.0",
                self.engine_label,
                reason,
                self.model_name,
                counter.load(Ordering::Relaxed)
            );
        }
        // Which path the abort of a streaming request took. Saber's own names -- prod has no
        // equivalent, because prod only has the second one.
        let _ = writeln!(
            out,
            "# HELP saber:stream_abort_total How a streaming request's abort was discovered."
        );
        let _ = writeln!(out, "# TYPE saber:stream_abort_total counter");
        for (path, counter) in [
            ("disconnect", &self.abort_on_disconnect),
            ("write_failure", &self.abort_on_write_failure),
        ] {
            let _ = writeln!(
                out,
                "saber:stream_abort_total{{engine=\"{}\",model_name=\"{}\",path=\"{}\"}} {}.0",
                self.engine_label,
                self.model_name,
                path,
                counter.load(Ordering::Relaxed)
            );
        }
        // Which family a refused request fell into. Prod has no equivalent: its two families come
        // from two different layers of its own stack (FastAPI's validator and the chat handler),
        // and here they are one module, so the split has to be exported to stay visible.
        let _ = writeln!(
            out,
            "# HELP saber:request_refused_total A request refused before the engine saw it."
        );
        let _ = writeln!(out, "# TYPE saber:request_refused_total counter");
        for (family, counter) in [
            ("shape", &self.refused_shape),
            ("handler", &self.refused_handler),
        ] {
            let _ = writeln!(
                out,
                "saber:request_refused_total{{engine=\"{}\",model_name=\"{}\",family=\"{}\"}} {}.0",
                self.engine_label,
                self.model_name,
                family,
                counter.load(Ordering::Relaxed)
            );
        }
        // D3: the image input path.
        for (name, help, value) in [
            (
                "saber:image_inputs_total",
                "Images admitted on the OpenAI plane (decoded, processed, sent to the engine).",
                &self.images,
            ),
            (
                "saber:image_patches_total",
                "Vision patches (16x16 x T=2) carried by the admitted images.",
                &self.image_patches,
            ),
        ] {
            let _ = writeln!(out, "# HELP {name} {help}");
            let _ = writeln!(out, "# TYPE {name} counter");
            let _ = writeln!(out, "{name}{l} {}.0", value.load(Ordering::Relaxed));
        }
        let _ = writeln!(
            out,
            "# HELP saber:image_refused_total An image refused on the OpenAI plane, by stage."
        );
        let _ = writeln!(out, "# TYPE saber:image_refused_total counter");
        for (reason, counter) in IMAGE_REFUSAL_LABELS.iter().zip(&self.image_refused) {
            let _ = writeln!(
                out,
                "saber:image_refused_total{{engine=\"{}\",model_name=\"{}\",reason=\"{}\"}} {}.0",
                self.engine_label,
                self.model_name,
                reason,
                counter.load(Ordering::Relaxed)
            );
        }
        // FN126: images the arena clamp downscaled (answered at the booted arena's pixel ceiling)
        let _ = writeln!(
            out,
            "# HELP saber_fn126_downscaled_total Images downscaled by prod's smart-resize at the \
             booted tower arena's pixel ceiling (FN126; answered, not refused)."
        );
        let _ = writeln!(out, "# TYPE saber_fn126_downscaled_total counter");
        let _ = writeln!(
            out,
            "saber_fn126_downscaled_total{l} {}.0",
            self.image_downscaled.load(Ordering::Relaxed)
        );
        // The engine's shutdown census, live (ws-v8): one family per counter, under the name the
        // census table in `docs/SHADOW.md` §2 gives it. Rendered in the order the engine declared
        // them, which groups each census line's counters together.
        for sample in &stats.census {
            let def = sample.def;
            let _ = writeln!(out, "# HELP {} {}", def.name, def.help);
            let _ = writeln!(out, "# TYPE {} {}", def.name, def.kind.word());
            let _ = writeln!(out, "{}{} {}.0", def.name, l, sample.value);
        }
        // The Flash-Next memory plane (FN8): the tier's and the PLE tier's families, verbatim.
        out.push_str(&stats.memory_plane);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FN126: an engine error carrying the tower arena's text is an `arena` image refusal (any
    /// other engine error is not an image's), and the clamp's downscaled count is its own family.
    #[test]
    fn fn126_the_arena_refusal_and_the_downscaled_count() {
        let m = Metrics::new("Qwen3.8-Flash-Next");
        m.record_engine_error(
            "vision: 31440 patches exceed this engine's tower arena of 16384 patches \
             (`--vision-max-patches`; prod's ceiling is 65536)",
        );
        m.record_engine_error(
            "FN126: an image of 70000 patches exceeds this engine's tower arena of 65536 patches",
        );
        m.record_engine_error("restore failed: something else");
        m.record_downscaled(3);
        let (_, _, by) = m.image_counts();
        let get = |l: &str| by.iter().find(|(x, _)| *x == l).map(|x| x.1);
        assert_eq!(get("arena"), Some(2));
        assert_eq!(
            get("decode"),
            Some(0),
            "the other engine error is not an image refusal"
        );
        assert_eq!(m.downscaled(), 3);
        let text = m.render(EngineStats::default());
        assert!(
            text.contains(
                "saber:image_refused_total{engine=\"0\",model_name=\"Qwen3.8-Flash-Next\",reason=\"arena\"} 2.0"
            ),
            "{text}"
        );
        assert!(
            text.contains(
                "saber_fn126_downscaled_total{engine=\"0\",model_name=\"Qwen3.8-Flash-Next\"} 3.0"
            ),
            "{text}"
        );
    }

    /// FN8: the memory plane's families ride the exposition verbatim, after prod's.
    #[test]
    fn the_memory_plane_families_are_appended() {
        let metrics = Metrics::new("Qwen3.8");
        let plane = "# TYPE saber_xtier_routed_total counter\n\
                     saber_xtier_routed_total{rank=\"0\",path=\"resident\"} 5\n";
        let text = metrics.render(EngineStats {
            memory_plane: plane.to_string(),
            ..EngineStats::default()
        });
        assert!(text.ends_with(plane), "{text}");
        let none = metrics.render(EngineStats::default());
        assert!(!none.contains("saber_xtier"), "no tier, no family");
    }

    /// The names and labels `eval_drafter.py` and BetterBench parse, verbatim.
    #[test]
    fn exposition_carries_the_names_prod_exports() {
        let metrics = Metrics::new("Qwen3.8");
        metrics.record_prompt_tokens(34);
        metrics.record_generation_tokens(33);
        metrics.record_finish(FinishReason::Stop);
        let text = metrics.render(EngineStats {
            num_requests_running: 2,
            num_requests_waiting: 1,
            spec_decode_num_drafts: 7,
            spec_decode_num_draft_tokens: 49,
            spec_decode_num_accepted_tokens: 26,
            spec_decode_accepted_per_pos: vec![7, 6, 5, 4, 2, 1, 1],
            census: Vec::new(),
            memory_plane: String::new(),
        });
        for line in [
            "saber:num_requests_running{engine=\"0\",model_name=\"Qwen3.8\"} 2.0",
            "saber:num_requests_waiting{engine=\"0\",model_name=\"Qwen3.8\"} 1.0",
            "saber:prompt_tokens_total{engine=\"0\",model_name=\"Qwen3.8\"} 34.0",
            "saber:generation_tokens_total{engine=\"0\",model_name=\"Qwen3.8\"} 33.0",
            "saber:spec_decode_num_drafts_total{engine=\"0\",model_name=\"Qwen3.8\"} 7.0",
            "saber:spec_decode_num_draft_tokens_total{engine=\"0\",model_name=\"Qwen3.8\"} 49.0",
            "saber:spec_decode_num_accepted_tokens_total{engine=\"0\",model_name=\"Qwen3.8\"} 26.0",
            "saber:request_success_total{engine=\"0\",finished_reason=\"stop\",model_name=\"Qwen3.8\"} 1.0",
            // The per-position curve, under prod's name and prod's label order.
            "saber:spec_decode_num_accepted_tokens_per_pos_total{engine=\"0\",model_name=\"Qwen3.8\",position=\"0\"} 7.0",
            "saber:spec_decode_num_accepted_tokens_per_pos_total{engine=\"0\",model_name=\"Qwen3.8\",position=\"6\"} 1.0",
        ] {
            assert!(text.contains(line), "missing exposition line: {line}\n{text}");
        }
        assert!(text.contains("# TYPE saber:num_requests_running gauge"));
        assert!(text.contains("# TYPE saber:spec_decode_num_drafts_total counter"));
        assert!(text.contains("# TYPE saber:spec_decode_num_accepted_tokens_per_pos_total counter"));
        // prod's identity: the positions sum to the accepted total, and the bonus is in none
        // of them (`saber_model::draft::SpecDecodeCounters`, measured on the live unit).
        let per_pos: u64 = text
            .lines()
            .filter(|l| l.starts_with("saber:spec_decode_num_accepted_tokens_per_pos_total{"))
            .filter_map(|l| l.rsplit(' ').next())
            .filter_map(|v| v.trim_end_matches(".0").parse::<u64>().ok())
            .sum();
        assert_eq!(
            per_pos, 26,
            "the per-position vector must sum to the accepted total"
        );
    }

    /// A server that has never drafted still exposes the family, and exposes no position rows.
    ///
    /// A scrape that shows nothing at all is indistinguishable from a build with no speculation
    /// in it, which is the reading `eval_drafter.py` refuses on (`prod_spec_accept.py` returns 3
    /// rather than reporting a drafter that accepts nothing).
    #[test]
    fn the_per_position_family_exists_before_the_first_draft() {
        let text = Metrics::new("Qwen3.8").render(EngineStats::default());
        assert!(
            text.contains("# TYPE saber:spec_decode_num_accepted_tokens_per_pos_total counter"),
            "the family must be declared even with no drafts:\n{text}"
        );
        assert!(
            !text.contains("saber:spec_decode_num_accepted_tokens_per_pos_total{"),
            "no position rows before a draft:\n{text}"
        );
    }
}
