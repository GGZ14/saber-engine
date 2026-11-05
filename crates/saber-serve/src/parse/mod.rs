//! The `qwen3` reasoning parser and the `qwen3_coder` (= `qwen3_xml`) tool parser.
//!
//! A port of the reference 0.27.1's `ParserEngine` (`upstream/parser/engine/parser_engine.py`) and
//! its qwen3 configuration, including the two fixes production bakes on top of it
//! (`ref/patch_qwen3_toolparse.py`, the reference server #47137). Gated record for record
//! against the real parser by `tests/parser_corpus.rs` over the corpus that
//! `the internal reference harness` records inside the prod image.
//!
//! The model emits
//!
//! ```text
//! <think>…</think><tool_call>
//! <function=NAME>
//! <parameter=KEY>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! and this module turns it into `reasoning`, `content` and OpenAI `tool_calls` whose
//! `arguments` are JSON typed by the request's tool schema.

pub mod compose;
pub mod lexer;
pub mod machine;
pub mod pyjson;

use std::collections::HashSet;
use std::sync::LazyLock;

use fancy_regex::Regex;
use serde_json::Value;

use lexer::LexerShape;
use machine::{EventType, Machine, ParserState, SemanticEvent};
use pyjson::{PyDict, PyVal};

/// `<parameter=NAME>VALUE`, closed either by `</parameter>` or by the next
/// `<parameter=`. The lookahead is why this crate depends on `fancy-regex`.
static PARAM_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?s)<\s*parameter\s*=\s*([^>]*)>(.*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
    )
    .expect("PARAM_RE compiles")
});

/// The still-open parameter at the end of a truncated body (streaming only).
static PARTIAL_PARAM_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<\s*parameter\s*=\s*([^>]+)>(.*)$").expect("PARTIAL_PARAM_RE compiles")
});

/// `"name": "…"` inside a raw JSON body, for calls that never emitted `<function=`.
static NAME_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#""name"\s*:\s*"([^"]*)""#).expect("NAME_RE compiles"));

/// One tool call under construction.
#[derive(Debug, Clone, Default)]
struct ToolCallSlot {
    id: String,
    name: String,
    args: String,
    name_sent: bool,
    /// Keys whose values are schema-typed `string` and may therefore stream before
    /// the closing tag arrives. `None` means "no schema".
    string_keys: Option<HashSet<String>>,
    streamed_json: String,
}

/// An incremental OpenAI `tool_calls` entry.
#[derive(Debug, Clone, PartialEq)]
pub struct DeltaToolCall {
    pub index: i32,
    pub id: Option<String>,
    pub kind: Option<&'static str>,
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// One streaming update, in the shape `/v1/chat/completions` sends it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeltaMessage {
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<DeltaToolCall>,
}

impl DeltaMessage {
    pub fn is_empty(&self) -> bool {
        self.content.is_none() && self.reasoning.is_none() && self.tool_calls.is_empty()
    }
}

/// A finished tool call.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// The result of a non-streaming parse.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedToolCalls {
    pub tools_called: bool,
    pub tool_calls: Vec<ToolCall>,
    pub content: Option<String>,
}

/// How tool calls in the request are to be treated (`--enable-auto-tool-choice`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolChoice {
    /// `"auto"`, or absent with tools present: parse tool calls out of the output.
    #[default]
    Auto,
    /// `"none"`: never surface tool calls; their markup is still stripped from
    /// content, which is what the engine-based parsers do in the reference server.
    None,
}

/// Construction-time options: everything the reference server reads off the request or the config.
#[derive(Debug, Clone)]
pub struct ParserOptions {
    /// `chat_template_kwargs.enable_thinking` — decides the initial state.
    pub enable_thinking: bool,
    /// The request's `tools`, as OpenAI JSON.
    pub tools: Vec<Value>,
    pub tool_choice: ToolChoice,
    /// Special tokens the lexer drops (everything the tokenizer marks special that is
    /// not part of the grammar). Supplied by the caller so it always matches the
    /// tokenizer actually loaded.
    pub drop_tokens: Vec<String>,
    /// The reference server's `skip_tool_parsing`: leave tool markup in `content` instead of
    /// promoting it. This is the reasoning-parser-only configuration — a server
    /// started with `--reasoning-parser qwen3` and *without*
    /// `--enable-auto-tool-choice`.
    pub skip_tool_parsing: bool,
}

impl Default for ParserOptions {
    fn default() -> Self {
        Self {
            enable_thinking: true,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            drop_tokens: Vec::new(),
            skip_tool_parsing: false,
        }
    }
}

/// Generates the `chatcmpl-tool-…` ids the reference server's `make_tool_call_id` produces.
pub trait IdSource: Send + Sync + std::fmt::Debug {
    fn tool_call_id(&self) -> String;
}

#[derive(Debug, Default)]
pub struct RandomIds;

impl IdSource for RandomIds {
    fn tool_call_id(&self) -> String {
        format!("chatcmpl-tool-{}", crate::ids::random_hex32())
    }
}

/// The qwen3 parser: reasoning and tool calls out of one state machine, which is how
/// The reference server's engine-based parsers work and how `Qwen3EngineToolParser` (`qwen3_coder` /
/// `qwen3_xml`) is built.
#[derive(Debug)]
pub struct Qwen3Parser {
    machine: Machine,
    tools: Vec<Value>,
    slots: Vec<ToolCallSlot>,
    deferred_content: String,
    content_has_nonws: bool,
    suppress_tool_calls: bool,
    reasoning_ended: bool,
    has_reasoning: bool,
    streaming_initialized: bool,
    /// `chat_template_kwargs.enable_thinking`. Also decides whether
    /// [`Qwen3Parser::extract_reasoning`] parses at all — see that method.
    thinking_enabled: bool,
    ids: Box<dyn IdSource>,
}

// The qwen3 config's flags, spelled out so the ported code reads like the original.
const STRIP_TRAILING_REASONING_WS: bool = false;
const DROP_WS_ONLY_CONTENT_BEFORE_TOOLS: bool = true;
const STRIP_CONTENT_WS_WITH_TOOLS: bool = true;
const VALIDATE_TOOL_NAMES: bool = false;
const STREAM_ARG_DELTAS: bool = true;

impl Qwen3Parser {
    pub fn new(options: ParserOptions) -> Self {
        Self::with_ids(options, Box::new(RandomIds))
    }

    pub fn with_ids(options: ParserOptions, ids: Box<dyn IdSource>) -> Self {
        let shape: LexerShape = machine::qwen3_lexer_shape(&options.drop_tokens);
        let initial = if options.enable_thinking {
            ParserState::Reasoning
        } else {
            ParserState::Content
        };
        let skip_tool_parsing = options.skip_tool_parsing;
        // `_has_reasoning` in the reference server: THINK_START/THINK_END are always declared for
        // qwen3, so it is true regardless of the initial state.
        let has_reasoning = true;
        let mut machine = Machine::new(shape, initial);
        machine.skip_tool_parsing = skip_tool_parsing;
        Self {
            machine,
            suppress_tool_calls: options.tool_choice == ToolChoice::None
                && !options.tools.is_empty(),
            tools: options.tools,
            slots: Vec::new(),
            deferred_content: String::new(),
            content_has_nonws: false,
            reasoning_ended: !has_reasoning,
            has_reasoning,
            streaming_initialized: false,
            thinking_enabled: initial == ParserState::Reasoning,
            ids,
        }
    }

    pub fn reasoning_ended(&self) -> bool {
        self.reasoning_ended
    }

    fn reset(&mut self, initial_state: Option<ParserState>) {
        self.machine.reset(initial_state);
        self.reasoning_ended = !self.has_reasoning;
        self.slots.clear();
        self.deferred_content.clear();
        self.content_has_nonws = false;
    }

    /// The reference server's `initialize_streaming`: resets once, on the first streaming call.
    fn initialize_streaming(&mut self, initial_state: Option<ParserState>) {
        if !self.streaming_initialized {
            self.streaming_initialized = true;
            self.reset(initial_state);
        }
    }

    // ── Streaming ────────────────────────────────────────────────────────────

    /// One engine step's text. `finished` closes the parse (the reference server's `parse_delta`).
    pub fn parse_delta(&mut self, delta_text: &str, finished: bool) -> Option<DeltaMessage> {
        let mut events = self.machine.feed(delta_text);
        if finished {
            events.extend(self.machine.finish());
        }
        self.events_to_delta(events, finished)
    }

    /// The reference server's `ParserEngineToolAdapter.extract_tool_calls_streaming`: the tool parser
    /// starts in CONTENT state because reasoning has already been split off.
    pub fn extract_tool_calls_streaming(&mut self, delta_text: &str) -> Option<DeltaMessage> {
        self.initialize_streaming(Some(ParserState::Content));
        let events = self.machine.feed(delta_text);
        self.events_to_delta(events, false)
    }

    /// The reference server's `extract_reasoning_streaming` (the reasoning adapter's streaming path).
    pub fn extract_reasoning_streaming(&mut self, delta_text: &str) -> Option<DeltaMessage> {
        self.initialize_streaming(None);
        let events = self.machine.feed(delta_text);
        self.events_to_delta(events, false)
    }

    /// The reference server's `finish_streaming`.
    pub fn finish_streaming(&mut self) -> Option<DeltaMessage> {
        let events = self.machine.finish();
        if !events.is_empty() || !self.deferred_content.is_empty() {
            return self.events_to_delta(events, true);
        }
        None
    }

    // ── Non-streaming ────────────────────────────────────────────────────────

    /// The reference server's `ParserEngine.parse`: `(reasoning, content, tool_calls)` in one pass.
    pub fn parse(&mut self, model_output: &str) -> (Option<String>, Option<String>, Vec<ToolCall>) {
        let (reasoning, content, info) = self.single_pass(model_output, None);
        let tool_calls = if info.tools_called {
            info.tool_calls
        } else {
            Vec::new()
        };
        (reasoning, content, tool_calls)
    }

    /// The reference server's `extract_reasoning`: reasoning split only, tool markup discarded.
    ///
    /// `Qwen3Parser.extract_reasoning` short-circuits when thinking is disabled and
    /// returns the output **verbatim** as content — including the empty string, which
    /// is why the content is `Some("")` rather than `None` there.
    pub fn extract_reasoning(&mut self, model_output: &str) -> (Option<String>, Option<String>) {
        if !self.thinking_enabled {
            return (None, Some(model_output.to_owned()));
        }
        self.reset(None);
        let mut events = self.machine.feed(model_output);
        events.extend(self.machine.finish());

        let mut reasoning = String::new();
        let mut content = String::new();
        for event in &events {
            match event.ty {
                EventType::ReasoningChunk => reasoning.push_str(&event.value),
                EventType::TextChunk => content.push_str(&event.value),
                EventType::ReasoningEnd => self.reasoning_ended = true,
                _ => {}
            }
        }
        if STRIP_TRAILING_REASONING_WS {
            reasoning = reasoning.trim_end().to_owned();
        }
        (none_if_empty(reasoning), none_if_empty(content))
    }

    /// The reference server's `ParserEngineToolAdapter.extract_tool_calls`, i.e. what the registered
    /// `qwen3_coder` parser does: parse reasoning-stripped content from CONTENT state.
    pub fn extract_tool_calls(&mut self, content: &str) -> ExtractedToolCalls {
        let (_, parsed_content, mut info) = self.single_pass(content, Some(ParserState::Content));
        if parsed_content.is_some() && info.content.is_none() {
            info.content = parsed_content;
        }
        info
    }

    fn single_pass(
        &mut self,
        text: &str,
        initial_state: Option<ParserState>,
    ) -> (Option<String>, Option<String>, ExtractedToolCalls) {
        self.reset(initial_state);
        let mut events = self.machine.feed(text);
        events.extend(self.machine.finish());

        let delta = self.events_to_delta(events, true);
        let info = self.build_extracted_result(&[]);

        let mut reasoning = delta.as_ref().and_then(|d| d.reasoning.clone());
        if STRIP_TRAILING_REASONING_WS {
            reasoning = reasoning.and_then(|r| none_if_empty(r.trim_end().to_owned()));
        }
        let content = delta
            .as_ref()
            .and_then(|d| d.content.clone())
            .and_then(|c| self.strip_content_whitespace(&c, info.tools_called));
        (reasoning, content, info)
    }

    // ── Events → delta ───────────────────────────────────────────────────────

    fn events_to_delta(
        &mut self,
        events: Vec<SemanticEvent>,
        finished: bool,
    ) -> Option<DeltaMessage> {
        if events.is_empty() && self.deferred_content.is_empty() {
            return None;
        }
        let mut tool_call_deltas: Vec<DeltaToolCall> = Vec::new();
        let mut content_parts: Vec<String> = Vec::new();
        let mut reasoning_parts: Vec<String> = Vec::new();
        let mut seen_tool_event = false;
        let suppress = self.suppress_tool_calls;

        for event in events {
            match event.ty {
                EventType::TextChunk => {
                    if seen_tool_event {
                        self.deferred_content.push_str(&event.value);
                    } else {
                        content_parts.push(event.value);
                    }
                }
                EventType::ReasoningChunk => reasoning_parts.push(event.value),
                EventType::ReasoningEnd => self.reasoning_ended = true,
                EventType::ToolCallStart => {
                    if !suppress {
                        seen_tool_event = true;
                        self.ensure_slot(event.tool_index);
                    }
                }
                EventType::ToolName => {
                    if !suppress {
                        seen_tool_event = true;
                        if let Some(slot) = self.slot_mut(event.tool_index) {
                            slot.name.push_str(&event.value);
                        }
                    }
                }
                EventType::ArgValueChunk => {
                    if !suppress {
                        seen_tool_event = true;
                        self.handle_arg_chunk(&event, &mut tool_call_deltas);
                    }
                }
                EventType::ToolCallEnd => {
                    if !suppress {
                        seen_tool_event = true;
                        self.handle_tool_end(&event, &mut tool_call_deltas);
                    }
                }
            }
        }

        if tool_call_deltas.len() > 1 {
            tool_call_deltas = coalesce_tool_call_deltas(tool_call_deltas);
        }

        if !self.deferred_content.is_empty() && (!seen_tool_event || tool_call_deltas.is_empty()) {
            content_parts.insert(0, std::mem::take(&mut self.deferred_content));
        }

        let mut content_str: String = content_parts.concat();

        if self.content_has_nonws {
            // Already past the whitespace-only prologue: pass everything through.
        } else if !content_str.is_empty() {
            if !content_str.trim().is_empty() {
                self.content_has_nonws = true;
            } else if !self.slots.is_empty() {
                if DROP_WS_ONLY_CONTENT_BEFORE_TOOLS {
                    content_str.clear();
                }
            } else if !finished {
                self.deferred_content = std::mem::take(&mut content_str);
            }
        }

        let content = none_if_empty(content_str);
        let reasoning = none_if_empty(reasoning_parts.concat());

        if content.is_some() || !tool_call_deltas.is_empty() || reasoning.is_some() {
            Some(DeltaMessage {
                content,
                reasoning,
                tool_calls: tool_call_deltas,
            })
        } else {
            None
        }
    }

    fn ensure_slot(&mut self, idx: i32) {
        if idx < 0 {
            return;
        }
        while self.slots.len() <= idx as usize {
            self.slots.push(ToolCallSlot::default());
        }
    }

    fn slot_mut(&mut self, idx: i32) -> Option<&mut ToolCallSlot> {
        if idx < 0 {
            return None;
        }
        self.slots.get_mut(idx as usize)
    }

    fn ensure_tool_id(&mut self, idx: usize) {
        if self.slots[idx].id.is_empty() {
            self.slots[idx].id = self.ids.tool_call_id();
        }
    }

    fn accept_tool_name(&self, name: &str) -> bool {
        if name.is_empty() {
            return false;
        }
        if !VALIDATE_TOOL_NAMES || self.tools.is_empty() {
            return true;
        }
        find_tool(&self.tools, name).is_some()
    }

    fn emit_name_delta(&mut self, idx: usize, deltas: &mut Vec<DeltaToolCall>, name: &str) {
        if !self.accept_tool_name(name) {
            return;
        }
        let string_keys = streamable_string_keys(find_tool_properties(&self.tools, name));
        {
            let slot = &mut self.slots[idx];
            slot.name = name.to_owned();
            slot.name_sent = true;
            slot.string_keys = string_keys;
        }
        self.ensure_tool_id(idx);
        deltas.push(DeltaToolCall {
            index: idx as i32,
            id: Some(self.slots[idx].id.clone()),
            kind: Some("function"),
            name: Some(name.to_owned()),
            arguments: None,
        });
    }

    fn handle_arg_chunk(&mut self, event: &SemanticEvent, deltas: &mut Vec<DeltaToolCall>) {
        let idx = match usize::try_from(event.tool_index) {
            Ok(idx) if idx < self.slots.len() => idx,
            _ => return,
        };
        if !event.value.is_empty() {
            self.slots[idx].args.push_str(&event.value);
        }

        if !self.slots[idx].name_sent {
            if !self.slots[idx].name.is_empty() {
                let name = self.slots[idx].name.clone();
                self.emit_name_delta(idx, deltas, &name);
            } else if !event.value.is_empty() {
                // The name is not known yet — a raw JSON body may still carry it.
                if let Some(name) = try_extract_name(&self.slots[idx].args) {
                    self.emit_name_delta(idx, deltas, &name);
                }
            }
        } else if !event.value.is_empty() {
            if let Some(arg_delta) = self.compute_arg_delta(idx) {
                deltas.push(DeltaToolCall {
                    index: idx as i32,
                    id: None,
                    kind: None,
                    name: None,
                    arguments: Some(arg_delta),
                });
            }
        }
    }

    fn handle_tool_end(&mut self, event: &SemanticEvent, deltas: &mut Vec<DeltaToolCall>) {
        let idx = match usize::try_from(event.tool_index) {
            Ok(idx) if idx < self.slots.len() => idx,
            _ => return,
        };
        let mut remaining = self.flush_arg_converter(idx);

        if !self.slots[idx].name_sent {
            let name = if self.slots[idx].name.is_empty() {
                try_extract_name(&self.slots[idx].args).unwrap_or_default()
            } else {
                self.slots[idx].name.clone()
            };
            if self.accept_tool_name(&name) {
                let string_keys = streamable_string_keys(find_tool_properties(&self.tools, &name));
                {
                    let slot = &mut self.slots[idx];
                    slot.name = name.clone();
                    slot.name_sent = true;
                    slot.string_keys = string_keys;
                }
                self.ensure_tool_id(idx);
                deltas.push(DeltaToolCall {
                    index: idx as i32,
                    id: Some(self.slots[idx].id.clone()),
                    kind: Some("function"),
                    name: Some(name),
                    arguments: Some(remaining.clone().unwrap_or_default()),
                });
                remaining = None;
            }
        }

        if let Some(remaining) = remaining {
            if self.slots[idx].name_sent {
                deltas.push(DeltaToolCall {
                    index: idx as i32,
                    id: None,
                    kind: None,
                    name: None,
                    arguments: Some(remaining),
                });
            }
        }
    }

    // ── Argument conversion ──────────────────────────────────────────────────

    fn compute_arg_delta(&mut self, idx: usize) -> Option<String> {
        if !STREAM_ARG_DELTAS {
            return None;
        }
        let mut current_json = arg_converter(&self.slots[idx].args, true);
        if current_json.is_empty() {
            return None;
        }
        if !self.slots[idx].name.is_empty() {
            let name = self.slots[idx].name.clone();
            current_json = self.fix_arg_types(&current_json, &name);
        }
        let prev = self.slots[idx].streamed_json.clone();
        let safe_json = safe_arg_prefix(&current_json, self.slots[idx].string_keys.as_ref());

        if safe_json.is_empty() || safe_json == prev {
            return None;
        }
        let diff = if prev.is_empty() {
            safe_json.clone()
        } else {
            if !safe_json.starts_with(&prev) {
                return None;
            }
            safe_json[prev.len()..].to_owned()
        };
        if diff.is_empty() {
            return None;
        }
        self.slots[idx].streamed_json = safe_json;
        Some(diff)
    }

    fn flush_arg_converter(&mut self, idx: usize) -> Option<String> {
        let mut final_json = arg_converter(&self.slots[idx].args, false);
        if !final_json.is_empty() {
            let name = self.slots[idx].name.clone();
            final_json = self.fix_arg_types(&final_json, &name);
        }
        let prev = self.slots[idx].streamed_json.clone();
        if !final_json.is_empty() && final_json.len() > prev.len() {
            if !prev.is_empty() && !final_json.starts_with(&prev) {
                return None;
            }
            let diff = final_json[prev.len()..].to_owned();
            self.slots[idx].streamed_json = final_json;
            return Some(diff);
        }
        None
    }

    /// The reference server's `_fix_arg_types`: coerce string values to the schema's types.
    fn fix_arg_types(&self, args_json: &str, func_name: &str) -> String {
        if self.tools.is_empty() || func_name.is_empty() {
            return args_json.to_owned();
        }
        let Some(PyVal::Obj(entries)) = pyjson::loads(args_json) else {
            return args_json.to_owned();
        };
        let Some(properties) = find_tool_properties(&self.tools, func_name) else {
            return args_json.to_owned();
        };
        let mut args = entries;
        let changed = coerce_dict(&mut args, properties);
        if changed {
            PyVal::Obj(args).dumps()
        } else {
            args_json.to_owned()
        }
    }

    // ── Final extraction ─────────────────────────────────────────────────────

    /// The reference server's `_build_extracted_result`, including the baked #47137 fix: when a call
    /// is truncated mid-parameter, emit exactly what streaming already sent.
    fn build_extracted_result(&mut self, deltas: &[Option<DeltaMessage>]) -> ExtractedToolCalls {
        let mut content_parts: Vec<String> = Vec::new();
        for delta in deltas.iter().flatten() {
            if let Some(content) = &delta.content {
                if !content.is_empty() {
                    content_parts.push(content.clone());
                }
            }
        }

        let mut tool_calls: Vec<ToolCall> = Vec::new();
        for idx in 0..self.slots.len() {
            if self.slots[idx].name.is_empty() && self.slots[idx].args.is_empty() {
                continue;
            }
            let mut name = self.slots[idx].name.trim().to_owned();
            let raw_body = self.slots[idx].args.clone();

            let mut args_json;
            if name.is_empty() && !raw_body.trim().is_empty() {
                let (extracted_name, extracted_args) = extract_name_and_args(&raw_body);
                name = extracted_name;
                args_json = extracted_args;
            } else if !raw_body.trim().is_empty() {
                args_json = arg_converter(&raw_body, false);
                // #47137: a truncated body parses further with `partial=True`; when it
                // does, the streamed arguments are the agreed answer.
                let partial_json = arg_converter(&raw_body, true);
                if partial_json != args_json && !self.slots[idx].streamed_json.is_empty() {
                    args_json = self.slots[idx].streamed_json.clone();
                }
            } else {
                args_json = "{}".to_owned();
            }

            if self.accept_tool_name(&name) {
                self.ensure_tool_id(idx);
                let args_json = self.fix_arg_types(&args_json, &name);
                tool_calls.push(ToolCall {
                    id: self.slots[idx].id.clone(),
                    name,
                    arguments: args_json,
                });
            }
        }

        let content_str = content_parts.concat();
        let content = self.strip_content_whitespace(&content_str, !tool_calls.is_empty());
        ExtractedToolCalls {
            tools_called: !tool_calls.is_empty(),
            tool_calls,
            content,
        }
    }

    fn strip_content_whitespace(&self, content: &str, tools_called: bool) -> Option<String> {
        let mut content = content.to_owned();
        if tools_called {
            if STRIP_CONTENT_WS_WITH_TOOLS {
                content = content.trim().to_owned();
            } else if DROP_WS_ONLY_CONTENT_BEFORE_TOOLS && content.trim().is_empty() {
                content.clear();
            }
        }
        none_if_empty(content)
    }
}

fn none_if_empty(s: String) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// The reference server's `_coalesce_tool_call_deltas`: one entry per index.
fn coalesce_tool_call_deltas(deltas: Vec<DeltaToolCall>) -> Vec<DeltaToolCall> {
    // The reference server returns the input untouched when nothing merged; the merged list is then
    // identical in order and contents, so this returns it unconditionally.
    let mut merged: Vec<DeltaToolCall> = Vec::new();
    for tc in deltas {
        match merged.iter_mut().find(|m| m.index == tc.index) {
            None => merged.push(tc),
            Some(existing) => {
                if tc.id.is_some() && existing.id.is_none() {
                    existing.id = tc.id;
                }
                if tc.kind.is_some() && existing.kind.is_none() {
                    existing.kind = tc.kind;
                }
                if tc.name.is_some() && existing.name.is_none() {
                    existing.name = tc.name;
                }
                if let Some(args) = tc.arguments {
                    match &mut existing.arguments {
                        None => existing.arguments = Some(args),
                        Some(existing_args) => existing_args.push_str(&args),
                    }
                }
            }
        }
    }
    merged
}

/// `_qwen3_arg_converter`: `<parameter=K>V</parameter>` pairs into a JSON object of
/// strings. `partial` additionally takes the unterminated trailing parameter.
fn arg_converter(raw_args: &str, partial: bool) -> String {
    let mut params = PyDict::new();
    for m in PARAM_RE.captures_iter(raw_args).flatten() {
        let name = m.get(1).map(|g| g.as_str()).unwrap_or("");
        let value = m.get(2).map(|g| g.as_str()).unwrap_or("");
        params.insert(name, PyVal::Str(trim_wrapping_newlines(value).to_owned()));
    }
    if partial {
        let remaining = PARAM_RE.replace_all(raw_args, "");
        if let Ok(Some(m)) = PARTIAL_PARAM_RE.captures(&remaining) {
            let name = m.get(1).map(|g| g.as_str()).unwrap_or("");
            let value = m.get(2).map(|g| g.as_str()).unwrap_or("");
            if !name.is_empty() {
                params.insert(name, PyVal::Str(trim_wrapping_newlines(value).to_owned()));
            }
        }
    }
    params.into_val().dumps()
}

/// One leading and one trailing newline are template markup, not value.
fn trim_wrapping_newlines(value: &str) -> &str {
    let value = value.strip_prefix('\n').unwrap_or(value);
    value.strip_suffix('\n').unwrap_or(value)
}

fn try_extract_name(args: &str) -> Option<String> {
    let m = NAME_RE.captures(args).ok().flatten()?;
    let name = m.get(1)?.as_str();
    if name.is_empty() {
        None
    } else {
        Some(name.to_owned())
    }
}

/// The reference server's `_extract_name_and_args` for a raw JSON body with no `<function=` tag.
fn extract_name_and_args(raw_body: &str) -> (String, String) {
    let trimmed = raw_body.trim();
    let Some(PyVal::Obj(entries)) = pyjson::loads(trimmed) else {
        return (String::new(), trimmed.to_owned());
    };
    let name = entries
        .iter()
        .find(|(k, _)| k == "name")
        .and_then(|(_, v)| match v {
            PyVal::Str(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_default();
    for key in ["arguments", "parameters"] {
        if let Some((_, value)) = entries.iter().find(|(k, _)| k == key) {
            let args = match value {
                PyVal::Str(s) => s.clone(),
                other => other.dumps(),
            };
            return (name, args);
        }
    }
    let without_name: Vec<(String, PyVal)> =
        entries.into_iter().filter(|(k, _)| k != "name").collect();
    (name, PyVal::Obj(without_name).dumps())
}

/// The reference server's `_safe_arg_prefix`: the prefix of `json_str` that will not change shape as
/// more input arrives.
fn safe_arg_prefix(json_str: &str, string_keys: Option<&HashSet<String>>) -> String {
    let bytes = json_str.as_bytes();
    let mut last_colon: isize = -1;
    let mut last_key: Option<String> = None;
    let mut pending_key: Option<String> = None;
    let mut in_string = false;
    let mut escape = false;
    let mut string_start: isize = -1;
    let mut depth: i32 = 0;

    for (i, c) in json_str.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_string = false;
                if depth == 1 && string_start >= 0 {
                    pending_key = Some(json_str[string_start as usize + 1..i].to_owned());
                }
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                string_start = i as isize;
            }
            '{' | '[' => depth += 1,
            '}' | ']' => depth -= 1,
            ':' if depth == 1 => {
                last_colon = i as isize;
                last_key = pending_key.take();
            }
            _ => {}
        }
    }

    if last_colon < 0 {
        return String::new();
    }
    let mut end = last_colon as usize + 1;
    while end < bytes.len() && matches!(bytes[end], b' ' | b'\t' | b'\n' | b'\r') {
        end += 1;
    }
    if end >= bytes.len() || bytes[end] != b'"' {
        return json_str[..end].to_owned();
    }
    if let Some(keys) = string_keys {
        // With a schema, only fields that stay strings are safe to emit unterminated.
        if !last_key.as_deref().is_some_and(|key| keys.contains(key)) {
            return json_str[..end].to_owned();
        }
    }

    let mut escape = false;
    for (i, c) in json_str[end + 1..].char_indices() {
        if escape {
            escape = false;
            continue;
        }
        match c {
            '\\' => escape = true,
            '"' => return json_str[..end + 1 + i].to_owned(),
            _ => {}
        }
    }
    json_str.to_owned()
}

/// Keys whose schema type is exactly `string`, which may stream unterminated.
fn streamable_string_keys(
    properties: Option<&serde_json::Map<String, Value>>,
) -> Option<HashSet<String>> {
    let properties = properties?;
    if properties.is_empty() {
        return None;
    }
    let mut out = HashSet::new();
    for (key, schema) in properties {
        let types = extract_types_from_schema(Some(schema));
        if types.len() == 1 && types.contains("string") {
            out.insert(key.clone());
        }
    }
    Some(out)
}

fn find_tool<'a>(tools: &'a [Value], name: &str) -> Option<&'a Value> {
    tools.iter().find(|tool| {
        tool.get("function")
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            == Some(name)
    })
}

fn find_tool_properties<'a>(
    tools: &'a [Value],
    name: &str,
) -> Option<&'a serde_json::Map<String, Value>> {
    let props = find_tool(tools, name)?
        .get("function")?
        .get("parameters")?
        .get("properties")?
        .as_object()?;
    if props.is_empty() {
        None
    } else {
        Some(props)
    }
}

/// The reference server's `extract_types_from_schema`: `type`, `enum` inference and `anyOf`/`oneOf`/
/// `allOf`, defaulting to `string`.
fn extract_types_from_schema(schema: Option<&Value>) -> HashSet<String> {
    let mut types: HashSet<String> = HashSet::new();
    let Some(Value::Object(schema)) = schema else {
        return HashSet::from(["string".to_owned()]);
    };
    match schema.get("type") {
        Some(Value::String(t)) => {
            types.insert(t.clone());
        }
        Some(Value::Array(items)) => {
            for item in items {
                if let Value::String(t) = item {
                    types.insert(t.clone());
                }
            }
        }
        _ => {}
    }
    if let Some(Value::Array(values)) = schema.get("enum") {
        for value in values {
            let name = match value {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
                Value::Number(_) => "number",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
            };
            types.insert(name.to_owned());
        }
    }
    for field in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(choices)) = schema.get(field) {
            for choice in choices {
                types.extend(extract_types_from_schema(Some(choice)));
            }
        }
    }
    if types.is_empty() {
        HashSet::from(["string".to_owned()])
    } else {
        types
    }
}

const TYPE_ALIASES: &[(&str, &str)] = &[
    ("str", "string"),
    ("text", "string"),
    ("varchar", "string"),
    ("char", "string"),
    ("enum", "string"),
    ("int", "integer"),
    ("int32", "integer"),
    ("int64", "integer"),
    ("uint", "integer"),
    ("uint32", "integer"),
    ("uint64", "integer"),
    ("long", "integer"),
    ("short", "integer"),
    ("unsigned", "integer"),
    ("float", "number"),
    ("float32", "number"),
    ("float64", "number"),
    ("double", "number"),
    ("bool", "boolean"),
    ("dict", "object"),
    ("arr", "array"),
    ("list", "array"),
    ("sequence", "array"),
];

/// The reference server's `coerce_to_schema_type`. `None` means "unchanged", which is how the
/// original signals identity (`coerced is not value`).
fn coerce_to_schema_type(value: &str, types: &HashSet<String>) -> Option<PyVal> {
    let normalized: HashSet<String> = types
        .iter()
        .map(|t| {
            let key = t.trim().to_lowercase();
            TYPE_ALIASES
                .iter()
                .find(|(alias, _)| *alias == key)
                .map(|(_, canonical)| (*canonical).to_owned())
                .unwrap_or(key)
        })
        .collect();

    for candidate in [
        "null", "integer", "number", "boolean", "object", "array", "string",
    ] {
        if !normalized.contains(candidate) {
            continue;
        }
        match candidate {
            "null" => {
                if value.to_lowercase() == "null" {
                    return Some(PyVal::Null);
                }
            }
            "string" => return None, // identity: the value is already a string
            "integer" => {
                if let Some(digits) = pyjson::parse_python_int(value) {
                    return Some(PyVal::Int(digits));
                }
            }
            "number" => {
                if let Some(v) = pyjson::parse_python_float(value) {
                    if !v.is_finite() {
                        continue;
                    }
                    // `val if val != int(val) else int(val)`
                    if v.fract() == 0.0 && v.abs() < 9.007_199_254_740_992e15 {
                        return Some(PyVal::int(v as i64));
                    }
                    if v.fract() == 0.0 {
                        // Python's `int(float)` is exact for any magnitude.
                        return Some(PyVal::Int(exact_int_from_f64(v)));
                    }
                    return Some(PyVal::Float(v));
                }
            }
            "boolean" => {
                let lower = value.to_lowercase();
                let lower = lower.trim();
                if lower == "true" || lower == "1" {
                    return Some(PyVal::Bool(true));
                }
                if lower == "false" || lower == "0" {
                    return Some(PyVal::Bool(false));
                }
            }
            "object" | "array" => {
                if let Some(parsed) = pyjson::loads(value) {
                    if parsed.is_json_finite() {
                        return Some(parsed);
                    }
                }
            }
            _ => {}
        }
    }

    match pyjson::loads(value) {
        Some(parsed) if parsed.is_json_finite() => Some(parsed),
        _ => None,
    }
}

/// `int(float)` in Python is exact at any magnitude; `as i64` is not.
fn exact_int_from_f64(v: f64) -> String {
    // Every finite f64 with no fractional part is an exact integer; build it from the
    // mantissa and exponent with big-decimal doubling rather than losing digits.
    let negative = v < 0.0;
    let mut mantissa = v.abs();
    let mut exponent = 0i32;
    while mantissa >= 9.007_199_254_740_992e15 {
        mantissa /= 2.0;
        exponent += 1;
    }
    let mut digits: Vec<u8> = mantissa.to_string().bytes().map(|b| b - b'0').collect();
    for _ in 0..exponent {
        // digits *= 2
        let mut carry = 0u8;
        for d in digits.iter_mut().rev() {
            let value = *d * 2 + carry;
            *d = value % 10;
            carry = value / 10;
        }
        if carry > 0 {
            digits.insert(0, carry);
        }
    }
    let text: String = digits.iter().map(|d| (d + b'0') as char).collect();
    if negative {
        format!("-{text}")
    } else {
        text
    }
}

/// The reference server's `_coerce_value` + `_coerce_dict`, mutating in place and reporting change.
fn coerce_dict(args: &mut [(String, PyVal)], properties: &serde_json::Map<String, Value>) -> bool {
    let mut changed = false;
    for (key, value) in args.iter_mut() {
        let Some(schema) = properties.get(key.as_str()) else {
            continue;
        };
        if !schema.is_object() {
            continue;
        }
        if coerce_value(value, schema) {
            changed = true;
        }
    }
    changed
}

fn coerce_value(value: &mut PyVal, schema: &Value) -> bool {
    match value {
        PyVal::Str(s) => {
            let types = extract_types_from_schema(Some(schema));
            match coerce_to_schema_type(s, &types) {
                Some(coerced) => {
                    *value = coerced;
                    true
                }
                None => false,
            }
        }
        PyVal::Obj(entries) => match schema.get("properties").and_then(Value::as_object) {
            Some(nested) => coerce_dict(entries, nested),
            None => false,
        },
        PyVal::Arr(items) => match schema.get("items") {
            Some(items_schema) if items_schema.is_object() => {
                let mut changed = false;
                for item in items.iter_mut() {
                    if coerce_value(item, items_schema) {
                        changed = true;
                    }
                }
                changed
            }
            _ => false,
        },
        other => {
            let types = extract_types_from_schema(Some(schema));
            let as_str = other.dumps();
            // `type(coerced) is not type(value) or coerced != value`
            match coerce_to_schema_type(&as_str, &types) {
                Some(coerced)
                    if std::mem::discriminant(&coerced) != std::mem::discriminant(other)
                        || coerced != *other =>
                {
                    *other = coerced;
                    true
                }
                _ => false,
            }
        }
    }
}
