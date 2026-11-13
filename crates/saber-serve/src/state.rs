//! Everything a request handler needs, assembled once at startup.

use std::sync::Arc;

use serde_json::{Map, Value};

use crate::detok::TextCodec;
use crate::engine::Engine;
use crate::metrics::Metrics;
use crate::protocol::ApiError;
use crate::sampling::{DefaultSampling, SamplingLimits};

/// Renders a chat request to the prompt string the model sees.
pub trait ChatRenderer: Send + Sync + std::fmt::Debug {
    fn render(
        &self,
        messages: &[Value],
        tools: Option<&[Value]>,
        add_generation_prompt: bool,
        kwargs: &Map<String, Value>,
    ) -> Result<String, ApiError>;
}

/// The production renderer: `saber-model`'s byte-exact `qwen-fixed-v22.3.jinja`.
#[derive(Debug)]
pub struct TemplateRenderer {
    inner: saber_model::Renderer,
}

impl TemplateRenderer {
    pub fn new(inner: saber_model::Renderer) -> Self {
        Self { inner }
    }

    /// `--chat-template <path>`.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, saber_model::ModelError> {
        Ok(Self::new(saber_model::Renderer::from_file(path)?))
    }
}

impl ChatRenderer for TemplateRenderer {
    fn render(
        &self,
        messages: &[Value],
        tools: Option<&[Value]>,
        add_generation_prompt: bool,
        kwargs: &Map<String, Value>,
    ) -> Result<String, ApiError> {
        let mut kwargs = kwargs.clone();
        kwargs.insert(
            "add_generation_prompt".to_owned(),
            Value::Bool(add_generation_prompt),
        );
        let request = saber_model::ChatRequest::from_template_kwargs(
            messages.to_vec(),
            tools.map(<[Value]>::to_vec),
            &kwargs,
        )
        .map_err(|e| ApiError::bad_request(e.to_string(), None))?;
        self.inner
            .render(&request)
            .map_err(|e| ApiError::bad_request(e.to_string(), None))
    }
}

/// A ChatML renderer with no template file, the counterpart of
/// [`crate::engine::MockEngine`] and [`crate::detok::ByteCodec`]: it exists so the
/// HTTP tests can run without a checkpoint. **Never** the production path — prod
/// parity is defined by the Jinja template, which `saber-model` owns.
#[derive(Debug, Default, Clone, Copy)]
pub struct MinimalChatRenderer;

impl ChatRenderer for MinimalChatRenderer {
    fn render(
        &self,
        messages: &[Value],
        _tools: Option<&[Value]>,
        add_generation_prompt: bool,
        _kwargs: &Map<String, Value>,
    ) -> Result<String, ApiError> {
        let mut out = String::new();
        for message in messages {
            let role = message.get("role").and_then(Value::as_str).ok_or_else(|| {
                ApiError::bad_request("message is missing 'role'", Some("messages"))
            })?;
            let content = match message.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Null) | None => String::new(),
                Some(other) => other.to_string(),
            };
            out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
        if add_generation_prompt {
            out.push_str("<|im_start|>assistant\n");
        }
        Ok(out)
    }
}

/// Every `--tool-call-parser` name prod's image registers (the reference 0.27.1,
/// `ToolParserManager.list_registered()` in `the reference image`, read 2026-09-23).
/// The reference server's own refusal lists them in this order, so the text of an unknown name's error is prod's.
pub const COMPAT_TOOL_PARSERS: &[&str] = &[
    "apertus",
    "cohere_command3",
    "cohere_command4",
    "deepseek_v3",
    "deepseek_v31",
    "deepseek_v32",
    "deepseek_v4",
    "ernie45",
    "functiongemma",
    "gemma4",
    "gigachat3",
    "glm45",
    "glm47",
    "granite",
    "granite-20b-fc",
    "granite4",
    "hermes",
    "hunyuan_a13b",
    "hy_v3",
    "inkling",
    "internlm",
    "jamba",
    "kimi_k2",
    "kimi_k3",
    "lfm2",
    "llama3_json",
    "llama4_json",
    "llama4_pythonic",
    "longcat",
    "mimo",
    "minicpm5",
    "minimax_m2",
    "minimax_m3",
    "mistral",
    "olmo3",
    "openai",
    "phi4_mini_json",
    "poolside_v1",
    "pythonic",
    "qwen3_coder",
    "qwen3_xml",
    "seed_oss",
    "step3",
    "step3p5",
    "xlam",
];

/// Every `--reasoning-parser` name prod's image registers (`ReasoningParserManager.list_registered()`).
pub const COMPAT_REASONING_PARSERS: &[&str] = &[
    "cohere_command3",
    "cohere_command4",
    "deepseek_r1",
    "deepseek_v3",
    "deepseek_v4",
    "ernie45",
    "gemma4",
    "glm45",
    "glm47",
    "granite",
    "holo2",
    "hunyuan_a13b",
    "hy_v3",
    "inkling",
    "kimi_k2",
    "kimi_k3",
    "mimo",
    "minimax_m2",
    "minimax_m2_append_think",
    "minimax_m3",
    "mistral",
    "nemotron_v3",
    "olmo3",
    "openai_gptoss",
    "poolside_v1",
    "qwen3",
    "seed_oss",
    "step3",
    "step3p5",
];

/// `--reasoning-parser`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningParserKind {
    /// The reference server's `Qwen3ParserReasoningAdapter`, registered as `qwen3` AND `mimo`.
    Qwen3,
}

impl ReasoningParserKind {
    /// The kind a registered name resolves to in prod's image, or `None` for a name that is not
    /// one of Saber's (see [`ParserFlags::resolve`] for the refusals' text).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "qwen3" | "mimo" => Some(Self::Qwen3),
            _ => None,
        }
    }

    /// The reference server's class name, for the boot line.
    pub fn class_name(self) -> &'static str {
        match self {
            Self::Qwen3 => "Qwen3ParserReasoningAdapter",
        }
    }
}

/// `--tool-call-parser`. `qwen3_coder`, `qwen3_xml` and `mimo` are ONE class in prod's image
/// (`Qwen3EngineToolParser`), so they are one parser here (see `qwen3-tool-parser-aliases`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolParserKind {
    Qwen3Coder,
}

impl ToolParserKind {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "qwen3_coder" | "qwen3_xml" | "mimo" => Some(Self::Qwen3Coder),
            _ => None,
        }
    }

    pub fn class_name(self) -> &'static str {
        match self {
            Self::Qwen3Coder => "Qwen3EngineToolParser",
        }
    }
}

/// Which defaults an ABSENT parser flag takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParserDefaults {
    /// Prod's unit (`serve-mxfp4.sh:1071`): auto tool choice on, `qwen3_coder`, `qwen3`. Saber's
    /// default, so a boot that names no parser flag serves exactly what it served before V9.
    #[default]
    Prod,
    /// The reference server's own: auto tool choice off, no tool parser, no reasoning parser — what a launcher
    /// written for the reference server's `serve` means when it leaves the flags out.
    Compat,
}

/// `saber serve`'s parser flags as given, before resolution (V9). `None` = the flag was absent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParserFlags {
    /// `--enable-auto-tool-choice` (`Some(true)`) / `--no-enable-auto-tool-choice` (`Some(false)`).
    pub enable_auto_tool_choice: Option<bool>,
    /// `--tool-call-parser NAME`. An empty string is the reference server's "none".
    pub tool_call_parser: Option<String>,
    /// `--reasoning-parser NAME`. An empty string is the reference server's default: no reasoning parser.
    pub reasoning_parser: Option<String>,
    pub defaults: ParserDefaults,
}

/// The flags, resolved: what the server constructs, and what it says at boot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedParsers {
    pub enable_auto_tool_choice: bool,
    /// The tool parser the server CONSTRUCTS — `None` without `--enable-auto-tool-choice`, as in
    /// The reference server (`ParserManager.get_tool_parser` returns before looking the name up).
    pub tool_parser: Option<ToolParserKind>,
    pub reasoning_parser: Option<ReasoningParserKind>,
    /// The names as given (after defaults), for the boot line.
    pub tool_call_parser_name: Option<String>,
    pub reasoning_parser_name: Option<String>,
}

impl ParserFlags {
    /// The reference server's checks, in the reference server's order and with the reference server's text, then Saber's own refusal of a name prod
    /// registers but Saber does not implement:
    ///
    /// 1. `validate_parsed_serve_args`: `--enable-auto-tool-choice` without a tool parser.
    /// 2. `validate_api_server_args`: an unknown tool parser (only with auto tool choice on -- the reference server
    ///    never looks the name up otherwise) and an unknown reasoning parser, each listing prod's
    ///    registered names.
    /// 3. A registered name that is not one of Saber's parsers (e.g. `hermes`): refused by name.
    ///    Serving it with the qwen3 grammar would be a different parser under the requested name.
    pub fn resolve(&self) -> Result<ResolvedParsers, String> {
        let prod = self.defaults == ParserDefaults::Prod;
        let auto = self.enable_auto_tool_choice.unwrap_or(prod);
        let tool_name = self
            .tool_call_parser
            .clone()
            .or_else(|| prod.then(|| "qwen3_coder".to_owned()))
            .filter(|n| !n.is_empty());
        let reasoning_name = self
            .reasoning_parser
            .clone()
            .or_else(|| prod.then(|| "qwen3".to_owned()))
            .filter(|n| !n.is_empty());
        if auto && tool_name.is_none() {
            return Err("Error: --enable-auto-tool-choice requires --tool-call-parser".to_owned());
        }
        if auto {
            let name = tool_name.as_deref().unwrap_or_default();
            if !COMPAT_TOOL_PARSERS.contains(&name) {
                return Err(format!(
                    "invalid tool call parser: {name} (chose from {{ {} }})",
                    COMPAT_TOOL_PARSERS.join(",")
                ));
            }
        }
        if let Some(name) = reasoning_name.as_deref() {
            if !COMPAT_REASONING_PARSERS.contains(&name) {
                return Err(format!(
                    "invalid reasoning parser: {name} (chose from {{ {} }})",
                    COMPAT_REASONING_PARSERS.join(",")
                ));
            }
        }
        let tool_parser = match (auto, tool_name.as_deref()) {
            (true, Some(name)) => Some(ToolParserKind::from_name(name).ok_or_else(|| {
                format!(
                    "--tool-call-parser {name}: registered in the reference 0.27.1 but not implemented by \
                     Saber, which serves only qwen3_coder (= qwen3_xml = mimo, the reference server's \
                     Qwen3EngineToolParser)"
                )
            })?),
            _ => None,
        };
        let reasoning_parser = match reasoning_name.as_deref() {
            Some(name) => Some(ReasoningParserKind::from_name(name).ok_or_else(|| {
                format!(
                    "--reasoning-parser {name}: registered in the reference 0.27.1 but not implemented by \
                     Saber, which serves only qwen3 (= mimo, the reference server's Qwen3ParserReasoningAdapter)"
                )
            })?),
            None => None,
        };
        Ok(ResolvedParsers {
            enable_auto_tool_choice: auto,
            tool_parser,
            reasoning_parser,
            tool_call_parser_name: tool_name,
            reasoning_parser_name: reasoning_name,
        })
    }
}

impl ResolvedParsers {
    /// Prod's profile, resolved.
    pub fn prod() -> Self {
        ParserFlags::default()
            .resolve()
            .expect("prod's parser profile resolves")
    }

    /// The boot line: what was asked for, what it resolved to, and every alias said out loud.
    pub fn banner(&self) -> String {
        let tool = match (&self.tool_call_parser_name, self.tool_parser) {
            (Some(name), Some(kind)) => {
                let alias = if name == "qwen3_coder" {
                    String::new()
                } else {
                    " -- the same parser as qwen3_coder in prod's image".to_owned()
                };
                format!("{name} = {}{alias}", kind.class_name())
            }
            (Some(name), None) => format!(
                "{name} named but INERT: no --enable-auto-tool-choice, so the reference server never constructs \
                 it and a request with tools must say tool_choice \"none\""
            ),
            (None, _) => "none (a request with tools must say tool_choice \"none\")".to_owned(),
        };
        let reasoning = match (&self.reasoning_parser_name, self.reasoning_parser) {
            (Some(name), Some(kind)) => {
                let alias = if name == "qwen3" {
                    String::new()
                } else {
                    " -- the same parser as qwen3 in prod's image".to_owned()
                };
                format!("{name} = {}{alias}", kind.class_name())
            }
            _ => "none (the think block stays in `content`)".to_owned(),
        };
        format!(
            "parsers        auto tool choice {}, tool call parser {tool}, reasoning parser {reasoning}",
            if self.enable_auto_tool_choice { "ON" } else { "off" }
        )
    }
}

/// The assembled server.
#[derive(Debug)]
pub struct ServerState {
    pub engine: Arc<dyn Engine>,
    pub codec: Arc<dyn TextCodec>,
    pub renderer: Arc<dyn ChatRenderer>,
    pub metrics: Arc<Metrics>,
    pub limits: SamplingLimits,
    pub reasoning_parser: Option<ReasoningParserKind>,
    pub tool_parser: Option<ToolParserKind>,
    /// `--enable-auto-tool-choice`. Without it, tool markup is left in `content`.
    pub enable_auto_tool_choice: bool,
    /// Special tokens the parser's lexer drops, from the loaded tokenizer.
    pub drop_tokens: Vec<String>,
    /// `enable_thinking` when the request does not say.
    pub default_enable_thinking: bool,
    /// What a request that sends no `temperature`/`top_p`/`top_k`/`min_p`/
    /// `repetition_penalty` gets — prod's `--override-generation-config`, not OpenAI's
    /// defaults. See [`DefaultSampling`].
    pub default_sampling: DefaultSampling,
    /// `default_sampling_params["stop_token_ids"]` — ids the SERVER adds to every
    /// request's stop set (`to_sampling_params` merges them, request first, dupes
    /// dropped). Prod's generation config carries none, so this is empty there.
    pub default_stop_token_ids: Vec<crate::engine::TokenId>,
    pub system_fingerprint: String,
    /// `created` for `/v1/models`, fixed at startup as the reference server does.
    pub created: u64,
    /// D3: the image policy (`--vision-max-patches`, `--limit-mm-per-prompt`). Text-only by
    /// default: an image part is refused by name.
    pub vision: crate::images::VisionPolicy,
    /// FN25: what an assistant turn's `tool_calls[].function.arguments` reaches the template as.
    /// [`ToolArguments::Verbatim`] (the builder's default) is the plane as it always was; the
    /// 27B's plane builder (`saber_sched::build_state_with_vision`) runs
    /// [`ToolArguments::ObjectsStrict`], prod 0.27.1's `_postprocess_messages` (FN27); the
    /// Flash-Next plane runs [`ToolArguments::Objects`], the reference 0.29's.
    pub tool_arguments: ToolArguments,
    /// FN28: which conversation the template renders. [`ConversationForm::AsSent`] (the
    /// builder's default; the Flash-Next plane) renders each message as sent; the 27B's plane
    /// builder runs [`ConversationForm::Prod0271`], the conversation prod 0.27.1 rebuilds.
    pub conversation_form: ConversationForm,
}

/// FN28 (the user, 2026-09-29 ~19:50 CDT: "ok, align that also for 27b"): the conversation the
/// template renders, applied after the [`ToolArguments`] law and before the template.
///
/// Prod the reference 0.27.1 never renders the request's messages: its request model's `mode="before"`
/// validator renames a message's `reasoning_content` to `reasoning` (`protocol.py` 507-532) and
/// `chat_utils._parse_chat_message_content` (1834-1908) REBUILDS every message from a fixed set of
/// keys -- see `crate::conversation::conversation_0271` for the law, per role, with the line
/// citations. The reference server 0.29's rebuild is the same code (`chat_utils.py` 1966-2040, `protocol.py`
/// 521-547, read in `saber-base:0.29.0-test2`), but the Flash-Next plane keeps `AsSent` until
/// the user says otherwise (its template's content format and served bytes are not this gate's).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ConversationForm {
    /// Each message reaches the template as sent (after the arguments law). FN28's control.
    #[default]
    AsSent,
    /// Prod 0.27.1's rebuilt conversation, in the `openai` content format the 27B's
    /// `qwen-fixed-v22.3.jinja` is detected as (it iterates `message.content`).
    Prod0271,
}

/// FN25: the tool-call `arguments` law of the plane's handler, between the shape check and the
/// template.
///
/// The OpenAI wire type makes `arguments` a JSON **string**; a tool-use chat template reads a
/// MAPPING (Flash-Next's iterates `tool_call.arguments|items`, so a string fails the render and the
/// request 400s). The reference server 0.29 (`chat_utils._postprocess_messages`, read in `saber-base:0.29.0`)
/// bridges the two on every assistant message: an empty `tool_calls` list is dropped; a truthy
/// `arguments` string is `json.loads`ed, and one that is not JSON, or JSON that is not an object,
/// becomes `{}` (logged, not refused -- the malformed turn lives in the history); a falsy one
/// (`""`, `null`, absent) becomes `{}`; a mapping passes as is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ToolArguments {
    /// The 27B plane's law from ws-v5 to FN27 (now the builder's default only): the messages
    /// reach the template as sent, and a non-empty `arguments` that is not JSON is prod 0.27.1's
    /// 400 (`crate::reqcheck`). FN27's control law.
    #[default]
    Verbatim,
    /// The reference server 0.29's `_postprocess_messages`, above (the Flash-Next plane, FN25).
    Objects,
    /// Prod the reference 0.27.1's `_postprocess_messages` (the 27B plane since FN27, on the user's word
    /// "ok, align 27b"): a truthy `arguments` string is `json.loads`ed and the parsed value
    /// replaces it whatever its type (`null` -> `{}`); MALFORMED JSON is 0.27.1's 400 (as under
    /// [`Self::Verbatim`], not 0.29's `{}`); a falsy one becomes `{}`; an empty `tool_calls` list
    /// is dropped. See `crate::reqcheck::tool_arguments_to_objects_0271` for the line citations.
    ObjectsStrict,
}

impl ServerState {
    /// The parsers this server constructs, as `ServeParser` composes them: the tool parser only
    /// with `--enable-auto-tool-choice`, exactly as the reference server's `ParserManager.get_tool_parser`.
    pub fn parser_profile(&self) -> crate::parse::compose::ParserProfile {
        crate::parse::compose::ParserProfile {
            reasoning: self.reasoning_parser,
            tool: if self.enable_auto_tool_choice {
                self.tool_parser
            } else {
                None
            },
        }
    }

    pub fn builder(
        engine: Arc<dyn Engine>,
        codec: Arc<dyn TextCodec>,
        renderer: Arc<dyn ChatRenderer>,
    ) -> ServerStateBuilder {
        ServerStateBuilder::new(engine, codec, renderer)
    }

    /// Model ids this server answers to, `/v1/models` order.
    pub fn model_ids(&self) -> Vec<String> {
        let model = self.engine.model();
        let mut ids = vec![model.served_model_name.clone()];
        ids.extend(model.aliases.iter().cloned());
        ids
    }

    /// Resolve the requested model, 404 exactly as prod does when it is unknown.
    pub fn resolve_model(&self, requested: Option<&str>) -> Result<String, ApiError> {
        let model = self.engine.model();
        match requested {
            None => Ok(model.served_model_name.clone()),
            Some(name) => {
                if self.model_ids().iter().any(|id| id == name) {
                    Ok(name.to_owned())
                } else {
                    Err(ApiError::not_found(
                        format!("The model `{name}` does not exist."),
                        Some("model"),
                    ))
                }
            }
        }
    }
}

/// Builder, so a test can stand the server up with three lines and no checkpoint.
#[derive(Debug)]
pub struct ServerStateBuilder {
    state: ServerState,
}

impl ServerStateBuilder {
    pub fn new(
        engine: Arc<dyn Engine>,
        codec: Arc<dyn TextCodec>,
        renderer: Arc<dyn ChatRenderer>,
    ) -> Self {
        let model_name = engine.model().served_model_name.clone();
        let max_model_len = engine.model().max_model_len;
        Self {
            state: ServerState {
                engine,
                codec,
                renderer,
                metrics: Arc::new(Metrics::new(model_name)),
                limits: SamplingLimits {
                    max_model_len,
                    ..SamplingLimits::default()
                },
                // the reference server's stock generation config until the launcher says otherwise. The
                // orchestrating binary sets `DefaultSampling::PROD` to match the unit.
                default_sampling: DefaultSampling::default(),
                default_stop_token_ids: Vec::new(),
                reasoning_parser: None,
                tool_parser: None,
                enable_auto_tool_choice: false,
                drop_tokens: Vec::new(),
                default_enable_thinking: true,
                system_fingerprint: format!("saber-{}", env!("CARGO_PKG_VERSION")),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                vision: crate::images::VisionPolicy::default(),
                tool_arguments: ToolArguments::Verbatim,
                conversation_form: ConversationForm::AsSent,
            },
        }
    }

    /// FN25: the tool-call `arguments` law (Flash-Next: [`ToolArguments::Objects`]; the 27B:
    /// [`ToolArguments::ObjectsStrict`], FN27).
    #[must_use]
    pub fn tool_arguments(mut self, law: ToolArguments) -> Self {
        self.state.tool_arguments = law;
        self
    }

    /// FN28: the conversation the template renders (the 27B: [`ConversationForm::Prod0271`]).
    #[must_use]
    pub fn conversation_form(mut self, form: ConversationForm) -> Self {
        self.state.conversation_form = form;
        self
    }

    #[must_use]
    pub fn reasoning_parser(mut self, kind: ReasoningParserKind) -> Self {
        self.state.reasoning_parser = Some(kind);
        self
    }

    #[must_use]
    pub fn tool_parser(mut self, kind: ToolParserKind) -> Self {
        self.state.tool_parser = Some(kind);
        self
    }

    #[must_use]
    /// All three parser flags at once, resolved by [`ParserFlags::resolve`].
    pub fn parsers(mut self, resolved: &ResolvedParsers) -> Self {
        self.state.reasoning_parser = resolved.reasoning_parser;
        self.state.tool_parser = resolved.tool_parser;
        self.state.enable_auto_tool_choice = resolved.enable_auto_tool_choice;
        self
    }

    pub fn enable_auto_tool_choice(mut self, enable: bool) -> Self {
        self.state.enable_auto_tool_choice = enable;
        self
    }

    #[must_use]
    pub fn drop_tokens(mut self, tokens: Vec<String>) -> Self {
        self.state.drop_tokens = tokens;
        self
    }

    /// `--max-logprobs`: the cap on `logprobs` and `top_logprobs` (and on `prompt_logprobs` when
    /// it is above [`Self::max_prompt_logprobs`]), checked at validation with the reference
    /// server's 400. `-1` is the whole vocabulary.
    #[must_use]
    pub fn max_logprobs(mut self, max: i32) -> Self {
        self.state.limits.max_logprobs = max;
        self
    }

    /// The floor of the `prompt_logprobs` cap (default
    /// [`crate::sampling::DEFAULT_MAX_PROMPT_LOGPROBS`] = 64, the seam's width). The cap a request
    /// meets is the larger of this and `--max-logprobs`.
    #[must_use]
    pub fn max_prompt_logprobs(mut self, max: i32) -> Self {
        self.state.limits.max_prompt_logprobs = max;
        self
    }

    /// The vocabulary `-1` resolves to (`model_config.get_vocab_size()`).
    #[must_use]
    pub fn vocab_size(mut self, vocab: usize) -> Self {
        self.state.limits.vocab_size = vocab;
        self
    }

    #[must_use]
    pub fn default_enable_thinking(mut self, enable: bool) -> Self {
        self.state.default_enable_thinking = enable;
        self
    }

    /// `--generation-config` / `--override-generation-config`.
    #[must_use]
    pub fn default_sampling(mut self, defaults: DefaultSampling) -> Self {
        self.state.default_sampling = defaults;
        self
    }

    /// Server-side `stop_token_ids`, merged into every request's.
    #[must_use]
    pub fn default_stop_token_ids(mut self, ids: Vec<crate::engine::TokenId>) -> Self {
        self.state.default_stop_token_ids = ids;
        self
    }

    #[must_use]
    pub fn system_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.state.system_fingerprint = fingerprint.into();
        self
    }

    /// D3: serve image parts, the tower's arena holding `max_patches` per image; `limit` images
    /// per prompt (the reference server's `--limit-mm-per-prompt`).
    #[must_use]
    pub fn vision(mut self, max_patches: u32, limit_per_prompt: u32) -> Self {
        self.state.vision.max_patches = (max_patches > 0).then_some(max_patches);
        self.state.vision.limit_per_prompt = limit_per_prompt;
        self
    }

    pub fn build(self) -> Arc<ServerState> {
        Arc::new(self.state)
    }
}
