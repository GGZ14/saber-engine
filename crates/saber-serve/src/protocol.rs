//! OpenAI wire types, shaped against the bytes production actually emits.
//!
//! Every field here was read off the live prod server (the reference 0.27.1, unit
//! `qwen_prod_nvfp4`) rather than from the OpenAI docs, because clients on this box
//! are written against *that*: BetterBench reads `usage.completion_tokens` and the
//! per-chunk `delta.content` / `delta.reasoning`; `ppl.py` and `kld.py` read
//! `prompt_logprobs`; pi-bench reads `tool_calls`.
//!
//! Two details that look like mistakes and are not:
//!
//! * the non-streaming `usage` carries `prompt_tokens_details: null`, while the
//!   streaming usage chunk omits the key entirely — prod does exactly that;
//! * the response field is **`reasoning`**, not `reasoning_content` (the reference 0.26+).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

// ── errors ──────────────────────────────────────────────────────────────────────

/// The error envelope prod returns, e.g.
/// `{"error":{"message":"temperature must be non-negative, got -1.0. (parameter=temperature, value=-1.0)","type":"BadRequestError","param":"temperature","code":400}}`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub param: Option<String>,
    pub code: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiErrorEnvelope {
    pub error: ApiErrorBody,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    pub body: ApiErrorBody,
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>, param: Option<&str>) -> Self {
        Self {
            body: ApiErrorBody {
                message: message.into(),
                kind: "BadRequestError".to_owned(),
                param: param.map(str::to_owned),
                code: 400,
            },
        }
    }

    /// A parameter rejection in prod's exact format, message and suffix alike.
    pub fn invalid_param(
        message: impl Into<String>,
        param: &str,
        value: impl std::fmt::Display,
    ) -> Self {
        let message = format!("{} (parameter={param}, value={value})", message.into());
        Self::bad_request(message, Some(param))
    }

    /// The OTHER 400 prod returns, and it is not the same body.
    ///
    /// A request whose SHAPE does not match `ChatCompletionRequest`'s types never reaches the
    /// handler: FastAPI raises `RequestValidationError` and `validation_exception_handler` builds
    /// the body, so `type` is `HTTPStatus.BAD_REQUEST.phrase` — the string **`Bad Request`**, not
    /// `BadRequestError` — and `param` is the first pydantic error's `loc`, dotted. A client that
    /// keys on `type` sees the difference, so [`crate::reqcheck`] keeps it.
    pub fn validation(message: impl Into<String>, param: Option<&str>) -> Self {
        Self {
            body: ApiErrorBody {
                message: message.into(),
                kind: "Bad Request".to_owned(),
                param: param.map(str::to_owned),
                code: 400,
            },
        }
    }

    pub fn not_found(message: impl Into<String>, param: Option<&str>) -> Self {
        Self {
            body: ApiErrorBody {
                message: message.into(),
                kind: "NotFoundError".to_owned(),
                param: param.map(str::to_owned),
                code: 404,
            },
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            body: ApiErrorBody {
                message: message.into(),
                kind: "InternalServerError".to_owned(),
                param: None,
                code: 500,
            },
        }
    }

    /// FN73: an engine-ended request after a fatal (a failed rank) is a 503 with the reason;
    /// before one it stays the 500 prod answers an engine exception with.
    pub fn engine_ended(message: impl Into<String>) -> Self {
        if crate::fatal::get().is_some() {
            Self {
                body: ApiErrorBody {
                    message: message.into(),
                    kind: "ServiceUnavailableError".to_owned(),
                    param: None,
                    code: 503,
                },
            }
        } else {
            Self::internal(message)
        }
    }

    pub fn status(&self) -> axum::http::StatusCode {
        axum::http::StatusCode::from_u16(self.body.code)
            .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR)
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.body.message)
    }
}

impl std::error::Error for ApiError {}

impl axum::response::IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let status = self.status();
        let envelope = ApiErrorEnvelope { error: self.body };
        (status, axum::Json(envelope)).into_response()
    }
}

// ── requests ────────────────────────────────────────────────────────────────────

/// `stop` accepts a string or a list of strings.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum StringOrVec {
    One(String),
    Many(Vec<String>),
}

impl StringOrVec {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StringOrVec::One(s) => vec![s],
            StringOrVec::Many(v) => v,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

/// `/v1/chat/completions` request. Unknown fields are ignored, as the reference server does.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    /// Messages stay as JSON: the chat template duck-types every field it touches, so
    /// a typed struct would silently drop anything that changes the rendered bytes.
    #[serde(default)]
    pub messages: Vec<Value>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    /// The reference server's `skip_special_tokens` (default `true`). It reaches the detokenizer only when the
    /// server runs NO parser: every engine adapter's `adjust_request` overrides it to `false` (V9).
    #[serde(default)]
    pub skip_special_tokens: Option<bool>,
    #[serde(default)]
    pub max_tokens: Option<i64>,
    #[serde(default)]
    pub max_completion_tokens: Option<i64>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<i64>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub n: Option<i64>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub stop: Option<StringOrVec>,
    /// The reference server extension: ids that end generation, merged with the server's own.
    #[serde(default)]
    pub stop_token_ids: Option<Vec<crate::engine::TokenId>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<i64>,
    /// The reference server extension; `ppl.py` and `kld.py` depend on it.
    #[serde(default)]
    pub prompt_logprobs: Option<i64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub ignore_eos: Option<bool>,
    #[serde(default)]
    pub include_stop_str_in_output: Option<bool>,
    #[serde(default)]
    pub add_generation_prompt: Option<bool>,
    /// `enable_thinking`, `reasoning_effort`, … — forwarded to the renderer.
    #[serde(default)]
    pub chat_template_kwargs: Option<Map<String, Value>>,
    /// OpenAI's / the reference server's top-level `reasoning_effort` (FN14: Flash-Next's template reads
    /// `xhigh | medium | low`): forwarded to the renderer as the `reasoning_effort` kwarg unless
    /// `chat_template_kwargs` already names one. A template that does not read it ignores it.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// The reference server's `return_token_ids` extension on chat (D5): the generated ids on each choice (and,
    /// streamed, on each chunk) and the prompt's ids on the response (the stream's first chunk).
    #[serde(default)]
    pub return_token_ids: Option<bool>,
}

/// `/v1/completions` request (the subset prod's harnesses use).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub prompt: Value,
    #[serde(default)]
    pub max_tokens: Option<i64>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<i64>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub n: Option<i64>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub stop: Option<StringOrVec>,
    #[serde(default)]
    pub stop_token_ids: Option<Vec<crate::engine::TokenId>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub logprobs: Option<i64>,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub prompt_logprobs: Option<i64>,
    #[serde(default)]
    pub echo: Option<bool>,
    #[serde(default)]
    pub ignore_eos: Option<bool>,
    /// The reference server's `return_token_ids` extension: put the generated ids on the choice.
    ///
    /// The reference capture (`tools/capture/capture_nospec.py`) asks prod for it and records the
    /// id, so a gate that compares Saber's serve path with that reference has to be able to
    /// answer the same question — otherwise the serve path can only be compared on TEXT, which is
    /// a different quantity from the one every other token gate uses.
    #[serde(default)]
    pub return_token_ids: Option<bool>,
}

// ── responses ───────────────────────────────────────────────────────────────────

/// Non-streaming `usage`: `prompt_tokens_details` is present and null, as in prod.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub total_tokens: u64,
    pub completion_tokens: u64,
    pub prompt_tokens_details: Option<Value>,
}

impl Usage {
    pub fn new(prompt_tokens: u64, completion_tokens: u64) -> Self {
        Self {
            prompt_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            completion_tokens,
            prompt_tokens_details: None,
        }
    }
}

/// Streaming `usage`: prod omits `prompt_tokens_details` on this one.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamUsage {
    pub prompt_tokens: u64,
    pub total_tokens: u64,
    pub completion_tokens: u64,
}

impl From<Usage> for StreamUsage {
    fn from(u: Usage) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens,
            total_tokens: u.total_tokens,
            completion_tokens: u.completion_tokens,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallWire {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCallWire,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCallWire {
    pub name: String,
    pub arguments: String,
}

/// The assistant message of a non-streaming response, field for field as prod emits
/// it (`tool_calls` is the only key prod omits when absent).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseMessage {
    pub role: String,
    pub content: Option<String>,
    pub refusal: Option<Value>,
    pub annotations: Option<Value>,
    pub audio: Option<Value>,
    pub function_call: Option<Value>,
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallWire>>,
}

impl Default for ResponseMessage {
    fn default() -> Self {
        Self {
            role: "assistant".to_owned(),
            content: None,
            refusal: None,
            annotations: None,
            audio: None,
            function_call: None,
            reasoning: None,
            tool_calls: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LogprobEntry {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<Vec<LogprobEntry>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatLogprobs {
    pub content: Vec<LogprobEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ResponseMessage,
    pub logprobs: Option<ChatLogprobs>,
    pub finish_reason: Option<String>,
    pub stop_reason: Option<Value>,
    /// The reference server's `ChatCompletionResponseChoice.token_ids` (D5): the choice's ids, the stop token
    /// included (prod's detokenizer appends the skipped stop id back), when the request set
    /// `return_token_ids`; `null` otherwise — prod's non-streaming body is a `model_dump()`, so
    /// the key is always there.
    pub token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub service_tier: Option<Value>,
    pub system_fingerprint: String,
    pub usage: Usage,
    /// The reference server extension: present (and null when not requested) on the response object.
    pub prompt_logprobs: Option<Vec<Option<Map<String, Value>>>>,
    /// The reference server extension (D5): the prompt's ids under `return_token_ids`, null otherwise.
    pub prompt_token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DeltaToolCallWire {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    pub function: DeltaFunctionCallWire,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DeltaFunctionCallWire {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// A streaming delta. Absent fields are omitted, exactly as prod does — a client that
/// counts chunks with content must not see `"content": null` on a reasoning chunk.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<DeltaToolCallWire>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub logprobs: Option<ChatLogprobs>,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<Value>,
    /// The reference server's stream choice `token_ids` (D5). Prod serializes chunks with `exclude_unset`, and a
    /// content or finish chunk SETS it (the step's ids under `return_token_ids`, else `None`), so
    /// it is `null` there and absent from the opening chunk: outer `None` = absent,
    /// `Some(None)` = null.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<Option<Vec<u32>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<StreamUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    /// The reference server (D5): set on the stream's FIRST chunk only — the prompt's ids under
    /// `return_token_ids`, `null` otherwise; absent from every later chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_token_ids: Option<Option<Vec<u32>>>,
}

// ── /v1/completions ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CompletionLogprobs {
    pub text_offset: Vec<usize>,
    pub token_logprobs: Vec<f32>,
    pub tokens: Vec<String>,
    pub top_logprobs: Vec<Map<String, Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub logprobs: Option<CompletionLogprobs>,
    pub finish_reason: Option<String>,
    pub stop_reason: Option<Value>,
    pub prompt_logprobs: Option<Vec<Option<Map<String, Value>>>>,
    /// The ids this choice emitted, when the request asked (`return_token_ids`). Absent
    /// otherwise, so a client that did not ask sees the response it always saw.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<Vec<u32>>,
}

/// A streamed `/v1/completions` choice (FN25), the reference server's `CompletionResponseStreamChoice` dumped with
/// `exclude_unset=True`: every field below is SET by the generator, so each is on the wire, `null`
/// when empty.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompletionStreamChoice {
    pub index: u32,
    pub text: String,
    pub logprobs: Option<CompletionLogprobs>,
    pub finish_reason: Option<String>,
    pub stop_reason: Option<Value>,
    /// The prompt's ids on the choice's first chunk under `return_token_ids` (or `echo`).
    pub prompt_token_ids: Option<Vec<u32>>,
    /// This chunk's ids under `return_token_ids`.
    pub token_ids: Option<Vec<u32>>,
}

/// A streamed `/v1/completions` chunk (FN25), the reference server's `CompletionStreamResponse`: the content
/// chunks carry no `usage`; the terminal chunk carries `system_fingerprint` when no usage chunk
/// follows; the usage chunk (`stream_options.include_usage`) has `choices: []`, the usage and the
/// fingerprint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompletionStreamChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionStreamChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<StreamUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub system_fingerprint: String,
    pub usage: Usage,
}

// ── /v1/models ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    pub root: String,
    pub parent: Option<String>,
    pub max_model_len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

/// Engine rejections become HTTP errors with the status a client can act on:
/// capacity is 503 (retry), a prompt over the window is 400 (do not retry).
impl From<crate::engine::EngineError> for ApiError {
    fn from(err: crate::engine::EngineError) -> Self {
        use crate::engine::EngineError as E;
        match err {
            E::AtCapacity | E::ShuttingDown => Self {
                body: ApiErrorBody {
                    message: err.to_string(),
                    kind: "ServiceUnavailableError".to_owned(),
                    param: None,
                    code: 503,
                },
            },
            E::PromptTooLong { .. } => Self::bad_request(err.to_string(), None),
            E::Unsupported { param, message } => Self::bad_request(message, Some(param)),
            E::Other(_) => Self::internal(err.to_string()),
        }
    }
}
