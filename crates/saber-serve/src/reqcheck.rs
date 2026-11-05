//! The SHAPE of a chat request, refused the way prod refuses it.
//!
//! `SamplingParams::validate` (ws-v1) is the reference server's `_verify_args`: the knobs. This module is the
//! other half, and ws-q1's agentic gate is what found it missing — prod refused **19 of 28**
//! tool-bearing corpus rows with HTTP 400 and Saber served every one of them, so the two stacks
//! were not being compared on the same requests at all. Row 205 is the one that survived
//! normalisation: its `tool_calls[0].function.arguments` is the literal string `not even json`,
//! which is a string (what the API type says) and is not JSON (what prod's handler needs).
//!
//! # Prod's two refusal families, and why they look different on the wire
//!
//! Every message and every status below was read off prod's own code in the prod image
//! (`the reference image`, the reference 0.27.1) and then off a live side
//! container (`tools/gates/request_parity.py`, ws-v5).
//!
//! * **The handler family** — `ValidationError` from a `model_validator`, and the
//!   `json.JSONDecodeError` that `_postprocess_messages` raises. These reach the client through
//!   `create_error_response`, so `type` is **`BadRequestError`**, `param` is the validator's own
//!   `parameter=` (and `null` for the JSON error), and the message is a fixed string. Saber
//!   reproduces these **byte for byte**.
//! * **The pydantic family** — a field that does not match `ChatCompletionRequest`'s types. FastAPI
//!   raises `RequestValidationError` and `validation_exception_handler` builds the body, so `type`
//!   is **`Bad Request`** (`HTTPStatus.BAD_REQUEST.phrase`, not `BadRequestError`), `param` is the
//!   first error's `loc` joined with dots, and the message is `"{n} validation error(s):"` followed
//!   by one Python `repr` of a pydantic error dict per line. Saber reproduces the **status, the
//!   type and the loc** — and not the rest: the line carries a `repr` of the offending input and a
//!   `https://errors.pydantic.dev/2.13/...` URL, and a union-typed field reports one error PER
//!   BRANCH (11 of them for a message with no `role`), which is pydantic's internal enumeration and
//!   not a contract. What is compared against prod is stated in the gate rather than implied here.
//!
//! # The order is prod's order, because it decides which message a doubly-bad request gets
//!
//! 1. the `mode="before"` model validators, in pydantic's **reverse definition order**:
//!    `check_system_message_content_type` (warns only), `check_cache_salt_support`,
//!    `check_generation_prompt`, `check_tool_usage`, `check_structured_outputs_count`,
//!    `check_logprobs`, `validate_stream_options`, `validate_response_format`;
//! 2. field validation — `messages` before `tools`, each in declaration order;
//! 3. the handler: the model check (404), then `_postprocess_messages` (the `arguments` JSON).
//!
//! `SamplingParams::validate` runs **after** all of it, because `to_sampling_params` is called once
//! the prompt has been rendered.
//!
//! # What prod does NOT refuse, which is half of what this module is for
//!
//! Measured, not assumed (`tools/gates/request_parity.py --shapes`): `role: tool` with **no**
//! `tool_call_id` is served; an assistant `tool_calls` entry with `arguments: ""` is served (the
//! handler substitutes `{}`); `tool_calls: []` is served (the handler drops the key); a `content`
//! parts list is served; an unknown role is served; a non-assistant message carrying a malformed
//! `arguments` is served, because only assistant messages reach the JSON decode.

use serde_json::Value;

use crate::parse::pyjson::{self, PyVal};
use crate::protocol::{ApiError, ChatCompletionRequest};
use crate::state::ToolArguments;

/// One segment of a pydantic `loc` tuple.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Loc {
    Field(&'static str),
    Index(usize),
    /// A union branch's model name, which pydantic puts in the `loc` and prod's
    /// `clean_loc_for_param` keeps (it filters bracketed markers, not model names).
    Branch(&'static str),
}

impl Loc {
    fn py(&self) -> String {
        match self {
            Loc::Field(f) => format!("'{f}'"),
            Loc::Branch(b) => format!("'{b}'"),
            Loc::Index(i) => i.to_string(),
        }
    }

    fn dotted(&self) -> String {
        match self {
            Loc::Field(f) => (*f).to_owned(),
            Loc::Branch(b) => (*b).to_owned(),
            Loc::Index(i) => i.to_string(),
        }
    }
}

/// `upstream.entrypoints.serve.utils.server_utils._is_internal_loc_segment`, verbatim: pydantic-core's
/// schema-kind vocabulary, plus anything with a bracket in it.
fn is_internal_loc_segment(segment: &str) -> bool {
    const MARKERS: &[&str] = &[
        "function-wrap",
        "function-after",
        "function-before",
        "function-plain",
        "json-or-python",
        "lax-or-strict",
        "chain",
        "default",
        "nullable",
        "tagged-union",
        "union",
        "call",
        "arguments",
        "is-instance",
        "is-subclass",
        "callable",
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "bytearray",
        "list",
        "tuple",
        "dict",
        "set",
        "frozenset",
        "complex",
        "none",
        "nonetype",
    ];
    if segment.contains(['[', ']', '{', '}', '(', ')']) {
        return true;
    }
    let lower = segment.to_ascii_lowercase();
    MARKERS.contains(&lower.as_str())
}

/// A pydantic-family refusal: one error, in prod's vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeError {
    /// pydantic's error `type`, e.g. `missing`, `string_type`, `dict_type`, `literal_error`.
    pub kind: &'static str,
    /// The `loc` tuple, with FastAPI's `body` prefix already on it.
    pub loc: Vec<Loc>,
    /// pydantic's `msg`.
    pub msg: String,
}

impl ShapeError {
    fn missing(loc: Vec<Loc>) -> Self {
        Self {
            kind: "missing",
            loc,
            msg: "Field required".to_owned(),
        }
    }

    fn string_type(loc: Vec<Loc>) -> Self {
        Self {
            kind: "string_type",
            loc,
            msg: "Input should be a valid string".to_owned(),
        }
    }

    fn dict_type(loc: Vec<Loc>) -> Self {
        Self {
            kind: "dict_type",
            loc,
            msg: "Input should be a valid dictionary".to_owned(),
        }
    }

    fn literal(loc: Vec<Loc>, expected: &str) -> Self {
        Self {
            kind: "literal_error",
            loc,
            msg: format!("Input should be {expected}"),
        }
    }

    /// `clean_loc_for_param`: the dotted path prod puts in `param`.
    ///
    /// **Including its false positive**, because parity is with what prod sends and not with what
    /// prod meant. The filter drops any `loc` segment in pydantic-core's internal schema-kind
    /// vocabulary, and that vocabulary contains `arguments`, `str`, `int`, `list`, `none` … — all
    /// of which are also REAL field names here. So prod's `param` for a tool call whose
    /// `function.arguments` is the wrong type is `…function`, with the field that actually failed
    /// filtered out of it; a client keying on `param` sees `…function`, so that is what this
    /// returns.
    pub fn param(&self) -> String {
        let kept: Vec<String> = self
            .loc
            .iter()
            .map(Loc::dotted)
            .filter(|s| !is_internal_loc_segment(s))
            .collect();
        if kept.is_empty() {
            return self
                .loc
                .iter()
                .map(Loc::dotted)
                .collect::<Vec<_>>()
                .join(".");
        }
        kept.join(".")
    }

    /// The body `validation_exception_handler` builds, in its shape: a count line and one error
    /// per line. The per-error `input` repr and the pydantic docs URL are NOT reproduced — see the
    /// module docs.
    pub fn message(&self) -> String {
        let loc = self.loc.iter().map(Loc::py).collect::<Vec<_>>().join(", ");
        format!(
            "1 validation error:\n  {{'type': '{}', 'loc': ({}), 'msg': '{}'}}",
            self.kind, loc, self.msg
        )
    }

    fn into_api_error(self) -> ApiError {
        let param = self.param();
        ApiError::validation(self.message(), Some(&param))
    }
}

/// `validate_stream_options`, `check_tool_usage` and the field validation of `messages` and
/// `tools`, followed by `_postprocess_messages`' JSON decode — in prod's order.
///
/// # Errors
/// The first refusal, as prod's own body.
pub fn validate_chat_shape(req: &ChatCompletionRequest) -> Result<(), ApiError> {
    validate_chat_shape_for(req, ToolArguments::Verbatim)
}

/// [`validate_chat_shape`] under a plane's tool-call `arguments` law (FN25): the pydantic family
/// and the content parts are the same on both; the handler's JSON decode of `arguments` is prod
/// 0.27.1's 400 under [`ToolArguments::Verbatim`] and the reference 0.29's coercion to `{}` under
/// [`ToolArguments::Objects`] (see [`tool_arguments_to_objects`]), which refuses nothing.
///
/// # Errors
/// The first refusal, as prod's own body.
pub fn validate_chat_shape_for(
    req: &ChatCompletionRequest,
    law: ToolArguments,
) -> Result<(), ApiError> {
    // ---- `mode="before"` model validators, in pydantic's reverse definition order -------------
    check_tool_usage(req)?;
    check_stream_options(req)?;

    // ---- field validation, in the layer order prod reports ------------------------------------
    //
    // `messages` is declared before `tools`, and pydantic COLLECTS every field error rather than
    // stopping at the first, so the one a doubly-bad body is named for is the earlier field's.
    // But a message's `tool_calls` is NOT among them: it is typed `Iterable[...]`, which pydantic
    // validates LAZILY, and the error only appears when `_materialize_tool_calls_after` walks it
    // in the `mode="after"` pass -- after every field. Measured: corpus rows 205, 206, 207, 215
    // and 223 carry a bare `legacy_tool` AND a malformed `tool_calls`, and prod names the TOOL
    // (`body.tools.N.function`) on every one of them.
    if let Some(e) = messages_shape(&req.messages) {
        return Err(e.into_api_error());
    }
    if let Some(tools) = &req.tools {
        if let Some(e) = tools_shape(tools) {
            return Err(e.into_api_error());
        }
    }
    // ---- `mode="after"`: the lazy `tool_calls` iterator, materialised -------------------------
    for msg in &req.messages {
        let Some(map) = msg.as_object() else { continue };
        if let Some(Value::Array(calls)) = map.get("tool_calls") {
            if let Some(e) = tool_calls_shape(calls) {
                return Err(e.into_api_error());
            }
        }
    }

    // ---- the handler: `parse_chat_messages`, which parses the content PARTS of every message
    // before it calls `_postprocess_messages` on the conversation it built ----------------------
    content_parts(&req.messages)?;
    // Every law refuses a custom tool call (0.27.1 1935-1943, 0.29 the same); only prod 0.27.1's
    // laws refuse malformed `arguments` JSON (0.29 makes it `{}`).
    postprocess_messages(
        &req.messages,
        matches!(law, ToolArguments::Verbatim | ToolArguments::ObjectsStrict),
    )
}

/// A tool call of pydantic's custom branch, exactly: `id` a string, `type` `"custom"`, `custom`
/// an object whose `name` and `input` are strings.
fn is_custom_call(call: &serde_json::Map<String, Value>) -> bool {
    call.get("type").and_then(Value::as_str) == Some("custom")
        && call.get("id").is_some_and(Value::is_string)
        && call
            .get("custom")
            .and_then(Value::as_object)
            .is_some_and(|c| {
                c.get("name").is_some_and(Value::is_string)
                    && c.get("input").is_some_and(Value::is_string)
            })
}

/// The reference server 0.29's `chat_utils._postprocess_messages` (FN25, the Flash-Next plane): the messages the
/// template reads, with every assistant turn's tool-call `arguments` a mapping. `None` when no
/// assistant message carries a `tool_calls` list, so a request without one reaches the template
/// as the very same JSON it always did.
///
/// Per assistant message whose `tool_calls` is a list: an empty list is dropped (the template's
/// normal assistant path); per entry, a truthy `arguments` that is a mapping passes, a string is
/// parsed (`serde_json` with `preserve_order`: key order and the LAST of a duplicated key's
/// values, as Python's `dict`), and anything that does not parse to an object -- malformed JSON,
/// `[]`, `42`, a number or list sent bare -- becomes `{}`; a falsy one (`""`, `null`, `0`,
/// `false`, `{}`, `[]`, absent) becomes `{}`. Entries that are not objects, or whose `function`
/// is not one, were refused by the shape check before this runs.
#[must_use]
pub fn tool_arguments_to_objects(messages: &[Value]) -> Option<Vec<Value>> {
    let touches = messages.iter().any(|m| {
        m.get("role").and_then(Value::as_str) == Some("assistant")
            && m.get("tool_calls").is_some_and(Value::is_array)
    });
    if !touches {
        return None;
    }
    let mut out = messages.to_vec();
    for msg in &mut out {
        let Some(map) = msg.as_object_mut() else {
            continue;
        };
        if map.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(Value::Array(calls)) = map.get_mut("tool_calls") else {
            continue;
        };
        if calls.is_empty() {
            map.shift_remove("tool_calls");
            continue;
        }
        for call in calls.iter_mut() {
            let Some(Value::Object(function)) = call.get_mut("function") else {
                continue;
            };
            let parsed = match function.get("arguments") {
                Some(v) if py_truthy(v) => match v {
                    Value::Object(_) => continue,
                    Value::String(s) => match serde_json::from_str::<Value>(s) {
                        Ok(Value::Object(m)) => Value::Object(m),
                        _ => Value::Object(serde_json::Map::new()),
                    },
                    _ => Value::Object(serde_json::Map::new()),
                },
                _ => Value::Object(serde_json::Map::new()),
            };
            function.insert("arguments".to_owned(), parsed);
        }
    }
    Some(out)
}

/// Prod the reference 0.27.1's `chat_utils._postprocess_messages` (FN27, the 27B plane): the messages the
/// template reads, as prod built them. `None` when no assistant message carries a `tool_calls`
/// list, so a request without one reaches the template as the very same JSON it always did.
///
/// Read in prod's image (`the reference image`, the reference 0.27.1,
/// `upstream/entrypoints/chat_utils.py`), line by line:
///
/// * 1918-1921: only an `assistant` message with a `tool_calls` key, and only a list;
/// * 1923-1926: an EMPTY list is dropped (`message.pop("tool_calls")`);
/// * 1929-1943: an entry that is not an object, or whose `type` is not `function` or whose
///   `function` is not an object, is a 400 -- pydantic's wire type refuses every such entry
///   first (`tool_calls_shape`), so it cannot reach this step;
/// * 1946: `if content := function.get("arguments")` -- Python truthiness;
/// * 1947-1949: a truthy value that is not a `dict`/`list` is `json.loads`ed and the PARSED VALUE
///   replaces it WHATEVER ITS TYPE (a mapping, but also `[1, 2]`, `42`, `"text"`, `false`), except
///   `null`, which becomes `{}`; a string that is not JSON raises `JSONDecodeError`, the 400 that
///   [`validate_chat_shape_for`] already returned under [`ToolArguments::ObjectsStrict`] -- so
///   every string reaching this function parses;
/// * 1947 (else): a truthy `dict` or `list` passes as is (pydantic makes `arguments` a string,
///   so this cannot arrive on the wire);
/// * 1950-1951: a falsy value (`""`, `null`, absent) becomes `{}`.
///
/// Where the reference 0.29 ([`tool_arguments_to_objects`]) differs: 0.29 maps malformed JSON AND JSON
/// that is not an object to `{}`; 0.27.1 refuses the first (400) and keeps the second.
///
/// The parsed value is CPython's (`pyjson`: key order, the last of a duplicated key's values in
/// the first one's place, `NaN`/`Infinity`, exact integers). Two values have no `serde_json`
/// form the template can read -- a non-finite float and an integer beyond 64 bits -- and a call
/// whose arguments hold one keeps its string as sent (the pre-FN27 rendering; counted in the
/// FN27 gate as the known edge, not silently rewritten).
#[must_use]
pub fn tool_arguments_to_objects_0271(messages: &[Value]) -> Option<Vec<Value>> {
    let touches = messages.iter().any(|m| {
        m.get("role").and_then(Value::as_str) == Some("assistant")
            && m.get("tool_calls").is_some_and(Value::is_array)
    });
    if !touches {
        return None;
    }
    let mut out = messages.to_vec();
    for msg in &mut out {
        let Some(map) = msg.as_object_mut() else {
            continue;
        };
        if map.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(Value::Array(calls)) = map.get_mut("tool_calls") else {
            continue;
        };
        if calls.is_empty() {
            map.shift_remove("tool_calls");
            continue;
        }
        for call in calls.iter_mut() {
            let Some(Value::Object(function)) = call.get_mut("function") else {
                continue;
            };
            let parsed = match function.get("arguments") {
                Some(v) if py_truthy(v) => match v {
                    Value::Object(_) | Value::Array(_) => continue,
                    Value::String(s) => match pyjson::loads(s) {
                        Some(PyVal::Null) => Value::Object(serde_json::Map::new()),
                        Some(p) => match pyval_to_json(&p) {
                            Some(v) => v,
                            // No serde form (the edge above): the string as sent.
                            None => continue,
                        },
                        // Refused by `validate_chat_shape_for` before this runs; a caller that
                        // skipped it keeps the string rather than inventing a value.
                        None => continue,
                    },
                    // A truthy number or `true`: `json.loads` raises `TypeError` on prod (a
                    // 500); pydantic refuses it first (`string_type`).
                    _ => continue,
                },
                _ => Value::Object(serde_json::Map::new()),
            };
            function.insert("arguments".to_owned(), parsed);
        }
    }
    Some(out)
}

/// A CPython `json.loads` value as the `serde_json` value the template reads: `None` for the two
/// that have none (a non-finite float, an integer outside `i64`/`u64`). Object entries keep
/// CPython's dict semantics: first position, last value.
fn pyval_to_json(v: &PyVal) -> Option<Value> {
    Some(match v {
        PyVal::Null => Value::Null,
        PyVal::Bool(b) => Value::Bool(*b),
        PyVal::Int(digits) => {
            if let Ok(i) = digits.parse::<i64>() {
                Value::from(i)
            } else {
                Value::from(digits.parse::<u64>().ok()?)
            }
        }
        PyVal::Float(f) => Value::Number(serde_json::Number::from_f64(*f)?),
        PyVal::Str(s) => Value::String(s.clone()),
        PyVal::Arr(items) => Value::Array(items.iter().map(pyval_to_json).collect::<Option<_>>()?),
        PyVal::Obj(entries) => {
            let mut m = serde_json::Map::new();
            for (k, v) in entries {
                m.insert(k.clone(), pyval_to_json(v)?);
            }
            Value::Object(m)
        }
    })
}

/// Python truthiness of a JSON value (`if content := function.get("arguments")`).
fn py_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// A content PARTS LIST whose elements are not objects.
///
/// The list is typed `Iterable[ChatCompletionContentPartParam]`, and pydantic validates an
/// `Iterable` LAZILY: the request passes field validation and the error surfaces when the parts
/// are first walked, inside the handler, as a bare `pydantic.ValidationError`. That is a
/// `ValueError`, so `create_error_response` gives it `type: BadRequestError` and `param: null` —
/// the handler family's envelope around pydantic's own rendering.
///
/// Measured on the live side container (ws-v5): corpus rows 217 and 235, whose assistant content
/// is `[{"type": "text", …}, 0.0001, 30.0, true, false, null, 7, -17]`, come back 400 with
/// **two** errors — one per branch of the assistant part union — naming the FIRST bad element
/// only. The branch list is role-dependent and only the assistant one has been measured, so the
/// message below is prod's for an assistant message and prod's SHAPE for any other role; the
/// status, the type and the `param` are prod's either way, and the gate says which is compared.
fn content_parts(messages: &[Value]) -> Result<(), ApiError> {
    for msg in messages {
        let Some(map) = msg.as_object() else { continue };
        let Some(Value::Array(parts)) = map.get("content") else {
            continue;
        };
        // FN28 (measured in prod's image, `ChatCompletionRequest.model_validate` + materialising
        // the lazy content iterator): an `input_text` / `output_text` part -- a Responses-API
        // shape `_parse_chat_message_content_part` would render as text -- is refused by the
        // OpenAI message types' part unions for the five OpenAI roles (user 7 / assistant 3 /
        // system, tool, developer 1 branch errors); only a custom role's message takes it. Status
        // and type are prod's; the per-branch listing is its first branch only.
        let openai_role = matches!(
            map.get("role").and_then(Value::as_str),
            Some("user" | "assistant" | "system" | "tool" | "developer")
        );
        if openai_role {
            let responses_part = parts.iter().enumerate().find_map(|(i, p)| {
                p.get("type")
                    .and_then(Value::as_str)
                    .filter(|t| matches!(*t, "input_text" | "output_text"))
                    .map(|t| (i, t))
            });
            if let Some((i, kind)) = responses_part {
                let n = match map.get("role").and_then(Value::as_str) {
                    Some("user") => 7,
                    Some("assistant") => 3,
                    _ => 1,
                };
                let errors = if n == 1 { "error" } else { "errors" };
                return Err(ApiError::bad_request(
                    format!(
                        "{n} validation {errors} for ValidatorIterator\n{i}.\
                         ChatCompletionContentPartTextParam.type\n  Input should be 'text' \
                         [type=literal_error, input_value='{kind}', input_type=str]"
                    ),
                    None,
                ));
            }
        }
        let Some((i, bad)) = parts.iter().enumerate().find(|(_, p)| !p.is_object()) else {
            continue;
        };
        // The assistant part union, which is the one the corpus exercises and the one that was
        // measured. A user-role part has a longer union (image_url, input_audio, file); this
        // names the pair either way rather than inventing a list nobody has read off the wire.
        let branches: &[&str] = &[
            "ChatCompletionContentPartTextParam",
            "ChatCompletionContentPartRefusalParam",
        ];
        let value = py_repr(bad);
        let ty = py_type_name(bad);
        let mut message = format!("{} validation errors for ValidatorIterator", branches.len());
        for branch in branches {
            message.push_str(&format!(
                "\n{i}.{branch}\n  Input should be a valid dictionary [type=dict_type, \
                 input_value={value}, input_type={ty}]\n    For further information visit \
                 https://errors.pydantic.dev/2.13/v/dict_type"
            ));
        }
        return Err(ApiError::bad_request(message, None));
    }
    Ok(())
}

/// `repr(x)` for a JSON scalar, as pydantic prints an `input_value`.
fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(s) => format!("{s:?}"),
        Value::Number(n) => match n.as_i64() {
            Some(i) => i.to_string(),
            None => n
                .as_f64()
                .map_or_else(|| n.to_string(), pyjson::py_repr_f64),
        },
        Value::Array(_) => "[...]".to_owned(),
        Value::Object(_) => "{...}".to_owned(),
    }
}

/// `type(x).__name__`, as pydantic prints an `input_type`.
fn py_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::String(_) => "str",
        Value::Number(n) if n.is_i64() || n.is_u64() => "int",
        Value::Number(_) => "float",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// `ChatCompletionRequest.check_tool_usage`, the parts that are reachable from this wire type.
fn check_tool_usage(req: &ChatCompletionRequest) -> Result<(), ApiError> {
    if req.tools.as_ref().is_some_and(Vec::is_empty) {
        return Err(ApiError::bad_request(
            "`tools` must not be an empty array. Either provide at least one tool or omit the \
             field entirely. (parameter=tools)",
            Some("tools"),
        ));
    }
    // "if `tool_choice` is not specified but tools are provided, default to auto", and a
    // `tool_choice` of `none` skips every check below it.
    let choice = match req.tool_choice.as_ref() {
        None if req.tools.as_ref().is_some_and(|t| !t.is_empty()) => return Ok(()), // auto
        None => return Ok(()),
        Some(Value::Null) => return Ok(()),
        Some(v) if v == "none" => return Ok(()),
        Some(v) => v,
    };
    let Some(tools) = req.tools.as_ref() else {
        return Err(ApiError::bad_request(
            "When using `tool_choice`, `tools` must be set. (parameter=tool_choice)",
            Some("tool_choice"),
        ));
    };
    let named = match choice {
        Value::Object(_) => true,
        v if v == "auto" || v == "required" => false,
        v => {
            return Err(ApiError::bad_request(
                format!(
                    "Invalid value for `tool_choice`: {}! Only named tools, \"none\", \"auto\" or \
                     \"required\" are supported. (parameter=tool_choice)",
                    py_scalar(v)
                ),
                Some("tool_choice"),
            ))
        }
    };
    if !named {
        return Ok(());
    }
    const USAGE: &str =
        "Correct usage: `{\"type\": \"function\", \"function\": {\"name\": \"my_function\"}}`";
    let function = choice.get("function");
    let Some(Value::Object(function)) = function else {
        return Err(ApiError::bad_request(
            format!(
                "Invalid value for `function`: `{}` in `tool_choice`! {USAGE} \
                 (parameter=tool_choice.function)",
                function.map_or_else(|| "None".to_owned(), py_scalar)
            ),
            Some("tool_choice.function"),
        ));
    };
    let Some(name) = function.get("name") else {
        return Err(ApiError::bad_request(
            format!(
                "Expected field `name` in `function` in `tool_choice`! {USAGE} \
                 (parameter=tool_choice.function.name)"
            ),
            Some("tool_choice.function.name"),
        ));
    };
    let name = match name {
        Value::String(s) if !s.is_empty() => s.clone(),
        other => {
            return Err(ApiError::bad_request(
                format!(
                    "Invalid `name` in `function`: `{}` in `tool_choice`! {USAGE} \
                     (parameter=tool_choice.function.name)",
                    py_scalar(other)
                ),
                Some("tool_choice.function.name"),
            ))
        }
    };
    let matches = tools.iter().any(|t| {
        t.get("function")
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            == Some(name.as_str())
    });
    if !matches {
        return Err(ApiError::bad_request(
            "The tool specified in `tool_choice` does not match any of the specified `tools` \
             (parameter=tool_choice)",
            Some("tool_choice"),
        ));
    }
    Ok(())
}

/// `validate_stream_options`: the option only exists on a stream.
fn check_stream_options(req: &ChatCompletionRequest) -> Result<(), ApiError> {
    // The reference server's test is `data.get("stream_options") and not data.get("stream")`, so an empty object
    // is falsy in Python and passes. `{"include_usage": false}` is a non-empty dict and does not.
    if req.stream_options.is_some() && !req.stream {
        return Err(ApiError::bad_request(
            "Stream options can only be defined when `stream=True`. (parameter=stream_options)",
            Some("stream_options"),
        ));
    }
    Ok(())
}

/// A Python `repr` of a scalar, for the messages that interpolate one.
fn py_scalar(v: &Value) -> String {
    match v {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// `tools: list[ChatCompletionToolsParam]`, in field declaration order (`type`, then `function`).
fn tools_shape(tools: &[Value]) -> Option<ShapeError> {
    for (i, tool) in tools.iter().enumerate() {
        let Value::Object(tool) = tool else {
            return Some(ShapeError::dict_type(vec![
                Loc::Field("body"),
                Loc::Field("tools"),
                Loc::Index(i),
            ]));
        };
        if let Some(kind) = tool.get("type") {
            if kind != "function" {
                return Some(ShapeError::literal(
                    vec![
                        Loc::Field("body"),
                        Loc::Field("tools"),
                        Loc::Index(i),
                        Loc::Field("type"),
                    ],
                    "'function'",
                ));
            }
        }
        let loc = |f: &'static str| {
            vec![
                Loc::Field("body"),
                Loc::Field("tools"),
                Loc::Index(i),
                Loc::Field("function"),
                Loc::Field(f),
            ]
        };
        match tool.get("function") {
            None => {
                return Some(ShapeError::missing(vec![
                    Loc::Field("body"),
                    Loc::Field("tools"),
                    Loc::Index(i),
                    Loc::Field("function"),
                ]))
            }
            Some(Value::Object(f)) => match f.get("name") {
                None => return Some(ShapeError::missing(loc("name"))),
                Some(Value::String(_)) => {}
                Some(_) => return Some(ShapeError::string_type(loc("name"))),
            },
            Some(_) => {
                return Some(ShapeError::dict_type(vec![
                    Loc::Field("body"),
                    Loc::Field("tools"),
                    Loc::Index(i),
                    Loc::Field("function"),
                ]))
            }
        }
    }
    None
}

/// The union `OpenAIChatCompletionMessageParam | CustomChatCompletionMessageParam`, reduced to
/// what a message has to satisfy to match SOME branch of it.
fn messages_shape(messages: &[Value]) -> Option<ShapeError> {
    // The FIRST branch of the message union, which is the one pydantic reports first and
    // therefore the one prod's `param` names. Measured on the live side container: a message with
    // no `role` comes back `param: body.messages.0.ChatCompletionDeveloperMessageParam.role`, with
    // eleven errors behind it -- one per branch.
    const FIRST: &str = "ChatCompletionDeveloperMessageParam";
    for (i, msg) in messages.iter().enumerate() {
        let at = |field: &'static str| {
            vec![
                Loc::Field("body"),
                Loc::Field("messages"),
                Loc::Index(i),
                Loc::Branch(FIRST),
                Loc::Field(field),
            ]
        };
        let Value::Object(map) = msg else {
            return Some(ShapeError::dict_type(vec![
                Loc::Field("body"),
                Loc::Field("messages"),
                Loc::Index(i),
                Loc::Branch(FIRST),
            ]));
        };
        let role = match map.get("role") {
            None => return Some(ShapeError::missing(at("role"))),
            Some(Value::String(s)) => s.as_str(),
            Some(_) => return Some(ShapeError::string_type(at("role"))),
        };
        // `content` is a string or a parts list on every branch; only the assistant branch's is
        // `Optional`, and `total=False` means it may be absent entirely.
        match map.get("content") {
            None | Some(Value::String(_)) | Some(Value::Array(_)) => {}
            Some(Value::Null) if role == "assistant" => {}
            Some(_) => return Some(ShapeError::string_type(at("content"))),
        }
        // `tool_calls` is NOT checked here: see `validate_chat_shape`. A non-list is the one
        // shape the `mode="before"` normaliser turns into one (`list(tool_calls)`), so a string
        // arrives at the field as a list of its characters and fails as a list of non-objects.
        if let Some(v) = map.get("tool_calls") {
            if !v.is_array() && !v.is_null() {
                return Some(ShapeError::dict_type(vec![
                    Loc::Field("body"),
                    Loc::Index(0),
                    Loc::Branch("ChatCompletionMessageFunctionToolCallParam"),
                ]));
            }
        }
    }
    None
}

/// `list[ChatCompletionMessageToolCallParam]`, whose `loc` prod reports as
/// `(<index>, '<branch>', <field>…)` — the tool call's own index, not the message's.
fn tool_calls_shape(calls: &[Value]) -> Option<ShapeError> {
    const BRANCH: &str = "ChatCompletionMessageFunctionToolCallParam";
    for (j, call) in calls.iter().enumerate() {
        let base = || vec![Loc::Field("body"), Loc::Index(j), Loc::Branch(BRANCH)];
        let Value::Object(call) = call else {
            return Some(ShapeError::dict_type(base()));
        };
        // FN28: a well-formed CUSTOM tool call (`{"id", "type": "custom", "custom": {"name",
        // "input"}}`) passes prod's pydantic union (its `ChatCompletionMessageCustomToolCallParam`
        // branch, measured) and is refused by the handler (`postprocess_messages`), not here.
        if is_custom_call(call) {
            continue;
        }
        let field = |f: &'static str| {
            let mut l = base();
            l.push(Loc::Field(f));
            l
        };
        match call.get("id") {
            None => return Some(ShapeError::missing(field("id"))),
            Some(Value::String(_)) => {}
            Some(_) => return Some(ShapeError::string_type(field("id"))),
        }
        // THE ORDER IS THE WIRE TYPE'S, not this function's convenience: openai's
        // `ChatCompletionMessageFunctionToolCallParam` declares `id`, `function`, `type`, and
        // pydantic reports a TypedDict's missing keys in declaration order — so a call that is
        // missing BOTH `function` and `type` (the corpus's "no wrapper" shape) is reported against
        // `function`, and a call missing only `type` against `type`. Measured, not assumed.
        let fun = match call.get("function") {
            None => return Some(ShapeError::missing(field("function"))),
            Some(Value::Object(f)) => f,
            Some(_) => return Some(ShapeError::dict_type(field("function"))),
        };
        let inner = |f: &'static str| {
            let mut l = field("function");
            l.push(Loc::Field(f));
            l
        };
        match fun.get("name") {
            None => return Some(ShapeError::missing(inner("name"))),
            Some(Value::String(_)) => {}
            Some(_) => return Some(ShapeError::string_type(inner("name"))),
        }
        match fun.get("arguments") {
            None => return Some(ShapeError::missing(inner("arguments"))),
            Some(Value::String(_)) => {}
            Some(_) => return Some(ShapeError::string_type(inner("arguments"))),
        }
        // `type` is REQUIRED on the wire type even though `_postprocess_messages` defaults it:
        // pydantic refuses the request before the handler ever runs.
        match call.get("type") {
            None => return Some(ShapeError::missing(field("type"))),
            Some(v) if v == "function" => {}
            Some(_) => return Some(ShapeError::literal(field("type"), "'function'")),
        }
    }
    None
}

/// `upstream.entrypoints.chat_utils._postprocess_messages`, which is where row 205 dies.
///
/// Assistant messages only, `arguments` only, and only when the string is non-empty: an empty
/// string is falsy in Python and the handler substitutes `{}` without parsing anything.
///
/// Per call IN ORDER (0.27.1 1928-1949): a call whose `type` is not `function` (a custom call --
/// every other shape was pydantic's 400) is `ValidationError("chat completions only support
/// assistant tool_calls of type 'function'.", parameter="tool_calls")`, which the server answers
/// as `create_error_response` does (FN28, measured): 400 `BadRequestError`, `param:
/// "tool_calls"`, the message with ` (parameter=tool_calls)` appended by its `__str__`; then,
/// when `decode` (0.27.1's laws), the call's `arguments` JSON decode.
fn postprocess_messages(messages: &[Value], decode: bool) -> Result<(), ApiError> {
    for msg in messages {
        let Some(map) = msg.as_object() else { continue };
        if map.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(Value::Array(calls)) = map.get("tool_calls") else {
            continue;
        };
        for call in calls {
            if call.get("type").and_then(Value::as_str) == Some("custom") {
                return Err(ApiError::bad_request(
                    "chat completions only support assistant tool_calls of type 'function'. \
                     (parameter=tool_calls)",
                    Some("tool_calls"),
                ));
            }
            if !decode {
                continue;
            }
            let Some(args) = call
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(Value::as_str)
            else {
                continue;
            };
            if args.is_empty() {
                continue;
            }
            if let Err(e) = pyjson::loads_err(args) {
                // `json.JSONDecodeError` is a `ValueError`, which `create_error_response` maps to
                // a 400 with `param: null` and `str(exc)` as the message.
                return Err(ApiError::bad_request(e.to_string(), None));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn req(body: Value) -> ChatCompletionRequest {
        serde_json::from_value(body).expect("a request body")
    }

    fn tool() -> Value {
        json!({"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}})
    }

    fn with_call(call: Value) -> Value {
        json!({"model": "Qwen3.8", "tools": [tool()], "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "content": "r", "tool_call_id": "c0"},
            {"role": "user", "content": "go"}]})
    }

    fn err(body: Value) -> ApiError {
        validate_chat_shape(&req(body)).expect_err("this shape is a 400 on prod")
    }

    fn ok(body: Value) {
        validate_chat_shape(&req(body)).expect("prod serves this shape");
    }

    /// **Row 205**, the one row of ws-q1's agentic corpus that prod refused and Saber served, in
    /// prod's own body: a `ValueError` through `create_error_response`, so `BadRequestError`, no
    /// `param`, and CPython's `json` message verbatim.
    #[test]
    fn a_tool_call_whose_arguments_are_not_json_is_prods_400() {
        let e = err(with_call(json!({
            "id": "call_0050", "type": "function",
            "function": {"name": "legacy_tool", "arguments": "not even json"}})));
        assert_eq!(e.body.code, 400);
        assert_eq!(e.body.kind, "BadRequestError");
        assert_eq!(e.body.param, None);
        assert_eq!(e.body.message, "Expecting value: line 1 column 1 (char 0)");
    }

    /// The controls for it, each measured against prod rather than assumed: an EMPTY `arguments`
    /// string is served (Python's `if content:` is false and the handler substitutes `{}`), a
    /// well-formed one is served, and the same malformed string on a NON-assistant message is
    /// served, because only assistant messages reach the decode.
    #[test]
    fn the_json_decode_is_assistant_only_and_skips_an_empty_string() {
        ok(with_call(json!({
            "id": "c0", "type": "function", "function": {"name": "f", "arguments": ""}})));
        ok(with_call(json!({
            "id": "c0", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}})));
        ok(json!({"model": "Qwen3.8", "tools": [tool()], "messages": [
            {"role": "system", "content": "s", "tool_calls": [
                {"id": "c0", "type": "function",
                 "function": {"name": "f", "arguments": "not even json"}}]},
            {"role": "user", "content": "hi"}]}));
    }

    /// The four message shapes ws-q1's `normalise()` had to rewrite, each with prod's verdict.
    /// Three are 400s from the pydantic family; the fourth is **served**, which is the finding.
    #[test]
    fn the_four_corpus_shapes_get_prods_verdicts() {
        // 1. a `tool_calls` entry with no `type`
        let e = err(with_call(
            json!({"id": "c0", "function": {"name": "f", "arguments": "{}"}}),
        ));
        assert_eq!(e.body.code, 400);
        assert_eq!(e.body.kind, "Bad Request");
        assert_eq!(
            e.body.param.as_deref(),
            Some("body.0.ChatCompletionMessageFunctionToolCallParam.type")
        );

        // 2. `arguments` as an OBJECT rather than the JSON string the API defines
        let e = err(with_call(json!({
            "id": "c0", "type": "function", "function": {"name": "f", "arguments": {"a": 1}}})));
        assert_eq!(e.body.code, 400);
        assert!(
            e.body.message.contains("Input should be a valid string"),
            "{}",
            e.body.message
        );

        // 3. a `tool_calls` entry with no `function` wrapper. It is missing `type` as well, and
        //    the field prod NAMES is `function`, because that is the one the wire type declares
        //    first (`id`, `function`, `type`) -- an order that is measured, not chosen.
        let e = err(with_call(
            json!({"id": "c0", "name": "f", "arguments": "{}"}),
        ));
        assert_eq!(e.body.code, 400);
        assert_eq!(
            e.body.param.as_deref(),
            Some("body.0.ChatCompletionMessageFunctionToolCallParam.function")
        );
        assert!(
            e.body.message.contains("'msg': 'Field required'"),
            "{}",
            e.body.message
        );

        // 4. `role: tool` with NO `tool_call_id` -- prod SERVES it. Measured against the live
        //    side container, not inferred: `CustomChatCompletionMessageParam.tool_call_id` is
        //    optional, so the request is valid and only the template cares.
        ok(json!({"model": "Qwen3.8", "tools": [tool()], "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c0", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "content": "r"},
            {"role": "user", "content": "go"}]}));
    }

    /// **Corpus rows 217 and 235**, the two the first prod capture caught this module serving: a
    /// content PARTS LIST with a float where a part should be. The list is lazily validated, so
    /// the error escapes the handler as a bare `pydantic.ValidationError` — a `ValueError`, hence
    /// `BadRequestError` and `param: null` — and this is prod's body for row 217, byte for byte,
    /// taken off the live side container on 2026-09-22.
    #[test]
    fn a_content_part_that_is_not_an_object_is_prods_400() {
        let e = err(json!({"model": "Qwen3.8", "tools": [tool()], "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "values: "}, 0.0001, 30.0, true, false, null, 7, -17]},
            {"role": "user", "content": "go"}]}));
        assert_eq!(e.body.code, 400);
        assert_eq!(e.body.kind, "BadRequestError");
        assert_eq!(e.body.param, None);
        assert_eq!(
            e.body.message,
            "2 validation errors for ValidatorIterator\n\
             1.ChatCompletionContentPartTextParam\n  Input should be a valid dictionary \
             [type=dict_type, input_value=0.0001, input_type=float]\n    For further information \
             visit https://errors.pydantic.dev/2.13/v/dict_type\n\
             1.ChatCompletionContentPartRefusalParam\n  Input should be a valid dictionary \
             [type=dict_type, input_value=0.0001, input_type=float]\n    For further information \
             visit https://errors.pydantic.dev/2.13/v/dict_type"
        );
        // Row 235's first bad element is `1e+16`, and the repr is CPython's, not Rust's.
        let e = err(json!({"model": "Qwen3.8", "tools": [tool()], "messages": [
            {"role": "assistant", "content": [
                {"type": "text", "text": "values: "}, 1e16, 1e-5, true]},
            {"role": "user", "content": "go"}]}));
        assert!(
            e.body.message.contains("input_value=1e+16"),
            "{}",
            e.body.message
        );
        // A parts list of real parts is served, which is the control: the refusal is about the
        // ELEMENT's type and not about the list.
        ok(json!({"model": "Qwen3.8", "tools": [tool()], "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "a"},
                                              {"type": "text", "text": "b"}]},
            {"role": "user", "content": "go"}]}));
    }

    /// `legacy_tool`: the corpus's bare tool with no `{"type": "function"}` wrapper. One error,
    /// one clean `loc`, and prod's `param` for it is reproduced exactly.
    #[test]
    fn a_tool_without_its_function_wrapper_is_prods_400() {
        let e = err(
            json!({"model": "Qwen3.8", "messages": [{"role": "user", "content": "hi"}],
                           "tools": [tool(), {"name": "legacy_tool", "description": "d"}]}),
        );
        assert_eq!(e.body.code, 400);
        assert_eq!(e.body.kind, "Bad Request");
        assert_eq!(e.body.param.as_deref(), Some("body.tools.1.function"));
        assert_eq!(
            e.body.message,
            "1 validation error:\n  {'type': 'missing', 'loc': ('body', 'tools', 1, 'function'), \
             'msg': 'Field required'}"
        );
        // A tool that is not a function tool at all is the literal error, not a missing field.
        let e = err(
            json!({"model": "Qwen3.8", "messages": [{"role": "user", "content": "hi"}],
                           "tools": [{"type": "custom", "custom": {"name": "c"}}]}),
        );
        assert_eq!(e.body.param.as_deref(), Some("body.tools.0.type"));
        assert!(
            e.body.message.contains("Input should be 'function'"),
            "{}",
            e.body.message
        );
    }

    /// The `mode="before"` validators that reach this wire type, and the order between them: a
    /// request that is bad in two ways gets the message for whichever prod checks first.
    #[test]
    fn the_before_validators_run_in_prods_order() {
        let e = err(
            json!({"model": "Qwen3.8", "messages": [{"role": "user", "content": "hi"}],
                           "tools": []}),
        );
        assert_eq!(e.body.kind, "BadRequestError");
        assert_eq!(e.body.param.as_deref(), Some("tools"));
        assert_eq!(
            e.body.message,
            "`tools` must not be an empty array. Either provide at least one tool or omit the \
             field entirely. (parameter=tools)"
        );

        // `stream_options` without `stream` -- the shape ws-q1's own driver sends on its
        // NON-streaming twin, and the reason prod answered `nostream_error` on every row it
        // answered at all.
        let e = err(
            json!({"model": "Qwen3.8", "messages": [{"role": "user", "content": "hi"}],
                           "stream_options": {"include_usage": true}}),
        );
        assert_eq!(e.body.kind, "BadRequestError");
        assert_eq!(e.body.param.as_deref(), Some("stream_options"));
        assert_eq!(
            e.body.message,
            "Stream options can only be defined when `stream=True`. (parameter=stream_options)"
        );
        ok(
            json!({"model": "Qwen3.8", "messages": [{"role": "user", "content": "hi"}],
                  "stream": true, "stream_options": {"include_usage": true}}),
        );

        // `check_tool_usage` runs BEFORE `validate_stream_options` (pydantic runs `mode="before"`
        // model validators in reverse definition order), so a request that breaks both gets the
        // tools message.
        let e = err(
            json!({"model": "Qwen3.8", "messages": [{"role": "user", "content": "hi"}],
                           "tools": [], "stream_options": {"include_usage": true}}),
        );
        assert_eq!(e.body.param.as_deref(), Some("tools"));
    }

    /// `tool_choice`, whose refusals are the handler family and are reproduced byte for byte.
    #[test]
    fn tool_choice_refusals_are_prods_own_strings() {
        let e = err(
            json!({"model": "Qwen3.8", "messages": [{"role": "user", "content": "hi"}],
                           "tool_choice": "banana", "tools": [tool()]}),
        );
        assert_eq!(
            e.body.message,
            "Invalid value for `tool_choice`: banana! Only named tools, \"none\", \"auto\" or \
             \"required\" are supported. (parameter=tool_choice)"
        );
        let e = err(
            json!({"model": "Qwen3.8", "messages": [{"role": "user", "content": "hi"}],
                           "tool_choice": {"type": "function", "function": {"name": "nope"}},
                           "tools": [tool()]}),
        );
        assert_eq!(
            e.body.message,
            "The tool specified in `tool_choice` does not match any of the specified `tools` \
             (parameter=tool_choice)"
        );
        // `none` skips every check below it, including the named-tool match.
        ok(
            json!({"model": "Qwen3.8", "messages": [{"role": "user", "content": "hi"}],
                  "tool_choice": "none", "tools": [tool()]}),
        );
    }

    /// The shapes prod SERVES, kept as a list because an over-refusal breaks parity exactly as
    /// badly as an under-refusal and is harder to notice.
    #[test]
    fn what_prod_serves_saber_serves() {
        ok(json!({"model": "Qwen3.8", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}]}));
        ok(json!({"model": "Qwen3.8", "messages": [{"role": "wizard", "content": "hi"}]}));
        ok(json!({"model": "Qwen3.8", "messages": [
            {"role": "assistant", "content": null}, {"role": "user", "content": "hi"}]}));
        ok(json!({"model": "Qwen3.8", "messages": [
            {"role": "assistant", "content": "x", "tool_calls": []},
            {"role": "user", "content": "hi"}]}));
        // No `tools` key at all, and no `tool_choice`: the plainest request there is.
        ok(json!({"model": "Qwen3.8", "messages": [{"role": "user", "content": "hi"}]}));
    }

    /// The control for the whole module: a request with NO malformed shape must not be refused,
    /// and one with a malformed shape must not be served. Both directions, one test.
    #[test]
    fn the_validator_has_a_control_that_fails() {
        let good = json!({"model": "Qwen3.8", "tools": [tool()], "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c0", "type": "function",
                 "function": {"name": "f", "arguments": "{\"x\": 1}"}}]},
            {"role": "tool", "content": "r", "tool_call_id": "c0"},
            {"role": "user", "content": "go"}]});
        ok(good.clone());
        let mut bad = good;
        bad["messages"][1]["tool_calls"][0]["function"]["arguments"] =
            Value::String("not even json".to_owned());
        assert!(
            validate_chat_shape(&req(bad)).is_err(),
            "the control did not fire"
        );
    }

    /// FN25: the reference 0.29's `_postprocess_messages`, branch by branch (read in
    /// `saber-base:0.29.0`, `chat_utils.py`).
    #[test]
    fn tool_arguments_become_mappings_as_upstream_029_makes_them() {
        let call = |args: Value| {
            json!({"id": "c", "type": "function",
                                        "function": {"name": "f", "arguments": args}})
        };
        let conv = |calls: Value| {
            vec![
                json!({"role": "user", "content": "hi"}),
                json!({"role": "assistant", "content": "", "tool_calls": calls}),
                json!({"role": "tool", "tool_call_id": "c", "content": "{\"a\": 1}"}),
            ]
        };
        let args_of = |msgs: &[Value]| msgs[1]["tool_calls"][0]["function"]["arguments"].clone();
        // no assistant tool_calls list: the very same JSON (None)
        assert!(tool_arguments_to_objects(&[json!({"role": "user", "content": "hi"})]).is_none());
        // a JSON string -> its mapping, key order kept, the last duplicate wins in place
        let out = tool_arguments_to_objects(&conv(json!([call(json!(
            "{\"b\": 1, \"a\": [1, \"x\"], \"b\": 2.5}"
        ))])))
        .expect("touched");
        assert_eq!(args_of(&out), json!({"b": 2.5, "a": [1, "x"]}));
        assert_eq!(
            args_of(&out)
                .as_object()
                .expect("map")
                .keys()
                .collect::<Vec<_>>(),
            ["b", "a"]
        );
        // a mapping passes as is
        let out = tool_arguments_to_objects(&conv(json!([call(json!({"z": 1, "y": null}))])))
            .expect("touched");
        assert_eq!(args_of(&out), json!({"z": 1, "y": null}));
        // malformed JSON, JSON that is not an object, falsy values, a bare number: {}
        for bad in [
            json!("not even json"),
            json!("[1, 2]"),
            json!("42"),
            json!(""),
            json!(null),
            json!(0),
            json!(7),
            json!([1]),
            json!({}),
        ] {
            let out =
                tool_arguments_to_objects(&conv(json!([call(bad.clone())]))).expect("touched");
            assert_eq!(args_of(&out), json!({}), "{bad}");
        }
        // absent arguments: {}
        let out = tool_arguments_to_objects(&conv(json!([
            {"id": "c", "type": "function", "function": {"name": "f"}}
        ])))
        .expect("touched");
        assert_eq!(args_of(&out), json!({}));
        // an empty list is dropped; the tool turn and the user turn are untouched
        let out = tool_arguments_to_objects(&conv(json!([]))).expect("touched");
        assert!(out[1].get("tool_calls").is_none());
        assert_eq!(out[2], conv(json!([]))[2]);
        // a non-assistant message's tool_calls are not the handler's business
        let msgs = vec![json!({"role": "user", "content": "x", "tool_calls": [call(json!("{}"))]})];
        assert!(tool_arguments_to_objects(&msgs).is_none());
    }

    /// FN25: under the Objects law a malformed `arguments` is the reference 0.29's `{}`, not 0.27.1's 400;
    /// every pydantic-family refusal is the same on both laws.
    #[test]
    fn the_objects_law_refuses_only_what_the_schema_refuses() {
        let body = |args: Value| {
            json!({"model": "m", "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c", "type": "function", "function": {"name": "f", "arguments": args}}]},
                {"role": "user", "content": "go"}]})
        };
        let malformed = req(body(json!("not even json")));
        assert!(validate_chat_shape_for(&malformed, ToolArguments::Verbatim).is_err());
        assert!(validate_chat_shape_for(&malformed, ToolArguments::Objects).is_ok());
        let object = req(body(json!({"x": 1})));
        for law in [ToolArguments::Verbatim, ToolArguments::Objects] {
            let e = validate_chat_shape_for(&object, law).expect_err("pydantic string_type");
            assert_eq!(e.body.kind, "Bad Request");
        }
    }

    /// FN27: prod 0.27.1's `_postprocess_messages` (`chat_utils.py` 1911-1951 in
    /// `the reference image`), branch by branch, with the reference 0.29's verdict beside each one.
    #[test]
    fn tool_arguments_become_what_prod_0271_made_them() {
        let call = |args: Value| json!({"id": "c", "type": "function", "function": {"name": "f", "arguments": args}});
        let conv = |calls: Value| {
            vec![
                json!({"role": "user", "content": "hi"}),
                json!({"role": "assistant", "content": "", "tool_calls": calls}),
                json!({"role": "tool", "tool_call_id": "c", "content": "{\"a\": 1}"}),
            ]
        };
        let args_of = |msgs: &[Value]| msgs[1]["tool_calls"][0]["function"]["arguments"].clone();
        let strict = |args: Value| {
            args_of(&tool_arguments_to_objects_0271(&conv(json!([call(args)]))).expect("touched"))
        };
        let v029 = |args: Value| {
            args_of(&tool_arguments_to_objects(&conv(json!([call(args)]))).expect("touched"))
        };
        // 1918-1921: nothing to touch -> the very same JSON (None)
        assert!(
            tool_arguments_to_objects_0271(&[json!({"role": "user", "content": "hi"})]).is_none()
        );
        let user_calls =
            vec![json!({"role": "user", "content": "x", "tool_calls": [call(json!("{}"))]})];
        assert!(tool_arguments_to_objects_0271(&user_calls).is_none());
        let null_calls = vec![json!({"role": "assistant", "content": "x", "tool_calls": null})];
        assert!(tool_arguments_to_objects_0271(&null_calls).is_none());
        // 1923-1926: an empty list is dropped, the other turns untouched (0.29 alike)
        let out = tool_arguments_to_objects_0271(&conv(json!([]))).expect("touched");
        assert!(out[1].get("tool_calls").is_none());
        assert_eq!(out[2], conv(json!([]))[2]);
        // 1947-1949: every spelling of one object is one mapping, key order kept, the last
        // duplicate in the first one's place (0.29 alike)
        for spelling in [
            "{\"b\":1,\"a\":[1,\"x\"]}",
            "{\"b\": 1, \"a\": [1, \"x\"]}",
            "{ \"b\" : 1 , \"a\" : [ 1 , \"x\" ] }",
            "\n{\n  \"b\": 1,\n  \"a\": [1, \"x\"]\n}\n",
            "{\"b\": 0, \"a\": [1, \"x\"], \"b\": 1}",
        ] {
            let got = strict(json!(spelling));
            assert_eq!(got, json!({"b": 1, "a": [1, "x"]}), "{spelling}");
            assert_eq!(
                got.as_object().expect("map").keys().collect::<Vec<_>>(),
                ["b", "a"]
            );
            assert_eq!(got, v029(json!(spelling)));
        }
        // unicode: the escape and the character are one string
        assert_eq!(
            strict(json!("{\"c\": \"Z\\u00fcrich\"}")),
            json!({"c": "Z\u{fc}rich"})
        );
        // 1947-1949: JSON that is NOT an object is KEPT as parsed on 0.27.1; 0.29 makes it {}
        for (text, parsed) in [
            ("[1, 2]", json!([1, 2])),
            ("42", json!(42)),
            ("\"just text\"", json!("just text")),
            ("false", json!(false)),
            ("0", json!(0)),
            ("-0.0", json!(-0.0)),
        ] {
            assert_eq!(strict(json!(text)), parsed, "{text}");
            assert_eq!(v029(json!(text)), json!({}), "{text}");
        }
        // 1949: `null` -> {} (0.29 alike)
        assert_eq!(strict(json!("null")), json!({}));
        assert_eq!(v029(json!("null")), json!({}));
        // 1950-1951: falsy -> {} (0.29 alike); absent too
        for falsy in [json!(""), json!(null)] {
            assert_eq!(strict(falsy.clone()), json!({}), "{falsy}");
            assert_eq!(v029(falsy.clone()), json!({}), "{falsy}");
        }
        let absent = conv(json!([{"id": "c", "type": "function", "function": {"name": "f"}}]));
        assert_eq!(
            args_of(&tool_arguments_to_objects_0271(&absent).expect("touched")),
            json!({})
        );
        // 1947 (else): a mapping or list passes as is (pydantic refuses both on the wire)
        assert_eq!(
            strict(json!({"z": 1, "y": null})),
            json!({"z": 1, "y": null})
        );
        assert_eq!(strict(json!([3])), json!([3]));
        // exact 64-bit integers survive; the two values with no serde form keep the string
        assert_eq!(
            strict(json!("{\"n\": 18446744073709551615}")),
            json!({"n": u64::MAX})
        );
        for edge in ["{\"x\": NaN}", "{\"n\": 123456789012345678901234}"] {
            assert_eq!(strict(json!(edge)), json!(edge), "{edge}");
        }
        // every call of every assistant turn, not the first only
        let two = vec![
            json!({"role": "assistant", "content": "", "tool_calls": [
                call(json!("{\"a\":1}")), call(json!("{\"b\":2}"))]}),
            json!({"role": "assistant", "content": "", "tool_calls": [call(json!("[]"))]}),
        ];
        let out = tool_arguments_to_objects_0271(&two).expect("touched");
        assert_eq!(
            out[0]["tool_calls"][1]["function"]["arguments"],
            json!({"b": 2})
        );
        assert_eq!(out[1]["tool_calls"][0]["function"]["arguments"], json!([]));
    }

    /// FN27: under ObjectsStrict a malformed `arguments` is prod 0.27.1's 400 (its own
    /// `JSONDecodeError` text, `BadRequestError`, `param: null`) -- where 0.29's Objects serves
    /// it as `{}` -- and every pydantic refusal is the same under all three laws.
    /// FN28, measured in prod's image (`the internal reference harness`'s `fn28:xml:*custom*`):
    /// a custom tool call passes pydantic and is the handler's 400 `BadRequestError`, param
    /// `tool_calls`, under EVERY law (0.29 refuses it too); calls are walked in order, so a
    /// malformed `arguments` BEFORE it is the JSON 400 instead (0.27.1's laws only).
    #[test]
    fn a_custom_tool_call_is_prods_handler_400() {
        let custom = json!({"id": "k", "type": "custom", "custom": {"name": "f", "input": "x"}});
        let bad =
            json!({"id": "b", "type": "function", "function": {"name": "f", "arguments": "{bad"}});
        let good =
            json!({"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}});
        let body = |calls: Value| {
            req(json!({"model": "m", "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": calls},
                {"role": "user", "content": "go"}]}))
        };
        let msg = "chat completions only support assistant tool_calls of type 'function'. \
                   (parameter=tool_calls)";
        for law in [
            ToolArguments::Verbatim,
            ToolArguments::ObjectsStrict,
            ToolArguments::Objects,
        ] {
            for calls in [
                json!([custom.clone()]),
                json!([good.clone(), custom.clone()]),
                json!([custom.clone(), bad.clone()]),
            ] {
                let e = validate_chat_shape_for(&body(calls), law).expect_err("custom");
                assert_eq!(
                    (e.body.code, e.body.kind.as_str()),
                    (400, "BadRequestError")
                );
                assert_eq!(e.body.message, msg);
                assert_eq!(e.body.param.as_deref(), Some("tool_calls"));
            }
        }
        let e = validate_chat_shape_for(&body(json!([bad, custom])), ToolArguments::ObjectsStrict)
            .expect_err("json first");
        assert_eq!(e.body.param, None);
        assert!(
            e.body.message.starts_with("Expecting property name"),
            "{}",
            e.body.message
        );
    }

    /// FN28, measured in prod's image: an `input_text` / `output_text` part is pydantic's 400 for
    /// the five OpenAI roles; a custom role takes it.
    #[test]
    fn responses_text_parts_are_refused_for_openai_roles() {
        for role in ["user", "assistant", "system", "tool", "developer", "weird"] {
            for kind in ["input_text", "output_text"] {
                let r = req(json!({"model": "m", "messages": [
                    {"role": role, "tool_call_id": "c", "content": [{"type": kind, "text": "a"}]}]}));
                let got = validate_chat_shape_for(&r, ToolArguments::ObjectsStrict);
                if role == "weird" {
                    assert!(got.is_ok(), "{role} {kind}");
                } else {
                    let e = got.expect_err(kind);
                    assert_eq!(
                        (e.body.code, e.body.kind.as_str()),
                        (400, "BadRequestError")
                    );
                }
            }
        }
    }

    #[test]
    fn the_strict_law_refuses_malformed_arguments_as_prod_0271() {
        let body = |args: Value| {
            json!({"model": "m", "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                    {"id": "d", "type": "function", "function": {"name": "f", "arguments": args}}]},
                {"role": "user", "content": "go"}]})
        };
        for (bad, message) in [
            ("not even json", "Expecting value: line 1 column 1 (char 0)"),
            (
                "{\"a\":1,}",
                "Expecting property name enclosed in double quotes: line 1 column 8 (char 7)",
            ),
            (
                "{'a': 1}",
                "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
            ),
            ("   ", "Expecting value: line 1 column 4 (char 3)"),
        ] {
            let r = req(body(json!(bad)));
            let e = validate_chat_shape_for(&r, ToolArguments::ObjectsStrict).expect_err(bad);
            assert_eq!(
                (e.body.code, e.body.kind.as_str()),
                (400, "BadRequestError")
            );
            assert_eq!(e.body.message, message, "{bad}");
            assert_eq!(e.body.param, None);
            let v = validate_chat_shape_for(&r, ToolArguments::Verbatim).expect_err(bad);
            assert_eq!(v.body.message, e.body.message);
            assert!(
                validate_chat_shape_for(&r, ToolArguments::Objects).is_ok(),
                "{bad}"
            );
        }
        for ok in ["", "{\"a\": 1}", "[1]", "null"] {
            validate_chat_shape_for(&req(body(json!(ok))), ToolArguments::ObjectsStrict).expect(ok);
        }
        let object = req(body(json!({"x": 1})));
        for law in [
            ToolArguments::Verbatim,
            ToolArguments::Objects,
            ToolArguments::ObjectsStrict,
        ] {
            let e = validate_chat_shape_for(&object, law).expect_err("pydantic string_type");
            assert_eq!(e.body.kind, "Bad Request");
        }
    }
}
