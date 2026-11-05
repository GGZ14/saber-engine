//! Chat-template rendering, byte-identical to the reference server's.
//!
//! **This is a P0 accuracy gate, not a convenience.** The served GSM8K band is 97–98% with
//! `qwen-fixed-v22.3.jinja` and 95–96% with anything else (`gsm8k-band-is-template-bound`), so a
//! one-byte difference in the prompt is an accuracy regression that no kernel work can recover.
//!
//! The reference is CPython + Jinja2 3.1.6 as configured by transformers 5.14.1 in
//! `chat_template_utils._cached_compile_jinja_template`:
//!
//! ```text
//! ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
//!                               extensions=[AssistantTracker, jinja2.ext.loopcontrols])
//! env.filters["tojson"]   = json.dumps(x, ensure_ascii=False, indent=None,
//!                                      separators=None, sort_keys=False)
//! env.filters["from_json"] = json.loads(x) if isinstance(x, str) else x   # the reference patch
//! env.globals["raise_exception"] = ...
//! env.globals["strftime_now"]    = ...
//! ```
//!
//! [`Renderer`] reproduces that on minijinja. The gap list — every place minijinja or Rust
//! disagrees with CPython/Jinja2 and what is installed to close it — is in [`pysem`] and in the
//! `install_*` functions below. The gate that proves the result is `tests/template_gate.rs`,
//! run over the P-1e corpus built by `the internal reference harness`.
//!
//! **Deliberately not implemented** (each fails loud rather than rendering something plausible):
//!
//! * `strftime_now` — a template that calls it gets "unknown function". Adding it would make
//!   rendering time-dependent, which the byte-identity gate could not then check.
//! * The `{% generation %}` / `{% endgeneration %}` block of transformers' `AssistantTracker`
//!   (used only for `return_assistant_tokens_mask`, which the serve path does not request).
//!   minijinja rejects it at compile time.
//! * `continue_final_message` — refused by [`ChatRequest::from_template_kwargs`]; it is a
//!   transformers-level message rewrite, not a template variable.
//! * `tojson(separators=...)` — the only `json.dumps` keyword with no caller.

pub mod pysem;

use std::borrow::Cow;
use std::path::Path;

use minijinja::value::{from_args, Kwargs, Rest, Value, ValueKind};
use minijinja::{Environment, Error, ErrorKind, State};

use crate::ModelError;

/// One chat-completion request, in the shape the template sees it.
///
/// Message and tool entries stay `serde_json::Value` on purpose: the template duck-types every
/// field it touches (`message.reasoning_content`, `message.thinking`, `tool_call.function`,
/// arbitrary roles, content that is a string *or* a list of parts *or* `null`), so a typed
/// struct would silently drop fields that change the rendered bytes.
#[derive(Debug, Clone, Default)]
pub struct ChatRequest {
    /// `messages` — the conversation after the reference server's `parse_chat_messages` normalisation.
    pub messages: Vec<serde_json::Value>,
    /// `tools` — OpenAI tool schemas, or `None`. `None` is rendered as Jinja `None`, not as
    /// undefined, because that is what transformers passes.
    pub tools: Option<Vec<serde_json::Value>>,
    /// `documents` — always `None` on the serve path, kept because transformers passes it.
    pub documents: Option<Vec<serde_json::Value>>,
    /// `add_generation_prompt` — `true` for every `/v1/chat/completions` request by default.
    pub add_generation_prompt: bool,
    /// Everything else the reference server's `resolve_chat_template_kwargs` lets through: `enable_thinking`,
    /// `reasoning_effort`, `tool_call_format`, `max_tool_arg_chars`, …
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl ChatRequest {
    /// Build a request from the messages/tools plus the *resolved* `chat_template_kwargs` map,
    /// which is exactly what `tokenizer.apply_chat_template` receives on the serve path.
    ///
    /// `continue_final_message` is handled by transformers outside the template; it is refused
    /// here rather than silently ignored, because Saber's serve path does not implement it.
    pub fn from_template_kwargs(
        messages: Vec<serde_json::Value>,
        tools: Option<Vec<serde_json::Value>>,
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Self, ModelError> {
        let mut extra = kwargs.clone();
        let add_generation_prompt = matches!(
            extra.remove("add_generation_prompt"),
            Some(serde_json::Value::Bool(true))
        );
        let documents = match extra.remove("documents") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::Array(a)) => Some(a),
            Some(other) => {
                return Err(ModelError::Template(format!(
                    "documents must be a list, got {other}"
                )))
            }
        };
        match extra.remove("continue_final_message") {
            None | Some(serde_json::Value::Null) | Some(serde_json::Value::Bool(false)) => {}
            Some(_) => {
                return Err(ModelError::Template(
                    "continue_final_message is not supported by the Saber serve path".to_owned(),
                ))
            }
        }
        // Harness-level flags that never reach the template.
        extra.remove("return_dict");
        extra.remove("tokenize");
        extra.remove("return_assistant_tokens_mask");
        Ok(Self {
            messages,
            tools,
            documents,
            add_generation_prompt,
            extra,
        })
    }
}

/// A compiled chat template.
pub struct Renderer {
    env: Environment<'static>,
}

impl std::fmt::Debug for Renderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Renderer").finish_non_exhaustive()
    }
}

const TEMPLATE_NAME: &str = "chat";

impl Renderer {
    /// Compile a template from source.
    pub fn from_source(source: &str) -> Result<Self, ModelError> {
        let mut env = Environment::new();
        // transformers: ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).
        // keep_trailing_newline stays false, which is also Jinja2's default.
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_keep_trailing_newline(false);
        // Jinja2's default Undefined: printable, `is defined`-testable, but attribute access
        // and iteration raise. minijinja's Lenient is the closest match; `iterable` is
        // overridden below to restore the raising behaviour.
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);
        env.set_unknown_method_callback(python_method_callback);
        install_filters(&mut env);
        install_tests(&mut env);
        install_globals(&mut env);
        // P27-8: `X[::-1]` is compiled as `__py_reverse(X)` (minijinja 2.24 indexes an EMPTY
        // sequence's negative-step slice out of bounds and panics; Python gives `[]`).
        env.add_template_owned(TEMPLATE_NAME, rewrite_reverse_slices(source))
            .map_err(|e| ModelError::Template(format!("{e:#}")))?;
        Ok(Self { env })
    }

    /// Compile a template from a file (`--chat-template <path>`).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let source = std::fs::read_to_string(path.as_ref())?;
        Self::from_source(&source)
    }

    /// Render one request to the exact prompt string the reference server would produce.
    pub fn render(&self, req: &ChatRequest) -> Result<String, ModelError> {
        let tmpl = self
            .env
            .get_template(TEMPLATE_NAME)
            .map_err(|e| ModelError::Template(format!("{e:#}")))?;

        let mut ctx = serde_json::Map::new();
        ctx.insert(
            "messages".to_owned(),
            serde_json::Value::Array(req.messages.clone()),
        );
        ctx.insert(
            "tools".to_owned(),
            req.tools
                .clone()
                .map_or(serde_json::Value::Null, serde_json::Value::Array),
        );
        ctx.insert(
            "documents".to_owned(),
            req.documents
                .clone()
                .map_or(serde_json::Value::Null, serde_json::Value::Array),
        );
        ctx.insert(
            "add_generation_prompt".to_owned(),
            serde_json::Value::Bool(req.add_generation_prompt),
        );
        for (k, v) in &req.extra {
            ctx.insert(k.clone(), v.clone());
        }

        // P27-8: a panic inside the template engine is a render ERROR, never an unwind into the
        // caller (the serve plane would drop the connection). The environment is immutable here,
        // so a caught panic leaves nothing half-written for the next render.
        let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tmpl.render(Value::from_serialize(serde_json::Value::Object(ctx)))
        }))
        .map_err(|p| ModelError::Template(format!("template engine panic: {}", panic_text(&*p))))?;
        rendered.map_err(|e| ModelError::Template(render_error_chain(&e)))
    }
}

/// The text of a caught panic payload (`&str` or `String`; anything else unnamed).
pub fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "(non-text payload)".to_owned())
}

/// P27-8: compile every `X[::-1]` inside a `{{ }}` / `{% %}` tag as `__py_reverse(X)`.
///
/// minijinja 2.24's negative-step slice (`value::ops::range_step_backwards`) yields index 0 of
/// an EMPTY sequence and panics (`index out of bounds: the len is 0`); CPython returns `[]`. The
/// 27B's template reverses `messages[head.count:]`, which is empty for a system-only request.
/// `X` is the postfix chain ending at the slice: identifier characters, `.`, and balanced
/// `(...)` / `[...]` groups walked backward; a slice of anything else (a literal) is left as
/// written. Text outside tags, comments, and string literals inside tags are never touched.
fn rewrite_reverse_slices(source: &str) -> String {
    const SLICE: &str = "[::-1]";
    let mut out = String::with_capacity(source.len() + 64);
    let mut i = 0;
    while i < source.len() {
        let open = &source[i..];
        let close = if open.starts_with("{{") {
            "}}"
        } else if open.starts_with("{%") {
            "%}"
        } else if open.starts_with("{#") {
            // A comment: copied verbatim to its end.
            let end = open.find("#}").map_or(source.len(), |e| i + e + 2);
            out.push_str(&source[i..end]);
            i = end;
            continue;
        } else {
            let c = open.chars().next().unwrap_or_default();
            out.push(c);
            i += c.len_utf8();
            continue;
        };
        // Inside a tag: copy, skipping string literals, rewriting each slice.
        out.push_str(&source[i..i + 2]);
        i += 2;
        let tag_start = out.len();
        let mut quote: Option<char> = None;
        while i < source.len() {
            let rest = &source[i..];
            let c = rest.chars().next().unwrap_or_default();
            if let Some(q) = quote {
                out.push(c);
                i += c.len_utf8();
                if c == '\\' {
                    if let Some(n) = source[i..].chars().next() {
                        out.push(n);
                        i += n.len_utf8();
                    }
                } else if c == q {
                    quote = None;
                }
                continue;
            }
            if rest.starts_with(close) {
                out.push_str(close);
                i += 2;
                break;
            }
            if c == '\'' || c == '"' {
                quote = Some(c);
            } else if rest.starts_with(SLICE) {
                if let Some(start) = postfix_start(&out, tag_start) {
                    let operand = out.split_off(start);
                    out.push_str("__py_reverse(");
                    out.push_str(&operand);
                    out.push(')');
                    i += SLICE.len();
                    continue;
                }
            }
            out.push(c);
            i += c.len_utf8();
        }
    }
    out
}

/// Where the postfix chain that ends `out` starts (not before `floor`), or `None` when `out`
/// does not end in one (a literal, a number, an operator).
fn postfix_start(out: &str, floor: usize) -> Option<usize> {
    let b = out.as_bytes();
    let mut j = out.len();
    while j > floor {
        let c = b[j - 1];
        if c.is_ascii_alphanumeric() || c == b'_' || c == b'.' {
            j -= 1;
        } else if c == b')' || c == b']' {
            // Walk back over the balanced group.
            let mut depth = 0usize;
            let mut k = j;
            loop {
                if k == floor {
                    return None;
                }
                k -= 1;
                match b[k] {
                    b')' | b']' => depth += 1,
                    b'(' | b'[' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
            }
            j = k;
        } else {
            break;
        }
    }
    let first = out[j..].bytes().next()?;
    (!first.is_ascii_digit() && first != b'.').then_some(j)
}

fn render_error_chain(err: &Error) -> String {
    let mut msg = format!("{err:#}");
    let mut source = std::error::Error::source(err);
    while let Some(next) = source {
        msg.push_str(&format!("\n  caused by: {next}"));
        source = next.source();
    }
    msg
}

// -------------------------------------------------------------------------------------------
// value bridging
// -------------------------------------------------------------------------------------------

/// minijinja `Value` -> `serde_json::Value`, preserving object key order.
fn to_json(v: &Value) -> Result<serde_json::Value, Error> {
    serde_json::to_value(v).map_err(|e| {
        Error::new(
            ErrorKind::InvalidOperation,
            format!("value is not JSON-serialisable: {e}"),
        )
    })
}

fn as_text(v: &Value) -> Result<Cow<'_, str>, Error> {
    if let Some(s) = v.as_str() {
        return Ok(Cow::Borrowed(s));
    }
    Ok(Cow::Owned(pysem::py_str(&to_json(v)?)))
}

fn opt_i64(v: Option<&Value>) -> Result<Option<i64>, Error> {
    match v {
        None => Ok(None),
        Some(v) if v.is_none() || v.is_undefined() => Ok(None),
        Some(v) => i64::try_from(v.clone()).map(Some),
    }
}

// -------------------------------------------------------------------------------------------
// filters
// -------------------------------------------------------------------------------------------

fn install_filters(env: &mut Environment<'static>) {
    // `tojson`: transformers replaces Jinja's HTML-escaping default with plain json.dumps.
    env.add_filter(
        "tojson",
        |v: &Value, kwargs: Kwargs| -> Result<Value, Error> {
            let opts = pysem::JsonDumpsOpts {
                ensure_ascii: kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false),
                indent: kwargs.get::<Option<usize>>("indent")?,
                sort_keys: kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false),
            };
            if kwargs.get::<Option<Value>>("separators")?.is_some() {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "tojson(separators=...) is not implemented",
                ));
            }
            kwargs.assert_all_used()?;
            Ok(Value::from(pysem::py_json_dumps_with(&to_json(v)?, opts)))
        },
    );

    // the reference's patch to the transformers environment; kept for template parity.
    env.add_filter("from_json", |v: &Value| -> Result<Value, Error> {
        match v.as_str() {
            Some(s) => {
                let parsed: serde_json::Value = serde_json::from_str(s).map_err(|e| {
                    Error::new(ErrorKind::InvalidOperation, format!("from_json: {e}"))
                })?;
                Ok(Value::from_serialize(&parsed))
            }
            None => Ok(v.clone()),
        }
    });

    // Jinja2 `string` is `soft_str` -> Python `str()`: `True`, not `true`; `None`, not `none`;
    // `1e+20`, not `100000000000000000000.0`. Worth 28/1000 records on the P-1e corpus.
    env.add_filter("string", |v: &Value| -> Result<Value, Error> {
        Ok(Value::from(pysem::py_str(&to_json(v)?)))
    });

    // Jinja2 `trim(value, chars=None)` -> Python `str.strip`, which also strips U+001C..U+001F.
    // Worth 44/1000 records on the P-1e corpus.
    env.add_filter(
        "trim",
        |v: &Value, args: Rest<Value>| -> Result<Value, Error> {
            let chars = args.first().and_then(Value::as_str).map(str::to_owned);
            let s = as_text(v)?;
            Ok(Value::from(pysem::py_strip(&s, chars.as_deref())))
        },
    );

    env.add_filter("lower", |v: &Value| -> Result<Value, Error> {
        Ok(Value::from(pysem::py_lower(&as_text(v)?)))
    });
    env.add_filter("upper", |v: &Value| -> Result<Value, Error> {
        Ok(Value::from(pysem::py_upper(&as_text(v)?)))
    });
}

// -------------------------------------------------------------------------------------------
// tests
// -------------------------------------------------------------------------------------------

fn install_tests(env: &mut Environment<'static>) {
    // Jinja2's `iterable` test is `try: iter(value); except TypeError: False`. A bare number is
    // not iterable; `None` is not iterable; an *undefined* raises UndefinedError rather than
    // testing false, and minijinja's built-in silently answers `true` for both.
    env.add_test("iterable", |v: &Value| -> Result<bool, Error> {
        if v.is_undefined() {
            return Err(Error::new(
                ErrorKind::UndefinedError,
                "'iterable' test applied to an undefined value",
            ));
        }
        if v.is_none() {
            return Ok(false);
        }
        Ok(matches!(
            v.kind(),
            ValueKind::String | ValueKind::Seq | ValueKind::Map | ValueKind::Iterable
        ))
    });
}

// -------------------------------------------------------------------------------------------
// globals
// -------------------------------------------------------------------------------------------

fn install_globals(env: &mut Environment<'static>) {
    // transformers: `def raise_exception(message): raise jinja2.exceptions.TemplateError(message)`
    env.add_function("raise_exception", |msg: &Value| -> Result<Value, Error> {
        Err(Error::new(
            ErrorKind::InvalidOperation,
            format!("{}", as_text(msg)?),
        ))
    });
    // `namespace()` is built into minijinja and matches Jinja's; nothing to install.
    // P27-8: `X[::-1]` (see `rewrite_reverse_slices`) with CPython's result: a list (a string
    // for a string); `None` / undefined give `[]`, as minijinja's slice did.
    env.add_function("__py_reverse", |v: &Value| -> Result<Value, Error> {
        match v.kind() {
            ValueKind::Undefined | ValueKind::None => Ok(Value::from(Vec::<Value>::new())),
            ValueKind::String => Ok(Value::from(
                v.as_str()
                    .unwrap_or_default()
                    .chars()
                    .rev()
                    .collect::<String>(),
            )),
            ValueKind::Seq | ValueKind::Iterable => {
                let mut items: Vec<Value> = v.try_iter()?.collect();
                items.reverse();
                Ok(Value::from(items))
            }
            kind => Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("value of type {kind} cannot be sliced"),
            )),
        }
    });
}

// -------------------------------------------------------------------------------------------
// Python string methods
// -------------------------------------------------------------------------------------------

/// Methods minijinja resolves through `set_unknown_method_callback`.
///
/// `minijinja-contrib`'s `pycompat` covers most of these but diverges from CPython on the ones
/// this template depends on, so the string methods are implemented here and `pycompat` is only
/// the fallback (it still supplies `dict.get/keys/values/items`, `list.count`, `str.format`,
/// the `is*` predicates and `title`/`capitalize`).
fn python_method_callback(
    state: &State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, Error> {
    if value.kind() == ValueKind::String {
        if let Some(rv) = string_method(value, method, args)? {
            return Ok(rv);
        }
    }
    minijinja_contrib::pycompat::unknown_method_callback(state, value, method, args)
}

#[allow(clippy::too_many_lines)] // one arm per Python method; splitting it hides the table
fn string_method(value: &Value, method: &str, args: &[Value]) -> Result<Option<Value>, Error> {
    let Some(s) = value.as_str() else {
        return Ok(None);
    };
    let invalid = |m: &str| Error::new(ErrorKind::InvalidOperation, m.to_owned());

    let rv = match method {
        "split" | "rsplit" => {
            let (sep, maxsplit): (Option<&Value>, Option<&Value>) = from_args(args)?;
            let sep_owned = match sep {
                None => None,
                Some(v) if v.is_none() => None,
                Some(v) => Some(
                    v.as_str()
                        .ok_or_else(|| invalid("separator must be a string"))?,
                ),
            };
            let maxsplit = opt_i64(maxsplit)?.unwrap_or(-1);
            let parts = if method == "split" {
                pysem::py_split(s, sep_owned, maxsplit)
            } else {
                pysem::py_rsplit(s, sep_owned, maxsplit)
            }
            .map_err(&invalid)?;
            Value::from(parts.into_iter().map(Value::from).collect::<Vec<_>>())
        }
        "strip" | "lstrip" | "rstrip" => {
            let (chars,): (Option<&Value>,) = from_args(args)?;
            let chars = match chars {
                None => None,
                Some(v) if v.is_none() => None,
                Some(v) => Some(
                    v.as_str()
                        .ok_or_else(|| invalid("chars must be a string"))?,
                ),
            };
            Value::from(match method {
                "strip" => pysem::py_strip(s, chars),
                "lstrip" => pysem::py_lstrip(s, chars),
                _ => pysem::py_rstrip(s, chars),
            })
        }
        "startswith" | "endswith" => {
            let (needle, start, end): (&Value, Option<&Value>, Option<&Value>) = from_args(args)?;
            let (start, end) = (opt_i64(start)?, opt_i64(end)?);
            let test = |n: &Value| -> Result<bool, Error> {
                let n = n.as_str().ok_or_else(|| {
                    invalid("startswith/endswith argument must be a string or tuple of strings")
                })?;
                Ok(if method == "startswith" {
                    pysem::py_startswith(s, n, start, end)
                } else {
                    pysem::py_endswith(s, n, start, end)
                })
            };
            let hit = if needle.as_str().is_some() {
                test(needle)?
            } else if matches!(needle.kind(), ValueKind::Seq | ValueKind::Iterable) {
                // Python accepts a *tuple* of candidates; any match wins.
                let mut hit = false;
                for candidate in needle.try_iter()? {
                    if test(&candidate)? {
                        hit = true;
                        break;
                    }
                }
                hit
            } else {
                return Err(invalid(
                    "startswith/endswith argument must be a string or tuple of strings",
                ));
            };
            Value::from(hit)
        }
        "find" | "rfind" | "index" | "rindex" => {
            let (needle, start, end): (&str, Option<&Value>, Option<&Value>) = from_args(args)?;
            let (start, end) = (opt_i64(start)?, opt_i64(end)?);
            let pos = if method.starts_with('r') {
                pysem::py_rfind(s, needle, start, end)
            } else {
                pysem::py_find(s, needle, start, end)
            };
            if pos < 0 && (method == "index" || method == "rindex") {
                return Err(invalid("substring not found"));
            }
            Value::from(pos)
        }
        "count" => {
            let (needle, start, end): (&str, Option<&Value>, Option<&Value>) = from_args(args)?;
            Value::from(pysem::py_count(s, needle, opt_i64(start)?, opt_i64(end)?))
        }
        "replace" => {
            let (old, new, count): (&str, &str, Option<&Value>) = from_args(args)?;
            Value::from(pysem::py_replace(
                s,
                old,
                new,
                opt_i64(count)?.unwrap_or(-1),
            ))
        }
        "removeprefix" => {
            let (prefix,): (&str,) = from_args(args)?;
            Value::from(s.strip_prefix(prefix).unwrap_or(s))
        }
        "removesuffix" => {
            let (suffix,): (&str,) = from_args(args)?;
            Value::from(s.strip_suffix(suffix).unwrap_or(s))
        }
        "lower" => {
            let () = from_args(args)?;
            Value::from(pysem::py_lower(s))
        }
        "upper" => {
            let () = from_args(args)?;
            Value::from(pysem::py_upper(s))
        }
        "join" => {
            let (items,): (&Value,) = from_args(args)?;
            let mut out = String::new();
            for (i, item) in items.try_iter()?.enumerate() {
                if i > 0 {
                    out.push_str(s);
                }
                out.push_str(&as_text(&item)?);
            }
            Value::from(out)
        }
        _ => return Ok(None),
    };
    Ok(Some(rv))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render_one(src: &str, ctx: serde_json::Value) -> String {
        let r = Renderer::from_source(src).expect("compile");
        let messages = ctx
            .get("messages")
            .and_then(|m| m.as_array())
            .cloned()
            .unwrap_or_default();
        let mut extra = serde_json::Map::new();
        if let serde_json::Value::Object(map) = ctx {
            for (k, v) in map {
                if k != "messages" {
                    extra.insert(k, v);
                }
            }
        }
        let req = ChatRequest {
            messages,
            tools: None,
            documents: None,
            add_generation_prompt: false,
            extra,
        };
        r.render(&req).expect("render")
    }

    #[test]
    fn tojson_uses_python_separators_and_float_repr() {
        let out = render_one(
            "{{ x | tojson }}",
            serde_json::json!({ "x": {"b": 1, "a": [1e-7, 1e20, 2.5], "u": "\u{e9}"} }),
        );
        assert_eq!(
            out,
            "{\"b\": 1, \"a\": [1e-07, 1e+20, 2.5], \"u\": \"\u{e9}\"}"
        );
    }

    #[test]
    fn string_filter_is_python_str() {
        assert_eq!(
            render_one("{{ x | string }}", serde_json::json!({"x": true})),
            "True"
        );
        assert_eq!(
            render_one("{{ x | string }}", serde_json::json!({"x": 2.5})),
            "2.5"
        );
    }

    #[test]
    fn python_string_methods() {
        assert_eq!(
            render_one("{{ x.split('.')[-1] }}", serde_json::json!({"x": "a.b.c"})),
            "c"
        );
        assert_eq!(
            render_one(
                "{{ x.split('.', 1) | join('|') }}",
                serde_json::json!({"x": "a.b.c"})
            ),
            "a|b.c"
        );
        // minijinja renders booleans Jinja-style ("True"/"False"), matching Python's str().
        assert_eq!(
            render_one(
                "{{ x.startswith(('z', 'a')) }}",
                serde_json::json!({"x": "abc"})
            ),
            "True"
        );
        assert_eq!(
            render_one("{{ x.lstrip('\\n') }}", serde_json::json!({"x": "\n\nq"})),
            "q"
        );
        assert_eq!(
            render_one(
                "{{ x.rfind('b') }}",
                serde_json::json!({"x": "\u{4e2d}b\u{4e2d}b"})
            ),
            "3"
        );
        assert_eq!(
            render_one(
                "{{ x.replace('a', 'z', 1) }}",
                serde_json::json!({"x": "aaa"})
            ),
            "zaa"
        );
        assert_eq!(
            render_one("{{ x.count('aa') }}", serde_json::json!({"x": "aaaa"})),
            "2"
        );
    }

    #[test]
    fn namespace_survives_macro_mutation() {
        let src = "{%- set ns = namespace(v=0) %}\
                   {%- macro bump() %}{%- set ns.v = ns.v + 1 %}{%- endmacro %}\
                   {{- bump() }}{{- bump() }}{{- ns.v }}";
        assert_eq!(render_one(src, serde_json::json!({})), "2");
    }

    #[test]
    fn raise_exception_fails_the_render() {
        let r = Renderer::from_source("{{ raise_exception('boom') }}").unwrap();
        let err = r.render(&ChatRequest::default()).unwrap_err();
        assert!(err.to_string().contains("boom"), "{err}");
    }

    #[test]
    fn iterable_test_matches_jinja() {
        assert_eq!(
            render_one("{{ x is iterable }}", serde_json::json!({"x": 5})),
            "False"
        );
        assert_eq!(
            render_one("{{ x is iterable }}", serde_json::json!({"x": null})),
            "False"
        );
        assert_eq!(
            render_one("{{ x is iterable }}", serde_json::json!({"x": "ab"})),
            "True"
        );
        let r = Renderer::from_source("{{ nope is iterable }}").unwrap();
        assert!(r.render(&ChatRequest::default()).is_err());
    }

    #[test]
    fn continue_final_message_is_refused() {
        let mut kwargs = serde_json::Map::new();
        kwargs.insert(
            "continue_final_message".to_owned(),
            serde_json::Value::Bool(true),
        );
        assert!(ChatRequest::from_template_kwargs(vec![], None, &kwargs).is_err());
    }

    #[test]
    fn p27_8_reverse_slices_are_rewritten_inside_tags_only() {
        let r = rewrite_reverse_slices;
        assert_eq!(
            r("{%- for m in messages[::-1] %}"),
            "{%- for m in __py_reverse(messages) %}"
        );
        assert_eq!(
            r("{%- for m in messages[head.count:][::-1] %}"),
            "{%- for m in __py_reverse(messages[head.count:]) %}"
        );
        assert_eq!(r("{{ a.b(c)[::-1] }}"), "{{ __py_reverse(a.b(c)) }}");
        assert_eq!(r("{{ (x + y)[::-1] }}"), "{{ __py_reverse((x + y)) }}");
        // Outside tags, in a comment, in a string literal, on a literal: untouched.
        for s in [
            "text x[::-1] {{ 1 }}",
            "{# x[::-1] #}",
            "{{ 'x[::-1]' }}",
            "{{ ''[::-1] }}",
            "{{ 12[::-1] }}",
        ] {
            assert_eq!(r(s), s, "{s}");
        }
    }

    #[test]
    fn p27_8_an_empty_reverse_slice_renders_pythons_empty_list() {
        // minijinja 2.24 panics on these (index 0 of an empty sequence); CPython gives [].
        let src = "{% for m in xs[::-1] %}{{ m }}{% endfor %}|{{ xs[::-1] | length }}|{{ s[::-1] }}|{{ ys[::-1] | join(',') }}|{{ ys[::-1][0] }}";
        let out = render_one(src, serde_json::json!({"xs": [], "s": "", "ys": [1, 2, 3]}));
        assert_eq!(out, "|0||3,2,1|3");
        let out = render_one("{{ s[::-1] }}", serde_json::json!({"s": "abc"}));
        assert_eq!(out, "cba");
    }

    #[test]
    fn p27_8_a_panic_in_the_engine_is_a_render_error() {
        // Negative control: a literal is not rewritten, so minijinja's own slice still panics on
        // it -- and the renderer returns an error instead of unwinding.
        let r = Renderer::from_source("{{ ''[::-1] }}").expect("compile");
        let err = r
            .render(&ChatRequest::default())
            .expect_err("the engine panics on an empty reverse slice");
        let text = err.to_string();
        assert!(text.contains("template engine panic"), "{text}");
        assert!(text.contains("index out of bounds"), "{text}");
    }
}
