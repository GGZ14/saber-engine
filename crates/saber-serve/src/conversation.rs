//! FN28: the conversation prod the reference 0.27.1 renders ([`crate::state::ConversationForm::Prod0271`]).
//!
//! Prod never hands the request's messages to the template. Two steps rebuild them first (read in
//! prod's image `the reference image`, the reference 0.27.1):
//!
//! 1. `upstream/entrypoints/openai/chat_completion/protocol.py` 507-532,
//!    `ChatCompletionRequest._normalize_messages_before` (a `mode="before"` validator, so before
//!    pydantic sees the body): on EVERY message, whatever its role, `reasoning_content` is popped,
//!    and when it is not `null` and `reasoning` is absent or `null` it becomes `reasoning`
//!    (529-531). A sent `reasoning_content` is therefore never dropped: it is `reasoning` unless a
//!    non-null `reasoning` was sent beside it, which wins.
//! 2. `upstream/entrypoints/chat_utils.py` 1834-1908, `_parse_chat_message_content`, run by
//!    `parse_chat_messages` (1954-1990) for every message; its result is the conversation
//!    `_postprocess_messages` (1911-1951, FN27's arguments law) then walks and the template renders.
//!    The rebuilt message carries, in this order, and NOTHING else:
//!    * `role` (1842, 1674/1691 `ConversationMessage(role=role, content=...)`);
//!    * `content` (1843-1857, 1652-1691): absent / `null` -> `[]`; a string -> one text part; a
//!      list -> its parts one by one (`_parse_chat_message_content_part`, 1712-1831) in the
//!      `openai` content format (`wrap_dicts`, 1852) -- the 27B's `qwen-fixed-v22.3.jinja` is
//!      detected as `openai` (`hf._detect_content_format`, measured): a plain string part and a
//!      `text` / `input_text` / `output_text` / `refusal` / `thinking` part become
//!      `{"type": "text", "text": ...}` followed by the part's extra fields (the ones outside
//!      `_KNOWN_CONTENT_PART_FIELDS`, e.g. `cache_control`; 1756-1764), a `text` / `refusal` part
//!      whose value is `null` or absent is skipped (1746-1754, `PART_TYPES_TO_SKIP_NONE_CONTENT`),
//!      a media part becomes `{"type": <modality>}` + extras (1810-1818; images:
//!      `crate::images::extract` already did it), a `tool_reference` part
//!      `{"type": "tool_reference", "name": ...}` (1796-1801);
//!    * an ASSISTANT's `tool_calls` when the key is present and not `null` (1864-1865), then
//!      `reasoning` and `reasoning_content`, BOTH the (normalised) `reasoning`, when that is not
//!      `null` (1867-1872); every other assistant key (`thinking`, `refusal`, `audio`, `function_call`,
//!      ...) is dropped;
//!    * a TOOL's `tool_call_id` when the key is present, even `null` (1874-1876), and its content
//!      list flattened to a string -- the texts joined with `"\n"` (`""` for none) -- unless some
//!      part is not a text part, when the list stays (1877-1896);
//!    * `name` when it is a string, `task` when it is a string -- any role (1898-1902);
//!    * a DEVELOPER's `tools`, `null` when not sent (1904-1905).
//!
//!    `reasoning` of any other role, a non-tool's `tool_call_id`, a non-assistant's `tool_calls`
//!    and every unknown key are dropped.
//!
//! The reference server 0.29 runs the very same code (`chat_utils.py` 1966-2040, `protocol.py` 521-547 in
//! `saber-base:0.29.0-test2`: the function body is byte-identical).
//!
//! Named edges, not reproduced (no corpus body has them; prod's verdicts are errors): a text part
//! carrying a `uuid` (prod: "Missing 'type' field"), a tool content list holding a `thinking` part
//! whose value is `null` (prod: `TypeError` in the join).

use serde_json::{Map, Value};

use crate::images::KNOWN_PART_FIELDS;

/// Prod 0.27.1's rebuilt conversation for `messages` (see the module docs for the law).
/// A message that is not an object is passed through (the shape check refused it earlier).
#[must_use]
pub fn conversation_0271(messages: &[Value]) -> Vec<Value> {
    messages.iter().map(message_0271).collect()
}

fn message_0271(message: &Value) -> Value {
    let Some(sent) = message.as_object() else {
        return message.clone();
    };
    // protocol.py 529-531: `reasoning_content` -> `reasoning`, any role.
    let mut reasoning = sent.get("reasoning").filter(|v| !v.is_null()).cloned();
    if reasoning.is_none() {
        reasoning = sent
            .get("reasoning_content")
            .filter(|v| !v.is_null())
            .cloned();
    }
    let role = sent.get("role").cloned().unwrap_or(Value::Null);
    let role_str = role.as_str().unwrap_or_default();
    let mut out = Map::new();
    out.insert("role".to_owned(), role.clone());
    let mut content = content_0271(sent.get("content"));
    if role_str == "tool" {
        content = flatten_tool_content(content);
    }
    out.insert("content".to_owned(), content);
    match role_str {
        "assistant" => {
            if let Some(calls) = sent.get("tool_calls").filter(|v| !v.is_null()) {
                out.insert("tool_calls".to_owned(), calls.clone());
            }
            if let Some(r) = reasoning {
                out.insert("reasoning".to_owned(), r.clone());
                out.insert("reasoning_content".to_owned(), r);
            }
        }
        "tool" => {
            if let Some(id) = sent.get("tool_call_id") {
                out.insert("tool_call_id".to_owned(), id.clone());
            }
        }
        _ => {}
    }
    for key in ["name", "task"] {
        if let Some(v) = sent.get(key).filter(|v| v.is_string()) {
            out.insert(key.to_owned(), v.clone());
        }
    }
    if role_str == "developer" {
        out.insert(
            "tools".to_owned(),
            sent.get("tools").cloned().unwrap_or(Value::Null),
        );
    }
    Value::Object(out)
}

/// `content` in the `openai` content format (chat_utils.py 1843-1857, 1712-1831).
fn content_0271(content: Option<&Value>) -> Value {
    match content {
        None | Some(Value::Null) => Value::Array(Vec::new()),
        Some(Value::String(s)) => Value::Array(vec![text_part(Value::String(s.clone()), None)]),
        Some(Value::Array(parts)) => Value::Array(parts.iter().filter_map(part_0271).collect()),
        Some(other) => other.clone(),
    }
}

fn text_part(text: Value, extras_of: Option<&Map<String, Value>>) -> Value {
    let mut out = Map::new();
    out.insert("type".to_owned(), Value::String("text".to_owned()));
    out.insert("text".to_owned(), text);
    if let Some(part) = extras_of {
        for (k, v) in part {
            if !KNOWN_PART_FIELDS.contains(&k.as_str()) {
                out.insert(k.clone(), v.clone());
            }
        }
    }
    Value::Object(out)
}

/// One content part; `None` when prod skips it (1746-1754).
fn part_0271(part: &Value) -> Option<Value> {
    let map = match part {
        Value::String(s) => return Some(text_part(Value::String(s.clone()), None)),
        Value::Object(map) => map,
        other => return Some(other.clone()),
    };
    if map.get("uuid").is_some_and(|u| !u.is_null()) {
        return Some(part.clone()); // named edge (module docs)
    }
    let kind = map.get("type").and_then(Value::as_str).unwrap_or_default();
    let field = match kind {
        "text" | "input_text" | "output_text" => "text",
        "refusal" => "refusal",
        "thinking" => "thinking",
        "tool_reference" => {
            let mut out = Map::new();
            out.insert(
                "type".to_owned(),
                Value::String("tool_reference".to_owned()),
            );
            out.insert(
                "name".to_owned(),
                map.get("name").cloned().unwrap_or(Value::Null),
            );
            return Some(Value::Object(out));
        }
        // Media parts: `crate::images::extract` normalised images already; the others are
        // refused by the shape check before this runs.
        _ => return Some(part.clone()),
    };
    let value = map.get(field).cloned().unwrap_or(Value::Null);
    if value.is_null() && matches!(kind, "text" | "refusal") {
        return None;
    }
    Some(text_part(value, Some(map)))
}

/// A tool message's content list -> `"\n"`-joined texts, unless a part is not text (1877-1896).
fn flatten_tool_content(content: Value) -> Value {
    let Value::Array(parts) = content else {
        return content;
    };
    let non_text = parts.iter().any(|p| {
        p.as_object()
            .is_some_and(|m| m.get("type") != Some(&Value::String("text".to_owned())))
    });
    if non_text {
        return Value::Array(parts);
    }
    let texts: Vec<String> = parts
        .iter()
        .filter_map(Value::as_object)
        .map(|m| match m.get("text") {
            Some(Value::String(s)) => s.clone(),
            None => String::new(),
            Some(other) => other.to_string(),
        })
        .collect();
    Value::String(texts.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn one(m: Value) -> Value {
        conversation_0271(&[m]).remove(0)
    }

    /// Every expectation below is prod 0.27.1's own output for the same message (measured with
    /// `ChatCompletionRequest.model_validate` + `parse_chat_messages(..., "openai")` in prod's
    /// image; FN28).
    #[test]
    fn assistant_reasoning_keys_as_prod_0271() {
        let tc = json!([{"id": "c", "type": "function", "function": {"name": "f", "arguments": {"a": 1}}}]);
        // reasoning + reasoning_content + thinking: reasoning wins, thinking dropped, content null -> []
        let got = one(
            json!({"role": "assistant", "content": null, "reasoning": "R1",
            "thinking": "T1", "reasoning_content": "RC1", "tool_calls": tc.clone(),
            "refusal": "x", "audio": null}),
        );
        assert_eq!(
            serde_json::to_string(&got).unwrap(),
            serde_json::to_string(
                &json!({"role": "assistant", "content": [], "tool_calls": tc,
                "reasoning": "R1", "reasoning_content": "R1"})
            )
            .unwrap()
        );
        // reasoning_content alone -> reasoning
        assert_eq!(
            one(json!({"role": "assistant", "content": "ok", "reasoning_content": "RC2"})),
            json!({"role": "assistant", "content": [{"type": "text", "text": "ok"}],
                "reasoning": "RC2", "reasoning_content": "RC2"})
        );
        // reasoning null + reasoning_content -> reasoning_content
        assert_eq!(
            one(
                json!({"role": "assistant", "content": "ok", "reasoning": null, "reasoning_content": "X"})
            )["reasoning"],
            json!("X")
        );
        // "" is not None: it wins over reasoning_content
        assert_eq!(
            one(
                json!({"role": "assistant", "content": "ok", "reasoning": "", "reasoning_content": "X"})
            )["reasoning_content"],
            json!("")
        );
        // thinking alone: dropped
        assert_eq!(
            one(json!({"role": "assistant", "content": "ok", "thinking": "T3"})),
            json!({"role": "assistant", "content": [{"type": "text", "text": "ok"}]})
        );
        // null tool_calls / reasoning: dropped
        assert_eq!(
            one(
                json!({"role": "assistant", "content": "ok", "reasoning": null, "tool_calls": null})
            ),
            json!({"role": "assistant", "content": [{"type": "text", "text": "ok"}]})
        );
    }

    #[test]
    fn content_parts_as_prod_0271() {
        assert_eq!(
            one(
                json!({"role": "user", "content": [{"type": "text", "text": "a"},
                {"type": "text", "text": "b", "cache_control": {"x": 1}},
                {"type": "thinking", "thinking": "zz"}, {"type": "refusal", "refusal": "no"},
                {"type": "text", "text": null}]})
            ),
            json!({"role": "user", "content": [{"type": "text", "text": "a"},
                {"type": "text", "text": "b", "cache_control": {"x": 1}},
                {"type": "text", "text": "zz"}, {"type": "text", "text": "no"}]})
        );
        assert_eq!(
            one(json!({"role": "user"})),
            json!({"role": "user", "content": []})
        );
        // an already-normalised image part passes
        let img = json!({"type": "image", "k": 1});
        assert_eq!(
            one(json!({"role": "user", "content": [img.clone()]}))["content"][0],
            img
        );
    }

    #[test]
    fn other_roles_as_prod_0271() {
        // tool: list flattened, non-string name dropped, tool_call_id kept
        assert_eq!(
            one(json!({"role": "tool", "tool_call_id": "c", "name": 7,
                "content": [{"type": "text", "text": "t1"}, {"type": "text", "text": "t2"}]})),
            json!({"role": "tool", "content": "t1\nt2", "tool_call_id": "c"})
        );
        assert_eq!(
            one(json!({"role": "tool", "content": "t"}))["content"],
            json!("t")
        );
        assert_eq!(
            one(json!({"role": "tool", "content": null}))["content"],
            json!("")
        );
        // system: extras dropped, string name kept
        assert_eq!(
            one(json!({"role": "system", "content": "sys", "name": "n1", "extra": 1})),
            json!({"role": "system", "content": [{"type": "text", "text": "sys"}], "name": "n1"})
        );
        // developer: tools = null
        assert_eq!(
            one(json!({"role": "developer", "content": "dev"})),
            json!({"role": "developer", "content": [{"type": "text", "text": "dev"}], "tools": null})
        );
        // an unknown role: reasoning / tool_call_id dropped, task kept
        assert_eq!(
            one(
                json!({"role": "weird", "content": "w", "reasoning": "r", "tool_call_id": "z", "task": "tk"})
            ),
            json!({"role": "weird", "content": [{"type": "text", "text": "w"}], "task": "tk"})
        );
    }
}
