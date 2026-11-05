//! FN66: WARM PROMPTS -- the user's repeating system prompt(s) prefilled once at boot, so the
//! first turn of every session is a prompt-cache hit instead of a full prefill.
//!
//! `--warm-prompts FILE` (both serve planes): after the server listens, every entry of the file
//! is rendered THROUGH THE CHAT REQUEST PATH ([`crate::server::chat_kwargs`] +
//! [`crate::server::render_chat_ids`]: the tool-call law, the conversation form, the template,
//! the tokenizer) as the head of a conversation, and its ids up to the start of the next user
//! turn are submitted to the engine as one greedy one-token request -- a prefill whose prefix the
//! engine's cache keeps (the 27B: `--enable-prefix-caching`'s forced cut at the prefix's last
//! whole block; Flash-Next: the prompt cache's warm cut at the prefix's last token). The sampled
//! token is discarded. A later request whose prompt begins with the same ids reloads the prefix.
//!
//! **The prefix law.** A real turn 1 renders `head + [user]` with the generation prompt; its ids
//! up to the user turn's `<|im_start|>` are the head's. The runner renders `head + [user
//! SENTINEL]` WITHOUT the generation prompt (a template may refuse a conversation with no user
//! turn, Flash-Next's does) and cuts before the last `<|im_start|>`: exactly those ids. The
//! sentinel's text never reaches the engine.
//!
//! **The file** (the user's, never invented here): a JSON array of entries, one JSON entry, JSON
//! lines (one entry a line), or plain text (the whole file = ONE system prompt). An entry is a
//! string (a system prompt), an array of chat messages (a conversation head: system and any
//! fixed turns), or an object `{"messages": [...], "tools": [...], "chat_template_kwargs": {...},
//! "reasoning_effort": "...", "label": "..."}` -- the fields a client sends that change the
//! rendered head (tools render into the system turn).

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use futures_util::StreamExt;
use serde_json::{json, Map, Value};

use crate::engine::{EngineRequest, TokenId};
use crate::protocol::ChatCompletionRequest;
use crate::sampling::SamplingParams;
use crate::state::ServerState;

/// The placeholder user turn the head is rendered before (cut off: never sent to the engine).
pub const SENTINEL: &str = "warm";

/// One warm prompt: a conversation head and the request fields that render with it.
#[derive(Debug, Clone)]
pub struct WarmPrompt {
    pub label: String,
    pub request: ChatCompletionRequest,
}

/// What the engine's cache holds now: (bytes, a description for the boot log).
pub type CacheProbe = Arc<dyn Fn() -> (u64, String) + Send + Sync>;

/// One warm prompt's result.
#[derive(Debug, Clone, PartialEq)]
pub struct WarmReport {
    pub label: String,
    /// The prefix's tokens (what the engine prefilled).
    pub tokens: usize,
    /// The request's wall: the prefill + one sampled token.
    pub ms: f64,
    /// The cache's bytes before and after (the engine's probe).
    pub bytes_before: u64,
    pub bytes_after: u64,
}

fn entry(v: Value, i: usize) -> Result<WarmPrompt, String> {
    let label = format!("#{}", i + 1);
    let (label, req) = match v {
        Value::String(s) => (
            label,
            json!({"messages": [{"role": "system", "content": s}]}),
        ),
        Value::Array(_) => (label, json!({ "messages": v })),
        Value::Object(ref o) if o.contains_key("messages") => {
            let label = o
                .get("label")
                .and_then(Value::as_str)
                .map_or(label, str::to_owned);
            let mut o: Map<String, Value> = o.clone();
            o.remove("label");
            (label, Value::Object(o))
        }
        Value::Object(_) if v.get("role").is_some() => (label, json!({ "messages": [v] })),
        _ => {
            return Err(format!(
                "entry {}: a string, an array of messages or an object with \"messages\"",
                i + 1
            ))
        }
    };
    let request: ChatCompletionRequest =
        serde_json::from_value(req).map_err(|e| format!("entry {}: {e}", i + 1))?;
    if request.messages.is_empty() {
        return Err(format!("entry {}: no messages", i + 1));
    }
    Ok(WarmPrompt { label, request })
}

fn is_message(v: &Value) -> bool {
    v.get("role").is_some()
}

/// Parse a warm-prompts file's text (module doc: JSON array / one JSON entry / JSON lines / plain
/// text = one system prompt).
pub fn parse(text: &str) -> Result<Vec<WarmPrompt>, String> {
    let t = text.trim();
    if t.is_empty() {
        return Err("the warm-prompts file is empty".to_owned());
    }
    if t.starts_with('[') || t.starts_with('{') || t.starts_with('"') {
        if let Ok(v) = serde_json::from_str::<Value>(t) {
            return match v {
                // an array of messages is ONE conversation head; else an array of entries
                Value::Array(a) if !a.is_empty() && a.iter().all(is_message) => {
                    Ok(vec![entry(Value::Array(a), 0)?])
                }
                Value::Array(a) => a
                    .into_iter()
                    .enumerate()
                    .map(|(i, e)| entry(e, i))
                    .collect(),
                v => Ok(vec![entry(v, 0)?]),
            };
        }
        let lines: Vec<&str> = t.lines().filter(|l| !l.trim().is_empty()).collect();
        let parsed: Result<Vec<Value>, _> = lines
            .iter()
            .map(|l| serde_json::from_str::<Value>(l))
            .collect();
        if let Ok(vs) = parsed {
            return vs
                .into_iter()
                .enumerate()
                .map(|(i, e)| entry(e, i))
                .collect();
        }
    }
    Ok(vec![entry(Value::String(text.to_owned()), 0)?])
}

/// Read and parse `path`.
pub fn load(path: &Path) -> anyhow::Result<Vec<WarmPrompt>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("--warm-prompts {}: {e}", path.display()))?;
    parse(&text).map_err(|e| anyhow::anyhow!("--warm-prompts {}: {e}", path.display()))
}

/// The ids a real turn 1 after `p`'s head shares with every other (module doc: the prefix law).
pub fn prefix_ids(state: &ServerState, p: &WarmPrompt) -> Result<Vec<TokenId>, String> {
    let mut req = p.request.clone();
    req.messages
        .push(json!({"role": "user", "content": SENTINEL}));
    req.add_generation_prompt = Some(false);
    crate::reqcheck::validate_chat_shape_for(&req, state.tool_arguments)
        .map_err(|e| e.to_string())?;
    let kwargs = crate::server::chat_kwargs(state, &req);
    let ids = crate::server::render_chat_ids(state, &req.messages, &req, &kwargs)
        .map_err(|e| e.to_string())?;
    let im = state
        .codec
        .encode("<|im_start|>")
        .map_err(|e| e.to_string())?;
    let [im] = im[..] else {
        return Err(format!("<|im_start|> encodes to {im:?}, not one id"));
    };
    let j = cut_before_last(&ids, im).ok_or("the rendered head has no user turn to cut at")?;
    Ok(ids[..j].to_vec())
}

/// The index of the last `tok` in `ids` when there is something before it.
#[must_use]
pub fn cut_before_last(ids: &[TokenId], tok: TokenId) -> Option<usize> {
    ids.iter().rposition(|&t| t == tok).filter(|&j| j > 0)
}

/// Wait until the probe reads the same bytes twice 50 ms apart (an async store has landed), at
/// most ~5 s.
async fn settle(probe: &CacheProbe) -> (u64, String) {
    let mut last = probe();
    for _ in 0..100 {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let now = probe();
        if now.0 == last.0 {
            return now;
        }
        last = now;
    }
    last
}

/// Run every warm prompt in order (one at a time: each prefix lands before the next starts) and
/// print one `warmprompt` line each + the `warmprompts done` line the launchers wait for.
pub async fn run(
    state: Arc<ServerState>,
    prompts: Vec<WarmPrompt>,
    probe: CacheProbe,
) -> Vec<WarmReport> {
    let n = prompts.len();
    let t_all = Instant::now();
    let mut out = Vec::with_capacity(n);
    let mut failed = 0usize;
    for (i, p) in prompts.into_iter().enumerate() {
        let ids = match prefix_ids(&state, &p) {
            Ok(ids) => ids,
            Err(e) => {
                println!("warmprompt {}/{n} {}: REFUSED: {e}", i + 1, p.label);
                failed += 1;
                continue;
            }
        };
        let (before, _) = settle(&probe).await;
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: Some(1),
            ..SamplingParams::default()
        };
        let tokens = ids.len();
        let t0 = Instant::now();
        let sub = state.engine.submit(EngineRequest {
            prompt_token_ids: ids,
            params,
        });
        let mut sub = match sub {
            Ok(s) => s,
            Err(e) => {
                println!("warmprompt {}/{n} {}: ENGINE REFUSED: {e}", i + 1, p.label);
                failed += 1;
                continue;
            }
        };
        while let Some(ev) = sub.events.next().await {
            if ev.finished.is_some() {
                break;
            }
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let (after, desc) = settle(&probe).await;
        println!(
            "warmprompt {}/{n} {}: {tokens} tokens, prefill {ms:.1} ms, cache +{:.1} MiB ({desc})",
            i + 1,
            p.label,
            after.saturating_sub(before) as f64 / f64::from(1u32 << 20)
        );
        out.push(WarmReport {
            label: p.label,
            tokens,
            ms,
            bytes_before: before,
            bytes_after: after,
        });
    }
    println!(
        "warmprompts done: {} of {n} prompts, {} tokens, {:.0} ms, {failed} refused",
        out.len(),
        out.iter().map(|r| r.tokens).sum::<usize>(),
        t_all.elapsed().as_secs_f64() * 1e3
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sys(p: &WarmPrompt) -> &str {
        p.request.messages[0]["content"].as_str().unwrap()
    }

    #[test]
    fn the_file_forms_parse_to_conversation_heads() {
        // plain text: the whole file is one system prompt (even with JSON-looking lines inside)
        let p = parse("You are Pi.\n{not json}\n").unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(sys(&p[0]), "You are Pi.\n{not json}\n");
        // one messages array = one head
        let p =
            parse(r#"[{"role":"system","content":"A"},{"role":"user","content":"x"}]"#).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].request.messages.len(), 2);
        // an array of entries: string, messages, object with tools + kwargs + label
        let p = parse(
            r#"["S1", [{"role":"system","content":"S2"}],
                {"label":"pi","messages":[{"role":"system","content":"S3"}],
                 "tools":[{"type":"function","function":{"name":"f","parameters":{}}}],
                 "chat_template_kwargs":{"enable_thinking":false}}]"#,
        )
        .unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(sys(&p[0]), "S1");
        assert_eq!(sys(&p[1]), "S2");
        assert_eq!(p[2].label, "pi");
        assert_eq!(p[2].request.tools.as_ref().unwrap().len(), 1);
        assert_eq!(
            p[2].request.chat_template_kwargs.as_ref().unwrap()["enable_thinking"],
            false
        );
        // JSON lines
        let p =
            parse("\"A\"\n\n{\"messages\":[{\"role\":\"system\",\"content\":\"B\"}]}\n").unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(sys(&p[1]), "B");
        // refusals by name
        assert!(parse("  \n").is_err());
        assert!(parse("[1]").is_err());
        assert!(parse(r#"{"messages":[]}"#).is_err());
    }

    #[test]
    fn the_cut_is_before_the_last_turn_start() {
        assert_eq!(cut_before_last(&[1, 5, 9, 1, 7, 8], 1), Some(3));
        assert_eq!(cut_before_last(&[1, 7, 8], 1), None, "nothing before it");
        assert_eq!(cut_before_last(&[7, 8], 1), None);
    }
}
