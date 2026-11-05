//! The daily reader (`docs/SHADOW.md` §3): a mirror log and a replay file in, the comparison and
//! the six stop conditions out.
//!
//! What it compares, per exchange (matched by the mirror's `id`):
//!
//! * the status, and whether each side SERVED the request (2xx, body complete, no `error` event,
//!   no `abort` finish);
//! * the `usage` counts — `prompt_tokens` must be identical (the template is byte-identical),
//!   `completion_tokens` differ with the text and are reported;
//! * for a **greedy** request (`temperature` < 1e-5), the ids: `token_ids` when the request asked
//!   for them (the reference server's `return_token_ids`), otherwise the decoded text — content, reasoning and
//!   the tool calls, per choice — which is the ids through one fixed tokenizer;
//! * content, reasoning and tool-call JSON identity over every request, and the finish-reason
//!   table;
//! * latency percentiles (first byte and whole response) on each side, which say nothing about
//!   parity under replay (the arrival process is prod's, the load is not) and are printed as such.
//!
//! Identity is REPORTED, not a stop: numerics flips at P10's floor are expected. The stops are
//! the runbook's six, each printed as CLEAR, FIRED (with the evidence) or NOT CHECKED (with what
//! was missing to check it).

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::Serialize;
use serde_json::Value;

use crate::record::{
    served_family, Answer, Exchange, LogEntry, MetricsSnapshot, ReplayMeta, Response,
};

/// Census counters that must not move over a window (`docs/SHADOW.md` §2). `eager_steps` is NOT
/// here: a prefill chunk outside a captured bucket runs eager by declaration (the card's TP=1 serve
/// reads `path graph 2 / eager 1` for one request), and the counter does not split decode from
/// prefill — it is reported in `census moved`.
pub const MUST_BE_ZERO: &[&str] = &[
    "saber:engine_budget_mismatches_total",
    "saber:engine_undeclared_shapes_total",
    "saber:engine_shape_violations_total",
    "saber:engine_forward_errors_total",
    "saber:engine_unknown_sequences_total",
    "saber:engine_oversized_step_refusals_total",
    "saber:engine_argmax_fallbacks_total",
    "saber:engine_penalty_lists_too_wide_total",
    "saber:engine_logprobs_skipped_on_verify_total",
    "saber:engine_stale_anchor_defects_total",
    "saber:engine_proposal_collect_failures_total",
    "saber:engine_park_failures_total",
    "saber:engine_restore_failures_total",
    "saber:engine_fork_refusals_total",
    "saber:engine_fork_first_block_failures_total",
    "saber:draft_uncollected_total",
    "saber:sched_executor_protocol_errors_total",
    "saber:sched_executor_token_count_errors_total",
    "saber:sched_park_refusals_total",
    "saber:sched_restore_failures_total",
    "saber:sched_restores_given_up_total",
    "saber:sched_parked_discards_total",
];

/// What the reader was given beyond the two files.
#[derive(Clone, Debug, Default)]
pub struct ReaderOptions {
    /// Saber's stdout for the window (the boot, the `listening` line, the census, `stopped`).
    pub saber_log: Option<String>,
    /// The power cap every card must read, watts.
    pub cap_w: f64,
    /// Exchange ids whose Saber refusal is already on record as prod's own (V5's row 205).
    pub allowed_refusals: Vec<u64>,
}

/// One stop condition's verdict.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(
    tag = "status",
    content = "detail",
    rename_all = "SCREAMING_SNAKE_CASE"
)]
pub enum Verdict {
    /// Checked, nothing found.
    Clear,
    /// Checked, and it fired: the evidence.
    Fired(String),
    /// Could not be checked: why.
    NotChecked(String),
}

impl Verdict {
    fn word(&self) -> &'static str {
        match self {
            Self::Clear => "CLEAR",
            Self::Fired(_) => "FIRED",
            Self::NotChecked(_) => "NOT CHECKED",
        }
    }
}

/// Percentiles of one latency series, ms.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Pct {
    /// Samples.
    pub n: usize,
    /// p50.
    pub p50: f64,
    /// p90.
    pub p90: f64,
    /// p99.
    pub p99: f64,
}

fn pct(mut v: Vec<f64>) -> Pct {
    if v.is_empty() {
        return Pct::default();
    }
    v.sort_by(f64::total_cmp);
    let at = |q: f64| v[((q * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
    Pct {
        n: v.len(),
        p50: at(0.50),
        p90: at(0.90),
        p99: at(0.99),
    }
}

/// The report.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Report {
    /// Exchanges in the log.
    pub recorded: usize,
    /// Of those, answered by the replay.
    pub replayed: usize,
    /// The replay's pace and compression factor.
    pub pace: String,
    /// Prod did not serve (non-2xx, error, abort, incomplete).
    pub prod_not_served: usize,
    /// Saber did not serve.
    pub saber_not_served: usize,
    /// Status codes identical.
    pub status_identical: usize,
    /// Greedy exchanges both sides served.
    pub greedy_compared: usize,
    /// Of those, identical ids (or text).
    pub greedy_identical: usize,
    /// How many of the greedy comparisons were on ids rather than text.
    pub greedy_on_ids: usize,
    /// Greedy exchanges that differed, by id.
    pub greedy_differing: Vec<u64>,
    /// Exchanges both sides served.
    pub both_served: usize,
    /// Content identical (of both served).
    pub content_identical: usize,
    /// Reasoning identical.
    pub reasoning_identical: usize,
    /// Exchanges with a tool call on either side.
    pub tool_exchanges: usize,
    /// Of those, tool-call JSON identical.
    pub tool_identical: usize,
    /// Both sides reported usage.
    pub usage_compared: usize,
    /// `prompt_tokens` identical.
    pub prompt_tokens_identical: usize,
    /// `completion_tokens` identical.
    pub completion_tokens_identical: usize,
    /// `(prod finish, saber finish)` → count.
    pub finish_table: BTreeMap<String, usize>,
    /// Prod: first byte.
    pub prod_first_byte_ms: Pct,
    /// Prod: whole response.
    pub prod_total_ms: Pct,
    /// Saber: first byte.
    pub saber_first_byte_ms: Pct,
    /// Saber: whole response.
    pub saber_total_ms: Pct,
    /// Prod's acceptance over the log's metrics polls: accepted / drafts, and / draft tokens.
    pub prod_acceptance: Option<(f64, f64)>,
    /// Saber's over the replay.
    pub saber_acceptance: Option<(f64, f64)>,
    /// The census deltas the replay moved, every `saber:` counter that moved.
    pub census_moved: BTreeMap<String, f64>,
    /// The six stop conditions, in the runbook's order.
    pub stops: Vec<(String, Verdict)>,
}

impl Report {
    /// Whether any stop condition fired.
    pub fn fired(&self) -> bool {
        self.stops
            .iter()
            .any(|(_, v)| matches!(v, Verdict::Fired(_)))
    }

    /// The text a person reads.
    pub fn render(&self) -> String {
        let mut o = String::new();
        let _ = writeln!(
            o,
            "shadow report: {} recorded, {} replayed (pace {})",
            self.recorded, self.replayed, self.pace
        );
        let _ = writeln!(
            o,
            "served         prod not served {}, Saber not served {}, status identical {}/{}",
            self.prod_not_served, self.saber_not_served, self.status_identical, self.replayed
        );
        let _ = writeln!(
            o,
            "greedy         {}/{} identical ({} on ids, {} on text){}",
            self.greedy_identical,
            self.greedy_compared,
            self.greedy_on_ids,
            self.greedy_compared - self.greedy_on_ids,
            if self.greedy_differing.is_empty() {
                String::new()
            } else {
                format!(", differing ids {:?}", self.greedy_differing)
            }
        );
        let _ = writeln!(
            o,
            "identity       content {}/{}, reasoning {}/{}, tool-call JSON {}/{}",
            self.content_identical,
            self.both_served,
            self.reasoning_identical,
            self.both_served,
            self.tool_identical,
            self.tool_exchanges
        );
        let _ = writeln!(
            o,
            "usage          prompt_tokens identical {}/{}, completion_tokens identical {}/{}",
            self.prompt_tokens_identical,
            self.usage_compared,
            self.completion_tokens_identical,
            self.usage_compared
        );
        let finish: Vec<String> = self
            .finish_table
            .iter()
            .map(|(k, v)| format!("{k} {v}"))
            .collect();
        let _ = writeln!(o, "finish         {}", finish.join(", "));
        let p = |x: &Pct| {
            format!(
                "p50 {:.1} / p90 {:.1} / p99 {:.1} (n {})",
                x.p50, x.p90, x.p99, x.n
            )
        };
        let _ = writeln!(
            o,
            "latency (ms)   prod first byte {}, total {}",
            p(&self.prod_first_byte_ms),
            p(&self.prod_total_ms)
        );
        let _ = writeln!(
            o,
            "               Saber first byte {}, total {}  (replay: prod's arrivals, not prod's load)",
            p(&self.saber_first_byte_ms),
            p(&self.saber_total_ms)
        );
        let acc = |a: Option<(f64, f64)>| {
            a.map_or_else(
                || "n/a".to_owned(),
                |(d, t)| format!("{d:.3} accepted/draft, {t:.3} accepted/draft token"),
            )
        };
        let _ = writeln!(
            o,
            "acceptance     prod {} | Saber {}",
            acc(self.prod_acceptance),
            acc(self.saber_acceptance)
        );
        if !self.census_moved.is_empty() {
            let moved: Vec<String> = self
                .census_moved
                .iter()
                .map(|(k, v)| format!("{} {v}", k.trim_start_matches("saber:")))
                .collect();
            let _ = writeln!(o, "census moved   {}", moved.join(", "));
        }
        let _ = writeln!(o, "stop conditions:");
        for (i, (name, v)) in self.stops.iter().enumerate() {
            let detail = match v {
                Verdict::Clear => String::new(),
                Verdict::Fired(d) | Verdict::NotChecked(d) => format!(" — {d}"),
            };
            let _ = writeln!(o, "  {}. {:<11} {name}{detail}", i + 1, v.word());
        }
        let _ = writeln!(
            o,
            "verdict        {}",
            if self.fired() {
                "STOP: a stop condition fired (docs/SHADOW.md §3)"
            } else {
                "no stop condition fired"
            }
        );
        o
    }
}

/// One choice as a client assembled it.
#[derive(Clone, Debug, Default, PartialEq)]
struct ChoiceOut {
    content: String,
    reasoning: String,
    /// Tool calls by their index: name and the concatenated arguments.
    tools: BTreeMap<u64, (String, String)>,
    finish: Option<String>,
    ids: Option<Vec<u64>>,
}

/// A response as a client assembled it.
#[derive(Clone, Debug, Default)]
struct Parsed {
    status: Option<u16>,
    choices: BTreeMap<u64, ChoiceOut>,
    usage: Option<(u64, u64)>,
    error_event: bool,
    complete: bool,
}

impl Parsed {
    fn served(&self) -> bool {
        self.status.is_some_and(|s| (200..300).contains(&s))
            && self.complete
            && !self.error_event
            && !self
                .choices
                .values()
                .any(|c| c.finish.as_deref() == Some("abort"))
    }
}

fn fold_choice(out: &mut ChoiceOut, c: &Value) {
    let part = c.get("delta").or_else(|| c.get("message")).unwrap_or(c);
    if let Some(s) = part.get("content").and_then(Value::as_str) {
        out.content.push_str(s);
    }
    if let Some(s) = c.get("text").and_then(Value::as_str) {
        out.content.push_str(s);
    }
    for key in ["reasoning", "reasoning_content"] {
        if let Some(s) = part.get(key).and_then(Value::as_str) {
            out.reasoning.push_str(s);
        }
    }
    if let Some(calls) = part.get("tool_calls").and_then(Value::as_array) {
        for (pos, call) in calls.iter().enumerate() {
            let idx = call
                .get("index")
                .and_then(Value::as_u64)
                .unwrap_or(pos as u64);
            let e = out.tools.entry(idx).or_default();
            if let Some(f) = call.get("function") {
                if let Some(n) = f.get("name").and_then(Value::as_str) {
                    e.0.push_str(n);
                }
                if let Some(a) = f.get("arguments").and_then(Value::as_str) {
                    e.1.push_str(a);
                }
            }
        }
    }
    if let Some(f) = c.get("finish_reason").and_then(Value::as_str) {
        out.finish = Some(f.to_owned());
    }
    if let Some(ids) = c.get("token_ids").and_then(Value::as_array) {
        out.ids
            .get_or_insert_with(Vec::new)
            .extend(ids.iter().filter_map(Value::as_u64));
    }
}

fn fold_object(p: &mut Parsed, v: &Value) {
    if v.get("error").is_some() && v.get("choices").is_none() {
        p.error_event = true;
        return;
    }
    if let Some(choices) = v.get("choices").and_then(Value::as_array) {
        for (pos, c) in choices.iter().enumerate() {
            let idx = c.get("index").and_then(Value::as_u64).unwrap_or(pos as u64);
            fold_choice(p.choices.entry(idx).or_default(), c);
        }
    }
    if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
        let g = |k: &str| u.get(k).and_then(Value::as_u64).unwrap_or(0);
        p.usage = Some((g("prompt_tokens"), g("completion_tokens")));
    }
}

fn parse(resp: &Response) -> Parsed {
    let mut p = Parsed {
        status: resp.status,
        complete: resp.complete && resp.error.is_none(),
        ..Parsed::default()
    };
    let body = resp.body();
    let text = String::from_utf8_lossy(&body);
    let sse =
        resp.headers.iter().any(|(k, v)| {
            k.eq_ignore_ascii_case("content-type") && v.contains("text/event-stream")
        }) || text.trim_start().starts_with("data:");
    if sse {
        for line in text.lines() {
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                continue;
            }
            match serde_json::from_str::<Value>(data) {
                Ok(v) => fold_object(&mut p, &v),
                Err(_) => p.error_event = true,
            }
        }
    } else if let Ok(v) = serde_json::from_str::<Value>(&text) {
        fold_object(&mut p, &v);
    }
    p
}

fn is_greedy(x: &Exchange) -> bool {
    serde_json::from_slice::<Value>(&x.body.bytes())
        .ok()
        .and_then(|v| v.get("temperature").and_then(Value::as_f64))
        .is_some_and(|t| t < 1e-5)
}

/// Prometheus text → `name{labels}` → value. A serving family ([`served_family`]) is keyed
/// under `saber:` whatever namespace the exporter used, so prod's polls and Saber's read alike.
pub fn parse_metrics(text: &str) -> BTreeMap<String, f64> {
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .filter_map(|l| {
            let (k, v) = l.rsplit_once(' ')?;
            let k = served_family(k).map_or_else(|| k.to_owned(), |rest| format!("saber:{rest}"));
            Some((k, v.parse().ok()?))
        })
        .collect()
}

/// The sum of a family over its label sets.
fn family(m: &BTreeMap<String, f64>, name: &str) -> Option<f64> {
    let mut found = false;
    let mut sum = 0.0;
    for (k, v) in m {
        let base = k.split('{').next().unwrap_or(k);
        if base == name {
            found = true;
            sum += v;
        }
    }
    found.then_some(sum)
}

fn acceptance(before: &BTreeMap<String, f64>, after: &BTreeMap<String, f64>) -> Option<(f64, f64)> {
    let d = |n: &str| Some(family(after, n)? - family(before, n).unwrap_or(0.0));
    let drafts = d("saber:spec_decode_num_drafts_total")?;
    let tokens = d("saber:spec_decode_num_draft_tokens_total")?;
    let accepted = d("saber:spec_decode_num_accepted_tokens_total")?;
    (drafts > 0.0).then(|| (accepted / drafts, accepted / tokens.max(1.0)))
}

/// Builds the report.
pub fn compare(
    log: &[LogEntry],
    meta: &ReplayMeta,
    answers: &[Answer],
    opts: &ReaderOptions,
) -> Report {
    let exchanges: Vec<&Exchange> = log
        .iter()
        .filter_map(|e| match e {
            LogEntry::Exchange(x) => Some(x.as_ref()),
            _ => None,
        })
        .collect();
    let polls: Vec<&MetricsSnapshot> = log
        .iter()
        .filter_map(|e| match e {
            LogEntry::Metrics(m) => Some(m),
            _ => None,
        })
        .collect();
    let by_id: BTreeMap<u64, &Answer> = answers.iter().map(|a| (a.id, a)).collect();

    let mut r = Report {
        recorded: exchanges.len(),
        pace: if meta.pace == "recorded" {
            format!("recorded, speed x{}", meta.speed)
        } else {
            meta.pace.clone()
        },
        ..Report::default()
    };
    let (mut refused, mut prompt_differs, mut not_transparent) =
        (Vec::new(), Vec::new(), Vec::new());
    let mut replayed_prompt_tokens = 0u64;
    let mut usage_unknown = 0usize;
    let (mut pf, mut pt, mut sf, mut st) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());

    for x in &exchanges {
        // Stop 6 is read off the RECORD, replayed or not.
        if !x.client_gone
            && (x.upstream_fnv != x.forwarded_fnv || x.upstream_bytes != x.forwarded_bytes)
        {
            not_transparent.push(format!(
                "#{}: upstream {} B {} vs forwarded {} B {}",
                x.id, x.upstream_bytes, x.upstream_fnv, x.forwarded_bytes, x.forwarded_fnv
            ));
        }
        let Some(a) = by_id.get(&x.id) else { continue };
        r.replayed += 1;
        let prod = parse(&x.response);
        let saber = parse(&a.response);
        if prod.status == saber.status {
            r.status_identical += 1;
        }
        pf.extend(x.response.t_first_byte_ms);
        pt.extend(x.response.t_done_ms);
        sf.extend(a.response.t_first_byte_ms);
        st.extend(a.response.t_done_ms);
        if !prod.served() {
            r.prod_not_served += 1;
        }
        if !saber.served() {
            r.saber_not_served += 1;
            if prod.served() && !opts.allowed_refusals.contains(&x.id) {
                refused.push(format!(
                    "#{} (Saber status {:?}{})",
                    x.id,
                    saber.status,
                    if saber.error_event {
                        ", error event"
                    } else {
                        ""
                    }
                ));
            }
        }
        match saber.usage {
            Some((p, _)) => replayed_prompt_tokens += p,
            // A request Saber answered without `usage` (a stream without `include_usage`) makes
            // the prompt-token half of the foreign-traffic check unreadable.
            None if saber.served() => usage_unknown += 1,
            None => {}
        }
        if let (Some(pu), Some(su)) = (prod.usage, saber.usage) {
            r.usage_compared += 1;
            if pu.0 == su.0 {
                r.prompt_tokens_identical += 1;
            } else {
                prompt_differs.push(format!("#{} ({} vs {})", x.id, pu.0, su.0));
            }
            if pu.1 == su.1 {
                r.completion_tokens_identical += 1;
            }
        }
        let finish = |p: &Parsed| {
            let f: Vec<&str> = p
                .choices
                .values()
                .map(|c| c.finish.as_deref().unwrap_or("-"))
                .collect();
            if f.is_empty() {
                format!(
                    "status {}",
                    p.status.map_or("-".to_owned(), |s| s.to_string())
                )
            } else {
                f.join("+")
            }
        };
        *r.finish_table
            .entry(format!("{} -> {}", finish(&prod), finish(&saber)))
            .or_default() += 1;
        if !(prod.served() && saber.served()) {
            continue;
        }
        r.both_served += 1;
        let pairs: Vec<(Option<&ChoiceOut>, Option<&ChoiceOut>)> = prod
            .choices
            .keys()
            .chain(saber.choices.keys())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|k| (prod.choices.get(k), saber.choices.get(k)))
            .collect();
        let all = |f: &dyn Fn(&ChoiceOut, &ChoiceOut) -> bool| {
            pairs.iter().all(|(a, b)| match (a, b) {
                (Some(a), Some(b)) => f(a, b),
                _ => false,
            })
        };
        if all(&|a, b| a.content == b.content) {
            r.content_identical += 1;
        }
        if all(&|a, b| a.reasoning == b.reasoning) {
            r.reasoning_identical += 1;
        }
        let has_tools = pairs.iter().any(|(a, b)| {
            a.is_some_and(|c| !c.tools.is_empty()) || b.is_some_and(|c| !c.tools.is_empty())
        });
        if has_tools {
            r.tool_exchanges += 1;
            if all(&|a, b| tools_equal(&a.tools, &b.tools)) {
                r.tool_identical += 1;
            }
        }
        if is_greedy(x) {
            r.greedy_compared += 1;
            let on_ids = pairs.iter().all(|(a, b)| {
                a.is_some_and(|c| c.ids.is_some()) && b.is_some_and(|c| c.ids.is_some())
            });
            if on_ids {
                r.greedy_on_ids += 1;
            }
            let same = if on_ids {
                all(&|a, b| a.ids == b.ids)
            } else {
                all(&|a, b| {
                    a.content == b.content
                        && a.reasoning == b.reasoning
                        && tools_equal(&a.tools, &b.tools)
                })
            };
            if same {
                r.greedy_identical += 1;
            } else {
                r.greedy_differing.push(x.id);
            }
        }
    }
    r.prod_first_byte_ms = pct(pf);
    r.prod_total_ms = pct(pt);
    r.saber_first_byte_ms = pct(sf);
    r.saber_total_ms = pct(st);
    if let (Some(first), Some(last)) = (polls.first(), polls.last()) {
        r.prod_acceptance = acceptance(&parse_metrics(&first.text), &parse_metrics(&last.text));
    }

    // ---- the six stops ------------------------------------------------------------------------
    let before = meta.metrics_before.as_deref().map(parse_metrics);
    let after = meta.metrics_after.as_deref().map(parse_metrics);
    if let (Some(b), Some(a)) = (&before, &after) {
        r.saber_acceptance = acceptance(b, a);
        // The census is Saber's own counters; the serving families are reported above.
        let census = |k: &&String| k.starts_with("saber:") && served_family(k).is_none();
        for k in a.keys().filter(census) {
            let d = a[k] - b.get(k).copied().unwrap_or(0.0);
            if d != 0.0 && !k.starts_with("saber:engine_clean") {
                r.census_moved.insert(k.clone(), d);
            }
        }
    }
    let stop1 = match (&before, &after) {
        (Some(b), Some(a)) => {
            let missing: Vec<&str> = MUST_BE_ZERO
                .iter()
                .copied()
                .filter(|n| family(a, n).is_none())
                .collect();
            if missing.len() == MUST_BE_ZERO.len() {
                Verdict::NotChecked(
                    "the target's /metrics carries no saber: census (a build before ws-v8)".into(),
                )
            } else {
                let mut moved: Vec<String> = MUST_BE_ZERO
                    .iter()
                    .filter_map(|n| {
                        let d = family(a, n)? - family(b, n).unwrap_or(0.0);
                        (d != 0.0).then(|| format!("{} +{d}", n.trim_start_matches("saber:")))
                    })
                    .collect();
                if family(a, "saber:engine_clean") == Some(0.0) {
                    moved.push("engine_clean 0".to_owned());
                }
                if !missing.is_empty() {
                    moved.push(format!("not on /metrics: {}", missing.join(", ")));
                }
                if moved.is_empty() {
                    Verdict::Clear
                } else {
                    Verdict::Fired(moved.join("; "))
                }
            }
        }
        _ => Verdict::NotChecked("the replay did not scrape the target's /metrics".into()),
    };
    let stop2 = if r.replayed == 0 {
        Verdict::NotChecked("nothing replayed".into())
    } else if refused.is_empty() {
        Verdict::Clear
    } else {
        Verdict::Fired(format!(
            "prod served, Saber did not: {}",
            refused.join(", ")
        ))
    };
    let stop3 = if r.usage_compared == 0 {
        Verdict::NotChecked("no exchange carried usage on both sides".into())
    } else if prompt_differs.is_empty() {
        Verdict::Clear
    } else {
        Verdict::Fired(format!(
            "prompt_tokens differ: {}",
            prompt_differs.join(", ")
        ))
    };
    let stop4 = match &opts.saber_log {
        None => Verdict::NotChecked("no --saber-log".into()),
        Some(log) => {
            let mut bad = Vec::new();
            if !log.contains("listening") {
                bad.push("no `listening` line".to_owned());
            }
            if !log.contains("stopped") {
                bad.push("no `stopped` line (not the window's SIGTERM)".to_owned());
            }
            for needle in ["stuck ", "panicked", "hipError", "HIP error"] {
                if log.contains(needle) {
                    bad.push(format!("`{needle}` in the log"));
                }
            }
            if bad.is_empty() {
                Verdict::Clear
            } else {
                Verdict::Fired(bad.join("; "))
            }
        }
    };
    let stop5 = {
        let mut bad = Vec::new();
        let mut checked = Vec::new();
        if meta.power_caps_w.is_empty() {
            checked.push("power cap not read");
        } else {
            for (card, w) in &meta.power_caps_w {
                if (w - opts.cap_w).abs() > 0.5 {
                    bad.push(format!("{card} at {w} W (want {})", opts.cap_w));
                }
            }
        }
        if let (Some(b), Some(a)) = (&before, &after) {
            let d = |n: &str| family(a, n).unwrap_or(0.0) - family(b, n).unwrap_or(0.0);
            let seen = d("saber:request_success_total") + d("saber:request_refused_total");
            if seen > meta.replayed as f64 {
                bad.push(format!(
                    "the target finished or refused {seen} requests and the replay sent {}: \
                     foreign traffic on the port",
                    meta.replayed
                ));
            }
            let prompt = d("saber:prompt_tokens_total");
            if usage_unknown == 0 && prompt > replayed_prompt_tokens as f64 {
                bad.push(format!(
                    "the target prefilled {prompt} prompt tokens and the replay's requests carry \
                     {replayed_prompt_tokens}: foreign traffic on the port"
                ));
            }
        } else {
            checked.push("foreign traffic not checked (no /metrics scrape)");
        }
        if !bad.is_empty() {
            Verdict::Fired(bad.join("; "))
        } else if checked.len() == 2 {
            Verdict::NotChecked(checked.join("; "))
        } else if checked.is_empty() {
            Verdict::Clear
        } else {
            // Half checked is reported as such: the half that ran is clear.
            Verdict::NotChecked(format!("{} (the rest is clear)", checked.join("; ")))
        }
    };
    let stop6 = if exchanges.is_empty() {
        Verdict::NotChecked("no exchange recorded".into())
    } else if not_transparent.is_empty() {
        Verdict::Clear
    } else {
        Verdict::Fired(format!(
            "the mirror changed what a client received: {}",
            not_transparent.join(", ")
        ))
    };
    r.stops = vec![
        ("a census field that must read 0 moved".to_owned(), stop1),
        (
            "prod served (2xx), Saber refused or aborted".to_owned(),
            stop2,
        ),
        ("usage.prompt_tokens differs".to_owned(), stop3),
        (
            "Saber exited other than by SIGTERM, a HIP error, or no `listening`".to_owned(),
            stop4,
        ),
        (
            "a power cap other than 210 W, or foreign traffic on the port".to_owned(),
            stop5,
        ),
        (
            "the mirror changed what prod's clients received".to_owned(),
            stop6,
        ),
    ];
    r
}

/// Tool calls equal: names equal and arguments equal as JSON (key order and whitespace do not
/// make two calls different; a different value does).
fn tools_equal(a: &BTreeMap<u64, (String, String)>, b: &BTreeMap<u64, (String, String)>) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|((ia, (na, aa)), (ib, (nb, ab)))| {
                ia == ib
                    && na == nb
                    && match (
                        serde_json::from_str::<Value>(aa),
                        serde_json::from_str::<Value>(ab),
                    ) {
                        (Ok(x), Ok(y)) => x == y,
                        _ => aa == ab,
                    }
            })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{Blob, Chunk};

    fn resp(body: &str, sse: bool) -> Response {
        Response {
            status: Some(200),
            headers: vec![(
                "content-type".into(),
                if sse {
                    "text/event-stream"
                } else {
                    "application/json"
                }
                .into(),
            )],
            chunks: vec![Chunk {
                t_ms: 1.0,
                data: Blob::of(body.as_bytes()),
            }],
            t_first_byte_ms: Some(1.0),
            t_done_ms: Some(2.0),
            complete: true,
            error: None,
        }
    }

    #[test]
    fn sse_and_json_assemble_to_the_same_choice() {
        let sse = "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n\
                   data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"hm\"}}]}\n\n\
                   data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n\
                   data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" there\"},\"finish_reason\":\"stop\"}]}\n\n\
                   data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3}}\n\n\
                   data: [DONE]\n\n";
        let json = r#"{"choices":[{"index":0,"message":{"content":"Hi there","reasoning":"hm"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3}}"#;
        let a = parse(&resp(sse, true));
        let b = parse(&resp(json, false));
        assert_eq!(a.choices, b.choices);
        assert_eq!(a.usage, Some((5, 3)));
        assert!(a.served() && b.served());
        // An error event mid-stream is not served.
        let bad = parse(&resp(
            "data: {\"error\":{\"message\":\"x\"}}\n\ndata: [DONE]\n\n",
            true,
        ));
        assert!(!bad.served());
    }

    #[test]
    fn tool_arguments_compare_as_json() {
        let mut a = BTreeMap::new();
        a.insert(0, ("f".to_owned(), r#"{"x": 1, "y": 2}"#.to_owned()));
        let mut b = BTreeMap::new();
        b.insert(0, ("f".to_owned(), r#"{"y":2,"x":1}"#.to_owned()));
        assert!(tools_equal(&a, &b));
        b.insert(0, ("f".to_owned(), r#"{"y":3,"x":1}"#.to_owned()));
        assert!(
            !tools_equal(&a, &b),
            "a different value is a different call"
        );
    }

    fn metrics(eager: u64, clean: u64, success: u64) -> String {
        let mut t = String::new();
        for n in MUST_BE_ZERO {
            let v = if *n == "saber:engine_budget_mismatches_total" {
                eager
            } else {
                0
            };
            t.push_str(&format!(
                "{n}{{engine=\"0\",model_name=\"Qwen3.8\"}} {v}.0\n"
            ));
        }
        t.push_str(&format!(
            "saber:engine_clean{{engine=\"0\",model_name=\"Qwen3.8\"}} {clean}.0\n"
        ));
        t.push_str(&format!(
            "saber:request_success_total{{engine=\"0\",finished_reason=\"stop\",model_name=\"Qwen3.8\"}} {success}.0\n"
        ));
        t
    }

    fn meta(before: String, after: String, replayed: u64, caps: &[f64]) -> ReplayMeta {
        ReplayMeta {
            pace: "sequential".into(),
            replayed,
            metrics_before: Some(before),
            metrics_after: Some(after),
            power_caps_w: caps
                .iter()
                .map(|w| ("after:card1".to_owned(), *w))
                .collect(),
            ..ReplayMeta::default()
        }
    }

    /// Stop 1 reads the census deltas; stop 5 the caps and the request count. Each with the
    /// control that must fire.
    #[test]
    fn stop_1_and_stop_5_read_the_census_and_the_caps() {
        let opts = ReaderOptions {
            cap_w: 210.0,
            ..ReaderOptions::default()
        };
        let clear = compare(
            &[],
            &meta(metrics(3, 1, 10), metrics(3, 1, 12), 2, &[210.0]),
            &[],
            &opts,
        );
        assert_eq!(clear.stops[0].1, Verdict::Clear, "{}", clear.render());
        assert_eq!(clear.stops[4].1, Verdict::Clear, "{}", clear.render());
        // CONTROLS: a budget mismatch during the window; clean going to 0; a cap at 300 W; a request
        // the replay did not send.
        let moved = compare(
            &[],
            &meta(metrics(3, 1, 10), metrics(4, 0, 13), 2, &[300.0]),
            &[],
            &opts,
        );
        match &moved.stops[0].1 {
            Verdict::Fired(d) => assert!(
                d.contains("budget_mismatches_total +1") && d.contains("engine_clean 0"),
                "{d}"
            ),
            other => panic!("{other:?}"),
        }
        match &moved.stops[4].1 {
            Verdict::Fired(d) => {
                assert!(d.contains("300 W") && d.contains("foreign traffic"), "{d}")
            }
            other => panic!("{other:?}"),
        }
        // A target without the census is NOT CHECKED, never CLEAR.
        let old = compare(&[], &meta(String::new(), String::new(), 0, &[]), &[], &opts);
        assert!(matches!(old.stops[0].1, Verdict::NotChecked(_)));
    }

    #[test]
    fn percentiles_are_order_statistics() {
        let p = pct((1..=100).map(f64::from).collect());
        assert_eq!((p.p50, p.p90, p.p99), (51.0, 90.0, 99.0));
        assert_eq!(pct(Vec::new()), Pct::default());
    }
}
