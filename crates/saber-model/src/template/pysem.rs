//! Python semantics that Jinja2 templates inherit from the host language.
//!
//! `qwen-fixed-v22.3.jinja` is rendered by CPython's Jinja2, so every string method, every
//! `tojson`, and every implicit `str()` in it has *Python* behaviour, not Rust's. This module
//! is the compatibility layer: pure functions, no minijinja types, unit-tested against values
//! taken from CPython 3.12.
//!
//! The divergences that actually bite (all covered below):
//!
//! | construct | Rust default | Python |
//! |---|---|---|
//! | `trim` / `strip()` | `char::is_whitespace` | also U+001C..U+001F |
//! | `find` / `rfind` | byte offset | code-point offset |
//! | `json.dumps` separators | `,` `:` | `, ` `: ` |
//! | `json.dumps` floats | `1e-7`, `100000000000000000000` | `1e-07`, `1e+20` |
//! | `str(True)` | `true` | `True` |
//! | `str.lower()` | contextual final sigma | per-character mapping |

/// `str.isspace()` for a single code point (CPython `Py_UNICODE_ISSPACE`).
///
/// Rust's `char::is_whitespace` is the Unicode `White_Space` property, which omits the four
/// C0 separators U+001C..U+001F that Python treats as whitespace.
#[must_use]
pub fn is_py_space(c: char) -> bool {
    matches!(c,
        '\u{09}'..='\u{0d}'
        | '\u{1c}'..='\u{1f}'
        | '\u{20}'
        | '\u{85}'
        | '\u{a0}'
        | '\u{1680}'
        | '\u{2000}'..='\u{200a}'
        | '\u{2028}'
        | '\u{2029}'
        | '\u{202f}'
        | '\u{205f}'
        | '\u{3000}'
    )
}

/// `str.strip()` / `str.strip(chars)`.
#[must_use]
pub fn py_strip<'a>(s: &'a str, chars: Option<&str>) -> &'a str {
    py_rstrip(py_lstrip(s, chars), chars)
}

/// `str.lstrip()` / `str.lstrip(chars)`.
#[must_use]
pub fn py_lstrip<'a>(s: &'a str, chars: Option<&str>) -> &'a str {
    match chars {
        None => s.trim_start_matches(is_py_space),
        Some(set) => s.trim_start_matches(|c| set.contains(c)),
    }
}

/// `str.rstrip()` / `str.rstrip(chars)`.
#[must_use]
pub fn py_rstrip<'a>(s: &'a str, chars: Option<&str>) -> &'a str {
    match chars {
        None => s.trim_end_matches(is_py_space),
        Some(set) => s.trim_end_matches(|c| set.contains(c)),
    }
}

/// `str.lower()`.
///
/// Applied per character on purpose: Rust's `str::to_lowercase` implements the *conditional*
/// Final_Sigma rule (U+03A3 at the end of a word becomes U+03C2), which CPython's `str.lower()`
/// does not. Per-character `char::to_lowercase` still performs the unconditional full mappings
/// (e.g. U+0130 -> "i\u{307}"), which CPython does perform.
#[must_use]
pub fn py_lower(s: &str) -> String {
    s.chars().flat_map(char::to_lowercase).collect()
}

/// `str.upper()`.
#[must_use]
pub fn py_upper(s: &str) -> String {
    s.chars().flat_map(char::to_uppercase).collect()
}

/// `str.split(sep, maxsplit)`.
///
/// `sep == None` splits on runs of whitespace and drops leading/trailing empties;
/// `sep == Some("")` is a `ValueError` in Python and is reported as `Err` here.
pub fn py_split(s: &str, sep: Option<&str>, maxsplit: i64) -> Result<Vec<String>, &'static str> {
    match sep {
        None => Ok(py_split_whitespace(s, maxsplit)),
        Some("") => Err("empty separator"),
        Some(sep) => Ok(if maxsplit < 0 {
            s.split(sep).map(str::to_owned).collect()
        } else {
            // Python's maxsplit counts splits; Rust's splitn counts pieces.
            let n = usize::try_from(maxsplit)
                .unwrap_or(usize::MAX)
                .saturating_add(1);
            s.splitn(n, sep).map(str::to_owned).collect()
        }),
    }
}

fn py_split_whitespace(s: &str, maxsplit: i64) -> Vec<String> {
    if maxsplit < 0 {
        return s.split_whitespace().map(str::to_owned).collect();
    }
    let limit = usize::try_from(maxsplit).unwrap_or(usize::MAX);
    let mut out: Vec<String> = Vec::new();
    let mut rest = s.trim_start_matches(is_py_space);
    while out.len() < limit && !rest.is_empty() {
        match rest.find(is_py_space) {
            Some(idx) => {
                out.push(rest[..idx].to_owned());
                rest = rest[idx..].trim_start_matches(is_py_space);
            }
            None => {
                out.push(rest.to_owned());
                rest = "";
            }
        }
    }
    if !rest.is_empty() {
        // Python keeps the remainder verbatim, including its trailing whitespace.
        out.push(rest.to_owned());
    }
    out
}

/// `str.rsplit(sep, maxsplit)` (separator form only, which is all templates use).
pub fn py_rsplit(s: &str, sep: Option<&str>, maxsplit: i64) -> Result<Vec<String>, &'static str> {
    match sep {
        None => {
            let mut v = py_split_whitespace(s, -1);
            if maxsplit >= 0 {
                let keep = usize::try_from(maxsplit).unwrap_or(usize::MAX);
                if v.len() > keep + 1 {
                    // not reachable from the production template; keep it honest anyway
                    let head = v.drain(..v.len() - keep).collect::<Vec<_>>().join(" ");
                    v.insert(0, head);
                }
            }
            Ok(v)
        }
        Some("") => Err("empty separator"),
        Some(sep) => {
            let mut v: Vec<String> = if maxsplit < 0 {
                s.rsplit(sep).map(str::to_owned).collect()
            } else {
                let n = usize::try_from(maxsplit)
                    .unwrap_or(usize::MAX)
                    .saturating_add(1);
                s.rsplitn(n, sep).map(str::to_owned).collect()
            };
            v.reverse();
            Ok(v)
        }
    }
}

/// Resolve Python's `start`/`end` slice arguments to a byte range of `s`.
fn py_slice_bounds(s: &str, start: Option<i64>, end: Option<i64>) -> (usize, usize, usize) {
    let idx: Vec<usize> = s.char_indices().map(|(i, _)| i).chain([s.len()]).collect();
    let n = idx.len() - 1; // number of code points
    let clamp = |v: i64| -> usize {
        let v = if v < 0 { v + n as i64 } else { v };
        v.clamp(0, n as i64) as usize
    };
    let cs = start.map_or(0, clamp);
    let ce = end.map_or(n, clamp);
    let ce = ce.max(cs);
    (idx[cs], idx[ce], cs)
}

/// `str.find(sub, start, end)` — returns a **code-point** index, or -1.
#[must_use]
pub fn py_find(s: &str, sub: &str, start: Option<i64>, end: Option<i64>) -> i64 {
    let (b0, b1, c0) = py_slice_bounds(s, start, end);
    match s[b0..b1].find(sub) {
        Some(off) => (c0 + s[b0..b0 + off].chars().count()) as i64,
        None => -1,
    }
}

/// `str.rfind(sub, start, end)` — returns a **code-point** index, or -1.
#[must_use]
pub fn py_rfind(s: &str, sub: &str, start: Option<i64>, end: Option<i64>) -> i64 {
    let (b0, b1, c0) = py_slice_bounds(s, start, end);
    match s[b0..b1].rfind(sub) {
        Some(off) => (c0 + s[b0..b0 + off].chars().count()) as i64,
        None => -1,
    }
}

/// `str.count(sub, start, end)` — non-overlapping occurrences.
#[must_use]
pub fn py_count(s: &str, sub: &str, start: Option<i64>, end: Option<i64>) -> i64 {
    let (b0, b1, _) = py_slice_bounds(s, start, end);
    let hay = &s[b0..b1];
    if sub.is_empty() {
        return hay.chars().count() as i64 + 1;
    }
    hay.matches(sub).count() as i64
}

/// `str.replace(old, new, count)`.
#[must_use]
pub fn py_replace(s: &str, old: &str, new: &str, count: i64) -> String {
    if count < 0 {
        s.replace(old, new)
    } else {
        s.replacen(old, new, usize::try_from(count).unwrap_or(usize::MAX))
    }
}

/// `str.startswith(prefix, start, end)` for a single prefix.
#[must_use]
pub fn py_startswith(s: &str, prefix: &str, start: Option<i64>, end: Option<i64>) -> bool {
    let (b0, b1, _) = py_slice_bounds(s, start, end);
    s[b0..b1].starts_with(prefix)
}

/// `str.endswith(suffix, start, end)` for a single suffix.
#[must_use]
pub fn py_endswith(s: &str, suffix: &str, start: Option<i64>, end: Option<i64>) -> bool {
    let (b0, b1, _) = py_slice_bounds(s, start, end);
    s[b0..b1].ends_with(suffix)
}

// ---------------------------------------------------------------------------------------------
// numbers
// ---------------------------------------------------------------------------------------------

/// CPython's `repr(float)` / `str(float)` (they are the same since 3.1), which is also what
/// `json.dumps` emits for floats.
///
/// Shortest round-trip digits, then: exponential notation when `decpt <= -4 || decpt > 16`,
/// with a signed, at-least-two-digit exponent; otherwise positional with a forced `.0`.
#[must_use]
pub fn py_float_repr(v: f64) -> String {
    if v.is_nan() {
        return "nan".to_owned();
    }
    if v.is_infinite() {
        return if v > 0.0 {
            "inf".to_owned()
        } else {
            "-inf".to_owned()
        };
    }
    // Rust's LowerExp gives the shortest round-trip digits in `d[.ddd]e<exp>` form.
    let formatted = format!("{v:e}");
    let (mantissa, exponent) = formatted
        .split_once('e')
        .expect("LowerExp always emits an exponent");
    let exp: i32 = exponent.parse().expect("LowerExp emits a decimal exponent");
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    let decpt = exp + 1; // digits are written as 0.<digits> * 10^decpt

    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        out.push('e');
        out.push(if e < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", e.abs()));
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..-decpt {
            out.push('0');
        }
        out.push_str(digits);
    } else if decpt as usize >= digits.len() {
        out.push_str(digits);
        for _ in 0..(decpt as usize - digits.len()) {
            out.push('0');
        }
        out.push_str(".0");
    } else {
        let split = decpt as usize;
        out.push_str(&digits[..split]);
        out.push('.');
        out.push_str(&digits[split..]);
    }
    out
}

// ---------------------------------------------------------------------------------------------
// json.dumps
// ---------------------------------------------------------------------------------------------

/// Options for [`py_json_dumps_with`], mirroring `json.dumps`'s keyword arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonDumpsOpts {
    /// `ensure_ascii` — escape every non-ASCII code point as `\uXXXX` (surrogate pairs above
    /// the BMP). The chat-template filter sets this to `false`.
    pub ensure_ascii: bool,
    /// `indent` — `None` for the compact `(", ", ": ")` separators, `Some(n)` for pretty output
    /// with `(",", ": ")` separators and `n` spaces per level.
    pub indent: Option<usize>,
    /// `sort_keys` — sort object keys instead of keeping insertion order.
    pub sort_keys: bool,
}

impl Default for JsonDumpsOpts {
    /// The defaults transformers passes: `ensure_ascii=False, indent=None, sort_keys=False`.
    fn default() -> Self {
        Self {
            ensure_ascii: false,
            indent: None,
            sort_keys: false,
        }
    }
}

/// `json.dumps(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False)` —
/// exactly the `tojson` filter transformers installs in the chat-template environment
/// (`transformers.utils.chat_template_utils._cached_compile_jinja_template`).
#[must_use]
pub fn py_json_dumps(v: &serde_json::Value) -> String {
    py_json_dumps_with(v, JsonDumpsOpts::default())
}

/// `json.dumps` with explicit options.
#[must_use]
pub fn py_json_dumps_with(v: &serde_json::Value, opts: JsonDumpsOpts) -> String {
    let mut out = String::new();
    write_json(&mut out, v, opts, 0);
    out
}

fn sorted_entries(
    map: &serde_json::Map<String, serde_json::Value>,
    opts: JsonDumpsOpts,
) -> Vec<(&String, &serde_json::Value)> {
    let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
    if opts.sort_keys {
        entries.sort_by(|a, b| a.0.cmp(b.0));
    }
    entries
}

fn write_json(out: &mut String, v: &serde_json::Value, opts: JsonDumpsOpts, depth: usize) {
    // `indent=None` -> item separator ", "; `indent=n` -> "," plus a newline and n*level spaces.
    let newline = |out: &mut String, level: usize| {
        if let Some(indent) = opts.indent {
            out.push('\n');
            for _ in 0..indent * level {
                out.push(' ');
            }
        }
    };
    match v {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(true) => out.push_str("true"),
        serde_json::Value::Bool(false) => out.push_str("false"),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else if let Some(f) = n.as_f64() {
                out.push_str(&py_json_float(f));
            } else {
                out.push_str(&n.to_string());
            }
        }
        serde_json::Value::String(s) => write_json_string(out, s, opts),
        serde_json::Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if opts.indent.is_none() {
                        out.push(' ');
                    }
                }
                newline(out, depth + 1);
                write_json(out, item, opts, depth + 1);
            }
            newline(out, depth);
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, val)) in sorted_entries(map, opts).into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if opts.indent.is_none() {
                        out.push(' ');
                    }
                }
                newline(out, depth + 1);
                write_json_string(out, k, opts);
                out.push_str(": ");
                write_json(out, val, opts, depth + 1);
            }
            newline(out, depth);
            out.push('}');
        }
    }
}

/// `json.dumps` uses `float.__repr__` but spells the non-finite values differently.
fn py_json_float(f: f64) -> String {
    if f.is_nan() {
        "NaN".to_owned()
    } else if f.is_infinite() {
        if f > 0.0 {
            "Infinity".to_owned()
        } else {
            "-Infinity".to_owned()
        }
    } else {
        py_float_repr(f)
    }
}

/// CPython `json.encoder.py_encode_basestring` (`ensure_ascii=False`): escape only `"`, `\`
/// and C0 control characters; everything else, including all non-ASCII, is emitted raw.
/// With `ensure_ascii=True` it is `py_encode_basestring_ascii`, which also escapes every
/// non-ASCII code point, using a UTF-16 surrogate pair above the BMP.
fn write_json_string(out: &mut String, s: &str, opts: JsonDumpsOpts) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if opts.ensure_ascii && !c.is_ascii() => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

// ---------------------------------------------------------------------------------------------
// str() / repr()
// ---------------------------------------------------------------------------------------------

/// Python `str(x)` for a JSON-shaped value — what Jinja2's `string` filter (`soft_str`) does.
#[must_use]
pub fn py_str(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// Python `repr(x)` for a JSON-shaped value.
///
/// Known gap: `repr` of a *non-printable non-ASCII* code point inside a container would be
/// `\uXXXX`-escaped by CPython; we emit it raw, because deciding "printable" needs the Unicode
/// category tables. Every printable character, and every ASCII character, is exact.
#[must_use]
pub fn py_repr(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "None".to_owned(),
        serde_json::Value::Bool(true) => "True".to_owned(),
        serde_json::Value::Bool(false) => "False".to_owned(),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else if let Some(f) = n.as_f64() {
                py_float_repr(f)
            } else {
                n.to_string()
            }
        }
        serde_json::Value::String(s) => py_str_repr(s),
        serde_json::Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(py_repr).collect();
            format!("[{}]", inner.join(", "))
        }
        serde_json::Value::Object(map) => {
            if map.is_empty() {
                return "{}".to_owned();
            }
            let inner: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", py_str_repr(k), py_repr(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

/// Python `repr()` of a `str`.
#[must_use]
pub fn py_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every expected value below is CPython 3.12 output.
    #[test]
    fn float_repr_matches_cpython() {
        for (v, want) in [
            (0.0_f64, "0.0"),
            (-0.0_f64, "-0.0"),
            (1.0, "1.0"),
            (2.5, "2.5"),
            (30.0, "30.0"),
            (120.5, "120.5"),
            (-17.25, "-17.25"),
            (1e-7, "1e-07"),
            (1e20, "1e+20"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.0 / 3.0, "0.3333333333333333"),
            (1.5e-300, "1.5e-300"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
            (123456789012345.6, "123456789012345.6"),
        ] {
            assert_eq!(py_float_repr(v), want, "repr({v:?})");
        }
    }

    #[test]
    fn json_dumps_matches_cpython() {
        // CPython: json.dumps({"b": 1, "a": [1, 2.5, None, True],
        //                      "s": 'q"\\\n\x07\xe9\u4e2d'}, ensure_ascii=False)
        let mut obj = serde_json::Map::new();
        obj.insert("b".into(), serde_json::json!(1));
        obj.insert("a".into(), serde_json::json!([1, 2.5, null, true]));
        obj.insert("s".into(), serde_json::json!("q\"\\\n\u{7}\u{e9}\u{4e2d}"));
        let v = serde_json::Value::Object(obj);
        assert_eq!(
            py_json_dumps(&v),
            "{\"b\": 1, \"a\": [1, 2.5, null, true], \"s\": \"q\\\"\\\\\\n\\u0007\u{e9}\u{4e2d}\"}"
        );
        assert_eq!(py_json_dumps(&serde_json::json!({})), "{}");
        assert_eq!(py_json_dumps(&serde_json::json!([])), "[]");
    }

    #[test]
    fn json_dumps_options_match_cpython() {
        let mut obj = serde_json::Map::new();
        obj.insert("b".into(), serde_json::json!([1, 2]));
        obj.insert("a".into(), serde_json::json!("\u{e9}"));
        let v = serde_json::Value::Object(obj);
        // json.dumps(v, ensure_ascii=True)
        assert_eq!(
            py_json_dumps_with(
                &v,
                JsonDumpsOpts {
                    ensure_ascii: true,
                    ..Default::default()
                }
            ),
            "{\"b\": [1, 2], \"a\": \"\\u00e9\"}"
        );
        // json.dumps({"a": "\U0001f600"}, ensure_ascii=True) -> UTF-16 surrogate pair
        assert_eq!(
            py_json_dumps_with(
                &serde_json::json!({ "a": "\u{1f600}" }),
                JsonDumpsOpts {
                    ensure_ascii: true,
                    ..Default::default()
                }
            ),
            "{\"a\": \"\\ud83d\\ude00\"}"
        );
        // json.dumps(v, indent=2, sort_keys=True)
        assert_eq!(
            py_json_dumps_with(
                &v,
                JsonDumpsOpts {
                    ensure_ascii: false,
                    indent: Some(2),
                    sort_keys: true
                }
            ),
            "{\n  \"a\": \"\u{e9}\",\n  \"b\": [\n    1,\n    2\n  ]\n}"
        );
    }

    #[test]
    fn strip_covers_python_only_whitespace() {
        assert!(is_py_space('\u{1c}'));
        assert!(!'\u{1c}'.is_whitespace());
        assert_eq!(py_strip("\u{1c}\u{1f} x \u{85}", None), "x");
        assert_eq!(py_strip("xxhixx", Some("x")), "hi");
        assert_eq!(py_lstrip("\n\n a", Some("\n")), " a");
        assert_eq!(py_rstrip("a\n\n", Some("\n")), "a");
    }

    #[test]
    fn split_matches_python() {
        assert_eq!(py_split("a,b,c", Some(","), -1).unwrap(), ["a", "b", "c"]);
        assert_eq!(py_split("a,b,c", Some(","), 1).unwrap(), ["a", "b,c"]);
        assert_eq!(py_split("a,b,c", Some(","), 0).unwrap(), ["a,b,c"]);
        assert_eq!(py_split("", Some(","), -1).unwrap(), [""]);
        assert_eq!(py_split(",", Some(","), -1).unwrap(), ["", ""]);
        assert_eq!(py_split("  a  b  ", None, -1).unwrap(), ["a", "b"]);
        assert_eq!(py_split("  a  b  ", None, 1).unwrap(), ["a", "b  "]);
        assert!(py_split("abc", Some(""), -1).is_err());
        assert_eq!(py_rsplit("a.b.c", Some("."), 1).unwrap(), ["a.b", "c"]);
    }

    #[test]
    fn find_and_count_use_code_points() {
        assert_eq!(py_find("\u{4e2d}\u{6587}x", "x", None, None), 2);
        assert_eq!(py_rfind("axbxc", "x", None, None), 3);
        assert_eq!(py_find("abc", "z", None, None), -1);
        assert_eq!(py_find("abcabc", "b", Some(2), None), 4);
        assert_eq!(py_count("aaaa", "aa", None, None), 2);
        assert_eq!(py_count("abc", "", None, None), 4);
    }

    #[test]
    fn str_and_repr_match_python() {
        assert_eq!(py_str(&serde_json::json!(true)), "True");
        assert_eq!(py_str(&serde_json::json!(null)), "None");
        assert_eq!(py_str(&serde_json::json!(2.5)), "2.5");
        assert_eq!(py_str(&serde_json::json!(5)), "5");
        assert_eq!(py_str(&serde_json::json!("hi")), "hi");
        assert_eq!(
            py_repr(&serde_json::json!([1, 2.5, true])),
            "[1, 2.5, True]"
        );
        assert_eq!(py_repr(&serde_json::json!({"a": 1})), "{'a': 1}");
        assert_eq!(py_str_repr("it's"), "\"it's\"");
        assert_eq!(py_str_repr("a\nb"), "'a\\nb'");
    }

    #[test]
    fn lower_is_per_character() {
        // CPython: 'ΑΣ'.lower() == 'ασ'; Rust's str::to_lowercase gives 'ας'.
        assert_eq!(py_lower("\u{391}\u{3a3}"), "\u{3b1}\u{3c3}");
        assert_ne!("\u{391}\u{3a3}".to_lowercase(), py_lower("\u{391}\u{3a3}"));
        assert_eq!(py_lower("\u{130}"), "i\u{307}");
        assert_eq!(py_upper("\u{df}"), "SS");
    }
}
