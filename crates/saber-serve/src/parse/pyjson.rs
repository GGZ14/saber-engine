//! A JSON value and encoder that reproduce CPython's `json` module byte for byte.
//!
//! Tool-call `arguments` are compared against the reference server's output, which is
//! `json.dumps(obj, ensure_ascii=False)` over a Python dict. Three details of that
//! output are not what `serde_json::to_string` produces, and every one of them shows
//! up in the parity corpus:
//!
//! * **separators** — Python's default is `", "` and `": "`, not `","` and `":"`.
//! * **integers are exact** — `int("9999999999999999999999")` keeps all its digits,
//!   where an `f64` round-trip would print `1e+22`. Integer literals are therefore
//!   carried as their digit string.
//! * **float repr** — Python switches to exponent form outside `[1e-4, 1e16)` and
//!   writes the exponent with a sign and at least two digits (`1e-05`), where Rust's
//!   `Display` never uses exponent form and `LowerExp` writes `1e-5`.
//!
//! `serde_json` is still used for *schemas* (read-only), never for values that are
//! serialised back into an `arguments` string.

use std::fmt::Write as _;

/// A parsed JSON value with Python's fidelity: exact integers, and floats that
/// remember nothing but their `f64` value (which is all CPython keeps either).
#[derive(Debug, Clone, PartialEq)]
pub enum PyVal {
    Null,
    Bool(bool),
    /// An integer literal, kept as digits so arbitrary precision survives.
    Int(String),
    Float(f64),
    Str(String),
    Arr(Vec<PyVal>),
    /// Object entries in insertion order — Python dicts preserve it and
    /// `json.dumps` emits it.
    Obj(Vec<(String, PyVal)>),
}

impl PyVal {
    pub fn int(v: i64) -> Self {
        PyVal::Int(v.to_string())
    }

    pub fn is_str(&self) -> bool {
        matches!(self, PyVal::Str(_))
    }

    /// `json.dumps(self, ensure_ascii=False)`.
    pub fn dumps(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    /// Whether `json.dumps(obj, allow_nan=False)` would succeed — i.e. no
    /// non-finite float anywhere inside. The reference server's `_is_json_finite`.
    pub fn is_json_finite(&self) -> bool {
        match self {
            PyVal::Float(f) => f.is_finite(),
            PyVal::Arr(items) => items.iter().all(PyVal::is_json_finite),
            PyVal::Obj(entries) => entries.iter().all(|(_, v)| v.is_json_finite()),
            _ => true,
        }
    }

    fn write(&self, out: &mut String) {
        match self {
            PyVal::Null => out.push_str("null"),
            PyVal::Bool(true) => out.push_str("true"),
            PyVal::Bool(false) => out.push_str("false"),
            PyVal::Int(digits) => out.push_str(digits),
            PyVal::Float(f) => out.push_str(&py_repr_f64(*f)),
            PyVal::Str(s) => encode_string(s, out),
            PyVal::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    item.write(out);
                }
                out.push(']');
            }
            PyVal::Obj(entries) => {
                out.push('{');
                for (i, (key, value)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    encode_string(key, out);
                    out.push_str(": ");
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

/// An insertion-ordered string map with Python `dict` update semantics: assigning to
/// an existing key overwrites the value and keeps the key's original position.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PyDict {
    entries: Vec<(String, PyVal)>,
}

impl PyDict {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: &str, value: PyVal) {
        if let Some(slot) = self.entries.iter_mut().find(|(k, _)| k == key) {
            slot.1 = value;
        } else {
            self.entries.push((key.to_owned(), value));
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn into_val(self) -> PyVal {
        PyVal::Obj(self.entries)
    }
}

/// `json.dumps` string escaping with `ensure_ascii=False`: quote, backslash and the
/// C0 controls only — everything else is written as-is.
fn encode_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// CPython's `repr(float)`, which is what `json.dumps` writes for a float.
///
/// Shortest round-trip digits (same as Rust's), then Python's presentation rule:
/// fixed notation while the decimal exponent is in `[-4, 16)`, exponent notation
/// outside it with an explicit sign and at least two exponent digits.
pub fn py_repr_f64(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_owned();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity" } else { "-Infinity" }.to_owned();
    }
    // `{:e}` gives the shortest round-trip digits as `[-]d[.ddd]e[-]X`.
    let sci = format!("{v:e}");
    let (mantissa, exp) = sci.split_once('e').expect("LowerExp always emits 'e'");
    let exp: i32 = exp.parse().expect("LowerExp emits a valid exponent");
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };

    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if v == 0.0 {
        out.push_str("0.0");
        return out;
    }
    if (-4..16).contains(&exp) {
        if exp < 0 {
            out.push_str("0.");
            for _ in 0..(-exp - 1) {
                out.push('0');
            }
            out.push_str(digits);
        } else {
            let point = exp as usize + 1;
            if digits.len() > point {
                out.push_str(&digits[..point]);
                out.push('.');
                out.push_str(&digits[point..]);
            } else {
                out.push_str(digits);
                for _ in 0..(point - digits.len()) {
                    out.push('0');
                }
                out.push_str(".0");
            }
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    out
}

/// `json.loads`. Returns `None` on anything CPython would raise `JSONDecodeError`
/// for, with one deliberate difference: an overflowing literal such as `1e999`
/// parses to an infinity here exactly as it does in CPython, and callers reject it
/// through [`PyVal::is_json_finite`].
pub fn loads(text: &str) -> Option<PyVal> {
    loads_err(text).ok()
}

/// CPython's `json.JSONDecodeError`, formatted the way `str(exc)` formats it — which is
/// exactly what reaches an OpenAI client, because `create_error_response` puts `str(exc)`
/// in the body of the 400 (`ValueError` → `BadRequestError`, `param: null`).
///
/// `pos` is a CHARACTER index, as CPython's is: the decoder indexes a `str`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyJsonError {
    /// `e.msg`, e.g. `Expecting value`.
    pub msg: String,
    /// `e.pos`, in characters.
    pub pos: usize,
    /// `e.lineno`, 1-based.
    pub lineno: usize,
    /// `e.colno`, 1-based.
    pub colno: usize,
}

impl std::fmt::Display for PyJsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: line {} column {} (char {})",
            self.msg, self.lineno, self.colno, self.pos
        )
    }
}

/// `json.loads`, with CPython's own failure message.
///
/// The messages and positions are the ones in `the internal reference harness`'s table, taken from
/// CPython 3.12 inside the prod image and replayed by `the_decoder_reports_cpythons_message`.
///
/// # Errors
/// Whatever CPython's decoder would raise, in its own words.
pub fn loads_err(text: &str) -> Result<PyVal, PyJsonError> {
    let chars: Vec<char> = text.chars().collect();
    let mut p = Parser {
        s: &chars,
        i: 0,
        fail: None,
    };
    p.skip_ws();
    let value = match p.value() {
        Some(v) => v,
        None => {
            let (msg, pos) = p
                .fail
                .clone()
                .unwrap_or_else(|| ("Expecting value".to_owned(), p.i));
            return Err(error_at(&chars, &msg, pos));
        }
    };
    p.skip_ws();
    if p.i != p.s.len() {
        // `json.decoder.JSONDecoder.decode` skips whitespace and raises at what is left.
        return Err(error_at(&chars, "Extra data", p.i));
    }
    Ok(value)
}

/// CPython's `JSONDecodeError.__init__`: `lineno = doc.count('\n', 0, pos) + 1` and
/// `colno = pos - doc.rfind('\n', 0, pos)`.
fn error_at(s: &[char], msg: &str, pos: usize) -> PyJsonError {
    let upto = &s[..pos.min(s.len())];
    let lineno = upto.iter().filter(|c| **c == '\n').count() + 1;
    let last_nl = upto.iter().rposition(|c| *c == '\n');
    let colno = match last_nl {
        // `rfind` returns the index of the newline; CPython's `pos - that` is 1-based already.
        Some(i) => pos - i,
        // and -1 when there is none, which makes `pos - (-1)` = `pos + 1`.
        None => pos + 1,
    };
    PyJsonError {
        msg: msg.to_owned(),
        pos,
        lineno,
        colno,
    }
}

struct Parser<'a> {
    s: &'a [char],
    i: usize,
    /// The FIRST failure recorded, which is the deepest one: CPython raises from where the
    /// scanner gave up, and an outer frame that also gives up must not overwrite it.
    fail: Option<(String, usize)>,
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(
            self.peek(),
            Some(' ') | Some('\t') | Some('\n') | Some('\r')
        ) {
            self.i += 1;
        }
    }

    fn lit(&mut self, word: &str, value: PyVal) -> Option<PyVal> {
        if self.s[self.i..]
            .iter()
            .collect::<String>()
            .starts_with(word)
        {
            self.i += word.chars().count();
            Some(value)
        } else {
            None
        }
    }

    /// Record the FIRST failure and give up. Every `return None` in this parser goes through
    /// here, so `loads_err` can say what CPython would have said and where.
    fn bail<T>(&mut self, msg: &str, pos: usize) -> Option<T> {
        if self.fail.is_none() {
            self.fail = Some((msg.to_owned(), pos));
        }
        None
    }

    fn value(&mut self) -> Option<PyVal> {
        let start = self.i;
        let v = self.value_inner();
        if v.is_none() {
            // The scanner's own message, and only when nothing deeper already spoke: CPython's
            // `_scan_once` raises `StopIteration(idx)` at the position the value should start at,
            // which `decode`/`JSONArray`/`JSONObject` turn into "Expecting value" THERE.
            return self.bail("Expecting value", start);
        }
        v
    }

    fn value_inner(&mut self) -> Option<PyVal> {
        match self.peek()? {
            '{' => self.object(),
            '[' => self.array(),
            '"' => self.string().map(PyVal::Str),
            't' => self.lit("true", PyVal::Bool(true)),
            'f' => self.lit("false", PyVal::Bool(false)),
            'n' => self.lit("null", PyVal::Null),
            // CPython's decoder also accepts these three, and models emit them.
            'N' => self.lit("NaN", PyVal::Float(f64::NAN)),
            'I' => self.lit("Infinity", PyVal::Float(f64::INFINITY)),
            _ => self.number(),
        }
    }

    fn object(&mut self) -> Option<PyVal> {
        self.i += 1; // '{'
        let mut entries: Vec<(String, PyVal)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.i += 1;
            return Some(PyVal::Obj(entries));
        }
        loop {
            self.skip_ws();
            // CPython checks for the quote itself so it can name the key, rather than reporting
            // "Expecting value" for `{a: 1}`.
            if self.peek() != Some('"') {
                return self.bail("Expecting property name enclosed in double quotes", self.i);
            }
            let key = self.string()?;
            self.skip_ws();
            if self.peek() != Some(':') {
                return self.bail("Expecting ':' delimiter", self.i);
            }
            self.i += 1;
            self.skip_ws();
            let value = self.value()?;
            // Later duplicate keys win but keep the first position — CPython's dict.
            if let Some(slot) = entries.iter_mut().find(|(k, _)| *k == key) {
                slot.1 = value;
            } else {
                entries.push((key, value));
            }
            self.skip_ws();
            match self.peek() {
                Some(',') => self.i += 1,
                Some('}') => {
                    self.i += 1;
                    return Some(PyVal::Obj(entries));
                }
                _ => return self.bail("Expecting ',' delimiter", self.i),
            }
        }
    }

    fn array(&mut self) -> Option<PyVal> {
        self.i += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.i += 1;
            return Some(PyVal::Arr(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value()?);
            self.skip_ws();
            match self.peek() {
                Some(',') => self.i += 1,
                Some(']') => {
                    self.i += 1;
                    return Some(PyVal::Arr(items));
                }
                _ => return self.bail("Expecting ',' delimiter", self.i),
            }
        }
    }

    fn string(&mut self) -> Option<String> {
        let open = self.i;
        if self.peek() != Some('"') {
            return self.bail("Expecting value", self.i);
        }
        self.i += 1;
        let mut out = String::new();
        loop {
            let Some(ch) = self.peek() else {
                return self.bail("Unterminated string starting at", open);
            };
            self.i += 1;
            match ch {
                '"' => return Some(out),
                '\\' => {
                    let back = self.i - 1;
                    let Some(esc) = self.peek() else {
                        return self.bail("Unterminated string starting at", open);
                    };
                    self.i += 1;
                    match esc {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'b' => out.push('\u{8}'),
                        'f' => out.push('\u{c}'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => {
                            let cp = self.hex4(back + 1)?;
                            // Surrogate pair, as CPython's decoder joins them.
                            if (0xD800..0xDC00).contains(&cp)
                                && self.peek() == Some('\\')
                                && self.s.get(self.i + 1).copied() == Some('u')
                            {
                                let save = self.i;
                                self.i += 2;
                                let low = self.hex4(save + 1)?;
                                if (0xDC00..0xE000).contains(&low) {
                                    let joined = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                                    out.push(char::from_u32(joined)?);
                                } else {
                                    self.i = save;
                                    out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                                }
                            } else {
                                out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                            }
                        }
                        _ => return self.bail("Invalid \\escape", back),
                    }
                }
                c if (c as u32) < 0x20 => {
                    return self.bail("Invalid control character at", self.i - 1)
                }
                c => out.push(c),
            }
        }
    }

    /// Four hex digits. `at` is where CPython reports a bad one from: the index of the `u`.
    fn hex4(&mut self, at: usize) -> Option<u32> {
        let mut value = 0u32;
        for _ in 0..4 {
            let Some(c) = self.peek() else {
                return self.bail("Invalid \\uXXXX escape", at);
            };
            self.i += 1;
            let Some(d) = c.to_digit(16) else {
                return self.bail("Invalid \\uXXXX escape", at);
            };
            value = value * 16 + d;
        }
        Some(value)
    }

    /// CPython's `NUMBER_RE`: `(-?(?:0|[1-9]\d*))(\.\d+)?([eE][-+]?\d+)?`.
    ///
    /// It is a REGEX, so it matches the longest prefix that fits and leaves the rest to the
    /// caller — which is why `json.loads("1.")` and `json.loads("01")` are both `Extra data`
    /// rather than `Expecting value`. A parser that consumed greedily and then failed would
    /// report the wrong message and the wrong position for both.
    fn number(&mut self) -> Option<PyVal> {
        let start = self.i;
        if self.peek() == Some('-') {
            self.i += 1;
            if self.peek() == Some('I') {
                return self.lit("Infinity", PyVal::Float(f64::NEG_INFINITY));
            }
        }
        let mut int_digits = 0;
        match self.peek() {
            // `0` on its own: a leading zero does not start a longer integer.
            Some('0') => {
                self.i += 1;
                int_digits = 1;
            }
            Some(c) if c.is_ascii_digit() => {
                while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    self.i += 1;
                    int_digits += 1;
                }
            }
            _ => {}
        }
        if int_digits == 0 {
            self.i = start;
            return self.bail("Expecting value", start);
        }
        let mut is_float = false;
        if self.peek() == Some('.') {
            let dot = self.i;
            self.i += 1;
            let mut frac = 0;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.i += 1;
                frac += 1;
            }
            if frac == 0 {
                // The optional group did not match; the integer still did.
                self.i = dot;
            } else {
                is_float = true;
            }
        }
        if matches!(self.peek(), Some('e') | Some('E')) {
            let e = self.i;
            self.i += 1;
            if matches!(self.peek(), Some('+') | Some('-')) {
                self.i += 1;
            }
            let mut exp = 0;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.i += 1;
                exp += 1;
            }
            if exp == 0 {
                // Same again: the exponent group is optional, so `1e` is the integer `1` with
                // `e` left over, not a malformed number.
                self.i = e;
            } else {
                is_float = true;
            }
        }
        let text: String = self.s[start..self.i].iter().collect();
        if is_float {
            Some(PyVal::Float(text.parse::<f64>().ok()?))
        } else {
            Some(PyVal::Int(normalize_int(&text)))
        }
    }
}

/// `int("-007")` is `-7`; the digit string has to be normalised or the dump differs.
fn normalize_int(text: &str) -> String {
    let (sign, digits) = match text.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", text),
    };
    let trimmed = digits.trim_start_matches('0');
    if trimmed.is_empty() {
        "0".to_owned()
    } else {
        format!("{sign}{trimmed}")
    }
}

/// `int(value)` for a raw model string: CPython accepts surrounding whitespace and a
/// leading sign, and nothing else (no underscores in this position, no `0x`).
pub fn parse_python_int(value: &str) -> Option<String> {
    let text = value.trim();
    let (sign, digits) = match text.strip_prefix(['-', '+']) {
        Some(rest) => (if text.starts_with('-') { "-" } else { "" }, rest),
        None => ("", text),
    };
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(normalize_int(&format!("{sign}{digits}")))
}

/// `float(value)`; rejects the forms CPython's `float()` rejects that Rust accepts.
pub fn parse_python_float(value: &str) -> Option<f64> {
    let text = value.trim();
    if text.is_empty() {
        return None;
    }
    // Rust parses "inf"/"NaN"; CPython does too, so only the exotic Rust-only forms
    // need rejecting. Rust does not accept "1_0" or hex floats, so a plain parse is
    // equivalent here.
    text.parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **CPython's own failure messages**, as `str(json.JSONDecodeError)` renders them — which is
    /// the string prod puts in the body of the 400 for a tool call whose `arguments` are not JSON
    /// (ws-q1's row 205). The table was taken from CPython 3.12 inside the prod image with
    /// `the internal reference harness`; regenerate it there, do not hand-edit it.
    ///
    /// Three of these look like typos and are the point: `json.loads("01")` and `json.loads("1.")`
    /// are **Extra data**, not `Expecting value`, because `NUMBER_RE` is a regex that matches the
    /// longest legal prefix and leaves the rest to the caller. A parser that consumed greedily and
    /// then failed reports the wrong message at the wrong position for both.
    #[test]
    fn the_decoder_reports_cpythons_message() {
        let cases: &[(&str, &str)] = &[
            ("not even json", "Expecting value: line 1 column 1 (char 0)"),
            ("", "Expecting value: line 1 column 1 (char 0)"),
            ("   ", "Expecting value: line 1 column 4 (char 3)"),
            (
                "{",
                "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
            ),
            ("}", "Expecting value: line 1 column 1 (char 0)"),
            ("[", "Expecting value: line 1 column 2 (char 1)"),
            ("]", "Expecting value: line 1 column 1 (char 0)"),
            ("[1,", "Expecting value: line 1 column 4 (char 3)"),
            ("[1 2]", "Expecting ',' delimiter: line 1 column 4 (char 3)"),
            (
                "{\"a\"}",
                "Expecting ':' delimiter: line 1 column 5 (char 4)",
            ),
            ("{\"a\": }", "Expecting value: line 1 column 7 (char 6)"),
            (
                "{\"a\" 1}",
                "Expecting ':' delimiter: line 1 column 6 (char 5)",
            ),
            (
                "{\"a\": 1,}",
                "Expecting property name enclosed in double quotes: line 1 column 9 (char 8)",
            ),
            (
                "{a: 1}",
                "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
            ),
            (
                "{'a': 1}",
                "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
            ),
            (
                "{\"a\": 1} trailing",
                "Extra data: line 1 column 10 (char 9)",
            ),
            ("{\"a\": 1}{}", "Extra data: line 1 column 9 (char 8)"),
            (
                "\"unterminated",
                "Unterminated string starting at: line 1 column 1 (char 0)",
            ),
            (
                "\"bad \\q escape\"",
                "Invalid \\escape: line 1 column 6 (char 5)",
            ),
            (
                "\"ctl \u{1} char\"",
                "Invalid control character at: line 1 column 6 (char 5)",
            ),
            (
                "\"\\u12\"",
                "Invalid \\uXXXX escape: line 1 column 3 (char 2)",
            ),
            ("tru", "Expecting value: line 1 column 1 (char 0)"),
            ("nul", "Expecting value: line 1 column 1 (char 0)"),
            ("01", "Extra data: line 1 column 2 (char 1)"),
            ("1.", "Extra data: line 1 column 2 (char 1)"),
            ("1e", "Extra data: line 1 column 2 (char 1)"),
            ("-", "Expecting value: line 1 column 1 (char 0)"),
            ("\n\n  oops", "Expecting value: line 3 column 3 (char 4)"),
            (
                "{\n  \"a\": 1,\n  \"b\" 2\n}",
                "Expecting ':' delimiter: line 3 column 7 (char 18)",
            ),
            ("[1, 2,, 3]", "Expecting value: line 1 column 7 (char 6)"),
            ("{\"a\": tru}", "Expecting value: line 1 column 7 (char 6)"),
        ];
        for (text, want) in cases {
            let got = loads_err(text).expect_err(text).to_string();
            assert_eq!(&got, want, "on {text:?}");
        }
        // The other half of the same table: what CPython ACCEPTS, including the three literals
        // the JSON spec does not have. A decoder that refused these would 400 requests prod
        // serves, which is the same parity failure in the other direction.
        for ok in [
            "NaN",
            "Infinity",
            "-Infinity",
            "{\"a\": 1, \"b\": [2, {\"c\": null}]}",
            "  {\"a\": 1}  ",
            "{}",
            "\"\"",
        ] {
            assert!(loads_err(ok).is_ok(), "CPython accepts {ok:?}");
        }
    }

    #[test]
    fn python_float_repr_matches_cpython() {
        // Values checked against CPython 3.12 `repr`.
        assert_eq!(py_repr_f64(0.5), "0.5");
        assert_eq!(py_repr_f64(1.0), "1.0");
        assert_eq!(py_repr_f64(-0.0), "-0.0");
        assert_eq!(py_repr_f64(100.0), "100.0");
        assert_eq!(py_repr_f64(0.0001), "0.0001");
        assert_eq!(py_repr_f64(0.00001), "1e-05");
        assert_eq!(py_repr_f64(1e16), "1e+16");
        assert_eq!(py_repr_f64(1e15), "1000000000000000.0");
        assert_eq!(py_repr_f64(1.5e-7), "1.5e-07");
        assert_eq!(py_repr_f64(1.25e22), "1.25e+22");
        assert_eq!(py_repr_f64(0.1 + 0.2), "0.30000000000000004");
    }

    #[test]
    fn dumps_uses_python_separators_and_escapes() {
        let mut d = PyDict::new();
        d.insert("a", PyVal::int(1));
        d.insert("b", PyVal::Str("x\"y\\z\n\u{1}é".to_owned()));
        d.insert("c", PyVal::Arr(vec![PyVal::Bool(true), PyVal::Null]));
        assert_eq!(
            d.into_val().dumps(),
            "{\"a\": 1, \"b\": \"x\\\"y\\\\z\\n\\u0001é\", \"c\": [true, null]}"
        );
    }

    #[test]
    fn big_integers_keep_every_digit() {
        let v = loads("{\"n\": 9999999999999999999999}").unwrap();
        assert_eq!(v.dumps(), "{\"n\": 9999999999999999999999}");
    }

    #[test]
    fn duplicate_keys_keep_first_position_last_value() {
        let v = loads("{\"a\": 1, \"b\": 2, \"a\": 3}").unwrap();
        assert_eq!(v.dumps(), "{\"a\": 3, \"b\": 2}");
    }

    #[test]
    fn rejects_trailing_data_like_cpython() {
        assert!(loads("{} {}").is_none());
        assert!(loads("{broken").is_none());
        assert!(loads("").is_none());
    }
}
