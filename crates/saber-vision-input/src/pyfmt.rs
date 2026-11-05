//! The two Python renderings prod's texts embed: `repr(str)` and `repr(float)`.

/// `repr(s)` for a Python `str`: single quotes unless the text holds `'` and no `"`; `\\`, the
/// quote, `\n` `\r` `\t` and other control characters escaped.
pub fn str_repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

/// `repr(x)` for a finite Python float in the range prod's texts use (shortest round-trip, a
/// trailing `.0` on integers).
pub fn float_repr(x: f64) -> String {
    let s = format!("{x}");
    if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") {
        s
    } else {
        format!("{s}.0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reprs() {
        assert_eq!(str_repr("data:image/png;base64"), "'data:image/png;base64'");
        assert_eq!(str_repr("it's"), "\"it's\"");
        assert_eq!(float_repr(201.0), "201.0");
        assert_eq!(float_repr(250.5), "250.5");
    }
}
