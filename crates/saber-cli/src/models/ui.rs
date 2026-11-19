//! Terminal output for `saber models` / `saber serve --model`: errors that say what is wrong and
//! the exact command that fixes it, sizes, durations, prompts. Never a stack trace.

use std::io::{IsTerminal, Write};

/// An error meant for a person: what went wrong, optional notes, and the commands that fix it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UserError {
    pub what: String,
    pub notes: Vec<String>,
    pub fixes: Vec<String>,
    /// The process exit code (2 = usage / registry, 1 = everything else).
    pub code: i32,
}

impl UserError {
    pub fn new(what: impl Into<String>) -> UserError {
        UserError {
            what: what.into(),
            notes: Vec::new(),
            fixes: Vec::new(),
            code: 1,
        }
    }

    #[must_use]
    pub fn fix(mut self, f: impl Into<String>) -> UserError {
        self.fixes.push(f.into());
        self
    }

    #[must_use]
    pub fn note(mut self, n: impl Into<String>) -> UserError {
        self.notes.push(n.into());
        self
    }

    #[must_use]
    pub fn usage(mut self) -> UserError {
        self.code = 2;
        self
    }

    /// The message as printed (`color` = ANSI styling).
    #[must_use]
    pub fn render(&self, color: bool) -> String {
        let st = Style(color);
        let mut s = format!("{} {}\n", st.red("error:"), self.what);
        for n in &self.notes {
            s.push_str(&format!("  {}\n", n));
        }
        for f in &self.fixes {
            s.push_str(&format!("  {} {}\n", st.cyan("fix:"), f));
        }
        s
    }
}

impl std::fmt::Display for UserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.render(false))
    }
}

pub type Res<T> = Result<T, UserError>;

/// ANSI styling, off when the stream is not a terminal or `NO_COLOR` is set.
#[derive(Clone, Copy)]
pub struct Style(pub bool);

impl Style {
    #[must_use]
    pub fn stderr() -> Style {
        Style(std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none())
    }
    #[must_use]
    pub fn stdout() -> Style {
        Style(std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none())
    }
    fn wrap(self, code: &str, s: &str) -> String {
        if self.0 {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    #[must_use]
    pub fn red(self, s: &str) -> String {
        self.wrap("1;31", s)
    }
    #[must_use]
    pub fn cyan(self, s: &str) -> String {
        self.wrap("36", s)
    }
    #[must_use]
    pub fn green(self, s: &str) -> String {
        self.wrap("32", s)
    }
    #[must_use]
    pub fn yellow(self, s: &str) -> String {
        self.wrap("33", s)
    }
    #[must_use]
    pub fn bold(self, s: &str) -> String {
        self.wrap("1", s)
    }
    #[must_use]
    pub fn dim(self, s: &str) -> String {
        self.wrap("2", s)
    }
}

/// `20.5 GB` (decimal units, as the Hub shows sizes).
#[must_use]
pub fn human(b: u64) -> String {
    saber_console::downloads::human(b)
}

/// `1:02:03` / `4:05` / `12 s`.
#[must_use]
pub fn duration(secs: f64) -> String {
    if !secs.is_finite() || secs < 0.0 {
        return "--".into();
    }
    let s = secs.round() as u64;
    if s < 60 {
        format!("{s} s")
    } else if s < 3600 {
        format!("{}:{:02}", s / 60, s % 60)
    } else {
        format!("{}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
    }
}

/// Ask a yes/no question on the terminal. `default_yes` is what Enter means. Without a terminal
/// (a script), the answer is an error telling the caller to pass `--yes`.
pub fn confirm(question: &str, default_yes: bool, flag_hint: &str) -> Res<bool> {
    if !std::io::stdin().is_terminal() {
        return Err(
            UserError::new(format!("{question} -- there is no terminal to ask on"))
                .fix(format!("{flag_hint} --yes   (answer yes without asking)")),
        );
    }
    let hint = if default_yes { "[Y/n]" } else { "[y/N]" };
    eprint!("{question} {hint} ");
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return Ok(false);
    }
    let a = line.trim().to_ascii_lowercase();
    Ok(match a.as_str() {
        "" => default_yes,
        "y" | "yes" => true,
        _ => false,
    })
}

/// `~/x` for a path under the home directory (display only).
#[must_use]
pub fn tilde(p: &std::path::Path) -> String {
    if let Some(h) = std::env::var_os("HOME") {
        if let Ok(rest) = p.strip_prefix(&h) {
            return format!("~/{}", rest.display());
        }
    }
    p.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_render_what_and_fix() {
        let e = UserError::new("the pack is missing")
            .note("looked in /x")
            .fix("saber models pull w4a8");
        assert_eq!(
            e.render(false),
            "error: the pack is missing\n  looked in /x\n  fix: saber models pull w4a8\n"
        );
        assert!(e.render(true).contains("\x1b[1;31merror:\x1b[0m"));
    }

    #[test]
    fn durations() {
        assert_eq!(duration(5.2), "5 s");
        assert_eq!(duration(125.0), "2:05");
        assert_eq!(duration(3725.0), "1:02:05");
        assert_eq!(duration(f64::NAN), "--");
    }
}
