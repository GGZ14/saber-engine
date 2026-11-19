//! Listing and downloading a repo's files: the Hugging Face Hub over HTTPS (through `curl`, which
//! resumes, retries and follows the CDN redirects), or a local mirror directory (`file:///dir`,
//! laid out `<dir>/<owner>/<repo>/...` like the pack cache itself).
//!
//! The token, when the user has one, reaches curl on its standard input (`-H @-`): it is never on
//! a command line, in a file this code writes, or in any output.

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use super::ui::{duration, human, Res, Style, UserError};

/// One file of a repo.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteFile {
    /// The path inside the repo (`tp2/rank0.pack`).
    pub rel: String,
    pub bytes: Option<u64>,
    pub sha256: Option<String>,
}

/// Where files come from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Endpoint {
    Http(String),
    File(PathBuf),
}

impl Endpoint {
    #[must_use]
    pub fn parse(s: &str) -> Endpoint {
        match s.strip_prefix("file://") {
            Some(p) => Endpoint::File(PathBuf::from(p)),
            None => Endpoint::Http(s.trim_end_matches('/').to_string()),
        }
    }
}

/// A repo on an endpoint, with the credentials to read it.
pub struct Repo<'a> {
    pub ep: Endpoint,
    pub repo: &'a str,
    pub revision: &'a str,
    pub token: Option<String>,
    /// The command a person re-runs after fixing a problem (`saber models pull w4a8`).
    pub retry: String,
}

/// Percent-encode a repo path segment by segment (keeps `/`).
fn enc(p: &str) -> String {
    p.split('/')
        .map(|seg| {
            seg.bytes()
                .map(|b| {
                    if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
                        (b as char).to_string()
                    } else {
                        format!("%{b:02X}")
                    }
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn curl() -> Command {
    let mut c = Command::new("curl");
    c.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    c
}

/// Feed the auth header (if any) to curl's stdin and close it.
fn feed_token(child: &mut std::process::Child, token: Option<&str>) {
    if let Some(mut si) = child.stdin.take() {
        if let Some(t) = token {
            let _ = writeln!(si, "Authorization: Bearer {t}");
        }
    }
}

fn spawn_err(e: &std::io::Error) -> UserError {
    UserError::new(format!("cannot run curl ({e}): downloads need it")).fix(
        "install curl with your package manager (e.g. `dnf install curl` or `apt install curl`)",
    )
}

impl Repo<'_> {
    fn url_resolve(&self, base: &str, rel: &str) -> String {
        format!(
            "{base}/{}/resolve/{}/{}",
            self.repo,
            enc(self.revision),
            enc(rel)
        )
    }

    /// A small GET through curl -> (HTTP status, body).
    fn get(&self, url: &str) -> Res<(u16, String)> {
        let mut c = curl();
        c.args([
            "-sS",
            "-L",
            "--connect-timeout",
            "15",
            "--max-time",
            "120",
            "-w",
            "\n%{http_code}",
        ]);
        if self.token.is_some() {
            c.args(["-H", "@-"]);
        }
        c.arg(url);
        let mut ch = c.spawn().map_err(|e| spawn_err(&e))?;
        feed_token(&mut ch, self.token.as_deref());
        let o = ch.wait_with_output().map_err(|e| spawn_err(&e))?;
        let out = String::from_utf8_lossy(&o.stdout).into_owned();
        if !o.status.success() {
            return Err(self.network_error(
                o.status.code(),
                &String::from_utf8_lossy(&o.stderr),
                url,
            ));
        }
        let (body, code) = out.rsplit_once('\n').unwrap_or(("", &out));
        Ok((code.trim().parse().unwrap_or(0), body.to_string()))
    }

    /// curl failed before any HTTP answer.
    fn network_error(&self, code: Option<i32>, stderr: &str, url: &str) -> UserError {
        let msg = saber_console::downloads::curl_error(code, stderr, url);
        let offline = matches!(code, Some(5..=7 | 28 | 35 | 56 | 60));
        let mut e = UserError::new(msg);
        if offline {
            e = e.note(
                "`saber models list`, `info` and `serve` of downloaded models work offline; only a pull needs the network",
            );
        }
        e.fix(format!(
            "check the connection, then run `{}` again (it resumes)",
            self.retry
        ))
    }

    /// An HTTP status other than 200, in words, with the fix.
    fn http_error(&self, status: u16, what: &str) -> UserError {
        let page = match &self.ep {
            Endpoint::Http(b) => format!("{b}/{}", self.repo),
            Endpoint::File(p) => p.display().to_string(),
        };
        match status {
            401 | 403 if self.token.is_none() => UserError::new(format!(
                "{page} refused {what} (HTTP {status}): the repo is private or gated, and no Hugging Face token was found"
            ))
            .note("(the Hub also answers 401 for a repo that does not exist: check the name with saber models info <name>)")
            .fix("hf auth login     (or: export HF_TOKEN=<your token>)")
            .fix(format!("then run `{}` again", self.retry)),
            401 | 403 => UserError::new(format!(
                "{page} refused {what} (HTTP {status}): your Hugging Face token does not have access"
            ))
            .fix(format!("open {page} and request access, or log in with a token that has it (hf auth login)")),
            404 => {
                let mut e = UserError::new(format!("{page} has no {what} (HTTP 404)"));
                if self.token.is_none() {
                    e = e.note("a private repo also answers 404 to a request without a token");
                }
                e.fix("check the entry's `repo` and `path` (saber models info <name>), or log in: hf auth login")
            }
            429 => UserError::new(format!("{page} is rate-limiting this machine (HTTP 429)"))
                .fix(format!("wait a minute, then run `{}` again (it resumes where it stopped)", self.retry))
                .fix("a logged-in token gets a higher limit: hf auth login"),
            s if s >= 500 => UserError::new(format!("{page} is having trouble (HTTP {s})"))
                .fix(format!("try `{}` again in a few minutes (it resumes)", self.retry)),
            s => UserError::new(format!("{page} answered HTTP {s} for {what}"))
                .fix(format!("run `{}` again; if it persists, check `saber models info <name>`", self.retry)),
        }
    }

    /// The files under `dir` (recursive; `""` = the repo root, not recursive) with sizes and
    /// sha256 where the endpoint has them.
    pub fn list(&self, dir: &str) -> Res<Vec<RemoteFile>> {
        match &self.ep {
            Endpoint::Http(base) => {
                let url = if dir.is_empty() {
                    format!(
                        "{base}/api/models/{}/tree/{}",
                        self.repo,
                        enc(self.revision)
                    )
                } else {
                    format!(
                        "{base}/api/models/{}/tree/{}/{}?recursive=true",
                        self.repo,
                        enc(self.revision),
                        enc(dir)
                    )
                };
                let what = if dir.is_empty() {
                    "file listing".to_string()
                } else {
                    format!("`{dir}/` folder")
                };
                let (s, body) = self.get(&url)?;
                if s != 200 {
                    return Err(self.http_error(s, &what));
                }
                Ok(saber_console::downloads::parse_tree(&body)
                    .into_iter()
                    .map(|(rel, bytes, sha256)| RemoteFile { rel, bytes, sha256 })
                    .collect())
            }
            Endpoint::File(root) => {
                let (o, n) = self.repo.split_once('/').unwrap_or(("", self.repo));
                let base = root.join(o).join(n);
                let d = if dir.is_empty() {
                    base.clone()
                } else {
                    base.join(dir)
                };
                let rd = std::fs::read_dir(&d).map_err(|_| {
                    UserError::new(format!("the mirror has no {}", d.display()))
                        .fix("check the entry's `endpoint`, `repo` and `path` (saber models info <name>)")
                })?;
                let mut v = Vec::new();
                for f in rd.flatten() {
                    let name = f.file_name().to_string_lossy().into_owned();
                    let Ok(m) = f.metadata() else { continue };
                    if !m.is_file() || name.starts_with('.') || name.ends_with(".part") {
                        continue;
                    }
                    let rel = if dir.is_empty() {
                        name
                    } else {
                        format!("{dir}/{name}")
                    };
                    v.push(RemoteFile {
                        rel,
                        bytes: Some(m.len()),
                        sha256: None,
                    });
                }
                v.sort_by(|a, b| a.rel.cmp(&b.rel));
                Ok(v)
            }
        }
    }

    /// `SHA256SUMS` at the repo root (`<sha>  <path>` lines), when the repo has one.
    pub fn sums(&self) -> Vec<(String, String)> {
        let text = match &self.ep {
            Endpoint::Http(base) => match self.get(&self.url_resolve(base, "SHA256SUMS")) {
                Ok((200, b)) => b,
                _ => return Vec::new(),
            },
            Endpoint::File(root) => {
                let (o, n) = self.repo.split_once('/').unwrap_or(("", self.repo));
                std::fs::read_to_string(root.join(o).join(n).join("SHA256SUMS")).unwrap_or_default()
            }
        };
        parse_sums(&text)
    }

    /// Download `f` to `dest` (resuming `dest.part`), reporting to `p`.
    pub fn download(&self, f: &RemoteFile, dest: &Path, p: &mut Progress) -> Res<()> {
        let part = PathBuf::from(format!("{}.part", dest.display()));
        let want = f.bytes;
        if let (Some(w), Ok(m)) = (want, std::fs::metadata(dest)) {
            if m.len() == w {
                p.add_done(w);
                return Ok(());
            }
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                UserError::new(format!("cannot create {} ({e})", parent.display()))
                    .fix("pick a writable packs folder: --packs-dir DIR (or SABER_PACKS=DIR)")
            })?;
        }
        if let (Some(w), Ok(m)) = (want, std::fs::metadata(&part)) {
            if m.len() > w {
                let _ = std::fs::remove_file(&part);
            }
        }
        let url = match &self.ep {
            Endpoint::Http(base) => self.url_resolve(base, &f.rel),
            Endpoint::File(root) => {
                let (o, n) = self.repo.split_once('/').unwrap_or(("", self.repo));
                format!("file://{}", root.join(o).join(n).join(&f.rel).display())
            }
        };
        let name = f.rel.rsplit('/').next().unwrap_or(&f.rel).to_string();
        let have_before = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        if want == Some(have_before) {
            // a complete .part from an interrupted run: nothing left to fetch
        } else {
            let mut c = curl();
            c.stdout(Stdio::null());
            c.args([
                "-sS",
                "-L",
                "--fail",
                "--connect-timeout",
                "20",
                "--retry",
                "3",
                "--retry-delay",
                "2",
                "--speed-limit",
                "1024",
                "--speed-time",
                "120",
                "-C",
                "-",
                "-o",
            ]);
            c.arg(&part);
            if self.token.is_some() && matches!(self.ep, Endpoint::Http(_)) {
                c.args(["-H", "@-"]);
            }
            c.arg(&url);
            let mut ch = c.spawn().map_err(|e| spawn_err(&e))?;
            feed_token(&mut ch, self.token.as_deref());
            let status = loop {
                if let Some(s) = ch.try_wait().map_err(|e| spawn_err(&e))? {
                    break s;
                }
                let now = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
                p.show(now, &name);
                std::thread::sleep(Duration::from_millis(250));
            };
            let mut err = String::new();
            if let Some(mut se) = ch.stderr.take() {
                let _ = std::io::Read::read_to_string(&mut se, &mut err);
            }
            if !status.success() {
                p.clear_line();
                let code = status.code();
                if code == Some(22) {
                    let s = err
                        .split("error:")
                        .nth(1)
                        .and_then(|x| x.split_whitespace().next())
                        .and_then(|x| x.parse().ok())
                        .unwrap_or(0);
                    return Err(self.http_error(s, &format!("`{}`", f.rel)));
                }
                if code == Some(23) {
                    return Err(UserError::new(format!(
                        "could not write {}: the disk is probably full",
                        part.display()
                    ))
                    .fix("free some space (saber models rm <name>), or pull to a bigger disk: --packs-dir DIR")
                    .fix(format!("then run `{}` again (it resumes)", self.retry)));
                }
                return Err(self.network_error(code, &err, &url));
            }
        }
        let got = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        if let Some(w) = want {
            if got != w {
                p.clear_line();
                return Err(UserError::new(format!(
                    "{} arrived as {} bytes, the listing says {w}",
                    f.rel, got
                ))
                .fix(format!("run `{}` again (it resumes)", self.retry)));
            }
        }
        std::fs::rename(&part, dest).map_err(|e| {
            UserError::new(format!("cannot move {} into place ({e})", part.display()))
        })?;
        p.add_done(got);
        Ok(())
    }
}

/// `<sha>  <path>` lines -> (path, sha).
#[must_use]
pub fn parse_sums(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let sha = it.next()?;
            let path = it.next()?.trim_start_matches('*');
            (sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit()))
                .then(|| (path.to_string(), sha.to_ascii_lowercase()))
        })
        .collect()
}

/// A progress line: done / total, percent, speed, ETA. On a terminal it redraws one line; in a
/// log it prints a line every 10 %.
pub struct Progress {
    total: u64,
    done: u64,
    start: Instant,
    last: Option<(Instant, u64)>,
    rate: Option<f64>,
    tty: bool,
    logged_decile: u64,
    verb: &'static str,
    drawn: bool,
}

impl Progress {
    #[must_use]
    pub fn new(total: u64, verb: &'static str) -> Progress {
        Progress {
            total,
            done: 0,
            start: Instant::now(),
            last: None,
            rate: None,
            tty: std::io::stderr().is_terminal(),
            logged_decile: 0,
            verb,
            drawn: false,
        }
    }

    pub fn add_done(&mut self, b: u64) {
        self.done += b;
    }

    /// Bytes per second so far (for the summary).
    #[must_use]
    pub fn mean_rate(&self) -> f64 {
        self.done as f64 / self.start.elapsed().as_secs_f64().max(1e-3)
    }

    /// Redraw with `cur` bytes of the current file done.
    pub fn show(&mut self, cur: u64, name: &str) {
        let now_b = self.done + cur;
        let t = Instant::now();
        if let Some((lt, lb)) = self.last {
            let dt = t.duration_since(lt).as_secs_f64();
            if dt >= 0.5 {
                let r = now_b.saturating_sub(lb) as f64 / dt;
                self.rate = Some(self.rate.map_or(r, |old| 0.7 * old + 0.3 * r));
                self.last = Some((t, now_b));
            }
        } else {
            self.last = Some((t, now_b));
        }
        let pct = if self.total > 0 {
            (now_b as f64 / self.total as f64 * 100.0).min(100.0)
        } else {
            100.0
        };
        let speed = self
            .rate
            .map_or("--".to_string(), |r| format!("{}/s", human(r as u64)));
        let eta = self
            .rate
            .filter(|r| *r > 1.0)
            .map_or("--".to_string(), |r| {
                duration(self.total.saturating_sub(now_b) as f64 / r)
            });
        if self.tty {
            let width = 24usize;
            let fill = ((pct / 100.0) * width as f64) as usize;
            let bar: String = "#".repeat(fill) + &".".repeat(width - fill);
            eprint!(
                "\r\x1b[2K  {} [{bar}] {:>9} / {:<9} {:>3.0}%  {:>10}  ETA {:<8} {}",
                self.verb,
                human(now_b),
                human(self.total),
                pct,
                speed,
                eta,
                name
            );
            let _ = std::io::stderr().flush();
            self.drawn = true;
        } else {
            let dec = (pct / 10.0) as u64;
            if dec > self.logged_decile {
                self.logged_decile = dec;
                eprintln!(
                    "  {} {} / {} ({:.0}%), {speed}, ETA {eta}  {name}",
                    self.verb,
                    human(now_b),
                    human(self.total),
                    pct
                );
            }
        }
    }

    pub fn clear_line(&mut self) {
        if self.tty && self.drawn {
            eprint!("\r\x1b[2K");
            let _ = std::io::stderr().flush();
            self.drawn = false;
        }
    }

    /// End the line with a summary.
    pub fn finish(&mut self, what: &str) {
        self.clear_line();
        let st = Style::stderr();
        eprintln!(
            "  {} {what}: {} in {} ({}/s)",
            st.green("done"),
            human(self.done),
            duration(self.start.elapsed().as_secs_f64()),
            human(self.mean_rate() as u64)
        );
    }
}

/// The sha256 of `p`, reporting to `prog` as it reads.
pub fn sha256_with_progress(p: &Path, prog: &mut Progress) -> std::io::Result<String> {
    use sha2::Digest;
    use std::io::Read;
    let name = p
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut f = std::fs::File::open(p)?;
    let mut h = sha2::Sha256::new();
    let mut buf = vec![0u8; 4 << 20];
    let mut n_done = 0u64;
    let mut last = Instant::now();
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        n_done += n as u64;
        if last.elapsed() >= Duration::from_millis(250) {
            prog.show(n_done, &name);
            last = Instant::now();
        }
    }
    prog.add_done(n_done);
    Ok(hex::encode(h.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_parse() {
        let s = parse_sums(
            "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a  LICENSE\n\
             86F4BAD3D09A17CCF70B252F3F51B7E977A39B1EC2B39796F2F9917B79071E95 *tp2/rank0.pack\n\
             junk line\n",
        );
        assert_eq!(s.len(), 2);
        assert_eq!(s[1].0, "tp2/rank0.pack");
        assert!(s[1].1.starts_with("86f4bad3"));
    }

    #[test]
    fn endpoints_parse() {
        assert_eq!(
            Endpoint::parse("https://huggingface.co/"),
            Endpoint::Http("https://huggingface.co".into())
        );
        assert_eq!(Endpoint::parse("file:///m"), Endpoint::File("/m".into()));
    }

    #[test]
    fn http_errors_say_what_to_do() {
        let r = Repo {
            ep: Endpoint::parse("https://huggingface.co"),
            repo: "o/r",
            revision: "main",
            token: None,
            retry: "saber models pull x".into(),
        };
        let t = r.http_error(401, "file listing").render(false);
        assert!(
            t.contains("no Hugging Face token") && t.contains("hf auth login"),
            "{t}"
        );
        let t = r.http_error(429, "x").render(false);
        assert!(
            t.contains("rate-limiting") && t.contains("saber models pull x"),
            "{t}"
        );
        let t = r.http_error(404, "`tp2/` folder").render(false);
        assert!(t.contains("HTTP 404") && t.contains("private repo"), "{t}");
        let t = r.http_error(503, "x").render(false);
        assert!(t.contains("having trouble"), "{t}");
    }

    #[test]
    fn a_mirror_lists_and_downloads() {
        let s = saber_test_scratch::ScratchDir::new("models-mirror");
        let src = s.join("mirror/o/r/tp1");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("rank0.pack"), b"SBRPACK1abc").unwrap();
        std::fs::write(src.join(".hidden"), b"x").unwrap();
        let r = Repo {
            ep: Endpoint::File(s.join("mirror")),
            repo: "o/r",
            revision: "main",
            token: None,
            retry: "x".into(),
        };
        let l = r.list("tp1").unwrap();
        assert_eq!(
            l,
            vec![RemoteFile {
                rel: "tp1/rank0.pack".into(),
                bytes: Some(11),
                sha256: None
            }]
        );
        let dest = s.join("out/rank0.pack");
        let mut p = Progress::new(11, "downloading");
        r.download(&l[0], &dest, &mut p).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"SBRPACK1abc");
        // idempotent: a second call finds it complete
        r.download(&l[0], &dest, &mut p).unwrap();
        assert!(r.list("nope").is_err());
    }
}
