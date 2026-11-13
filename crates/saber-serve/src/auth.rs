//! Optional Bearer API keys. OFF by default: a server started without
//! `--api-key` / `--api-key-file` serves every route exactly as before.
//!
//! When keys are configured, every route needs `Authorization: Bearer <key>` except `/health`
//! and `/metrics` (a liveness probe and a scraper stay open) -- unless `--api-auth-all` closes
//! those too. Each key carries a LABEL; the matched label rides the request (an
//! [`axum::Extension<KeyLabel>`]) into the recent-requests ring. A refusal is the OpenAI
//! envelope with 401. The comparison is constant-time in the key's bytes.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

/// The routes that stay open while auth is on (unless `--api-auth-all`).
pub const OPEN_ROUTES: &[&str] = &["/health", "/metrics"];

/// The key label reserved for the console's `/v1` proxy: with keys on, the
/// engine keeps a request's `x-saber-console-ref` ([`crate::requests::CLIENT_REF_HEADER`]) only
/// when the request presented the key with this label.
pub const CONSOLE_KEY_LABEL: &str = "console";

/// The console ref a request keeps: every ref with auth off (the loopback console is the only
/// client that sends one), else only the console key's.
#[must_use]
pub fn keep_client_ref(
    keys: Option<&ApiKeys>,
    label: Option<&str>,
    client_ref: Option<String>,
) -> Option<String> {
    match keys {
        None => client_ref,
        Some(_) => client_ref.filter(|_| label == Some(CONSOLE_KEY_LABEL)),
    }
}

/// The label of the key a request presented.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyLabel(pub String);

/// The configured keys.
#[derive(Clone, PartialEq, Eq)]
pub struct ApiKeys {
    keys: Vec<(String, String)>,
    /// `--api-auth-all`: `/health` and `/metrics` need a key too.
    pub protect_open_routes: bool,
}

impl std::fmt::Debug for ApiKeys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // never print a key
        f.debug_struct("ApiKeys")
            .field(
                "labels",
                &self
                    .keys
                    .iter()
                    .map(|(l, _)| l.as_str())
                    .collect::<Vec<_>>(),
            )
            .field("protect_open_routes", &self.protect_open_routes)
            .finish()
    }
}

impl ApiKeys {
    /// Keys from `LABEL=KEY` specs (`--api-key`, repeatable) and the lines of a key file
    /// (`--api-key-file`: `LABEL=KEY` or `LABEL KEY` per line, `#` comments, blanks ignored).
    /// `None` when no key was given (auth off). A key shorter than 8 bytes, an empty label or a
    /// duplicate key is refused.
    pub fn from_specs<'a>(
        specs: impl IntoIterator<Item = &'a str>,
        file_text: Option<&str>,
        protect_open_routes: bool,
    ) -> Result<Option<Self>, String> {
        let mut keys: Vec<(String, String)> = Vec::new();
        let mut add = |line: &str, whence: &str| -> Result<(), String> {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return Ok(());
            }
            let (label, key) = line
                .split_once('=')
                .or_else(|| line.split_once(char::is_whitespace))
                .ok_or_else(|| format!("{whence}: expected LABEL=KEY"))?;
            let (label, key) = (label.trim(), key.trim());
            if label.is_empty() {
                return Err(format!("{whence}: empty label"));
            }
            if key.len() < 8 {
                return Err(format!(
                    "{whence}: the key for `{label}` is shorter than 8 bytes"
                ));
            }
            if keys.iter().any(|(_, k)| k == key) {
                return Err(format!("{whence}: the key for `{label}` is a duplicate"));
            }
            keys.push((label.to_owned(), key.to_owned()));
            Ok(())
        };
        for s in specs {
            add(s, "--api-key")?;
        }
        if let Some(text) = file_text {
            for (i, line) in text.lines().enumerate() {
                add(line, &format!("--api-key-file line {}", i + 1))?;
            }
        }
        if keys.is_empty() {
            if protect_open_routes {
                return Err("--api-auth-all needs --api-key or --api-key-file".into());
            }
            return Ok(None);
        }
        Ok(Some(Self {
            keys,
            protect_open_routes,
        }))
    }

    #[must_use]
    pub fn labels(&self) -> Vec<&str> {
        self.keys.iter().map(|(l, _)| l.as_str()).collect()
    }

    /// The label of `presented`, checking every key in constant time per key.
    #[must_use]
    pub fn check(&self, presented: &str) -> Option<&str> {
        let mut hit = None;
        for (label, key) in &self.keys {
            if ct_eq(key.as_bytes(), presented.as_bytes()) && hit.is_none() {
                hit = Some(label.as_str());
            }
        }
        hit
    }

    fn route_is_open(&self, path: &str) -> bool {
        !self.protect_open_routes && OPEN_ROUTES.contains(&path)
    }
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn unauthorized(message: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "Unauthorized",
            "param": null,
            "code": 401,
        }
    });
    (
        axum::http::StatusCode::UNAUTHORIZED,
        [(axum::http::header::WWW_AUTHENTICATE, "Bearer")],
        axum::Json(body),
    )
        .into_response()
}

/// The middleware (`axum::middleware::from_fn_with_state`), installed only when keys exist.
pub async fn require_key(
    State(keys): State<Arc<ApiKeys>>,
    mut req: Request,
    next: Next,
) -> Response {
    if keys.route_is_open(req.uri().path()) {
        return next.run(req).await;
    }
    let presented = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            v.strip_prefix("Bearer ")
                .or_else(|| v.strip_prefix("bearer "))
        })
        .map(str::trim);
    let Some(presented) = presented else {
        return unauthorized("missing API key: send `Authorization: Bearer <key>`");
    };
    match keys.check(presented) {
        Some(label) => {
            let label = KeyLabel(label.to_owned());
            req.extensions_mut().insert(label);
            next.run(req).await
        }
        None => unauthorized("invalid API key"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_and_files_parse_and_refuse() {
        assert_eq!(
            ApiKeys::from_specs([], None, false),
            Ok(None),
            "no keys = off"
        );
        assert!(
            ApiKeys::from_specs([], None, true).is_err(),
            "auth-all with no key"
        );
        let k = ApiKeys::from_specs(
            ["ci=0123456789"],
            Some("# comment\n\nconsole abcdefghij\nbench=zzzzzzzzzz\n"),
            false,
        )
        .unwrap()
        .unwrap();
        assert_eq!(k.labels(), vec!["ci", "console", "bench"]);
        assert_eq!(k.check("abcdefghij"), Some("console"));
        assert_eq!(k.check("abcdefghiX"), None);
        assert_eq!(k.check(""), None);
        assert!(ApiKeys::from_specs(["x=short"], None, false).is_err());
        assert!(ApiKeys::from_specs(["=0123456789"], None, false).is_err());
        assert!(ApiKeys::from_specs(["a=0123456789", "b=0123456789"], None, false).is_err());
        assert!(ApiKeys::from_specs(["nokey"], None, false).is_err());
        assert!(
            !format!("{k:?}").contains("abcdefghij"),
            "Debug never prints a key"
        );
    }

    #[test]
    fn a_console_ref_is_kept_only_from_the_console_key_when_auth_is_on() {
        let r = || Some("c18f-2".to_string());
        // auth off: kept (nothing to authenticate it against)
        assert_eq!(keep_client_ref(None, None, r()), r());
        let k = ApiKeys::from_specs(["console=0123456789", "ci=abcdefghij"], None, false)
            .unwrap()
            .unwrap();
        assert_eq!(keep_client_ref(Some(&k), Some("console"), r()), r());
        // negative controls: another key's label, and no label, drop the spoofed ref
        assert_eq!(keep_client_ref(Some(&k), Some("ci"), r()), None);
        assert_eq!(keep_client_ref(Some(&k), None, r()), None);
        assert_eq!(keep_client_ref(Some(&k), Some("console"), None), None);
    }

    #[test]
    fn open_routes_follow_the_flag() {
        let k = ApiKeys::from_specs(["a=0123456789"], None, false)
            .unwrap()
            .unwrap();
        assert!(k.route_is_open("/health") && k.route_is_open("/metrics"));
        assert!(!k.route_is_open("/v1/models") && !k.route_is_open("/v1/stats"));
        let k = ApiKeys::from_specs(["a=0123456789"], None, true)
            .unwrap()
            .unwrap();
        assert!(!k.route_is_open("/health"));
    }
}
