//! The security plane over HTTP.
//!
//! Proves: EVERY route the console's router declares (read from the router's own source, so a
//! new route cannot skip this test) refuses a call without the admin token and with a wrong
//! one, the `/v1` proxy too; a right token passes; a mutating call without the CSRF header is
//! refused even with the token; the failed-auth limiter locks an address out (429, even with
//! the right token) and expires; the admin token is rotated in place (old refused, new
//! accepted, shown once, the config holds only its hash); `/v1` keys per label are created,
//! accepted, revoked (refused) and deleted; each change is in the audit log with its label; a
//! non-loopback console refuses to start without a token; only the SPA's static files are open.

use std::sync::Arc;
use std::time::Duration;

use saber_console::apply::Console;
use saber_console::auth::{hash_secret, Admin, Auth, Limiter};
use saber_console::config::Config;
use saber_console::http::{router, AppState};
use saber_console::logring::LogRing;
use saber_console::proxy::Proxy;
use saber_console::scrape::MetricsStore;
use saber_console::supervisor::Supervisor;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const TOKEN: &str = "sa-admin-route-test-0123456789";

/// Every `(method, path)` a router source declares: each `.route("path", ...)` call and the
/// method routers in it (`get(..)`, `.post(..)`, `axum::routing::delete(..)`, `any(..)`).
fn routes_in(src: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = src;
    while let Some(i) = rest.find(".route(") {
        rest = &rest[i + ".route(".len()..];
        // the call's text: up to its closing paren
        let mut depth = 1;
        let mut end = 0;
        for (j, c) in rest.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = j;
                        break;
                    }
                }
                _ => {}
            }
        }
        let call = &rest[..end];
        let Some(q0) = call.find('"') else { continue };
        let Some(q1) = call[q0 + 1..].find('"') else {
            continue;
        };
        let path = call[q0 + 1..q0 + 1 + q1].to_string();
        let handlers = &call[q0 + 1 + q1..];
        let mut any = false;
        for (m, pat) in [
            ("GET", "get("),
            ("POST", "post("),
            ("PUT", "put("),
            ("DELETE", "delete("),
            ("POST", "any("),
        ] {
            // `get(` must be a method router, not part of a longer name (`budget(`)
            let mut k = 0;
            while let Some(p) = handlers[k..].find(pat) {
                let at = k + p;
                let before = handlers[..at].chars().last().unwrap_or(' ');
                if !before.is_ascii_alphanumeric() && before != '_' {
                    out.push((m.to_string(), path.clone()));
                    any = true;
                    break;
                }
                k = at + pat.len();
            }
        }
        assert!(any, "no method router found for {path}: {call}");
    }
    out
}

fn concrete(path: &str) -> String {
    // `{id}` -> 1, `{*rest}` -> chat/completions, other params -> x
    let mut out = String::new();
    let mut rest = path;
    while let Some(i) = rest.find('{') {
        out.push_str(&rest[..i]);
        let j = rest[i..].find('}').unwrap() + i;
        let name = &rest[i + 1..j];
        out.push_str(match name {
            "id" => "1",
            "*rest" => "chat/completions",
            _ => "x",
        });
        rest = &rest[j + 1..];
    }
    out.push_str(rest);
    out
}

struct Reply {
    code: u16,
    body: String,
    head: String,
}

async fn call(
    port: u16,
    method: &str,
    path: &str,
    token: Option<&str>,
    csrf: bool,
    body: &str,
) -> Reply {
    let mut s = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    let a = token.map_or(String::new(), |t| format!("Authorization: Bearer {t}\r\n"));
    let h = if csrf { "X-Saber-Console: 1\r\n" } else { "" };
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 192.168.7.7:{port}\r\n{a}{h}Content-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut b = Vec::new();
    tokio::time::timeout(Duration::from_secs(20), s.read_to_end(&mut b))
        .await
        .unwrap()
        .unwrap();
    let text = String::from_utf8_lossy(&b).into_owned();
    let code: u16 = text[9..12].parse().unwrap();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    Reply {
        code,
        body: body.to_string(),
        head: head.to_string(),
    }
}

fn json_of(r: &Reply) -> Value {
    // a chunked body: the JSON is the line that starts with `{`
    r.body
        .lines()
        .find(|l| l.trim_start().starts_with('{'))
        .and_then(|l| serde_json::from_str(l).ok())
        .unwrap_or(Value::Null)
}

struct Fixture {
    port: u16,
    sup: Arc<Supervisor>,
    auth: Arc<Auth>,
    _dir: saber_test_scratch::ScratchDir,
}

async fn console(limiter: Limiter) -> Fixture {
    let dir = saber_test_scratch::ScratchDir::new("console-auth-routes");
    let mut cfg = Config::mock(1);
    cfg.engine.autostart = false;
    cfg.console.admin_token = TOKEN.into();
    cfg.console.host = "0.0.0.0".into();
    cfg.save(&dir.path().join("config.toml")).unwrap();
    let cfg = Config::load(&dir.path().join("config.toml"))
        .unwrap()
        .unwrap();
    let admin = cfg.admin();
    assert_eq!(admin.source(), "config");
    let (sup, _rx) = Supervisor::new(
        cfg,
        dir.path().join("config.toml"),
        dir.path().to_path_buf(),
        dir.path().join("saber"),
        Arc::new(LogRing::new(100)),
    );
    let metrics = Arc::new(MetricsStore::new(10, 5));
    let auth = Arc::new(Auth::with_limiter(admin, limiter));
    let state = AppState {
        console: Arc::new(Console::new(sup.clone(), metrics.clone(), &json!({}), 0)),
        sup: sup.clone(),
        metrics,
        schema: Arc::new(json!({"flags": []})),
        auth: auth.clone(),
        // a non-loopback console (no Host rule; the token is the guard)
        loopback: false,
        open_lan: false,
        proxy: Arc::new(Proxy::new(None)),
        usage: None,
    };
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = l.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(
            l,
            router(state).into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .unwrap();
    });
    Fixture {
        port,
        sup,
        auth,
        _dir: dir,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_route_refuses_without_the_token() {
    let mut routes = routes_in(include_str!("../src/http.rs"));
    routes.extend(routes_in(include_str!("../src/wizard.rs")));
    routes.sort();
    routes.dedup();
    // the parser found the router (a guard against a silent zero): the console has 50+ routes
    assert!(routes.len() >= 50, "{} routes: {routes:?}", routes.len());
    for must in [
        ("GET", "/api/status"),
        ("POST", "/api/config/apply"),
        ("POST", "/api/engine/start"),
        ("GET", "/api/audit"),
        ("POST", "/api/auth/keys"),
        ("GET", "/metrics"),
        ("POST", "/v1/{*rest}"),
        ("POST", "/api/setup/launch"),
        ("DELETE", "/api/packs/{name}"),
    ] {
        assert!(
            routes.contains(&(must.0.to_string(), must.1.to_string())),
            "{must:?} not parsed: {routes:?}"
        );
    }
    let f = console(Limiter::new(
        100_000,
        Duration::from_secs(60),
        Duration::from_secs(60),
    ))
    .await;
    for (m, p) in &routes {
        // the sign-in routes take the token in their body (`sign_in_remembers_this_browser`)
        if p == "/api/auth/login" || p == "/api/auth/logout" {
            continue;
        }
        let path = concrete(p);
        for tok in [None, Some("sa-admin-wrong-token-000000000")] {
            let r = call(f.port, m, &path, tok, true, "{}").await;
            assert_eq!(
                r.code, 401,
                "{m} {path} with {tok:?} answered {} {}",
                r.code, r.body
            );
            if p.starts_with("/v1/") {
                assert_eq!(json_of(&r)["error"]["code"], 401, "{}", r.body);
            }
        }
        // the same call with the right token passes auth: GETs answer; a mutating call without
        // the CSRF header is refused (403) before its handler runs
        if p.starts_with("/v1/") {
            let r = call(f.port, m, &path, Some(TOKEN), false, "{}").await;
            assert_eq!(r.code, 503, "{m} {path}: past auth, no engine: {}", r.body);
        } else if m == "GET" && (p.ends_with("/stream") || p == "/api/system") {
            // an SSE stream never ends (its ?token= form is checked below); the system probe
            // asks the host's container runtime (setup_downloads.rs runs it on fixed facts)
        } else if m == "GET" {
            let r = call(f.port, m, &path, Some(TOKEN), false, "").await;
            assert!(
                r.code != 401 && r.code != 403 && r.code != 429,
                "{m} {path} with the token: {} {}",
                r.code,
                r.body
            );
        } else {
            let r = call(f.port, m, &path, Some(TOKEN), false, "{}").await;
            assert_eq!(
                r.code, 403,
                "{m} {path} without the CSRF header: {}",
                r.body
            );
        }
    }
    // an EventSource's ?token= works for a stream (read the head only)
    let mut s = TcpStream::connect(("127.0.0.1", f.port)).await.unwrap();
    s.write_all(
        format!(
            "GET /api/logs/stream?token={TOKEN} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    let mut b = [0u8; 64];
    let n = tokio::time::timeout(Duration::from_secs(5), s.read(&mut b))
        .await
        .unwrap()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&b[..n]).starts_with("HTTP/1.1 200"),
        "{}",
        String::from_utf8_lossy(&b[..n])
    );
    // the SPA's files are the only thing served without the token (the login screen)
    let r = call(f.port, "GET", "/", None, false, "").await;
    assert!(r.code == 200 || r.code == 404, "{} {}", r.code, r.body);
    assert!(r.code != 401);
    let r = call(f.port, "GET", "/api/nope", None, false, "").await;
    assert_eq!(r.code, 401, "an unknown /api path is not a probe for free");
}

/// `call` with extra header lines (a `Cookie:`), no Authorization.
async fn call_raw(port: u16, method: &str, path: &str, headers: &str, body: &str) -> Reply {
    let mut s = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 192.168.7.7:{port}\r\n{headers}Content-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut b = Vec::new();
    tokio::time::timeout(Duration::from_secs(20), s.read_to_end(&mut b))
        .await
        .unwrap()
        .unwrap();
    let text = String::from_utf8_lossy(&b).into_owned();
    let code: u16 = text[9..12].parse().unwrap();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    Reply {
        code,
        body: body.to_string(),
        head: head.to_string(),
    }
}

fn set_cookie(r: &Reply) -> Option<String> {
    r.head
        .lines()
        .find_map(|l| l.to_ascii_lowercase().starts_with("set-cookie:").then(|| l[11..].trim().to_string()))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sign_in_remembers_this_browser() {
    let f = console(Limiter::new(
        100_000,
        Duration::from_secs(60),
        Duration::from_secs(60),
    ))
    .await;
    let csrf = "X-Saber-Console: 1\r\n";
    let good = format!("{{\"token\": \"{TOKEN}\"}}");
    // a login needs the CSRF header; a wrong token is refused and sets no cookie
    let r = call_raw(f.port, "POST", "/api/auth/login", "", &good).await;
    assert_eq!(r.code, 403, "{}", r.body);
    let r = call_raw(f.port, "POST", "/api/auth/login", csrf, "{\"token\": \"nope\"}").await;
    assert_eq!(r.code, 401, "{}", r.body);
    assert!(set_cookie(&r).is_none());
    // the right token sets a signed, HttpOnly, SameSite=Strict, 30-day cookie
    let r = call_raw(f.port, "POST", "/api/auth/login", csrf, &good).await;
    assert_eq!(r.code, 200, "{}", r.body);
    let c = set_cookie(&r).expect("a Set-Cookie");
    for part in ["saber_session=", "HttpOnly", "SameSite=Strict", "Max-Age=2592000", "Path=/"] {
        assert!(c.contains(part), "{part} in {c}");
    }
    assert!(!c.contains(TOKEN), "the cookie never holds the token");
    let value = c.split(';').next().unwrap().to_string();
    // the cookie alone is a credential (no Authorization header)
    let r = call_raw(f.port, "GET", "/api/status", &format!("Cookie: {value}\r\n"), "").await;
    assert_eq!(r.code, 200, "{}", r.body);
    let r = call_raw(f.port, "GET", "/api/status", "Cookie: saber_session=1.00\r\n", "").await;
    assert_eq!(r.code, 401);
    let r = call_raw(f.port, "GET", "/api/status", "", "").await;
    assert_eq!(r.code, 401);
    // a mutating call still needs the CSRF header even with the cookie
    let r = call_raw(
        f.port,
        "POST",
        "/api/engine/stop",
        &format!("Cookie: {value}\r\n"),
        "{}",
    )
    .await;
    assert_eq!(r.code, 403, "{}", r.body);
    // logout clears it
    let r = call_raw(f.port, "POST", "/api/auth/logout", csrf, "{}").await;
    assert_eq!(r.code, 200);
    assert!(set_cookie(&r).is_some_and(|c| c.contains("Max-Age=0")));
    // rotating the token ends every remembered browser
    f.auth.set_admin(saber_console::auth::Admin::Flag("a-brand-new-token-0000".into()));
    let r = call_raw(f.port, "GET", "/api/status", &format!("Cookie: {value}\r\n"), "").await;
    assert_eq!(r.code, 401, "{}", r.body);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_auth_is_rate_limited_per_address() {
    let f = console(Limiter::new(
        5,
        Duration::from_secs(60),
        Duration::from_millis(1500),
    ))
    .await;
    // no credential at all (a signed-out page polling) is refused but not counted
    for _ in 0..12 {
        let r = call(f.port, "GET", "/api/status", None, false, "").await;
        assert_eq!(r.code, 401);
    }
    let r = call(f.port, "GET", "/api/status", Some(TOKEN), false, "").await;
    assert_eq!(
        r.code, 200,
        "no lockout from calls without a token: {}",
        r.body
    );
    for i in 0..5 {
        let r = call(
            f.port,
            "GET",
            "/api/status",
            Some("wrong-token-wrong-0001"),
            false,
            "",
        )
        .await;
        assert_eq!(r.code, 401, "attempt {i}");
    }
    let r = call(f.port, "GET", "/api/status", Some(TOKEN), false, "").await;
    assert_eq!(
        r.code, 429,
        "locked out even with the right token: {}",
        r.body
    );
    assert!(
        r.head.to_ascii_lowercase().contains("retry-after:"),
        "{}",
        r.head
    );
    let r = call(
        f.port,
        "POST",
        "/v1/chat/completions",
        Some(TOKEN),
        false,
        "{}",
    )
    .await;
    assert_eq!(r.code, 429, "the proxy shares the lockout");
    tokio::time::sleep(Duration::from_millis(1700)).await;
    let r = call(f.port, "GET", "/api/status", Some(TOKEN), false, "").await;
    assert_eq!(r.code, 200, "the lockout expires: {}", r.body);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rotate_keys_revoke_and_the_audit_log() {
    let f = console(Limiter::default()).await;
    let p = f.port;
    // /api/auth: who am I, no secret anywhere
    let r = call(p, "GET", "/api/auth", Some(TOKEN), false, "").await;
    let v = json_of(&r);
    assert_eq!(
        (v["who"].as_str(), v["required"].as_bool()),
        (Some("admin"), Some(true))
    );
    assert_eq!(v["source"], "config");
    // the config API shows no secret and no hash
    let r = call(p, "GET", "/api/config", Some(TOKEN), false, "").await;
    assert!(
        !r.body.contains(TOKEN) && !r.body.contains("sha256$"),
        "{}",
        r.body
    );
    assert!(r.body.contains("(set)"), "{}", r.body);

    // a key per label: shown once, then only its label
    let r = call(
        p,
        "POST",
        "/api/auth/keys",
        Some(TOKEN),
        true,
        r#"{"label":"laptop"}"#,
    )
    .await;
    assert_eq!(r.code, 200, "{}", r.body);
    let key = json_of(&r)["key"].as_str().unwrap().to_string();
    assert!(key.starts_with("sk-saber-"));
    let r = call(
        p,
        "POST",
        "/api/auth/keys",
        Some(TOKEN),
        true,
        r#"{"label":"laptop"}"#,
    )
    .await;
    assert_eq!(r.code, 409, "a label is unique");
    let r = call(
        p,
        "POST",
        "/api/auth/keys",
        Some(TOKEN),
        true,
        r#"{"label":"bad label!"}"#,
    )
    .await;
    assert_eq!(r.code, 400);
    let r = call(p, "GET", "/api/auth/keys", Some(TOKEN), false, "").await;
    assert!(
        !r.body.contains(&key) && r.body.contains("laptop"),
        "{}",
        r.body
    );
    let on_disk = std::fs::read_to_string(&f.sup.config_path).unwrap();
    assert!(
        !on_disk.contains(&key) && !on_disk.contains(TOKEN),
        "{on_disk}"
    );
    // the key opens the proxy (no engine: 503 past auth), not the admin plane
    let r = call(p, "POST", "/v1/chat/completions", Some(&key), false, "{}").await;
    assert_eq!(r.code, 503, "{}", r.body);
    let r = call(p, "GET", "/api/status", Some(&key), false, "").await;
    assert_eq!(r.code, 401, "a /v1 key is not an admin token");
    // revoke: refused; delete: gone
    let r = call(
        p,
        "POST",
        "/api/auth/keys/laptop/revoke",
        Some(TOKEN),
        true,
        "",
    )
    .await;
    assert_eq!(r.code, 200, "{}", r.body);
    let r = call(p, "POST", "/v1/chat/completions", Some(&key), false, "{}").await;
    assert_eq!(r.code, 401, "a revoked key is refused");
    let r = call(p, "GET", "/api/auth/keys", Some(TOKEN), false, "").await;
    assert_eq!(json_of(&r)["keys"][0]["revoked"], true);
    // a config save cannot bring a revoked key back (credentials are not a config field)
    let r = call(p, "GET", "/api/config", Some(TOKEN), false, "").await;
    let mut c = json_of(&r)["config"].clone();
    c["proxy"]["keys"][0]["revoked"] = json!(false);
    let r = call(p, "PUT", "/api/config", Some(TOKEN), true, &c.to_string()).await;
    assert_eq!(r.code, 200, "{}", r.body);
    let r = call(p, "POST", "/v1/chat/completions", Some(&key), false, "{}").await;
    assert_eq!(r.code, 401, "still revoked after a config save");
    let r = call(p, "DELETE", "/api/auth/keys/laptop", Some(TOKEN), true, "").await;
    assert_eq!(r.code, 200, "{}", r.body);
    let r = call(p, "DELETE", "/api/auth/keys/laptop", Some(TOKEN), true, "").await;
    assert_eq!(r.code, 404);

    // rotate the admin token in place
    let r = call(p, "POST", "/api/auth/admin-token", Some(TOKEN), true, "").await;
    assert_eq!(r.code, 200, "{}", r.body);
    let new = json_of(&r)["token"].as_str().unwrap().to_string();
    assert_ne!(new, TOKEN);
    let r = call(p, "GET", "/api/status", Some(TOKEN), false, "").await;
    assert_eq!(r.code, 401, "the old token is refused at once");
    let r = call(p, "GET", "/api/status", Some(&new), false, "").await;
    assert_eq!(r.code, 200);
    let stored = Config::load(&f.sup.config_path).unwrap().unwrap();
    assert!(stored.console.admin_token.is_empty());
    assert!(
        stored.admin().check(&new),
        "the config holds the new token's hash"
    );
    assert!(f.auth.check_admin(&new));
    // a non-loopback console cannot drop its token
    let r = call(p, "DELETE", "/api/auth/admin-token", Some(&new), true, "").await;
    assert_eq!(r.code, 409, "{}", r.body);

    // the audit log: every change, its label, newest first
    let r = call(p, "GET", "/api/audit?limit=50", Some(&new), false, "").await;
    let e = json_of(&r)["entries"].as_array().unwrap().clone();
    let acts: Vec<&str> = e.iter().map(|x| x["action"].as_str().unwrap()).collect();
    for want in [
        "auth.key.create",
        "auth.key.revoke",
        "auth.key.delete",
        "config.save",
        "auth.admin-token.rotate",
    ] {
        assert!(acts.contains(&want), "{want} not in {acts:?}");
    }
    assert!(
        e.iter()
            .all(|x| x["who"] == "admin" && x["ip"] == "127.0.0.1"),
        "{e:?}"
    );
    assert_eq!(e[0]["action"], "auth.admin-token.remove", "{e:?}");
    assert_eq!(
        e[0]["status"], 409,
        "a refused call is logged with its status"
    );
    let created = e
        .iter()
        .find(|x| x["action"] == "auth.key.create" && x["status"] == 200)
        .unwrap();
    assert_eq!(created["detail"], "label=laptop");
    assert!(
        !r.body.contains(&new) && !r.body.contains(&key),
        "no secret in the log"
    );
}

#[test]
fn a_non_loopback_console_refuses_to_start_without_a_token() {
    let dir = saber_test_scratch::ScratchDir::new("console-bind-refuse");
    let args = |token: Option<&str>| saber_console::ConsoleArgs {
        host: Some("0.0.0.0".into()),
        port: Some(0),
        admin_token: token.map(str::to_string),
        admin: None,
        config_path: dir.path().join("absent.toml"),
        default_config: Config::default(),
        override_config: None,
        home: dir.path().to_path_buf(),
        exe: dir.path().join("saber"),
        schema: json!({}),
        no_auth: false,
    };
    let e = saber_console::run(args(None)).unwrap_err().to_string();
    assert!(e.contains("refusing to start"), "{e}");
    let e = saber_console::run(args(Some(""))).unwrap_err().to_string();
    assert!(
        e.contains("refusing to start"),
        "an empty token is none: {e}"
    );
    // a stored hash is a token (the bind check passes; nothing else is started here)
    let h = Admin::Hash(hash_secret("cfg-token-0123456789").unwrap());
    assert!(saber_console::check_bind("0.0.0.0", &h).is_ok());
}
