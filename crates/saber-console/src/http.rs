//! The console's HTTP plane: the JSON/SSE API under `/api/*` and the embedded SPA.
//!
//! Security: a non-loopback bind needs an admin token (checked in [`crate::run`]);
//! when a token is set EVERY `/api/*` route and the console's `/metrics` carry it
//! (`Authorization: Bearer`, or `?token=` for an `EventSource`), checked in constant time
//! against the stored salted hash ([`crate::auth`]); a client address that fails it
//! [`crate::auth::Limiter::MAX_FAILS`] times a minute is refused (429) for a minute. Every
//! mutating call also needs the `X-Saber-Console: 1` header, which a cross-site form or
//! `fetch` cannot send without a CORS preflight this server never answers; on a loopback bind
//! the `Host` header must name the loopback (DNS-rebinding guard). Every mutating call is
//! written to the audit log ([`crate::audit`]) with the credential's label and the address.
//! Only the SPA's static files are served without the token (the login screen).

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Query, Request, State};
use axum::http::{header, HeaderMap, Method, StatusCode, Uri};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::Stream;
use futures_util::StreamExt;
use rust_embed::RustEmbed;
use serde::Deserialize;
use serde_json::json;
use tokio_stream::wrappers::BroadcastStream;

use crate::config::Config;
use crate::scrape::MetricsStore;
use crate::supervisor::Supervisor;
use crate::usage::{RowFilter, UsageStore};

#[derive(RustEmbed)]
#[folder = "../../ui/dist"]
struct Assets;

#[derive(Clone)]
pub struct AppState {
    pub sup: Arc<Supervisor>,
    pub metrics: Arc<MetricsStore>,
    pub schema: Arc<serde_json::Value>,
    /// The admin credential (swapped in place by a rotate) and the limiter.
    pub auth: Arc<crate::auth::Auth>,
    pub loopback: bool,
    /// `--no-auth` beyond the loopback: no token, but only private-network peers and hostnames
    /// are served ([`lan_guard`]).
    pub open_lan: bool,
    /// The `/v1` proxy and the usage store.
    pub proxy: Arc<crate::proxy::Proxy>,
    pub usage: Option<Arc<UsageStore>>,
    /// Validation, apply / rollback, history, profiles, packs.
    pub console: Arc<crate::apply::Console>,
}

pub fn router(state: AppState) -> Router {
    let api = Router::new()
        .route("/api/status", get(status))
        .route("/api/config", get(get_config).put(put_config))
        .route("/api/schema", get(schema))
        .route("/api/command", get(command))
        .route("/api/engine/start", post(start))
        .route("/api/engine/stop", post(stop))
        .route("/api/engine/restart", post(restart))
        .route("/api/logs", get(logs))
        .route("/api/logs/stream", get(logs_stream))
        .route("/api/metrics", get(metrics))
        .route("/api/metrics/latest", get(metrics_latest))
        .route("/api/metrics/stream", get(metrics_stream))
        .route("/api/series", get(series))
        .route("/api/hw", get(hw))
        .route("/api/logs/download", get(logs_download))
        .route("/api/usage/status", get(usage_status))
        .route("/api/usage/summary", get(usage_summary))
        .route("/api/usage/requests", get(usage_requests))
        .route("/api/usage/requests/{uid}", get(usage_request))
        .route("/api/usage/monthly", get(usage_monthly))
        .route("/api/usage/export", get(usage_export))
        .route("/metrics", get(console_metrics))
        // the settings plane
        .route("/api/config/validate", post(config_validate))
        .route("/api/config/tp", post(config_tp))
        .route("/api/config/apply", post(config_apply))
        .route("/api/apply", get(apply_status))
        .route("/api/history", get(history_list))
        .route("/api/history/{id}", get(history_get))
        .route("/api/history/{id}/rollback", post(history_rollback))
        .route("/api/profiles", get(profiles_list).post(profiles_save))
        .route("/api/profiles/import", post(profiles_import))
        .route("/api/profiles/compose", post(profiles_compose))
        .route(
            "/api/profiles/{name}",
            axum::routing::put(profiles_rename).delete(profiles_delete),
        )
        .route("/api/profiles/{name}/duplicate", post(profiles_duplicate))
        .route("/api/profiles/{name}/export", get(profiles_export))
        .route("/api/packs", get(packs))
        // System check, catalog, downloads, pack delete, the setup wizard
        .merge(crate::wizard::routes())
        // Credentials and the audit log
        .route("/api/auth", get(auth_info))
        .route(
            "/api/auth/admin-token",
            post(admin_token_rotate).delete(admin_token_remove),
        )
        .route("/api/auth/keys", get(keys_list).post(keys_create))
        .route("/api/auth/keys/{label}", axum::routing::delete(keys_delete))
        .route("/api/auth/keys/{label}/revoke", post(keys_revoke))
        .route("/api/audit", get(audit_list))
        .route_layer(middleware::from_fn_with_state(state.clone(), auth))
        // sign in / out: no admin credential yet (login checks the token itself)
        .route("/api/auth/login", post(auth_login))
        .route("/api/auth/logout", post(auth_logout));
    // the /v1 proxy checks its own keys (crate::proxy), not the admin plane's header rules
    let v1 = Router::new()
        .route("/v1/{*rest}", axum::routing::any(crate::proxy::handle))
        .with_state(state.clone());
    api.merge(v1)
        .fallback(static_file)
        .layer(middleware::from_fn_with_state(state.clone(), host_guard))
        .layer(middleware::from_fn_with_state(state.clone(), lan_guard))
        .with_state(state)
}

fn query_token(uri: &Uri) -> Option<String> {
    uri.query()?.split('&').find_map(|kv| {
        kv.strip_prefix("token=")
            .map(|v| percent_decode(v).unwrap_or_else(|| v.to_string()))
    })
}

/// `%XX` and `+` decoding of a query value (`encodeURIComponent`'s output).
fn percent_decode(v: &str) -> Option<String> {
    let b = v.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' => {
                let h = std::str::from_utf8(b.get(i + 1..i + 3)?).ok()?;
                out.push(u8::from_str_radix(h, 16).ok()?);
                i += 3;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8(out).ok()
}

/// The client's address (the listener's `ConnectInfo`; a router served without it, e.g. a test,
/// counts every client as one address).
#[must_use]
pub fn client_ip(req: &Request) -> std::net::IpAddr {
    req.extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map_or(std::net::IpAddr::from([0, 0, 0, 0]), |c| c.0.ip())
}

/// 429 while an address is locked out (`oai`: the OpenAI error envelope, for `/v1`).
#[must_use]
pub fn too_many(retry_after_s: u64, oai: bool) -> Response {
    let msg = format!(
        "too many failed authentication attempts from this address: try again in {retry_after_s} s"
    );
    let body = if oai {
        json!({"error": {"message": msg, "type": "rate_limit_error", "code": 429}})
    } else {
        json!({"error": msg})
    };
    (
        StatusCode::TOO_MANY_REQUESTS,
        [(header::RETRY_AFTER, retry_after_s.to_string())],
        Json(body),
    )
        .into_response()
}

/// Who made an admin call (the audit log's `who`): the credential's label.
#[derive(Clone, Debug)]
pub struct Who(pub String);

async fn auth(State(st): State<AppState>, mut req: Request, next: Next) -> Response {
    let ip = client_ip(&req);
    let who = if st.auth.required() {
        if let Some(left) = st.auth.limiter.blocked(ip) {
            return too_many(left, false);
        }
        let bearer = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(|v| v.trim().to_string());
        let given = bearer.or_else(|| query_token(req.uri()));
        let presented = given.as_deref().is_some_and(|g| !g.is_empty());
        let session = session_cookie(req.headers())
            .is_some_and(|v| st.auth.session_check(&v, crate::now_ms()));
        if !session && !given.is_some_and(|g| st.auth.check_admin(&g)) {
            // a WRONG token counts (a guess); no token at all (a signed-out page) does not
            if presented {
                st.auth.limiter.fail(ip);
            }
            return (
                StatusCode::UNAUTHORIZED,
                [(header::WWW_AUTHENTICATE, "Bearer")],
                Json(json!({"error": "admin token required"})),
            )
                .into_response();
        }
        st.auth.limiter.ok(ip);
        "admin"
    } else {
        "local"
    };
    if req.method() != Method::GET && req.headers().get("x-saber-console").is_none() {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"error": "mutating calls need the X-Saber-Console: 1 header"})),
        )
            .into_response();
    }
    let action = crate::audit::action_of(req.method().as_str(), req.uri().path());
    let route = format!("{} {}", req.method(), req.uri().path());
    req.extensions_mut().insert(Who(who.to_string()));
    let resp = next.run(req).await;
    if let Some(action) = action {
        let detail = resp
            .headers()
            .get(AUDIT_DETAIL)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let e = crate::audit::AuditEntry {
            ts_ms: crate::now_ms(),
            who: who.to_string(),
            ip: Some(ip.to_string()),
            action,
            route,
            status: resp.status().as_u16(),
            detail,
        };
        let c = st.console.clone();
        // a file append: off the async workers
        tokio::task::spawn_blocking(move || c.audit.record(&e));
    }
    resp
}

/// The value of the remember-me cookie, when the browser sent one.
fn session_cookie(headers: &HeaderMap) -> Option<String> {
    headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(';'))
        .find_map(|kv| {
            kv.trim()
                .strip_prefix(crate::auth::SESSION_COOKIE)
                .and_then(|r| r.strip_prefix('='))
                .map(str::to_string)
        })
}

fn session_set_cookie(value: &str, max_age_s: u64) -> String {
    // HttpOnly: a page script can never read it; SameSite=Strict: another site's request never
    // carries it (the mutating calls also need the X-Saber-Console header)
    format!(
        "{}={value}; Path=/; Max-Age={max_age_s}; HttpOnly; SameSite=Strict",
        crate::auth::SESSION_COOKIE
    )
}

#[derive(Deserialize)]
struct LoginBody {
    token: String,
}

/// `POST /api/auth/login {"token"}`: check the admin token and remember this browser (a signed
/// 30-day cookie). A wrong token counts toward the failed-auth limit.
async fn auth_login(State(st): State<AppState>, req: Request) -> Response {
    let ip = client_ip(&req);
    if req.headers().get("x-saber-console").is_none() {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"error": "login needs the X-Saber-Console: 1 header"})),
        )
            .into_response();
    }
    if let Some(left) = st.auth.limiter.blocked(ip) {
        return too_many(left, false);
    }
    let Ok(bytes) = axum::body::to_bytes(req.into_body(), 4096).await else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let Ok(body) = serde_json::from_slice::<LoginBody>(&bytes) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "expected {\"token\": ...}"})),
        )
            .into_response();
    };
    if !st.auth.required() {
        return Json(json!({"ok": true, "required": false})).into_response();
    }
    if !st.auth.check_admin(body.token.trim()) {
        st.auth.limiter.fail(ip);
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "admin token required"})),
        )
            .into_response();
    }
    st.auth.limiter.ok(ip);
    let mut resp = Json(json!({"ok": true, "required": true})).into_response();
    if let Some(v) = st.auth.session_mint(crate::now_ms()) {
        if let Ok(h) = session_set_cookie(&v, crate::auth::SESSION_TTL.as_secs()).parse() {
            resp.headers_mut().insert(header::SET_COOKIE, h);
        }
    }
    resp
}

/// `POST /api/auth/logout`: forget this browser (the cookie is cleared).
async fn auth_logout(req: Request) -> Response {
    if req.headers().get("x-saber-console").is_none() {
        return StatusCode::FORBIDDEN.into_response();
    }
    let mut resp = Json(json!({"ok": true})).into_response();
    if let Ok(h) = session_set_cookie("", 0).parse() {
        resp.headers_mut().insert(header::SET_COOKIE, h);
    }
    resp
}

/// True for an address on a private network: loopback, RFC 1918, link-local, unique-local.
#[must_use]
pub fn is_private_ip(ip: std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(v) => v.is_loopback() || v.is_private() || v.is_link_local(),
        IpAddr::V6(v) => {
            if let Some(v4) = v.to_ipv4_mapped() {
                return is_private_ip(IpAddr::V4(v4));
            }
            v.is_loopback()
                || (v.segments()[0] & 0xfe00) == 0xfc00
                || (v.segments()[0] & 0xffc0) == 0xfe80
        }
    }
}

/// A `Host` an open (`--no-auth`) console answers: an IP literal, `localhost`, or a local
/// name (no dot, or `.local` / `.lan` / `.home.arpa` / `.internal`). A public name that resolves
/// to this machine (DNS rebinding) is refused.
#[must_use]
pub fn host_is_lan(h: &str) -> bool {
    let host = if let Some(rest) = h.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        h.rsplit_once(':').map_or(h, |(a, _)| a)
    };
    host.parse::<std::net::IpAddr>().is_ok()
        || !host.contains('.')
        || [".local", ".lan", ".home.arpa", ".internal"]
            .iter()
            .any(|sfx| host.ends_with(sfx))
}

/// `--no-auth` beyond the loopback: only private-network peers and local hostnames are served,
/// and a browser's cross-site request (an `Origin` that is not this host) is refused.
async fn lan_guard(State(st): State<AppState>, req: Request, next: Next) -> Response {
    if st.open_lan {
        let peer_ok = is_private_ip(client_ip(&req));
        let host = req
            .headers()
            .get(header::HOST)
            .and_then(|v| v.to_str().ok());
        let host_ok = host.is_some_and(host_is_lan);
        let origin_ok = req
            .headers()
            .get(header::ORIGIN)
            .and_then(|v| v.to_str().ok())
            .is_none_or(|o| {
                o.split_once("://")
                    .is_some_and(|(_, authority)| Some(authority) == host)
            });
        if !peer_ok || !host_ok || !origin_ok {
            return (
                StatusCode::FORBIDDEN,
                "this console without a token answers private-network clients on local hostnames only",
            )
                .into_response();
        }
    }
    next.run(req).await
}

/// A handler's note for the audit entry of its call (a response header, never a secret).
pub const AUDIT_DETAIL: &str = "x-saber-audit-detail";

fn host_is_loopback(h: &str) -> bool {
    let host = if let Some(rest) = h.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        h.rsplit_once(':').map_or(h, |(a, _)| a)
    };
    matches!(host, "localhost" | "127.0.0.1" | "::1")
}

async fn host_guard(State(st): State<AppState>, req: Request, next: Next) -> Response {
    if st.loopback {
        let ok = req
            .headers()
            .get(header::HOST)
            .and_then(|v| v.to_str().ok())
            .is_some_and(host_is_loopback);
        if !ok {
            return (
                StatusCode::FORBIDDEN,
                "this console answers loopback hosts only",
            )
                .into_response();
        }
    }
    next.run(req).await
}

async fn status(State(st): State<AppState>) -> impl IntoResponse {
    Json(st.sup.status())
}

async fn get_config(State(st): State<AppState>) -> impl IntoResponse {
    let c = st
        .sup
        .config
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    Json(json!({"path": st.sup.config_path.display().to_string(), "config": c.redacted()}))
}

async fn put_config(State(st): State<AppState>, Json(c): Json<Config>) -> Response {
    // the credentials are the Security page's (`/api/auth/*`), never a config save's
    let c = c.with_credentials_of(&st.sup.config_now());
    if let Err(e) = c.validate() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": e.to_string()})),
        )
            .into_response();
    }
    if let Err(e) = c.save(&st.sup.config_path) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("{e:#}")})),
        )
            .into_response();
    }
    if let Some(u) = &st.usage {
        u.set_utc_offset_minutes(c.usage.utc_offset_minutes);
    }
    *st.sup.config.lock().unwrap_or_else(|e| e.into_inner()) = c;
    let alive = st.sup.status().state.alive();
    Json(json!({"saved": true, "restart_required": alive})).into_response()
}

async fn schema(State(st): State<AppState>) -> impl IntoResponse {
    let mut v = st.schema.as_ref().clone();
    if let Some(o) = v.as_object_mut() {
        // the on/off flags whose `false` composes as `off` (the Settings toggle's meaning)
        o.insert("on_off_flags".into(), json!(crate::compose::ON_OFF_FLAGS));
    }
    Json(v)
}

async fn command(State(st): State<AppState>) -> Response {
    match st.sup.preview() {
        Ok(c) => {
            Json(json!({"engine": c.engine, "engine_line": c.engine_line(), "spawn": c.spawn}))
                .into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

fn op_result(r: anyhow::Result<()>, st: &AppState) -> Response {
    match r {
        Ok(()) => Json(st.sup.status()).into_response(),
        Err(e) => (
            StatusCode::CONFLICT,
            Json(json!({"error": format!("{e:#}")})),
        )
            .into_response(),
    }
}

async fn start(State(st): State<AppState>) -> Response {
    let r = st.sup.start().await;
    op_result(r, &st)
}

async fn stop(State(st): State<AppState>) -> Response {
    let r = st.sup.stop().await;
    op_result(r, &st)
}

async fn restart(State(st): State<AppState>) -> Response {
    // a restart drains first, which can take minutes: answer at once and let the status poll
    // and the log tail show the progress
    let sup = st.sup.clone();
    tokio::spawn(async move {
        if let Err(e) = sup.restart().await {
            sup.logs
                .push("console", format!("saber ui: restart failed: {e:#}"));
        }
    });
    (StatusCode::ACCEPTED, Json(json!({"restarting": true}))).into_response()
}

#[derive(Deserialize)]
struct LogQuery {
    since: Option<u64>,
    limit: Option<usize>,
}

async fn logs(State(st): State<AppState>, Query(q): Query<LogQuery>) -> impl IntoResponse {
    Json(
        st.sup
            .logs
            .since(q.since.unwrap_or(0), q.limit.unwrap_or(1000).min(10_000)),
    )
}

async fn logs_stream(
    State(st): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let s = BroadcastStream::new(st.sup.logs.subscribe()).filter_map(|r| async move {
        r.ok()
            .and_then(|l| Event::default().event("log").json_data(l).ok())
            .map(Ok)
    });
    Sse::new(s).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

#[derive(Deserialize)]
struct MetricsQuery {
    seconds: Option<f64>,
    max_points: Option<usize>,
}

async fn metrics(State(st): State<AppState>, Query(q): Query<MetricsQuery>) -> impl IntoResponse {
    Json(st.metrics.window(
        q.seconds.unwrap_or(900.0),
        q.max_points.unwrap_or(2000).clamp(1, 20_000),
    ))
}

async fn metrics_latest(State(st): State<AppState>) -> impl IntoResponse {
    match st.metrics.latest() {
        Some((t, series)) => Json(json!({"t": t, "series": series})),
        None => Json(json!({"t": null, "series": {}})),
    }
}

async fn metrics_stream(
    State(st): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let s = BroadcastStream::new(st.metrics.subscribe()).filter_map(|r| async move {
        r.ok()
            .and_then(|p| Event::default().event("point").json_data(p).ok())
            .map(Ok)
    });
    Sse::new(s).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

// ----

#[derive(Deserialize)]
struct SeriesQuery {
    /// Comma-separated family-name prefixes (`saber_xtier_,hw_`); empty = every series.
    names: Option<String>,
    seconds: Option<f64>,
    max_points: Option<usize>,
}

async fn series(State(st): State<AppState>, Query(q): Query<SeriesQuery>) -> impl IntoResponse {
    let names = q.names.unwrap_or_default();
    let prefixes: Vec<&str> = names.split(',').filter(|s| !s.is_empty()).collect();
    Json(st.metrics.series_window(
        q.seconds.unwrap_or(900.0),
        q.max_points.unwrap_or(600).clamp(1, 20_000),
        &prefixes,
    ))
}

async fn hw(State(st): State<AppState>) -> impl IntoResponse {
    match st.metrics.hw_latest() {
        Some((t, h)) => Json(json!({"t": t, "gpus": h.gpus, "host": h.host})),
        None => Json(json!({"t": null, "gpus": [], "host": {}})),
    }
}

async fn logs_download(State(st): State<AppState>) -> Response {
    let mut out = String::new();
    for l in st.sup.logs.since(0, 1_000_000) {
        out.push_str(&format!("{}\t{}\t{}\n", l.t_ms, l.stream, l.text));
    }
    (
        [
            (header::CONTENT_TYPE, "text/plain; charset=utf-8"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"saber-engine.log\"",
            ),
        ],
        out,
    )
        .into_response()
}

fn no_usage() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({"error": "the usage history is off (usage.enabled = false, or the database did not open)"})),
    )
        .into_response()
}

fn internal(e: &anyhow::Error) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({"error": format!("{e:#}")})),
    )
        .into_response()
}

/// Run a blocking store read off the async workers.
#[allow(clippy::result_large_err)] // the Err is the finished HTTP reply
async fn with_store<T: Send + 'static>(
    st: &AppState,
    f: impl FnOnce(&UsageStore) -> anyhow::Result<T> + Send + 'static,
) -> Result<T, Response> {
    let Some(u) = st.usage.clone() else {
        return Err(no_usage());
    };
    match tokio::task::spawn_blocking(move || f(&u)).await {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(internal(&e)),
        Err(e) => Err(internal(&anyhow::anyhow!("{e}"))),
    }
}

/// `15m` / `24h` / `7d` / `1y` / `all` / plain seconds -> milliseconds (`all` = since 1970).
#[must_use]
pub fn parse_range_ms(r: &str, now: i64) -> Option<i64> {
    if r == "all" {
        return Some(now);
    }
    let (num, unit) = r.split_at(r.find(|c: char| !c.is_ascii_digit()).unwrap_or(r.len()));
    let n: i64 = num.parse().ok()?;
    let mult = match unit {
        "" | "s" => 1_000,
        "m" => 60_000,
        "h" => 3_600_000,
        "d" => 86_400_000,
        "w" => 7 * 86_400_000,
        "y" => 365 * 86_400_000,
        _ => return None,
    };
    n.checked_mul(mult)
}

#[derive(Deserialize)]
struct SummaryQuery {
    range: Option<String>,
    from: Option<i64>,
    to: Option<i64>,
    step_ms: Option<i64>,
}

async fn usage_summary(State(st): State<AppState>, Query(q): Query<SummaryQuery>) -> Response {
    let now = crate::now_ms() as i64;
    let to = q.to.unwrap_or(now + 1);
    let from = match (q.from, q.range.as_deref()) {
        (Some(f), _) => f,
        (None, r) => match parse_range_ms(r.unwrap_or("24h"), now) {
            Some(ms) => to - ms,
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "range: 15m, 24h, 7d, 30d, 1y, all or seconds"})),
                )
                    .into_response()
            }
        },
    };
    let step = q.step_ms;
    match with_store(&st, move |u| crate::usage::summary(u, from, to, step)).await {
        Ok(v) => Json(v).into_response(),
        Err(r) => r,
    }
}

async fn usage_requests(State(st): State<AppState>, Query(mut f): Query<RowFilter>) -> Response {
    if f.limit == 0 {
        f.limit = 100;
    }
    f.limit = f.limit.min(1000);
    let (limit, offset) = (f.limit, f.offset);
    match with_store(&st, move |u| u.rows(&f)).await {
        Ok((total, rows)) => {
            Json(json!({"total": total, "limit": limit, "offset": offset, "rows": rows}))
                .into_response()
        }
        Err(r) => r,
    }
}

async fn usage_request(
    State(st): State<AppState>,
    axum::extract::Path(uid): axum::extract::Path<String>,
) -> Response {
    match with_store(&st, move |u| u.row(&uid)).await {
        Ok(Some(r)) => Json(r).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "no such request (pruned, or never recorded)"})),
        )
            .into_response(),
        Err(r) => r,
    }
}

#[derive(Deserialize)]
struct MonthlyQuery {
    months: Option<usize>,
    days: Option<usize>,
}

async fn usage_monthly(State(st): State<AppState>, Query(q): Query<MonthlyQuery>) -> Response {
    let now = crate::now_ms() as i64;
    let months = q.months.unwrap_or(12).clamp(1, 120);
    let days = q.days.unwrap_or(371).clamp(1, 3660);
    match with_store(&st, move |u| crate::usage::monthly(u, now, months, days)).await {
        Ok(v) => Json(v).into_response(),
        Err(r) => r,
    }
}

#[derive(Deserialize)]
struct ExportQuery {
    format: Option<String>,
    #[serde(flatten)]
    filter: RowFilter,
}

async fn usage_export(State(st): State<AppState>, Query(q): Query<ExportQuery>) -> Response {
    let mut f = q.filter;
    f.limit = if f.limit == 0 {
        100_000
    } else {
        f.limit.min(100_000)
    };
    let json_out = q.format.as_deref() == Some("json");
    match with_store(&st, move |u| u.rows(&f)).await {
        Ok((_, rows)) => {
            let (ct, name, body) = if json_out {
                (
                    "application/json",
                    "saber-usage.json",
                    serde_json::to_string_pretty(&rows).unwrap_or_default(),
                )
            } else {
                (
                    "text/csv; charset=utf-8",
                    "saber-usage.csv",
                    crate::usage::csv(&rows),
                )
            };
            (
                [
                    (header::CONTENT_TYPE, ct.to_string()),
                    (
                        header::CONTENT_DISPOSITION,
                        format!("attachment; filename=\"{name}\""),
                    ),
                ],
                body,
            )
                .into_response()
        }
        Err(r) => r,
    }
}

async fn usage_status(State(st): State<AppState>) -> Response {
    let cfg = st.sup.config_now().usage;
    let Some(u) = st.usage.clone() else {
        return Json(json!({"enabled": false, "settings": cfg})).into_response();
    };
    let rows = tokio::task::spawn_blocking({
        let u = u.clone();
        move || u.row_count().ok()
    })
    .await
    .ok()
    .flatten();
    Json(json!({
        "enabled": true,
        "path": u.path.display().to_string(),
        "rows": rows,
        "writer": u.writer_stats(),
        "settings": cfg,
    }))
    .into_response()
}

async fn console_metrics(State(st): State<AppState>) -> Response {
    match with_store(&st, crate::usage::prometheus).await {
        Ok(text) => (
            [(
                header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            )],
            text,
        )
            .into_response(),
        Err(r) => r,
    }
}

// ---- the settings plane ----

fn bad(msg: impl std::fmt::Display) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error": msg.to_string()})),
    )
        .into_response()
}

/// A draft config: the whole config, or `{config, note}`.
#[derive(Deserialize)]
struct Draft {
    config: Config,
    #[serde(default)]
    note: String,
}

/// Run a blocking settings call (file reads) off the async workers.
#[allow(clippy::result_large_err)] // the Err is the finished HTTP reply
async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> T + Send + 'static,
) -> Result<T, Response> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| internal(&anyhow::anyhow!("{e}")))
}

async fn config_validate(State(st): State<AppState>, Json(d): Json<Draft>) -> Response {
    let c = st.console.clone();
    let cfg = d.config.with_credentials_of(&st.sup.config_now());
    match blocking(move || c.check(&cfg)).await {
        Ok(check) => Json(check).into_response(),
        Err(r) => r,
    }
}

#[derive(Deserialize)]
struct TpReq {
    config: Config,
    tp: u32,
    /// TP=1: the card (default: keep the config's).
    #[serde(default)]
    device: Option<u32>,
}

/// The Settings page's TP switch: the draft with `engine.tp` (and the card) set and its pack
/// flags moved to the packs cut for that TP ([`crate::catalog::with_tp`]). Nothing is saved.
async fn config_tp(State(st): State<AppState>, Json(b): Json<TpReq>) -> Response {
    let c = st.console.clone();
    let r = blocking(move || -> anyhow::Result<Config> {
        let p = b.config.engine.preset;
        if !p.tp_choices().contains(&b.tp) {
            anyhow::bail!(
                "the {p:?} preset does not serve TP={}{}",
                b.tp,
                if p == crate::config::Preset::Flashnext {
                    ": Flash-Next runs on two cards only (TP=2)"
                } else {
                    ""
                }
            );
        }
        let cat = crate::wizard::load_catalog(&c, &b.config)?;
        Ok(crate::catalog::with_tp(&b.config, b.tp, b.device, &cat))
    })
    .await;
    match r {
        Ok(Ok(cfg)) => Json(json!({"config": cfg})).into_response(),
        Ok(Err(e)) => bad(format!("{e:#}")),
        Err(r) => r,
    }
}

async fn config_apply(State(st): State<AppState>, Json(d): Json<Draft>) -> Response {
    let c = st.console.clone();
    let Draft { config, note } = d;
    // the check reads files: off the workers; the job itself runs on the runtime
    let r = blocking(move || {
        let rt = tokio::runtime::Handle::current();
        let _g = rt.enter();
        c.apply(config, crate::history::Origin::Apply, note)
    })
    .await;
    match r {
        Ok(Ok(job)) => (StatusCode::ACCEPTED, Json(job)).into_response(),
        Ok(Err(check)) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({"error": "the config does not validate", "check": check})),
        )
            .into_response(),
        Err(r) => r,
    }
}

async fn apply_status(State(st): State<AppState>) -> Response {
    Json(json!({"job": st.console.job(), "status": st.sup.status()})).into_response()
}

async fn history_list(State(st): State<AppState>) -> Response {
    let c = st.console.clone();
    match blocking(move || (c.history.list(), c.history.dir().display().to_string())).await {
        Ok((entries, dir)) => Json(json!({"dir": dir, "entries": entries})).into_response(),
        Err(r) => r,
    }
}

async fn history_get(
    State(st): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<u64>,
) -> Response {
    let c = st.console.clone();
    match blocking(move || c.history.get(id)).await {
        Ok(Ok(e)) => {
            let config = e.config.redacted();
            let toml = config.to_toml().unwrap_or_default();
            Json(json!({"meta": e.meta, "config": config, "toml": toml})).into_response()
        }
        Ok(Err(e)) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("{e:#}")})),
        )
            .into_response(),
        Err(r) => r,
    }
}

#[allow(clippy::result_large_err)] // the closure's Err is the finished HTTP reply
async fn history_rollback(
    State(st): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<u64>,
) -> Response {
    let c = st.console.clone();
    let r = blocking(move || {
        let rt = tokio::runtime::Handle::current();
        let _g = rt.enter();
        let e = c.history.get(id).map_err(|e| bad(format!("{e:#}")))?;
        c.apply(
            e.config,
            crate::history::Origin::Rollback,
            format!("rollback to #{id}"),
        )
        .map_err(|check| {
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"error": "that config no longer validates", "check": check})),
            )
                .into_response()
        })
    })
    .await;
    match r {
        Ok(Ok(job)) => (StatusCode::ACCEPTED, Json(job)).into_response(),
        Ok(Err(r)) | Err(r) => r,
    }
}

async fn profiles_list(State(st): State<AppState>) -> Response {
    let c = st.console.clone();
    match blocking(move || c.profiles.list()).await {
        Ok(Ok(user)) => Json(json!({
            "builtin": {"engine": crate::profiles::builtin("engine"), "serve": crate::profiles::builtin("serve")},
            // the 27B on one card: the same names, sized for one card's KV pool
            "builtin_tp1": {"serve": crate::profiles::builtin_for("serve", 1)},
            "user": user,
            "sticky": crate::profiles::STICKY,
        }))
        .into_response(),
        Ok(Err(e)) => internal(&e),
        Err(r) => r,
    }
}

#[derive(Deserialize)]
struct SaveProfile {
    name: String,
    #[serde(default)]
    description: String,
    config: Config,
    #[serde(default)]
    replace: bool,
}

async fn profiles_save(State(st): State<AppState>, Json(b): Json<SaveProfile>) -> Response {
    let c = st.console.clone();
    let p = crate::profiles::snapshot(&b.config, &b.name, &b.description);
    match blocking(move || c.profiles.save(p, b.replace)).await {
        Ok(Ok(())) => Json(json!({"saved": b.name})).into_response(),
        Ok(Err(e)) => bad(format!("{e:#}")),
        Err(r) => r,
    }
}

#[derive(Deserialize)]
struct Rename {
    name: String,
}

async fn profiles_rename(
    State(st): State<AppState>,
    axum::extract::Path(from): axum::extract::Path<String>,
    Json(b): Json<Rename>,
) -> Response {
    let c = st.console.clone();
    let to = b.name.clone();
    match blocking(move || c.profiles.rename(&from, &to)).await {
        Ok(Ok(())) => Json(json!({"renamed": b.name})).into_response(),
        Ok(Err(e)) => bad(format!("{e:#}")),
        Err(r) => r,
    }
}

async fn profiles_delete(
    State(st): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Response {
    let c = st.console.clone();
    match blocking(move || c.profiles.delete(&name)).await {
        Ok(Ok(())) => Json(json!({"deleted": true})).into_response(),
        Ok(Err(e)) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("{e:#}")})),
        )
            .into_response(),
        Err(r) => r,
    }
}

async fn profiles_duplicate(
    State(st): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Response {
    let c = st.console.clone();
    match blocking(move || c.profiles.duplicate(&name)).await {
        Ok(Ok(n)) => Json(json!({"name": n})).into_response(),
        Ok(Err(e)) => bad(format!("{e:#}")),
        Err(r) => r,
    }
}

async fn profiles_export(
    State(st): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Response {
    let c = st.console.clone();
    let n = name.clone();
    let p = match blocking(move || c.profiles.get(&n)).await {
        Ok(Ok(Some(p))) => p,
        Ok(Ok(None)) => {
            // a built-in exports too (as a starting point for a user profile)
            match ["engine", "serve"]
                .iter()
                .flat_map(|s| crate::profiles::builtin(s))
                .find(|p| p.name == name)
            {
                Some(p) => p,
                None => {
                    return (
                        StatusCode::NOT_FOUND,
                        Json(json!({"error": "no such profile"})),
                    )
                        .into_response()
                }
            }
        }
        Ok(Err(e)) => return internal(&e),
        Err(r) => return r,
    };
    match crate::profiles::export_toml(&p) {
        Ok(t) => {
            let file: String = p
                .name
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '-' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();
            (
                [
                    (
                        header::CONTENT_TYPE,
                        "application/toml; charset=utf-8".to_string(),
                    ),
                    (
                        header::CONTENT_DISPOSITION,
                        format!("attachment; filename=\"saber-profile-{file}.toml\""),
                    ),
                ],
                t,
            )
                .into_response()
        }
        Err(e) => internal(&e),
    }
}

async fn profiles_import(State(st): State<AppState>, body: String) -> Response {
    let c = st.console.clone();
    match blocking(move || c.profiles.import(&body)).await {
        Ok(Ok(names)) => Json(json!({"imported": names})).into_response(),
        Ok(Err(e)) => bad(format!("{e:#}")),
        Err(r) => r,
    }
}

#[derive(Deserialize)]
struct ComposeReq {
    config: Config,
    /// A built-in name (with `builtin: true`) or a user profile's name.
    name: String,
    #[serde(default)]
    builtin: bool,
}

async fn profiles_compose(State(st): State<AppState>, Json(b): Json<ComposeReq>) -> Response {
    let c = st.console.clone();
    let r = blocking(move || -> anyhow::Result<Config> {
        let sec = b.config.engine.preset.subcommand();
        let p = if b.builtin {
            crate::profiles::builtin(sec)
                .into_iter()
                .find(|p| p.name == b.name)
                .ok_or_else(|| {
                    anyhow::anyhow!("no built-in profile `{}` for `saber {sec}`", b.name)
                })?
        } else {
            c.profiles
                .get(&b.name)?
                .ok_or_else(|| anyhow::anyhow!("no profile named `{}`", b.name))?
        };
        crate::profiles::apply_profile(&b.config, &p)
    })
    .await;
    match r {
        Ok(Ok(cfg)) => Json(json!({"config": cfg})).into_response(),
        Ok(Err(e)) => bad(format!("{e:#}")),
        Err(r) => r,
    }
}

async fn packs(State(st): State<AppState>) -> Response {
    let c = st.console.clone();
    let cfg = st.sup.config_now();
    match blocking(move || c.packs(&cfg)).await {
        Ok((dir, rows)) => {
            Json(json!({"dir": dir.display().to_string(), "packs": rows})).into_response()
        }
        Err(r) => r,
    }
}

// ---- credentials and the audit log ----

/// Change the stored config's credentials (validated, sealed, saved, swapped in memory).
#[allow(clippy::result_large_err)] // the Err is the finished HTTP reply
async fn update_credentials(
    st: &AppState,
    f: impl FnOnce(&mut Config) -> Result<(), (StatusCode, String)> + Send + 'static,
) -> Result<Config, Response> {
    let sup = st.sup.clone();
    let r = blocking(move || {
        let mut cfg = sup.config_now();
        f(&mut cfg)?;
        cfg.seal()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
        cfg.validate()
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
        cfg.save(&sup.config_path)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
        sup.set_config(cfg.clone());
        Ok::<_, (StatusCode, String)>(cfg)
    })
    .await?;
    r.map_err(|(code, m)| (code, Json(json!({"error": m}))).into_response())
}

fn key_view(k: &crate::config::ProxyKey) -> serde_json::Value {
    json!({"label": k.label, "created_ms": k.created_ms, "revoked": k.revoked})
}

async fn auth_info(State(st): State<AppState>, who: Option<axum::Extension<Who>>) -> Response {
    let cfg = st.sup.config_now();
    let admin = st.auth.admin();
    Json(json!({
        "who": who.map(|w| w.0 .0),
        "required": admin.is_set(),
        "source": admin.source(),
        "loopback": st.loopback,
        "open_lan": st.open_lan,
        "rate_limit": {
            "max_fails": crate::auth::Limiter::MAX_FAILS,
            "window_s": crate::auth::Limiter::WINDOW.as_secs(),
            "lockout_s": crate::auth::Limiter::LOCKOUT.as_secs(),
        },
        "keys": cfg.proxy.keys.iter().map(key_view).collect::<Vec<_>>(),
        "proxy_open": cfg.proxy.keys.is_empty() && !admin.is_set(),
        "engine_key_set": st.sup.engine_bearer().is_some(),
        "audit_path": st.console.audit.path().display().to_string(),
    }))
    .into_response()
}

/// Rotate (or set) the console's admin token: the new token is in this reply ONLY; the config
/// keeps its salted hash; the running console checks it from now on.
async fn admin_token_rotate(State(st): State<AppState>) -> Response {
    if st.auth.admin().source() == "flag" {
        return (
            StatusCode::CONFLICT,
            Json(json!({"error": "this console's token came from --admin-token: change it there (the flag wins over the config)"})),
        )
            .into_response();
    }
    let token = match crate::setup::random_key("sa-admin-") {
        Ok(t) => t,
        Err(e) => return internal(&anyhow::anyhow!("{e}")),
    };
    let t = token.clone();
    match update_credentials(&st, move |c| {
        c.console.admin_token = t;
        c.console.admin_token_hash.clear();
        Ok(())
    })
    .await
    {
        Ok(cfg) => {
            st.auth.set_admin(cfg.admin());
            Json(json!({"token": token, "note": "shown once: the console keeps only its hash"}))
                .into_response()
        }
        Err(r) => r,
    }
}

async fn admin_token_remove(State(st): State<AppState>) -> Response {
    if !st.loopback {
        return (
            StatusCode::CONFLICT,
            Json(json!({"error": "this console listens beyond loopback: it needs an admin token"})),
        )
            .into_response();
    }
    if st.auth.admin().source() == "flag" {
        return (
            StatusCode::CONFLICT,
            Json(json!({"error": "this console's token came from --admin-token"})),
        )
            .into_response();
    }
    match update_credentials(&st, |c| {
        c.console.admin_token.clear();
        c.console.admin_token_hash.clear();
        if !c.console.host.is_empty() && !crate::is_loopback(&c.console.host) {
            return Err((
                StatusCode::CONFLICT,
                format!(
                    "console.host is {}: the next start needs a token; set console.host to \
                     127.0.0.1 first",
                    c.console.host
                ),
            ));
        }
        Ok(())
    })
    .await
    {
        Ok(cfg) => {
            st.auth.set_admin(cfg.admin());
            Json(json!({"removed": true})).into_response()
        }
        Err(r) => r,
    }
}

async fn keys_list(State(st): State<AppState>) -> Response {
    let cfg = st.sup.config_now();
    Json(json!({"keys": cfg.proxy.keys.iter().map(key_view).collect::<Vec<_>>()})).into_response()
}

#[derive(Deserialize)]
struct NewKey {
    label: String,
}

/// A new `/v1` key for `label`: the key is in this reply ONLY.
async fn keys_create(State(st): State<AppState>, Json(b): Json<NewKey>) -> Response {
    let label = b.label.trim().to_string();
    if label.is_empty()
        || label.len() > 64
        || !label
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        return bad("a key label is 1-64 characters of A-Z a-z 0-9 - _ .");
    }
    let key = match crate::setup::random_key("sk-saber-") {
        Ok(k) => k,
        Err(e) => return internal(&anyhow::anyhow!("{e}")),
    };
    let (k2, l2) = (key.clone(), label.clone());
    match update_credentials(&st, move |c| {
        if c.proxy.keys.iter().any(|k| k.label == l2) {
            return Err((
                StatusCode::CONFLICT,
                format!("a key labelled `{l2}` exists: revoke and delete it first"),
            ));
        }
        let mut k = crate::config::ProxyKey::new(&l2, &k2);
        k.created_ms = crate::now_ms();
        c.proxy.keys.push(k);
        Ok(())
    })
    .await
    {
        Ok(_) => (
            [(AUDIT_DETAIL, format!("label={label}"))],
            Json(json!({"label": label, "key": key, "note": "shown once: the console keeps only its hash"})),
        )
            .into_response(),
        Err(r) => r,
    }
}

async fn keys_revoke(
    State(st): State<AppState>,
    axum::extract::Path(label): axum::extract::Path<String>,
) -> Response {
    let l = label.clone();
    match update_credentials(&st, move |c| {
        let k = c
            .proxy
            .keys
            .iter_mut()
            .find(|k| k.label == l)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("no key labelled `{l}`")))?;
        k.revoked = true;
        Ok(())
    })
    .await
    {
        Ok(_) => (
            [(AUDIT_DETAIL, format!("label={label}"))],
            Json(json!({"revoked": label})),
        )
            .into_response(),
        Err(r) => r,
    }
}

async fn keys_delete(
    State(st): State<AppState>,
    axum::extract::Path(label): axum::extract::Path<String>,
) -> Response {
    let l = label.clone();
    match update_credentials(&st, move |c| {
        let n = c.proxy.keys.len();
        c.proxy.keys.retain(|k| k.label != l);
        if c.proxy.keys.len() == n {
            return Err((StatusCode::NOT_FOUND, format!("no key labelled `{l}`")));
        }
        Ok(())
    })
    .await
    {
        Ok(_) => (
            [(AUDIT_DETAIL, format!("label={label}"))],
            Json(json!({"deleted": label})),
        )
            .into_response(),
        Err(r) => r,
    }
}

#[derive(Deserialize)]
struct AuditQuery {
    limit: Option<usize>,
}

async fn audit_list(State(st): State<AppState>, Query(q): Query<AuditQuery>) -> Response {
    let c = st.console.clone();
    let limit = q.limit.unwrap_or(200).clamp(1, 5000);
    match blocking(move || c.audit.recent(limit)).await {
        Ok(entries) => Json(json!({"entries": entries})).into_response(),
        Err(r) => r,
    }
}

async fn static_file(
    State(st): State<AppState>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
) -> Response {
    let path = uri.path().trim_start_matches('/');
    if path.starts_with("api/") {
        // an unknown /api path tells an unauthenticated caller nothing
        let ok = !st.auth.required()
            || session_cookie(&headers).is_some_and(|v| st.auth.session_check(&v, crate::now_ms()))
            || headers
                .get(header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
                .is_some_and(|t| st.auth.check_admin(t.trim()));
        if !ok {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({"error": "admin token required"})),
            )
                .into_response();
        }
    }
    if method != Method::GET && method != Method::HEAD {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }
    if path.starts_with("api/") {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "no such endpoint"})),
        )
            .into_response();
    }
    let (file, name) = match Assets::get(if path.is_empty() { "index.html" } else { path }) {
        Some(f) => (f, if path.is_empty() { "index.html" } else { path }),
        // the SPA's client-side routes
        None => match Assets::get("index.html") {
            Some(f) => (f, "index.html"),
            None => {
                return (
                    StatusCode::NOT_FOUND,
                    "the UI was not built into this binary",
                )
                    .into_response()
            }
        },
    };
    let etag = format!("\"{}\"", hex_prefix(&file.metadata.sha256_hash()));
    if headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v == etag)
    {
        return StatusCode::NOT_MODIFIED.into_response();
    }
    let mime = file.metadata.mimetype().to_string();
    // hashed asset names are immutable; index.html always revalidates
    let cache = if name.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    };
    Response::builder()
        .header(header::CONTENT_TYPE, mime)
        .header(header::ETAG, etag)
        .header(header::CACHE_CONTROL, cache)
        .header("x-content-type-options", "nosniff")
        .body(Body::from(file.data.into_owned()))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

fn hex_prefix(b: &[u8]) -> String {
    b.iter().take(8).map(|x| format!("{x:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_open_console_serves_private_peers_and_local_names_only() {
        use std::net::IpAddr;
        for ip in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.9",
            "192.168.12.36",
            "169.254.1.1",
            "::1",
            "fd00::5",
            "fe80::1",
            "::ffff:192.168.1.5",
        ] {
            assert!(is_private_ip(ip.parse::<IpAddr>().unwrap()), "{ip}");
        }
        for ip in ["8.8.8.8", "93.184.216.34", "172.32.0.1", "2606:4700::1111"] {
            assert!(!is_private_ip(ip.parse::<IpAddr>().unwrap()), "{ip}");
        }
        for h in [
            "192.168.12.47:8090",
            "localhost:8090",
            "[::1]:8090",
            "myserver:8090",
            "box.local",
            "nas.lan:8090",
            "x.home.arpa",
        ] {
            assert!(host_is_lan(h), "{h}");
        }
        for h in [
            "evil.example.com",
            "evil.example.com:8090",
            "rebind.attacker.net",
        ] {
            assert!(!host_is_lan(h), "{h}");
        }
    }

    #[test]
    fn loopback_hosts() {
        for h in [
            "localhost:8090",
            "127.0.0.1:8090",
            "[::1]:8090",
            "localhost",
        ] {
            assert!(host_is_loopback(h), "{h}");
        }
        for h in ["evil.example:8090", "192.168.1.5:8090", "[fe80::1]:80"] {
            assert!(!host_is_loopback(h), "{h}");
        }
    }

    #[test]
    fn token_query() {
        let u: Uri = "/api/logs/stream?x=1&token=s3cret".parse().unwrap();
        assert_eq!(query_token(&u).as_deref(), Some("s3cret"));
        let u: Uri = "/api/logs/stream?token=a%2Db_c%7E".parse().unwrap();
        assert_eq!(query_token(&u).as_deref(), Some("a-b_c~"));
    }

    #[test]
    fn ranges() {
        assert_eq!(parse_range_ms("24h", 0), Some(86_400_000));
        assert_eq!(parse_range_ms("7d", 0), Some(7 * 86_400_000));
        assert_eq!(parse_range_ms("90", 0), Some(90_000));
        assert_eq!(parse_range_ms("all", 5), Some(5));
        assert_eq!(parse_range_ms("x", 0), None);
        assert_eq!(parse_range_ms("3q", 0), None);
    }
}
