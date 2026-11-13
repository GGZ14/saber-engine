//! The setup wizard's HTTP routes: the system check (`/api/system`), the model catalog with fit and
//! install state (`/api/catalog`), the download manager (`/api/downloads*`), pack delete, and
//! the setup wizard (`/api/setup*`). Mounted by [`crate::http::router`] behind the same admin
//! auth and mutating-call header rules as every other `/api` route.

use std::convert::Infallible;
use std::path::PathBuf;
use std::time::Duration;

use axum::extract::{Path as UrlPath, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::Stream;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_stream::wrappers::BroadcastStream;

use crate::apply::Console;
use crate::catalog::{self, Catalog, Model, Roots};
use crate::config::{Config, FlagValue, ProxyKey};
use crate::downloads::{self, Spec};
use crate::http::AppState;
use crate::setup;
use crate::system::{self, ProbeInput};

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/system", get(system_check))
        .route("/api/catalog", get(catalog_view))
        .route("/api/downloads", get(dl_list).post(dl_start))
        .route("/api/downloads/stream", get(dl_stream))
        .route("/api/downloads/clear", post(dl_clear))
        .route("/api/downloads/{id}/pause", post(dl_pause))
        .route("/api/downloads/{id}/resume", post(dl_resume))
        .route("/api/downloads/{id}/cancel", post(dl_cancel))
        .route("/api/packs/{name}", axum::routing::delete(pack_delete))
        .route("/api/setup", get(setup_get))
        .route("/api/setup/goto", post(setup_goto))
        .route("/api/setup/model", post(setup_model))
        .route("/api/setup/tp", post(setup_tp))
        .route("/api/setup/profile", post(setup_profile))
        .route("/api/setup/network", post(setup_network))
        .route("/api/setup/keygen", post(setup_keygen))
        .route("/api/setup/launch", post(setup_launch))
        .route("/api/setup/finish", post(setup_finish))
        .route("/api/setup/skip", post(setup_skip))
        .route("/api/setup/rerun", post(setup_rerun))
}

fn err(code: StatusCode, msg: impl std::fmt::Display) -> Response {
    (code, Json(json!({"error": msg.to_string()}))).into_response()
}

#[allow(clippy::result_large_err)] // the Err is the finished HTTP reply
async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> T + Send + 'static,
) -> Result<T, Response> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))
}

/// The catalog the config names (or the embedded one).
pub fn load_catalog(c: &Console, cfg: &Config) -> anyhow::Result<Catalog> {
    let p = &cfg.downloads.catalog;
    if p.as_os_str().is_empty() {
        Ok(Catalog::embedded())
    } else {
        Catalog::load(&crate::expand_home(p, c.sup.home()))
    }
}

fn roots(c: &Console, cfg: &Config) -> Roots {
    Roots::for_config(cfg, c.sup.home())
}

/// The KV volume: the `--fn-kv-dir` override, else `<nvme>/saber-kv`.
fn kv_dir(c: &Console, cfg: &Config) -> PathBuf {
    match cfg.engine.flags.get("--fn-kv-dir") {
        Some(FlagValue::Str(s)) if !s.is_empty() => PathBuf::from(s),
        _ => cfg
            .engine
            .paths
            .resolved(c.sup.home())
            .nvme
            .join("saber-kv"),
    }
}

/// Bytes `model` still has to download (its components' sizes, the installed ones excluded).
fn remaining_bytes(m: &Model, r: &Roots) -> u64 {
    m.components
        .iter()
        .filter(|x| x.installed_at(r).is_none())
        .map(|x| x.approx_bytes)
        .sum()
}

/// The supported (gfx12, not integrated) GPUs in this machine's KFD topology.
fn usable_gpus(c: &Console) -> u32 {
    system::probe_gpus(&c.sys_root)
        .iter()
        .filter(|g| g.supported && !g.integrated)
        .count() as u32
}

/// The TP `m` runs at for this config: the Model step's pick when the model is hosted at it,
/// else the machine's recommendation (TP=2 with two cards, TP=1 with one).
pub fn chosen_tp(cfg: &Config, m: &Model, gpus: u32) -> u32 {
    let choices = m.tp_choices();
    if m.preset == crate::config::Preset::Mock {
        return 2;
    }
    if choices.contains(&cfg.setup.tp) {
        return cfg.setup.tp;
    }
    // a config already set up at a TP (the Settings page, a seeded file) keeps it
    if cfg.setup.tp == 0 && cfg.engine.preset == m.preset && choices.contains(&cfg.engine.tp) {
        return cfg.engine.tp;
    }
    m.recommended_tp(gpus)
}

/// The catalog model the wizard picked, in its form at the chosen TP.
fn picked(c: &Console, cfg: &Config) -> Option<(Model, u32)> {
    let cat = load_catalog(c, cfg).ok()?;
    let m = cat.model(&cfg.setup.model)?;
    let tp = chosen_tp(cfg, m, usable_gpus(c));
    Some((m.at_tp(tp), tp))
}

fn host_facts(c: &Console, cfg: &Config) -> system::HostFacts {
    if let Some(h) = &c.host_facts {
        return h.clone();
    }
    let e = &cfg.engine;
    system::gather_host(
        &e.container.runtime,
        &e.container.image,
        &c.sup.home().join("saber/.cache"),
        e.port,
        c.sup.status().state.alive(),
    )
}

fn probe_for(c: &Console, cfg: &Config, with_commands: bool) -> system::SystemReport {
    let r = roots(c, cfg);
    let model = picked(c, cfg).map(|(m, _)| m);
    system::probe(&ProbeInput {
        root: c.sys_root.clone(),
        packs_dir: r.packs.clone(),
        kv_dir: kv_dir(c, cfg),
        need_disk_bytes: model.as_ref().map(|m| remaining_bytes(m, &r)),
        needs_gpus: model.as_ref().is_none_or(|m| m.preset.uses_gpus()),
        // the picked form's cards (a TP=1 pick needs one); no pick: the hosted models' two
        min_gpus: model.as_ref().map_or(2, |m| m.fit.min_gpus.max(1)),
        host: if with_commands {
            host_facts(c, cfg)
        } else {
            system::HostFacts::default()
        },
    })
}

async fn system_check(State(st): State<AppState>) -> Response {
    let c = st.console.clone();
    let cfg = st.sup.config_now();
    match blocking(move || probe_for(&c, &cfg, true)).await {
        Ok(r) => Json(r).into_response(),
        Err(r) => r,
    }
}

/// The catalog with, per model: what is installed, the fit verdict, what is left to download.
fn catalog_json(c: &Console, cfg: &Config) -> anyhow::Result<Value> {
    let cat = load_catalog(c, cfg)?;
    let r = roots(c, cfg);
    let sys = probe_for(c, cfg, false);
    let jobs = c.downloads.list();
    let gpus = usable_gpus(c);
    let free = system::disk_space(&r.packs).1;
    let models: Vec<Value> = cat
        .models
        .iter()
        .map(|entry| {
            let tp = chosen_tp(cfg, entry, gpus);
            let m = &entry.at_tp(tp);
            // every TP the model is hosted at, with its own fit and download (the Model step's
            // TP choice)
            let forms: Vec<Value> = entry
                .tp_choices()
                .iter()
                .rev()
                .map(|&t| {
                    let f = entry.at_tp(t);
                    let need = remaining_bytes(&f, &r);
                    json!({
                        "tp": t,
                        "cards": t,
                        "note": f.note,
                        "download_gb": f.download_gb,
                        "fit": catalog::fit(&f.fit, &sys, free, need),
                        "installed": f.components.iter().all(|x| x.installed_at(&r).is_some()),
                        "remaining_bytes": need,
                        "possible": entry.preset == crate::config::Preset::Mock || t <= gpus,
                    })
                })
                .collect();
            let comps: Vec<Value> = m
                .components
                .iter()
                .map(|x| {
                    let at = x.installed_at(&r);
                    let target = x.target(&r).display().to_string();
                    let staging = x.staging(&r, &m.id);
                    let job = jobs
                        .iter()
                        .rev()
                        .find(|j| j.model == m.id && j.component == x.id);
                    json!({
                        "id": x.id,
                        "label": x.label,
                        "dest": x.dest,
                        "target": target,
                        "installed": at.is_some(),
                        "installed_at": at.map(|p| p.display().to_string()),
                        "approx_bytes": x.approx_bytes,
                        "partial": staging.is_dir(),
                        "source": if x.source.hf_repo.is_empty() { "url".to_string() } else { format!("huggingface.co/{}", x.source.hf_repo) },
                        "job": job,
                    })
                })
                .collect();
            let need = remaining_bytes(m, &r);
            let fit = catalog::fit(&m.fit, &sys, free, need);
            let installed = m.components.iter().all(|x| x.installed_at(&r).is_some());
            json!({
                "id": m.id,
                "name": m.name,
                "title": m.title,
                "kind": m.kind,
                "preset": m.preset,
                "description": m.description,
                "note": m.note,
                "download_gb": m.download_gb,
                "fit_rules": m.fit,
                "fit": fit,
                "installed": installed,
                "remaining_bytes": need,
                "components": comps,
                "tp": tp,
                "tp_choices": entry.tp_choices(),
                "recommended_tp": entry.recommended_tp(gpus),
                "tp_forms": forms,
            })
        })
        .collect();
    Ok(json!({
        "gpus": gpus,
        "device": cfg.engine.device,
        "source": if cfg.downloads.catalog.as_os_str().is_empty() { "built in".to_string() } else { cfg.downloads.catalog.display().to_string() },
        "roots": r,
        "free_bytes": system::disk_space(&r.packs).1,
        "models": models,
    }))
}

async fn catalog_view(State(st): State<AppState>) -> Response {
    let c = st.console.clone();
    let cfg = st.sup.config_now();
    match blocking(move || catalog_json(&c, &cfg)).await {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        Err(r) => r,
    }
}

// ---- downloads ----

#[derive(Deserialize)]
struct StartReq {
    model: String,
    /// One component (default: every component not installed yet).
    #[serde(default)]
    component: Option<String>,
    /// The form to download (default: the TP the wizard picked, else the recommendation).
    #[serde(default)]
    tp: Option<u32>,
}

fn specs_for(c: &Console, cfg: &Config, req: &StartReq) -> anyhow::Result<Vec<Spec>> {
    let cat = load_catalog(c, cfg)?;
    let entry = cat
        .model(&req.model)
        .ok_or_else(|| anyhow::anyhow!("no model `{}` in the catalog", req.model))?;
    let tp = req
        .tp
        .unwrap_or_else(|| chosen_tp(cfg, entry, usable_gpus(c)));
    if !entry.tp_choices().contains(&tp) {
        anyhow::bail!("{} is not hosted for TP={tp}", entry.name);
    }
    let m = &entry.at_tp(tp);
    let r = roots(c, cfg);
    let home = c.sup.home();
    let token = if cfg.downloads.hf_token_file.as_os_str().is_empty() {
        None
    } else {
        let p = crate::expand_home(&cfg.downloads.hf_token_file, home);
        Some(
            std::fs::read_to_string(&p)
                .map_err(|e| anyhow::anyhow!("reading the token file {}: {e}", p.display()))?
                .trim()
                .to_string(),
        )
    };
    let ep = if cfg.downloads.hf_endpoint.is_empty() {
        cat.hf_endpoint.clone()
    } else {
        cfg.downloads.hf_endpoint.clone()
    };
    let mut out = Vec::new();
    for x in &m.components {
        if req.component.as_deref().is_some_and(|w| w != x.id) {
            continue;
        }
        if x.installed_at(&r).is_some() {
            continue;
        }
        out.push(Spec {
            model: m.id.clone(),
            model_name: m.name.clone(),
            component: x.clone(),
            roots: r.clone(),
            hf_endpoint: ep.clone(),
            token: token.clone(),
            curl: crate::expand_home(&cfg.downloads.curl, home),
            reserve_bytes: cfg.downloads.reserve_bytes,
        });
    }
    if let Some(w) = &req.component {
        if out.is_empty() && !m.components.iter().any(|x| &x.id == w) {
            anyhow::bail!("model `{}` has no component `{w}`", m.id);
        }
    }
    Ok(out)
}

async fn dl_start(State(st): State<AppState>, Json(req): Json<StartReq>) -> Response {
    let c = st.console.clone();
    let cfg = st.sup.config_now();
    let r = blocking(move || -> anyhow::Result<Vec<downloads::Job>> {
        let specs = specs_for(&c, &cfg, &req)?;
        Ok(specs.into_iter().map(|s| c.downloads.enqueue(s)).collect())
    })
    .await;
    match r {
        Ok(Ok(jobs)) if jobs.is_empty() => {
            Json(json!({"jobs": [], "note": "everything is already installed"})).into_response()
        }
        Ok(Ok(jobs)) => (StatusCode::ACCEPTED, Json(json!({"jobs": jobs}))).into_response(),
        Ok(Err(e)) => err(StatusCode::BAD_REQUEST, format!("{e:#}")),
        Err(r) => r,
    }
}

async fn dl_list(State(st): State<AppState>) -> Response {
    Json(json!({"jobs": st.console.downloads.list()})).into_response()
}

async fn dl_stream(
    State(st): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let first = st.console.downloads.list();
    let head = futures_util::stream::once(async move {
        Ok::<_, Infallible>(
            Event::default()
                .event("downloads")
                .json_data(first)
                .unwrap_or_default(),
        )
    });
    let s = BroadcastStream::new(st.console.downloads.subscribe()).filter_map(|r| async move {
        r.ok()
            .and_then(|l| Event::default().event("downloads").json_data(l).ok())
            .map(Ok)
    });
    Sse::new(head.chain(s)).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

fn job_result(r: anyhow::Result<downloads::Job>) -> Response {
    match r {
        Ok(j) => Json(j).into_response(),
        Err(e) => err(StatusCode::CONFLICT, format!("{e:#}")),
    }
}

async fn dl_pause(State(st): State<AppState>, UrlPath(id): UrlPath<u64>) -> Response {
    job_result(st.console.downloads.pause(id))
}
async fn dl_resume(State(st): State<AppState>, UrlPath(id): UrlPath<u64>) -> Response {
    // a download restored after a console restart carries no token: read the file again
    let cfg = st.sup.config_now();
    if !cfg.downloads.hf_token_file.as_os_str().is_empty() {
        let p = crate::expand_home(&cfg.downloads.hf_token_file, st.console.sup.home());
        if let Ok(t) = std::fs::read_to_string(&p) {
            st.console
                .downloads
                .set_token(id, Some(t.trim().to_string()));
        }
    }
    job_result(st.console.downloads.resume(id))
}
async fn dl_cancel(State(st): State<AppState>, UrlPath(id): UrlPath<u64>) -> Response {
    job_result(st.console.downloads.cancel(id))
}
async fn dl_clear(State(st): State<AppState>) -> Response {
    st.console.downloads.clear_finished();
    Json(json!({"jobs": st.console.downloads.list()})).into_response()
}

#[derive(Deserialize)]
struct DeleteReq {
    /// Must repeat the pack's name (the confirmation guard).
    #[serde(default)]
    confirm: String,
}

/// Why `name` may not be deleted (`None` = it may).
fn delete_refusal(
    c: &Console,
    cfg: &Config,
    name: &str,
    confirm: &str,
) -> Option<(StatusCode, String)> {
    if !catalog::safe_name(name) {
        return Some((
            StatusCode::BAD_REQUEST,
            format!("`{name}` is not a pack name"),
        ));
    }
    if confirm != name {
        return Some((
            StatusCode::BAD_REQUEST,
            format!("type the pack's name (`{name}`) to confirm the delete"),
        ));
    }
    let dir = c.packs_dir(cfg).join(name);
    match std::fs::symlink_metadata(&dir) {
        Err(_) => {
            return Some((
                StatusCode::NOT_FOUND,
                format!("no pack `{name}` in {}", c.packs_dir(cfg).display()),
            ))
        }
        Ok(m) if m.file_type().is_symlink() || !m.is_dir() => {
            return Some((
                StatusCode::BAD_REQUEST,
                format!(
                    "{} is not a pack directory (a link or a file): remove it by hand",
                    dir.display()
                ),
            ))
        }
        Ok(_) => {}
    }
    let (_, rows) = c.packs(cfg);
    if let Some(row) = rows.iter().find(|p| p.info.name == name) {
        if let Some(role) = row.active_as {
            return Some((
                StatusCode::CONFLICT,
                format!(
                    "`{name}` is the {role} pack the engine is configured to serve. Switch to \
                     another pack (and apply) before deleting it."
                ),
            ));
        }
    }
    let status = c.sup.status();
    if status.state.alive() && status.command.iter().any(|a| a.contains(name)) {
        return Some((
            StatusCode::CONFLICT,
            format!("the running engine uses `{name}`: stop it first"),
        ));
    }
    if c.downloads.busy_with(&dir) {
        return Some((
            StatusCode::CONFLICT,
            format!("a download into `{name}` is in progress: cancel it first"),
        ));
    }
    None
}

async fn pack_delete(
    State(st): State<AppState>,
    UrlPath(name): UrlPath<String>,
    body: Option<Json<DeleteReq>>,
) -> Response {
    let c = st.console.clone();
    let cfg = st.sup.config_now();
    let confirm = body.map(|b| b.0.confirm).unwrap_or_default();
    let r = blocking(move || {
        if let Some(r) = delete_refusal(&c, &cfg, &name, &confirm) {
            return Err(r);
        }
        let dir = c.packs_dir(&cfg).join(&name);
        let bytes = crate::estimate::read_pack(&dir).size_bytes;
        std::fs::remove_dir_all(&dir)
            .map(|()| (name, bytes))
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("deleting {}: {e}", dir.display()),
                )
            })
    })
    .await;
    match r {
        Ok(Ok((name, bytes))) => {
            st.sup.logs.push(
                "console",
                format!(
                    "saber ui: deleted the pack {name} ({})",
                    downloads::human(bytes)
                ),
            );
            Json(json!({"deleted": name, "freed_bytes": bytes})).into_response()
        }
        Ok(Err((code, m))) => err(code, m),
        Err(r) => r,
    }
}

// ---- the wizard ----

/// The gates' inputs for `cfg`.
pub fn facts(c: &Console, cfg: &Config) -> setup::Facts {
    let form = picked(c, cfg);
    let m = form.as_ref().map(|(m, _)| m);
    let tp = form.as_ref().map_or(cfg.engine.tp, |(_, t)| *t);
    let gpus = usable_gpus(c);
    let r = roots(c, cfg);
    let mut config_errors: Vec<String> = Vec::new();
    if let Err(e) = cfg.validate() {
        config_errors.push(e.to_string());
    } else if m.is_some() {
        let check = c.check(cfg);
        config_errors.extend(
            check
                .report
                .errors
                .iter()
                .map(|i| format!("{}: {}", i.field, i.message)),
        );
    }
    setup::Facts {
        model_known: m.is_some(),
        model_installed: m
            .is_some_and(|m| m.components.iter().all(|x| x.installed_at(&r).is_some())),
        profile_known: crate::profiles::builtin_for(cfg.engine.preset.subcommand(), tp)
            .iter()
            .any(|p| p.name == cfg.setup.profile),
        config_errors,
        engine_ready: c.sup.status().state == crate::supervisor::EngineState::Ready,
        tp,
        gpus,
        tp_unfit: m.is_some_and(|m| m.preset.uses_gpus() && tp > gpus),
    }
}

fn setup_json(c: &Console, cfg: &Config) -> Value {
    let f = facts(c, cfg);
    json!({
        "needed": !cfg.setup.completed,
        "config_exists": c.sup.config_path.exists(),
        "config_path": c.sup.config_path.display().to_string(),
        "state": cfg.setup,
        "steps": setup::view(&cfg.setup, &f),
        "facts": f,
        "blocker": setup::blocker(&cfg.setup.step, &cfg.setup, &f),
        "network": {
            "engine_port": cfg.engine.port,
            "console_host": cfg.console.host,
            "console_port": cfg.console.port,
            "admin_token_set": !cfg.console.admin_token.is_empty() || !cfg.console.admin_token_hash.is_empty(),
            // A stored key is a hash; its text was shown once, when it was made
            "keys": cfg.proxy.keys.iter().map(|k| json!({"label": k.label, "set": true, "revoked": k.revoked, "created_ms": k.created_ms})).collect::<Vec<_>>(),
            "running_console_port": c.console_port,
        },
    })
}

/// Validate, save the file, swap the in-memory config.
fn save(c: &Console, cfg: Config) -> Result<(), (StatusCode, String)> {
    cfg.validate()
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    cfg.save(&c.sup.config_path)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
    c.sup.set_config(cfg);
    Ok(())
}

/// Run `f` over the current config off the workers; save what it returns; answer the setup
/// view (or the error).
async fn mutate(
    st: &AppState,
    f: impl FnOnce(&Console, Config) -> Result<Config, (StatusCode, String)> + Send + 'static,
) -> Response {
    let c = st.console.clone();
    let cfg = st.sup.config_now();
    let r = blocking(move || {
        let next = f(&c, cfg)?;
        save(&c, next.clone())?;
        Ok::<_, (StatusCode, String)>(setup_json(&c, &next))
    })
    .await;
    match r {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err((code, m))) => err(code, m),
        Err(r) => r,
    }
}

async fn setup_get(State(st): State<AppState>) -> Response {
    let c = st.console.clone();
    let cfg = st.sup.config_now();
    match blocking(move || setup_json(&c, &cfg)).await {
        Ok(v) => Json(v).into_response(),
        Err(r) => r,
    }
}

#[derive(Deserialize)]
struct GotoReq {
    step: String,
}

async fn setup_goto(State(st): State<AppState>, Json(b): Json<GotoReq>) -> Response {
    mutate(&st, move |c, mut cfg| {
        let f = facts(c, &cfg);
        let next = setup::goto(&cfg.setup, &b.step, &f).map_err(|e| (StatusCode::CONFLICT, e))?;
        // leaving the Model step forward: serve what is installed now (a download may have
        // finished since the pick)
        let from = setup::index(&cfg.setup.step).unwrap_or(0);
        let model_ix = setup::index("model").unwrap_or(1);
        if from <= model_ix && setup::index(&next.step).unwrap_or(0) > model_ix {
            let cat = load_catalog(c, &cfg)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
            if let Some(m) = cat.model(&cfg.setup.model) {
                let tp = chosen_tp(&cfg, m, usable_gpus(c));
                cfg = catalog::config_for_tp(&cfg, m, tp, &roots(c, &cfg))
                    .map_err(|e| (StatusCode::CONFLICT, format!("{e:#}")))?;
            }
        }
        cfg.setup = next;
        Ok(cfg)
    })
    .await
}

#[derive(Deserialize)]
struct ModelReq {
    id: String,
    /// The TP to run it at (default: the one picked before when the model is hosted at it,
    /// else the machine's recommendation).
    #[serde(default)]
    tp: Option<u32>,
}

/// Compose the engine config for the picked model at `tp` when its form is installed (a form
/// still downloading is composed when the step is left).
fn compose_pick(
    c: &Console,
    mut next: Config,
    m: &Model,
    tp: u32,
) -> Result<Config, (StatusCode, String)> {
    let r = roots(c, &next);
    let form = m.at_tp(tp);
    if form.components.iter().all(|x| x.installed_at(&r).is_some()) {
        let mut composed = catalog::config_for_tp(&next, m, tp, &r)
            .map_err(|e| (StatusCode::CONFLICT, format!("{e:#}")))?;
        composed.setup = next.setup.clone();
        next = composed;
    }
    Ok(next)
}

/// Why this machine cannot run `m` at `tp` (`None` = it can).
fn tp_refusal(m: &Model, tp: u32, gpus: u32) -> Option<String> {
    if !m.tp_choices().contains(&tp) {
        return Some(if m.preset == crate::config::Preset::Flashnext {
            format!(
                "{} runs on two cards only (TP=2): no single-card pack is hosted",
                m.name
            )
        } else {
            format!("{} is not hosted for TP={tp}", m.name)
        });
    }
    (m.preset.uses_gpus() && tp > gpus).then(|| {
        format!(
            "TP={tp} needs {tp} supported GPUs and this machine has {gpus}{}",
            if gpus == 1 && m.tp_choices().contains(&1) {
                ": pick TP=1 (one card)"
            } else {
                ""
            }
        )
    })
}

async fn setup_model(State(st): State<AppState>, Json(b): Json<ModelReq>) -> Response {
    mutate(&st, move |c, cfg| {
        let cat = load_catalog(c, &cfg)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
        let m = cat.model(&b.id).ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                format!("no model `{}` in the catalog", b.id),
            )
        })?;
        let gpus = usable_gpus(c);
        if let Some(t) = b.tp {
            if let Some(why) = tp_refusal(m, t, gpus) {
                return Err((StatusCode::CONFLICT, why));
            }
        }
        let same = cfg.engine.preset.subcommand() == m.preset.subcommand();
        let mut next = cfg.clone();
        next.setup = setup::pick_model(&cfg.setup, &m.id, same);
        let tp = b.tp.unwrap_or_else(|| chosen_tp(&next, m, gpus));
        next.setup.tp = if m.preset == crate::config::Preset::Mock {
            0
        } else {
            tp
        };
        compose_pick(c, next, m, tp)
    })
    .await
}

#[derive(Deserialize)]
struct TpPick {
    tp: u32,
    /// TP=1: the card (default: keep the config's).
    #[serde(default)]
    device: Option<u32>,
}

/// The Model step's TP choice for the picked model (one card or two).
async fn setup_tp(State(st): State<AppState>, Json(b): Json<TpPick>) -> Response {
    mutate(&st, move |c, cfg| {
        let cat = load_catalog(c, &cfg)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
        let m = cat
            .model(&cfg.setup.model)
            .ok_or_else(|| (StatusCode::CONFLICT, "Pick a model first.".to_string()))?;
        let gpus = usable_gpus(c);
        if let Some(why) = tp_refusal(m, b.tp, gpus) {
            return Err((StatusCode::CONFLICT, why));
        }
        let mut next = cfg.clone();
        next.setup.tp = b.tp;
        if let Some(d) = b.device {
            if gpus > 0 && d >= gpus {
                return Err((
                    StatusCode::CONFLICT,
                    format!("card {d} does not exist: this machine has {gpus} supported GPU(s)"),
                ));
            }
            next.engine.device = d;
        }
        if b.tp == 2 {
            next.engine.device = 0;
        }
        compose_pick(c, next, m, b.tp)
    })
    .await
}

#[derive(Deserialize)]
struct ProfileReq {
    name: String,
}

async fn setup_profile(State(st): State<AppState>, Json(b): Json<ProfileReq>) -> Response {
    mutate(&st, move |c, cfg| {
        // the model's composition first (a download may have finished since it was picked)
        let mut base = cfg.clone();
        if let Ok(cat) = load_catalog(c, &cfg) {
            if let Some(m) = cat.model(&cfg.setup.model) {
                let tp = chosen_tp(&cfg, m, usable_gpus(c));
                if let Ok(x) = catalog::config_for_tp(&cfg, m, tp, &roots(c, &cfg)) {
                    base = x;
                }
            }
        }
        let sec = base.engine.preset.subcommand();
        let p = crate::profiles::builtin_for(sec, base.engine.tp)
            .into_iter()
            .find(|p| p.name == b.name)
            .ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    format!("no built-in profile `{}` for `saber {sec}`", b.name),
                )
            })?;
        let mut next = crate::profiles::apply_profile(&base, &p)
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("{e:#}")))?;
        next.setup.profile = p.name;
        Ok(next)
    })
    .await
}

#[derive(Deserialize)]
struct NetworkReq {
    engine_port: u16,
    #[serde(default)]
    console_host: String,
    #[serde(default)]
    console_port: u16,
    /// `None` = keep the stored token; `Some("")` = remove it.
    #[serde(default)]
    admin_token: Option<String>,
    /// The `/v1` proxy keys (replaces the list). A key sent without its text keeps the stored
    /// hash of the key with that label (stored keys are hashes).
    #[serde(default)]
    keys: Vec<NetKey>,
}

#[derive(Deserialize)]
struct NetKey {
    label: String,
    #[serde(default)]
    key: String,
}

async fn setup_network(State(st): State<AppState>, Json(b): Json<NetworkReq>) -> Response {
    let flag_token = st.auth.admin().source() == "flag";
    let auth = st.auth.clone();
    let r = mutate(&st, move |_c, mut cfg| {
        cfg.engine.port = b.engine_port;
        cfg.console.host = b.console_host.trim().to_string();
        if cfg.console.host == "127.0.0.1" {
            cfg.console.host.clear();
        }
        cfg.console.port = if b.console_port == 8090 {
            0
        } else {
            b.console_port
        };
        if let Some(t) = b.admin_token {
            cfg.console.admin_token = t.trim().to_string();
            cfg.console.admin_token_hash.clear();
        }
        let old = std::mem::take(&mut cfg.proxy.keys);
        for k in b.keys {
            let label = k.label.trim();
            if k.key.trim().is_empty() {
                if let Some(o) = old.iter().find(|o| o.label == label) {
                    cfg.proxy.keys.push(o.clone());
                    continue;
                }
            }
            let mut n = ProxyKey::new(label, k.key.trim());
            n.created_ms = crate::now_ms();
            cfg.proxy.keys.push(n);
        }
        cfg.seal()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
        Ok(cfg)
    })
    .await;
    // the token checks from now on (unless --admin-token, which wins)
    if r.status().is_success() && !flag_token {
        auth.set_admin(st.sup.config_now().admin());
    }
    r
}

#[derive(Deserialize)]
struct KeygenReq {
    /// `api` (a `/v1` key) or `admin` (the console's admin token).
    #[serde(default)]
    kind: String,
}

async fn setup_keygen(Json(b): Json<KeygenReq>) -> Response {
    let prefix = if b.kind == "admin" {
        "sa-admin-"
    } else {
        "sk-saber-"
    };
    match setup::random_key(prefix) {
        Ok(k) => Json(json!({"key": k})).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

async fn setup_launch(State(st): State<AppState>) -> Response {
    let c = st.console.clone();
    let cfg = st.sup.config_now();
    let r = blocking(move || {
        let rt = tokio::runtime::Handle::current();
        let _g = rt.enter();
        c.apply(cfg, crate::history::Origin::Apply, "setup wizard".into())
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

#[derive(Deserialize)]
struct FinishReq {
    #[serde(default)]
    without_start: bool,
}

async fn setup_finish(State(st): State<AppState>, Json(b): Json<FinishReq>) -> Response {
    mutate(&st, move |c, mut cfg| {
        let f = facts(c, &cfg);
        cfg.setup = if b.without_start {
            setup::finish_without_start(&cfg.setup, &f)
        } else {
            setup::goto(&cfg.setup, "done", &f)
        }
        .map_err(|e| (StatusCode::CONFLICT, e))?;
        Ok(cfg)
    })
    .await
}

async fn setup_skip(State(st): State<AppState>) -> Response {
    mutate(&st, |_c, mut cfg| {
        cfg.setup = setup::skip(&cfg.setup);
        Ok(cfg)
    })
    .await
}

async fn setup_rerun(State(st): State<AppState>) -> Response {
    mutate(&st, |_c, mut cfg| {
        cfg.setup = setup::rerun(&cfg.setup);
        Ok(cfg)
    })
    .await
}
