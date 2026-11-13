//! Live: the settings plane over HTTP against a fake engine binary (a script
//! that serves `/v1/models` on its `--port`, exits 2 on `--bad-flag` the way clap refuses an
//! unknown argument, and never answers on `--hang`).
//!
//! Proves: validation errors block an apply (422, nothing saved); an apply from stopped boots
//! and reaches ready; a field that does not reach the command line saves without a restart; a
//! deliberately bad config (exit during load) and a config that never reaches ready (timeout)
//! are both AUTO-ROLLED BACK to the last working config, which is running again; a manual
//! rollback; the history records each outcome; the profile CRUD + import/export round trip and
//! the built-in composition; the packs listing; the mutating-call header rule.

use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use saber_console::apply::Console;
use saber_console::config::{Config, FlagValue};
use saber_console::http::{router, AppState};
use saber_console::logring::LogRing;
use saber_console::proxy::Proxy;
use saber_console::scrape::MetricsStore;
use saber_console::supervisor::Supervisor;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const FAKE: &str = r#"#!/bin/sh
port=""; prev=""
for a in "$@"; do
  case "$a" in
    --bad-flag) echo "error: unexpected argument '--bad-flag' found" >&2; exit 2;;
    --hang) exec sleep 600;;
  esac
  [ "$prev" = "--port" ] && port="$a"
  prev="$a"
done
echo "fake engine on $port"
exec python3 -I -c 'import http.server,json,sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b=json.dumps({"data":[{"id":"fake"}]}).encode() if self.path=="/v1/models" else b"num_requests_running 0\nnum_requests_waiting 0\n"
        self.send_response(200); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self,*a): pass
http.server.HTTPServer(("127.0.0.1",int(sys.argv[1])),H).serve_forever()' "$port"
"#;

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn schema() -> Value {
    let f = |name: &str, kind: &str| {
        let r = saber_console::schema::range_of(name);
        json!({"name": name, "section": "serve", "kind": kind, "min": r.map(|r| r.0),
               "max": r.map(|r| r.1), "group": "Batching", "tier": "basic"})
    };
    json!({"flags": [
        f("--engine-max-seqs", "uint"),
        f("--engine-max-ctx", "uint"),
        f("--step-delay-ms", "uint"),
        f("--port", "uint"),
        f("--host", "string"),
    ]})
}

async fn request(port: u16, method: &str, path: &str, body: &str, header: bool) -> (u16, String) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    let h = if header { "X-Saber-Console: 1\r\n" } else { "" };
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{h}Content-Type: application/json\r\n\
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
    let body = text.split_once("\r\n\r\n").map_or("", |x| x.1).to_string();
    // a chunked body: drop the size lines
    let body = if text
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        body.split("\r\n")
            .enumerate()
            .filter(|(i, _)| i % 2 == 1)
            .map(|(_, x)| x)
            .collect::<String>()
    } else {
        body
    };
    (code, body)
}

async fn post(port: u16, path: &str, body: &Value) -> (u16, Value) {
    let (c, b) = request(port, "POST", path, &body.to_string(), true).await;
    (c, serde_json::from_str(&b).unwrap_or(Value::String(b)))
}

async fn get(port: u16, path: &str) -> Value {
    let (c, b) = request(port, "GET", path, "", false).await;
    assert_eq!(c, 200, "{path}: {b}");
    serde_json::from_str(&b).unwrap()
}

/// Wait for the current apply job to finish; returns it.
async fn wait_job(port: u16, within: Duration) -> Value {
    let t0 = Instant::now();
    loop {
        let v = get(port, "/api/apply").await;
        if v["job"]["state"] != "running" {
            return v["job"].clone();
        }
        assert!(t0.elapsed() < within, "the apply did not finish: {v}");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

fn step<'a>(job: &'a Value, id: &str) -> &'a Value {
    job["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["id"] == id)
        .unwrap_or_else(|| panic!("no step {id} in {job}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn apply_restart_rollback_and_the_settings_api() {
    let dir = saber_test_scratch::ScratchDir::new("console-settings");
    let bin = dir.path().join("fake-saber");
    std::fs::write(&bin, FAKE).unwrap();
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    // a packs dir with a target and a drafter
    let packs = dir.path().join("nvme/saber-packs");
    for (n, m) in [
        (
            "qwen38-27b-a",
            r#"{"tp_size": 2, "ranks": [{"rank": 0, "file_bytes": 4096}]}"#,
        ),
        (
            "qwen38-dflash2-drafter-b",
            r#"{"tp_size": 2, "builder": "drafter_head_pair.py", "ranks": []}"#,
        ),
    ] {
        std::fs::create_dir_all(packs.join(n)).unwrap();
        std::fs::write(packs.join(n).join("manifest.json"), m).unwrap();
        std::fs::write(packs.join(n).join("rank0.pack"), vec![0u8; 4096]).unwrap();
    }

    // not a pack: hidden directories are skipped
    std::fs::create_dir_all(packs.join(".cache")).unwrap();
    let engine_port = free_port();
    let mut good = Config::mock(engine_port);
    good.engine.autostart = false;
    good.engine.paths.binary = bin.clone();
    good.engine.paths.repo = dir.path().to_path_buf();
    good.engine.paths.nvme = dir.path().join("nvme");
    good.engine.drain_timeout_s = 3;
    good.engine.stop_timeout_s = 5;
    good.console.data_dir = dir.path().join("data");
    good.console.ready_timeout_s = 20;
    good.usage.enabled = false;
    let cfg_path = dir.path().join("config.toml");
    let (sup, _rx) = Supervisor::new(
        good.clone(),
        cfg_path.clone(),
        dir.path().to_path_buf(),
        bin.clone(),
        Arc::new(LogRing::new(1000)),
    );
    let metrics = Arc::new(MetricsStore::new(100, 5));
    let schema = schema();
    let console_port = free_port();
    let state = AppState {
        console: Arc::new(Console::new(
            sup.clone(),
            metrics.clone(),
            &schema,
            console_port,
        )),
        sup: sup.clone(),
        metrics,
        schema: Arc::new(schema),
        auth: Arc::new(saber_console::auth::Auth::open()),
        loopback: true,
        proxy: Arc::new(Proxy::new(None)),
        usage: None,
    };
    let l = tokio::net::TcpListener::bind(("127.0.0.1", console_port))
        .await
        .unwrap();
    tokio::spawn(async move { axum::serve(l, router(state)).await.unwrap() });
    let p = console_port;

    // ---- validation: typed errors per field; a failing validate blocks the apply ----
    let mut bad_type = good.clone();
    bad_type
        .engine
        .flags
        .insert("--engine-max-seqs".into(), FlagValue::Str("many".into()));
    bad_type
        .engine
        .flags
        .insert("--no-such-flag".into(), FlagValue::Bool(true));
    let (c, v) = post(p, "/api/config/validate", &json!({"config": bad_type})).await;
    assert_eq!(c, 200);
    assert_eq!(v["report"]["ok"], false);
    let fields: Vec<&str> = v["report"]["errors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["field"].as_str().unwrap())
        .collect();
    assert_eq!(fields, ["--engine-max-seqs", "--no-such-flag"], "{v}");
    assert_eq!(v["estimate"]["is_estimate"], true);
    assert!(v["command_line"]
        .as_str()
        .unwrap()
        .contains("--engine mock"));
    assert!(v["effective"]
        .as_array()
        .unwrap()
        .iter()
        .any(|e| e["name"] == "--port" && e["source"] == "preset"));
    let (c, v) = post(p, "/api/config/apply", &json!({"config": bad_type})).await;
    assert_eq!(c, 422, "{v}");
    assert!(!cfg_path.exists(), "a refused apply saves nothing");
    // the mutating-call header rule
    let (c, _) = request(
        p,
        "POST",
        "/api/config/validate",
        &json!({"config": good}).to_string(),
        false,
    )
    .await;
    assert_eq!(c, 403);

    // ---- apply from stopped: save -> start -> loading -> ready ----
    let (c, v) = post(
        p,
        "/api/config/apply",
        &json!({"config": good, "note": "first"}),
    )
    .await;
    assert_eq!(c, 202, "{v}");
    let job = wait_job(p, Duration::from_secs(30)).await;
    assert_eq!(job["state"], "done", "{job}");
    assert_eq!(step(&job, "drain")["state"], "skipped");
    assert_eq!(step(&job, "ready")["state"], "done");
    let first_id = job["history_id"].as_u64().unwrap();
    let st = get(p, "/api/status").await;
    assert_eq!(st["state"], "ready");
    let gen1 = st["generation"].as_u64().unwrap();
    assert!(cfg_path.exists());

    // ---- a field that does not reach the command line: saved, no restart ----
    let mut m = good.clone();
    m.metrics.interval_s = 10;
    let (_, chk) = post(p, "/api/config/validate", &json!({"config": m})).await;
    assert_eq!(chk["restart_required"], false);
    assert_eq!(chk["diff_base"], "running");
    assert!(chk["diff"]
        .as_array()
        .unwrap()
        .iter()
        .any(|d| d["field"] == "metrics.interval_s"));
    let (c, _) = post(p, "/api/config/apply", &json!({"config": m})).await;
    assert_eq!(c, 202);
    let job = wait_job(p, Duration::from_secs(10)).await;
    assert_eq!(job["state"], "done");
    assert_eq!(job["restart"], false);
    assert!(
        job["message"].as_str().unwrap().contains("no restart"),
        "{job}"
    );
    assert_eq!(
        get(p, "/api/status").await["generation"].as_u64(),
        Some(gen1)
    );

    // ---- a flag change restarts (drain -> stop -> start -> ready) ----
    let mut slow = m.clone();
    slow.engine
        .flags
        .insert("--step-delay-ms".into(), FlagValue::Int(5));
    let (_, chk) = post(p, "/api/config/validate", &json!({"config": slow})).await;
    assert_eq!(chk["restart_required"], true);
    assert!(chk["diff"]
        .as_array()
        .unwrap()
        .iter()
        .any(|d| d["field"] == "--step-delay-ms" && d["new"] == "5"));
    post(p, "/api/config/apply", &json!({"config": slow})).await;
    let job = wait_job(p, Duration::from_secs(30)).await;
    assert_eq!(job["state"], "done", "{job}");
    assert_eq!(step(&job, "drain")["state"], "done");
    assert_eq!(step(&job, "stop")["state"], "done");
    let working_id = job["history_id"].as_u64().unwrap();
    assert!(get(p, "/api/status").await["generation"].as_u64() > Some(gen1));

    // ---- a deliberately bad config (the engine exits 2 during load): AUTO-ROLLBACK ----
    let mut bad = slow.clone();
    bad.engine.extra_args = vec!["--bad-flag".into()];
    let (c, _) = post(p, "/api/config/apply", &json!({"config": bad})).await;
    assert_eq!(c, 202);
    let job = wait_job(p, Duration::from_secs(40)).await;
    assert_eq!(job["state"], "rolled-back", "{job}");
    assert_eq!(job["rolled_back_to"].as_u64(), Some(working_id));
    assert_eq!(step(&job, "loading")["state"], "failed");
    let lm = step(&job, "loading")["message"].as_str().unwrap();
    assert!(lm.contains("code 2") && lm.contains("--bad-flag"), "{job}");
    assert_eq!(step(&job, "rollback")["state"], "done");
    assert!(
        job["message"].as_str().unwrap().contains("rolled back"),
        "{job}"
    );
    let bad_id = job["history_id"].as_u64().unwrap();
    let st = get(p, "/api/status").await;
    assert_eq!(st["state"], "ready", "the last working config runs again");
    assert!(!st["command"]
        .as_array()
        .unwrap()
        .iter()
        .any(|a| a == "--bad-flag"));
    assert_eq!(sup.config_now().engine.extra_args, Vec::<String>::new());
    assert_eq!(Config::load(&cfg_path).unwrap().unwrap(), slow);

    // ---- a config that never reaches ready (timeout): AUTO-ROLLBACK ----
    let mut hang = slow.clone();
    hang.engine.extra_args = vec!["--hang".into()];
    hang.console.ready_timeout_s = 5;
    post(p, "/api/config/apply", &json!({"config": hang})).await;
    let job = wait_job(p, Duration::from_secs(60)).await;
    assert_eq!(job["state"], "rolled-back", "{job}");
    assert!(step(&job, "loading")["message"]
        .as_str()
        .unwrap()
        .contains("did not reach ready"));
    assert_eq!(get(p, "/api/status").await["state"], "ready");

    // ---- history: outcomes recorded, newest first ----
    let h = get(p, "/api/history").await;
    let entries = h["entries"].as_array().unwrap();
    let by_id = |id: u64| entries.iter().find(|e| e["id"] == id).unwrap().clone();
    assert_eq!(by_id(first_id)["outcome"], "ready");
    assert_eq!(by_id(working_id)["outcome"], "ready");
    assert_eq!(by_id(bad_id)["outcome"], "failed");
    assert!(by_id(bad_id)["message"]
        .as_str()
        .unwrap()
        .contains("code 2"));
    assert_eq!(entries[0]["origin"], "auto-rollback");
    assert_eq!(entries[0]["outcome"], "ready");
    let e = get(p, &format!("/api/history/{first_id}")).await;
    assert!(e["toml"].as_str().unwrap().contains("version = 4"));

    // ---- one-click (manual) rollback to the first config ----
    let (c, v) = post(p, &format!("/api/history/{first_id}/rollback"), &json!({})).await;
    assert_eq!(c, 202, "{v}");
    let job = wait_job(p, Duration::from_secs(30)).await;
    assert_eq!(job["state"], "done", "{job}");
    assert_eq!(job["origin"], "rollback");
    assert!(sup.config_now().engine.flags.is_empty());
    let (c, _) = post(p, "/api/history/999999/rollback", &json!({})).await;
    assert_eq!(c, 400);

    // ---- profiles: built-ins, save / rename / duplicate / export / import / compose / delete ----
    let pr = get(p, "/api/profiles").await;
    assert_eq!(pr["builtin"]["serve"].as_array().unwrap().len(), 5);
    assert_eq!(pr["user"].as_array().unwrap().len(), 0);
    let (c, _) = post(
        p,
        "/api/profiles",
        &json!({"name": "slow", "description": "d", "config": slow}),
    )
    .await;
    assert_eq!(c, 200);
    let (c, _) = post(p, "/api/profiles", &json!({"name": "slow", "config": slow})).await;
    assert_eq!(c, 400, "duplicate names are refused");
    let (c, _) = request(
        p,
        "PUT",
        "/api/profiles/slow",
        &json!({"name": "slower"}).to_string(),
        true,
    )
    .await;
    assert_eq!(c, 200);
    let (c, v) = post(p, "/api/profiles/slower/duplicate", &json!({})).await;
    assert_eq!((c, v["name"].as_str()), (200, Some("slower copy")));
    let (c, toml_text) = request(p, "GET", "/api/profiles/slower/export", "", false).await;
    assert_eq!(c, 200);
    assert!(toml_text.contains("--step-delay-ms"), "{toml_text}");
    let (c, v) = request(p, "POST", "/api/profiles/import", &toml_text, true).await;
    assert_eq!(c, 200, "{v}");
    let v: Value = serde_json::from_str(&v).unwrap();
    assert_eq!(v["imported"][0], "slower (imported)");
    let pr = get(p, "/api/profiles").await;
    let user = pr["user"].as_array().unwrap();
    assert_eq!(user.len(), 3);
    let a = user.iter().find(|x| x["name"] == "slower").unwrap();
    let b = user
        .iter()
        .find(|x| x["name"] == "slower (imported)")
        .unwrap();
    assert_eq!(a["flags"], b["flags"], "export -> import round trip");
    let (c, v) = post(
        p,
        "/api/profiles/compose",
        &json!({"config": good, "name": "Max throughput", "builtin": true}),
    )
    .await;
    assert_eq!(c, 200, "{v}");
    assert_eq!(v["config"]["engine"]["flags"]["--engine-max-ctx"], 32768);
    let (c, v) = post(
        p,
        "/api/profiles/compose",
        &json!({"config": good, "name": "slower"}),
    )
    .await;
    assert_eq!(c, 200);
    assert_eq!(v["config"]["engine"]["flags"]["--step-delay-ms"], 5);
    let (c, _) = request(p, "DELETE", "/api/profiles/slower%20copy", "", true).await;
    assert_eq!(c, 200);
    let (c, _) = request(p, "DELETE", "/api/profiles/nope", "", true).await;
    assert_eq!(c, 404);

    // ---- the packs listing ----
    let pk = get(p, "/api/packs").await;
    let rows = pk["packs"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    let t = rows.iter().find(|r| r["name"] == "qwen38-27b-a").unwrap();
    assert_eq!(t["role"], "target");
    assert_eq!(t["rank_weight_bytes"], 4096);
    assert!(t["size_bytes"].as_u64().unwrap() >= 4096);
    assert_eq!(t["arg"], "/packs/qwen38-27b-a");
    let d = rows
        .iter()
        .find(|r| r["name"] == "qwen38-dflash2-drafter-b")
        .unwrap();
    assert_eq!(d["role"], "drafter");

    sup.stop().await.unwrap();
}
