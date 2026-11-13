//! Live: the download manager against a local HTTP server that speaks the
//! Hub's tree API and serves byte ranges (a small fake pack, never a real model), and the
//! setup wizard + system check + catalog + pack delete over the HTTP API with a fake engine.
//!
//! Proves: a pack downloads, verifies and lands by atomic rename (no staging dir left); a pause
//! keeps the `.part` and the resume continues with a Range request; a sha256 mismatch fails in
//! plain English and deletes the corrupt part; the disk-space precheck refuses before any byte
//! is fetched; a cancel deletes the staging dir; an explicit url/size/sha list of loose files
//! lands in place; the API contract of /api/system, /api/catalog, /api/downloads, the delete
//! guard, and a full wizard walk from an empty config to ready.

use std::collections::HashMap;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use saber_console::apply::Console;
use saber_console::catalog::{Catalog, Roots};
use saber_console::config::Config;
use saber_console::downloads::{DlState, Downloads, Spec};
use saber_console::http::{router, AppState};
use saber_console::logring::LogRing;
use saber_console::proxy::Proxy;
use saber_console::scrape::MetricsStore;
use saber_console::supervisor::Supervisor;
use saber_console::system::HostFacts;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

// ---------------------------------------------------------------- the fake file server

#[derive(Default)]
struct Server {
    /// repo path -> bytes
    files: HashMap<String, Vec<u8>>,
    /// repo path -> the sha256 the listing claims (overrides the real one)
    fake_sha: HashMap<String, String>,
    slow: AtomicBool,
    resolve_hits: AtomicUsize,
    /// `Range:` headers seen on resolve requests
    ranges: Mutex<Vec<String>>,
}

fn sha_hex(b: &[u8]) -> String {
    use sha2::Digest;
    hex::encode(sha2::Sha256::digest(b))
}

fn blob(n: usize, seed: u8, magic: bool) -> Vec<u8> {
    let mut v: Vec<u8> = (0..n)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect();
    if magic {
        v[..8].copy_from_slice(b"SBRPACK1");
    }
    v
}

impl Server {
    fn tree(&self, prefix: &str, recursive: bool) -> String {
        let mut out = Vec::new();
        let mut names: Vec<_> = self.files.keys().collect();
        names.sort();
        for p in names {
            let rel = if prefix.is_empty() {
                p.as_str()
            } else if let Some(r) = p.strip_prefix(&format!("{prefix}/")) {
                r
            } else {
                continue;
            };
            if !recursive && rel.contains('/') {
                continue;
            }
            let b = &self.files[p];
            let sha = self.fake_sha.get(p).cloned().unwrap_or_else(|| sha_hex(b));
            out.push(
                json!({"type": "file", "path": p, "size": 134, "oid": "gitsha",
                            "lfs": {"oid": sha, "size": b.len(), "pointerSize": 134}}),
            );
        }
        Value::Array(out).to_string()
    }
}

async fn serve_conn(srv: Arc<Server>, mut s: TcpStream) {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = match s.read(&mut tmp).await {
            Ok(0) | Err(_) => return,
            Ok(n) => n,
        };
        buf.extend_from_slice(&tmp[..n]);
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }
    let head = String::from_utf8_lossy(&buf).into_owned();
    let line = head.lines().next().unwrap_or("");
    let target = line.split_whitespace().nth(1).unwrap_or("/").to_string();
    let range = head.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        k.eq_ignore_ascii_case("range")
            .then(|| v.trim().to_string())
    });
    let (path, query) = target.split_once('?').unwrap_or((&target, ""));
    let reply = |code: &str, ctype: &str, extra: String, body: &[u8]| {
        let mut r = format!(
            "HTTP/1.1 {code}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n{extra}\r\n",
            body.len()
        )
        .into_bytes();
        r.extend_from_slice(body);
        r
    };
    if let Some(rest) = path.strip_prefix("/api/models/o/r/tree/main") {
        let prefix = rest.trim_start_matches('/');
        let body = srv.tree(prefix, query.contains("recursive=true"));
        let _ = s
            .write_all(&reply(
                "200 OK",
                "application/json",
                String::new(),
                body.as_bytes(),
            ))
            .await;
        return;
    }
    if let Some(rel) = path
        .strip_prefix("/o/r/resolve/main/")
        .or_else(|| path.strip_prefix("/plain/"))
    {
        srv.resolve_hits.fetch_add(1, Ordering::SeqCst);
        let Some(b) = srv.files.get(rel) else {
            let _ = s
                .write_all(&reply(
                    "404 Not Found",
                    "text/plain",
                    String::new(),
                    b"nope",
                ))
                .await;
            return;
        };
        let start = range
            .as_deref()
            .and_then(|r| r.strip_prefix("bytes="))
            .and_then(|r| r.split('-').next())
            .and_then(|x| x.parse::<usize>().ok())
            .unwrap_or(0);
        if let Some(r) = &range {
            srv.ranges.lock().unwrap().push(r.clone());
        }
        let body = &b[start.min(b.len())..];
        let (code, extra) = if range.is_some() {
            (
                "206 Partial Content",
                format!(
                    "Content-Range: bytes {start}-{}/{}\r\n",
                    b.len() - 1,
                    b.len()
                ),
            )
        } else {
            ("200 OK", String::new())
        };
        let head = format!(
            "HTTP/1.1 {code}\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n{extra}\r\n",
            body.len()
        );
        if s.write_all(head.as_bytes()).await.is_err() {
            return;
        }
        for chunk in body.chunks(16 * 1024) {
            if s.write_all(chunk).await.is_err() {
                return;
            }
            if srv.slow.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(40)).await;
            }
        }
        return;
    }
    let _ = s
        .write_all(&reply(
            "404 Not Found",
            "text/plain",
            String::new(),
            b"no route",
        ))
        .await;
}

async fn start_server(srv: Arc<Server>) -> u16 {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = l.local_addr().unwrap().port();
    tokio::spawn(async move {
        loop {
            let Ok((s, _)) = l.accept().await else { return };
            tokio::spawn(serve_conn(srv.clone(), s));
        }
    });
    port
}

/// The fake pack: a manifest and two rank files of 640 KiB each (slow mode: ~6 s a file).
fn fake_pack_server() -> Server {
    let mut s = Server::default();
    s.files.insert(
        "tiny-pack/manifest.json".into(),
        br#"{"tp_size": 2, "ranks": []}"#.to_vec(),
    );
    s.files
        .insert("tiny-pack/rank0.pack".into(), blob(640 * 1024, 1, true));
    s.files
        .insert("tiny-pack/rank1.pack".into(), blob(640 * 1024, 2, true));
    s.files
        .insert("tok.json".into(), b"{\"tokenizer\": true}".to_vec());
    s
}

fn catalog_json(port: u16, tok_sha: &str) -> String {
    json!({
        "version": 1,
        "hf_endpoint": format!("http://127.0.0.1:{port}"),
        "models": [
            {"id": "mock-dl", "name": "Mock with a pack", "kind": "mock", "preset": "mock",
             "fit": {"min_gpus": 0},
             "components": [
                {"id": "target", "label": "Tiny pack", "dest": "packs", "name": "tiny-pack",
                 "approx_bytes": 1310720,
                 "source": {"hf_repo": "o/r", "path": "tiny-pack"},
                 "pack_magic": ["rank0.pack", "rank1.pack"], "required": ["manifest.json"]}
             ]},
            {"id": "loose", "name": "Loose files", "kind": "mock", "preset": "mock",
             "components": [
                {"id": "tok", "dest": "repo_cache", "layout": "files",
                 "files": [{"path": "tok.json", "url": format!("http://127.0.0.1:{port}/plain/tok.json"),
                            "size": 19, "sha256": tok_sha, "config_path": "tokenizer"}]}
             ]},
            {"id": "big", "name": "Needs real GPUs", "kind": "27B", "preset": "qwen27b",
             "fit": {"min_gpus": 2, "arch_prefixes": ["gfx12"], "min_vram_gib": 30},
             "components": [
                {"id": "target", "dest": "packs", "name": "big-pack", "approx_bytes": 1000,
                 "source": {"hf_repo": "o/r", "path": "nothing-here"}}
             ]}
        ]
    })
    .to_string()
}

fn spec(cat: &Catalog, model: &str, roots: &Roots, port: u16) -> Spec {
    let m = cat.model(model).unwrap();
    Spec {
        model: m.id.clone(),
        model_name: m.name.clone(),
        component: m.components[0].clone(),
        roots: roots.clone(),
        hf_endpoint: format!("http://127.0.0.1:{port}"),
        token: Some("hf_test_token".into()),
        curl: PathBuf::new(),
        reserve_bytes: 0,
    }
}

async fn wait_state(
    d: &Downloads,
    id: u64,
    want: &[DlState],
    within: Duration,
) -> saber_console::downloads::Job {
    let t0 = Instant::now();
    loop {
        let j = d.get(id).unwrap();
        if want.contains(&j.state) {
            return j;
        }
        assert!(t0.elapsed() < within, "download #{id} stuck: {j:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn roots_in(dir: &Path) -> Roots {
    let r = Roots {
        packs: dir.join("nvme/saber-packs"),
        models: dir.join("nvme/models"),
        repo_cache: dir.join("repo/.cache"),
    };
    for p in [&r.packs, &r.models, &r.repo_cache] {
        std::fs::create_dir_all(p).unwrap();
    }
    r
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn download_verify_resume_and_failures() {
    let dir = saber_test_scratch::ScratchDir::new("console-dl");
    let srv = Arc::new(fake_pack_server());
    let port = start_server(srv.clone()).await;
    let tok_sha = sha_hex(&srv.files["tok.json"]);
    let cat = Catalog::parse(&catalog_json(port, &tok_sha)).unwrap();
    let roots = roots_in(dir.path());
    let d = Downloads::new(None);
    tokio::spawn(d.clone().worker());

    // ---- 1. the happy path: listed, fetched, verified, renamed into place ----
    let j = d.enqueue(spec(&cat, "mock-dl", &roots, port));
    let j = wait_state(
        &d,
        j.id,
        &[DlState::Done, DlState::Failed],
        Duration::from_secs(30),
    )
    .await;
    assert_eq!(j.state, DlState::Done, "{j:?}");
    assert_eq!(j.files_total, 3);
    assert_eq!(j.total_bytes, Some(2 * 640 * 1024 + 27));
    let pack = roots.packs.join("tiny-pack");
    for f in ["manifest.json", "rank0.pack", "rank1.pack"] {
        assert_eq!(
            std::fs::read(pack.join(f)).unwrap(),
            srv.files[&format!("tiny-pack/{f}")],
            "{f}"
        );
    }
    assert!(
        !roots.packs.join(".saber-download-tiny-pack").exists(),
        "staging left behind"
    );
    // a second start of an installed dir refuses by name
    let j2 = d.enqueue(spec(&cat, "mock-dl", &roots, port));
    let j2 = wait_state(
        &d,
        j2.id,
        &[DlState::Failed, DlState::Done],
        Duration::from_secs(10),
    )
    .await;
    assert!(j2.error.unwrap().contains("already exists"));
    std::fs::remove_dir_all(&pack).unwrap();

    // ---- 2. pause keeps the .part; resume continues with a Range request ----
    srv.slow.store(true, Ordering::SeqCst);
    srv.ranges.lock().unwrap().clear();
    let j = d.enqueue(spec(&cat, "mock-dl", &roots, port));
    let t0 = Instant::now();
    loop {
        let x = d.get(j.id).unwrap();
        if x.state == DlState::Downloading && x.done_bytes > 64 * 1024 {
            assert!(x.speed_bps.is_some_and(|s| s > 0.0), "{x:?}");
            assert!(x.eta_s.is_some(), "{x:?}");
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(20), "{x:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    d.pause(j.id).unwrap();
    let p = wait_state(&d, j.id, &[DlState::Paused], Duration::from_secs(10)).await;
    let staging = roots.packs.join(".saber-download-tiny-pack");
    let parts: Vec<_> = walk(&staging)
        .into_iter()
        .filter(|f| f.to_string_lossy().ends_with(".part"))
        .collect();
    assert_eq!(parts.len(), 1, "{parts:?}");
    let part_len = std::fs::metadata(&parts[0]).unwrap().len();
    assert!(part_len > 0 && part_len < 640 * 1024, "{part_len}");
    assert!(p.done_bytes >= part_len);
    // the token went in a 0600 header file, never on the command line
    let hdr = staging.join(".auth-header");
    assert_eq!(
        std::fs::metadata(&hdr).unwrap().permissions().mode() & 0o777,
        0o600
    );
    srv.slow.store(false, Ordering::SeqCst);
    d.resume(j.id).unwrap();
    let j = wait_state(
        &d,
        j.id,
        &[DlState::Done, DlState::Failed],
        Duration::from_secs(30),
    )
    .await;
    assert_eq!(j.state, DlState::Done, "{j:?}");
    let ranges = srv.ranges.lock().unwrap().clone();
    assert!(
        ranges.iter().any(|r| r == &format!("bytes={part_len}-")),
        "no resume from {part_len}: {ranges:?}"
    );
    assert_eq!(
        std::fs::read(pack.join("rank0.pack")).unwrap(),
        srv.files["tiny-pack/rank0.pack"]
    );
    assert_eq!(
        std::fs::read(pack.join("rank1.pack")).unwrap(),
        srv.files["tiny-pack/rank1.pack"]
    );
    std::fs::remove_dir_all(&pack).unwrap();

    // ---- 3. a sha256 mismatch fails in plain English and deletes the corrupt part ----
    let mut bad = fake_pack_server();
    bad.fake_sha
        .insert("tiny-pack/rank1.pack".into(), "00".repeat(32));
    let bad = Arc::new(bad);
    let bport = start_server(bad.clone()).await;
    let j = d.enqueue(spec(&cat, "mock-dl", &roots, bport));
    let j = wait_state(
        &d,
        j.id,
        &[DlState::Done, DlState::Failed],
        Duration::from_secs(30),
    )
    .await;
    assert_eq!(j.state, DlState::Failed);
    let e = j.error.unwrap();
    assert!(
        e.contains("rank1.pack is corrupt") && e.contains("Retry"),
        "{e}"
    );
    assert!(!pack.exists());
    assert!(!staging.join("rank1.pack.part").exists() && !staging.join("rank1.pack").exists());
    // the good files stay staged: the retry fetches only the bad one
    assert!(staging.join("rank0.pack").exists());
    d.cancel(j.id).unwrap();
    assert!(
        !staging.exists(),
        "cancel of a failed job deletes the staging dir"
    );

    // ---- 4. the disk-space precheck refuses before any byte is fetched ----
    let tiny = Downloads::new(Some(Arc::new(|_p: &Path| Some(1000))));
    tokio::spawn(tiny.clone().worker());
    let hits = srv.resolve_hits.load(Ordering::SeqCst);
    let j = tiny.enqueue(spec(&cat, "mock-dl", &roots, port));
    let j = wait_state(
        &tiny,
        j.id,
        &[DlState::Failed, DlState::Done],
        Duration::from_secs(10),
    )
    .await;
    let e = j.error.unwrap();
    assert!(
        e.contains("Not enough disk space") && e.contains("1.3 MB"),
        "{e}"
    );
    assert_eq!(
        srv.resolve_hits.load(Ordering::SeqCst),
        hits,
        "fetched despite the precheck"
    );

    // ---- 5. cancel mid-download deletes what it fetched ----
    srv.slow.store(true, Ordering::SeqCst);
    let j = d.enqueue(spec(&cat, "mock-dl", &roots, port));
    let t0 = Instant::now();
    while d.get(j.id).unwrap().done_bytes < 32 * 1024 {
        assert!(t0.elapsed() < Duration::from_secs(20));
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    d.cancel(j.id).unwrap();
    wait_state(&d, j.id, &[DlState::Cancelled], Duration::from_secs(10)).await;
    assert!(!staging.exists() && !pack.exists());
    srv.slow.store(false, Ordering::SeqCst);

    // ---- 6. an explicit url/size/sha list of loose files lands in the root ----
    let j = d.enqueue(spec(&cat, "loose", &roots, port));
    let j = wait_state(
        &d,
        j.id,
        &[DlState::Done, DlState::Failed],
        Duration::from_secs(20),
    )
    .await;
    assert_eq!(j.state, DlState::Done, "{j:?}");
    assert_eq!(
        std::fs::read(roots.repo_cache.join("tok.json")).unwrap(),
        srv.files["tok.json"]
    );
    assert_eq!(
        std::fs::read_dir(&roots.repo_cache).unwrap().count(),
        1,
        "staging left behind"
    );

    // ---- 7. a 404 listing (nothing under the repo path) fails by name ----
    let j = d.enqueue(spec(&cat, "big", &roots, port));
    let j = wait_state(
        &d,
        j.id,
        &[DlState::Failed, DlState::Done],
        Duration::from_secs(10),
    )
    .await;
    assert!(j.error.unwrap().contains("lists no files"));
}

fn walk(d: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(d) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(walk(&p));
            } else {
                out.push(p);
            }
        }
    }
    out
}

// ---------------------------------------------------------------- the API + the wizard

const FAKE: &str = r#"#!/bin/sh
port=""; prev=""
for a in "$@"; do
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

async fn request(port: u16, method: &str, path: &str, body: &str) -> (u16, Value) {
    request_as(port, method, path, body, None).await
}

async fn request_as(
    port: u16,
    method: &str,
    path: &str,
    body: &str,
    token: Option<&str>,
) -> (u16, Value) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    let a = token.map_or(String::new(), |t| format!("Authorization: Bearer {t}\r\n"));
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{a}X-Saber-Console: 1\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut b = Vec::new();
    tokio::time::timeout(Duration::from_secs(30), s.read_to_end(&mut b))
        .await
        .unwrap()
        .unwrap();
    let text = String::from_utf8_lossy(&b).into_owned();
    let code: u16 = text[9..12].parse().unwrap();
    let body = text.split_once("\r\n\r\n").map_or("", |x| x.1).to_string();
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
    (
        code,
        serde_json::from_str(&body).unwrap_or(Value::String(body)),
    )
}

async fn post(port: u16, path: &str, body: Value) -> (u16, Value) {
    request(port, "POST", path, &body.to_string()).await
}

async fn get(port: u16, path: &str) -> Value {
    let (c, v) = request(port, "GET", path, "").await;
    assert_eq!(c, 200, "{path}: {v}");
    v
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

/// A minimal fake sysfs: one CPU node and one gfx1201 card.
fn fake_sys(root: &Path) {
    let w = |p: &str, s: &str| {
        let f = root.join(p);
        std::fs::create_dir_all(f.parent().unwrap()).unwrap();
        std::fs::write(f, s).unwrap();
    };
    let n = "sys/class/kfd/kfd/topology/nodes";
    w(&format!("{n}/0/properties"), "gfx_target_version 0\n");
    w(
        &format!("{n}/1/properties"),
        "gfx_target_version 120001\ndrm_render_minor 128\n",
    );
    w(
        &format!("{n}/1/mem_banks/0/properties"),
        "size_in_bytes 34208743424\n",
    );
    w("dev/kfd", "");
    w("dev/dri/renderD128", "");
    w("proc/meminfo", "MemTotal: 65536000 kB\n");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_wizard_from_an_empty_config_to_ready() {
    let dir = saber_test_scratch::ScratchDir::new("console-wizard");
    let srv = Arc::new(fake_pack_server());
    let fport = start_server(srv.clone()).await;
    let cat_path = dir.path().join("models.json");
    std::fs::write(
        &cat_path,
        catalog_json(fport, &sha_hex(&srv.files["tok.json"])),
    )
    .unwrap();
    let bin = dir.path().join("fake-saber");
    std::fs::write(&bin, FAKE).unwrap();
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    let sysroot = dir.path().join("sys-root");
    fake_sys(&sysroot);

    // the console's defaults for this run (`saber ui --engine mock` on a fresh box)
    let engine_port = free_port();
    let mut cfg = Config::mock(engine_port);
    cfg.engine.autostart = false;
    cfg.engine.paths.binary = bin.clone();
    cfg.engine.paths.repo = dir.path().join("repo");
    cfg.engine.paths.nvme = dir.path().join("nvme");
    cfg.engine.drain_timeout_s = 3;
    cfg.engine.stop_timeout_s = 5;
    cfg.console.data_dir = dir.path().join("data");
    cfg.console.ready_timeout_s = 20;
    cfg.usage.enabled = false;
    cfg.downloads.catalog = cat_path.clone();
    cfg.downloads.reserve_bytes = 0;
    let cfg_path = dir.path().join("config.toml");
    let (sup, _rx) = Supervisor::new(
        cfg.clone(),
        cfg_path.clone(),
        dir.path().to_path_buf(),
        bin.clone(),
        Arc::new(LogRing::new(1000)),
    );
    let metrics = Arc::new(MetricsStore::new(100, 5));
    let schema = schema();
    let console_port = free_port();
    let mut console = Console::new(sup.clone(), metrics.clone(), &schema, console_port);
    console.sys_root = sysroot.clone();
    console.host_facts = Some(HostFacts {
        runtime: "podman".into(),
        runtime_version: Some("podman version 5.8.2".into()),
        image: "localhost/saber-dev:latest".into(),
        image_present: Some(false),
        conflicts: vec![],
        engine_port,
        port_busy: false,
        engine_ours: false,
    });
    let console = Arc::new(console);
    tokio::spawn(console.downloads.clone().worker());
    let state = AppState {
        console: console.clone(),
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

    // ---- a fresh box: the wizard is needed, at the system step ----
    let s = get(p, "/api/setup").await;
    assert_eq!(s["needed"], true);
    assert_eq!(s["config_exists"], false);
    assert_eq!(s["state"]["step"], "system");
    assert_eq!(s["steps"].as_array().unwrap().len(), 6);

    // ---- /api/system against the fake tree: one card = warn, the image missing = fail ----
    let sys = get(p, "/api/system").await;
    assert_eq!(sys["gpus"][0]["gfx"], "gfx1201");
    let ck = |id: &str| {
        sys["checks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == id)
            .unwrap()
            .clone()
    };
    assert_eq!(ck("gpus")["status"], "warn");
    assert_eq!(ck("image")["status"], "fail");
    assert!(ck("image")["fix"]
        .as_str()
        .unwrap()
        .contains("tools/build-image.sh"));
    assert_eq!(ck("kfd")["status"], "pass");

    // ---- /api/catalog: fit verdicts and install state ----
    let c = get(p, "/api/catalog").await;
    let m = |id: &str| {
        c["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == id)
            .unwrap()
            .clone()
    };
    assert_eq!(m("big")["fit"]["verdict"], "no-fit", "{c}");
    assert_eq!(m("mock-dl")["fit"]["verdict"], "fits");
    assert_eq!(m("mock-dl")["installed"], false);
    assert_eq!(m("mock-dl")["remaining_bytes"], 1310720);

    // ---- the walk: model gate refuses until the download is done ----
    let (code, v) = post(p, "/api/setup/goto", json!({"step": "model"})).await;
    assert_eq!(code, 200, "{v}");
    assert!(cfg_path.exists(), "every step saves the config");
    let (code, _) = post(p, "/api/setup/model", json!({"id": "nope"})).await;
    assert_eq!(code, 400);
    let (code, v) = post(p, "/api/setup/model", json!({"id": "mock-dl"})).await;
    assert_eq!(code, 200, "{v}");
    assert_eq!(v["state"]["model"], "mock-dl");
    let (code, v) = post(p, "/api/setup/goto", json!({"step": "profile"})).await;
    assert_eq!(code, 409);
    assert!(
        v["error"]
            .as_str()
            .unwrap()
            .contains("Download the model first"),
        "{v}"
    );

    // the download over the API
    let (code, v) = post(p, "/api/downloads", json!({"model": "mock-dl"})).await;
    assert_eq!(code, 202, "{v}");
    let id = v["jobs"][0]["id"].as_u64().unwrap();
    let t0 = Instant::now();
    loop {
        let l = get(p, "/api/downloads").await;
        let j = l["jobs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|j| j["id"] == id)
            .unwrap()
            .clone();
        if j["state"] == "done" {
            break;
        }
        assert_ne!(j["state"], "failed", "{j}");
        assert!(t0.elapsed() < Duration::from_secs(30), "{j}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let (code, _) = post(p, "/api/downloads/999/pause", json!({})).await;
    assert_eq!(code, 409);
    assert_eq!(get(p, "/api/catalog").await["models"][0]["installed"], true);

    // forward to profile now composes the config from what is installed (the pack flag)
    let (code, v) = post(p, "/api/setup/goto", json!({"step": "profile"})).await;
    assert_eq!(code, 200, "{v}");
    let saved = Config::load(&cfg_path).unwrap().unwrap();
    assert_eq!(saved.setup.step, "profile");
    // a reload resumes here
    assert_eq!(get(p, "/api/setup").await["state"]["step"], "profile");
    let (code, _) = post(p, "/api/setup/goto", json!({"step": "network"})).await;
    assert_eq!(code, 409);
    let (code, v) = post(p, "/api/setup/profile", json!({"name": "Balanced"})).await;
    assert_eq!(code, 200, "{v}");
    let (code, _) = post(p, "/api/setup/goto", json!({"step": "network"})).await;
    assert_eq!(code, 200);

    // network: a non-loopback bind without a token is refused; a generated key is accepted
    let (code, v) = post(
        p,
        "/api/setup/network",
        json!({"engine_port": engine_port, "console_host": "0.0.0.0", "console_port": 8090}),
    )
    .await;
    assert_eq!(code, 400, "{v}");
    assert!(v["error"].as_str().unwrap().contains("admin token"), "{v}");
    let (_, k) = post(p, "/api/setup/keygen", json!({"kind": "api"})).await;
    let key = k["key"].as_str().unwrap().to_string();
    assert!(key.starts_with("sk-saber-"));
    let (_, a) = post(p, "/api/setup/keygen", json!({"kind": "admin"})).await;
    let admin = a["key"].as_str().unwrap().to_string();
    let (code, v) = post(
        p,
        "/api/setup/network",
        json!({"engine_port": engine_port, "console_host": "0.0.0.0", "console_port": 8090,
               "admin_token": admin, "keys": [{"label": "default", "key": key}]}),
    )
    .await;
    assert_eq!(code, 200, "{v}");
    assert_eq!(v["network"]["admin_token_set"], true);
    let saved = Config::load(&cfg_path).unwrap().unwrap();
    assert_eq!(saved.console.host, "0.0.0.0");
    assert_eq!(saved.console.port, 0, "the default port is stored as 0");
    // Stored as a salted hash, never as text
    assert!(saved.proxy.keys[0].key.is_empty());
    assert!(saved.proxy.keys[0].matches(&key));
    assert!(saved.admin().check(&admin));
    let on_disk = std::fs::read_to_string(&cfg_path).unwrap();
    assert!(
        !on_disk.contains(&key) && !on_disk.contains(&admin),
        "{on_disk}"
    );
    assert_eq!(
        std::fs::metadata(&cfg_path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    // The token checks at once (the SPA keeps the one it just sent)
    let (code, _) = post(p, "/api/setup/goto", json!({"step": "review"})).await;
    assert_eq!(code, 401, "the wizard's token is live");
    // back to loopback without a token (the rest of the test talks to it unauthenticated)
    let (code, v) = request_as(
        p,
        "POST",
        "/api/setup/network",
        &json!({"engine_port": engine_port, "admin_token": "", "keys": []}).to_string(),
        Some(&admin),
    )
    .await;
    assert_eq!(code, 200, "{v}");
    let (code, v) = post(p, "/api/setup/goto", json!({"step": "review"})).await;
    assert_eq!(code, 200, "{v}");

    // the delete guard: the configured pack is refused, a confirmation is required
    let (code, v) = request(p, "DELETE", "/api/packs/tiny-pack", "{}").await;
    assert_eq!(code, 400, "{v}");
    assert!(v["error"]
        .as_str()
        .unwrap()
        .contains("type the pack's name"));
    let (code, _) = request(p, "DELETE", "/api/packs/..", r#"{"confirm": ".."}"#).await;
    assert!(code == 400 || code == 404);
    // (the mock preset passes no --pack, so the downloaded pack is not "active": a second
    // pack named by a flag is)
    let mut c2 = Config::load(&cfg_path).unwrap().unwrap();
    c2.engine.flags.insert(
        "--pack".into(),
        saber_console::config::FlagValue::Str("/packs/tiny-pack".into()),
    );
    sup.set_config(c2.clone());
    let (code, v) = request(
        p,
        "DELETE",
        "/api/packs/tiny-pack",
        r#"{"confirm": "tiny-pack"}"#,
    )
    .await;
    assert_eq!(code, 409, "{v}");
    assert!(
        v["error"].as_str().unwrap().contains("configured to serve"),
        "{v}"
    );
    c2.engine.flags.remove("--pack");
    sup.set_config(c2);

    // review: finishing needs ready; launch boots the (fake) engine through the apply flow
    let (code, v) = post(p, "/api/setup/finish", json!({})).await;
    assert_eq!(code, 409, "{v}");
    let (code, v) = post(p, "/api/setup/launch", json!({})).await;
    assert_eq!(code, 202, "{v}");
    let t0 = Instant::now();
    loop {
        let a = get(p, "/api/apply").await;
        if a["job"]["state"] != "running" {
            assert_eq!(a["job"]["state"], "done", "{a}");
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(30), "{a}");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert_eq!(get(p, "/api/status").await["state"], "ready");
    let (code, v) = post(p, "/api/setup/finish", json!({})).await;
    assert_eq!(code, 200, "{v}");
    assert_eq!(v["needed"], false);
    assert!(Config::load(&cfg_path).unwrap().unwrap().setup.completed);

    // the running engine's pack (none for the mock) and an unconfigured pack deletes
    let (code, v) = request(
        p,
        "DELETE",
        "/api/packs/tiny-pack",
        r#"{"confirm": "tiny-pack"}"#,
    )
    .await;
    assert_eq!(code, 200, "{v}");
    assert!(v["freed_bytes"].as_u64().unwrap() > 1_000_000);
    assert!(!dir.path().join("nvme/saber-packs/tiny-pack").exists());

    // re-run from Settings, then skip
    let (_, v) = post(p, "/api/setup/rerun", json!({})).await;
    assert_eq!(v["needed"], true);
    assert_eq!(v["state"]["model"], "mock-dl", "the picks are kept");
    let (_, v) = post(p, "/api/setup/skip", json!({})).await;
    assert_eq!(v["needed"], false);
    assert_eq!(v["state"]["skipped"], true);

    sup.stop().await.unwrap();
}

/// The terminal wizard (`tools/saber-setup`) and the console share config.toml: what the
/// Python tool writes (migrating a legacy setup.json) parses and validates here, and what the
/// console writes reads back in the Python tool.
#[test]
fn the_terminal_wizard_and_the_console_share_config_toml() {
    let dir = saber_test_scratch::ScratchDir::new("console-pysetup");
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/saber-setup");
    let d = dir.path().display().to_string();
    let py = |code: &str| {
        let out = std::process::Command::new("python3")
            .arg("-I")
            .arg("-c")
            .arg(format!(
                "import importlib.machinery as m, importlib.util as u, json\n\
                 l = m.SourceFileLoader('s', {script:?})\n\
                 s = u.module_from_spec(u.spec_from_loader('s', l)); l.exec_module(s)\n\
                 s.CONF = {d:?} + '/setup.json'; s.CONSOLE_CONF = {d:?} + '/config.toml'\n{code}"
            ))
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    std::fs::write(
        dir.path().join("setup.json"),
        r#"{"model": "flashnext", "ctxfn": "131072", "thinking": false, "routes": true, "port": "8085"}"#,
    )
    .unwrap();
    py("s.save_cfg(s.load_cfg())");
    let c = Config::load(&dir.path().join("config.toml"))
        .unwrap()
        .unwrap();
    assert_eq!(c.engine.preset, saber_console::config::Preset::Flashnext);
    assert_eq!(c.engine.port, 8085);
    assert_eq!(
        c.engine.flags.get("--fn-max-ctx"),
        Some(&saber_console::config::FlagValue::Int(131072))
    );
    assert!(c.setup.completed);
    assert_eq!(c.setup.model, "flashnext");
    // the console edits it (a profile, a key); the Python tool reads the console's values
    let mut c2 = c.clone();
    c2.engine.flags.insert(
        "--fn-thinking".into(),
        saber_console::config::FlagValue::Str("on".into()),
    );
    c2.proxy.keys.push(saber_console::config::ProxyKey::new(
        "ops",
        "sk-saber-0123456789",
    ));
    c2.save(&dir.path().join("config.toml")).unwrap();
    let out = py(
        "c = s.load_cfg(); print(json.dumps([c['model'], c['thinking'], c['ctxfn'], c['port']]))",
    );
    assert_eq!(out.trim(), r#"["flashnext", true, "131072", "8085"]"#);
    // and a Python save keeps what it does not own
    py("c = s.load_cfg(); c['ctxfn'] = '65536'; s.save_cfg(c)");
    let c3 = Config::load(&dir.path().join("config.toml"))
        .unwrap()
        .unwrap();
    assert_eq!(c3.proxy.keys.len(), 1);
    assert_eq!(
        c3.engine.flags.get("--fn-max-ctx"),
        Some(&saber_console::config::FlagValue::Int(65536))
    );
    // the 27B on one card: the Python tool's TP=1 overrides load and validate here
    py("c = s.default_cfg('27b'); s.set_tp(c, '1'); s.save_cfg(c)");
    let c4 = Config::load(&dir.path().join("config.toml"))
        .unwrap()
        .unwrap();
    assert_eq!(c4.engine.preset, saber_console::config::Preset::Qwen27b);
    assert_eq!(
        c4.engine.flags.get("--engine-tp"),
        Some(&saber_console::config::FlagValue::Int(1))
    );
    assert_eq!(
        c4.engine.flags.get("--engine-peer-device"),
        Some(&saber_console::config::FlagValue::Bool(false))
    );
    c4.validate().unwrap();
}

/// The download list survives a console restart. A finished download comes back as
/// it was; a queued one comes back PAUSED (nothing starts by itself) and finishes when resumed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_download_list_survives_a_restart() {
    let dir = saber_test_scratch::ScratchDir::new("console-dl-persist");
    let srv = Arc::new(fake_pack_server());
    let port = start_server(srv.clone()).await;
    let tok_sha = sha_hex(&srv.files["tok.json"]);
    let cat = Catalog::parse(&catalog_json(port, &tok_sha)).unwrap();
    let roots = roots_in(dir.path());
    let data = dir.path().join("console");
    let d = Downloads::persistent(None, &data);
    tokio::spawn(d.clone().worker());
    let done = d.enqueue(spec(&cat, "mock-dl", &roots, port));
    let done = wait_state(
        &d,
        done.id,
        &[DlState::Done, DlState::Failed],
        Duration::from_secs(30),
    )
    .await;
    assert_eq!(done.state, DlState::Done, "{done:?}");
    // a second list (no worker: it stays queued), as a console that stopped with it queued
    let d2 = Downloads::persistent(None, &dir.path().join("console2"));
    let queued = d2.enqueue(spec(&cat, "loose", &roots, port));
    assert_eq!(queued.state, DlState::Queued);
    drop(d2);

    // "restart": new managers read the files
    let back = Downloads::persistent(None, &data);
    let j = back
        .get(done.id)
        .expect("the finished download is listed after a restart");
    assert_eq!((j.state, j.files_total), (DlState::Done, 3));
    assert_eq!(j.target, done.target);
    let back2 = Downloads::persistent(None, &dir.path().join("console2"));
    let q = back2.get(queued.id).expect("the queued download is listed");
    assert_eq!(
        q.state,
        DlState::Paused,
        "restored paused, not started: {q:?}"
    );
    assert!(
        q.error.as_deref().unwrap_or("").contains("restarted"),
        "{q:?}"
    );
    // new ids continue after the restored ones
    let j3 = back2.enqueue(spec(&cat, "mock-dl", &roots, port));
    assert!(j3.id > queued.id);
    back2.cancel(j3.id).unwrap();
    // a resume continues it
    tokio::spawn(back2.clone().worker());
    back2.resume(q.id).unwrap();
    let q = wait_state(
        &back2,
        q.id,
        &[DlState::Done, DlState::Failed],
        Duration::from_secs(30),
    )
    .await;
    assert_eq!(q.state, DlState::Done, "{q:?}");
    // clearing finished entries is persisted too
    back2.clear_finished();
    assert!(Downloads::persistent(None, &dir.path().join("console2"))
        .list()
        .is_empty());
}
