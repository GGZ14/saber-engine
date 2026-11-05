//! The replay (`docs/SHADOW.md` §1, mode R, steps 2–3): the recorded requests sent to a Saber
//! endpoint, byte for byte, with the answers stored beside them.
//!
//! The body is the recorded body, unchanged — same messages, same tools, same sampling
//! parameters, same seed — and so are the headers the record kept. Two paces:
//!
//! * `recorded` — each exchange is sent at its recorded arrival offset divided by `speed`,
//!   concurrency uncapped (the scheduler's `--engine-max-seqs` is the cap, as prod's
//!   `--max-num-seqs` was). The compression factor is written into the replay file;
//! * `sequential` — one at a time, in recorded order. Deterministic, for a gate.
//!
//! Around the whole replay it scrapes the target's `/metrics` (the `saber:` census families and
//! the serving families) and reads every card's `power1_cap`, so the reader can check stop conditions
//! 1 and 5 without a person looking at a log.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;

use crate::record::{unix_ms, Answer, Blob, Chunk, Exchange, ReplayMeta, Response};

/// How the replay is paced.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pace {
    /// At the recorded arrival offsets, divided by the speed (2.0 = twice as fast).
    Recorded {
        /// The compression factor.
        speed: f64,
    },
    /// One at a time, in recorded order.
    Sequential,
}

/// How a replay is set up.
#[derive(Clone, Debug)]
pub struct ReplayConfig {
    /// Saber's base URL (`http://127.0.0.1:8090` — never `:8080`).
    pub target: String,
    /// The pace.
    pub pace: Pace,
    /// `/sys/class/drm`, where `card*/device/hwmon/hwmon*/power1_cap` is read (`None`: skip).
    pub power_cap_root: Option<PathBuf>,
    /// Per-request timeout.
    pub timeout: Duration,
}

/// Replays `exchanges` against the target.
///
/// # Errors
/// The HTTP client cannot be built.
pub async fn replay(
    exchanges: &[Exchange],
    cfg: &ReplayConfig,
) -> anyhow::Result<(ReplayMeta, Vec<Answer>)> {
    let http = reqwest::Client::builder().timeout(cfg.timeout).build()?;
    let mut meta = ReplayMeta {
        target: cfg.target.clone(),
        pace: match cfg.pace {
            Pace::Recorded { .. } => "recorded".to_owned(),
            Pace::Sequential => "sequential".to_owned(),
        },
        speed: match cfg.pace {
            Pace::Recorded { speed } => speed,
            Pace::Sequential => 0.0,
        },
        ..ReplayMeta::default()
    };
    if let Some(root) = &cfg.power_cap_root {
        meta.power_caps_w.extend(power_caps(root, "before"));
    }
    meta.metrics_before = scrape(&http, &cfg.target).await;
    meta.started_unix_ms = unix_ms();
    let t0 = Instant::now();
    let first = exchanges.first().map_or(0.0, |e| e.t_mono_ms);

    let mut answers = Vec::with_capacity(exchanges.len());
    match cfg.pace {
        Pace::Sequential => {
            for x in exchanges {
                answers.push(send(&http, &cfg.target, x, t0).await);
            }
        }
        Pace::Recorded { speed } => {
            let speed = if speed > 0.0 { speed } else { 1.0 };
            let http = Arc::new(http.clone());
            let mut jobs = Vec::with_capacity(exchanges.len());
            for x in exchanges {
                let at = Duration::from_secs_f64(((x.t_mono_ms - first) / speed).max(0.0) / 1e3);
                let (http, target, x) = (Arc::clone(&http), cfg.target.clone(), x.clone());
                jobs.push(tokio::spawn(async move {
                    tokio::time::sleep_until(tokio::time::Instant::from_std(t0 + at)).await;
                    send(&http, &target, &x, t0).await
                }));
            }
            for j in jobs {
                answers.push(j.await?);
            }
        }
    }
    meta.replayed = answers.len() as u64;
    meta.metrics_after = scrape(&http, &cfg.target).await;
    if let Some(root) = &cfg.power_cap_root {
        meta.power_caps_w.extend(power_caps(root, "after"));
    }
    Ok((meta, answers))
}

async fn scrape(http: &reqwest::Client, target: &str) -> Option<String> {
    let r = http.get(format!("{target}/metrics")).send().await.ok()?;
    r.text().await.ok()
}

async fn send(http: &reqwest::Client, target: &str, x: &Exchange, t0: Instant) -> Answer {
    let sent = Instant::now();
    let t_sent_ms = sent.duration_since(t0).as_secs_f64() * 1e3;
    let method = reqwest::Method::from_bytes(x.method.as_bytes()).unwrap_or(reqwest::Method::POST);
    let mut req = http
        .request(method, format!("{target}{}", x.path))
        .body(x.body.bytes());
    for (k, v) in &x.headers {
        if k != "authorization" {
            req = req.header(k.as_str(), v.as_str());
        }
    }
    let mut response = Response::default();
    match req.send().await {
        Err(e) => {
            response.error = Some(e.to_string());
        }
        Ok(r) => {
            response.status = Some(r.status().as_u16());
            response.headers = r
                .headers()
                .iter()
                .map(|(n, v)| (n.to_string(), v.to_str().unwrap_or("").to_owned()))
                .collect();
            let mut stream = r.bytes_stream();
            response.complete = true;
            while let Some(item) = stream.next().await {
                let t = sent.elapsed().as_secs_f64() * 1e3;
                match item {
                    Ok(b) => {
                        response.t_first_byte_ms.get_or_insert(t);
                        response.chunks.push(Chunk {
                            t_ms: t,
                            data: Blob::of(&b),
                        });
                    }
                    Err(e) => {
                        response.complete = false;
                        response.error = Some(e.to_string());
                        break;
                    }
                }
            }
        }
    }
    response.t_done_ms = Some(sent.elapsed().as_secs_f64() * 1e3);
    Answer {
        id: x.id,
        t_sent_ms,
        response,
    }
}

/// Every `card*/device/hwmon/hwmon*/power1_cap` under `root`, in watts, labelled `<when>:<card>`.
pub fn power_caps(root: &Path, when: &str) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    let Ok(cards) = std::fs::read_dir(root) else {
        return out;
    };
    let mut cards: Vec<_> = cards.filter_map(Result::ok).collect();
    cards.sort_by_key(std::fs::DirEntry::file_name);
    for card in cards {
        let name = card.file_name().to_string_lossy().into_owned();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let Ok(hwmons) = std::fs::read_dir(card.path().join("device/hwmon")) else {
            continue;
        };
        for h in hwmons.filter_map(Result::ok) {
            if let Ok(s) = std::fs::read_to_string(h.path().join("power1_cap")) {
                if let Ok(uw) = s.trim().parse::<f64>() {
                    out.push((format!("{when}:{name}"), uw / 1e6));
                }
            }
        }
    }
    out
}
