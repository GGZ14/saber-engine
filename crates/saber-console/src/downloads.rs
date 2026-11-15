//! The download manager (`/api/downloads`): fetch a catalog component (a pack directory, a
//! checkpoint, loose tokenizer files) into place with resume, progress / speed / ETA, sha256
//! verification, a disk-space precheck and an atomic rename, one download at a time.
//!
//! The same job as `tools/fetch_packs.sh` (which stays as the standalone CLI path, through the
//! `hf` CLI), reimplemented so the console needs no Python package: files are listed with the
//! Hub's tree API (sizes and LFS sha256 come from it) and fetched with `curl` (`-C -` resumes a
//! `.part` file; TLS, redirects to the CDN and retries are curl's). Everything lands in a hidden
//! staging directory on the destination's own file system and is renamed into place only when
//! every file is complete, verified and the pack checks pass, so a half-downloaded pack is never
//! visible to the engine or the packs listing. Pausing kills curl and keeps the `.part` files;
//! the staging path is deterministic, so a later start (even after a console restart) resumes.
//! No GPU is touched.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, watch, Notify};

use crate::catalog::{Component, Layout, Roots};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DlState {
    Queued,
    Listing,
    Downloading,
    Verifying,
    Installing,
    Paused,
    Done,
    Failed,
    Cancelled,
}

impl DlState {
    /// Still in the queue or the worker (not finished).
    #[must_use]
    pub fn active(self) -> bool {
        matches!(
            self,
            DlState::Queued
                | DlState::Listing
                | DlState::Downloading
                | DlState::Verifying
                | DlState::Installing
                | DlState::Paused
        )
    }
}

/// One download, as the API shows it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Job {
    pub id: u64,
    pub model: String,
    pub model_name: String,
    pub component: String,
    pub label: String,
    /// Where it lands (`<packs>/<name>`, or the root for loose files).
    pub target: String,
    pub state: DlState,
    pub total_bytes: Option<u64>,
    pub done_bytes: u64,
    pub speed_bps: Option<f64>,
    pub eta_s: Option<f64>,
    pub files_total: usize,
    pub files_done: usize,
    pub current_file: Option<String>,
    /// A plain-English reason (failed) or note.
    pub error: Option<String>,
    pub created_ms: u64,
    pub started_ms: Option<u64>,
    pub finished_ms: Option<u64>,
}

/// What a download fetches.
#[derive(Clone, Debug)]
pub struct Spec {
    pub model: String,
    pub model_name: String,
    pub component: Component,
    pub roots: Roots,
    pub hf_endpoint: String,
    /// A Hugging Face token (sent as a header from a 0600 file, never on the command line).
    pub token: Option<String>,
    pub curl: PathBuf,
    pub reserve_bytes: u64,
}

impl Spec {
    fn target(&self) -> PathBuf {
        self.component.target(&self.roots)
    }
    fn staging(&self) -> PathBuf {
        self.component.staging(&self.roots, &self.model)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ctl {
    Run,
    Pause,
    Cancel,
}

struct Entry {
    job: Job,
    spec: Spec,
    ctl: watch::Sender<Ctl>,
}

/// Free bytes at a path (injectable: the low-disk test passes its own).
pub type FreeSpace = Arc<dyn Fn(&Path) -> Option<u64> + Send + Sync>;

pub struct Downloads {
    entries: Mutex<Vec<Entry>>,
    seq: AtomicU64,
    wake: Notify,
    tx: broadcast::Sender<Vec<Job>>,
    free_space: FreeSpace,
    /// The list survives a console restart (`<data dir>/downloads.json`).
    store: Option<PathBuf>,
    /// The last persisted (id, state, files done) list: progress ticks do not rewrite the file.
    persisted: Mutex<Vec<(u64, DlState, usize)>>,
}

/// One persisted download: the job and what it fetches (never the Hugging Face token: it is
/// read from its file again when the download is resumed).
#[derive(Serialize, Deserialize)]
struct Saved {
    job: Job,
    model: String,
    model_name: String,
    component: crate::catalog::Component,
    roots: crate::catalog::Roots,
    hf_endpoint: String,
    curl: PathBuf,
    reserve_bytes: u64,
}

/// One resolved file.
#[derive(Clone, Debug, PartialEq)]
pub struct RFile {
    pub rel: String,
    pub url: String,
    pub size: Option<u64>,
    pub sha256: Option<String>,
}

#[derive(Debug)]
enum Stop {
    Paused,
    Cancelled,
    Failed(String),
}

impl Downloads {
    #[must_use]
    pub fn new(free_space: Option<FreeSpace>) -> Arc<Downloads> {
        let (tx, _) = broadcast::channel(64);
        Arc::new(Downloads {
            entries: Mutex::new(Vec::new()),
            seq: AtomicU64::new(0),
            wake: Notify::new(),
            tx,
            free_space: free_space
                .unwrap_or_else(|| Arc::new(|p: &Path| crate::system::disk_space(p).1)),
            store: None,
            persisted: Mutex::new(Vec::new()),
        })
    }

    /// A download list kept in `<data_dir>/downloads.json`: what it held when the
    /// console last stopped comes back, a download that was running or queued as PAUSED (its
    /// `.part` files are kept, so a resume continues where it was; nothing starts by itself).
    #[must_use]
    pub fn persistent(free_space: Option<FreeSpace>, data_dir: &Path) -> Arc<Downloads> {
        let path = data_dir.join("downloads.json");
        let (tx, _) = broadcast::channel(64);
        let saved: Vec<Saved> = std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        let mut max_id = 0;
        let entries = saved
            .into_iter()
            .map(|sv| {
                let mut job = sv.job;
                max_id = max_id.max(job.id);
                if job.state.active() && job.state != DlState::Paused {
                    job.state = DlState::Paused;
                    job.error = Some("the console restarted: resume to continue".into());
                }
                job.speed_bps = None;
                job.eta_s = None;
                let (ctl, _) = watch::channel(Ctl::Run);
                Entry {
                    job,
                    spec: Spec {
                        model: sv.model,
                        model_name: sv.model_name,
                        component: sv.component,
                        roots: sv.roots,
                        hf_endpoint: sv.hf_endpoint,
                        token: None,
                        curl: sv.curl,
                        reserve_bytes: sv.reserve_bytes,
                    },
                    ctl,
                }
            })
            .collect();
        Arc::new(Downloads {
            entries: Mutex::new(entries),
            seq: AtomicU64::new(max_id),
            wake: Notify::new(),
            tx,
            free_space: free_space
                .unwrap_or_else(|| Arc::new(|p: &Path| crate::system::disk_space(p).1)),
            store: Some(path),
            persisted: Mutex::new(Vec::new()),
        })
    }

    /// Write the list when an entry was added / removed or changed state (not on progress).
    fn persist(&self) {
        let Some(path) = &self.store else {
            return;
        };
        let (sig, saved): (Vec<_>, Vec<Saved>) = {
            let g = self.lock();
            (
                g.iter()
                    .map(|e| (e.job.id, e.job.state, e.job.files_done))
                    .collect(),
                g.iter()
                    .map(|e| Saved {
                        job: e.job.clone(),
                        model: e.spec.model.clone(),
                        model_name: e.spec.model_name.clone(),
                        component: e.spec.component.clone(),
                        roots: e.spec.roots.clone(),
                        hf_endpoint: e.spec.hf_endpoint.clone(),
                        curl: e.spec.curl.clone(),
                        reserve_bytes: e.spec.reserve_bytes,
                    })
                    .collect(),
            )
        };
        {
            let mut p = self.persisted.lock().unwrap_or_else(|e| e.into_inner());
            if *p == sig {
                return;
            }
            *p = sig;
        }
        let write = || -> std::io::Result<()> {
            if let Some(d) = path.parent() {
                std::fs::create_dir_all(d)?;
            }
            let tmp = path.with_extension("json.tmp");
            std::fs::write(&tmp, serde_json::to_vec_pretty(&saved)?)?;
            std::fs::rename(&tmp, path)
        };
        if let Err(e) = write() {
            eprintln!("saber ui: saving {}: {e}", path.display());
        }
    }

    /// A restored download has no token: re-read it from the config's file on resume.
    pub fn set_token(&self, id: u64, token: Option<String>) {
        if let Some(e) = self.lock().iter_mut().find(|e| e.job.id == id) {
            if e.spec.token.is_none() {
                e.spec.token = token;
            }
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Entry>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[must_use]
    pub fn list(&self) -> Vec<Job> {
        self.lock().iter().map(|e| e.job.clone()).collect()
    }

    #[must_use]
    pub fn get(&self, id: u64) -> Option<Job> {
        self.lock()
            .iter()
            .find(|e| e.job.id == id)
            .map(|e| e.job.clone())
    }

    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<Vec<Job>> {
        self.tx.subscribe()
    }

    fn publish(&self) {
        self.persist();
        let _ = self.tx.send(self.list());
    }

    fn update(&self, id: u64, f: impl FnOnce(&mut Job)) {
        {
            let mut g = self.lock();
            if let Some(e) = g.iter_mut().find(|e| e.job.id == id) {
                f(&mut e.job);
            }
        }
        self.publish();
    }

    /// True while a download for `target` is queued, running or paused.
    #[must_use]
    pub fn busy_with(&self, target: &Path) -> bool {
        let t = target.display().to_string();
        self.lock()
            .iter()
            .any(|e| e.job.state.active() && e.job.target == t)
    }

    /// Queue `spec` (the same target already in the queue returns that job).
    pub fn enqueue(&self, spec: Spec) -> Job {
        let target = spec.target().display().to_string();
        let job = {
            let mut g = self.lock();
            if let Some(e) = g.iter_mut().find(|e| {
                e.job.state.active()
                    && e.job.target == target
                    && e.job.component == spec.component.id
            }) {
                if e.job.state == DlState::Paused {
                    e.job.state = DlState::Queued;
                    e.ctl.send_replace(Ctl::Run);
                }
                e.job.clone()
            } else {
                let id = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
                let (ctl, _) = watch::channel(Ctl::Run);
                let job = Job {
                    id,
                    model: spec.model.clone(),
                    model_name: spec.model_name.clone(),
                    component: spec.component.id.clone(),
                    label: if spec.component.label.is_empty() {
                        spec.component.id.clone()
                    } else {
                        spec.component.label.clone()
                    },
                    target,
                    state: DlState::Queued,
                    total_bytes: (spec.component.approx_bytes > 0)
                        .then_some(spec.component.approx_bytes),
                    done_bytes: 0,
                    speed_bps: None,
                    eta_s: None,
                    files_total: 0,
                    files_done: 0,
                    current_file: None,
                    error: None,
                    created_ms: crate::now_ms(),
                    started_ms: None,
                    finished_ms: None,
                };
                g.push(Entry {
                    job: job.clone(),
                    spec,
                    ctl,
                });
                job
            }
        };
        self.publish();
        self.wake.notify_one();
        job
    }

    /// Pause a queued or running download (its `.part` files are kept).
    pub fn pause(&self, id: u64) -> anyhow::Result<Job> {
        let j = {
            let mut g = self.lock();
            let e = g
                .iter_mut()
                .find(|e| e.job.id == id)
                .ok_or_else(|| anyhow::anyhow!("no download #{id}"))?;
            match e.job.state {
                DlState::Queued => e.job.state = DlState::Paused,
                DlState::Listing | DlState::Downloading | DlState::Verifying => {
                    e.ctl.send_replace(Ctl::Pause);
                }
                DlState::Paused => {}
                s => anyhow::bail!("download #{id} is {s:?}: nothing to pause"),
            }
            e.job.clone()
        };
        self.publish();
        Ok(j)
    }

    /// Resume a paused download, or retry a failed / cancelled one (partial files are reused).
    pub fn resume(&self, id: u64) -> anyhow::Result<Job> {
        let j = {
            let mut g = self.lock();
            let e = g
                .iter_mut()
                .find(|e| e.job.id == id)
                .ok_or_else(|| anyhow::anyhow!("no download #{id}"))?;
            match e.job.state {
                DlState::Paused | DlState::Failed | DlState::Cancelled => {
                    e.job.state = DlState::Queued;
                    e.job.error = None;
                    e.job.finished_ms = None;
                    e.ctl.send_replace(Ctl::Run);
                }
                s => anyhow::bail!("download #{id} is {s:?}"),
            }
            e.job.clone()
        };
        self.publish();
        self.wake.notify_one();
        Ok(j)
    }

    /// Cancel: stop it and delete what it downloaded so far.
    pub fn cancel(&self, id: u64) -> anyhow::Result<Job> {
        let (j, staging) = {
            let mut g = self.lock();
            let e = g
                .iter_mut()
                .find(|e| e.job.id == id)
                .ok_or_else(|| anyhow::anyhow!("no download #{id}"))?;
            let staging = match e.job.state {
                DlState::Queued | DlState::Paused | DlState::Failed => {
                    e.job.state = DlState::Cancelled;
                    e.job.finished_ms = Some(crate::now_ms());
                    e.job.speed_bps = None;
                    e.job.eta_s = None;
                    Some(e.spec.staging())
                }
                DlState::Listing | DlState::Downloading | DlState::Verifying => {
                    // the worker deletes the staging dir when curl is gone
                    e.ctl.send_replace(Ctl::Cancel);
                    None
                }
                s => anyhow::bail!("download #{id} is {s:?}: nothing to cancel"),
            };
            (e.job.clone(), staging)
        };
        if let Some(s) = staging {
            let _ = std::fs::remove_dir_all(s);
        }
        self.publish();
        Ok(j)
    }

    /// Forget finished entries (done / failed / cancelled).
    pub fn clear_finished(&self) {
        self.lock().retain(|e| e.job.state.active());
        self.publish();
    }

    fn next_queued(&self) -> Option<(u64, Spec, watch::Receiver<Ctl>)> {
        let mut g = self.lock();
        let e = g.iter_mut().find(|e| e.job.state == DlState::Queued)?;
        e.job.state = DlState::Listing;
        e.job.started_ms.get_or_insert(crate::now_ms());
        e.ctl.send_replace(Ctl::Run);
        Some((e.job.id, e.spec.clone(), e.ctl.subscribe()))
    }

    /// The single-flight worker: run queued downloads one at a time, forever.
    pub async fn worker(self: Arc<Self>) {
        loop {
            let notified = self.wake.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let Some((id, spec, ctl)) = self.next_queued() else {
                notified.await;
                continue;
            };
            self.publish();
            let r = self.run(id, &spec, ctl).await;
            let now = crate::now_ms();
            match r {
                Ok(()) => self.update(id, |j| {
                    j.state = DlState::Done;
                    j.finished_ms = Some(now);
                    j.speed_bps = None;
                    j.eta_s = Some(0.0);
                    j.current_file = None;
                    if let Some(t) = j.total_bytes {
                        j.done_bytes = t;
                    }
                }),
                Err(Stop::Paused) => self.update(id, |j| {
                    j.state = DlState::Paused;
                    j.speed_bps = None;
                    j.eta_s = None;
                }),
                Err(Stop::Cancelled) => {
                    let _ = tokio::fs::remove_dir_all(spec.staging()).await;
                    self.update(id, |j| {
                        j.state = DlState::Cancelled;
                        j.finished_ms = Some(now);
                        j.speed_bps = None;
                        j.eta_s = None;
                        j.current_file = None;
                    });
                }
                Err(Stop::Failed(msg)) => self.update(id, |j| {
                    j.state = DlState::Failed;
                    j.error = Some(msg);
                    j.finished_ms = Some(now);
                    j.speed_bps = None;
                    j.eta_s = None;
                }),
            }
        }
    }

    async fn run(&self, id: u64, spec: &Spec, mut ctl: watch::Receiver<Ctl>) -> Result<(), Stop> {
        let staging = spec.staging();
        let target = spec.target();
        if spec.component.layout == Layout::Dir && target.exists() {
            return Err(Stop::Failed(format!(
                "{} already exists. Delete it in the model manager first if you want to download \
                 it again.",
                target.display()
            )));
        }
        let root = spec.roots.root(spec.component.dest).to_path_buf();
        tokio::fs::create_dir_all(&staging).await.map_err(|e| {
            Stop::Failed(format!(
                "Cannot create {}: {e}. Check that the drive is mounted and writable.",
                staging.display()
            ))
        })?;
        let hdr = write_header_file(&staging, spec.token.as_deref())
            .map_err(|e| Stop::Failed(format!("writing the auth header file: {e}")))?;
        let curl = if spec.curl.as_os_str().is_empty() {
            PathBuf::from("curl")
        } else {
            spec.curl.clone()
        };

        // 1. the file list
        let files = resolve_files(spec, &curl, hdr.as_deref(), &mut ctl).await?;
        if files.is_empty() {
            return Err(Stop::Failed(format!(
                "The source lists no files for {} (wrong repo path in the catalog?)",
                spec.component.id
            )));
        }
        let total: u64 = files.iter().filter_map(|f| f.size).sum();
        let unknown = files.iter().any(|f| f.size.is_none());

        // 2. the disk-space precheck (what is still missing + the reserve)
        let present: u64 = files
            .iter()
            .map(|f| {
                let fin = staging.join(&f.rel);
                if f.size.is_some_and(|s| file_len(&fin) == Some(s)) {
                    f.size.unwrap_or(0)
                } else {
                    file_len(&part_of(&fin))
                        .unwrap_or(0)
                        .min(f.size.unwrap_or(u64::MAX))
                }
            })
            .sum();
        let need = total.saturating_sub(present);
        if let Some(free) = (self.free_space)(&root) {
            if free < need.saturating_add(spec.reserve_bytes) {
                return Err(Stop::Failed(format!(
                    "Not enough disk space on {}: the download needs {} more (plus {} kept \
                     free), and {} is free. Free up space, or point the NVMe path at a bigger \
                     drive in Settings.",
                    root.display(),
                    human(need),
                    human(spec.reserve_bytes),
                    human(free)
                )));
            }
        }
        self.update(id, |j| {
            j.total_bytes = (!unknown || total > 0).then_some(total);
            j.files_total = files.len();
            j.done_bytes = present;
            j.state = DlState::Downloading;
        });

        // 3. every file: resume its .part, verify, move it into the staging tree
        let mut done_before: u64 = 0;
        let mut speed = Speed::new(present);
        for (i, f) in files.iter().enumerate() {
            let fin = staging.join(&f.rel);
            if f.size.is_some_and(|s| file_len(&fin) == Some(s)) {
                done_before += f.size.unwrap_or(0);
                self.update(id, |j| j.files_done = i + 1);
                continue;
            }
            if let Some(dir) = fin.parent() {
                tokio::fs::create_dir_all(dir)
                    .await
                    .map_err(|e| Stop::Failed(format!("creating {}: {e}", dir.display())))?;
            }
            let part = part_of(&fin);
            let have = file_len(&part);
            let complete = f.size.is_some() && have == f.size;
            if !complete {
                if f.size.is_none() || have.zip(f.size).is_some_and(|(h, s)| h > s) {
                    // no size to resume against (or a part larger than the file): start over
                    let _ = tokio::fs::remove_file(&part).await;
                }
                self.update(id, |j| {
                    j.current_file = Some(f.rel.clone());
                    j.state = DlState::Downloading;
                });
                self.fetch(
                    id,
                    &curl,
                    hdr.as_deref(),
                    f,
                    &part,
                    done_before,
                    &mut speed,
                    &mut ctl,
                )
                .await?;
            }
            // verify
            let got = file_len(&part).unwrap_or(0);
            if let Some(s) = f.size {
                if got != s {
                    let _ = tokio::fs::remove_file(&part).await;
                    return Err(Stop::Failed(format!(
                        "{} arrived with {} bytes, the source says {s}. The partial file was \
                         deleted; press Retry to download it again.",
                        f.rel, got
                    )));
                }
            }
            if let Some(want) = &f.sha256 {
                self.update(id, |j| {
                    j.state = DlState::Verifying;
                    j.current_file = Some(f.rel.clone());
                    j.speed_bps = None;
                });
                let p = part.clone();
                let sum = tokio::task::spawn_blocking(move || sha256_file(&p))
                    .await
                    .map_err(|e| Stop::Failed(format!("verifying: {e}")))?
                    .map_err(|e| Stop::Failed(format!("reading {}: {e}", part.display())))?;
                if *ctl.borrow() == Ctl::Cancel {
                    return Err(Stop::Cancelled);
                }
                if !sum.eq_ignore_ascii_case(want) {
                    let _ = tokio::fs::remove_file(&part).await;
                    return Err(Stop::Failed(format!(
                        "{} is corrupt: its sha256 is {sum}, the source says {want}. The file \
                         was deleted; press Retry to download it again.",
                        f.rel
                    )));
                }
            }
            tokio::fs::rename(&part, &fin)
                .await
                .map_err(|e| Stop::Failed(format!("moving {}: {e}", f.rel)))?;
            done_before += got;
            self.update(id, |j| {
                j.files_done = i + 1;
                j.done_bytes = done_before;
                j.state = DlState::Downloading;
            });
            match *ctl.borrow() {
                Ctl::Cancel => return Err(Stop::Cancelled),
                Ctl::Pause => return Err(Stop::Paused),
                Ctl::Run => {}
            }
        }

        // 4. the pack checks, then into place
        self.update(id, |j| {
            j.state = DlState::Installing;
            j.current_file = None;
            j.speed_bps = None;
        });
        if let Some(h) = &hdr {
            let _ = std::fs::remove_file(h);
        }
        match spec.component.layout {
            Layout::Dir => {
                let bad = spec.component.verify_tree(&staging);
                if !bad.is_empty() {
                    return Err(Stop::Failed(format!(
                        "The download finished but the pack does not check out: {}. It was \
                         kept in {} (not installed).",
                        bad.join("; "),
                        staging.display()
                    )));
                }
                std::fs::rename(&staging, &target).map_err(|e| {
                    Stop::Failed(format!(
                        "moving {} into place at {}: {e}",
                        staging.display(),
                        target.display()
                    ))
                })?;
            }
            Layout::Files => {
                for f in &files {
                    std::fs::rename(staging.join(&f.rel), root.join(&f.rel)).map_err(|e| {
                        Stop::Failed(format!("moving {} into {}: {e}", f.rel, root.display()))
                    })?;
                }
                let _ = std::fs::remove_dir_all(&staging);
            }
        }
        Ok(())
    }

    /// One curl run for one file, with progress ticks; returns when curl exits or on pause /
    /// cancel (curl is killed, the `.part` stays).
    #[allow(clippy::too_many_arguments)]
    async fn fetch(
        &self,
        id: u64,
        curl: &Path,
        hdr: Option<&Path>,
        f: &RFile,
        part: &Path,
        done_before: u64,
        speed: &mut Speed,
        ctl: &mut watch::Receiver<Ctl>,
    ) -> Result<(), Stop> {
        let mut cmd = tokio::process::Command::new(curl);
        cmd.args(curl_common(hdr));
        if file_len(part).is_some_and(|l| l > 0) {
            cmd.args(["-C", "-"]);
        }
        cmd.arg("-o")
            .arg(part)
            .arg(&f.url)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = cmd.spawn().map_err(|e| {
            Stop::Failed(format!(
                "Cannot run {} ({e}). Install curl (`sudo dnf install curl` or `sudo apt \
                 install curl`).",
                curl.display()
            ))
        })?;
        let mut stderr = child.stderr.take();
        let err_task = tokio::spawn(async move {
            let mut s = String::new();
            if let Some(e) = stderr.as_mut() {
                use tokio::io::AsyncReadExt;
                let _ = e.read_to_string(&mut s).await;
            }
            s
        });
        let mut tick = tokio::time::interval(Duration::from_millis(400));
        loop {
            tokio::select! {
                st = child.wait() => {
                    let st = st.map_err(|e| Stop::Failed(format!("curl: {e}")))?;
                    let err = err_task.await.unwrap_or_default();
                    if st.success() {
                        return Ok(());
                    }
                    // a pause / cancel that raced the exit
                    match *ctl.borrow() {
                        Ctl::Cancel => return Err(Stop::Cancelled),
                        Ctl::Pause => return Err(Stop::Paused),
                        Ctl::Run => {}
                    }
                    return Err(Stop::Failed(curl_error(st.code(), &err, &f.url)));
                }
                _ = tick.tick() => {
                    let now = done_before + file_len(part).unwrap_or(0);
                    let (bps, eta) = speed.sample(now, self.get(id).and_then(|j| j.total_bytes));
                    self.update(id, |j| {
                        j.done_bytes = now;
                        j.speed_bps = bps;
                        j.eta_s = eta;
                    });
                }
                r = ctl.changed() => {
                    if r.is_err() {
                        continue;
                    }
                    let c = *ctl.borrow();
                    if c != Ctl::Run {
                        let _ = child.kill().await;
                        let _ = child.wait().await;
                        let now = done_before + file_len(part).unwrap_or(0);
                        self.update(id, |j| j.done_bytes = now);
                        return Err(if c == Ctl::Pause { Stop::Paused } else { Stop::Cancelled });
                    }
                }
            }
        }
    }
}

/// Bytes/s as an exponential moving average, and the ETA.
struct Speed {
    last: Option<(Instant, u64)>,
    ema: Option<f64>,
}

impl Speed {
    fn new(start: u64) -> Speed {
        Speed {
            last: Some((Instant::now(), start)),
            ema: None,
        }
    }

    fn sample(&mut self, now_bytes: u64, total: Option<u64>) -> (Option<f64>, Option<f64>) {
        let t = Instant::now();
        if let Some((t0, b0)) = self.last {
            let dt = t.duration_since(t0).as_secs_f64();
            if dt > 0.05 {
                let r = now_bytes.saturating_sub(b0) as f64 / dt;
                self.ema = Some(self.ema.map_or(r, |e| e * 0.7 + r * 0.3));
                self.last = Some((t, now_bytes));
            }
        } else {
            self.last = Some((t, now_bytes));
        }
        let eta = match (self.ema, total) {
            (Some(s), Some(tot)) if s > 1.0 => Some(tot.saturating_sub(now_bytes) as f64 / s),
            _ => None,
        };
        (self.ema, eta)
    }
}

fn part_of(fin: &Path) -> PathBuf {
    let mut s = fin.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

fn file_len(p: &Path) -> Option<u64> {
    std::fs::metadata(p)
        .ok()
        .filter(|m| m.is_file())
        .map(|m| m.len())
}

/// `1.2 GB`.
#[must_use]
pub fn human(b: u64) -> String {
    let f = b as f64;
    if f >= 1e9 {
        format!("{:.1} GB", f / 1e9)
    } else if f >= 1e6 {
        format!("{:.1} MB", f / 1e6)
    } else if f >= 1e3 {
        format!("{:.0} kB", f / 1e3)
    } else {
        format!("{b} bytes")
    }
}

/// The sha256 of a file, hex.
pub fn sha256_file(p: &Path) -> std::io::Result<String> {
    use sha2::Digest;
    use std::io::Read;
    let mut f = std::fs::File::open(p)?;
    let mut h = sha2::Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex::encode(h.finalize()))
}

/// The Authorization header in a 0600 file curl reads with `-H @file` (keeps the token off the
/// command line, where `ps` shows it).
fn write_header_file(dir: &Path, token: Option<&str>) -> std::io::Result<Option<PathBuf>> {
    let Some(t) = token.filter(|t| !t.trim().is_empty()) else {
        return Ok(None);
    };
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let p = dir.join(".auth-header");
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&p)?;
    writeln!(f, "Authorization: Bearer {}", t.trim())?;
    Ok(Some(p))
}

fn curl_common(hdr: Option<&Path>) -> Vec<String> {
    let mut a: Vec<String> = [
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "20",
        "--retry",
        "3",
        "--retry-delay",
        "2",
        // abort a stalled transfer (under 1 kB/s for two minutes) instead of hanging forever
        "--speed-limit",
        "1024",
        "--speed-time",
        "120",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    if let Some(h) = hdr {
        a.push("-H".into());
        a.push(format!("@{}", h.display()));
    }
    a
}

/// curl's exit code in plain English.
#[must_use]
pub fn curl_error(code: Option<i32>, stderr: &str, url: &str) -> String {
    let host = url
        .split("://")
        .nth(1)
        .and_then(|r| r.split('/').next())
        .unwrap_or(url);
    let detail = stderr.trim().lines().last().unwrap_or("").trim();
    let what = match code {
        Some(6) => format!("Could not find the server {host}: is this machine online?"),
        Some(7) => {
            format!("Could not connect to {host}: the server is down or a firewall blocks it.")
        }
        Some(22) if detail.contains("401") || detail.contains("403") => {
            "The server refused the download (not authorized). If the repo is private, put a \
             Hugging Face token in a file and set downloads.hf_token_file in Settings."
                .to_string()
        }
        Some(22) if detail.contains("404") => {
            "The file is not on the server (404): the catalog may name a repo or path that \
             moved."
                .to_string()
        }
        Some(22) => "The server answered with an error.".to_string(),
        Some(23) => "Could not write the file: is the disk full?".to_string(),
        Some(28) => "The download timed out or stalled. Check the network and press Retry: it \
                     resumes where it stopped."
            .to_string(),
        Some(33) => "The server does not support resuming. Cancel and start again.".to_string(),
        Some(35 | 60) => format!("A secure connection to {host} failed (TLS)."),
        Some(52 | 55 | 56 | 18) => "The connection dropped. Press Retry: it resumes where it \
                                    stopped."
            .to_string(),
        Some(c) => format!("curl exited with code {c}."),
        None => "curl was killed.".to_string(),
    };
    if detail.is_empty() {
        what
    } else {
        format!("{what} ({detail})")
    }
}

async fn run_curl_capture(
    curl: &Path,
    args: &[String],
    ctl: &mut watch::Receiver<Ctl>,
) -> Result<(String, String), Stop> {
    let mut cmd = tokio::process::Command::new(curl);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let child = cmd.spawn().map_err(|e| {
        Stop::Failed(format!(
            "Cannot run {} ({e}). Install curl (`sudo dnf install curl` or `sudo apt install \
             curl`).",
            curl.display()
        ))
    })?;
    let out = child.wait_with_output();
    tokio::pin!(out);
    loop {
        tokio::select! {
            o = &mut out => {
                let o = o.map_err(|e| Stop::Failed(format!("curl: {e}")))?;
                let stdout = String::from_utf8_lossy(&o.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&o.stderr).into_owned();
                if !o.status.success() {
                    let url = args.last().cloned().unwrap_or_default();
                    return Err(Stop::Failed(format!(
                        "Listing the files failed: {}",
                        curl_error(o.status.code(), &stderr, &url)
                    )));
                }
                return Ok((stdout, stderr));
            }
            r = ctl.changed() => {
                if r.is_err() { continue; }
                match *ctl.borrow() {
                    Ctl::Cancel => return Err(Stop::Cancelled),
                    Ctl::Pause => return Err(Stop::Paused),
                    Ctl::Run => {}
                }
            }
        }
    }
}

/// Percent-encode a repo path segment by segment (keeps `/`).
fn enc_path(p: &str) -> String {
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

/// One page of the Hub tree API (`[{type, path, size, lfs: {oid}}]`) -> files.
#[must_use]
pub fn parse_tree(json: &str) -> Vec<(String, Option<u64>, Option<String>)> {
    let v: serde_json::Value = serde_json::from_str(json).unwrap_or(serde_json::Value::Null);
    v.as_array()
        .map(|a| {
            a.iter()
                .filter(|e| e["type"] == "file")
                .filter_map(|e| {
                    let path = e["path"].as_str()?.to_string();
                    let size = e["lfs"]["size"].as_u64().or_else(|| e["size"].as_u64());
                    let sha = e["lfs"]["oid"]
                        .as_str()
                        .or_else(|| e["lfs"]["sha256"].as_str())
                        .map(str::to_string);
                    Some((path, size, sha))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// `Link: <url>; rel="next"` from a header dump.
fn next_link(headers: &str) -> Option<String> {
    headers.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        if !k.trim().eq_ignore_ascii_case("link") {
            return None;
        }
        v.split(',').find_map(|part| {
            let part = part.trim();
            if !part.contains("rel=\"next\"") {
                return None;
            }
            Some(
                part.split('>')
                    .next()?
                    .trim_start_matches('<')
                    .trim()
                    .to_string(),
            )
        })
    })
}

/// The component's files: its explicit list (sizes / hashes filled from the Hub where the
/// catalog has none), or the Hub listing of its repo path.
async fn resolve_files(
    spec: &Spec,
    curl: &Path,
    hdr: Option<&Path>,
    ctl: &mut watch::Receiver<Ctl>,
) -> Result<Vec<RFile>, Stop> {
    let c = &spec.component;
    let src = &c.source;
    let ep = spec.hf_endpoint.trim_end_matches('/');
    let prefix = src.path.trim_matches('/');
    let resolve = |rel_in_repo: &str| {
        format!(
            "{ep}/{}/resolve/{}/{}",
            src.hf_repo,
            enc_path(&src.revision),
            enc_path(rel_in_repo)
        )
    };
    let join = |rel: &str| {
        if prefix.is_empty() {
            rel.to_string()
        } else {
            format!("{prefix}/{rel}")
        }
    };
    let needs_listing = !src.hf_repo.is_empty()
        && (c.files.is_empty()
            || c.files
                .iter()
                .any(|f| f.size.is_none() || f.sha256.is_none()));
    let mut listed: Vec<(String, Option<u64>, Option<String>)> = Vec::new();
    if needs_listing {
        let recursive = c.files.is_empty();
        let mut url = format!(
            "{ep}/api/models/{}/tree/{}{}{}",
            src.hf_repo,
            enc_path(&src.revision),
            if prefix.is_empty() { "" } else { "/" },
            enc_path(prefix)
        );
        if recursive {
            url.push_str("?recursive=true");
        }
        let hdump = spec.staging().join(".listing-headers");
        for _page in 0..1000 {
            let mut args = curl_common(hdr);
            args.push("-D".into());
            args.push(hdump.display().to_string());
            args.push(url.clone());
            let (body, _) = run_curl_capture(curl, &args, ctl).await?;
            listed.extend(parse_tree(&body));
            let h = std::fs::read_to_string(&hdump).unwrap_or_default();
            match next_link(&h) {
                Some(n) => url = n,
                None => break,
            }
        }
        let _ = std::fs::remove_file(&hdump);
    }
    let mut out = Vec::new();
    if c.files.is_empty() {
        for (path, size, sha) in listed {
            let rel = if prefix.is_empty() {
                path.clone()
            } else {
                match path.strip_prefix(&format!("{prefix}/")) {
                    Some(r) => r.to_string(),
                    None => continue,
                }
            };
            if !crate::catalog::safe_rel(&rel) {
                // dotfiles (.gitattributes) and anything odd stay out
                continue;
            }
            out.push(RFile {
                url: resolve(&path),
                rel,
                size,
                sha256: sha,
            });
        }
    } else {
        for f in &c.files {
            let in_repo = join(if f.from.is_empty() { &f.path } else { &f.from });
            let l = listed.iter().find(|(p, ..)| *p == in_repo);
            out.push(RFile {
                rel: f.path.clone(),
                url: if f.url.is_empty() {
                    resolve(&in_repo)
                } else {
                    f.url.clone()
                },
                size: f.size.or_else(|| l.and_then(|x| x.1)),
                sha256: f.sha256.clone().or_else(|| l.and_then(|x| x.2.clone())),
            });
        }
        if needs_listing
            && c.files.iter().any(|f| f.url.is_empty())
            && out.iter().any(|f| f.size.is_none())
        {
            let missing: Vec<_> = out
                .iter()
                .filter(|f| f.size.is_none())
                .map(|f| f.rel.clone())
                .collect();
            return Err(Stop::Failed(format!(
                "The source does not have {} (the catalog may be out of date).",
                missing.join(", ")
            )));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_parse_and_links() {
        let j = r#"[{"type":"directory","path":"p"},
            {"type":"file","path":"p/manifest.json","size":12,"oid":"abc"},
            {"type":"file","path":"p/rank0.pack","size":134,"lfs":{"oid":"deadbeef","size":9000,"pointerSize":134}}]"#;
        let f = parse_tree(j);
        assert_eq!(f.len(), 2);
        assert_eq!(f[0], ("p/manifest.json".into(), Some(12), None));
        assert_eq!(
            f[1],
            ("p/rank0.pack".into(), Some(9000), Some("deadbeef".into()))
        );
        assert!(parse_tree("not json").is_empty());
        let h = "HTTP/1.1 200 OK\r\nLink: <https://h/api/x?cursor=2>; rel=\"next\"\r\n";
        assert_eq!(next_link(h).as_deref(), Some("https://h/api/x?cursor=2"));
        assert_eq!(next_link("HTTP/1.1 200 OK\r\n"), None);
    }

    #[test]
    fn plain_english_errors() {
        assert!(curl_error(Some(6), "", "https://huggingface.co/x").contains("huggingface.co"));
        assert!(curl_error(
            Some(22),
            "curl: (22) The requested URL returned error: 404",
            "u"
        )
        .contains("404"));
        assert!(curl_error(Some(22), "error: 401", "u").contains("token"));
        assert!(curl_error(Some(28), "", "u").contains("Retry"));
        assert_eq!(enc_path("a b/c+d"), "a%20b/c%2Bd");
        assert_eq!(human(26_000_000_000), "26.0 GB");
    }

    #[test]
    fn sha_of_a_file() {
        let d = saber_test_scratch::ScratchDir::new("console-dl-sha");
        let p = d.path().join("x");
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_file(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
