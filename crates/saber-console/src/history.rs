//! Config history: every applied config, versioned, in `<data dir>/history/`.
//!
//! One TOML file an entry (`000042.toml`: a `[meta]` table + the whole `[config]`), the newest
//! `keep` kept (default 50) -- and the newest entry that reached `ready` is never pruned, so a
//! run of failed applies cannot delete the config to roll back to. The outcome is updated in
//! place when the apply finishes.

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::config::Config;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Outcome {
    /// Applied; the engine has not reached ready yet.
    Pending,
    /// The engine reached ready on it (a "working" config).
    Ready,
    /// It did not reach ready (timeout, or the engine exited during load).
    Failed,
    /// Saved with no restart (the engine was stopped, or nothing that reaches the command line
    /// changed); not proven either way.
    Saved,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Origin {
    Apply,
    Rollback,
    AutoRollback,
    /// The config the console found running when history began (first apply).
    Baseline,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Meta {
    pub id: u64,
    pub ts_ms: u64,
    pub origin: Origin,
    pub outcome: Outcome,
    #[serde(default)]
    pub note: String,
    /// Why it failed / what it rolled back from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// The engine command line it composed to.
    #[serde(default)]
    pub command: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub meta: Meta,
    pub config: Config,
}

pub struct History {
    dir: PathBuf,
    keep: std::sync::atomic::AtomicUsize,
    lock: std::sync::Mutex<()>,
}

fn file_id(p: &Path) -> Option<u64> {
    let n = p.file_name()?.to_str()?;
    n.strip_suffix(".toml")?.parse().ok()
}

impl History {
    #[must_use]
    pub fn new(data_dir: &Path, keep: usize) -> History {
        History {
            dir: data_dir.join("history"),
            keep: std::sync::atomic::AtomicUsize::new(keep.max(1)),
            lock: std::sync::Mutex::new(()),
        }
    }

    pub fn set_keep(&self, keep: usize) {
        self.keep
            .store(keep.max(1), std::sync::atomic::Ordering::Relaxed);
    }

    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn guard(&self) -> std::sync::MutexGuard<'_, ()> {
        self.lock.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn ids(&self) -> Vec<u64> {
        let mut v: Vec<u64> = std::fs::read_dir(&self.dir)
            .map(|rd| rd.flatten().filter_map(|e| file_id(&e.path())).collect())
            .unwrap_or_default();
        v.sort_unstable();
        v
    }

    fn path(&self, id: u64) -> PathBuf {
        self.dir.join(format!("{id:06}.toml"))
    }

    fn read(&self, id: u64) -> anyhow::Result<Entry> {
        let p = self.path(id);
        let t = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        toml::from_str(&t).with_context(|| format!("parsing {}", p.display()))
    }

    fn write(&self, e: &Entry) -> anyhow::Result<()> {
        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("creating {}", self.dir.display()))?;
        let p = self.path(e.meta.id);
        let tmp = p.with_extension("toml.tmp");
        {
            // owner-only, like config.toml (the engine key rides the config)
            use std::io::Write;
            use std::os::unix::fs::OpenOptionsExt;
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(&tmp)
                .with_context(|| format!("writing {}", tmp.display()))?;
            f.write_all(toml::to_string_pretty(e)?.as_bytes())?;
        }
        std::fs::rename(&tmp, &p)?;
        Ok(())
    }

    /// Record a config; returns its id. Prunes past `keep`.
    pub fn record(
        &self,
        config: &Config,
        origin: Origin,
        outcome: Outcome,
        note: &str,
        command: &str,
    ) -> anyhow::Result<u64> {
        let _g = self.guard();
        let id = self.ids().last().map_or(1, |l| l + 1);
        // never a secret as text in the history
        let mut config = config.clone();
        config.seal()?;
        self.write(&Entry {
            meta: Meta {
                id,
                ts_ms: crate::now_ms(),
                origin,
                outcome,
                note: note.to_string(),
                message: None,
                command: command.to_string(),
            },
            config,
        })?;
        self.prune();
        Ok(id)
    }

    fn prune(&self) {
        let keep = self.keep.load(std::sync::atomic::Ordering::Relaxed);
        let ids = self.ids();
        if ids.len() <= keep {
            return;
        }
        let working = ids
            .iter()
            .rev()
            .find(|id| {
                self.read(**id)
                    .is_ok_and(|e| e.meta.outcome == Outcome::Ready)
            })
            .copied();
        let cut = ids.len() - keep;
        for id in &ids[..cut] {
            if Some(*id) != working {
                let _ = std::fs::remove_file(self.path(*id));
            }
        }
    }

    pub fn set_outcome(
        &self,
        id: u64,
        outcome: Outcome,
        message: Option<String>,
    ) -> anyhow::Result<()> {
        let _g = self.guard();
        let mut e = self.read(id)?;
        e.meta.outcome = outcome;
        if message.is_some() {
            e.meta.message = message;
        }
        self.write(&e)
    }

    /// Every entry's meta, newest first.
    pub fn list(&self) -> Vec<Meta> {
        let _g = self.guard();
        self.ids()
            .into_iter()
            .rev()
            .filter_map(|id| self.read(id).ok().map(|e| e.meta))
            .collect()
    }

    pub fn get(&self, id: u64) -> anyhow::Result<Entry> {
        let _g = self.guard();
        self.read(id)
    }

    /// The newest entry that reached ready, older than `before` when given.
    pub fn last_working(&self, before: Option<u64>) -> Option<Entry> {
        let _g = self.guard();
        self.ids()
            .into_iter()
            .rev()
            .filter(|id| before.is_none_or(|b| *id < b))
            .filter_map(|id| self.read(id).ok())
            .find(|e| e.meta.outcome == Outcome::Ready)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_lists_prunes_and_keeps_the_last_working() {
        let dir = saber_test_scratch::ScratchDir::new("console-history");
        let h = History::new(dir.path(), 5);
        let mut c = Config::mock(8099);
        let first = h
            .record(&c, Origin::Apply, Outcome::Pending, "one", "saber serve")
            .unwrap();
        h.set_outcome(first, Outcome::Ready, None).unwrap();
        for i in 0..8 {
            c.engine.port = 9000 + i;
            let id = h
                .record(&c, Origin::Apply, Outcome::Pending, &format!("n{i}"), "")
                .unwrap();
            h.set_outcome(id, Outcome::Failed, Some("exited 2".into()))
                .unwrap();
        }
        let l = h.list();
        // 5 kept + the last working one that would have been pruned
        assert_eq!(l.len(), 6);
        assert_eq!(l[0].id, 9);
        assert_eq!(l.last().unwrap().id, first);
        assert_eq!(l[0].message.as_deref(), Some("exited 2"));
        let w = h.last_working(None).unwrap();
        assert_eq!(w.meta.id, first);
        assert_eq!(w.config.engine.port, 8099);
        assert!(h.last_working(Some(first)).is_none());
        assert_eq!(h.get(9).unwrap().config.engine.port, 9007);
        // a newer working entry lets the old one go
        let id = h
            .record(&c, Origin::Rollback, Outcome::Ready, "", "")
            .unwrap();
        assert_eq!(h.last_working(None).unwrap().meta.id, id);
        h.record(&c, Origin::Apply, Outcome::Saved, "", "").unwrap();
        assert!(h.list().iter().all(|m| m.id != first));
        assert_eq!(h.list().len(), 5);
    }
}
