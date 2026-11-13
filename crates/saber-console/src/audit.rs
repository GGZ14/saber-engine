//! The audit log: every mutating admin call (config saves and applies, rollbacks,
//! engine start / stop / restart, token and key changes, the wizard's steps) and the console's
//! own auto-rollbacks, as one JSON line each in `<data dir>/audit.jsonl` (owner-only, 0600).
//! `who` is the credential's label (`admin`, `local` for an open loopback console, `console`
//! for the console's own actions), never a secret.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuditEntry {
    pub ts_ms: u64,
    pub who: String,
    /// The client's address (`None` for the console's own actions).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ip: Option<String>,
    /// `engine.start`, `config.apply`, `auth.key.revoke`, ...
    pub action: String,
    /// `POST /api/engine/start`.
    #[serde(default)]
    pub route: String,
    /// The reply's status (0 for an action the console took itself).
    #[serde(default)]
    pub status: u16,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub detail: String,
}

pub struct Audit {
    path: PathBuf,
    lock: Mutex<()>,
}

/// Lines kept before the file is cut to its newer half.
const MAX_LINES: usize = 20_000;

impl Audit {
    #[must_use]
    pub fn new(data_dir: &Path) -> Audit {
        Audit {
            path: data_dir.join("audit.jsonl"),
            lock: Mutex::new(()),
        }
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one entry (a failure to write is logged, never fatal to the call).
    pub fn record(&self, e: &AuditEntry) {
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(err) = self.append(e) {
            eprintln!("saber ui: audit log {}: {err:#}", self.path.display());
        }
    }

    fn append(&self, e: &AuditEntry) -> anyhow::Result<()> {
        use std::os::unix::fs::OpenOptionsExt;
        if let Some(d) = self.path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(&self.path)?;
        let mut line = serde_json::to_string(e)?;
        line.push('\n');
        f.write_all(line.as_bytes())?;
        if f.metadata()?.len() > (MAX_LINES as u64) * 400 {
            drop(f);
            self.trim()?;
        }
        Ok(())
    }

    fn trim(&self) -> anyhow::Result<()> {
        let all = self.read_all();
        if all.len() <= MAX_LINES / 2 {
            return Ok(());
        }
        let keep = &all[all.len() - MAX_LINES / 2..];
        let tmp = self.path.with_extension("jsonl.tmp");
        {
            use std::os::unix::fs::OpenOptionsExt;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .mode(0o600)
                .open(&tmp)?;
            for e in keep {
                writeln!(f, "{}", serde_json::to_string(e)?)?;
            }
        }
        std::fs::rename(tmp, &self.path)?;
        Ok(())
    }

    fn read_all(&self) -> Vec<AuditEntry> {
        let Ok(f) = std::fs::File::open(&self.path) else {
            return Vec::new();
        };
        std::io::BufReader::new(f)
            .lines()
            .map_while(Result::ok)
            .filter_map(|l| serde_json::from_str(&l).ok())
            .collect()
    }

    /// The newest `limit` entries, newest first.
    #[must_use]
    pub fn recent(&self, limit: usize) -> Vec<AuditEntry> {
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut all = self.read_all();
        all.reverse();
        all.truncate(limit);
        all
    }
}

/// The action name of a mutating admin route (`None` = not audited: a read, or a dry run).
#[must_use]
pub fn action_of(method: &str, path: &str) -> Option<String> {
    if method == "GET" || method == "HEAD" {
        return None;
    }
    let p = path.trim_start_matches("/api/");
    let parts: Vec<&str> = p.split('/').collect();
    let a = match parts.as_slice() {
        ["config"] => "config.save".to_string(),
        ["config", "validate"] | ["profiles", "compose"] | ["setup", "keygen"] => return None,
        ["config", "apply"] => "config.apply".into(),
        ["engine", op] => format!("engine.{op}"),
        ["history", _, "rollback"] => "config.rollback".into(),
        ["auth", "admin-token"] if method == "DELETE" => "auth.admin-token.remove".into(),
        ["auth", "admin-token"] => "auth.admin-token.rotate".into(),
        ["auth", "keys"] => "auth.key.create".into(),
        ["auth", "keys", _, "revoke"] => "auth.key.revoke".into(),
        ["auth", "keys", _] if method == "DELETE" => "auth.key.delete".into(),
        ["auth", "login"] => return None,
        ["packs", _] if method == "DELETE" => "packs.delete".into(),
        ["profiles", ..] => format!("profiles.{}", method.to_ascii_lowercase()),
        ["downloads", ..] => format!("downloads.{}", parts.last().unwrap_or(&"start")),
        ["setup", step] => format!("setup.{step}"),
        _ => format!("{} {}", method.to_ascii_lowercase(), p),
    };
    Some(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actions() {
        assert_eq!(action_of("GET", "/api/config"), None);
        assert_eq!(
            action_of("PUT", "/api/config").as_deref(),
            Some("config.save")
        );
        assert_eq!(action_of("POST", "/api/config/validate"), None);
        assert_eq!(
            action_of("POST", "/api/config/apply").as_deref(),
            Some("config.apply")
        );
        assert_eq!(
            action_of("POST", "/api/engine/stop").as_deref(),
            Some("engine.stop")
        );
        assert_eq!(
            action_of("POST", "/api/history/7/rollback").as_deref(),
            Some("config.rollback")
        );
        assert_eq!(
            action_of("POST", "/api/auth/keys/laptop/revoke").as_deref(),
            Some("auth.key.revoke")
        );
        assert_eq!(
            action_of("DELETE", "/api/auth/keys/laptop").as_deref(),
            Some("auth.key.delete")
        );
        assert_eq!(
            action_of("POST", "/api/setup/launch").as_deref(),
            Some("setup.launch")
        );
    }

    #[test]
    fn append_read_newest_first_and_0600() {
        use std::os::unix::fs::PermissionsExt;
        let d = saber_test_scratch::ScratchDir::new("console-audit");
        let a = Audit::new(d.path());
        for i in 0..3 {
            a.record(&AuditEntry {
                ts_ms: i,
                who: "admin".into(),
                ip: Some("127.0.0.1".into()),
                action: format!("a{i}"),
                route: String::new(),
                status: 200,
                detail: String::new(),
            });
        }
        let r = a.recent(2);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].action, "a2");
        assert_eq!(r[1].action, "a1");
        let mode = std::fs::metadata(a.path()).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }
}
