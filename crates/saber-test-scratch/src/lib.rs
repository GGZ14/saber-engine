//! Test scratch directories under the cargo target dir, deleted on drop (FN51h, 2026-10-02).
//!
//! `/tmp` is a 31 GiB tmpfs on this box. The saber-mem / xtier / fn* tests used to make their
//! scratch under `std::env::temp_dir()` and never delete it: ~300 directories per host-check run,
//! 11,325 of them (~15 GiB) by FN51g, swapped into zram (5.95 GiB of RAM held) -- which is what
//! broke the 262k served default's 16 GiB MemAvailable floor. So every test that needs scratch on
//! disk takes a [`ScratchDir`]:
//!
//! * it lives under `<target>/test-scratch/` (the cargo target dir: `$CARGO_TARGET_DIR` when set,
//!   else the workspace's `target/`), never under `/tmp` -- [`ScratchDir::new`] PANICS if the
//!   resolved path is under `/tmp` or `std::env::temp_dir()`;
//! * it is unique per process and per call (`<name>-<pid>-<seq>`), so tests running beside each
//!   other never share one;
//! * it is removed (recursively) when dropped, unless `SABER_KEEP_SCRATCH=1` is set, which keeps
//!   it for a post-mortem and prints where it is.
//!
//! `tools/ci/structural.sh` (check 10) refuses a `temp_dir()` call in test code anywhere else.

use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static SEQ: AtomicU64 = AtomicU64::new(0);

/// The workspace root: this crate sits at `<root>/crates/saber-test-scratch`.
fn workspace_root() -> PathBuf {
    let m = Path::new(env!("CARGO_MANIFEST_DIR"));
    m.parent()
        .and_then(Path::parent)
        .expect("saber-test-scratch sits two levels under the workspace root")
        .to_path_buf()
}

/// The cargo target dir the tests are running from: `$CARGO_TARGET_DIR` at run time (cargo hands
/// its environment to the test binaries), else at build time, else `<workspace>/target`. A
/// relative value is relative to the workspace root, as cargo reads it.
pub fn target_dir() -> PathBuf {
    let set = std::env::var_os("CARGO_TARGET_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            option_env!("CARGO_TARGET_DIR")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        });
    match set {
        Some(p) if p.is_absolute() => p,
        Some(p) => workspace_root().join(p),
        None => workspace_root().join("target"),
    }
}

/// The directory every [`ScratchDir`] is made in.
pub fn scratch_root() -> PathBuf {
    target_dir().join("test-scratch")
}

/// True when `p` is under `/tmp` or the system temp dir (lexically and after resolving links).
fn under_tmp(p: &Path) -> bool {
    let tmps = [PathBuf::from("/tmp"), std::env::temp_dir()];
    let resolved = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    tmps.iter().any(|t| {
        let tc = t.canonicalize().unwrap_or_else(|_| t.clone());
        p.starts_with(t)
            || p.starts_with(&tc)
            || resolved.starts_with(t)
            || resolved.starts_with(&tc)
    })
}

fn keep_requested() -> bool {
    std::env::var_os("SABER_KEEP_SCRATCH").is_some_and(|v| v == "1")
}

/// A scratch directory that exists from [`ScratchDir::new`] until it is dropped.
///
/// It derefs to [`Path`], so `dir.join("x")`, `&*dir` and `dir.display()` read as they did on the
/// `PathBuf` it replaces.
#[derive(Debug)]
pub struct ScratchDir {
    path: PathBuf,
    keep: bool,
}

impl ScratchDir {
    /// Make `<target>/test-scratch/<name>-<pid>-<seq>`. Panics if that would be under `/tmp`.
    pub fn new(name: &str) -> ScratchDir {
        assert!(
            !name.is_empty() && !name.contains('/') && name != "." && name != "..",
            "scratch name {name:?} must be one plain path component"
        );
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = scratch_root().join(format!("{name}-{}-{seq}", std::process::id()));
        assert!(
            !under_tmp(&path),
            "test scratch {} resolves under /tmp (a RAM-backed tmpfs here): point \
             CARGO_TARGET_DIR at a disk path (FN51h)",
            path.display()
        );
        if path.exists() {
            // A stale directory from a crashed run that had this pid: start clean.
            let _ = std::fs::remove_dir_all(&path);
        }
        std::fs::create_dir_all(&path)
            .unwrap_or_else(|e| panic!("creating test scratch {}: {e}", path.display()));
        assert!(
            !under_tmp(&path),
            "test scratch {} resolves under /tmp once made (a symlinked target dir?) (FN51h)",
            path.display()
        );
        ScratchDir {
            path,
            keep: keep_requested(),
        }
    }

    /// The directory.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Deref for ScratchDir {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.path
    }
}

impl AsRef<Path> for ScratchDir {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        if self.keep {
            eprintln!("SABER_KEEP_SCRATCH=1: keeping {}", self.path.display());
            return;
        }
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scratch_dir_is_under_target_unique_and_gone_on_drop() {
        let a = ScratchDir::new("selftest");
        let b = ScratchDir::new("selftest");
        assert_ne!(a.path(), b.path(), "two calls never share a directory");
        assert!(a.starts_with(scratch_root()));
        assert!(!under_tmp(&a));
        std::fs::write(a.join("f"), b"x").unwrap();
        std::fs::create_dir_all(a.join("d/e")).unwrap();
        let p = a.path().to_path_buf();
        drop(a);
        if !keep_requested() {
            assert!(!p.exists(), "removed on drop");
        }
    }

    #[test]
    fn the_tmp_guard_sees_tmp() {
        assert!(under_tmp(Path::new("/tmp/x/y")));
        assert!(under_tmp(&std::env::temp_dir().join("z")));
        assert!(!under_tmp(&workspace_root().join("target/test-scratch/q")));
    }
}
