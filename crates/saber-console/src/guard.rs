//! The one-server-on-the-cards guard, ported from the launchers (`tools/serve_*.sh`): refuse to
//! start while the launchers' systemd user unit is active, while another saber server container
//! runs, while a GPU lock file is held (`<repo>/.cache/gpu.lock`, `gpu.lock.<card>`), or while
//! the engine's port is taken. Every probe is read-only: a lock is tested with a shared,
//! non-blocking `flock` that is released at once, so a queued holder is never displaced.

use std::path::Path;
use std::process::Command;

/// The prod unit the launchers refuse to share the cards with.
pub const PROD_UNIT: &str = "qwen_prod_nvfp4";

/// Container names that mean "someone holds the cards" (the launchers' pattern, plus the 27B
/// server's own name, which `serve_flashnext.sh` misses).
pub const CARD_HOLDER_PREFIXES: &[&str] = &[
    "saber-fn",
    "saber-flashnext-serve",
    "saber-27b-serve",
    "saber-27b-tp1-serve",
];

/// Why the console refused to start the engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal(pub String);

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The card guard (GPU presets only): the first conflict, if any.
pub fn check_cards(container_runtime: &str, lock_dir: &Path) -> Result<(), Refusal> {
    match card_conflicts(container_runtime, lock_dir)
        .into_iter()
        .next()
    {
        Some(c) => Err(Refusal(c)),
        None => Ok(()),
    }
}

/// Every conflict on the cards (the setup wizard's system check lists them all): prod's unit,
/// other saber servers / hold containers, held GPU locks.
#[must_use]
pub fn card_conflicts(container_runtime: &str, lock_dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    if Command::new("systemctl")
        .args(["--user", "is-active", "--quiet", PROD_UNIT])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        out.push(format!(
            "prod's unit {PROD_UNIT} is active; refusing to boot a second server on the cards"
        ));
    }
    if let Ok(o) = Command::new(container_runtime)
        .args(["ps", "--format", "{{.Names}}"])
        .output()
    {
        let names = String::from_utf8_lossy(&o.stdout);
        let holders: Vec<&str> = names
            .lines()
            .filter(|n| CARD_HOLDER_PREFIXES.iter().any(|p| n.starts_with(p)))
            .collect();
        if !holders.is_empty() {
            out.push(format!(
                "another saber server or an agent's hold holds the cards ({}); stop it or wait",
                holders.join(", ")
            ));
        }
    }
    for lock in ["gpu.lock", "gpu.lock.0", "gpu.lock.1"] {
        let p = lock_dir.join(lock);
        if p.exists() && lock_held(&p) {
            out.push(format!(
                "the GPU lock {} is held (a hold or a GPU job is running); wait for it",
                p.display()
            ));
        }
    }
    out
}

/// True when someone holds `path` exclusively (`flock -n -s` fails).
fn lock_held(path: &Path) -> bool {
    match Command::new("flock")
        .arg("-n")
        .arg("-s")
        .arg(path)
        .arg("true")
        .status()
    {
        Ok(s) => !s.success(),
        Err(_) => false,
    }
}

/// The port guard (every preset): something already listens on `port`.
pub fn check_port(port: u16) -> Result<(), Refusal> {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    if std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(300)).is_ok() {
        return Err(Refusal(format!("port {port} is already in use")));
    }
    Ok(())
}

/// A container of this name exists (running or not). The console refuses to replace a server
/// it did not start (the launchers `rm -f` it; a console never kills what it does not own).
pub fn check_name_free(container_runtime: &str, name: &str) -> Result<(), Refusal> {
    let out = Command::new(container_runtime)
        .args(["ps", "-a", "--format", "{{.Names}}"])
        .output();
    if let Ok(out) = out {
        if String::from_utf8_lossy(&out.stdout)
            .lines()
            .any(|n| n == name)
        {
            return Err(Refusal(format!(
                "a container named {name} already exists (started outside the console); stop it \
                 with `{container_runtime} rm -f {name}` or the launcher's `stop`"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_guard_sees_a_listener() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        assert!(check_port(port).is_err());
        // (no "free after drop" check: a connect to a just-freed EPHEMERAL port can
        // self-connect on Linux, which is a property of the test's port, not of the guard)
        drop(l);
    }

    /// Read-only probe of THIS host (`cargo test -p saber-console -- --ignored live_card_guard`).
    #[test]
    #[ignore = "reads the live box: systemctl, podman ps, the GPU locks"]
    fn live_card_guard() {
        // the checkout whose .cache holds the GPU lock (a worktree sits inside it)
        let lock_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .map(|a| a.join(".cache"))
            .find(|d| d.join("gpu.lock").exists())
            .unwrap_or_default();
        println!("lock dir {}", lock_dir.display());
        println!("card guard: {:?}", check_cards("podman", &lock_dir));
        println!("port 8080: {:?}", check_port(8080));
        println!("name: {:?}", check_name_free("podman", "saber-27b-serve"));
    }

    #[test]
    fn lock_probe_sees_an_exclusive_holder() {
        let dir = saber_test_scratch::ScratchDir::new("console-guard-lock");
        let p = dir.path().join("gpu.lock.0");
        std::fs::write(&p, b"").unwrap();
        assert!(!lock_held(&p));
        let mut holder = Command::new("flock")
            .arg("-x")
            .arg(&p)
            .args(["sleep", "5"])
            .spawn()
            .unwrap();
        let mut held = false;
        for _ in 0..50 {
            if lock_held(&p) {
                held = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let _ = holder.kill();
        let _ = holder.wait();
        assert!(held);
    }
}
