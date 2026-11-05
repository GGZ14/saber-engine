//! The idle check, in one place: **VRAM first, `/metrics` second.**
//!
//! Every GPU command in this binary used to ask prod's `/metrics` for `num_requests_running`
//! and treat a refused connection as "the GPU is idle". On 2026-09-19 prod was restarted by hand and
//! had already claimed 30.6 GiB of *both* cards while it loaded the model — and a loading server
//! answers nothing on 8080, so the check passed and a 15.4 GiB weight upload died with
//! `hipMalloc failed: OutOfMemory` a minute later (`docs/LEDGER.md`, workstream H part 9).
//!
//! So the order is fixed and the reason is the failure: **free VRAM is the fact**, an idle request
//! counter is a proxy for it, and a proxy that reads "idle" while 30 GiB is held is worse than no
//! check at all. `flock ~/saber/.cache/gpu.lock.<card>` (the per-card locks, `tools/capture/gpu_idle.sh`) is also not this check — it serialises the agents
//! that take it and says nothing about a server started outside it.
//!
//! The VRAM number comes from `hipMemGetInfo` on the device this process is about to use, not from
//! `/sys/class/drm/cardN`: in a container `HIP_VISIBLE_DEVICES=1` maps card 1 to ordinal 0, so a
//! sysfs card index is a second opinion about which card is meant. The shell half of this check
//! (`tools/capture/gpu_idle.sh`) does read sysfs, because a shell script has no HIP — and it takes
//! the card index explicitly for the same reason.
//!
//! Nothing here ever stops, restarts or otherwise touches prod. It refuses, prints why, and leaves.

use std::io::{Read as _, Write as _};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use saber_hip::HipThread;

/// Where prod's `/metrics` lives on this box.
///
/// A constant rather than an argument on every subcommand: the address is a property of the machine
/// and every command that has one already defaults to it. `saber tune` keeps its own flag because a
/// sweep is sometimes pointed at a second engine.
pub const DEFAULT_METRICS: &str = "127.0.0.1:8080";

/// What the check saw, so a caller can print or record it.
#[derive(Debug, Clone, Copy)]
pub struct Idle {
    /// Free device memory, bytes.
    pub free: u64,
    /// Total device memory, bytes.
    pub total: u64,
    /// `num_requests_running` summed over engines, or `None` when no server answered.
    pub requests_running: Option<f64>,
}

impl Idle {
    /// Bytes held by everything else on this device.
    #[must_use]
    pub fn held(&self) -> u64 {
        self.total.saturating_sub(self.free)
    }

    /// The line every GPU command prints before it allocates.
    #[must_use]
    pub fn line(&self, device: i32) -> String {
        format!(
            "idle      HIP device {device}: {} MiB free of {} MiB ({} MiB held), prod \
             num_requests_running = {}",
            self.free >> 20,
            self.total >> 20,
            self.held() >> 20,
            match self.requests_running {
                Some(n) => format!("{n}"),
                None => "unreachable (a loading server answers nothing, which is why VRAM is \
                         checked first)"
                    .to_string(),
            }
        )
    }
}

/// Read the device's memory and prod's request count. Never fails on the `/metrics` half.
///
/// # Errors
/// `hipMemGetInfo` failing, which means the context is not usable and nothing else matters.
pub fn look(hip: &HipThread, metrics_addr: &str) -> anyhow::Result<Idle> {
    let (free, total) = hip.mem_info()?;
    Ok(Idle {
        free,
        total,
        requests_running: requests_running(metrics_addr),
    })
}

/// The check every GPU command makes before it allocates: print, then refuse if the card is busy.
///
/// `need` is what this command is about to allocate, and `slack` is what it leaves for the driver
/// and the compositor (a gigabyte is the number the step capture uses). A command that fits beside
/// prod passes `need = 0` and relies on the request count, which is what the small kernel gates
/// have always done — the difference is that they now also *print* what is held, so a log says
/// whether the number they produced was measured beside a resident server.
///
/// # Errors
/// More than `total - need - slack` held by someone else, or prod with requests in flight.
pub fn require(
    hip: &HipThread,
    device: i32,
    need: u64,
    slack: u64,
    metrics_addr: &str,
) -> anyhow::Result<Idle> {
    let idle = look(hip, metrics_addr)?;
    println!("{}", idle.line(device));
    if need > 0 && need + slack > idle.free {
        anyhow::bail!(
            "this run needs {} MiB (plus {} MiB of slack) and HIP device {device} has {} MiB free \
             of {} MiB: something else is holding it — prod, or another agent's container. Nothing \
             here will stop prod; run this when the card is free.",
            need >> 20,
            slack >> 20,
            idle.free >> 20,
            idle.total >> 20
        );
    }
    if let Some(n) = idle.requests_running {
        if n > 0.0 {
            anyhow::bail!(
                "prod has {n} request(s) in flight on {metrics_addr}; refusing to touch the GPU"
            );
        }
    }
    Ok(idle)
}

/// [`require`] at [`DEFAULT_METRICS`] with a gigabyte of slack — what every GPU command wants.
///
/// # Errors
/// As [`require`].
pub fn require_default(hip: &HipThread, device: i32, need: u64) -> anyhow::Result<Idle> {
    require(hip, device, need, 1 << 30, DEFAULT_METRICS)
}

/// An exposition line of the running-requests gauge, whatever its namespace.
fn is_requests_running(line: &str) -> bool {
    line.split_once(':').is_some_and(|(ns, rest)| {
        !ns.is_empty()
            && ns.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')
            && rest.starts_with("num_requests_running")
    })
}

/// `<namespace>:num_requests_running` (prod's exporter and Saber's name it under their own
/// namespace), summed over engines. `None` means no server answered.
///
/// Plain TCP rather than a HTTP client: one GET, no dependency, and a server that is not there
/// answers nothing — which is exactly why this is the SECOND half of the check and not the first.
#[must_use]
pub fn requests_running(addr: &str) -> Option<f64> {
    let sock: SocketAddr = addr.parse().ok()?;
    let mut stream = TcpStream::connect_timeout(&sock, Duration::from_secs(3)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    stream
        .write_all(b"GET /metrics HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .ok()?;
    let mut body = String::new();
    stream.read_to_string(&mut body).ok()?;
    let mut total = 0.0;
    let mut seen = false;
    for line in body.lines() {
        if is_requests_running(line) {
            if let Some(v) = line
                .split_whitespace()
                .next_back()
                .and_then(|t| t.parse::<f64>().ok())
            {
                total += v;
                seen = true;
            }
        }
    }
    seen.then_some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The line is the record: it must name the device, both numbers, and what prod said — including
    /// when prod said nothing, which is the case that used to read as "idle".
    #[test]
    fn the_line_says_what_was_held_and_that_an_unreachable_server_is_not_evidence() {
        let busy = Idle {
            free: 1 << 30,
            total: 32 << 30,
            requests_running: None,
        };
        let s = busy.line(1);
        assert!(s.contains("1024 MiB free"), "{s}");
        assert!(s.contains("31744 MiB held"), "{s}");
        assert!(s.contains("VRAM is checked first"), "{s}");
        assert_eq!(busy.held(), 31 << 30);

        let quiet = Idle {
            free: 32 << 30,
            total: 32 << 30,
            requests_running: Some(0.0),
        };
        assert!(quiet.line(0).contains("running = 0"), "{}", quiet.line(0));
        assert_eq!(quiet.held(), 0);
    }

    /// No server, no panic: the second half of the check is allowed to be absent.
    #[test]
    fn an_address_that_answers_nothing_is_none_not_an_error() {
        // Port 1 on loopback: nothing listens, and a garbage address must not panic either.
        assert!(requests_running("127.0.0.1:1").is_none());
        assert!(requests_running("not-an-address").is_none());
    }
}
