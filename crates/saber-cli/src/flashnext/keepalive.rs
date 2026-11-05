//! FN54: `--fn-control host-keepalive` -- the user-space stand-in for a PM-QoS request (an ARM,
//! default OFF). One thread of the rank-0 process spins (`spin_loop`, no syscalls) on a fixed CPU
//! for the life of the boot, so at least one core of the package stays in C0 and the package /
//! data fabric never reaches its deep idle states between the ranks' waits. It does NOT stop the
//! other cores' C3 (350 us exit on this host: `cpuidle` acpi_idle); that needs root
//! (`/dev/cpu_dma_latency` held open at 0, or `cpupower idle-set -D 20`). Nothing the engine
//! computes depends on it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// The CPU the spinner is pinned to: core 11 = CCD1's last physical core on the Ryzen 9 9900X
/// (FN40: the ranks' hot threads sit on CCD0 ~97 % of the time).
pub const KEEPALIVE_CPU: u32 = 11;

/// The running spinner; dropping it stops the thread.
pub struct KeepAlive {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Drop for KeepAlive {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Start the spinner pinned to `cpu` (through `taskset -p` on its own tid: no FFI here). Returns
/// the guard and the boot line (pinned or not, named).
pub fn start(cpu: u32) -> anyhow::Result<(KeepAlive, String)> {
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let s2 = Arc::clone(&stop);
    let handle = std::thread::Builder::new()
        .name("fn-keepalive".into())
        .spawn(move || {
            let tid = std::fs::read_link("/proc/thread-self")
                .ok()
                .and_then(|p| p.file_name().map(|t| t.to_string_lossy().into_owned()));
            let pin = match tid {
                Some(t) => match std::process::Command::new("taskset")
                    .args(["-p", "-c", &cpu.to_string(), &t])
                    .output()
                {
                    Ok(o) if o.status.success() => format!("tid {t} pinned to cpu {cpu}"),
                    Ok(o) => format!(
                        "tid {t} UNPINNED (taskset rc {:?}: {})",
                        o.status.code(),
                        String::from_utf8_lossy(&o.stderr).trim()
                    ),
                    Err(e) => format!("tid {t} UNPINNED (taskset: {e})"),
                },
                None => "UNPINNED (no /proc/thread-self)".into(),
            };
            let _ = tx.send(pin);
            while !s2.load(Ordering::Relaxed) {
                for _ in 0..1024 {
                    std::hint::spin_loop();
                }
            }
        })?;
    let pin = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap_or_else(|_| "pin status unknown".into());
    Ok((
        KeepAlive {
            stop,
            handle: Some(handle),
        },
        format!(
            "fnkeepalive: HOST KEEP-AWAKE ARM (FN54) -- one spinning thread, {pin}; the root form is \
             `/dev/cpu_dma_latency` held at 0 or `cpupower idle-set -D 20`"
        ),
    ))
}

/// FN90a: `--fn-ple-drive-keepalive-ms N` -- the PLE drive's keep-awake (an ARM, default off). The
/// kernel's APST puts an idle NVMe drive into a non-operational power state after
/// `nvme_core.apst_primary_timeout_ms` (100 ms here) of no commands, and the next read pays the
/// state's exit latency; a decode step whose late PLE rows miss the RAM cache after such a gap
/// stalls the device between the layer-0 and layer-1 graphs. One thread reads one 4 KiB
/// `O_DIRECT` block of `path` every `period` (offsets walk the file's first 64 MiB), so the drive
/// never idles that long. Nothing the engine computes reads it. Dropping it stops the thread and
/// prints its census line (reads, latency p50 / p99 / max).
pub struct DriveKeepAlive {
    stop: Option<std::sync::mpsc::Sender<()>>,
    /// The reader's (read latencies ms, gaps between consecutive reads' ends ms).
    handle: Option<std::thread::JoinHandle<(Vec<f64>, Vec<f64>)>>,
    period_ms: u64,
    path: std::path::PathBuf,
}

/// FN121 (`--fn-ple-prefetch keep`): the keep-awake's coverage -- one file per distinct block
/// device among `files` (the PLE files + the embedding file: every drive path either rank reads on
/// a decode step) and the boot line naming them.
pub fn drive_cover(
    files: &[std::path::PathBuf],
) -> anyhow::Result<(Vec<std::path::PathBuf>, String)> {
    use std::os::unix::fs::MetadataExt as _;
    let mut devs: Vec<(u64, std::path::PathBuf, usize)> = Vec::new();
    for f in files {
        let d = std::fs::metadata(f)
            .map_err(|e| anyhow::anyhow!("{}: {e}", f.display()))?
            .dev();
        match devs.iter_mut().find(|x| x.0 == d) {
            Some(x) => x.2 += 1,
            None => devs.push((d, f.clone(), 1)),
        }
    }
    anyhow::ensure!(!devs.is_empty(), "FN121 keep: no drive file to cover");
    let line = format!(
        "fnple121: keep-awake coverage -- {} file(s) on {} block device(s) [{}]: one reader a device \
         covers both ranks (they read the same files)",
        files.len(),
        devs.len(),
        devs.iter()
            .map(|(d, f, n)| format!("dev {}:{} {} files via {}", d >> 8, d & 0xff, n, f.display()))
            .collect::<Vec<_>>()
            .join("; ")
    );
    Ok((devs.into_iter().map(|x| x.1).collect(), line))
}

impl Drop for DriveKeepAlive {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(h) = self.handle.take() {
            if let Ok((mut lat, gaps)) = h.join() {
                let gmax = gaps.iter().copied().fold(0.0f64, f64::max);
                lat.sort_by(f64::total_cmp);
                let q = |p: f64| {
                    lat.get(((lat.len() as f64 * p) as usize).min(lat.len().saturating_sub(1)))
                        .copied()
                        .unwrap_or(0.0)
                };
                println!(
                    "fnplekeep90a: {} keep-awake reads every {} ms, latency ms p50 {:.3} p99 {:.3} max {:.3}, {} over 1 ms | FN121: gap between reads max {:.1} ms, {} over 90 ms ({})",
                    lat.len(),
                    self.period_ms,
                    q(0.5),
                    q(0.99),
                    lat.last().copied().unwrap_or(0.0),
                    lat.iter().filter(|&&x| x > 1.0).count(),
                    gmax,
                    gaps.iter().filter(|&&x| x > 90.0).count(),
                    self.path.display()
                );
            }
        }
    }
}

/// Start the PLE drive's keep-awake reader over `path` (FN90a). Returns the guard and the boot line.
pub fn start_drive(
    path: &std::path::Path,
    period_ms: u64,
) -> anyhow::Result<(DriveKeepAlive, String)> {
    use saber_mem::dio::{open_read, preadv_once, AlignedBuf, OpenMode};
    anyhow::ensure!(
        period_ms > 0,
        "--fn-ple-drive-keepalive-ms 0 is off (pass no flag)"
    );
    let file = open_read(path, OpenMode::Direct)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let len = file.metadata()?.len();
    let span = (len.min(64 << 20) / 4096).max(1);
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let period = std::time::Duration::from_millis(period_ms);
    let handle = std::thread::Builder::new()
        .name("fn-plekeep".into())
        .spawn(move || {
            let mut buf = AlignedBuf::zeroed(4096);
            let mut lat = Vec::new();
            let mut gaps = Vec::new();
            let mut last: Option<std::time::Instant> = None;
            let mut k = 0u64;
            loop {
                let t = std::time::Instant::now();
                if preadv_once(&file, &mut buf[..], (k % span) * 4096).is_ok() {
                    let e = std::time::Instant::now();
                    lat.push((e - t).as_secs_f64() * 1e3);
                    // FN121: the process's drive-idle clock (the PLE readers' records read it)
                    saber_mem::ple_tier::note_io_end(e);
                    if let Some(l) = last {
                        gaps.push((e - l).as_secs_f64() * 1e3);
                    }
                    last = Some(e);
                }
                k += 1;
                match rx.recv_timeout(period) {
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    _ => break,
                }
            }
            (lat, gaps)
        })?;
    Ok((
        DriveKeepAlive {
            stop: Some(tx),
            handle: Some(handle),
            period_ms,
            path: path.to_path_buf(),
        },
        format!(
            "fnplekeep90a: PLE DRIVE KEEP-AWAKE ARM (FN90a) -- one 4 KiB O_DIRECT read of {} every {period_ms} ms; \
             the root form is `nvme_core.default_ps_max_latency_us=0`",
            path.display()
        ),
    ))
}

#[cfg(test)]
mod tests {
    /// FN121: the keep-awake's coverage -- files on one block device need ONE reader (the first
    /// file names it); a file that does not exist is refused by name (the control).
    #[test]
    fn fn121_one_reader_covers_every_file_on_one_device() {
        let scratch = saber_test_scratch::ScratchDir::new("saber-fn121-cover");
        let d = scratch.path().to_path_buf();
        let a = d.join("a.bin");
        let b = d.join("b.bin");
        std::fs::write(&a, [0u8; 16]).unwrap();
        std::fs::write(&b, [0u8; 16]).unwrap();
        let (devs, line) = super::drive_cover(&[a.clone(), b]).unwrap();
        assert_eq!(devs, vec![a]);
        assert!(line.contains("on 1 block device"), "{line}");
        assert!(super::drive_cover(&[d.join("missing.bin")]).is_err());
    }
}
