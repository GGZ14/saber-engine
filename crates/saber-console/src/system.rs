//! The setup wizard's system check (`/api/system`): GPUs (gfx arch, VRAM, the PCIe link of the
//! card and of its slot, peer-to-peer links) from the KFD topology and DRM sysfs, `/dev/kfd`
//! and render-node access, the container runtime and the dev image, free disk on the packs and
//! KV volumes, host RAM, and any server that already holds the cards. Each check is
//! pass / warn / fail with a plain-English fix.
//!
//! Nothing here opens a GPU device or calls a ROCm tool: the host has no ROCm, and the probe
//! must never touch GPU compute. Device access is tested with `access(2)`, which opens nothing.
//! Every sysfs / dev path is read under a `root` so the tests run against a fake tree.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Serialize;

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Link {
    /// GT/s (`32.0 GT/s PCIe` -> 32.0).
    pub speed_gts: Option<f64>,
    pub width: Option<u32>,
}

impl Link {
    /// `Gen5 x8`.
    #[must_use]
    pub fn label(&self) -> String {
        let gen = self.speed_gts.map(|s| match s {
            s if s >= 63.0 => "Gen6".to_string(),
            s if s >= 31.0 => "Gen5".to_string(),
            s if s >= 15.0 => "Gen4".to_string(),
            s if s >= 7.0 => "Gen3".to_string(),
            s if s >= 4.0 => "Gen2".to_string(),
            _ => "Gen1".to_string(),
        });
        match (gen, self.width) {
            (Some(g), Some(w)) => format!("{g} x{w}"),
            (Some(g), None) => g,
            (None, Some(w)) => format!("x{w}"),
            (None, None) => "unknown".into(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct GpuInfo {
    /// The KFD topology node.
    pub node: u32,
    /// `gfx1201`.
    pub gfx: String,
    /// gfx12 (RDNA4): what Saber's kernels are built for.
    pub supported: bool,
    /// An integrated GPU (small carve-out VRAM); never counted for serving.
    pub integrated: bool,
    /// A marketing name when the PCI id is known.
    pub name: String,
    /// `1002:7551`.
    pub pci_id: String,
    pub bdf: String,
    pub render_minor: Option<u32>,
    pub vram_bytes: Option<u64>,
    /// The card's own link (to its on-board switch on cards that have one).
    pub link_current: Link,
    pub link_max: Link,
    /// The narrowest capability on the path to the CPU (the slot): what transfers really get.
    pub slot_max: Link,
    pub slot_current: Link,
    /// KFD nodes of the GPUs this one has a peer-to-peer link to.
    pub p2p_peers: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CheckStatus {
    Pass,
    Warn,
    Fail,
    Info,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Check {
    pub id: &'static str,
    pub title: String,
    pub status: CheckStatus,
    pub detail: String,
    /// What to do about a warn / fail, in plain English.
    pub fix: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Disk {
    /// `packs` / `kv`.
    pub label: &'static str,
    pub path: String,
    /// The nearest existing directory the free space was read at.
    pub probed: String,
    pub free_bytes: Option<u64>,
    pub total_bytes: Option<u64>,
}

/// What the probe learns from commands (the container runtime, the card guard). Gathered by
/// [`gather_host`]; the tests pass their own.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct HostFacts {
    pub runtime: String,
    /// `podman version 5.x` (`None` = not found).
    pub runtime_version: Option<String>,
    pub image: String,
    /// `None` = could not be asked (no runtime).
    pub image_present: Option<bool>,
    /// Other servers / holds / GPU locks on the cards (the card guard's refusals).
    pub conflicts: Vec<String>,
    pub engine_port: u16,
    /// Something listens on the engine's port.
    pub port_busy: bool,
    /// The console's own engine is running (so a busy port is ours).
    pub engine_ours: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SystemReport {
    pub gpus: Vec<GpuInfo>,
    pub host_ram_bytes: Option<u64>,
    pub cpus: Option<u32>,
    pub disks: Vec<Disk>,
    pub host: HostFacts,
    pub checks: Vec<Check>,
    /// The worst status (`fail` > `warn` > `pass`).
    pub overall: Option<CheckStatus>,
    pub generated_ms: u64,
}

/// Where the probe looks and what it is checking for.
#[derive(Clone, Debug)]
pub struct ProbeInput {
    /// `/` on a real box; a fake tree in tests.
    pub root: PathBuf,
    pub packs_dir: PathBuf,
    pub kv_dir: PathBuf,
    /// Bytes the chosen model still has to download (the disk check's bar).
    pub need_disk_bytes: Option<u64>,
    /// The chosen model runs on the GPUs (the mock does not: GPU checks become info).
    pub needs_gpus: bool,
    /// The supported cards the chosen model's form needs (2 at TP=2, 1 for the 27B at TP=1).
    pub min_gpus: u32,
    pub host: HostFacts,
}

fn read(p: &Path) -> Option<String> {
    std::fs::read_to_string(p)
        .ok()
        .map(|s| s.trim().to_string())
}

fn props(p: &Path) -> BTreeMap<String, String> {
    read(p)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| l.split_once(char::is_whitespace))
        .map(|(k, v)| (k.to_string(), v.trim().to_string()))
        .collect()
}

/// `120001` -> `gfx1201`.
#[must_use]
pub fn gfx_name(v: u64) -> String {
    let (major, minor, step) = (v / 10000, (v / 100) % 100, v % 100);
    format!("gfx{major}{minor:x}{step:x}")
}

/// Known AMD device ids (anything else shows its PCI id).
fn marketing(pci: &str) -> &'static str {
    match pci {
        "1002:7551" => "Radeon AI PRO R9700",
        "1002:7550" => "Radeon RX 9070 XT",
        "1002:7590" => "Radeon RX 9060 XT",
        _ => "",
    }
}

fn parse_speed(s: &str) -> Option<f64> {
    s.split_whitespace().next()?.parse().ok()
}

fn link_at(dir: &Path, which: &str) -> Link {
    Link {
        speed_gts: read(&dir.join(format!("{which}_link_speed"))).and_then(|s| parse_speed(&s)),
        width: read(&dir.join(format!("{which}_link_width"))).and_then(|s| s.parse().ok()),
    }
}

fn is_bdf(name: &str) -> bool {
    // 0000:03:00.0
    let b = name.as_bytes();
    b.len() == 12 && b[4] == b':' && b[7] == b':' && b[10] == b'.'
}

/// The narrowest link on the path from the device up to the root complex.
fn slot_links(dev: &Path) -> (Link, Link) {
    let mut max = Link::default();
    let mut cur = Link::default();
    let mut d = dev.to_path_buf();
    loop {
        let name = d.file_name().map(|n| n.to_string_lossy().into_owned());
        if !name.as_deref().is_some_and(is_bdf) {
            break;
        }
        for (acc, which) in [(&mut max, "max"), (&mut cur, "current")] {
            let l = link_at(&d, which);
            if let Some(s) = l.speed_gts {
                acc.speed_gts = Some(acc.speed_gts.map_or(s, |a: f64| a.min(s)));
            }
            if let Some(w) = l.width {
                acc.width = Some(acc.width.map_or(w, |a: u32| a.min(w)));
            }
        }
        if !d.pop() {
            break;
        }
    }
    (max, cur)
}

/// The GPUs in the KFD topology under `root`.
#[must_use]
pub fn probe_gpus(root: &Path) -> Vec<GpuInfo> {
    let nodes = root.join("sys/class/kfd/kfd/topology/nodes");
    let mut ids: Vec<u32> = std::fs::read_dir(&nodes)
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| e.file_name().to_string_lossy().parse().ok())
                .collect()
        })
        .unwrap_or_default();
    ids.sort_unstable();
    let mut out = Vec::new();
    let mut gpu_nodes = std::collections::BTreeSet::new();
    for &n in &ids {
        let nd = nodes.join(n.to_string());
        let p = props(&nd.join("properties"));
        let v: u64 = p
            .get("gfx_target_version")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if v == 0 {
            continue; // a CPU node
        }
        gpu_nodes.insert(n);
        let gfx = gfx_name(v);
        let minor: Option<u32> = p.get("drm_render_minor").and_then(|s| s.parse().ok());
        let mut g = GpuInfo {
            node: n,
            supported: gfx.starts_with("gfx12"),
            gfx,
            render_minor: minor,
            ..GpuInfo::default()
        };
        // VRAM: DRM's own counter, else the KFD memory bank
        let kfd_mem = props(&nd.join("mem_banks/0/properties"))
            .get("size_in_bytes")
            .and_then(|s| s.parse::<u64>().ok());
        if let Some(m) = minor {
            let dev = root.join(format!("sys/class/drm/renderD{m}/device"));
            g.vram_bytes = read(&dev.join("mem_info_vram_total"))
                .and_then(|s| s.parse().ok())
                .or(kfd_mem);
            // uevent is KEY=VALUE
            let ue: BTreeMap<String, String> = read(&dev.join("uevent"))
                .unwrap_or_default()
                .lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            g.pci_id = ue
                .get("PCI_ID")
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();
            g.bdf = ue.get("PCI_SLOT_NAME").cloned().unwrap_or_default();
            g.name = marketing(&g.pci_id).to_string();
            g.link_current = link_at(&dev, "current");
            g.link_max = link_at(&dev, "max");
            if let Ok(real) = std::fs::canonicalize(&dev) {
                let (m, c) = slot_links(&real);
                g.slot_max = m;
                g.slot_current = c;
            }
        } else {
            g.vram_bytes = kfd_mem;
        }
        g.integrated = g.vram_bytes.is_some_and(|v| v < (4u64 << 30));
        // peer links (newer kernels list GPU peers under p2p_links; older ones under io_links)
        for dir in ["p2p_links", "io_links"] {
            if let Ok(rd) = std::fs::read_dir(nd.join(dir)) {
                for e in rd.flatten() {
                    let lp = props(&e.path().join("properties"));
                    if let Some(to) = lp.get("node_to").and_then(|s| s.parse::<u32>().ok()) {
                        if to != n && !g.p2p_peers.contains(&to) {
                            g.p2p_peers.push(to);
                        }
                    }
                }
            }
        }
        out.push(g);
    }
    // only GPU peers count (io_links also point at the CPU node)
    for g in &mut out {
        g.p2p_peers.retain(|p| gpu_nodes.contains(p));
        g.p2p_peers.sort_unstable();
    }
    out
}

/// `MemTotal` and the CPU count from `/proc` under `root`.
fn host_ram(root: &Path) -> (Option<u64>, Option<u32>) {
    let mem = read(&root.join("proc/meminfo")).and_then(|t| {
        t.lines()
            .find_map(|l| l.strip_prefix("MemTotal:"))
            .and_then(|v| v.split_whitespace().next()?.parse::<u64>().ok())
            .map(|kb| kb * 1024)
    });
    let cpus = read(&root.join("proc/cpuinfo"))
        .map(|t| t.lines().filter(|l| l.starts_with("processor")).count() as u32);
    (mem, cpus.filter(|&c| c > 0))
}

/// Free / total bytes of the file system holding `path` (its nearest existing ancestor).
#[must_use]
pub fn disk_space(path: &Path) -> (PathBuf, Option<u64>, Option<u64>) {
    let mut p = path.to_path_buf();
    while !p.exists() {
        if !p.pop() {
            return (PathBuf::new(), None, None);
        }
    }
    let (f, t) = statvfs(&p);
    (p, f, t)
}

fn statvfs(p: &Path) -> (Option<u64>, Option<u64>) {
    use std::os::unix::ffi::OsStrExt;
    let Ok(c) = std::ffi::CString::new(p.as_os_str().as_bytes()) else {
        return (None, None);
    };
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `c` is a NUL-terminated path and `s` a writable statvfs.
    let rc = unsafe { libc::statvfs(c.as_ptr(), &mut s) };
    if rc != 0 {
        return (None, None);
    }
    let frs = s.f_frsize as u64;
    (Some(s.f_bavail as u64 * frs), Some(s.f_blocks as u64 * frs))
}

/// `access(path, R_OK | W_OK)`: opens nothing (the probe never touches the GPU).
fn can_rw(p: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(c) = std::ffi::CString::new(p.as_os_str().as_bytes()) else {
        return false;
    };
    // SAFETY: `c` is a NUL-terminated path; access(2) reads it and nothing else.
    unsafe { libc::access(c.as_ptr(), libc::R_OK | libc::W_OK) == 0 }
}

/// Ask the container runtime and the card guard (blocking: a few short commands).
#[must_use]
pub fn gather_host(
    runtime: &str,
    image: &str,
    lock_dir: &Path,
    engine_port: u16,
    engine_ours: bool,
) -> HostFacts {
    use std::process::Command;
    let quiet = |args: &[&str]| {
        Command::new(runtime)
            .args(args)
            .stdin(std::process::Stdio::null())
            .output()
            .ok()
    };
    let version = quiet(&["--version"])
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    let image_present = version.as_ref().map(|_| {
        let args: &[&str] = if runtime.ends_with("docker") {
            &["image", "inspect", image]
        } else {
            &["image", "exists", image]
        };
        quiet(args).is_some_and(|o| o.status.success())
    });
    let conflicts = crate::guard::card_conflicts(runtime, lock_dir);
    let port_busy = crate::guard::check_port(engine_port).is_err();
    HostFacts {
        runtime: runtime.to_string(),
        runtime_version: version,
        image: image.to_string(),
        image_present,
        conflicts,
        engine_port,
        port_busy,
        engine_ours,
    }
}

const GIB: f64 = (1u64 << 30) as f64;

fn gb(b: u64) -> String {
    format!("{:.0} GB", b as f64 / 1e9)
}

/// Run the probe and grade every check.
#[must_use]
pub fn probe(inp: &ProbeInput) -> SystemReport {
    let gpus = probe_gpus(&inp.root);
    let (host_ram_bytes, cpus) = host_ram(&inp.root);
    let mut checks = Vec::new();
    let soft = |s: CheckStatus| {
        if inp.needs_gpus || s == CheckStatus::Pass {
            s
        } else {
            CheckStatus::Info
        }
    };
    let usable: Vec<&GpuInfo> = gpus
        .iter()
        .filter(|g| g.supported && !g.integrated)
        .collect();
    let listing = gpus
        .iter()
        .map(|g| {
            let mut s = g.gfx.clone();
            if !g.name.is_empty() {
                s = format!("{} ({s})", g.name);
            }
            if let Some(v) = g.vram_bytes {
                s.push_str(&format!(", {:.0} GiB", v as f64 / GIB));
            }
            if g.integrated {
                s.push_str(", integrated: not used");
            } else if !g.supported {
                s.push_str(", not supported");
            }
            s
        })
        .collect::<Vec<_>>()
        .join("; ");
    let (st, fix) = match usable.len() {
        1 if inp.min_gpus <= 1 => (CheckStatus::Pass, None),
        0 => (
            CheckStatus::Fail,
            Some(
                "Saber needs AMD RDNA4 (gfx12) cards with the amdgpu driver loaded. If the cards \
                 are installed, check that the amdgpu kernel driver is loaded (`lsmod | grep \
                 amdgpu`)."
                    .to_string(),
            ),
        ),
        1 => (
            CheckStatus::Warn,
            Some(
                "One supported card: the 27B runs on it at TP=1 (pick \"One card\" in the Model \
                 step); Flash-Next and the 27B's TP=2 need two cards. The mock engine runs without \
                 a GPU."
                    .to_string(),
            ),
        ),
        _ => (CheckStatus::Pass, None),
    };
    checks.push(Check {
        id: "gpus",
        title: "GPUs".into(),
        status: soft(st),
        detail: if gpus.is_empty() {
            "No GPU found in the KFD topology".into()
        } else if usable.len() == 1 && inp.min_gpus <= 1 {
            format!("1 supported (enough for the 27B on one card, TP=1): {listing}")
        } else {
            format!("{} supported: {listing}", usable.len())
        },
        fix,
    });

    // VRAM
    if !usable.is_empty() {
        let min = usable.iter().filter_map(|g| g.vram_bytes).min();
        let st = match min {
            Some(v) if (v as f64) / GIB >= 30.0 => CheckStatus::Pass,
            Some(_) => CheckStatus::Warn,
            None => CheckStatus::Warn,
        };
        checks.push(Check {
            id: "vram",
            title: "GPU memory".into(),
            status: soft(st),
            detail: min.map_or("could not be read".into(), |v| {
                format!("{:.0} GiB on the smallest card", v as f64 / GIB)
            }),
            fix: (st != CheckStatus::Pass).then(|| {
                "The hosted models need about 30 GiB of VRAM a card (32 GB cards). Smaller cards \
                 can still run the mock engine."
                    .to_string()
            }),
        });
    }

    // PCIe
    if !usable.is_empty() {
        let worst = usable
            .iter()
            .map(|g| {
                if g.slot_max.width.is_some() {
                    g.slot_max.clone()
                } else {
                    g.link_max.clone()
                }
            })
            .min_by(|a, b| {
                let k = |l: &Link| l.speed_gts.unwrap_or(0.0) * f64::from(l.width.unwrap_or(0));
                k(a).total_cmp(&k(b))
            })
            .unwrap_or_default();
        let slow = worst.width.is_some_and(|w| w < 8) || worst.speed_gts.is_some_and(|s| s < 15.0);
        let per = usable
            .iter()
            .map(|g| {
                format!(
                    "{}: slot {} (card {})",
                    if g.bdf.is_empty() {
                        g.gfx.clone()
                    } else {
                        g.bdf.clone()
                    },
                    g.slot_max.label(),
                    g.link_max.label()
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        let st = if worst.width.is_none() {
            CheckStatus::Info
        } else if slow {
            CheckStatus::Warn
        } else {
            CheckStatus::Pass
        };
        checks.push(Check {
            id: "pcie",
            title: "PCIe links".into(),
            status: soft(st),
            detail: format!(
                "{per}. An idle link shows a lower current speed: that is power saving, not a fault."
            ),
            fix: slow.then(|| {
                "A slot narrower than x8 or older than Gen4 slows the cards' all-reduce. Move the \
                 cards to CPU-attached x8/x16 slots (see the motherboard manual)."
                    .to_string()
            }),
        });
    }

    // P2P
    if usable.len() >= 2 {
        let nodes: Vec<u32> = usable.iter().map(|g| g.node).collect();
        let missing: Vec<String> = usable
            .iter()
            .flat_map(|g| {
                nodes
                    .iter()
                    .filter(move |&&o| o != g.node && !g.p2p_peers.contains(&o))
                    .map(move |o| format!("{} -> {o}", g.node))
            })
            .collect();
        let st = if missing.is_empty() {
            CheckStatus::Pass
        } else {
            CheckStatus::Warn
        };
        checks.push(Check {
            id: "p2p",
            title: "Peer-to-peer between cards".into(),
            status: soft(st),
            detail: if missing.is_empty() {
                "every card pair has a peer-to-peer link (KFD topology)".into()
            } else {
                format!("no peer link for KFD nodes {}", missing.join(", "))
            },
            fix: (!missing.is_empty()).then(|| {
                "Without peer-to-peer the cards exchange data through host RAM (slower). Enable \
                 Above 4G decoding and Resizable BAR in the BIOS and keep IOMMU off or in \
                 passthrough mode."
                    .to_string()
            }),
        });
    }

    // /dev/kfd + render nodes
    let kfd = inp.root.join("dev/kfd");
    let kfd_ok = kfd.exists() && can_rw(&kfd);
    let renders: Vec<(u32, bool)> = usable
        .iter()
        .filter_map(|g| g.render_minor)
        .map(|m| (m, can_rw(&inp.root.join(format!("dev/dri/renderD{m}")))))
        .collect();
    let render_bad: Vec<String> = renders
        .iter()
        .filter(|(_, ok)| !ok)
        .map(|(m, _)| format!("/dev/dri/renderD{m}"))
        .collect();
    let st = if kfd_ok && render_bad.is_empty() {
        CheckStatus::Pass
    } else {
        CheckStatus::Fail
    };
    checks.push(Check {
        id: "kfd",
        title: "GPU device access".into(),
        status: soft(st),
        detail: if !kfd.exists() {
            "/dev/kfd does not exist (the amdgpu compute driver is not loaded)".into()
        } else if !kfd_ok {
            "/dev/kfd exists but this user cannot open it".into()
        } else if !render_bad.is_empty() {
            format!("no access to {}", render_bad.join(", "))
        } else {
            format!(
                "/dev/kfd and {} render node(s) are accessible",
                renders.len()
            )
        },
        fix: (st == CheckStatus::Fail).then(|| {
            if kfd.exists() {
                "Add your user to the render and video groups: `sudo usermod -aG render,video \
                 $USER`, then log out and back in."
                    .to_string()
            } else {
                "Load the amdgpu driver (a recent kernel has it built in; check `dmesg | grep \
                 amdgpu` for errors)."
                    .to_string()
            }
        }),
    });

    // container runtime + image
    let h = &inp.host;
    checks.push(Check {
        id: "runtime",
        title: "Container runtime".into(),
        status: soft(if h.runtime_version.is_some() {
            CheckStatus::Pass
        } else {
            CheckStatus::Fail
        }),
        detail: h
            .runtime_version
            .clone()
            .unwrap_or_else(|| format!("`{}` was not found", h.runtime)),
        fix: h.runtime_version.is_none().then(|| {
            "The engine runs inside a container (the host has no ROCm). Install podman (`sudo \
             dnf install podman` or `sudo apt install podman`), or set the runtime to docker in \
             Settings."
                .to_string()
        }),
    });
    let st = match h.image_present {
        Some(true) => CheckStatus::Pass,
        Some(false) => CheckStatus::Fail,
        None => CheckStatus::Warn,
    };
    checks.push(Check {
        id: "image",
        title: "Saber dev image".into(),
        status: soft(st),
        detail: match h.image_present {
            Some(true) => format!("{} is present", h.image),
            Some(false) => format!("{} is not built yet", h.image),
            None => "cannot check without a container runtime".into(),
        },
        fix: (st != CheckStatus::Pass).then(|| {
            "Build it once from the repo: `tools/build-image.sh` (needs network; about 20 \
             minutes)."
                .to_string()
        }),
    });

    // disks
    let mut disks = Vec::new();
    for (label, path) in [("packs", &inp.packs_dir), ("kv", &inp.kv_dir)] {
        let (probed, free, total) = disk_space(path);
        disks.push(Disk {
            label,
            path: path.display().to_string(),
            probed: probed.display().to_string(),
            free_bytes: free,
            total_bytes: total,
        });
    }
    let packs_free = disks[0].free_bytes;
    let need = inp.need_disk_bytes.unwrap_or(30_000_000_000);
    let st = match packs_free {
        Some(f) if f >= need => CheckStatus::Pass,
        Some(_) => CheckStatus::Warn,
        None => CheckStatus::Warn,
    };
    checks.push(Check {
        id: "disk",
        title: "Disk space".into(),
        status: st,
        detail: disks
            .iter()
            .map(|d| {
                format!(
                    "{} at {}: {}",
                    if d.label == "packs" {
                        "models"
                    } else {
                        "KV cache"
                    },
                    d.path,
                    d.free_bytes
                        .map_or("unknown".into(), |f| format!("{} free", gb(f)))
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
            + &inp.need_disk_bytes.map_or(String::new(), |n| {
                format!(" (the chosen model needs {})", gb(n))
            }),
        fix: (st != CheckStatus::Pass).then(|| {
            "Free up space on that drive, or point the NVMe path at a bigger drive in Settings \
             (paths.nvme). A fast NVMe drive is strongly recommended."
                .to_string()
        }),
    });

    // host RAM
    checks.push(Check {
        id: "ram",
        title: "Host memory".into(),
        status: match host_ram_bytes {
            Some(_) => CheckStatus::Info,
            None => CheckStatus::Warn,
        },
        detail: host_ram_bytes.map_or("could not be read".into(), |r| {
            format!(
                "{:.0} GiB{}",
                r as f64 / GIB,
                cpus.map_or(String::new(), |c| format!(", {c} CPU threads"))
            )
        }),
        fix: None,
    });

    // conflicts
    let mut conflicts = h.conflicts.clone();
    if h.port_busy && !h.engine_ours {
        conflicts.push(format!(
            "port {} (the engine's port) is already in use",
            h.engine_port
        ));
    }
    checks.push(Check {
        id: "conflicts",
        title: "Other servers".into(),
        status: if conflicts.is_empty() {
            CheckStatus::Pass
        } else {
            CheckStatus::Fail
        },
        detail: if conflicts.is_empty() {
            if h.engine_ours {
                "only this console's engine is running".into()
            } else {
                "nothing else holds the cards or the engine's port".into()
            }
        } else {
            conflicts.join("; ")
        },
        fix: (!conflicts.is_empty()).then(|| {
            "Only one server may use the cards. Stop the other server (or wait for the GPU job \
             to finish), or pick another engine port in the Network step."
                .to_string()
        }),
    });

    let overall = checks.iter().map(|c| c.status).max_by_key(|s| match s {
        CheckStatus::Fail => 3,
        CheckStatus::Warn => 2,
        CheckStatus::Pass => 1,
        CheckStatus::Info => 0,
    });
    SystemReport {
        gpus,
        host_ram_bytes,
        cpus,
        disks,
        host: inp.host.clone(),
        checks,
        overall,
        generated_ms: crate::now_ms(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake sysfs/dev/proc tree shaped like a two-card host: a CPU node, two R9700s behind on-board
    /// switches on x8 root ports, P2P between them, and an iGPU.
    pub(crate) fn fake_tree(root: &Path, p2p: bool, slot_width: u32) {
        let w = |p: &str, s: &str| {
            let f = root.join(p);
            std::fs::create_dir_all(f.parent().unwrap()).unwrap();
            std::fs::write(f, s).unwrap();
        };
        let kfd = "sys/class/kfd/kfd/topology/nodes";
        w(
            &format!("{kfd}/0/properties"),
            "cpu_cores_count 24\nsimd_count 0\ngfx_target_version 0\n",
        );
        w(
            &format!("{kfd}/0/mem_banks/0/properties"),
            "heap_type 0\nsize_in_bytes 64856244224\n",
        );
        w(
            &format!("{kfd}/0/io_links/0/properties"),
            "type 2\nnode_from 0\nnode_to 1\n",
        );
        let gpus = [
            (1u32, 128u32, "0000:00:01.1", "0000:03:00.0", 2u32),
            (2, 129, "0000:00:01.3", "0000:07:00.0", 1),
        ];
        for (node, minor, port, bdf, peer) in gpus {
            w(
                &format!("{kfd}/{node}/properties"),
                &format!("simd_count 128\ngfx_target_version 120001\ndrm_render_minor {minor}\n"),
            );
            w(
                &format!("{kfd}/{node}/mem_banks/0/properties"),
                "heap_type 1\nsize_in_bytes 34208743424\n",
            );
            w(
                &format!("{kfd}/{node}/io_links/0/properties"),
                &format!("type 2\nnode_from {node}\nnode_to 0\n"),
            );
            if p2p {
                w(
                    &format!("{kfd}/{node}/p2p_links/0/properties"),
                    &format!("type 2\nnode_from {node}\nnode_to {peer}\n"),
                );
                w(
                    &format!("{kfd}/{node}/p2p_links/1/properties"),
                    &format!("type 2\nnode_from {node}\nnode_to 3\n"),
                );
            }
            // root port (slot) -> switch up -> switch down -> the card
            let sw_up = format!("0000:{:02x}:00.0", if node == 1 { 1 } else { 5 });
            let sw_dn = format!("0000:{:02x}:00.0", if node == 1 { 2 } else { 6 });
            let base = format!("sys/devices/pci0000:00/{port}");
            let lk = |d: &str, cs: &str, cw: u32, ms: &str, mw: u32| {
                w(
                    &format!("{d}/current_link_speed"),
                    &format!("{cs} GT/s PCIe\n"),
                );
                w(&format!("{d}/current_link_width"), &format!("{cw}\n"));
                w(&format!("{d}/max_link_speed"), &format!("{ms} GT/s PCIe\n"));
                w(&format!("{d}/max_link_width"), &format!("{mw}\n"));
            };
            lk(&base, "2.5", slot_width, "32.0", slot_width);
            lk(&format!("{base}/{sw_up}"), "2.5", slot_width, "32.0", 16);
            lk(&format!("{base}/{sw_up}/{sw_dn}"), "32.0", 16, "32.0", 16);
            let card = format!("{base}/{sw_up}/{sw_dn}/{bdf}");
            lk(&card, "32.0", 16, "32.0", 16);
            w(&format!("{card}/mem_info_vram_total"), "34208743424\n");
            w(
                &format!("{card}/uevent"),
                &format!("DRIVER=amdgpu\nPCI_ID=1002:7551\nPCI_SLOT_NAME={bdf}\n"),
            );
            let link = root.join(format!("sys/class/drm/renderD{minor}"));
            std::fs::create_dir_all(&link).unwrap();
            std::os::unix::fs::symlink(
                format!("../../../devices/pci0000:00/{port}/{sw_up}/{sw_dn}/{bdf}"),
                link.join("device"),
            )
            .unwrap();
            w(&format!("dev/dri/renderD{minor}"), "");
        }
        // the iGPU
        w(
            &format!("{kfd}/3/properties"),
            "simd_count 4\ngfx_target_version 100306\ndrm_render_minor 130\n",
        );
        let ig = "sys/devices/pci0000:00/0000:00:08.1/0000:7f:00.0";
        w(&format!("{ig}/mem_info_vram_total"), "2147483648\n");
        w(
            &format!("{ig}/uevent"),
            "PCI_ID=1002:13C0\nPCI_SLOT_NAME=0000:7f:00.0\n",
        );
        let link = root.join("sys/class/drm/renderD130");
        std::fs::create_dir_all(&link).unwrap();
        std::os::unix::fs::symlink(
            "../../../devices/pci0000:00/0000:00:08.1/0000:7f:00.0",
            link.join("device"),
        )
        .unwrap();
        w("dev/kfd", "");
        w(
            "proc/meminfo",
            "MemTotal:       63336176 kB\nMemFree:  100 kB\n",
        );
        w("proc/cpuinfo", &"processor\t: 0\n\n".repeat(24));
    }

    fn good_host() -> HostFacts {
        HostFacts {
            runtime: "podman".into(),
            runtime_version: Some("podman version 5.6.0".into()),
            image: "localhost/saber-dev:latest".into(),
            image_present: Some(true),
            conflicts: vec![],
            engine_port: 8080,
            port_busy: false,
            engine_ours: false,
        }
    }

    fn input(root: &Path, host: HostFacts) -> ProbeInput {
        ProbeInput {
            root: root.to_path_buf(),
            packs_dir: root.join("nvme/saber-packs"),
            kv_dir: root.join("nvme/saber-kv"),
            need_disk_bytes: Some(1),
            needs_gpus: true,
            min_gpus: 2,
            host,
        }
    }

    fn status(r: &SystemReport, id: &str) -> CheckStatus {
        r.checks.iter().find(|c| c.id == id).unwrap().status
    }

    #[test]
    fn gfx_names() {
        assert_eq!(gfx_name(120001), "gfx1201");
        assert_eq!(gfx_name(100306), "gfx1036");
        assert_eq!(gfx_name(90010), "gfx90a");
        assert_eq!(gfx_name(110000), "gfx1100");
    }

    #[test]
    fn a_box_like_this_one_passes() {
        let dir = saber_test_scratch::ScratchDir::new("console-sys-good");
        fake_tree(dir.path(), true, 8);
        let r = probe(&input(dir.path(), good_host()));
        assert_eq!(r.gpus.len(), 3);
        let g = &r.gpus[0];
        assert_eq!(g.gfx, "gfx1201");
        assert_eq!(g.name, "Radeon AI PRO R9700");
        assert_eq!(g.bdf, "0000:03:00.0");
        assert_eq!(g.vram_bytes, Some(34_208_743_424));
        assert_eq!(g.link_max.label(), "Gen5 x16");
        // the slot is the narrowest capability on the way up: x8, and idle at 2.5 GT/s
        assert_eq!(g.slot_max.label(), "Gen5 x8");
        assert_eq!(g.slot_current.speed_gts, Some(2.5));
        // GPU peers only (the iGPU's node 3 counts as a GPU node; the CPU node 0 does not)
        assert_eq!(g.p2p_peers, [2, 3]);
        assert!(r.gpus[2].integrated);
        assert_eq!(r.host_ram_bytes, Some(63_336_176 * 1024));
        assert_eq!(r.cpus, Some(24));
        for id in [
            "gpus",
            "vram",
            "pcie",
            "p2p",
            "kfd",
            "runtime",
            "image",
            "conflicts",
        ] {
            assert_eq!(status(&r, id), CheckStatus::Pass, "{id}: {:?}", r.checks);
        }
        assert!(r
            .checks
            .iter()
            .all(|c| c.status != CheckStatus::Pass || c.fix.is_none()));
    }

    #[test]
    fn faults_are_graded_with_fixes() {
        let dir = saber_test_scratch::ScratchDir::new("console-sys-bad");
        fake_tree(dir.path(), false, 4);
        let mut h = good_host();
        h.runtime_version = None;
        h.image_present = None;
        h.conflicts = vec!["another saber server holds the cards (saber-27b-serve)".into()];
        h.port_busy = true;
        let r = probe(&input(dir.path(), h));
        assert_eq!(status(&r, "pcie"), CheckStatus::Warn);
        assert_eq!(status(&r, "p2p"), CheckStatus::Warn);
        assert_eq!(status(&r, "runtime"), CheckStatus::Fail);
        assert_eq!(status(&r, "image"), CheckStatus::Warn);
        let c = r.checks.iter().find(|c| c.id == "conflicts").unwrap();
        assert_eq!(c.status, CheckStatus::Fail);
        assert!(
            c.detail.contains("saber-27b-serve") && c.detail.contains("port 8080"),
            "{c:?}"
        );
        assert_eq!(r.overall, Some(CheckStatus::Fail));
        for c in &r.checks {
            if matches!(c.status, CheckStatus::Warn | CheckStatus::Fail) {
                assert!(c.fix.as_deref().is_some_and(|f| f.len() > 20), "{c:?}");
            }
        }
        // our own engine on the port is not a conflict
        let mut h = good_host();
        h.port_busy = true;
        h.engine_ours = true;
        let r = probe(&input(dir.path(), h));
        assert_eq!(status(&r, "conflicts"), CheckStatus::Pass);
    }

    #[test]
    fn no_gpus_fails_and_the_mock_softens_it() {
        let dir = saber_test_scratch::ScratchDir::new("console-sys-none");
        std::fs::create_dir_all(dir.path().join("proc")).unwrap();
        let r = probe(&input(dir.path(), good_host()));
        assert!(r.gpus.is_empty());
        assert_eq!(status(&r, "gpus"), CheckStatus::Fail);
        assert_eq!(status(&r, "kfd"), CheckStatus::Fail);
        let mut i = input(dir.path(), good_host());
        i.needs_gpus = false;
        let r = probe(&i);
        assert_eq!(status(&r, "gpus"), CheckStatus::Info);
        assert_eq!(status(&r, "kfd"), CheckStatus::Info);
    }

    #[test]
    fn disk_space_walks_up_to_an_existing_dir() {
        let dir = saber_test_scratch::ScratchDir::new("console-sys-disk");
        let (p, free, total) = disk_space(&dir.path().join("a/b/c"));
        assert_eq!(p, dir.path());
        assert!(free.unwrap() > 0 && total.unwrap() >= free.unwrap());
    }

    /// Read-only probe of THIS host (`cargo test -p saber-console -- --ignored live_system`).
    #[test]
    #[ignore = "reads the live host's sysfs and asks podman"]
    fn live_system_probe() {
        let h = gather_host(
            "podman",
            "localhost/saber-dev:latest",
            Path::new("/nonexistent"),
            8080,
            false,
        );
        let r = probe(&ProbeInput {
            root: PathBuf::from("/"),
            packs_dir: PathBuf::from("/mnt/nvme/saber-packs"),
            kv_dir: PathBuf::from("/mnt/nvme/saber-kv"),
            need_disk_bytes: None,
            needs_gpus: true,
            min_gpus: 2,
            host: h,
        });
        println!("{}", serde_json::to_string_pretty(&r).unwrap());
    }
}
