//! The hardware sampler: per-GPU power, temperatures, clocks, VRAM and busy % from
//! amdgpu's sysfs (`/sys/class/drm/card*/device`, its `hwmon`), and the host's load, memory and
//! CPU busy from `/proc`. The engine exports none of these; the console reads them itself.
//! Read-only: nothing here writes sysfs (the power cap is never touched, only displayed).
//!
//! Host DRAM bandwidth has no unprivileged counter on these hosts (AMD's DF PMU needs perf
//! privileges), so it is not sampled.

use std::path::{Path, PathBuf};

use serde::Serialize;

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct GpuSample {
    /// The DRM card number (`card1` = 1).
    pub card: u32,
    /// The PCI address (`0000:03:00.0`).
    pub bdf: String,
    pub power_w: Option<f64>,
    pub power_cap_w: Option<f64>,
    pub temp_edge_c: Option<f64>,
    pub temp_junction_c: Option<f64>,
    pub temp_mem_c: Option<f64>,
    pub sclk_mhz: Option<f64>,
    pub mclk_mhz: Option<f64>,
    pub fan_rpm: Option<f64>,
    pub vram_used: Option<f64>,
    pub vram_total: Option<f64>,
    pub busy_pct: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct HostSample {
    pub load1: Option<f64>,
    pub load5: Option<f64>,
    pub load15: Option<f64>,
    pub mem_total: Option<f64>,
    pub mem_available: Option<f64>,
    pub swap_total: Option<f64>,
    pub swap_free: Option<f64>,
    pub cpu_busy_pct: Option<f64>,
    pub cpus: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct HwSample {
    pub gpus: Vec<GpuSample>,
    pub host: HostSample,
}

impl HwSample {
    /// The sample as series for the metrics ring (`hw_*` names, gauges).
    #[must_use]
    pub fn series(&self) -> Vec<(String, f64)> {
        let mut o = Vec::new();
        for g in &self.gpus {
            let l = format!("{{gpu=\"{}\"}}", g.card);
            for (n, v) in [
                ("hw_gpu_power_watts", g.power_w),
                ("hw_gpu_power_cap_watts", g.power_cap_w),
                ("hw_gpu_temp_edge_celsius", g.temp_edge_c),
                ("hw_gpu_temp_junction_celsius", g.temp_junction_c),
                ("hw_gpu_temp_mem_celsius", g.temp_mem_c),
                ("hw_gpu_sclk_mhz", g.sclk_mhz),
                ("hw_gpu_mclk_mhz", g.mclk_mhz),
                ("hw_gpu_fan_rpm", g.fan_rpm),
                ("hw_gpu_vram_used_bytes", g.vram_used),
                ("hw_gpu_vram_total_bytes", g.vram_total),
                ("hw_gpu_busy_percent", g.busy_pct),
            ] {
                if let Some(v) = v {
                    o.push((format!("{n}{l}"), v));
                }
            }
        }
        let h = &self.host;
        for (n, v) in [
            ("hw_host_load1", h.load1),
            ("hw_host_load5", h.load5),
            ("hw_host_load15", h.load15),
            ("hw_host_mem_total_bytes", h.mem_total),
            ("hw_host_mem_available_bytes", h.mem_available),
            ("hw_host_swap_total_bytes", h.swap_total),
            ("hw_host_swap_free_bytes", h.swap_free),
            ("hw_host_cpu_busy_percent", h.cpu_busy_pct),
        ] {
            if let Some(v) = v {
                o.push((n.to_string(), v));
            }
        }
        o
    }

    /// Total GPU power now (W), when any card reports it.
    #[must_use]
    pub fn total_power_w(&self) -> Option<f64> {
        let v: Vec<f64> = self.gpus.iter().filter_map(|g| g.power_w).collect();
        (!v.is_empty()).then(|| v.iter().sum())
    }
}

fn read(p: &Path) -> Option<String> {
    std::fs::read_to_string(p)
        .ok()
        .map(|s| s.trim().to_string())
}

fn num(p: &Path) -> Option<f64> {
    read(p)?.parse::<f64>().ok()
}

/// Samples a filesystem root (`/` on the host; a fake tree in tests).
pub struct HwSampler {
    root: PathBuf,
    prev_cpu: Option<(u64, u64)>,
}

impl HwSampler {
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> HwSampler {
        HwSampler {
            root: root.into(),
            prev_cpu: None,
        }
    }

    pub fn sample(&mut self) -> HwSample {
        HwSample {
            gpus: self.gpus(),
            host: self.host(),
        }
    }

    fn gpus(&self) -> Vec<GpuSample> {
        let drm = self.root.join("sys/class/drm");
        let Ok(rd) = std::fs::read_dir(&drm) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let Some(n) = name
                .strip_prefix("card")
                .and_then(|n| n.parse::<u32>().ok())
            else {
                continue;
            };
            let dev = e.path().join("device");
            if read(&dev.join("vendor")).as_deref() != Some("0x1002") {
                continue;
            }
            let bdf = std::fs::read_link(&dev)
                .ok()
                .and_then(|p| p.file_name().map(|f| f.to_string_lossy().to_string()))
                .unwrap_or_default();
            let mut g = GpuSample {
                card: n,
                bdf,
                vram_used: num(&dev.join("mem_info_vram_used")),
                vram_total: num(&dev.join("mem_info_vram_total")),
                busy_pct: num(&dev.join("gpu_busy_percent")),
                ..GpuSample::default()
            };
            if let Some(hw) = std::fs::read_dir(dev.join("hwmon"))
                .ok()
                .and_then(|mut d| d.find_map(|x| x.ok()))
                .map(|x| x.path())
            {
                g.power_w = num(&hw.join("power1_average"))
                    .or_else(|| num(&hw.join("power1_input")))
                    .map(|uw| uw / 1e6);
                g.power_cap_w = num(&hw.join("power1_cap")).map(|uw| uw / 1e6);
                g.fan_rpm = num(&hw.join("fan1_input"));
                for i in 1..=4 {
                    let label = read(&hw.join(format!("temp{i}_label")));
                    let v = num(&hw.join(format!("temp{i}_input"))).map(|m| m / 1e3);
                    match label.as_deref() {
                        Some("edge") => g.temp_edge_c = v,
                        Some("junction") => g.temp_junction_c = v,
                        Some("mem") => g.temp_mem_c = v,
                        _ => {}
                    }
                }
                for i in 1..=3 {
                    let label = read(&hw.join(format!("freq{i}_label")));
                    let v = num(&hw.join(format!("freq{i}_input"))).map(|hz| hz / 1e6);
                    match label.as_deref() {
                        Some("sclk") => g.sclk_mhz = v,
                        Some("mclk") => g.mclk_mhz = v,
                        _ => {}
                    }
                }
            }
            out.push(g);
        }
        out.sort_by_key(|g| g.card);
        out
    }

    fn host(&mut self) -> HostSample {
        let mut h = HostSample::default();
        if let Some(l) = read(&self.root.join("proc/loadavg")) {
            let mut it = l.split_whitespace().map(|x| x.parse::<f64>().ok());
            h.load1 = it.next().flatten();
            h.load5 = it.next().flatten();
            h.load15 = it.next().flatten();
        }
        if let Some(m) = read(&self.root.join("proc/meminfo")) {
            for line in m.lines() {
                let mut it = line.split_whitespace();
                let (Some(k), Some(v)) = (it.next(), it.next().and_then(|v| v.parse::<f64>().ok()))
                else {
                    continue;
                };
                let bytes = Some(v * 1024.0);
                match k {
                    "MemTotal:" => h.mem_total = bytes,
                    "MemAvailable:" => h.mem_available = bytes,
                    "SwapTotal:" => h.swap_total = bytes,
                    "SwapFree:" => h.swap_free = bytes,
                    _ => {}
                }
            }
        }
        if let Some(s) = read(&self.root.join("proc/stat")) {
            let mut cpus = 0u32;
            for line in s.lines() {
                if line.starts_with("cpu") && !line.starts_with("cpu ") {
                    cpus += 1;
                }
                if let Some(rest) = line.strip_prefix("cpu ") {
                    let v: Vec<u64> = rest
                        .split_whitespace()
                        .filter_map(|x| x.parse().ok())
                        .collect();
                    let total: u64 = v.iter().take(8).sum();
                    let idle = v.get(3).copied().unwrap_or(0) + v.get(4).copied().unwrap_or(0);
                    if let Some((pt, pi)) = self.prev_cpu {
                        let dt = total.saturating_sub(pt);
                        if dt > 0 {
                            let di = idle.saturating_sub(pi);
                            h.cpu_busy_pct = Some(100.0 * (dt - di.min(dt)) as f64 / dt as f64);
                        }
                    }
                    self.prev_cpu = Some((total, idle));
                }
            }
            h.cpus = (cpus > 0).then_some(f64::from(cpus));
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_fake_sysfs_tree() {
        let dir = saber_test_scratch::ScratchDir::new("console-hw");
        let root = dir.path();
        let pci = root.join("sys/devices/pci0000:00/0000:03:00.0");
        let hw = pci.join("hwmon/hwmon4");
        std::fs::create_dir_all(&hw).unwrap();
        std::fs::create_dir_all(root.join("sys/class/drm/card1")).unwrap();
        std::os::unix::fs::symlink(&pci, root.join("sys/class/drm/card1/device")).unwrap();
        std::fs::create_dir_all(root.join("sys/class/drm/card1-DP-1")).unwrap();
        let w = |p: &Path, s: &str| std::fs::write(p, s).unwrap();
        w(&pci.join("vendor"), "0x1002\n");
        w(&pci.join("mem_info_vram_used"), "1073741824\n");
        w(&pci.join("mem_info_vram_total"), "34359738368\n");
        w(&pci.join("gpu_busy_percent"), "87\n");
        w(&hw.join("power1_average"), "209000000\n");
        w(&hw.join("power1_cap"), "210000000\n");
        w(&hw.join("temp1_label"), "edge\n");
        w(&hw.join("temp1_input"), "61000\n");
        w(&hw.join("temp2_label"), "junction\n");
        w(&hw.join("temp2_input"), "78000\n");
        w(&hw.join("freq1_label"), "sclk\n");
        w(&hw.join("freq1_input"), "2400000000\n");
        std::fs::create_dir_all(root.join("proc")).unwrap();
        w(&root.join("proc/loadavg"), "0.50 1.25 2.00 1/500 99\n");
        w(
            &root.join("proc/meminfo"),
            "MemTotal: 1000 kB\nMemAvailable: 400 kB\n",
        );
        w(
            &root.join("proc/stat"),
            "cpu  100 0 100 800 0 0 0 0\ncpu0 1\ncpu1 1\n",
        );
        let mut s = HwSampler::new(root);
        let a = s.sample();
        assert_eq!(a.gpus.len(), 1);
        let g = &a.gpus[0];
        assert_eq!((g.card, g.bdf.as_str()), (1, "0000:03:00.0"));
        assert_eq!(g.power_w, Some(209.0));
        assert_eq!(g.power_cap_w, Some(210.0));
        assert_eq!((g.temp_edge_c, g.temp_junction_c), (Some(61.0), Some(78.0)));
        assert_eq!(g.sclk_mhz, Some(2400.0));
        assert_eq!(g.busy_pct, Some(87.0));
        assert_eq!(a.host.load1, Some(0.5));
        assert_eq!(a.host.mem_available, Some(400.0 * 1024.0));
        assert_eq!(a.host.cpus, Some(2.0));
        assert_eq!(a.host.cpu_busy_pct, None, "the first sample has no delta");
        w(
            &root.join("proc/stat"),
            "cpu  150 0 150 900 0 0 0 0\ncpu0 1\ncpu1 1\n",
        );
        let b = s.sample();
        assert_eq!(b.host.cpu_busy_pct, Some(50.0));
        assert_eq!(b.total_power_w(), Some(209.0));
        assert!(b
            .series()
            .iter()
            .any(|(k, v)| k == "hw_gpu_power_watts{gpu=\"1\"}" && *v == 209.0));
    }
}
