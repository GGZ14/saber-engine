//! `DeviceProfile` — what the CapabilityProbe reports, as data.
//!
//! `ARCHITECTURE.md` § *1. Control plane*: gfx target, WMMA shapes, LDS bytes, fp8/int4/2:4 support
//! and **measured** PCIe/DRAM bandwidths. Two rules are carried here explicitly:
//!
//! * **Wave size is a translation-unit property, not a device property** — `-mwavefrontsize64` is
//!   TU-wide and the per-function attribute is silently ignored. The device lists which wave sizes
//!   it *supports*; each [`crate::Implementation`] declares the one it was compiled with, and the
//!   planner checks the declaration against this list.
//! * Bandwidths are measurements with provenance, never nominal link rates. A number measured on
//!   other hardware is marked as such ([`BandwidthSource::MeasuredElsewhere`]) so a plan cannot
//!   quietly size a tier against an EPYC host's PCIe.

use serde::{Deserialize, Serialize};
use std::fmt;

/// 64 MiB — the gfx1201 Infinity Cache. The 09-14 cliff is a hard edge at this number.
pub const GFX1201_INFINITY_CACHE_BYTES: u64 = 64 * 1024 * 1024;

/// 64 KiB of LDS per workgroup on gfx1201.
pub const GFX1201_LDS_BYTES: u32 = 64 * 1024;

/// 31.86 GiB — the ReBAR-addressable VRAM the census reports on an R9700.
pub const GFX1201_VRAM_BYTES: u64 = 34_209_414_512;

/// One WMMA / SWMMAC instruction shape the target supports.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WmmaShape {
    /// Rows of the accumulator tile.
    pub m: u32,
    /// Columns of the accumulator tile.
    pub n: u32,
    /// Depth consumed per instruction.
    pub k: u32,
    /// A-operand dtype (`bf16`, `fp16`, `fp8_e4m3`, `iu8`, `iu4`).
    pub a_dtype: String,
    /// B-operand dtype.
    pub b_dtype: String,
    /// Accumulator dtype (`f32`, `i32`).
    pub acc_dtype: String,
    /// True for the 2:4 structured-sparse `swmmac` form (compute, not bandwidth — see
    /// `mxfp4-sparsity-verdict`).
    pub sparse_2_4: bool,
}

impl fmt::Display for WmmaShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}x{}x{} {}x{}->{}{}",
            self.m,
            self.n,
            self.k,
            self.a_dtype,
            self.b_dtype,
            self.acc_dtype,
            if self.sparse_2_4 { " (2:4)" } else { "" }
        )
    }
}

/// Coarse capability bits the planner and `explain` read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceCaps {
    /// Dense fp8 WMMA exists at *some* shape (gfx1201: 16x16x16, **not** 16x16x32).
    pub fp8_wmma: bool,
    /// Dense int8 WMMA (16x16x32 on gfx1201).
    pub int8_wmma: bool,
    /// Dense int4 WMMA (16x16x32 on gfx1201, ~2.1x fp8).
    pub int4_wmma: bool,
    /// 2:4 structured-sparse `swmmac` for fp8 / iu4.
    pub sparse_2_4_swmmac: bool,
    /// LDS-DMA (`global_load_lds`). Absent on gfx1201.
    pub lds_dma: bool,
    /// Non-temporal global loads. Present on gfx1201.
    pub nontemporal_loads: bool,
    /// A transpose read out of LDS. Absent on RDNA4.
    pub lds_transpose_read: bool,
}

/// Where a bandwidth number came from. A plan that sizes a tier against an imported number must be
/// able to say so out loud.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BandwidthSource {
    /// Measured on this box, with the harness and date that produced it.
    MeasuredHere {
        /// Harness name.
        harness: String,
        /// ISO-8601 date.
        when: String,
    },
    /// Measured on *other* hardware and imported pending re-measurement (open item 1).
    MeasuredElsewhere {
        /// What it was measured on.
        hardware: String,
        /// Why it is still here.
        note: String,
    },
    /// A link rate off a spec sheet. Never valid for sizing.
    Nominal,
}

impl fmt::Display for BandwidthSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BandwidthSource::MeasuredHere { harness, when } => {
                write!(f, "measured here ({harness}, {when})")
            }
            BandwidthSource::MeasuredElsewhere { hardware, note } => {
                write!(f, "measured on {hardware} — {note}")
            }
            BandwidthSource::Nominal => f.write_str("nominal (not measured)"),
        }
    }
}

/// One bandwidth number and where it came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bandwidth {
    /// GB/s (10^9 bytes per second, the unit every harness on this box reports).
    pub gbs: f64,
    /// Provenance of the number.
    pub source: BandwidthSource,
}

impl Bandwidth {
    /// A number measured on this box.
    pub fn here(gbs: f64, harness: &str, when: &str) -> Self {
        Self {
            gbs,
            source: BandwidthSource::MeasuredHere {
                harness: harness.to_string(),
                when: when.to_string(),
            },
        }
    }

    /// A number imported from other hardware, pending re-measurement.
    pub fn elsewhere(gbs: f64, hardware: &str, note: &str) -> Self {
        Self {
            gbs,
            source: BandwidthSource::MeasuredElsewhere {
                hardware: hardware.to_string(),
                note: note.to_string(),
            },
        }
    }
}

impl fmt::Display for Bandwidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1} GB/s [{}]", self.gbs, self.source)
    }
}

/// The three bandwidths the memory plane budgets against.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeasuredBandwidths {
    /// VRAM read bandwidth.
    pub dram: Bandwidth,
    /// Pinned host to device over PCIe, per card.
    pub pcie_h2d: Bandwidth,
    /// Device-to-device across the IOD (no XGMI on this box).
    pub p2p: Bandwidth,
}

/// Everything the planner needs to know about the target device.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeviceProfile {
    /// LLVM target, e.g. `gfx1201`.
    pub gfx_target: String,
    /// Marketing name, for `explain` only.
    pub name: String,
    /// Compute units.
    pub cu_count: u32,
    /// Every WMMA/SWMMAC shape the target implements.
    pub wmma_shapes: Vec<WmmaShape>,
    /// LDS bytes available to one workgroup.
    pub lds_bytes: u32,
    /// Wave sizes the target supports. *Which one a kernel uses is a property of the kernel.*
    pub wave_sizes: Vec<u32>,
    /// Capability bits.
    pub caps: DeviceCaps,
    /// Measured bandwidths with provenance.
    pub bandwidths: MeasuredBandwidths,
    /// Infinity Cache size in bytes. The GDN speculative-state bucket check is against this.
    pub infinity_cache_bytes: u64,
    /// VRAM bytes per card.
    pub vram_bytes: u64,
}

impl DeviceProfile {
    /// The known-good gfx1201 (Radeon AI PRO R9700) profile.
    ///
    /// Numbers, each traceable to a file on this box:
    ///
    /// * DRAM 635 GB/s measured (`rdna4-dram-peak-635` — *not* the 576 the startup sweep prints;
    ///   a small repeated buffer reads off the Infinity Cache and fakes >100% of peak).
    /// * PCIe H2D 27.7 GB/s — both GPU root ports are Gen5 x8 (`wht6-ar-decomposition`, 09-16).
    ///   Host→device on *this* box is open item 1; the 28.4 GB/s figure in the tiering docs was
    ///   measured on an EPYC Gen4 host and is marked as imported.
    /// * P2P 27.7 GB/s across the IOD, `p2p_links` type 2 (no XGMI).
    /// * Infinity Cache 64 MiB — the 09-14 GDN cliff.
    /// * LDS 64 KiB/workgroup; wave32 and wave64 both supported (wave64 is ~10% slower per FLOP).
    /// * No dense 16x16x32 fp8 WMMA; 2:4 `swmmac` for fp8/iu4 *is* present; no LDS-DMA; no LDS
    ///   transpose read; non-temporal loads present (`gfx1201-hw-limits`, `rdna4-fp8-wmma-2x`).
    pub fn gfx1201_reference() -> Self {
        let wmma = |m: u32, n: u32, k: u32, a: &str, acc: &str, sparse: bool| WmmaShape {
            m,
            n,
            k,
            a_dtype: a.to_string(),
            b_dtype: a.to_string(),
            acc_dtype: acc.to_string(),
            sparse_2_4: sparse,
        };
        Self {
            gfx_target: "gfx1201".to_string(),
            name: "Radeon AI PRO R9700".to_string(),
            cu_count: 64,
            wmma_shapes: vec![
                wmma(16, 16, 16, "bf16", "f32", false),
                wmma(16, 16, 16, "fp16", "f32", false),
                wmma(16, 16, 16, "fp8_e4m3", "f32", false),
                wmma(16, 16, 16, "fp8_e5m2", "f32", false),
                wmma(16, 16, 32, "iu8", "i32", false),
                wmma(16, 16, 32, "iu4", "i32", false),
                wmma(16, 16, 32, "fp8_e4m3", "f32", true),
                wmma(16, 16, 64, "iu4", "i32", true),
            ],
            lds_bytes: GFX1201_LDS_BYTES,
            wave_sizes: vec![32, 64],
            caps: DeviceCaps {
                fp8_wmma: true,
                int8_wmma: true,
                int4_wmma: true,
                sparse_2_4_swmmac: true,
                lds_dma: false,
                nontemporal_loads: true,
                lds_transpose_read: false,
            },
            bandwidths: MeasuredBandwidths {
                dram: Bandwidth::here(635.0, "rdna4-dram-peak sweep", "2026-08-28"),
                pcie_h2d: Bandwidth::elsewhere(
                    28.4,
                    "EPYC 74F3 / H12SSL-NT, Gen4 x16",
                    "open item 1: re-measure H2D on this box before any tier sizing depends on it",
                ),
                p2p: Bandwidth::here(27.7, "wht6 all-reduce decomposition", "2026-09-16"),
            },
            infinity_cache_bytes: GFX1201_INFINITY_CACHE_BYTES,
            vram_bytes: GFX1201_VRAM_BYTES,
        }
    }

    /// Whether this target supports a wave size an implementation declares.
    pub fn supports_wave_size(&self, wave: u32) -> bool {
        self.wave_sizes.contains(&wave)
    }

    /// Canonical digest of the profile, folded into the plan's input hash.
    pub fn digest(&self) -> String {
        crate::hash::digest_of(self)
    }
}
