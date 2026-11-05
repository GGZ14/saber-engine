//! Device attributes and the capability probe.
//!
//! Everything here goes through `hipDeviceGetAttribute`, `hipDeviceGetName`, `hipDeviceTotalMem`
//! and `hipDeviceGetPCIBusId` -- **never `hipGetDeviceProperties`**. `hipDeviceProp_t` is a large,
//! versioned struct (`hipGetDevicePropertiesR0600` behind a macro) whose layout has changed more
//! than once; reproducing it by hand in a crate that deliberately links no ROCm would be one
//! silently wrong offset away from garbage. Individual attribute queries cost one call each and
//! cannot be misaligned.
//!
//! The attribute ids are the values of `hipDeviceAttribute_t` in HIP 7.14.60850, read out of
//! `/opt/rocm/include/hip/hip_runtime_api.h` in the dev image by compiling and running the enum
//! rather than by counting declarations. The CUDA-compatible block (< 10000) is ordered to match
//! CUDA's and is stable; the AMD block starts at 10000.
//!
//! One casualty worth naming: **`hipDeviceAttributeGcnArch` and `hipDeviceAttributeGcnArchName` no
//! longer exist** -- HIP 7 turned them into `hipDeviceAttributeUnused4` and `...Unused5`, and the
//! only remaining HIP-side source of the target *name* is `hipDeviceProp_t::gcnArchName`, i.e. the
//! struct this module refuses to reproduce. Measured on this box: `gcnArchName` is `"gfx1201"`
//! while `(ComputeCapabilityMajor, ComputeCapabilityMinor)` is `(12, 0)` -- the minor field is the
//! ISA *minor*, not the stepping, so the pair alone names the wrong target (`gfx1200`).
//!
//! So the gfx target comes from the **KFD topology** instead:
//! `/sys/class/kfd/kfd/topology/nodes/*/properties` publishes `gfx_target_version` (120001 for
//! gfx1201, 90010 for gfx90a) and `location_id`, which is the device's PCI bus/device/function and
//! matches `hipDeviceGetPCIBusId`. That is the kernel driver's own statement of the ISA, it needs
//! no HIP context, and it is exact including the hex stepping. When no node matches, the target is
//! derived from `(major, minor, AsicRevision)` -- correct here, but an inference -- and
//! [`ArchSource`] records which route produced it.

use crate::error::Result;
use crate::thread::HipThread;

/// `hipDeviceAttribute_t` ids used by the engine.
///
/// A plain `i32`-valued enum rather than a set of constants so that a typo is a compile error.
/// Values are from HIP 7.14.60850; a runtime that renumbers them is a different major ROCm and is
/// a re-verification, not a patch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum DeviceAttribute {
    /// Peak clock frequency, kHz.
    ClockRate = 5,
    /// Device can run several kernels concurrently.
    ConcurrentKernels = 8,
    /// Size of L2 cache, bytes.
    L2CacheSize = 19,
    /// Major compute capability. On AMD this is the gfx family: 12 for gfx12xx.
    ComputeCapabilityMajor = 23,
    /// Maximum block dimension in X.
    MaxBlockDimX = 26,
    /// Maximum grid dimension in X.
    MaxGridDimX = 29,
    /// Maximum work-items per workgroup.
    MaxThreadsPerBlock = 56,
    /// Maximum resident work-items per compute unit.
    MaxThreadsPerMultiProcessor = 57,
    /// Global memory bus width, bits.
    MemoryBusWidth = 59,
    /// Peak memory clock, kHz.
    MemoryClockRate = 60,
    /// Minor compute capability. On AMD this encodes the rest of the gfx number.
    ComputeCapabilityMinor = 61,
    /// Number of compute units. **Halves in WGP mode**, where one workgroup processor is two CUs.
    MultiprocessorCount = 63,
    /// Device supports coherent access to pageable host memory.
    PageableMemoryAccess = 65,
    /// PCI bus id.
    PciBusId = 67,
    /// PCI device (slot) id.
    PciDeviceId = 68,
    /// PCI domain id.
    PciDomainId = 69,
    /// 32-bit registers available per block.
    MaxRegistersPerBlock = 71,
    /// Maximum LDS per workgroup, bytes. 65536 on gfx1201.
    MaxSharedMemoryPerBlock = 74,
    /// Constant memory, bytes.
    TotalConstantMemory = 83,
    /// Global memory, bytes. (`hipDeviceTotalMem` reports the same thing without truncation to
    /// `int`, and is what [`DeviceInfo`] uses.)
    TotalGlobalMem = 84,
    /// Wavefront size. **A device-level default only**: wave size is a translation-unit property,
    /// `-mwavefrontsize64` is TU-wide, and the per-function attribute is silently ignored, so each
    /// kernel implementation declares its own in the plugin manifest.
    WarpSize = 87,
    /// Frequency of the device-side `clock()` timer, kHz. (AMD block.)
    ClockInstructionRate = 10000,
    /// Maximum LDS per compute unit, bytes. (AMD block.)
    MaxSharedMemoryPerMultiprocessor = 10002,
    /// Whether the BAR covers all of VRAM. (AMD block.)
    IsLargeBar = 10011,
    /// ASIC revision. (AMD block.)
    AsicRevision = 10012,
    /// Physical compute units, independent of CU/WGP mode. (AMD block.)
    PhysicalMultiProcessorCount = 10015,
    /// Whether fine-grained (host-coherent) allocations are supported -- what the all-reduce flag
    /// regions need. (AMD block.)
    FineGrainSupport = 10016,
    /// Constant wall-clock frequency, kHz. (AMD block.)
    WallClockRate = 10017,
    /// Number of XCCs in the current compute partition; 1 on RDNA. (AMD block.)
    NumberOfXccs = 10018,
    /// GPU manufacturer device id. (AMD block.)
    PciChipId = 10020,
}

impl DeviceAttribute {
    /// The raw attribute id.
    #[must_use]
    pub const fn id(self) -> i32 {
        self as i32
    }
}

/// A decoded HIP version triple.
///
/// HIP packs versions as `major * 10_000_000 + minor * 100_000 + patch`, so 7.14.60850 arrives as
/// 71_460_850.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipVersion {
    /// Major.
    pub major: i32,
    /// Minor.
    pub minor: i32,
    /// Patch.
    pub patch: i32,
    /// The packed value HIP returned, kept so nothing is lost if the packing ever changes.
    pub raw: i32,
}

impl HipVersion {
    /// Decode a packed HIP version.
    #[must_use]
    pub const fn decode(raw: i32) -> HipVersion {
        HipVersion {
            major: raw / 10_000_000,
            minor: (raw / 100_000) % 100,
            patch: raw % 100_000,
            raw,
        }
    }
}

impl std::fmt::Display for HipVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}.{}.{} ({})",
            self.major, self.minor, self.patch, self.raw
        )
    }
}

/// Where a device's gfx target came from.
///
/// This enum **is** the which-path counter for the only route in the probe that has a fallback
/// (CLAUDE.md rule 5): a route with a fallback branch declares a counter or it does not merge.
/// `saber probe` prints it, so a derived target never passes for a reported one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchSource {
    /// Exact. The amdgpu/KFD topology node's `gfx_target_version`, matched to the device by PCI
    /// location. This is the kernel driver's own statement of the ISA the device runs.
    KfdTopology,
    /// Derived, from `(ComputeCapabilityMajor, ComputeCapabilityMinor, AsicRevision)`. Correct on
    /// this box (12, 0, 1 -> gfx1201) but it is an inference about how ROCclr fills those fields,
    /// not a report, so it is labelled.
    HipAttributes,
}

impl std::fmt::Display for ArchSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArchSource::KfdTopology => f.write_str("KFD topology (exact)"),
            ArchSource::HipAttributes => f.write_str("derived from HIP attributes"),
        }
    }
}

/// Everything the probe reports about one device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// HIP device ordinal.
    pub index: i32,
    /// Marketing name from `hipDeviceGetName`.
    pub name: String,
    /// The gfx target this device runs, e.g. `gfx1201`. See [`DeviceInfo::arch_source`] for where
    /// it came from; it is never guessed silently.
    pub arch: String,
    /// Which route produced [`DeviceInfo::arch`] — the which-path counter for the one fallback in
    /// the probe.
    pub arch_source: ArchSource,
    /// Major compute capability, raw. On AMD this is the ISA major version (12 for gfx12xx).
    pub cc_major: i32,
    /// Minor compute capability, raw. On AMD this is the ISA minor version — **not** the stepping,
    /// which is why it alone cannot name the target (gfx1201 reports `(12, 0)`).
    pub cc_minor: i32,
    /// `hipDeviceAttributeAsicRevision`, which on this runtime carries the ISA stepping (1 for
    /// gfx1201). Reported raw because the arch derivation depends on it.
    pub asic_revision: i32,
    /// Total VRAM, bytes, from `hipDeviceTotalMem`.
    pub total_mem: u64,
    /// Free VRAM, bytes, from `hipMemGetInfo` -- which needs a context on the device, so it can
    /// legitimately fail (for example when another process has claimed nearly all of VRAM). `None`
    /// carries the reason in the log rather than a zero that reads like a measurement.
    pub free_mem: Option<u64>,
    /// Compute units as HIP counts them (halved in WGP mode).
    pub cu_count: i32,
    /// Physical compute units, mode-independent.
    pub physical_cu_count: i32,
    /// Device-default wavefront size. Per-kernel wave size lives in the plugin manifest.
    pub warp_size: i32,
    /// Maximum LDS per workgroup, bytes.
    pub lds_per_block: i32,
    /// Maximum LDS per compute unit, bytes.
    pub lds_per_cu: i32,
    /// Maximum work-items per workgroup.
    pub max_threads_per_block: i32,
    /// 32-bit registers per workgroup.
    pub max_registers_per_block: i32,
    /// Peak core clock, kHz.
    pub clock_khz: i32,
    /// Peak memory clock, kHz.
    pub memory_clock_khz: i32,
    /// Memory bus width, bits.
    pub memory_bus_bits: i32,
    /// L2 cache, bytes.
    pub l2_bytes: i32,
    /// PCI bus id string, e.g. `0000:03:00.0`.
    pub pci_bus_id: String,
    /// Whether fine-grained (host-coherent) device allocations are available.
    pub fine_grain: bool,
    /// Whether the BAR covers all of VRAM.
    pub large_bar: bool,
}

impl DeviceInfo {
    /// Probe one device. Attribute queries need no context; `hipMemGetInfo` does, so it is
    /// attempted separately by [`DeviceInfo::probe_free_mem`].
    pub fn probe(hip: &HipThread, index: i32) -> Result<DeviceInfo> {
        let attr = |a: DeviceAttribute| hip.device_attribute(index, a.id());

        let cc_major = attr(DeviceAttribute::ComputeCapabilityMajor)?;
        let cc_minor = attr(DeviceAttribute::ComputeCapabilityMinor)?;
        let asic_revision = attr(DeviceAttribute::AsicRevision)?;
        let pci_bus_id = hip.device_pci_bus_id(index)?;
        let (arch, arch_source) = resolve_arch(&pci_bus_id, cc_major, cc_minor, asic_revision);

        let info = DeviceInfo {
            index,
            name: hip.device_name(index)?,
            arch,
            arch_source,
            cc_major,
            cc_minor,
            asic_revision,
            total_mem: hip.device_total_mem(index)?,
            free_mem: None,
            cu_count: attr(DeviceAttribute::MultiprocessorCount)?,
            physical_cu_count: attr(DeviceAttribute::PhysicalMultiProcessorCount)?,
            warp_size: attr(DeviceAttribute::WarpSize)?,
            lds_per_block: attr(DeviceAttribute::MaxSharedMemoryPerBlock)?,
            lds_per_cu: attr(DeviceAttribute::MaxSharedMemoryPerMultiprocessor)?,
            max_threads_per_block: attr(DeviceAttribute::MaxThreadsPerBlock)?,
            max_registers_per_block: attr(DeviceAttribute::MaxRegistersPerBlock)?,
            clock_khz: attr(DeviceAttribute::ClockRate)?,
            memory_clock_khz: attr(DeviceAttribute::MemoryClockRate)?,
            memory_bus_bits: attr(DeviceAttribute::MemoryBusWidth)?,
            l2_bytes: attr(DeviceAttribute::L2CacheSize)?,
            pci_bus_id,
            fine_grain: attr(DeviceAttribute::FineGrainSupport)? != 0,
            large_bar: attr(DeviceAttribute::IsLargeBar)? != 0,
        };
        Ok(info)
    }

    /// Fill in [`DeviceInfo::free_mem`] by making `index` current and calling `hipMemGetInfo`.
    ///
    /// Separate from [`DeviceInfo::probe`] because it is the only part that creates a context on
    /// the device, and a caller sharing the box with a live server may not want to.
    pub fn probe_free_mem(&mut self, hip: &HipThread) -> Result<()> {
        hip.set_device(self.index)?;
        match hip.mem_info() {
            Ok((free, _total)) => {
                self.free_mem = Some(free);
                Ok(())
            }
            Err(e) => {
                tracing::warn!(device = self.index, error = %e,
                    "hipMemGetInfo failed; reporting free VRAM as unknown rather than as zero");
                self.free_mem = None;
                Ok(())
            }
        }
    }
}

/// The KFD topology directory. A constant rather than a parameter: it is the kernel's fixed
/// interface, and making it configurable would only add a way to point the probe at a lie.
const KFD_TOPOLOGY_NODES: &str = "/sys/class/kfd/kfd/topology/nodes";

/// Format an ISA triple the way LLVM names targets: major and minor in decimal, stepping in **hex**
/// (which is what makes `gfx90a` `gfx90a` and not `gfx9010`).
fn gfx_name(major: u32, minor: u32, stepping: u32) -> String {
    format!("gfx{major}{minor}{stepping:x}")
}

/// Parse a PCI bus id string (`0000:03:00.0`) into the KFD's `location_id` encoding,
/// `(bus << 8) | (device << 3) | function`.
fn location_id_of(pci_bus_id: &str) -> Option<u32> {
    // `domain:bus:device.function`, all hex.
    let (head, func) = pci_bus_id.rsplit_once('.')?;
    let mut parts = head.split(':');
    let _domain = parts.next()?;
    let bus = parts.next()?;
    let dev = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    let bus = u32::from_str_radix(bus.trim(), 16).ok()?;
    let dev = u32::from_str_radix(dev.trim(), 16).ok()?;
    let func = u32::from_str_radix(func.trim().trim_end_matches('\0'), 16).ok()?;
    Some((bus << 8) | (dev << 3) | func)
}

/// Read `gfx_target_version` and `location_id` out of every KFD topology node that has a GPU.
///
/// `gfx_target_version` is packed `major * 10_000 + minor * 100 + stepping`, so gfx1201 is 120001
/// and gfx90a is 90010. Nodes with version 0 are CPU nodes and are skipped.
fn kfd_targets_by_location() -> Vec<(u32, String)> {
    let mut out = Vec::new();
    let dir = match std::fs::read_dir(KFD_TOPOLOGY_NODES) {
        Ok(d) => d,
        Err(e) => {
            tracing::debug!(path = KFD_TOPOLOGY_NODES, error = %e, "KFD topology unreadable");
            return out;
        }
    };
    for entry in dir.flatten() {
        let path = entry.path().join("properties");
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let mut version: Option<u32> = None;
        let mut location: Option<u32> = None;
        for line in text.lines() {
            let mut it = line.split_ascii_whitespace();
            match (it.next(), it.next()) {
                (Some("gfx_target_version"), Some(v)) => version = v.parse().ok(),
                (Some("location_id"), Some(v)) => location = v.parse().ok(),
                _ => {}
            }
        }
        if let (Some(v), Some(loc)) = (version, location) {
            if v != 0 {
                out.push((loc, gfx_name(v / 10_000, (v / 100) % 100, v % 100)));
            }
        }
    }
    out
}

/// Resolve the gfx target, preferring the kernel's answer and labelling the fallback.
fn resolve_arch(
    pci_bus_id: &str,
    cc_major: i32,
    cc_minor: i32,
    asic_revision: i32,
) -> (String, ArchSource) {
    if let Some(loc) = location_id_of(pci_bus_id) {
        if let Some((_, name)) = kfd_targets_by_location()
            .into_iter()
            .find(|(l, _)| *l == loc)
        {
            return (name, ArchSource::KfdTopology);
        }
    }
    let derived = gfx_name(
        cc_major.max(0) as u32,
        cc_minor.max(0) as u32,
        asic_revision.max(0) as u32,
    );
    tracing::warn!(
        pci_bus_id,
        derived = %derived,
        "no KFD topology node matched this device; the gfx target is derived from HIP attributes"
    );
    (derived, ArchSource::HipAttributes)
}

/// The whole machine as the probe sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Topology {
    /// The loaded runtime's soname.
    pub soname: String,
    /// `hipDriverGetVersion`, decoded.
    pub driver: HipVersion,
    /// `hipRuntimeGetVersion`, decoded.
    pub runtime: HipVersion,
    /// One entry per device, in ordinal order.
    pub devices: Vec<DeviceInfo>,
    /// `peer_access[i][j]` is `hipDeviceCanAccessPeer(i, j)`. The diagonal is `false`: HIP reports
    /// a device cannot "peer" with itself, and flattening that to `true` would hide a real answer.
    pub peer_access: Vec<Vec<bool>>,
}

impl Topology {
    /// Probe every device, its P2P reachability, and the runtime versions.
    ///
    /// `with_free_mem` controls whether a context is created on each device to read free VRAM.
    /// Attribute and P2P queries create none.
    pub fn probe(hip: &HipThread, with_free_mem: bool) -> Result<Topology> {
        let n = hip.device_count()?;
        let mut devices = Vec::with_capacity(n.max(0) as usize);
        for i in 0..n {
            let mut d = DeviceInfo::probe(hip, i)?;
            if with_free_mem {
                d.probe_free_mem(hip)?;
            }
            devices.push(d);
        }

        let mut peer_access = vec![vec![false; devices.len()]; devices.len()];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    peer_access[i as usize][j as usize] = hip.can_access_peer(i, j)?;
                }
            }
        }

        Ok(Topology {
            soname: hip.info().soname.clone(),
            driver: HipVersion::decode(hip.info().driver_version),
            runtime: HipVersion::decode(hip.info().runtime_version),
            devices,
            peer_access,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attribute_ids_are_the_ones_read_out_of_hip_7_14() {
        // Read from /opt/rocm/include/hip/hip_runtime_api.h in the dev image by compiling the
        // enum, not by counting declarations. If these drift, every number the probe prints is
        // quietly about a different attribute.
        assert_eq!(DeviceAttribute::ClockRate.id(), 5);
        assert_eq!(DeviceAttribute::L2CacheSize.id(), 19);
        assert_eq!(DeviceAttribute::ComputeCapabilityMajor.id(), 23);
        assert_eq!(DeviceAttribute::MaxThreadsPerBlock.id(), 56);
        assert_eq!(DeviceAttribute::ComputeCapabilityMinor.id(), 61);
        assert_eq!(DeviceAttribute::MultiprocessorCount.id(), 63);
        assert_eq!(DeviceAttribute::MaxSharedMemoryPerBlock.id(), 74);
        assert_eq!(DeviceAttribute::TotalGlobalMem.id(), 84);
        assert_eq!(DeviceAttribute::WarpSize.id(), 87);
        assert_eq!(DeviceAttribute::ClockInstructionRate.id(), 10000);
        assert_eq!(
            DeviceAttribute::MaxSharedMemoryPerMultiprocessor.id(),
            10002
        );
        assert_eq!(DeviceAttribute::PhysicalMultiProcessorCount.id(), 10015);
        assert_eq!(DeviceAttribute::FineGrainSupport.id(), 10016);
        assert_eq!(DeviceAttribute::PciChipId.id(), 10020);
    }

    #[test]
    fn hip_versions_decode_the_way_hip_packs_them() {
        let v = HipVersion::decode(71_460_850);
        assert_eq!((v.major, v.minor, v.patch), (7, 14, 60850));
        assert_eq!(v.raw, 71_460_850);
        assert!(v.to_string().starts_with("7.14.60850"));
    }

    #[test]
    fn gfx_names_put_the_stepping_in_hex() {
        // gfx_target_version 120001 -> (12, 0, 1). This is the value the KFD publishes for the
        // R9700s in this box, checked against hipDeviceProp_t::gcnArchName == "gfx1201".
        assert_eq!(gfx_name(12, 0, 1), "gfx1201");
        // 90010 -> (9, 0, 10): the stepping is hex, which is the whole reason gfx90a is spelled
        // that way. A decimal stepping would produce "gfx9010" and match no code object.
        assert_eq!(gfx_name(9, 0, 10), "gfx90a");
        assert_eq!(gfx_name(9, 4, 2), "gfx942");
        assert_eq!(gfx_name(11, 0, 0), "gfx1100");
    }

    #[test]
    fn pci_bus_ids_decode_to_the_kfd_location_id() {
        // Measured pairs from this box: device 0 is 0000:03:00.0 and its KFD node reports
        // location_id 768 (0x300); device 1 is 0000:07:00.0 with 1792 (0x700).
        assert_eq!(location_id_of("0000:03:00.0"), Some(768));
        assert_eq!(location_id_of("0000:07:00.0"), Some(1792));
        // The iGPU node on this box: 0x7f00.
        assert_eq!(location_id_of("0000:7f:00.0"), Some(32512));
        // Function and device digits are not ignored.
        assert_eq!(location_id_of("0000:03:01.2"), Some(768 + 8 + 2));
        // Malformed input is None, never a plausible-looking 0.
        assert_eq!(location_id_of(""), None);
        assert_eq!(location_id_of("0000:03:00"), None);
        assert_eq!(location_id_of("zz:zz:zz.z"), None);
        assert_eq!(location_id_of("0:0000:03:00.0"), None);
    }

    #[test]
    fn the_arch_fallback_is_labelled_not_silent() {
        // No KFD node can match an address that is not a device, so this exercises the fallback
        // route and must come back marked as derived.
        let (arch, source) = resolve_arch("ffff:ff:1f.7", 12, 0, 1);
        assert_eq!(arch, "gfx1201");
        assert_eq!(source, ArchSource::HipAttributes);
        assert!(source.to_string().contains("derived"));
    }
}
