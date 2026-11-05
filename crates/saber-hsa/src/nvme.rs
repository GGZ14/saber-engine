//! FN70 — drive bytes into VRAM: `nvme_read_to_vram(file, offset, len, dst, done)`.
//!
//! Three ROUTES, picked ONCE at [`NvmeVram::open`] by probe reads and counted per call (rule 5),
//! in this order of preference ([`choose_route`]):
//!
//! * [`NvmeRoute::Pool`] (FN71b) — peer DMA into the card's amdgpu P2PDMA POOL (kernel patch
//!   FN71 0001, `amdgpu.p2pmem_mb = N`: N MiB of VRAM pinned at boot and registered with
//!   `pci_p2pdma_add_resource`). Two views of one range: the drive side is the `mmap` of
//!   `/sys/bus/pci/devices/<bdf>/p2pmem/allocate` (P2PDMA `struct page`s an `O_DIRECT` `pread`
//!   can pin), the GPU side is `GEM_CREATE(AMDGPU_GEM_CREATE_P2PMEM_LOCAL)` on the card's render
//!   node -> PRIME fd -> `hsa_amd_interop_map_buffer`. The two addresses DIFFER (offset x on one
//!   is offset x on the other: FN71b [B]), so a destination is named by its GPU address and
//!   [`PoolMap::host_of`] translates it for the `pread`. The pool is mapped WHOLE, so one reader
//!   a card owns it (`available` must equal `size` at open). FN71b: 9.5-10.7 GB/s, byte-equal,
//!   and host DRAM untouched (a CPU stream keeps 51.7 of 52.0 GB/s beside it; 32.4 beside the
//!   bounce).
//! * [`NvmeRoute::P2p`] — peer DMA. The destination is host-visible VRAM (a PUBLIC allocation
//!   inside the large BAR, [`crate::Hsa::alloc_vram_host_visible`]: one address for the GPU and
//!   the host), and an `O_DIRECT` `pread` names it: the NVMe controller writes the GPU's BAR
//!   directly, no host DRAM touched. This needs the kernel to pin BAR pages for the block layer
//!   (`pin_user_pages(FOLL_PCI_P2PDMA)` -> P2PDMA `struct page`s), which needs the GPU driver to
//!   REGISTER its BAR as a P2PDMA provider (`pci_p2pdma_add_resource`). amdgpu (Linux 7.1.3) does
//!   not: the BAR mapping is `VM_PFNMAP`, so the pin fails and the read returns `EFAULT`
//!   (LEDGER FN70). The probe records that errno as [`NvmeVram::p2p_refusal`] and the reader runs
//!   the bounce; a kernel that registers the BAR turns the P2P route on with no code change.
//!   After the read the route reads one word back through the BAR (a non-posted read behind the
//!   controller's posted writes on the root complex -> GPU path) before it publishes completion.
//! * [`NvmeRoute::Bounce`] — `O_DIRECT` `pread` into a pinned staging slot (ROCr system memory,
//!   `slots` x `slot_bytes`), then an SDMA H2D on [`SdmaLane`] (FN63b's lane: ticket signals,
//!   no HIP stream). Host DRAM carries every byte twice (the drive's write, the engine's read).
//!
//! **Completion visible to HIP** (both routes, FN63b's pattern): `done = Some((word, value))`
//! writes the 32-bit `value` to the device word `word` (a registered device region) on the lane
//! AFTER the bytes landed -- a HIP stream waits with `hipStreamWaitValue32(word, value, GEQ)`;
//! the host waits with [`NvmeVram::wait`] on the returned ticket.
//!
//! **The P2P fence rule** (FN71b [E1]/[E2], both pool views, card 1): the drive's writes reach
//! VRAM behind the GPU's caches. A kernel LAUNCHED after the completion (a stream that waited
//! on the word, then dispatched) reads the new bytes even with the old lines warm in GL2 (6/6);
//! a kernel ALREADY RUNNING that saw the word with a plain volatile spin reads the OLD bytes
//! (~100 % stale, 16 KiB and 2 MiB); a system-scope acquire after the word
//! (`__hip_atomic_load(.., __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM)` or
//! `__threadfence_system()`) in every reading thread makes them EQUAL. So: consume after a
//! launch boundary, or acquire at system scope inside a resident kernel.
//!
//! **Bit-exact by construction**: bytes are bytes (no conversion on either route); the card
//! example (`examples/fn70_nvme.rs`) reads them back and compares with a host `pread`.
//!
//! Alignment: `offset`, `len` and every destination address are multiples of 4096 (the
//! `O_DIRECT` law, `saber_mem::dio`); the file must be opened with `O_DIRECT`.

use std::fs::{File, OpenOptions};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::lane::SdmaLane;
use crate::{Agent, EngineChoice, Error, Hsa, Result};

/// The `O_DIRECT` alignment every offset, length and address keeps.
pub const NVME_ALIGN: u64 = 4096;

/// Which route the reader runs (picked at open, never per call).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvmeRoute {
    /// Drive -> the card's P2PDMA pool (FN71b; the destination must lie in [`NvmeVram::pool`]).
    Pool,
    /// Drive -> VRAM peer DMA (the destination must lie in [`NvmeVram::window`]).
    P2p,
    /// Drive -> pinned staging -> SDMA H2D.
    Bounce,
}

/// The which-path counters (rule 5).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NvmeCounts {
    /// Calls served by the P2PDMA pool route.
    pub pool_reads: u64,
    /// Bytes they moved.
    pub pool_bytes: u64,
    /// Calls served by peer DMA into the host-visible window.
    pub p2p_reads: u64,
    /// Bytes they moved.
    pub p2p_bytes: u64,
    /// Calls served by the bounce.
    pub bounce_reads: u64,
    /// Staging pieces (one `pread` + one SDMA copy each).
    pub bounce_pieces: u64,
    /// Bytes they moved.
    pub bounce_bytes: u64,
    /// Pieces that had to wait on the host for their staging slot's previous copy.
    pub stage_waits: u64,
    /// FN71c: pool -> VRAM device copies issued on the lane ([`NvmeVram::copy_from_pool`]).
    pub pool_copies: u64,
    /// Bytes they moved.
    pub pool_copy_bytes: u64,
}

/// Whether (and where) [`NvmeVram::open_with`] looks for the card's P2PDMA pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolAsk {
    /// Never (the window / bounce routes only).
    Off,
    /// `/sys/bus/pci/devices/<bdf of bdfid>/p2pmem/` (domain 0), its `allocate` file in place.
    Auto,
    /// An explicit sysfs dir (size / available / published) and `allocate` path (a container
    /// bind-mounts the `allocate` file rw elsewhere: sysfs is read-only there).
    At {
        /// The `p2pmem` directory.
        sysfs_dir: PathBuf,
        /// The `allocate` file to mmap.
        allocate: PathBuf,
    },
}

/// What [`NvmeVram::open_with`] builds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvmeOptions {
    /// Pinned staging slots for the bounce.
    pub slots: usize,
    /// Bytes a slot (a multiple of [`NVME_ALIGN`]).
    pub slot_bytes: usize,
    /// A host-visible VRAM window for FN70's BAR route (0 = none); allocated only when the pool
    /// route is refused.
    pub window_bytes: usize,
    /// The pool route.
    pub pool: PoolAsk,
}

/// The pool's two views: the GPU address (`hsa_amd_interop_map_buffer`) and the host mapping
/// of the `allocate` file, same length; offset x of one is offset x of the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolMap {
    /// Device address of pool offset 0.
    pub gpu: u64,
    /// Host address of pool offset 0 (the drive's `pread` destination).
    pub host: u64,
    /// Pool bytes.
    pub bytes: u64,
}

impl PoolMap {
    /// The host address of the GPU range `dst .. dst + len`, or `None` when any byte lies
    /// outside the pool.
    #[must_use]
    pub fn host_of(&self, dst: u64, len: u64) -> Option<u64> {
        let end = dst.checked_add(len)?;
        if len == 0 || dst < self.gpu || end > self.gpu.checked_add(self.bytes)? {
            return None;
        }
        Some(self.host + (dst - self.gpu))
    }
}

/// FN71c: check and translate a batch of pool reads -- each `(file offset, GPU destination)` of
/// `len` bytes -> `(file offset, HOST address)`; every offset, length and destination keeps the
/// `O_DIRECT` alignment and every destination lies inside the pool, or the whole batch is refused
/// (nothing is read).
pub fn plan_pool_reads(m: &PoolMap, items: &[(u64, u64)], len: u64) -> Result<Vec<(u64, u64)>> {
    if len == 0 || !aligned(len) {
        return Err(Error::Invalid(format!(
            "pool read length {len}: a non-zero multiple of {NVME_ALIGN}"
        )));
    }
    items
        .iter()
        .map(|&(off, dst)| {
            if !aligned(off) || !aligned(dst) {
                return Err(Error::Invalid(format!(
                    "pool read at {off} -> {dst:#x}: both multiples of {NVME_ALIGN}"
                )));
            }
            m.host_of(dst, len).map(|h| (off, h)).ok_or_else(|| {
                Error::Invalid(format!(
                    "pool destination {dst:#x} + {len} lies outside the pool {:#x} + {}",
                    m.gpu, m.bytes
                ))
            })
        })
        .collect()
}

/// The sysfs `p2pmem` directory of PCI `bdfid` (`(bus << 8) | (dev << 3) | func`, domain 0).
#[must_use]
pub fn pool_sysfs_dir(bdfid: u32) -> PathBuf {
    PathBuf::from(format!("/sys/bus/pci/devices/{}/p2pmem", bdf_string(bdfid)))
}

/// `0000:BB:DD.F` of a ROCr `bdfid`.
#[must_use]
pub fn bdf_string(bdfid: u32) -> String {
    format!(
        "0000:{:02x}:{:02x}.{:x}",
        (bdfid >> 8) & 0xff,
        (bdfid >> 3) & 0x1f,
        bdfid & 0x7
    )
}

/// The pool's bytes from its sysfs `size` / `available` / `published`, or why it cannot be
/// opened (absent, unpublished, empty, or partly handed out to another mapping).
pub fn parse_pool_sysfs(
    size: &str,
    available: &str,
    published: &str,
) -> std::result::Result<u64, String> {
    let num = |s: &str, what: &str| {
        s.trim()
            .parse::<u64>()
            .map_err(|_| format!("p2pmem/{what} '{}' is not a number", s.trim()))
    };
    let size = num(size, "size")?;
    let avail = num(available, "available")?;
    if published.trim() != "1" {
        return Err(format!(
            "p2pmem/published is '{}' (want 1)",
            published.trim()
        ));
    }
    if size == 0 || size % NVME_ALIGN != 0 {
        return Err(format!("p2pmem/size {size}: empty or not 4 KiB aligned"));
    }
    if avail != size {
        return Err(format!(
            "p2pmem: {avail} of {size} B available (another reader maps the pool)"
        ));
    }
    Ok(size)
}

/// The route from the two probes (pool first, then FN70's host-visible window, else the
/// bounce) and the refusal text naming why each faster route is off (None = the pool serves).
#[must_use]
pub fn choose_route(
    pool: &std::result::Result<(), String>,
    window: &std::result::Result<(), String>,
) -> (NvmeRoute, Option<String>) {
    match (pool, window) {
        (Ok(()), _) => (NvmeRoute::Pool, None),
        (Err(p), Ok(())) => (NvmeRoute::P2p, Some(format!("pool: {p}"))),
        (Err(p), Err(w)) => (NvmeRoute::Bounce, Some(format!("pool: {p}; window: {w}"))),
    }
}

// ---- the pool's two ioctls (layouts from include/uapi/drm; no libdrm on the host)
#[repr(C)]
#[derive(Clone, Copy)]
struct GemCreateIn {
    bo_size: u64,
    alignment: u64,
    domains: u64,
    domain_flags: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct PrimeHandle {
    handle: u32,
    flags: u32,
    fd: i32,
}
/// `DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_CREATE, union drm_amdgpu_gem_create)` (32 B).
const IOCTL_GEM_CREATE: u64 = (3 << 30) | (32 << 16) | (0x64 << 8) | 0x40;
/// `DRM_IOWR(0x2d, struct drm_prime_handle)` (12 B).
const IOCTL_PRIME_HANDLE_TO_FD: u64 = (3 << 30) | (12 << 16) | (0x64 << 8) | 0x2d;
/// `AMDGPU_GEM_DOMAIN_VRAM`.
const DOMAIN_VRAM: u64 = 0x4;
/// `AMDGPU_GEM_CREATE_P2PMEM_LOCAL` (patch FN71 0001).
const GEM_CREATE_P2PMEM_LOCAL: u64 = 1 << 48;

/// The card's render node (`/dev/dri/renderD*` whose device is `bdf`).
fn render_node_for(bdf: &str) -> Option<PathBuf> {
    let rd = std::fs::read_dir("/sys/class/drm").ok()?;
    for e in rd.flatten() {
        let name = e.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("renderD") {
            continue;
        }
        let dev = std::fs::canonicalize(e.path().join("device")).ok()?;
        if dev.file_name().is_some_and(|f| f.to_string_lossy() == bdf) {
            return Some(Path::new("/dev/dri").join(&*name));
        }
    }
    None
}

fn read_trim(p: &Path) -> std::result::Result<String, String> {
    std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))
}

/// The opened pool: both views + the fds that keep them alive. Unmapped by [`NvmeVram`]'s drop
/// (the GPU view needs the runtime).
struct P2pPool {
    map: PoolMap,
    _render: OwnedFd,
    _file: OwnedFd,
}

/// FN76: the card's whole P2PDMA pool as ONE amdgpu BO exported as a dma-buf (GEM_CREATE with
/// the FN71 `P2PMEM_LOCAL` flag on the card's render node, then PRIME_HANDLE_TO_FD). The pool is
/// mapped whole, so one holder a card at a time: [`NvmeVram`] drops its own before this succeeds.
pub struct PoolBo {
    /// The dma-buf (hand a dup to the importer).
    pub dmabuf: OwnedFd,
    /// The pool's bytes.
    pub bytes: u64,
    _render: OwnedFd,
}

/// FN76: open the card's pool as a [`PoolBo`] (see there); `Err` names why not.
pub fn pool_bo(bdfid: u32, ask: &PoolAsk) -> std::result::Result<PoolBo, String> {
    let (render, dmabuf, bytes, _) = open_pool_bo(bdfid, ask)?;
    Ok(PoolBo {
        dmabuf,
        bytes,
        _render: render,
    })
}

fn open_pool_bo(
    bdfid: u32,
    ask: &PoolAsk,
) -> std::result::Result<(OwnedFd, OwnedFd, u64, PathBuf), String> {
    let (dir, allocate) = match ask {
        PoolAsk::Off => return Err("not asked (PoolAsk::Off)".into()),
        PoolAsk::Auto => {
            let d = pool_sysfs_dir(bdfid);
            let a = d.join("allocate");
            (d, a)
        }
        PoolAsk::At {
            sysfs_dir,
            allocate,
        } => (sysfs_dir.clone(), allocate.clone()),
    };
    if !dir.exists() {
        return Err(format!(
            "{} absent (no amdgpu.p2pmem_mb on this boot / not the FN71 kernel)",
            dir.display()
        ));
    }
    let bytes = parse_pool_sysfs(
        &read_trim(&dir.join("size"))?,
        &read_trim(&dir.join("available"))?,
        &read_trim(&dir.join("published"))?,
    )?;
    let bdf = bdf_string(bdfid);
    let node = render_node_for(&bdf).ok_or_else(|| format!("no render node for {bdf}"))?;
    let render: OwnedFd = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&node)
        .map_err(|e| format!("{}: {e}", node.display()))?
        .into();
    let mut gc = GemCreateIn {
        bo_size: 0,
        alignment: 0,
        domains: DOMAIN_VRAM,
        domain_flags: GEM_CREATE_P2PMEM_LOCAL,
    };
    // SAFETY: the amdgpu GEM_CREATE ioctl on an open render node with a 32-byte in/out
    // union; on success the kernel writes the handle into the first 4 bytes.
    let r = unsafe {
        libc::ioctl(
            render.as_raw_fd(),
            IOCTL_GEM_CREATE as _,
            (&raw mut gc).cast::<libc::c_void>(),
        )
    };
    if r != 0 {
        return Err(format!(
            "GEM_CREATE(P2PMEM_LOCAL) on {}: {} (the FN71 module patch?)",
            node.display(),
            std::io::Error::last_os_error()
        ));
    }
    let mut ph = PrimeHandle {
        handle: gc.bo_size as u32,
        flags: (libc::O_CLOEXEC | libc::O_RDWR) as u32,
        fd: -1,
    };
    // SAFETY: PRIME_HANDLE_TO_FD with a 12-byte in/out struct on the same render node.
    let r = unsafe {
        libc::ioctl(
            render.as_raw_fd(),
            IOCTL_PRIME_HANDLE_TO_FD as _,
            (&raw mut ph).cast::<libc::c_void>(),
        )
    };
    if r != 0 || ph.fd < 0 {
        return Err(format!(
            "PRIME_HANDLE_TO_FD: {}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: a fresh fd the kernel just returned; owned here.
    let dmabuf = unsafe { OwnedFd::from_raw_fd(ph.fd) };
    Ok((render, dmabuf, bytes, allocate))
}

impl P2pPool {
    fn open(
        hsa: &Hsa,
        gpu: Agent,
        bdfid: u32,
        ask: &PoolAsk,
    ) -> std::result::Result<P2pPool, String> {
        let (render, dmabuf, bytes, allocate) = open_pool_bo(bdfid, ask)?;
        let (gptr, gsz) = hsa
            .interop_map_buffer(gpu, dmabuf.as_raw_fd())
            .map_err(|e| format!("interop map of the pool's dma-buf: {e}"))?;
        drop(dmabuf);
        let unmap_gpu = || {
            // SAFETY: just mapped, nothing in flight.
            let _ = unsafe { hsa.interop_unmap_buffer(gptr) };
        };
        if gsz as u64 != bytes {
            unmap_gpu();
            return Err(format!("interop map is {gsz} B, the pool {bytes} B"));
        }
        let file: OwnedFd = match OpenOptions::new().read(true).write(true).open(&allocate) {
            Ok(f) => f.into(),
            Err(e) => {
                unmap_gpu();
                return Err(format!("{}: {e}", allocate.display()));
            }
        };
        // SAFETY: a fresh shared mapping of the whole pool file (no fixed address).
        let host = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if host == libc::MAP_FAILED {
            let e = std::io::Error::last_os_error();
            unmap_gpu();
            return Err(format!("mmap {} ({bytes} B): {e}", allocate.display()));
        }
        Ok(P2pPool {
            map: PoolMap {
                gpu: gptr as u64,
                host: host as u64,
                bytes,
            },
            _render: render,
            _file: file,
        })
    }

    fn close(&self, hsa: &Hsa) {
        // SAFETY: both views came from `open`; the reader drained every copy first.
        unsafe {
            libc::munmap(self.map.host as *mut libc::c_void, self.map.bytes as usize);
            let _ = hsa.interop_unmap_buffer(self.map.gpu as *mut u8);
        }
    }
}

/// The reader (see the module docs). Drops in order: drain the lane, free the staging and the
/// window, then the lane.
pub struct NvmeVram {
    lane: SdmaLane,
    route: NvmeRoute,
    refusal: Option<String>,
    pool: Option<P2pPool>,
    window: Option<(u64, u64)>,
    stage: u64,
    slot_bytes: u64,
    slot_ticket: Vec<u64>,
    next_slot: usize,
    /// The counters.
    pub counts: NvmeCounts,
}

impl std::fmt::Debug for NvmeVram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NvmeVram({:?}, window {:?}, {} x {} B staging)",
            self.route,
            self.window,
            self.slot_ticket.len(),
            self.slot_bytes
        )
    }
}

fn aligned(x: u64) -> bool {
    x % NVME_ALIGN == 0
}

/// One `pread` loop into `dst` (a host-addressable range: pinned staging or the BAR window).
///
/// # Safety
/// `dst .. dst + len` is mapped, writable and owned by the caller for the call.
unsafe fn pread_into(file: &File, offset: u64, dst: u64, len: u64) -> std::io::Result<()> {
    let n = usize::try_from(len).map_err(|_| std::io::Error::other("length"))?;
    // SAFETY: the caller's contract.
    let buf = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, n) };
    file.read_exact_at(buf, offset)
}

impl NvmeVram {
    /// Open ROCr's lane on the GPU at PCI `bdfid`, allocate `slots` x `slot_bytes` of pinned
    /// staging and -- when `window_bytes > 0` -- a host-visible VRAM window, then PROBE the P2P
    /// route with one `NVME_ALIGN` read of `probe` at `probe_offset` into the window (compared with
    /// the same bytes read into the staging). `window_bytes == 0` = the bounce only. FN71b: the
    /// card's P2PDMA pool is tried FIRST (`PoolAsk::Auto`); [`Self::open_with`] names it.
    pub fn open(
        soname: &str,
        bdfid: u32,
        slots: usize,
        slot_bytes: usize,
        window_bytes: usize,
        probe: (&File, u64),
    ) -> Result<NvmeVram> {
        let opts = NvmeOptions {
            slots,
            slot_bytes,
            window_bytes,
            pool: PoolAsk::Auto,
        };
        Self::open_with(soname, bdfid, &opts, probe)
    }

    /// [`Self::open`] with every choice explicit: the pool route is tried first (`opts.pool`),
    /// FN70's host-visible window only when the pool is refused, the bounce always exists.
    pub fn open_with(
        soname: &str,
        bdfid: u32,
        opts: &NvmeOptions,
        probe: (&File, u64),
    ) -> Result<NvmeVram> {
        let (slots, slot_bytes, window_bytes) = (opts.slots, opts.slot_bytes, opts.window_bytes);
        let slot = slot_bytes as u64;
        if slots == 0 || slot == 0 || !aligned(slot) {
            return Err(Error::Invalid(format!(
                "staging {slots} x {slot_bytes} B: at least one slot, a multiple of {NVME_ALIGN}"
            )));
        }
        if !aligned(probe.1) || !aligned(window_bytes as u64) {
            return Err(Error::Invalid(format!(
                "probe offset {} / window {window_bytes}: multiples of {NVME_ALIGN}",
                probe.1
            )));
        }
        let mut lane = SdmaLane::open(soname, bdfid, 256, EngineChoice::Auto)?;
        let (stage, pool) = {
            let (hsa, cpu, gpu) = lane.runtime();
            let stage = hsa.alloc_fine_sys(cpu, gpu, slots * slot_bytes)? as u64;
            (stage, P2pPool::open(hsa, gpu, bdfid, &opts.pool))
        };
        lane.register_raw(stage, slot * slots as u64, true);
        let mut me = NvmeVram {
            lane,
            route: NvmeRoute::Bounce,
            refusal: None,
            pool: None,
            window: None,
            stage,
            slot_bytes: slot,
            slot_ticket: vec![0; slots],
            next_slot: 0,
            counts: NvmeCounts::default(),
        };
        let pool_probe = match pool {
            Err(why) => Err(why),
            Ok(p) => {
                let m = p.map;
                me.lane.register_raw(m.gpu, m.bytes, false);
                me.pool = Some(p);
                me.probe_pool(probe.0, probe.1, m)
            }
        };
        let window_probe = if pool_probe.is_ok() {
            Err("not needed (the pool serves)".to_string())
        } else if window_bytes == 0 {
            Err("no window asked (window_bytes 0)".to_string())
        } else {
            let alloc = {
                let (hsa, cpu, gpu) = me.lane.runtime();
                hsa.alloc_vram_host_visible(cpu, gpu, window_bytes)
            };
            match alloc {
                Err(e) => Err(format!("host-visible VRAM window: {e}")),
                Ok(p) => {
                    let w = p as u64;
                    me.window = Some((w, window_bytes as u64));
                    me.lane.register_raw(w, window_bytes as u64, false);
                    me.probe_p2p(probe.0, probe.1, w)
                }
            }
        };
        let (route, refusal) = choose_route(&pool_probe, &window_probe);
        me.route = route;
        me.refusal = refusal;
        Ok(me)
    }

    /// The pool probe: one aligned read through the HOST view, then the lane copies the same
    /// bytes back out through the GPU view (proves the drive reaches the pool AND the two views
    /// alias at the translated offset), compared with the staging's own read.
    fn probe_pool(
        &mut self,
        file: &File,
        offset: u64,
        m: PoolMap,
    ) -> std::result::Result<(), String> {
        let n = NVME_ALIGN;
        let mut truth = vec![0u8; n as usize];
        // SAFETY: the staging holds at least one aligned slot; nothing is in flight yet.
        unsafe { pread_into(file, offset, self.stage, n) }
            .map_err(|e| format!("probe read into the staging: {e}"))?;
        // SAFETY: the staging slot just read, NVME_ALIGN bytes.
        truth.copy_from_slice(unsafe {
            std::slice::from_raw_parts(self.stage as *const u8, n as usize)
        });
        // poison the staging so the copy-back must land
        // SAFETY: as above.
        unsafe { std::ptr::write_bytes(self.stage as *mut u8, 0xa5, n as usize) };
        let host = m
            .host_of(m.gpu, n)
            .ok_or_else(|| "pool smaller than one probe".to_string())?;
        // SAFETY: inside the pool's host view (checked above), owned by this reader.
        unsafe { pread_into(file, offset, host, n) }.map_err(|e| {
            format!(
                "O_DIRECT pread into the P2PDMA pool refused: {e} (errno {:?})",
                e.raw_os_error()
            )
        })?;
        let t = self.lane.next_ticket();
        self.lane
            .copy(t, self.stage, m.gpu, n, &[])
            .map_err(|e| format!("probe copy-back from the pool's GPU view: {e}"))?;
        self.lane
            .wait(t)
            .map_err(|e| format!("probe copy-back wait: {e}"))?;
        // SAFETY: the staging slot the copy just landed in.
        let back = unsafe { std::slice::from_raw_parts(self.stage as *const u8, n as usize) };
        if back == truth.as_slice() {
            Ok(())
        } else {
            Err("pool probe: the GPU view holds DIFFERENT bytes than the drive wrote".into())
        }
    }

    /// The P2P probe: one aligned read into the window, compared with the staging's copy.
    fn probe_p2p(&self, file: &File, offset: u64, w: u64) -> std::result::Result<(), String> {
        let n = NVME_ALIGN;
        // SAFETY: the staging holds at least one aligned slot; the window at least NVME_ALIGN
        // bytes (both allocated by `open`, nothing in flight yet).
        unsafe { pread_into(file, offset, self.stage, n) }
            .map_err(|e| format!("probe read into the staging: {e}"))?;
        // SAFETY: as above; the window is host-visible by its allocation, and a kernel that cannot
        // pin it returns EFAULT here (no fault is taken: the kernel touches it, not this thread).
        unsafe { pread_into(file, offset, w, n) }.map_err(|e| {
            format!(
                "O_DIRECT pread into host-visible VRAM refused: {e} (errno {:?}; the GPU's BAR is \
                 not a P2PDMA provider in this kernel)",
                e.raw_os_error()
            )
        })?;
        // SAFETY: both ranges are NVME_ALIGN bytes this reader owns; the window read goes over
        // the BAR (uncached), which is the fence the route uses anyway.
        let (a, b) = unsafe {
            (
                std::slice::from_raw_parts(self.stage as *const u8, n as usize),
                std::slice::from_raw_parts(w as *const u8, n as usize),
            )
        };
        if a == b {
            Ok(())
        } else {
            Err("P2P probe read landed DIFFERENT bytes than the staging read".into())
        }
    }

    /// The route picked at open.
    #[must_use]
    pub fn route(&self) -> NvmeRoute {
        self.route
    }

    /// Why the P2P route is off (None = it is on).
    #[must_use]
    pub fn p2p_refusal(&self) -> Option<&str> {
        self.refusal.as_deref()
    }

    /// FN71b: the P2PDMA pool's two views (Pool-route destinations are GPU addresses inside
    /// `gpu .. gpu + bytes`; also a device region the lane may copy into and write words to).
    #[must_use]
    pub fn pool(&self) -> Option<PoolMap> {
        self.pool.as_ref().map(|p| p.map)
    }

    /// FN70's host-visible VRAM window `(base, bytes)` (opened only when the pool is refused).
    #[must_use]
    pub fn window(&self) -> Option<(u64, u64)> {
        self.window
    }

    /// Force the bounce (the control arm of a P2P consumer); counted like any bounce.
    pub fn force_bounce(&mut self) {
        if self.route != NvmeRoute::Bounce {
            self.refusal = Some("forced to the bounce (control)".into());
        }
        self.route = NvmeRoute::Bounce;
    }

    /// Register a device allocation (a destination or a completion word's region) the caller
    /// owns and keeps alive while this reader lives.
    pub fn register_device(&mut self, base: u64, bytes: u64) {
        self.lane.register_raw(base, bytes, false);
    }

    /// Read `len` bytes of `file` at `offset` into device address `dst`; when `done` is
    /// `Some((word, value))`, write `value` to the device word `word` after every byte landed.
    /// Returns the last lane ticket (0 when nothing was queued: a P2P read without `done`, which
    /// is complete on return). The bounce returns while the last copies may still be in flight:
    /// [`Self::wait`] the ticket (or wait on the word on a stream) before reading `dst`.
    pub fn nvme_read_to_vram(
        &mut self,
        file: &File,
        offset: u64,
        len: u64,
        dst: u64,
        done: Option<(u64, u32)>,
    ) -> Result<u64> {
        if len == 0 || !aligned(offset) || !aligned(len) || !aligned(dst) {
            return Err(Error::Invalid(format!(
                "nvme read {len} B at {offset} -> {dst:#x}: all three multiples of {NVME_ALIGN}"
            )));
        }
        match self.route {
            NvmeRoute::Pool => self.read_pool(file, offset, len, dst, done),
            NvmeRoute::P2p => self.read_p2p(file, offset, len, dst, done),
            NvmeRoute::Bounce => self.read_bounce(file, offset, len, dst, done),
        }
    }

    /// The pool route: `dst` is a GPU address in the pool; the drive writes the HOST view at the
    /// same offset. After the `pread` returns, one non-posted read of the last word through the
    /// host view (behind the controller's posted writes), then the completion word on the lane.
    /// Consumers follow the module's fence rule (launch boundary, or a system-scope acquire).
    fn read_pool(
        &mut self,
        file: &File,
        offset: u64,
        len: u64,
        dst: u64,
        done: Option<(u64, u32)>,
    ) -> Result<u64> {
        let m = self
            .pool
            .as_ref()
            .map(|p| p.map)
            .ok_or_else(|| Error::Invalid("pool route without a pool".into()))?;
        let host = m.host_of(dst, len).ok_or_else(|| {
            Error::Invalid(format!(
                "pool destination {dst:#x} + {len} lies outside the pool {:#x} + {}",
                m.gpu, m.bytes
            ))
        })?;
        // SAFETY: inside the pool's host view (translated + bounds-checked above); the caller
        // owns the destination range for the call.
        unsafe { pread_into(file, offset, host, len) }
            .map_err(|e| Error::Invalid(format!("pool read {len} B at {offset}: {e}")))?;
        // SAFETY: the last word of the range just read, inside the host view.
        let _ = unsafe { ((host + len - 4) as *const u32).read_volatile() };
        self.counts.pool_reads += 1;
        self.counts.pool_bytes += len;
        match done {
            None => Ok(0),
            Some((word, value)) => {
                let t = self.lane.next_ticket();
                self.lane.write_word(t, word, value, &[])?;
                Ok(t)
            }
        }
    }

    fn read_p2p(
        &mut self,
        file: &File,
        offset: u64,
        len: u64,
        dst: u64,
        done: Option<(u64, u32)>,
    ) -> Result<u64> {
        let (w, wl) = self
            .window
            .ok_or_else(|| Error::Invalid("P2P route without a window".into()))?;
        if dst < w || dst.checked_add(len).is_none_or(|e| e > w + wl) {
            return Err(Error::Invalid(format!(
                "P2P destination {dst:#x} + {len} lies outside the host-visible window \
                 {w:#x} + {wl}"
            )));
        }
        // SAFETY: inside the window this reader allocated (checked above); the caller owns the
        // destination range for the call (no copy in flight writes it).
        unsafe { pread_into(file, offset, dst, len) }
            .map_err(|e| Error::Invalid(format!("P2P read {len} B at {offset}: {e}")))?;
        // the fence: one non-posted read of the last word through the BAR
        // SAFETY: the word lies in the range just read, inside the window.
        let _ = unsafe { ((dst + len - 4) as *const u32).read_volatile() };
        self.counts.p2p_reads += 1;
        self.counts.p2p_bytes += len;
        match done {
            None => Ok(0),
            Some((word, value)) => {
                let t = self.lane.next_ticket();
                self.lane.write_word(t, word, value, &[])?;
                Ok(t)
            }
        }
    }

    fn read_bounce(
        &mut self,
        file: &File,
        offset: u64,
        len: u64,
        dst: u64,
        done: Option<(u64, u32)>,
    ) -> Result<u64> {
        let mut tickets = Vec::with_capacity(len.div_ceil(self.slot_bytes) as usize);
        let mut o = 0u64;
        while o < len {
            let n = self.slot_bytes.min(len - o);
            let s = self.next_slot;
            self.next_slot = (s + 1) % self.slot_ticket.len();
            let prev = self.slot_ticket[s];
            if !self.lane.done(prev) {
                self.counts.stage_waits += 1;
            }
            self.lane.wait(prev)?;
            let src = self.stage + s as u64 * self.slot_bytes;
            // SAFETY: slot `s` of the staging this reader owns; its previous copy landed (waited
            // above), so no engine reads it while the host writes it.
            unsafe { pread_into(file, offset + o, src, n) }
                .map_err(|e| Error::Invalid(format!("bounce read {n} B at {}: {e}", offset + o)))?;
            let t = self.lane.next_ticket();
            self.lane.copy(t, dst + o, src, n, &[])?;
            self.slot_ticket[s] = t;
            tickets.push(t);
            self.counts.bounce_pieces += 1;
            o += n;
        }
        self.counts.bounce_reads += 1;
        self.counts.bounce_bytes += len;
        let last = *tickets.last().unwrap_or(&0);
        match done {
            None => Ok(last),
            Some((word, value)) => {
                let t = self.lane.next_ticket();
                self.lane.write_word(t, word, value, &tickets)?;
                Ok(t)
            }
        }
    }

    /// FN71c: read a BATCH of `len`-byte pieces into the pool -- `items` = `(file offset, GPU
    /// destination in the pool)` -- with up to `qd` threads, each one `O_DIRECT` `pread` into the
    /// pool's host view at the translated offset, then one non-posted read-back of every piece's
    /// last word (the FN71b fence's host half). Complete on return: a kernel launched after it,
    /// or a lane copy issued after it, reads the new bytes. Pool route only (the caller checks
    /// [`Self::route`]); the destinations must not be read by any engine while this runs.
    pub fn read_pool_many(
        &mut self,
        file: &File,
        items: &[(u64, u64)],
        len: u64,
        qd: usize,
    ) -> Result<()> {
        if self.route != NvmeRoute::Pool {
            return Err(Error::Invalid(format!(
                "read_pool_many on the {:?} route",
                self.route
            )));
        }
        let m = self
            .pool
            .as_ref()
            .map(|p| p.map)
            .ok_or_else(|| Error::Invalid("pool route without a pool".into()))?;
        let plan = plan_pool_reads(&m, items, len)?;
        if plan.is_empty() {
            return Ok(());
        }
        let workers = qd.clamp(1, plan.len());
        let next = std::sync::atomic::AtomicUsize::new(0);
        let err: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
        std::thread::scope(|sc| {
            for _ in 0..workers {
                sc.spawn(|| loop {
                    let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(off, host)) = plan.get(k) else {
                        break;
                    };
                    // SAFETY: `host .. host + len` lies inside the pool's host view (translated
                    // and bounds-checked by `plan_pool_reads`); pieces are distinct destinations
                    // the caller owns for the call.
                    if let Err(e) = unsafe { pread_into(file, off, host, len) } {
                        if let Ok(mut g) = err.lock() {
                            g.get_or_insert_with(|| format!("pool read {len} B at {off}: {e}"));
                        }
                        break;
                    }
                });
            }
        });
        if let Some(e) = err.into_inner().ok().flatten() {
            return Err(Error::Invalid(e));
        }
        for &(_, host) in &plan {
            // SAFETY: the last word of a range just read, inside the host view.
            let _ = unsafe { ((host + len - 4) as *const u32).read_volatile() };
        }
        self.counts.pool_reads += plan.len() as u64;
        self.counts.pool_bytes += plan.len() as u64 * len;
        Ok(())
    }

    /// FN71c: copy `len` bytes from the pool's GPU view (`src`) into a registered device range
    /// (`dst`, [`Self::register_device`]) on the lane's SDMA engine after every ticket in `deps`;
    /// returns the ticket ([`Self::wait`] / [`Self::done`]). Issue it after the bytes landed in
    /// the pool ([`Self::read_pool_many`] returned).
    pub fn copy_from_pool(&mut self, dst: u64, src: u64, len: u64, deps: &[u64]) -> Result<u64> {
        let m = self
            .pool
            .as_ref()
            .map(|p| p.map)
            .ok_or_else(|| Error::Invalid("copy_from_pool without a pool".into()))?;
        if m.host_of(src, len).is_none() {
            return Err(Error::Invalid(format!(
                "pool source {src:#x} + {len} lies outside the pool"
            )));
        }
        let t = self.lane.next_ticket();
        self.lane.copy_d2d(t, dst, src, len, deps)?;
        self.counts.pool_copies += 1;
        self.counts.pool_copy_bytes += len;
        Ok(t)
    }

    /// Whether ticket `t` has landed.
    #[must_use]
    pub fn done(&self, t: u64) -> bool {
        self.lane.done(t)
    }

    /// The host waits until ticket `t` (a return of [`Self::nvme_read_to_vram`]) has landed.
    pub fn wait(&mut self, t: u64) -> Result<()> {
        self.lane.wait(t)
    }

    /// The lane's own counters (copies, words, host waits).
    #[must_use]
    pub fn lane_counts(&self) -> crate::LaneCounts {
        self.lane.counts
    }
}

impl Drop for NvmeVram {
    fn drop(&mut self) {
        if let Err(e) = self.lane.wait_all() {
            eprintln!("saber-hsa nvme: drain at drop: {e}");
        }
        let (hsa, _, _) = self.lane.runtime();
        if let Some(p) = &self.pool {
            p.close(hsa);
        }
        // SAFETY: both came from this reader's `open`; nothing in flight (drained above).
        unsafe {
            let _ = hsa.free(self.stage as *mut u8);
            if let Some((w, _)) = self.window {
                let _ = hsa.free(w as *mut u8);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reader_without_a_runtime_is_not_available_not_a_panic() {
        let f = File::open("/proc/self/exe").expect("a file");
        let e = NvmeVram::open("libsaber-no-such-hsa.so", 0x700, 4, 1 << 20, 0, (&f, 0))
            .expect_err("no runtime");
        assert!(matches!(e, Error::NotAvailable(_)), "{e}");
        let e = NvmeVram::open("libsaber-no-such-hsa.so", 0x700, 4, 1000, 0, (&f, 0))
            .expect_err("unaligned slot");
        assert!(matches!(e, Error::Invalid(_)), "{e}");
        let e = NvmeVram::open("libsaber-no-such-hsa.so", 0x700, 0, 4096, 0, (&f, 0))
            .expect_err("no slot");
        assert!(matches!(e, Error::Invalid(_)), "{e}");
        let e = NvmeVram::open("libsaber-no-such-hsa.so", 0x700, 1, 4096, 0, (&f, 100))
            .expect_err("unaligned probe");
        assert!(matches!(e, Error::Invalid(_)), "{e}");
    }

    #[test]
    fn a_pool_batch_translates_every_piece_or_refuses_the_whole_batch() {
        // FN71c: the boot fill's ring and the valve's bounce name GPU addresses in the pool
        let m = PoolMap {
            gpu: 0x7fe7_7f60_0000,
            host: 0x7f00_1000_0000,
            bytes: 1 << 30,
        };
        let rec = 2_617_344; // the Flash-Next expert stride (639 x 4096)
        let items: Vec<(u64, u64)> = (0..4).map(|k| (4096 * (1 + k), m.gpu + k * rec)).collect();
        let p = plan_pool_reads(&m, &items, rec).expect("in the pool");
        assert_eq!(p.len(), 4);
        for (k, &(off, h)) in p.iter().enumerate() {
            assert_eq!(off, 4096 * (1 + k as u64));
            assert_eq!(h, m.host + k as u64 * rec);
        }
        // the last slot that fits is in, the next one refuses the WHOLE batch
        let last = m.bytes / rec - 1;
        assert!(plan_pool_reads(&m, &[(0, m.gpu + last * rec)], rec).is_ok());
        let mut bad = items.clone();
        bad.push((0, m.gpu + (last + 1) * rec));
        assert!(plan_pool_reads(&m, &bad, rec).is_err());
        // alignment of the offset, the destination and the length
        assert!(plan_pool_reads(&m, &[(100, m.gpu)], rec).is_err());
        assert!(plan_pool_reads(&m, &[(0, m.gpu + 100)], rec).is_err());
        assert!(plan_pool_reads(&m, &[(0, m.gpu)], rec + 1).is_err());
        assert!(plan_pool_reads(&m, &[(0, m.gpu)], 0).is_err());
        // a HOST address is not a pool destination
        assert!(plan_pool_reads(&m, &[(0, m.host)], rec).is_err());
        assert!(plan_pool_reads(&m, &[], rec).expect("empty").is_empty());
    }

    #[test]
    fn a_pool_destination_translates_to_the_host_view_at_the_same_offset() {
        // FN71b's measured layout: the two views differ (HIP/HSA vs the sysfs mmap).
        let m = PoolMap {
            gpu: 0x7fe7_7f60_0000,
            host: 0x7f00_1000_0000,
            bytes: 2 << 30,
        };
        assert_eq!(m.host_of(m.gpu, 4096), Some(m.host));
        let rec = 2_621_440;
        assert_eq!(m.host_of(m.gpu + 512 * rec, rec), Some(m.host + 512 * rec));
        // the last byte of the pool is in, one past is out
        assert_eq!(
            m.host_of(m.gpu + m.bytes - 4096, 4096),
            Some(m.host + m.bytes - 4096)
        );
        assert_eq!(m.host_of(m.gpu + m.bytes - 4096, 8192), None);
        assert_eq!(m.host_of(m.gpu - 4096, 4096), None);
        assert_eq!(m.host_of(m.gpu, 0), None);
        assert_eq!(m.host_of(u64::MAX - 10, 4096), None);
        // the HOST address is not a valid destination (a caller passing the wrong view fails)
        assert_eq!(m.host_of(m.host, 4096), None);
    }

    #[test]
    fn the_route_prefers_the_pool_then_the_window_then_the_bounce() {
        let ok: std::result::Result<(), String> = Ok(());
        let no = |s: &str| -> std::result::Result<(), String> { Err(s.to_string()) };
        assert_eq!(choose_route(&ok, &ok), (NvmeRoute::Pool, None));
        assert_eq!(choose_route(&ok, &no("w")), (NvmeRoute::Pool, None));
        let (r, why) = choose_route(&no("absent"), &ok);
        assert_eq!(r, NvmeRoute::P2p);
        assert_eq!(why.as_deref(), Some("pool: absent"));
        let (r, why) = choose_route(&no("absent"), &no("EFAULT"));
        assert_eq!(r, NvmeRoute::Bounce);
        assert_eq!(why.as_deref(), Some("pool: absent; window: EFAULT"));
    }

    #[test]
    fn the_pool_sysfs_law_refuses_unpublished_partial_and_empty_pools() {
        assert_eq!(
            parse_pool_sysfs("2147483648\n", "2147483648\n", "1\n"),
            Ok(2 << 30)
        );
        assert!(parse_pool_sysfs("2147483648", "0", "1")
            .unwrap_err()
            .contains("another reader"));
        assert!(parse_pool_sysfs("2147483648", "2147483648", "0")
            .unwrap_err()
            .contains("published"));
        assert!(parse_pool_sysfs("0", "0", "1")
            .unwrap_err()
            .contains("empty"));
        assert!(parse_pool_sysfs("x", "0", "1").is_err());
    }

    #[test]
    fn the_pool_paths_follow_the_rocr_bdfid() {
        assert_eq!(bdf_string(0x0700), "0000:07:00.0");
        assert_eq!(bdf_string(0x0300), "0000:03:00.0");
        assert_eq!(
            pool_sysfs_dir(0x0700),
            PathBuf::from("/sys/bus/pci/devices/0000:07:00.0/p2pmem")
        );
        // the ioctl numbers the FN71 verify program used (_IOWR('d', ..))
        assert_eq!(IOCTL_GEM_CREATE, 0xC020_6440);
        assert_eq!(IOCTL_PRIME_HANDLE_TO_FD, 0xC00C_642D);
        assert_eq!(std::mem::size_of::<GemCreateIn>(), 32);
        assert_eq!(std::mem::size_of::<PrimeHandle>(), 12);
    }
}
