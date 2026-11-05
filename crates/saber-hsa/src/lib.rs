//! saber-hsa — the ROCr (HSA runtime) surface under HIP: agents, AQL queues, signals, code-object
//! executables, kernarg memory and SDMA copies, `dlopen`ed like `saber-hip` opens HIP.
//!
//! FN63 (a PROTOTYPE, not an engine backend): the question is whether submitting Saber's own
//! code objects straight to a hardware AQL queue -- one 64-byte packet a dispatch, a pre-baked
//! packet ARRAY a step, one doorbell -- removes the costs HIP adds on top of ROCr (the per-node
//! graph work, the ROCclr command queues, the `AsyncEventsLoop` thread). The kernel bytes do not
//! change: the same `.hsaco` is loaded through `hsa_executable_load_agent_code_object`.
//!
//! **Nothing here is linked at build time**, for the reason `saber-hip` gives: the host has no
//! ROCm, and the ~30 entry points this crate needs are hand-declared (their constants and struct
//! layouts were read from the dev image's `/opt/rocm/include/hsa/{hsa,hsa_ext_amd}.h`, ROCr
//! 1.21; `tools/capture/fn63/hsa_headers*.sh` prints them again).
//!
//! In ONE process with HIP the two share one ROCr: `dlopen("libhsa-runtime64.so.1")` returns the
//! library HIP already loaded, `hsa_init` is reference counted, and a `hipMalloc` pointer is a
//! valid address for a packet this crate submits (same process, same GPU VM). That is what lets
//! the prototype replay the HIP step's own kernarg images over the HIP step's own buffers.
//!
//! This crate and `saber-hip` are the only places in the engine allowed to contain
//! `unsafe extern "C"` (structural rule 2).

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::atomic::{AtomicU32, Ordering};

use libloading::Library;

pub mod lane;
pub mod nvme;
pub use lane::{EngineChoice, Engines, LaneCounts, SdmaLane};
pub use nvme::{
    bdf_string, plan_pool_reads, pool_bo, pool_sysfs_dir, NvmeCounts, NvmeOptions, NvmeRoute,
    NvmeVram, PoolAsk, PoolBo, PoolMap,
};

/// The soname this crate binds to (ROCr 1.x, ROCm 7).
pub const DEFAULT_HSA_SONAME: &str = "libhsa-runtime64.so.1";

/// Errors. An unknown status code is reported, never matched exhaustively (rule 5).
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The runtime could not be opened or lacks a symbol.
    #[error("hsa runtime not available: {0}")]
    NotAvailable(String),
    /// An HSA call returned a status other than success.
    #[error("{op} failed: hsa status 0x{code:x} ({msg})")]
    Hsa {
        /// The entry point.
        op: &'static str,
        /// The `hsa_status_t`.
        code: u32,
        /// `hsa_status_string`, when the runtime has one.
        msg: String,
    },
    /// A caller-side refusal (no such agent, a packet array larger than the queue, ...).
    #[error("{0}")]
    Invalid(String),
}

/// Result alias.
pub type Result<T> = std::result::Result<T, Error>;

// ---- constants (hsa.h / hsa_ext_amd.h, ROCr 1.21) ------------------------------------------- //

const HSA_STATUS_SUCCESS: u32 = 0;
const HSA_STATUS_ERROR_OUT_OF_RESOURCES: u32 = 0x1008;
const HSA_AGENT_INFO_NAME: u32 = 0;
const HSA_AGENT_INFO_PROFILE: u32 = 4;
const HSA_AGENT_INFO_DEVICE: u32 = 17;
const HSA_AMD_AGENT_INFO_BDFID: u32 = 0xA006;
const HSA_DEVICE_TYPE_CPU: u32 = 0;
const HSA_DEVICE_TYPE_GPU: u32 = 1;
const HSA_SYSTEM_INFO_TIMESTAMP: u32 = 2;
const HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY: u32 = 3;
const HSA_AMD_MEMORY_POOL_INFO_SEGMENT: u32 = 0;
const HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS: u32 = 1;
const HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED: u32 = 5;
const HSA_AMD_SEGMENT_GLOBAL: u32 = 0;
const POOL_FLAG_KERNARG_INIT: u32 = 1;
const POOL_FLAG_FINE_GRAINED: u32 = 2;
const POOL_FLAG_COARSE_GRAINED: u32 = 4;
/// `HSA_AMD_MEMORY_POOL_PCIE_FLAG` (an allocation flag, not a pool flag): public = host-visible VRAM.
const POOL_ALLOC_PCIE: u32 = 1;
const HSA_QUEUE_TYPE_MULTI: u32 = 0;
const HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT: u32 = 0;
const SYMBOL_INFO_KERNEL_OBJECT: u32 = 22;
const SYMBOL_INFO_KERNARG_SEGMENT_SIZE: u32 = 11;
const SYMBOL_INFO_GROUP_SEGMENT_SIZE: u32 = 13;
const SYMBOL_INFO_PRIVATE_SEGMENT_SIZE: u32 = 14;
const HSA_SIGNAL_CONDITION_LT: u32 = 2;

/// `hsa_packet_type_t`.
pub const PACKET_TYPE_INVALID: u16 = 1;
/// `HSA_PACKET_TYPE_KERNEL_DISPATCH`.
pub const PACKET_TYPE_KERNEL_DISPATCH: u16 = 2;
/// `HSA_PACKET_TYPE_BARRIER_AND`.
pub const PACKET_TYPE_BARRIER_AND: u16 = 3;
const HEADER_BARRIER_BIT: u16 = 8;
const HEADER_ACQUIRE_SHIFT: u16 = 9;
const HEADER_RELEASE_SHIFT: u16 = 11;

/// `hsa_fence_scope_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum Scope {
    /// No fence.
    None = 0,
    /// Agent scope: visible to the rest of this GPU (what a dependent dispatch needs).
    Agent = 1,
    /// System scope: visible to the host and the peers.
    System = 2,
}

/// `hsa_wait_state_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum WaitState {
    /// Sleep in the kernel on the signal's interrupt (after a short spin inside ROCr).
    Blocked = 0,
    /// Busy-poll the signal on the calling thread.
    Active = 1,
}

/// The 16-bit packet header word.
#[must_use]
pub fn header(kind: u16, barrier: bool, acquire: Scope, release: Scope) -> u16 {
    kind | (u16::from(barrier) << HEADER_BARRIER_BIT)
        | ((acquire as u16) << HEADER_ACQUIRE_SHIFT)
        | ((release as u16) << HEADER_RELEASE_SHIFT)
}

// ---- handles ----------------------------------------------------------------------------------- //

/// `hsa_agent_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Agent(pub u64);

/// `hsa_signal_t`. Created and destroyed through [`Hsa`]; a handle is plain data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Signal(pub u64);

/// `hsa_amd_memory_pool_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Pool(pub u64);

/// `hsa_queue_t` (hsa.h, little-endian large model).
#[repr(C)]
struct RawQueue {
    kind: u32,
    features: u32,
    base_address: *mut c_void,
    doorbell_signal: u64,
    size: u32,
    reserved1: u32,
    id: u64,
}

/// `hsa_amd_profiling_dispatch_time_t` / `..._async_copy_time_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Times {
    /// Start, in the system timestamp domain.
    pub start: u64,
    /// End, in the system timestamp domain.
    pub end: u64,
}

/// One AQL kernel-dispatch packet's fields (hsa.h `hsa_kernel_dispatch_packet_t`).
#[derive(Debug, Clone, Copy)]
pub struct Dispatch {
    /// The header word (`header(PACKET_TYPE_KERNEL_DISPATCH, ..)`).
    pub header: u16,
    /// Work-group size in work-items.
    pub block: [u32; 3],
    /// Grid in WORK-GROUPS (HIP's convention); the packet carries work-items.
    pub grid_blocks: [u32; 3],
    /// Private segment bytes per work-item.
    pub private_bytes: u32,
    /// Group segment bytes (static + dynamic LDS).
    pub group_bytes: u32,
    /// `HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT`.
    pub kernel_object: u64,
    /// Address of the full kernarg segment (explicit + hidden).
    pub kernarg: u64,
    /// Completion signal handle, 0 for none.
    pub completion: u64,
}

impl Dispatch {
    /// The 64 packet bytes. Word 0 (header + setup) is what [`Queue::submit`] stores LAST.
    #[must_use]
    pub fn bytes(&self) -> [u8; 64] {
        let mut p = [0u8; 64];
        p[0..2].copy_from_slice(&self.header.to_le_bytes());
        p[2..4].copy_from_slice(&3u16.to_le_bytes()); // setup: 3 dimensions, as HIP dispatches
        for d in 0..3 {
            let wg = u16::try_from(self.block[d]).unwrap_or(u16::MAX);
            p[4 + 2 * d..6 + 2 * d].copy_from_slice(&wg.to_le_bytes());
            let g = self.grid_blocks[d].saturating_mul(self.block[d]);
            p[12 + 4 * d..16 + 4 * d].copy_from_slice(&g.to_le_bytes());
        }
        p[24..28].copy_from_slice(&self.private_bytes.to_le_bytes());
        p[28..32].copy_from_slice(&self.group_bytes.to_le_bytes());
        p[32..40].copy_from_slice(&self.kernel_object.to_le_bytes());
        p[40..48].copy_from_slice(&self.kernarg.to_le_bytes());
        p[56..64].copy_from_slice(&self.completion.to_le_bytes());
        p
    }
}

/// One barrier-AND packet: the queue does not pass it until every dependency signal reads 0.
#[must_use]
pub fn barrier_and(header_word: u16, deps: &[Signal], completion: u64) -> [u8; 64] {
    let mut p = [0u8; 64];
    p[0..2].copy_from_slice(&header_word.to_le_bytes());
    for (i, s) in deps.iter().take(5).enumerate() {
        p[8 + 8 * i..16 + 8 * i].copy_from_slice(&s.0.to_le_bytes());
    }
    p[56..64].copy_from_slice(&completion.to_le_bytes());
    p
}

// ---- the library -------------------------------------------------------------------------------- //

type AgentCb = unsafe extern "C" fn(u64, *mut c_void) -> u32;
type PoolCb = unsafe extern "C" fn(u64, *mut c_void) -> u32;

macro_rules! fns {
    ($($field:ident : $sym:literal => $ty:ty;)*) => {
        struct Fns { $($field: $ty,)* }
        impl Fns {
            fn load(lib: &Library) -> Result<Fns> {
                // SAFETY: each symbol is looked up by its exact C name and given the signature
                // hsa.h / hsa_ext_amd.h declare for it.
                unsafe {
                    Ok(Fns { $($field: *lib.get::<$ty>(concat!($sym, "\0").as_bytes())
                        .map_err(|e| Error::NotAvailable(format!("{}: {e}", $sym)))?,)* })
                }
            }
        }
    };
}

fns! {
    init: "hsa_init" => unsafe extern "C" fn() -> u32;
    shut_down: "hsa_shut_down" => unsafe extern "C" fn() -> u32;
    status_string: "hsa_status_string" => unsafe extern "C" fn(u32, *mut *const c_char) -> u32;
    system_get_info: "hsa_system_get_info" => unsafe extern "C" fn(u32, *mut c_void) -> u32;
    iterate_agents: "hsa_iterate_agents" => unsafe extern "C" fn(AgentCb, *mut c_void) -> u32;
    agent_get_info: "hsa_agent_get_info" => unsafe extern "C" fn(u64, u32, *mut c_void) -> u32;
    iterate_pools: "hsa_amd_agent_iterate_memory_pools"
        => unsafe extern "C" fn(u64, PoolCb, *mut c_void) -> u32;
    pool_get_info: "hsa_amd_memory_pool_get_info"
        => unsafe extern "C" fn(u64, u32, *mut c_void) -> u32;
    pool_allocate: "hsa_amd_memory_pool_allocate"
        => unsafe extern "C" fn(u64, usize, u32, *mut *mut c_void) -> u32;
    pool_free: "hsa_amd_memory_pool_free" => unsafe extern "C" fn(*mut c_void) -> u32;
    allow_access: "hsa_amd_agents_allow_access"
        => unsafe extern "C" fn(u32, *const u64, *const u32, *const c_void) -> u32;
    memory_copy: "hsa_memory_copy" => unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> u32;
    queue_create: "hsa_queue_create" => unsafe extern "C" fn(
        u64, u32, u32, *const c_void, *mut c_void, u32, u32, *mut *mut RawQueue) -> u32;
    queue_destroy: "hsa_queue_destroy" => unsafe extern "C" fn(*mut RawQueue) -> u32;
    queue_add_write_index: "hsa_queue_add_write_index_screlease"
        => unsafe extern "C" fn(*const RawQueue, u64) -> u64;
    queue_load_read_index: "hsa_queue_load_read_index_scacquire"
        => unsafe extern "C" fn(*const RawQueue) -> u64;
    signal_create: "hsa_signal_create"
        => unsafe extern "C" fn(i64, u32, *const u64, *mut u64) -> u32;
    signal_destroy: "hsa_signal_destroy" => unsafe extern "C" fn(u64) -> u32;
    signal_store_relaxed: "hsa_signal_store_relaxed" => unsafe extern "C" fn(u64, i64);
    signal_store_screlease: "hsa_signal_store_screlease" => unsafe extern "C" fn(u64, i64);
    signal_load_scacquire: "hsa_signal_load_scacquire" => unsafe extern "C" fn(u64) -> i64;
    signal_wait_scacquire: "hsa_signal_wait_scacquire"
        => unsafe extern "C" fn(u64, u32, i64, u64, u32) -> i64;
    reader_from_memory: "hsa_code_object_reader_create_from_memory"
        => unsafe extern "C" fn(*const c_void, usize, *mut u64) -> u32;
    reader_destroy: "hsa_code_object_reader_destroy" => unsafe extern "C" fn(u64) -> u32;
    executable_create_alt: "hsa_executable_create_alt"
        => unsafe extern "C" fn(u32, u32, *const c_char, *mut u64) -> u32;
    executable_load_agent_code_object: "hsa_executable_load_agent_code_object"
        => unsafe extern "C" fn(u64, u64, u64, *const c_char, *mut u64) -> u32;
    executable_freeze: "hsa_executable_freeze" => unsafe extern "C" fn(u64, *const c_char) -> u32;
    executable_destroy: "hsa_executable_destroy" => unsafe extern "C" fn(u64) -> u32;
    executable_get_symbol_by_name: "hsa_executable_get_symbol_by_name"
        => unsafe extern "C" fn(u64, *const c_char, *const u64, *mut u64) -> u32;
    symbol_get_info: "hsa_executable_symbol_get_info"
        => unsafe extern "C" fn(u64, u32, *mut c_void) -> u32;
    profiling_set_enabled: "hsa_amd_profiling_set_profiler_enabled"
        => unsafe extern "C" fn(*mut RawQueue, c_int) -> u32;
    profiling_dispatch_time: "hsa_amd_profiling_get_dispatch_time"
        => unsafe extern "C" fn(u64, u64, *mut Times) -> u32;
    profiling_copy_enable: "hsa_amd_profiling_async_copy_enable" => unsafe extern "C" fn(bool) -> u32;
    profiling_copy_time: "hsa_amd_profiling_get_async_copy_time"
        => unsafe extern "C" fn(u64, *mut Times) -> u32;
    async_copy: "hsa_amd_memory_async_copy" => unsafe extern "C" fn(
        *mut c_void, u64, *const c_void, u64, usize, u32, *const u64, u64) -> u32;
}

/// FN63b: entry points a ROCr may lack (looked up one by one: a missing one is `None`, never an
/// open failure, so FN63's prototype keeps opening on an older runtime).
struct OptFns {
    copy_on_engine: Option<
        unsafe extern "C" fn(
            *mut c_void,
            u64,
            *const c_void,
            u64,
            usize,
            u32,
            *const u64,
            u64,
            u32,
            bool,
        ) -> u32,
    >,
    engine_status: Option<unsafe extern "C" fn(u64, u64, *mut u32) -> u32>,
    /// FN71b: `hsa_amd_interop_map_buffer(num_agents, agents, fd, flags, size*, ptr*, msize*, meta*)`.
    interop_map: Option<
        unsafe extern "C" fn(
            u32,
            *const u64,
            c_int,
            u32,
            *mut usize,
            *mut *mut c_void,
            *mut usize,
            *mut *const c_void,
        ) -> u32,
    >,
    /// FN71b: `hsa_amd_interop_unmap_buffer(ptr)`.
    interop_unmap: Option<unsafe extern "C" fn(*mut c_void) -> u32>,
}

impl OptFns {
    fn load(lib: &Library) -> OptFns {
        // SAFETY: each symbol is looked up by its exact C name and given the signature
        // hsa_ext_amd.h (ROCr 1.21) declares for it (`hsa_amd_sdma_engine_id_t` is a C enum = u32).
        unsafe {
            OptFns {
                copy_on_engine: lib
                    .get(b"hsa_amd_memory_async_copy_on_engine\0")
                    .ok()
                    .map(|s| *s),
                engine_status: lib
                    .get(b"hsa_amd_memory_copy_engine_status\0")
                    .ok()
                    .map(|s| *s),
                interop_map: lib.get(b"hsa_amd_interop_map_buffer\0").ok().map(|s| *s),
                interop_unmap: lib.get(b"hsa_amd_interop_unmap_buffer\0").ok().map(|s| *s),
            }
        }
    }
}

/// The opened runtime. `hsa_init` on open, `hsa_shut_down` on drop (both reference counted, so
/// sharing the process with HIP is safe).
pub struct Hsa {
    f: Fns,
    opt: OptFns,
    _lib: Library,
}

impl std::fmt::Debug for Hsa {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Hsa")
    }
}

/// What [`Hsa::agents`] reports for one agent.
#[derive(Debug, Clone)]
pub struct AgentInfo {
    /// The handle.
    pub agent: Agent,
    /// `HSA_AGENT_INFO_NAME` (`gfx1201`, or the CPU's name).
    pub name: String,
    /// GPU (true) or CPU.
    pub gpu: bool,
    /// PCI bus/device/function as `(bus << 8) | (dev << 3) | func`; 0 for the CPU.
    pub bdfid: u32,
    /// `HSA_AGENT_INFO_PROFILE` (0 base, 1 full).
    pub profile: u32,
}

/// A loaded kernel: what a dispatch packet needs.
#[derive(Debug, Clone, Copy)]
pub struct KernelSym {
    /// `KERNEL_OBJECT`.
    pub object: u64,
    /// `KERNARG_SEGMENT_SIZE` (explicit + hidden).
    pub kernarg_bytes: u32,
    /// `GROUP_SEGMENT_SIZE` (static LDS).
    pub group_bytes: u32,
    /// `PRIVATE_SEGMENT_SIZE`.
    pub private_bytes: u32,
}

impl Hsa {
    /// `dlopen` + `hsa_init`.
    pub fn open(soname: &str) -> Result<Hsa> {
        // SAFETY: loading a shared library runs its initialisers; ROCr's are the ones HIP runs.
        let lib = unsafe { Library::new(soname) }
            .map_err(|e| Error::NotAvailable(format!("{soname}: {e}")))?;
        let f = Fns::load(&lib)?;
        let opt = OptFns::load(&lib);
        let h = Hsa { f, opt, _lib: lib };
        // SAFETY: no arguments.
        h.check("hsa_init", unsafe { (h.f.init)() })?;
        Ok(h)
    }

    fn check(&self, op: &'static str, code: u32) -> Result<()> {
        if code == HSA_STATUS_SUCCESS {
            return Ok(());
        }
        let mut p: *const c_char = std::ptr::null();
        // SAFETY: out-pointer to a static string owned by the runtime.
        let msg = if unsafe { (self.f.status_string)(code, &mut p) } == HSA_STATUS_SUCCESS
            && !p.is_null()
        {
            // SAFETY: NUL-terminated static string.
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        } else {
            "unknown status".into()
        };
        Err(Error::Hsa { op, code, msg })
    }

    /// The system timestamp (`HSA_SYSTEM_INFO_TIMESTAMP`), the domain profiling times are in.
    pub fn timestamp(&self) -> Result<u64> {
        let mut v = 0u64;
        // SAFETY: the attribute's value is a u64.
        self.check("hsa_system_get_info", unsafe {
            (self.f.system_get_info)(HSA_SYSTEM_INFO_TIMESTAMP, (&raw mut v).cast())
        })?;
        Ok(v)
    }

    /// Ticks per second of [`Hsa::timestamp`].
    pub fn timestamp_hz(&self) -> Result<u64> {
        let mut v = 0u64;
        // SAFETY: the attribute's value is a u64.
        self.check("hsa_system_get_info", unsafe {
            (self.f.system_get_info)(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, (&raw mut v).cast())
        })?;
        Ok(v)
    }

    /// Every agent.
    pub fn agents(&self) -> Result<Vec<AgentInfo>> {
        unsafe extern "C" fn cb(agent: u64, data: *mut c_void) -> u32 {
            // SAFETY: `data` is the Vec below, alive for the iteration.
            unsafe { &mut *data.cast::<Vec<u64>>() }.push(agent);
            HSA_STATUS_SUCCESS
        }
        let mut raw: Vec<u64> = Vec::new();
        // SAFETY: the callback only pushes into the Vec it is handed.
        self.check("hsa_iterate_agents", unsafe {
            (self.f.iterate_agents)(cb, (&raw mut raw).cast())
        })?;
        let mut out = Vec::new();
        for a in raw {
            let mut dev = 0u32;
            let mut name = [0u8; 64];
            let mut bdf = 0u32;
            let mut profile = 0u32;
            // SAFETY: each attribute's value type per hsa.h (u32 enum, char[64], u32, u32 enum).
            unsafe {
                self.check(
                    "hsa_agent_get_info",
                    (self.f.agent_get_info)(a, HSA_AGENT_INFO_DEVICE, (&raw mut dev).cast()),
                )?;
                self.check(
                    "hsa_agent_get_info",
                    (self.f.agent_get_info)(a, HSA_AGENT_INFO_NAME, name.as_mut_ptr().cast()),
                )?;
                self.check(
                    "hsa_agent_get_info",
                    (self.f.agent_get_info)(a, HSA_AGENT_INFO_PROFILE, (&raw mut profile).cast()),
                )?;
                if dev == HSA_DEVICE_TYPE_GPU {
                    self.check(
                        "hsa_agent_get_info",
                        (self.f.agent_get_info)(a, HSA_AMD_AGENT_INFO_BDFID, (&raw mut bdf).cast()),
                    )?;
                }
            }
            if dev != HSA_DEVICE_TYPE_GPU && dev != HSA_DEVICE_TYPE_CPU {
                continue;
            }
            let end = name.iter().position(|&c| c == 0).unwrap_or(name.len());
            out.push(AgentInfo {
                agent: Agent(a),
                name: String::from_utf8_lossy(&name[..end]).into_owned(),
                gpu: dev == HSA_DEVICE_TYPE_GPU,
                bdfid: bdf,
                profile,
            });
        }
        Ok(out)
    }

    /// The agent's global pools as `(pool, flags)`, flags = `HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_*`.
    fn global_pools(&self, agent: Agent) -> Result<Vec<(Pool, u32)>> {
        unsafe extern "C" fn cb(pool: u64, data: *mut c_void) -> u32 {
            // SAFETY: `data` is the Vec below.
            unsafe { &mut *data.cast::<Vec<u64>>() }.push(pool);
            HSA_STATUS_SUCCESS
        }
        let mut raw: Vec<u64> = Vec::new();
        // SAFETY: as above.
        self.check("hsa_amd_agent_iterate_memory_pools", unsafe {
            (self.f.iterate_pools)(agent.0, cb, (&raw mut raw).cast())
        })?;
        let mut out = Vec::new();
        for p in raw {
            let (mut seg, mut flags, mut alloc) = (0u32, 0u32, false);
            // SAFETY: u32 enum, u32 flags, bool.
            unsafe {
                self.check(
                    "hsa_amd_memory_pool_get_info",
                    (self.f.pool_get_info)(
                        p,
                        HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                        (&raw mut seg).cast(),
                    ),
                )?;
                if seg != HSA_AMD_SEGMENT_GLOBAL {
                    continue;
                }
                self.check(
                    "hsa_amd_memory_pool_get_info",
                    (self.f.pool_get_info)(
                        p,
                        HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                        (&raw mut flags).cast(),
                    ),
                )?;
                self.check(
                    "hsa_amd_memory_pool_get_info",
                    (self.f.pool_get_info)(
                        p,
                        HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                        (&raw mut alloc).cast(),
                    ),
                )?;
            }
            if alloc {
                out.push((Pool(p), flags));
            }
        }
        Ok(out)
    }

    /// Allocate `bytes` from the CPU agent's KERNARG pool (fine-grained system memory, GPU-visible)
    /// and grant `gpu` access. The returned pointer is host-writable.
    pub fn alloc_kernarg_sys(&self, cpu: Agent, gpu: Agent, bytes: usize) -> Result<*mut u8> {
        let pool = self
            .global_pools(cpu)?
            .into_iter()
            .find(|(_, f)| f & POOL_FLAG_KERNARG_INIT != 0)
            .ok_or_else(|| Error::Invalid("the CPU agent has no kernarg pool".into()))?;
        let p = self.pool_alloc(pool.0, bytes)?;
        self.allow(gpu, p)?;
        Ok(p.cast())
    }

    /// Allocate `bytes` of the GPU's COARSE-grained VRAM (not host-writable: fill it with
    /// [`Hsa::copy`]).
    pub fn alloc_vram(&self, gpu: Agent, bytes: usize) -> Result<*mut u8> {
        let pool = self
            .global_pools(gpu)?
            .into_iter()
            .find(|(_, f)| f & POOL_FLAG_COARSE_GRAINED != 0)
            .ok_or_else(|| Error::Invalid("the GPU agent has no coarse-grained pool".into()))?;
        Ok(self.pool_alloc(pool.0, bytes)?.cast())
    }

    /// Fine-grained system memory (host-coherent), GPU-visible.
    pub fn alloc_fine_sys(&self, cpu: Agent, gpu: Agent, bytes: usize) -> Result<*mut u8> {
        let pool = self
            .global_pools(cpu)?
            .into_iter()
            .find(|(_, f)| f & POOL_FLAG_FINE_GRAINED != 0 && f & POOL_FLAG_KERNARG_INIT == 0)
            .ok_or_else(|| Error::Invalid("the CPU agent has no fine-grained pool".into()))?;
        let p = self.pool_alloc(pool.0, bytes)?;
        self.allow(gpu, p)?;
        Ok(p.cast())
    }

    /// FN70: `bytes` of the GPU's coarse-grained VRAM allocated PUBLIC
    /// (`HSA_AMD_MEMORY_POOL_PCIE_FLAG`: inside the large BAR) and made accessible to the CPU
    /// agent -- one address the GPU and the host both use. The host reaches it through the BAR
    /// mapping (uncached, write-combined); it is the only VRAM a host-side `pread` can name.
    pub fn alloc_vram_host_visible(&self, cpu: Agent, gpu: Agent, bytes: usize) -> Result<*mut u8> {
        let pool = self
            .global_pools(gpu)?
            .into_iter()
            .find(|(_, f)| f & POOL_FLAG_COARSE_GRAINED != 0)
            .ok_or_else(|| Error::Invalid("the GPU agent has no coarse-grained pool".into()))?;
        let p = self.pool_alloc_flags(pool.0, bytes, POOL_ALLOC_PCIE)?;
        if let Err(e) = self.allow(cpu, p) {
            // SAFETY: just allocated, never shared.
            let _ = unsafe { self.free(p.cast()) };
            return Err(e);
        }
        Ok(p.cast())
    }

    fn pool_alloc(&self, pool: Pool, bytes: usize) -> Result<*mut c_void> {
        self.pool_alloc_flags(pool, bytes, 0)
    }

    fn pool_alloc_flags(&self, pool: Pool, bytes: usize, flags: u32) -> Result<*mut c_void> {
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: out-pointer.
        self.check("hsa_amd_memory_pool_allocate", unsafe {
            (self.f.pool_allocate)(pool.0, bytes.max(64), flags, &mut p)
        })?;
        Ok(p)
    }

    fn allow(&self, gpu: Agent, p: *mut c_void) -> Result<()> {
        let agents = [gpu.0];
        // SAFETY: one agent, no flags, a pointer this crate allocated.
        self.check("hsa_amd_agents_allow_access", unsafe {
            (self.f.allow_access)(1, agents.as_ptr(), std::ptr::null(), p)
        })
    }

    /// Free a pool allocation.
    ///
    /// # Safety
    /// `p` came from one of this crate's allocators and nothing in flight reads it.
    pub unsafe fn free(&self, p: *mut u8) -> Result<()> {
        // SAFETY: the caller's contract.
        self.check("hsa_amd_memory_pool_free", unsafe {
            (self.f.pool_free)(p.cast())
        })
    }

    /// `hsa_memory_copy` (synchronous; any direction).
    ///
    /// # Safety
    /// Both ranges are valid for `bytes`.
    pub unsafe fn copy(&self, dst: *mut u8, src: *const u8, bytes: usize) -> Result<()> {
        // SAFETY: the caller's contract.
        self.check("hsa_memory_copy", unsafe {
            (self.f.memory_copy)(dst.cast(), src.cast(), bytes)
        })
    }

    /// A signal. `consumers` empty = any agent may wait, the host included (ROCr backs it with
    /// an interrupt event); `[gpu]` = only the GPU consumes it (no interrupt).
    pub fn signal(&self, initial: i64, consumers: &[Agent]) -> Result<Signal> {
        let c: Vec<u64> = consumers.iter().map(|a| a.0).collect();
        let mut s = 0u64;
        // SAFETY: out-pointer; consumer list borrowed for the call.
        self.check("hsa_signal_create", unsafe {
            (self.f.signal_create)(
                initial,
                u32::try_from(c.len()).unwrap_or(0),
                if c.is_empty() {
                    std::ptr::null()
                } else {
                    c.as_ptr()
                },
                &mut s,
            )
        })?;
        Ok(Signal(s))
    }

    /// Destroy a signal.
    pub fn signal_destroy(&self, s: Signal) -> Result<()> {
        // SAFETY: a handle this crate created; nothing in flight references it (caller drained).
        self.check("hsa_signal_destroy", unsafe {
            (self.f.signal_destroy)(s.0)
        })
    }

    /// `hsa_signal_store_relaxed`.
    pub fn signal_store(&self, s: Signal, v: i64) {
        // SAFETY: a live handle.
        unsafe { (self.f.signal_store_relaxed)(s.0, v) }
    }

    /// `hsa_signal_load_scacquire`.
    pub fn signal_load(&self, s: Signal) -> i64 {
        // SAFETY: a live handle.
        unsafe { (self.f.signal_load_scacquire)(s.0) }
    }

    /// Wait until `s < 1` (a completion signal set to 1 that the packet decrements).
    pub fn wait_done(&self, s: Signal, state: WaitState, timeout_ticks: u64) -> i64 {
        // SAFETY: a live handle.
        unsafe {
            (self.f.signal_wait_scacquire)(
                s.0,
                HSA_SIGNAL_CONDITION_LT,
                1,
                timeout_ticks,
                state as u32,
            )
        }
    }

    /// `hsa_amd_profiling_get_dispatch_time` for a packet that carried `s` (profiling on).
    pub fn dispatch_time(&self, gpu: Agent, s: Signal) -> Result<Times> {
        let mut t = Times::default();
        // SAFETY: out-pointer.
        self.check("hsa_amd_profiling_get_dispatch_time", unsafe {
            (self.f.profiling_dispatch_time)(gpu.0, s.0, &mut t)
        })?;
        Ok(t)
    }

    /// Enable async-copy profiling (process-wide).
    pub fn copy_profiling(&self, on: bool) -> Result<()> {
        // SAFETY: no pointers.
        self.check("hsa_amd_profiling_async_copy_enable", unsafe {
            (self.f.profiling_copy_enable)(on)
        })
    }

    /// `hsa_amd_profiling_get_async_copy_time`.
    pub fn copy_time(&self, s: Signal) -> Result<Times> {
        let mut t = Times::default();
        // SAFETY: out-pointer.
        self.check("hsa_amd_profiling_get_async_copy_time", unsafe {
            (self.f.profiling_copy_time)(s.0, &mut t)
        })?;
        Ok(t)
    }

    /// `hsa_amd_memory_async_copy` (SDMA by default): `completion` is decremented when it lands.
    ///
    /// # Safety
    /// Both ranges are valid for `bytes`, visible to the copy agents, and stay alive until the
    /// completion signal fires.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn async_copy(
        &self,
        dst: u64,
        dst_agent: Agent,
        src: u64,
        src_agent: Agent,
        bytes: usize,
        deps: &[Signal],
        completion: Signal,
    ) -> Result<()> {
        let d: Vec<u64> = deps.iter().map(|s| s.0).collect();
        // SAFETY: the caller's contract.
        self.check("hsa_amd_memory_async_copy", unsafe {
            (self.f.async_copy)(
                dst as *mut c_void,
                dst_agent.0,
                src as *const c_void,
                src_agent.0,
                bytes,
                u32::try_from(d.len()).unwrap_or(0),
                if d.is_empty() {
                    std::ptr::null()
                } else {
                    d.as_ptr()
                },
                completion.0,
            )
        })
    }

    /// FN63b: `hsa_amd_memory_copy_engine_status` -- the SDMA engines available for a copy from
    /// `src` to `dst` (a mask of `HSA_AMD_SDMA_ENGINE_*` bits). `Ok(None)` when the runtime lacks
    /// the entry point; `Ok(Some(0))` when it reports no engine available.
    pub fn copy_engine_status(&self, dst: Agent, src: Agent) -> Result<Option<u32>> {
        let Some(f) = self.opt.engine_status else {
            return Ok(None);
        };
        let mut mask = 0u32;
        // SAFETY: out-pointer.
        let code = unsafe { f(dst.0, src.0, &mut mask) };
        if code == HSA_STATUS_ERROR_OUT_OF_RESOURCES {
            return Ok(Some(0));
        }
        self.check("hsa_amd_memory_copy_engine_status", code)?;
        Ok(Some(mask))
    }

    /// FN71b: map a dma-buf `fd` (the amdgpu P2PDMA pool's PRIME fd) into `gpu`'s address space
    /// with `hsa_amd_interop_map_buffer`; returns `(device pointer, bytes)`. The runtime takes
    /// its own reference on the buffer, so the caller may close `fd` afterwards.
    pub fn interop_map_buffer(&self, gpu: Agent, fd: c_int) -> Result<(*mut u8, usize)> {
        let f = self.opt.interop_map.ok_or_else(|| {
            Error::NotAvailable("hsa_amd_interop_map_buffer missing from this runtime".into())
        })?;
        let agents = [gpu.0];
        let (mut size, mut ptr, mut msize) = (0usize, std::ptr::null_mut(), 0usize);
        let mut meta: *const c_void = std::ptr::null();
        // SAFETY: one agent, out-pointers owned by this frame; `fd` is the caller's open dma-buf.
        self.check("hsa_amd_interop_map_buffer", unsafe {
            f(
                1,
                agents.as_ptr(),
                fd,
                0,
                &mut size,
                &mut ptr,
                &mut msize,
                &mut meta,
            )
        })?;
        Ok((ptr.cast(), size))
    }

    /// FN71b: undo [`Hsa::interop_map_buffer`].
    ///
    /// # Safety
    /// `p` came from `interop_map_buffer` and nothing in flight reads or writes it.
    pub unsafe fn interop_unmap_buffer(&self, p: *mut u8) -> Result<()> {
        let f = self.opt.interop_unmap.ok_or_else(|| {
            Error::NotAvailable("hsa_amd_interop_unmap_buffer missing from this runtime".into())
        })?;
        // SAFETY: the caller's contract.
        self.check("hsa_amd_interop_unmap_buffer", unsafe { f(p.cast()) })
    }

    /// FN63b: whether `hsa_amd_memory_async_copy_on_engine` exists.
    #[must_use]
    pub fn has_copy_on_engine(&self) -> bool {
        self.opt.copy_on_engine.is_some()
    }

    /// FN63b: `hsa_amd_memory_async_copy_on_engine` with `force_copy_on_sdma` -- the copy runs on
    /// the SDMA engine `engine` (one `HSA_AMD_SDMA_ENGINE_*` bit).
    ///
    /// # Safety
    /// As [`Hsa::async_copy`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn async_copy_on_engine(
        &self,
        dst: u64,
        dst_agent: Agent,
        src: u64,
        src_agent: Agent,
        bytes: usize,
        deps: &[Signal],
        completion: Signal,
        engine: u32,
    ) -> Result<()> {
        let f = self
            .opt
            .copy_on_engine
            .ok_or_else(|| Error::NotAvailable("hsa_amd_memory_async_copy_on_engine".into()))?;
        // `Signal` is `repr(transparent)` over the u64 handle: the slice is the C array
        // SAFETY: the caller's contract.
        self.check("hsa_amd_memory_async_copy_on_engine", unsafe {
            f(
                dst as *mut c_void,
                dst_agent.0,
                src as *const c_void,
                src_agent.0,
                bytes,
                u32::try_from(deps.len()).unwrap_or(0),
                if deps.is_empty() {
                    std::ptr::null()
                } else {
                    deps.as_ptr().cast()
                },
                completion.0,
                engine,
                true,
            )
        })
    }

    /// Load one code object (a bare amdgcn ELF, NOT a clang offload bundle) for `gpu`, frozen.
    pub fn load_executable(&self, gpu: Agent, profile: u32, elf: &[u8]) -> Result<Executable<'_>> {
        let mut reader = 0u64;
        // SAFETY: the image is borrowed for the call; ROCr copies what it keeps.
        self.check("hsa_code_object_reader_create_from_memory", unsafe {
            (self.f.reader_from_memory)(elf.as_ptr().cast(), elf.len(), &mut reader)
        })?;
        let mut exec = 0u64;
        // SAFETY: out-pointers; no options.
        let r = (|| {
            self.check("hsa_executable_create_alt", unsafe {
                (self.f.executable_create_alt)(
                    profile,
                    HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                    std::ptr::null(),
                    &mut exec,
                )
            })?;
            self.check("hsa_executable_load_agent_code_object", unsafe {
                (self.f.executable_load_agent_code_object)(
                    exec,
                    gpu.0,
                    reader,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                )
            })?;
            self.check("hsa_executable_freeze", unsafe {
                (self.f.executable_freeze)(exec, std::ptr::null())
            })
        })();
        // SAFETY: the reader is no longer needed once loaded (or on failure).
        let _ = unsafe { (self.f.reader_destroy)(reader) };
        r?;
        Ok(Executable {
            hsa: self,
            raw: exec,
            gpu,
        })
    }

    /// An AQL queue on `gpu` of `size` packets (a power of two), profiling on or off.
    pub fn queue(&self, gpu: Agent, size: u32, profiling: bool) -> Result<Queue<'_>> {
        let mut q: *mut RawQueue = std::ptr::null_mut();
        // SAFETY: no callback; the runtime picks segment sizes (u32::MAX = "default").
        self.check("hsa_queue_create", unsafe {
            (self.f.queue_create)(
                gpu.0,
                size,
                HSA_QUEUE_TYPE_MULTI,
                std::ptr::null(),
                std::ptr::null_mut(),
                u32::MAX,
                u32::MAX,
                &mut q,
            )
        })?;
        if profiling {
            // SAFETY: a live queue.
            self.check("hsa_amd_profiling_set_profiler_enabled", unsafe {
                (self.f.profiling_set_enabled)(q, 1)
            })?;
        }
        Ok(Queue { hsa: self, raw: q })
    }
}

impl Drop for Hsa {
    fn drop(&mut self) {
        // SAFETY: balanced with the hsa_init in `open`.
        let _ = unsafe { (self.f.shut_down)() };
    }
}

/// A frozen executable (one code object).
pub struct Executable<'h> {
    hsa: &'h Hsa,
    raw: u64,
    gpu: Agent,
}

impl Executable<'_> {
    /// Look a kernel up by its symbol (`<name>.kd`).
    pub fn kernel(&self, symbol: &str) -> Result<KernelSym> {
        let name = CString::new(symbol).map_err(|_| Error::Invalid(format!("NUL in {symbol}")))?;
        let mut sym = 0u64;
        let agent = self.gpu.0;
        // SAFETY: out-pointer; the agent is borrowed for the call.
        self.hsa
            .check("hsa_executable_get_symbol_by_name", unsafe {
                (self.hsa.f.executable_get_symbol_by_name)(
                    self.raw,
                    name.as_ptr(),
                    &agent,
                    &mut sym,
                )
            })?;
        let mut k = KernelSym {
            object: 0,
            kernarg_bytes: 0,
            group_bytes: 0,
            private_bytes: 0,
        };
        // SAFETY: KERNEL_OBJECT is a u64, the three sizes are u32.
        unsafe {
            let f = self.hsa.f.symbol_get_info;
            self.hsa.check(
                "hsa_executable_symbol_get_info",
                f(sym, SYMBOL_INFO_KERNEL_OBJECT, (&raw mut k.object).cast()),
            )?;
            self.hsa.check(
                "hsa_executable_symbol_get_info",
                f(
                    sym,
                    SYMBOL_INFO_KERNARG_SEGMENT_SIZE,
                    (&raw mut k.kernarg_bytes).cast(),
                ),
            )?;
            self.hsa.check(
                "hsa_executable_symbol_get_info",
                f(
                    sym,
                    SYMBOL_INFO_GROUP_SEGMENT_SIZE,
                    (&raw mut k.group_bytes).cast(),
                ),
            )?;
            self.hsa.check(
                "hsa_executable_symbol_get_info",
                f(
                    sym,
                    SYMBOL_INFO_PRIVATE_SEGMENT_SIZE,
                    (&raw mut k.private_bytes).cast(),
                ),
            )?;
        }
        Ok(k)
    }
}

impl Drop for Executable<'_> {
    fn drop(&mut self) {
        // SAFETY: nothing in flight (the caller drained its queues first).
        let _ = unsafe { (self.hsa.f.executable_destroy)(self.raw) };
    }
}

/// An AQL queue.
pub struct Queue<'h> {
    hsa: &'h Hsa,
    raw: *mut RawQueue,
}

impl Queue<'_> {
    /// Packets the ring holds.
    #[must_use]
    pub fn size(&self) -> u32 {
        // SAFETY: a live queue; `size` is immutable after creation.
        unsafe { (*self.raw).size }
    }

    /// Write `packets` into the ring and ring the doorbell ONCE: the HSA analogue of a graph
    /// launch. Each packet's 60-byte body is written first and its header word is stored last
    /// with release semantics, in order, so the packet processor never reads a half-written
    /// packet. Returns the index of the last packet.
    ///
    /// # Safety
    /// Every address in every packet (kernel object, kernarg, signals) is live until the packets
    /// complete.
    pub unsafe fn submit(&self, packets: &[[u8; 64]]) -> Result<u64> {
        let n = packets.len() as u64;
        let size = u64::from(self.size());
        if n == 0 || n > size {
            return Err(Error::Invalid(format!(
                "a packet array of {n} does not fit a {size}-packet queue"
            )));
        }
        // SAFETY: a live queue (the caller's contract covers the packets).
        unsafe {
            let first = (self.hsa.f.queue_add_write_index)(self.raw, n);
            // Back-pressure: wait until the ring has room for the whole array.
            while first + n - (self.hsa.f.queue_load_read_index)(self.raw) > size {
                std::hint::spin_loop();
            }
            let base = (*self.raw).base_address.cast::<u8>();
            for (i, p) in packets.iter().enumerate() {
                let slot = base.add((((first + i as u64) % size) * 64) as usize);
                std::ptr::copy_nonoverlapping(p.as_ptr().add(4), slot.add(4), 60);
                let word = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
                (*slot.cast::<AtomicU32>()).store(word, Ordering::Release);
            }
            (self.hsa.f.signal_store_screlease)(
                (*self.raw).doorbell_signal,
                (first + n - 1) as i64,
            );
            Ok(first + n - 1)
        }
    }
}

impl Drop for Queue<'_> {
    fn drop(&mut self) {
        // SAFETY: nothing in flight (the caller waited on its completion signals).
        let _ = unsafe { (self.hsa.f.queue_destroy)(self.raw) };
    }
}

/// A packet array's raw bytes as `u32` words for an invalid packet (what a ring slot holds when
/// free). Exposed for tests of the header law.
#[must_use]
pub fn invalid_header() -> u16 {
    header(PACKET_TYPE_INVALID, false, Scope::None, Scope::None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_bits_follow_hsa_h() {
        // type 2 | barrier bit 8 | acquire agent (1 << 9) | release system (2 << 11)
        assert_eq!(
            header(
                PACKET_TYPE_KERNEL_DISPATCH,
                true,
                Scope::Agent,
                Scope::System
            ),
            2 | (1 << 8) | (1 << 9) | (2 << 11)
        );
        assert_eq!(
            header(PACKET_TYPE_BARRIER_AND, false, Scope::None, Scope::None),
            3
        );
        assert_eq!(invalid_header(), 1);
    }

    #[test]
    fn a_dispatch_packet_carries_work_items_not_blocks() {
        let p = Dispatch {
            header: 0x1234,
            block: [256, 1, 1],
            grid_blocks: [10, 2, 1],
            private_bytes: 16,
            group_bytes: 4096,
            kernel_object: 0xdead_beef,
            kernarg: 0x1000,
            completion: 7,
        }
        .bytes();
        assert_eq!(u16::from_le_bytes([p[0], p[1]]), 0x1234);
        assert_eq!(u16::from_le_bytes([p[2], p[3]]), 3);
        assert_eq!(u16::from_le_bytes([p[4], p[5]]), 256);
        assert_eq!(u32::from_le_bytes([p[12], p[13], p[14], p[15]]), 2560);
        assert_eq!(u32::from_le_bytes([p[16], p[17], p[18], p[19]]), 2);
        assert_eq!(u32::from_le_bytes([p[28], p[29], p[30], p[31]]), 4096);
        assert_eq!(
            u64::from_le_bytes(p[32..40].try_into().unwrap()),
            0xdead_beef
        );
        assert_eq!(u64::from_le_bytes(p[56..64].try_into().unwrap()), 7);
    }

    #[test]
    fn a_barrier_and_packet_takes_at_most_five_dependencies() {
        let deps: Vec<Signal> = (1..=6).map(Signal).collect();
        let p = barrier_and(3, &deps, 9);
        for i in 0..5 {
            assert_eq!(
                u64::from_le_bytes(p[8 + 8 * i..16 + 8 * i].try_into().unwrap()),
                (i + 1) as u64
            );
        }
        assert_eq!(u64::from_le_bytes(p[48..56].try_into().unwrap()), 0);
        assert_eq!(u64::from_le_bytes(p[56..64].try_into().unwrap()), 9);
    }

    #[test]
    fn opening_a_nonexistent_runtime_is_not_available_not_a_panic() {
        assert!(matches!(
            Hsa::open("libsaber-no-such-hsa.so.0"),
            Err(Error::NotAvailable(_))
        ));
    }
}
