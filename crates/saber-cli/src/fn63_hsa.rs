//! FN63 -- an HSA-direct replay of ONE captured step beside its HIP graph (a PROTOTYPE).
//!
//! `saber step-capture --fn63 DIR` records the eager step's dispatches (plugin, kernel, grid,
//! block, the exact kernarg image `hipModuleLaunchKernel` got), loads the SAME committed code
//! objects a second time through ROCr (`hsa_executable_load_agent_code_object`, never rebuilt),
//! fills each kernel's hidden arguments from its own code-object metadata, and replays the step
//! as a pre-baked AQL packet ARRAY -- N packets written, one doorbell -- over the SAME buffers:
//!
//! * `hip-graph`  the control: the captured hipGraph, `hipGraphLaunch` + `hipStreamSynchronize`;
//! * `hsa-b`      every dispatch with the BARRIER bit (in order, = the graph's chain), kernargs in
//!   VRAM, the host waits ACTIVE (busy-poll);
//! * `hsa-b-sys`  the same, kernargs in the CPU agent's kernarg pool (fine-grained system memory);
//! * `hsa-c`      barrier-AND packets only where the dataflow needs them (a node depends on an
//!   earlier node when their buffer extents overlap; adjacent = the barrier bit);
//! * `hsa-b-blk`  `hsa-b`, the host waits BLOCKED (interrupt): the wake latency arm;
//! * `hsa-busy`   (untimed) one completion signal per dispatch, for the summed kernel busy time.
//!
//! Every arm is byte-compared (3 replays from a reset state) against the eager step's whole arena
//! and state, with a negative control (`hsa-b` minus one dispatch) that must MISMATCH. Per-thread
//! CPU is sampled from `/proc/self/task/*/schedstat` per block of reps. The SDMA copy batch (45 x
//! 2.6 MB, FN59's shape) runs through `hsa_amd_memory_async_copy` beside `hipMemcpyAsync`.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context};
use saber_exec::DispatchRecord;
use saber_hip::{EventFlags, HipThread, Job, MemcpyKind, Stream, StreamFlags};
use saber_hsa::{
    barrier_and, header, Agent, Dispatch, Hsa, Queue, Scope, Signal, WaitState,
    PACKET_TYPE_BARRIER_AND, PACKET_TYPE_KERNEL_DISPATCH,
};
use saber_kernels::codeobj::{amdgcn_elf, CodeObject, KernelMeta};

/// Options from the command line.
pub(crate) struct Fn63Opts {
    /// Results directory (`~/saber-cache/ws-fn63/...`).
    pub out: PathBuf,
    /// Timed reps per arm.
    pub reps: usize,
    /// Reps per interleaved block.
    pub block: usize,
    /// ROCr soname or path.
    pub hsa_lib: String,
}

/// What the step-capture hands over.
pub(crate) struct Fn63Input<'a> {
    pub hip: &'a HipThread,
    pub stream: &'a Stream,
    pub device: i32,
    pub prebuilt: &'a Path,
    pub records: Vec<DispatchRecord>,
    pub graph_nodes: usize,
    pub eager_arena: &'a [u8],
    pub eager_states: &'a [Vec<u8>],
    /// Known buffer extents `(address, bytes)` (arena buffers, state and input buffers).
    pub extents: Vec<(u64, u64)>,
}

/// The step's whole arena and every state buffer.
pub(crate) type Snapshot = (Vec<u8>, Vec<Vec<u8>>);

/// `state(true)` resets the step's state; `state(false)` snapshots it.
pub(crate) type StateFn<'a> = dyn FnMut(bool) -> anyhow::Result<Option<Snapshot>> + 'a;

// ---- hidden arguments ------------------------------------------------------------------------ //

/// Hidden argument kinds this prototype fills (the rest are refused by name).
const HIDDEN_FILLED: &[&str] = &[
    "hidden_block_count_x",
    "hidden_block_count_y",
    "hidden_block_count_z",
    "hidden_group_size_x",
    "hidden_group_size_y",
    "hidden_group_size_z",
    "hidden_remainder_x",
    "hidden_remainder_y",
    "hidden_remainder_z",
    "hidden_global_offset_x",
    "hidden_global_offset_y",
    "hidden_global_offset_z",
    "hidden_grid_dims",
    "hidden_dynamic_lds_size",
    // Zero is what HIP passes for a non-cooperative, non-device-enqueue launch.
    "hidden_none",
    "hidden_multigrid_sync_arg",
    "hidden_default_queue",
    "hidden_completion_action",
];

/// The full kernarg segment of one dispatch: the explicit image, then the hidden block filled
/// from the code object's own metadata, the way ROCclr fills it for `hipModuleLaunchKernel`.
pub(crate) fn full_kernarg(
    meta: &KernelMeta,
    explicit: &[u8],
    grid: [u32; 3],
    block: [u32; 3],
    dyn_lds: u32,
) -> anyhow::Result<Vec<u8>> {
    let mut img = vec![0u8; meta.kernarg_segment_size as usize];
    if explicit.len() > img.len() {
        bail!(
            "{}: explicit image {} B > segment {} B",
            meta.name,
            explicit.len(),
            img.len()
        );
    }
    img[..explicit.len()].copy_from_slice(explicit);
    for a in meta
        .args
        .iter()
        .filter(|a| a.value_kind.starts_with("hidden_"))
    {
        let at = a.offset as usize;
        let put = |img: &mut Vec<u8>, b: &[u8]| -> anyhow::Result<()> {
            if b.len() != a.size as usize || at + b.len() > img.len() {
                bail!("{}: {} size {} at {at}", meta.name, a.value_kind, a.size);
            }
            img[at..at + b.len()].copy_from_slice(b);
            Ok(())
        };
        let d = |s: &str| match s.as_bytes().last() {
            Some(b'x') => 0usize,
            Some(b'y') => 1,
            _ => 2,
        };
        match a.value_kind.as_str() {
            k if k.starts_with("hidden_block_count_") => put(&mut img, &grid[d(k)].to_le_bytes())?,
            k if k.starts_with("hidden_group_size_") => {
                put(&mut img, &u16::try_from(block[d(k)])?.to_le_bytes())?;
            }
            k if k.starts_with("hidden_remainder_") || k.starts_with("hidden_global_offset_") => {}
            "hidden_grid_dims" => put(&mut img, &3u16.to_le_bytes())?,
            "hidden_dynamic_lds_size" => put(&mut img, &dyn_lds.to_le_bytes())?,
            k if HIDDEN_FILLED.contains(&k) => {}
            k => bail!(
                "{}: hidden argument `{k}` has no HSA-path value (HIP fills it from the device)",
                meta.name
            ),
        }
    }
    Ok(img)
}

// ---- dependencies ---------------------------------------------------------------------------- //

/// Per node, the extents its pointer arguments name: a pointer inside a known buffer covers from
/// itself to the end of the furthest known buffer containing it; an unknown pointer (a weight)
/// covers one byte (equality).
fn node_extents(rec: &DispatchRecord, known: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    for &off in &rec.pointer_slots {
        let o = off as usize;
        let Some(b) = rec.kernarg.get(o..o + 8) else {
            continue;
        };
        let p = u64::from_le_bytes(b.try_into().unwrap_or([0; 8]));
        if p == 0 {
            continue;
        }
        let end = known
            .iter()
            .filter(|(a, n)| *a <= p && p < a + n)
            .map(|(a, n)| a + n)
            .max()
            .unwrap_or(p + 1);
        out.push((p, end));
    }
    out
}

/// For node j, the earlier nodes it must wait for, reduced (no dep implied by another), and
/// whether the barrier bit expresses it (the reduced set holds j-1 -> barrier bit).
pub(crate) struct DepPlan {
    /// Per node: `None` = barrier bit; `Some(v)` = barrier-AND over `v` (empty = free).
    pub wait: Vec<Option<Vec<usize>>>,
}

pub(crate) fn plan_deps(ext: &[Vec<(u64, u64)>]) -> DepPlan {
    let n = ext.len();
    let words = n.div_ceil(64);
    let mut anc: Vec<Vec<u64>> = vec![vec![0u64; words]; n];
    let set = |v: &mut Vec<u64>, i: usize| v[i / 64] |= 1 << (i % 64);
    let has = |v: &Vec<u64>, i: usize| v[i / 64] >> (i % 64) & 1 == 1;
    let mut wait = Vec::with_capacity(n);
    for j in 0..n {
        let mut d: Vec<usize> = (0..j)
            .filter(|&i| {
                ext[i]
                    .iter()
                    .any(|(a, b)| ext[j].iter().any(|(c, e)| a < e && c < b))
            })
            .collect();
        // keep only the maximal ones
        let dd = d.clone();
        d.retain(|&x| !dd.iter().any(|&y| y != x && has(&anc[y], x)));
        let last = j + 1 == n;
        if last || (j > 0 && d.contains(&(j - 1))) {
            for i in 0..j {
                set(&mut anc[j], i);
            }
            wait.push(None);
        } else {
            let mut a = vec![0u64; words];
            for &x in &d {
                set(&mut a, x);
                for w in 0..words {
                    a[w] |= anc[x][w];
                }
            }
            anc[j] = a;
            wait.push(Some(d));
        }
    }
    DepPlan { wait }
}

// ---- CPU per thread -------------------------------------------------------------------------- //

fn my_tid() -> i32 {
    std::fs::read_link("/proc/thread-self")
        .ok()
        .and_then(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .and_then(|s| s.parse().ok())
        })
        .unwrap_or(-1)
}

/// `tid -> (comm, ns on cpu)`.
fn thread_cpu() -> BTreeMap<i32, (String, u64)> {
    let mut out = BTreeMap::new();
    let Ok(rd) = std::fs::read_dir("/proc/self/task") else {
        return out;
    };
    for e in rd.flatten() {
        let Some(tid) = e.file_name().to_str().and_then(|s| s.parse::<i32>().ok()) else {
            continue;
        };
        let p = e.path();
        let comm = std::fs::read_to_string(p.join("comm"))
            .unwrap_or_default()
            .trim()
            .to_string();
        let ns = std::fs::read_to_string(p.join("schedstat"))
            .ok()
            .and_then(|s| s.split_whitespace().next().and_then(|x| x.parse().ok()))
            .unwrap_or(0);
        out.insert(tid, (comm, ns));
    }
    out
}

// ---- arms -------------------------------------------------------------------------------------- //

struct Arm {
    name: &'static str,
    packets: Vec<[u8; 64]>,
    /// Signals set back to 1 before every submit (the first and done signals included).
    resets: Vec<Signal>,
    first: Signal,
    done: Signal,
    wait: WaitState,
}

#[derive(Default)]
struct Samples {
    host_us: Vec<f64>,
    gpu_us: Vec<f64>,
    wake_us: Vec<f64>,
    issue_us: Vec<f64>,
    cpu: BTreeMap<i32, (String, u64)>,
    reps: usize,
}

fn pct(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() as f64 - 1.0) * p).round() as usize]
}

struct Node {
    obj: u64,
    grid: [u32; 3],
    block: [u32; 3],
    group: u32,
    private: u32,
    image: Vec<u8>,
}

#[allow(clippy::too_many_lines)]
pub(crate) fn run(
    inp: Fn63Input<'_>,
    opts: &Fn63Opts,
    state: &mut StateFn<'_>,
    replay: &mut dyn FnMut() -> anyhow::Result<()>,
) -> anyhow::Result<usize> {
    let hip = inp.hip;
    let mut failures = 0usize;
    let mut log = String::new();
    macro_rules! out {
        ($($t:tt)*) => {{ let s = format!($($t)*); println!("{s}"); let _ = writeln!(log, "{s}"); }};
    }
    std::fs::create_dir_all(&opts.out).ok();
    let n = inp.records.len();
    out!(
        "fn63 records {n} dispatches; graph nodes {}",
        inp.graph_nodes
    );
    if n != inp.graph_nodes {
        out!("fn63 MISMATCH: the graph has non-kernel nodes or the recorder missed launches");
        failures += 1;
    }

    // ---- ROCr in the same process --------------------------------------------------------------
    let hsa = Hsa::open(&opts.hsa_lib)?;
    let bus = hip.device_pci_bus_id(inp.device)?;
    let bdf = parse_bdf(&bus).with_context(|| format!("pci bus id {bus}"))?;
    let agents = hsa.agents()?;
    let gpu_info = agents
        .iter()
        .find(|a| a.gpu && a.bdfid == bdf)
        .with_context(|| format!("no HSA GPU agent at {bus} (bdfid {bdf:#x})"))?
        .clone();
    let cpu = agents
        .iter()
        .find(|a| !a.gpu)
        .context("no CPU agent")?
        .agent;
    let gpu = gpu_info.agent;
    let hz = hsa.timestamp_hz()? as f64;
    out!(
        "fn63 hsa agent {} bdfid {bdf:#x} (HIP {bus}), profile {}, timestamp {hz:.0} Hz, {} agents",
        gpu_info.name,
        gpu_info.profile,
        agents.len()
    );

    // ---- the same code objects, loaded by ROCr ----------------------------------------------
    let mut metas: BTreeMap<String, CodeObject> = BTreeMap::new();
    let mut images: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for r in &inp.records {
        if !images.contains_key(&r.plugin) {
            let path = inp.prebuilt.join(format!("{}.hsaco", r.plugin));
            let bytes = std::fs::read(&path).with_context(|| path.display().to_string())?;
            metas.insert(r.plugin.clone(), CodeObject::parse(&bytes)?);
            images.insert(r.plugin.clone(), bytes);
        }
    }
    let mut execs = BTreeMap::new();
    for (plugin, bytes) in &images {
        let elf = amdgcn_elf(bytes)?;
        execs.insert(
            plugin.clone(),
            hsa.load_executable(gpu, gpu_info.profile, elf)?,
        );
    }
    out!(
        "fn63 loaded {} code objects through hsa_executable_load_agent_code_object",
        execs.len()
    );

    let mut nodes = Vec::with_capacity(n);
    let mut hidden_kinds: BTreeMap<String, usize> = BTreeMap::new();
    for r in &inp.records {
        let meta = metas[&r.plugin]
            .kernel(&r.kernel)
            .with_context(|| format!("{} not in {}", r.kernel, r.plugin))?;
        for a in meta
            .args
            .iter()
            .filter(|a| a.value_kind.starts_with("hidden_"))
        {
            *hidden_kinds.entry(a.value_kind.clone()).or_default() += 1;
        }
        let sym = execs[&r.plugin].kernel(&meta.symbol)?;
        if sym.kernarg_bytes != meta.kernarg_segment_size {
            bail!(
                "{}: ROCr says {} kernarg bytes, the metadata {}",
                r.kernel,
                sym.kernarg_bytes,
                meta.kernarg_segment_size
            );
        }
        nodes.push(Node {
            obj: sym.object,
            grid: r.grid,
            block: r.block,
            group: sym.group_bytes + r.shared_bytes,
            private: sym.private_bytes,
            image: full_kernarg(meta, &r.kernarg, r.grid, r.block, r.shared_bytes)?,
        });
    }
    out!("fn63 hidden args filled: {hidden_kinds:?}");

    // ---- kernarg segments: VRAM (written once through ROCr) and the system kernarg pool ------
    let stride: Vec<usize> = nodes
        .iter()
        .map(|x| x.image.len().max(1).div_ceil(64) * 64)
        .collect();
    let total: usize = stride.iter().sum();
    let ka_sys = hsa.alloc_kernarg_sys(cpu, gpu, total)?;
    let ka_vram = hsa.alloc_vram(gpu, total)?;
    let mut off = Vec::with_capacity(n);
    let mut o = 0usize;
    for (x, s) in nodes.iter().zip(&stride) {
        // SAFETY: `ka_sys` holds `total` host-writable bytes; each image fits its stride.
        unsafe { std::ptr::copy_nonoverlapping(x.image.as_ptr(), ka_sys.add(o), x.image.len()) };
        off.push(o);
        o += s;
    }
    // SAFETY: both allocations hold `total` bytes.
    unsafe { hsa.copy(ka_vram, ka_sys, total)? };
    out!(
        "fn63 kernarg segments {total} B ({} B/node avg), sys + vram",
        total / n.max(1)
    );

    // ---- signals -----------------------------------------------------------------------------
    let gpu_only = [gpu];
    let mk = |consumers: &[Agent]| hsa.signal(1, consumers);
    let first = mk(&gpu_only)?;
    let done_spin = mk(&[])?;
    let done_blk = mk(&[])?;
    let dispatch = |j: usize, barrier: bool, ka: *mut u8, completion: u64| -> [u8; 64] {
        let acq = if j == 0 { Scope::System } else { Scope::Agent };
        let rel = if j + 1 == n {
            Scope::System
        } else {
            Scope::Agent
        };
        Dispatch {
            header: header(PACKET_TYPE_KERNEL_DISPATCH, barrier, acq, rel),
            block: nodes[j].block,
            grid_blocks: nodes[j].grid,
            private_bytes: nodes[j].private,
            group_bytes: nodes[j].group,
            kernel_object: nodes[j].obj,
            // SAFETY: inside the allocation.
            kernarg: unsafe { ka.add(off[j]) } as u64,
            completion,
        }
        .bytes()
    };
    let in_order = |ka: *mut u8, done: Signal, skip: Option<usize>| -> Vec<[u8; 64]> {
        (0..n)
            .filter(|j| Some(*j) != skip)
            .map(|j| {
                let c = if j == 0 {
                    first.0
                } else if j + 1 == n {
                    done.0
                } else {
                    0
                };
                dispatch(j, true, ka, c)
            })
            .collect()
    };

    // dataflow arm
    let mut known = inp.extents.clone();
    known.sort_unstable();
    let ext: Vec<Vec<(u64, u64)>> = inp
        .records
        .iter()
        .map(|r| node_extents(r, &known))
        .collect();
    let deps = plan_deps(&ext);
    let mut sig_of: BTreeMap<usize, Signal> = BTreeMap::new();
    sig_of.insert(0, first);
    for w in deps.wait.iter().flatten() {
        for &d in w {
            if let std::collections::btree_map::Entry::Vacant(e) = sig_of.entry(d) {
                e.insert(mk(&gpu_only)?);
            }
        }
    }
    let (mut n_bar, mut n_and, mut n_free) = (0usize, 0usize, 0usize);
    let mut c_packets = Vec::new();
    for (j, w) in deps.wait.iter().enumerate() {
        let c = if j + 1 == n {
            done_spin.0
        } else {
            sig_of.get(&j).map_or(0, |s| s.0)
        };
        match w {
            None => {
                n_bar += 1;
                c_packets.push(dispatch(j, true, ka_vram, c));
            }
            Some(v) if v.is_empty() => {
                n_free += 1;
                c_packets.push(dispatch(j, false, ka_vram, c));
            }
            Some(v) => {
                for chunk in v.chunks(5) {
                    let s: Vec<Signal> = chunk.iter().map(|d| sig_of[d]).collect();
                    c_packets.push(barrier_and(
                        header(PACKET_TYPE_BARRIER_AND, false, Scope::None, Scope::None),
                        &s,
                        0,
                    ));
                    n_and += 1;
                }
                c_packets.push(dispatch(j, false, ka_vram, c));
            }
        }
    }
    out!(
        "fn63 dataflow (hsa-c): {n_bar} barrier-bit nodes, {n_free} free nodes, {n_and} barrier-AND \
         packets, {} dependency signals; {} packets",
        sig_of.len(),
        c_packets.len()
    );
    let c_resets: Vec<Signal> = sig_of.values().copied().chain([done_spin]).collect();

    // busy arm: one profiled signal per dispatch
    let mut busy_sig = Vec::with_capacity(n);
    for j in 0..n {
        busy_sig.push(if j == 0 {
            first
        } else if j + 1 == n {
            done_spin
        } else {
            mk(&gpu_only)?
        });
    }
    let busy_packets: Vec<[u8; 64]> = (0..n)
        .map(|j| dispatch(j, true, ka_vram, busy_sig[j].0))
        .collect();

    let arms = [
        Arm {
            name: "hsa-b",
            packets: in_order(ka_vram, done_spin, None),
            resets: vec![first, done_spin],
            first,
            done: done_spin,
            wait: WaitState::Active,
        },
        Arm {
            name: "hsa-b-sys",
            packets: in_order(ka_sys, done_spin, None),
            resets: vec![first, done_spin],
            first,
            done: done_spin,
            wait: WaitState::Active,
        },
        Arm {
            name: "hsa-c",
            packets: c_packets,
            resets: c_resets,
            first,
            done: done_spin,
            wait: WaitState::Active,
        },
        Arm {
            name: "hsa-b-blk",
            packets: in_order(ka_vram, done_blk, None),
            resets: vec![first, done_blk],
            first,
            done: done_blk,
            wait: WaitState::Blocked,
        },
    ];
    let neg = Arm {
        name: "neg-control",
        packets: in_order(ka_vram, done_spin, Some(n / 2)),
        resets: vec![first, done_spin],
        first,
        done: done_spin,
        wait: WaitState::Active,
    };
    let busy = Arm {
        name: "hsa-busy",
        packets: busy_packets,
        resets: busy_sig.clone(),
        first,
        done: done_spin,
        wait: WaitState::Active,
    };

    let queue = hsa.queue(gpu, 4096, true)?;
    let timeout = (hz * 10.0) as u64;
    let submit = |q: &Queue<'_>, a: &Arm| -> anyhow::Result<(f64, f64, u64)> {
        for s in &a.resets {
            hsa.signal_store(*s, 1);
        }
        let t0 = Instant::now();
        // SAFETY: every kernarg, kernel object and signal outlives the wait below.
        unsafe { q.submit(&a.packets)? };
        let issue = t0.elapsed().as_secs_f64() * 1e6;
        let v = hsa.wait_done(a.done, a.wait, timeout);
        let wake_ts = hsa.timestamp()?;
        let host = t0.elapsed().as_secs_f64() * 1e6;
        if v >= 1 {
            bail!("{}: the step did not complete in 10 s (signal {v})", a.name);
        }
        Ok((issue, host, wake_ts))
    };

    // ---- byte identity: every arm vs the eager step ------------------------------------------
    let compare = |snap: &(Vec<u8>, Vec<Vec<u8>>)| -> (usize, usize) {
        let a = snap
            .0
            .iter()
            .zip(inp.eager_arena)
            .filter(|(x, y)| x != y)
            .count();
        let s = snap
            .1
            .iter()
            .zip(inp.eager_states)
            .map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p != q).count())
            .sum();
        (a, s)
    };
    for a in arms.iter().chain([&neg, &busy]) {
        for r in 1..=3 {
            state(true)?;
            submit(&queue, a)?;
            let snap = state(false)?.context("snapshot")?;
            let (ab, sb) = compare(&snap);
            let exact = ab == 0 && sb == 0;
            let want_exact = a.name != "neg-control";
            out!(
                "fn63 bytes {:<11} replay {r}: arena {ab} of {} differ, state {sb} -- {}",
                a.name,
                snap.0.len(),
                match (exact, want_exact) {
                    (true, true) => "BIT-EXACT",
                    (false, false) => "MISMATCH (the control must)",
                    (true, false) => "CONTROL FAILED TO DIFFER",
                    (false, true) => "MISMATCH",
                }
            );
            if exact != want_exact {
                failures += 1;
            }
        }
    }

    // ---- kernel busy: one profiled signal per dispatch ---------------------------------------
    let mut busy_sum = Vec::new();
    let mut span_busy = Vec::new();
    let mut gaps: Vec<f64> = Vec::new();
    for _ in 0..30 {
        submit(&queue, &busy)?;
        let t: Vec<saber_hsa::Times> = busy_sig
            .iter()
            .map(|s| hsa.dispatch_time(gpu, *s))
            .collect::<Result<_, _>>()?;
        busy_sum.push(t.iter().map(|x| (x.end - x.start) as f64).sum::<f64>() / hz * 1e6);
        span_busy.push((t[n - 1].end - t[0].start) as f64 / hz * 1e6);
        for w in t.windows(2) {
            gaps.push((w[1].start as f64 - w[0].end as f64) / hz * 1e6);
        }
    }
    let busy_us = pct(&busy_sum, 0.5);
    out!(
        "fn63 busy (per-dispatch signals, barrier bit): kernels sum p50 {busy_us:.1} us, span p50 \
         {:.1} us, gap end->start p50 {:.2} p90 {:.2} p99 {:.2} us",
        pct(&span_busy, 0.5),
        pct(&gaps, 0.5),
        pct(&gaps, 0.9),
        pct(&gaps, 0.99)
    );

    // ---- timing: interleaved blocks ------------------------------------------------------------
    let ev0 = hip.create_event(EventFlags::Default)?;
    let ev1 = hip.create_event(EventFlags::Default)?;
    let hip_tid = hip.tid();
    let me = my_tid();
    let names = ["hip-graph", "hsa-b", "hsa-b-sys", "hsa-c", "hsa-b-blk"];
    let mut samples: Vec<Samples> = (0..names.len()).map(|_| Samples::default()).collect();
    // warm every arm
    for _ in 0..5 {
        replay()?;
        hip.sync_stream(inp.stream)?;
        for a in &arms {
            submit(&queue, a)?;
        }
    }
    let rounds = opts.reps.div_ceil(opts.block.max(1));
    for round in 0..rounds {
        for k in 0..names.len() {
            let idx = (k + round) % names.len();
            let before = thread_cpu();
            for _ in 0..opts.block {
                let s = &mut samples[idx];
                if idx == 0 {
                    hip.record_event(&ev0, inp.stream)?;
                    let t0 = Instant::now();
                    replay()?;
                    let issue = t0.elapsed().as_secs_f64() * 1e6;
                    hip.record_event(&ev1, inp.stream)?;
                    hip.sync_stream(inp.stream)?;
                    s.host_us.push(t0.elapsed().as_secs_f64() * 1e6);
                    s.issue_us.push(issue);
                    s.gpu_us.push(f64::from(hip.elapsed_ms(&ev0, &ev1)?) * 1e3);
                } else {
                    let a = &arms[idx - 1];
                    let (issue, host, wake) = submit(&queue, a)?;
                    let t_first = hsa.dispatch_time(gpu, a.first)?;
                    let t_done = hsa.dispatch_time(gpu, a.done)?;
                    s.host_us.push(host);
                    s.issue_us.push(issue);
                    s.gpu_us
                        .push((t_done.end - t_first.start) as f64 / hz * 1e6);
                    s.wake_us.push((wake as f64 - t_done.end as f64) / hz * 1e6);
                }
                s.reps += 1;
            }
            let after = thread_cpu();
            for (tid, (comm, ns)) in after {
                let was = before.get(&tid).map_or(0, |x| x.1);
                let e = samples[idx].cpu.entry(tid).or_insert((comm, 0));
                e.1 += ns.saturating_sub(was);
            }
        }
    }
    out!("");
    out!(
        "fn63 timing ({} reps/arm, blocks of {}, rotated; card {}, one queue, profiling on for HSA)",
        samples[0].reps,
        opts.block,
        inp.device
    );
    out!("fn63 nodes {n}; kernel busy p50 {busy_us:.1} us (per-dispatch profiled pass)");
    for (i, s) in samples.iter().enumerate() {
        let g = pct(&s.gpu_us, 0.5);
        out!(
            "fn63 arm {:<10} gpu span p50 {:9.1} us p90 {:9.1} | host wall p50 {:9.1} p90 {:9.1} | \
             issue p50 {:7.1} us | per-node overhead (span - busy)/N {:6.3} us | wake p50 {:6.1} \
             p99 {:6.1} us",
            names[i],
            g,
            pct(&s.gpu_us, 0.9),
            pct(&s.host_us, 0.5),
            pct(&s.host_us, 0.9),
            pct(&s.issue_us, 0.5),
            (g - busy_us) / n as f64,
            pct(&s.wake_us, 0.5),
            pct(&s.wake_us, 0.99),
        );
        let reps = s.reps.max(1) as f64;
        let mut rows: Vec<_> = s.cpu.iter().filter(|(_, (_, ns))| *ns > 0).collect();
        rows.sort_by_key(|a| std::cmp::Reverse(a.1 .1));
        let total: u64 = s.cpu.values().map(|x| x.1).sum();
        let mut line = format!(
            "fn63 cpu  {:<10} total {:8.1} us/step:",
            names[i],
            total as f64 / reps / 1e3
        );
        for (tid, (comm, ns)) in rows.iter().take(8) {
            let tag = if **tid == me {
                "*submit"
            } else if **tid == hip_tid {
                "*hipthread"
            } else {
                ""
            };
            let _ = write!(line, " {comm}[{tid}]{tag} {:.1}", *ns as f64 / reps / 1e3);
        }
        out!("{line}");
    }

    // ---- SDMA: 45 x 2.6 MB, H2D / D2H alternating ----------------------------------------------
    match sdma_bench(hip, &hsa, cpu, gpu, hz, 200) {
        Ok(lines) => {
            for l in lines {
                out!("{l}");
            }
        }
        Err(e) => out!("fn63 sdma FAILED: {e:#}"),
    }

    drop(queue);
    for s in sig_of
        .values()
        .chain(busy_sig.iter())
        .chain([&done_spin, &done_blk])
    {
        let _ = hsa.signal_destroy(*s);
    }
    // SAFETY: every queue is drained and destroyed.
    unsafe {
        hsa.free(ka_vram)?;
        hsa.free(ka_sys)?;
    }
    drop(execs);
    out!("fn63 failures {failures}");
    std::fs::write(opts.out.join("fn63.txt"), log)?;
    Ok(failures)
}

/// FN63b: the HSA agent id (`HSA_AMD_AGENT_INFO_BDFID`) of the HIP thread's CURRENT device.
pub(crate) fn hsa_bdfid(hip: &HipThread) -> anyhow::Result<u32> {
    let d = hip.current_device()?;
    let bus = hip.device_pci_bus_id(d)?;
    parse_bdf(&bus).with_context(|| format!("pci bus id {bus}"))
}

pub(crate) fn parse_bdf(bus: &str) -> Option<u32> {
    // "0000:03:00.0" -> (bus << 8) | (dev << 3) | func
    let mut it = bus.rsplit(':');
    let devfn = it.next()?;
    let b = u32::from_str_radix(it.next()?, 16).ok()?;
    let (d, f) = devfn.split_once('.')?;
    let d = u32::from_str_radix(d, 16).ok()?;
    let f = u32::from_str_radix(f, 16).ok()?;
    Some((b << 8) | (d << 3) | f)
}

const COPIES: usize = 45;
const STRIDE: u64 = 2_600_960;

#[allow(clippy::too_many_lines)]
fn sdma_bench(
    hip: &HipThread,
    hsa: &Hsa,
    cpu: Agent,
    gpu: Agent,
    hz: f64,
    iters: usize,
) -> anyhow::Result<Vec<String>> {
    let mut lines = Vec::new();
    let dev = hip.malloc(2 * COPIES as u64 * STRIDE)?;
    let host = hip.host_malloc(2 * COPIES as u64 * STRIDE)?;
    let copy = hip.create_stream(StreamFlags::NonBlocking)?;
    let mut jobs: Vec<Job<'_>> = Vec::new();
    for k in 0..COPIES {
        jobs.push(Job::MemcpyPinned {
            dev: &dev,
            dev_offset: k as u64 * STRIDE,
            host: &host,
            host_offset: (COPIES + k) as u64 * STRIDE,
            bytes: STRIDE,
            kind: if k % 2 == 0 {
                MemcpyKind::HostToDevice
            } else {
                MemcpyKind::DeviceToHost
            },
            stream: &copy,
        });
    }
    let sigs: Vec<Signal> = (0..COPIES)
        .map(|_| hsa.signal(1, &[]))
        .collect::<Result<_, _>>()?;
    hsa.copy_profiling(true)?;
    let d0 = dev.addr().as_u64();
    let h0 = host.addr() as u64;
    let me = my_tid();
    let hip_tid = hip.tid();
    let mut res: [(Vec<f64>, Vec<f64>, u64, u64); 3] = Default::default();
    let names = ["hipMemcpyAsync", "hsa chained", "hsa unchained"];
    for i in 0..(iters + 10) {
        for arm in 0..3 {
            let arm = (arm + i) % 3;
            let before = thread_cpu();
            let t0 = Instant::now();
            let issue;
            if arm == 0 {
                hip.run_batch(&jobs)?;
                issue = t0.elapsed();
                hip.sync_stream(&copy)?;
            } else {
                for s in &sigs {
                    hsa.signal_store(*s, 1);
                }
                for k in 0..COPIES {
                    let dv = d0 + k as u64 * STRIDE;
                    let hv = h0 + (COPIES + k) as u64 * STRIDE;
                    let deps: &[Signal] = if arm == 1 && k > 0 {
                        &sigs[k - 1..k]
                    } else {
                        &[]
                    };
                    // SAFETY: both ranges are live allocations of `STRIDE` bytes at these
                    // offsets; the waits below drain every copy before the buffers drop.
                    unsafe {
                        if k % 2 == 0 {
                            hsa.async_copy(dv, gpu, hv, cpu, STRIDE as usize, deps, sigs[k])?;
                        } else {
                            hsa.async_copy(hv, cpu, dv, gpu, STRIDE as usize, deps, sigs[k])?;
                        }
                    }
                }
                issue = t0.elapsed();
                for s in &sigs {
                    if hsa.wait_done(*s, WaitState::Active, (hz * 10.0) as u64) >= 1 {
                        bail!("an SDMA copy did not land in 10 s");
                    }
                }
            }
            let land = t0.elapsed();
            let after = thread_cpu();
            if i >= 10 {
                res[arm].0.push(issue.as_secs_f64() * 1e6);
                res[arm].1.push(land.as_secs_f64() * 1e6);
                for (tid, (_, ns)) in &after {
                    let d = ns.saturating_sub(before.get(tid).map_or(0, |x| x.1));
                    if *tid == me || *tid == hip_tid {
                        res[arm].2 += d;
                    } else {
                        res[arm].3 += d;
                    }
                }
            }
        }
    }
    let t = hsa.copy_time(sigs[0]).ok();
    for (k, r) in res.iter().enumerate() {
        lines.push(format!(
            "fn63 sdma {:<15} 45 x {STRIDE} B: issue p50 {:8.1} p99 {:8.1} us | landed p50 {:8.1} \
             p99 {:8.1} us | cpu caller {:.1} other {:.1} us/batch",
            names[k],
            pct(&r.0, 0.5),
            pct(&r.0, 0.99),
            pct(&r.1, 0.5),
            pct(&r.1, 0.99),
            r.2 as f64 / iters as f64 / 1e3,
            r.3 as f64 / iters as f64 / 1e3,
        ));
    }
    if let Some(t) = t {
        lines.push(format!(
            "fn63 sdma one hsa copy (profiled): {:.1} us for {STRIDE} B",
            (t.end - t.start) as f64 / hz * 1e6
        ));
    }
    hsa.copy_profiling(false)?;
    for s in sigs {
        let _ = hsa.signal_destroy(s);
    }
    drop(jobs);
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(ptrs: &[u64]) -> DispatchRecord {
        let mut k = Vec::new();
        let mut slots = Vec::new();
        for (i, p) in ptrs.iter().enumerate() {
            slots.push((i * 8) as u32);
            k.extend_from_slice(&p.to_le_bytes());
        }
        DispatchRecord {
            plugin: String::new(),
            kernel: String::new(),
            op: String::new(),
            grid: [1; 3],
            block: [1; 3],
            shared_bytes: 0,
            kernarg: k,
            pointer_slots: slots,
        }
    }

    #[test]
    fn the_dataflow_plan_uses_extents_not_pointer_equality() {
        // buffer A [0x1000, 0x2000) holds a view at 0x1800; node 1 writes the view, node 2 reads A
        let known = vec![(0x1000, 0x1000), (0x5000, 0x100)];
        let recs = [
            rec(&[0x1000]),
            rec(&[0x5000]),
            rec(&[0x1800]),
            rec(&[0x9000]),
        ];
        let ext: Vec<_> = recs.iter().map(|r| node_extents(r, &known)).collect();
        let p = plan_deps(&ext);
        assert_eq!(p.wait[0], Some(vec![]));
        assert_eq!(p.wait[1], Some(vec![]), "disjoint buffers: free");
        assert_eq!(
            p.wait[2],
            Some(vec![0]),
            "a view inside A depends on A's writer"
        );
        assert!(
            p.wait[3].is_none(),
            "the last node always takes the barrier bit"
        );
    }

    #[test]
    fn an_adjacent_dependency_is_the_barrier_bit_and_implied_deps_are_dropped() {
        let known = vec![(0x1000, 0x100), (0x2000, 0x100)];
        let recs = [
            rec(&[0x1000]),
            rec(&[0x1000, 0x2000]),
            rec(&[0x2000, 0x1000]),
            rec(&[0x7]),
        ];
        let ext: Vec<_> = recs.iter().map(|r| node_extents(r, &known)).collect();
        let p = plan_deps(&ext);
        assert!(p.wait[1].is_none());
        assert!(p.wait[2].is_none());
    }

    #[test]
    fn bdf_parses_the_hip_pci_bus_id() {
        assert_eq!(parse_bdf("0000:03:00.0"), Some(0x300));
        assert_eq!(parse_bdf("0000:c3:00.1"), Some(0xc301));
    }

    /// The pre-flight for the card: every hidden argument of every committed code object is one
    /// this path knows how to fill (a kernel that needs a hostcall buffer or the queue pointer
    /// would be refused by name at replay; this says so on the host first).
    #[test]
    fn every_committed_kernels_hidden_args_are_fillable() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
        let mut refused = Vec::new();
        for e in std::fs::read_dir(&dir).unwrap().flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("hsaco") {
                continue;
            }
            let co = CodeObject::parse(&std::fs::read(&p).unwrap()).unwrap();
            for k in &co.kernels {
                for a in k
                    .args
                    .iter()
                    .filter(|a| a.value_kind.starts_with("hidden_"))
                {
                    *kinds.entry(a.value_kind.clone()).or_default() += 1;
                }
                if let Err(e) = full_kernarg(k, &[], [1; 3], [64, 1, 1], 0) {
                    refused.push(format!("{}: {e}", p.display()));
                }
            }
        }
        println!("hidden kinds: {kinds:?}");
        assert!(refused.is_empty(), "{refused:#?}");
    }
}
