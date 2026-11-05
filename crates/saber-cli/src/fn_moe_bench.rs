//! `saber flashnext-moe-bench` (FN8): Flash-Next's MoE block over the memory-plane tier, end to
//! end on one card, through the REAL mover (`saber_mem::device_mover`).
//!
//! Per step and per layer (48): FN4's six launches — the bf16 router GEMM (FN5's `moe.router`
//! route; timed on random weights, its logits are not consumed), `moe_route_topk10_sort_e512` over
//! logits synthesized from FN3's replay routes (the routed ids ARE the replay's), grouped gate_up,
//! silu-quant, grouped down, combine — with the grouped GEMMs reading FN4's `xptr` tables filled by
//! `ExpertTier::fill_xptr` (VRAM slot, or the pinned RAM slot over UVA in place, or a staging-ring
//! slot), and `end_step`'s promotions / demotions on the one copy stream, overlapped with the next
//! step. Modes:
//!
//! * `gate`: the byte laws: every routed layer's combine output == the GOLDEN run (every routed
//!   expert read fresh from NVMe into a VRAM ring the mover never touches) at checkpoint
//!   steps, the neighbour-record control must differ; after each run every VRAM and RAM
//!   slot's bytes == its record's sha256 (read back once per run).
//! * `paths`: ms per layer / per step by path: all-VRAM control, the default tier (copies
//!   overlapped), the same with the copy stream drained between steps (the overlap
//!   control), the demand tier (RAM 1,024), the forced-miss first step.
//! * `sweep`: the conc-8 insert policy: `max_inserts` x `read_through`, paired on one trace.
//! * `stage`: prefill chunks: UVA in place vs the staging ring, bytes identical, time per chunk.
//! * `ple`: the PLE staged buffer on the card: gather -> pinned -> H2D on the copy stream,
//!   cold and warm, alone and beside a layer's compute.
//! * `boot`: (no GPU) the boot lines and the metrics exposition the engine will print.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context};
use saber_exec::loader::{FunctionTable, PluginSet};
use saber_exec::LoadCounters;
use saber_hip::{
    DevicePtr, Event, EventFlags, Function, HipThread, Job, MemcpyKind, PinnedBuf, Stream,
    StreamFlags,
};
use saber_kernels::flashnext::FlashNextRules;
use saber_kernels::manifest::KernelDecl;
use saber_kernels::moe::{self, Cell, Geometry, MoeGemm};
use saber_mem::device_mover::{DeviceMover, HipCopyQueue};
use saber_mem::dio::OpenMode;
use saber_mem::expert_mover::parallel_reads;
use saber_mem::expert_tier::{ExpertTier, Placement, RouteProfile, TierParams};
use saber_mem::flashnext_budget::{
    geometry, weights_from_plan, Ep2Budget, HostBudget, RuntimeParams,
};
use saber_mem::xtier::XtierFile;

use crate::xtier_flags::{StageLaw, TierLaw, XtierFlags, NARROW_WIDTH};

const LAYERS: usize = 48;
const LOC: usize = 256;
const XN: usize = 3 * (LOC + 1); // one xptr table (gate_up or down), u64 entries
const XBYTES: u64 = (XN * 8) as u64;
const H: u32 = moe::HIDDEN;

/// The bench's arguments.
#[derive(Debug, Clone)]
pub struct BenchArgs {
    pub mode: String,
    pub pack: PathBuf,
    pub rank: u32,
    pub routes: PathBuf,
    pub profile: PathBuf,
    pub prebuilt: PathBuf,
    pub device: i32,
    pub tokens: Vec<u32>,
    pub steps: usize,
    pub warmup: usize,
    pub ple_ids: PathBuf,
    pub ckpt_root: Option<PathBuf>,
    pub sweep: String,
    pub flags: XtierFlags,
}

// ------------------------------------------------------------------------------------------ routes

/// FN3's replay routes (`fn7_prep.py routes`): per prompt, `[48][T][10]` global expert ids.
fn load_routes(p: &Path) -> anyhow::Result<Vec<(usize, Vec<i32>)>> {
    let raw = std::fs::read(p).with_context(|| format!("routes {}", p.display()))?;
    let u = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().expect("4 bytes")) as usize;
    let mut o = 4;
    let mut v = Vec::new();
    for _ in 0..u(0) {
        let t = u(o);
        o += 4;
        let r: Vec<i32> = raw[o..o + 48 * t * 40]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect();
        o += 48 * t * 40;
        v.push((t, r));
    }
    Ok(v)
}

/// One step: per layer, `tokens × 10` global ids in token order.
#[derive(Debug, Clone)]
struct Step {
    tokens: u32,
    ids: Vec<Vec<u32>>,
}

/// Build a trace. `kind`: `b1` (each prompt token a step), `conc8` (8 prompts interleaved),
/// `conc8x8` (8 prompts × 8 consecutive tokens = 64 rows), `chunk<T>` (the concatenated prompt
/// stream cut into T-token chunks, cycling to fill the last).
fn trace(routes: &[(usize, Vec<i32>)], kind: &str) -> anyhow::Result<Vec<Step>> {
    let tok = |pi: usize, k: usize, l: usize| -> [u32; 10] {
        let (t, r) = &routes[pi];
        let o = (l * t + k) * 10;
        std::array::from_fn(|j| r[o + j] as u32)
    };
    let mut steps = Vec::new();
    let group = |steps: &mut Vec<Step>, per: usize| {
        for (g, grp) in routes.chunks(8).enumerate() {
            let maxt = grp.iter().map(|x| x.0).max().unwrap_or(0);
            let mut k = 0;
            while k < maxt {
                let mut ids = vec![Vec::new(); LAYERS];
                let mut n = 0;
                for (j, x) in grp.iter().enumerate() {
                    for kk in k..(k + per).min(x.0) {
                        n += 1;
                        for (l, v) in ids.iter_mut().enumerate() {
                            v.extend_from_slice(&tok(g * 8 + j, kk, l));
                        }
                    }
                }
                if n > 0 {
                    steps.push(Step { tokens: n, ids });
                }
                k += per;
            }
        }
    };
    match kind {
        "b1" => {
            for (pi, (t, _)) in routes.iter().enumerate() {
                for k in 0..*t {
                    steps.push(Step {
                        tokens: 1,
                        ids: (0..LAYERS).map(|l| tok(pi, k, l).to_vec()).collect(),
                    });
                }
            }
        }
        "conc8" => group(&mut steps, 1),
        "conc8x8" => group(&mut steps, 8),
        c if c.starts_with("chunk") => {
            let t: usize = c[5..].parse()?;
            let all: Vec<(usize, usize)> = routes
                .iter()
                .enumerate()
                .flat_map(|(pi, (n, _))| (0..*n).map(move |k| (pi, k)))
                .collect();
            let chunks = all.len().div_ceil(t);
            for c in 0..chunks {
                let mut ids = vec![Vec::new(); LAYERS];
                for i in 0..t {
                    let (pi, k) = all[(c * t + i) % all.len()];
                    for (l, v) in ids.iter_mut().enumerate() {
                        v.extend_from_slice(&tok(pi, k, l));
                    }
                }
                steps.push(Step {
                    tokens: t as u32,
                    ids,
                });
            }
        }
        _ => bail!("unknown trace {kind}"),
    }
    Ok(steps)
}

fn trace_for(t: u32) -> String {
    match t {
        1 => "b1".into(),
        8 => "conc8".into(),
        64 => "conc8x8".into(),
        n => format!("chunk{n}"),
    }
}

// ----------------------------------------------------------------------------------------- kernels

enum Arg {
    P(u64),
    I(i32),
}

struct Kern<'a> {
    f: &'a Function<'a>,
    d: &'a KernelDecl,
}

fn kern<'a>(t: &'a FunctionTable<'a>, plugin: &str, name: &str) -> anyhow::Result<Kern<'a>> {
    let (f, d) = t
        .get(plugin, name)
        .with_context(|| format!("{plugin}::{name} not loaded"))?;
    Ok(Kern { f, d })
}

impl Kern<'_> {
    fn launch(
        &self,
        hip: &HipThread,
        grid: [u32; 3],
        threads: u32,
        st: &Stream,
        vals: &[Arg],
    ) -> anyhow::Result<()> {
        let exp: Vec<_> = self
            .d
            .arg
            .iter()
            .filter(|a| !a.value_kind.starts_with("hidden"))
            .collect();
        ensure!(
            exp.len() == vals.len(),
            "{}: {} explicit args, {} given",
            self.d.name,
            exp.len(),
            vals.len()
        );
        let size = exp.iter().map(|a| a.offset + a.size).max().unwrap_or(0) as usize;
        let mut img = vec![0u8; size];
        for (a, v) in exp.iter().zip(vals) {
            let o = a.offset as usize;
            match v {
                Arg::P(x) => {
                    ensure!(a.size == 8, "{}: arg at {o} is not a pointer", self.d.name);
                    img[o..o + 8].copy_from_slice(&x.to_le_bytes());
                }
                Arg::I(x) => {
                    ensure!(a.size == 4, "{}: arg at {o} is not an i32", self.d.name);
                    img[o..o + 4].copy_from_slice(&x.to_le_bytes());
                }
            }
        }
        hip.launch(self.f, grid, [threads, 1, 1], 0, st, &mut img)?;
        Ok(())
    }
}

fn a(p: &DevicePtr) -> u64 {
    p.addr().as_u64()
}

/// Deterministic fill.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

fn bf16(f: f32) -> [u8; 2] {
    let u = f.to_bits();
    (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
}

/// Every device buffer of the MoE block at the run's largest token count.
struct Bufs {
    tmax: u32,
    x8: DevicePtr,
    xs: DevicePtr,
    xb: DevicePtr,
    router_w: DevicePtr,
    router_c: DevicePtr,
    router_part: DevicePtr,
    router_cnt: DevicePtr,
    sgate: DevicePtr,
    ids: DevicePtr,
    w: DevicePtr,
    pos: DevicePtr,
    row_tok: DevicePtr,
    row_route: DevicePtr,
    te: DevicePtr,
    t0: DevicePtr,
    tr: DevicePtr,
    ecnt: DevicePtr,
    meta: DevicePtr,
    blk: DevicePtr,
    rank: DevicePtr,
    arrive: DevicePtr,
    h: DevicePtr,
    q: DevicePtr,
    sc: DevicePtr,
    y: DevicePtr,
    p1: DevicePtr,
    p2: DevicePtr,
    c1: DevicePtr,
    c2: DevicePtr,
    out: [DevicePtr; 2],
    /// `[slot][48][2][771]` u64 (slot 0/1: the tier by step parity, 2: golden/control, 3: stage).
    xptr: DevicePtr,
    /// The shared expert's six sub-records per layer.
    shared: DevicePtr,
    shared_off: Vec<[u64; 6]>,
}

fn upload(hip: &HipThread, st: &Stream, d: &DevicePtr, bytes: &[u8]) -> anyhow::Result<()> {
    hip.memcpy_h2d_blocking(d, bytes, st)?;
    Ok(())
}

impl Bufs {
    fn new(hip: &HipThread, st: &Stream, tmax: u32, pack: &Path) -> anyhow::Result<Bufs> {
        let g = Geometry {
            tokens: tmax,
            e_lo: 0,
            e_loc: LOC as u32,
            tile_blocks: 1,
            shared: true,
        };
        let (r, pcap, tcap) = (
            u64::from(g.routes()),
            u64::from(g.rows_cap()),
            u64::from(g.tiles_cap()),
        );
        let t = u64::from(tmax);
        let m = |n: u64| hip.malloc(n.max(4));
        let mut rng = Lcg(0xf8f8);
        let x8: Vec<u8> = (0..t * u64::from(H))
            .map(|_| {
                let v = rng.next();
                ((v % 0x60) as u8) | if (v >> 12) & 1 == 1 { 0x80 } else { 0 }
            })
            .collect();
        let xs: Vec<u8> = (0..t)
            .flat_map(|_| (0.01f32 + (rng.next() % 10_000) as f32 * 1e-5).to_le_bytes())
            .collect();
        let xb: Vec<u8> = (0..t * u64::from(H))
            .flat_map(|_| bf16(((rng.next() % 2000) as f32 - 1000.0) * 1e-3))
            .collect();
        let rw: Vec<u8> = (0..(LAYERS as u64) * 512 * u64::from(H))
            .flat_map(|_| bf16(((rng.next() % 2000) as f32 - 1000.0) * 2e-5))
            .collect();
        let sg: Vec<u8> = (0..t)
            .flat_map(|_| bf16(((rng.next() % 2000) as f32 - 1000.0) * 2e-3))
            .collect();
        let b = Bufs {
            tmax,
            x8: m(x8.len() as u64)?,
            xs: m(xs.len() as u64)?,
            xb: m(xb.len() as u64)?,
            router_w: m(rw.len() as u64)?,
            router_c: m(t * 512 * 2)?,
            router_part: m(4 * t * 512 * 4)?,
            router_cnt: m(4096 * 4)?,
            sgate: m(sg.len() as u64)?,
            ids: m(r * 4)?,
            w: m(r * 4)?,
            pos: m(r * 4)?,
            row_tok: m(pcap * 4)?,
            row_route: m(pcap * 4)?,
            te: m(tcap * 4)?,
            t0: m(tcap * 4)?,
            tr: m(tcap * 4)?,
            ecnt: m(LOC as u64 * 4)?,
            meta: m(32)?,
            blk: m(g.blk_cnt_len() * 4)?,
            rank: m(r * 4)?,
            arrive: m(4)?,
            h: m(pcap * 1280 * 2)?,
            q: m(pcap * 640)?,
            sc: m(pcap * 4)?,
            y: m(pcap * u64::from(H) * 2)?,
            p1: m(4 * pcap * 1280 * 4)?,
            p2: m(4 * pcap * u64::from(H) * 4)?,
            c1: m(tcap * 10 * 4)?,
            c2: m(tcap * 20 * 4)?,
            out: [m(t * u64::from(H) * 2)?, m(t * u64::from(H) * 2)?],
            xptr: m(4 * LAYERS as u64 * 2 * XBYTES)?,
            shared: m(1)?,
            shared_off: Vec::new(),
        };
        upload(hip, st, &b.x8, &x8)?;
        upload(hip, st, &b.xs, &xs)?;
        upload(hip, st, &b.xb, &xb)?;
        upload(hip, st, &b.router_w, &rw)?;
        upload(hip, st, &b.sgate, &sg)?;
        for z in [&b.arrive, &b.c1, &b.c2, &b.router_cnt, &b.meta, &b.h, &b.y] {
            hip.memset(z, 0, st)?;
        }
        hip.memset(&b.row_tok, 0xFF, st)?;
        hip.sync_stream(st)?;
        let mut b = b;
        b.load_shared(hip, st, pack)?;
        Ok(b)
    }

    /// The shared expert's trios of every layer from `shared/dense.pack` (one VRAM buffer).
    fn load_shared(&mut self, hip: &HipThread, st: &Stream, pack: &Path) -> anyhow::Result<()> {
        let dj: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(pack.join("shared/dense.json"))?)?;
        let recs = dj["records"].as_array().context("dense.json records")?;
        let find = |name: &str| -> anyhow::Result<(u64, u64)> {
            let r = recs
                .iter()
                .find(|r| r["name"] == name)
                .with_context(|| format!("no record {name}"))?;
            ensure!(
                r["storage"]["pack"] == "dense.pack",
                "{name} not in dense.pack"
            );
            Ok((
                r["storage"]["data_off"].as_u64().context("data_off")?,
                r["storage"]["length"]
                    .as_u64()
                    .or_else(|| r["bytes"].as_u64())
                    .context("length")?,
            ))
        };
        let lens: [u64; 6] = [
            1280 * 2560 / 2,
            80 * 1280,
            1280,
            2560 * 640 / 2,
            20 * 2560,
            2560,
        ];
        let mut offs = [0u64; 6];
        let mut acc = 0;
        for (i, l) in lens.iter().enumerate() {
            offs[i] = acc;
            acc += l.div_ceil(256) * 256;
        }
        let per = acc;
        let mut host = vec![0u8; (per * LAYERS as u64) as usize];
        let f = std::fs::File::open(pack.join("shared/dense.pack"))?;
        use std::os::unix::fs::FileExt;
        for l in 0..LAYERS {
            let mut o = [0u64; 6];
            for (i, (m, p)) in ["gate_up", "down"]
                .iter()
                .flat_map(|m| ["w", "ws", "wref"].map(move |p| (m, p)))
                .enumerate()
            {
                let (off, len) = find(&format!("layer.{l}.mlp.shared_expert.{m}.{p}"))?;
                ensure!(len >= lens[i], "layer {l} {m}.{p}: {len} < {}", lens[i]);
                let dst = (l as u64 * per + offs[i]) as usize;
                f.read_exact_at(&mut host[dst..dst + lens[i] as usize], off)?;
                o[i] = l as u64 * per + offs[i];
            }
            self.shared_off.push(o);
        }
        self.shared = hip.malloc(host.len() as u64)?;
        upload(hip, st, &self.shared, &host)?;
        Ok(())
    }
}

/// The MoE block's kernels.
struct Moe<'a> {
    route: Kern<'a>,
    silu: Kern<'a>,
    comb: Kern<'a>,
    grouped: Vec<(Cell, Kern<'a>, Kern<'a>)>,
    table: &'a FunctionTable<'a>,
    rules: FlashNextRules,
}

impl<'a> Moe<'a> {
    fn new(t: &'a FunctionTable<'a>) -> anyhow::Result<Moe<'a>> {
        let mut grouped = Vec::new();
        for c in moe::CELLS {
            grouped.push((
                c,
                kern(
                    t,
                    moe::PLUGIN_GROUPED,
                    &moe::grouped_symbol(MoeGemm::GateUp, c),
                )?,
                kern(
                    t,
                    moe::PLUGIN_GROUPED,
                    &moe::grouped_symbol(MoeGemm::Down, c),
                )?,
            ));
        }
        Ok(Moe {
            route: kern(t, moe::PLUGIN_ROUTE, moe::ROUTE_SYMBOL)?,
            silu: kern(t, moe::PLUGIN_GROUPED, moe::SILU_SYMBOL)?,
            comb: kern(t, moe::PLUGIN_ROUTE, moe::COMBINE_SYMBOL)?,
            grouped,
            table: t,
            rules: FlashNextRules::prod()?,
        })
    }

    fn cell(&self, c: Cell, gu: bool) -> &Kern<'a> {
        let e = self
            .grouped
            .iter()
            .find(|(k, _, _)| *k == c)
            .expect("every cell loaded");
        if gu {
            &e.1
        } else {
            &e.2
        }
    }

    /// The six launches of one layer at `t` tokens.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        hip: &HipThread,
        st: &Stream,
        b: &Bufs,
        l: usize,
        t: u32,
        logits: u64,
        xgu: u64,
        xdn: u64,
        out: u64,
    ) -> anyhow::Result<()> {
        let (cg, cd) = (
            moe::measured_cell(MoeGemm::GateUp, t),
            moe::measured_cell(MoeGemm::Down, t),
        );
        ensure!(cg.tm == cd.tm, "the route kernel's tile law needs one tm");
        let g = Geometry {
            tokens: t,
            e_lo: 0,
            e_loc: LOC as u32,
            tile_blocks: cg.tm,
            shared: true,
        };
        // 1. the router GEMM (FN5's route; its logits are not consumed).
        let (lr, splits) = self.rules.bgemm_route("moe.router", t, 512)?;
        let plugin = self.rules.plugin_for(&lr.symbol).context("router plugin")?;
        let rk = kern(self.table, plugin, &lr.symbol)?;
        let rw = a(&b.router_w) + (l as u64) * 512 * u64::from(H) * 2;
        let (m, n, k) = (t as i32, 512, H as i32);
        if lr.symbol.starts_with("fn_bgemm_sk") {
            ensure!(u64::from(splits) * u64::from(t) * 512 * 4 <= b.router_part.len());
            rk.launch(
                hip,
                lr.grid,
                lr.threads,
                st,
                &[
                    Arg::P(a(&b.xb)),
                    Arg::P(rw),
                    Arg::P(a(&b.router_part)),
                    Arg::P(a(&b.router_cnt)),
                    Arg::P(a(&b.router_c)),
                    Arg::I(m),
                    Arg::I(n),
                    Arg::I(k),
                ],
            )?;
        } else {
            rk.launch(
                hip,
                lr.grid,
                lr.threads,
                st,
                &[
                    Arg::P(a(&b.xb)),
                    Arg::P(rw),
                    Arg::P(a(&b.router_c)),
                    Arg::I(m),
                    Arg::I(n),
                    Arg::I(k),
                ],
            )?;
        }
        // 2. route + sort.
        self.route.launch(
            hip,
            [g.route_workgroups(), 1, 1],
            256,
            st,
            &[
                Arg::P(logits),
                Arg::I(t as i32),
                Arg::I(0),
                Arg::I(LOC as i32),
                Arg::I(cg.tm as i32),
                Arg::I(1),
                Arg::P(a(&b.ids)),
                Arg::P(a(&b.w)),
                Arg::P(a(&b.pos)),
                Arg::P(a(&b.row_tok)),
                Arg::P(a(&b.row_route)),
                Arg::P(a(&b.te)),
                Arg::P(a(&b.t0)),
                Arg::P(a(&b.tr)),
                Arg::P(a(&b.ecnt)),
                Arg::P(a(&b.meta)),
                Arg::P(a(&b.blk)),
                Arg::P(a(&b.rank)),
                Arg::P(a(&b.arrive)),
            ],
        )?;
        let pmax = g.rows_cap() as i32;
        let tiles = g.tiles_cap();
        // 3. gate_up (A gathered by token).
        self.cell(cg, true).launch(
            hip,
            [1280 / 128, tiles, cg.ks],
            256,
            st,
            &[
                Arg::P(a(&b.x8)),
                Arg::P(xgu),
                Arg::P(a(&b.te)),
                Arg::P(a(&b.t0)),
                Arg::P(a(&b.tr)),
                Arg::P(a(&b.row_tok)),
                Arg::P(a(&b.meta)),
                Arg::P(a(&b.xs)),
                Arg::P(a(&b.p1)),
                Arg::P(a(&b.c1)),
                Arg::P(a(&b.h)),
                Arg::I(1280),
                Arg::I(H as i32),
                Arg::I(pmax),
            ],
        )?;
        // 4. silu-quant.
        self.silu.launch(
            hip,
            [g.rows_cap(), 1, 1],
            256,
            st,
            &[
                Arg::P(a(&b.h)),
                Arg::P(a(&b.q)),
                Arg::P(a(&b.sc)),
                Arg::I(640),
                Arg::P(a(&b.row_tok)),
                Arg::P(a(&b.meta)),
            ],
        )?;
        // 5. down.
        self.cell(cd, false).launch(
            hip,
            [H / 128, tiles, cd.ks],
            256,
            st,
            &[
                Arg::P(a(&b.q)),
                Arg::P(xdn),
                Arg::P(a(&b.te)),
                Arg::P(a(&b.t0)),
                Arg::P(a(&b.tr)),
                Arg::P(a(&b.row_tok)),
                Arg::P(a(&b.meta)),
                Arg::P(a(&b.sc)),
                Arg::P(a(&b.p2)),
                Arg::P(a(&b.c2)),
                Arg::P(a(&b.y)),
                Arg::I(H as i32),
                Arg::I(640),
                Arg::I(pmax),
            ],
        )?;
        // 6. combine.
        self.comb.launch(
            hip,
            [t, 1, 1],
            256,
            st,
            &[
                Arg::P(a(&b.y)),
                Arg::P(a(&b.w)),
                Arg::P(a(&b.pos)),
                Arg::P(a(&b.sgate)),
                Arg::P(a(&b.meta)),
                Arg::P(out),
                Arg::I(H as i32),
            ],
        )?;
        Ok(())
    }
}

// ------------------------------------------------------------------------------------------ the rig

/// The logits of a whole trace on the device: `[step][48][T][512]` bf16, top-10 of each token set
/// to the replay's ids in rank order (4.0 - 0.25 k), the rest -4.0.
struct Logits {
    buf: DevicePtr,
    off: Vec<u64>,
}

fn logits_for(hip: &HipThread, st: &Stream, steps: &[Step]) -> anyhow::Result<Logits> {
    let mut off = Vec::with_capacity(steps.len());
    let mut total = 0u64;
    for s in steps {
        off.push(total);
        total += LAYERS as u64 * u64::from(s.tokens) * 512 * 2;
    }
    let mut host = Vec::with_capacity(total as usize);
    let lo = bf16(-4.0);
    for s in steps {
        for l in 0..LAYERS {
            for t in 0..s.tokens as usize {
                let mut row = [0u8; 1024];
                for e in 0..512 {
                    row[2 * e..2 * e + 2].copy_from_slice(&lo);
                }
                for k in 0..10 {
                    let e = s.ids[l][t * 10 + k] as usize;
                    row[2 * e..2 * e + 2].copy_from_slice(&bf16(4.0 - 0.25 * k as f32));
                }
                host.extend_from_slice(&row);
            }
        }
    }
    let buf = hip.malloc(total.max(4))?;
    upload(hip, st, &buf, &host)?;
    Ok(Logits { buf, off })
}

/// Timing events of one step.
struct Clocks {
    l0: Vec<Event>,
    l1: Vec<Event>,
    s0: Event,
    s1: Event,
    c0: [Event; 2],
    c1: [Event; 2],
}

impl Clocks {
    fn new(hip: &HipThread) -> anyhow::Result<Clocks> {
        let e = || hip.create_event(EventFlags::Default);
        Ok(Clocks {
            l0: (0..LAYERS).map(|_| e()).collect::<Result<_, _>>()?,
            l1: (0..LAYERS).map(|_| e()).collect::<Result<_, _>>()?,
            s0: e()?,
            s1: e()?,
            c0: [e()?, e()?],
            c1: [e()?, e()?],
        })
    }
}

/// What one run measured.
#[derive(Debug, Default, Clone)]
struct RunStats {
    steps: u64,
    layer_ms: Vec<f64>,
    compute_ms: f64,
    wall_ms: f64,
    host_ms: f64,
    copy_ms: f64,
    copy_steps: u64,
    resident: u64,
    ram: u64,
    nvme: u64,
    inserts: u64,
    demotions: u64,
    bytes_h2d: u64,
    bytes_d2h: u64,
    gate_waits: u64,
    first_step_ms: f64,
    first_step_nvme: u64,
    staged: u64,
    staged_unused: u64,
    stage_overflow: u64,
}

impl RunStats {
    fn per(&self, x: f64) -> f64 {
        x / self.steps.max(1) as f64
    }
    fn line(&self, name: &str, t: u32) -> String {
        let routed = (self.resident + self.ram + self.nvme).max(1) as f64;
        format!(
            "path {name:<14} T={t:<5} steps {:>5} | ms/layer {:.4} | compute ms/step {:.3} | wall ms/step {:.3} \
             | host ms/step {:.3} | copy-stream ms/step {:.3} | resident {:.1}% ram {:.1}% nvme {:.2}% \
             ({:.1}/{:.1}/{:.2} per step) | inserts/step {:.2} demotions/step {:.2} | link MB/step h2d {:.1} d2h {:.1} \
             | gate waits/step {:.2}{}",
            self.steps,
            self.per(self.compute_ms) / LAYERS as f64,
            self.per(self.compute_ms),
            self.per(self.wall_ms),
            self.per(self.host_ms),
            self.copy_ms / self.copy_steps.max(1) as f64,
            100.0 * self.resident as f64 / routed,
            100.0 * self.ram as f64 / routed,
            100.0 * self.nvme as f64 / routed,
            self.per(self.resident as f64),
            self.per(self.ram as f64),
            self.per(self.nvme as f64),
            self.per(self.inserts as f64),
            self.per(self.demotions as f64),
            self.per(self.bytes_h2d as f64) / 1e6,
            self.per(self.bytes_d2h as f64) / 1e6,
            self.per(self.gate_waits as f64),
            if self.staged + self.stage_overflow + self.staged_unused > 0 {
                format!(
                    " | staged {:.1}/step (unused {:.1}, over the ring -> UVA {:.1})",
                    self.per(self.staged as f64),
                    self.per(self.staged_unused as f64),
                    self.per(self.stage_overflow as f64)
                )
            } else {
                String::new()
            }
        )
    }
}

/// How a step runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Way {
    /// Every routed RAM expert copied (untimed) into the control ring first: all-VRAM reads.
    Control,
    /// The tier; copies overlapped with the next step.
    Tier,
    /// The tier; the copy stream drained before the next step (the overlap control).
    Serial,
    /// The demand path: the host waits for each layer's routes, `resolve` may read NVMe.
    Demand,
    /// The tier with the prefill staging ring.
    Staged,
}

struct Rig<'h, 'x> {
    hip: &'h HipThread,
    xt: &'x XtierFile,
    compute: Stream,
    mover: DeviceMover<'x, HipCopyQueue<'h>>,
    stride: u64,
    sub: [(u64, u64); 6],
    /// The golden / control ring (LOC slots) and its pinned NVMe staging.
    ring: DevicePtr,
    gstage: PinnedBuf,
    /// The staging ring (2 halves × slots).
    stage_ring: Option<DevicePtr>,
    /// Stage only the step's ROUTED RAM experts (an oracle: the engine cannot know a layer's
    /// routes before its router; the bound for any staging policy).
    stage_routed: bool,
    stage_ev: Vec<Event>,
    consumed: [Event; 2],
    /// Host xptr staging `[4][48][2][771]` u64.
    xhost: PinnedBuf,
    /// What each VRAM / RAM slot holds right now (host mirror, for restores).
    cur_vram: Vec<u32>,
    cur_ram: Vec<u32>,
}

const NONE: u32 = u32::MAX;

impl<'h, 'x> Rig<'h, 'x> {
    fn xput(&mut self, slot: usize, l: usize, gu: &[u64], dn: &[u64]) {
        let base = ((slot * LAYERS + l) * 2 * XN) * 8;
        let s = self.xhost.as_mut_slice();
        for (i, v) in gu.iter().chain(dn.iter()).enumerate() {
            s[base + i * 8..base + i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
    }
    fn xdev(&self, b: &Bufs, slot: usize, l: usize) -> (u64, u64) {
        let o = a(&b.xptr) + ((slot * LAYERS + l) * 2) as u64 * XBYTES;
        (o, o + XBYTES)
    }
    /// H2D of `n` layers' tables of `slot` starting at `l0`, on the compute stream.
    fn xupload(&self, b: &Bufs, slot: usize, l0: usize, n: usize) -> anyhow::Result<()> {
        let off = ((slot * LAYERS + l0) * 2) as u64 * XBYTES;
        self.hip.run_batch(&[Job::MemcpyPinned {
            dev: &b.xptr,
            dev_offset: off,
            host: &self.xhost,
            host_offset: off,
            bytes: n as u64 * 2 * XBYTES,
            kind: MemcpyKind::HostToDevice,
            stream: &self.compute,
        }])?;
        Ok(())
    }
    fn tables(&self, tier: &ExpertTier, b: &Bufs, l: usize) -> (Vec<u64>, Vec<u64>) {
        let (mut gu, mut dn) = (vec![0u64; XN], vec![0u64; XN]);
        tier.fill_xptr(l as u32, &mut gu, &mut dn);
        let sh = a(&b.shared);
        for k in 0..3 {
            gu[3 * LOC + k] = sh + b.shared_off[l][k];
            dn[3 * LOC + k] = sh + b.shared_off[l][3 + k];
        }
        (gu, dn)
    }
    fn ring_addr(&self, i: usize) -> u64 {
        a(&self.ring) + i as u64 * self.stride
    }
    fn override_entry(&self, gu: &mut [u64], dn: &mut [u64], e: usize, base: u64) {
        for k in 0..3 {
            gu[3 * e + k] = base + self.sub[k].0;
            dn[3 * e + k] = base + self.sub[3 + k].0;
        }
    }

    /// Record `rec` from NVMe into golden ring slot `i` for a batch (blocking).
    fn golden_fill(&mut self, recs: &[(u64, usize)]) -> anyhow::Result<()> {
        let per = self.gstage.len() / self.stride as usize;
        for chunk in recs.chunks(per) {
            let reads: Vec<(u64, u32)> = chunk
                .iter()
                .enumerate()
                .map(|(k, &(r, _))| (r, k as u32))
                .collect();
            parallel_reads(
                self.xt,
                self.gstage.as_mut_slice(),
                self.stride as usize,
                &reads,
                32,
            )?;
            let jobs: Vec<Job<'_>> = chunk
                .iter()
                .enumerate()
                .map(|(k, &(_, i))| Job::MemcpyPinned {
                    dev: &self.ring,
                    dev_offset: i as u64 * self.stride,
                    host: &self.gstage,
                    host_offset: k as u64 * self.stride,
                    bytes: self.stride,
                    kind: MemcpyKind::HostToDevice,
                    stream: &self.compute,
                })
                .chain(std::iter::once(Job::SyncStream {
                    stream: &self.compute,
                }))
                .collect();
            self.hip.run_batch(&jobs)?;
        }
        Ok(())
    }

    /// Bring the slots to `tier`'s warm placement, reading only the slots whose record differs.
    fn restore(&mut self, tier: &ExpertTier) -> anyhow::Result<(usize, usize)> {
        self.mover.drain()?;
        let p = tier.params();
        let mut ram = Vec::new();
        for s in 0..=p.ram_slots {
            let want = tier.ram_record(s).unwrap_or(NONE);
            if want != NONE && self.cur_ram[s as usize] != want {
                ram.push((u64::from(want), s));
            }
        }
        let st = self.stride as usize;
        for chunk in ram.chunks(256) {
            parallel_reads(
                self.xt,
                self.mover.queue_mut().ram.as_mut_slice(),
                st,
                chunk,
                32,
            )?;
        }
        for &(r, s) in &ram {
            self.cur_ram[s as usize] = r as u32;
        }
        let mut vram = Vec::new();
        for s in 0..p.vram_slots {
            let want = tier.vram_record(s).unwrap_or(NONE);
            if want != NONE && self.cur_vram[s as usize] != want {
                vram.push((u64::from(want), s));
            }
        }
        let per = self.gstage.len() / st;
        for chunk in vram.chunks(per) {
            let reads: Vec<(u64, u32)> = chunk
                .iter()
                .enumerate()
                .map(|(k, &(r, _))| (r, k as u32))
                .collect();
            parallel_reads(self.xt, self.gstage.as_mut_slice(), st, &reads, 32)?;
            let q = self.mover.queue();
            let jobs: Vec<Job<'_>> = chunk
                .iter()
                .enumerate()
                .map(|(k, &(_, s))| Job::MemcpyPinned {
                    dev: &q.vram,
                    dev_offset: u64::from(s) * self.stride,
                    host: &self.gstage,
                    host_offset: (k * st) as u64,
                    bytes: self.stride,
                    kind: MemcpyKind::HostToDevice,
                    stream: &q.stream,
                })
                .chain(std::iter::once(Job::SyncStream { stream: &q.stream }))
                .collect();
            self.hip.run_batch(&jobs)?;
        }
        for &(r, s) in &vram {
            self.cur_vram[s as usize] = r as u32;
        }
        Ok((vram.len(), ram.len()))
    }

    /// After a run: the mirror follows the tier.
    fn sync_mirror(&mut self, tier: &ExpertTier) {
        let p = tier.params();
        for s in 0..p.vram_slots {
            self.cur_vram[s as usize] = tier.vram_record(s).unwrap_or(NONE);
        }
        for s in 0..=p.ram_slots {
            self.cur_ram[s as usize] = tier.ram_record(s).unwrap_or(NONE);
        }
    }

    /// THE SLOT BYTE LAW: every VRAM slot (read back once) and every RAM slot == its record's
    /// sha256. Returns (checked, bad).
    fn verify_slots(&mut self, tier: &mut ExpertTier) -> anyhow::Result<(u64, u64)> {
        self.mover.drain()?;
        // FN17: the deferred law's last copies have landed (drained): publish them, so every
        // slot they wrote is checked against its record too
        tier.settle();
        let st = self.stride as usize;
        let p = tier.params().clone();
        let xt = self.xt;
        let mut bad = 0u64;
        let mut checked = 0u64;
        // RAM: in place, threaded.
        {
            let slab = self.mover.queue().ram.as_slice();
            let slots: Vec<(u32, u32)> = (0..tier.ram_slab_slots())
                .filter_map(|s| tier.ram_record(s).map(|r| (s, r)))
                .collect();
            let (c, b) = verify_threaded(xt, slab, st, &slots);
            checked += c;
            bad += b;
        }
        // VRAM: D2H through the golden staging, a batch at a time.
        let per = self.gstage.len() / st;
        let slots: Vec<(u32, u32)> = (0..p.vram_slots)
            .filter_map(|s| tier.vram_record(s).map(|r| (s, r)))
            .collect();
        for chunk in slots.chunks(per) {
            let q = self.mover.queue();
            let jobs: Vec<Job<'_>> = chunk
                .iter()
                .enumerate()
                .map(|(k, &(s, _))| Job::MemcpyPinned {
                    dev: &q.vram,
                    dev_offset: u64::from(s) * self.stride,
                    host: &self.gstage,
                    host_offset: (k * st) as u64,
                    bytes: self.stride,
                    kind: MemcpyKind::DeviceToHost,
                    stream: &q.stream,
                })
                .chain(std::iter::once(Job::SyncStream { stream: &q.stream }))
                .collect();
            self.hip.run_batch(&jobs)?;
            let local: Vec<(u32, u32)> = chunk
                .iter()
                .enumerate()
                .map(|(k, &(_, r))| (k as u32, r))
                .collect();
            let (c, b) = verify_threaded(xt, self.gstage.as_slice(), st, &local);
            checked += c;
            bad += b;
        }
        Ok((checked, bad))
    }
}

pub(crate) fn verify_threaded(
    xt: &XtierFile,
    slab: &[u8],
    st: usize,
    slots: &[(u32, u32)],
) -> (u64, u64) {
    let n = 16usize;
    let bad = std::sync::atomic::AtomicU64::new(0);
    std::thread::scope(|sc| {
        for part in slots.chunks(slots.len().div_ceil(n).max(1)) {
            let bad = &bad;
            sc.spawn(move || {
                for &(s, r) in part {
                    let b = &slab[s as usize * st..(s as usize + 1) * st];
                    if !xt.verify(u64::from(r), b) {
                        bad.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            });
        }
    });
    (slots.len() as u64, bad.into_inner())
}

/// Run `steps` of `path` over the tier. `checkpoint(step index)` asks for the per-layer golden
/// comparison (gate mode). Returns the stats and the gate's (compared, identical, controls caught).
#[allow(clippy::too_many_arguments)]
fn run_path(
    rig: &mut Rig<'_, '_>,
    tier: &mut ExpertTier,
    moe: &Moe<'_>,
    b: &Bufs,
    lg: &Logits,
    steps: &[Step],
    warm: usize,
    path: Way,
    stage: Option<StageLaw>,
    ck: &mut Clocks,
    checkpoint: &dyn Fn(usize) -> bool,
    gate: &mut (u64, u64, u64, u64),
) -> anyhow::Result<RunStats> {
    let hip = rig.hip;
    let mut rs = RunStats::default();
    let mut copy_pending: Option<usize> = None;
    let complete = tier.complete();
    for (si, s) in steps.iter().enumerate() {
        let timed = si >= warm;
        let t = s.tokens;
        ensure!(t <= b.tmax, "step of {t} tokens over the run's {}", b.tmax);
        let h0 = Instant::now();
        tier.begin_step();
        let parity = si % 2;
        let mut c = saber_mem::expert_tier::PathCounts::default();
        let lgs = a(&lg.buf) + lg.off[si];
        let lstride = u64::from(t) * 512 * 2;
        let upfront = path != Way::Demand;
        if upfront {
            // Complete tier / frozen tier: the step's placements are known before it starts.
            for (l, ids) in s.ids.iter().enumerate() {
                let x = tier.resolve(l as u32, ids, &mut rig.mover)?;
                c.resident += x.resident;
                c.ram += x.ram;
                c.nvme += x.nvme;
                tier.check_dispatch(l as u32, ids)?;
            }
            ensure!(
                !complete || c.nvme == 0,
                "the complete tier read NVMe mid-step"
            );
            for l in 0..LAYERS {
                let (gu, dn) = rig.tables(tier, b, l);
                rig.xput(parity, l, &gu, &dn);
            }
            rig.xupload(b, parity, 0, LAYERS)?;
        }
        // The staging ring's first layer is issued before the step's first launch.
        let stage_on = path == Way::Staged && stage.is_some_and(|x| x.stages(t));
        if stage_on {
            let law = stage.expect("stage law");
            for l in 0..LAYERS {
                let (mut gu, mut dn) = rig.tables(tier, b, l);
                let routed: std::collections::BTreeSet<u32> = s.ids[l]
                    .iter()
                    .filter(|&&e| (e as usize) < LOC)
                    .copied()
                    .collect();
                let mut k = 0usize;
                for e in 0..LOC as u32 {
                    if rig.stage_routed && !routed.contains(&e) {
                        continue;
                    }
                    if let Some(Placement::Ram(_)) = tier.placement(l as u32, e) {
                        if k < law.slots as usize {
                            let half = (l % 2) * law.slots as usize;
                            let base = a(rig.stage_ring.as_ref().expect("ring"))
                                + (half + k) as u64 * rig.stride;
                            rig.override_entry(&mut gu, &mut dn, e as usize, base);
                            k += 1;
                            if routed.contains(&e) {
                                rs.staged += u64::from(timed);
                            } else {
                                rs.staged_unused += u64::from(timed);
                            }
                        } else if routed.contains(&e) {
                            rs.stage_overflow += u64::from(timed);
                        }
                    }
                }
                rig.xput(3, l, &gu, &dn);
            }
            rig.xupload(b, 3, 0, LAYERS)?;
            stage_issue(rig, tier, 0, law, rig.stage_routed.then_some(&s.ids[0][..]))?;
        }
        hip.record_event(&ck.s0, &rig.compute)?;
        for l in 0..LAYERS {
            let ids = &s.ids[l];
            let (mut xg, mut xd) = rig.xdev(b, parity, l);
            match path {
                Way::Demand => {
                    // The route readback: earlier layers are done before resolve may drop a RAM
                    // record or read NVMe into its slot.
                    hip.sync_stream(&rig.compute)?;
                    let x = tier.resolve(l as u32, ids, &mut rig.mover)?;
                    c.resident += x.resident;
                    c.ram += x.ram;
                    c.nvme += x.nvme;
                    tier.check_dispatch(l as u32, ids)?;
                    let (gu, dn) = rig.tables(tier, b, l);
                    rig.xput(parity, l, &gu, &dn);
                    rig.xupload(b, parity, l, 1)?;
                }
                Way::Control => {
                    // Every routed RAM expert copied into the control ring (untimed: before the
                    // layer's start event, on the same stream).
                    let (mut gu, mut dn) = rig.tables(tier, b, l);
                    let mut local: Vec<u32> = ids
                        .iter()
                        .filter(|&&e| (e as usize) < LOC)
                        .copied()
                        .collect();
                    local.sort_unstable();
                    local.dedup();
                    let mut jobs = Vec::new();
                    let mut k = 0;
                    let mut ovr = Vec::new();
                    for e in local {
                        if let Some(Placement::Ram(r)) = tier.placement(l as u32, e) {
                            ovr.push((e as usize, rig.ring_addr(k)));
                            jobs.push((r, k));
                            k += 1;
                        }
                    }
                    for (e, base) in ovr {
                        rig.override_entry(&mut gu, &mut dn, e, base);
                    }
                    rig.xput(2, l, &gu, &dn);
                    rig.xupload(b, 2, l, 1)?;
                    let q = rig.mover.queue();
                    let js: Vec<Job<'_>> = jobs
                        .iter()
                        .map(|&(r, k)| Job::MemcpyPinned {
                            dev: &rig.ring,
                            dev_offset: k as u64 * rig.stride,
                            host: &q.ram,
                            host_offset: u64::from(r) * rig.stride,
                            bytes: rig.stride,
                            kind: MemcpyKind::HostToDevice,
                            stream: &rig.compute,
                        })
                        .collect();
                    hip.run_batch(&js)?;
                    (xg, xd) = rig.xdev(b, 2, l);
                }
                _ => {}
            }
            if stage_on {
                let law = stage.expect("stage law");
                // Issue the NEXT layer's staging now: it lands while this layer computes.
                if l + 1 < LAYERS {
                    stage_issue(
                        rig,
                        tier,
                        l + 1,
                        law,
                        rig.stage_routed.then_some(&s.ids[l + 1][..]),
                    )?;
                }
                hip.stream_wait_event(&rig.compute, &rig.stage_ev[l])?;
                (xg, xd) = rig.xdev(b, 3, l);
            }
            let waited = rig.mover.gate(l as u32, &rig.compute)?;
            rs.gate_waits += u64::from(waited && timed);
            hip.record_event(&ck.l0[l], &rig.compute)?;
            moe.layer(
                hip,
                &rig.compute,
                b,
                l,
                t,
                lgs + l as u64 * lstride,
                xg,
                xd,
                a(&b.out[0]),
            )?;
            hip.record_event(&ck.l1[l], &rig.compute)?;
            if stage_on {
                hip.record_event(&rig.consumed[l % 2], &rig.compute)?;
            }
        }
        hip.record_event(&ck.s1, &rig.compute)?;
        let host_issue = h0.elapsed().as_secs_f64() * 1e3;
        hip.sync_event(&ck.s1)?;
        // The gate's checkpoint: before end_step moves anything, compare every layer against the
        // golden run (records fresh from NVMe) and the neighbour control.
        if checkpoint(si) {
            gate_step(rig, tier, moe, b, lg, si, s, gate)?;
        }
        let wall = f64::from(hip.elapsed_ms(&ck.s0, &ck.s1)?);
        let mut comp = 0.0;
        let mut per_layer = vec![0.0; LAYERS];
        for (l, pl) in per_layer.iter_mut().enumerate() {
            let x = f64::from(hip.elapsed_ms(&ck.l0[l], &ck.l1[l])?);
            comp += x;
            *pl = x;
        }
        // The previous step's copy-stream time (its events have landed by now).
        if let Some(pp) = copy_pending.take() {
            hip.sync_event(&ck.c1[pp])?;
            if timed {
                rs.copy_ms += f64::from(hip.elapsed_ms(&ck.c0[pp], &ck.c1[pp])?);
                rs.copy_steps += 1;
            }
        }
        if si == 0 {
            rs.first_step_ms = h0.elapsed().as_secs_f64() * 1e3;
            rs.first_step_nvme = c.nvme;
        }
        // The step's end: fence, then the LRU's copies on the copy stream (not for the frozen
        // control or the staging comparison).
        let mut moved = (0, 0, 0, 0);
        if matches!(path, Way::Tier | Way::Serial | Way::Demand) {
            rig.mover.after(&rig.compute)?;
            let q = rig.mover.queue();
            hip.record_event(&ck.c0[parity], &q.stream)?;
            let r = tier.end_step(&mut rig.mover)?;
            let q = rig.mover.queue();
            hip.record_event(&ck.c1[parity], &q.stream)?;
            copy_pending = Some(parity);
            moved = (
                r.moves.inserts,
                r.moves.demotions,
                r.moves.bytes_h2d,
                r.moves.bytes_d2h,
            );
            if path == Way::Serial {
                rig.mover.drain()?;
            }
        }
        if timed {
            rs.steps += 1;
            rs.wall_ms += wall;
            rs.compute_ms += comp;
            rs.host_ms += host_issue;
            if rs.layer_ms.is_empty() {
                rs.layer_ms = vec![0.0; LAYERS];
            }
            for (x, y) in rs.layer_ms.iter_mut().zip(per_layer) {
                *x += y;
            }
            rs.resident += c.resident;
            rs.ram += c.ram;
            rs.nvme += c.nvme;
            rs.inserts += moved.0;
            rs.demotions += moved.1;
            rs.bytes_h2d += moved.2;
            rs.bytes_d2h += moved.3;
        }
        if path == Way::Demand {
            // A demand step's wall is the host's: it waited for every layer.
            if timed {
                rs.wall_ms += h0.elapsed().as_secs_f64() * 1e3 - wall;
            }
        }
    }
    if let Some(pp) = copy_pending.take() {
        hip.sync_event(&ck.c1[pp])?;
        rs.copy_ms += f64::from(hip.elapsed_ms(&ck.c0[pp], &ck.c1[pp])?);
        rs.copy_steps += 1;
    }
    rig.mover.drain()?;
    rig.sync_mirror(tier);
    Ok(rs)
}

/// Issue layer `l`'s staging copies on the copy stream: wait until the ring half's previous user
/// (layer l-2) finished, H2D the layer's RAM-resident experts (up to the half's slots), mark.
fn stage_issue(
    rig: &Rig<'_, '_>,
    tier: &ExpertTier,
    l: usize,
    law: StageLaw,
    only: Option<&[u32]>,
) -> anyhow::Result<()> {
    let q = rig.mover.queue();
    let ring = rig.stage_ring.as_ref().expect("ring");
    let half = (l % 2) * law.slots as usize;
    let mut jobs = Vec::new();
    if l >= 2 {
        jobs.push(Job::StreamWaitEvent {
            stream: &q.stream,
            event: &rig.consumed[l % 2],
        });
    }
    let mut k = 0usize;
    for e in 0..LOC as u32 {
        if k >= law.slots as usize {
            break;
        }
        if only.is_some_and(|o| !o.contains(&e)) {
            continue;
        }
        if let Some(Placement::Ram(r)) = tier.placement(l as u32, e) {
            jobs.push(Job::MemcpyPinned {
                dev: ring,
                dev_offset: (half + k) as u64 * rig.stride,
                host: &q.ram,
                host_offset: u64::from(r) * rig.stride,
                bytes: rig.stride,
                kind: MemcpyKind::HostToDevice,
                stream: &q.stream,
            });
            k += 1;
        }
    }
    jobs.push(Job::RecordEvent {
        event: &rig.stage_ev[l],
        stream: &q.stream,
    });
    rig.hip.run_batch(&jobs)?;
    Ok(())
}

/// One checkpoint of the byte gate: per layer, re-run the tier path's layer into `out[0]`, the
/// golden layer (every routed local expert read fresh from NVMe into the ring) into `out[1]`,
/// compare; at the first layer with two routed experts also the neighbour control.
#[allow(clippy::too_many_arguments)]
fn gate_step(
    rig: &mut Rig<'_, '_>,
    tier: &ExpertTier,
    moe: &Moe<'_>,
    b: &Bufs,
    lg: &Logits,
    si: usize,
    s: &Step,
    gate: &mut (u64, u64, u64, u64),
) -> anyhow::Result<()> {
    let hip = rig.hip;
    let t = s.tokens;
    let lstride = u64::from(t) * 512 * 2;
    let lgs = a(&lg.buf) + lg.off[si];
    let n = (u64::from(t) * u64::from(H) * 2) as usize;
    let mut control_done = false;
    for l in 0..LAYERS {
        let (xg, xd) = rig.xdev(b, si % 2, l);
        moe.layer(
            hip,
            &rig.compute,
            b,
            l,
            t,
            lgs + l as u64 * lstride,
            xg,
            xd,
            a(&b.out[0]),
        )?;
        let mut local: Vec<u32> = s.ids[l]
            .iter()
            .filter(|&&e| (e as usize) < LOC)
            .copied()
            .collect();
        local.sort_unstable();
        local.dedup();
        let recs: Vec<(u64, usize)> = local
            .iter()
            .enumerate()
            .map(|(k, &e)| (u64::from(tier.index(l as u32, e).expect("local")), k))
            .collect();
        rig.golden_fill(&recs)?;
        let (mut gu, mut dn) = rig.tables(tier, b, l);
        for g in gu.iter_mut().take(3 * LOC) {
            *g = 0;
        }
        for d in dn.iter_mut().take(3 * LOC) {
            *d = 0;
        }
        for (k, &e) in local.iter().enumerate() {
            let base = rig.ring_addr(k);
            rig.override_entry(&mut gu, &mut dn, e as usize, base);
        }
        rig.xput(2, l, &gu, &dn);
        rig.xupload(b, 2, l, 1)?;
        let (gg, gd) = rig.xdev(b, 2, l);
        moe.layer(
            hip,
            &rig.compute,
            b,
            l,
            t,
            lgs + l as u64 * lstride,
            gg,
            gd,
            a(&b.out[1]),
        )?;
        hip.sync_stream(&rig.compute)?;
        let mut o0 = vec![0u8; n];
        let mut o1 = vec![0u8; n];
        d2h_prefix(hip, &rig.compute, &b.out[0], &mut o0)?;
        d2h_prefix(hip, &rig.compute, &b.out[1], &mut o1)?;
        gate.0 += 1;
        gate.1 += u64::from(o0 == o1);
        if o0 != o1 {
            eprintln!("GATE step {si} layer {l} T={t}: tier bytes DIFFER from the golden run");
        }
        if !control_done && local.len() >= 2 {
            // THE CONTROL: every routed expert pointed at its neighbour's record.
            for (k, &e) in local.iter().enumerate() {
                let base = rig.ring_addr((k + 1) % local.len());
                rig.override_entry(&mut gu, &mut dn, e as usize, base);
            }
            rig.xput(2, l, &gu, &dn);
            rig.xupload(b, 2, l, 1)?;
            moe.layer(
                hip,
                &rig.compute,
                b,
                l,
                t,
                lgs + l as u64 * lstride,
                gg,
                gd,
                a(&b.out[1]),
            )?;
            hip.sync_stream(&rig.compute)?;
            d2h_prefix(hip, &rig.compute, &b.out[1], &mut o1)?;
            gate.2 += 1;
            gate.3 += u64::from(o0 != o1);
            control_done = true;
        }
    }
    Ok(())
}

fn d2h_prefix(hip: &HipThread, st: &Stream, src: &DevicePtr, dst: &mut [u8]) -> anyhow::Result<()> {
    hip.memcpy_d2h_blocking(dst, src, st)?;
    Ok(())
}

// ------------------------------------------------------------------------------------------- setup

fn mem_total() -> anyhow::Result<u64> {
    let m = std::fs::read_to_string("/proc/meminfo")?;
    let kb: u64 = m
        .lines()
        .find(|l| l.starts_with("MemTotal:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .context("MemTotal")?
        .parse()?;
    Ok(kb * 1024)
}

/// The budget, the host budget and the flags, resolved the way the engine's boot will.
pub struct Plan {
    pub budget: Ep2Budget,
    pub host: HostBudget,
    pub law: TierLaw,
}

pub fn plan(pack: &Path, rank: u32, flags: &XtierFlags) -> anyhow::Result<Plan> {
    let pj = std::fs::read_to_string(pack.join("plan.json"))
        .with_context(|| format!("{}/plan.json", pack.display()))?;
    let base = Ep2Budget::plan(
        rank,
        30_500 << 20,
        weights_from_plan(&pj, rank)?.0,
        RuntimeParams::default().classes(),
        geometry::EXPERT_STRIDE,
        geometry::EXPERT_RECORDS,
    )?;
    let ledger_slots = base.slots as u32;
    let ring = flags
        .resolve(u32::MAX / 2, 0, false, NARROW_WIDTH)
        .ok()
        .and_then(|l| l.stage)
        .map_or(0, |s| s.ring_slots());
    let host = HostBudget::this_box(mem_total()?);
    let complement = geometry::EXPERT_RECORDS - u64::from(ledger_slots - ring.min(ledger_slots));
    let l2_default = host.default_l2_slots(complement, geometry::EXPERT_STRIDE);
    let law = flags
        .resolve(ledger_slots, l2_default as u32, false, NARROW_WIDTH)
        .map_err(anyhow::Error::msg)?;
    let budget = base.with_stage_ring(u64::from(ring))?;
    Ok(Plan { budget, host, law })
}

fn boot_lines(p: &Plan, tier: &ExpertTier) {
    println!("{}", p.budget.boot_line());
    println!(
        "{}",
        p.host.boot_line(
            u64::from(p.law.tier.ram_slots),
            geometry::EXPERT_RECORDS - u64::from(p.law.tier.vram_slots),
            geometry::EXPERT_STRIDE
        )
    );
    println!("{}", tier.boot_line());
    println!("{}", p.law.boot_line(tier.rank()));
}

/// `boot` mode (no GPU): the boot lines and the exposition the engine will serve, over a short
/// host replay (counting mover) so the counters are not all zero.
pub fn boot(args: &BenchArgs) -> anyhow::Result<()> {
    let p = plan(&args.pack, args.rank, &args.flags)?;
    let xt = XtierFile::open(
        &args.pack.join(format!("rank{}/experts.xtier", args.rank)),
        OpenMode::Direct,
    )?;
    let prof = load_profile(args)?;
    let mut tier = ExpertTier::new(xt.header(), p.law.tier.clone(), prof.as_ref(), 0, 0)?;
    boot_lines(&p, &tier);
    let routes = load_routes(&args.routes)?;
    let steps = trace(&routes, "conc8")?;
    let mut mv = saber_mem::expert_mover::DrainMover::new(&xt, 4);
    mv.read = false;
    for s in steps.iter().take(args.steps.max(1)) {
        tier.begin_step();
        for (l, ids) in s.ids.iter().enumerate() {
            tier.resolve(l as u32, ids, &mut mv)?;
        }
        tier.end_step(&mut mv)?;
    }
    let metrics = saber_serve::metrics::Metrics::new("Qwen3.8-Flash-Next");
    let text = metrics.render(saber_serve::engine::EngineStats {
        memory_plane: tier.render_prometheus(),
        ..Default::default()
    });
    for l in text.lines().filter(|l| {
        l.starts_with("saber_xtier") && !l.starts_with("saber_xtier_layer")
            || l.starts_with("saber:num_requests_running")
    }) {
        println!("{l}");
    }
    Ok(())
}

fn load_profile(args: &BenchArgs) -> anyhow::Result<Option<RouteProfile>> {
    if args.flags.no_xtier_profile {
        return Ok(None);
    }
    let p = args
        .flags
        .xtier_profile
        .clone()
        .unwrap_or_else(|| args.profile.clone());
    Ok(Some(RouteProfile::from_json(&std::fs::read_to_string(
        &p,
    )?)?))
}

// --------------------------------------------------------------------------------------------- run

pub fn run(hip: &HipThread, args: &BenchArgs) -> anyhow::Result<()> {
    if args.mode == "boot" {
        return boot(args);
    }
    hip.set_device(args.device)?;
    crate::gpu_idle::require_default(hip, args.device, 24 << 30)?;
    let p = plan(&args.pack, args.rank, &args.flags)?;
    let xt = XtierFile::open(
        &args.pack.join(format!("rank{}/experts.xtier", args.rank)),
        OpenMode::Direct,
    )?;
    let h = xt.header().clone();
    let prof = load_profile(args)?;
    let law = p.law.clone();
    let tp = law.tier.clone();
    // Kernels.
    let lc = LoadCounters::new();
    let plugins = PluginSet::load_named(
        hip,
        &args.prebuilt,
        &["moe_route", "moe_grouped", "flashnext_hc", "dflash_bgemm"],
        &lc,
    )?;
    let table = plugins.functions(hip)?;
    let moe = Moe::new(&table)?;
    let compute = hip.create_stream(StreamFlags::NonBlocking)?;
    // The tier's memory: allocated ONCE (I1).
    let t_alloc = Instant::now();
    let q = HipCopyQueue::new(
        hip,
        h.stride,
        tp.vram_slots,
        saber_mem::expert_tier::ExpertTier::slab_slots(&tp) - 1,
        h.layers,
    )?;
    let identity = q.ram_device_addr == q.ram.addr() as u64;
    println!(
        "# arena {:.2} GiB vram + {:.2} GiB pinned in {:.1} s; pinned UVA address {} the host address",
        q.vram.len() as f64 / (1u64 << 30) as f64,
        q.ram.len() as f64 / (1u64 << 30) as f64,
        t_alloc.elapsed().as_secs_f64(),
        if identity { "==" } else { "!=" }
    );
    let vram_base = q.vram_base();
    let ram_base = q.ram_device_addr;
    let mover = DeviceMover::new(&xt, q, 32).with_defer(tp.defer_spares > 0);
    let tmax = *args.tokens.iter().max().context("--tokens")?;
    let stage_slots = law
        .stage
        .map_or(args.flags.stage_slots(NARROW_WIDTH), |s| s.slots);
    let mut rig = Rig {
        hip,
        xt: &xt,
        mover,
        stride: h.stride,
        sub: h.sub,
        ring: hip.malloc(LOC as u64 * h.stride)?,
        gstage: hip.host_malloc(64 * h.stride)?,
        stage_ring: if stage_slots > 0 && args.mode == "stage" {
            Some(hip.malloc(2 * u64::from(stage_slots) * h.stride)?)
        } else {
            None
        },
        stage_routed: false,
        stage_ev: (0..LAYERS)
            .map(|_| hip.create_event(EventFlags::DisableTiming))
            .collect::<Result<_, _>>()?,
        consumed: [
            hip.create_event(EventFlags::DisableTiming)?,
            hip.create_event(EventFlags::DisableTiming)?,
        ],
        xhost: hip.host_malloc(4 * LAYERS as u64 * 2 * XBYTES)?,
        compute,
        cur_vram: vec![NONE; tp.vram_slots as usize],
        cur_ram: vec![NONE; tp.ram_slots as usize + 1],
    };
    let b = Bufs::new(hip, &rig.compute, tmax, &args.pack)?;
    let mut ck = Clocks::new(hip)?;
    let base_tier = ExpertTier::new(&h, tp.clone(), prof.as_ref(), vram_base, ram_base)?;
    boot_lines(&p, &base_tier);
    let t_warm = Instant::now();
    let (wv, wr) = rig.restore(&base_tier)?;
    println!(
        "# warm: {wv} vram + {wr} ram records from NVMe in {:.1} s",
        t_warm.elapsed().as_secs_f64()
    );
    let routes = load_routes(&args.routes)?;
    let fresh = |params: &TierParams| {
        ExpertTier::new(&h, params.clone(), prof.as_ref(), vram_base, ram_base)
    };
    let never = |_: usize| false;
    let mut nogate = (0, 0, 0, 0);
    match args.mode.as_str() {
        "gate" => {
            let mut total = (0u64, 0u64, 0u64, 0u64);
            let mut slot_bad = 0;
            for &t in &args.tokens {
                let mut steps = trace(&routes, &trace_for(t))?;
                steps.truncate(args.steps.max(1));
                let n = steps.len();
                let lg = logits_for(hip, &rig.compute, &steps)?;
                let mut tier = fresh(&tp)?;
                rig.restore(&tier)?;
                let cks: Vec<usize> = [n / 3, 2 * n / 3, n - 1]
                    .into_iter()
                    .filter(|&x| t < 512 || x == n - 1)
                    .collect();
                let mut g = (0, 0, 0, 0);
                let rs = run_path(
                    &mut rig,
                    &mut tier,
                    &moe,
                    &b,
                    &lg,
                    &steps,
                    0,
                    Way::Tier,
                    None,
                    &mut ck,
                    &|i| cks.contains(&i),
                    &mut g,
                )?;
                let (checked, bad) = rig.verify_slots(&mut tier)?;
                slot_bad += bad;
                println!(
                    "GATE T={t}: {n} steps through the mover ({} inserts, {} demotions): layer outputs \
                     IDENTICAL to the golden run {}/{} at steps {cks:?}; neighbour control DIFFERS {}/{}; \
                     slots == sha256 {}/{checked}",
                    rs.inserts,
                    rs.demotions,
                    g.1,
                    g.0,
                    g.3,
                    g.2,
                    checked - bad
                );
                total = (total.0 + g.0, total.1 + g.1, total.2 + g.2, total.3 + g.3);
            }
            let ok = total.0 == total.1 && total.2 == total.3 && slot_bad == 0 && total.0 > 0;
            println!(
                "MOVER BYTE GATE: {} ({}/{} identical, {}/{} controls caught, {slot_bad} bad slots)",
                if ok { "PASS" } else { "FAIL" },
                total.1,
                total.0,
                total.3,
                total.2
            );
            if !ok {
                bail!("the mover byte gate failed");
            }
        }
        "paths" => {
            for &t in &args.tokens {
                let mut steps = trace(&routes, &trace_for(t))?;
                let mut warm = args.warmup;
                if t >= 512 {
                    // One chunk, repeated: 2 warm-up + 3 timed (its logits are 100 MB a copy).
                    let one = steps[0].clone();
                    steps = vec![one; 5];
                    warm = 2;
                }
                steps.truncate(args.steps.max(1) + warm);
                let warm = warm.min(steps.len() / 4);
                let lg = logits_for(hip, &rig.compute, &steps)?;
                let warm = warm.min(steps.len() - 1);
                let mut out = Vec::new();
                for (name, path, params) in [
                    ("all-vram", Way::Control, tp.clone()),
                    ("tier", Way::Tier, tp.clone()),
                    ("tier-serial", Way::Serial, tp.clone()),
                    (
                        "tier-noinsert",
                        Way::Tier,
                        TierParams {
                            read_through: 0.0,
                            ..tp.clone()
                        },
                    ),
                    (
                        "demand-1024",
                        Way::Demand,
                        TierParams {
                            ram_slots: 1024.min(tp.ram_slots),
                            ..tp.clone()
                        },
                    ),
                ] {
                    if t >= 512 && path == Way::Demand {
                        continue;
                    }
                    let mut tier = fresh(&params)?;
                    rig.restore(&tier)?;
                    let rs = run_path(
                        &mut rig,
                        &mut tier,
                        &moe,
                        &b,
                        &lg,
                        &steps,
                        warm,
                        path,
                        None,
                        &mut ck,
                        &never,
                        &mut nogate,
                    )?;
                    println!("{}", rs.line(name, t));
                    out.push((name, rs));
                }
                // The forced-miss first step: every routed record from NVMe.
                let mut tier = fresh(&tp)?;
                rig.restore(&tier)?;
                tier.evict_all();
                let first = &steps[..3.min(steps.len())];
                let rs = run_path(
                    &mut rig,
                    &mut tier,
                    &moe,
                    &b,
                    &lg,
                    first,
                    0,
                    Way::Demand,
                    None,
                    &mut ck,
                    &never,
                    &mut nogate,
                )?;
                let distinct: usize = first[0]
                    .ids
                    .iter()
                    .map(|v| {
                        let mut x: Vec<u32> =
                            v.iter().filter(|&&e| (e as usize) < LOC).copied().collect();
                        x.sort_unstable();
                        x.dedup();
                        x.len()
                    })
                    .sum();
                println!(
                    "path forced-miss    T={t:<5} first step {:.1} ms (host wall, NVMe on every layer) | \
                     nvme {} == distinct routed local records of step 0 ({distinct})? then {} steps: {}",
                    rs.first_step_ms,
                    rs.first_step_nvme,
                    first.len(),
                    rs.line("forced", t)
                );
                let (checked, bad) = rig.verify_slots(&mut tier)?;
                println!(
                    "slots after forced-miss: {}/{checked} == sha256",
                    checked - bad
                );
                if let Some((_, c)) = out.iter().find(|(n, _)| *n == "all-vram") {
                    for (n, r) in &out {
                        println!(
                            "T={t} {n:<12} compute {:+.3} ms/step ({:+.1}%), wall {:.3} ms/step, step cost {:.3} \
                             ms (wall + the drained copies for the serial control) vs all-vram compute {:.3}",
                            r.per(r.compute_ms) - c.per(c.compute_ms),
                            100.0 * (r.per(r.compute_ms) / c.per(c.compute_ms).max(1e-9) - 1.0),
                            r.per(r.wall_ms),
                            r.per(r.wall_ms)
                                + if *n == "tier-serial" {
                                    r.copy_ms / r.copy_steps.max(1) as f64
                                } else {
                                    0.0
                                },
                            c.per(c.compute_ms)
                        );
                    }
                }
            }
        }
        "sweep" => {
            let t0 = args.tokens[0];
            let mut steps = trace(&routes, &trace_for(t0))?;
            if args.steps > 1 {
                steps.truncate(args.warmup + args.steps);
            }
            let lg = logits_for(hip, &rig.compute, &steps)?;
            let warm = args.warmup.min(steps.len() / 2);
            let mut cfgs = vec![(64u32, 0.5f64)];
            if args.sweep == "inserts" {
                for mi in [0u32, 1, 2, 4, 8] {
                    cfgs.push((mi, 0.5));
                }
            } else {
                for mi in [0u32, 8, 16, 32, 64] {
                    for rt in [0.25, 0.5, 1.0] {
                        cfgs.push((mi, rt));
                    }
                }
            }
            cfgs.push((64, 0.5));
            for (mi, rt) in cfgs {
                let params = TierParams {
                    // max_inserts 0 = no inserts: the tier refuses 0, so read-through 0 stands in.
                    max_inserts: mi.max(1),
                    read_through: if mi == 0 { 0.0 } else { rt },
                    ..tp.clone()
                };
                let mut tier = fresh(&params)?;
                rig.restore(&tier)?;
                let rs = run_path(
                    &mut rig,
                    &mut tier,
                    &moe,
                    &b,
                    &lg,
                    &steps,
                    warm,
                    Way::Tier,
                    None,
                    &mut ck,
                    &never,
                    &mut nogate,
                )?;
                println!(
                    "sweep max-inserts {mi:<2} read-through {rt:.2}: {}",
                    rs.line("tier", t0)
                );
            }
        }
        "stage" => {
            let law0 = law.stage.unwrap_or(StageLaw {
                above_rows: args.flags.stage_min_rows(NARROW_WIDTH),
                slots: args.flags.stage_slots(NARROW_WIDTH),
                buffers: 2,
            });
            for &t in &args.tokens {
                let steps = trace(&routes, &trace_for(t))?;
                let mut steps: Vec<Step> = steps.into_iter().take(3).collect();
                let reps = args.warmup + args.steps.max(1);
                while steps.len() < reps {
                    steps.push(steps[steps.len() % 3.min(steps.len())].clone());
                }
                let lg = logits_for(hip, &rig.compute, &steps)?;
                let forced = StageLaw {
                    above_rows: 1,
                    ..law0
                };
                let mut tier = fresh(&tp)?;
                rig.restore(&tier)?;
                let mut res = Vec::new();
                for (name, path) in [
                    ("uva", Way::Tier),
                    ("staged", Way::Staged),
                    ("staged-routed", Way::Staged),
                    ("all-vram", Way::Control),
                ] {
                    rig.stage_routed = name == "staged-routed";
                    // Frozen tier (no end_step) for a paired placement: Tier here = UVA in place.
                    let rs = run_path(
                        &mut rig,
                        &mut tier,
                        &moe,
                        &b,
                        &lg,
                        &steps,
                        args.warmup,
                        if path == Way::Tier { Way::Staged } else { path },
                        if path == Way::Tier {
                            None
                        } else {
                            Some(forced)
                        },
                        &mut ck,
                        &never,
                        &mut nogate,
                    )?;
                    println!("{}", rs.line(name, t));
                    res.push((name, rs));
                }
                // Bytes: staged == UVA, every layer of one chunk.
                let (same, n) = stage_bytes(&mut rig, &mut tier, &moe, &b, &lg, &steps[0], forced)?;
                println!(
                    "STAGE GATE T={t}: staged layer outputs IDENTICAL to UVA in place {same}/{n} \
                     (ring 2 x {} slots)",
                    law0.slots
                );
                let u = res[0].1.per(res[0].1.wall_ms);
                let s = res[1].1.per(res[1].1.wall_ms);
                println!(
                    "STAGE T={t}: chunk wall uva {u:.2} ms, staged {s:.2} ms ({:+.1}%), all-vram compute {:.2} ms",
                    100.0 * (s / u - 1.0),
                    res[3].1.per(res[3].1.compute_ms)
                );
                let o = res[2].1.per(res[2].1.wall_ms);
                println!(
                    "STAGE T={t}: oracle (routed-only) staging {o:.2} ms ({:+.1}% vs uva)",
                    100.0 * (o / u - 1.0)
                );
                rig.stage_routed = false;
            }
        }
        "ple" => {
            let tier = fresh(&tp)?;
            rig.restore(&tier)?;
            ple_card(&mut rig, &tier, &moe, &b, args, &routes)?;
        }
        m => bail!("unknown mode {m}"),
    }
    Ok(())
}

/// The staging byte law: every layer of `s` run over UVA in place and over the staging ring,
/// outputs compared. Returns (identical, layers).
fn stage_bytes(
    rig: &mut Rig<'_, '_>,
    tier: &mut ExpertTier,
    moe: &Moe<'_>,
    b: &Bufs,
    lg: &Logits,
    s: &Step,
    law: StageLaw,
) -> anyhow::Result<(u64, u64)> {
    let hip = rig.hip;
    let t = s.tokens;
    let n = (u64::from(t) * u64::from(H) * 2) as usize;
    let lstride = u64::from(t) * 512 * 2;
    let lgs = a(&lg.buf) + lg.off[0];
    let mut same = 0;
    for l in 0..LAYERS {
        let (gu, dn) = rig.tables(tier, b, l);
        rig.xput(0, l, &gu, &dn);
        rig.xupload(b, 0, l, 1)?;
        let (xg, xd) = rig.xdev(b, 0, l);
        moe.layer(
            hip,
            &rig.compute,
            b,
            l,
            t,
            lgs + l as u64 * lstride,
            xg,
            xd,
            a(&b.out[0]),
        )?;
        // Staged: this layer's RAM experts into ring half l % 2, tables pointed there.
        let (mut sg, mut sd) = (gu.clone(), dn.clone());
        let mut k = 0usize;
        for e in 0..LOC as u32 {
            if k >= law.slots as usize {
                break;
            }
            if let Some(Placement::Ram(_)) = tier.placement(l as u32, e) {
                let base = a(rig.stage_ring.as_ref().context("ring")?)
                    + ((l % 2) * law.slots as usize + k) as u64 * rig.stride;
                rig.override_entry(&mut sg, &mut sd, e as usize, base);
                k += 1;
            }
        }
        rig.xput(3, l, &sg, &sd);
        rig.xupload(b, 3, l, 1)?;
        hip.sync_stream(&rig.compute)?;
        stage_issue(rig, tier, l, law, None)?;
        hip.stream_wait_event(&rig.compute, &rig.stage_ev[l])?;
        let (yg, yd) = rig.xdev(b, 3, l);
        moe.layer(
            hip,
            &rig.compute,
            b,
            l,
            t,
            lgs + l as u64 * lstride,
            yg,
            yd,
            a(&b.out[1]),
        )?;
        hip.record_event(&rig.consumed[l % 2], &rig.compute)?;
        hip.sync_stream(&rig.compute)?;
        let mut o0 = vec![0u8; n];
        let mut o1 = vec![0u8; n];
        d2h_prefix(hip, &rig.compute, &b.out[0], &mut o0)?;
        d2h_prefix(hip, &rig.compute, &b.out[1], &mut o1)?;
        same += u64::from(o0 == o1);
    }
    Ok((same, LAYERS as u64))
}

/// The PLE staged buffer on the card: gather (cold, warm) into pinned staging, H2D on the copy
/// stream, alone and beside a 2,048-token MoE layer on the compute stream.
fn ple_card(
    rig: &mut Rig<'_, '_>,
    tier0: &ExpertTier,
    moe: &Moe<'_>,
    b: &Bufs,
    args: &BenchArgs,
    routes: &[(usize, Vec<i32>)],
) -> anyhow::Result<()> {
    use saber_mem::ple_tier::{PleGate, PleIndex, PleSourceKind, PleTier};
    let hip = rig.hip;
    let idx = PleIndex::from_json(
        &std::fs::read_to_string(args.pack.join("ple/ple_index.json"))?,
        PleSourceKind::CkptBf16,
        PleGate::Default,
        args.ckpt_root.as_deref(),
    )?;
    let raw = std::fs::read(&args.ple_ids)?;
    let ids: Vec<u64> = raw
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().expect("8")))
        .collect();
    let t = 2048usize;
    let chunk: Vec<u64> = (0..t * 16).map(|i| ids[i % ids.len()]).collect();
    let law = &args.flags;
    let mut tier = PleTier::new(
        idx,
        OpenMode::Direct,
        law.ple_qd_prefill,
        ((law.ple_cache_mib << 20) / 320) as u32,
    )?;
    println!("{}", tier.boot_line());
    let bytes = t * 16 * 320;
    let mut staged = hip.host_malloc(bytes as u64)?;
    let dev = hip.malloc(bytes as u64)?;
    let steps = trace(routes, "chunk2048")?;
    let lg = logits_for(hip, &rig.compute, &steps[..1])?;
    let (xg, xd) = rig.xdev(b, 0, 0);
    let (gu, dn) = rig.tables(tier0, b, 0);
    rig.xput(0, 0, &gu, &dn);
    rig.xupload(b, 0, 0, 1)?;
    let e0 = hip.create_event(EventFlags::Default)?;
    let e1 = hip.create_event(EventFlags::Default)?;
    let c0 = hip.create_event(EventFlags::Default)?;
    let c1 = hip.create_event(EventFlags::Default)?;
    let layer = |hip: &HipThread| -> anyhow::Result<()> {
        moe.layer(
            hip,
            &rig.compute,
            b,
            0,
            t as u32,
            a(&lg.buf),
            xg,
            xd,
            a(&b.out[0]),
        )
    };
    // The layer alone.
    let mut alone = Vec::new();
    for _ in 0..5 {
        hip.record_event(&e0, &rig.compute)?;
        layer(hip)?;
        hip.record_event(&e1, &rig.compute)?;
        hip.sync_event(&e1)?;
        alone.push(f64::from(hip.elapsed_ms(&e0, &e1)?));
    }
    alone.sort_by(f64::total_cmp);
    println!(
        "ple: layer-0 MoE train at T=2048 alone {:.2} ms (median of 5)",
        alone[2]
    );
    let q = rig.mover.queue();
    for (name, cold) in [
        ("cold", true),
        ("warm", false),
        ("cold", true),
        ("warm", false),
    ] {
        if cold {
            tier.evict_all();
        }
        // Beside the layer: the compute stream runs the layer while the host gathers and the copy
        // stream carries the buffer.
        let w0 = Instant::now();
        hip.record_event(&e0, &rig.compute)?;
        layer(hip)?;
        hip.record_event(&e1, &rig.compute)?;
        let g0 = Instant::now();
        let c = tier.gather(&chunk, staged.as_mut_slice())?;
        let gather_ms = g0.elapsed().as_secs_f64() * 1e3;
        hip.run_batch(&[
            Job::RecordEvent {
                event: &c0,
                stream: &q.stream,
            },
            Job::MemcpyH2d {
                dst: &dev,
                dst_offset: 0,
                src: staged.slice(0, bytes).context("staged slice")?,
                stream: &q.stream,
            },
            Job::RecordEvent {
                event: &c1,
                stream: &q.stream,
            },
        ])?;
        hip.sync_event(&c1)?;
        hip.sync_event(&e1)?;
        let wall = w0.elapsed().as_secs_f64() * 1e3;
        let h2d = f64::from(hip.elapsed_ms(&c0, &c1)?);
        let lay = f64::from(hip.elapsed_ms(&e0, &e1)?);
        println!(
            "ple {name}: chunk 2048 x 16 rows: gather {gather_ms:.2} ms ({} distinct, {} cache hits, {} nvme reads, \
             read {:.2} ms) + H2D {h2d:.3} ms ({:.1} MB) | beside the layer ({lay:.2} ms on the card): wall {wall:.2} ms \
             = {:.0}% of gather+h2d hidden",
            c.distinct,
            c.cache_hits,
            c.nvme_reads,
            c.read_s * 1e3,
            bytes as f64 / 1e6,
            100.0 * ((gather_ms + h2d + lay) - wall).max(0.0) / (gather_ms + h2d).max(1e-9)
        );
    }
    // Bytes: the H2D'd buffer == the host gather.
    let mut back = vec![0u8; bytes];
    hip.memcpy_d2h_blocking(&mut back, &dev, &rig.compute)?;
    println!(
        "PLE STAGED BUFFER GATE: device bytes == host gather: {}",
        if back == staged.as_slice()[..bytes] {
            "IDENTICAL"
        } else {
            "DIFFER"
        }
    );
    Ok(())
}
