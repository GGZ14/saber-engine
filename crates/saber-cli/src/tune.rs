//! `saber tune` — the P0 autotune sweep: turn `saber explain`'s untuned work list into bindings.
//!
//! > No cost model. Constraint match → autotune lookup → **error if not tuned**.
//! >
//! > — ARCHITECTURE.md, *Cuts*
//!
//! `saber explain` with a cold cache prints 44 untuned routes and refuses to produce a plan. This
//! is the command that answers them, and every step of it is one of the rules the architecture
//! states as a measurement rather than a preference:
//!
//! 1. **Resolve, then take [`Resolved::missing`]** — the sweep measures exactly what the planner
//!    asked for, at the keys the planner will look up. Nothing else is measured, and a key is
//!    never invented here.
//! 2. **Enumerate the variants from the rules** ([`saber_exec::enumerate_variants`]), because one
//!    implementation row covers many instantiations and the tie between two rows that both match
//!    is a measurement, never registry order.
//! 3. **Interleave** the variants A B C A B C …, so drift over the sweep lands on all of them.
//! 4. **Discard the shape's warm-up** — handed to [`Autotuner::record_measurements`], which owns
//!    that protocol, because the first-measured variant per shape reads ~17% inflated.
//! 5. **Refuse to write a record unless both cards read the manifest's power cap.** A record
//!    measured uncapped is invalid at the enforced 210 W, and a MODE1 reset clears the policy
//!    silently.
//!
//! GPU rules (`CLAUDE.md`): prod serves on both cards. Run on HIP device 1 only, under
//! the card's lock (`~/saber/.cache/gpu.lock.<card>`, via `tools/capture/gpu_idle.sh`), with prod's `num_requests_running` at 0, every allocation inside
//! `--budget-mib`, and never touch the power cap.
//!
//! **Absolute microseconds here are measured beside a serving production engine.** They are not
//! comparable across runs and they are not a kernel benchmark; what the interleaving makes
//! meaningful is the *ranking within one shape*, which is the only thing a record decides.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use saber_exec::{
    enumerate_variants, variant_id_of, LaunchCounters, LaunchCtx, Launcher, LoadCounters,
    PluginSet, QkCell, ShapeOutcome, SweepConfig, Sweeper, SymbolIndex, TuneOp, VariantSamples,
    VariantSet,
};
use saber_hip::{HipThread, StreamFlags};
use saber_kernels::glue::{GateShape, KvShape, QkShape};
use saber_kernels::rules::{resolve_with_layout, ALayout, LaunchRules, Op, Shape};
use saber_kernels::saber::{
    self, AttnShape, GdnPrefillShape, GdnShape, KvDtype, NormShape, SaberRules,
};
use saber_model::graph::Phase;
use saber_plan::{
    check_plugins_declared, explain, registry_from_prebuilt, AutotuneCache, AutotuneKey, Autotuner,
    DeviceProfile, Manifest, Measurement, ModelSpec, Provenance, Registry, Requirement, Resolved,
    VariantParams,
};

/// The epsilon the reference capture feeds the fused norm producers. It scales no loop bound and
/// selects no instantiation, so it cannot move a measurement; it is here so the descriptor is
/// complete rather than defaulted.
const PRODUCER_EPS: f32 = 1e-6;

/// What every record this command writes says about itself.
const HARNESS: &str = "saber-tune v0 interleaved hipEvent";

/// The smallest IPC slot `saber tune` ever asks `ar-gate` for.
///
/// Prod's is 84 MiB and two of them plus the local pack are allocated **per rank**, which is
/// 168 MiB of peer-visible memory on cards that are serving. Every message a *decode* bucket
/// all-reduces is under the 128 KiB compression threshold, so 4 MiB is ample for a decode-only
/// plan and the slot size does not change which kernel runs.
///
/// A prefill bucket is a different message entirely -- 20 MiB at 2048 tokens, 80 MiB at 8192 --
/// and the one-shot kernel's launch rule **refuses** a message larger than the slot, because
/// past that size prod falls back to RCCL. A slot fixed at 4 MiB therefore does not merely make
/// the prefill all-reduce slow to measure, it makes it unmeasurable, and it reports that as a
/// refusal rather than as a number. So the floor is a floor: [`ar_slot_bytes`] sizes the request
/// from the largest message the plan actually carries.
const AR_SLOT_FLOOR_BYTES: u64 = 4 << 20;

/// The IPC slot and per-rank budget the all-reduce cases need, in bytes and MiB.
///
/// The slot must hold the largest message or the kernel's own launch rule refuses it; the gate's
/// budget covers three buffers of that size (local, peer, reference).
fn ar_slot_and_budget(max_message_bytes: u64) -> (u64, u64) {
    let mib = 1 << 20;
    let slot = max_message_bytes.div_ceil(mib) * mib;
    let slot = slot.max(AR_SLOT_FLOOR_BYTES);
    let budget_mib = ((max_message_bytes * 3).div_ceil(mib)).max(AR_BUDGET_FLOOR_MIB);
    (slot, budget_mib)
}

/// The smallest per-rank budget `ar-gate` is ever asked for.
const AR_BUDGET_FLOOR_MIB: u64 = 128;

/// Warm-up rounds for the all-reduce timing.
///
/// Higher than the sweep's 5, and measured rather than chosen: the FIRST case in an `ar-gate`
/// process also pays the one-off IPC setup (peer access enable, first touch of the imported
/// region), and at 5 it read 18.4 us against the 12.2 us the same case gives once that is behind
/// it. This is the sweep's "first-measured reads inflated" rule at the process level.
const AR_WARMUP: usize = 12;

/// What a two-rank all-reduce record says took the measurement. Never compare across harnesses.
///
/// The GPU condition is appended the same way the sweep's own harness string carries it: an
/// idle card and a card sharing a 210 W cap with a server are not the same instrument, and an
/// 80 MiB prefill all-reduce at 1.4 ms/call is exactly the kind of number that moves with it.
const AR_HARNESS: &str = "saber tune via ar-gate (2 ranks)";

/// What else was running on the cards while this sweep measured.
///
/// It goes in every record's provenance because it changes what the number means: a kernel timed
/// beside a serving engine shares its memory system, its power budget and its queue, and one
/// timed on an idle box does not. `LEDGER.md`'s absolute microseconds are qualified by exactly
/// this, and a cache that does not carry it invites a comparison across the two.
#[derive(Debug, Clone, Copy, PartialEq)]
enum GpuState {
    /// The engine answered and had nothing in flight.
    ProdIdle,
    /// Nothing answered on the metrics port: prod is down and the cards are ours.
    ProdDown,
    /// `--no-prod-check`: not looked at. Asserted, not observed, and the record says so.
    NotChecked,
}

impl GpuState {
    /// The phrase that goes in the harness string.
    fn label(self) -> &'static str {
        match self {
            GpuState::ProdIdle => "GPU idle (prod serving, no requests in flight)",
            GpuState::ProdDown => "GPU idle (prod down)",
            // Asserted, not observed. It reads differently on purpose: a record that says this
            // is a record nobody checked the conditions for.
            GpuState::NotChecked => "GPU state NOT CHECKED (--no-prod-check)",
        }
    }
}

/// Why the all-reduce carries no measurement, when it carries none.
///
/// The sweep process itself cannot time it -- the 2-rank all-reduce needs a peer process,
/// IPC-exported workspaces and a peer's flags to spin on -- so it is measured out of process by
/// `saber ar-gate --timing`. A record carrying THIS string is one that pass did not cover:
/// `--no-ar-timing`, or a case the gate did not return.
const AR_NOT_MEASURED: &str = "not timed in this run: the 2-rank all-reduce is measured out of \
     process by `saber ar-gate --timing`, which did not cover this case";

/// Linear value heads per linear key head: 48 / 16 in `config.json`, and the ratio survives the
/// TP divide. The model spec carries the value-head count as the op's `n`; this is what turns it
/// back into the pair the GDN kernels take.
const GDN_V_PER_K_HEADS: u32 = 3;

/// The prefill attention's OPT mode: which of the QK and PV legs run in 8 bits.
///
/// 3 = both, which is what the serving unit sets (`SABER_ATTN_FP8=3`, `podman inspect prodnvfp4`,
/// quoted in the rules file's header). It is a **numerics** choice and not a tile, so the sweep
/// binds the tier's value and does not let a stopwatch pick it; the other three modes are timed
/// beside it because "what does the 8-bit leg buy" is worth an answer.
const ATTN_PREFILL_FP8_MODE: u32 = 3;

/// The GDN prefill chunk, from the rules file's `[gdn] chunk`.
const GDN_PREFILL_CHUNK: u32 = 64;

/// `SOFTPLUS_THRESHOLD`, as the reference capture script (`the internal reference harness`) passes it.
const SOFTPLUS_THRESHOLD: f32 = 20.0;

/// Arguments of `saber tune`.
pub struct TuneArgs {
    /// The Saber manifest whose plan is being tuned.
    pub manifest: PathBuf,
    /// The committed code objects and their manifests.
    pub prebuilt: PathBuf,
    /// Where to write the cache.
    pub out: PathBuf,
    /// An existing cache to extend. Routes it already answers are not re-measured.
    pub autotune_in: Option<PathBuf>,
    /// Only sweep routes whose model op name contains this substring.
    pub only: Option<String>,
    /// Untimed rounds before timing starts. Each round launches every variant once.
    pub warmup: usize,
    /// Timed rounds. Each contributes one sample per variant.
    pub samples: usize,
    /// A shape whose inputs exceed this is skipped and reported. Prod owns the card.
    pub budget_mib: u64,
    /// HIP device ordinal within the visible set.
    pub device: i32,
    /// The cap in watts, when `/sys/class/drm/card*/device/hwmon/*/power1_cap` is not readable
    /// from inside the container. Read it on the host and pass it; it is then asserted, not
    /// measured, and the run says so.
    pub power_cap_w: Option<u32>,
    /// `host:port` of the production engine's metrics endpoint, checked idle before any timing.
    pub metrics_addr: String,
    /// Do not check the production engine at all.
    ///
    /// For a box where prod is deliberately down: the check would report "no server" anyway, and
    /// saying so on the command line is the difference between a condition that was *observed*
    /// and one that was *assumed*. Either way the run prints which it was, and every record's
    /// provenance carries it.
    pub no_prod_check: bool,
    /// Enumerate and print the work list without touching a GPU.
    pub list: bool,
    /// Print every variant's median under each shape's row, not only the winner and runner-up.
    pub dump_variants: bool,
    /// The HIP runtime the two all-reduce rank processes load. Forwarded to `saber ar-gate`.
    pub hip_lib: String,
    /// Skip the two-rank all-reduce measurement and bind those routes without a number.
    ///
    /// For a box with one visible card, or a run that must not touch the second one.
    pub no_ar_timing: bool,
    /// Measure with the Infinity Cache HOT: one set of buffers, re-read every sample.
    ///
    /// This is the pre-2026-09-19 behaviour and it is kept for one purpose -- the paired A/B that
    /// shows which winners the hot cache was choosing. A record written this way says `IC-HOT` in
    /// its provenance, because it is not a measurement of what the step does.
    pub ic_hot: bool,
}

// ------------------------------------------------------------------ the power cap

/// One card's enforced power cap, and where the number came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PowerCap {
    /// `card0`, or `--power-cap-w` when the value was asserted rather than read.
    pub card: String,
    /// The file it was read from, if it was read.
    pub path: Option<PathBuf>,
    /// Watts.
    pub watts: u32,
}

/// Read `power1_cap` for every DRM card that has one. Microwatts on disk, watts here.
///
/// Cards without a `power1_cap` (the display iGPU) are simply absent from the result: a card that
/// cannot be capped is not a card a kernel was measured on.
pub fn read_power_caps(sysfs: &Path) -> Vec<PowerCap> {
    let mut out = Vec::new();
    let Ok(cards) = std::fs::read_dir(sysfs) else {
        return out;
    };
    let mut names: Vec<String> = cards
        .filter_map(std::result::Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        // `card0`, not `card0-DP-1`: the connectors are not devices.
        .filter(|n| n.starts_with("card") && n[4..].bytes().all(|b| b.is_ascii_digit()))
        .collect();
    names.sort();
    for name in names {
        let hwmon = sysfs.join(&name).join("device").join("hwmon");
        let Ok(entries) = std::fs::read_dir(&hwmon) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = entries
            .filter_map(std::result::Result::ok)
            .map(|e| e.path().join("power1_cap"))
            .filter(|p| p.exists())
            .collect();
        paths.sort();
        for path in paths {
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(uw) = text.trim().parse::<u64>() else {
                continue;
            };
            out.push(PowerCap {
                card: name.clone(),
                path: Some(path),
                // Rounded to the nearest watt: the policy is set in watts.
                watts: u32::try_from((uw + 500_000) / 1_000_000).unwrap_or(u32::MAX),
            });
        }
    }
    out
}

/// Refuse the run unless **both** cards are at the cap the plan's records will claim.
///
/// This is not belt-and-braces: the 210 W policy with −75 mV is re-applied by a timer, and a MODE1
/// GPU reset clears it silently. A sweep that ran uncapped would write records that are wrong by
/// ~15% on prefill shapes and would never be detectable again, because the key would still say
/// 210 W.
pub fn check_power_caps(caps: &[PowerCap], required_w: u32) -> Result<(), String> {
    if caps.len() < 2 {
        return Err(format!(
            "{} card(s) reported a power cap; this box has two, and a record is only valid if \
             both were capped at {required_w} W. Inside the container /sys may not be visible — \
             read the caps on the host and pass --power-cap-w",
            caps.len()
        ));
    }
    let wrong: Vec<String> = caps
        .iter()
        .filter(|c| c.watts != required_w)
        .map(|c| format!("{} at {} W", c.card, c.watts))
        .collect();
    if !wrong.is_empty() {
        return Err(format!(
            "the manifest's records are keyed at {required_w} W but {} — a measurement taken at a \
             different cap is not a measurement of this plan. Restore the fixed policy \
             (server-power-profile) before sweeping",
            wrong.join(", ")
        ));
    }
    Ok(())
}

// ------------------------------------------------------------------ prod idle

/// `num_requests_running`, from the shared check (`crate::gpu_idle`).
///
/// The counter alone is NOT an idle check — a loading server answers nothing and holds the whole
/// card — so `run` now prints the VRAM the card is holding beside it. See `docs/LEDGER.md`,
/// workstream H part 9.
fn prod_requests_running(addr: &str) -> Option<f64> {
    crate::gpu_idle::requests_running(addr)
}

// ------------------------------------------------------------------ the work list

/// One measurement the sweep owes: a key, the routes that share it, and what to launch.
#[derive(Debug)]
struct WorkItem {
    key: AutotuneKey,
    /// Every `(bucket, model op)` this one key binds. Routes collapse onto one key when they
    /// share a geometry — `o_proj` and `gdn.out_proj` are the same (n, k), so at a DECODE
    /// bucket they are the same measurement. At a prefill bucket they are not: `a_layout` is
    /// part of the key, and o_proj's edge is row-major where out_proj's is A-tiled.
    routes: Vec<String>,
    candidates: Vec<String>,
    work: Work,
    /// What the A-side edge of this route actually carries.
    ///
    /// The variant enumerator decides the GEMM family from `a_tiled(m, rules)` — the collapsed
    /// rule, one level down from the planner's. For a route whose producer is not fused that
    /// answer is wrong at a prefill M, and the sweep then offers an A-tiled variant no
    /// candidate implementation claims. It showed up exactly that way, as a hard error mid
    /// sweep rather than as a wrong measurement, which is the right failure mode.
    a_layout: ALayout,
}

/// What this key needs: a measurement, or a binding the harness cannot measure.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Work {
    /// Time it.
    Sweep(TuneOp),
    /// Bind it without a number. The reason is printed and stored in the record's provenance.
    Unmeasurable {
        /// The op to resolve for the rules' only legal variant.
        op: saber::Op,
        /// Its geometry.
        shape: saber::Shape,
        /// Why this route carries no measurement. Per route, because the reasons differ: the
        /// all-reduce needs a peer process, the prefill kernels need inputs this sweep has no
        /// builder for. A single blanket string would make the second look like the first.
        reason: &'static str,
    },
}

impl WorkItem {
    fn label(&self) -> String {
        let s = &self.key.shape;
        let num = |k: &str| s.num(k).unwrap_or(0);
        match self.work {
            Work::Sweep(TuneOp::Mxfp4 { shape, .. }) => {
                format!("{} n={} k={} M={}", self.key.op, shape.n, shape.k, shape.m)
            }
            Work::Sweep(TuneOp::AttnDecode(a)) => format!(
                "{} b={} q_len={} ctx={}",
                self.key.op, a.num_seqs, a.q_len, a.max_ctx
            ),
            Work::Sweep(TuneOp::GdnFused(g)) | Work::Sweep(TuneOp::GdnPair(g)) => {
                format!("{} b={} maxq={} h={}", self.key.op, g.n_seqs, g.maxq, g.h)
            }
            Work::Sweep(TuneOp::GlueQkNormRope(q)) => format!(
                "{} M={} nq={} nkv={} ctx={}",
                self.key.op, q.shape.m, q.shape.nq, q.shape.nkv, q.max_ctx
            ),
            Work::Sweep(TuneOp::GlueGateQuant(g)) => {
                format!("{} M={} n={}", self.key.op, g.m, g.n)
            }
            Work::Sweep(TuneOp::GlueKvAppend(k)) => {
                format!("{} T={} nkv={}", self.key.op, k.tokens, k.nkv)
            }
            Work::Sweep(TuneOp::AttnPrefill(a)) => {
                format!("{} chunk={} ctx={}", self.key.op, a.q_len, a.max_ctx)
            }
            Work::Sweep(TuneOp::GdnConvPrep(g))
            | Work::Sweep(TuneOp::GdnKktSolve(g))
            | Work::Sweep(TuneOp::GdnChunkScan(g)) => {
                format!("{} T={} h={}", self.key.op, g.t, g.h)
            }
            Work::Sweep(TuneOp::GdnNorm(n)) => {
                format!("{} rows={} ch={}", self.key.op, n.rows, n.channels)
            }
            // The all-reduce is sized by its message; the prefill routes by their chunk. One
            // label each, so the sweep's output names the geometry it bound rather than
            // printing `numel=0` for five different kernels.
            Work::Unmeasurable { .. } if self.key.op == "allreduce" => {
                format!("{} numel={} rows={}", self.key.op, num("numel"), num("m"))
            }
            Work::Unmeasurable { .. } => {
                format!("{} M={} n={}", self.key.op, num("m"), num("n"))
            }
        }
    }

    /// The `(n, k, m)` a work list is ordered by, so consecutive items share input buffers.
    fn order(&self) -> (u32, u32, u32) {
        match self.work {
            Work::Sweep(TuneOp::Mxfp4 { shape, .. }) => (shape.n, shape.k, shape.m),
            Work::Sweep(TuneOp::AttnDecode(a)) => (1 << 24, a.max_blocks, a.num_seqs),
            Work::Sweep(TuneOp::GdnFused(g)) => (1 << 25, g.maxq, g.n_seqs),
            Work::Sweep(TuneOp::GdnPair(g)) => (1 << 26, g.maxq, g.n_seqs),
            Work::Sweep(TuneOp::AttnPrefill(a)) => (1 << 27, a.max_blocks, a.q_len),
            // The trio share their big buffers, so they sort adjacent at one T and the pool
            // uploads q/k/v/g/beta once for all three.
            Work::Sweep(TuneOp::GdnConvPrep(g)) => (1 << 28, g.t, 1),
            Work::Sweep(TuneOp::GdnKktSolve(g)) => (1 << 28, g.t, 2),
            Work::Sweep(TuneOp::GdnChunkScan(g)) => (1 << 28, g.t, 3),
            // The glue last, and by ROW COUNT: the projection is the big buffer and it is the
            // only one the three share, so one M uploads it once for all three.
            Work::Sweep(TuneOp::GlueQkNormRope(q)) => (1 << 29, q.shape.m, 1),
            Work::Sweep(TuneOp::GlueGateQuant(g)) => (1 << 29, g.m, 2),
            Work::Sweep(TuneOp::GlueKvAppend(k)) => (1 << 29, k.tokens, 3),
            Work::Sweep(TuneOp::GdnNorm(n)) => (1 << 29, u32::try_from(n.rows).unwrap_or(0), 0),
            Work::Unmeasurable { .. } => (1 << 30, 0, 0),
        }
    }
}

/// The op and geometry a requirement's key names, as `saber_kernels::rules` spells them.
/// Ops this command can BIND — by timing them, or (for the all-reduce alone) by recording that
/// nobody timed them.
///
/// The MXFP4 family and every saber route the plan binds, decode and prefill alike: the paged
/// KV cache, the block table, the conv-state cache, the fp32 recurrent state, the per-chunk
/// cumsum and the chunked gram are all built by [`saber_exec::tune`] at the shapes
/// the reference capture script (`the internal reference harness`) uses. The all-reduce is the one exception and cannot be
/// otherwise: it is a two-rank op whose workspaces are all `IpcPeerVisible` and whose kernel
/// spins on a peer's flags, so a single-process sweeper has no peer to reduce with. It is
/// measured out of process by `saber ar-gate --timing` instead — never with a fabricated
/// number.
fn sweep_can_launch(op: &str) -> bool {
    matches!(
        op,
        "gemm_w4a8"
            | "add_rms_quant"
            | "silu_mul_quant"
            | "gdn_norm_quant"
            | "attn_decode_paged"
            | "attn_prefill_paged"
            | "gdn_fused_update"
            | "gdn_conv_update"
            | "gdn_conv_prep"
            | "gdn_kkt_solve"
            | "gdn_chunk_scan"
            | "gdn_gated_rmsnorm"
            | "allreduce"
            // The attention glue, bound without a number: one legal variant per band, and no
            // input builder in this process.
            | "qk_norm_rope"
            | "attn_gate_quant"
            | "kv_append"
    )
}

/// What the sweep has to do for one requirement: which op, at which geometry, and whether it can
/// be timed here at all.
///
/// The saber shapes are built at the reference capture script (`the internal reference harness`)'s geometry, which is the geometry
/// the bit-exact gate proved those kernels on. Two values come from the **bucket** rather than
/// from the shape keys, because the keys are necessary conditions and these are not: `max_ctx`
/// (the bucket's context band) and the GDN `maxq` (its speculative width).
fn work_for(req: &Requirement) -> anyhow::Result<Work> {
    work_for_keys(&req.key.op, &req.key.shape)
}

/// [`work_for`] from the KEY alone.
///
/// The bucket's batch and spec width are in the shape keys already -- they are part of what a
/// record is a measurement of -- so a caller that holds a bound op rather than a requirement (the
/// gate, deciding which capture case exercises which binding) needs nothing else.
pub(crate) fn work_for_keys(op: &str, keys: &saber_plan::ShapeKeys) -> anyhow::Result<Work> {
    let num = |name: &str| -> anyhow::Result<u32> {
        let v = keys
            .num(name)
            .ok_or_else(|| anyhow::anyhow!("{op}: shape has no `{name}`"))?;
        u32::try_from(v).map_err(|_| anyhow::anyhow!("{op}: `{name}` = {v} is not a u32"))
    };
    let dtype = |name: &str| -> Option<String> {
        keys.get(name).and_then(|v| v.as_str().map(str::to_owned))
    };
    Ok(match op {
        "attn_decode_paged" => {
            let q_heads = num("n")?;
            let gqa = num("gqa")?;
            let head_dim = num("k")?;
            let block_size = num("block_size")?;
            let kv_heads = q_heads / gqa.max(1);
            // Both of these are in the KEY now (2026-09-19): the record is a measurement of the
            // geometry its key names, so the sweep must read them from there and not re-derive
            // them from the bucket.
            let max_ctx = num("max_ctx")?;
            let blocks = max_ctx.div_ceil(block_size);
            let kv_dtype = match dtype("kv_dtype").as_deref() {
                Some("bf16") => KvDtype::Bf16,
                _ => KvDtype::Fp8E4m3,
            };
            Work::Sweep(TuneOp::AttnDecode(AttnShape {
                num_seqs: num("batch")?,
                q_len: num("q_len")?,
                // A swept decode cell is a full verify width; the prefill one below is a full
                // chunk. A sweep measures the bucket, so its live rows ARE the bucket's.
                live_rows: num("q_len")?,
                q_heads,
                kv_heads,
                head_dim,
                block_size,
                max_blocks: blocks,
                max_ctx,
                // 0 = let the split law choose, which is what prod passes.
                splits: 0,
                kv_dtype,
                kv_block_stride: i64::from(kv_heads * block_size * 2 * head_dim),
                kv_head_stride: i64::from(block_size * 2 * head_dim),
                scale_bits: (f64::from(head_dim).powf(-0.5) as f32).to_bits(),
                // Prod's per-tensor K/V scales are 1.0 and fold away.
                has_descales: false,
                ragged_rows: 0,
            }))
        }
        "gdn_fused_update" | "gdn_conv_update" => {
            let h = num("n")?;
            let head_k = num("k")?;
            let head_v = num("head_v")?;
            let conv_width = num("conv_width")?;
            let hg = h / GDN_V_PER_K_HEADS;
            let maxq = num("spec_width")?;
            let n_seqs = num("batch")?;
            let units = 2 * hg + h;
            let dim = units * head_k;
            let state_len = (conv_width - 1) + (maxq - 1);
            let shape = GdnShape {
                n_seqs,
                h,
                hg,
                head_k,
                head_v,
                conv_width,
                maxq,
                o_rows: n_seqs * maxq,
                xpitch: i64::from(dim),
                cs_seq: i64::from(dim * state_len),
                cs_dim: i64::from(state_len),
                cs_tok: 1,
                // `cache_idx` is `sidx[:, 0]`: the same allocation, read with stride `maxq`.
                ci_stride: i64::from(maxq),
                ab_stride: i64::from(h),
                st_slot: i64::from(h * head_v * head_k),
                st_head: i64::from(head_v * head_k),
                sidx_stride: i64::from(maxq),
                scale_bits: (f64::from(head_k).powf(-0.5) as f32).to_bits(),
                softplus_thr_bits: SOFTPLUS_THRESHOLD.to_bits(),
                has_naccept: true,
                has_bias: true,
                state_f16: false,
            };
            if op == "gdn_fused_update" {
                Work::Sweep(TuneOp::GdnFused(shape))
            } else {
                Work::Sweep(TuneOp::GdnPair(shape))
            }
        }
        "attn_prefill_paged" => {
            let q_heads = num("n")?;
            let gqa = num("gqa")?;
            let head_dim = num("k")?;
            let block_size = num("block_size")?;
            let kv_heads = q_heads / gqa.max(1);
            let max_ctx = num("max_ctx")?;
            let kv_dtype = match dtype("kv_dtype").as_deref() {
                Some("bf16") => KvDtype::Bf16,
                _ => KvDtype::Fp8E4m3,
            };
            Work::Sweep(TuneOp::AttnPrefill(AttnShape {
                // One sequence of `m` new tokens against `max_ctx` of context: the chunk the
                // scheduler hands the prefill bucket, which is what `m` means here.
                num_seqs: 1,
                q_len: num("m")?,
                live_rows: num("m")?,
                q_heads,
                kv_heads,
                head_dim,
                block_size,
                max_blocks: max_ctx.div_ceil(block_size),
                max_ctx,
                // The prefill kernel does not split KV; the field is unread on this path.
                splits: 0,
                kv_dtype,
                kv_block_stride: i64::from(kv_heads * block_size * 2 * head_dim),
                kv_head_stride: i64::from(block_size * 2 * head_dim),
                scale_bits: (f64::from(head_dim).powf(-0.5) as f32).to_bits(),
                has_descales: false,
                ragged_rows: 0,
            }))
        }
        "gdn_conv_prep" | "gdn_kkt_solve" | "gdn_chunk_scan" => {
            let h = num("n")?;
            let head_k = num("k")?;
            let head_v = num("head_v").unwrap_or(head_k);
            let conv_width = num("conv_width").unwrap_or(4);
            let hg = h / GDN_V_PER_K_HEADS;
            let t = num("m")?;
            let units = 2 * hg + h;
            let dim = units * head_k;
            let shape = GdnPrefillShape {
                // One sequence per prefill chunk, as `capture_ref.py` runs it.
                n_seqs: 1,
                t,
                h,
                hg,
                head_k,
                head_v,
                conv_width,
                chunk: num("chunk").unwrap_or(GDN_PREFILL_CHUNK),
                xpitch: i64::from(dim),
                // A prefill carries no speculative window, so the conv state is the kernel's
                // width minus one and nothing more.
                cs_seq: i64::from(dim * (conv_width - 1)),
                cs_dim: i64::from(conv_width - 1),
                cs_tok: 1,
                ci_stride: 1,
                ab_stride: i64::from(h),
                scale_bits: (f64::from(head_k).powf(-0.5) as f32).to_bits(),
                softplus_thr_bits: SOFTPLUS_THRESHOLD.to_bits(),
                has_bias: true,
                // A fresh chunk has no carried conv state.
                has_init: true,
                state_f16: false,
            };
            Work::Sweep(match op {
                "gdn_conv_prep" => TuneOp::GdnConvPrep(shape),
                "gdn_kkt_solve" => TuneOp::GdnKktSolve(shape),
                _ => TuneOp::GdnChunkScan(shape),
            })
        }
        "gdn_gated_rmsnorm" => {
            let channels = num("channels")?;
            // One row per (token, head): the norm runs over the scan's output.
            let rows = u64::from(num("m")?) * u64::from(num("n")?);
            Work::Sweep(TuneOp::GdnNorm(NormShape {
                rows,
                channels,
                xrow: i64::from(channels),
                zrow: i64::from(channels),
                orow: i64::from(channels),
                eps_bits: PRODUCER_EPS.to_bits(),
            }))
        }
        // The attention glue. Every field is derived from the key the plan asked with, so the
        // descriptor names the symbol and the launch count that will actually run -- including
        // which side of the qk band and which A-side layout.
        "qk_norm_rope" | "attn_gate_quant" | "kv_append" => {
            let m = num("m")?;
            let nq = num("n")?;
            let head_dim = num("head_dim")?;
            let nkv = nq / num("gqa")?.max(1);
            // [q|gate] per q head, then k, then v.
            let qkv_n = 2 * nq * head_dim + 2 * nkv * head_dim;
            // The SHAPE only. Which `glue::Op` it is comes from the `TuneOp`, so
            // `enumerate_saber` decides `mrope_interleaved` in one place rather than two.
            Work::Sweep(match op {
                "qk_norm_rope" => {
                    // The band, the batch and the verify width come from the KEY -- the record is
                    // a measurement of the geometry its key names, and `cos_sin`/`pos` are
                    // functions of all three (I20; `layer.attn.qk_norm_rope` carries
                    // `BucketKey::MaxCtx` for exactly this). `spec_width == 0` IS the prefill
                    // bucket, as `Bucket::prefill` builds it, so the phase is read rather than
                    // guessed.
                    let spec_width = num("spec_width")?;
                    let (phase, num_seqs, q_len) = if spec_width == 0 {
                        (Phase::Prefill, 1, m)
                    } else {
                        (Phase::Decode, num("batch")?, spec_width)
                    };
                    TuneOp::GlueQkNormRope(QkCell {
                        shape: QkShape {
                            m,
                            nq,
                            nkv,
                            head_dim,
                            qkv_n,
                        },
                        num_seqs,
                        q_len,
                        max_ctx: num("max_ctx")?,
                        phase,
                    })
                }
                "attn_gate_quant" => TuneOp::GlueGateQuant(GateShape {
                    m,
                    n: nq * head_dim,
                    head_dim,
                    gs: i64::from(qkv_n),
                }),
                _ => {
                    let block_size = num("block_size")?;
                    let hd = i64::from(head_dim);
                    let bs = i64::from(block_size);
                    TuneOp::GlueKvAppend(KvShape {
                        tokens: m,
                        nkv,
                        head_dim,
                        block_size,
                        key_stride: i64::from(nkv) * hd,
                        // V is read IN PLACE out of the projection, so its row stride is the
                        // projection's width and not the compact one.
                        value_stride: i64::from(qkv_n),
                        // `[blocks, kv_heads, block_size, 2 * head_dim]`.
                        block_stride: i64::from(nkv) * bs * 2 * hd,
                        page_stride: 2 * hd,
                        head_stride: bs * 2 * hd,
                    })
                }
            })
        }
        "allreduce" => {
            // The message is `hidden x rows`; `numel` is stated per row, so the necessary
            // condition holds at every bucket.
            let rows = u64::from(num("m")?);
            let shape = saber::Shape::Ar(saber::ArShape {
                n_elem: u64::from(num("numel")?) * rows,
                elem_bytes: 2,
                world_size: num("world_size")?,
            });
            let op = if keys.num("exact") == Some(1) {
                saber::Op::AllReduceExact {
                    dtype: saber::ArPayload::Bf16,
                }
            } else {
                saber::Op::AllReduceWht {
                    bits: num("bits")?,
                    dtype: saber::ArDtype::Bf16,
                }
            };
            Work::Unmeasurable {
                op,
                shape,
                reason: AR_NOT_MEASURED,
            }
        }
        _ => {
            let (gemm, shape) = op_and_shape(op, keys)?;
            Work::Sweep(TuneOp::Mxfp4 { op: gemm, shape })
        }
    })
}

fn op_and_shape(op: &str, keys: &saber_plan::ShapeKeys) -> anyhow::Result<(Op, Shape)> {
    let num = |name: &str| -> anyhow::Result<u32> {
        let v = keys
            .num(name)
            .ok_or_else(|| anyhow::anyhow!("{op}: shape has no `{name}`"))?;
        u32::try_from(v).map_err(|_| anyhow::anyhow!("{op}: `{name}` = {v} is not a u32"))
    };
    let m = num("m")?;
    Ok(match op {
        // The dense tier ships a per-row reference exponent, which is why the unfolded row is
        // refused at resolution rather than being a candidate here.
        "gemm_w4a8" => (Op::gemm(), Shape::new(m, num("n")?, num("k")?)),
        "add_rms_quant" => (
            Op::AddRmsQuant { eps: PRODUCER_EPS },
            Shape::new(m, 0, num("k")?),
        ),
        "silu_mul_quant" => (Op::SiluMulQuant, Shape::new(m, num("n")?, 0)),
        "gdn_norm_quant" => {
            let n = num("n")?;
            let zs = keys.num("zs").unwrap_or(i64::from(n));
            (
                Op::GdnNormQuant {
                    zs,
                    eps: PRODUCER_EPS,
                },
                Shape::new(m, n, 0),
            )
        }
        other => anyhow::bail!(
            "`{other}` is not an op this sweep can launch. Add it to `op_and_shape` when its \
             plugin lands; inventing a descriptor for it would measure the wrong kernel"
        ),
    })
}

/// Collapse the missing routes onto their distinct keys, in a device-friendly order.
///
/// Sorted by `(n, k, m)` so consecutive items share weights: the pool in [`Sweeper`] then holds
/// one shape's `w`/`ws`/`wref` across all four bucket M values instead of re-uploading 300 MiB
/// per route.
fn work_list(
    missing: &[Requirement],
    only: Option<&str>,
) -> anyhow::Result<(Vec<WorkItem>, BTreeMap<String, usize>)> {
    let mut by_key: BTreeMap<String, WorkItem> = BTreeMap::new();
    // Routes this sweep cannot measure, by op. NOT a silent skip: both callers print it, and a
    // sweep that covers less than the plan needs has to say so or the cache looks complete when
    // it is not.
    let mut untunable: BTreeMap<String, usize> = BTreeMap::new();
    for req in missing {
        if let Some(only) = only {
            if !req.op_name.contains(only) {
                continue;
            }
        }
        if !sweep_can_launch(&req.key.op) {
            *untunable.entry(req.key.op.clone()).or_default() += 1;
            continue;
        }
        let route = format!("{} {}", req.bucket.label(), req.op_name);
        let digest = req.key.digest();
        if let Some(item) = by_key.get_mut(&digest) {
            item.routes.push(route);
            continue;
        }
        let a_layout = match req.key.shape.get("a_layout").and_then(|v| v.as_str()) {
            Some("row_major") => ALayout::RowMajor,
            // Absent for everything that is not a GEMM, and `FollowsM` is what the enumerator
            // already assumes for those.
            _ => ALayout::FollowsM,
        };
        by_key.insert(
            digest,
            WorkItem {
                key: req.key.clone(),
                routes: vec![route],
                candidates: req.candidates.clone(),
                work: work_for(req)?,
                a_layout,
            },
        );
    }
    let mut items: Vec<WorkItem> = by_key.into_values().collect();
    items.sort_by_key(WorkItem::order);
    Ok((items, untunable))
}

/// The rules to enumerate one item's variants under.
///
/// `enumerate_variants` takes rules, not a layout, and decides the GEMM family with
/// `a_tiled(m, rules)`. A row-major edge is precisely a route for which that threshold does not
/// apply, so it is enumerated with the threshold switched off — which is the same statement
/// `rules::resolve_with_layout(.., ALayout::RowMajor)` makes in typed form, and the same edit
/// the enumerator already makes internally to reach the folded tile at a large M.
fn rules_for(item: &WorkItem, rules: &LaunchRules) -> LaunchRules {
    match item.a_layout {
        ALayout::FollowsM => rules.clone(),
        ALayout::RowMajor => {
            let mut r = rules.clone();
            r.knobs.a_tiled_min_m = 0;
            r
        }
    }
}

/// Print what the sweep is not covering, in one block, before anything is measured.
fn report_untunable(untunable: &BTreeMap<String, usize>) {
    if untunable.is_empty() {
        return;
    }
    let total: usize = untunable.values().sum();
    println!(
        "{total} route(s) are NOT in this sweep: their ops have no shape builder here yet, so the \
         cache this run writes will not complete the plan."
    );
    for (op, n) in untunable {
        println!("  {op:<24} {n} route(s)");
    }
    println!();
}

/// Partition the measured variants into the ones a record may BIND and the ones a split would
/// move the output on.
///
/// A variant with no `split_k` parameter is always bindable: the folded and A-tiled families do
/// not split a reduction. A variant that has one must match the reference's, because that is the
/// one knob in these tables that is not output-preserving (I11: 3 bytes of 163840 at
/// `gemm 8 10240 5120`). `ref_split` of `None` means the op has no split to compare -- every
/// variant is bindable, which is also what this code did before 2026-09-20.
fn split_preserving(
    measured: &[VariantSamples],
    ref_split: Option<u32>,
) -> (Vec<&VariantSamples>, Vec<&VariantSamples>) {
    let Some(want) = ref_split else {
        return (measured.iter().collect(), Vec::new());
    };
    let mut keep = Vec::new();
    let mut drop = Vec::new();
    for vs in measured {
        match vs.params.get("split_k") {
            Some(got) if *got != i64::from(want) => drop.push(vs),
            _ => keep.push(vs),
        }
    }
    (keep, drop)
}

/// The split the reference launcher takes for this op, when the op has one.
///
/// `None` for everything that is not an MXFP4 GEMM (no saber route splits a reduction the way
/// split-K does) and for a GEMM the rules refuse. The caller uses it to keep a measurement from
/// binding a variant whose output differs from the reference's -- see the call site.
fn prod_split(op: &TuneOp, layout: ALayout, rules: &LaunchRules) -> Option<u32> {
    match *op {
        // The decode descriptor's grid is `[nblk, 1, ks]`, so the split is read off the launch
        // the reference would make rather than re-derived from the ladder.
        TuneOp::Mxfp4 { op: gemm, shape } => resolve_with_layout(gemm, shape, rules, layout)
            .ok()
            .map(|d| d.grid[2]),
        // No saber route divides a reduction the way split-K does; the split-KV attention
        // decode splits the CONTEXT and its combine sums the partials in a fixed order.
        _ => None,
    }
}

/// Which registry implementation ships this plugin, among the candidates the planner allows.
///
/// The record names an *implementation*, not a plugin, and `bind` refuses a record whose
/// implementation is not in the candidate set — so the mapping is made against that set here, and
/// a plugin outside it is an error rather than a guess.
fn impl_for_plugin(
    registry: &Registry,
    candidates: &[String],
    plugin: &str,
) -> anyhow::Result<String> {
    let mut hits: Vec<&str> = registry
        .implementations()
        .iter()
        .filter(|i| i.plugin == plugin && candidates.iter().any(|c| c == &i.name))
        .map(|i| i.name.as_str())
        .collect();
    hits.sort_unstable();
    match hits.len() {
        1 => Ok(hits[0].to_string()),
        0 => anyhow::bail!(
            "plugin `{plugin}` produced a variant but no candidate implementation claims it \
             (candidates: {})",
            candidates.join(", ")
        ),
        _ => anyhow::bail!(
            "plugin `{plugin}` is claimed by {} candidate implementations ({}); the record could \
             not say which one ran",
            hits.len(),
            hits.join(", ")
        ),
    }
}

// ------------------------------------------------------------------ the command

/// Everything the planner needs, owned so that `Resolved` can borrow it.
struct Ctx {
    manifest: Manifest,
    registry: Registry,
    device: DeviceProfile,
    model: ModelSpec,
}

impl Ctx {
    fn load(args: &TuneArgs) -> anyhow::Result<Ctx> {
        let manifest = Manifest::from_path(&args.manifest)?;
        let registry = registry_from_prebuilt(&args.prebuilt)?;
        let device = DeviceProfile::gfx1201_reference();
        let model = ModelSpec::qwen38_27b_dense(manifest.device.tp)?;
        // Scoped to the rows this model can bind: the registry also carries the rollback tier's
        // Paro producers, which this deployment never launches and does not declare.
        check_plugins_declared(&registry, &manifest, &model)?;
        Ok(Ctx {
            manifest,
            registry,
            device,
            model,
        })
    }

    fn resolve(&self) -> anyhow::Result<Resolved<'_>> {
        Ok(saber_plan::resolve(
            &self.manifest,
            &self.device,
            &self.model,
            &self.registry,
        )?)
    }
}

/// Print the work list and the variants each item would measure. No GPU, no records.
pub fn list(args: &TuneArgs) -> anyhow::Result<()> {
    let ctx = Ctx::load(args)?;
    let resolved = ctx.resolve()?;
    let prior = load_input_cache(args)?;
    // The same merge the sweep runs, so `--list` reports the work the sweep will actually do.
    let (cache, stale) = resolved.carry_forward(&prior);
    report_carried(args, &prior, &cache, &stale);
    let missing = resolved.missing(&cache);
    let (items, untunable) = work_list(&missing, args.only.as_deref())?;
    let rules = LaunchRules::prod_mxfp4()?;
    let saber_rules = SaberRules::prod_saber()?;
    let symbols = SymbolIndex::from_manifest_dir(&args.prebuilt)?;

    report_untunable(&untunable);
    // The no-GPU path is where a bad cell should be found, so `--list` asks the same question the
    // sweep does.
    report_index_laws(&items)?;
    println!(
        "{} untuned route(s) -> {} distinct autotune key(s); {} symbol(s) indexed from {}",
        missing.len(),
        items.len(),
        symbols.len(),
        args.prebuilt.display()
    );
    println!();
    let mut total = 0usize;
    for item in &items {
        let Work::Sweep(op) = item.work else {
            // Not measurable *by this process*. The all-reduce still gets a real number, out of
            // process, unless the run turns that pass off; the prefill routes get none at all.
            let how = match item.work {
                Work::Unmeasurable {
                    shape: saber::Shape::Ar(_),
                    ..
                } if !args.no_ar_timing => "measured out of process by `saber ar-gate --timing`",
                Work::Unmeasurable {
                    shape: saber::Shape::Ar(_),
                    ..
                } => "--no-ar-timing: bound without a number",
                _ => "no sweep builder: bound without a number",
            };
            println!(
                "{:<40} NOT SWEEPABLE HERE, {how} ({} route(s))",
                item.label(),
                item.routes.len()
            );
            continue;
        };
        let item_rules = rules_for(item, &rules);
        let set = enumerate_variants(&op, &item_rules, &saber_rules, &symbols);
        total += set.variants.len();
        println!(
            "{:<34} {} candidate(s) x {} variant(s), {} route(s)",
            item.label(),
            item.candidates.len(),
            set.variants.len(),
            item.routes.len()
        );
        for v in &set.variants {
            println!("    {:<46} {}", v.kernel, v.label());
        }
        for r in &set.refused {
            println!("    refused: {r}");
        }
    }
    println!();
    println!(
        "{total} variant sweep(s) at {} warm-up + {} timed round(s) = {} launches",
        args.warmup,
        args.samples,
        total * (args.warmup + args.samples)
    );
    Ok(())
}

/// One row of the sweep table.
struct Row {
    label: String,
    routes: usize,
    candidates: usize,
    variants: usize,
    winner: String,
    winner_us: f64,
    runner_up: Option<(String, f64)>,
    margin_pct: Option<f64>,
    /// What the production launcher picks for this shape, and its time when it was measured.
    prod_kernel: Option<String>,
    prod_us: Option<f64>,
}

impl Row {
    fn differs_from_prod(&self) -> bool {
        self.prod_kernel.as_ref().is_some_and(|p| p != &self.winner)
    }
}

/// One all-reduce case, as `saber ar-gate` spells it.
fn ar_case_line(shape: &saber::ArShape, op: &saber::Op) -> String {
    match op {
        saber::Op::AllReduceWht { bits, .. } => {
            format!("ar_wht {} {} {bits} bf16", shape.n_elem, shape.elem_bytes)
        }
        _ => format!("ar_exact {} {} bf16", shape.n_elem, shape.elem_bytes),
    }
}

/// Measure every all-reduce case by running `saber ar-gate --timing` as two rank processes.
///
/// It cannot be done in this process: every one of the op's workspaces is `IpcPeerVisible`, and
/// `hipIpcOpenMemHandle` refuses a handle the calling process exported. So the sweep shells out to
/// the gate that already runs both ranks, and reads the `AR-TIMING` line it prints — the same
/// binary, the same launcher, the same descriptor, two processes.
///
/// Returns `task line -> (kernel, microseconds)`.
fn measure_all_reduce(
    args: &TuneArgs,
    cases: &[String],
    slot_bytes: u64,
    budget_mib: u64,
) -> anyhow::Result<BTreeMap<String, (String, f64)>> {
    let dir = args
        .out
        .parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    std::fs::create_dir_all(&dir)?;
    let case_file = dir.join("ar_cases.txt");
    std::fs::write(&case_file, cases.join("\n") + "\n")?;

    let exe = std::env::current_exe()?;
    let out = std::process::Command::new(&exe)
        .arg("ar-gate")
        .arg("--cases")
        .arg(&case_file)
        .arg("--prebuilt")
        .arg(&args.prebuilt)
        .arg("--hip-lib")
        .arg(&args.hip_lib)
        .arg("--timing")
        .arg("--warmup")
        .arg(AR_WARMUP.to_string())
        .arg("--samples")
        .arg(args.samples.to_string())
        .arg("--ar-slot-bytes")
        .arg(slot_bytes.to_string())
        .arg("--budget-mib")
        .arg(budget_mib.to_string())
        .output()
        .map_err(|e| anyhow::anyhow!("spawning `saber ar-gate`: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout);
    if !out.status.success() {
        anyhow::bail!(
            "`saber ar-gate --timing` exited {}: the all-reduce did not pass its own gate, so \
             there is no number to record.\n{}{}",
            out.status,
            text,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    parse_ar_timing(&text)
}

/// The `AR-TIMING` lines out of `saber ar-gate --timing`'s output.
///
/// Tab-separated and prefixed, so it survives the gate's own table being reformatted: a harness
/// that scrapes a human table is a harness that breaks the next time someone widens a column.
fn parse_ar_timing(text: &str) -> anyhow::Result<BTreeMap<String, (String, f64)>> {
    let mut found = BTreeMap::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("AR-TIMING\t") else {
            continue;
        };
        let f: Vec<&str> = rest.split('\t').collect();
        let (Some(task), Some(kernel), Some(max)) = (f.first(), f.get(1), f.get(4)) else {
            anyhow::bail!("ar-gate printed a malformed timing line: `{line}`");
        };
        let us: f64 = max
            .strip_prefix("max=")
            .ok_or_else(|| anyhow::anyhow!("ar-gate printed no `max=` in: `{line}`"))?
            .parse()
            .map_err(|_| anyhow::anyhow!("ar-gate printed an unparseable time: `{line}`"))?;
        found.insert((*task).to_string(), ((*kernel).to_string(), us));
    }
    Ok(found)
}

/// Run the sweep.
pub fn run(hip: &HipThread, args: &TuneArgs) -> anyhow::Result<()> {
    let ctx = Ctx::load(args)?;
    let required_cap = ctx.manifest.device.power_cap_w;

    // ---- the cap, before anything is measured -------------------------------------------
    let sysfs = Path::new("/sys/class/drm");
    let mut caps = read_power_caps(sysfs);
    let asserted = caps.len() < 2 && args.power_cap_w.is_some();
    if asserted {
        let w = args.power_cap_w.expect("checked");
        caps = (0..2)
            .map(|i| PowerCap {
                card: format!("--power-cap-w[{i}]"),
                path: None,
                watts: w,
            })
            .collect();
    }
    println!("power cap");
    for c in &caps {
        println!(
            "  {:<18} {} W   {}",
            c.card,
            c.watts,
            c.path.as_ref().map_or(
                "asserted on the command line (sysfs not readable here)".to_string(),
                |p| p.display().to_string()
            )
        );
    }
    if let Err(why) = check_power_caps(&caps, required_cap) {
        anyhow::bail!("refusing to write any record: {why}");
    }
    if let (Some(w), false) = (args.power_cap_w, asserted) {
        if w != required_cap {
            anyhow::bail!(
                "--power-cap-w {w} disagrees with the caps actually read ({required_cap} W)"
            );
        }
    }
    println!("  both cards at the manifest's {required_cap} W; records may be written");

    // ---- prod, before anything is measured ------------------------------------------------
    //
    // A connection refused is NOT a failure here: prod being down is a legitimate condition to
    // measure under, and the one that matters is "nothing else is using the GPU". What is not
    // legitimate is measuring beside a server with requests in flight, which is the only case
    // this refuses. Which of the three ran is recorded on every record, because absolute
    // microseconds are not comparable across them.
    // VRAM first: the sweep runs beside prod on purpose, so it does not refuse on held memory —
    // but a record measured while 30 GiB was held is a different measurement from one taken on a
    // free card, and the line below is what says which it was.
    match crate::gpu_idle::look(hip, &args.metrics_addr) {
        Ok(idle) => println!("{}", idle.line(args.device)),
        Err(e) => println!(
            "idle      VRAM unreadable ({e}); the request count below is the only \
                            evidence this run has"
        ),
    }
    let gpu_state = if args.no_prod_check {
        println!(
            "prod            *** NOT CHECKED *** (--no-prod-check): the cards are ASSERTED idle, \
             not observed to be. Every record written by this run says so."
        );
        GpuState::NotChecked
    } else {
        match prod_requests_running(&args.metrics_addr) {
            Some(n) if n > 0.0 => anyhow::bail!(
                "prod has {n} request(s) in flight on {}; refusing to time anything",
                args.metrics_addr
            ),
            Some(_) => {
                println!(
                    "prod            idle (num_requests_running = 0 on {})",
                    args.metrics_addr
                );
                GpuState::ProdIdle
            }
            None => {
                println!(
                    "prod            no /metrics on {} — prod is down; the GPU is idle, which is \
                     a condition this sweep may measure under and records as such",
                    args.metrics_addr
                );
                GpuState::ProdDown
            }
        }
    };

    // ---- what to measure --------------------------------------------------------------------
    let resolved = ctx.resolve()?;
    let prior = load_input_cache(args)?;
    // The carried cache is this resolution's keys and NOTHING else, which is what a full sweep
    // produces without trying: it starts empty. An incremental one starts from a file, so the
    // records the file holds and the plan does not have to be dropped here or the op ends up
    // with a record under each policy digest (I19; the guard's `attn_gate_quant carries 2 policy
    // digests` on 2026-09-21).
    let (mut cache, stale) = resolved.carry_forward(&prior);
    report_carried(args, &prior, &cache, &stale);
    let missing = resolved.missing(&cache);
    let (items, untunable) = work_list(&missing, args.only.as_deref())?;
    report_untunable(&untunable);
    if items.is_empty() {
        println!("nothing to do: no route this sweep can measure is untuned");
        return Ok(());
    }

    let rules = LaunchRules::prod_mxfp4()?;
    let saber_rules = SaberRules::prod_saber()?;
    let symbols = SymbolIndex::from_manifest_dir(&args.prebuilt)?;

    hip.set_device(args.device)?;
    let load_counters = LoadCounters::new();
    // Exactly the plugins this manifest declares, not everything in the directory. The plan binds
    // a declared plugin set and its build hashes are in every autotune key, so loading a code
    // object the plan never names could only add a way to fail.
    let names: Vec<&str> = ctx
        .manifest
        .plugins
        .iter()
        .map(|p| p.name.as_str())
        .collect();
    let plugins = PluginSet::load_named(hip, &args.prebuilt, &names, &load_counters)?;
    let functions = plugins.functions(hip)?;
    let launch_ctx = LaunchCtx::new(hip, &functions);
    println!(
        "plugins         {} loaded ({} symbols); {}",
        plugins.plugins().len(),
        functions.len(),
        load_counters.snapshot().line()
    );
    // The sweep's inputs against the step's index laws, over every case, BEFORE the first launch:
    // a cell whose `seqused_k`, `cu`, `block_table`, `slot` or `cache_idx` disagrees with
    // `saber_exec::forward::index_table` is refused here naming the op, and the divergences the
    // cell cannot state are printed. `Sweeper::sweep` runs the same check per case; this one runs
    // first so the refusal costs no GPU time and names every op at once.
    report_index_laws(&items)?;

    let selected: usize = items.iter().map(|i| i.routes.len()).sum();
    println!(
        "work            {} untuned route(s); {selected} selected -> {} distinct key(s) (routes \
         that share a geometry share a key, so `attn.o_proj` and `gdn.out_proj` are one \
         measurement)",
        missing.len(),
        items.len()
    );
    println!(
        "protocol        {} warm-up round(s) then {} timed round(s), variants interleaved \
         A B C A B C; hipEvent around the launch only; the shape's first sample is discarded and \
         the median of the rest is the record",
        args.warmup, args.samples
    );
    println!(
        "cache state    {}",
        if args.ic_hot {
            "IC-HOT: every sample re-reads ONE set of buffers, so anything under the 64 MB              Infinity Cache is served from it. This is NOT what the step does -- it is the A/B              arm, for showing what the hot number was."
        } else {
            "IC-cold: every streamed buffer >= 1 MiB is rotated through enough copies that a              sample never re-reads a warm line, which is the state a step streaming 15.4 GB of              weights per token is always in"
        }
    );
    println!(
        "conditions      device {}, {}, cap {required_cap} W, budget {} MiB. \
         ONLY THE RANKING WITHIN ONE SHAPE IS MEANINGFUL, and only that is what a record \
         decides: absolute microseconds move with what else is on the card.",
        args.device,
        gpu_state.label(),
        args.budget_mib
    );

    let stream = hip.create_stream(StreamFlags::NonBlocking)?;
    let launch_counters = LaunchCounters::new();
    let launcher = Launcher::new("tune", &launch_counters);
    let mut sweeper = Sweeper::new(&launch_ctx, &stream, launcher);
    let cfg = SweepConfig {
        warmup: args.warmup,
        samples: args.samples,
        budget_bytes: args.budget_mib << 20,
        ic_cold: !args.ic_hot,
    };

    let provenance = Provenance::new(
        "saber tune",
        &today_utc(),
        &format!(
            "{HARNESS} ({} warm-up + {} interleaved samples, {}, {required_cap} W{}, {})",
            args.warmup,
            args.samples,
            if args.ic_hot { "IC-HOT" } else { "IC-cold" },
            if asserted { ", cap asserted" } else { "" },
            gpu_state.label()
        ),
    );
    let ar_harness = format!(
        "{AR_HARNESS} ({AR_WARMUP} warm-up + {} timed round(s), {required_cap} W{}, {})",
        args.samples,
        if asserted { ", cap asserted" } else { "" },
        gpu_state.label()
    );
    let mut tuner = Autotuner::new(cache.clone(), provenance);

    println!();
    println!(
        "{:<34} {:>5} {:>5} {:<46} {:>9} {:<40} {:>8}",
        "op / shape / M", "route", "c x v", "winner", "us", "runner-up", "margin"
    );

    // The two-rank all-reduce, first and in its own two processes. Before the sweep, so the
    // input pool this process is about to fill is not resident on the cards while the rank
    // processes are asking them for peer-visible memory.
    let ar_work: Vec<(String, u64)> = items
        .iter()
        .filter_map(|i| match i.work {
            Work::Unmeasurable { op, shape, .. } => match shape {
                saber::Shape::Ar(s) => Some((ar_case_line(&s, &op), s.n_elem * s.elem_bytes)),
                _ => None,
            },
            Work::Sweep(_) => None,
        })
        .collect();
    let ar_cases: Vec<String> = ar_work.iter().map(|(c, _)| c.clone()).collect();
    // Sized from the plan's own traffic, not from a constant: a prefill bucket all-reduces
    // 80 MiB and the one-shot kernel refuses a message wider than the slot.
    let (ar_slot, ar_budget_mib) =
        ar_slot_and_budget(ar_work.iter().map(|(_, b)| *b).max().unwrap_or(0));
    let ar_times = if ar_cases.is_empty() || args.no_ar_timing {
        if !ar_cases.is_empty() {
            println!(
                "all-reduce      --no-ar-timing: {} route(s) will be BOUND WITHOUT A MEASUREMENT",
                ar_cases.len()
            );
        }
        BTreeMap::new()
    } else {
        println!(
            "all-reduce      measuring {} case(s) through `saber ar-gate --timing`: two rank \
             processes on devices 0 and 1, {} MiB IPC slot ({} MiB per-rank budget, sized to \
             the largest message this plan carries), {AR_WARMUP} warm-up + {} timed round(s), a \
             rendezvous before every round and the MAX of the two ranks",
            ar_cases.len(),
            ar_slot >> 20,
            ar_budget_mib,
            args.samples
        );
        measure_all_reduce(args, &ar_cases, ar_slot, ar_budget_mib)?
    };

    let mut rows: Vec<Row> = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    let mut unmeasured = 0usize;
    /// `(route label, the two winners and the worst swing)`, for the summary.
    type UnstableRow = (String, saber_plan::Unstable);
    let mut unstable: Vec<UnstableRow> = Vec::new();
    let sweep_started = std::time::Instant::now();
    let mut last_shape = (u32::MAX, u32::MAX);
    for item in &items {
        // A new weight/cache set: free the pool before allocating the next one.
        if item.order().0 != last_shape.0 || item.order().1 != last_shape.1 {
            // A new (n, k): the weights in the pool are dead. Free them before allocating the
            // next set — there is ~1 GiB of VRAM beside a 0.95-utilisation server.
            sweeper.release_inputs();
            last_shape = (item.order().0, item.order().1);
        }
        let Work::Sweep(op) = item.work else {
            let Work::Unmeasurable { op, shape, reason } = item.work else {
                unreachable!("Work has two arms and Sweep was handled above");
            };
            let desc = saber::resolve(op, shape, &saber_rules)?;
            let impl_name = impl_for_plugin(&ctx.registry, &item.candidates, &desc.plugin)?;
            let mut params = VariantParams::new(variant_id_of(&desc.kernel));
            params = params.param("n_launches", i64::from(desc.n_launches));
            // An all-reduce case the two-rank pass timed carries a real number. Everything
            // else -- the glue routes, and the all-reduce when that pass did not run -- is
            // bound without one, and says per route why.
            let timed = match item.work {
                Work::Unmeasurable {
                    op,
                    shape: saber::Shape::Ar(ar),
                    ..
                } => ar_times.get(&ar_case_line(&ar, &op)),
                _ => None,
            };
            match timed {
                Some((kernel, us)) => {
                    // The gate resolved the same descriptor from the same rules; if it did not,
                    // the number belongs to a different kernel and must not be recorded.
                    if kernel != &desc.kernel {
                        anyhow::bail!(
                            "ar-gate timed `{kernel}` where this plan binds `{}`: the two \
                             disagree about which kernel this message takes",
                            desc.kernel
                        );
                    }
                    tuner.record_external(
                        item.key.clone(),
                        &impl_name,
                        params,
                        *us,
                        args.samples,
                        &ar_harness,
                    );
                    println!(
                        "{:<34} {:>5} {:>5} {:<46} {:>9.2} {:<40} {:>8}",
                        item.label(),
                        item.routes.len(),
                        "1x1",
                        desc.kernel,
                        us,
                        "(2 ranks, max of the two; see ar-gate)",
                        "-"
                    );
                }
                None => {
                    // Bound, not measured, and loud about it at every later step.
                    tuner.record_unmeasured(item.key.clone(), &impl_name, params, reason);
                    println!(
                        "{:<34} {:>5} {:>5} {:<46} {:>9} {:<40} {:>8}",
                        item.label(),
                        item.routes.len(),
                        "1x0",
                        desc.kernel,
                        "NOT MEAS",
                        // The reason, not a blanket one: the all-reduce has a two-process
                        // harness and this run did not use it, the prefill routes have no
                        // builder at all. They are bound the same way for different reasons
                        // and the line should say which.
                        if reason == AR_NOT_MEASURED {
                            "(2-rank op, not timed in this run)"
                        } else {
                            "(no prefill shape builder)"
                        },
                        "-"
                    );
                    unmeasured += 1;
                }
            }
            continue;
        };
        let item_rules = rules_for(item, &rules);
        let set: VariantSet = enumerate_variants(&op, &item_rules, &saber_rules, &symbols);
        if set.variants.is_empty() {
            let why = set
                .refused
                .first()
                .cloned()
                .unwrap_or_else(|| "no legal variant".to_string());
            skipped.push((item.label(), why));
            continue;
        }
        let outcome = sweeper.sweep(&op, &set.variants, &cfg)?;
        let measured = match outcome {
            ShapeOutcome::Skipped { reason, .. } => {
                skipped.push((item.label(), reason));
                continue;
            }
            ShapeOutcome::Swept(m) => m,
        };
        // A SECOND, independent pass over the same enumeration. It doubles the GPU time and buys
        // the one thing a single pass cannot say: whether its own winner is reproducible. I17
        // found `attn_decode_paged` at batch 4 naming `nw=4` in one sweep and `nw=2` in the next,
        // with the same variant measuring 81.3 us and 73.96 us -- 9% apart on a shape whose
        // within-run spread is near 0.5%, and the tie-break deciding it silently both times.
        let second = match sweeper.sweep(&op, &set.variants, &cfg)? {
            ShapeOutcome::Skipped { reason, .. } => {
                skipped.push((item.label(), reason));
                continue;
            }
            ShapeOutcome::Swept(m) => m,
        };

        // What the production launcher would pick for this shape. It is not a vote: it only
        // breaks a tie the noise floor cannot decide, and then the engine agrees with production
        // exactly where the measurement has no opinion. For a saber route the reference IS the
        // rules' own answer at the prod knobs, which is what `resolve` returns with no overlay.
        //
        // The GEMM reference takes the edge's `a_layout`, like the enumerator does: `resolve`
        // alone assumes `FollowsM` and answers A-tiled above the threshold, so a row-major edge
        // (`o_proj`, whose A is the attention output) read as a disagreement at every prefill M
        // while both sides in fact agreed. That is a bug in the report, not a finding.
        let prod_kernel = match op {
            // No production launcher to compare against: prod runs Triton and ATen for all three,
            // which is why the glue was written from reference SEMANTICS rather than ported. A
            // comparison column would have to invent a kernel name to disagree with.
            TuneOp::GlueQkNormRope(_) | TuneOp::GlueGateQuant(_) | TuneOp::GlueKvAppend(_) => None,
            TuneOp::Mxfp4 { op, shape } => resolve_with_layout(op, shape, &rules, item.a_layout)
                .ok()
                .map(|d| d.kernel),
            TuneOp::AttnDecode(a) => {
                saber::resolve(saber::Op::AttnDecode, saber::Shape::Attn(a), &saber_rules)
                    .ok()
                    .map(|d| d.kernel)
            }
            TuneOp::GdnFused(g) => saber::resolve(
                saber::Op::GdnFusedUpdate { ab_bf16: true },
                saber::Shape::Gdn(g),
                &saber_rules,
            )
            .ok()
            .map(|d| d.kernel),
            // The pair is TWO symbols, and the rules choose both: `hpb` for the conv and `vs`
            // for the recurrent update. Naming only the conv would make every pair look like a
            // disagreement and hide the axis that actually moved.
            TuneOp::GdnPair(g) => saber::resolve(
                saber::Op::GdnDecodePair { ab_bf16: true },
                saber::Shape::Gdn(g),
                &saber_rules,
            )
            .ok()
            .zip(saber::gdn_decode_pair_recurrent(g, true, None, &saber_rules).ok())
            .map(|(conv, rec)| format!("{} + {}", conv.kernel, rec.kernel)),
            // The prefill-class routes: the rules' answer at the tier's own knobs.
            TuneOp::AttnPrefill(a) => saber::resolve(
                saber::Op::AttnPrefill {
                    fp8_mode: ATTN_PREFILL_FP8_MODE,
                },
                saber::Shape::Attn(a),
                &saber_rules,
            )
            .ok()
            .map(|d| d.kernel),
            TuneOp::GdnConvPrep(g) => saber::resolve(
                saber::Op::GdnConvPrep { ab_bf16: true },
                saber::Shape::GdnPrefill(g),
                &saber_rules,
            )
            .ok()
            .map(|d| d.kernel),
            TuneOp::GdnKktSolve(g) => saber::resolve(
                saber::Op::GdnKktSolve,
                saber::Shape::GdnPrefill(g),
                &saber_rules,
            )
            .ok()
            .map(|d| d.kernel),
            TuneOp::GdnChunkScan(g) => saber::resolve(
                saber::Op::GdnChunkScan,
                saber::Shape::GdnPrefill(g),
                &saber_rules,
            )
            .ok()
            .map(|d| d.kernel),
            TuneOp::GdnNorm(n) => saber::resolve(
                saber::Op::GdnGatedRmsNorm { act: 0 },
                saber::Shape::Norm(n),
                &saber_rules,
            )
            .ok()
            .map(|d| d.kernel),
        };

        // Hand the raw samples to the planner's protocol. The order is the measurement order, so
        // the discard lands on the sample that was actually taken first.
        // Only OUTPUT-PRESERVING variants may be bound.
        //
        // Every knob in these tables changes how the same arithmetic is scheduled -- BK is a
        // staging width, TN a column tile, TM a row tile, NW a warp count -- except one. A split
        // divides the k-sum and adds the parts, so ks=1 and ks=2 associate it differently and the
        // last bf16 bit moves: I11 measured that (3 bytes of 163840) and declined to ship a rule
        // for it. The CACHE was doing it anyway, silently, since the first sweep -- four cells
        // where the measurement beat the rules by 2.5-4.1% and the replayed output differed by
        // 3-12 bytes, which `saber gate --autotune` found the day it could see the plan's
        // bindings (2026-09-20).
        //
        // So a variant whose split differs from the reference launcher's is MEASURED and printed
        // -- the finding is worth keeping -- and not offered to the recorder. The engine's bytes
        // stay prod's bytes, and what that costs is on the line below.
        let ref_split = prod_split(&op, item.a_layout, &rules);
        let (bindable, dropped) = split_preserving(&measured, ref_split);
        let mut measurements = Vec::with_capacity(bindable.len());
        for vs in bindable {
            let impl_name = impl_for_plugin(&ctx.registry, &item.candidates, &vs.plugin)?;
            let mut params = VariantParams::new(vs.variant_id);
            for (k, v) in &vs.params {
                params = params.param(k, *v);
            }
            let m = Measurement::new(&impl_name, params, vs.samples.clone());
            let is_ref = prod_kernel.as_deref() == Some(vs.kernel.as_str());
            measurements.push(if is_ref { m.reference() } else { m });
        }
        if measurements.is_empty() {
            anyhow::bail!(
                "{}: every measured variant changes the split; nothing output-preserving was                  offered, which cannot happen unless the reference's own answer was not                  enumerated",
                item.label()
            );
        }
        // The same variants in the same order, from the second pass, so the two are comparable
        // variant for variant. A variant dropped by the split filter in one pass and not the
        // other cannot happen -- the filter is a function of the reference, not of the samples --
        // and `record_two_passes` refuses the pair if it ever did.
        let (bindable_2, _) = split_preserving(&second, ref_split);
        let mut second_measurements = Vec::with_capacity(bindable_2.len());
        for vs in bindable_2 {
            let impl_name = impl_for_plugin(&ctx.registry, &item.candidates, &vs.plugin)?;
            let mut params = VariantParams::new(vs.variant_id);
            for (k, v) in &vs.params {
                params = params.param(k, *v);
            }
            let m = Measurement::new(&impl_name, params, vs.samples.clone());
            let is_ref = prod_kernel.as_deref() == Some(vs.kernel.as_str());
            second_measurements.push(if is_ref { m.reference() } else { m });
        }
        let record =
            tuner.record_two_passes(item.key.clone(), measurements, second_measurements)?;
        if let Some(u) = &record.unstable {
            unstable.push((item.label(), u.clone()));
        }
        let winner_id = record.variant.variant_id;
        let winner = measured
            .iter()
            .find(|v| v.variant_id == winner_id)
            .expect("the winning variant came from this sweep");
        let runner_up = record.runner_up.as_ref().map(|r| {
            let kernel = measured
                .iter()
                .find(|v| v.variant_id == r.variant_id)
                .map_or_else(|| format!("variant {}", r.variant_id), |v| v.kernel.clone());
            (kernel, r.measured_us)
        });

        // What the prod launcher's own pick cost when it was measured.
        let prod_us = prod_kernel.as_ref().and_then(|k| {
            measured
                .iter()
                .find(|v| &v.kernel == k)
                .map(|v| median(&v.samples))
        });

        let row = Row {
            label: item.label(),
            routes: item.routes.len(),
            candidates: item.candidates.len(),
            variants: measured.len(),
            winner: winner.kernel.clone(),
            winner_us: record.measured_us,
            runner_up,
            margin_pct: record.margin_pct(),
            prod_kernel: prod_kernel.clone(),
            prod_us,
        };
        println!(
            "{:<34} {:>5} {:>5} {:<46} {:>9.2} {:<40} {:>8}",
            row.label,
            row.routes,
            format!("{}x{}", row.candidates, row.variants),
            row.winner,
            row.winner_us,
            row.runner_up
                .as_ref()
                .map_or_else(|| "-".to_string(), |(k, us)| format!("{k} {us:.2}")),
            row.margin_pct
                .map_or_else(|| "-".to_string(), |m| format!("{m:+.1}%")),
        );
        for vs in &dropped {
            let us = median(&vs.samples);
            let faster = record.measured_us - us;
            if faster > 0.0 {
                println!(
                    "      {:<46} {us:>9.2}  NOT BOUND: changes the split ({faster:+.2} us vs the \
                     bound variant)",
                    vs.kernel
                );
            }
        }
        if args.dump_variants {
            // Every variant's median, so a reader can see how close the field was rather than
            // only who won. A shape whose top three sit inside one noise floor looks identical to
            // a shape decided by 30% in the table above, and it is not the same situation.
            let mut ranked: Vec<(&str, f64)> = measured
                .iter()
                .map(|v| (v.kernel.as_str(), median(&v.samples)))
                .collect();
            ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
            let best = ranked[0].1;
            for (kernel, us) in ranked {
                println!(
                    "      {:<46} {us:>9.2}  {:+7.2} us{}",
                    kernel,
                    us - best,
                    if Some(kernel) == prod_kernel.as_deref() {
                        "   <- prod launcher"
                    } else {
                        ""
                    }
                );
            }
        }
        rows.push(row);
    }

    println!();
    for (label, why) in &skipped {
        println!("SKIPPED {label}: {why}");
    }
    let (scratch, persistent) = sweeper.launcher().arena_bytes();
    println!(
        "sweep           {:.1} s wall for {} launch(es) over {} shape(s), allocation and fill \
         included",
        sweep_started.elapsed().as_secs_f64(),
        launch_counters.snapshot().launched,
        rows.len()
    );
    println!("launch counters {}", launch_counters.snapshot().line());
    println!(
        "arenas          scratch {scratch} B, persistent {persistent} B; inputs {} B held at exit",
        sweeper.pool_bytes()
    );

    // ---- write, reload, bind ---------------------------------------------------------------
    cache = tuner.into_cache();
    if let Some(dir) = args.out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    cache.save(&args.out)?;
    println!();
    if unstable.is_empty() {
        println!(
            "stability       every shape's two passes named the same winner: {} of {} \
             reproducible",
            items.len() - unmeasured,
            items.len() - unmeasured
        );
    } else {
        println!(
            "stability       {} shape(s) NOT REPRODUCIBLE: the two passes named different \
             winners, so the RULES keep the cell and the record says so",
            unstable.len()
        );
        for (label, u) in &unstable {
            println!(
                "                {label}: pass 1 {} @ {:.2} us, pass 2 {} @ {:.2} us, worst \
                 swing on one variant {:.1}%",
                u.first.variant_id,
                u.first.measured_us,
                u.second.variant_id,
                u.second.measured_us,
                u.worst_swing_pct
            );
        }
    }
    // A carried record keeps its `unstable` pair, and a cell that was not reproducible in the
    // sweep that measured it is not reproducible now — this run simply did not re-measure it. The
    // stability line above counts what THIS run swept, so the carried ones are stated separately
    // rather than disappearing into a "reproducible" count that never looked at them.
    let measured_now: std::collections::BTreeSet<String> =
        items.iter().map(|i| i.key.digest()).collect();
    let carried_unstable = cache
        .entries()
        .filter(|e| e.record.unstable.is_some() && !measured_now.contains(&e.key.digest()))
        .count();
    if carried_unstable > 0 {
        println!(
            "                {carried_unstable} carried cell(s) still carry an `unstable` pair \
             from the sweep that measured them; this run did not re-measure them"
        );
    }
    println!(
        "cache           {} record(s) written to {} ({unmeasured} of them BOUND WITHOUT A \
         MEASUREMENT)",
        cache.len(),
        args.out.display()
    );

    // Reload from disk rather than binding the in-memory cache: what the next command will read is
    // the file, and a round trip that only works in memory is not a round trip.
    let reloaded = AutotuneCache::load(&args.out)?;
    // The guard, run by the sweep on its own output. `tools/ci/structural.sh` and
    // `the_committed_cache_binds_the_committed_manifest` call the same function; what they cannot
    // do is stop the file being written, and the file being written is what cost a full sweep on
    // 2026-09-21.
    if let Err(split) = reloaded.one_policy_digest_per_op() {
        anyhow::bail!(
            "the cache just written to {} is split across policies: {split}. The merge dropped \
             too little — this is `Resolved::carry_forward`'s bug, not the cache's.",
            args.out.display()
        );
    }
    let still_missing = resolved.missing(&reloaded);
    if !still_missing.is_empty() {
        println!(
            "{} route(s) are still untuned; `saber explain` will keep refusing to bind:",
            still_missing.len()
        );
        for r in &still_missing {
            println!("  {} {}", r.bucket.label(), r.op_name);
        }
        report_differences(&rows);
        return Ok(());
    }
    let plan = resolved.bind(&reloaded)?;
    println!();
    print!("{}", explain(&plan));
    report_differences(&rows);
    Ok(())
}

/// The finding the sweep exists to produce: every route where the measurement disagrees with the
/// launcher prod ships. Not an error — the reference launcher's ladder was measured on a different
/// day, at different shapes, against a different dispatch path.
fn report_differences(rows: &[Row]) {
    let diff: Vec<&Row> = rows.iter().filter(|r| r.differs_from_prod()).collect();
    println!();
    println!(
        "winner vs the production launcher: {} of {} shape(s) disagree",
        diff.len(),
        rows.len()
    );
    if diff.is_empty() {
        println!("  (every shape's measured winner is what `resolve` picks for the prod knobs)");
        return;
    }
    for r in diff {
        let prod = r.prod_kernel.as_deref().unwrap_or("(refused)");
        let delta = match (r.prod_us, r.winner_us) {
            (Some(p), w) if w > 0.0 => format!("{:+.1}%", (p - w) / w * 100.0),
            _ => "-".to_string(),
        };
        println!(
            "  {:<34} measured {:<46} {:>8.2} us   prod {:<46} {:>8}   {delta}",
            r.label,
            r.winner,
            r.winner_us,
            prod,
            r.prod_us
                .map_or_else(|| "not measured".to_string(), |us| format!("{us:.2} us"))
        );
    }
}

/// Every case's index-law inputs against the step's own laws, and the declared divergences.
///
/// The `attn_prefill_paged` records in every cache up to I17 were measurements of a cell the
/// engine does not run — `seqused_k = max_ctx` where `forward::index_table` gives a prefill chunk
/// its own length — and no key, fingerprint or guard in this repo covered that class, because the
/// record is keyed on the SHAPE and it was the INPUT that was wrong (I18). This is the check for
/// it, at sweep time, naming the op.
///
/// # Errors
/// A case whose index-law buffer is neither derived from the step's law nor declared to differ.
fn report_index_laws(items: &[WorkItem]) -> anyhow::Result<()> {
    let mut declared: BTreeMap<(String, &'static str), &'static str> = BTreeMap::new();
    let mut checked = 0usize;
    for item in items {
        let Work::Sweep(op) = &item.work else {
            continue;
        };
        checked += 1;
        for note in saber_exec::check_index_laws(op)
            .map_err(|e| anyhow::anyhow!("{} refuses its own inputs: {e}", item.label()))?
        {
            declared.insert((item.key.op.clone(), note.buffer), note.why);
        }
    }
    println!(
        "index laws      {checked} case(s) built their `seqused_k`/`cu`/`block_table`/`slot` from \
         `forward::index_table`, the law the step runs",
    );
    for ((op, buffer), why) in &declared {
        println!("                DECLARED {op}.{buffer}: {why}");
    }
    Ok(())
}

fn load_input_cache(args: &TuneArgs) -> anyhow::Result<AutotuneCache> {
    Ok(match &args.autotune_in {
        Some(p) => AutotuneCache::load(p)?,
        None => AutotuneCache::new(),
    })
}

/// What the incremental merge kept and what it dropped, before a single kernel is launched.
///
/// A drop is a record this repo paid GPU time for, so it is never silent. The two reasons are
/// different facts and are counted apart: **superseded** is the incremental sweep working — the
/// route is still in the plan and its key moved, so this run re-measures it — while **not
/// required** means the file answered a plan this manifest is not.
fn report_carried(
    args: &TuneArgs,
    prior: &saber_plan::AutotuneCache,
    carried: &saber_plan::AutotuneCache,
    stale: &[saber_plan::Stale],
) {
    let Some(path) = &args.autotune_in else {
        return;
    };
    let superseded = stale
        .iter()
        .filter(|s| matches!(s.why, saber_plan::StaleWhy::Superseded(_)))
        .count();
    println!(
        "carried         {} of {} record(s) from {} ({superseded} superseded by a moved key, {} \
         not a route in this plan)",
        carried.len(),
        prior.len(),
        path.display(),
        stale.len() - superseded,
    );
    for s in stale {
        println!("                DROPPED {s}");
    }
    // The property that makes the incremental path equal to `--full`: whatever this run measures,
    // the file it writes cannot carry two policies for one op, because the carried half already
    // does not.
    if let Err(split) = carried.one_policy_digest_per_op() {
        println!("                *** {split} — the merge is wrong, not the cache ***");
    }
}

fn median(samples: &[f64]) -> f64 {
    let mut s = samples.to_vec();
    s.sort_by(f64::total_cmp);
    if s.is_empty() {
        return 0.0;
    }
    if s.len() % 2 == 1 {
        s[s.len() / 2]
    } else {
        f64::midpoint(s[s.len() / 2 - 1], s[s.len() / 2])
    }
}

/// Today, UTC, as `YYYY-MM-DD`.
///
/// `saber-plan` keeps `Provenance::when` a string so that it stays dependency-free; this is the
/// civil-date conversion that would otherwise be a calendar crate (Hinnant's algorithm).
fn today_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let days = (secs / 86_400) as i64;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use saber_plan::{AutotuneRecord, ShapeKeys};
    use std::collections::BTreeSet;

    fn samples(kernel: &str, split: Option<i64>, us: f64) -> VariantSamples {
        let mut params = BTreeMap::new();
        if let Some(ks) = split {
            params.insert("split_k".to_string(), ks);
        }
        VariantSamples {
            plugin: "mxfp4_gemm_decode".to_string(),
            kernel: kernel.to_string(),
            params,
            variant_id: variant_id_of(kernel),
            samples: vec![us; 5],
        }
    }

    /// Only an output-preserving variant may be bound: a split changes how the k-sum associates,
    /// so a record that moved it would make the engine's bytes differ from prod's.
    ///
    /// The cells this stopped, from the IC-cold cache of 2026-09-20: `n=8192 k=5120` at M=20 and
    /// M=40 (the measurement wanted ks=1 where the rules say ks=2, by 3.9% and 3.0%), `n=5120
    /// k=3072` at M=20 and `n=5120 k=8704` at M=10 (ks=2 against the rules' ks=4, by 3.6% and
    /// 2.2%). `saber gate --autotune` had replayed all four and found 3-12 differing bytes.
    #[test]
    fn a_variant_that_changes_the_split_is_measured_and_not_bound() {
        let measured = vec![
            samples("mxfp4_gemm_decode_bk128_ks1_tm2_wperm1_nt1", Some(1), 53.5),
            samples("mxfp4_gemm_decode_bk128_ks2_tm2_wperm1_nt1", Some(2), 55.6),
            samples("mxfp4_gemm_decode_bk64_ks4_tm2_wperm1_nt1", Some(4), 61.0),
        ];
        // The reference launcher takes ks=2 here, so ks=1 and ks=4 are measured and dropped.
        let (keep, drop) = split_preserving(&measured, Some(2));
        assert_eq!(
            keep.iter().map(|v| v.kernel.as_str()).collect::<Vec<_>>(),
            vec!["mxfp4_gemm_decode_bk128_ks2_tm2_wperm1_nt1"]
        );
        assert_eq!(drop.len(), 2, "both other splits are dropped");
        // The FASTER one is among the dropped: that is what the rule costs, and it is printed.
        assert!(drop.iter().any(|v| v.params["split_k"] == 1));

        // A family with no split -- folded, A-tiled -- is untouched by the rule.
        let folded = vec![
            samples("mxfp4_gemm_folded_tn2_wperm1_epifast1", None, 300.0),
            samples("mxfp4_gemm_folded_tn4_wperm1_epifast1", None, 320.0),
        ];
        let (keep, drop) = split_preserving(&folded, Some(1));
        assert_eq!(keep.len(), 2);
        assert!(drop.is_empty());
    }

    /// The negative control: with no reference split to compare against -- which is what this code
    /// did before 2026-09-20, and what it still does for every saber op -- the faster
    /// split-changing variant IS bindable. If this passed while the test above also passed by
    /// accident, the rule would be doing nothing.
    #[test]
    fn without_a_reference_split_the_faster_variant_is_bindable_again() {
        let measured = vec![
            samples("mxfp4_gemm_decode_bk128_ks1_tm2_wperm1_nt1", Some(1), 53.5),
            samples("mxfp4_gemm_decode_bk128_ks2_tm2_wperm1_nt1", Some(2), 55.6),
        ];
        let (keep, drop) = split_preserving(&measured, None);
        assert_eq!(keep.len(), 2, "nothing is filtered without a reference");
        assert!(drop.is_empty());
    }

    fn args(out: PathBuf) -> TuneArgs {
        TuneArgs {
            manifest: repo().join("examples/qwen38-27b-nvfp4-tp2.toml"),
            prebuilt: repo().join("prebuilt/gfx1201"),
            out,
            autotune_in: None,
            only: None,
            warmup: 5,
            samples: 20,
            budget_mib: 520,
            device: 0,
            power_cap_w: None,
            metrics_addr: "127.0.0.1:8080".to_string(),
            list: false,
            dump_variants: false,
            // The tests never spawn the rank processes: they have no GPU.
            no_ar_timing: true,
            ic_hot: false,
            no_prod_check: true,
            hip_lib: saber_hip::DEFAULT_HIP_SONAME.to_string(),
        }
    }

    fn repo() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
    }

    /// Fixtures go under `target/`, never `/tmp`: `/tmp` is tmpfs here and the repo has already
    /// been bitten once by a test that silently exercised a different path because of it.
    fn scratch_dir(name: &str) -> PathBuf {
        let dir = repo().join("target").join("tune-tests").join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn cap(card: &str, watts: u32) -> PowerCap {
        PowerCap {
            card: card.to_string(),
            path: None,
            watts,
        }
    }

    /// The committed cache must bind the committed manifest, with **zero** untuned routes.
    ///
    /// This is a CI guard, not a measurement. Everything the autotune key carries — the manifest,
    /// the device profile, the model spec, the bindable registry rows and the plugin build hashes
    /// — is committed to this repo, so "the cache went stale" is a fact a test can establish
    /// rather than something the next person discovers by running a sweep and reading 56 misses.
    /// It has already happened twice: once when the model spec gained the saber ops, and once
    /// when a plugin for another tier moved the registry fingerprint.
    ///
    /// If this fails, re-run the sweep and commit the cache it writes. Since I18 the re-sweep is
    /// INCREMENTAL: a record is keyed on the digest of the tables its own route could bind, so a
    /// one-table edit misses only that table's routes and `--autotune-in` re-measures exactly
    /// those. `tools/capture/run_tune.sh` feeds the committed cache back in by default:
    ///
    /// ```text
    /// tools/capture/run_tune.sh              # incremental
    /// tools/capture/run_tune.sh --full       # everything, which a registry change needs
    /// cp .cache/autotune/gfx1201-210w.json examples/autotune-gfx1201-210w.json
    /// ```
    /// The all-reduce slot is sized by the plan's traffic, not by a constant.
    ///
    /// This is the difference between a measured prefill all-reduce and a refused one: the
    /// one-shot 2-rank kernel refuses a message wider than the IPC scratch slot, because past
    /// that width prod falls back to RCCL. With the old fixed 4 MiB slot the gate answered
    /// `REFUSED (launch rule: 83886080 B exceeds the 4194304 B IPC scratch slot)` for both
    /// prefill buckets, and the routes were bound with no number for a reason that was really an
    /// artifact of how the sweep asked.
    #[test]
    fn the_all_reduce_slot_is_sized_from_the_largest_message() {
        // A decode-only plan: every message is far below the floor, so the floor is what is
        // asked for and nothing about the old behaviour changes.
        let (slot, budget) = ar_slot_and_budget(204_800);
        assert_eq!(slot, AR_SLOT_FLOOR_BYTES);
        assert_eq!(budget, AR_BUDGET_FLOOR_MIB);

        // 8192 prefill tokens x 5120 hidden x 2 bytes = 80 MiB. The slot must hold it whole and
        // the budget must cover three of them.
        let eighty_mib = 8192 * 5120 * 2;
        let (slot, budget) = ar_slot_and_budget(eighty_mib);
        assert_eq!(slot, 80 << 20, "the slot holds the whole message");
        assert!(
            slot >= eighty_mib,
            "a message wider than the slot is refused"
        );
        assert_eq!(budget, 240, "local + peer + reference");

        // Rounded up to a whole MiB, never down onto a message that would then be refused.
        let (slot, _) = ar_slot_and_budget((5 << 20) + 1);
        assert_eq!(slot, 6 << 20);
    }

    #[test]
    fn the_committed_cache_binds_the_committed_manifest() {
        let a = args(scratch_dir("committed").join("unused.json"));
        let ctx = Ctx::load(&a).expect("manifest + artifacts");
        let resolved = ctx.resolve().expect("resolve");
        let cache = AutotuneCache::load(&repo().join("examples/autotune-gfx1201-210w.json"))
            .expect("the committed cache parses");

        let missing = resolved.missing(&cache);
        assert!(
            missing.is_empty(),
            "examples/autotune-gfx1201-210w.json is STALE: {} of {} routes have no record \
             (first: {} {}). Re-run `saber tune` and commit the cache it writes.",
            missing.len(),
            resolved.requirements().len(),
            missing[0].bucket.label(),
            missing[0].op_name
        );

        // Binding is the other half: a record can exist and still name an implementation the
        // constraint match does not allow.
        let plan = resolved.bind(&cache).expect("the committed cache binds");
        assert!(plan.verify_hash());
        let ops: usize = plan.buckets().iter().map(|b| b.ops.len()).sum();
        assert_eq!(ops, resolved.requirements().len(), "every route bound");

        // The per-route POLICY digest is real, not nominal (I18). A route is keyed on the tables
        // its own candidates could bind, so routes in different plugins must carry different
        // policy digests -- if they all carried one, an edit to any table would re-key all 85
        // records again and `--autotune-in` would buy nothing. This is that split, asserted on
        // the committed cache rather than on a constructed one.
        let policies = cache.policy_digests_by_op();
        // THE guard, as one function: `saber tune` runs it on the file it writes and
        // `an_incremental_merge_lands_where_a_full_sweep_would` runs it on the merge's output, so
        // the three cannot drift apart. It fired on main on 2026-09-21 (I19).
        cache
            .one_policy_digest_per_op()
            .unwrap_or_else(|split| panic!("{split}"));
        let distinct: std::collections::BTreeSet<&str> =
            policies.values().flat_map(|d| d.iter().copied()).collect();
        assert!(
            distinct.len() >= 3,
            "the cache carries {} distinct policy digest(s): the per-plugin split is not \
             happening, so a one-line rules edit still costs a full sweep",
            distinct.len()
        );
        // The glue routes and the GEMM routes cannot share one, or editing `attn_glue` would
        // re-key the linears.
        let of = |op: &str| -> Option<String> {
            policies
                .get(op)
                .and_then(|d| d.iter().next().map(|s| (*s).to_string()))
        };
        if let (Some(glue), Some(gemm)) = (of("qk_norm_rope"), of("gemm_w4a8")) {
            assert_ne!(
                glue, gemm,
                "attn_glue and mxfp4 must not share a policy digest"
            );
        }

        // And the cache says who measured it, so a hand-written one cannot pass unnoticed.
        for entry in cache.entries() {
            assert_eq!(entry.record.provenance.who, "saber tune", "{:?}", entry.key);
        }
    }

    /// The incremental merge, over the **committed** cache and the committed tables rather than a
    /// fixture: a one-row rules edit, then the file `saber tune --autotune-in` would write.
    ///
    /// `an_incremental_merge_lands_where_a_full_sweep_would` (saber-plan) states the law; this is
    /// the same law against the 85 records on disk and the three real policy digests, which is
    /// what fired on 2026-09-21. It asserts the count, too: a merge that dropped everything would
    /// satisfy the guard and cost a full sweep, which is the failure the guard cannot see.
    #[test]
    fn the_committed_cache_survives_a_one_row_rules_edit() {
        let a = args(scratch_dir("incremental").join("unused.json"));
        let ctx = Ctx::load(&a).expect("manifest + artifacts");
        let committed = AutotuneCache::load(&repo().join("examples/autotune-gfx1201-210w.json"))
            .expect("the committed cache parses");

        // One row of `attn_glue_gfx1201.toml` moved — F's rope row, the edit that re-keyed the
        // glue routes on 2026-09-21.
        let owned: Vec<(String, String)> = saber_kernels::committed_rule_tables()
            .iter()
            .map(|(n, t)| {
                let text = if *n == "attn_glue_gfx1201.toml" {
                    format!("{t}\n# one row moved\n")
                } else {
                    (*t).to_owned()
                };
                ((*n).to_owned(), text)
            })
            .collect();
        let borrowed: Vec<(&str, &str)> = owned
            .iter()
            .map(|(n, t)| (n.as_str(), t.as_str()))
            .collect();
        let mut edited = ctx.registry.clone();
        edited.set_policy_tables(&borrowed);
        let after = saber_plan::resolve(&ctx.manifest, &ctx.device, &ctx.model, &edited)
            .expect("resolve over the edited policy");

        let re_keyed = after.missing(&committed);
        assert!(!re_keyed.is_empty(), "an attn_glue edit re-keys the glue");
        let (merged, stale) = after.carry_forward(&committed);

        // The guard, on the merge's own output — the same function the check above calls.
        merged
            .one_policy_digest_per_op()
            .unwrap_or_else(|split| panic!("{split}"));

        // Only the glue records moved: everything else is still the measurement it was, so the
        // sweep this edit costs is the glue routes and nothing more.
        assert!(
            stale
                .iter()
                .all(|s| matches!(s.why, saber_plan::StaleWhy::Superseded(_))),
            "{stale:?}"
        );
        for s in &stale {
            assert!(
                ["qk_norm_rope", "attn_gate_quant", "kv_append"].contains(&s.key.op.as_str()),
                "an attn_glue edit cannot reach {}",
                s.key.op
            );
        }
        assert_eq!(
            merged.len() + stale.len(),
            committed.len(),
            "every committed record is either carried or dropped, never both and never neither"
        );
        assert!(
            merged.len() > committed.len() / 2,
            "{} of {} record(s) survived a ONE-ROW edit; the merge is dropping routes the edit \
             cannot reach, which is a full sweep wearing an incremental sweep's name",
            merged.len(),
            committed.len()
        );
        // And what is left to measure is exactly what was dropped.
        assert_eq!(
            after.missing(&merged).len(),
            re_keyed.len(),
            "the work list after the merge is the routes whose key moved"
        );
    }

    /// **Every case the committed plan sweeps builds its index-law inputs by the step's own law.**
    ///
    /// The `attn_prefill_paged` records in every cache up to and including I17 were measurements
    /// of a cell the engine does not run: `seqused_k` was `max_ctx` at both phases, where
    /// `forward::index_table` gives a prefill chunk `done_before = 0` and therefore its own length
    /// (I18, against F's census; 1208.9 µs became 285.1 at m=2048/kv=2, and the m=8192 control
    /// moved 0.7%). Nothing in the key or the provenance could say so — the record is keyed on the
    /// SHAPE, the shape was right, the INPUT was wrong.
    ///
    /// This is the whole sweep asked that question at once. It runs on the host over the committed
    /// manifest, so the answer is about the routes the shipped cache holds records for.
    #[test]
    fn every_swept_case_derives_its_index_inputs_from_the_steps_law() {
        let a = args(scratch_dir("indexlaws").join("cache.json"));
        let ctx = Ctx::load(&a).expect("manifest + artifacts");
        let resolved = ctx.resolve().expect("resolve");
        let missing = resolved.missing(&AutotuneCache::new());
        let (items, _) = work_list(&missing, None).expect("work list");

        let mut swept = 0usize;
        let mut declared: BTreeSet<(String, &'static str)> = BTreeSet::new();
        for item in &items {
            let Work::Sweep(op) = &item.work else {
                continue;
            };
            swept += 1;
            for note in
                saber_exec::check_index_laws(op).unwrap_or_else(|e| panic!("{}: {e}", item.label()))
            {
                declared.insert((item.key.op.clone(), note.buffer));
            }
        }
        assert!(swept > 60, "only {swept} case(s) checked");

        // The divergences that remain are enumerated, not counted: a new one has to be added here
        // by hand, with the reason at its law map, or this test fails.
        //
        // `qk_norm_rope`'s `cos_sin` and `pos` were here until I20 and are DERIVED now. The cell
        // could not state the band the step sizes the rope table by, nor the `done_before` it
        // reads `pos` at, because the band was in neither the shape nor the key; it is in the key
        // now (`BucketKey::MaxCtx` on `layer.attn.qk_norm_rope`) and `QkCell` carries it to the
        // fill. What was measured before, at `b1s5c8192`: a **640-byte** table where the
        // step binds **1 MiB**, at positions `0..5` where the step feeds `8187..8191`.
        // (I19's entry says 1.2 KiB against 2 MiB; the measured sizes are
        // `max_pos * ROTARY_DIM * 2` = 5 x 64 x 2 and 8192 x 64 x 2.)
        //
        // The two that stay are `naccept`, and they stay for a reason no field can fix: the cell
        // carries no acceptance history, and both values are legal answers to one law.
        let names: BTreeSet<(&str, &str)> = declared
            .iter()
            .map(|(op, b)| (op.as_str(), *b))
            .collect::<BTreeSet<_>>();
        assert_eq!(
            names,
            [
                ("gdn_conv_update", "naccept"),
                ("gdn_fused_update", "naccept"),
            ]
            .into_iter()
            .collect::<BTreeSet<_>>(),
            "the set of buffers the sweep states differently from the step has changed"
        );
    }

    /// The half of I20's index-law change that a count cannot state: the `qk_norm_rope` cells the
    /// committed plan sweeps bind the step's OWN rope table and the step's OWN positions.
    ///
    /// Asserted against the numbers, not against "derived": a cell that derived a 640-byte
    /// table from a band of 5 would satisfy the checker and still be the thing I19 named. At
    /// `b1s5c8192` the band is 8192, the table is 8192 positions of 64 rotary dims in bf16 =
    /// 1 MiB, and the five rows sit at 8187..8191.
    #[test]
    fn the_qk_cells_carry_the_buckets_context_band() {
        let a = args(scratch_dir("qkband").join("cache.json"));
        let ctx = Ctx::load(&a).expect("manifest + artifacts");
        let resolved = ctx.resolve().expect("resolve");
        let missing = resolved.missing(&AutotuneCache::new());
        let (items, _) = work_list(&missing, None).expect("work list");

        let mut seen = 0usize;
        for item in &items {
            let Work::Sweep(TuneOp::GlueQkNormRope(c)) = &item.work else {
                continue;
            };
            seen += 1;
            assert_eq!(c.max_ctx, 8192, "every committed bucket bands at 8192");
            assert_eq!(c.num_seqs * c.q_len, c.shape.m, "{}", item.label());
            // The rope table is the BAND's, never the chunk's.
            assert_eq!(c.max_pos(), 8192, "{}", item.label());
            // ... and a decode bucket's sequences sit at the top of it.
            let want_done = if c.q_len == c.shape.m && c.num_seqs == 1 && c.shape.m > 8 {
                0 // the prefill chunks: p2048 and p8192, first thing in their sequence
            } else {
                8192 - c.q_len
            };
            assert_eq!(c.done_before(), want_done, "{}", item.label());
        }
        assert_eq!(seen, 6, "one qk_norm_rope cell per bucket");
    }

    #[test]
    fn the_work_list_collapses_routes_that_share_a_key() {
        let a = args(scratch_dir("worklist").join("cache.json"));
        let ctx = Ctx::load(&a).expect("manifest + artifacts");
        let resolved = ctx.resolve().expect("resolve");
        let missing = resolved.missing(&AutotuneCache::new());
        // 4 decode buckets x 14 ops (8 merged linears, paged attention, the THREE
        // attention-glue routes, the all-reduce, and exactly one of the two GDN decode routes)
        // + 2 prefill buckets x 17 (the same 8 linears and 3 glue routes, query-tiled
        // attention, the GDN chunk trio, the gated norm, the all-reduce).
        assert_eq!(
            missing.len(),
            90,
            "4 decode + 2 prefill buckets, all untuned"
        );
        let (items, untunable) = work_list(&missing, None).expect("work list");
        // Nothing is outside the sweep any more: the saber decode routes are measured and the
        // all-reduce is bound without a number, which is a work item rather than a skip.
        assert!(untunable.is_empty(), "{untunable:?}");
        // 8 distinct GEMM geometries x 6 bucket M, plus per decode bucket paged attention,
        // one GDN decode route and the all-reduce, plus per prefill bucket query-tiled
        // attention, the GDN chunk trio, the gated norm and the all-reduce.
        // 67, not 68: `lm_head` runs on the SAMPLED positions, so both prefill buckets ask
        // for it at M=1 and the two collapse onto one key and one measurement. Plus the
        // attention glue's 18 routes (3 ops x 6 buckets) on 18 keys of their own -- the M
        // differs in every bucket, so none of them share. 83, not 85, since G2: with the tiled
        // GDN norm `gdn.out_proj` and `attn.o_proj` share their key at the prefill buckets too.
        assert_eq!(items.len(), 83);
        assert_eq!(items.iter().map(|i| i.routes.len()).sum::<usize>(), 90);
        let gemm = items
            .iter()
            .filter(|i| matches!(i.work, Work::Sweep(TuneOp::Mxfp4 { .. })))
            .count();
        // 8 rows x 6 buckets = 48 geometries, of which 4 collapse: `layer.gdn.out_proj` and
        // `layer.attn.o_proj` are both n=5120 k=3072, so at a DECODE bucket they share one key
        // and one measurement. Under the reference's law they did not at a prefill bucket --
        // `a_layout` separated them, out_proj's edge being row-major -- and the per-edge
        // obligation did real work in the keys. G2's tiled GDN norm makes both edges A-tiled, so
        // they share at all six buckets: 48 - 6 - 1 (lm_head's two prefill M=1 routes) = 41.
        assert_eq!(gemm, 41);
        // Bound without a number: one all-reduce key per bucket (6), plus the attention glue's
        // three at each of the six (18). The ten prefill saber keys were here too until
        // 2026-09-19; they are swept now.
        assert_eq!(
            items
                .iter()
                .filter(|i| matches!(i.work, Work::Unmeasurable { .. }))
                .count(),
            6
        );
        // And the glue's 18 are SWEPT, not bound: the sweeper builds their inputs now.
        assert_eq!(
            items
                .iter()
                .filter(|i| matches!(
                    i.work,
                    Work::Sweep(
                        TuneOp::GlueQkNormRope(_)
                            | TuneOp::GlueGateQuant(_)
                            | TuneOp::GlueKvAppend(_)
                    )
                ))
                .count(),
            18
        );
        assert_eq!(
            items
                .iter()
                .filter(|i| matches!(
                    i.work,
                    Work::Sweep(
                        TuneOp::AttnPrefill(_)
                            | TuneOp::GdnConvPrep(_)
                            | TuneOp::GdnKktSolve(_)
                            | TuneOp::GdnChunkScan(_)
                            | TuneOp::GdnNorm(_)
                    )
                ))
                .count(),
            10,
            "five prefill routes at each of the two prefill buckets"
        );
        assert_eq!(
            items.iter().filter(|i| i.key.op == "allreduce").count(),
            6,
            "one all-reduce key per bucket"
        );
        // The fused GDN route exists at batch 1 only; every other bucket takes the pair.
        assert_eq!(
            items
                .iter()
                .filter(|i| matches!(i.work, Work::Sweep(TuneOp::GdnFused(_))))
                .count(),
            1
        );
        assert_eq!(
            items
                .iter()
                .filter(|i| matches!(i.work, Work::Sweep(TuneOp::GdnPair(_))))
                .count(),
            3
        );
        // Sorted so that consecutive items share their input buffers.
        let mut seen: BTreeSet<(u32, u32)> = BTreeSet::new();
        let mut last = (u32::MAX, u32::MAX);
        for i in &items {
            let (a, b, _) = i.order();
            if (a, b) != last {
                assert!(
                    seen.insert((a, b)),
                    "shape {last:?} was revisited after moving on"
                );
                last = (a, b);
            }
        }
    }

    /// The shapes the saber routes are measured at come from the bucket where the shape keys are
    /// only a necessary condition, and from `capture_ref.py` everywhere else.
    #[test]
    fn the_saber_shapes_are_built_from_the_bucket_and_the_capture() {
        let a = args(scratch_dir("sabershapes").join("cache.json"));
        let ctx = Ctx::load(&a).expect("ctx");
        let resolved = ctx.resolve().expect("resolve");
        let missing = resolved.missing(&AutotuneCache::new());
        let (items, _) = work_list(&missing, None).expect("work list");

        let attn = items
            .iter()
            .find_map(|i| match i.work {
                Work::Sweep(TuneOp::AttnDecode(s)) if s.num_seqs == 8 => Some(s),
                _ => None,
            })
            .expect("the 8x5 bucket's attention route");
        assert_eq!(attn.q_heads, 12, "24 q heads at TP=2");
        assert_eq!(attn.kv_heads, 2, "q_heads / gqa");
        assert_eq!(attn.head_dim, 256);
        // The context band is the BUCKET's; the shape keys do not carry it.
        assert_eq!(attn.max_ctx, 8192);
        assert_eq!(attn.max_blocks, 8192 / 16);
        assert_eq!(attn.kv_block_stride, 2 * 16 * 2 * 256);
        assert_eq!(attn.kv_head_stride, 16 * 2 * 256);
        assert_eq!(attn.splits, 0, "let the split law choose, as prod does");

        let gdn = items
            .iter()
            .find_map(|i| match i.work {
                Work::Sweep(TuneOp::GdnPair(s)) if s.n_seqs == 8 => Some(s),
                _ => None,
            })
            .expect("the 8x5 bucket's GDN route");
        assert_eq!(gdn.h, 24, "48 value heads at TP=2");
        assert_eq!(gdn.hg, 8, "16 key heads at TP=2");
        assert_eq!(gdn.maxq, 5, "the bucket's speculative width");
        assert_eq!(gdn.o_rows, 40);
        assert_eq!(gdn.xpitch, (2 * 8 + 24) * 128, "the 5120-wide qkvz split");
        assert_eq!(gdn.cs_dim, i64::from(4 - 1 + 5 - 1), "conv state length");
        assert_eq!(gdn.st_head, 128 * 128);
        assert_eq!(gdn.ci_stride, 5, "cache_idx is sidx[:, 0], stride maxq");

        // The all-reduce is bound, not swept, and its message is hidden x rows.
        let ar = items
            .iter()
            .filter(|i| i.key.op == "allreduce")
            .find_map(|i| match i.work {
                Work::Unmeasurable { shape, .. } => Some(shape),
                _ => None,
            })
            .expect("an all-reduce route");
        let saber::Shape::Ar(ar) = ar else {
            panic!("the all-reduce carries an Ar shape");
        };
        assert_eq!(ar.world_size, 2);
        assert_eq!(ar.elem_bytes, 2, "bf16");
        assert_eq!(ar.n_elem % 5120, 0, "hidden x rows");

        // ONE reason a route carries no number now: the all-reduce needs a second PROCESS.
        // The five prefill saber routes were a second reason until 2026-09-19 and the attention
        // glue a third until 2026-09-20; both have input builders in the sweeper now, so both
        // carry measurements and the test says so rather than keeping a stale expectation.
        let unmeasured: std::collections::BTreeMap<&str, &str> = items
            .iter()
            .filter_map(|i| match i.work {
                Work::Unmeasurable { reason, .. } => Some((i.key.op.as_str(), reason)),
                Work::Sweep(_) => None,
            })
            .collect();
        assert_eq!(
            unmeasured.keys().copied().collect::<Vec<_>>(),
            vec!["allreduce"],
            "the 2-rank op is the only route out of this process's reach"
        );
        assert_eq!(unmeasured["allreduce"], AR_NOT_MEASURED);
        // The glue is NOT in that map: it is swept, so it carries a measurement like every
        // other route the sweeper can build inputs for.
        for op in ["qk_norm_rope", "attn_gate_quant", "kv_append"] {
            assert!(!unmeasured.contains_key(op), "{op} should be swept");
        }

        // And every prefill route IS swept, at the geometry `capture_ref.py` proved it on.
        let prefill: std::collections::BTreeMap<&str, TuneOp> = items
            .iter()
            .filter_map(|i| match i.work {
                Work::Sweep(op)
                    if matches!(
                        op,
                        TuneOp::AttnPrefill(_)
                            | TuneOp::GdnConvPrep(_)
                            | TuneOp::GdnKktSolve(_)
                            | TuneOp::GdnChunkScan(_)
                            | TuneOp::GdnNorm(_)
                    ) =>
                {
                    Some((i.key.op.as_str(), op))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            prefill.keys().copied().collect::<Vec<_>>(),
            vec![
                "attn_prefill_paged",
                "gdn_chunk_scan",
                "gdn_conv_prep",
                "gdn_gated_rmsnorm",
                "gdn_kkt_solve",
            ]
        );
        let TuneOp::GdnChunkScan(scan) = prefill["gdn_chunk_scan"] else {
            panic!("the chunk scan carries a prefill shape");
        };
        assert_eq!(scan.n_seqs, 1, "one sequence per prefill chunk");
        assert_eq!(scan.chunk, 64);
        assert_eq!(scan.h, 24);
        assert_eq!(scan.hg, 8);
        // A prefill carries no speculative window, so the conv state is width - 1 and no more.
        let TuneOp::GdnConvPrep(prep) = prefill["gdn_conv_prep"] else {
            panic!("the conv prep carries a prefill shape");
        };
        assert_eq!(prep.cs_dim, i64::from(prep.conv_width - 1));
        assert_eq!(prep.xpitch, (2 * 8 + 24) * 128, "the 5120-wide qkvz split");
        // The norm runs one row per (token, head).
        let TuneOp::GdnNorm(norm) = prefill["gdn_gated_rmsnorm"] else {
            panic!("the norm carries a norm shape");
        };
        assert_eq!(norm.channels, 128);
        assert_eq!(norm.rows % 24, 0, "tokens x heads");
    }

    #[test]
    fn only_filters_the_work_list() {
        let a = args(scratch_dir("only").join("cache.json"));
        let ctx = Ctx::load(&a).expect("ctx");
        let resolved = ctx.resolve().expect("resolve");
        let missing = resolved.missing(&AutotuneCache::new());
        let (items, _) = work_list(&missing, Some("lm_head")).expect("work list");
        // 5, not 6: the two prefill buckets both run lm_head at M=1 (one sampled position
        // per sequence), so they share a key.
        assert_eq!(items.len(), 5);
        assert!(items.iter().all(|i| matches!(
            i.work,
            Work::Sweep(TuneOp::Mxfp4 {
                shape: Shape { n: 124_160, .. },
                ..
            })
        )));
    }

    #[test]
    fn a_key_the_sweep_cannot_launch_is_an_error_not_a_guess() {
        let key = AutotuneKey::new(
            "flash_attention",
            ShapeKeys::new().with("m", 8i64).with("n", 128i64),
            210,
            "plan",
            "plugins",
            "policy",
        )
        .expect("key");
        let req = Requirement {
            bucket: saber_plan::Bucket::decode(1, 5, 8192, true),
            op_name: "layer.attn.core".to_string(),
            key,
            candidates: vec![],
        };
        // It is REPORTED, not guessed at and not silently dropped: the op comes back in the
        // untunable set with its route count, and no work item is produced for it.
        let (items, untunable) = work_list(&[req], None).expect("an unbuildable op is a report");
        assert!(items.is_empty(), "nothing may be invented for it");
        assert_eq!(untunable["flash_attention"], 1);
        // And an op the sweep CAN launch, whose shape is malformed, is still a hard error --
        // widening `by_value` must not have widened this.
        let bad = Requirement {
            bucket: saber_plan::Bucket::decode(1, 5, 8192, true),
            op_name: "layer.mlp.down_proj".to_string(),
            key: AutotuneKey::new(
                "gemm_w4a8",
                saber_plan::ShapeKeys::new().with("m", 5i64),
                210,
                "input",
                "plugins",
                "policy",
            )
            .expect("key"),
            candidates: vec![],
        };
        let err = work_list(&[bad], None).expect_err("a gemm with no N is malformed");
        assert!(format!("{err}").contains('n'), "{err}");
    }

    #[test]
    fn the_power_cap_gate_refuses_anything_but_both_cards_at_the_manifest_cap() {
        // The condition the ledger's timing section runs under.
        assert!(check_power_caps(&[cap("card0", 210), cap("card1", 210)], 210).is_ok());
        // Negative controls, each of which has actually happened on this box.
        let uncapped = check_power_caps(&[cap("card0", 210), cap("card1", 300)], 210)
            .expect_err("one card off the policy");
        assert!(uncapped.contains("card1 at 300 W"), "{uncapped}");
        let reset = check_power_caps(&[cap("card0", 300), cap("card1", 300)], 210)
            .expect_err("a MODE1 reset cleared the policy on both");
        assert!(reset.contains("300 W"), "{reset}");
        let one = check_power_caps(&[cap("card0", 210)], 210).expect_err("only one card visible");
        assert!(one.contains("--power-cap-w"), "{one}");
        assert!(check_power_caps(&[], 210).is_err(), "no sysfs at all");
    }

    #[test]
    fn power_caps_are_read_in_watts_from_microwatts() {
        // A fake sysfs tree with the real layout, including a connector and a card with no hwmon.
        let root = scratch_dir("sysfs");
        let hw = root.join("card0/device/hwmon/hwmon2");
        std::fs::create_dir_all(&hw).unwrap();
        std::fs::write(hw.join("power1_cap"), "210000000\n").unwrap();
        let hw = root.join("card1/device/hwmon/hwmon3");
        std::fs::create_dir_all(&hw).unwrap();
        std::fs::write(hw.join("power1_cap"), "210000000\n").unwrap();
        std::fs::create_dir_all(root.join("card2/device")).unwrap();
        std::fs::create_dir_all(root.join("card0-DP-1")).unwrap();
        let caps = read_power_caps(&root);
        assert_eq!(caps.len(), 2, "{caps:?}");
        assert_eq!(caps[0].card, "card0");
        assert_eq!(caps[0].watts, 210);
        assert_eq!(caps[1].card, "card1");
        assert!(check_power_caps(&caps, 210).is_ok());
    }

    /// record -> save -> load -> bind, and the plan hash is the same one the in-memory cache gives.
    #[test]
    fn a_recorded_sweep_round_trips_through_the_file_to_the_same_plan() {
        let dir = scratch_dir("roundtrip");
        let path = dir.join("autotune.json");
        let a = args(path.clone());
        let ctx = Ctx::load(&a).expect("ctx");
        let resolved = ctx.resolve().expect("resolve");
        let missing = resolved.missing(&AutotuneCache::new());
        let (items, _) = work_list(&missing, None).expect("work list");

        // Two synthetic variants per key, with the faster one second — so the winner is only right
        // if the warm-up discard and the ranking both work.
        let provenance = Provenance::new("test", "2026-09-18", HARNESS);
        let mut tuner = Autotuner::new(AutotuneCache::new(), provenance);
        for (i, item) in items.iter().enumerate() {
            let slow = 100.0 + i as f64;
            let fast = slow - 10.0;
            let impl_name = item
                .candidates
                .last()
                .expect("every route has a candidate")
                .clone();
            let m = vec![
                // The first variant's first sample is the shape's warm-up: inflated, and dropped.
                Measurement::new(
                    &impl_name,
                    VariantParams::new(1).param("split_k", 1),
                    vec![slow * 1.17, slow, slow, slow],
                ),
                Measurement::new(
                    &impl_name,
                    VariantParams::new(2).param("split_k", 2),
                    vec![fast, fast, fast, fast],
                ),
            ];
            let rec = tuner
                .record_measurements(item.key.clone(), m)
                .expect("record");
            assert_eq!(rec.variant.variant_id, 2, "the faster variant wins");
            assert!((rec.measured_us - fast).abs() < 1e-9);
        }
        let mut cache = tuner.into_cache();
        assert_eq!(cache.len(), items.len());

        // The saber routes are not in this sweep (no paged KV cache, no recurrent state here), so
        // fill them with fabricated records to get a bindable plan. Fabricating is legal in a test
        // and nowhere else; what is under test is the record -> file -> plan round trip, not the
        // measurement.
        for req in resolved.missing(&cache) {
            cache.insert(
                req.key.clone(),
                AutotuneRecord {
                    impl_name: req.candidates[0].clone(),
                    variant: VariantParams::new(1),
                    measured_us: 50.0,
                    samples_kept: 4,
                    runner_up: None,
                    unstable: None,
                    provenance: Provenance::new("test", "2026-09-18", HARNESS),
                },
            );
        }

        let in_memory = resolved.bind(&cache).expect("bind in memory");
        cache.save(&path).expect("save");
        let reloaded = AutotuneCache::load(&path).expect("load");
        assert_eq!(reloaded, cache, "the file is the cache");
        assert!(
            resolved.missing(&reloaded).is_empty(),
            "every route answered"
        );
        let from_file = resolved.bind(&reloaded).expect("bind from file");
        assert_eq!(from_file.plan_hash, in_memory.plan_hash);
        assert!(from_file.verify_hash());

        // And the plan says `measured`, with the provenance and the margin the sweep recorded.
        let op = from_file.buckets()[0].ops.ops()[0];
        assert_eq!(op.promotion.state, saber_plan::PromotionState::Measured);
        assert_eq!(
            op.promotion.provenance.as_ref().map(|p| p.harness.as_str()),
            Some(HARNESS)
        );
        assert!(op.margin_pct.expect("a runner-up") > 0.0);

        // Negative control: drop one record and the plan refuses to bind again.
        let mut holed = AutotuneCache::new();
        for (i, e) in reloaded.entries().enumerate() {
            if i > 0 {
                holed.insert(e.key.clone(), e.record.clone());
            }
        }
        assert!(resolved.bind(&holed).is_err(), "a cold key must stay loud");
    }

    #[test]
    fn a_record_naming_an_implementation_outside_the_candidates_is_refused() {
        // `bind` must not accept a record for a route the constraint match rules out — the guard
        // that keeps a stale cache from binding a kernel that cannot run this geometry.
        let dir = scratch_dir("stale");
        let a = args(dir.join("cache.json"));
        let ctx = Ctx::load(&a).expect("ctx");
        let resolved = ctx.resolve().expect("resolve");
        let missing = resolved.missing(&AutotuneCache::new());
        let mut cache = AutotuneCache::new();
        for r in &missing {
            cache.insert(
                r.key.clone(),
                AutotuneRecord::new(
                    "gemm_w4a8_mxfp4_atiled", // legal only at M >= 513; no bucket here is
                    VariantParams::new(7),
                    42.0,
                    Provenance::new("test", "2026-09-18", HARNESS),
                ),
            );
        }
        assert!(resolved.bind(&cache).is_err());
    }

    #[test]
    fn the_plugin_to_implementation_map_is_taken_from_the_candidate_set() {
        let a = args(scratch_dir("implmap").join("cache.json"));
        let ctx = Ctx::load(&a).expect("ctx");
        let cands = vec![
            "gemm_w4a8_mxfp4_decode".to_string(),
            "gemm_w4a8_mxfp4_folded".to_string(),
        ];
        assert_eq!(
            impl_for_plugin(&ctx.registry, &cands, "mxfp4_gemm_decode").unwrap(),
            "gemm_w4a8_mxfp4_decode"
        );
        // A plugin outside the candidate set is an error: the planner would refuse the record.
        assert!(impl_for_plugin(&ctx.registry, &cands, "mxfp4_gemm_atiled").is_err());
    }

    #[test]
    fn the_date_is_a_real_civil_date() {
        let t = today_utc();
        assert_eq!(t.len(), 10, "{t}");
        let parts: Vec<u32> = t.split('-').map(|p| p.parse().unwrap()).collect();
        assert!(parts[0] >= 2026);
        assert!((1..=12).contains(&parts[1]));
        assert!((1..=31).contains(&parts[2]));
    }
}

#[cfg(test)]
mod ar_tests {
    use super::*;

    /// The case line the gate parses, from the geometry the plan binds.
    #[test]
    fn an_all_reduce_case_line_states_the_message_and_the_family() {
        let shape = saber::ArShape {
            n_elem: 25_600,
            elem_bytes: 2,
            world_size: 2,
        };
        assert_eq!(
            ar_case_line(
                &shape,
                &saber::Op::AllReduceExact {
                    dtype: saber::ArPayload::Bf16
                }
            ),
            "ar_exact 25600 2 bf16"
        );
        assert_eq!(
            ar_case_line(
                &shape,
                &saber::Op::AllReduceWht {
                    bits: 6,
                    dtype: saber::ArDtype::Bf16
                }
            ),
            "ar_wht 25600 2 6 bf16",
            "the width is in the line: wht4 and wht6 are different kernels"
        );
    }

    /// The harness reads a tab-separated marker line, not the gate's human table.
    #[test]
    fn the_timing_lines_are_parsed_and_the_max_is_what_is_taken() {
        let out = concat!(
            "ar-gate        2 case(s)\n",
            "ar_exact 25600 2 bf16   saber_ar_oneshot_2rank_exact_bf16   51200 OK\n",
            "AR-TIMING\tar_exact 25600 2 bf16\tsaber_ar_oneshot_2rank_exact_bf16",
            "\trank0=12.202\trank1=12.074\tmax=12.202\n",
            "AR-TIMING\tar_wht 204800 2 6 bf16\tsaber_ar_oneshot_2rank_wht6_bf16",
            "\trank0=25.982\trank1=26.004\tmax=26.004\n",
            "2 passed, 0 FAILED\n"
        );
        let got = parse_ar_timing(out).expect("parses");
        assert_eq!(got.len(), 2);
        let (kernel, us) = &got["ar_exact 25600 2 bf16"];
        assert_eq!(kernel, "saber_ar_oneshot_2rank_exact_bf16");
        assert!((us - 12.202).abs() < 1e-9);
        // The slower rank is what the step pays, and it is not always rank 0.
        let (_, us) = &got["ar_wht 204800 2 6 bf16"];
        assert!((us - 26.004).abs() < 1e-9, "the max, which here is rank 1");
    }

    /// A malformed line is an error, not a silently missing record: a route with no number would
    /// otherwise be bound as unmeasured with no explanation.
    #[test]
    fn a_malformed_timing_line_is_refused() {
        assert!(parse_ar_timing("AR-TIMING\tonly\tthree\tfields").is_err());
        assert!(parse_ar_timing("AR-TIMING\ta\tb\trank0=1\trank1=2\tmax=nope").is_err());
        assert!(
            parse_ar_timing("AR-TIMING\ta\tb\tc\td\te").is_err(),
            "no max="
        );
        // And a run with no timing lines at all is empty rather than wrong.
        assert!(parse_ar_timing("nothing here").unwrap().is_empty());
    }
}
