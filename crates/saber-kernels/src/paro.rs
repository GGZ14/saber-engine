//! Paro's rotation producers as data — the third plugin.
//!
//! The PARO tier is prod's ROLLBACK unit, so these are a serve path, not an experiment.
//! `ref/paro/ref_paro.hip` decides in C++ which instantiation of the
//! four producers runs — `pq_rot_gpw` (:1026 of `par_kernels.h`), `pq_tok_waves` (:105), the
//! `fused` / `i8` ladders in `launch_add_rms_rot` (:294) and `launch_ew_rot` (:323), and the
//! `W`/`TILED` ladders in the two `_tok` launchers. Here each is a row in
//! `rules/paro_gfx1201.toml` and [`resolve`] is a pure function over them.
//!
//! Two families, and which one runs is the TIER rather than the shape (see the rules file): the
//! per-GROUP producers serve the int4 `paro` tier, the per-TOKEN ones serve
//! `paro_mxfp4`, which is what `paromx` actually runs.

use serde::Deserialize;

use crate::abi::{self, ArgKind, SigArg};
use crate::rules::{ArgValue, LaunchDescriptor, Slot, WorkspaceReq};
use crate::{Error, Result};

// ---------------------------------------------------------------------------- rules (data)

#[derive(Debug, Clone, Deserialize)]
pub struct ParoRules {
    pub schema_version: u32,
    pub name: String,
    pub knobs: Knobs,
    pub rot_gpw: Vec<GpwRow>,
    pub tok_waves: Vec<WavesRow>,
    pub add_rms_rot: Vec<RotRow>,
    pub ew_rot: Vec<EwRow>,
    pub add_rms_rot_tok: Vec<TokRow>,
    pub ew_rot_tok: Vec<EwTokRow>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Knobs {
    pub group: u32,
    pub krot_max: u32,
    pub rot_waves: u32,
    pub rot_block: u32,
    pub lds_max_bytes: u64,
    pub decode_max_m: u32,
    pub i8: bool,
    pub hw_cvt: bool,
    /// **Not a knob of this table.** `a_tiled_min_m` has one source,
    /// [`crate::rules::a_tiled_min_m`]; this field exists only so that a table which states it
    /// again is REFUSED by name rather than shadowing it.
    #[serde(default, rename = "a_tiled_min_m")]
    a_tiled_min_m_restated: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GpwRow {
    pub m_max: Option<u32>,
    pub gpw: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WavesRow {
    pub m_max: Option<u32>,
    pub k_min: Option<u32>,
    pub waves: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RotRow {
    pub rot: bool,
    pub i8: bool,
    pub symbol: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EwRow {
    pub mode: u32,
    pub rot: bool,
    pub i8: bool,
    pub symbol: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokRow {
    pub waves: u32,
    pub tiled: bool,
    pub symbol: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EwTokRow {
    pub mode: u32,
    pub waves: u32,
    pub tiled: bool,
    pub symbol: String,
}

const PROD_RULES: &str = include_str!("../rules/paro_gfx1201.toml");

/// The same text, for the plan input hash: the policy this module enforces.
pub const PROD_RULES_TEXT: &str = PROD_RULES;

/// One plugin holds all four producer families: they come from one reference header and one
/// translation unit, which is also why they share a build hash.
const PLUGIN: &str = "paro_producers";

impl ParoRules {
    /// The rules the `paromx` rollback unit runs (it sets no `SABER_PQ_*` at all).
    pub fn prod_paro() -> Result<Self> {
        Self::from_toml(PROD_RULES)
    }

    pub fn from_toml(text: &str) -> Result<Self> {
        let r: ParoRules = toml::from_str(text).map_err(|e| Error::Parse(e.to_string()))?;
        r.check()?;
        Ok(r)
    }

    fn check(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(Error::Rule(format!(
                "paro rules schema {} (this build speaks 1)",
                self.schema_version
            )));
        }
        let k = &self.knobs;
        if k.rot_block != k.rot_waves * 32 {
            return Err(Error::Rule(format!(
                "rot_block {} is not rot_waves {} x 32",
                k.rot_block, k.rot_waves
            )));
        }
        if !k.group.is_power_of_two() {
            return Err(Error::Rule("group must be a power of two".into()));
        }
        // ONE source for the band edge. This table used to carry its own copy of the number the
        // A-tiled GEMM reads; a copy is what P6 cost an experiment, so a table that states it is
        // refused by name rather than shadowing `rules::a_tiled_min_m`.
        if let Some(v) = k.a_tiled_min_m_restated {
            return Err(Error::Rule(format!(
                "[knobs] a_tiled_min_m = {v}: this table does not own that number. \
                 `rules/mxfp4_gfx1201.toml [knobs] a_tiled_min_m` is the one source and \
                 `saber_kernels::rules::a_tiled_min_m()` is the one reader"
            )));
        }
        // The per-token producer writes the fragment layout the A-tiled GEMM reads. If the band
        // edge ever fell into this tier's decode band a shape would be both, which is silent
        // garbage rather than an error -- so the shared number is checked against THIS table's
        // own `decode_max_m`, which is the part of the pair this file does own.
        let a_tiled_min_m = crate::rules::a_tiled_min_m();
        if a_tiled_min_m <= k.decode_max_m {
            return Err(Error::Rule(format!(
                "a_tiled_min_m {a_tiled_min_m} must exceed decode_max_m {}",
                k.decode_max_m
            )));
        }
        if self.rot_gpw.is_empty() || self.tok_waves.is_empty() {
            return Err(Error::Rule(
                "rot_gpw and tok_waves both need a default row".into(),
            ));
        }
        Ok(())
    }

    /// `pq_rot_gpw(M)`: groups per wave.
    pub fn rot_gpw(&self, m: u32) -> u32 {
        self.rot_gpw
            .iter()
            .find(|r| r.m_max.is_none_or(|x| m <= x))
            .map(|r| r.gpw)
            .unwrap_or(1)
    }

    /// `pq_tok_waves(M, K)`: waves per row for the per-token producers. First match wins.
    pub fn tok_waves(&self, m: u32, k: u32) -> u32 {
        self.tok_waves
            .iter()
            .find(|r| r.m_max.is_none_or(|x| m <= x) && r.k_min.is_none_or(|x| k >= x))
            .map(|r| r.waves)
            .unwrap_or(8)
    }

    /// The launcher's `fused`: at and below `decode_max_m` the producer also rotates and
    /// per-group quantises. Prod computes exactly this and never passes anything else.
    pub fn rot(&self, m: u32) -> bool {
        m <= self.knobs.decode_max_m
    }

    /// `_tiled(M)`: the producer and the A-tiled GEMM take the same layout decision from M, off
    /// the ONE threshold ([`crate::rules::a_tiled_min_m`]).
    pub fn tiled(&self, m: u32) -> bool {
        let edge = crate::rules::a_tiled_min_m();
        edge > 0 && m >= edge
    }

    /// Every symbol the four instantiation tables name, in table order.
    ///
    /// This is the authoritative list of what the plugin must contain: `resolve` can return no
    /// other symbol, and a consumer that wants to know "which code object holds this producer?"
    /// asks it rather than testing a name prefix.
    pub fn symbols(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::new();
        out.extend(self.add_rms_rot.iter().map(|r| r.symbol.as_str()));
        out.extend(self.ew_rot.iter().map(|r| r.symbol.as_str()));
        out.extend(self.add_rms_rot_tok.iter().map(|r| r.symbol.as_str()));
        out.extend(self.ew_rot_tok.iter().map(|r| r.symbol.as_str()));
        out
    }

    /// Which `.hsaco` holds `symbol`, from the tables above.
    ///
    /// The reference's map ([`crate::saber::plugin_for`]) is prefix-ordered because its symbols are spread
    /// over nine code objects. These four families come from ONE reference header and ONE
    /// translation unit, so the map is membership, not spelling: a symbol these rules never name
    /// has no plugin here, and that is a refusal rather than a guess that would send the loader
    /// looking for it in the wrong blob.
    pub fn plugin_for(&self, symbol: &str) -> Result<&'static str> {
        if self.symbols().contains(&symbol) {
            Ok(PLUGIN)
        } else {
            Err(Error::Rule(format!(
                "no Paro plugin holds symbol `{symbol}`; these rules name {} symbols and                  that is not one of them",
                self.symbols().len()
            )))
        }
    }

    /// The one plugin these rules' symbols live in, derived from the tables rather than typed.
    ///
    /// A caller matching an artifact against this is asking the map, which is what keeps the
    /// planner's attribution out of the business of guessing from a file stem.
    pub fn owner(&self) -> Result<&'static str> {
        let first = self
            .symbols()
            .first()
            .copied()
            .ok_or_else(|| Error::Rule("paro rules name no symbols at all".into()))?;
        self.plugin_for(first)
    }
}

// ---------------------------------------------------------------------------- shapes and ops

/// `add_rms_rot` / `add_rms_rot_tok`: `[M, K]` with `P` differently-rotated activation copies (a
/// merged linear carries one per partition; `P <= 3`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParoShape {
    pub m: u32,
    pub k: u32,
    pub p: u32,
    /// Rotation depth, `rec.shape[1]`; the entry point refuses outside `[1, krot_max]`.
    pub krot: u32,
    /// `f32` bits, so the shape stays `Eq`.
    pub eps_bits: u32,
}

/// `ew_rot` / `ew_rot_tok`: `[M, N]`, single partition. `ys` is the second operand's row stride in
/// elements (the gate for mode 1, `z` for mode 2); mode 0 reads `gate_up [M, 2N]` and ignores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParoEwShape {
    pub m: u32,
    pub n: u32,
    pub krot: u32,
    pub ys: i64,
    pub eps_bits: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Rms(ParoShape),
    Ew(ParoEwShape),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Per-GROUP residual-add + RMSNorm producer (the int4 `paro` tier). `ROT` is derived
    /// from M: prod computes `fused = M <= decode_max_m` and never passes anything else.
    AddRmsRot,
    /// Per-GROUP elementwise producer. mode 0 silu-mul, 1 attention gate, 2 GDN gated RMSNorm.
    EwRot { mode: u32 },
    /// Per-TOKEN residual-add + RMSNorm producer (the `paro_mxfp4` tier — what paromx runs).
    AddRmsRotTok,
    /// Per-TOKEN elementwise producer.
    EwRotTok { mode: u32 },
}

// ---------------------------------------------------------------------------- kernarg builder

struct KArgs {
    sig: &'static [SigArg],
    values: Vec<ArgValue>,
}

impl KArgs {
    fn new(sig: &'static [SigArg]) -> Self {
        KArgs {
            sig,
            values: Vec::with_capacity(sig.len()),
        }
    }

    fn buf(mut self, name: &str) -> Self {
        self.values.push(ArgValue::Buffer {
            buffer: name.to_owned(),
        });
        self
    }

    /// A pointer the chosen instantiation provably never dereferences — `a`, `asg` and `rs` on
    /// the `ROT=false` arm, which writes `hs` only. Named, so the descriptor says WHICH buffer is
    /// absent rather than carrying an unexplained zero.
    fn maybe_buf(self, name: &str, present: bool) -> Self {
        if present {
            self.buf(name)
        } else {
            self.null_buf(name)
        }
    }

    fn null_buf(mut self, name: &str) -> Self {
        self.values.push(ArgValue::NullBuffer {
            buffer: name.to_owned(),
        });
        self
    }

    fn i32(mut self, v: i64) -> Self {
        self.values.push(ArgValue::I32 { ival: v });
        self
    }

    fn i64(mut self, v: i64) -> Self {
        self.values.push(ArgValue::I64 { ival: v });
        self
    }

    fn f32(mut self, v: f32) -> Self {
        self.values.push(ArgValue::F32 { fval: v });
        self
    }

    fn finish(self) -> Result<(Vec<Slot>, u32)> {
        if self.values.len() != self.sig.len() {
            return Err(Error::Rule(format!(
                "paro kernarg builder: {} values for a {}-argument signature",
                self.values.len(),
                self.sig.len()
            )));
        }
        let (slots, size) = abi::pack(self.sig);
        let mut out = Vec::with_capacity(self.sig.len());
        for ((s, (off, sz)), v) in self.sig.iter().zip(slots).zip(self.values) {
            let want = match v {
                ArgValue::Buffer { .. } | ArgValue::NullBuffer { .. } => ArgKind::Buffer,
                ArgValue::I32 { .. } => ArgKind::I32,
                ArgValue::I64 { .. } => ArgKind::I64,
                ArgValue::F32 { .. } => ArgKind::F32,
            };
            if want != s.kind {
                return Err(Error::Rule(format!(
                    "paro kernarg builder: `{}` is {:?} in the signature, bound as {want:?}",
                    s.name, s.kind
                )));
            }
            out.push(Slot {
                name: s.name.to_owned(),
                offset: off,
                size: sz,
                value: v,
            });
        }
        Ok((out, size))
    }
}

// ---------------------------------------------------------------------------- resolve

/// The one decision function for the Paro producers. Pure: (op, shape, rules) in,
/// descriptor out.
pub fn resolve(op: Op, shape: Shape, rules: &ParoRules) -> Result<LaunchDescriptor> {
    match (op, shape) {
        (Op::AddRmsRot, Shape::Rms(s)) => add_rms_rot(s, rules),
        (Op::AddRmsRotTok, Shape::Rms(s)) => add_rms_rot_tok(s, rules),
        (Op::EwRot { mode }, Shape::Ew(s)) => ew_rot(s, mode, rules),
        (Op::EwRotTok { mode }, Shape::Ew(s)) => ew_rot_tok(s, mode, rules),
        (o, _) => Err(Error::Rule(format!(
            "{o:?} was handed a shape of the wrong kind"
        ))),
    }
}

/// The two conditions every entry point throws on, in its order.
fn common_checks(width: u32, krot: u32, m: u32, what: &str, rules: &ParoRules) -> Result<()> {
    let k = &rules.knobs;
    if width == 0 || width % k.group != 0 {
        return Err(Error::Rule(format!(
            "Paro: {what} must be a multiple of {}",
            k.group
        )));
    }
    if krot < 1 || krot > k.krot_max {
        return Err(Error::Rule("Paro: krot out of range".into()));
    }
    if m == 0 {
        return Err(Error::Rule(format!("{what}: empty batch")));
    }
    Ok(())
}

/// The per-token launchers stage the whole rotated row in dynamic LDS and refuse above 60 KiB.
fn lds_bytes(width: u32, what: &str, rules: &ParoRules) -> Result<u32> {
    let need = width as u64 * 2;
    if need > rules.knobs.lds_max_bytes {
        return Err(Error::Rule(format!(
            "Paro: {what} needs {} B of LDS; too large",
            need
        )));
    }
    Ok(need as u32)
}

fn add_rms_rot(s: ParoShape, rules: &ParoRules) -> Result<LaunchDescriptor> {
    common_checks(s.k, s.krot, s.m, "K", rules)?;
    if s.p == 0 {
        return Err(Error::Rule("add_rms_rot: P must be at least 1".into()));
    }
    let kn = &rules.knobs;
    let rot = rules.rot(s.m);
    let i8 = kn.i8 && rot;
    let sym = rules
        .add_rms_rot
        .iter()
        .find(|r| r.rot == rot && r.i8 == i8)
        .ok_or_else(|| {
            Error::Rule(format!(
                "add_rms_rot: no rot={rot} i8={i8} instantiation (the launcher's ladder is \
                 `fused && i8` / `fused` / else, so ROT=false never carries I8)"
            ))
        })?;
    // gpw only matters on the ROT arm; the plain arm is one workgroup per row.
    let gpw = rules.rot_gpw(s.m);
    let groups = s.k / kn.group;
    let grid = if rot {
        [s.m, groups.div_ceil(kn.rot_waves * gpw), s.p]
    } else {
        [s.m, 1, 1]
    };
    let (args, bytes) = KArgs::new(abi::SIG_PARO_ADD_RMS_ROT)
        .buf("y")
        .buf("res")
        .buf("w")
        .f32(f32::from_bits(s.eps_bits))
        .buf("t")
        .buf("cs")
        .buf("hs")
        .buf("ro")
        .maybe_buf("a", rot)
        .maybe_buf("asg", rot)
        .maybe_buf("rs", rot)
        .i32(s.m as i64)
        .i32(s.k as i64)
        .i32(s.krot as i64)
        .i32(gpw as i64)
        .finish()?;
    Ok(LaunchDescriptor {
        op: "paro_add_rms_rot".into(),
        plugin: PLUGIN.into(),
        kernel: sym.symbol.clone(),
        grid,
        block: [kn.rot_block, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: no_workspaces(),
        n_launches: 1,
    })
}

fn ew_rot(s: ParoEwShape, mode: u32, rules: &ParoRules) -> Result<LaunchDescriptor> {
    common_checks(s.n, s.krot, s.m, "N", rules)?;
    let kn = &rules.knobs;
    let rot = rules.rot(s.m);
    let i8 = kn.i8 && rot;
    let sym = rules
        .ew_rot
        .iter()
        .find(|r| r.mode == mode && r.rot == rot && r.i8 == i8)
        .ok_or_else(|| {
            Error::Rule(format!(
                "ew_rot: no mode={mode} rot={rot} i8={i8} instantiation"
            ))
        })?;
    // The reference forces gpw = 1 on the plain arm, so the grid is one workgroup per group-block.
    let gpw = if rot { rules.rot_gpw(s.m) } else { 1 };
    let groups = s.n / kn.group;
    let (args, bytes) = KArgs::new(abi::SIG_PARO_EW_ROT)
        .buf("x")
        .buf("y")
        .i64(s.ys)
        .buf("w")
        .f32(f32::from_bits(s.eps_bits))
        .buf("t")
        .buf("cs")
        .buf("hs")
        .maybe_buf("a", rot)
        .maybe_buf("asg", rot)
        .maybe_buf("rs", rot)
        .i32(s.m as i64)
        .i32(s.n as i64)
        .i32(s.krot as i64)
        .i32(gpw as i64)
        .finish()?;
    Ok(LaunchDescriptor {
        op: "paro_ew_rot".into(),
        plugin: PLUGIN.into(),
        kernel: sym.symbol.clone(),
        grid: [s.m, groups.div_ceil(kn.rot_waves * gpw), 1],
        block: [kn.rot_block, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: no_workspaces(),
        n_launches: 1,
    })
}

fn add_rms_rot_tok(s: ParoShape, rules: &ParoRules) -> Result<LaunchDescriptor> {
    common_checks(s.k, s.krot, s.m, "K", rules)?;
    if s.p == 0 {
        return Err(Error::Rule("add_rms_rot_tok: P must be at least 1".into()));
    }
    let smem = lds_bytes(s.k, "add_rms_rot_tok", rules)?;
    let waves = rules.tok_waves(s.m, s.k);
    let tiled = rules.tiled(s.m);
    let sym = rules
        .add_rms_rot_tok
        .iter()
        .find(|r| r.waves == waves && r.tiled == tiled)
        .ok_or_else(|| {
            Error::Rule(format!(
                "add_rms_rot_tok: no waves={waves} tiled={tiled} instantiation"
            ))
        })?;
    let (args, bytes) = KArgs::new(abi::SIG_PARO_ADD_RMS_ROT_TOK)
        .buf("y")
        .buf("res")
        .buf("w")
        .f32(f32::from_bits(s.eps_bits))
        .buf("t")
        .buf("cs")
        .buf("hs")
        .buf("ro")
        .buf("a")
        .buf("as")
        .i32(s.m as i64)
        .i32(s.k as i64)
        .i32(s.krot as i64)
        .finish()?;
    Ok(LaunchDescriptor {
        op: "paro_add_rms_rot_tok".into(),
        plugin: PLUGIN.into(),
        kernel: sym.symbol.clone(),
        grid: [s.m, s.p, 1],
        block: [waves * 32, 1, 1],
        shared_bytes: smem,
        kernarg_bytes: bytes,
        args,
        workspaces: no_workspaces(),
        n_launches: 1,
    })
}

fn ew_rot_tok(s: ParoEwShape, mode: u32, rules: &ParoRules) -> Result<LaunchDescriptor> {
    common_checks(s.n, s.krot, s.m, "N", rules)?;
    let smem = lds_bytes(s.n, "ew_rot_tok", rules)?;
    let waves = rules.tok_waves(s.m, s.n);
    let tiled = rules.tiled(s.m);
    let sym = rules
        .ew_rot_tok
        .iter()
        .find(|r| r.mode == mode && r.waves == waves && r.tiled == tiled)
        .ok_or_else(|| {
            Error::Rule(format!(
                "ew_rot_tok: no mode={mode} waves={waves} tiled={tiled} instantiation"
            ))
        })?;
    let (args, bytes) = KArgs::new(abi::SIG_PARO_EW_ROT_TOK)
        .buf("x")
        .buf("y")
        .i64(s.ys)
        .buf("w")
        .f32(f32::from_bits(s.eps_bits))
        .buf("t")
        .buf("cs")
        .buf("hs")
        .buf("a")
        .buf("as")
        .i32(s.m as i64)
        .i32(s.n as i64)
        .i32(s.krot as i64)
        .finish()?;
    Ok(LaunchDescriptor {
        op: "paro_ew_rot_tok".into(),
        plugin: PLUGIN.into(),
        kernel: sym.symbol.clone(),
        // Single partition: all three consumers are P = 1.
        grid: [s.m, 1, 1],
        block: [waves * 32, 1, 1],
        shared_bytes: smem,
        kernarg_bytes: bytes,
        args,
        workspaces: no_workspaces(),
        n_launches: 1,
    })
}

/// Every buffer these producers touch is the layer's own tensor. Spelled out rather than left
/// implicit: "this op needs no workspace" is a claim the descriptor makes and the gate checks.
fn no_workspaces() -> Vec<WorkspaceReq> {
    Vec::new()
}
