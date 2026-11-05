//! The attention layer's glue as data — the fourth plugin.
//!
//! Three ops the P1 decoder block either computed in torch or did not model at all
//! (LEDGER, "P1 decoder block (workstream P)", *What this block does NOT do*): `qk_norm_rope`,
//! `attn_gate_quant` and the KV append. They are the last things between a bit-exact block and a
//! block the engine can run on its own, so they block P2.
//!
//! **Unlike the other three plugins, nothing here is a port of a HIP source.** Prod runs a Triton
//! kernel or an ATen composition for every one of them, so the device code is written from
//! The reference server's reference semantics and the gate compares it against what prod actually executes. The
//! consequence for these rules is that a constant is a claim about *prod's own configuration*,
//! not about a launcher's C++ ladder — so each one names where it comes from.

use serde::Deserialize;

use crate::abi::{self, ArgKind, SigArg};
use crate::rules::{ArgValue, LaunchDescriptor, Slot, WorkspaceReq};
use crate::{Error, Result};

const PLUGIN: &str = "attn_glue";
const PROD_RULES: &str = include_str!("../rules/attn_glue_gfx1201.toml");

/// The committed table, verbatim, for the plan's policy digest.
pub const PROD_RULES_TEXT: &str = PROD_RULES;

// ---------------------------------------------------------------------------- rules (data)

#[derive(Debug, Clone, Deserialize)]
pub struct GlueRules {
    pub schema_version: u32,
    pub name: String,
    pub model: Model,
    pub qk_norm_rope: QkRow,
    pub attn_gate_quant: GateRow,
    pub kv_append: KvRow,
    /// ws-h38: the fused `qk_norm_rope` + `kv_append` symbols, one row per rope decomposition
    /// ([`rope_append`]). Absent = the fusion is never offered.
    #[serde(default)]
    pub rope_append: Vec<RopeAppendRow>,
}

/// One decomposition's fused rope + append (ws-h38). `kind` is the [`QkBand`] kind it pairs with:
/// the fusion runs the rope exactly as the band would and appends inside the same workgroup, so
/// it never moves the rope onto the other decomposition.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeAppendRow {
    pub kind: QkKind,
    pub interleaved: String,
    pub chunked: String,
    /// The interleaved symbol with the V half never written: the fusion gate's negative control.
    pub control: String,
}

/// The model geometry the symbols are compiled for. `head_dim` and `rotary_dim` are template
/// parameters, not arguments: a shape off them is refused rather than served slowly.
#[derive(Debug, Clone, Deserialize)]
pub struct Model {
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub mrope_section: [u32; 3],
    pub mrope_interleaved: bool,
    pub block_size: u32,
    pub eps: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QkRow {
    pub threads: u32,
    pub n_launches: u32,
    pub reduction: u32,
    /// Which decomposition runs, by the number of (token, head) pairs. First match wins; the last
    /// row carries no `pairs_max` and is the default.
    pub band: Vec<QkBand>,
    pub controls: Vec<String>,
}

/// The two decompositions of `qk_norm_rope`. They compute the same bytes -- both run ATen's
/// reduction in the same 32 lanes in the same order -- and differ only in how much of the machine
/// one launch can keep busy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QkKind {
    /// One workgroup per (token, head). 32 of its `threads` lanes reduce, through LDS.
    Head,
    /// One WAVE per (token, head), `threads / 32` of them per workgroup. No LDS, no barrier.
    Wave,
}

/// How positions are applied to a head.
///
/// A selection key, not decoration: the two are different arithmetic over a different number of
/// position rows, and they are different compiled symbols.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RopeKind {
    /// Three position rows partitioned by `mrope_section` over `rotary_dim` of `head_dim` dims.
    /// The target's, and the only one whose `interleaved`/`chunked` split means anything.
    Mrope,
    /// One position row over the whole head. The drafter's.
    Plain,
}

/// WHICH rope an op is asking for: the kind, and whether the projection it reads carries a
/// per-head gate beside each q head.
///
/// These two plus `head_dim` are the dimensions a band row selects on, and they exist as a named
/// pair because the bug they close is a row being picked for a shape it cannot serve: before
/// this, `qk_band` looked only at the (token, head) pair count, so a row added for the drafter
/// (128-wide heads, plain rope, no gate) became a third rung the TARGET could land on at a large
/// token count -- a wrong kernel for a real shape, with every address still in range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QkFlavour {
    /// How positions are applied.
    pub rope: RopeKind,
    /// The projection holds `[q|gate]` per q head, so a q head is `2 * head_dim` wide in it.
    pub gated: bool,
}

impl QkFlavour {
    /// Qwen3.8-27B's own: 3-row mrope over 64 of 256 dims, reading a `[q|gate]` projection.
    pub const TARGET: Self = Self {
        rope: RopeKind::Mrope,
        gated: true,
    };
    /// The DFlash2 drafter's: one position row over the whole head, no gate.
    pub const DRAFTER: Self = Self {
        rope: RopeKind::Plain,
        gated: false,
    };
}

#[derive(Debug, Clone, Deserialize)]
pub struct QkBand {
    /// Head width this row's symbols are compiled for. A row is only a candidate for a shape of
    /// exactly this width.
    pub head_dim: u32,
    /// Rope kind this row's symbols implement.
    pub rope: RopeKind,
    /// Whether this row's symbols read a `[q|gate]` projection.
    pub gated: bool,
    pub pairs_max: Option<u64>,
    pub kind: QkKind,
    pub interleaved: String,
    pub chunked: String,
}

impl QkBand {
    /// The `(rope, gated)` pair this row serves.
    #[must_use]
    pub fn flavour(&self) -> QkFlavour {
        QkFlavour {
            rope: self.rope,
            gated: self.gated,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct GateRow {
    pub threads: u32,
    pub n_launches: u32,
    pub tiled0: String,
    pub tiled1: String,
    /// **Not a row of this table.** `a_tiled_min_m` has one source,
    /// [`crate::rules::a_tiled_min_m`], and this field exists only so that a table which states
    /// it again is REFUSED by name rather than shadowing it. See [`GateRow::a_tiled_min_m`].
    #[serde(default, rename = "a_tiled_min_m")]
    a_tiled_min_m_restated: Option<u32>,
}

impl GateRow {
    /// The M at and above which this producer writes the fragment-tiled layout.
    ///
    /// Read from [`crate::rules::a_tiled_min_m`], never from this table: the producer writes the
    /// A side the A-tiled GEMM reads, so one threshold or the consumer reads a layout the
    /// producer did not write. It was a second copy here, pinned to the first by a host test,
    /// until P6 showed what a copy costs (see [`crate::rules::a_tiled_min_m`]).
    #[must_use]
    pub fn a_tiled_min_m(&self) -> u32 {
        crate::rules::a_tiled_min_m()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct KvRow {
    pub threads: u32,
    pub n_launches: u32,
    pub kernel: String,
    pub fp8: bool,
}

impl GlueRules {
    /// The rules the `qwen_prod_nvfp4` unit's own configuration implies.
    pub fn prod() -> Result<Self> {
        Self::from_toml(PROD_RULES)
    }

    pub fn from_toml(text: &str) -> Result<Self> {
        let r: GlueRules = toml::from_str(text).map_err(|e| Error::Parse(e.to_string()))?;
        r.check()?;
        Ok(r)
    }

    fn check(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(Error::Rule(format!(
                "attn_glue rules schema {} (this build speaks 1)",
                self.schema_version
            )));
        }
        let m = &self.model;
        // `MRotaryEmbedding.__init__` asserts exactly this, and the interleaved subclass raises
        // on it: the three sections partition the rotary HALF, because each one covers a
        // (cos, sin) pair. A table that fails it would index past the cache.
        let half = m.rotary_dim / 2;
        if m.mrope_section.iter().sum::<u32>() != half {
            return Err(Error::Rule(format!(
                "mrope_section {:?} must sum to rotary_dim/2 = {half}",
                m.mrope_section
            )));
        }
        if m.rotary_dim == 0 || m.rotary_dim % 2 != 0 || m.rotary_dim > m.head_dim {
            return Err(Error::Rule(format!(
                "rotary_dim must be a positive even integer <= head_dim, got rotary_dim={}, \
                 head_dim={}",
                m.rotary_dim, m.head_dim
            )));
        }
        if m.head_dim % 8 != 0 {
            return Err(Error::Rule("head_dim must be a multiple of 8".into()));
        }
        if !m.block_size.is_power_of_two() {
            return Err(Error::Rule("block_size must be a power of two".into()));
        }
        // ONE source for the band edge. A table that states it again would be a second copy of
        // the number the A-tiled GEMM reads, and a copy is what P6 cost an experiment: refuse it
        // by name instead of letting it shadow `rules::a_tiled_min_m`.
        if let Some(v) = self.attn_gate_quant.a_tiled_min_m_restated {
            return Err(Error::Rule(format!(
                "[attn_gate_quant] a_tiled_min_m = {v}: this table does not own that number. \
                 `rules/mxfp4_gfx1201.toml [knobs] a_tiled_min_m` is the one source and \
                 `saber_kernels::rules::a_tiled_min_m()` is the one reader — the producer writes \
                 the A side the A-tiled GEMM reads, so a second copy is a layout mismatch waiting \
                 for an editor who moves one of them"
            )));
        }
        if self.qk_norm_rope.reduction > 5 {
            return Err(Error::Rule(format!(
                "reduction {} names no compiled variance order (0..5)",
                self.qk_norm_rope.reduction
            )));
        }
        // The named order has to be the one every band's symbols actually carry, or the rules
        // would document one reduction and launch another.
        let want = format!("_red{}", self.qk_norm_rope.reduction);
        if self.qk_norm_rope.band.is_empty() {
            return Err(Error::Rule(
                "qk_norm_rope needs at least one band row".into(),
            ));
        }
        for b in &self.qk_norm_rope.band {
            for s in [&b.interleaved, &b.chunked] {
                if !s.ends_with(&want) {
                    return Err(Error::Rule(format!(
                        "`{s}` does not carry the reduction order the rules name ({want})"
                    )));
                }
            }
        }
        // The ladder is a ladder PER (head_dim, rope, gated) -- the three things a row's symbols
        // are compiled for -- and the invariants are per group, not over the file:
        //
        //   * a group whose last row is bounded would refuse a large batch outright, which is a
        //     rules bug rather than a launch decision;
        //   * a group with an unbounded row anywhere but last shadows every row after it, since
        //     the selection is first-match-wins. Silent dead rows are how a measured band turns
        //     into a lie.
        //
        // A plain-rope row has no mrope decomposition to choose between, so its two symbol fields
        // must name the same kernel: the flag is meaningless there and a row that offered two
        // different symbols for it would be claiming a choice the op does not have.
        let mut groups: Vec<(u32, QkFlavour)> = Vec::new();
        for b in &self.qk_norm_rope.band {
            let g = (b.head_dim, b.flavour());
            if !groups.contains(&g) {
                groups.push(g);
            }
            if b.rope == RopeKind::Plain && b.interleaved != b.chunked {
                return Err(Error::Rule(format!(
                    "qk_norm_rope band row (head_dim {}, plain) names two symbols, `{}` and `{}`: plain rope has one position row and no interleaved/chunked split",
                    b.head_dim, b.interleaved, b.chunked
                )));
            }
        }
        for g in &groups {
            let rows: Vec<&QkBand> = self
                .qk_norm_rope
                .band
                .iter()
                .filter(|b| (b.head_dim, b.flavour()) == *g)
                .collect();
            let (hd, fl) = g;
            let last = rows.last().expect("a group exists because a row is in it");
            if last.pairs_max.is_some() {
                return Err(Error::Rule(format!(
                    "the last qk_norm_rope band row for head_dim {hd} {:?} gated={} must carry no pairs_max: it is that shape's default",
                    fl.rope, fl.gated
                )));
            }
            if let Some(i) = rows[..rows.len() - 1]
                .iter()
                .position(|b| b.pairs_max.is_none())
            {
                return Err(Error::Rule(format!(
                    "qk_norm_rope band row {i} for head_dim {hd} {:?} gated={} carries no pairs_max but is not the last of its group: it shadows every row after it",
                    fl.rope, fl.gated
                )));
            }
        }
        if self.qk_norm_rope.threads % 32 != 0 {
            return Err(Error::Rule(
                "qk_norm_rope threads must be a whole number of 32-lane waves".into(),
            ));
        }
        // The wave kernel's waves-per-workgroup is a COMPILE-TIME constant (it must be: reading
        // blockDim pulls in the implicit kernarg block), so the rules' `threads` and the symbol's
        // own `_wgN_` have to agree or the grid would be sized for a workgroup shape the code
        // object does not have.
        let wg = format!("_wg{}_", self.qk_norm_rope.threads / 32);
        for b in &self.qk_norm_rope.band {
            if b.kind == QkKind::Wave && !b.interleaved.contains(&wg) {
                return Err(Error::Rule(format!(
                    "`{}` is a wave-band symbol but does not carry {wg}, which is what threads/32 = {} makes it",
                    b.interleaved,
                    self.qk_norm_rope.threads / 32
                )));
            }
        }
        // The fused rope + append: one row per decomposition, each carrying the rules' variance
        // order, the wave row the `_wgN_` the knob makes it -- the same two promises the band
        // rows keep, because the fused symbol runs the band's own rope.
        for (i, r) in self.rope_append.iter().enumerate() {
            if self.rope_append[..i].iter().any(|p| p.kind == r.kind) {
                return Err(Error::Rule(format!(
                    "two rope_append rows for the {:?} decomposition",
                    r.kind
                )));
            }
            for s in [&r.interleaved, &r.chunked] {
                if !s.contains(&want) {
                    return Err(Error::Rule(format!(
                        "`{s}` does not carry the reduction order the rules name ({want})"
                    )));
                }
                if r.kind == QkKind::Wave && !s.contains(&wg) {
                    return Err(Error::Rule(format!(
                        "`{s}` is a wave rope_append symbol but does not carry {wg}"
                    )));
                }
            }
        }
        if self.kv_append.threads * 2 != m.head_dim {
            return Err(Error::Rule(format!(
                "kv_append runs head_dim/2 = {} threads, not {}",
                m.head_dim / 2,
                self.kv_append.threads
            )));
        }
        Ok(())
    }

    /// Every symbol these rules can launch, including the two variance-order controls.
    ///
    /// The controls are listed deliberately: they are in the code object, the metadata gate
    /// checks them, and a map that left them out would make `plugin_for` refuse a symbol the
    /// plugin demonstrably holds.
    pub fn symbols(&self) -> Vec<&str> {
        let mut out = vec![
            self.attn_gate_quant.tiled0.as_str(),
            self.attn_gate_quant.tiled1.as_str(),
            self.kv_append.kernel.as_str(),
        ];
        for b in &self.qk_norm_rope.band {
            out.push(b.interleaved.as_str());
            out.push(b.chunked.as_str());
        }
        out.extend(self.qk_norm_rope.controls.iter().map(|s| s.as_str()));
        for r in &self.rope_append {
            out.push(r.interleaved.as_str());
            out.push(r.chunked.as_str());
            out.push(r.control.as_str());
        }
        out
    }

    /// The band row a shape falls in: among the rows compiled for THIS `head_dim` and flavour,
    /// the first whose `pairs_max` covers the pair count, else that group's default.
    ///
    /// A `Result`, not an `expect`: a shape no group serves is a refusal that names what the
    /// rules hold, and the alternative -- falling through to whatever row happens to be last --
    /// is the bug this signature exists to make impossible.
    pub fn qk_band(&self, head_dim: u32, flavour: QkFlavour, pairs: u64) -> Result<&QkBand> {
        self.qk_norm_rope
            .band
            .iter()
            .find(|b| {
                b.head_dim == head_dim
                    && b.flavour() == flavour
                    && b.pairs_max.is_none_or(|x| pairs <= x)
            })
            .ok_or_else(|| {
                Error::Rule(format!(
                    "qk_norm_rope: no band row is compiled for head_dim {head_dim} {:?} gated={}; the rules hold {}",
                    flavour.rope,
                    flavour.gated,
                    self.qk_norm_rope
                        .band
                        .iter()
                        .map(|b| format!("{} {:?} gated={}", b.head_dim, b.rope, b.gated))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })
    }

    /// Which `.hsaco` holds `symbol`. Membership, not spelling — one rules file, one code object.
    pub fn plugin_for(&self, symbol: &str) -> Result<&'static str> {
        if self.symbols().contains(&symbol) {
            Ok(PLUGIN)
        } else {
            Err(Error::Rule(format!(
                "no attn_glue plugin holds symbol `{symbol}`; these rules name {} symbols and \
                 that is not one of them",
                self.symbols().len()
            )))
        }
    }

    pub fn owner(&self) -> Result<&'static str> {
        let first = self
            .symbols()
            .first()
            .copied()
            .ok_or_else(|| Error::Rule("attn_glue rules name no symbols at all".into()))?;
        self.plugin_for(first)
    }

    /// The A-side layout decision under the COMMITTED law, shared with the MXFP4 producers and
    /// the A-tiled GEMM. A plan built on edited mxfp4 rules asks [`GlueRules::tiled_under`].
    pub fn tiled(&self, m: u32) -> bool {
        self.tiled_under(m, crate::rules::ATiledLaw::committed())
    }

    /// The A-side layout decision under the layout law of the mxfp4 rules a plan was built with
    /// (G6): this producer switches wherever the others do, mid band included.
    pub fn tiled_under(&self, m: u32, law: crate::rules::ATiledLaw) -> bool {
        law.tiled(m)
    }
}

// ---------------------------------------------------------------------------- shapes and ops

/// `qk_norm_rope`: `m` tokens of a `[m, qkv_n]` fused projection carrying `nq` `[q|gate]` head
/// pairs, then `nkv` k heads, then `nkv` v heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QkShape {
    pub m: u32,
    pub nq: u32,
    pub nkv: u32,
    pub head_dim: u32,
    pub qkv_n: u32,
}

/// `attn_gate_quant`: `[m, n]` attention output with `n = nq * head_dim`; the gate is read in
/// place out of the projection at `gs` row stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateShape {
    pub m: u32,
    pub n: u32,
    pub head_dim: u32,
    pub gs: i64,
}

/// `kv_append`: `tokens` rows of `nkv` heads into a paged cache of `block_size` slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvShape {
    pub tokens: u32,
    pub nkv: u32,
    pub head_dim: u32,
    pub block_size: u32,
    /// Element strides of the source `[T, nkv, head_dim]` tensors and of the paged cache view.
    pub key_stride: i64,
    pub value_stride: i64,
    pub block_stride: i64,
    pub page_stride: i64,
    pub head_stride: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Qk(QkShape),
    Gate(GateShape),
    Kv(KvShape),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Per-head q/k RMS norm + partial mrope, one launch. `interleaved` is the CONFIG's
    /// `mrope_interleaved`, not a shape: for a text-only request the three position rows are
    /// equal and the choice cannot be observed in the output, which is exactly why it must come
    /// from the table rather than from the data.
    QkNormRope {
        interleaved: bool,
        /// Which rope, and whether the projection is gated. `QkFlavour::TARGET` is the model's.
        flavour: QkFlavour,
    },
    /// `core * sigmoid(gate)` + per-token e4m3, feeding o_proj's A side.
    AttnGateQuant,
    /// The paged fp8 KV write.
    KvAppend,
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
                "attn_glue kernarg builder: {} values for a {}-argument signature",
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
                    "attn_glue kernarg builder: `{}` is {:?} in the signature, bound as {want:?}",
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

/// Every buffer these three ops touch is the layer's own tensor — the paged KV cache included,
/// which is the model's memory and not a scratch arena, the same rule the GDN state follows.
fn no_workspaces() -> Vec<WorkspaceReq> {
    Vec::new()
}

// ---------------------------------------------------------------------------- resolve

/// The one decision function for the attention glue. Pure: (op, shape, rules) in, descriptor out.
/// The gate producer's layout follows the COMMITTED law; [`resolve_under`] takes a plan's own.
pub fn resolve(op: Op, shape: Shape, rules: &GlueRules) -> Result<LaunchDescriptor> {
    resolve_under(op, shape, rules, crate::rules::ATiledLaw::committed())
}

/// [`resolve`] under the layout law of the mxfp4 rules the plan was built with (G6): the gate
/// producer writes the layout the o_proj GEMM of the same plan reads.
pub fn resolve_under(
    op: Op,
    shape: Shape,
    rules: &GlueRules,
    law: crate::rules::ATiledLaw,
) -> Result<LaunchDescriptor> {
    match (op, shape) {
        (
            Op::QkNormRope {
                interleaved,
                flavour,
            },
            Shape::Qk(s),
        ) => qk_norm_rope(s, interleaved, flavour, rules),
        (Op::AttnGateQuant, Shape::Gate(s)) => attn_gate_quant(s, rules, law),
        (Op::KvAppend, Shape::Kv(s)) => kv_append(s, rules),
        (o, _) => Err(Error::Rule(format!(
            "{o:?} was handed a shape of the wrong kind"
        ))),
    }
}

fn qk_norm_rope(
    s: QkShape,
    interleaved: bool,
    flavour: QkFlavour,
    rules: &GlueRules,
) -> Result<LaunchDescriptor> {
    let m = &rules.model;
    if s.m == 0 || s.nq == 0 || s.nkv == 0 {
        return Err(Error::Rule(
            "qk_norm_rope: empty batch or head count".into(),
        ));
    }
    // The projection has to be wide enough to hold [q|gate] per q head (or just q, where the
    // flavour is ungated), then k, then v. Narrower and the k heads would be read out of the v
    // block, which is silent nonsense rather than a fault: every address is still in the tensor.
    let q_wide = if flavour.gated { 2 } else { 1 };
    let need = (q_wide * s.nq + 2 * s.nkv) * s.head_dim;
    if s.qkv_n < need {
        return Err(Error::Rule(format!(
            "qk_norm_rope: qkv_n {} is short of the {need} this head layout needs \
             ({q_wide}*nq + 2*nkv heads of {})",
            s.qkv_n, s.head_dim
        )));
    }
    let q = &rules.qk_norm_rope;
    // The decision is the number of (token, head) pairs, not M: a one-token batch of 28 heads and
    // a 28-token batch of one head give a launch the same amount to do, and the per-head
    // decomposition exists precisely for the case where there is not enough of it.
    let pairs = s.m as u64 * (s.nq + s.nkv) as u64;
    // The head width is checked HERE, by the band lookup, rather than against `[model]`: the
    // model's `head_dim` is the target's, and a rules file may hold rows for more than one.
    let band = rules.qk_band(s.head_dim, flavour, pairs)?;
    let sym = if interleaved {
        &band.interleaved
    } else {
        &band.chunked
    };
    let waves = q.threads / 32;
    // The per-head decomposition runs ONE THREAD PER ELEMENT OF THE HEAD, so its block is the
    // BAND's `head_dim` and not the `[qk_norm_rope] threads` knob. For the target the two are the
    // same number (256) and this changes nothing; for the drafter's 128-wide head they are not,
    // and the knob would launch a 128-thread code object with 256 threads -- past its own
    // `max_flat_workgroup_size`, which is a launch failure rather than a slow path. The knob keeps
    // sizing the WAVE band, where it means workgroups-worth-of-waves and is what `_wgN_` checks.
    let (grid, block) = match band.kind {
        QkKind::Head => ([s.m, s.nq + s.nkv, 1], band.head_dim),
        QkKind::Wave => ([pairs.div_ceil(waves as u64) as u32, 1, 1], q.threads),
    };
    let (args, bytes) = KArgs::new(abi::SIG_QK_NORM_ROPE)
        .buf("qkv")
        .buf("qw")
        .buf("kw")
        .buf("cos_sin")
        .buf("pos")
        .buf("q_out")
        .buf("k_out")
        .i32(s.m as i64)
        .i32(s.nq as i64)
        .i32(s.nkv as i64)
        .i32(s.qkv_n as i64)
        .f32(m.eps)
        .i32(m.mrope_section[0] as i64)
        .i32(m.mrope_section[1] as i64)
        .i32(m.mrope_section[2] as i64)
        .finish()?;
    Ok(LaunchDescriptor {
        op: "qk_norm_rope".into(),
        plugin: PLUGIN.into(),
        kernel: sym.clone(),
        grid,
        block: [block, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: no_workspaces(),
        n_launches: q.n_launches,
    })
}

fn attn_gate_quant(
    s: GateShape,
    rules: &GlueRules,
    law: crate::rules::ATiledLaw,
) -> Result<LaunchDescriptor> {
    let g = &rules.attn_gate_quant;
    if s.m == 0 {
        return Err(Error::Rule("attn_gate_quant: empty batch".into()));
    }
    // The kernel moves 8 elements per lane and reads the gate as one uint4 at a head-local
    // offset, so both the row and the head must be a whole number of 8-groups.
    if s.n == 0 || s.n % 8 != 0 || s.head_dim % 8 != 0 {
        return Err(Error::Rule(format!(
            "attn_gate_quant: n {} and head_dim {} must both be multiples of 8",
            s.n, s.head_dim
        )));
    }
    if s.n % s.head_dim != 0 {
        return Err(Error::Rule(format!(
            "attn_gate_quant: n {} is not a whole number of {}-wide heads",
            s.n, s.head_dim
        )));
    }
    // The gate lives in the projection, one head stride of 2*head_dim apart. A row stride that
    // cannot hold the q half means the caller passed a contiguous copy, which this instantiation
    // does not read -- and would read past the row if it tried.
    if s.gs < 2 * (s.n as i64) {
        return Err(Error::Rule(format!(
            "attn_gate_quant: gate row stride {} cannot hold {} interleaved [q|gate] heads",
            s.gs,
            s.n / s.head_dim
        )));
    }
    let tiled = rules.tiled_under(s.m, law);
    let (args, bytes) = KArgs::new(abi::SIG_ATTN_GATE_QUANT)
        .buf("core")
        .buf("gate")
        .buf("q")
        .buf("scale")
        .i32(s.n as i64)
        .i32(s.head_dim as i64)
        .i32(s.gs)
        .finish()?;
    Ok(LaunchDescriptor {
        op: "attn_gate_quant".into(),
        plugin: PLUGIN.into(),
        kernel: if tiled {
            g.tiled1.clone()
        } else {
            g.tiled0.clone()
        },
        grid: [s.m, 1, 1],
        block: [g.threads, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: no_workspaces(),
        n_launches: g.n_launches,
    })
}

fn kv_append(s: KvShape, rules: &GlueRules) -> Result<LaunchDescriptor> {
    let m = &rules.model;
    let k = &rules.kv_append;
    if !k.fp8 {
        return Err(Error::Rule(
            "kv_append: only the fp8 paged cache is compiled (--kv-cache-dtype fp8)".into(),
        ));
    }
    if s.head_dim != m.head_dim || s.block_size != m.block_size {
        return Err(Error::Rule(format!(
            "kv_append: head_dim {} block_size {} are compiled in; this shape is {} / {}",
            m.head_dim, m.block_size, s.head_dim, s.block_size
        )));
    }
    if s.tokens == 0 || s.nkv == 0 {
        return Err(Error::Rule("kv_append: nothing to write".into()));
    }
    // K and V are the two halves of one page, so V's base is K's plus head_dim. A page stride
    // narrower than that would overlap them.
    if s.page_stride < 2 * (s.head_dim as i64) {
        return Err(Error::Rule(format!(
            "kv_append: page stride {} cannot hold K and V of {} elements each",
            s.page_stride, s.head_dim
        )));
    }
    let (args, bytes) = KArgs::new(abi::SIG_KV_APPEND)
        .buf("key")
        .buf("value")
        .buf("kv")
        .buf("slot")
        .buf("k_scale")
        .buf("v_scale")
        .i64(s.key_stride)
        .i64(s.value_stride)
        .i64(s.block_stride)
        .i64(s.page_stride)
        .i64(s.head_stride)
        .finish()?;
    Ok(LaunchDescriptor {
        op: "kv_append".into(),
        plugin: PLUGIN.into(),
        kernel: k.kernel.clone(),
        grid: [s.tokens, s.nkv, 1],
        block: [k.threads, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: no_workspaces(),
        n_launches: k.n_launches,
    })
}

/// **`qk_norm_rope` + `kv_append` in ONE dispatch** (ws-h38, fusion-map row 7).
///
/// The fused symbol is the rope's own impl followed, in the same workgroup, by the append of the
/// K rows that workgroup just wrote (and the matching V rows, read in place out of the
/// projection): the append's work item `(token t, kv head h)` reads one row of `k_out`, which the
/// rope writes in exactly one of its own work items, `(t, nq + h)`, in either decomposition. So
/// the two grids are row-compatible, the append needs one workgroup barrier and no grid barrier,
/// and every byte is the two ops' own arithmetic -- gated bit-exact by `step-tp2 --fusion
/// rope-append`.
///
/// Each half is resolved exactly as it would be alone ([`resolve`] on the rope's shape and on the
/// append's), and the fused descriptor takes the ROPE's grid and block, so the decomposition the
/// band picks is the one that runs. Refused -- and the planner keeps the two ops -- unless the
/// rules carry a `rope_append` row for that decomposition, the flavour is the target's (gated
/// mrope; the drafter's rope has no fused symbol), both halves see the same tokens, kv heads and
/// head width, and the append reads the COMPACT `k_out` the rope writes
/// (`key_stride = nkv * head_dim`).
pub fn rope_append(
    qk: QkShape,
    interleaved: bool,
    flavour: QkFlavour,
    kv: KvShape,
    rules: &GlueRules,
) -> Result<LaunchDescriptor> {
    if flavour != QkFlavour::TARGET {
        return Err(Error::Rule(format!(
            "rope_append: only the target's rope ({:?}) has a fused symbol, not {flavour:?}",
            QkFlavour::TARGET
        )));
    }
    if qk.m != kv.tokens || qk.nkv != kv.nkv || qk.head_dim != kv.head_dim {
        return Err(Error::Rule(format!(
            "rope_append: the rope covers {} token(s) x {} kv head(s) of {}, the append {} x {} \
             of {}; the fused workgroup appends only the rows it roped",
            qk.m, qk.nkv, qk.head_dim, kv.tokens, kv.nkv, kv.head_dim
        )));
    }
    let compact = i64::from(kv.nkv) * i64::from(kv.head_dim);
    if kv.key_stride != compact {
        return Err(Error::Rule(format!(
            "rope_append: the append reads k at row stride {}, the rope writes it compact \
             ({compact})",
            kv.key_stride
        )));
    }
    let rope = qk_norm_rope(qk, interleaved, flavour, rules)?;
    let append = kv_append(kv, rules)?;
    let band = rules.qk_band(
        qk.head_dim,
        flavour,
        u64::from(qk.m) * u64::from(qk.nq + qk.nkv),
    )?;
    let row = rules
        .rope_append
        .iter()
        .find(|r| r.kind == band.kind)
        .ok_or_else(|| {
            Error::Rule(format!(
                "rope_append: no fused symbol for the {:?} decomposition",
                band.kind
            ))
        })?;
    let sym = if interleaved {
        &row.interleaved
    } else {
        &row.chunked
    };
    let m = &rules.model;
    let (args, bytes) = KArgs::new(abi::SIG_QK_ROPE_APPEND)
        .buf("qkv")
        .buf("qw")
        .buf("kw")
        .buf("cos_sin")
        .buf("pos")
        .buf("q_out")
        .buf("k_out")
        .i32(qk.m as i64)
        .i32(qk.nq as i64)
        .i32(qk.nkv as i64)
        .i32(qk.qkv_n as i64)
        .f32(m.eps)
        .i32(m.mrope_section[0] as i64)
        .i32(m.mrope_section[1] as i64)
        .i32(m.mrope_section[2] as i64)
        .buf("value")
        .buf("kv")
        .buf("slot")
        .buf("k_scale")
        .buf("v_scale")
        .i64(kv.key_stride)
        .i64(kv.value_stride)
        .i64(kv.block_stride)
        .i64(kv.page_stride)
        .i64(kv.head_stride)
        .finish()?;
    Ok(LaunchDescriptor {
        op: "qk_rope_append".into(),
        plugin: PLUGIN.into(),
        kernel: sym.clone(),
        grid: rope.grid,
        block: rope.block,
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: no_workspaces(),
        n_launches: rope.n_launches.max(append.n_launches),
    })
}
