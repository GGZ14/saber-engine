//! The Flash-Next tier pack plan: EP=2 over two ranks, dense TP=2 (`docs/FLASHNEXT-PLAN.md` §3, §6).
//!
//! The pack has THREE artefacts, laid out for the memory plane's three tiers from the start
//! (the user's design of 2026-09-28: frequently used experts in VRAM, the rest in host RAM, the PLE
//! n-gram table on the SSD):
//!
//! 1. **Dense records** — `rank0/`, `rank1/` (TP-split) and `shared/` (replicated, one copy on
//!    disk, loaded by both ranks). Always resident. Source kinds: `bf16` passthrough (a TP slice of
//!    one checkpoint tensor), `bf16_merged` (a dim-0 concatenation, e.g. the GDN `in_proj` merges
//!    as on the 27B), `f32_from_bf16` (`A_log`, `dt_bias`), `i64` (the PLE hash buffers) and
//!    `quark_mxfp4` (the shared expert: a byte copy of Quark's nibbles with `permute_w`, the scale
//!    transpose and `wref` — no requantization).
//! 2. **The expert tier** — `rank<r>/experts.xtier`: rank `r` OWNS routed experts
//!    `[256r, 256r + 256)` of every layer (EP=2; §6). One fixed-size record per (layer, expert):
//!    gate_up (`w`, `ws`, `wref`) then down (`w`, `ws`, `wref`), each sub-record 256-aligned, the
//!    record padded to 4 KiB, so the expert LRU pulls an expert with ONE aligned NVMe / RAM read.
//!    Source kind `quark_mxfp4`, the same byte copy.
//! 3. **The PLE table** — `ple/`: an index over the checkpoint's own 128 table shards (row-major
//!    160-wide bf16 rows), read in place (the disk decision: a copy does not fit, see
//!    [`PleTier`]); the optional fp8 variant indexes Qwen's first-party
//!    `model-fp8-mtp-ple.safetensors` as a SEPARATE source kind, gated later.
//!
//! **FN144, the FP4-Local source** ([`PackPlan::fp4local`], `nvidia/Qwen3.8-Flash-Next-FP4-Local`,
//! the LOSSLESS path): the same record set and geometry, with no requantization and no extra
//! rounding anywhere:
//!
//! * routed experts: NVFP4 into an xtier **v2** `NVFP4_E4M3G16` tier -- `w` (the e2m1 bytes in the
//!   same `permute_w` fragment order: the packing is identical), `ws` the CSF-DECODED E4M3 g16
//!   scales transposed to `[K/16, N]` like today's `ws`, `gs` the f32 globals (`weight_scale_2`)
//!   in place of `wref`: 676 pages a record against 639;
//! * MXFP8 trunk + shared experts: the checkpoint's e4m3 `[N, K]` + UE8M0 `[N, K/32]` bytes
//!   carried AS STORED (identity records `w` / `ws`, TP-sliced like the bf16 records they replace;
//!   no bs128 re-rounding) -- `fp4local_mxfp8`;
//! * GDN `in_proj_a` / `in_proj_b` and the QSA indexer's `index_qk_proj`: MXFP8 dequantized to bf16
//!   at pack time -- exact (e4m3 x 2^e has 4 significant bits and fits bf16's range at these
//!   scales; the builder checks every element round-trips) -- `fp4local_mxfp8_bf16`;
//! * MTP routed experts: W4A16 NVFP4 with PLAIN E4M3 scales -> the same `w` / `ws` / `gs` trio,
//!   resident -- `fp4local_nvfp4`; the MTP attention / fc / shared expert are bf16 (the `ignore`
//!   globs) and stay `bf16`;
//! * the target HC norms and the PLE norms are f32 in this checkpoint: carried as f32
//!   (`fp4local_f32`, identity);
//! * the PLE table: NVFP4 rows (`[R, 80]` + E4M3 `[R, 10]` per shard, one f32 `weight_scale_2`
//!   for the table) -- indexed, P5 re-lays it;
//! * no loader alias (FN9's QSA merged projection is a bf16 identity merge; the fp8 trunk's merge
//!   is FN133 P4's).
//!
//! This module is the record SET and the byte ledger; the bytes are
//! `tools/pack/build_flashnext_pack.py`'s and the checks `tools/pack/check_flashnext_pack.py`'s.

use serde_json::{json, Value};

use super::config::{LayerKind, ModeloptAlgo, Qwen4ExpConfig, ScaleEncoding};

/// Expert-parallel ranks.
pub const EP: u32 = 2;
/// Tensor-parallel ranks for the dense layers.
pub const TP: u32 = 2;
/// MX block along K.
pub const MX_BLOCK: u64 = 32;
/// Sub-record alignment inside an expert record.
pub const SUB_ALIGN: u64 = 256;
/// Record alignment for `O_DIRECT` (the filesystem block; the drive's logical block is 512).
pub const RECORD_ALIGN: u64 = 4096;

/// Where a dense record lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Placement {
    /// One rank's TP slice.
    Rank(u32),
    /// Replicated: stored once, loaded by every rank.
    Shared,
}

impl Placement {
    /// Directory name under the pack root.
    #[must_use]
    pub fn dir(self) -> String {
        match self {
            Placement::Rank(r) => format!("rank{r}"),
            Placement::Shared => "shared".into(),
        }
    }
}

/// FN1's budget lines (`saber-cache/ws-fn1/budget.txt`), one per record class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    /// Routed experts, MXFP4 (the expert tier).
    Experts,
    /// GDN projections, conv, norms, A_log / dt_bias.
    Gdn,
    /// Attention q (+gate) / o.
    AttnQo,
    /// Attention k / v.
    AttnKv,
    /// QSA indexer + q/k norms.
    Indexer,
    /// Hyper-connection mixers (per layer + the final mixer).
    Hc,
    /// Router + shared-expert gate.
    Router,
    /// The shared expert, MXFP4.
    SharedExpert,
    /// PLE projections, norms, conv, hash buffers.
    PleDense,
    /// `embed_tokens` (vocab-parallel).
    Embed,
    /// `lm_head` bf16 (vocab-parallel).
    LmHead,
    /// MTP routed experts, bf16.
    MtpExperts,
    /// MTP dense (fc, norms, HC, attention, router, shared expert).
    MtpDense,
    /// The vision tower (optional; not in the per-card weight total, as in FN1).
    Vision,
}

impl Class {
    /// The class's ledger key.
    #[must_use]
    pub fn key(self) -> &'static str {
        match self {
            Class::Experts => "experts",
            Class::Gdn => "gdn",
            Class::AttnQo => "attn.qo",
            Class::AttnKv => "attn.kv",
            Class::Indexer => "indexer",
            Class::Hc => "hc",
            Class::Router => "router",
            Class::SharedExpert => "shared_expert",
            Class::PleDense => "ple.dense",
            Class::Embed => "embed",
            Class::LmHead => "lm_head",
            Class::MtpExperts => "mtp.experts",
            Class::MtpDense => "mtp.dense",
            Class::Vision => "vision",
        }
    }
}

/// How a record's bytes are produced from the checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceKind {
    /// The checkpoint's bf16 bytes, a TP slice of one tensor. Identity layout: may be read in place.
    Bf16,
    /// A dim-0 concatenation of bf16 slices (no arithmetic). Identity per extent: may be read in
    /// place as an extent list.
    Bf16Merged,
    /// bf16 widened to f32 (`A_log`, `dt_bias`, which the GDN kernels read as f32). Written.
    F32FromBf16,
    /// Raw non-float bytes (the PLE hash buffers, I64). May be read in place.
    Raw,
    /// Quark MXFP4: the checkpoint's e2m1 nibbles and e8m0 scales, BYTE COPIED — `permute_w`
    /// (WMMA fragment order), the scale transpose to `[K/32, N]`, `wref` = the row max of the
    /// scales. No requantization. Written.
    QuarkMxfp4,
    /// FN144: an FP4-Local record (see [`Fp4Local`]).
    ModeloptFp4Local(Fp4Local),
}

/// FN144: how an FP4-Local record is made, every form lossless.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp4Local {
    /// MXFP8 carried as stored: part `w` = the e4m3 `[N, K]` slice, part `ws` = the UE8M0
    /// `[N, K/32]` slice of `weight_scale`. Identity (read in place).
    Mxfp8,
    /// MXFP8 dequantized to bf16 (`e4m3 * 2^(e - 127)`, exact). Written.
    Mxfp8ToBf16,
    /// W4A16 NVFP4 (plain E4M3 scales): `w` permuted, `ws` transposed `[K/16, N]`, `gs` the f32
    /// globals. Written.
    Nvfp4Trio,
    /// A tensor this checkpoint stores as f32 where Quark's is bf16 (the 97 target HC norms and
    /// the PLE norms): the checkpoint's f32 bytes. Identity.
    F32Copy,
}

/// FN144: FP4-Local's f32 tensors that are bf16 in Quark (checked against the headers by
/// `saber flashnext-pack --source fp4-local --dry-run`).
#[must_use]
pub fn fp4local_f32(tensor: &str) -> bool {
    tensor.starts_with("model.language_model.")
        && (tensor.ends_with(".hc_norm.weight")
            || tensor.ends_with(".ple.norm_conv.weight")
            || tensor.ends_with(".ple.norm_key.weight")
            || tensor.ends_with(".ple.norm_query.weight"))
}

impl SourceKind {
    /// The manifest name.
    #[must_use]
    pub fn key(self) -> &'static str {
        match self {
            SourceKind::Bf16 => "bf16",
            SourceKind::Bf16Merged => "bf16_merged",
            SourceKind::F32FromBf16 => "f32_from_bf16",
            SourceKind::Raw => "raw",
            SourceKind::QuarkMxfp4 => "quark_mxfp4",
            SourceKind::ModeloptFp4Local(Fp4Local::Mxfp8) => "fp4local_mxfp8",
            SourceKind::ModeloptFp4Local(Fp4Local::Mxfp8ToBf16) => "fp4local_mxfp8_bf16",
            SourceKind::ModeloptFp4Local(Fp4Local::Nvfp4Trio) => "fp4local_nvfp4",
            SourceKind::ModeloptFp4Local(Fp4Local::F32Copy) => "fp4local_f32",
        }
    }
    /// Whether the record's bytes are checkpoint bytes in order (so a reference can stand in
    /// for a copy).
    #[must_use]
    pub fn identity(self) -> bool {
        matches!(
            self,
            SourceKind::Bf16
                | SourceKind::Bf16Merged
                | SourceKind::Raw
                | SourceKind::ModeloptFp4Local(Fp4Local::Mxfp8 | Fp4Local::F32Copy)
        )
    }
}

/// A slice of one checkpoint tensor: rows `[r0, r1)` and columns `[c0, c1)` in ELEMENT units of
/// the tensor's own dim 0 / flattened remaining dims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slice {
    /// Checkpoint tensor name.
    pub tensor: String,
    /// Full-tensor shape (checked against the header by the builder).
    pub shape: Vec<u64>,
    /// Row range.
    pub rows: (u64, u64),
    /// Column range (over the product of the trailing dims).
    pub cols: (u64, u64),
}

impl Slice {
    fn whole(tensor: String, shape: &[u64]) -> Self {
        let rows = shape[0];
        let cols: u64 = shape[1..].iter().product::<u64>().max(1);
        Self {
            tensor,
            shape: shape.to_vec(),
            rows: (0, rows),
            cols: (0, cols),
        }
    }
    fn rows(tensor: String, shape: &[u64], r0: u64, r1: u64) -> Self {
        let mut s = Self::whole(tensor, shape);
        s.rows = (r0, r1);
        s
    }
    fn cols(tensor: String, shape: &[u64], c0: u64, c1: u64) -> Self {
        let mut s = Self::whole(tensor, shape);
        s.cols = (c0, c1);
        s
    }
    fn elems(&self) -> u64 {
        (self.rows.1 - self.rows.0) * (self.cols.1 - self.cols.0)
    }
}

/// A loader alias: a merged weight as identity extents over other records' checkpoint bytes.
///
/// The QSA merged projection (FN6's EXEC §13 rule, FN9's pack rule): rank `r`'s
/// `<layer>.self_attn.qkv_idx.weight` = rows `[q_proj half r (q|gate x 12 heads, 6,144) | k_proj
/// half r (256) | v_proj half r (256) | index_qk_proj WHOLE (640, replicated)]` x 2,560 bf16 =
/// N 7,296, the ONE GEMM whose output every consumer reads in place (qk_norm_rope, the gate at
/// 7,296, the append's V at 6,400, the indexer at 6,656).
#[derive(Clone, Debug, PartialEq)]
pub struct Alias {
    /// Alias name.
    pub name: String,
    /// Where it lives (a rank).
    pub placement: Placement,
    /// `[rows, cols]`.
    pub shape: Vec<u64>,
    /// The member slices, in row order.
    pub sources: Vec<Slice>,
}

/// One dense record.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseRecord {
    /// Record name (the checkpoint tensor's name for a passthrough; the op's for a merge).
    pub name: String,
    /// Budget class.
    pub class: Class,
    /// Rank slice or shared.
    pub placement: Placement,
    /// How it is built.
    pub kind: SourceKind,
    /// Record dtype (safetensors spelling, plus `FP4_E2M1_X2` / `E8M0` for the MXFP4 trio).
    pub dtype: &'static str,
    /// Record shape (logical: an MXFP4 `w` is `[N, K]` with two values per byte).
    pub shape: Vec<u64>,
    /// Sources, in concatenation order (for `quark_mxfp4` the WEIGHT tensors; the scales are the
    /// same names + `_scale`).
    pub sources: Vec<Slice>,
    /// The trio member, when `kind` is `QuarkMxfp4` (`w`, `ws`, `wref`), FP4-Local MXFP8 (`w`,
    /// `ws`) or FP4-Local NVFP4 (`w`, `ws`, `gs`).
    pub part: Option<&'static str>,
    /// Payload bytes.
    pub bytes: u64,
}

/// Offsets inside one expert-tier record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertRecordLayout {
    /// (offset, bytes) of gate_up `w` `[2I, H]` e2m1, `permute_w` order.
    pub gate_up_w: (u64, u64),
    /// gate_up `ws` `[H/32, 2I]` e8m0.
    pub gate_up_ws: (u64, u64),
    /// gate_up `wref` `[2I]` e8m0.
    pub gate_up_wref: (u64, u64),
    /// down `w` `[H, I]` e2m1, `permute_w` order.
    pub down_w: (u64, u64),
    /// down `ws` `[I/32, H]`.
    pub down_ws: (u64, u64),
    /// down `wref` `[H]`.
    pub down_wref: (u64, u64),
    /// Sum of the six payloads.
    pub payload: u64,
    /// Record stride on disk (payload + sub-alignment, rounded to 4 KiB).
    pub stride: u64,
    /// The record format (`MXFP4_E8M0`, or FN144's `NVFP4_E4M3G16` whose `*_wref` slots hold the
    /// f32 globals `gs`: gate_up 2 x f32 (gate, up), down 1 x f32).
    pub format: &'static str,
}

impl ExpertRecordLayout {
    /// The record layout for experts of `hidden` × `inter`.
    #[must_use]
    pub fn new(hidden: u64, inter: u64) -> Self {
        let gu_n = 2 * inter;
        let sizes = [
            gu_n * hidden / 2,
            (hidden / MX_BLOCK) * gu_n,
            gu_n,
            hidden * inter / 2,
            (inter / MX_BLOCK) * hidden,
            hidden,
        ];
        let mut off = 0u64;
        let mut at = [(0u64, 0u64); 6];
        for (i, s) in sizes.iter().enumerate() {
            at[i] = (off, *s);
            off = (off + s).div_ceil(SUB_ALIGN) * SUB_ALIGN;
        }
        let end = at[5].0 + at[5].1;
        Self {
            gate_up_w: at[0],
            gate_up_ws: at[1],
            gate_up_wref: at[2],
            down_w: at[3],
            down_ws: at[4],
            down_wref: at[5],
            payload: sizes.iter().sum(),
            stride: end.div_ceil(RECORD_ALIGN) * RECORD_ALIGN,
            format: "MXFP4_E8M0",
        }
    }

    /// FN144: the v2 NVFP4 record layout (`saber_mem::xtier::record_layout_nvfp4`'s numbers):
    /// gate_up `w` `[2I, H]` e2m1, `ws` E4M3 `[H/16, 2I]`, `gs` 2 x f32; down `w` `[H, I]`, `ws`
    /// `[I/16, H]`, `gs` 1 x f32. Payload = the end of the last sub-record.
    #[must_use]
    pub fn nvfp4(hidden: u64, inter: u64) -> Self {
        let gu_n = 2 * inter;
        let sizes = [
            gu_n * hidden / 2,
            (hidden / 16) * gu_n,
            8,
            hidden * inter / 2,
            (inter / 16) * hidden,
            4,
        ];
        let mut off = 0u64;
        let mut at = [(0u64, 0u64); 6];
        for (i, s) in sizes.iter().enumerate() {
            at[i] = (off, *s);
            off = (off + s).div_ceil(SUB_ALIGN) * SUB_ALIGN;
        }
        let end = at[5].0 + at[5].1;
        Self {
            gate_up_w: at[0],
            gate_up_ws: at[1],
            gate_up_wref: at[2],
            down_w: at[3],
            down_ws: at[4],
            down_wref: at[5],
            payload: end,
            stride: end.div_ceil(RECORD_ALIGN) * RECORD_ALIGN,
            format: "NVFP4_E4M3G16",
        }
    }
    /// The checkpoint bytes one expert carries (`w` + `ws`, what FN1's budget counts).
    #[must_use]
    pub fn checkpoint_bytes(&self) -> u64 {
        self.gate_up_w.1 + self.gate_up_ws.1 + self.down_w.1 + self.down_ws.1
    }
    /// The `wref` bytes (derived; not in the checkpoint).
    #[must_use]
    pub fn wref_bytes(&self) -> u64 {
        self.gate_up_wref.1 + self.down_wref.1
    }
}

/// One rank's expert tier file.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpertTier {
    /// Owning rank.
    pub rank: u32,
    /// First owned expert id (`256 r`).
    pub first_expert: u32,
    /// Owned experts per layer (256).
    pub experts: u32,
    /// Layers (48).
    pub layers: u32,
    /// The fixed record layout.
    pub record: ExpertRecordLayout,
}

impl ExpertTier {
    /// Records in the file.
    #[must_use]
    pub fn records(&self) -> u64 {
        u64::from(self.layers) * u64::from(self.experts)
    }
    /// Record index of (layer, global expert id) — the LRU's address law.
    #[must_use]
    pub fn index(&self, layer: u32, expert: u32) -> Option<u64> {
        if layer >= self.layers
            || expert < self.first_expert
            || expert >= self.first_expert + self.experts
        {
            return None;
        }
        Some(u64::from(layer) * u64::from(self.experts) + u64::from(expert - self.first_expert))
    }
}

/// The PLE table tier (an index over the checkpoint's shards; see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct PleTier {
    /// Rows in the (padded) table: 320,001,536.
    pub rows: u64,
    /// Rows per checkpoint shard: 2,500,012.
    pub rows_per_shard: u64,
    /// Shards: 128.
    pub shards: u32,
    /// Elements per row: 160.
    pub row_dim: u32,
    /// Rows looked up per token: 16.
    pub rows_per_token: u32,
    /// Checkpoint tensor name of shard `k` = `tensor_prefix` + k + `.weight`.
    pub tensor_prefix: String,
}

impl PleTier {
    /// Bytes per bf16 row: 320.
    #[must_use]
    pub fn row_bytes_bf16(&self) -> u64 {
        u64::from(self.row_dim) * 2
    }
    /// Bytes per fp8 row (the optional first-party variant): 160.
    #[must_use]
    pub fn row_bytes_fp8(&self) -> u64 {
        u64::from(self.row_dim)
    }
    /// FN144: bytes per NVFP4 row of the FP4-Local table (80 e2m1 + 10 E4M3 scales, no global): 90.
    #[must_use]
    pub fn row_bytes_nvfp4(&self) -> u64 {
        u64::from(self.row_dim) / 2 + u64::from(self.row_dim) / 16
    }
    /// (shard, local row) of a global row id (upstream `load_weights`: shard k holds rows
    /// `[k * rows_per_shard, ...)`).
    #[must_use]
    pub fn locate(&self, row: u64) -> Option<(u32, u64)> {
        if row >= self.rows {
            return None;
        }
        Some((
            (row / self.rows_per_shard) as u32,
            row % self.rows_per_shard,
        ))
    }
}

/// Which checkpoint a plan is built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackSource {
    /// AMD Quark MXFP4 (the served pack).
    Quark,
    /// FN144: NVIDIA ModelOpt FP4-Local (the lossless path).
    Fp4Local,
}

impl PackSource {
    /// The manifest spelling.
    #[must_use]
    pub fn key(self) -> &'static str {
        match self {
            PackSource::Quark => "quark",
            PackSource::Fp4Local => "fp4_local",
        }
    }
}

/// The whole plan.
#[derive(Clone, Debug, PartialEq)]
pub struct PackPlan {
    /// The checkpoint kind.
    pub source: PackSource,
    /// Dense records in forward order.
    pub dense: Vec<DenseRecord>,
    /// Loader ALIASES (FN9): one GEMM's merged weight re-addressed as identity extents over bytes
    /// the `dense` records already own -- no byte rewritten, no ledger line (the engine loads the
    /// alias INSTEAD of its members, so the members' ledger lines are its bytes).
    pub aliases: Vec<Alias>,
    /// One expert tier per rank.
    pub experts: Vec<ExpertTier>,
    /// The PLE table.
    pub ple: PleTier,
}

struct Builder<'a> {
    c: &'a Qwen4ExpConfig,
    out: Vec<DenseRecord>,
    aliases: Vec<Alias>,
    /// FN144: building from the FP4-Local checkpoint.
    fp4: bool,
}

impl Builder<'_> {
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        name: String,
        class: Class,
        placement: Placement,
        kind: SourceKind,
        dtype: &'static str,
        shape: Vec<u64>,
        sources: Vec<Slice>,
    ) {
        let elems: u64 = shape.iter().product();
        let bytes = match dtype {
            "BF16" => elems * 2,
            "F32" => elems * 4,
            "I64" => elems * 8,
            "FP4_E2M1_X2" => elems / 2,
            _ => elems,
        };
        self.out.push(DenseRecord {
            name,
            class,
            placement,
            kind,
            dtype,
            shape,
            sources,
            part: None,
            bytes,
        });
    }

    /// FN144: the FP4-Local form of a WEIGHT tensor (`None` = bf16 in this checkpoint too).
    fn form(&self, tensor: &str) -> Option<Fp4Local> {
        if !self.fp4 {
            return None;
        }
        let module = tensor.strip_suffix(".weight")?;
        let l = self.c.quant.modelopt_layer(module)?;
        match l.algo {
            ModeloptAlgo::Mxfp8 => {
                let small = module.ends_with(".linear_attn.in_proj_a")
                    || module.ends_with(".linear_attn.in_proj_b")
                    || module.ends_with(".indexer.index_qk_proj");
                Some(if small {
                    Fp4Local::Mxfp8ToBf16
                } else {
                    Fp4Local::Mxfp8
                })
            }
            // NVFP4 / W4A16 modules are expert blocks (the tier, the MTP trio) or the PLE table,
            // never reached through a dense passthrough; reaching one is a plan bug.
            a => panic!("fp4local: `{module}` ({}) has no dense form", a.key()),
        }
    }

    /// A dense record that is bf16 on Quark: bf16 (`bf16_kind`), or its FP4-Local form.
    fn emit(
        &mut self,
        name: String,
        class: Class,
        placement: Placement,
        bf16_kind: SourceKind,
        shape: Vec<u64>,
        sources: Vec<Slice>,
    ) {
        let form = self.form(&sources[0].tensor);
        assert!(
            sources.iter().all(|s| self.form(&s.tensor) == form),
            "{name}: a merge over sources of different forms"
        );
        match form {
            None if self.fp4 && sources.iter().all(|s| fp4local_f32(&s.tensor)) => self.push(
                name,
                class,
                placement,
                SourceKind::ModeloptFp4Local(Fp4Local::F32Copy),
                "F32",
                shape,
                sources,
            ),
            None => self.push(name, class, placement, bf16_kind, "BF16", shape, sources),
            Some(Fp4Local::Mxfp8ToBf16) => self.push(
                name,
                class,
                placement,
                SourceKind::ModeloptFp4Local(Fp4Local::Mxfp8ToBf16),
                "BF16",
                shape,
                sources,
            ),
            Some(Fp4Local::Mxfp8) => {
                let kind = SourceKind::ModeloptFp4Local(Fp4Local::Mxfp8);
                let scales: Vec<Slice> = sources
                    .iter()
                    .map(|s| {
                        assert!(
                            s.shape.len() == 2
                                && s.cols.0 % MX_BLOCK == 0
                                && s.cols.1 % MX_BLOCK == 0,
                            "{}: an MXFP8 slice must cut K on a 32 boundary",
                            s.tensor
                        );
                        Slice {
                            tensor: format!("{}_scale", s.tensor),
                            shape: vec![s.shape[0], s.shape[1] / MX_BLOCK],
                            rows: s.rows,
                            cols: (s.cols.0 / MX_BLOCK, s.cols.1 / MX_BLOCK),
                        }
                    })
                    .collect();
                let sshape = vec![shape[0], shape[1] / MX_BLOCK];
                self.push(
                    name.clone(),
                    class,
                    placement,
                    kind,
                    "F8_E4M3",
                    shape,
                    sources,
                );
                self.out.last_mut().expect("just pushed").part = Some("w");
                self.push(
                    format!("{name}_scale"),
                    class,
                    placement,
                    kind,
                    "UE8M0",
                    sshape,
                    scales,
                );
                self.out.last_mut().expect("just pushed").part = Some("ws");
            }
            Some(Fp4Local::Nvfp4Trio | Fp4Local::F32Copy) => {
                unreachable!("form() never returns the trio or the f32 copy")
            }
        }
    }

    /// FN144: a W4A16 NVFP4 trio (`w` permuted, `ws` transposed, `gs` globals) over the
    /// projections `(tensor prefix, rows)` of one merged GEMM with `k` inputs.
    fn nvfp4_trio(
        &mut self,
        op: &str,
        class: Class,
        placement: Placement,
        projs: &[(String, u64)],
        k: u64,
    ) {
        let n: u64 = projs.iter().map(|p| p.1).sum();
        let ws: Vec<Slice> = projs
            .iter()
            .map(|(t, rows)| Slice::whole(format!("{t}.weight"), &[*rows, k / 2]))
            .collect();
        let sc: Vec<Slice> = projs
            .iter()
            .map(|(t, rows)| Slice::whole(format!("{t}.weight_scale"), &[*rows, k / 16]))
            .collect();
        let gs: Vec<Slice> = projs
            .iter()
            .map(|(t, _)| Slice::whole(format!("{t}.weight_scale_2"), &[1]))
            .collect();
        let kind = SourceKind::ModeloptFp4Local(Fp4Local::Nvfp4Trio);
        for (part, dtype, shape, src) in [
            ("w", "FP4_E2M1_X2", vec![n, k], ws),
            ("ws", "E4M3", vec![k / 16, n], sc),
            ("gs", "F32", vec![projs.len() as u64], gs),
        ] {
            self.push(
                format!("{op}.{part}"),
                class,
                placement,
                kind,
                dtype,
                shape,
                src,
            );
            self.out.last_mut().expect("just pushed").part = Some(part);
        }
    }

    /// A replicated bf16 tensor, stored once in `shared/`.
    fn shared(&mut self, name: &str, class: Class, shape: &[u64]) {
        self.emit(
            name.to_string(),
            class,
            Placement::Shared,
            SourceKind::Bf16,
            shape.to_vec(),
            vec![Slice::whole(name.to_string(), shape)],
        );
    }

    /// A bf16 tensor split on dim 0 across the TP ranks.
    fn dim0(&mut self, name: &str, class: Class, shape: &[u64]) {
        let per = shape[0] / u64::from(TP);
        for r in 0..TP {
            let mut s = shape.to_vec();
            s[0] = per;
            let lo = u64::from(r) * per;
            self.emit(
                name.to_string(),
                class,
                Placement::Rank(r),
                SourceKind::Bf16,
                s,
                vec![Slice::rows(name.to_string(), shape, lo, lo + per)],
            );
        }
    }

    /// A bf16 2-D tensor split on dim 1 (row-parallel K).
    fn dim1(&mut self, name: &str, class: Class, shape: &[u64]) {
        let per = shape[1] / u64::from(TP);
        for r in 0..TP {
            let lo = u64::from(r) * per;
            self.emit(
                name.to_string(),
                class,
                Placement::Rank(r),
                SourceKind::Bf16,
                vec![shape[0], per],
                vec![Slice::cols(name.to_string(), shape, lo, lo + per)],
            );
        }
    }

    /// A dim-0 merge of segmented, TP-split sources: each `(tensor, shape, segments)` contributes
    /// rank `r`'s half of every segment, in order.
    fn merged(&mut self, name: &str, class: Class, parts: &[(String, Vec<u64>, Vec<u64>)]) {
        for r in 0..TP {
            let mut sources = Vec::new();
            let mut rows = 0u64;
            let trailing = parts[0].1[1..].to_vec();
            for (t, shape, segs) in parts {
                let mut base = 0;
                for seg in segs {
                    let per = seg / u64::from(TP);
                    let lo = base + u64::from(r) * per;
                    sources.push(Slice::rows(t.clone(), shape, lo, lo + per));
                    rows += per;
                    base += seg;
                }
            }
            let mut shape = vec![rows];
            shape.extend(&trailing);
            let kind = if sources.len() == 1 {
                SourceKind::Bf16
            } else {
                SourceKind::Bf16Merged
            };
            self.emit(
                name.to_string(),
                class,
                Placement::Rank(r),
                kind,
                shape,
                sources,
            );
        }
    }

    /// The MXFP4 trio of one merged Quark GEMM (the shared expert), replicated.
    fn quark_trio(&mut self, op: &str, class: Class, weights: &[(String, u64, u64)]) {
        if self.fp4 {
            // FN144: the shared expert is MXFP8 in FP4-Local (trained so): carried as stored,
            // replicated, gate_up merged.
            let n: u64 = weights.iter().map(|w| w.1).sum();
            let k = weights[0].2;
            let sources: Vec<Slice> = weights
                .iter()
                .map(|(t, rows, kk)| Slice::whole(t.clone(), &[*rows, *kk]))
                .collect();
            let kind = if sources.len() == 1 {
                SourceKind::Bf16
            } else {
                SourceKind::Bf16Merged
            };
            self.emit(
                format!("{op}.w"),
                class,
                Placement::Shared,
                kind,
                vec![n, k],
                sources,
            );
            return;
        }
        let n: u64 = weights.iter().map(|w| w.1).sum();
        let k = weights[0].2;
        let sources: Vec<Slice> = weights
            .iter()
            .map(|(t, rows, kk)| Slice::whole(t.clone(), &[*rows, kk / 2]))
            .collect();
        for (part, dtype, shape) in [
            ("w", "FP4_E2M1_X2", vec![n, k]),
            ("ws", "E8M0", vec![k / MX_BLOCK, n]),
            ("wref", "E8M0", vec![n]),
        ] {
            self.push(
                format!("{op}.{part}"),
                class,
                Placement::Shared,
                SourceKind::QuarkMxfp4,
                dtype,
                shape,
                sources.clone(),
            );
            self.out.last_mut().expect("just pushed").part = Some(part);
        }
    }

    fn hc(&mut self, prefix: &str, class: Class) {
        let w = u64::from(self.c.hc_width());
        let lr = u64::from(self.c.hc.lowrank);
        let s = u64::from(self.c.hc.count);
        for hc in ["attn_hyper_connection", "mlp_hyper_connection"] {
            let p = format!("{prefix}.{hc}");
            self.shared(&format!("{p}.block_inject_weight.weight"), class, &[s, w]);
            self.shared(&format!("{p}.hc_norm.weight"), class, &[w]);
            self.shared(
                &format!("{p}.input_mix_weight_down.weight"),
                class,
                &[lr, w],
            );
            self.shared(&format!("{p}.input_mix_weight_up.weight"), class, &[w, lr]);
        }
    }

    fn qsa(&mut self, p: &str, qo: Class, kv: Class, idx: Class) {
        let c = self.c;
        let h = u64::from(c.hidden_size);
        let hd = u64::from(c.attention.head_dim);
        let kvr = u64::from(c.attention.num_kv_heads) * hd;
        let a = format!("{p}.self_attn");
        self.dim0(
            &format!("{a}.q_proj.weight"),
            qo,
            &[u64::from(c.q_proj_rows()), h],
        );
        self.dim0(&format!("{a}.k_proj.weight"), kv, &[kvr, h]);
        self.dim0(&format!("{a}.v_proj.weight"), kv, &[kvr, h]);
        self.shared(&format!("{a}.q_norm.weight"), idx, &[hd]);
        self.shared(&format!("{a}.k_norm.weight"), idx, &[hd]);
        let ih = u64::from(c.qsa.head_dim);
        self.shared(
            &format!("{a}.indexer.index_qk_proj.weight"),
            idx,
            &[u64::from(c.qsa.qk_proj_rows()), h],
        );
        self.shared(&format!("{a}.indexer.q_layernorm.weight"), idx, &[ih]);
        self.shared(&format!("{a}.indexer.k_layernorm.weight"), idx, &[ih]);
        self.dim1(
            &format!("{a}.o_proj.weight"),
            qo,
            &[h, u64::from(c.attention.num_heads) * hd],
        );
        // FN9: the merged projection alias per rank (q|gate half, k half, v half, index_qk whole).
        // FN144: none on FP4-Local (the members are fp8 / bf16; the fp8 merge is FN133 P4's).
        if self.fp4 {
            return;
        }
        let qr = u64::from(c.q_proj_rows());
        let ir = u64::from(c.qsa.qk_proj_rows());
        for r in 0..TP {
            let (qp, kp) = (qr / u64::from(TP), kvr / u64::from(TP));
            let (q0, k0) = (u64::from(r) * qp, u64::from(r) * kp);
            let sources = vec![
                Slice::rows(format!("{a}.q_proj.weight"), &[qr, h], q0, q0 + qp),
                Slice::rows(format!("{a}.k_proj.weight"), &[kvr, h], k0, k0 + kp),
                Slice::rows(format!("{a}.v_proj.weight"), &[kvr, h], k0, k0 + kp),
                Slice::whole(format!("{a}.indexer.index_qk_proj.weight"), &[ir, h]),
            ];
            self.aliases.push(Alias {
                name: format!("{a}.qkv_idx.weight"),
                placement: Placement::Rank(r),
                shape: vec![qp + 2 * kp + ir, h],
                sources,
            });
        }
    }
}

impl PackPlan {
    /// The EP=2 / TP=2 plan for a Quark MXFP4 Flash-Next checkpoint.
    #[must_use]
    pub fn ep2(c: &Qwen4ExpConfig) -> Self {
        Self::build(c, false)
    }

    /// FN144: the EP=2 / TP=2 plan for the FP4-Local checkpoint (the lossless path; module docs).
    pub fn fp4local(c: &Qwen4ExpConfig) -> Result<Self, String> {
        if !c.quant.is_fp4_local() {
            return Err("fp4local: the checkpoint's quantization_config is not FP4-Local (no CSF NVFP4 experts)".into());
        }
        let lm = "model.language_model.layers";
        for l in 0..c.num_layers() {
            let m = format!("{lm}.{l}.mlp.experts");
            match c.quant.modelopt_layer(&m) {
                Some(x) if x.algo == ModeloptAlgo::Nvfp4 && x.encoding == ScaleEncoding::Csf => {}
                other => return Err(format!("fp4local: `{m}` is {other:?}, not CSF NVFP4")),
            }
        }
        for ml in 0..c.mtp.num_layers {
            let m = format!("mtp.layers.{ml}.mlp.experts");
            if c.quant.modelopt_layer(&m).map(|x| x.algo) != Some(ModeloptAlgo::W4a16Nvfp4) {
                return Err(format!("fp4local: `{m}` is not W4A16_NVFP4"));
            }
        }
        Ok(Self::build(c, true))
    }

    fn build(c: &Qwen4ExpConfig, fp4: bool) -> Self {
        let mut b = Builder {
            c,
            out: Vec::new(),
            aliases: Vec::new(),
            fp4,
        };
        let h = u64::from(c.hidden_size);
        let v = u64::from(c.vocab_size);
        let w = u64::from(c.hc_width());
        let lr = u64::from(c.hc.lowrank);
        let e_inter = u64::from(c.moe.intermediate);
        let s_inter = u64::from(c.moe.shared_intermediate);
        let lm = "model.language_model";

        b.dim0(&format!("{lm}.embed_tokens.weight"), Class::Embed, &[v, h]);
        let ple_layers = c.ple.layer_indices();
        for (l, kind) in c.layers.iter().enumerate() {
            let p = format!("{lm}.layers.{l}");
            b.hc(&p, Class::Hc);
            if ple_layers.contains(&(l as u32)) {
                let pl = format!("{p}.ple");
                b.shared(&format!("{pl}.key_proj.weight"), Class::PleDense, &[w, h]);
                b.shared(&format!("{pl}.value_proj.weight"), Class::PleDense, &[h, h]);
                for n in ["norm_conv", "norm_key", "norm_query"] {
                    b.shared(&format!("{pl}.{n}.weight"), Class::PleDense, &[w]);
                }
                b.shared(
                    &format!("{pl}.conv1d.weight"),
                    Class::PleDense,
                    &[w, 1, u64::from(c.ple.conv_kernel)],
                );
                let heads = u64::from(c.ple.ngram_heads());
                for (n, len) in [
                    ("layer_multipliers", u64::from(c.ple.ngram_size)),
                    ("ngram_heads_offsets", heads),
                    ("ngram_heads_vocab_sizes", heads),
                ] {
                    let t = format!("{pl}.ple_embedding.{n}");
                    b.push(
                        t.clone(),
                        Class::PleDense,
                        Placement::Shared,
                        SourceKind::Raw,
                        "I64",
                        vec![len],
                        vec![Slice::whole(t, &[len])],
                    );
                }
            }
            match kind {
                LayerKind::Gdn => {
                    let g = &c.gdn;
                    let la = format!("{p}.linear_attn");
                    let (kd, vd) = (u64::from(g.key_dim()), u64::from(g.value_dim()));
                    let qkv = u64::from(g.qkv_rows());
                    let nv = u64::from(g.num_value_heads);
                    b.merged(
                        &format!("layer.{l}.gdn.in_proj_qkvz.w"),
                        Class::Gdn,
                        &[
                            (
                                format!("{la}.in_proj_qkv.weight"),
                                vec![qkv, h],
                                vec![kd, kd, vd],
                            ),
                            (format!("{la}.in_proj_z.weight"), vec![vd, h], vec![vd]),
                        ],
                    );
                    // `b` before `a`: the 27B's merge order (graph.rs `in_proj_ba`).
                    b.merged(
                        &format!("layer.{l}.gdn.in_proj_ba.w"),
                        Class::Gdn,
                        &[
                            (format!("{la}.in_proj_b.weight"), vec![nv, h], vec![nv]),
                            (format!("{la}.in_proj_a.weight"), vec![nv, h], vec![nv]),
                        ],
                    );
                    b.merged(
                        &format!("{la}.conv1d.weight"),
                        Class::Gdn,
                        &[(
                            format!("{la}.conv1d.weight"),
                            vec![qkv, 1, u64::from(g.conv_kernel)],
                            vec![kd, kd, vd],
                        )],
                    );
                    for t in ["A_log", "dt_bias"] {
                        let name = format!("{la}.{t}");
                        let per = nv / u64::from(TP);
                        for r in 0..TP {
                            let lo = u64::from(r) * per;
                            b.push(
                                name.clone(),
                                Class::Gdn,
                                Placement::Rank(r),
                                SourceKind::F32FromBf16,
                                "F32",
                                vec![per],
                                vec![Slice::rows(name.clone(), &[nv], lo, lo + per)],
                            );
                        }
                    }
                    b.shared(
                        &format!("{la}.norm.weight"),
                        Class::Gdn,
                        &[u64::from(g.value_head_dim)],
                    );
                    b.dim1(&format!("{la}.out_proj.weight"), Class::Gdn, &[h, vd]);
                }
                LayerKind::Qsa => b.qsa(&p, Class::AttnQo, Class::AttnKv, Class::Indexer),
            }
            let m = format!("{p}.mlp");
            b.shared(
                &format!("{m}.gate.weight"),
                Class::Router,
                &[u64::from(c.moe.num_experts), h],
            );
            b.shared(
                &format!("{m}.shared_expert_gate.weight"),
                Class::Router,
                &[1, h],
            );
            let se = format!("{m}.shared_expert");
            b.quark_trio(
                &format!("layer.{l}.mlp.shared_expert.gate_up"),
                Class::SharedExpert,
                &[
                    (format!("{se}.gate_proj.weight"), s_inter, h),
                    (format!("{se}.up_proj.weight"), s_inter, h),
                ],
            );
            b.quark_trio(
                &format!("layer.{l}.mlp.shared_expert.down"),
                Class::SharedExpert,
                &[(format!("{se}.down_proj.weight"), h, s_inter)],
            );
        }
        let fm = format!("{lm}.hyper_connection_mixer");
        b.shared(&format!("{fm}.hc_norm.weight"), Class::Hc, &[w]);
        b.shared(
            &format!("{fm}.input_mix_weight_down.weight"),
            Class::Hc,
            &[lr, w],
        );
        b.shared(
            &format!("{fm}.input_mix_weight_up.weight"),
            Class::Hc,
            &[w, lr],
        );
        b.dim0("lm_head.weight", Class::LmHead, &[v, h]);

        // ---- MTP -----------------------------------------------------------------------------
        b.shared("mtp.pre_fc_norm_embedding.weight", Class::MtpDense, &[h]);
        b.shared("mtp.pre_fc_norm_hidden.weight", Class::MtpDense, &[w]);
        b.dim0("mtp.fc_embedding.weight", Class::MtpDense, &[h, h]);
        b.dim0("mtp.fc_hidden.weight", Class::MtpDense, &[h, h]);
        for ml in 0..c.mtp.num_layers {
            let p = format!("mtp.layers.{ml}");
            b.hc(&p, Class::MtpDense);
            b.qsa(&p, Class::MtpDense, Class::MtpDense, Class::MtpDense);
            let m = format!("{p}.mlp");
            b.shared(
                &format!("{m}.gate.weight"),
                Class::MtpDense,
                &[u64::from(c.moe.num_experts), h],
            );
            b.shared(
                &format!("{m}.shared_expert_gate.weight"),
                Class::MtpDense,
                &[1, h],
            );
            // The bf16 shared expert, replicated (§6: never split by K), gate_up merged.
            let se = format!("{m}.shared_expert");
            b.push(
                format!("mtp.layer.{ml}.mlp.shared_expert.gate_up.w"),
                Class::MtpDense,
                Placement::Shared,
                SourceKind::Bf16Merged,
                "BF16",
                vec![2 * s_inter, h],
                vec![
                    Slice::whole(format!("{se}.gate_proj.weight"), &[s_inter, h]),
                    Slice::whole(format!("{se}.up_proj.weight"), &[s_inter, h]),
                ],
            );
            b.shared(
                &format!("{se}.down_proj.weight"),
                Class::MtpDense,
                &[h, s_inter],
            );
            // The MTP routed experts: EP-owned like the target's, bf16, resident.
            let per = u64::from(c.moe.num_experts / EP);
            for r in 0..EP {
                for e in (u64::from(r) * per)..(u64::from(r + 1) * per) {
                    let x = format!("{m}.experts.{e}");
                    if fp4 {
                        // FN144: W4A16 NVFP4, plain E4M3 scales -> the trio, resident.
                        b.nvfp4_trio(
                            &format!("mtp.layer.{ml}.expert.{e}.gate_up"),
                            Class::MtpExperts,
                            Placement::Rank(r),
                            &[
                                (format!("{x}.gate_proj"), e_inter),
                                (format!("{x}.up_proj"), e_inter),
                            ],
                            h,
                        );
                        b.nvfp4_trio(
                            &format!("mtp.layer.{ml}.expert.{e}.down"),
                            Class::MtpExperts,
                            Placement::Rank(r),
                            &[(format!("{x}.down_proj"), h)],
                            e_inter,
                        );
                        continue;
                    }
                    b.push(
                        format!("mtp.layer.{ml}.expert.{e}.gate_up.w"),
                        Class::MtpExperts,
                        Placement::Rank(r),
                        SourceKind::Bf16Merged,
                        "BF16",
                        vec![2 * e_inter, h],
                        vec![
                            Slice::whole(format!("{x}.gate_proj.weight"), &[e_inter, h]),
                            Slice::whole(format!("{x}.up_proj.weight"), &[e_inter, h]),
                        ],
                    );
                    b.push(
                        format!("{x}.down_proj.weight"),
                        Class::MtpExperts,
                        Placement::Rank(r),
                        SourceKind::Bf16,
                        "BF16",
                        vec![h, e_inter],
                        vec![Slice::whole(format!("{x}.down_proj.weight"), &[h, e_inter])],
                    );
                }
            }
        }
        let fm = "mtp.hyper_connection_mixer";
        b.shared(&format!("{fm}.hc_norm.weight"), Class::MtpDense, &[w]);
        b.shared(
            &format!("{fm}.input_mix_weight_down.weight"),
            Class::MtpDense,
            &[lr, w],
        );
        b.shared(
            &format!("{fm}.input_mix_weight_up.weight"),
            Class::MtpDense,
            &[w, lr],
        );

        let record = if fp4 {
            ExpertRecordLayout::nvfp4(h, e_inter)
        } else {
            ExpertRecordLayout::new(h, e_inter)
        };
        let per = c.moe.num_experts / EP;
        let experts = (0..EP)
            .map(|r| ExpertTier {
                rank: r,
                first_expert: r * per,
                experts: per,
                layers: c.num_layers(),
                record,
            })
            .collect();
        let ple_layer = ple_layers[0];
        let ple = PleTier {
            rows: c.ple_table_rows(),
            rows_per_shard: c.ple_rows_per_shard(),
            shards: c.ple.split_parts,
            row_dim: c.ple.row_dim(),
            rows_per_token: c.ple.ngram_heads(),
            tensor_prefix: format!(
                "{lm}.layers.{ple_layer}.ple.ple_embedding.ngram_embedding.shard_"
            ),
        };
        Self {
            source: if fp4 {
                PackSource::Fp4Local
            } else {
                PackSource::Quark
            },
            dense: b.out,
            aliases: b.aliases,
            experts,
            ple,
        }
    }

    /// Per placement per class: resident bytes. Keys: `rank0`, `rank1`, `shared`.
    #[must_use]
    pub fn ledger(&self) -> Vec<(String, Class, u64)> {
        let mut out: Vec<(String, Class, u64)> = Vec::new();
        let mut add = |p: String, c: Class, b: u64| {
            if let Some(e) = out.iter_mut().find(|e| e.0 == p && e.1 == c) {
                e.2 += b;
            } else {
                out.push((p, c, b));
            }
        };
        for r in &self.dense {
            add(r.placement.dir(), r.class, r.bytes);
        }
        for t in &self.experts {
            add(
                format!("rank{}", t.rank),
                Class::Experts,
                t.records() * (t.record.checkpoint_bytes() + t.record.wref_bytes()),
            );
        }
        out.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
        out
    }

    /// The checkpoint bytes each class CONSUMES, counting every checkpoint byte once (a slice's
    /// bytes, over every rank and shared, with no replication): the number that must equal FN1's
    /// category totals to the byte.
    #[must_use]
    pub fn consumed_by_class(&self) -> Vec<(Class, u64)> {
        let mut out: Vec<(Class, u64)> = Vec::new();
        let mut add = |c: Class, b: u64| {
            if let Some(e) = out.iter_mut().find(|e| e.0 == c) {
                e.1 += b;
            } else {
                out.push((c, b));
            }
        };
        for r in &self.dense {
            match r.kind {
                // One trio consumes its sources once: count at the `w` record, w + scales.
                SourceKind::QuarkMxfp4 => {
                    if r.part == Some("w") {
                        let n: u64 = r.shape[0];
                        let k = r.shape[1];
                        add(r.class, n * k / 2 + n * (k / MX_BLOCK));
                    }
                }
                SourceKind::F32FromBf16 => {
                    add(r.class, r.sources.iter().map(Slice::elems).sum::<u64>() * 2);
                }
                SourceKind::Raw => add(r.class, r.bytes),
                // FN144: identity w / ws: their own bytes; the bf16 dequant consumed e4m3 + scales;
                // the trio at its `w` record: e2m1 + E4M3 per 16 + 4 B a projection.
                SourceKind::ModeloptFp4Local(Fp4Local::Mxfp8 | Fp4Local::F32Copy) => {
                    add(r.class, r.bytes)
                }
                SourceKind::ModeloptFp4Local(Fp4Local::Mxfp8ToBf16) => {
                    let e = r.sources.iter().map(Slice::elems).sum::<u64>();
                    add(r.class, e + e / MX_BLOCK);
                }
                SourceKind::ModeloptFp4Local(Fp4Local::Nvfp4Trio) => {
                    if r.part == Some("w") {
                        let (n, k) = (r.shape[0], r.shape[1]);
                        add(
                            r.class,
                            n * k / 2 + n * (k / 16) + 4 * r.sources.len() as u64,
                        );
                    }
                }
                SourceKind::Bf16 | SourceKind::Bf16Merged => {
                    add(r.class, r.sources.iter().map(Slice::elems).sum::<u64>() * 2);
                }
            }
        }
        for t in &self.experts {
            add(Class::Experts, t.records() * t.record.checkpoint_bytes());
        }
        out.sort_by_key(|e| e.0);
        out
    }

    /// The plan as JSON for the builder.
    #[must_use]
    pub fn to_json(&self, c: &Qwen4ExpConfig) -> Value {
        let slice = |s: &Slice| {
            json!({"tensor": s.tensor, "shape": s.shape, "rows": [s.rows.0, s.rows.1],
                   "cols": [s.cols.0, s.cols.1]})
        };
        let dense: Vec<Value> = self
            .dense
            .iter()
            .map(|r| {
                json!({
                    "name": r.name, "class": r.class.key(), "placement": r.placement.dir(),
                    "kind": r.kind.key(), "identity": r.kind.identity(), "dtype": r.dtype,
                    "shape": r.shape, "part": r.part, "bytes": r.bytes,
                    "sources": r.sources.iter().map(slice).collect::<Vec<_>>(),
                })
            })
            .collect();
        let lay = |x: (u64, u64)| json!([x.0, x.1]);
        let experts: Vec<Value> = self
            .experts
            .iter()
            .map(|t| {
                json!({
                    "rank": t.rank, "first_expert": t.first_expert, "experts": t.experts,
                    "layers": t.layers, "records": t.records(),
                    "record": {
                        "gate_up_w": lay(t.record.gate_up_w), "gate_up_ws": lay(t.record.gate_up_ws),
                        "gate_up_wref": lay(t.record.gate_up_wref), "down_w": lay(t.record.down_w),
                        "down_ws": lay(t.record.down_ws), "down_wref": lay(t.record.down_wref),
                        "payload": t.record.payload, "stride": t.record.stride,
                        "checkpoint_bytes": t.record.checkpoint_bytes(),
                        "format": t.record.format,
                    },
                    "source": if self.source == PackSource::Fp4Local {
                        "model.language_model.layers.{layer}.mlp.experts.{expert}.{gate_proj,up_proj,down_proj}.{weight,weight_scale.nvfp4_csf_{fixed,exceptions},weight_scale_2}"
                    } else {
                        "model.language_model.layers.{layer}.mlp.experts.{expert}.{gate_proj,up_proj,down_proj}.weight[_scale]"
                    },
                })
            })
            .collect();
        let ledger: Vec<Value> = self
            .ledger()
            .iter()
            .map(|(p, cl, b)| json!({"placement": p, "class": cl.key(), "bytes": b}))
            .collect();
        let consumed: Vec<Value> = self
            .consumed_by_class()
            .iter()
            .map(|(cl, b)| json!({"class": cl.key(), "bytes": b}))
            .collect();
        json!({
            "model": "qwen4_exp",
            "source": self.source.key(),
            "layout": {"ep": EP, "tp": TP, "owner": "rank r owns experts [256r, 256r+256) of every layer"},
            "declared_defaults": c.declared_defaults().iter()
                .map(|d| json!({"key": d.key, "value": d.value, "source": d.source}))
                .collect::<Vec<_>>(),
            "dense": dense,
            "aliases": self.aliases.iter().map(|x| json!({
                "name": x.name, "placement": x.placement.dir(), "kind": "bf16_merged",
                "dtype": "BF16", "shape": x.shape,
                "sources": x.sources.iter().map(slice).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "experts": experts,
            "ple": {
                "rows": self.ple.rows, "rows_per_shard": self.ple.rows_per_shard,
                "shards": self.ple.shards, "row_dim": self.ple.row_dim,
                "rows_per_token": self.ple.rows_per_token,
                "row_bytes_bf16": self.ple.row_bytes_bf16(), "row_bytes_fp8": self.ple.row_bytes_fp8(),
                "row_bytes_nvfp4": self.ple.row_bytes_nvfp4(),
                "tensor_prefix": self.ple.tensor_prefix,
            },
            "ledger": ledger,
            "consumed_by_class": consumed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUARK: &str = include_str!("../../tests/data/qwen4_exp_quark.config.json");
    const FP4LOCAL: &str = include_str!("../../tests/data/qwen4_exp_fp4local.config.json");

    /// FN144: the FP4-Local plan -- same names per placement as Quark's but the fp8 pairs, every
    /// MXFP8 module carried as stored except the three small ones (bf16), the NVFP4 tier at 676
    /// pages, the MTP experts as NVFP4 trios, no alias; Quark's plan refuses to be built from it.
    #[test]
    fn fn144_fp4local_plan_is_lossless_and_geometry_unchanged() {
        let c = Qwen4ExpConfig::from_config_json(FP4LOCAL).unwrap();
        let p = PackPlan::fp4local(&c).unwrap();
        let q = PackPlan::ep2(&Qwen4ExpConfig::from_config_json(QUARK).unwrap());
        assert!(PackPlan::fp4local(&Qwen4ExpConfig::from_config_json(QUARK).unwrap()).is_err());
        assert_eq!(p.source, PackSource::Fp4Local);
        assert!(p.aliases.is_empty());
        let t = &p.experts[0];
        assert_eq!(t.record.stride, 676 * 4096);
        assert_eq!(t.record.format, "NVFP4_E4M3G16");
        assert_eq!(t.record.payload, 2_765_060);
        assert_eq!(t.record.gate_up_ws.1, 160 * 1280);
        assert_eq!((t.record.gate_up_wref.1, t.record.down_wref.1), (8, 4));
        assert_eq!(q.experts[0].record.stride, 639 * 4096);
        let kinds = |k: Fp4Local| {
            p.dense
                .iter()
                .filter(|r| r.kind == SourceKind::ModeloptFp4Local(k))
                .count()
        };
        // 36 GDN x (qkvz + out) x 2 ranks + 12 QSA x (q, k, v, o) x 2 + 48 shared x 2, each a
        // w / ws pair.
        assert_eq!(
            kinds(Fp4Local::Mxfp8),
            2 * (36 * 2 * 2 + 12 * 4 * 2 + 48 * 2)
        );
        // 36 in_proj_ba x 2 ranks + 12 index_qk_proj (shared).
        assert_eq!(kinds(Fp4Local::Mxfp8ToBf16), 36 * 2 + 12);
        assert_eq!(kinds(Fp4Local::Nvfp4Trio), 512 * 2 * 3);
        // 48 x 2 layer HC norms + the final mixer's + the PLE's three norms.
        assert_eq!(kinds(Fp4Local::F32Copy), 48 * 2 + 1 + 3);
        for r in &p.dense {
            if r.kind == SourceKind::ModeloptFp4Local(Fp4Local::Mxfp8) {
                assert!(r.kind.identity(), "{}", r.name);
                match r.part {
                    Some("w") => assert_eq!(r.dtype, "F8_E4M3"),
                    Some("ws") => {
                        assert_eq!(r.dtype, "UE8M0");
                        assert!(r
                            .sources
                            .iter()
                            .all(|s| s.tensor.ends_with(".weight_scale")));
                    }
                    other => panic!("{}: part {other:?}", r.name),
                }
            }
            // Nothing in the MTP attention / fc / shared expert is quantized (the ignore globs).
            if r.name.starts_with("mtp.") && r.class == Class::MtpDense {
                assert!(!r.kind.key().starts_with("fp4local"), "{}", r.name);
            }
        }
        // The fp8 out_proj is K-split on a 32 boundary: rank 1's scale columns [96, 192).
        let op = p
            .dense
            .iter()
            .find(|r| {
                r.name == "model.language_model.layers.0.linear_attn.out_proj.weight_scale"
                    && r.placement == Placement::Rank(1)
            })
            .unwrap();
        assert_eq!(op.sources[0].cols, (96, 192));
        assert_eq!(op.shape, vec![2560, 96]);
        // The rest of the record set is Quark's, name for name (bf16 records keep their names).
        let names = |pl: &PackPlan| {
            let mut v: Vec<String> = pl
                .dense
                .iter()
                .filter(|r| {
                    r.kind.key() == "bf16"
                        || r.kind.key() == "raw"
                        || r.kind.key() == "f32_from_bf16"
                })
                .map(|r| format!("{}:{}", r.placement.dir(), r.name))
                .collect();
            v.sort();
            v
        };
        let qn = names(&q);
        assert!(names(&p).iter().all(|n| qn.binary_search(n).is_ok()));
        let j = p.to_json(&c);
        assert_eq!(j["source"], "fp4_local");
        assert_eq!(j["ple"]["row_bytes_nvfp4"], 90);
    }

    fn plan() -> (Qwen4ExpConfig, PackPlan) {
        let c = Qwen4ExpConfig::from_config_json(QUARK).unwrap();
        let p = PackPlan::ep2(&c);
        (c, p)
    }

    #[test]
    fn expert_record_is_one_aligned_read() {
        let r = ExpertRecordLayout::new(2560, 640);
        assert_eq!(
            r.checkpoint_bytes(),
            2_611_200,
            "FN1's 2.6112 MB per expert-layer"
        );
        assert_eq!(r.wref_bytes(), 3_840);
        assert_eq!(r.payload, 2_615_040);
        assert_eq!(r.stride, 2_617_344);
        assert_eq!(r.stride % RECORD_ALIGN, 0);
        for s in [
            r.gate_up_w,
            r.gate_up_ws,
            r.gate_up_wref,
            r.down_w,
            r.down_ws,
            r.down_wref,
        ] {
            assert_eq!(s.0 % SUB_ALIGN, 0);
        }
        assert!(r.down_wref.0 + r.down_wref.1 <= r.stride);
    }

    #[test]
    fn ep_ownership_partitions_every_expert() {
        let (_, p) = plan();
        assert_eq!(p.experts.len(), 2);
        for l in 0..48 {
            for e in 0..512 {
                let owners: Vec<u32> = p
                    .experts
                    .iter()
                    .filter(|t| t.index(l, e).is_some())
                    .map(|t| t.rank)
                    .collect();
                assert_eq!(owners, vec![e / 256], "layer {l} expert {e}");
            }
        }
        assert_eq!(p.experts[1].index(47, 511), Some(47 * 256 + 255));
        assert_eq!(p.experts[0].index(0, 256), None);
    }

    #[test]
    fn ple_tier_locates_every_row() {
        let (_, p) = plan();
        assert_eq!(
            (p.ple.shards, p.ple.rows_per_shard, p.ple.row_dim),
            (128, 2_500_012, 160)
        );
        assert_eq!(p.ple.row_bytes_bf16(), 320);
        assert_eq!(p.ple.locate(0), Some((0, 0)));
        assert_eq!(p.ple.locate(2_500_012), Some((1, 0)));
        assert_eq!(p.ple.locate(p.ple.rows), None);
        assert_eq!(p.ple.locate(p.ple.rows - 1).unwrap().0, 127);
    }

    #[test]
    fn names_are_unique_per_placement() {
        let (_, p) = plan();
        let mut seen = std::collections::BTreeSet::new();
        for r in &p.dense {
            assert!(
                seen.insert((r.placement, r.name.clone())),
                "{} twice",
                r.name
            );
        }
    }

    /// FN1's category totals (budget.txt, first line, GiB to 4 places) — and exact bytes from the
    /// checkpoint shapes — must be the plan's consumed bytes per class.
    #[test]
    fn consumed_bytes_match_fn1_categories() {
        let (_, p) = plan();
        let got: std::collections::BTreeMap<&str, u64> = p
            .consumed_by_class()
            .iter()
            .map(|(c, b)| (c.key(), *b))
            .collect();
        let gib = |b: u64| (b as f64 / f64::from(1u32 << 30) * 1e4).round() / 1e4;
        // FN1: experts 59.7656, gdn 3.8864, attn.qo 1.0547, attn.kv 0.0586, hc 1.1933,
        // embed 1.1841, lm_head 1.1841, router 0.1174, shared 0.1167, ple.dense 0.0612.
        assert_eq!(gib(got["experts"]), 59.7656);
        assert_eq!(gib(got["gdn"]), 3.8864);
        assert_eq!(gib(got["attn.qo"]), 1.0547);
        assert_eq!(gib(got["attn.kv"]), 0.0586);
        assert_eq!(gib(got["hc"]), 1.1933);
        assert_eq!(gib(got["embed"]), 1.1841);
        assert_eq!(gib(got["lm_head"]), 1.1841);
        assert_eq!(gib(got["router"]), 0.1174);
        assert_eq!(gib(got["shared_expert"]), 0.1167);
        assert_eq!(gib(got["ple.dense"]), 0.0612);
        // FN1's indexer class also holds the attention q/k norms (budget.py), as here.
        assert_eq!(gib(got["indexer"]), 0.0366);
        // mtp.experts 4.6875 + mtp.{tp .1215, hc .0368, attn.kv .0049, small .0031, router .0024}
        assert_eq!(gib(got["mtp.experts"]), 4.6875);
        assert_eq!(gib(got["mtp.dense"]), 0.1687);
    }

    #[test]
    fn the_qsa_merged_projection_is_an_alias_of_n_7296_per_rank() {
        let (c, p) = plan();
        // 12 target QSA layers + the MTP layer, one alias per rank each.
        assert_eq!(p.aliases.len(), 13 * 2);
        for a in &p.aliases {
            assert_eq!(a.shape, vec![7_296, 2_560], "{}", a.name);
            let rows: u64 = a.sources.iter().map(|s| s.rows.1 - s.rows.0).sum();
            assert_eq!(rows, 7_296);
            let r = match a.placement {
                Placement::Rank(r) => u64::from(r),
                other => panic!("{other:?}"),
            };
            assert_eq!(
                a.sources[0].rows,
                (6_144 * r, 6_144 * (r + 1)),
                "q|gate half"
            );
            assert_eq!(a.sources[1].rows, (256 * r, 256 * (r + 1)), "k half");
            assert_eq!(a.sources[2].rows, (256 * r, 256 * (r + 1)), "v half");
            assert_eq!(a.sources[3].rows, (0, 640), "the indexer, replicated whole");
            // every member names bytes a dense record already owns: no new byte, no ledger line
            for s in &a.sources {
                assert!(
                    p.dense
                        .iter()
                        .any(|d| d.sources.iter().any(|x| x.tensor == s.tensor
                            && x.rows.0 <= s.rows.0
                            && s.rows.1 <= x.rows.1)),
                    "{} is not inside a dense record",
                    s.tensor
                );
            }
        }
        assert_eq!(
            u64::from(saber_kernels::fn6::QSA_PROJ_N_TP2),
            p.aliases[0].shape[0],
            "== FN6's declared N"
        );
        let j = p.to_json(&c);
        assert_eq!(j["aliases"].as_array().unwrap().len(), 26);
        let ledger_before: u64 = p.ledger().iter().map(|x| x.2).sum();
        let mut q = p.clone();
        q.aliases.clear();
        assert_eq!(
            q.ledger().iter().map(|x| x.2).sum::<u64>(),
            ledger_before,
            "aliases add no ledger bytes"
        );
    }
}
