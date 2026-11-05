//! The Flash-Next layer executor (FN12): `layer_graph`'s op list as launches.
//!
//! Every op is the FN9 / FN9b harness's (`tools/capture/fn9/fn9_layers.py`, the bar gate A holds
//! this to byte for byte): the same kernels, the same grids, the same arguments in the same order;
//! the HC mixes take FN11's v2 routes through `FlashNextRules::mix_plan` (byte-identical to FN5's
//! symbols of the same `sps`, FN11's kernel-vs-kernel oracle); the bf16 GEMMs FN5's routes
//! (`bgemm_route`); the MoE block FN4's train + FN9's shared-gate GEMV (7 launches) at
//! `moe::measured_cell`; the GDN core FN9b's exact-gate scan; the attention FN6's prefill body.
//!
//! What the harness allocated fresh and zeroed per op, the executor takes from the rank's arena
//! under the SEED LAW ([`Seed`]): counters, route tables, the `-1` row map and the per-sequence
//! states start seeded; everything else is scratch that its own op writes before anything reads it
//! (the poison arm, `--fn-poison`, proves it on the card: a 0xA5-filled arena gives the same bytes).

use anyhow::{ensure, Context};
use saber_kernels::flashnext::{FlashNextRules, MixMode, MixOut};
use saber_kernels::moe::{self, MoeGemm};
use saber_model::qwen4_exp::layer_graph::Group;
pub use saber_model::qwen4_exp::layer_graph::{moe_block_front, moe_block_fused, MoeTrain};

use super::kern::{Arena, Arg, Bump, Decls, Rec, Seed};

pub const H: u64 = 2560;
pub const W: u64 = 10240;
pub const HD: u64 = 256;
pub const EPS: f32 = 1e-6;
pub const VOCAB: u64 = 248_320;
/// The indexer's raw-key ring rows per sequence per layer at spec 0..4 (EXEC §13: `4 ceil((4+spec)/4)`).
pub const RING: u64 = 8;
/// The PLE conv history rows (`9 + spec` at MTP-4, EXEC §13).
pub const PLE_HIST: u64 = 13;
/// FN15b: the verify-width law's maximum (MTP-4: the committed token + 4 drafts; fn6's GQA-12 cap).
pub const WMAX: u32 = 5;

/// FN24: which kernel CLASS a verify step's rows run in (`--fn-verify-class`).
///
/// `Width1` (the default): every M-keyed cell of a width-`w` verify step of `b` sequences (M = b x w
/// rows) is the cell a SPEC-0 step of the same `b` runs (M = b) -- the bf16 GEMM band + split-K,
/// the HC mix band (`sps`), the MXFP4 expert cell (split-K), the GDN decode form (fused / pair) --
/// and the GDN recurrent state carried across the window is rounded to f16 after every row (the
/// `fn_gdn_*_fc_*` bodies, `flashnext_gdn_fc`): exactly SPEC 0's store + reload between tokens. So a
/// verify row's bytes are SPEC 0's bytes for the same token (every cell is row-independent at a
/// fixed class: a WMMA row, a split-K partial order, a per-row mix / norm), and MTP-k's greedy ids
/// ARE SPEC 0's. The one M-keyed law it does not move is the all-reduce's exact / wht6 band (a
/// payload law; FN78: `--fn-decode-ar exact` keeps every decode form exact). `Wide` = FN15b's routing (each cell at M = b x w, the fp32
/// carry): the pre-FN24 control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VerifyClass {
    #[default]
    Width1,
    Wide,
}

impl std::str::FromStr for VerifyClass {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim() {
            "width1" => Ok(VerifyClass::Width1),
            "wide" => Ok(VerifyClass::Wide),
            t => Err(format!("--fn-verify-class width1 | wide, not {t}")),
        }
    }
}

impl std::fmt::Display for VerifyClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            VerifyClass::Width1 => "width1",
            VerifyClass::Wide => "wide",
        })
    }
}

/// FN24: the f16-carry GDN decode bodies (`flashnext_gdn_fc`): the shipped fused / recurrent update
/// with the carried state rounded to f16 after each row (== SPEC 0's store + reload); at one row a
/// sequence they compute the shipped bodies' bytes.
pub const GDN_FUSED_FC: &str = "fn_gdn_fused_update_fc_f16s_w4k128v128_ab1";
pub const GDN_RECURRENT_FC: &str = "fn_gdn_recurrent_update_fc_f16s_k128_v128_ab1_norm0_vs1";

/// FN134 / FN138: the dims the slab4 MTP layer's MoE runs at -- [`Dims::mtp_form`] (EP split,
/// whole shape) for the EP slab, the target's own tp2 dims for the half slab (refused unless the
/// target IS at the tp2 form: the half tables have 513 entries).
pub fn mtp_x4_dims(lw: &LayerW, target: Dims) -> anyhow::Result<Dims> {
    if lw.mtp_xtp2 {
        ensure!(
            target.xtp2 && target.eloc == 512 && target.shared,
            "--fn-mtp-split tp2: the MTP half slab runs at the target's tp2 dims ({target:?})"
        );
        Ok(target)
    } else {
        Ok(target.mtp_form())
    }
}

/// Per-rank dense dimensions (TP=1 or the TP=2 shard).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dims {
    pub tp: u32,
    pub rank: u32,
    pub hk: u64,
    pub hv: u64,
    pub conv: u64,
    pub zn: u64,
    pub gn: u64,
    pub nq: u64,
    pub nkv: u64,
    pub qn: u64,
    pub v_off: u64,
    pub idx_off: u64,
    pub eloc: u32,
    pub e_lo: u32,
    pub shared: bool,
    /// FN80b-2 (`--fn-experts tp2`): the TARGET's routed experts are TP-split -- every rank holds
    /// half `rank` of EVERY expert's intermediate dimension (`rank<r>/experts.tp2.xtier`) and of
    /// the shared expert, so the route keeps all 512 experts local (`eloc` 512, `e_lo` 0, the
    /// shared rows on both ranks) and the expert GEMMs run at half width (gate_up N 640, down K
    /// 320). The MTP layer's experts STAY EP ([`Dims::mtp_experts`]).
    pub xtp2: bool,
    /// FN150 (`--fn-experts-fmt nvfp4`): the TARGET's routed-expert records are the FP4-Local v2
    /// NVFP4 form (`xptr[3e..3e+3]` = `w` / `ws` E4M3 g16 / `gs` f32), so the seven-launch train's
    /// grouped GEMMs run FN148's form (b) cells (`moe::nvfp4_symbol`, + `gsplit`); every other
    /// expert body (fused back half, `far`, prefill tiles) refuses by name. False for the MTP
    /// layer ([`Dims::mtp_form`]).
    /// FN152: the grouped cells are FN151's `_p1` twins (`moe::Nvfp4Body::P1`) and the fused back
    /// half is FN152's (`moe::nvfp4_back_symbol`); `xnv_plain` = FN150's fn148-only route (control).
    pub xnv: bool,
    /// FN152: FN150's route (FN148's plain cells, no fused back half) -- the byte control.
    pub xnv_plain: bool,
}

impl Dims {
    #[must_use]
    pub fn new(tp: u32, rank: u32) -> Dims {
        let t = u64::from(tp);
        let (hk, hv, conv, zn) = (16 / t, 48 / t, 10240 / t, 6144 / t);
        let (nq, nkv) = (24 / t, 2 / t);
        let qn = nq * 2 * HD + 2 * nkv * HD + 640;
        Dims {
            tp,
            rank,
            hk,
            hv,
            conv,
            zn,
            gn: conv + zn + 2 * hv,
            nq,
            nkv,
            qn,
            v_off: nq * 2 * HD + nkv * HD,
            idx_off: nq * 2 * HD + 2 * nkv * HD,
            eloc: 512 / tp,
            e_lo: rank * (512 / tp),
            shared: rank == 0,
            xtp2: false,
            xnv: false,
            xnv_plain: false,
        }
    }
    /// FN150: the dims of the NVFP4 expert records (`--fn-experts-fmt nvfp4`).
    #[must_use]
    pub fn experts_nvfp4(self) -> Dims {
        Dims { xnv: true, ..self }
    }
    /// FN152: [`Self::experts_nvfp4`] on FN150's route (FN148's plain cells only): the control.
    #[must_use]
    pub fn experts_nvfp4_plain(self) -> Dims {
        Dims {
            xnv: true,
            xnv_plain: true,
            ..self
        }
    }
    /// FN80b-2: the dims of the TP-split expert form (`--fn-experts tp2`; TP=2 only): every
    /// routed expert local on both ranks, the shared expert's half on both ranks.
    #[must_use]
    pub fn experts_tp2(self) -> Dims {
        assert_eq!(self.tp, 2, "--fn-experts tp2 is a TP=2 form");
        Dims {
            eloc: 512,
            e_lo: 0,
            shared: true,
            xtp2: true,
            ..self
        }
    }
    /// The MTP layer's routed experts (always EP=2 at TP=2: FLASHNEXT-PLAN §13 item 5):
    /// `(eloc, e_lo, shared)`.
    #[must_use]
    pub fn mtp_experts(&self) -> (u32, u32, bool) {
        let t = self.tp;
        (512 / t, self.rank * (512 / t), self.rank == 0)
    }
    /// FN134: the dims the MTP layer's routed experts run at when they ride the TRUNK's train
    /// (`--fn-mtp-experts slab4`: the seven-launch W4A8 route + grouped GEMMs over the slab's EP
    /// tables): [`Self::mtp_experts`]'s EP split and the whole-expert shape, whatever the target's
    /// expert form. Equal to `self` under EP (and at TP=1) by construction, so the EP slab4 path
    /// is byte for byte today's; under `--fn-experts tp2` it undoes [`Self::experts_tp2`] for the
    /// MTP layer only (the slab holds EP's records: 257 entries, the shared expert on rank 0).
    #[must_use]
    pub fn mtp_form(self) -> Dims {
        let (eloc, e_lo, shared) = self.mtp_experts();
        Dims {
            eloc,
            e_lo,
            shared,
            xtp2: false,
            xnv: false,
            xnv_plain: false,
            ..self
        }
    }
    /// FN80b-1's shape key of the target's routed-expert cells (`TpHalf` under `--fn-experts tp2`).
    #[must_use]
    pub fn xshape(&self) -> moe::ExpertShape {
        if self.xtp2 {
            moe::ExpertShape::TpHalf
        } else {
            moe::ExpertShape::Whole
        }
    }
    /// The target's expert intermediate width a rank computes (640 whole, 320 under TP split).
    #[must_use]
    pub fn x_inter(&self) -> u64 {
        u64::from(self.xshape().inter())
    }
    #[must_use]
    pub fn tps(&self) -> &'static str {
        if self.tp == 1 {
            "tp1"
        } else {
            "tp2"
        }
    }
}

/// One HC mix's weights.
#[derive(Debug, Clone, Copy, Default)]
pub struct HcW {
    /// FN108: `wdown` / `wup` are the e4m3 records (codes + f32 row-group scales) and the mix
    /// launches the `_h8` twins (`flashnext_hc8`).
    pub w8: bool,
    pub w: u64,
    pub wdown: u64,
    pub wup: u64,
    /// 0 for the final mixer (no injection rows).
    pub winj: u64,
}

/// The mixer of a layer.
#[derive(Debug, Clone, Copy)]
pub enum MixerW {
    Gdn {
        win: u64,
        conv: u64,
        alog: u64,
        dtb: u64,
        nw: u64,
        wout: u64,
    },
    Qsa {
        wproj: u64,
        wo: u64,
        qw: u64,
        kw: u64,
        iqw: u64,
        ikw: u64,
    },
}

/// Layer 1's PLE weights.
#[derive(Debug, Clone, Copy, Default)]
pub struct PleW {
    pub pkey: u64,
    pub pval: u64,
    pub pk: u64,
    pub pq: u64,
    pub pc: u64,
    pub pcw: u64,
}

/// FN61c: one target layer's fp8 W8 trunk records (`--fn-trunk w8`, the `-trunk8` pack variant):
/// the merged projection's W8 PREFIX (GDN `in_proj_qkvz` 8,192 rows / QSA q|gate + k + v 6,656
/// rows, preshuffled e4m3 + fp32 128 x 128 block scales, one upload) and the out / o projection.
/// The merged rows' bf16 SUFFIX (`in_proj_ba` 48, the indexer 640) stays in the bf16 record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct T8W {
    /// The merged projection's W8 prefix: weight, scales, rows.
    pub w: u64,
    pub s: u64,
    pub n: u64,
    /// The out_proj / o_proj: weight, scales.
    pub wo: u64,
    pub so: u64,
    /// FN61e: the merged projection's bf16 SUFFIX rows (in_proj_ba / the indexer) as their own
    /// upload when the bf16 trunk records are DROPPED (`--fn-trunk w8`: every form reads W8, the
    /// mixer's `win` / `wproj` / `wout` / `wo` are 0); 0 = FN61d's `w8-decode` (the suffix is read
    /// in the resident merged bf16 record).
    pub sfx: u64,
}

/// One decoder layer's device addresses.
#[derive(Debug, Clone, Copy)]
pub struct LayerW {
    pub l: u32,
    pub ahc: HcW,
    pub mhc: HcW,
    pub mixer: MixerW,
    pub router: u64,
    pub sg: u64,
    pub ple: Option<PleW>,
    /// The xptr tables (gate_up, down) this chunk's grouped GEMMs read.
    pub xgu: u64,
    pub xdn: u64,
    /// FN15: the MTP layer (bf16 routed + shared experts on FN6's grouped bf16 band, 16-row
    /// tiles; the mlp mix hands a bf16 block input). False for every target layer.
    pub mtp: bool,
    /// FN50: the MTP layer's routed experts are Qwen's fp8 (`xptr[2e + 1]` = their fp32 block
    /// scales; the shared expert's entry stays `[bf16, 0]`): the mixed-table train
    /// ([`Emitter::moe_experts_fp8`]). False for every target layer and the bf16 default.
    pub mtp_fp8: bool,
    /// FN110: the MTP layer's routed experts + shared expert are MXFP4 records in the expert
    /// tier's NATIVE form (`--fn-mtp-experts slab4`; the pack's additional `rank<r>/mtp_x4.*`):
    /// the mlp mix hands the TARGET's Q8 block input and the MoE block is the trunk's seven-launch
    /// train (route + FN4's grouped W4A8 GEMMs, `xptr[3e..3e+3]` = w / ws / wref). False for every
    /// target layer, the bf16 control and the fp8 default.
    pub mtp_x4: bool,
    /// FN138 (`--fn-mtp-split tp2`, under `--fn-experts tp2` + slab4): the slab holds this rank's
    /// HALF of every MTP expert (FN80a's split, sliced at load from both ranks' `mtp_x4` files;
    /// 513 entries, the shared half at 512 on both ranks) and the MTP MoE runs at the TARGET's
    /// tp2 dims (all 512 experts local, the TpHalf cells) instead of [`Dims::mtp_form`]. False for
    /// every target layer and the EP split (the control).
    pub mtp_xtp2: bool,
    /// FN61c: the fp8 W8 trunk records (`--fn-trunk w8`; None = bf16, the default and the MTP
    /// layer). Read by DECODE forms only (decode, verify): prefill keeps the bf16 records.
    pub t8: Option<T8W>,
}

/// FN15: the MTP drafter's device weights (upstream `Qwen4ExpMultiTokenPredictor`, LEDGER FN15).
#[derive(Debug, Clone, Copy)]
pub struct MtpW {
    /// `pre_fc_norm_embedding` (2,560) and `pre_fc_norm_hidden` (ONE norm over 10,240).
    pub ne: u64,
    pub nh: u64,
    /// `fc_embedding` / `fc_hidden`, both column halves on every rank (`[2560, 2560]`).
    pub fce: u64,
    pub fch: u64,
    /// The MTP decoder layer (a QSA layer, `mtp: true`; xgu / xdn = the resident bf16 tables).
    pub lw: LayerW,
    /// `mtp.hyper_connection_mixer` (no inject rows).
    pub fm: HcW,
}

/// A sequence's per-layer persistent state (the chunk-boundary carriers).
#[derive(Debug, Clone, Copy, Default)]
pub struct LayerState {
    /// GDN conv state `[1, conv, 3]` bf16 (updated in place by the conv prep).
    pub cst: u64,
    /// GDN recurrent state ping-pong `[1, hv, 128, 128]` f16: chunk c reads `h[c % 2]`, writes the other.
    pub h: [u64; 2],
    /// QSA KV pages `[nb, nkv, 16, 512]` bf16.
    pub kv: u64,
    /// Indexer raw-key ring `[RING, 128]` bf16 + positions `[RING, 3]` i64.
    pub ring: u64,
    pub ringp: u64,
    /// Indexer compressed store `[nb * 4, 128]` bf16.
    pub comp: u64,
    /// PLE conv history ping-pong `[1, 13, W]` bf16.
    pub ple_st: [u64; 2],
}

/// The per-step constant tables (uploaded once per chunk / decode step, shared by every layer).
///
/// A PREFILL chunk is one sequence (index 0 of its own state bases, the FN12 tables). A DECODE
/// step (FN13) is `b` sequences of one row each; sequence `s` of the batch is POOL SLOT `s` of every
/// per-sequence pool (conv / GDN state at slot `s + 1` -- slot 0 is the kernels' null slot --, KV
/// pages `s * nb ..`, indexer ring rows `s * RING ..`, PLE history `s`), so every per-sequence table
/// below has `b` entries.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChunkConsts {
    /// Rows in the chunk / step.
    pub m: u32,
    /// The chunk's first absolute position (prefill).
    pub p0: u32,
    /// The chunk's index in its sequence (0 = the first; it picks the ping-pong parity).
    pub chunk: u32,
    /// A decode step (the decode-form ops).
    pub decode: bool,
    /// Sequences in the step (1 for a prefill chunk).
    pub b: u32,
    /// i64 `[3, m]` mrope positions (text: three equal rows).
    pub pos3: u64,
    /// i64 `[m]` absolute positions.
    pub pos: u64,
    /// i64 `[m]` the KV slot of each row (prefill: == pos at the identity block table).
    pub slot: u64,
    /// i32 `[m]` row -> sequence.
    pub zeros: u64,
    /// FN17: i32 `[m]` the PLE conv's row -> sequence (== `zeros` but for pad rows, which form ONE
    /// pseudo-sequence at the pad slot: the PLE history is indexed by the row's sequence).
    pub ple_seq: u64,
    /// i32 `[b + 1]` cumulative rows.
    pub cu: u64,
    /// i32 `[1]` = 0 (prefill's conv cache index).
    pub zero1: u64,
    /// i32 `[b]` each sequence's first row.
    pub seq_row0: u64,
    /// i32 `[b]` each sequence's indexer ring block.
    pub ring_row0: u64,
    /// i32 `[b]` the PLE history's dropped rows (0: SPEC 0).
    pub drop: u64,
    /// i32 `[b]` decode: the conv-state slot per sequence (s + 1).
    pub cidx: u64,
    /// i32 `[b]` decode: the GDN-state slot per sequence (s + 1).
    pub sidx: u64,
    /// i64 `[b]` each sequence's first position of this step.
    pub p0v: u64,
    /// u8 `[1]` has_init for the conv prep (chunk > 0).
    pub hi8: u64,
    /// i32 `[b]` has_init for the PLE conv.
    pub hi32: u64,
    /// i32 `[b]` rows per sequence (the PLE conv's seq_rows).
    pub mv: u64,
    /// i32 `[b]` each sequence's context after this step (the attention's seqused_k).
    pub ctx: u64,
    /// i64 `[m]` ring slot per row (-1 = not stored).
    pub rslot: u64,
    /// i32 `[b, nb]` block table.
    pub bt: u64,
    /// FN24: a verify step's attention as `b x w` pseudo-sequences of ONE row (the width-1 class):
    /// i32 `[m]` each row's own context (`position + 1`: SPEC 0's `seqused_k` for that token) and
    /// i32 `[m, nb]` each row's sequence's block table. == `ctx` / `bt` at width 1.
    pub ctx_rows: u64,
    pub bt_rows: u64,
    pub nb: u32,
    /// FN51f: the row length of the step's `bt` (the index store / score table): `nb`, or a
    /// decode-form step's band ([`super::kvwire::band_nb`]) on a long boot.
    pub bt_nb: u32,
    /// The bf16 cos/sin cache.
    pub cs: u64,
    /// Decode: the attention's segmentation context (the split law's `max_ctx`).
    pub max_ctx: u32,
    /// Decode: the fused update's self-resetting grid barrier (one i32, zero at rest).
    pub bar: u64,
    /// FN15b: rows per sequence of a decode step (the verify width; 1 at SPEC 0 and prefill).
    pub w: u32,
    /// FN15b: i32 `[b]` tokens each sequence committed in the PREVIOUS step (the GDN snapshot and
    /// the conv window the step starts from: `sidx[s * pw + naccept - 1]`); 0 = the null table.
    pub naccept: u64,
    /// FN16: the GDN snapshot table's stride (`pool_w`: every snapshot of a sequence is in the
    /// table, so a step NARROWER than the last one's commit resumes from the right snapshot; 1 at
    /// SPEC 0, where it equals FN15b's `w`).
    pub pw: u32,
    /// FN19: a decode step whose rows all sit at positions <= 2,050 (`fn19::short_rows`): the
    /// dense decode (== the gather over identity lists, byte for byte), no selection launches.
    pub qsa_short: bool,
    /// FN45b2 (the tiered KV): i64 `[13, m]` each QSA layer's append slot (0 = the flat pools:
    /// `slot` / the identity table), i32 `[13, m, kvwire::KBTR]` each layer's per-row table of a
    /// dense verify step (0 = none), and the byte offset of the chunk's sequence row in a layer's
    /// page table (a prefill chunk: its pool slot's row; a decode step: 0, rows = pool slots).
    pub kslot: u64,
    pub kbtr: u64,
    pub kv_row0: u64,
}

pub(super) fn put_table(
    b: &mut Vec<u8>,
    offs: &mut Vec<(&'static str, usize)>,
    name: &'static str,
    v: &[u8],
) {
    while b.len() % 256 != 0 {
        b.push(0);
    }
    offs.push((name, b.len()));
    b.extend_from_slice(v);
}

fn le64(v: &[i64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 8);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

/// FN51e: sized up front (a 262,144-token boot's decode tables are 16,385 entries a row: the
/// `flat_map` collect re-grew a 2 MiB vector every b8 verify step).
pub(super) fn le32(v: &[i32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

/// FN51e: the block table of `slots` (row `r` = slot `slots[r]`'s identity blocks `0..nb`).
#[cfg(test)]
fn block_rows(slots: &[i32], nb: u32) -> Vec<i32> {
    let mut v = Vec::with_capacity(slots.len() * nb as usize);
    for &i in slots {
        v.extend((0..nb as i32).map(|j| i * nb as i32 + j));
    }
    v
}

/// FN51e: [`put_table`] of `le32(&block_rows(slots, nb))`, written in place (no intermediate
/// vectors: 2 MiB a b8 verify step at a 262,144-token boot's 16,385 entries a row).
fn put_block_rows(
    b: &mut Vec<u8>,
    offs: &mut Vec<(&'static str, usize)>,
    name: &'static str,
    slots: &[i32],
    nb: u32,
    row: u32,
) {
    put_table(b, offs, name, &[]);
    b.reserve(slots.len() * row as usize * 4);
    for &i in slots {
        let base = i * nb as i32;
        for j in 0..row as i32 {
            b.extend_from_slice(&(base + j).to_le_bytes());
        }
    }
}

/// FN17: the PAD LAW of a decode step -- which pool slot each batch sequence reads and writes.
/// Batch sequences `0..live` are the live ones (pool slots `0..live`, the ready set); a PAD
/// sequence (`live..b`, the bucket's padding) goes to the `pad` slot when set (the pools' SPARE
/// slot). FN14's pad row `i` read and wrote pool slot `i`, which may hold a multi-chunk prompt still
/// prefilling between its chunks; `pad: None` is that pre-FN17 law, the gate's control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PadLaw {
    /// Live sequences (the first `live` of the batch).
    pub live: usize,
    /// The pad slot (`None` = the pre-FN17 control: pad `i` at slot `i`).
    pub pad: Option<u32>,
    /// FN96: the pools' index law -- (conv slots, GDN slots) a sequence slot's region holds under
    /// `--fn-pools-layout seq` (`cidx` = `1 + slot x conv`, `sidx` = `1 + slot x gdn + t`);
    /// `None` = FN13's (`slot + 1`, `1 + slot x pool_w + t`).
    pub idx: Option<(u32, u32)>,
}

impl PadLaw {
    /// Every batch sequence live (no pads: the unit tests' and FN16's form).
    #[cfg(test)]
    #[must_use]
    pub fn every_live(n: usize) -> PadLaw {
        PadLaw {
            live: n,
            pad: None,
            idx: None,
        }
    }
    /// The pool slot of batch sequence `s`.
    #[must_use]
    pub fn slot(&self, s: usize) -> u32 {
        match self.pad {
            Some(p) if s >= self.live => p,
            _ => s as u32,
        }
    }
    /// The table index of the pad pseudo-sequence (PLE) whenever a pad slot is set -- with or
    /// without pads in THIS step: a captured graph bakes every table's address, so the layout of
    /// a (bucket, width) must not depend on how many of its rows are live (hold 1 found it: a b4
    /// graph captured on a padded step read `ple_seq` from a stale upload on an unpadded one).
    #[must_use]
    pub fn pad_entry(&self) -> Option<usize> {
        self.pad.map(|p| p as usize)
    }
}

impl ChunkConsts {
    /// The host bytes of every table of a PREFILL chunk (one sequence), and each table's offset.
    #[must_use]
    pub fn host(m: u32, p0: u32, chunk: u32, nb: u32) -> (Vec<u8>, Vec<(&'static str, usize)>) {
        let mut b: Vec<u8> = Vec::new();
        let mut offs = Vec::new();
        let pos: Vec<i64> = (0..m).map(|i| i64::from(p0 + i)).collect();
        let mut p3 = pos.clone();
        p3.extend_from_slice(&pos);
        p3.extend_from_slice(&pos);
        put_table(&mut b, &mut offs, "pos3", &le64(&p3));
        put_table(&mut b, &mut offs, "pos", &le64(&pos));
        put_table(&mut b, &mut offs, "zeros", &vec![0u8; 4 * m as usize]);
        put_table(&mut b, &mut offs, "cu", &le32(&[0, m as i32]));
        put_table(&mut b, &mut offs, "zero1", &[0u8; 8]);
        put_table(&mut b, &mut offs, "p0v", &le64(&[i64::from(p0)]));
        put_table(&mut b, &mut offs, "hi8", &[u8::from(chunk > 0), 0, 0, 0]);
        put_table(&mut b, &mut offs, "hi32", &le32(&[i32::from(chunk > 0)]));
        put_table(&mut b, &mut offs, "mv", &le32(&[m as i32]));
        put_table(&mut b, &mut offs, "ctx", &le32(&[(p0 + m) as i32]));
        let rslot: Vec<i64> = pos
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                if i as u64 + RING >= u64::from(m) {
                    p % RING as i64
                } else {
                    -1
                }
            })
            .collect();
        put_table(&mut b, &mut offs, "rslot", &le64(&rslot));
        put_table(
            &mut b,
            &mut offs,
            "bt",
            &le32(&(0..nb as i32).collect::<Vec<i32>>()),
        );
        (b, offs)
    }

    /// The host bytes of a DECODE step: sequence `s` at position `pos[s]`, one row each, pool slot
    /// `s`; `nb` blocks per sequence (SPEC 0: [`ChunkConsts::host_verify`] at width 1, every
    /// sequence having committed one token, GDN snapshot slot `s + 1`).
    #[cfg(test)]
    #[must_use]
    pub fn host_decode(pos: &[u32], nb: u32) -> (Vec<u8>, Vec<(&'static str, usize)>) {
        let n = pos.len();
        Self::host_verify(pos, 1, &vec![1; n], &vec![0; n], nb, 1)
    }

    /// FN15b: the host bytes of a VERIFY step: sequence `s` (pool slot `s`) feeds `w` rows at
    /// positions `pos[s] .. pos[s] + w` (the committed token + `w - 1` drafts). Row `r = s * w + t`:
    /// KV slot `s * nb * 16 + pos[s] + t`, GDN snapshot slot `1 + s * pool_w + t` (the pools hold
    /// `pool_w >= w` snapshots per sequence; the step starts from `naccept[s] - 1`), indexer ring
    /// slot `s * RING + (pos[s] + t) % RING`; per sequence: conv slot `s + 1`, `naccept[s]` (the
    /// tokens the previous step committed), `drop[s]` (its rejected rows, the PLE history's),
    /// context after the step `pos[s] + w`. `w = 1`, `pool_w = 1`, `naccept = 1`, `drop = 0` is
    /// FN13's SPEC-0 table byte for byte (plus the all-ones `naccept`, the kernels' null).
    #[cfg(test)]
    #[must_use]
    pub fn host_verify(
        pos: &[u32],
        w: u32,
        naccept: &[u32],
        drop: &[u32],
        nb: u32,
        pool_w: u32,
    ) -> (Vec<u8>, Vec<(&'static str, usize)>) {
        Self::host_verify_at(
            pos,
            w,
            naccept,
            drop,
            nb,
            pool_w,
            PadLaw::every_live(pos.len()),
        )
    }

    /// FN17: [`ChunkConsts::host_verify`] under a PAD LAW: batch sequence `s` is pool slot
    /// `pads.slot(s)` in EVERY per-sequence table (KV slot, ring block, ring slot, block table, conv
    /// slot, GDN snapshots) and in the PLE history's row -> sequence table (`ple_seq`: the pad rows
    /// form ONE pseudo-sequence at the pad slot -- the PLE conv indexes its history by the row's
    /// sequence, so the pseudo-sequence's `seq_row0` / `mv` / `drop` / `hi32` entries sit at index
    /// `pad slot` of the same tables, past the batch's `b` entries every other kernel reads). The
    /// live rows' tables are FN16's byte for byte.
    #[must_use]
    pub fn host_verify_at(
        pos: &[u32],
        w: u32,
        naccept: &[u32],
        drop: &[u32],
        nb: u32,
        pool_w: u32,
        pads: PadLaw,
    ) -> (Vec<u8>, Vec<(&'static str, usize)>) {
        Self::host_verify_band(pos, w, naccept, drop, nb, nb, true, pool_w, pads)
    }

    /// FN51f: [`ChunkConsts::host_verify_at`] with `bt` rows of `bt_nb` entries (the band's row
    /// length; each row still the identity from `slot * nb`) and, without `rows_table` (the tiered
    /// KV: its per-row K/V table is `kbtr`, nothing reads `bt_rows`), no `bt_rows` -- the long
    /// boot's step blob shrinks to the 10,240 boot's at the 10,240 band.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn host_verify_band(
        pos: &[u32],
        w: u32,
        naccept: &[u32],
        drop: &[u32],
        nb: u32,
        bt_nb: u32,
        rows_table: bool,
        pool_w: u32,
        pads: PadLaw,
    ) -> (Vec<u8>, Vec<(&'static str, usize)>) {
        let n = pos.len();
        let wu = w as usize;
        // FN51e: the blob's block tables dominate it at a long boot (no re-growth copies)
        let mut b: Vec<u8> = Vec::with_capacity(n * (wu + 1) * bt_nb as usize * 4 + (64 << 10));
        let mut offs = Vec::new();
        let live = pads.live.min(n);
        let p: Vec<i64> = (0..n * wu)
            .map(|r| i64::from(pos[r / wu]) + (r % wu) as i64)
            .collect();
        let row_seq: Vec<i32> = (0..(n * wu) as i32).map(|r| r / w as i32).collect();
        let s: Vec<i32> = (0..n as i32).collect();
        // the pool slot of each batch sequence (the pad law)
        let ps: Vec<i32> = (0..n).map(|i| pads.slot(i) as i32).collect();
        let row_slot: Vec<i32> = row_seq.iter().map(|&q| ps[q as usize]).collect();
        let mut p3 = p.clone();
        p3.extend_from_slice(&p);
        p3.extend_from_slice(&p);
        put_table(&mut b, &mut offs, "pos3", &le64(&p3));
        put_table(&mut b, &mut offs, "pos", &le64(&p));
        let slot: Vec<i64> = (0..n * wu)
            .map(|r| (i64::from(row_slot[r]) * i64::from(nb)) * 16 + p[r])
            .collect();
        put_table(&mut b, &mut offs, "slot", &le64(&slot));
        put_table(&mut b, &mut offs, "zeros", &le32(&row_seq));
        put_table(
            &mut b,
            &mut offs,
            "cu",
            &le32(&(0..=n as i32).map(|x| x * w as i32).collect::<Vec<_>>()),
        );
        put_table(&mut b, &mut offs, "zero1", &[0u8; 8]);
        // FN17: the per-sequence tables the PLE conv reads through `ple_seq` carry the pad
        // pseudo-sequence at index `pad slot` (past the batch's `b` entries)
        let pe = pads.pad_entry();
        let ext = |mut v: Vec<i32>, at_pad: i32| -> Vec<i32> {
            if let Some(ix) = pe {
                v.resize(ix + 1, 0);
                v[ix] = at_pad;
            }
            v
        };
        put_table(
            &mut b,
            &mut offs,
            "seq_row0",
            &le32(&ext(
                s.iter().map(|x| x * w as i32).collect::<Vec<_>>(),
                (live * wu) as i32,
            )),
        );
        put_table(&mut b, &mut offs, "ring_row0", &le32(&ps));
        let dr: Vec<i32> = drop.iter().map(|&x| x as i32).collect();
        put_table(&mut b, &mut offs, "drop", &le32(&ext(dr, 0)));
        // FN96: the region-strided index under the sequence-major pools
        let (ck, gk) = pads
            .idx
            .map_or((1, pool_w as i32), |(c, g)| (c as i32, g as i32));
        let s1: Vec<i32> = ps.iter().map(|x| 1 + x * ck).collect();
        put_table(&mut b, &mut offs, "cidx", &le32(&s1));
        // FN16: EVERY snapshot of a sequence (stride pool_w, not w): the step resumes from
        // `naccept - 1`, which may be >= w when this step is narrower than the last one's commit
        let pwu = pool_w.max(w) as usize;
        let sidx: Vec<i32> = (0..n * pwu)
            .map(|r| 1 + ps[r / pwu] * gk + (r % pwu) as i32)
            .collect();
        put_table(&mut b, &mut offs, "sidx", &le32(&sidx));
        let na: Vec<i32> = naccept.iter().map(|&x| x as i32).collect();
        put_table(&mut b, &mut offs, "naccept", &le32(&na));
        let p0: Vec<i64> = pos.iter().map(|&x| i64::from(x)).collect();
        put_table(&mut b, &mut offs, "p0v", &le64(&p0));
        put_table(&mut b, &mut offs, "hi8", &[1, 0, 0, 0]);
        put_table(&mut b, &mut offs, "hi32", &le32(&ext(vec![1; n], 1)));
        put_table(
            &mut b,
            &mut offs,
            "mv",
            &le32(&ext(vec![w as i32; n], ((n - live) * wu) as i32)),
        );
        let ctx: Vec<i32> = pos.iter().map(|&x| (x + w) as i32).collect();
        put_table(&mut b, &mut offs, "ctx", &le32(&ctx));
        let rslot: Vec<i64> = (0..n * wu)
            .map(|r| i64::from(row_slot[r]) * RING as i64 + p[r] % RING as i64)
            .collect();
        put_table(&mut b, &mut offs, "rslot", &le64(&rslot));
        put_block_rows(&mut b, &mut offs, "bt", &ps, nb, bt_nb);
        if w > 1 {
            // FN24: the per-row attention tables (a row = SPEC 0's step of its token)
            let cr: Vec<i32> = p.iter().map(|&x| (x + 1) as i32).collect();
            put_table(&mut b, &mut offs, "ctx_rows", &le32(&cr));
            if rows_table {
                put_block_rows(&mut b, &mut offs, "bt_rows", &row_slot, nb, bt_nb);
            }
        }
        if let Some(ix) = pe {
            // FN17: the PLE conv's row -> sequence: live rows their own, pad rows the
            // pseudo-sequence at the pad slot
            let ple_seq: Vec<i32> = row_seq
                .iter()
                .map(|&q| if (q as usize) < live { q } else { ix as i32 })
                .collect();
            put_table(&mut b, &mut offs, "ple_seq", &le32(&ple_seq));
        }
        (b, offs)
    }

    /// Addresses at `base` for the layout [`ChunkConsts::host`] / [`ChunkConsts::host_decode`]
    /// returned (a prefill layout aliases the per-sequence tables of its one sequence).
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn at(
        base: u64,
        offs: &[(&'static str, usize)],
        m: u32,
        p0: u32,
        chunk: u32,
        nb: u32,
        cs: u64,
    ) -> ChunkConsts {
        let get = |n: &str| {
            offs.iter()
                .find(|(k, _)| *k == n)
                .map(|x| base + x.1 as u64)
        };
        let o = |n: &str| get(n).expect("table");
        let decode = get("sidx").is_some();
        ChunkConsts {
            m,
            p0,
            chunk,
            decode,
            b: if decode { m } else { 1 },
            pos3: o("pos3"),
            pos: o("pos"),
            slot: get("slot").unwrap_or_else(|| o("pos")),
            zeros: o("zeros"),
            ple_seq: get("ple_seq").unwrap_or_else(|| o("zeros")),
            cu: o("cu"),
            zero1: o("zero1"),
            seq_row0: get("seq_row0").unwrap_or_else(|| o("zero1")),
            ring_row0: get("ring_row0").unwrap_or_else(|| o("zero1")),
            drop: get("drop").unwrap_or_else(|| o("zero1")),
            cidx: get("cidx").unwrap_or_else(|| o("zero1")),
            sidx: get("sidx").unwrap_or(0),
            p0v: o("p0v"),
            hi8: o("hi8"),
            hi32: o("hi32"),
            mv: o("mv"),
            ctx: o("ctx"),
            rslot: o("rslot"),
            bt: o("bt"),
            ctx_rows: get("ctx_rows").unwrap_or_else(|| o("ctx")),
            bt_rows: get("bt_rows").unwrap_or_else(|| o("bt")),
            nb,
            bt_nb: nb,
            cs,
            max_ctx: 0,
            bar: 0,
            w: 1,
            naccept: get("naccept").unwrap_or(0),
            pw: 1,
            qsa_short: false,
            kslot: get("kslot").unwrap_or(0),
            kbtr: get("kbtr").unwrap_or(0),
            kv_row0: 0,
        }
    }
}

/// What a mix hands on.
#[derive(Debug, Clone, Copy)]
pub struct MixOutBufs {
    /// The stream the mix read (written by the down in expand / combine / final mode).
    pub hid: u64,
    /// The next sub-layer's pending injection logits `[m, 4]` bf16.
    pub injo: u64,
    /// The block input `[m, H]` bf16.
    pub bi: u64,
    /// e4m3 block input + per-row scale (Q8 only).
    pub q8: u64,
    pub scale: u64,
}

/// FN110: the MTP drafter's routed-expert record form (`--fn-mtp-experts`): `Fp8` = FN50's
/// default, `Bf16` = the control, `Slab4` = the expert tier's native MXFP4 record form in a fixed
/// VRAM slab (the trunk's grouped W4A8 train; numerics: the drafts move, the target's verify does
/// not).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpX {
    Bf16,
    Fp8,
    Slab4,
    /// FN138: the slab4 records at the TARGET's tp2 split (`--fn-mtp-split tp2`): each rank's
    /// half of every MTP expert, the MoE at the TpHalf cells (an ARM; numerics: the drafts move).
    Slab4Tp2,
}

impl MtpX {
    /// The flag's value (`bf16` / `fp8` / `slab4`).
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        Ok(match s {
            "bf16" => Self::Bf16,
            "fp8" => Self::Fp8,
            "slab4" => Self::Slab4,
            o => anyhow::bail!("--fn-mtp-experts {o}: bf16 | fp8 | slab4"),
        })
    }

    /// FN138: the form from `--fn-mtp-experts` (resolved) and `--fn-mtp-split` (`ep` | `tp2`;
    /// `tp2` is slab4's split only).
    pub fn resolve(experts: &str, split: &str) -> anyhow::Result<Self> {
        let x = Self::parse(experts)?;
        match (x, split) {
            (_, "ep") => Ok(x),
            (Self::Slab4, "tp2") => Ok(Self::Slab4Tp2),
            (_, "tp2") => anyhow::bail!(
                "--fn-mtp-split tp2 is the slab4 records' split (--fn-mtp-experts {experts})"
            ),
            (_, o) => anyhow::bail!("--fn-mtp-split {o}: ep | tp2"),
        }
    }

    /// The slab4 record form (either split).
    #[must_use]
    pub fn slab4(self) -> bool {
        matches!(self, Self::Slab4 | Self::Slab4Tp2)
    }
}

/// The pending triple a layer hands the next: (stream, MoE output, injection logits).
#[derive(Debug, Clone, Copy)]
pub struct Pending {
    pub hid: u64,
    pub mo: u64,
    pub injo: u64,
    /// FN37: the combine x' and its norm factor rr, already written by the MoE output's reduce.
    pub pre: Option<HcPre>,
    /// FN85 (`--fn-ar-prologue norm`): `mo` is THIS rank's MoE partial, not yet reduced -- the next
    /// layer's attention-side combine down reduces it in its prologue (`_rp`).
    pub rp: bool,
}

/// FN85: one rank's exchange channel for a reduce run INSIDE its consumer's prologue (the `_rp`
/// combine downs): the peer's and my mailbox windows (fine-grained, the one-shot's two slots of
/// `stride16` 16-byte words), the peer's and my flag words, my channel counter `ctl` [calls,
/// readers] (plain device memory), the one-shot's drain / acquire modes. ONE keyed set a family
/// (T10): every `_rp` down of every graph shares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RpChan {
    pub peer_scr: u64,
    pub my_scr: u64,
    pub peer_flags: u64,
    pub my_flags: u64,
    pub ctl: u64,
    pub stride16: u64,
    pub drain: i32,
    pub acq: i32,
}

/// FN137 (`--fn-moe-ar-push on`): one rank's fused MoE-out reduce channel (`moe_grouped_far`'s
/// `FarArgs` less the per-call SUM): the peer's / this rank's window (slot 0; slot 1 at +
/// `stride16` 16-byte words), the peer's / this rank's 20 slice flags, the channel counter and the
/// diagnostic words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FarChan {
    pub peer: u64,
    pub my: u64,
    pub pflags: u64,
    pub mflags: u64,
    pub ctr: u64,
    pub diag: u64,
    pub stride16: u64,
}

impl FarChan {
    /// The kernel's by-value `FarArgs` (72 bytes; `abi::STRUCT_FAR_ARGS`) for a call writing its
    /// SUM to `red` (the stamp pointer 0: the served form never stamps).
    #[must_use]
    pub fn blob(&self, red: u64) -> Vec<u8> {
        let mut b = Vec::with_capacity(72);
        for x in [
            self.peer,
            self.my,
            self.pflags,
            self.mflags,
            self.ctr,
            red,
            self.diag,
            0,
        ] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b.extend_from_slice(&(self.stride16 as i32).to_le_bytes());
        b.extend_from_slice(&0i32.to_le_bytes());
        b
    }
}

/// FN85: the next combine mix reduces its `blk` (this rank's partial) in its prologue over `ch`
/// and stores the reduced block output to `out` (0 = not stored).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RpIn {
    pub ch: RpChan,
    pub out: u64,
}

/// FN37 (the producer-rr law): what the block output's producer wrote for the NEXT mix -- the
/// combined stream x' `[m, W]` bf16 (= the down's `hid` in combine / final mode) and its norm
/// factor rr `[m, 4]` fp32, both in the layer's carry slot. The mix then runs the down's norm-free
/// twin (MODE 3), which reads them instead of computing them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HcPre {
    pub x: u64,
    pub rr: u64,
}

/// FN49b: an HC v2 symbol of a DPP cfg (`..._v2_m1w3t1d`, `..._v2_s1d`) -> its `ds_bpermute`
/// twin (the cfg without `d`: same geometry, same bytes); every other symbol unchanged.
#[must_use]
pub fn hc_dpp_control(sym: &str) -> String {
    match sym.strip_suffix('d') {
        Some(base)
            if sym.starts_with("fn_hc_")
                && sym.contains("_v2_")
                && (base.ends_with("t1") || base.ends_with("_s1")) =>
        {
            base.to_string()
        }
        _ => sym.to_string(),
    }
}

/// FN108: the `flashnext_hc8` twin of a planned HC v2 symbol (`<sym>_h8`: the same arguments, the
/// mixing matrix read as an e4m3 record) when the connection's mixers are fp8; else `sym`.
#[must_use]
pub fn hc8_symbol(sym: &str, w8: bool) -> String {
    if w8 {
        format!("{sym}_h8")
    } else {
        sym.to_string()
    }
}

/// FN108: the up of an fp8-mixer mix -- the prefill `p4s` cfg has no `_h8` twin (MF 4 spills with
/// the fp8 operand), so its rows run on `p2s` (32-row blocks, the same 20 column groups and 8
/// waves; FN21: both byte-identical to FN5's up), the row-block counters with it.
pub fn hc8_up_plan(up: &mut saber_kernels::flashnext::Launch, up_counters: &mut u32, m: u64) {
    if let Some(base) = up.symbol.strip_suffix("_v2_p4s") {
        up.symbol = format!("{base}_v2_p2s");
        up.grid[1] = (m as u32).div_ceil(32);
        if *up_counters > 0 {
            *up_counters = up.grid[1];
        }
    }
}

/// FN37: the rr twin of a planned combine / final down (`fn_hc_down_{combine,final}_v2_<cfg>` ->
/// `fn_hc_down_{combine,final}rr_v2_<cfg>`); `None` for any other symbol (the FN5 bands, mix,
/// expand).
#[must_use]
pub fn rr_down_symbol(sym: &str) -> Option<String> {
    // FN49b: the rr twins read the producer's rr (no norm butterfly): a DPP cfg maps to its base
    let sym = &hc_dpp_control(sym);
    for m in ["combine", "final"] {
        let from = format!("fn_hc_down_{m}_v2_");
        if let Some(cfg) = sym.strip_prefix(&from) {
            return Some(format!("fn_hc_down_{m}rr_v2_{cfg}"));
        }
    }
    None
}

/// FN85: the `_rp` twin of a planned combine down (`fn_hc_down_combine_v2_<cfg>_rp`: the exact
/// two-rank reduce in its prologue). The decode bands' cfgs carry one (`m1w3t1d`, `m1w3t1` = the
/// route control's, `m1w7t1`); any other planned down has none.
#[must_use]
pub fn rp_down_symbol(sym: &str) -> String {
    format!("{sym}_rp")
}

/// FN85: the DECODE forms' rows the arm takes (the decode HC bands; the rp mailbox window).
pub const DECODE_RP_MAX_ROWS: u64 = 64;

/// FN85: whether the planned combine down `sym` has an `_rp` twin.
#[must_use]
pub fn has_rp_twin(sym: &str) -> bool {
    ["m1w3t1d", "m1w3t1", "m1w7t1"]
        .iter()
        .any(|c| sym == format!("fn_hc_down_combine_v2_{c}"))
}

/// Where layer 0's (or a materialized) stream comes from.
#[derive(Debug, Clone, Copy)]
pub enum LayerIn {
    /// The embedding `[m, H]` bf16 (layer 0: expand).
    Embed(u64),
    /// The previous layer's pending triple.
    Pending(Pending),
    /// FN15: a materialized multi-stream `[m, W]` bf16 (the MTP layer's input): a plain mix.
    Stream(u64),
}

/// What the capture reads of a layer.
#[derive(Debug, Clone, Copy)]
pub struct LayerOut {
    pub pend: Pending,
    /// The mixer output (after its all-reduce at TP=2).
    pub att: u64,
    /// Route ids `[m * 10]` i32 (the route kernel's `topk_ids`).
    pub ids: u64,
    /// Router logits `[m, 512]` bf16.
    pub logits: u64,
    /// The PLE addition `[m, W]` bf16 (layer 1).
    pub ple: Option<u64>,
    /// Launches this layer emitted (the forward's, capture excluded).
    pub launches: u32,
}

/// A two-rank reduce the emitter calls between phases (TP=2), or none (TP=1).
pub trait Reduce {
    /// Reduce `ins[r]` (`n_elem` bf16 per rank) into new buffers; appends each rank's launch.
    fn reduce(
        &mut self,
        ems: &mut [Emitter<'_>],
        ins: &[u64],
        n_elem: u64,
        carry: bool,
    ) -> anyhow::Result<Vec<u64>>;

    /// FN37: [`Reduce::reduce`] of a BLOCK OUTPUT whose next consumer is a combine-mode mix over
    /// `next[r]` = (res `[m, W]`, inj `[m, 4]`): a reduce that can also writes that mix's x' and rr
    /// (in the layer's carry slot) and returns them. The default writes neither (the control).
    fn reduce_hc(
        &mut self,
        ems: &mut [Emitter<'_>],
        ins: &[u64],
        n_elem: u64,
        carry: bool,
        next: &[(u64, u64)],
    ) -> anyhow::Result<(Vec<u64>, Vec<Option<HcPre>>)> {
        let _ = next;
        let outs = self.reduce(ems, ins, n_elem, carry)?;
        let n = outs.len();
        Ok((outs, vec![None; n]))
    }

    /// FN85: the exchange channel per emitter for a reduce of `n_elem` bf16 a rank run INSIDE the
    /// consumer's prologue, or `None` when this reduce cannot take that form (lossy band, rows
    /// past the window, no channel). Counts the call as a prologue reduce when it answers.
    fn rp_channel(&mut self, ems: &[Emitter<'_>], n_elem: u64) -> Option<Vec<RpChan>> {
        let _ = (ems, n_elem);
        None
    }

    /// FN137: the fused MoE-out reduce's channel per emitter for a reduce of `n_elem` bf16 a rank
    /// run INSIDE the tp2 back half (`--fn-moe-ar-push on`), or `None` (the arm off, or this
    /// reduce is not the exact decode-form one: counted as a stand-down). Counts the call.
    fn far_channel(&mut self, ems: &[Emitter<'_>], n_elem: u64) -> Option<Vec<FarChan>> {
        let _ = (ems, n_elem);
        None
    }

    /// FN137: the arm asked for a fused reduce at a back half that cannot carry it (counted).
    fn far_stood_down(&mut self) {}

    /// FN164: which forms the mixer-out reduce rides the out / o projection at
    /// (`--fn-mixer-ar-push`; `Off` = the standalone reduce, the control).
    fn mix_arm(&self) -> super::dev::MixPush {
        super::dev::MixPush::Off
    }

    /// FN164: the fused MIXER-OUT reduce's channel per emitter for a reduce of `n_elem` bf16 a rank
    /// run INSIDE the out / o projection (`flashnext_t8far`), or `None` (the arm off, or this reduce
    /// is not the exact decode-form one: counted as a stand-down). Counts the call.
    fn mix_channel(&mut self, ems: &[Emitter<'_>], n_elem: u64) -> Option<Vec<FarChan>> {
        let _ = (ems, n_elem);
        None
    }

    /// FN164: the arm asked for a fused mixer reduce at a form that cannot carry it (counted).
    fn mix_stood_down(&mut self) {}
}

/// Control arms that must move the bytes (gate A's negative control).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Ctl {
    #[default]
    None,
    /// The GDN gated norm's arguments swapped (z stride off by one head) -- a wiring bug on purpose.
    SwapArg,
    /// FN13's gate-C control: the certified saber chunk scan (split-and-clamped gates, FN9b's fault)
    /// in place of the exact-gate scan at prefill.
    CertifiedScan,
    /// FN13's chunk-law control: the GDN recurrent state a chunk > 0 reads is zeroed first (a
    /// dropped carry; the driver applies it).
    DropCarry,
}

/// One rank's emitter: the arena, the launch list, the kernel declarations.
pub struct Emitter<'d> {
    /// FN61c: the W8 records of the layer being emitted (set by [`emit_layer`] for a decode-form
    /// target layer under `--fn-trunk w8`, None otherwise) and the W8 head (weight, scales) the
    /// target's post reads when the last layer emitted ran W8 (`t8_post`).
    pub t8_cur: Option<T8W>,
    pub t8_head: Option<(u64, u64)>,
    pub t8_post: bool,
    /// FN61e: PREFILL forms read the W8 records too (`--fn-trunk w8` / `w8-head`; false under
    /// FN61d's `w8-decode`, whose prefill keeps the resident bf16 records).
    pub t8_prefill: bool,
    /// FN68 (`--fn-kv-dtype fp8`): the QSA layers' K/V pages are e4m3 bytes -- the append is
    /// `fn_kv_append_ring_h256_fp8` and every attention / gather symbol its `_fp8kv` twin (same
    /// kernargs, same element strides; the combine and the indexer unchanged).
    pub kv8: bool,
    /// FN68d (`--fn-kv-mtp-dtype`): the MTP layer's (48's) pages are e4m3 too -- false = bf16
    /// pages for the MTP layer alone (its own one-layer tier / pool; the bf16 append + attention
    /// symbols at layer 48, every target layer unchanged). Equal to `kv8` = one dtype.
    pub kv8_mtp: bool,
    /// FN68d: the MTP layer's twin tier's device page table (`[1, 9, nbs]` i32; 0 = layer 48
    /// reads the main tier's table at its tier layer 12).
    pub kv_mtp_table: u64,
    /// FN68b: this rank's (K, V) page scale per layer (`[0, 49)`; empty = FN68's 1.0) and the
    /// descale table's device base (`kvscale::KvScales::table`; 0 = the null descale pointers,
    /// FN68's kernargs).
    pub kv_scale: Vec<(f32, f32)>,
    pub kv_desc: u64,
    /// FN61g: the bf16 head is NOT resident (`--fn-trunk w8-head`): EVERY post -- the target's
    /// and the drafter's (FN22's full head, FN33's check arm) -- reads the W8 head `t8_head`, and
    /// FN33's rerank scores its candidates from the same W8 bytes (`fn_t8r_a8w8_gather_m16n64`).
    pub head_bf16_dropped: bool,
    pub d: &'d Decls,
    pub rules: &'d FlashNextRules,
    pub dims: Dims,
    pub arena: Arena,
    pub recs: Vec<Rec>,
    pub layer: i32,
    /// FN45b2: the tiered KV's device page table (`[13, 9, nbs]` i32; 0 = the flat pools) and one
    /// layer's table bytes -- a QSA layer's K/V readers take `kv_table + tier_layer * kv_ls`.
    pub kv_table: u64,
    pub kv_ls: u64,
    pub capture: bool,
    pub ctl: Ctl,
    /// Forward launches emitted since construction (capture excluded; survives a flush).
    pub emitted: u32,
    /// The GDN recurrent state stored in f32 (`fn_gdn_chunk_scan_xg_*`) instead of f16 (the default).
    pub state_f32: bool,
    /// Dynamic LDS bytes of the NEXT launch (taken by `run`).
    pub next_shared: u32,
    /// Buffers of layer `tap_layer` recorded for the boundary-law diagnostic (name, addr, bytes).
    pub tap_layer: i32,
    /// FN150: also tap the seven-launch train's expert-half buffers (`moe.gu` / `moe.q` /
    /// `moe.sc` / `moe.y`) at `tap_layer` (the engine-path byte test `saber fn150-moe-check`).
    pub tap_moe: bool,
    pub taps: Vec<(String, u64, u64)>,
    /// Which-path counter (FN14): decode attentions whose table cell says `declc`, run as the pair.
    pub declc_as_pair: u32,
    /// FN15b: the GDN conv state's per-slot length `3 + W_max - 1` (3 at SPEC 0: FN13's layout;
    /// `3 + k` under `--fn-mtp k`), for EVERY form (the prefill prep writes entries 0..3 of it).
    pub conv_l: u64,
    /// FN19: `--fn-qsa` (the model's sparse law above 2,048 by default; `dense` = the control).
    pub qsa: saber_kernels::fn19::QsaMode,
    /// FN19 (the dump): the last sparse QSA layer's selection `(sel, iq, rows)`: `sel` `[rows, 512]`
    /// i32 ascending group lists, `iq` the indexer queries `[rows, 4, 128]` bf16 the scores read
    /// (the capture takes both after the layer runs: the exact selection gate's inputs).
    pub qsa_sel: Option<(u64, u64, u64)>,
    /// FN24: `--fn-verify-class` (the verify step's cell class; `Width1` = SPEC 0's).
    pub vclass: VerifyClass,
    /// FN24: while a TARGET verify step is emitted in the width-1 class, `(rows, class rows)`:
    /// every M-keyed route of an op over `rows` rows reads the cell of `class rows` (set by
    /// [`emit_layer`] from the step's consts, cleared by [`emit_post3`] and [`Emitter::mtp_pre`]).
    pub route_m: Option<(u64, u64)>,
    /// FN24 (`--fn-glue-control`): the pre-switch glue symbols (`qk_norm_rope_wg8_h256_rot64_mi1_red5`,
    /// `fn_idx_q_compress_h128`) -- the control of FN21's byte-identical switch.
    pub glue_v1: bool,
    /// FN24 which-path counter: target layers of a width >= 2 step emitted in SPEC 0's class
    /// (`[0]`) / in their own M's class (`[1]`, the `Wide` control).
    pub vclass_layers: [u64; 2],
    /// FN78: the layer being emitted is a DECODE form (a target decode / verify step or a drafter
    /// pass; set by [`emit_layer`] from the step's consts, false for a prefill chunk): the TP = 2
    /// reduce reads it for the decode-form band law ([`super::dev::ArPair::decode_exact`]).
    pub ar_decode: bool,
    /// FN24 which-path counter: GEMMs whose class cell ran as its row-slice twin
    /// (`flashnext::class_symbol_at`: the m16 tile above 16 rows).
    pub mtile_twin: u64,
    /// FN30: `--fn-moe-train` (the target's MoE train; `Fused` = FN26's back half at a decode step
    /// of <= 64 rows, the default; `Legacy` = FN4's seven launches, the control).
    pub moe_train: MoeTrain,
    /// FN30: the device address of this rank's fused-MoE `sync` region ([`moe_sync_bytes`], zeroed
    /// ONCE at boot, outside the arena: the counters are back at zero after every launch); 0 = none
    /// (a fused block then refuses).
    pub moe_sync: u64,
    /// FN114c (`--fn-shared-expert split`): the device address of this rank's shared-expert HALF
    /// tables (48 layers x 6 u64: gate_up w / ws / wref, down w / ws / wref at FN80a's half
    /// layout, [`SHX_TABLE_BYTES`] a layer); 0 = `rank0` (today: the whole shared expert on rank 0
    /// only). Non-zero: every TARGET MoE block on the fused decode / verify train lays out the
    /// shared rows on BOTH ranks and runs `moe_back_ss_*` (the shared tile at this rank's half);
    /// prefill chunks and the drafter keep rank 0's whole shared expert.
    pub shared_split: u64,
    /// FN120b (`--fn-glue`): the run-wide seed region (None = `off`, today: per-layer seed fills).
    pub glue: Option<GlueSeeds>,
    /// FN120c (`--fn-dataflow gdn`): the persistent GDN-tail tables (None = `off`, the train).
    pub df: Option<DfTables>,
    /// FN120c: the launches the dataflow form removed from the layer being emitted (2 a folded
    /// GDN tail; the declared law's count minus these is what the layer emits).
    pub df_saved: u32,
    /// FN30 which-path counter: target MoE blocks emitted as the fused train (`[0]`) / as the seven
    /// launches (`[1]`: prefill chunks, `--fn-moe-train legacy`; the drafter's bf16 experts are not
    /// counted).
    pub moe_blocks: [u64; 2],
    /// FN31 which-path counter: fused blocks whose router GEMM + route ran as the front half.
    pub moe_front_blocks: u64,
    /// FN47b (`--fn-pregate-capture`, a DIAGNOSTIC; every forward byte unchanged): while set,
    /// `(log, next_router)` -- a target decode layer `l` also runs (a) its own router on its
    /// attention-side mix output (the stream the previous layer handed it) into slot (0, l) and
    /// (b) layer l + 1's router (`next_router`, 0 = none) on its MoE input into slot (1, l + 1)
    /// of the log `[2][LAYERS][PREGATE_ROWS][512]` bf16. Those launches are NOT forward launches
    /// (`emitted` excludes them; [`Self::pregate_launches`] counts them).
    pub pregate: Option<(u64, u64)>,
    /// FN47b which-path counter: the capture's router launches.
    pub pregate_launches: u64,
    /// FN49b (`--fn-route-control` = `Shipped`): the route's lowering. `Dpp` (THE DEFAULT): every
    /// MoE route of T 1..16 rows -- the seven-launch train's too (target, drafter, short prefill
    /// chunks) -- is FN31's LDS route on FN49a's DPP butterflies (`moe::route_symbol`), the front
    /// half `moe_front_*_dpp`; `Shipped` = main before FN49b (FN4's route in the seven, the
    /// `ds_bpermute` LDS route / front in the fused train). Tables byte-identical by FN31's + FN49a's
    /// oracles.
    pub route_form: moe::RouteForm,
    /// FN84 (`--fn-moe-miss-order`, `id` = FN26's symbol, the default): the fused back half's work
    /// order. Every order is the shipped bytes (LEDGER FN84); only the tickets move.
    pub miss_order: moe::MissOrder,
    /// FN84: this rank's pinned-RAM expert slab, `[lo, hi)` device addresses (the order law's RAM
    /// predicate; an ordering hint, no byte depends on it). `(0, 0)` = no tier: nothing is RAM.
    pub miss_ram: (u64, u64),
    /// FN84 which-path counter: fused back-half launches by order ([`moe::MissOrder::index`]).
    pub miss_launches: [u64; 4],
    /// FN85 (`--fn-ar-prologue norm`): at a TP = 2 target decode / verify form the exact reduce
    /// of a block output whose consumer is a combine down runs INSIDE that down's prologue (the
    /// `_rp` twin); the reduce's own launch, node and edge go. Off = the control (the default).
    pub ar_prologue: bool,
    /// FN85: the channel the NEXT mix's combine down reduces its `blk` over (taken by [`Self::mix`]).
    pub rp_next: Option<RpIn>,
    /// FN137: the channel the NEXT fused back half reduces its MoE output over (taken by
    /// [`Self::moe_back`], which then returns the SUM in the layer's carry).
    pub far_next: Option<FarChan>,
    /// FN137 which-path counter (`fnar137`): back halves that carried the MoE-out reduce.
    pub ar_far: u64,
    /// FN137: the reduce launches the arm removed from the layer being emitted.
    pub ar_far_saved: u32,
    /// FN164: the channel the NEXT out / o projection (N = ldc = 2,560) reduces its partial over
    /// (taken by [`Self::w8_gemm_q`], which then leaves the SUM's address in [`Self::mix_red`]).
    pub mix_next: Option<FarChan>,
    /// FN164: the SUM the fused out / o projection wrote (the address the reduce would have).
    pub mix_red: Option<u64>,
    /// FN164 which-path counter (`fnar164`): out / o projections that carried the mixer-out reduce.
    pub ar_mix: u64,
    /// FN164: the reduce launches the arm removed from the layer being emitted.
    pub ar_mix_saved: u32,
    /// FN85 which-path counter (`fnarpro`): reduces run inside a consumer's prologue (`[0]`) /
    /// reduces the arm left standalone because the consumer or the channel cannot take them (`[1]`).
    pub ar_rp: [u64; 2],
    /// FN85: the reduce launches the arm removed from the layer being emitted (the declared law's
    /// count minus these is what the layer emits).
    pub ar_rp_saved: u32,
    /// FN86b (`--fn-fuse`): the launch-fusion families this emitter runs fused
    /// (`saber_kernels::flashnext::fuse86`; from the rules at construction).
    pub fuse86: u32,
    /// FN86b which-path counter (`fnfuse86`): per family (index = family - 1) the launches its
    /// fused symbol ran (`[0]`) and the chain launches it removed (`[1]`) -- or, with the family
    /// off, `[0]` 0 and `[1]` the chain launches the family WOULD have removed at its sites (the
    /// control's count).
    pub fuse86_n: [[u64; 2]; 10],
    /// FN86b: the launches the fusion arm removed from the layer being emitted (the declared
    /// law's count minus these is what the layer emits).
    pub fuse86_saved: u32,
    /// ATTN-OPT-3 family 7: the decode attention's combine left to the gate (its `SABERArgs` blob
    /// and split count), taken by [`Self::qsa_core`]'s family-1 site, which runs the combine + gate +
    /// quant as one launch.
    pub f7_pending: Option<(Vec<u8>, u64)>,
    /// FN111 (`--fn-tiles`): the re-tiled cells this emitter runs (`saber_kernels::flashnext::
    /// tiles111`; from the rules at construction).
    pub tiles111: u32,
    /// FN111 which-path counter (`fntile111`): per tile the launches its fn111_ symbol ran (`[0]`)
    /// and the sites that kept the shipped cell (`[1]`).
    pub tiles111_n: [[u64; 2]; 3],
    /// FN122b which-path counter (`fnpfx122`): the fp8-mixer prefill downs that ran a
    /// `flashnext_hc8p` tile (`[0]`) and those that kept the shipped cfg (`[1]`).
    pub pf122_n: [u64; 2],
    /// FN122c which-path counter: the mixes whose down / up ran dequantized on the bf16 bodies.
    pub pf122_dq: [u64; 2],
    /// FN122c: the dequant launches the dq family added to the layer being emitted (the declared
    /// law's count plus these is what the layer emits).
    pub pf122_added: u32,
    /// FN124 which-path counter: fp8 QSA prefill gathers on the qsa124 body (`[0]`) / the shipped
    /// symbol (`[1]`); the in-situ twins emitted; the twin's counter (device; None = no twin).
    pub pf124_n: [u64; 2],
    /// FN128 (`--fn-sel128`): the decode-form selection launches on an added `flashnext_qsa128`
    /// symbol / on the parent, per kernel (score, top-k, union) -- the which-path count.
    pub sel128_n: [[u64; 2]; 3],
    /// FN131 (`--fn-hc-down`): the fp8-mixer decode downs on an added `flashnext_hc131` symbol
    /// (`[0]`) / on the shipped `flashnext_hc8` down of a routed cfg (`[1]`) -- the which-path count.
    pub down131_n: [u64; 2],
    /// FN141 (`--fn-k141`): per family (up, t8, router) the launches on an added symbol (`[0]`) /
    /// on the shipped parent at a gated site (`[1]`) -- the which-path count.
    pub k141_n: [[u64; 2]; 3],
    /// FN163 (`--fn-route163`): per band (b8 route, b1 front) the launches on the added symbol
    /// (`[0]`) / the sites that kept the shipped one (`[1]`) -- the `fnroute163` which-path count.
    pub route163_n: [[u64; 2]; 2],
    pub pf124_twins: u64,
    pub pf124_twin: Option<u64>,
}

/// FN47b: the capture log's rows per (method, layer) slot (decode / verify steps of more rows are
/// not captured).
pub const PREGATE_ROWS: u64 = 64;

/// FN47b: the byte offset of (method, layer)'s slot in the capture log.
#[must_use]
pub fn pregate_slot(method: u64, layer: u64) -> u64 {
    (method * 48 + layer) * PREGATE_ROWS * 512 * 2
}

/// FN30: the fused MoE back half's `sync` region: ONE slot per step row count `T` in
/// `1..=BackTrain::MAX_TOKENS` (every (bucket, width) of a decode / verify step maps to its
/// `T = b x w`; steps sharing a `T` share a slot -- they run in stream order and the counters are
/// back at zero after every launch), each slot `BackTrain::sync_words(tiles_cap(64))` i32 at its
/// start + FN31's front-half word (`FrontTrain::SYNC_WORDS`) as its LAST word, rounded to 256 B for
/// this rank's local experts.
#[must_use]
pub fn moe_sync_slot_bytes(eloc: u32) -> u64 {
    let tc = moe_tiles_cap(u64::from(moe::BackTrain::MAX_TOKENS), 1, u64::from(eloc));
    ((moe::BackTrain::sync_words(tc as u32) + moe::FrontTrain::SYNC_WORDS) * 4).div_ceil(256) * 256
}

/// FN30: the whole `sync` region of a rank with `eloc` local experts (the ledger class
/// `fn.moe.sync`): TP=2 (256 local experts) 64 x 3,840 B = 240 KiB.
#[must_use]
pub fn moe_sync_bytes(eloc: u32) -> u64 {
    u64::from(moe::BackTrain::MAX_TOKENS) * moe_sync_slot_bytes(eloc)
}

/// FN120b: the run-wide seed region's zero / ones parts at FN120b's boot (6 + 2 MiB, sized by
/// observation at the b8 w3 census). FN120e: a boot SIZES the region from the dry run over every
/// form it can serve ([`GlueRegion::of_need`], `run::glue_census`); these constants are only
/// [`GlueRegion::fixed`] (the unit tests' region).
#[cfg_attr(not(test), allow(dead_code))]
pub const GLUE_ZERO_BYTES: u64 = 6 << 20;
#[cfg_attr(not(test), allow(dead_code))]
pub const GLUE_ONES_BYTES: u64 = 2 << 20;
/// FN120e: the margin each part carries over the widest form's high-water (one 2 MiB page), and
/// the granule both parts round up to.
pub const GLUE_MARGIN_BYTES: u64 = 2 << 20;

/// FN120e: the glue region's two parts a rank (the ledger class `fn.glue120` = their sum).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlueRegion {
    pub zero: u64,
    pub ones: u64,
}

impl GlueRegion {
    /// THE SIZE LAW (FN120e): each part = the widest form's high-water + [`GLUE_MARGIN_BYTES`],
    /// rounded up to a whole 2 MiB page.
    #[must_use]
    pub fn of_need(zero: u64, ones: u64) -> GlueRegion {
        let part = |n: u64| (n + GLUE_MARGIN_BYTES).div_ceil(GLUE_MARGIN_BYTES) * GLUE_MARGIN_BYTES;
        GlueRegion {
            zero: part(zero),
            ones: part(ones),
        }
    }
    /// FN120b's fixed region (6 + 2 MiB).
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn fixed() -> GlueRegion {
        GlueRegion {
            zero: GLUE_ZERO_BYTES,
            ones: GLUE_ONES_BYTES,
        }
    }
    #[must_use]
    pub fn total(&self) -> u64 {
        self.zero + self.ones
    }
}

/// FN120e: a run whose seeds passed the region -- the take that would have passed it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlueSpill {
    /// The ones part (false = the zero part).
    pub ones: bool,
    /// The part's end the take wanted, and the part's bytes.
    pub want: u64,
    pub cap: u64,
}

/// FN120b (`--fn-glue seeds|fold`): THE GLUE FOLD. A captured decode / verify RUN takes every
/// seeded buffer (the seed law: zero, or all-ones for `row_tok`) from ONE region outside the
/// arena, at a distinct address per layer (the bumps do not restart at a layer), so the per-layer
/// seed fills disappear: the region is seeded at boot and the run RESTORES it with two fills at
/// its end (after its last reader), keeping the invariant "every byte of the region holds its
/// seed between runs". Every buffer still holds its seed value when its first reader runs (bytes
/// identical by construction; only addresses move). `fold` also lays the route ids of every target
/// layer at the route-id log's own offsets (layer `l` at `l x m x 40` from the zero part's base),
/// so ONE copy a run replaces the per-layer copy.
///
/// FN120e: the region is sized at boot ([`GlueRegion`]); a take that would pass it is NEVER a
/// fatal on a served boot: the run SPILLS -- that take and every later seed of the run come from
/// the arena under the per-layer fills (`--fn-glue off`'s law; the seeds already taken stay in
/// the region and are restored at the run's end as always), counted and logged
/// (`fnglue120 rank<r>: FALLBACK`). `strict` (`--fn-glue fold-strict`, and every boot that is not
/// `--fn-serve`: the gates and the harness) keeps FN120b's loud refusal.
#[derive(Debug, Clone)]
pub struct GlueSeeds {
    pub zero: Bump,
    pub ones: Bump,
    /// This run's high-water marks (bytes from each part's base) -- the restore's extents.
    pub zhi: u64,
    pub ohi: u64,
    /// `fold`: the ids block's per-layer stride (`m x 40`); 0 = `seeds` (or the run has no ids).
    pub ids_stride: u64,
    /// `fold` is the form.
    pub idlog: bool,
    /// A run is open (between [`GlueSeeds::begin`] and [`GlueSeeds::end`]).
    pub on: bool,
    /// FN120e: refuse a take past the region (the gates) instead of spilling (the served boot).
    pub strict: bool,
    /// FN120e: this run spilled (the first take past the region); None = it fit.
    pub spill: Option<GlueSpill>,
    /// FN120e: emitted runs that spilled / seeds the spills sent to the arena (a held run's dry
    /// re-emission counts again: these count emissions).
    pub spills: u64,
    pub spilled_seeds: u64,
}

impl GlueSeeds {
    /// FN120b's fixed region at `base`, strict (the unit tests).
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new(base: u64, idlog: bool) -> GlueSeeds {
        GlueSeeds::sized(base, GlueRegion::fixed(), idlog, true)
    }
    /// FN120e: the region `r` at `base` (the zero part first, the ones part after it).
    #[must_use]
    pub fn sized(base: u64, r: GlueRegion, idlog: bool, strict: bool) -> GlueSeeds {
        GlueSeeds {
            zero: Bump::new(base, r.zero),
            ones: Bump::new(base + r.zero, r.ones),
            zhi: 0,
            ohi: 0,
            ids_stride: 0,
            idlog,
            on: false,
            strict,
            spill: None,
            spills: 0,
            spilled_seeds: 0,
        }
    }
    /// The ones part's offset from the region's base (the restore's second fill).
    #[must_use]
    pub fn ones_at(&self) -> u64 {
        self.ones.base - self.zero.base
    }
    /// Does `a` lie inside the zero part (FN120e: a spilled run's later seeds are the arena's)?
    #[must_use]
    pub fn holds(&self, a: u64) -> bool {
        a >= self.zero.base && a < self.zero.base + self.zero.cap
    }
    /// Open a run of `m` rows: both parts restart; under `fold` the ids block (48 x m x 40) is
    /// reserved at the zero part's base (FN120e: an ids block past the region runs the run as
    /// `seeds` -- the ids are ordinary seeds, the per-layer copies run -- unless strict).
    pub fn begin(&mut self, m: u64) -> anyhow::Result<()> {
        ensure!(!self.on, "FN120b: a glue run opened twice");
        self.zero.reset();
        self.ones.reset();
        self.spill = None;
        self.ids_stride = if self.idlog { m * 40 } else { 0 };
        if self.ids_stride > 0 {
            let ids = u64::from(super::plan::LAYERS) * self.ids_stride;
            if ids > self.zero.cap && !self.strict {
                self.ids_stride = 0;
            } else {
                self.zero.take(ids)?;
            }
        }
        self.zhi = self.zero.top;
        self.ohi = 0;
        self.on = true;
        Ok(())
    }
    /// A seed of `bytes` from the region: Some(address), or None = the run has spilled (the
    /// caller takes it from the arena under the per-layer fills). Strict: a take past the region
    /// refuses loudly (FN120b's law).
    pub fn take(&mut self, bytes: u64, ones: bool) -> anyhow::Result<Option<u64>> {
        if self.spill.is_some() {
            self.spilled_seeds += 1;
            return Ok(None);
        }
        let b = if ones { &self.ones } else { &self.zero };
        let want = b.top.div_ceil(super::kern::ALIGN) * super::kern::ALIGN + bytes.max(4);
        if want > b.cap && !self.strict {
            self.spill = Some(GlueSpill {
                ones,
                want,
                cap: b.cap,
            });
            self.spills += 1;
            self.spilled_seeds += 1;
            return Ok(None);
        }
        if ones {
            let a = self.ones.take(bytes)?;
            self.ohi = self.ohi.max(self.ones.top);
            Ok(Some(a))
        } else {
            let a = self.zero.take(bytes)?;
            self.zhi = self.zhi.max(self.zero.top);
            Ok(Some(a))
        }
    }
    /// Close the run: (zero bytes, ones bytes) to restore, each rounded up to 256.
    pub fn end(&mut self) -> anyhow::Result<(u64, u64)> {
        ensure!(self.on, "FN120b: a glue run closed twice");
        self.on = false;
        Ok((self.zhi.div_ceil(256) * 256, self.ohi.div_ceil(256) * 256))
    }
}

/// FN120c: the dataflow table region a rank (the ledger class `fn.df120`): 512 sites of 4 KiB
/// (a b1 verify GDN tail takes one site a layer and form; 36 layers x the b1 widths fit 3x over).
pub const DF120_BYTES: u64 = 2 << 20;
/// FN120c: one site -- [0, 256) the control words (zero at boot, the kernel leaves them zero but
/// the sticky ones), [256, 384) the stage descriptors, [384, ..) the kernarg images then the tasks.
pub const DF120_SITE_BYTES: u64 = 4096;
const DF120_STAGES_AT: usize = 256;
const DF120_IMGS_AT: usize = 384;
/// FN120c: the persistent launch's symbol (`flashnext_df120b`).
pub const DF120_KERNEL: &str = "fn120_df_gdn_b1";
/// FN120c: its grid -- THIS kernel's measured co-residency (card 1, fn120c_probe `res`: 64
/// workgroups check in, 80 refuse -- not FN120's 96 for the forced-budget kernels); the fused
/// update's own grid barrier needs its <= 32 virtual blocks resident at once.
pub const DF120_GRID: u64 = 64;
/// FN120c: the check-in / dependency timeout in REALTIME ticks (100 MHz): 20 ms.
pub const DF120_TIMEOUT_TICKS: i64 = 2_000_000;
/// FN120c: the sticky control words a site reports (`ctl[4..7]`: refusals, dependency timeouts,
/// launches begun).
pub const DF120_STICKY: [usize; 3] = [4, 5, 6];

/// FN120c (`--fn-dataflow gdn`): the host-built tables of the persistent GDN-tail launches. A
/// site's bytes (stages, the three bodies' kernarg images, tasks in topological order) are keyed
/// by CONTENT: a re-emission of the same layer and form (a dry re-emit, a re-capture) finds its
/// site, so a captured graph's argument never moves and nothing in flight is rewritten. New sites
/// are uploaded after the capture closes (`pending`, before the graph's first launch).
#[derive(Debug, Clone)]
pub struct DfTables {
    pub base: u64,
    next: u64,
    slots: std::collections::HashMap<Vec<u8>, u64>,
    /// (offset in the region, bytes) not yet on the card.
    pub pending: Vec<(u64, Vec<u8>)>,
    /// A captured target run is being emitted (only then does a site fold).
    pub armed: bool,
    /// `gdn-nodep` (the gate's CONTROL): no task waits for its producer stage (must differ).
    pub nodep: bool,
    /// Which-path count: GDN tails folded into one launch (`[0]`) / armed tails that kept the
    /// train (`[1]`: not the b1 verify chain).
    pub sites: [u64; 2],
}

impl DfTables {
    #[must_use]
    pub fn new(base: u64, nodep: bool) -> DfTables {
        DfTables {
            base,
            nodep,
            next: 0,
            slots: std::collections::HashMap::new(),
            pending: Vec::new(),
            armed: false,
            sites: [0; 2],
        }
    }
    /// The distinct sites (each one a control block the end line reads).
    #[must_use]
    pub fn used(&self) -> u64 {
        self.next / DF120_SITE_BYTES
    }
    /// The site holding `bytes` (found by content, or a new one queued for upload).
    pub fn site(&mut self, bytes: Vec<u8>) -> anyhow::Result<u64> {
        if let Some(&a) = self.slots.get(&bytes) {
            return Ok(a);
        }
        ensure!(
            self.next + DF120_SITE_BYTES <= DF120_BYTES,
            "FN120c: the dataflow region is full ({} sites)",
            self.used()
        );
        let off = self.next;
        self.next += DF120_SITE_BYTES;
        self.pending.push((off, bytes.clone()));
        self.slots.insert(bytes, self.base + off);
        Ok(self.base + off)
    }
}

/// FN120c: the body kinds of `flashnext_df120b` (its `DF_*`) and their threads.
fn df120_kind(sym: &str) -> Option<(u32, u32)> {
    use saber_kernels::flashnext::{fuse86, tiles111};
    if sym == fuse86::Q8W8_SFX {
        Some((1, 128))
    } else if sym == fuse86::F5_FUSED {
        Some((2, 256))
    } else if sym == tiles111::SYMS[0] {
        Some((3, 128))
    } else {
        None
    }
}

/// FN120c: the persistent kernel unpacks a body's kernarg image at each argument's NATURAL
/// alignment (`fn120b_layout`); the manifest's offsets must be exactly that (fail loud).
fn df120_abi(d: &Decls, sym: &str) -> anyhow::Result<()> {
    let (_, k) = d.get(sym)?;
    let mut end = 0u64;
    for a in k.arg.iter().filter(|a| !a.value_kind.starts_with("hidden")) {
        let size = u64::from(a.size);
        ensure!(
            matches!(size, 4 | 8),
            "FN120c: {sym} arg {:?} is {size} B (scalars and pointers only)",
            a.name
        );
        let at = end.div_ceil(size) * size;
        ensure!(
            u64::from(a.offset) == at,
            "FN120c: {sym} arg {:?} at {} in the manifest, {at} by natural alignment",
            a.name,
            a.offset
        );
        end = at + size;
    }
    Ok(())
}

/// FN114c: one layer's shared-expert half table (6 u64: gate_up w / ws / wref, down w / ws / wref).
pub const SHX_TABLE_BYTES: u64 = 6 * 8;

/// FN114c: the fused back half with the shared tile at this rank's half (`moe_grouped_ss`).
#[must_use]
pub fn shared_split_symbol(bt: &moe::BackTrain) -> String {
    format!("moe_back_ss_gu{}_dn{}_p1", bt.gu_ks, bt.dn_ks)
}

/// FN4's static tile bound of a route over `m` tokens at `tm` blocks a tile (`tiles_cap`).
#[must_use]
pub fn moe_tiles_cap(m: u64, tm: u64, eloc: u64) -> u64 {
    let r = m * 10;
    let me = eloc.min(r);
    me + r / (16 * tm) + 1 + cdiv(m, 16).div_ceil(tm)
}

/// FN61c: the W8 trunk's output-stride cells (`flashnext_trunk8`).
pub const T8_A8W8_LDC: &str = "fn_t8_a8w8_bs128_ldc";
pub const T8_BGEMM_LDC: &str = "fn_t8_bgemm_m16_ldc";
/// FN61c: launches a W8 layer adds over its bf16 form at a decode step (merged projection: quant +
/// W8 prefix + bf16 suffix = 3 for 1; out / o projection: quant + W8 = 2 for 1).
pub const T8_LAYER_EXTRA: u32 = 3;

/// FN61e: the W8A8 cell by rows (`flashnext_trunk8p`, each element the FN61c parent's arithmetic,
/// byte-gated on card 1 vs the parent -- `tools/gates/fn61e_hold.sh`, LEDGER FN61e): (max rows,
/// symbol, rows a tile, columns a tile, rows-fastest grid). Up to 16 rows the FN61c / dflash cells
/// stay (the bytes and the shipped speed of b1..b16); past the last row the prefill tile. Picked
/// by hold k1 (card 1, 210 W, IC-cold, us): the 32-wide 2-wave tiles lose to 64-wide 4-wave ones
/// at every M, d2w4 <= 32 rows (out m24 20.2 -> 16.8), d4w4 <= 64 (out m64 23.3 -> 20.4, qkvz
/// 42.3 -> 38.5), p4w8 past 64 (qkvz m2048 1,307 -> 752, m8192 5,195 -> 3,048).
pub const T8_TILES: &[(u64, &str, u64, u64, bool)] = &[
    (32, "fn_t8p_a8w8_d2w4_ldc", 32, 64, false),
    (64, "fn_t8p_a8w8_d4w4_ldc", 64, 64, false),
    (u64::MAX, "fn_t8p_a8w8_p4w8_ldc", 64, 128, true),
];
/// FN61e: the bf16 suffix's 64-row tile -- byte-identical but SLOWER than FN61c's 16-row cell at
/// every M measured (k1: idx m2048 149 -> 283 us, ba m24 8.2 -> 16.9): not routed; the suffix
/// stays on `fn_t8_bgemm_m16_ldc` at every form.
#[cfg(test)]
pub const T8_BGEMM_M64: &str = "fn_t8p_bgemm_m64_ldc";

/// FN61e: the W8A8 GEMM's (symbol, grid, threads, takes ldc) at `m` rows.
#[must_use]
pub fn t8_cell(m: u64, n: u64, ldc: u64) -> (&'static str, [u64; 3], u64, bool) {
    if m <= 16 {
        return if ldc != n {
            (T8_A8W8_LDC, [n / 128, 1, 1], 128, true)
        } else {
            (
                "dflash_gemm_a8w8_bs128_m16n128k128",
                [n / 128, 1, 1],
                128,
                false,
            )
        };
    }
    let &(_, sym, bm, bn, mfast) = T8_TILES
        .iter()
        .find(|t| m <= t.0)
        .expect("the last tile takes any rows");
    let grid = if mfast {
        [cdiv(m, bm), n / bn, 1]
    } else {
        [n / bn, cdiv(m, bm), 1]
    };
    (sym, grid, bn / 16 * 32, true)
}

/// The fused decode update's limits (`saber_gfx1201.toml [gdn]`: fused_max_items / fused_max_wg).
pub const FUSED_MAX_ITEMS: u64 = 32;

/// FN86c family 5: the gated norm + act quant arguments the decode update's tail takes (`z` the
/// gate at row stride `zs` elements, `nw` the norm weight, `nrm` / `q8` / `s8` family 1's outputs).
#[derive(Clone, Copy, Debug)]
pub struct F5Tail {
    pub z: u64,
    pub zs: u64,
    pub nw: u64,
    pub nrm: u64,
    pub q8: u64,
    pub s8: u64,
}
pub const FUSED_MAX_WG: u64 = 32;

fn cdiv(a: u64, b: u64) -> u64 {
    a.div_ceil(b)
}

impl<'d> Emitter<'d> {
    #[must_use]
    pub fn new(d: &'d Decls, rules: &'d FlashNextRules, dims: Dims, arena: Arena) -> Emitter<'d> {
        Emitter {
            t8_cur: None,
            t8_head: None,
            t8_post: false,
            t8_prefill: false,
            kv8: false,
            kv8_mtp: false,
            kv_mtp_table: 0,
            kv_scale: Vec::new(),
            kv_desc: 0,
            head_bf16_dropped: false,
            d,
            rules,
            dims,
            arena,
            recs: Vec::new(),
            layer: -1,
            capture: false,
            ctl: Ctl::None,
            emitted: 0,
            state_f32: false,
            next_shared: 0,
            tap_layer: -1,
            tap_moe: false,
            kv_table: 0,
            kv_ls: 0,
            taps: Vec::new(),
            declc_as_pair: 0,
            conv_l: 3,
            qsa: saber_kernels::fn19::QsaMode::default(),
            qsa_sel: None,
            vclass: VerifyClass::default(),
            route_m: None,
            glue_v1: false,
            vclass_layers: [0; 2],
            ar_decode: false,
            mtile_twin: 0,
            moe_train: MoeTrain::default(),
            moe_sync: 0,
            shared_split: 0,
            glue: None,
            df: None,
            df_saved: 0,
            moe_blocks: [0; 2],
            moe_front_blocks: 0,
            pregate: None,
            pregate_launches: 0,
            route_form: moe::RouteForm::default(),
            miss_order: moe::MissOrder::default(),
            miss_ram: (0, 0),
            miss_launches: [0; 4],
            ar_prologue: rules.hc.ar_prologue,
            rp_next: None,
            far_next: None,
            ar_far: 0,
            ar_far_saved: 0,
            mix_next: None,
            mix_red: None,
            ar_mix: 0,
            ar_mix_saved: 0,
            ar_rp: [0; 2],
            ar_rp_saved: 0,
            fuse86: rules.hc.fuse86,
            fuse86_n: [[0; 2]; 10],
            fuse86_saved: 0,
            f7_pending: None,
            tiles111: rules.hc.tiles111,
            tiles111_n: [[0; 2]; 3],
            pf122_n: [0; 2],
            pf122_dq: [0; 2],
            pf122_added: 0,
            pf124_n: [0; 2],
            sel128_n: [[0; 2]; 3],
            down131_n: [0; 2],
            k141_n: [[0; 2]; 3],
            route163_n: [[0; 2]; 2],
            pf124_twins: 0,
            pf124_twin: None,
        }
    }

    /// FN24: the rows whose CELL an op over `m` rows takes (the step's class rows while a target
    /// verify step is emitted in the width-1 class; `m` otherwise).
    #[must_use]
    pub fn rm(&self, m: u64) -> u64 {
        match self.route_m {
            Some((rows, class)) if rows == m => class,
            _ => m,
        }
    }

    /// FN85: the HC symbol this emitter runs for a planned one (FN49b's control strips the `d`).
    #[must_use]
    pub fn hc_route(&self, sym: &str) -> String {
        if self.route_form == moe::RouteForm::Shipped {
            hc_dpp_control(sym)
        } else {
            sym.to_owned()
        }
    }

    /// FN24: the class rows of a TARGET step at `cc`: a decode step of `m / w` sequences at width
    /// `w >= 2` runs SPEC 0's `m / w` in the width-1 class; everything else (a prefill chunk,
    /// SPEC 0, the `Wide` control) its own `m`.
    #[must_use]
    pub fn class_rows(&self, cc: &ChunkConsts) -> u32 {
        if cc.decode && cc.w > 1 && self.vclass == VerifyClass::Width1 {
            cc.m / cc.w
        } else {
            cc.m
        }
    }

    /// FN68: an attention / gather symbol at this emitter's page dtype (`_bf16kv` -> `_fp8kv`
    /// under `--fn-kv-dtype fp8`; the twin is named, never a fallback: a missing one refuses).
    #[must_use]
    pub fn kv_sym(&self, sym: &str) -> String {
        if self.kv8_here() {
            sym.replace("_bf16kv", "_fp8kv")
        } else {
            sym.to_string()
        }
    }

    /// FN68b: the append's (K, V) scale at the layer being emitted (1.0 = FN68's law).
    #[must_use]
    pub fn kv_append_scales(&self) -> (f32, f32) {
        usize::try_from(self.layer)
            .ok()
            .and_then(|l| self.kv_scale.get(l).copied())
            .unwrap_or((1.0, 1.0))
    }

    /// FN68b: the attention's (k_descale, v_descale) pointers at the layer being emitted for a
    /// launch indexing `entries` = `num_seqs * kv_heads` descale slots (null under bf16 or FN68's
    /// unit law: the kernels then read 1.0).
    pub fn kv_desc_ptrs(&self, entries: u64) -> anyhow::Result<(u64, u64)> {
        use super::kvscale::{KvScales, DESC_N, TABLE_LAYERS};
        if !self.kv8_here() || self.kv_desc == 0 {
            return Ok((0, 0));
        }
        let l = usize::try_from(self.layer)
            .ok()
            .filter(|&l| l < TABLE_LAYERS)
            .ok_or_else(|| anyhow::anyhow!("kv descale: layer {}", self.layer))?;
        ensure!(
            entries as usize <= DESC_N,
            "kv descale: {entries} sequence entries past the table's {DESC_N}"
        );
        Ok((
            self.kv_desc + KvScales::table_off(l, 0),
            self.kv_desc + KvScales::table_off(l, 1),
        ))
    }

    /// FN68d: the pages of the layer being emitted are e4m3 (`kv8`; the MTP layer, 48, by
    /// `kv8_mtp`).
    #[must_use]
    pub fn kv8_here(&self) -> bool {
        if self.layer == 48 {
            self.kv8_mtp
        } else {
            self.kv8
        }
    }

    /// FN68: bytes of one K/V element at this emitter's page dtype.
    #[must_use]
    pub fn kv_eb(&self) -> u64 {
        if self.kv8_here() {
            1
        } else {
            2
        }
    }

    /// FN19's step law (`saber_kernels::fn19::sparse_step`) for this emitter's `--fn-qsa` at `cc`.
    #[must_use]
    pub fn qsa_sparse(&self, cc: &ChunkConsts) -> bool {
        saber_kernels::fn19::sparse_step(self.qsa, cc.decode, cc.p0, cc.m, cc.max_ctx)
            && !(cc.decode && cc.qsa_short)
    }

    /// FN45b2: this QSA layer's K/V bindings `(block table, append slots, per-row table, its
    /// stride)`: the tier's per-layer tables, or the step's own (the flat pools).
    fn kv_binds(&self, cc: &ChunkConsts, m: u64) -> anyhow::Result<(u64, u64, u64, u64)> {
        if self.kv_table == 0 {
            ensure!(cc.kslot == 0, "a tier step table on a flat-pools boot");
            return Ok((cc.bt, cc.slot, cc.bt_rows, u64::from(cc.nb)));
        }
        ensure!(
            cc.kslot != 0,
            "the tiered KV is on but the step carries no per-layer tables (layer {})",
            self.layer
        );
        let tl = u64::from(
            super::kvwire::tier_layer(self.layer)
                .with_context(|| format!("layer {} is not a QSA layer", self.layer))?,
        );
        let kb = u64::from(super::kvwire::KBTR);
        // FN68d: the MTP layer in its own one-layer tier reads ITS table (layer 0 of it); the
        // step constants' append slots / per-row tables keep the 13-layer layout (row 12)
        let table = if self.layer == 48 && self.kv_mtp_table != 0 {
            self.kv_mtp_table + cc.kv_row0
        } else {
            self.kv_table + tl * self.kv_ls + cc.kv_row0
        };
        Ok((
            table,
            cc.kslot + tl * m * 8,
            if cc.kbtr == 0 {
                0
            } else {
                cc.kbtr + tl * m * kb * 4
            },
            kb,
        ))
    }

    pub(super) fn run(
        &mut self,
        sym: &str,
        grid: [u64; 3],
        block: u64,
        args: Vec<Arg>,
        group: Group,
    ) -> anyhow::Result<()> {
        let (plugin, _) = self.d.get(sym)?;
        let img = self.d.pack(sym, &args)?;
        self.recs.push(Rec {
            plugin,
            sym: sym.to_string(),
            grid: [grid[0] as u32, grid[1] as u32, grid[2] as u32],
            block: block as u32,
            args: img,
            group,
            layer: self.layer,
            capture: self.capture,
            shared: std::mem::take(&mut self.next_shared),
        });
        if !self.capture {
            self.emitted += 1;
        }
        Ok(())
    }

    fn s(&mut self, bytes: u64) -> anyhow::Result<u64> {
        self.arena.take(bytes, Seed::None)
    }
    fn z(&mut self, bytes: u64) -> anyhow::Result<u64> {
        if let Some(g) = self.glue.as_mut().filter(|g| g.on) {
            // FN120e: None = the run spilled -> the arena's per-layer fills (`off`'s law)
            if let Some(a) = g.take(bytes, false)? {
                return Ok(a);
            }
        }
        self.arena.take(bytes, Seed::Zero)
    }
    /// The all-ones seed law (`row_tok`'s -1), routed like [`Self::z`].
    fn o(&mut self, bytes: u64) -> anyhow::Result<u64> {
        if let Some(g) = self.glue.as_mut().filter(|g| g.on) {
            if let Some(a) = g.take(bytes, true)? {
                return Ok(a);
            }
        }
        self.arena.take(bytes, Seed::Ones)
    }
    /// FN120b (`--fn-glue fold`): this TARGET layer's route ids at their route-id-log offset
    /// inside the run's seed region (layer `l` at `l x m x 40`, the log's own layout), so ONE
    /// copy a run replaces the per-layer copy; None = the ids are an ordinary zero seed.
    fn glue_ids(&mut self, bytes: u64) -> Option<u64> {
        let l = u64::try_from(self.layer).ok()?;
        let g = self.glue.as_ref().filter(|g| g.on && g.ids_stride > 0)?;
        (l < u64::from(super::plan::LAYERS) && bytes == g.ids_stride)
            .then(|| g.zero.base + l * g.ids_stride)
    }

    /// FN120c (`--fn-dataflow gdn`): fold the records emitted since `r0` -- when they are exactly
    /// the b1 verify GDN tail (`fn86_t8_q8w8_sfx_ldc` -> `fn86_gdn_fused_update_nq8_fc_*` ->
    /// `fn111_t8_a8w8_n64_ldc`, no capture-only launch, no dynamic LDS) -- into ONE persistent
    /// launch over a host-built table: each body's virtual blocks are tasks, a stage waits for
    /// the whole previous stage (striped counters), the kernarg images are the train's own bytes.
    /// Anything else keeps the train (counted).
    fn df_fold(&mut self, r0: usize) -> anyhow::Result<()> {
        if !self.df.as_ref().is_some_and(|d| d.armed) {
            return Ok(());
        }
        let tail = &self.recs[r0..];
        let kinds: Vec<Option<(u32, u32)>> = tail.iter().map(|r| df120_kind(&r.sym)).collect();
        let fits = tail.len() == 3
            && kinds
                .iter()
                .zip(1u32..)
                .all(|(k, want)| k.is_some_and(|(kd, _)| kd == want))
            && tail
                .iter()
                .zip(&kinds)
                .all(|(r, k)| !r.capture && r.shared == 0 && k.is_some_and(|(_, t)| t == r.block))
            && u64::from(tail[1].grid[0]) <= DF120_GRID
            && tail[1].grid[1] == 1
            && tail[1].grid[2] == 1;
        let df = self.df.as_mut().context("FN120c: the tables")?;
        if !fits {
            df.sites[1] += 1;
            return Ok(());
        }
        for r in tail {
            df120_abi(self.d, &r.sym)?;
        }
        // the site: stages at 256, images from 384 (8-aligned), tasks after (16-aligned)
        let mut site = vec![0u8; DF120_SITE_BYTES as usize];
        let mut img = DF120_IMGS_AT;
        let mut counts = Vec::with_capacity(3);
        for (s, (r, k)) in tail.iter().zip(&kinds).enumerate() {
            let (kind, thr) = k.context("FN120c: a kind")?;
            img = img.div_ceil(8) * 8;
            ensure!(
                img + r.args.len() <= DF120_SITE_BYTES as usize,
                "FN120c: the site's images overflow"
            );
            site[img..img + r.args.len()].copy_from_slice(&r.args);
            let words = [
                kind,
                thr,
                r.grid[0],
                r.grid[1],
                r.grid[2],
                (img - DF120_IMGS_AT) as u32,
                0,
                0,
            ];
            let at = DF120_STAGES_AT + s * 32;
            for (j, w) in words.iter().enumerate() {
                site[at + j * 4..at + j * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
            img += r.args.len();
            counts.push(r.grid.iter().map(|&g| u64::from(g)).product::<u64>());
        }
        let tasks_at = img.div_ceil(16) * 16;
        let mut t = tasks_at;
        for (s, &n) in counts.iter().enumerate() {
            let (dep, need) = if s == 0 || df.nodep {
                (u32::MAX, 0u32)
            } else {
                (s as u32 - 1, counts[s - 1] as u32)
            };
            for vb in 0..n as u32 {
                ensure!(
                    t + 16 <= DF120_SITE_BYTES as usize,
                    "FN120c: the site's tasks overflow"
                );
                for (j, w) in [s as u32, vb, dep, need].iter().enumerate() {
                    site[t + j * 4..t + j * 4 + 4].copy_from_slice(&w.to_le_bytes());
                }
                t += 16;
            }
        }
        let ntasks: u64 = counts.iter().sum();
        let a = df.site(site)?;
        df.sites[0] += 1;
        self.recs.truncate(r0);
        self.emitted -= 3;
        self.df_saved += 2;
        self.run(
            DF120_KERNEL,
            [DF120_GRID, 1, 1],
            256,
            vec![
                Arg::P(a + DF120_STAGES_AT as u64),
                Arg::P(a + tasks_at as u64),
                Arg::I(ntasks as i32),
                Arg::P(a + DF120_IMGS_AT as u64),
                Arg::P(a),
                Arg::I(3),
                Arg::L(DF120_TIMEOUT_TICKS),
            ],
            Group::Core,
        )
    }

    /// Launches of the forward emitted so far (capture excluded).
    #[must_use]
    pub fn forward_launches(&self) -> u32 {
        self.emitted
    }

    /// FN47b: one capture router GEMM (`w` on the `[m, H]` bf16 stream `a` into `c`), counted as
    /// a capture launch, not a forward one.
    fn pregate_gemm(&mut self, a: u64, w: u64, c: u64, m: u64) -> anyhow::Result<()> {
        let n0 = self.emitted;
        self.gemm("moe.router", a, w, c, m, 512, H, Group::Moe)?;
        let d = self.emitted - n0;
        self.emitted = n0;
        self.pregate_launches += u64::from(d);
        Ok(())
    }

    // ------------------------------------------------------------------------ bf16 GEMM
    /// `C[m, n] = A[m, k] . W[n, k]^T` on FN5's route for `shape`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &mut self,
        shape: &str,
        a: u64,
        w: u64,
        c: u64,
        m: u64,
        n: u64,
        k: u64,
        g: Group,
    ) -> anyhow::Result<()> {
        // FN24: the CELL is the class rows' (== m unless a width-1-class verify step); the launch
        // (grid, partials) is m's -- every bf16 GEMM symbol tiles any M (`band_rows`)
        let (l, sp) = self
            .rules
            .bgemm_route(shape, self.rm(m) as u32, n as u32)
            .with_context(|| format!("bgemm {shape} m {m} n {n}"))?;
        // the class cell's symbol at m rows (the m16 tile's row-slice twin above 16 rows)
        let sym = saber_kernels::flashnext::class_symbol_at(&l.symbol, m as u32);
        if sym != l.symbol {
            self.mtile_twin += 1;
        }
        self.gemm_sym(sym, a, w, c, m, n, k, u64::from(sp), g)
    }

    /// FN33: `shape`'s bf16 GEMM CELL at `class_m` rows (its N = `full_n`, a rules shape), launched
    /// at `m` rows x `n` columns -- the drafter's exact rerank runs the head's own symbol on the
    /// candidates' gathered rows: every routed bf16 symbol walks one output element's K in the same
    /// ascending steps into one accumulator whatever M, N or the column (split 1 at every head
    /// band; asserted), so a candidate's score is the full head's logit bit for bit. Returns the
    /// symbol.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_cell_at(
        &mut self,
        shape: &str,
        class_m: u64,
        full_n: u64,
        a: u64,
        w: u64,
        c: u64,
        m: u64,
        n: u64,
        k: u64,
        g: Group,
    ) -> anyhow::Result<String> {
        let (l, sp) = self
            .rules
            .bgemm_route(shape, self.rm(class_m) as u32, full_n as u32)
            .with_context(|| format!("bgemm {shape} m {class_m} n {full_n}"))?;
        ensure!(
            sp == 1,
            "the rerank needs a one-split head cell ({shape} m {class_m}: {sp})"
        );
        let sym = saber_kernels::flashnext::class_symbol_at(&l.symbol, m as u32).to_string();
        self.gemm_sym(&sym, a, w, c, m, n, k, 1, g)?;
        Ok(sym)
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_sym(
        &mut self,
        sym: &str,
        a: u64,
        w: u64,
        c: u64,
        m: u64,
        n: u64,
        k: u64,
        sp: u64,
        g: Group,
    ) -> anyhow::Result<()> {
        let (rows, cols, thr) =
            saber_kernels::flashnext::band_rows(sym).with_context(|| sym.to_string())?;
        let grid = [cdiv(n, u64::from(cols)), cdiv(m, u64::from(rows)), 1];
        let mi = |x: u64| Arg::I(x as i32);
        if sym.starts_with("fn_bgemm_sk_") {
            // FN141 (`--fn-k141 router`): the router's 80-step slices on the 16-step ring
            // (same arguments, grid, partials and counters)
            let sym = if saber_kernels::flashnext::k141::router_takes(sym, n, k, sp)
                && self.k141_site(saber_kernels::flashnext::k141::ROUTER)
            {
                saber_kernels::flashnext::k141::ROUTER_SYM
            } else {
                sym
            };
            let (part, cnt) = if sp > 1 {
                (self.s(sp * m * n * 4)?, self.z(grid[0] * grid[1] * 4)?)
            } else {
                (0, 0)
            };
            self.run(
                sym,
                [grid[0], grid[1], sp],
                u64::from(thr),
                vec![
                    Arg::P(a),
                    Arg::P(w),
                    Arg::P(part),
                    Arg::P(cnt),
                    Arg::P(c),
                    mi(m),
                    mi(n),
                    mi(k),
                ],
                g,
            )
        } else {
            self.run(
                sym,
                grid,
                u64::from(thr),
                vec![Arg::P(a), Arg::P(w), Arg::P(c), mi(m), mi(n), mi(k)],
                g,
            )
        }
    }

    // ------------------------------------------------------------------------ FN61c W8 trunk
    /// FN61c: `C[m, n] (row stride ldc) = A[m, k] . W8[n, k]^T` at fp8 W8A8 -- the per-128
    /// activation quant (`dflash_act_quant_g128`) then the drafter's block-scale GEMM
    /// (`dflash_gemm_a8w8_bs128_*`, unchanged) when `ldc == n`, its output-stride twin
    /// (`fn_t8_a8w8_bs128_ldc`) otherwise. Every output row's bytes depend on its own row only (the
    /// quant is per row, the GEMM's k walk is fixed): no class, no split. 2 launches.
    #[allow(clippy::too_many_arguments)]
    pub fn w8_gemm(
        &mut self,
        a: u64,
        w: u64,
        ws: u64,
        c: u64,
        m: u64,
        n: u64,
        k: u64,
        ldc: u64,
        g: Group,
    ) -> anyhow::Result<()> {
        use saber_kernels::fn50;
        ensure!(
            n % 128 == 0 && k % 128 == 0,
            "W8 GEMM {m} x {n} x {k}: 128 tiles"
        );
        let (q, s) = (self.s(m * k)?, self.s(m * (k / 128) * 4)?);
        self.run(
            fn50::ACT_QUANT_SYMBOL,
            [fn50::act_quant_blocks(m, k), 1, 1],
            fn50::ACT_QUANT_THREADS,
            vec![
                Arg::P(a),
                Arg::P(q),
                Arg::P(s),
                Arg::I(m as i32),
                Arg::I(k as i32),
            ],
            g,
        )?;
        self.w8_gemm_q(q, s, w, ws, c, m, n, k, ldc, g)
    }

    /// FN86b: the W8 GEMM of [`Self::w8_gemm`] on an activation a fused producer already
    /// quantised (`q` e4m3 `m x k`, `s` the fp32 scales a 128 group) -- the GEMM launch alone.
    #[allow(clippy::too_many_arguments)]
    pub fn w8_gemm_q(
        &mut self,
        q: u64,
        s: u64,
        w: u64,
        ws: u64,
        c: u64,
        m: u64,
        n: u64,
        k: u64,
        ldc: u64,
        g: Group,
    ) -> anyhow::Result<()> {
        let mut args = vec![
            Arg::P(q),
            Arg::P(w),
            Arg::P(s),
            Arg::P(ws),
            Arg::P(c),
            Arg::I(m as i32),
            Arg::I(n as i32),
            Arg::I(k as i32),
        ];
        // FN61e: <= 16 rows = FN61c's cells; past them the multi-row tiles (same bytes, gated)
        let (sym, grid, thr, with_ldc) = t8_cell(m, n, ldc);
        // FN111 t1: the dflash m16 cell (out / o projections) on 64-column tiles, byte-equal
        {
            use saber_kernels::flashnext::{k141, tiles111 as t111};
            if sym == "dflash_gemm_a8w8_bs128_m16n128k128"
                && n <= t111::T1_MAX_N
                && n % 64 == 0
                && self.tile111_site(t111::T1)
            {
                args.push(Arg::I(ldc as i32));
                // FN141 (`--fn-k141 t8`): the n64 tile as a register ring (same arguments, grid)
                let s = if k141::t8_takes(t111::SYMS[0], n, k, m) && self.k141_site(k141::T8) {
                    k141::T8_SYM
                } else {
                    t111::SYMS[0]
                };
                let s = self.mix_take(s, m, n, ldc, &mut args)?;
                return self.run(&s, [n / 64, cdiv(m, 16), 1], 128, args, g);
            }
        }
        if with_ldc {
            args.push(Arg::I(ldc as i32));
        }
        let sym = self.mix_take(sym, m, n, ldc, &mut args)?;
        self.run(&sym, grid, thr, args, g)
    }

    /// FN164: at the out / o projection (N = ldc = 2,560) of a layer whose mixer-out reduce rides
    /// it (`mix_next` set by [`emit_layer`]), the chosen cell's fused twin + the channel blob whose
    /// SUM is a fresh scratch of the reduce's size (the address the standalone reduce's output
    /// takes: the arena's next scratch after the GEMM's); else the cell unchanged.
    fn mix_take(
        &mut self,
        sym: &str,
        m: u64,
        n: u64,
        ldc: u64,
        args: &mut Vec<Arg>,
    ) -> anyhow::Result<String> {
        if n != H || ldc != H || self.mix_next.is_none() {
            return Ok(sym.to_string());
        }
        let ch = self.mix_next.take().context("FN164: the channel")?;
        let far = super::dev::mix_symbol(sym).with_context(|| {
            format!("--fn-mixer-ar-push: the out / o projection at {m} rows ran {sym}, which has no fused twin")
        })?;
        ensure!(
            args.len() == 9,
            "--fn-mixer-ar-push: {sym} takes the ldc signature"
        );
        let red = self.s(m * H * 2)?;
        args.push(Arg::Blob(ch.blob(red)));
        self.mix_red = Some(red);
        self.ar_mix += 1;
        Ok(far)
    }

    /// FN61c: the bf16 suffix of a merged projection at its row stride (`fn_t8_bgemm_m16_ldc`:
    /// FN5's `fn_bgemm_sk_m16` at one split + ldc). 1 launch.
    #[allow(clippy::too_many_arguments)]
    pub fn bgemm_ldc(
        &mut self,
        a: u64,
        w: u64,
        c: u64,
        m: u64,
        n: u64,
        k: u64,
        ldc: u64,
        g: Group,
    ) -> anyhow::Result<()> {
        let mi = |x: u64| Arg::I(x as i32);
        self.run(
            T8_BGEMM_LDC,
            [cdiv(n, 64), cdiv(m, 16), 1],
            128,
            vec![
                Arg::P(a),
                Arg::P(w),
                Arg::P(c),
                mi(m),
                mi(n),
                mi(k),
                mi(ldc),
            ],
            g,
        )
    }

    /// FN86b: one site of a launch-fusion family -- whether it runs fused (the family is on), and
    /// the which-path count (`fnfuse86`: fused sites / chain sites; each fused site of the built
    /// families removes ONE launch from the layer).
    pub fn fuse86_site(&mut self, fam: u32) -> bool {
        self.fuse86_site_n(fam, 1)
    }

    /// [`Self::fuse86_site`] for a family that removes `saved` launches at a site (ATTN-OPT-3's f6: 2).
    pub fn fuse86_site_n(&mut self, fam: u32, saved: u32) -> bool {
        let i = fam.trailing_zeros() as usize;
        let on = self.fuse86 & fam != 0;
        self.fuse86_n[i][usize::from(!on)] += 1;
        if on {
            self.fuse86_saved += saved;
        }
        on
    }

    /// ATTN-OPT-3 family 7: whether this decode attention's combine is left to the gate (a TARGET
    /// layer whose gate is family 1's q8 site), counted at the site.
    fn f7_site(&mut self, cc: &ChunkConsts) -> bool {
        use saber_kernels::flashnext::fuse86 as f86;
        cc.decode
            && self.t8_cur.is_some()
            && self.fuse86 & f86::F1 != 0
            && self.rules.kernels.attn_gate_bf16 == f86::ATTN_GATE
            && self.fuse86_site(f86::F7)
    }

    /// FN141: one gated site of a drained-ring family -- whether its fn141_ symbol runs, counted
    /// (`fnk141`).
    pub fn k141_site(&mut self, fam: u32) -> bool {
        let i = fam.trailing_zeros() as usize;
        let on = self.rules.hc.k141 & fam != 0;
        self.k141_n[i][usize::from(!on)] += 1;
        on
    }

    /// FN164: whether the out / o projection (N = ldc = 2,560) at `m` rows runs a cell with a fused
    /// mixer-out twin ([`super::dev::MIX_PARENTS`]) -- [`Self::w8_gemm_q`]'s pick without counting:
    /// <= 16 rows FN111's n64 tile (or FN141's ring over it) when the t1 re-tile is on, 17..64 rows
    /// FN61e's d2w4 / d4w4 tiles; the m16 cell (t1 off) and prefill tiles have none.
    #[must_use]
    pub fn mix_cell_fuses(&self, m: u64) -> bool {
        use saber_kernels::flashnext::tiles111 as t111;
        match m {
            0 => false,
            1..=16 => self.tiles111 & t111::T1 != 0 && H <= t111::T1_MAX_N && H % 64 == 0,
            17..=64 => matches!(
                t8_cell(m, H, H).0,
                "fn_t8p_a8w8_d2w4_ldc" | "fn_t8p_a8w8_d4w4_ldc"
            ),
            _ => false,
        }
    }

    /// FN163: one gated site of a route band -- whether its fn163_ symbol runs, counted
    /// (`fnroute163`).
    pub fn route163_site(&mut self, band: u32) -> bool {
        let i = band.trailing_zeros() as usize;
        let on = self.rules.hc.route163 & band != 0;
        self.route163_n[i][usize::from(!on)] += 1;
        on
    }

    /// FN111: one site of a re-tile -- whether its fn111_ symbol runs, counted (`fntile111`).
    pub fn tile111_site(&mut self, t: u32) -> bool {
        let i = t.trailing_zeros() as usize;
        let on = self.tiles111 & t != 0;
        self.tiles111_n[i][usize::from(!on)] += 1;
        on
    }

    /// FN86b family 4: a merged projection = the W8 prefix (`n8` rows of `t`, at row stride `ldc`)
    /// and the bf16 suffix (`ns` rows of `sfx`, written at column `n8`) of the same input `a`.
    /// Off: quant, W8 GEMM, suffix GEMM (3 launches); on: `fn86_q8_bgemm_m16_ldc` (the quant's
    /// workgroups and the suffix's in one launch), then the W8 GEMM (2).
    #[allow(clippy::too_many_arguments)]
    pub fn w8_merged(
        &mut self,
        a: u64,
        t: &T8W,
        sfx: u64,
        o: u64,
        m: u64,
        n8: u64,
        ns: u64,
        k: u64,
        ldc: u64,
    ) -> anyhow::Result<()> {
        use saber_kernels::fn50;
        if !self.fuse86_site(saber_kernels::flashnext::fuse86::F4) {
            self.w8_gemm(a, t.w, t.s, o, m, n8, k, ldc, Group::Proj)?;
            return self.bgemm_ldc(a, sfx, o + n8 * 2, m, ns, k, ldc, Group::Proj);
        }
        ensure!(
            n8 % 128 == 0 && k % 128 == 0,
            "W8 GEMM {m} x {n8} x {k}: 128 tiles"
        );
        let (q, s) = (self.s(m * k)?, self.s(m * (k / 128) * 4)?);
        // FN86c family 2 (composes on family 4): the act quant in the W8 GEMM's A staging and the
        // suffix as its extra workgroups -- ONE launch; `q` / `s` stay taken (the arena unchanged)
        if m <= saber_kernels::flashnext::fuse86::F2_MAX_ROWS
            && self.fuse86_site(saber_kernels::flashnext::fuse86::F2)
        {
            let _ = (q, s);
            // FN111 t2: the W8 prefix on 64-column tiles (byte-equal; missed the bar, not in all)
            let t2 = n8 % 64 == 0 && self.tile111_site(saber_kernels::flashnext::tiles111::T2);
            let nbx8 = if t2 { n8 / 64 } else { n8 / 128 };
            let mi = |x: u64| Arg::I(x as i32);
            return self.run(
                if t2 {
                    saber_kernels::flashnext::tiles111::SYMS[1]
                } else {
                    saber_kernels::flashnext::fuse86::Q8W8_SFX
                },
                [nbx8 + cdiv(ns, 64), cdiv(m, 16), 1],
                128,
                vec![
                    Arg::P(a),
                    Arg::P(t.w),
                    Arg::P(t.s),
                    Arg::P(o),
                    mi(m),
                    mi(n8),
                    mi(k),
                    mi(ldc),
                    mi(nbx8),
                    Arg::P(sfx),
                    Arg::P(o + n8 * 2),
                    mi(ns),
                ],
                Group::Proj,
            );
        }
        let nqb = fn50::act_quant_blocks(m, k);
        let (nbx, nby) = (cdiv(ns, 64), cdiv(m, 16));
        ensure!(
            fn50::ACT_QUANT_THREADS == 128,
            "family 4 runs the quant's 128 threads"
        );
        let mi = |x: u64| Arg::I(x as i32);
        self.run(
            saber_kernels::flashnext::fuse86::Q8_BGEMM_LDC,
            [nqb + nbx * nby, 1, 1],
            128,
            vec![
                Arg::P(a),
                Arg::P(q),
                Arg::P(s),
                mi(m),
                mi(k),
                mi(nqb),
                Arg::P(sfx),
                Arg::P(o + n8 * 2),
                mi(ns),
                mi(ldc),
                mi(nbx),
            ],
            Group::Proj,
        )?;
        self.w8_gemm_q(q, s, t.w, t.s, o, m, n8, k, ldc, Group::Proj)
    }

    /// FN122b (`--fn-prefill-tiles hc`): a planned fp8-mixer prefill down on the shipped
    /// `m2w21t1` cfg (sps 1, the class) runs `flashnext_hc8p`'s tile for its row count
    /// (`prefill122::hc_cfg`): the same body at another row block / waves / A-slab chunk, byte for
    /// byte (FN122b twin); the grid's row blocks and the counters follow the tile. Counted
    /// (`fnpfx122`) at every site the shipped cfg was planned.
    fn hc8_prefill_down(&mut self, p: &mut saber_kernels::flashnext::MixPlan, m: u64) {
        use saber_kernels::flashnext::prefill122 as p122;
        let shipped = format!("_v2_{}", p122::HC_SHIPPED_CFG);
        let Some(base) = p.down.symbol.strip_suffix(&shipped) else {
            return;
        };
        if p.sps != 1 {
            return;
        }
        let pick = (self.rules.hc.prefill122 & p122::HC != 0)
            .then(|| p122::hc_cfg(m as u32))
            .flatten();
        let Some((cfg, mf, waves, groups)) = pick else {
            self.pf122_n[1] += 1;
            return;
        };
        self.pf122_n[0] += 1;
        let blocks = (m as u32).div_ceil(16 * mf);
        p.down.symbol = format!("{base}_v2_{cfg}");
        p.down.grid = [p.down.grid[0], blocks, groups];
        p.down.threads = 32 * waves;
        p.down_counters = blocks * groups;
    }

    /// FN122c (`--fn-prefill-tiles dq`): a fp8-mixer mix of >= `DQ_MIN_ROWS` rows whose planned
    /// down is the shipped `m2w21t1` (sps 1) and whose planned up is the bf16 prefill `p4s` runs
    /// both on the BF16 bodies over its matrices dequantized once (`hc8_dequant`; FN108's law: an
    /// `_h8` symbol on a record == its bf16 parent on bf16(code x scale)): the down on
    /// `flashnext_hc8p`'s q body (`DQ_DOWN`), the up on the shipped `p4s`. A producer-rr mix (no
    /// qb rr twin) keeps the fp8 forms. Returns (down, up) dequantized; counted (`fnpfx122`).
    fn hc8_dequant_route(
        &mut self,
        p: &mut saber_kernels::flashnext::MixPlan,
        m: u64,
        no_pre: bool,
    ) -> (bool, bool) {
        use saber_kernels::flashnext::prefill122 as p122;
        if self.rules.hc.prefill122 & p122::DQ == 0
            || m < u64::from(p122::dq_min_rows(self.rules.hc.prefill122))
            || p.sps != 1
        {
            return (false, false);
        }
        let shipped = format!("_v2_{}", p122::HC_SHIPPED_CFG);
        let down = no_pre && p.down.symbol.ends_with(&shipped);
        if down {
            let base = p
                .down
                .symbol
                .strip_suffix(&shipped)
                .expect("checked")
                .to_string();
            let (cfg, mf, waves, groups) = p122::DQ_DOWN;
            let blocks = (m as u32).div_ceil(16 * mf);
            p.down.symbol = format!("{base}_v2_{cfg}");
            p.down.grid = [p.down.grid[0], blocks, groups];
            p.down.threads = 32 * waves;
            p.down_counters = blocks * groups;
            self.pf122_dq[0] += 1;
        }
        let up = p.up.symbol.ends_with(&format!("_v2_{}", p122::DQ_UP_CFG));
        if up {
            self.pf122_dq[1] += 1;
        }
        (down, up)
    }

    /// FN122c: an hc8 record (e4m3 codes [rows, cols] + f32 scales [rows, cols / g]) as the bf16
    /// matrix its `_h8` symbols multiply, in this pass's scratch -- one `fn122_hc8_dequant`
    /// launch (counted in `pf122_added` for the layer's launch law).
    fn hc8_dequant(&mut self, rec: u64, rows: u32, cols: u32, g: u32) -> anyhow::Result<u64> {
        use saber_kernels::flashnext::prefill122 as p122;
        let n = u64::from(rows) * u64::from(cols);
        let out = self.s(n * 2)?;
        self.run(
            p122::DEQUANT,
            [cdiv(n / 8, 256), 1, 1],
            256,
            vec![
                Arg::P(rec),
                Arg::P(out),
                Arg::I(rows as i32),
                Arg::I(cols as i32),
                Arg::I(g as i32),
            ],
            Group::Hc,
        )?;
        self.pf122_added += 1;
        Ok(out)
    }

    // ------------------------------------------------------------------------ HC mix (v2 routes)
    /// One mix of `m` rows. `hid_out` / `injo_out`: where the down writes the stream and the next
    /// injection logits (the carry region for the MoE side, scratch otherwise).
    #[allow(clippy::too_many_arguments)]
    pub fn mix(
        &mut self,
        hc: &HcW,
        mode: MixMode,
        out: MixOut,
        m: u64,
        src: u64,
        blk: Option<u64>,
        inj: Option<u64>,
        carry: bool,
        pre: Option<HcPre>,
    ) -> anyhow::Result<MixOutBufs> {
        let mut p = self
            .rules
            .mix_plan_at(mode, out, self.rm(m) as u32, m as u32)?;
        // FN49b: the control plans the HC cfgs without FN49b's DPP `d` (same grid / threads)
        if self.route_form == moe::RouteForm::Shipped {
            p.down.symbol = hc_dpp_control(&p.down.symbol);
            p.up.symbol = hc_dpp_control(&p.up.symbol);
        }
        // FN108: the fp8 mixers' up has no p4s twin (it spills at MF 4): its rows run on p2s_h8
        // (FN21: p2s and p4s are both byte-identical to FN5's up) -- 32-row blocks, same columns
        // FN122c (`dq`): the down / up on the bf16 bodies over the matrices dequantized once a mix
        let (dq_down, dq_up) = if hc.w8 {
            self.hc8_dequant_route(&mut p, m, pre.is_none())
        } else {
            (false, false)
        };
        if hc.w8 && !dq_up {
            hc8_up_plan(&mut p.up, &mut p.up_counters, m);
        }
        if hc.w8 && !dq_down {
            self.hc8_prefill_down(&mut p, m);
        }
        // FN85: this mix's `blk` is THIS rank's partial; the down's `_rp` twin reduces it in its
        // prologue (the same vadd as the one-shot, the exchange over `rp.ch`)
        let rp = self.rp_next.take();
        ensure!(
            !(hc.w8 && rp.is_some()),
            "the prologue-reduce mix (FN85) has no fp8-mixer twin (--fn-trunk w8-hc8)"
        );
        if rp.is_some() {
            ensure!(
                mode == MixMode::Combine && pre.is_none() && blk.is_some(),
                "a prologue-reduce mix is a combine mix of a partial (no producer rr)"
            );
        }
        // FN37: the producer wrote x' and rr -> the down's norm-free twin reads them (its `src` is
        // x', read as a materialized stream; `blk` / `hid` unused)
        let rrsym = match pre {
            Some(_) => {
                ensure!(
                    matches!(mode, MixMode::Combine | MixMode::Final),
                    "a producer-rr mix is a combine / final mix"
                );
                Some(rr_down_symbol(&p.down.symbol).with_context(|| {
                    format!("no rr twin of the planned down `{}`", p.down.symbol)
                })?)
            }
            None => None,
        };
        let (hid, injo) = match (pre, carry) {
            (Some(x), true) => {
                let slot = (self.layer.max(0) as usize) % 2;
                (x.x, self.arena.carry[slot].take(m * 4 * 2)?)
            }
            (Some(x), false) => (x.x, self.s(m * 4 * 2)?),
            (None, true) => {
                let slot = (self.layer.max(0) as usize) % 2;
                (
                    self.arena.carry[slot].take(m * W * 2)?,
                    self.arena.carry[slot].take(m * 4 * 2)?,
                )
            }
            (None, false) => (self.s(m * W * 2)?, self.s(m * 4 * 2)?),
        };
        let part = self.s(p.part_bytes)?;
        let rr = match pre {
            Some(x) => x.rr,
            None => self.s(m * 4 * 4)?,
        };
        let (src, blk) = match pre {
            Some(x) => (x.x, Some(x.x)),
            None => (src, blk),
        };
        let cnta = self.z(u64::from(p.down_counters) * 4)?;
        let lora = self.s(m * 320 * 2)?;
        let bi = self.s(m * H * 2)?;
        let (q8, scale, amax, cntb) = if out == MixOut::Bf16 {
            (0, 0, 0, 0)
        } else {
            (
                self.s(p.q8_bytes)?,
                self.s(m * 4)?,
                self.s(p.amax_bytes)?,
                self.z(u64::from(p.up_counters) * 4)?,
            )
        };
        let fin = mode == MixMode::Final;
        // FN122c: the mix's fp8 matrices as bf16, once (each launch = bf16(code x scale))
        let wdown = if dq_down {
            self.hc8_dequant(hc.wdown, 320, W as u32, 128)?
        } else {
            hc.wdown
        };
        let wup = if dq_up {
            self.hc8_dequant(hc.wup, W as u32, 320, 80)?
        } else {
            hc.wup
        };
        let d = &p.down;
        let mut dargs = vec![
            Arg::P(src),
            Arg::P(blk.unwrap_or(src)),
            Arg::P(inj.unwrap_or(injo)),
            Arg::P(hid),
            Arg::P(hc.w),
            Arg::P(wdown),
            Arg::P(if fin { 0 } else { hc.winj }),
            Arg::P(part),
            Arg::P(rr),
            Arg::P(cnta),
            Arg::P(lora),
            Arg::P(injo),
            Arg::I(m as i32),
            Arg::I(p.sps as i32),
            Arg::F(EPS),
        ];
        let dsym = match rp {
            Some(x) => {
                let c = x.ch;
                dargs.extend([
                    Arg::P(c.peer_scr),
                    Arg::P(c.my_scr),
                    Arg::P(x.out),
                    Arg::P(c.peer_flags),
                    Arg::P(c.my_flags),
                    Arg::P(c.ctl),
                    Arg::I(c.stride16 as i32),
                    Arg::I(c.drain),
                    Arg::I(c.acq),
                ]);
                rp_down_symbol(&d.symbol)
            }
            None => rrsym.unwrap_or_else(|| d.symbol.clone()),
        };
        // FN108: the planned symbol's fp8-mixer twin (same arguments, `wdown` / `wup` = the records)
        let dsym = hc8_symbol(&dsym, hc.w8 && !dq_down);
        // FN131 (`--fn-hc-down split`): a routed decode down runs its phase-split twin (the same
        // arguments and grid, more threads; byte-equal) -- counted at every routed site
        let (dsym, dthreads) = match saber_kernels::flashnext::down131::route(&dsym, p.sps) {
            Some((new, thr)) => {
                let on = self.rules.hc.down131 & saber_kernels::flashnext::down131::SPLIT != 0;
                self.down131_n[usize::from(!on)] += 1;
                if on {
                    (new, thr)
                } else {
                    (dsym, d.threads)
                }
            }
            None => (dsym, d.threads),
        };
        self.run(
            &dsym,
            [
                u64::from(d.grid[0]),
                u64::from(d.grid[1]),
                u64::from(d.grid[2]),
            ],
            u64::from(dthreads),
            dargs,
            Group::Hc,
        )?;
        let hid = if mode == MixMode::Mix { src } else { hid };
        let u = &p.up;
        let usym = hc8_symbol(&u.symbol, hc.w8 && !dq_up);
        // FN141 (`--fn-k141 up`): the decode s1d up on its twin (same arguments, grid, threads)
        let usym = match saber_kernels::flashnext::k141::up_route(&usym) {
            Some(new) if self.k141_site(saber_kernels::flashnext::k141::UP) => new.to_string(),
            _ => usym,
        };
        self.run(
            &usym,
            [
                u64::from(u.grid[0]),
                u64::from(u.grid[1]),
                u64::from(u.grid[2]),
            ],
            u64::from(u.threads),
            vec![
                Arg::P(hid),
                Arg::P(rr),
                Arg::P(hc.w),
                Arg::P(lora),
                Arg::P(wup),
                Arg::P(bi),
                Arg::P(q8),
                Arg::P(scale),
                Arg::P(amax),
                Arg::P(cntb),
                Arg::P(0),
                Arg::I(m as i32),
            ],
            Group::Hc,
        )?;
        Ok(MixOutBufs {
            hid,
            injo,
            bi,
            q8,
            scale,
        })
    }

    /// The materialized residual (capture only): `bf16(res + blk * 2 sigmoid(inj / 4))`.
    pub fn materialize(&mut self, p: &Pending, m: u64, out: u64) -> anyhow::Result<()> {
        let was = self.capture;
        self.capture = true;
        let r = self.run(
            "fn_hc_combine",
            [cdiv(m * W / 8, 256), 1, 1],
            256,
            vec![
                Arg::P(p.hid),
                Arg::P(p.mo),
                Arg::P(p.injo),
                Arg::P(out),
                Arg::I(m as i32),
            ],
            Group::Hc,
        );
        self.capture = was;
        r
    }

    // ------------------------------------------------------------------------ GDN core
    fn gdn_core(
        &mut self,
        mx: &MixerW,
        bi: u64,
        m: u64,
        st: &LayerState,
        cc: &ChunkConsts,
    ) -> anyhow::Result<u64> {
        let MixerW::Gdn {
            win,
            conv,
            alog,
            dtb,
            nw,
            wout,
        } = *mx
        else {
            unreachable!()
        };
        let dm = self.dims;
        let (gn, hv, hk) = (dm.gn, dm.hv, dm.hk);
        // FN120c: the GDN tail's first record (`df_fold` takes the records from here)
        let r0 = self.recs.len();
        let o = self.s(m * gn * 2)?;
        if let Some(t) = self.t8_cur {
            // FN61c: qkvz at W8A8 into the merged row (stride gn), in_proj_ba bf16 after it
            let pre = dm.conv + dm.zn;
            ensure!(t.n == pre, "W8 in_proj rows {} vs {pre}", t.n);
            // FN86b family 4: quant + suffix in one launch when on
            let sfx = if t.sfx != 0 { t.sfx } else { win + pre * H * 2 };
            self.w8_merged(bi, &t, sfx, o, m, pre, gn - pre, H, gn)?;
        } else {
            self.gemm(
                &format!("gdn.in_proj.{}", dm.tps()),
                bi,
                win,
                o,
                m,
                gn,
                H,
                Group::Proj,
            )?;
        }
        let q = self.s(m * hk * 128 * 2)?;
        let k = self.s(m * hk * 128 * 2)?;
        let v = self.s(m * hv * 128 * 2)?;
        let g = self.s(m * hv * 4)?;
        let beta = self.s(m * hv * 4)?;
        let a = self.s(m * hv * 64 * 2)?;
        let core = self.s(m * hv * 128 * 2)?;
        let nch = cdiv(m, 64) + 1;
        let (b_off, a_off) = (dm.conv + dm.zn, dm.conv + dm.zn + hv);
        let l = |x: u64| Arg::L(x as i64);
        // FN86c family 5: the fc decode update with family 1's norm + act quant in its tail (one
        // launch fewer); the norm's buffers are taken here, in the chain's order (nrm, out, q8, s8:
        // nothing between the update and the norm takes arena bytes)
        let zs = if self.ctl == Ctl::SwapArg {
            gn + 128
        } else {
            gn
        };
        let f5 = cc.decode
            && self.t8_cur.is_some()
            && self.rules.kernels.gdn_norm_bf16_sigmoid
                == saber_kernels::flashnext::fuse86::GDN_NORM
            && self.fuse86 & saber_kernels::flashnext::fuse86::F1 != 0
            && self.gdn_decode_fc(m, cc)
            && self.fuse86_site(saber_kernels::flashnext::fuse86::F5);
        let f5bufs = if f5 {
            ensure!(
                dm.zn == hv * 128,
                "family 5: the norm's N {} is the update's {hv} heads x 128",
                dm.zn
            );
            let nrm = self.s(m * dm.zn * 2)?;
            let out = self.s(m * H * 2)?;
            let (q8, s8) = (self.s(m * dm.zn)?, self.s(m * (dm.zn / 128) * 4)?);
            Some((nrm, out, q8, s8))
        } else {
            None
        };
        if cc.decode {
            let nq = f5bufs.map(|(nrm, _, q8, s8)| F5Tail {
                z: o + 2 * dm.conv,
                zs,
                nw,
                nrm,
                q8,
                s8,
            });
            self.gdn_decode(mx, o, q, k, v, core, m, st, cc, nq)?;
        } else {
            self.run(
                "saber_gdn_conv_prep_w4_h128_hpb8_ab1",
                [nch, dm.conv / 1024, 1],
                256,
                vec![
                    Arg::P(o),
                    l(gn),
                    Arg::P(conv),
                    Arg::P(0),
                    Arg::P(st.cst),
                    l(self.conv_l * dm.conv),
                    l(self.conv_l),
                    l(1),
                    Arg::P(cc.zero1),
                    l(1),
                    Arg::P(cc.hi8),
                    Arg::P(o + 2 * a_off),
                    Arg::P(o + 2 * b_off),
                    l(gn),
                    Arg::P(alog),
                    Arg::P(dtb),
                    Arg::P(q),
                    Arg::P(k),
                    Arg::P(v),
                    Arg::P(g),
                    Arg::P(beta),
                    Arg::P(cc.cu),
                    Arg::I(1),
                    Arg::I(hv as i32),
                    Arg::I(hk as i32),
                    Arg::F(20.0),
                ],
                Group::Core,
            )?;
            self.run(
                "saber_gdn_kkt_solve_k128_c64_r3",
                [nch, hk, 1],
                128,
                vec![
                    Arg::P(k),
                    Arg::P(beta),
                    Arg::P(g),
                    Arg::P(a),
                    Arg::P(cc.cu),
                    Arg::I(1),
                    Arg::I(hv as i32),
                    Arg::I(hk as i32),
                ],
                Group::Core,
            )?;
            let h0 = st.h[(cc.chunk % 2) as usize];
            let ht = st.h[((cc.chunk + 1) % 2) as usize];
            self.run(
                if self.ctl == Ctl::CertifiedScan {
                    ensure!(
                        !self.state_f32,
                        "the certified-scan control runs the f16 state"
                    );
                    "saber_gdn_chunk_scan_f16s_k128_v128_c64"
                } else if self.state_f32 {
                    "fn_gdn_chunk_scan_xg_k128_v128_c64"
                } else {
                    "fn_gdn_chunk_scan_xg_f16s_k128_v128_c64"
                },
                [2, hv, 1],
                256,
                vec![
                    Arg::P(q),
                    Arg::P(k),
                    Arg::P(v),
                    Arg::P(a),
                    Arg::P(g),
                    Arg::P(beta),
                    Arg::P(h0),
                    Arg::P(core),
                    Arg::P(ht),
                    Arg::P(cc.cu),
                    Arg::I(hv as i32),
                    Arg::I(hk as i32),
                    Arg::F(1.0 / (128f32).sqrt()),
                ],
                Group::Core,
            )?;
            if self.layer == self.tap_layer {
                let sb = if self.state_f32 { 4 } else { 2 };
                for (n, a_, b_) in [
                    ("bi", bi, m * H * 2),
                    ("inproj", o, m * gn * 2),
                    ("q", q, m * hk * 128 * 2),
                    ("k", k, m * hk * 128 * 2),
                    ("v", v, m * hv * 128 * 2),
                    ("g", g, m * hv * 4),
                    ("beta", beta, m * hv * 4),
                    ("a", a, m * hv * 64 * 2),
                    ("h0", h0, hv * 128 * 128 * sb),
                    ("ht", ht, hv * 128 * 128 * sb),
                    ("core", core, m * hv * 128 * 2),
                ] {
                    self.taps.push((n.to_string(), a_, b_));
                }
            }
        }
        let nrm = match f5bufs {
            Some((nrm, ..)) => nrm,
            None => self.s(m * dm.zn * 2)?,
        };
        // FN86b family 1: the gated norm + the W8 out_proj's act quant as ONE launch (the norm's
        // bf16 output still stored; the arena takes nrm, out, q, s in the chain's order)
        if let Some(t) = self.t8_cur.filter(|_| {
            self.rules.kernels.gdn_norm_bf16_sigmoid == saber_kernels::flashnext::fuse86::GDN_NORM
        }) {
            if self.fuse86_site(saber_kernels::flashnext::fuse86::F1) {
                ensure!(
                    dm.zn % 128 == 0 && dm.zn / 8 <= 5 * 256,
                    "family 1: the norm's N {} in 128 groups of <= 5 x 256 lanes",
                    dm.zn
                );
                let (out, q8, s8) = match f5bufs {
                    // FN86c family 5: the update's tail wrote nrm, q8, s8
                    Some((_, out, q8, s8)) => (out, q8, s8),
                    None => {
                        let out = self.s(m * H * 2)?;
                        let (q8, s8) = (self.s(m * dm.zn)?, self.s(m * (dm.zn / 128) * 4)?);
                        self.run(
                            saber_kernels::flashnext::fuse86::GDN_NORM_Q8,
                            [m, 1, 1],
                            256,
                            vec![
                                Arg::P(core),
                                Arg::P(o + 2 * dm.conv),
                                l(zs),
                                Arg::P(nw),
                                Arg::P(nrm),
                                Arg::P(q8),
                                Arg::P(s8),
                                Arg::I(dm.zn as i32),
                                Arg::F(EPS),
                            ],
                            Group::Core,
                        )?;
                        (out, q8, s8)
                    }
                };
                self.w8_gemm_q(q8, s8, t.wo, t.so, out, m, H, dm.zn, H, Group::Proj)?;
                self.df_fold(r0)?;
                if self.layer == self.tap_layer {
                    self.taps.push(("nrm".into(), nrm, m * dm.zn * 2));
                    self.taps.push(("out".into(), out, m * H * 2));
                }
                return Ok(out);
            }
        }
        ensure!(f5bufs.is_none(), "family 5 runs inside family 1's site");
        self.run(
            &self.rules.kernels.gdn_norm_bf16_sigmoid.clone(),
            [m, 1, 1],
            256,
            vec![
                Arg::P(core),
                Arg::P(o + 2 * dm.conv),
                l(zs),
                Arg::P(nw),
                Arg::P(nrm),
                Arg::I(dm.zn as i32),
                Arg::F(EPS),
            ],
            Group::Core,
        )?;
        let out = self.s(m * H * 2)?;
        if let Some(t) = self.t8_cur {
            self.w8_gemm(nrm, t.wo, t.so, out, m, H, dm.zn, H, Group::Proj)?;
        } else {
            self.gemm(
                &format!("gdn.out_proj.{}", dm.tps()),
                nrm,
                wout,
                out,
                m,
                H,
                dm.zn,
                Group::Proj,
            )?;
        }
        if self.layer == self.tap_layer {
            self.taps.push(("nrm".into(), nrm, m * dm.zn * 2));
            self.taps.push(("out".into(), out, m * H * 2));
        }
        Ok(out)
    }

    /// The GDN decode form (FN13): one row per sequence, the state in place at slot `s + 1`.
    /// `b * hv <= fused_max_items` (32): the fused update (conv + grid barrier + recurrent, one
    /// launch); above it the conv_update + recurrent_update PAIR (layer_graph's decode law). Both
    /// take ONE decay exponent per token (`eg = e^g`, g <= 0 -- FN9b's safe form; no split, no
    /// clamp).
    #[allow(clippy::too_many_arguments)]
    /// FN86c: whether this decode / verify GDN update runs the f16-carry (`fc`) bodies -- the only
    /// bodies family 5 derives (`gdn_decode`'s own predicate).
    fn gdn_decode_fc(&self, m: u64, cc: &ChunkConsts) -> bool {
        self.route_m.is_some() && u64::from(cc.w.max(1)) > 1 && m % u64::from(cc.w.max(1)) == 0
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_decode(
        &mut self,
        mx: &MixerW,
        o: u64,
        q: u64,
        k: u64,
        v: u64,
        core: u64,
        m: u64,
        st: &LayerState,
        cc: &ChunkConsts,
        nq: Option<F5Tail>,
    ) -> anyhow::Result<()> {
        let MixerW::Gdn {
            conv, alog, dtb, ..
        } = *mx
        else {
            unreachable!()
        };
        ensure!(
            !self.state_f32,
            "the decode form runs the f16 stored state (the default)"
        );
        let dm = self.dims;
        let (gn, hv, hk) = (dm.gn, dm.hv, dm.hk);
        let (b_off, a_off) = (dm.conv + dm.zn, dm.conv + dm.zn + hv);
        let l = |x: u64| Arg::L(x as i64);
        let i = |x: u64| Arg::I(x as i32);
        let scale = 1.0 / (128f32).sqrt();
        // FN15b: a verify step (w rows per sequence) runs the conv window at `3 + w - 1`, starts
        // from `naccept - 1`, and keeps one GDN snapshot per row (`sidx` stride w)
        let w = u64::from(cc.w.max(1));
        let nseq = m / w;
        ensure!(
            nseq * w == m,
            "decode: {m} rows are not whole windows of {w}"
        );
        let (csl, stl) = (self.conv_l, 3 + w - 1);
        // FN16: the snapshot table's stride (every snapshot of a sequence; == w when pw == w)
        let sstride = u64::from(cc.pw.max(cc.w).max(1));
        ensure!(stl <= csl, "a width-{w} window over a conv state of {csl}");
        // FN24: the decode FORM is the class rows' (SPEC 0's `nseq` in a width-1-class verify: the
        // fused update at b1 -- it work-loops sequences x heads and runs any window, as the pair
        // does), and a width-1-class window carries the state at f16 between its rows (the `fc`
        // bodies: SPEC 0's store + reload, byte for byte)
        let items = self.rm(m) * hv;
        let fc = self.route_m.is_some() && w > 1;
        ensure!(
            nq.is_none() || fc,
            "family 5 derives the fc bodies only (w {w})"
        );
        // FN86c family 5: the norm + quant's arguments after the update's own
        let tail = |a: &mut Vec<Arg>| {
            if let Some(t) = nq {
                a.extend([
                    Arg::P(t.z),
                    l(t.zs),
                    Arg::P(t.nw),
                    Arg::P(t.nrm),
                    Arg::P(t.q8),
                    Arg::P(t.s8),
                    Arg::F(EPS),
                ]);
            }
        };
        if items <= FUSED_MAX_ITEMS {
            let nwg = items.clamp(1, FUSED_MAX_WG);
            let mut args = vec![
                Arg::P(o),
                l(gn),
                Arg::P(conv),
                Arg::P(0),
                Arg::P(st.cst),
                l(csl * dm.conv),
                l(csl),
                l(1),
                i(stl),
                Arg::P(cc.cidx),
                l(1),
                Arg::P(cc.naccept),
                Arg::P(cc.cu),
                // N sequences of `w` rows (SPEC 0: m sequences of one row -- the old args)
                i(nseq),
                i(hv),
                i(hk),
                i(w),
                Arg::P(q),
                Arg::P(k),
                Arg::P(v),
                Arg::P(o + 2 * a_off),
                Arg::P(o + 2 * b_off),
                l(gn),
                Arg::P(alog),
                Arg::P(dtb),
                Arg::P(st.h[0]),
                l(hv * 128 * 128),
                l(128 * 128),
                Arg::P(core),
                Arg::P(cc.sidx),
                l(sstride),
                Arg::F(scale),
                Arg::F(20.0),
                Arg::P(cc.bar),
                i(nwg),
                i(m),
            ];
            tail(&mut args);
            self.run(
                if nq.is_some() {
                    saber_kernels::flashnext::fuse86::F5_FUSED
                } else if fc {
                    GDN_FUSED_FC
                } else {
                    "saber_gdn_fused_update_f16s_w4k128v128_ab1"
                },
                [nwg, 1, 1],
                256,
                args,
                Group::Core,
            )
        } else {
            let hpb = 8;
            self.run(
                "saber_gdn_conv_update_w4_h128_hpb8",
                [nseq, (2 * hk + hv) / hpb, 1],
                hpb * 32,
                vec![
                    Arg::P(o),
                    l(gn),
                    Arg::P(conv),
                    Arg::P(0),
                    Arg::P(st.cst),
                    l(csl * dm.conv),
                    l(csl),
                    l(1),
                    i(stl),
                    Arg::P(cc.cidx),
                    l(1),
                    Arg::P(cc.naccept),
                    Arg::P(q),
                    Arg::P(k),
                    Arg::P(v),
                    Arg::P(cc.cu),
                    i(hv),
                    i(hk),
                    i(w),
                ],
                Group::Core,
            )?;
            let mut args = vec![
                Arg::P(q),
                Arg::P(k),
                Arg::P(v),
                Arg::P(o + 2 * a_off),
                Arg::P(o + 2 * b_off),
                l(gn),
                Arg::P(alog),
                Arg::P(dtb),
                Arg::P(st.h[0]),
                l(hv * 128 * 128),
                l(128 * 128),
                Arg::P(core),
                Arg::P(cc.cu),
                Arg::P(cc.sidx),
                l(sstride),
                Arg::P(cc.naccept),
                Arg::P(0),
                Arg::P(0),
                Arg::F(0.0),
                i(0),
                i(hv),
                i(hk),
                Arg::F(scale),
                Arg::F(20.0),
            ];
            tail(&mut args);
            self.run(
                if nq.is_some() {
                    saber_kernels::flashnext::fuse86::F5_RECUR
                } else if fc {
                    GDN_RECURRENT_FC
                } else {
                    "saber_gdn_recurrent_update_f16s_k128_v128_ab1_norm0_vs1"
                },
                [nseq, hv, 1],
                256,
                args,
                Group::Core,
            )
        }
    }

    // ------------------------------------------------------------------------ QSA core
    /// FN116: the QSA layer's K-SIDE front only -- the merged projection, q / k norm + rope, the
    /// indexer (queries + its ring / compressed store) and the KV append: every byte a LATER pass
    /// reads of these rows (their attention output, MoE and mixers are read by nothing). The same
    /// emission, allocations and order as [`Self::qsa_core`]'s front.
    pub fn qsa_kv_front(
        &mut self,
        mx: &MixerW,
        bi: u64,
        m: u64,
        st: &LayerState,
        cc: &ChunkConsts,
    ) -> anyhow::Result<()> {
        self.qsa_front(mx, bi, m, st, cc).map(|_| ())
    }

    /// The front of [`Self::qsa_core`]: (merged projection rows, queries, keys, indexer queries,
    /// the KV block table).
    fn qsa_front(
        &mut self,
        mx: &MixerW,
        bi: u64,
        m: u64,
        st: &LayerState,
        cc: &ChunkConsts,
    ) -> anyhow::Result<(u64, u64, u64, u64, u64)> {
        let MixerW::Qsa {
            wproj,
            qw,
            kw,
            iqw,
            ikw,
            ..
        } = *mx
        else {
            unreachable!()
        };
        let dm = self.dims;
        let (nq, nkv, qn) = (dm.nq, dm.nkv, dm.qn);
        let qk = self.s(m * qn * 2)?;
        if let Some(t) = self.t8_cur {
            // FN61c: q|gate + k + v at W8A8 into the merged row (stride qn), the indexer bf16
            let pre = dm.idx_off;
            ensure!(t.n == pre, "W8 QSA rows {} vs {pre}", t.n);
            // FN86b family 4: quant + the indexer's suffix in one launch when on
            let sfx = if t.sfx != 0 {
                t.sfx
            } else {
                wproj + pre * H * 2
            };
            self.w8_merged(bi, &t, sfx, qk, m, pre, qn - pre, H, qn)?;
        } else {
            self.gemm(
                &format!("qsa.proj.{}", dm.tps()),
                bi,
                wproj,
                qk,
                m,
                qn,
                H,
                Group::Proj,
            )?;
        }
        let qo = self.s(m * nq * HD * 2)?;
        let ko = self.s(m * nkv * HD * 2)?;
        let i = |x: u64| Arg::I(x as i32);
        let l = |x: u64| Arg::L(x as i64);
        // ATTN-OPT-3 family 6: at a TARGET decode / verify step the norm + rope, the indexer and the
        // KV / ring append as ONE launch (byte-equal to the three below); prefill chunks, the drafter
        // and verify widths past F6_MAX_W keep the chain
        {
            use saber_kernels::flashnext::fuse86 as f86;
            if cc.decode
                && self.t8_cur.is_some()
                && !self.glue_v1
                && cc.w.max(1) <= f86::F6_MAX_W
                && self.fuse86_site_n(f86::F6, f86::F6_SAVED)
            {
                let q_out = self.s(m * 4 * 128 * 2)?;
                let (kv_bt, kv_slot, _, _) = self.kv_binds(cc, m)?;
                let (sym, ks, vs) = if self.kv8_here() {
                    let (ks, vs) = self.kv_append_scales();
                    (f86::F6_FRONT_FP8, ks, vs)
                } else {
                    (f86::F6_FRONT_BF16, 1.0, 1.0)
                };
                self.run(
                    sym,
                    [
                        cdiv(m * (nq + nkv), 8)
                            + cdiv(m, u64::from(saber_kernels::fn6::IDX_Q_COMPRESS_V2_ROWS)),
                        1,
                        1,
                    ],
                    256,
                    vec![
                        Arg::P(qk),
                        Arg::P(qw),
                        Arg::P(kw),
                        Arg::P(cc.cs),
                        Arg::P(cc.pos3),
                        Arg::P(qo),
                        Arg::P(ko),
                        i(m),
                        i(nq),
                        i(nkv),
                        i(qn),
                        Arg::F(EPS),
                        Arg::I(11),
                        Arg::I(10),
                        Arg::P(qk + 2 * dm.idx_off),
                        l(qn),
                        Arg::P(iqw),
                        Arg::P(ikw),
                        Arg::P(cc.zeros),
                        Arg::P(cc.pos),
                        Arg::P(cc.seq_row0),
                        Arg::P(cc.p0v),
                        Arg::P(cc.ring_row0),
                        Arg::P(st.ring),
                        Arg::P(st.ringp),
                        i(RING),
                        Arg::P(cc.bt),
                        i(u64::from(cc.bt_nb)),
                        Arg::P(q_out),
                        Arg::P(st.comp),
                        Arg::P(qk + 2 * dm.v_off),
                        l(qn),
                        Arg::P(st.kv),
                        Arg::P(kv_slot),
                        l(nkv * 16 * 512),
                        l(512),
                        l(16 * 512),
                        Arg::P(qk + 2 * (dm.idx_off + 512)),
                        l(qn),
                        Arg::P(cc.rslot),
                        Arg::F(ks),
                        Arg::F(vs),
                    ],
                    Group::Core,
                )?;
                return Ok((qk, qo, ko, q_out, kv_bt));
            }
        }
        // FN21 (switched by FN24): Flash-Next's own instantiation, byte-identical to attn_glue's
        // `qk_norm_rope_wg8_h256_rot64_mi1_red5` (5,345 checks), same kernargs, same launch
        self.run(
            if self.glue_v1 {
                "qk_norm_rope_wg8_h256_rot64_mi1_red5"
            } else {
                saber_kernels::fn6::QK_NORM_ROPE_V2
            },
            [cdiv(m * (nq + nkv), 8), 1, 1],
            256,
            vec![
                Arg::P(qk),
                Arg::P(qw),
                Arg::P(kw),
                Arg::P(cc.cs),
                Arg::P(cc.pos3),
                Arg::P(qo),
                Arg::P(ko),
                i(m),
                i(nq),
                i(nkv),
                i(qn),
                Arg::F(EPS),
                Arg::I(11),
                Arg::I(11),
                Arg::I(10),
            ],
            Group::Core,
        )?;
        let q_out = self.s(m * 4 * 128 * 2)?;
        // FN21 (switched by FN24): byte-identical to `fn_idx_q_compress_h128`, same kernargs, four
        // rows a 256-thread workgroup
        let (isym, igrid, ithr) = if self.glue_v1 {
            ("fn_idx_q_compress_h128", m, 160)
        } else {
            (
                saber_kernels::fn6::IDX_Q_COMPRESS_V2,
                cdiv(m, u64::from(saber_kernels::fn6::IDX_Q_COMPRESS_V2_ROWS)),
                256,
            )
        };
        self.run(
            isym,
            [igrid, 1, 1],
            ithr,
            vec![
                Arg::P(qk + 2 * dm.idx_off),
                l(qn),
                Arg::P(iqw),
                Arg::P(ikw),
                Arg::P(cc.cs),
                Arg::P(cc.pos3),
                Arg::P(cc.zeros),
                Arg::P(cc.pos),
                Arg::P(cc.seq_row0),
                Arg::P(cc.p0v),
                Arg::P(cc.ring_row0),
                Arg::P(st.ring),
                Arg::P(st.ringp),
                i(RING),
                Arg::P(cc.bt),
                i(u64::from(cc.bt_nb)),
                Arg::P(q_out),
                Arg::P(st.comp),
                i(m),
                Arg::I(11),
                Arg::I(10),
            ],
            Group::Core,
        )?;
        let (kv_bt, kv_slot, _, _) = self.kv_binds(cc, m)?;
        let mut app = vec![
            Arg::P(ko),
            l(nkv * HD),
            Arg::P(qk + 2 * dm.v_off),
            l(qn),
            Arg::P(st.kv),
            Arg::P(kv_slot),
            l(nkv * 16 * 512),
            l(512),
            l(16 * 512),
            Arg::P(qk + 2 * (dm.idx_off + 512)),
            l(qn),
            Arg::P(st.ring),
            Arg::P(st.ringp),
            Arg::P(cc.rslot),
            Arg::P(cc.pos3),
            i(m),
        ];
        // FN68: the e4m3 append at one per-tensor scale for K and for V; FN68b: the layer's
        // calibrated (K, V) scale of this rank's head (1.0 = FN68's law, `--fn-kv-scales unit`),
        // the attention reading it back through the descale pointers (`kv_desc_ptrs`)
        let asym = if self.kv8_here() {
            // FN68b: the layer's calibrated scales (1.0 under `--fn-kv-scales unit`)
            let (ks, vs) = self.kv_append_scales();
            app.extend([Arg::F(ks), Arg::F(vs)]);
            "fn_kv_append_ring_h256_fp8"
        } else {
            "fn_kv_append_ring_h256_bf16"
        };
        self.run(asym, [m, nkv, 1], 128, app, Group::Core)?;
        Ok((qk, qo, ko, q_out, kv_bt))
    }

    fn qsa_core(
        &mut self,
        mx: &MixerW,
        bi: u64,
        m: u64,
        st: &LayerState,
        cc: &ChunkConsts,
    ) -> anyhow::Result<u64> {
        let MixerW::Qsa { wo, .. } = *mx else {
            unreachable!()
        };
        let (qk, qo, ko, q_out, kv_bt) = self.qsa_front(mx, bi, m, st, cc)?;
        let dm = self.dims;
        let (nq, nkv, qn) = (dm.nq, dm.nkv, dm.qn);
        let i = |x: u64| Arg::I(x as i32);
        let nb = u64::from(cc.nb);
        let core = self.s(m * nq * HD * 2)?;
        if self.qsa_sparse(cc) {
            self.qsa_sparse_attn(q_out, qo, core, m, st, cc)?;
        } else if cc.decode {
            self.attn_decode(qo, core, m, st, cc)?;
        } else {
            let ctx = u64::from(cc.p0) + m;
            let mut blob = Vec::with_capacity(136);
            let (kd, vd) = self.kv_desc_ptrs(nkv)?;
            for v in [qo, st.kv, kv_bt, cc.ctx, core, kd, vd, 0, 0] {
                blob.extend_from_slice(&v.to_le_bytes());
            }
            for v in [1, m, nq, nkv, HD, 16, nb] {
                blob.extend_from_slice(&(v as i32).to_le_bytes());
            }
            blob.extend_from_slice(&[0u8; 4]);
            blob.extend_from_slice(&((nkv * 16 * 512) as i64).to_le_bytes());
            blob.extend_from_slice(&((16 * 512) as i64).to_le_bytes());
            blob.extend_from_slice(&(1.0f32 / 16.0).to_le_bytes());
            blob.extend_from_slice(&0i32.to_le_bytes());
            blob.extend_from_slice(&(ctx as i32).to_le_bytes());
            blob.extend_from_slice(&[0u8; 4]);
            ensure!(blob.len() == 136, "SABERArgs is 136 bytes");
            self.run(
                &self.kv_sym(saber_kernels::fn6::PREFILL_SYMBOL),
                [cdiv(m, 32), nkv, 1],
                768,
                vec![Arg::Blob(blob)],
                Group::Core,
            )?;
        }
        let gated = self.s(m * nq * HD * 2)?;
        // FN86b family 1 (the QSA half): the gate + the W8 o_proj's act quant as ONE launch (the
        // gated core still stored; the arena takes gated, out, q, s in the chain's order)
        let fused_out = match self.t8_cur.filter(|_| {
            self.rules.kernels.attn_gate_bf16 == saber_kernels::flashnext::fuse86::ATTN_GATE
        }) {
            Some(t) if self.fuse86_site(saber_kernels::flashnext::fuse86::F1) => {
                let n = nq * HD;
                ensure!(
                    n % 128 == 0 && n / 8 <= 3 * 256,
                    "family 1: the gate's N {n} in 128 groups of <= 3 x 256 lanes"
                );
                let out = self.s(m * H * 2)?;
                let (q8, s8) = (self.s(m * n)?, self.s(m * (n / 128) * 4)?);
                if let Some((blob, sp)) = self.f7_pending.take() {
                    // ATTN-OPT-3 family 7: the decode attention's combine + this gate + quant
                    self.next_shared = (sp * 4) as u32;
                    self.run(
                        saber_kernels::flashnext::fuse86::F7_COMBINE_GATE_Q8,
                        [m * nq, 1, 1],
                        256,
                        vec![
                            Arg::Blob(blob),
                            Arg::I(sp as i32),
                            Arg::I(16),
                            Arg::P(qk),
                            Arg::P(gated),
                            Arg::P(q8),
                            Arg::P(s8),
                            i(n),
                            i(HD),
                            i(qn),
                        ],
                        Group::Core,
                    )?;
                } else {
                    self.run(
                        saber_kernels::flashnext::fuse86::ATTN_GATE_Q8,
                        [m, 1, 1],
                        256,
                        vec![
                            Arg::P(core),
                            Arg::P(qk),
                            Arg::P(gated),
                            Arg::P(q8),
                            Arg::P(s8),
                            i(n),
                            i(HD),
                            i(qn),
                        ],
                        Group::Core,
                    )?;
                }
                self.w8_gemm_q(q8, s8, t.wo, t.so, out, m, H, n, H, Group::Proj)?;
                Some(out)
            }
            _ => None,
        };
        ensure!(
            self.f7_pending.is_none(),
            "family 7: the combine was left to a gate that did not run fused"
        );
        let out = if let Some(out) = fused_out {
            out
        } else {
            self.run(
                &self.rules.kernels.attn_gate_bf16.clone(),
                [m, 1, 1],
                256,
                vec![
                    Arg::P(core),
                    Arg::P(qk),
                    Arg::P(gated),
                    i(nq * HD),
                    i(HD),
                    i(qn),
                ],
                Group::Core,
            )?;
            let out = self.s(m * H * 2)?;
            if let Some(t) = self.t8_cur {
                self.w8_gemm(gated, t.wo, t.so, out, m, H, nq * HD, H, Group::Proj)?;
            } else {
                self.gemm(
                    &format!("gdn.out_proj.{}", dm.tps()),
                    gated,
                    wo,
                    out,
                    m,
                    H,
                    nq * HD,
                    Group::Proj,
                )?;
            }
            out
        };
        if self.layer == self.tap_layer {
            // FN15b's per-op tap of a QSA layer (the MTP pass's bisect): the block input, the merged
            // projection, q / k after norm + rope, the K/V pages (the sequence's first `nb` blocks),
            // the attention core, the gated core, the o_proj partial (before the reduce)
            for (n, a_, b_) in [
                ("bi", bi, m * H * 2),
                ("qk", qk, m * qn * 2),
                ("qo", qo, m * nq * HD * 2),
                ("ko", ko, m * nkv * HD * 2),
                ("kv", st.kv, nb * nkv * 16 * 512 * self.kv_eb()),
                ("core", core, m * nq * HD * 2),
                ("gated", gated, m * nq * HD * 2),
                ("out", out, m * H * 2),
            ] {
                // FN68b: under the tier (`kv_table` set) the pages live in the tier's hot pool,
                // outside the pools the tap reads: the page tap is the flat pools' only
                if n == "kv" && self.kv_table != 0 {
                    continue;
                }
                self.taps.push((n.to_string(), a_, b_));
            }
        }
        Ok(out)
    }

    /// FN19: the QSA SPARSE attention -- the indexer's block scores and the exact top-512 per
    /// [`saber_kernels::fn19::SEL_ROWS`]-row slice, the per-CTA union key list, the gather attention
    /// (prefill: FN6's prefill body over the list; decode: its decode body + FN6's combine at the
    /// DENSE route's split count, so a sequence whose rows all select the identity keeps FN13's
    /// bytes). `iq` = the indexer queries `[m, 4, 128]` (`fn_idx_q_compress_h128`), `qo` the
    /// attention's queries, `core` its output.
    fn qsa_sparse_attn(
        &mut self,
        iq: u64,
        qo: u64,
        core: u64,
        m: u64,
        st: &LayerState,
        cc: &ChunkConsts,
    ) -> anyhow::Result<()> {
        use saber_kernels::fn19 as q19;
        let dm = self.dims;
        let (nq, nkv) = (dm.nq, dm.nkv);
        let nb = u64::from(cc.nb);
        let i = |x: u64| Arg::I(x as i32);
        let l = |x: u64| Arg::L(x as i64);
        // rows per sequence / per score tile / per attention CTA; CTAs per sequence; the context
        // bound of a CTA's list; the last row's position (the score grid's extent)
        let (rps, rt, rcta, cps, ncta, ctx_bound, max_pos) = if cc.decode {
            let w = u64::from(cc.w.max(1));
            ensure!(
                m % w == 0 && m <= u64::from(q19::SEL_ROWS),
                "a sparse decode step of {m} rows"
            );
            let nseq = m / w;
            (
                w,
                w,
                w,
                1,
                nseq,
                u64::from(cc.max_ctx),
                cc.max_ctx.saturating_sub(1),
            )
        } else {
            let ncta = cdiv(m, u64::from(q19::PREFILL_CTA_ROWS));
            let ctx = u64::from(cc.p0) + m;
            (
                m,
                u64::from(q19::SCORE_TILE_ROWS),
                u64::from(q19::PREFILL_CTA_ROWS),
                ncta,
                ncta,
                ctx,
                (ctx - 1) as u32,
            )
        };
        let ss = u64::from(q19::score_stride(max_pos));
        let ws = u64::from(q19::who_stride(max_pos));
        let ks = u64::from(q19::key_stride(ctx_bound as u32, rcta as u32));
        let slice = m.min(u64::from(q19::SEL_ROWS));
        let scores = self.s(slice * ss * 4)?;
        let sel = self.s(m * u64::from(q19::TOPK) * 4)?;
        let who = self.z(ncta * ws * 4)?;
        let kslot = self.s(ncta * ks * 4)?;
        let kmask = self.s(ncta * ks * 4)?;
        let kbits = self.s(ncta * ks * 4)?;
        let nkeys = self.s(ncta * 4)?;
        let (kv_bt, _, _, _) = self.kv_binds(cc, m)?;
        // FN128 (`--fn-sel128`): a decode-form step whose CTAs hold <= 8 rows runs the added
        // `flashnext_qsa128` symbols (each byte-equal to its parent by the card-1 oracle); prefill
        // and wider CTAs keep the parents
        use saber_kernels::flashnext::sel128 as s128;
        let small = cc.decode && rcta <= u64::from(s128::MAX_ROWS) && rt <= 8;
        let on = |bit: u32| small && self.rules.hc.sel128 & bit != 0;
        let (score_sym, score_thr) = if on(s128::SCORE) {
            (s128::SCORE_SYMBOL, u64::from(s128::SCORE_THREADS))
        } else {
            (q19::SCORE_SYMBOL, 256)
        };
        let topk_new = if on(s128::TOPK) {
            s128::topk_cell(ss)
        } else {
            None
        };
        let (topk_sym, topk_thr) = match topk_new {
            Some((s, nt)) => (s, u64::from(nt)),
            None => (q19::TOPK_SYMBOL, 256),
        };
        let (union_sym, union_thr) = if on(s128::UNION) {
            (s128::UNION_SYMBOL, u64::from(s128::UNION_THREADS))
        } else {
            (q19::UNION_SYMBOL, 256)
        };
        if cc.decode {
            self.sel128_n[0][usize::from(score_sym == q19::SCORE_SYMBOL)] += 1;
            self.sel128_n[1][usize::from(topk_new.is_none())] += 1;
            self.sel128_n[2][usize::from(union_sym == q19::UNION_SYMBOL)] += 1;
        }
        let mut r0 = 0;
        while r0 < m {
            let rows = (m - r0).min(u64::from(q19::SEL_ROWS));
            self.run(
                score_sym,
                [ss / u64::from(q19::SCORE_TILE_GROUPS), cdiv(rows, rt), 1],
                score_thr,
                vec![
                    Arg::P(iq + r0 * 4 * 128 * 2),
                    Arg::P(st.comp),
                    Arg::P(cc.bt),
                    i(u64::from(cc.bt_nb)),
                    Arg::P(cc.pos + r0 * 8),
                    Arg::P(scores),
                    l(ss),
                    i(rows),
                    i(rps),
                    i(rt),
                ],
                Group::Core,
            )?;
            self.run(
                topk_sym,
                [rows, 1, 1],
                topk_thr,
                vec![
                    Arg::P(scores),
                    l(ss),
                    Arg::P(cc.pos + r0 * 8),
                    Arg::P(sel + r0 * u64::from(q19::TOPK) * 4),
                    i(rows),
                ],
                Group::Core,
            )?;
            r0 += rows;
        }
        self.run(
            union_sym,
            [ncta, 1, 1],
            union_thr,
            vec![
                Arg::P(sel),
                Arg::P(cc.pos),
                i(m),
                i(rcta),
                i(cps),
                Arg::P(kv_bt),
                i(nb),
                Arg::P(who),
                l(ws),
                Arg::P(kslot),
                Arg::P(kmask),
                Arg::P(kbits),
                Arg::P(nkeys),
                l(ks),
            ],
            Group::Core,
        )?;
        self.qsa_sel = Some((sel, iq, m));
        let mut qa = Vec::with_capacity(q19::QSA_KEYS_BYTES);
        for v in [kslot, kbits, nkeys, ks] {
            qa.extend_from_slice(&v.to_le_bytes());
        }
        if !cc.decode {
            let ctx = u64::from(cc.p0) + m;
            let (kd, vd) = self.kv_desc_ptrs(nkv)?;
            let blob = |out: u64| -> anyhow::Result<Vec<u8>> {
                let mut blob = Vec::with_capacity(136);
                for v in [qo, st.kv, kv_bt, cc.ctx, out, kd, vd, 0, 0] {
                    blob.extend_from_slice(&v.to_le_bytes());
                }
                for v in [1, m, nq, nkv, HD, 16, nb] {
                    blob.extend_from_slice(&(v as i32).to_le_bytes());
                }
                blob.extend_from_slice(&[0u8; 4]);
                blob.extend_from_slice(&((nkv * 16 * 512) as i64).to_le_bytes());
                blob.extend_from_slice(&((16 * 512) as i64).to_le_bytes());
                blob.extend_from_slice(&(1.0f32 / 16.0).to_le_bytes());
                blob.extend_from_slice(&0i32.to_le_bytes());
                blob.extend_from_slice(&(ctx as i32).to_le_bytes());
                blob.extend_from_slice(&[0u8; 4]);
                ensure!(blob.len() == 136, "SABERArgs is 136 bytes");
                Ok(blob)
            };
            // FN124 (`--fn-prefill-tiles qsa`): the fp8-KV gather on flashnext_qsa124's body (the
            // masked sub-tile skip, byte-equal); `qsatwin` runs a qsa124 body BESIDE the shipped
            // symbol into scratch and counts the differing words (the in-situ twin)
            use saber_kernels::flashnext::prefill122 as p122;
            let pf = self.rules.hc.prefill122;
            let fp8 = self.kv8_here();
            let lever = fp8 && pf & p122::QSA != 0 && pf & p122::QSATWIN == 0;
            self.pf124_n[usize::from(!lever)] += 1;
            let sym = if lever {
                p122::QSA_SYMBOL.to_string()
            } else {
                self.kv_sym(q19::PREFILL_SYMBOL)
            };
            self.run(
                &sym,
                [ncta, nkv, 1],
                768,
                vec![Arg::Blob(blob(core)?), Arg::Blob(qa.clone())],
                Group::Core,
            )?;
            if fp8 && pf & p122::QSATWIN != 0 {
                // a dry run (the boot's arena sizing) has no region: its records never launch
                let ctr = self.pf124_twin.unwrap_or(0);
                let bytes = m * nq * HD * 2;
                let core2 = self.s(bytes)?;
                // `qsatwin` with `qsa` = the lever beside the shipped symbol (0 words must
                // differ); `qsatwin` alone = the CONTROL beside it (must differ)
                let twin = if pf & p122::QSA != 0 {
                    p122::QSA_SYMBOL
                } else {
                    p122::QSA_CONTROL
                };
                self.run(
                    twin,
                    [ncta, nkv, 1],
                    768,
                    vec![Arg::Blob(blob(core2)?), Arg::Blob(qa)],
                    Group::Core,
                )?;
                let words = bytes / 4;
                self.run(
                    p122::QSA_CMP,
                    [cdiv(words, 256).min(1024), 1, 1],
                    256,
                    vec![
                        Arg::P(core2),
                        Arg::P(core),
                        Arg::L(words as i64),
                        Arg::P(ctr),
                    ],
                    Group::Core,
                )?;
                self.pf122_added += 2;
                self.pf124_twins += 1;
            }
            return Ok(());
        }
        // decode: the DENSE route's split count (the identity list is then FN13's bytes); the
        // declc cells run the pair (the sparse decode has no declc form)
        let w = u64::from(cc.w.max(1));
        let nseq = m / w;
        let rt_ = saber_kernels::fn6::decode_route(w as u32, cc.max_ctx, nseq as u32)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let sp = u64::from(rt_.splits);
        let scr = self.s(m * nq * sp * (HD * 2 + 8))?;
        let mut blob = Vec::with_capacity(136);
        // seqused_k = the lists' key counts: the segmentation runs over the union's keys
        let (kd, vd) = self.kv_desc_ptrs(nseq * nkv)?;
        for v in [qo, st.kv, kv_bt, nkeys, core, kd, vd, 0, scr] {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        for v in [nseq, w, nq, nkv, HD, 16, nb] {
            blob.extend_from_slice(&(v as i32).to_le_bytes());
        }
        blob.extend_from_slice(&[0u8; 4]);
        blob.extend_from_slice(&((nkv * 16 * 512) as i64).to_le_bytes());
        blob.extend_from_slice(&((16 * 512) as i64).to_le_bytes());
        blob.extend_from_slice(&(1.0f32 / 16.0).to_le_bytes());
        blob.extend_from_slice(&0i32.to_le_bytes());
        blob.extend_from_slice(&(cc.max_ctx as i32).to_le_bytes());
        blob.extend_from_slice(&[0u8; 4]);
        ensure!(blob.len() == 136, "SABERArgs is 136 bytes");
        self.run(
            &self.kv_sym(q19::DECODE_SYMBOL),
            [sp, 1, nseq],
            128,
            vec![Arg::Blob(blob.clone()), Arg::I(sp as i32), Arg::Blob(qa)],
            Group::Core,
        )?;
        // ATTN-OPT-3 family 7: the combine runs inside the gate's launch (qsa_core's family-1 site)
        if self.f7_site(cc) {
            ensure!(
                self.f7_pending.is_none(),
                "family 7: a combine already pending"
            );
            self.f7_pending = Some((blob, sp));
            return Ok(());
        }
        self.next_shared = (sp * 4) as u32;
        self.run(
            saber_kernels::fn6::COMBINE_SYMBOL,
            [m * nq, 1, 1],
            256,
            vec![Arg::Blob(blob), Arg::I(sp as i32), Arg::I(16)],
            Group::Core,
        )
    }

    /// The decode attention (FN13): FN10's routed form at (q_len 1, the plan's context band, b
    /// sequences) -- the decode + combine PAIR (57/60 cells); the declc form is refused by name.
    fn attn_decode(
        &mut self,
        qo: u64,
        core: u64,
        m: u64,
        st: &LayerState,
        cc: &ChunkConsts,
    ) -> anyhow::Result<()> {
        let dm = self.dims;
        let (nq, nkv) = (dm.nq, dm.nkv);
        ensure!(
            cc.max_ctx > 0,
            "decode: the attention's context band is not set"
        );
        let w = u64::from(cc.w.max(1));
        let nseq = m / w;
        // FN24: in the width-1 class a verify step's attention runs its m rows as m pseudo-
        // sequences of ONE row each (`ctx_rows` / `bt_rows`) at SPEC 0's route for `nseq`
        // sequences: every row's segmentation (tiles per split, splits used), its wave mates (the
        // lazy rescale is a wave decision) and its combine chains are SPEC 0's for that token
        let per_row = self.route_m.is_some() && w > 1;
        // FN45b2: under the tier, each layer's own tables; a per-row step's table is `kbtr` (dense
        // rows only: its stride is `KBTR`, every such row sits at <= 2,050)
        let (kv_bt, _, kv_btr, kv_btr_nb) = self.kv_binds(cc, m)?;
        let mut nb = u64::from(cc.nb);
        let (lseq, lq, lctx, lbt) = if per_row {
            if self.kv_table != 0 {
                ensure!(
                    kv_btr != 0 && cc.qsa_short,
                    "a tiered per-row decode step past 2,050 (or without its kbtr table)"
                );
                nb = kv_btr_nb;
            }
            (m, 1, cc.ctx_rows, kv_btr)
        } else {
            (nseq, w, cc.ctx, kv_bt)
        };
        let mut rt = saber_kernels::fn6::decode_route(lq as u32, cc.max_ctx, nseq as u32)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if rt.form == saber_kernels::fn6::AttnForm::DecodeDeclc {
            // FN14: the table's declc cells (b8 w1 above 2,048) run the decode + combine PAIR at
            // the row's own split count (the pair is correct in every cell; the declc is one launch
            // fewer and is not wired) -- counted, so a plan that hits it says so (rule 5)
            rt.form = saber_kernels::fn6::AttnForm::DecodePair;
            rt.symbol = format!(
                "fn_attn_decode_h256_gqa12_bf16kv_nw{}",
                saber_kernels::fn6::ROUTED_DECODE_NW
            );
            self.declc_as_pair += 1;
        }
        let sp = u64::from(rt.splits);
        let scr = self.s(m * nq * sp * (HD * 2 + 8))?;
        let dsym = self.kv_sym(&rt.symbol);
        let mut blob = Vec::with_capacity(136);
        let (kd, vd) = self.kv_desc_ptrs(lseq * nkv)?;
        for v in [qo, st.kv, lbt, lctx, core, kd, vd, 0, scr] {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        for v in [lseq, lq, nq, nkv, HD, 16, nb] {
            blob.extend_from_slice(&(v as i32).to_le_bytes());
        }
        blob.extend_from_slice(&[0u8; 4]);
        blob.extend_from_slice(&((nkv * 16 * 512) as i64).to_le_bytes());
        blob.extend_from_slice(&((16 * 512) as i64).to_le_bytes());
        blob.extend_from_slice(&(1.0f32 / 16.0).to_le_bytes());
        blob.extend_from_slice(&0i32.to_le_bytes());
        blob.extend_from_slice(&(cc.max_ctx as i32).to_le_bytes());
        blob.extend_from_slice(&[0u8; 4]);
        ensure!(blob.len() == 136, "SABERArgs is 136 bytes");
        self.run(
            &dsym,
            [sp, 1, lseq],
            128,
            vec![Arg::Blob(blob.clone()), Arg::I(sp as i32)],
            Group::Core,
        )?;
        // ATTN-OPT-3 family 7: the combine runs inside the gate's launch (qsa_core's family-1 site)
        if self.f7_site(cc) {
            ensure!(
                self.f7_pending.is_none(),
                "family 7: a combine already pending"
            );
            self.f7_pending = Some((blob, sp));
            return Ok(());
        }
        self.next_shared = (sp * 4) as u32;
        self.run(
            saber_kernels::fn6::COMBINE_SYMBOL,
            [m * nq, 1, 1],
            256,
            vec![Arg::Blob(blob), Arg::I(sp as i32), Arg::I(16)],
            Group::Core,
        )
    }

    // ------------------------------------------------------------------------ MoE (7 launches)
    /// The route half: router GEMM, shared-gate GEMV, route + sort (3 launches); `fused` (FN30):
    /// router GEMM + `moe::route_symbol(m)` only (2 launches) -- the gate GEMV rides the back half.
    pub fn moe_route(
        &mut self,
        lw: &LayerW,
        mx: &MixOutBufs,
        m: u64,
        fused: bool,
    ) -> anyhow::Result<MoeRoute> {
        self.moe_route_tm(lw, mx, m, None, fused)
    }

    /// [`Self::moe_route`] at a forced `tile_blocks` (FN15: the MTP experts' bf16 band takes
    /// 16-row tiles, `tm` 1, and FN10's split rows).
    pub fn moe_route_tm(
        &mut self,
        lw: &LayerW,
        mx: &MixOutBufs,
        m: u64,
        tm_force: Option<u64>,
        fused: bool,
    ) -> anyhow::Result<MoeRoute> {
        let dm = self.dims;
        let logits = self.s(m * 512 * 2)?;
        // the one predicate the declared counts share (`fused` already means a decode step)
        let front = fused && moe_block_front(self.moe_train, true, m as u32);
        if !front {
            self.gemm(
                "moe.router",
                mx.bi,
                lw.router,
                logits,
                m,
                512,
                H,
                Group::Moe,
            )?;
        }
        let sg = self.s(m * 2)?;
        if !fused {
            self.gemm_sym("fn_bgemm_sk_m16", lw.sg, mx.bi, sg, 1, m, H, 1, Group::Moe)?;
        }
        // FN24: the expert cell of the class rows (T = 1's split-K in a width-1-class b1 verify;
        // every decode cell runs any T: FN4's grouped == N dense calls, row for row)
        // FN80b-2: the TP-split target takes EP's measured cells (the class choice of FN4/FN20,
        // split counts included); its down half names FN80b-1's DBK-64 cells at tm 1 and the
        // committed pf tiles above (`moe::ExpertShape::TpHalf`, in `moe_experts` / `moe_back`)
        let gu_c = moe::measured_cell(MoeGemm::GateUp, self.rm(m) as u32);
        let dn_c = moe::measured_cell(MoeGemm::Down, self.rm(m) as u32);
        ensure!(
            gu_c.tm == dn_c.tm,
            "the route's tile_blocks is both GEMMs' tm"
        );
        let tm = tm_force.unwrap_or(u64::from(gu_c.tm));
        let (gu_ks, dn_ks) = if tm_force.is_some() {
            let (a, b) = saber_kernels::fn6::mtp_expert_splits(
                saber_kernels::fn6::MtpExperts::Bf16,
                m as u32,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            (u64::from(a), u64::from(b))
        } else {
            (u64::from(gu_c.ks), u64::from(dn_c.ks))
        };
        // the MTP layer's experts stay EP (FLASHNEXT-PLAN §13 item 5)
        let (eloc, e_lo, shared) = if tm_force.is_some() {
            dm.mtp_experts()
        } else {
            // FN114c: the fused train's shared rows on BOTH ranks under `--fn-shared-expert split`
            (
                dm.eloc,
                dm.e_lo,
                dm.shared || (fused && self.shared_split != 0),
            )
        };
        let eloc = u64::from(eloc);
        let r = m * 10;
        let me = eloc.min(r);
        let pcap = r + 15 * me + cdiv(m, 16) * 16;
        let tc = moe_tiles_cap(m, tm, eloc);
        let ids = match self.glue_ids(r * 4) {
            Some(a) => a,
            None => self.z(r * 4)?,
        };
        let wts = self.z(r * 4)?;
        let pos = self.z(r * 4)?;
        let row_tok = self.o(pcap * 4)?;
        let row_route = self.z(pcap * 4)?;
        let te = self.z(tc * 4)?;
        let t0 = self.z(tc * 4)?;
        let tr = self.z(tc * 4)?;
        let ecnt = self.z(eloc * 4)?;
        let meta = self.z(8 * 4)?;
        let blk = self.z(cdiv(m, 32) * eloc * 4)?;
        let rank = self.z(r * 4)?;
        let arrive = self.z(4)?;
        let i = |x: u64| Arg::I(x as i32);
        if front {
            // FN31's front half: the router GEMM at the CLASS rows' split (FN24: a b1 verify keeps
            // T = 1's split 4 -- the split is bytes) + the route in one workgroup's LDS
            let rm = self.rm(m) as u32;
            let (_, sp) = self
                .rules
                .bgemm_route("moe.router", rm, 512)
                .context("bgemm moe.router")?;
            ensure!(
                sp == moe::router_split(rm),
                "the rules' router split {sp} at {rm} rows != moe::router_split"
            );
            let ft = moe::FrontTrain {
                split: moe::router_split(rm),
                waves: moe::FrontTrain::WAVES,
                unrolled: true,
                route: self.route_form,
            };
            let part = self.s(ft.part_floats(m as u32) * 4)?;
            ensure!(self.moe_sync != 0, "the front half needs the sync region");
            let slot = moe_sync_slot_bytes(dm.eloc);
            let fsync = self.moe_sync + (m - 1) * slot + slot - 4 * moe::FrontTrain::SYNC_WORDS;
            let g = ft.grid(m as u32);
            // FN163 (`--fn-route163 b1`): the split-4 front's register-rank twin (same args, grid)
            let mut fsym = ft.symbol();
            if let Some(t) = saber_kernels::flashnext::route163::front_route(&fsym) {
                if self.route163_site(saber_kernels::flashnext::route163::B1) {
                    fsym = t.to_string();
                }
            }
            self.run(
                &fsym,
                [u64::from(g[0]), u64::from(g[1]), u64::from(g[2])],
                u64::from(ft.threads()),
                vec![
                    Arg::P(mx.bi),
                    Arg::P(lw.router),
                    Arg::P(part),
                    Arg::P(fsync),
                    Arg::P(logits),
                    i(H),
                    i(m),
                    i(u64::from(e_lo)),
                    i(eloc),
                    i(tm),
                    i(u64::from(shared)),
                    Arg::P(ids),
                    Arg::P(wts),
                    Arg::P(pos),
                    Arg::P(row_tok),
                    Arg::P(row_route),
                    Arg::P(te),
                    Arg::P(t0),
                    Arg::P(tr),
                    Arg::P(ecnt),
                    Arg::P(meta),
                ],
                Group::Moe,
            )?;
            self.moe_front_blocks += 1;
            return Ok(MoeRoute {
                logits,
                sg,
                ids,
                wts,
                pos,
                row_tok,
                te,
                t0,
                tr,
                meta,
                pcap,
                tc,
                tm,
                gu_ks,
                dn_ks,
            });
        }
        // FN30: under the fused train `moe::route_symbol(T)` (FN31's LDS route at T 1..16:
        // FN4's tables by construction, 3,534 oracle cases); FN49b: the seven-launch train
        // takes it too at the DPP default (FN49a's oracle: 4,548 cases, 0 bytes)
        let mut rsym = if fused || self.route_form == moe::RouteForm::Dpp {
            moe::route_symbol_form(m as u32, self.route_form)
        } else {
            moe::ROUTE_SYMBOL
        };
        let mut rthr = 256;
        {
            // FN111 t3: FN4's route at 17..64 tokens -> the LDS + DPP body at 16 waves (byte-equal)
            use saber_kernels::flashnext::tiles111 as t111;
            if rsym == moe::ROUTE_SYMBOL
                && self.route_form == moe::RouteForm::Dpp
                && (t111::T3_MIN_T..=t111::T3_MAX_T).contains(&(m as u32))
                && self.tile111_site(t111::T3)
            {
                rsym = t111::SYMS[2];
                rthr = t111::T3_THREADS;
            }
            // FN163 (`--fn-route163 b8`): the shipped selection at 32 waves + one packed scan
            // (byte-equal to every shipped route at 9..64 rows; same args, grid)
            use saber_kernels::flashnext::route163 as r163;
            if self.route_form == moe::RouteForm::Dpp
                && r163::route_takes(rsym, m as u32)
                && self.route163_site(r163::B8)
            {
                rsym = r163::ROUTE_SYM;
                rthr = r163::ROUTE_THREADS;
            }
        }
        self.run(
            rsym,
            [cdiv(m, 32), 1, 1],
            rthr,
            vec![
                Arg::P(logits),
                i(m),
                i(u64::from(e_lo)),
                i(eloc),
                i(tm),
                i(u64::from(shared)),
                Arg::P(ids),
                Arg::P(wts),
                Arg::P(pos),
                Arg::P(row_tok),
                Arg::P(row_route),
                Arg::P(te),
                Arg::P(t0),
                Arg::P(tr),
                Arg::P(ecnt),
                Arg::P(meta),
                Arg::P(blk),
                Arg::P(rank),
                Arg::P(arrive),
            ],
            Group::Moe,
        )?;
        Ok(MoeRoute {
            logits,
            sg,
            ids,
            wts,
            pos,
            row_tok,
            te,
            t0,
            tr,
            meta,
            pcap,
            tc,
            tm,
            gu_ks,
            dn_ks,
        })
    }

    /// The expert half over the xptr tables `(xgu, xdn)`: grouped gate_up, silu-quant, grouped
    /// down, combine (4 launches). Returns the MoE output (the carry when `carry_out`).
    pub fn moe_experts(
        &mut self,
        rt: &MoeRoute,
        mx: &MixOutBufs,
        m: u64,
        xgu: u64,
        xdn: u64,
        carry_out: bool,
    ) -> anyhow::Result<u64> {
        let MoeRoute {
            sg,
            wts,
            pos,
            row_tok,
            te,
            t0,
            tr,
            meta,
            pcap,
            tc,
            tm,
            ..
        } = *rt;
        let grouped = |em: &mut Self,
                       g: MoeGemm,
                       a: u64,
                       a_s: u64,
                       xp: u64,
                       c: u64,
                       n: u64,
                       k: u64,
                       ks: u64|
         -> anyhow::Result<()> {
            let (part, cnt) = if ks > 1 {
                (em.s(ks * pcap * n * 4)?, em.z(tc * (n / 128) * 4)?)
            } else {
                (0, 0)
            };
            // FN80b-2: the cell's symbol at the rank's expert shape (FN80b-1's key: the TP half's
            // down at DBK 64; identical to `moe::grouped_symbol` for the whole expert)
            let cell = moe::Cell {
                ks: ks as u32,
                tm: tm as u32,
            };
            let shape = em.dims.xshape();
            // FN150: NVFP4 records run FN148's form (b) at the same cell (+ `gsplit`); FN152:
            // FN151's `_p1` twin of that cell where one exists (`xnv_plain` = FN150's route)
            let sym = if em.dims.xnv {
                let body = if em.dims.xnv_plain {
                    moe::Nvfp4Body::Plain
                } else {
                    moe::Nvfp4Body::P1
                };
                moe::nvfp4_symbol_body(shape, g, cell, body).map_err(|e| anyhow::anyhow!("{e}"))?
            } else {
                match g {
                    MoeGemm::GateUp => shape.gate_up_symbol(cell, moe::DECODE_BODY),
                    MoeGemm::Down => shape
                        .down_symbol(cell, moe::DECODE_BODY)
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                }
            };
            let mut args = vec![
                Arg::P(a),
                Arg::P(xp),
                Arg::P(te),
                Arg::P(t0),
                Arg::P(tr),
                Arg::P(row_tok),
                Arg::P(meta),
                Arg::P(a_s),
                Arg::P(part),
                Arg::P(cnt),
                Arg::P(c),
                Arg::I(n as i32),
                Arg::I(k as i32),
                Arg::I(pcap as i32),
            ];
            if em.dims.xnv {
                args.push(Arg::I(moe::nvfp4_gsplit(g, n as u32) as i32));
            }
            em.run(&sym, [n / 128, tc, ks], 256, args, Group::Moe)
        };
        // FN80b-2: the TP-split form computes this rank's half of every expert (gate_up N 640 =
        // gate rows then up rows of the half, the silu at n 320, down K 320); the partial outputs
        // ride the post-MoE all-reduce the layer already pays
        let inter = self.dims.x_inter();
        let gu = self.s(pcap * 2 * inter * 2)?;
        grouped(
            self,
            MoeGemm::GateUp,
            mx.q8,
            mx.scale,
            xgu,
            gu,
            2 * inter,
            H,
            rt.gu_ks,
        )?;
        let q = self.s(pcap * inter)?;
        let sc = self.s(pcap * 4)?;
        self.run(
            moe::SILU_SYMBOL,
            [pcap, 1, 1],
            256,
            vec![
                Arg::P(gu),
                Arg::P(q),
                Arg::P(sc),
                Arg::I(inter as i32),
                Arg::P(row_tok),
                Arg::P(meta),
            ],
            Group::Moe,
        )?;
        let y = self.s(pcap * H * 2)?;
        grouped(self, MoeGemm::Down, q, sc, xdn, y, H, inter, rt.dn_ks)?;
        if self.tap_moe && self.layer == self.tap_layer {
            for (n, a_, b_) in [
                ("moe.gu", gu, pcap * 2 * inter * 2),
                ("moe.q", q, pcap * inter),
                ("moe.sc", sc, pcap * 4),
                ("moe.y", y, pcap * H * 2),
            ] {
                self.taps.push((n.to_string(), a_, b_));
            }
        }
        let out = if carry_out {
            let slot = (self.layer.max(0) as usize) % 2;
            self.arena.carry[slot].take(m * H * 2)?
        } else {
            self.s(m * H * 2)?
        };
        self.run(
            moe::COMBINE_SYMBOL,
            [m, 1, 1],
            256,
            vec![
                Arg::P(y),
                Arg::P(wts),
                Arg::P(pos),
                Arg::P(sg),
                Arg::P(meta),
                Arg::P(out),
                Arg::I(H as i32),
            ],
            Group::Moe,
        )?;
        Ok(out)
    }

    /// FN30: FN26's fused back half (`moe::BackTrain`, ONE launch: the shared-expert gate GEMV,
    /// gate_up, silu-quant, down and the combine, ticket-ordered) over the route `rt` taken with
    /// `fused`; the split pair is the route's own class cell (`gu_ks` / `dn_ks`: FN24's width-1
    /// class keeps T = 1's ks4 / ks2 in a b1 verify), the body FN26's `DECODE_BODY`. Byte-identical
    /// to [`Self::moe_experts`] + the gate GEMV (FN26: 0 of 150,984 rows at T 1..40, both pairs).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_back(
        &mut self,
        rt: &MoeRoute,
        lw: &LayerW,
        mx: &MixOutBufs,
        m: u64,
        xgu: u64,
        xdn: u64,
        carry_out: bool,
    ) -> anyhow::Result<u64> {
        let MoeRoute {
            sg,
            wts,
            pos,
            row_tok,
            te,
            t0,
            tr,
            meta,
            pcap,
            tc,
            tm,
            gu_ks,
            dn_ks,
            ..
        } = *rt;
        ensure!(
            !self.dims.xnv_plain,
            "--fn-experts-fmt nvfp4 on FN150's plain route: no fused back half (FN152's is P1's)"
        );
        let bt = moe::BackTrain {
            gu_ks: gu_ks as u32,
            dn_ks: dn_ks as u32,
            prefetch: moe::BACK_PREFETCH_BODY == moe::DecodeBody::Prefetch,
        };
        ensure!(
            tm == 1 && moe::BackTrain::CELLS.contains(&(bt.gu_ks, bt.dn_ks)),
            "no fused back half at tm {tm} ks {gu_ks}/{dn_ks}"
        );
        ensure!(
            (1..=u64::from(moe::BackTrain::MAX_TOKENS)).contains(&m),
            "the fused back half serves T <= {}, not {m}",
            moe::BackTrain::MAX_TOKENS
        );
        ensure!(
            self.moe_sync != 0,
            "the fused MoE train needs the boot's sync region (fn.moe.sync)"
        );
        let slot = moe_sync_slot_bytes(self.dims.eloc);
        ensure!(
            (moe::BackTrain::sync_words(tc as u32) + moe::FrontTrain::SYNC_WORDS) * 4 <= slot,
            "sync words of tc {tc} past the slot"
        );
        let sync = self.moe_sync + (m - 1) * slot;
        // FN80b-2: the rank's expert shape (FN80b-1's `moe_back_h2_*` at the TP half)
        let shape = self.dims.xshape();
        let gun = u64::from(shape.gate_up_n());
        let p_gu = if gu_ks > 1 {
            self.s(gu_ks * pcap * gun * 4)?
        } else {
            0
        };
        let p_dn = if dn_ks > 1 {
            self.s(dn_ks * pcap * H * 4)?
        } else {
            0
        };
        // FN4's split-K arrival counters (the kernels zero their own; seeded as the train's are)
        let cnt_gu = self.z(tc * (gun / 64) * 4)?;
        let cnt_dn = self.z(tc * (H / 64) * 4)?;
        let gu = self.s(pcap * gun * 2)?;
        let q = self.s(pcap * u64::from(shape.inter()))?;
        let sc = self.s(pcap * 4)?;
        let y = self.s(pcap * H * 2)?;
        // FN137: under the fused reduce the partial is scratch and the SUM takes the reduce's
        // place in the layer's carry (the standalone reduce's `carry` output)
        let far = self.far_next.take();
        let out = if carry_out && far.is_none() {
            let slot = (self.layer.max(0) as usize) % 2;
            self.arena.carry[slot].take(m * H * 2)?
        } else {
            self.s(m * H * 2)?
        };
        let red = if far.is_some() {
            let slot = (self.layer.max(0) as usize) % 2;
            self.arena.carry[slot].take(m * H * 2)?
        } else {
            0
        };
        let i = |x: u64| Arg::I(x as i32);
        // FN84: the order law's symbol + (ram_lo, ram_hi, kint), or FN26's symbol at `id`
        let order = self.miss_order;
        let mo = match order.symbol(&bt) {
            Some(s) => {
                ensure!(
                    bt.prefetch,
                    "the FN84 order law is instantiated at the prefetch body only"
                );
                let k = order.kint(self.rules, m as u32).ok_or_else(|| {
                    anyhow::anyhow!(
                        "--fn-moe-miss-order {}: no [[moe_miss]] row holds {m} rows",
                        order.name()
                    )
                })?;
                Some((s, k))
            }
            None => None,
        };
        self.miss_launches[order.index()] += 1;
        // FN80b-2: the TP half's back half is FN80b-1's `moe_back_h2_*` (no FN84 order twin yet)
        ensure!(
            shape == moe::ExpertShape::Whole || mo.is_none(),
            "--fn-experts tp2: the FN84 miss order has no half-width back half (--fn-moe-miss-order)"
        );
        // FN114c: the shared tile at this rank's half (`moe_back_ss_*`, FN26's routed tiles)
        let ss = self.shared_split != 0;
        ensure!(
            !ss || (shape == moe::ExpertShape::Whole && mo.is_none() && bt.prefetch),
            "--fn-shared-expert split: the EP whole-expert train at the id order and the _p1 body only"
        );
        ensure!(
            far.is_none() || (shape == moe::ExpertShape::TpHalf && mo.is_none() && !ss && bt.prefetch),
            "--fn-moe-ar-push on: the fused reduce rides FN80b-1's id-order half-width _p1 back half only"
        );
        // FN152: NVFP4 records run FN151's body in the fused back half (id order, no shared split)
        ensure!(
            !self.dims.xnv || (mo.is_none() && !ss),
            "--fn-experts-fmt nvfp4: the fused back half is FN152's id-order train only (no FN84 \
             miss order, no shared split)"
        );
        let sym = match &mo {
            Some((s, _)) => s.clone(),
            None if self.dims.xnv => moe::nvfp4_back_symbol(shape, &bt, far.is_some())
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            None if ss => shared_split_symbol(&bt),
            None if far.is_some() => super::dev::far_symbol(bt.gu_ks, bt.dn_ks),
            None => shape.back_symbol(&bt).map_err(|e| anyhow::anyhow!("{e}"))?,
        };
        let mut args = vec![
            Arg::P(mx.q8),
            Arg::P(mx.scale),
            Arg::P(xgu),
            Arg::P(xdn),
            Arg::P(te),
            Arg::P(t0),
            Arg::P(tr),
            Arg::P(row_tok),
            Arg::P(meta),
            Arg::P(p_gu),
            Arg::P(p_dn),
            Arg::P(cnt_gu),
            Arg::P(cnt_dn),
            Arg::P(gu),
            Arg::P(q),
            Arg::P(sc),
            Arg::P(y),
            Arg::P(wts),
            Arg::P(pos),
            Arg::P(lw.sg),
            Arg::P(mx.bi),
            Arg::P(sg),
            Arg::P(out),
            Arg::P(sync),
            i(m),
            i(pcap),
            i(tc),
        ];
        if ss {
            args.push(Arg::P(
                self.shared_split + u64::from(lw.l) * SHX_TABLE_BYTES,
            ));
        }
        if let Some((_, k)) = mo {
            args.extend([
                Arg::L(self.miss_ram.0 as i64),
                Arg::L(self.miss_ram.1 as i64),
                Arg::I(k),
            ]);
        }
        if let Some(ch) = &far {
            args.push(Arg::Blob(ch.blob(red)));
        }
        self.run(
            &sym,
            [u64::from(shape.back_grid(&bt, tc as u32)), 1, 1],
            u64::from(moe::BackTrain::THREADS),
            args,
            Group::Moe,
        )?;
        if far.is_some() {
            self.ar_far += 1;
            return Ok(red);
        }
        Ok(out)
    }

    /// FN50: the MTP expert half over the MIXED table (Qwen's fp8 routed experts, `xptr[2e + 1]` =
    /// their fp32 block scales; the bf16 shared expert `[w, 0]`): the per-128 activation quant of
    /// the token rows, gate_up as two launches over the same tiles (`fn50::fp8_symbol` -- bf16
    /// tiles exit -- then `fn50::bf16_symbol` at the bf16 arm's split -- fp8 tiles exit), the bf16
    /// SiluAndMul, the quant of the compact rows, down as two launches, FN4's combine (8 launches;
    /// `fn50::MOE_BLOCK_LAUNCHES_FP8` with the route half).
    pub fn moe_experts_fp8(
        &mut self,
        rt: &MoeRoute,
        mx: &MixOutBufs,
        m: u64,
        xgu: u64,
        xdn: u64,
        carry_out: bool,
    ) -> anyhow::Result<u64> {
        use saber_kernels::fn50;
        let MoeRoute {
            sg,
            wts,
            pos,
            row_tok,
            te,
            t0,
            tr,
            meta,
            pcap,
            tc,
            ..
        } = *rt;
        let quant = |em: &mut Self, x: u64, rows: u64, n: u64| -> anyhow::Result<(u64, u64)> {
            let (q, s) = (em.s(rows * n)?, em.s(rows * (n / 128) * 4)?);
            em.run(
                fn50::ACT_QUANT_SYMBOL,
                [fn50::act_quant_blocks(rows, n), 1, 1],
                fn50::ACT_QUANT_THREADS,
                vec![
                    Arg::P(x),
                    Arg::P(q),
                    Arg::P(s),
                    Arg::I(rows as i32),
                    Arg::I(n as i32),
                ],
                Group::Moe,
            )?;
            Ok((q, s))
        };
        let grouped = |em: &mut Self,
                       gather: bool,
                       (a8, as8): (u64, u64),
                       a: u64,
                       xp: u64,
                       c: u64,
                       n: u64,
                       k: u64,
                       ks: u64|
         -> anyhow::Result<()> {
            let tabs = [
                Arg::P(xp),
                Arg::P(te),
                Arg::P(t0),
                Arg::P(tr),
                Arg::P(row_tok),
                Arg::P(meta),
            ];
            let mut args = vec![Arg::P(a8)];
            args.extend(tabs.iter().cloned());
            args.extend([Arg::P(as8), Arg::P(c), Arg::I(n as i32), Arg::I(k as i32)]);
            em.run(
                fn50::fp8_symbol(gather),
                [n / 128, tc, 1],
                128,
                args,
                Group::Moe,
            )?;
            let (part, cnt) = if ks > 1 {
                (em.s(ks * pcap * n * 4)?, em.z(tc * (n / 64) * 4)?)
            } else {
                (0, 0)
            };
            let mut args = vec![Arg::P(a)];
            args.extend(tabs);
            args.extend([
                Arg::P(part),
                Arg::P(cnt),
                Arg::P(c),
                Arg::I(n as i32),
                Arg::I(k as i32),
                Arg::I(pcap as i32),
            ]);
            em.run(
                fn50::bf16_symbol(gather),
                [n / 64, tc, ks],
                128,
                args,
                Group::Moe,
            )
        };
        let gu = self.s(pcap * 1280 * 2)?;
        let a8 = quant(self, mx.bi, m, H)?;
        grouped(self, true, a8, mx.bi, xgu, gu, 1280, H, rt.gu_ks)?;
        let act = self.s(pcap * 640 * 2)?;
        self.run(
            "fn_moe_silu_mul_bf16_t256",
            [pcap, 1, 1],
            256,
            vec![
                Arg::P(gu),
                Arg::P(act),
                Arg::I(640),
                Arg::P(row_tok),
                Arg::P(meta),
            ],
            Group::Moe,
        )?;
        let d8 = quant(self, act, pcap, 640)?;
        let y = self.s(pcap * H * 2)?;
        grouped(self, false, d8, act, xdn, y, H, 640, rt.dn_ks)?;
        let out = if carry_out {
            let slot = (self.layer.max(0) as usize) % 2;
            self.arena.carry[slot].take(m * H * 2)?
        } else {
            self.s(m * H * 2)?
        };
        self.run(
            moe::COMBINE_SYMBOL,
            [m, 1, 1],
            256,
            vec![
                Arg::P(y),
                Arg::P(wts),
                Arg::P(pos),
                Arg::P(sg),
                Arg::P(meta),
                Arg::P(out),
                Arg::I(H as i32),
            ],
            Group::Moe,
        )?;
        Ok(out)
    }

    /// FN15: the MTP expert half over the resident bf16 tables (`xptr[2e]` = the expert's weight):
    /// FN6's grouped bf16 gate_up (A rows gathered by token), `fn_moe_silu_mul_bf16_t256`, the
    /// grouped bf16 down, FN4's combine (4 launches, the block's 7 with the route half).
    pub fn moe_experts_bf16(
        &mut self,
        rt: &MoeRoute,
        mx: &MixOutBufs,
        m: u64,
        xgu: u64,
        xdn: u64,
        carry_out: bool,
    ) -> anyhow::Result<u64> {
        let MoeRoute {
            sg,
            wts,
            pos,
            row_tok,
            te,
            t0,
            tr,
            meta,
            pcap,
            tc,
            ..
        } = *rt;
        let grouped = |em: &mut Self,
                       gather: bool,
                       a: u64,
                       xp: u64,
                       c: u64,
                       n: u64,
                       k: u64,
                       ks: u64|
         -> anyhow::Result<()> {
            let (part, cnt) = if ks > 1 {
                (em.s(ks * pcap * n * 4)?, em.z(tc * (n / 64) * 4)?)
            } else {
                (0, 0)
            };
            em.run(
                saber_kernels::fn6::MtpExperts::Bf16.symbol(gather),
                [n / 64, tc, ks],
                128,
                vec![
                    Arg::P(a),
                    Arg::P(xp),
                    Arg::P(te),
                    Arg::P(t0),
                    Arg::P(tr),
                    Arg::P(row_tok),
                    Arg::P(meta),
                    Arg::P(part),
                    Arg::P(cnt),
                    Arg::P(c),
                    Arg::I(n as i32),
                    Arg::I(k as i32),
                    Arg::I(pcap as i32),
                ],
                Group::Moe,
            )
        };
        let gu = self.s(pcap * 1280 * 2)?;
        grouped(self, true, mx.bi, xgu, gu, 1280, H, rt.gu_ks)?;
        let act = self.s(pcap * 640 * 2)?;
        self.run(
            "fn_moe_silu_mul_bf16_t256",
            [pcap, 1, 1],
            256,
            vec![
                Arg::P(gu),
                Arg::P(act),
                Arg::I(640),
                Arg::P(row_tok),
                Arg::P(meta),
            ],
            Group::Moe,
        )?;
        let y = self.s(pcap * H * 2)?;
        grouped(self, false, act, xdn, y, H, 640, rt.dn_ks)?;
        let out = if carry_out {
            let slot = (self.layer.max(0) as usize) % 2;
            self.arena.carry[slot].take(m * H * 2)?
        } else {
            self.s(m * H * 2)?
        };
        self.run(
            moe::COMBINE_SYMBOL,
            [m, 1, 1],
            256,
            vec![
                Arg::P(y),
                Arg::P(wts),
                Arg::P(pos),
                Arg::P(sg),
                Arg::P(meta),
                Arg::P(out),
                Arg::I(H as i32),
            ],
            Group::Moe,
        )?;
        Ok(out)
    }

    /// FN15: the MTP pass BEFORE its layer (4 launches, `fn6::MTP_PRE_LAYER_LAUNCHES`):
    /// `fn_mtp_norms` (the embedding norm + ONE norm over the whole 10,240 stream), `fc_embedding`,
    /// `fc_hidden` at M = 4 x rows (the four HC streams share the weight), `fn_mtp_add`
    /// (`x[t, s] = bf16(fe[t] + fh[t, s])`, HC-outer). `emb` `[m, H]`, `hid` `[m, W]` bf16.
    /// Returns `x` `[m, W]` in carry slot 1 (the MTP layer, index 48, resets slot 0).
    pub fn mtp_pre(&mut self, w: &MtpW, emb: u64, hid: u64, m: u64) -> anyhow::Result<u64> {
        self.route_m = None;
        self.layer = 48;
        self.arena.new_layer();
        self.arena.carry[1].reset();
        let en = self.s(m * H * 2)?;
        let hn = self.s(m * W * 2)?;
        self.run(
            "fn_mtp_norms",
            [2 * m, 1, 1],
            256,
            vec![
                Arg::P(emb),
                Arg::P(w.ne),
                Arg::P(en),
                Arg::P(hid),
                Arg::P(w.nh),
                Arg::P(hn),
                Arg::I(m as i32),
            ],
            Group::Proj,
        )?;
        let fe = self.s(m * H * 2)?;
        self.gemm("mtp.fc", en, w.fce, fe, m, H, H, Group::Proj)?;
        let fh = self.s(4 * m * H * 2)?;
        self.gemm("mtp.fc", hn, w.fch, fh, 4 * m, H, H, Group::Proj)?;
        let x = self.arena.carry[1].take(m * W * 2)?;
        self.run(
            "fn_mtp_add",
            [cdiv(m * 1280, 256), 1, 1],
            256,
            vec![Arg::P(fe), Arg::P(fh), Arg::P(x), Arg::I(m as i32)],
            Group::Proj,
        )?;
        Ok(x)
    }

    // ------------------------------------------------------------------------ PLE (layer 1)
    /// Returns (the stream after the PLE add, the PLE addition).
    fn ple(
        &mut self,
        pw: &PleW,
        m: u64,
        pend: &Pending,
        rows: u64,
        st: &LayerState,
        cc: &ChunkConsts,
    ) -> anyhow::Result<(u64, u64)> {
        let key = self.s(m * W * 2)?;
        let val = self.s(m * H * 2)?;
        self.gemm("ple.key_proj", rows, pw.pkey, key, m, W, H, Group::Ple)?;
        // FN38: `ple.value_proj` = `mtp.fc`'s bands to 2,048 rows, the wide rung above
        self.gemm("ple.value_proj", rows, pw.pval, val, m, H, H, Group::Ple)?;
        let gv = self.s(m * W * 2)?;
        let xc = self.s(m * W * 2)?;
        let xs = self.s(m * W * 2)?;
        self.run(
            "fn_ple_gate_combine",
            [m, 1, 1],
            256,
            vec![
                Arg::P(pend.hid),
                Arg::P(pend.mo),
                Arg::P(pend.injo),
                Arg::P(key),
                Arg::P(val),
                Arg::P(pw.pk),
                Arg::P(pw.pq),
                Arg::P(pw.pc),
                Arg::P(xs),
                Arg::P(gv),
                Arg::P(xc),
            ],
            Group::Ple,
        )?;
        let out = self.s(m * W * 2)?;
        let po = self.s(m * W * 2)?;
        let sin = st.ple_st[(cc.chunk % 2) as usize];
        let sout = st.ple_st[((cc.chunk + 1) % 2) as usize];
        self.run(
            "fn_ple_conv",
            [m, 5, 1],
            256,
            vec![
                Arg::P(xc),
                Arg::P(gv),
                Arg::P(xs),
                Arg::P(pw.pcw),
                Arg::P(cc.ple_seq),
                Arg::P(cc.seq_row0),
                Arg::P(cc.mv),
                Arg::P(sin),
                Arg::P(sout),
                Arg::P(cc.drop),
                Arg::P(cc.hi32),
                Arg::I(PLE_HIST as i32),
                Arg::P(out),
                Arg::P(po),
            ],
            Group::Ple,
        )?;
        Ok((out, po))
    }
}

/// The route half's buffers, which the expert half and the capture read.
#[derive(Debug, Clone, Copy)]
pub struct MoeRoute {
    pub logits: u64,
    pub sg: u64,
    pub ids: u64,
    pub wts: u64,
    pub pos: u64,
    pub row_tok: u64,
    pub te: u64,
    pub t0: u64,
    pub tr: u64,
    pub meta: u64,
    pub pcap: u64,
    pub tc: u64,
    pub tm: u64,
    pub gu_ks: u64,
    pub dn_ks: u64,
}

/// Where each rank's grouped GEMMs read their xptr tables, decided between the route half and the
/// expert half: the complete tier (and TP=1) answer from what is already resident; the demand path
/// (after `evict_all`) runs the route half, reads the routes back, resolves them (NVMe reads) and
/// uploads fresh tables.
pub trait XptrHook {
    /// `ids[r]`: rank r's route ids buffer (`m * 10` i32). Returns `(xgu, xdn)` per rank.
    fn tables(
        &mut self,
        ems: &mut [Emitter<'_>],
        layer: u32,
        ids: &[u64],
        m: u64,
    ) -> anyhow::Result<Vec<(u64, u64)>>;
}

/// The weights' own tables (TP=1, or a complete tier filled before the chunk).
pub struct Resident<'a>(pub &'a [LayerW]);

impl XptrHook for Resident<'_> {
    fn tables(
        &mut self,
        _ems: &mut [Emitter<'_>],
        _layer: u32,
        _ids: &[u64],
        _m: u64,
    ) -> anyhow::Result<Vec<(u64, u64)>> {
        Ok(self.0.iter().map(|w| (w.xgu, w.xdn)).collect())
    }
}

/// One decoder layer on 1 or 2 ranks (the harness's `run_layer`). `rows`: the PLE staged rows
/// `[m, 2560]` bf16 per rank (layer 1). The carry region of layer `l % 2` receives the pending
/// triple; the reduced attention output goes to scratch, the reduced MoE output to the carry.
#[allow(clippy::too_many_arguments)]
pub fn emit_layer(
    ems: &mut [Emitter<'_>],
    lws: &[LayerW],
    sts: &[LayerState],
    ins: &[LayerIn],
    rows: &[u64],
    cc: &[ChunkConsts],
    ar: &mut Option<&mut dyn Reduce>,
    hook: &mut dyn XptrHook,
) -> anyhow::Result<Vec<LayerOut>> {
    let n = ems.len();
    // TP=2 is the emitters' own dims: the one-process form emits both ranks, a worker process its
    // own rank only (FN13); the reduce is emitted either way.
    let tp2 = ems[0].dims.tp == 2;
    let m = u64::from(cc[0].m);
    let l = lws[0].l;
    let n0: Vec<u32> = ems.iter().map(Emitter::forward_launches).collect();
    // FN24: a TARGET verify step (w >= 2) routes every M-keyed cell at its class rows; the MTP
    // layer (the drafter) and every other form keep their own M
    for (r, em) in ems.iter_mut().enumerate() {
        let wide = cc[r].decode && cc[r].w > 1 && !lws[r].mtp;
        let class = u64::from(em.class_rows(&cc[r]));
        em.route_m = (wide && class != m).then_some((m, class));
        // FN78: the reduce's band law reads the form (decode forms reduce exactly, <= 64 rows)
        em.ar_decode = cc[r].decode;
        if wide {
            em.vclass_layers[usize::from(em.route_m.is_none())] += 1;
        }
    }
    // FN85: the arm runs at a TP = 2 TARGET decode / verify form of <= 64 rows (the drafter's MTP
    // layer, prefill chunks and the producer-rr arm keep the standalone reduce)
    let rp_form = tp2
        && m <= DECODE_RP_MAX_ROWS
        && ems
            .iter()
            .zip(lws)
            .zip(cc)
            .all(|((e, w), c)| e.ar_prologue && c.decode && !w.mtp && !e.rules.hc.rr_producer);
    for em in ems.iter_mut() {
        em.ar_rp_saved = 0;
        em.ar_far_saved = 0;
        em.far_next = None;
        em.ar_mix_saved = 0;
        em.mix_next = None;
        em.mix_red = None;
        em.fuse86_saved = 0;
        em.df_saved = 0;
        em.pf122_added = 0;
    }
    // FN85: an incoming MoE partial (the previous layer's `Pending.rp`) is reduced by this layer's
    // attention-side combine down over the channel
    let rp_in: Vec<Option<RpChan>> = if ins.iter().any(|i| matches!(i, LayerIn::Pending(p) if p.rp))
    {
        ensure!(
            rp_form,
            "L{l}: a prologue-reduce partial outside the arm's form"
        );
        let red = ar.as_mut().context("TP=2 needs the reduce")?;
        let ch = red
            .rp_channel(ems, m * H)
            .context("the reduce gave no prologue channel for an rp partial")?;
        ch.into_iter().map(Some).collect()
    } else {
        vec![None; n]
    };
    // FN164: the mixer-out reduce rides the out / o projection (`--fn-mixer-ar-push on|b1`) at a
    // TP = 2 TARGET decode / verify form of the W8 trunk whose reduce is the standalone exact one
    // (not FN85's prologue, not FN37's producer-rr) on a cell with a fused twin; `b1` = the
    // one-sequence forms only
    let mut mix_used = false;
    if tp2 {
        let red = ar.as_mut().context("TP=2 needs the reduce")?;
        let arm = red.mix_arm();
        let asked = arm != super::dev::MixPush::Off
            && (0..n).all(|r| {
                cc[r].decode
                    && !lws[r].mtp
                    && lws[r].t8.is_some()
                    && (arm == super::dev::MixPush::On || cc[r].b == 1)
            });
        if asked {
            let form = !rp_form
                && ems
                    .iter()
                    .all(|e| !e.rules.hc.rr_producer && e.mix_cell_fuses(m));
            match form.then(|| red.mix_channel(ems, m * H)).flatten() {
                Some(ch) => {
                    for (r, em) in ems.iter_mut().enumerate() {
                        em.mix_next = Some(ch[r]);
                        em.ar_mix_saved += 1;
                    }
                    mix_used = true;
                }
                None if !form => red.mix_stood_down(),
                None => {}
            }
        }
    }
    let mut am = Vec::with_capacity(n);
    let mut att = Vec::with_capacity(n);
    let mut ple = vec![None; n];
    for r in 0..n {
        let em = &mut ems[r];
        em.layer = l as i32;
        em.arena.new_layer();
        let slot = (l % 2) as usize;
        em.arena.carry[slot].reset();
        let lw = &lws[r];
        let (mode, src, blk, inj) = match (&ins[r], lw.ple) {
            (LayerIn::Embed(x), _) => (MixMode::Expand, *x, None, None),
            (LayerIn::Stream(x), _) => (MixMode::Mix, *x, None, None),
            (LayerIn::Pending(p), Some(pw)) => {
                let (x, po) = em.ple(&pw, m, p, rows[r], &sts[r], &cc[r])?;
                ple[r] = Some(po);
                (MixMode::Mix, x, None, None)
            }
            (LayerIn::Pending(p), None) => (MixMode::Combine, p.hid, Some(p.mo), Some(p.injo)),
        };
        let pre = match (&ins[r], lw.ple) {
            (LayerIn::Pending(p), None) => p.pre,
            _ => None,
        };
        // FN85: the previous layer left its MoE partial for THIS combine down to reduce
        if let LayerIn::Pending(p) = &ins[r] {
            if p.rp {
                ensure!(
                    mode == MixMode::Combine && pre.is_none(),
                    "L{l}: a prologue-reduce partial reached a {mode:?} mix"
                );
                let ch = rp_in[r].context("a prologue-reduce partial without its channel")?;
                let out = em.s(m * H * 2)?;
                em.rp_next = Some(RpIn { ch, out });
            }
        }
        let mo = em.mix(&lw.ahc, mode, MixOut::Bf16, m, src, blk, inj, false, pre)?;
        if let Some((log, _)) = em.pregate.filter(|_| cc[r].decode && !lw.mtp) {
            ensure!(
                m <= PREGATE_ROWS,
                "the pre-gate capture holds <= {PREGATE_ROWS} rows"
            );
            em.pregate_gemm(mo.bi, lw.router, log + pregate_slot(0, u64::from(l)), m)?;
        }
        // FN61c: the W8 trunk at DECODE forms; FN61e: and at PREFILL under `w8` / `w8-head` (the
        // bf16 records are dropped then: a dropped layer can never take the bf16 emission). The
        // MTP layer has none.
        em.t8_cur = lw.t8.filter(|_| cc[r].decode || em.t8_prefill);
        ensure!(
            lw.t8.is_none_or(|t| t.sfx == 0 || em.t8_cur.is_some()),
            "L{}: the bf16 trunk records were dropped but this form reads bf16",
            lw.l
        );
        em.t8_post = em.t8_cur.is_some();
        let a = match lw.mixer {
            MixerW::Gdn { .. } => em.gdn_core(&lw.mixer, mo.bi, m, &sts[r], &cc[r]),
            MixerW::Qsa { .. } => em.qsa_core(&lw.mixer, mo.bi, m, &sts[r], &cc[r]),
        };
        em.t8_cur = None;
        let a = a?;
        am.push(mo);
        att.push(a);
    }
    // FN37: the attention output's reduce also writes the MoE-side mix's x' and rr
    let mut att_pre = vec![None; n];
    // FN85: under the arm the MoE-side combine down reduces the attention partial in its prologue
    let mut att_rp: Vec<Option<RpIn>> = vec![None; n];
    if mix_used {
        // FN164: every rank's out / o projection took the channel and wrote the SUM
        for (r, em) in ems.iter_mut().enumerate() {
            ensure!(
                em.mix_next.is_none(),
                "L{l}: a fused mixer channel no out / o projection took"
            );
            att[r] = em
                .mix_red
                .take()
                .with_context(|| format!("L{l}: the fused mixer reduce left no SUM"))?;
        }
    } else if tp2 {
        let red = ar.as_mut().context("TP=2 needs the reduce")?;
        let twins = ems.iter().all(|e| {
            e.rules
                .mix_plan_at(MixMode::Combine, MixOut::Q8, e.rm(m) as u32, m as u32)
                .is_ok_and(|p| has_rp_twin(&e.hc_route(&p.down.symbol)))
        });
        let ch = if rp_form && twins {
            red.rp_channel(ems, m * H)
        } else {
            None
        };
        if let Some(ch) = ch {
            for (r, em) in ems.iter_mut().enumerate() {
                let out = em.s(m * H * 2)?;
                att_rp[r] = Some(RpIn { ch: ch[r], out });
                em.ar_rp[0] += 1;
                em.ar_rp_saved += 1;
            }
        } else {
            if rp_form {
                for em in ems.iter_mut() {
                    em.ar_rp[1] += 1;
                }
            }
            let next: Vec<(u64, u64)> = am.iter().map(|o| (o.hid, o.injo)).collect();
            (att, att_pre) = red.reduce_hc(ems, &att, m * H, false, &next)?;
        }
    }
    for (r, em) in ems.iter_mut().enumerate() {
        if em.layer == em.tap_layer {
            em.taps.push(("att".into(), att[r], m * H * 2));
        }
    }
    let mut mms = Vec::with_capacity(n);
    let mut rts = Vec::with_capacity(n);
    let mut fuseds = Vec::with_capacity(n);
    for r in 0..n {
        let em = &mut ems[r];
        em.rp_next = att_rp[r];
        let mm = em.mix(
            &lws[r].mhc,
            MixMode::Combine,
            if lws[r].mtp && !lws[r].mtp_x4 {
                MixOut::Bf16
            } else {
                MixOut::Q8
            },
            m,
            am[r].hid,
            Some(att[r]),
            Some(am[r].injo),
            true,
            att_pre[r],
        )?;
        // FN85: the reduced attention output is what the rp down stored (the capture's `att`)
        if let Some(x) = att_rp[r] {
            att[r] = x.out;
        }
        if let Some((log, next)) = em
            .pregate
            .filter(|p| p.1 != 0 && cc[r].decode && !lws[r].mtp)
        {
            em.pregate_gemm(mm.bi, next, log + pregate_slot(1, u64::from(l) + 1), m)?;
        }
        // FN30: the target's MoE block at a decode / verify step of <= 64 rows runs FN26's
        // fused train (the drafter's bf16 experts and prefill chunks keep the seven)
        let fused = !lws[r].mtp && moe_block_fused(em.moe_train, cc[r].decode, m as u32);
        if !lws[r].mtp {
            em.moe_blocks[usize::from(!fused)] += 1;
        }
        let rt = if lws[r].mtp && !lws[r].mtp_x4 {
            em.moe_route_tm(&lws[r], &mm, m, Some(1), false)?
        } else if lws[r].mtp {
            // FN134: the slab4 MTP layer routes over its EP tables whatever the target's form;
            // FN138: under `--fn-mtp-split tp2` over its half tables at the target's tp2 dims
            let keep = em.dims;
            em.dims = mtp_x4_dims(&lws[r], keep)?;
            let rt = em.moe_route(&lws[r], &mm, m, fused);
            em.dims = keep;
            rt?
        } else {
            em.moe_route(&lws[r], &mm, m, fused)?
        };
        fuseds.push(fused);
        mms.push(mm);
        rts.push(rt);
    }
    let ids: Vec<u64> = rts.iter().map(|t| t.ids).collect();
    // FN85: the MoE partial is left for the NEXT layer's attention-side combine down when that
    // layer is a target layer without PLE (the PLE layer's combine and the final mixer keep the
    // standalone reduce), at the arm's form; it then lives in the carry like the reduced output
    let moe_rp = rp_form
        && l + 1 < super::plan::LAYERS
        && l + 1 != super::plan::PLE_LAYER
        && ems.iter().all(|e| {
            e.rules
                .mix_plan_at(MixMode::Combine, MixOut::Bf16, e.rm(m) as u32, m as u32)
                .is_ok_and(|p| has_rp_twin(&e.hc_route(&p.down.symbol)))
        });
    for em in ems.iter_mut() {
        if moe_rp {
            // counted at the producer: THIS layer's declared reduce is the one not emitted
            em.ar_rp[0] += 1;
            em.ar_rp_saved += 1;
        } else if rp_form {
            em.ar_rp[1] += 1;
        }
    }
    let xp = hook.tables(ems, l, &ids, m)?;
    // FN137: the MoE-out reduce rides the fused tp2 back half (`--fn-moe-ar-push on`) at a target
    // decode / verify form whose reduce is the standalone exact one (not FN85's prologue, not
    // FN37's producer-rr), on FN80b-1's id-order half-width body
    let far_form = tp2
        && !moe_rp
        && (0..n).all(|r| {
            let e = &ems[r];
            fuseds[r]
                && !lws[r].mtp
                && !e.rules.hc.rr_producer
                && e.dims.xshape() == moe::ExpertShape::TpHalf
                && e.miss_order
                    .symbol(&moe::BackTrain {
                        gu_ks: rts[r].gu_ks as u32,
                        dn_ks: rts[r].dn_ks as u32,
                        prefetch: true,
                    })
                    .is_none()
                && e.shared_split == 0
                && moe::BACK_PREFETCH_BODY == moe::DecodeBody::Prefetch
        });
    let mut far_used = false;
    if tp2 && !moe_rp {
        let red = ar.as_mut().context("TP=2 needs the reduce")?;
        if far_form {
            if let Some(ch) = red.far_channel(ems, m * H) {
                for (r, em) in ems.iter_mut().enumerate() {
                    em.far_next = Some(ch[r]);
                    em.ar_far_saved += 1;
                }
                far_used = true;
            }
        } else if fuseds.iter().all(|f| *f) && !lws[0].mtp {
            red.far_stood_down();
        }
    }
    let mut mos = Vec::with_capacity(n);
    for r in 0..n {
        mos.push(if lws[r].mtp && lws[r].mtp_fp8 {
            ems[r].moe_experts_fp8(&rts[r], &mms[r], m, xp[r].0, xp[r].1, !tp2)?
        } else if lws[r].mtp && lws[r].mtp_x4 {
            // FN110: the trunk's grouped W4A8 train over the MTP's resident MXFP4 slab (FN134: at
            // the whole-expert shape and EP split under `--fn-experts tp2` too)
            let keep = ems[r].dims;
            ems[r].dims = mtp_x4_dims(&lws[r], keep)?;
            let mo = ems[r].moe_experts(&rts[r], &mms[r], m, xp[r].0, xp[r].1, !tp2);
            ems[r].dims = keep;
            mo?
        } else if lws[r].mtp {
            ems[r].moe_experts_bf16(&rts[r], &mms[r], m, xp[r].0, xp[r].1, !tp2)?
        } else if fuseds[r] {
            ems[r].moe_back(
                &rts[r],
                &lws[r],
                &mms[r],
                m,
                xp[r].0,
                xp[r].1,
                !tp2 || moe_rp,
            )?
        } else {
            ems[r].moe_experts(&rts[r], &mms[r], m, xp[r].0, xp[r].1, !tp2 || moe_rp)?
        });
    }
    // FN37: the MoE output's reduce also writes the next layer's (or the final mixer's) x' and rr
    let mut mos_pre = vec![None; n];
    if far_used {
        ensure!(
            ems.iter().all(|e| e.far_next.is_none()),
            "L{l}: a fused-reduce channel no back half took"
        );
    }
    if tp2 && !moe_rp && !far_used {
        let red = ar.as_mut().context("TP=2 needs the reduce")?;
        let next: Vec<(u64, u64)> = mms.iter().map(|o| (o.hid, o.injo)).collect();
        (mos, mos_pre) = red.reduce_hc(ems, &mos, m * H, true, &next)?;
    }
    Ok((0..n)
        .map(|r| LayerOut {
            pend: Pending {
                hid: mms[r].hid,
                mo: mos[r],
                injo: mms[r].injo,
                pre: mos_pre[r],
                rp: moe_rp,
            },
            att: att[r],
            ids: rts[r].ids,
            logits: rts[r].logits,
            ple: ple[r],
            launches: ems[r].forward_launches() - n0[r],
        })
        .collect())
}

/// FN116: a QSA layer over a STREAM input (the MTP layer) up to its KV append only: the
/// attention-side mix + [`Emitter::qsa_kv_front`] -- the emission [`emit_layer`] makes for those
/// ops, in its order (one rank's emitter; no reduce is reached). The rows' K / V / indexer state is
/// written; nothing else of the layer runs.
pub fn emit_layer_kv_front(
    em: &mut Emitter<'_>,
    lw: &LayerW,
    st: &LayerState,
    x: u64,
    cc: &ChunkConsts,
) -> anyhow::Result<()> {
    ensure!(
        matches!(lw.mixer, MixerW::Qsa { .. }) && lw.ple.is_none(),
        "L{}: the K-side front is a QSA layer's without PLE",
        lw.l
    );
    let m = u64::from(cc.m);
    em.route_m = None;
    em.ar_decode = cc.decode;
    em.layer = lw.l as i32;
    em.arena.new_layer();
    em.arena.carry[(lw.l % 2) as usize].reset();
    let mo = em.mix(
        &lw.ahc,
        MixMode::Mix,
        MixOut::Bf16,
        m,
        x,
        None,
        None,
        false,
        None,
    )?;
    em.t8_cur = lw.t8.filter(|_| cc.decode || em.t8_prefill);
    em.t8_post = em.t8_cur.is_some();
    let r = em.qsa_kv_front(&lw.mixer, mo.bi, m, st, cc);
    em.t8_cur = None;
    r
}

/// The final mixer over the last layer's pending triple, then the bf16 head on the last row
/// (`all_rows`: on every row -- a decode step's `m` sequences).
/// `head`: this rank's `lm_head` rows (TP=1: the whole head, two launches of 124,160 rows; TP=2:
/// the rank's half). Returns (final hidden `[m, H]`, logits `[1, vocab / tp]`).
pub fn emit_post(
    em: &mut Emitter<'_>,
    fm: &HcW,
    head: u64,
    pend: &Pending,
    m: u64,
    all_rows: bool,
) -> anyhow::Result<(u64, u64)> {
    let (bi, lg, _) = emit_post3(em, fm, head, pend, m, all_rows)?;
    Ok((bi, lg))
}

/// [`emit_post`] that also returns the final mixer's COMBINED stream `[m, W]` (the pre-final-mixer
/// multi-stream: the MTP drafter's input -- FN15's aux capture; free, the Final mix writes it).
pub fn emit_post3(
    em: &mut Emitter<'_>,
    fm: &HcW,
    head: u64,
    pend: &Pending,
    m: u64,
    all_rows: bool,
) -> anyhow::Result<(u64, u64, u64)> {
    emit_post3_opt(em, fm, head, pend, m, all_rows, true)
}

/// [`emit_post3`] with the head optional (FN33: an int2 drafter pass scores its candidates in its
/// tail instead; `with_head` false returns logits 0).
pub fn emit_post3_opt(
    em: &mut Emitter<'_>,
    fm: &HcW,
    head: u64,
    pend: &Pending,
    m: u64,
    all_rows: bool,
    with_head: bool,
) -> anyhow::Result<(u64, u64, u64)> {
    em.layer = -1;
    em.arena.new_layer();
    let fx = em.mix(
        fm,
        MixMode::Final,
        MixOut::Bf16,
        m,
        pend.hid,
        Some(pend.mo),
        Some(pend.injo),
        false,
        pend.pre,
    )?;
    if !with_head {
        em.route_m = None;
        return Ok((fx.bi, 0, fx.hid));
    }
    let lg = emit_head(em, head, fx.bi, m, all_rows)?;
    Ok((fx.bi, lg, fx.hid))
}

/// [`emit_post3_opt`]'s head over the final mixer's output `fx` (`m` rows; the last one unless
/// `all_rows`), in the same arena layer (FN119g: emitted on its own after the mixer, so the
/// drafter's K-side front can start from the stream while the head runs: the same launches, the
/// same addresses). Returns the logits.
pub fn emit_head(
    em: &mut Emitter<'_>,
    head: u64,
    fx: u64,
    m: u64,
    all_rows: bool,
) -> anyhow::Result<u64> {
    let half = VOCAB / 2;
    let per = u64::from(2 / em.dims.tp);
    let rows = if all_rows { m } else { 1 };
    let lg = em.s(rows * per * half * 2)?;
    let first = fx + (m - rows) * H * 2;
    // FN61c: the target's decode-form post after a W8 layer reads the W8 head; FN61g: every post
    // does once the bf16 head is dropped (`w8-head`: the drafter's head = the verify step's bytes)
    let t8h = em.t8_head.filter(|_| em.t8_post || em.head_bf16_dropped);
    em.t8_post = false;
    ensure!(
        t8h.is_some() || !em.head_bf16_dropped,
        "the bf16 head is not resident (--fn-trunk w8-head) but this post reads it"
    );
    if let Some((hw, hs)) = t8h {
        ensure!(per == 1, "the W8 head is TP=2 only");
        em.w8_gemm(first, hw, hs, lg, rows, half, H, half, Group::Head)?;
    } else {
        for s in 0..per {
            ensure!(per == 1 || rows == 1, "TP=1 head halves at one row only");
            em.gemm(
                "head.tp2",
                first,
                head + s * half * H * 2,
                lg + s * half * 2,
                rows,
                half,
                H,
                Group::Head,
            )?;
        }
    }
    // FN24: the target step ends at its head (the drafter's pass, next, keeps its own M)
    em.route_m = None;
    Ok(lg)
}

#[cfg(test)]
mod fn51e_tests {
    use super::*;

    /// The pre-FN51e law of the identity block tables (`flat_map` + `collect`).
    fn old_rows(slots: &[i32], nb: u32) -> Vec<u8> {
        let v: Vec<i32> = slots
            .iter()
            .flat_map(|&i| (0..nb as i32).map(move |j| i * nb as i32 + j))
            .collect();
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// FN51f: THE TABLE PREFIX LAW -- a 262,144 boot's decode-form blob at the 10,240 band is the
    /// 10,240 boot's blob table for table (same names, same order) except: `slot` (the flat
    /// pools' KV slot, unread under the tier), `bt` (each row the band's 641 entries, the
    /// identity from `slot * nbs` = the long boot's own first 641 entries a row), no `bt_rows`
    /// (unread under the tier: `kbtr`) -- and it is the 10,240 boot's size class, not 2 MiB.
    #[test]
    fn fn51f_the_band_blob_is_the_short_boots_blob_but_the_flat_slots() {
        let (long, short) = (16_385u32, 641u32);
        for (n, w) in [(1usize, 1u32), (8, 1), (8, 4), (4, 3), (2, 2)] {
            let pos: Vec<u32> = (0..n as u32).map(|i| 300 + i * 977).collect();
            let pads = PadLaw {
                live: n.saturating_sub(1).max(1),
                pad: Some(8),
                idx: None,
            };
            let (na, dr) = (vec![1; n], vec![0; n]);
            let (a, ao) = ChunkConsts::host_verify_at(&pos, w, &na, &dr, short, w, pads);
            let (b, bo) =
                ChunkConsts::host_verify_band(&pos, w, &na, &dr, long, short, false, w, pads);
            let (c, co) = ChunkConsts::host_verify_at(&pos, w, &na, &dr, long, w, pads);
            fn names(o: &[(&'static str, usize)]) -> Vec<&'static str> {
                o.iter().map(|x| x.0).collect()
            }
            let mut want = names(&ao);
            want.retain(|x| *x != "bt_rows");
            assert_eq!(names(&bo), want, "n {n} w {w}");
            fn span(o: &[(&'static str, usize)], v: &[u8], k: &str) -> Vec<u8> {
                let i = o.iter().position(|x| x.0 == k).expect(k);
                let end = o.get(i + 1).map_or(v.len(), |x| x.1);
                v[o[i].1..end].to_vec()
            }
            for k in want.iter().filter(|k| !["slot", "bt"].contains(k)) {
                assert_eq!(span(&ao, &a, k), span(&bo, &b, k), "table {k} n {n} w {w}");
            }
            let bt = span(&bo, &b, "bt");
            assert!(bt.len() >= n * short as usize * 4 && bt.len() < n * short as usize * 4 + 512);
            let full = span(&co, &c, "bt");
            for r in 0..n {
                let row = &bt[r * short as usize * 4..(r + 1) * short as usize * 4];
                let lrow = &full[r * long as usize * 4..r * long as usize * 4 + row.len()];
                assert_eq!(row, lrow, "bt row {r}: the long boot's prefix");
            }
            assert!(b.len() < a.len() + 64 && c.len() > 8 * b.len().min(c.len() / 8));
        }
    }

    /// FN51e: the sized table builders write the same bytes as the pre-FN51e `flat_map` law at
    /// every decode shape (and the control: a shifted slot moves them).
    #[test]
    fn fn51e_the_sized_decode_tables_are_the_flat_map_bytes() {
        for &nb in &[641u32, 16_385] {
            for slots in [vec![0, 1, 2, 3, 4, 5, 6, 7], vec![0, 8, 8, 8], vec![3]] {
                assert_eq!(le32(&block_rows(&slots, nb)), old_rows(&slots, nb));
                let (mut b, mut offs) = (vec![7u8; 3], Vec::new());
                put_block_rows(&mut b, &mut offs, "bt", &slots, nb, nb);
                let mut want = vec![7u8; 3];
                put_table(&mut want, &mut Vec::new(), "bt", &old_rows(&slots, nb));
                assert_eq!((b, offs[0].1), (want, 256));
                let mut moved = slots.clone();
                moved[0] += 1;
                assert_ne!(le32(&block_rows(&moved, nb)), old_rows(&slots, nb));
            }
        }
        let v = [i64::MIN, -1, 0, 1, 1 << 40, i64::MAX];
        let old: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(le64(&v), old);
    }

    /// FN51e: the host cost of one decode step's tables at the 10,240 vs the 262,144 boot's row
    /// width (`cargo test --release ... -- --ignored --nocapture`; a timing, not a gate).
    #[test]
    #[ignore]
    fn fn51e_time_decode_tables() {
        for &(n, w) in &[(1usize, 4u32), (8, 3), (8, 1)] {
            for &nb in &[641u32, 16_385] {
                let pos: Vec<u32> = (0..n as u32).map(|i| 300 + i * 7).collect();
                let pads = PadLaw {
                    live: n,
                    pad: Some(8),
                    idx: None,
                };
                let row = le32(&(0..nb as i32).collect::<Vec<_>>());
                let t0 = std::time::Instant::now();
                let it = 200;
                let mut sz = 0;
                for _ in 0..it {
                    let (mut tb, offs) =
                        ChunkConsts::host_verify_at(&pos, w, &vec![1; n], &vec![0; n], nb, w, pads);
                    for (name, per) in [("bt", 1usize), ("bt_rows", w as usize)] {
                        if let Some(&(_, o)) = offs.iter().find(|x| x.0 == name) {
                            for r in 0..n * per {
                                let at = o + r * nb as usize * 4;
                                tb[at..at + nb as usize * 4].copy_from_slice(&row);
                            }
                        }
                    }
                    sz = tb.len();
                    std::hint::black_box(&tb);
                }
                eprintln!(
                    "n {n} w {w} nb {nb}: {:.3} ms/step, blob {} KiB",
                    t0.elapsed().as_secs_f64() * 1e3 / f64::from(it),
                    sz / 1024
                );
            }
        }
    }
}
